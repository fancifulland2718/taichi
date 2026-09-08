"""Optional cuTENSOR runtime and FP32 contraction execution resource."""

import ctypes
import os
from types import MappingProxyType
import threading
import weakref

from taichi_forge.hardware._bundled_runtime_provider import (
    BundledRuntimeProviderDefinition,
    open_runtime as _open_runtime,
    passive_status as _passive_status,
    probe_provider as _probe_provider,
    resolve_library_path as _resolve_library_path,
)
from taichi_forge.hardware._external_cuda_submission import external_cuda_submission
from taichi_forge.hardware._native_adapter import (
    runtime_generation_matches,
    validate_runtime_generation,
)
from taichi_forge.hardware._runtime import active_backend
from taichi_forge.lang import impl
from taichi_forge.lang._ndarray import Ndarray, ScalarNdarray
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.types.primitive_types import f32, u8


DEFINITION = BundledRuntimeProviderDefinition(
    provider_id="cutensor",
    provider_name="NVIDIA cuTENSOR",
    adapter_stem="taichi_forge_cutensor_provider_abi2_api200_207",
    query_symbol="taichi_forge_cutensor_provider_query",
    provider_abi_name="taichi-forge-cutensor-provider-c-abi2",
    environment_variable="TI_CUTENSOR_LIBRARY_PATH",
    library_names=(
        ("cutensor64_2.dll", "cutensor.dll")
        if os.name == "nt"
        else ("libcutensor.so.2", "libcutensor.so")
    ),
    package_distributions=("cutensor-cu13", "cutensor-cu12"),
    supported_version_family="2.0.x-2.7.x",
)


class _TensorDesc(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("rank", ctypes.c_uint32),
        ("extents", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("modes", ctypes.POINTER(ctypes.c_int32)),
    ]


class _PlanDesc(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("compute_mode", ctypes.c_uint32),
        ("alignment_bytes", ctypes.c_uint32),
        ("workspace_preference", ctypes.c_uint32),
        ("workspace_limit_bytes", ctypes.c_uint64),
        ("a", _TensorDesc),
        ("b", _TensorDesc),
        ("c", _TensorDesc),
        ("d", _TensorDesc),
    ]


class _PlanInfo(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("workspace_estimate_bytes", ctypes.c_uint64),
        ("workspace_required_bytes", ctypes.c_uint64),
    ]


class _ExecDesc(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("alpha", ctypes.c_float),
        ("beta", ctypes.c_float),
        ("a", ctypes.c_uint64),
        ("b", ctypes.c_uint64),
        ("c", ctypes.c_uint64),
        ("d", ctypes.c_uint64),
        ("workspace", ctypes.c_uint64),
        ("workspace_bytes", ctypes.c_uint64),
        ("cuda_stream", ctypes.c_uint64),
    ]


_CreatePlan = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(_PlanDesc),
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(_PlanInfo),
)
_Execute = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(_ExecDesc))
_DestroyPlan = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)


class _ExecutionApi(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("execution_abi_version", ctypes.c_uint32),
        ("create_contraction_plan", _CreatePlan),
        ("execute_contraction", _Execute),
        ("destroy_contraction_plan", _DestroyPlan),
    ]


def resolve_library_path(library_path=None):
    return _resolve_library_path(DEFINITION, library_path)


def probe_provider(library_path=None):
    return _probe_provider(DEFINITION, library_path)


def passive_status():
    return _passive_status(DEFINITION)


def _require_cuda_program(name):
    program = impl.get_runtime().prog
    if program is None or active_backend() != "cuda":
        raise TaichiRuntimeError(f"{name} requires an initialized Taichi CUDA runtime")
    return program


def _device_pointer(value):
    return int(impl.get_runtime().prog.get_ndarray_data_ptr_as_int(value.arr))


def _packed_strides(shape):
    stride = 1
    output = [0] * len(shape)
    for index in range(len(shape) - 1, -1, -1):
        output[index] = stride
        stride *= shape[index]
    return tuple(output)


def _normalize_tensor(shape, modes, name):
    shape = tuple(int(value) for value in shape)
    if not shape or len(shape) > 32 or any(value <= 0 for value in shape):
        raise ValueError(f"cuTENSOR {name} shape must have 1-32 positive dimensions")
    normalized_modes = (
        tuple(ord(value) for value in modes)
        if isinstance(modes, str)
        else tuple(int(value) for value in modes)
    )
    if len(normalized_modes) != len(shape) or len(set(normalized_modes)) != len(
        normalized_modes
    ):
        raise ValueError(f"cuTENSOR {name} modes must be unique and match its rank")
    return shape, normalized_modes, _packed_strides(shape)


def _native_tensor(shape, modes, strides):
    extents_array = (ctypes.c_int64 * len(shape))(*shape)
    strides_array = (ctypes.c_int64 * len(strides))(*strides)
    modes_array = (ctypes.c_int32 * len(modes))(*modes)
    desc = _TensorDesc(
        ctypes.sizeof(_TensorDesc),
        len(shape),
        extents_array,
        strides_array,
        modes_array,
    )
    return desc, (extents_array, strides_array, modes_array)


def _validate_array(value, shape, name):
    if (
        not isinstance(value, Ndarray)
        or value.dtype != f32
        or value.element_shape != ()
    ):
        raise TaichiRuntimeError(f"cuTENSOR {name} must be a scalar f32 Taichi ndarray")
    if tuple(value.shape) != shape:
        raise TaichiRuntimeError(
            f"cuTENSOR {name} shape must be {shape}, got {tuple(value.shape)}"
        )


class CutensorProvider:
    """Owner of one retained cuTENSOR runtime and its contraction plans."""

    def __init__(self, library_path=None):
        program = _require_cuda_program("CutensorProvider")
        runtime = None
        try:
            runtime = _open_runtime(DEFINITION, library_path)
            execution_api = runtime.query_execution_api(_ExecutionApi)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            if runtime is not None:
                try:
                    runtime.close()
                except RuntimeError:
                    pass
            raise TaichiRuntimeError(str(exc) or type(exc).__name__) from exc
        self._runtime = runtime
        self._execution_api = execution_api
        self._runtime_prog = program
        self._runtime_generation = int(impl.runtime_generation())
        self._lock = threading.RLock()
        self._plans = weakref.WeakSet()
        self.identity = MappingProxyType(
            {
                "provider_abi": DEFINITION.provider_abi_name,
                "provider_version": f"{runtime.runtime_info['version_major']}.{runtime.runtime_info['version_minor']}.{runtime.runtime_info['version_patch']}",
                "vendor_library": runtime.runtime_info["library_path"],
                "execution_abi_version": int(execution_api.execution_abi_version),
            }
        )
        impl.get_runtime().register_runtime_object(self)

    @property
    def closed(self):
        return self._runtime is None

    def _validate_lifetime(self):
        if self._runtime is None:
            raise TaichiRuntimeError("CutensorProvider has been closed")
        validate_runtime_generation(
            self, "CutensorProvider belongs to a previous Taichi runtime generation"
        )

    def contraction_plan(
        self,
        a_shape,
        a_modes,
        b_shape,
        b_modes,
        c_shape,
        c_modes,
        d_shape,
        d_modes,
        *,
        compute="f32",
        alignment_bytes=128,
        workspace_preference="default",
        workspace_limit_bytes=128 << 20,
    ):
        with self._lock:
            self._validate_lifetime()
            return CutensorContractionPlan(
                self,
                a_shape,
                a_modes,
                b_shape,
                b_modes,
                c_shape,
                c_modes,
                d_shape,
                d_modes,
                compute=compute,
                alignment_bytes=alignment_bytes,
                workspace_preference=workspace_preference,
                workspace_limit_bytes=workspace_limit_bytes,
            )

    def close(self):
        with self._lock:
            if self._runtime is None:
                return None
            if any(not plan.closed for plan in self._plans):
                raise TaichiRuntimeError(
                    "CutensorProvider cannot close while contraction plans are live"
                )
            runtime = self._runtime
            if runtime_generation_matches(self):
                self._runtime_prog.synchronize()
            try:
                runtime.close()
            except RuntimeError as exc:
                raise TaichiRuntimeError(str(exc)) from exc
            self._runtime = None
        return None

    destroy = close

    def _invalidate_runtime(self):
        # Retire native plans before Program.finalize() destroys their device.
        # Graphs from this generation are invalidated by the same cold boundary.
        with self._lock:
            if self.closed:
                return
            self._runtime_prog.synchronize()
            for plan in tuple(self._plans):
                with plan._lock:
                    plan._close_native()
            self._runtime.close()
            self._runtime = None

    def __enter__(self):
        self._validate_lifetime()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class CutensorContractionPlan:
    """Reusable FP32 cuTENSOR contraction plan over scalar CUDA ndarrays."""

    def __init__(
        self,
        provider,
        a_shape,
        a_modes,
        b_shape,
        b_modes,
        c_shape,
        c_modes,
        d_shape,
        d_modes,
        *,
        compute,
        alignment_bytes,
        workspace_preference,
        workspace_limit_bytes,
    ):
        if not isinstance(provider, CutensorProvider):
            raise TypeError("provider must be a CutensorProvider")
        compute_modes = {"f32": 0, "tf32": 1}
        workspace_modes = {"min": 1, "default": 2, "max": 3}
        if compute not in compute_modes:
            raise ValueError("cuTENSOR compute must be 'f32' or 'tf32'")
        if workspace_preference not in workspace_modes:
            raise ValueError(
                "cuTENSOR workspace_preference must be 'min', 'default', or 'max'"
            )
        if (
            isinstance(alignment_bytes, bool)
            or not isinstance(alignment_bytes, int)
            or alignment_bytes <= 0
        ):
            raise ValueError("cuTENSOR alignment_bytes must be a positive integer")
        if (
            isinstance(workspace_limit_bytes, bool)
            or not isinstance(workspace_limit_bytes, int)
            or workspace_limit_bytes < 0
        ):
            raise ValueError(
                "cuTENSOR workspace_limit_bytes must be a nonnegative integer"
            )
        tensors = tuple(
            _normalize_tensor(shape, modes, name)
            for shape, modes, name in (
                (a_shape, a_modes, "A"),
                (b_shape, b_modes, "B"),
                (c_shape, c_modes, "C"),
                (d_shape, d_modes, "D"),
            )
        )
        native_tensors = tuple(_native_tensor(*tensor) for tensor in tensors)
        desc = _PlanDesc(
            ctypes.sizeof(_PlanDesc),
            compute_modes[compute],
            alignment_bytes,
            workspace_modes[workspace_preference],
            workspace_limit_bytes,
            *(item[0] for item in native_tensors),
        )
        handle = ctypes.c_void_p()
        info = _PlanInfo()
        info.struct_size = ctypes.sizeof(_PlanInfo)
        try:
            provider._runtime.check_result(
                provider._execution_api.create_contraction_plan(
                    provider._runtime.handle,
                    ctypes.byref(desc),
                    ctypes.byref(handle),
                    ctypes.byref(info),
                )
            )
        except RuntimeError as exc:
            raise TaichiRuntimeError(str(exc)) from exc
        if not handle.value:
            raise TaichiRuntimeError("cuTENSOR returned a null contraction plan")
        self.provider = provider
        self._handle = handle
        self._lock = threading.RLock()
        self._runtime_prog = provider._runtime_prog
        self._runtime_generation = provider._runtime_generation
        self._shapes = tuple(item[0] for item in tensors)
        self._tensors = tensors
        self._alignment_bytes = alignment_bytes
        self._compute = compute
        self._capture_leases = 0
        self.workspace_estimate_bytes = int(info.workspace_estimate_bytes)
        self.workspace_required_bytes = int(info.workspace_required_bytes)
        try:
            self._workspace = (
                ScalarNdarray(u8, (self.workspace_required_bytes,))
                if self.workspace_required_bytes
                else None
            )
            provider._plans.add(self)
        except Exception:
            provider._execution_api.destroy_contraction_plan(handle)
            self._handle = None
            raise

    @property
    def closed(self):
        return self._handle is None

    def _validate_lifetime(self):
        if self._handle is None:
            raise TaichiRuntimeError("CutensorContractionPlan has been closed")
        self.provider._validate_lifetime()
        validate_runtime_generation(
            self, "cuTENSOR plan belongs to a previous Taichi runtime generation"
        )

    def execute(self, a, b, c, d, *, alpha=1.0, beta=0.0):
        with self.provider._lock, self._lock:
            self._validate_lifetime()
            for value, shape, name in zip((a, b, c, d), self._shapes, "ABCD"):
                _validate_array(value, shape, name)
            resources = [a, b, c, d]
            if self._workspace is not None:
                resources.append(self._workspace)
            with external_cuda_submission(self._runtime_prog, resources) as submission:
                workspace_pointer = (
                    0 if self._workspace is None else _device_pointer(self._workspace)
                )
                pointers = tuple(_device_pointer(value) for value in (a, b, c, d))
                if pointers[3] in pointers[:2]:
                    raise TaichiRuntimeError("cuTENSOR D must not alias A or B")
                desc = _ExecDesc(
                    ctypes.sizeof(_ExecDesc),
                    0,
                    float(alpha),
                    float(beta),
                    *pointers,
                    workspace_pointer,
                    self.workspace_required_bytes,
                    0,
                )
                try:
                    result = submission.invoke(
                        self.provider._execution_api.execute_contraction,
                        self._handle,
                        ctypes.byref(desc),
                    )
                    self.provider._runtime.check_result(result)
                except RuntimeError as exc:
                    raise TaichiRuntimeError(str(exc)) from exc
        return d

    def record(
        self,
        *,
        a="a",
        b="b",
        c="c",
        d="d",
        workspace="cutensor_workspace",
        alpha=1.0,
        beta=0.0,
    ):
        """Record this fixed plan into a root CUDA Graph.

        Arrays are supplied at Graph.bind(), and may change values each replay.
        The recording retains this plan and workspace until all Graphs retire.
        C/D may alias only with identical shapes, strides and modes. Capturing
        does not execute the contraction or advance beta*C feedback.
        """
        from taichi_forge.hardware._cutensor_capture import CutensorCaptureRecording

        return CutensorCaptureRecording(
            self, a=a, b=b, c=c, d=d, workspace=workspace, alpha=alpha, beta=beta
        )

    def close(self):
        with self.provider._lock, self._lock:
            if self._handle is None:
                return None
            if self._capture_leases:
                raise TaichiRuntimeError(
                    "cuTENSOR plan cannot close while capture leases are live"
                )
            if runtime_generation_matches(self):
                self._runtime_prog.synchronize()
                self._close_native()
        return None

    def _close_native(self):
        if self._handle is None:
            return
        try:
            self.provider._runtime.check_result(
                self.provider._execution_api.destroy_contraction_plan(self._handle)
            )
        except RuntimeError as exc:
            raise TaichiRuntimeError(str(exc)) from exc
        self._handle = None
        self._workspace = None

    destroy = close

    def __enter__(self):
        self._validate_lifetime()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


__all__ = (
    "CutensorContractionPlan",
    "CutensorProvider",
    "DEFINITION",
    "passive_status",
    "probe_provider",
    "resolve_library_path",
)
