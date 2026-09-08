"""Optional cuSPARSELt runtime and FP16 2:4 matmul execution resource."""

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
from taichi_forge.types.primitive_types import f16, u8


DEFINITION = BundledRuntimeProviderDefinition(
    provider_id="cusparselt",
    provider_name="NVIDIA cuSPARSELt",
    adapter_stem="taichi_forge_cusparselt_provider_abi2_api080_090",
    query_symbol="taichi_forge_cusparselt_provider_query",
    provider_abi_name="taichi-forge-cusparselt-provider-c-abi2",
    environment_variable="TI_CUSPARSELT_LIBRARY_PATH",
    library_names=(
        ("cusparseLt64_0.dll", "cusparseLt.dll")
        if os.name == "nt"
        else ("libcusparseLt.so.0", "libcusparseLt.so")
    ),
    package_distributions=("nvidia-cusparselt-cu13", "nvidia-cusparselt-cu12"),
    supported_version_family="0.8.x-0.9.x",
)


class _PlanDesc(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("alignment_bytes", ctypes.c_uint32),
        ("m", ctypes.c_int64),
        ("n", ctypes.c_int64),
        ("k", ctypes.c_int64),
    ]


class _PlanInfo(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("compressed_bytes", ctypes.c_uint64),
        ("compression_buffer_bytes", ctypes.c_uint64),
        ("workspace_bytes", ctypes.c_uint64),
    ]


class _CompressDesc(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("dense_a", ctypes.c_uint64),
        ("compressed_a", ctypes.c_uint64),
        ("compression_buffer", ctypes.c_uint64),
        ("compression_buffer_bytes", ctypes.c_uint64),
        ("cuda_stream", ctypes.c_uint64),
    ]


class _ExecDesc(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("alpha", ctypes.c_float),
        ("beta", ctypes.c_float),
        ("compressed_a", ctypes.c_uint64),
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
_Compress = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(_CompressDesc)
)
_Execute = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(_ExecDesc))
_DestroyPlan = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)


class _ExecutionApi(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("execution_abi_version", ctypes.c_uint32),
        ("create_matmul_plan", _CreatePlan),
        ("compress_sparse_a", _Compress),
        ("execute_matmul", _Execute),
        ("destroy_matmul_plan", _DestroyPlan),
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


def _validate_array(value, shape, name):
    if (
        not isinstance(value, Ndarray)
        or value.dtype != f16
        or value.element_shape != ()
    ):
        raise TaichiRuntimeError(
            f"cuSPARSELt {name} must be a scalar f16 Taichi ndarray"
        )
    if tuple(value.shape) != shape:
        raise TaichiRuntimeError(
            f"cuSPARSELt {name} shape must be {shape}, got {tuple(value.shape)}"
        )


class CusparseLtProvider:
    """Owner of one retained cuSPARSELt runtime and its matmul plans."""

    def __init__(self, library_path=None):
        program = _require_cuda_program("CusparseLtProvider")
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
            raise TaichiRuntimeError("CusparseLtProvider has been closed")
        validate_runtime_generation(
            self, "CusparseLtProvider belongs to a previous Taichi runtime generation"
        )

    def matmul_plan(self, m, n, k, *, alignment_bytes=16):
        with self._lock:
            self._validate_lifetime()
            return CusparseLtMatmulPlan(self, m, n, k, alignment_bytes=alignment_bytes)

    def close(self):
        with self._lock:
            if self._runtime is None:
                return None
            if any(not plan.closed for plan in self._plans):
                raise TaichiRuntimeError(
                    "CusparseLtProvider cannot close while matmul plans are live"
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


class CusparseLtMatmulPlan:
    """Reusable row-major FP16 2:4 A times transposed-storage B matmul."""

    def __init__(self, provider, m, n, k, *, alignment_bytes):
        if not isinstance(provider, CusparseLtProvider):
            raise TypeError("provider must be a CusparseLtProvider")
        dimensions = (m, n, k)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in dimensions
        ):
            raise ValueError("cuSPARSELt m, n, and k must be positive integers")
        if any(value % 16 for value in dimensions):
            raise ValueError("cuSPARSELt FP16 m, n, and k must be multiples of 16")
        if (
            isinstance(alignment_bytes, bool)
            or not isinstance(alignment_bytes, int)
            or alignment_bytes <= 0
        ):
            raise ValueError("cuSPARSELt alignment_bytes must be a positive integer")
        desc = _PlanDesc(ctypes.sizeof(_PlanDesc), alignment_bytes, m, n, k)
        handle = ctypes.c_void_p()
        info = _PlanInfo()
        info.struct_size = ctypes.sizeof(_PlanInfo)
        try:
            provider._runtime.check_result(
                provider._execution_api.create_matmul_plan(
                    provider._runtime.handle,
                    ctypes.byref(desc),
                    ctypes.byref(handle),
                    ctypes.byref(info),
                )
            )
        except RuntimeError as exc:
            raise TaichiRuntimeError(str(exc)) from exc
        if not handle.value:
            raise TaichiRuntimeError("cuSPARSELt returned a null matmul plan")
        self.provider = provider
        self._handle = handle
        self._lock = threading.RLock()
        self._runtime_prog = provider._runtime_prog
        self._runtime_generation = provider._runtime_generation
        self.m, self.n, self.k = dimensions
        self._alignment_bytes = alignment_bytes
        self._capture_leases = 0
        self._capture_mode = None
        self.compressed_bytes = int(info.compressed_bytes)
        self.compression_buffer_bytes = int(info.compression_buffer_bytes)
        self.workspace_bytes = int(info.workspace_bytes)
        try:
            self._compressed_a = ScalarNdarray(u8, (self.compressed_bytes,))
            self._compression_buffer = (
                ScalarNdarray(u8, (self.compression_buffer_bytes,))
                if self.compression_buffer_bytes
                else None
            )
            self._workspace = (
                ScalarNdarray(u8, (self.workspace_bytes,))
                if self.workspace_bytes
                else None
            )
            self._compressed_ready = False
            provider._plans.add(self)
        except Exception:
            provider._execution_api.destroy_matmul_plan(handle)
            self._handle = None
            raise

    @property
    def closed(self):
        return self._handle is None

    def _validate_lifetime(self):
        if self._handle is None:
            raise TaichiRuntimeError("CusparseLtMatmulPlan has been closed")
        self.provider._validate_lifetime()
        validate_runtime_generation(
            self, "cuSPARSELt plan belongs to a previous Taichi runtime generation"
        )

    def compress(self, a):
        """Compress an already-valid 2:4 sparse A matrix with shape (m, k)."""

        with self.provider._lock, self._lock:
            self._validate_lifetime()
            if self._capture_leases:
                raise TaichiRuntimeError(
                    "cuSPARSELt compressed snapshot cannot change while capture leases are live; "
                    "retire the recordings or use a new plan for the new weight epoch"
                )
            _validate_array(a, (self.m, self.k), "A")
            resources = [a, self._compressed_a]
            if self._compression_buffer is not None:
                resources.append(self._compression_buffer)
            with external_cuda_submission(self._runtime_prog, resources) as submission:
                compression_pointer = (
                    0
                    if self._compression_buffer is None
                    else _device_pointer(self._compression_buffer)
                )
                desc = _CompressDesc(
                    ctypes.sizeof(_CompressDesc),
                    0,
                    _device_pointer(a),
                    _device_pointer(self._compressed_a),
                    compression_pointer,
                    self.compression_buffer_bytes,
                    0,
                )
                self._compressed_ready = False
                try:
                    result = submission.invoke(
                        self.provider._execution_api.compress_sparse_a,
                        self._handle,
                        ctypes.byref(desc),
                    )
                    self.provider._runtime.check_result(result)
                except RuntimeError as exc:
                    raise TaichiRuntimeError(str(exc)) from exc
            self._compressed_ready = True
        return self

    def execute(self, b, c, d, *, alpha=1.0, beta=0.0):
        """Execute A @ B with B stored as a row-major (n, k) transpose."""

        with self.provider._lock, self._lock:
            self._validate_lifetime()
            if not self._compressed_ready:
                raise TaichiRuntimeError(
                    "cuSPARSELt A must be compressed before matmul execution"
                )
            _validate_array(b, (self.n, self.k), "B transposed storage")
            _validate_array(c, (self.m, self.n), "C")
            _validate_array(d, (self.m, self.n), "D")
            resources = [self._compressed_a, b, c, d]
            if self._workspace is not None:
                resources.append(self._workspace)
            with external_cuda_submission(self._runtime_prog, resources) as submission:
                workspace_pointer = (
                    0 if self._workspace is None else _device_pointer(self._workspace)
                )
                pointers = tuple(_device_pointer(value) for value in (b, c, d))
                if pointers[2] == pointers[0]:
                    raise TaichiRuntimeError(
                        "cuSPARSELt D must not alias B transposed storage"
                    )
                desc = _ExecDesc(
                    ctypes.sizeof(_ExecDesc),
                    0,
                    float(alpha),
                    float(beta),
                    _device_pointer(self._compressed_a),
                    *pointers,
                    workspace_pointer,
                    self.workspace_bytes,
                    0,
                )
                try:
                    result = submission.invoke(
                        self.provider._execution_api.execute_matmul,
                        self._handle,
                        ctypes.byref(desc),
                    )
                    self.provider._runtime.check_result(result)
                except RuntimeError as exc:
                    raise TaichiRuntimeError(str(exc)) from exc
        return d

    def record(self, *, a=None, b="b", c="c", d="d", alpha=1.0, beta=0.0):
        """Record a retained matmul, optionally preceded by compression of A.

        With a=None, compress() must have established a snapshot. It cannot be
        overwritten until all recordings/Graphs release it. With a binding name,
        A is compressed each replay; its current values must already satisfy 2:4.
        B/C/D remain live bindings, C/D may alias. No implicit pruning or value
        inspection is performed. Neither bind nor capture advances the math.
        """
        from taichi_forge.hardware._cusparselt_capture import CusparseLtCaptureRecording

        return CusparseLtCaptureRecording(
            self, a=a, b=b, c=c, d=d, alpha=alpha, beta=beta
        )

    def close(self):
        with self.provider._lock, self._lock:
            if self._handle is None:
                return None
            if self._capture_leases:
                raise TaichiRuntimeError(
                    "cuSPARSELt plan cannot close while capture leases are live"
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
                self.provider._execution_api.destroy_matmul_plan(self._handle)
            )
        except RuntimeError as exc:
            raise TaichiRuntimeError(str(exc)) from exc
        self._handle = None
        self._compressed_a = self._compression_buffer = self._workspace = None
        self._compressed_ready = False

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
    "CusparseLtMatmulPlan",
    "CusparseLtProvider",
    "DEFINITION",
    "passive_status",
    "probe_provider",
    "resolve_library_path",
)
