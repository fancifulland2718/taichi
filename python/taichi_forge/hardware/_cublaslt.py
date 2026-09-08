"""Internal retained cuBLASLt proof loaded from the user's CUDA runtime.

This module is deliberately not re-exported from :mod:`taichi_forge.hardware`.
It does not add a wheel dependency or rewrite ordinary Taichi matrix code.  A
caller explicitly creates a row-major f32 plan; the process retains the shared
library, one provider handle belongs to one Taichi runtime generation, and the
plan retains descriptors, a selected algorithm, and its exact workspace.
"""

import _ctypes
import ctypes
import hashlib
import importlib.metadata
import math
import numbers
import os
from pathlib import Path
import shutil
import sys
import threading
import weakref

from taichi_forge._hardware_telemetry import hardware_provider_call
from taichi_forge.graph._ir import GraphAccess, ResourceEffect
from taichi_forge.graph._native import BackendCommandRecording
from taichi_forge.hardware._native_adapter import (
    native_recording_node,
    runtime_generation_matches,
    static_resource_effect,
    validate_exact_bindings,
    validate_runtime_generation,
)
from taichi_forge.hardware._external_cuda_submission import external_cuda_submission
from taichi_forge.hardware._retained import (
    HardwareExecutionCostModel,
    RetainedExecutionContract,
    attach_retained_execution_contract,
    fixed_cost,
    make_retained_plan_identity,
    scale_cost,
)
from taichi_forge.hardware._runtime import active_backend
from taichi_forge.lang import impl
from taichi_forge.lang._ndarray import Ndarray
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.types.primitive_types import f32, u8


CUBLASLT_PROVIDER_ABI = "cublaslt-dynamic-symbols-v1"
_ENVIRONMENT_VARIABLE = "TI_CUBLASLT_LIBRARY_PATH"
_CUDA_R_32F = 0
_CUBLAS_COMPUTE_32F = 68
_CUBLAS_OP_N = 0
_CUBLAS_OP_T = 1
_CUBLASLT_ORDER_ROW = 1
_MATRIX_LAYOUT_ORDER = 1
_MATRIX_LAYOUT_BATCH_COUNT = 5
_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = 6
_MATMUL_DESC_TRANSA = 3
_MATMUL_DESC_TRANSB = 4
_MATMUL_DESC_EPILOGUE = 7
_EPILOGUE_RELU = 2
_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1
_CUBLAS_STATUS_SUCCESS = 0


def _version_string(version):
    return f"{version // 10000}.{version // 100 % 100}.{version % 100}"


class _MatmulAlgo(ctypes.Structure):
    _fields_ = (("data", ctypes.c_uint64 * 8),)


class _HeuristicResult(ctypes.Structure):
    _fields_ = (
        ("algo", _MatmulAlgo),
        ("workspace_size", ctypes.c_size_t),
        ("state", ctypes.c_int),
        ("waves_count", ctypes.c_float),
        ("reserved", ctypes.c_int * 4),
    )


def _library_names():
    if os.name == "nt":
        return (
            "cublasLt64_13.dll",
            "cublasLt64_12.dll",
            "cublasLt64_11.dll",
            "cublasLt.dll",
        )
    return (
        "libcublasLt.so.13",
        "libcublasLt.so.12",
        "libcublasLt.so.11",
        "libcublasLt.so",
    )


def _path_candidate(value):
    path = Path(os.fspath(value)).expanduser()
    if path.is_file():
        return str(path.resolve())
    if path.is_dir():
        for relative in ((), ("bin",), ("bin", "x64"), ("lib",), ("lib64",)):
            for name in _library_names():
                candidate = path.joinpath(*relative, name)
                if candidate.is_file():
                    return str(candidate.resolve())
    return None


def _distribution_candidates():
    candidates = []
    for distribution_name in ("nvidia-cublas-cu13", "nvidia-cublas-cu12"):
        try:
            distribution = importlib.metadata.distribution(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            continue
        for item in distribution.files or ():
            if Path(item).name not in _library_names():
                continue
            candidate = Path(distribution.locate_file(item))
            if candidate.is_file():
                candidates.append(str(candidate.resolve()))
    return tuple(dict.fromkeys(candidates))


def resolve_library_path(library_path=None):
    """Resolve an existing user runtime without installing or importing it."""

    explicit = library_path
    if explicit is None:
        explicit = os.environ.get(_ENVIRONMENT_VARIABLE)
    if explicit:
        return _path_candidate(explicit) or os.fspath(explicit)
    candidates = _distribution_candidates()
    if candidates:
        return candidates[0]
    for name in _library_names():
        candidate = shutil.which(name)
        if candidate:
            return str(Path(candidate).resolve())
    return _library_names()[0]


def _unload_library(library):
    handle = getattr(library, "_handle", 0)
    if not handle:
        return
    if sys.platform == "win32":
        _ctypes.FreeLibrary(handle)
    else:
        _ctypes.dlclose(handle)
    library._handle = 0


class _CublasLtLibrary:
    def __init__(self, candidate):
        self.candidate = candidate
        try:
            self.library = ctypes.CDLL(candidate)
        except OSError as exc:
            raise TaichiRuntimeError(
                f"could not load cuBLASLt library {candidate!r}: {exc}"
            ) from exc
        try:
            self.create = self.library.cublasLtCreate
            self.destroy = self.library.cublasLtDestroy
            self.get_version = self.library.cublasLtGetVersion
            self.status_string = getattr(self.library, "cublasLtGetStatusString", None)
            self.matmul_desc_create = self.library.cublasLtMatmulDescCreate
            self.matmul_desc_destroy = self.library.cublasLtMatmulDescDestroy
            self.matmul_desc_set_attribute = self.library.cublasLtMatmulDescSetAttribute
            self.layout_create = self.library.cublasLtMatrixLayoutCreate
            self.layout_destroy = self.library.cublasLtMatrixLayoutDestroy
            self.layout_set_attribute = self.library.cublasLtMatrixLayoutSetAttribute
            self.preference_create = self.library.cublasLtMatmulPreferenceCreate
            self.preference_destroy = self.library.cublasLtMatmulPreferenceDestroy
            self.preference_set_attribute = (
                self.library.cublasLtMatmulPreferenceSetAttribute
            )
            self.heuristic = self.library.cublasLtMatmulAlgoGetHeuristic
            self.matmul = self.library.cublasLtMatmul
        except AttributeError as exc:
            _unload_library(self.library)
            raise TaichiRuntimeError(
                "cuBLASLt library is missing the retained f32 matmul symbol slice"
            ) from exc
        self._configure_abi()
        self.version = int(self.get_version())

    def _configure_abi(self):
        self.create.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
        self.create.restype = ctypes.c_int
        self.destroy.argtypes = (ctypes.c_void_p,)
        self.destroy.restype = ctypes.c_int
        self.get_version.argtypes = ()
        self.get_version.restype = ctypes.c_size_t
        if self.status_string is not None:
            self.status_string.argtypes = (ctypes.c_int,)
            self.status_string.restype = ctypes.c_char_p
        self.matmul_desc_create.argtypes = (
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_int,
        )
        self.matmul_desc_create.restype = ctypes.c_int
        self.matmul_desc_destroy.argtypes = (ctypes.c_void_p,)
        self.matmul_desc_destroy.restype = ctypes.c_int
        self.matmul_desc_set_attribute.argtypes = (
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
        )
        self.matmul_desc_set_attribute.restype = ctypes.c_int
        self.layout_create.argtypes = (
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_int64,
        )
        self.layout_create.restype = ctypes.c_int
        self.layout_destroy.argtypes = (ctypes.c_void_p,)
        self.layout_destroy.restype = ctypes.c_int
        self.layout_set_attribute.argtypes = (
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
        )
        self.layout_set_attribute.restype = ctypes.c_int
        self.preference_create.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
        self.preference_create.restype = ctypes.c_int
        self.preference_destroy.argtypes = (ctypes.c_void_p,)
        self.preference_destroy.restype = ctypes.c_int
        self.preference_set_attribute.argtypes = (
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
        )
        self.preference_set_attribute.restype = ctypes.c_int
        self.heuristic.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(_HeuristicResult),
            ctypes.POINTER(ctypes.c_int),
        )
        self.heuristic.restype = ctypes.c_int
        self.matmul.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(_MatmulAlgo),
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
        )
        self.matmul.restype = ctypes.c_int

    def require(self, status, operation):
        status = int(status)
        if status == _CUBLAS_STATUS_SUCCESS:
            return
        detail = ""
        if self.status_string is not None:
            value = self.status_string(status)
            if value:
                detail = ": " + value.decode("utf-8", errors="replace")
        error = TaichiRuntimeError(
            f"cuBLASLt {operation} failed with status {status}{detail}"
        )
        error.vendor_status = status
        error.vendor_operation = operation
        raise error


_LIBRARY_LOCK = threading.RLock()
_PROCESS_LIBRARIES = {}


def _load_process_library(library_path=None):
    candidate = resolve_library_path(library_path)
    key = (
        os.path.normcase(os.path.abspath(candidate))
        if Path(candidate).is_file()
        else candidate
    )
    with _LIBRARY_LOCK:
        library = _PROCESS_LIBRARIES.get(key)
        if library is None:
            library = _CublasLtLibrary(candidate)
            _PROCESS_LIBRARIES[key] = library
        return library


def passive_status():
    """Report only already-retained libraries; never trigger discovery."""

    with _LIBRARY_LOCK:
        libraries = tuple(_PROCESS_LIBRARIES.values())
    latest = libraries[-1] if libraries else None
    return {
        "provider_id": "cublaslt",
        "library_loaded": latest is not None,
        "provider_abi": CUBLASLT_PROVIDER_ABI,
        "provider_version": None if latest is None else _version_string(latest.version),
        "native_facts": {
            "status_policy": "process_cache_only",
            "external_component_probed": False,
            "provider_enablement_changed": False,
            "provider_selection_changed": False,
            "library_candidate": None if latest is None else latest.candidate,
            "retained_library_count": len(libraries),
        },
    }


def probe_provider(library_path=None):
    """Transiently inspect symbols/version without selecting the provider."""

    native_facts = {
        "probe_policy": "transient_vendor_runtime_query",
        "provider_enablement_changed": False,
        "provider_selection_changed": False,
        "symbol_slice": "retained_f32_matmul",
    }
    result = {
        "provider_id": "cublaslt",
        "external_component_probed": False,
        "discovery": "missing",
        "unavailable_reason": "vendor_runtime_not_found",
        "provider_abi": CUBLASLT_PROVIDER_ABI,
        "provider_version": None,
        "last_error": None,
        "failure_scope": None,
        "native_facts": native_facts,
    }
    candidate = resolve_library_path(library_path)
    native_facts["library_candidate"] = candidate
    try:
        library = _CublasLtLibrary(candidate)
    except (OSError, TaichiRuntimeError) as exc:
        result.update(
            discovery="incompatible",
            unavailable_reason="vendor_runtime_probe_failed",
            last_error=str(exc) or type(exc).__name__,
            failure_scope="provider",
        )
        return result
    try:
        result.update(
            external_component_probed=True,
            discovery="available",
            unavailable_reason="none",
            provider_version=_version_string(library.version),
        )
        return result
    finally:
        _unload_library(library.library)


def _positive_dimension(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 < value <= 0x7FFFFFFF
    ):
        raise ValueError(f"cuBLASLt {name} must be in [1, INT_MAX]")
    return value


def _workspace_limit(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("cuBLASLt workspace_limit_bytes must be a nonnegative integer")
    return value


def _set_attribute(library, function, owner, attribute, value, operation):
    library.require(
        function(owner, attribute, ctypes.byref(value), ctypes.sizeof(value)),
        operation,
    )


def _device_pointer(program, value):
    return int(program.get_ndarray_data_ptr_as_int(value.arr))


class CublasLtProvider:
    """One cuBLASLt handle bound to one Taichi CUDA runtime generation."""

    def __init__(self, library_path=None):
        program = impl.get_runtime().prog
        if program is None or active_backend() != "cuda":
            raise TaichiRuntimeError(
                "CublasLtProvider requires an initialized Taichi CUDA runtime"
            )
        self._library = _load_process_library(library_path)
        self._runtime_prog = program
        self._runtime_generation = int(impl.runtime_generation())
        self._lock = threading.RLock()
        self._handle = ctypes.c_void_p()
        self._plans = weakref.WeakSet()
        self._library.require(
            self._library.create(ctypes.byref(self._handle)), "handle creation"
        )
        if not self._handle.value:
            raise TaichiRuntimeError("cuBLASLt returned a null provider handle")
        impl.get_runtime().register_runtime_object(self)

    @property
    def closed(self):
        return not bool(self._handle and self._handle.value)

    @property
    def version(self):
        return self._library.version

    def _validate_lifetime(self):
        if self.closed:
            raise TaichiRuntimeError("CublasLtProvider has been closed")
        validate_runtime_generation(
            self, "CublasLtProvider belongs to a previous Taichi runtime generation"
        )

    def plan(
        self,
        m,
        n,
        k,
        *,
        batch_count=1,
        transpose_a=False,
        transpose_b=False,
        alpha=1.0,
        beta=0.0,
        workspace_limit_bytes=32 << 20,
        a="a",
        b="b",
        output="output",
    ):
        with self._lock:
            self._validate_lifetime()
            plan = CublasLtMatmulPlan(
                self,
                m,
                n,
                k,
                batch_count=batch_count,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
                alpha=alpha,
                beta=beta,
                workspace_limit_bytes=workspace_limit_bytes,
                a=a,
                b=b,
                output=output,
            )
            self._plans.add(plan)
            return plan

    def close(self):
        with self._lock:
            if self.closed:
                return None
            if any(not plan.closed for plan in self._plans):
                raise TaichiRuntimeError(
                    "CublasLtProvider cannot close while matmul plans are live"
                )
            if runtime_generation_matches(self):
                self._runtime_prog.synchronize()
            handle = self._handle
            self._handle = None
            self._library.require(self._library.destroy(handle), "handle destruction")
        return None

    destroy = close

    def _invalidate_runtime(self):
        with self._lock:
            if self.closed:
                return
            self._runtime_prog.synchronize()
            for plan in tuple(self._plans):
                plan._close_native()
            handle = self._handle
            self._handle = None
            self._library.require(self._library.destroy(handle), "handle destruction")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class CublasLtMatmulPlan(BackendCommandRecording):
    """Retained row-major f32 single or strided-batched matmul plan."""

    # Live resource owners use identity, not the inherited recording fields:
    # otherwise WeakSet merges distinct descriptors with equal binding names.
    __eq__ = object.__eq__
    __hash__ = object.__hash__

    def __init__(
        self,
        provider,
        m,
        n,
        k,
        *,
        batch_count,
        transpose_a,
        transpose_b,
        alpha,
        beta,
        workspace_limit_bytes,
        a,
        b,
        output,
    ):
        if not isinstance(provider, CublasLtProvider):
            raise TypeError("provider must be a CublasLtProvider")
        self.m = _positive_dimension(m, "m")
        self.n = _positive_dimension(n, "n")
        self.k = _positive_dimension(k, "k")
        self.batch_count = _positive_dimension(batch_count, "batch_count")
        if not isinstance(transpose_a, bool) or not isinstance(transpose_b, bool):
            raise TypeError("cuBLASLt transpose flags must be bool")
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b
        if any(
            isinstance(value, bool) or not isinstance(value, numbers.Real)
            for value in (alpha, beta)
        ):
            raise TypeError("cuBLASLt alpha and beta must be real numbers")
        self.alpha = float(alpha)
        self.beta = float(beta)
        if not math.isfinite(self.alpha) or not math.isfinite(self.beta):
            raise ValueError("cuBLASLt alpha and beta must be finite")
        self.workspace_limit_bytes = _workspace_limit(workspace_limit_bytes)
        names = (a, b, output)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("cuBLASLt binding names must be nonempty strings")
        if len(set(names)) != 3:
            raise ValueError("cuBLASLt binding names must be unique")
        self.a, self.b, self.output = names
        super().__init__(
            backend="cuda",
            binding_names=names,
            command_count=1,
            queue="compute",
            stream_binding="runtime_ordered",
            barrier_policy="declared_effects",
            workspace_ownership="provider_generation",
            replay_mode="rerecord",
            no_host_readback=True,
        )
        self.provider = provider
        self._runtime_prog = provider._runtime_prog
        self._runtime_generation = provider._runtime_generation
        self._lock = threading.RLock()
        self._matmul_desc = None
        self._layouts = []
        self.workspace = None
        self._capture_leases = 0
        try:
            self._create_native_plan()
        except Exception:
            self._close_native()
            raise
        algo_bytes = ctypes.string_at(
            ctypes.byref(self._heuristic.algo), ctypes.sizeof(_MatmulAlgo)
        )
        identity = make_retained_plan_identity(
            "linalg.matmul.cublaslt_explicit",
            "cublaslt",
            "cuda",
            provider_scope={
                "provider_abi": CUBLASLT_PROVIDER_ABI,
                "provider_version": _version_string(provider.version),
                "provider_binary_identity": None,
                "library_candidate": provider._library.candidate,
            },
            problem_scope={
                "m": self.m,
                "n": self.n,
                "k": self.k,
                "batch_count": self.batch_count,
                "transpose_a": self.transpose_a,
                "transpose_b": self.transpose_b,
            },
            execution_scope={
                "algorithm_sha256": hashlib.sha256(algo_bytes).hexdigest(),
                "workspace_bytes": self.workspace_bytes,
                "workspace_limit_bytes": self.workspace_limit_bytes,
                "row_major": self._physical_layout == "row_major",
                "physical_layout": self._physical_layout,
                "epilogue": self._physical_epilogue,
                "alpha": self.alpha,
                "beta": self.beta,
            },
        )
        attach_retained_execution_contract(
            self,
            RetainedExecutionContract(
                identity=identity,
                cost_model=HardwareExecutionCostModel(
                    (
                        fixed_cost("provider_library_load", "process"),
                        fixed_cost("provider_handle", "runtime_generation"),
                        fixed_cost(
                            "descriptors_heuristic_and_workspace",
                            "provider_generation",
                        ),
                        fixed_cost("ctypes_dispatch", "invocation"),
                        fixed_cost("submission_registration", "invocation"),
                        scale_cost("matmul_execution", "batch_count", "m", "n", "k"),
                    )
                ),
                workspace_ownership="provider_generation",
                concurrency_policy="runtime_ordered",
                automatic_selection_policy="forbidden",
            ),
        )

    @property
    def closed(self):
        return self._matmul_desc is None

    @property
    def a_shape(self):
        matrix = (self.k, self.m) if self.transpose_a else (self.m, self.k)
        return matrix if self.batch_count == 1 else (self.batch_count, *matrix)

    @property
    def b_shape(self):
        matrix = (self.n, self.k) if self.transpose_b else (self.k, self.n)
        return matrix if self.batch_count == 1 else (self.batch_count, *matrix)

    @property
    def output_shape(self):
        matrix = (self.m, self.n)
        return matrix if self.batch_count == 1 else (self.batch_count, *matrix)

    def _create_layout(self, rows, columns, *, column_major=False):
        library = self.provider._library
        layout = ctypes.c_void_p()
        library.require(
            library.layout_create(
                ctypes.byref(layout),
                _CUDA_R_32F,
                rows,
                columns,
                rows if column_major else columns,
            ),
            "matrix layout creation",
        )
        try:
            order = ctypes.c_int(0 if column_major else _CUBLASLT_ORDER_ROW)
            _set_attribute(
                library,
                library.layout_set_attribute,
                layout,
                _MATRIX_LAYOUT_ORDER,
                order,
                "matrix storage order",
            )
            if self.batch_count > 1:
                count = ctypes.c_int(self.batch_count)
                stride = ctypes.c_int64(rows * columns)
                _set_attribute(
                    library,
                    library.layout_set_attribute,
                    layout,
                    _MATRIX_LAYOUT_BATCH_COUNT,
                    count,
                    "layout batch count",
                )
                _set_attribute(
                    library,
                    library.layout_set_attribute,
                    layout,
                    _MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                    stride,
                    "layout batch stride",
                )
        except Exception:
            library.layout_destroy(layout)
            raise
        self._layouts.append(layout)
        return layout

    def _create_descriptors(self, *, layout="row_major", epilogue="identity"):
        if layout not in ("row_major", "transposed_column_major"):
            raise ValueError("Unknown cuBLASLt physical layout")
        if epilogue not in ("identity", "relu"):
            raise ValueError("Unknown cuBLASLt physical epilogue")
        self._physical_layout = layout
        self._physical_epilogue = epilogue
        column_major = layout == "transposed_column_major"
        self._capture_operand_order = (1, 0, 2) if column_major else (0, 1, 2)
        library = self.provider._library
        desc = ctypes.c_void_p()
        library.require(
            library.matmul_desc_create(
                ctypes.byref(desc), _CUBLAS_COMPUTE_32F, _CUDA_R_32F
            ),
            "matmul descriptor creation",
        )
        self._matmul_desc = desc
        transpose_a, transpose_b = self.transpose_a, self.transpose_b
        if column_major:
            transpose_a, transpose_b = transpose_b, transpose_a
        trans_a = ctypes.c_int(_CUBLAS_OP_T if transpose_a else _CUBLAS_OP_N)
        trans_b = ctypes.c_int(_CUBLAS_OP_T if transpose_b else _CUBLAS_OP_N)
        _set_attribute(
            library,
            library.matmul_desc_set_attribute,
            desc,
            _MATMUL_DESC_TRANSA,
            trans_a,
            "transpose-A attribute",
        )
        _set_attribute(
            library,
            library.matmul_desc_set_attribute,
            desc,
            _MATMUL_DESC_TRANSB,
            trans_b,
            "transpose-B attribute",
        )
        if epilogue == "relu":
            _set_attribute(
                library,
                library.matmul_desc_set_attribute,
                desc,
                _MATMUL_DESC_EPILOGUE,
                ctypes.c_uint32(_EPILOGUE_RELU),
                "ReLU epilogue",
            )
        shapes = (self.a_shape[-2:], self.b_shape[-2:], self.output_shape[-2:])
        if column_major:
            shapes = tuple(shapes[i][::-1] for i in self._capture_operand_order)
        self._a_layout = self._create_layout(*shapes[0], column_major=column_major)
        self._b_layout = self._create_layout(*shapes[1], column_major=column_major)
        self._c_layout = self._create_layout(*shapes[2], column_major=column_major)
        self._d_layout = self._create_layout(*shapes[2], column_major=column_major)
        return desc

    def _create_native_plan(self):
        library = self.provider._library
        desc = self._create_descriptors()
        preference = ctypes.c_void_p()
        library.require(
            library.preference_create(ctypes.byref(preference)),
            "matmul preference creation",
        )
        try:
            workspace_limit = ctypes.c_size_t(self.workspace_limit_bytes)
            _set_attribute(
                library,
                library.preference_set_attribute,
                preference,
                _MATMUL_PREF_MAX_WORKSPACE_BYTES,
                workspace_limit,
                "workspace preference",
            )
            result = _HeuristicResult()
            returned = ctypes.c_int()
            library.require(
                library.heuristic(
                    self.provider._handle,
                    desc,
                    self._a_layout,
                    self._b_layout,
                    self._c_layout,
                    self._d_layout,
                    preference,
                    1,
                    ctypes.byref(result),
                    ctypes.byref(returned),
                ),
                "algorithm heuristic",
            )
            if returned.value != 1 or result.state != _CUBLAS_STATUS_SUCCESS:
                raise TaichiRuntimeError(
                    "cuBLASLt found no compatible f32 matmul algorithm"
                )
            self._heuristic = result
            self.workspace_bytes = int(result.workspace_size)
            if self.workspace_bytes > self.workspace_limit_bytes:
                raise TaichiRuntimeError(
                    "cuBLASLt heuristic exceeded the configured workspace limit"
                )
            if self.workspace_bytes:
                self.workspace = impl.ndarray(u8, shape=self.workspace_bytes)
        finally:
            library.require(
                library.preference_destroy(preference),
                "matmul preference destruction",
            )

    @property
    def resource_effects(self):
        output_access = (
            GraphAccess.WRITE if self.beta == 0.0 else GraphAccess.READ_WRITE
        )
        effects = (
            ResourceEffect(self.a, GraphAccess.READ),
            ResourceEffect(self.b, GraphAccess.READ),
            ResourceEffect(self.output, output_access),
        )
        if self.workspace is not None:
            effects += (static_resource_effect(self.workspace, GraphAccess.READ_WRITE),)
        return effects

    @staticmethod
    def _validate_array(value, name, shape):
        if (
            not isinstance(value, Ndarray)
            or value.dtype != f32
            or tuple(value.element_shape) != ()
            or tuple(value.shape) != shape
        ):
            raise TaichiRuntimeError(
                f"cuBLASLt binding {name!r} must be a compact scalar f32 "
                f"ndarray with shape {shape}"
            )

    def execute(self, bindings):
        validate_exact_bindings(self, bindings, "CUDA cuBLASLt")
        validate_runtime_generation(
            self, "cuBLASLt plan belongs to a previous Taichi runtime generation"
        )
        if self.closed or active_backend() != "cuda":
            raise TaichiRuntimeError(
                "cuBLASLt matmul requires a live plan on the CUDA backend"
            )
        a = bindings[self.a]
        b = bindings[self.b]
        output = bindings[self.output]
        self._validate_array(a, self.a, self.a_shape)
        self._validate_array(b, self.b, self.b_shape)
        self._validate_array(output, self.output, self.output_shape)
        if output is a or output is b:
            raise TaichiRuntimeError("cuBLASLt output must not alias either input")

        runtime = impl.get_runtime()
        program = runtime.prog
        values = [a, b, output]
        if self.workspace is not None:
            values.append(self.workspace)
        with external_cuda_submission(program, values) as submission:
            a_pointer = _device_pointer(program, a)
            b_pointer = _device_pointer(program, b)
            output_pointer = _device_pointer(program, output)
            workspace_pointer = (
                0
                if self.workspace is None
                else _device_pointer(program, self.workspace)
            )
            alpha = ctypes.c_float(self.alpha)
            beta = ctypes.c_float(self.beta)
            with self.provider._lock, self._lock, hardware_provider_call("cublaslt"):
                status = submission.invoke(
                    self.provider._library.matmul,
                    self.provider._handle,
                    self._matmul_desc,
                    ctypes.byref(alpha),
                    a_pointer,
                    self._a_layout,
                    b_pointer,
                    self._b_layout,
                    ctypes.byref(beta),
                    output_pointer,
                    self._c_layout,
                    output_pointer,
                    self._d_layout,
                    ctypes.byref(self._heuristic.algo),
                    workspace_pointer,
                    self.workspace_bytes,
                    None,
                )
                self.provider._library.require(status, "matmul execution")
            return output

    def run(self, **bindings):
        return self.execute(bindings)

    def _capture(self, *, workspace="__cublaslt_workspace"):
        from taichi_forge.hardware._cublaslt_capture import CublasLtCaptureRecording

        return CublasLtCaptureRecording(self, workspace=workspace)

    def _as_graph_native_node(self):
        return native_recording_node(
            self,
            lifetime_leases=lambda item: (
                item,
                item.provider,
                *((item.workspace,) if item.workspace is not None else ()),
            ),
            debug_info=lambda item: {
                "kind": "cuda_cublaslt_matmul_f32",
                "shape": (item.batch_count, item.m, item.n, item.k),
                "transpose_a": item.transpose_a,
                "transpose_b": item.transpose_b,
                "workspace_bytes": item.workspace_bytes,
                "provider_version": _version_string(item.provider.version),
            },
        )

    def _close_native(self):
        if self.closed:
            return
        library = self.provider._library
        desc = self._matmul_desc
        layouts = tuple(reversed(self._layouts))
        self._matmul_desc = None
        self._layouts = []
        for layout in layouts:
            library.require(library.layout_destroy(layout), "matrix layout destruction")
        library.require(
            library.matmul_desc_destroy(desc), "matmul descriptor destruction"
        )
        self.workspace = None

    def close(self):
        with self.provider._lock, self._lock:
            if self.closed:
                return None
            if self._capture_leases:
                raise TaichiRuntimeError(
                    "cuBLASLt plan cannot close while Graph capture leases are live"
                )
            if runtime_generation_matches(self):
                self._runtime_prog.synchronize()
            self._close_native()
        return None

    destroy = close

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


__all__ = (
    "CUBLASLT_PROVIDER_ABI",
    "CublasLtMatmulPlan",
    "CublasLtProvider",
    "passive_status",
    "probe_provider",
    "resolve_library_path",
)
