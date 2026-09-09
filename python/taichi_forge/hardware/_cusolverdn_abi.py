"""Cold bindings to the user-installed cuSOLVER dense generic C API.

No Toolkit headers, vendor binary, compiler or runtime import is added to the
Forge wheel. This module does not own Graph or application solver semantics.
"""

from contextlib import contextmanager
import _ctypes
import ctypes as ct
import importlib.metadata
import os
from pathlib import Path
import threading

from taichi_forge.lang.exception import TaichiRuntimeError


ABI = "cusolverdn-generic-cholesky-v1"
_LIBRARIES = {}
_LOCK = threading.RLock()


def resolve_library_path(library_path=None):
    explicit = library_path or os.environ.get("TI_CUSOLVERDN_LIBRARY_PATH")
    names = (
        ("cusolver64_12.dll", "cusolver64_11.dll")
        if os.name == "nt"
        else ("libcusolver.so.12", "libcusolver.so.11", "libcusolver.so")
    )
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_dir():
            for parent in (
                path,
                path / "bin",
                path / "bin/x64",
                path / "lib",
                path / "lib64",
            ):
                for name in names:
                    if (parent / name).is_file():
                        return str((parent / name).resolve())
        return str(path.resolve())
    for package in ("nvidia-cusolver-cu13", "nvidia-cusolver-cu12"):
        try:
            distribution = importlib.metadata.distribution(package)
        except importlib.metadata.PackageNotFoundError:
            continue
        for item in distribution.files or ():
            if Path(item).name in names:
                return str(Path(distribution.locate_file(item)).resolve())
    return names[0]


def check(result, operation):
    if result:
        raise TaichiRuntimeError(f"{operation} failed with status {int(result)}")


def _function(library, name, arguments):
    function = getattr(library, name)
    function.restype = ct.c_int
    function.argtypes = arguments
    return function


def _unload(library):
    if library._handle:
        (_ctypes.FreeLibrary if os.name == "nt" else _ctypes.dlclose)(library._handle)
        library._handle = 0


class DenseLibrary:
    def __init__(self, path):
        self.library = ct.CDLL(path)
        self.path = path
        try:
            p, i, n, z = ct.c_void_p, ct.c_int, ct.c_int64, ct.c_size_t
            self.create = _function(self.library, "cusolverDnCreate", [ct.POINTER(p)])
            self.destroy = _function(self.library, "cusolverDnDestroy", [p])
            self.set_stream = _function(self.library, "cusolverDnSetStream", [p, p])
            self.create_params = _function(
                self.library, "cusolverDnCreateParams", [ct.POINTER(p)]
            )
            self.destroy_params = _function(
                self.library, "cusolverDnDestroyParams", [p]
            )
            self.buffer_size = _function(
                self.library,
                "cusolverDnXpotrf_bufferSize",
                [p, p, i, n, i, p, n, i, ct.POINTER(z), ct.POINTER(z)],
            )
            self.factor = _function(
                self.library,
                "cusolverDnXpotrf",
                [p, p, i, n, i, p, n, i, p, z, p, z, p],
            )
            self.solve = _function(
                self.library, "cusolverDnXpotrs", [p, p, i, n, n, i, p, n, i, p, n, p]
            )
            prop = _function(self.library, "cusolverGetProperty", [i, ct.POINTER(i)])
            version = []
            for key in range(3):
                value = i()
                check(prop(key, ct.byref(value)), "cuSOLVER version query")
                version.append(value.value)
            self.version = ".".join(map(str, version))
            if os.name == "nt":
                filename = ct.WinDLL("kernel32", use_last_error=True).GetModuleFileNameW
                filename.argtypes = [p, ct.c_wchar_p, ct.c_uint]
                filename.restype = ct.c_uint
                buffer = ct.create_unicode_buffer(32768)
                if not filename(self.library._handle, buffer, len(buffer)):
                    raise ct.WinError(ct.get_last_error())
                self.path = str(Path(buffer.value).resolve())
        except Exception:
            _unload(self.library)
            raise


def load_library(path=None):
    candidate = resolve_library_path(path)
    with _LOCK:
        if candidate not in _LIBRARIES:
            _LIBRARIES[candidate] = DenseLibrary(candidate)
        return _LIBRARIES[candidate]


def passive_status():
    with _LOCK:
        libraries = tuple(_LIBRARIES.values())
    return {
        "provider_id": "cusolverdn",
        "library_loaded": bool(libraries),
        "provider_abi": ABI,
        "provider_version": libraries[-1].version if libraries else None,
        "native_facts": {
            "status_policy": "passive_existing_loader",
            "external_component_probed": False,
            "provider_enablement_changed": False,
            "provider_selection_changed": False,
        },
    }


def probe_provider(library_path=None):
    candidate = resolve_library_path(library_path)
    facts = {
        "probe_policy": "transient_vendor_runtime_query",
        "provider_enablement_changed": False,
        "provider_selection_changed": False,
        "execution_qualified": False,
        "library_candidate": candidate,
    }
    result = {
        "provider_id": "cusolverdn",
        "external_component_probed": False,
        "provider_abi": ABI,
        "provider_version": None,
        "native_facts": facts,
    }
    try:
        library = DenseLibrary(candidate)
    except (OSError, AttributeError, TaichiRuntimeError) as exc:
        result.update(
            discovery="incompatible",
            unavailable_reason="vendor_runtime_probe_failed",
            last_error=str(exc) or type(exc).__name__,
            failure_scope="provider",
        )
        return result
    try:
        facts["library_candidate"] = library.path
        result.update(
            external_component_probed=True,
            discovery="available",
            unavailable_reason="none",
            provider_version=library.version,
        )
        return result
    finally:
        _unload(library.library)


class DriverBinding:
    """A context bound once to owned storage; push/pop is activation, not probing."""

    def __init__(self, pointer):
        self.library = (
            ct.WinDLL("nvcuda.dll") if os.name == "nt" else ct.CDLL("libcuda.so.1")
        )
        p = ct.c_void_p
        self.push = _function(self.library, "cuCtxPushCurrent_v2", [p])
        self.pop = _function(self.library, "cuCtxPopCurrent_v2", [ct.POINTER(p)])
        self.copy = _function(
            self.library,
            "cuMemcpyDtoDAsync_v2",
            [ct.c_uint64, ct.c_uint64, ct.c_size_t, p],
        )
        self.clear = _function(
            self.library, "cuMemsetD32Async", [ct.c_uint64, ct.c_uint, ct.c_size_t, p]
        )
        query = _function(
            self.library, "cuPointerGetAttribute", [p, ct.c_int, ct.c_uint64]
        )
        retain = _function(
            self.library, "cuDevicePrimaryCtxRetain", [ct.POINTER(p), ct.c_int]
        )
        self.release = _function(
            self.library, "cuDevicePrimaryCtxRelease_v2", [ct.c_int]
        )
        self.device = ct.c_int()
        check(query(ct.byref(self.device), 9, pointer), "CUDA storage device query")
        self.context = p()
        # Forge uses the primary context and legacy default stream. VMM-backed
        # allocations have no allocation context attribute; resolve their device
        # once and retain that primary context rather than pushing a null handle.
        check(
            retain(ct.byref(self.context), self.device), "CUDA primary context retain"
        )

    def close(self):
        if self.context:
            check(self.release(self.device), "CUDA primary context release")
            self.context = None

    @contextmanager
    def activate(self):
        check(self.push(self.context), "CUDA context activation")
        try:
            yield
        finally:
            previous = ct.c_void_p()
            check(self.pop(ct.byref(previous)), "CUDA context restoration")
