"""Shared loader for Forge-owned optional vendor runtime adapters."""

from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import sys
import threading


PROVIDER_ABI_VERSION = 2
EXECUTION_ABI_VERSION = 1

_SUCCESS = 0
_RUNTIME_UNAVAILABLE = 3
_FEATURE_VERSION_QUERY = 1 << 0
_FEATURE_REQUIRED_SYMBOL_AUDIT = 1 << 1
_FEATURE_TRANSIENT_PROBE = 1 << 2
_FEATURE_EXECUTION_API = 1 << 3
_REQUIRED_FEATURES = _FEATURE_VERSION_QUERY | _FEATURE_REQUIRED_SYMBOL_AUDIT | _FEATURE_TRANSIENT_PROBE
_ACTIVE_RUNTIME_LOCK = threading.RLock()
_ACTIVE_RUNTIMES = {}


@dataclass(frozen=True)
class BundledRuntimeProviderDefinition:
    provider_id: str
    provider_name: str
    adapter_stem: str
    query_symbol: str
    provider_abi_name: str
    environment_variable: str
    library_names: tuple[str, ...]
    package_distributions: tuple[str, ...]
    supported_version_family: str


class _ProviderInfo(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("provider_abi_version", ctypes.c_uint32),
        ("features", ctypes.c_uint64),
        ("required_symbol_count", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("provider_id", ctypes.c_char_p),
        ("provider_name", ctypes.c_char_p),
        ("supported_version_family", ctypes.c_char_p),
        ("build_identity", ctypes.c_char_p),
    ]


class _RuntimeInfo(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version_major", ctypes.c_uint32),
        ("version_minor", ctypes.c_uint32),
        ("version_patch", ctypes.c_uint32),
        ("cuda_runtime_version", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("library_path", ctypes.c_char_p),
        ("build_version", ctypes.c_char_p),
    ]


_ProbeRuntime = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(_RuntimeInfo))
_CreateRuntime = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(_RuntimeInfo),
)
_DestroyRuntime = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
_QueryExecutionApi = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.c_size_t,
    ctypes.c_void_p,
)
_GetLastError = ctypes.CFUNCTYPE(ctypes.c_size_t, ctypes.POINTER(ctypes.c_char), ctypes.c_size_t)


class _ProviderApi(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("provider_abi_version", ctypes.c_uint32),
        ("info", _ProviderInfo),
        ("probe_runtime", _ProbeRuntime),
        ("create_runtime", _CreateRuntime),
        ("destroy_runtime", _DestroyRuntime),
        ("query_execution_api", _QueryExecutionApi),
        ("get_last_error", _GetLastError),
    ]


@dataclass(frozen=True)
class _LoadedApi:
    library: object
    path: str
    api: _ProviderApi


class _ProviderRuntimeError(RuntimeError):
    def __init__(self, result, message):
        super().__init__(message)
        self.result = int(result)


def _runtime_registry_identity(runtime_info):
    return (
        runtime_info["library_path"],
        int(runtime_info["version_major"]),
        int(runtime_info["version_minor"]),
        int(runtime_info["version_patch"]),
        runtime_info["build_version"],
    )


def _retain_runtime(definition, runtime_info):
    identity = _runtime_registry_identity(runtime_info)
    with _ACTIVE_RUNTIME_LOCK:
        runtimes = _ACTIVE_RUNTIMES.setdefault(definition.provider_id, {})
        runtimes[identity] = runtimes.get(identity, 0) + 1


def _release_runtime(definition, runtime_info):
    identity = _runtime_registry_identity(runtime_info)
    with _ACTIVE_RUNTIME_LOCK:
        runtimes = _ACTIVE_RUNTIMES.get(definition.provider_id)
        if not runtimes or runtimes.get(identity, 0) <= 0:
            raise RuntimeError(f"{definition.provider_name} retained-runtime registry underflow")
        if runtimes[identity] == 1:
            del runtimes[identity]
        else:
            runtimes[identity] -= 1
        if not runtimes:
            del _ACTIVE_RUNTIMES[definition.provider_id]


def _active_runtime_snapshot(provider_id):
    with _ACTIVE_RUNTIME_LOCK:
        runtimes = dict(_ACTIVE_RUNTIMES.get(provider_id, {}))
    identities = tuple(
        {
            "library_path": identity[0],
            "version": f"{identity[1]}.{identity[2]}.{identity[3]}",
            "build_version": identity[4],
            "lease_count": count,
        }
        for identity, count in sorted(runtimes.items())
    )
    return sum(runtimes.values()), identities


class BundledRuntime:
    """Retained adapter/vendor runtime pair with deterministic destruction."""

    def __init__(self, definition, loaded, handle, runtime_info):
        self.definition = definition
        self.loaded = loaded
        self._handle = handle
        self.runtime_info = dict(runtime_info)
        _retain_runtime(definition, self.runtime_info)

    @property
    def closed(self):
        return self._handle is None

    @property
    def handle(self):
        if self._handle is None:
            raise RuntimeError(f"{self.definition.provider_name} runtime has been closed")
        return self._handle

    def query_execution_api(self, api_type, *, version=EXECUTION_ABI_VERSION):
        output = api_type()
        result = int(
            self.loaded.api.query_execution_api(
                self.handle,
                version,
                ctypes.sizeof(api_type),
                ctypes.byref(output),
            )
        )
        if result != _SUCCESS:
            raise _ProviderRuntimeError(result, _provider_error(self.loaded.api, self.definition))
        if output.struct_size < ctypes.sizeof(api_type):
            raise RuntimeError(f"{self.definition.provider_name} returned a truncated execution API")
        if output.execution_abi_version != version:
            raise RuntimeError(f"{self.definition.provider_name} returned a mismatched execution ABI")
        return output

    def check_result(self, result):
        result = int(result)
        if result != _SUCCESS:
            raise _ProviderRuntimeError(result, _provider_error(self.loaded.api, self.definition))
        return None

    def close(self):
        if self._handle is None:
            return None
        handle = self._handle
        result = int(self.loaded.api.destroy_runtime(handle))
        if result != _SUCCESS:
            raise _ProviderRuntimeError(result, _provider_error(self.loaded.api, self.definition))
        self._handle = None
        _release_runtime(self.definition, self.runtime_info)
        return None

    destroy = close

    def __enter__(self):
        self.handle
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _decode(value):
    return "" if not value else value.decode("utf-8", errors="replace")


def _adapter_filename(definition):
    if os.name == "nt":
        return f"{definition.adapter_stem}.dll"
    return f"lib{definition.adapter_stem}.so"


def _runtime_package_roots():
    roots = []
    spec = importlib.util.find_spec("taichi_forge_runtime")
    if spec is not None and spec.submodule_search_locations is not None:
        roots.extend(Path(path) for path in spec.submodule_search_locations)
    roots.append(Path(__file__).resolve().parents[1])
    return tuple(dict.fromkeys(roots))


def _bundled_provider_candidates(definition):
    candidates = []
    for root in _runtime_package_roots():
        candidate = root / "_lib" / "hardware_providers" / _adapter_filename(definition)
        if candidate.is_file():
            candidates.append(str(candidate.resolve()))
    return tuple(dict.fromkeys(candidates))


def _path_candidate(value, library_names):
    if value is None:
        return None
    path = Path(os.fspath(value)).expanduser()
    if path.is_file():
        return path.resolve()
    if path.is_dir():
        for relative in ((), ("bin",), ("lib",), ("lib64",)):
            for name in library_names:
                candidate = path.joinpath(*relative, name)
                if candidate.is_file():
                    return candidate.resolve()
    return None


def _distribution_library_candidates(definition):
    candidates = []
    for distribution_name in definition.package_distributions:
        try:
            distribution = importlib.metadata.distribution(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            continue
        for item in distribution.files or ():
            if Path(item).name not in definition.library_names:
                continue
            candidate = Path(distribution.locate_file(item))
            if candidate.is_file():
                candidates.append(candidate.resolve())
    return tuple(dict.fromkeys(candidates))


def resolve_library_path(definition, library_path=None):
    """Resolve an existing user runtime without installing or importing it."""

    explicit = library_path
    if explicit is None:
        explicit = os.environ.get(definition.environment_variable)
    if explicit:
        candidate = _path_candidate(explicit, definition.library_names)
        return str(candidate) if candidate is not None else os.fspath(explicit)

    candidates = _distribution_library_candidates(definition)
    if candidates:
        return str(candidates[0])
    return ""


@lru_cache(maxsize=32)
def _resolved_sha256(candidate, size, mtime_ns):
    del size, mtime_ns
    digest = hashlib.sha256()
    with Path(candidate).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _binary_sha256(path):
    candidate = Path(path)
    if not candidate.is_file():
        return None
    candidate = candidate.resolve()
    stat = candidate.stat()
    return _resolved_sha256(str(candidate), int(stat.st_size), int(stat.st_mtime_ns))


@contextmanager
def _vendor_dll_directories(definition, runtime_path):
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        yield
        return
    directories = []
    candidate = _path_candidate(runtime_path, definition.library_names)
    if candidate is not None:
        directories.append(candidate.parent)
    for package_candidate in _distribution_library_candidates(definition):
        directories.append(package_candidate.parent)
    nvidia_spec = importlib.util.find_spec("nvidia")
    if nvidia_spec is not None and nvidia_spec.submodule_search_locations is not None:
        for root_value in nvidia_spec.submodule_search_locations:
            root = Path(root_value)
            for component in (
                "cuda_runtime",
                "cublas",
                "cusparse",
                "cusparselt",
                "cutensor",
            ):
                directory = root / component / "bin"
                if directory.is_dir():
                    directories.append(directory.resolve())
    handles = []
    try:
        for directory in dict.fromkeys(directories):
            handles.append(os.add_dll_directory(str(directory)))
        yield
    finally:
        for handle in reversed(handles):
            handle.close()


def _load_library(path):
    return ctypes.CDLL(path)


def _provider_error(api, definition):
    required = int(api.get_last_error(None, 0))
    if required <= 1:
        return f"optional {definition.provider_name} adapter call failed"
    buffer = ctypes.create_string_buffer(required)
    api.get_last_error(buffer, required)
    return buffer.value.decode("utf-8", errors="replace")


def _check_api(api, definition):
    if api.struct_size < ctypes.sizeof(_ProviderApi):
        raise RuntimeError(f"{definition.provider_name} returned a truncated API")
    if api.provider_abi_version != PROVIDER_ABI_VERSION:
        raise RuntimeError(f"{definition.provider_name} returned a mismatched ABI")
    if api.info.struct_size < ctypes.sizeof(_ProviderInfo):
        raise RuntimeError(f"{definition.provider_name} returned truncated identity facts")
    if api.info.provider_abi_version != PROVIDER_ABI_VERSION:
        raise RuntimeError(f"{definition.provider_name} identity uses a mismatched ABI")
    if _decode(api.info.provider_id) != definition.provider_id:
        raise RuntimeError(f"{definition.provider_name} adapter identity mismatch")
    if _decode(api.info.supported_version_family) != definition.supported_version_family:
        raise RuntimeError(f"{definition.provider_name} adapter version-family mismatch")
    features = int(api.info.features)
    if features & _REQUIRED_FEATURES != _REQUIRED_FEATURES:
        raise RuntimeError(f"{definition.provider_name} adapter is missing probe features")
    if not features & _FEATURE_EXECUTION_API:
        raise RuntimeError(f"{definition.provider_name} adapter is missing execution support")
    if api.info.required_symbol_count <= 0:
        raise RuntimeError(f"{definition.provider_name} adapter does not audit execution symbols")
    for name in (
        "probe_runtime",
        "create_runtime",
        "destroy_runtime",
        "query_execution_api",
        "get_last_error",
    ):
        if not bool(getattr(api, name)):
            raise RuntimeError(f"{definition.provider_name} adapter is missing ABI entry {name}")


def _query_provider(definition, path):
    resolved = str(Path(path).expanduser().resolve())
    library = _load_library(resolved)
    try:
        query = getattr(library, definition.query_symbol)
    except AttributeError as exc:
        raise RuntimeError(f"{definition.provider_name} provider query symbol is missing") from exc
    query.argtypes = [
        ctypes.c_uint32,
        ctypes.c_size_t,
        ctypes.POINTER(_ProviderApi),
    ]
    query.restype = ctypes.c_int
    api = _ProviderApi()
    result = int(query(PROVIDER_ABI_VERSION, ctypes.sizeof(_ProviderApi), ctypes.byref(api)))
    if result != _SUCCESS:
        message = (
            _provider_error(api, definition)
            if bool(api.get_last_error)
            else f"provider query failed with result {result}"
        )
        raise RuntimeError(message)
    _check_api(api, definition)
    return _LoadedApi(library, resolved, api)


def _probe_runtime(definition, loaded, runtime_path):
    info = _RuntimeInfo()
    info.struct_size = ctypes.sizeof(_RuntimeInfo)
    argument = None if not runtime_path else os.fsencode(runtime_path)
    result = int(loaded.api.probe_runtime(argument, ctypes.byref(info)))
    if result != _SUCCESS:
        raise _ProviderRuntimeError(result, _provider_error(loaded.api, definition))
    return {
        "version_major": int(info.version_major),
        "version_minor": int(info.version_minor),
        "version_patch": int(info.version_patch),
        "cuda_runtime_version": int(info.cuda_runtime_version),
        "library_path": _decode(info.library_path) or runtime_path or "system_default",
        "build_version": _decode(info.build_version),
    }


def _create_runtime(definition, loaded, runtime_path):
    info = _RuntimeInfo()
    info.struct_size = ctypes.sizeof(_RuntimeInfo)
    handle = ctypes.c_void_p()
    argument = None if not runtime_path else os.fsencode(runtime_path)
    result = int(loaded.api.create_runtime(argument, ctypes.byref(handle), ctypes.byref(info)))
    if result != _SUCCESS or not handle.value:
        raise _ProviderRuntimeError(result, _provider_error(loaded.api, definition))
    runtime = {
        "version_major": int(info.version_major),
        "version_minor": int(info.version_minor),
        "version_patch": int(info.version_patch),
        "cuda_runtime_version": int(info.cuda_runtime_version),
        "library_path": _decode(info.library_path) or runtime_path or "system_default",
        "build_version": _decode(info.build_version),
    }
    return BundledRuntime(definition, loaded, handle, runtime)


def _format_version(runtime):
    return f"{runtime['version_major']}.{runtime['version_minor']}.{runtime['version_patch']}"


def probe_provider(definition, library_path=None):
    """Probe one adapter and vendor runtime without retaining either handle."""

    native_facts = {
        "probe_policy": "transient_adapter_and_vendor_runtime_query",
        "provider_enablement_changed": False,
        "provider_selection_changed": False,
        "execution_qualified": False,
        "execution_api_available": False,
        "supported_version_family": definition.supported_version_family,
    }
    result = {
        "provider_id": definition.provider_id,
        "external_component_probed": False,
        "discovery": "missing",
        "unavailable_reason": "bundled_provider_adapter_not_installed",
        "provider_abi": definition.provider_abi_name,
        "provider_version": None,
        "last_error": None,
        "failure_scope": None,
        "native_facts": native_facts,
    }
    try:
        runtime_path = resolve_library_path(definition, library_path)
        candidates = _bundled_provider_candidates(definition)
    except (OSError, TypeError, ValueError) as exc:
        result.update(
            discovery="incompatible",
            unavailable_reason="library_path_resolution_failed",
            last_error=str(exc) or type(exc).__name__,
            failure_scope="provider",
        )
        return result
    native_facts.update(
        provider_source="forge_runtime_wheel",
        vendor_library_candidate=runtime_path or "system_default",
        provider_candidates=tuple(candidates),
    )
    if not candidates:
        return result

    result["external_component_probed"] = True
    failures = []
    failure_results = []
    with _vendor_dll_directories(definition, runtime_path):
        for candidate in candidates:
            try:
                loaded = _query_provider(definition, candidate)
                runtime = _probe_runtime(definition, loaded, runtime_path)
                adapter_hash = _binary_sha256(loaded.path)
                if adapter_hash is None:
                    raise RuntimeError("Forge adapter identity is unavailable")
            except (
                AttributeError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                failures.append(f"{candidate}: {str(exc) or type(exc).__name__}")
                failure_results.append(getattr(exc, "result", None))
                continue
            info = loaded.api.info
            result.update(
                discovery="available",
                unavailable_reason="none",
                provider_version=_format_version(runtime),
            )
            native_facts.update(
                library_candidate=loaded.path,
                provider_adapter_binary_sha256=adapter_hash,
                library_loaded_transiently=True,
                probe_created_execution_resource=False,
                execution_resource_created=False,
                execution_api_available=True,
                vendor_runtime_abi_compatible=True,
                vendor_library_resolved=runtime["library_path"],
                provider_name=_decode(info.provider_name),
                build_identity=_decode(info.build_identity),
                feature_bits=int(info.features),
                required_symbol_count=int(info.required_symbol_count),
                cuda_runtime_version=(runtime["cuda_runtime_version"] or None),
                vendor_build_version=runtime["build_version"] or None,
            )
            if failures:
                native_facts["rejected_candidates"] = tuple(failures)
            return result

    runtime_missing = bool(failure_results) and all(code == _RUNTIME_UNAVAILABLE for code in failure_results)
    result.update(
        discovery="missing" if runtime_missing else "incompatible",
        unavailable_reason=(
            "external_library_not_found" if runtime_missing else f"no_compatible_{definition.provider_id}_provider"
        ),
        last_error="; ".join(failures),
        failure_scope="provider",
    )
    native_facts["library_loaded_transiently"] = False
    return result


def open_runtime(definition, library_path=None):
    """Load and retain one compatible adapter plus user-managed vendor runtime."""

    runtime_path = resolve_library_path(definition, library_path)
    candidates = _bundled_provider_candidates(definition)
    if not candidates:
        raise RuntimeError(f"Forge runtime wheel does not contain a {definition.provider_name} adapter")
    failures = []
    with _vendor_dll_directories(definition, runtime_path):
        for candidate in candidates:
            try:
                loaded = _query_provider(definition, candidate)
                return _create_runtime(definition, loaded, runtime_path)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                failures.append(f"{candidate}: {str(exc) or type(exc).__name__}")
    raise RuntimeError(f"no compatible {definition.provider_name} runtime: " + "; ".join(failures))


def passive_status(definition):
    """Report the optional-provider contract without loading either shared library."""

    active_runtime_count, identities = _active_runtime_snapshot(definition.provider_id)
    versions = tuple(dict.fromkeys(identity["version"] for identity in identities))
    return {
        "provider_id": definition.provider_id,
        "library_loaded": bool(active_runtime_count),
        "provider_abi": definition.provider_abi_name,
        "provider_version": versions[0] if len(versions) == 1 else None,
        "native_facts": {
            "status_policy": "passive_retained_runtime_registry",
            "external_component_probed": False,
            "provider_enablement_changed": False,
            "provider_selection_changed": False,
            "execution_api_available": bool(active_runtime_count),
            "active_runtime_count": active_runtime_count,
            "loaded_runtime_identities": identities,
        },
    }


__all__ = (
    "BundledRuntimeProviderDefinition",
    "BundledRuntime",
    "EXECUTION_ABI_VERSION",
    "PROVIDER_ABI_VERSION",
    "open_runtime",
    "passive_status",
    "probe_provider",
    "resolve_library_path",
)
