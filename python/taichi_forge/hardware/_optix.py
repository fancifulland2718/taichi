"""Optional OptiX runtime behind bundled Forge C-ABI adapters."""

import ctypes
from dataclasses import dataclass
from functools import partial
import importlib.util
import os
from pathlib import Path
from types import MappingProxyType
import weakref

from taichi_forge._hardware_telemetry import (
    hardware_failure_phase,
    instrument_hardware_recording,
)
from taichi_forge.graph._ir import GraphAccess, ResourceEffect
from taichi_forge.graph._native import BackendCommandRecording
from taichi_forge.hardware._memory import HardwareMemoryComponent, make_memory_report
from taichi_forge.hardware._native_adapter import (
    native_recording_node,
    runtime_generation_matches,
    static_resource_effect,
    validate_exact_bindings,
    validate_runtime_generation,
)
from taichi_forge.hardware._runtime import active_backend
from taichi_forge.lang import impl
from taichi_forge.lang._ndarray import Ndarray
from taichi_forge.lang._storage_view import describe_storage
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.types.primitive_types import f32, i32, u32


PROVIDER_ABI_VERSION = 1
PROVIDER_ABI_NAME = "taichi-forge-optix-provider-c-abi1"
PROVIDER_QUERY_SYMBOL = "taichi_forge_optix_provider_query"
SUPPORTED_OPTIX_ABIS = (93, 105, 118)

_SUCCESS = 0
_OPTIX_UNAVAILABLE = 4
_REQUIRED_FEATURES = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
_TYPED_HITS = 1 << 6
_WORD_ALIGNED_QUERY_STORAGE = 1 << 7
_loaded_providers = weakref.WeakSet()


def _provider_filename(optix_abi):
    stem = f"taichi_forge_optix_provider_abi1_optix{optix_abi}"
    if os.name == "nt":
        return f"{stem}.dll"
    return f"lib{stem}.so"


def _runtime_package_roots():
    roots = []
    spec = importlib.util.find_spec("taichi_forge_runtime")
    if spec is not None and spec.submodule_search_locations is not None:
        roots.extend(Path(path) for path in spec.submodule_search_locations)
    roots.append(Path(__file__).resolve().parents[1])
    return roots


def _bundled_provider_candidates():
    candidates = []
    seen = set()
    for root in _runtime_package_roots():
        directory = root / "_lib" / "hardware_providers"
        for optix_abi in reversed(SUPPORTED_OPTIX_ABIS):
            candidate = directory / _provider_filename(optix_abi)
            key = os.path.normcase(str(candidate))
            if key not in seen and candidate.is_file():
                candidates.append(str(candidate))
                seen.add(key)
    return tuple(candidates)


class _ProviderInfo(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("provider_abi_version", ctypes.c_uint32),
        ("optix_abi_version", ctypes.c_uint32),
        ("optix_version", ctypes.c_uint32),
        ("features", ctypes.c_uint64),
        ("provider_name", ctypes.c_char_p),
        ("build_identity", ctypes.c_char_p),
    ]


class _ContextDesc(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("device_ordinal", ctypes.c_uint32),
        ("cuda_context", ctypes.c_uint64),
        ("validation_mode", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("runtime_library_path", ctypes.c_char_p),
    ]


class _TriangleSceneDesc(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("vertex_count", ctypes.c_uint32),
        ("triangle_count", ctypes.c_uint32),
        ("allow_update", ctypes.c_uint32),
        ("vertices", ctypes.c_uint64),
        ("indices", ctypes.c_uint64),
        ("cuda_stream", ctypes.c_uint64),
    ]


class _TraceDesc(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("ray_count", ctypes.c_uint32),
        ("rays", ctypes.c_uint64),
        ("hits", ctypes.c_uint64),
        ("cuda_stream", ctypes.c_uint64),
    ]


class _SceneMemory(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("gas_bytes", ctypes.c_uint64),
        ("ias_bytes", ctypes.c_uint64),
        ("build_update_scratch_bytes", ctypes.c_uint64),
        ("instance_bytes", ctypes.c_uint64),
        ("launch_params_bytes", ctypes.c_uint64),
        ("shared_pipeline_sbt_bytes", ctypes.c_uint64),
    ]


class _TypedTraceDesc(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("ray_count", ctypes.c_uint32),
        ("rays", ctypes.c_uint64),
        ("hits", ctypes.c_uint64),
        ("hit_indices", ctypes.c_uint64),
        ("cuda_stream", ctypes.c_uint64),
    ]


_ProbeRuntime = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p)
_CreateContext = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.POINTER(_ContextDesc), ctypes.POINTER(ctypes.c_void_p)
)
_DestroyContext = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
_CreateScene = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(_TriangleSceneDesc),
    ctypes.POINTER(ctypes.c_void_p),
)
_UpdateScene = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(_TriangleSceneDesc)
)
_Trace = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(_TraceDesc))
_GetSceneMemory = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(_SceneMemory)
)
_DestroyScene = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
_PrepareTyped = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
_TraceTyped = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(_TypedTraceDesc)
)
_GetLastError = ctypes.CFUNCTYPE(
    ctypes.c_size_t, ctypes.POINTER(ctypes.c_char), ctypes.c_size_t
)


class _ProviderApi(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("provider_abi_version", ctypes.c_uint32),
        ("info", _ProviderInfo),
        ("probe_runtime", _ProbeRuntime),
        ("create_context", _CreateContext),
        ("destroy_context", _DestroyContext),
        ("create_triangle_scene", _CreateScene),
        ("update_triangle_scene", _UpdateScene),
        ("trace", _Trace),
        ("get_scene_memory", _GetSceneMemory),
        ("destroy_triangle_scene", _DestroyScene),
        ("get_last_error", _GetLastError),
        ("prepare_typed", _PrepareTyped),
        ("trace_typed", _TraceTyped),
    ]


@dataclass(frozen=True)
class _LoadedApi:
    library: object
    path: str
    api: _ProviderApi


def _decode(value):
    return "" if not value else value.decode("utf-8", errors="replace")


def _provider_error(api):
    required = int(api.get_last_error(None, 0))
    if required <= 1:
        return "optional OptiX provider call failed"
    buffer = ctypes.create_string_buffer(required)
    api.get_last_error(buffer, required)
    return buffer.value.decode("utf-8", errors="replace")


def _check_api(api):
    if api.struct_size < _ProviderApi.prepare_typed.offset:
        raise RuntimeError("OptiX provider returned a truncated Forge API table")
    if api.provider_abi_version != PROVIDER_ABI_VERSION:
        raise RuntimeError("OptiX provider returned a mismatched Forge ABI")
    if api.info.struct_size < ctypes.sizeof(_ProviderInfo):
        raise RuntimeError("OptiX provider returned truncated identity facts")
    if api.info.provider_abi_version != PROVIDER_ABI_VERSION:
        raise RuntimeError("OptiX provider identity uses a mismatched Forge ABI")
    if api.info.optix_abi_version not in SUPPORTED_OPTIX_ABIS:
        raise RuntimeError(
            "OptiX provider SDK ABI is outside Forge's bundled adapter range"
        )
    if int(api.info.features) & _REQUIRED_FEATURES != _REQUIRED_FEATURES:
        raise RuntimeError("OptiX provider does not implement the complete ray ABI")
    for name in (
        "probe_runtime",
        "create_context",
        "destroy_context",
        "create_triangle_scene",
        "update_triangle_scene",
        "trace",
        "get_scene_memory",
        "destroy_triangle_scene",
        "get_last_error",
    ):
        if not bool(getattr(api, name)):
            raise RuntimeError(f"OptiX provider is missing ABI entry {name}")


def _load_library(path):
    return ctypes.CDLL(path)


def _query_provider(path):
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("OptiX provider path must be a string or path-like value")
    resolved = str(Path(path).expanduser().resolve())
    library = _load_library(resolved)
    try:
        query = getattr(library, PROVIDER_QUERY_SYMBOL)
    except AttributeError as exc:
        raise RuntimeError("OptiX provider query symbol is missing") from exc
    query.argtypes = [
        ctypes.c_uint32,
        ctypes.c_size_t,
        ctypes.POINTER(_ProviderApi),
    ]
    query.restype = ctypes.c_int
    api = _ProviderApi()
    result = int(
        query(PROVIDER_ABI_VERSION, ctypes.sizeof(_ProviderApi), ctypes.byref(api))
    )
    if result != _SUCCESS:
        message = (
            _provider_error(api)
            if bool(api.get_last_error)
            else f"provider query failed with result {result}"
        )
        raise RuntimeError(message)
    _check_api(api)
    return _LoadedApi(library, resolved, api)


def _resolved_path(path):
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("OptiX library path must be a string or path-like value")
    return str(Path(path).expanduser().resolve())


def _provider_and_runtime_candidates(library_path=None):
    runtime_path = os.environ.get("TAICHI_FORGE_OPTIX_LIBRARY") or None
    if library_path is not None:
        runtime_path = _resolved_path(library_path)
    if runtime_path:
        runtime_path = _resolved_path(runtime_path)
    return _bundled_provider_candidates(), runtime_path, "forge_runtime_wheel"


def _runtime_library_argument(runtime_path):
    return None if runtime_path is None else os.fsencode(runtime_path)


def _probe_provider_runtime(loaded, runtime_path):
    result = int(loaded.api.probe_runtime(_runtime_library_argument(runtime_path)))
    if result != _SUCCESS:
        raise RuntimeError(_provider_error(loaded.api))
    return True


def _format_optix_version(value):
    value = int(value)
    return f"{value // 10000}.{(value // 100) % 100}.{value % 100}"


def probe_provider(path=None):
    """Probe bundled adapters and the vendor runtime without retaining them."""

    native_facts = {
        "probe_policy": "transient_adapter_and_vendor_runtime_query",
        "provider_enablement_changed": False,
        "provider_selection_changed": False,
        "execution_qualified": False,
        "supported_optix_abi_versions": SUPPORTED_OPTIX_ABIS,
    }
    result = {
        "provider_id": "optix",
        "external_component_probed": False,
        "discovery": "missing",
        "unavailable_reason": "bundled_provider_adapter_not_installed",
        "provider_abi": PROVIDER_ABI_NAME,
        "provider_version": None,
        "last_error": None,
        "failure_scope": None,
        "native_facts": native_facts,
    }
    try:
        candidates, runtime_path, provider_source = _provider_and_runtime_candidates(
            path
        )
    except (OSError, TypeError, ValueError) as exc:
        result.update(
            discovery="incompatible",
            unavailable_reason="library_path_resolution_failed",
            last_error=str(exc) or type(exc).__name__,
            failure_scope="provider",
        )
        return result
    native_facts.update(
        provider_source=provider_source,
        vendor_library_candidate=runtime_path or "system_default",
        provider_candidates=tuple(candidates),
    )
    if not candidates:
        return result
    result["external_component_probed"] = True
    failures = []
    for candidate in candidates:
        try:
            loaded = _query_provider(candidate)
            runtime_compatible = _probe_provider_runtime(loaded, runtime_path)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            failures.append(f"{candidate}: {str(exc) or type(exc).__name__}")
            continue
        info = loaded.api.info
        result.update(
            discovery="present",
            unavailable_reason="execution_not_qualified",
            provider_version=_format_optix_version(info.optix_version),
        )
        native_facts.update(
            library_candidate=loaded.path,
            library_loaded_transiently=True,
            runtime_probe_only=True,
            context_created=False,
            vendor_runtime_abi_compatible=runtime_compatible,
            optix_abi_version=int(info.optix_abi_version),
            optix_version=int(info.optix_version),
            provider_name=_decode(info.provider_name),
            build_identity=_decode(info.build_identity),
            feature_bits=int(info.features),
        )
        if failures:
            native_facts["rejected_newer_candidates"] = tuple(failures)
        return result
    result.update(
        discovery="incompatible",
        unavailable_reason="no_compatible_optix_provider",
        last_error="; ".join(failures),
        failure_scope="provider",
    )
    native_facts["library_loaded_transiently"] = False
    return result


def passive_status():
    loaded = tuple(provider for provider in _loaded_providers if not provider.closed)
    native_facts = {
        "status_policy": "passive_loaded_optix_plugins",
        "external_component_probed": False,
        "provider_enablement_changed": False,
        "provider_selection_changed": False,
        "loaded_provider_count": len(loaded),
    }
    if not loaded:
        return {
            "provider_id": "optix",
            "library_loaded": False,
            "provider_abi": PROVIDER_ABI_NAME,
            "provider_version": None,
            "native_facts": native_facts,
        }
    provider = loaded[0]
    native_facts.update(provider.identity)
    return {
        "provider_id": "optix",
        "library_loaded": True,
        "provider_abi": PROVIDER_ABI_NAME,
        "provider_version": provider.identity["provider_version"],
        "native_facts": native_facts,
    }


def _ray_storage(value, width, dtypes, name):
    description = describe_storage(value)
    descriptor = description.descriptor
    if descriptor is None or not description.supported:
        raise TaichiRuntimeError(f"OptiX ray {name} requires canonical dense storage")
    shape = tuple(descriptor.index_shape)
    element_shape = tuple(descriptor.element_shape)
    if descriptor.scalar_type not in dtypes:
        raise TaichiRuntimeError(f"OptiX ray {name} must use dtype {dtypes}")
    if element_shape == () and len(shape) == 2 and shape[1] == width:
        count = shape[0]
    elif element_shape == (width,) and len(shape) == 1:
        count = shape[0]
    else:
        raise TaichiRuntimeError(
            f"OptiX ray {name} must have scalar shape (N, {width}) or "
            f"AOS vector-{width} shape (N,)"
        )
    if count <= 0:
        raise TaichiRuntimeError(f"OptiX ray {name} must not be empty")
    return description, count


def _invoke_checked(api, function, *args):
    if int(function(*args)) != _SUCCESS:
        raise TaichiRuntimeError(_provider_error(api))


@dataclass(frozen=True)
class _PreparedOptixCall:
    owner: object
    storage: object
    call: object
    owners: tuple

    def __call__(self):
        self.owner._validate_lifetime()
        with hardware_failure_phase("provider_execution_failure"):
            self.owner._runtime_prog._invoke_external_cuda_prepared(
                self.storage, self.call
            )


def _prepare_storage(owner, values, descriptions, writable):
    packet = owner._runtime_prog._prepare_external_cuda_storage(
        tuple(item.descriptor for item in descriptions), writable
    )
    owners = (
        *values,
        *descriptions,
        *(value.arr for value in values if isinstance(value, Ndarray)),
    )
    return packet, owners


class OptixProvider:
    """Owner of one bundled OptiX adapter and CUDA context view."""

    def __init__(self, library_path=None, *, validation=False):
        program = impl.get_runtime().prog
        if program is None or active_backend() != "cuda":
            raise TaichiRuntimeError(
                "OptixProvider requires an initialized Taichi CUDA runtime"
            )
        if not isinstance(validation, bool):
            raise TypeError("validation must be a bool")
        with hardware_failure_phase("provider_load_failure"):
            candidates, runtime_path, provider_source = (
                _provider_and_runtime_candidates(library_path)
            )
            if not candidates:
                raise TaichiRuntimeError(
                    "Forge runtime wheel does not contain an OptiX provider adapter"
                )
            queried_candidates = []
            load_failures = []
            for candidate in candidates:
                try:
                    queried_candidates.append(_query_provider(candidate))
                except (
                    AttributeError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as exc:
                    load_failures.append(
                        f"{candidate}: {str(exc) or type(exc).__name__}"
                    )
            if not queried_candidates:
                raise TaichiRuntimeError(
                    "no loadable OptiX provider adapter: " + "; ".join(load_failures)
                )
        failures = list(load_failures)
        loaded = None
        context = None
        attempted = []
        with hardware_failure_phase("provider_plan_failure"):
            for candidate_loaded in queried_candidates:
                attempted.append(candidate_loaded.path)
                candidate_context = ctypes.c_void_p()
                desc = _ContextDesc(
                    ctypes.sizeof(_ContextDesc),
                    0,
                    0,
                    int(validation),
                    0,
                    _runtime_library_argument(runtime_path),
                )
                result = int(
                    candidate_loaded.api.create_context(
                        ctypes.byref(desc), ctypes.byref(candidate_context)
                    )
                )
                if result == _SUCCESS and candidate_context.value:
                    loaded = candidate_loaded
                    context = candidate_context
                    break
                message = _provider_error(candidate_loaded.api)
                failures.append(f"{candidate_loaded.path}: {message}")
                if result != _OPTIX_UNAVAILABLE:
                    raise TaichiRuntimeError(message)
            if loaded is None or context is None:
                raise TaichiRuntimeError(
                    "no compatible OptiX provider adapter: " + "; ".join(failures)
                )
        self._loaded = loaded
        self._context = context
        self._runtime_prog = program
        self._runtime_generation = int(impl.runtime_generation())
        self._scenes = weakref.WeakSet()
        self._typed_prepared = False
        self._shared_pipeline_sbt_bytes = 0
        info = loaded.api.info
        self.identity = MappingProxyType(
            {
                "library_candidate": loaded.path,
                "vendor_library_candidate": runtime_path or "system_default",
                "provider_source": provider_source,
                "provider_candidates_attempted": tuple(attempted),
                "provider_abi": PROVIDER_ABI_NAME,
                "provider_version": _format_optix_version(info.optix_version),
                "optix_abi_version": int(info.optix_abi_version),
                "optix_version": int(info.optix_version),
                "provider_name": _decode(info.provider_name),
                "build_identity": _decode(info.build_identity),
                "feature_bits": int(info.features),
            }
        )
        _loaded_providers.add(self)

    @property
    def closed(self):
        return self._context is None

    def triangle_scene(self, vertices, indices, *, allow_update=True):
        self._validate_lifetime()
        return OptixTriangleScene(self, vertices, indices, allow_update=allow_update)

    def _prepare_typed(self, scene):
        self._validate_lifetime()
        if self._typed_prepared:
            return
        api = self._loaded.api
        if (
            api.struct_size < ctypes.sizeof(_ProviderApi)
            or not int(api.info.features) & _TYPED_HITS
            or not bool(api.prepare_typed)
            or not bool(api.trace_typed)
        ):
            raise TaichiRuntimeError(
                "OptiX adapter does not support typed hits; use a newer Forge adapter"
            )
        with hardware_failure_phase("provider_plan_failure"):
            # The existing submission gate also orders explicit scene/context
            # close against this cold adapter preparation; no work is replayed.
            scope = self._runtime_prog._begin_external_cuda_submission()
            try:
                scene._validate_lifetime()
                _invoke_checked(api, api.prepare_typed, self._context)
                memory = _SceneMemory()
                memory.struct_size = ctypes.sizeof(memory)
                _invoke_checked(
                    api, api.get_scene_memory, scene._scene, ctypes.byref(memory)
                )
                self._shared_pipeline_sbt_bytes = int(memory.shared_pipeline_sbt_bytes)
            finally:
                del scope
        self._typed_prepared = True

    def _validate_lifetime(self):
        if self._context is None:
            raise TaichiRuntimeError("OptixProvider has been closed")
        validate_runtime_generation(
            self, "OptixProvider belongs to a previous Taichi runtime generation"
        )

    def close(self):
        if self._context is None:
            return None
        live = tuple(scene for scene in self._scenes if not scene.closed)
        if live:
            raise TaichiRuntimeError(
                "OptixProvider cannot close while triangle scenes are live"
            )
        context = self._context
        self._context = None
        if runtime_generation_matches(self):
            self._runtime_prog.synchronize()
            result = int(self._loaded.api.destroy_context(context))
            if result != _SUCCESS:
                self._context = context
                raise TaichiRuntimeError(_provider_error(self._loaded.api))
        return None

    destroy = close

    def __enter__(self):
        self._validate_lifetime()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


@instrument_hardware_recording("ray.query.batch.optix")
class OptixRayQueryRecording(BackendCommandRecording):
    """One runtime-ordered OptiX launch against a fixed scene generation."""

    def __init__(self, scene, ray_count, *, rays="rays", hits="hits", hit_indices=None):
        if not isinstance(scene, OptixTriangleScene):
            raise TypeError("OptiX ray query recording requires an OptixTriangleScene")
        if (
            isinstance(ray_count, bool)
            or not isinstance(ray_count, int)
            or not 1 <= ray_count <= 0xFFFFFFFF
        ):
            raise ValueError("OptiX ray count must be in [1, UINT32_MAX]")
        names = (rays, hits) if hit_indices is None else (rays, hits, hit_indices)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("OptiX ray bindings must be nonempty strings")
        if len(set(names)) != len(names):
            raise ValueError("OptiX ray bindings must be unique")
        super().__init__(
            backend="cuda",
            binding_names=names,
            command_count=1,
            queue="compute",
            stream_binding="runtime_ordered",
            barrier_policy="internal",
            workspace_ownership="provider_generation",
            replay_mode="rerecord",
            no_host_readback=True,
        )
        object.__setattr__(self, "scene", scene)
        object.__setattr__(self, "ray_count", ray_count)
        object.__setattr__(self, "rays", rays)
        object.__setattr__(self, "hits", hits)
        object.__setattr__(self, "hit_indices", hit_indices)
        if hit_indices is not None:
            scene.provider._prepare_typed(scene)

    @property
    def resource_effects(self):
        effects = (
            ResourceEffect(self.rays, GraphAccess.READ),
            ResourceEffect(self.hits, GraphAccess.WRITE),
            static_resource_effect(self.scene._effect_name, GraphAccess.READ),
        )
        if self.hit_indices is not None:
            effects += (ResourceEffect(self.hit_indices, GraphAccess.WRITE),)
        return effects

    def execute(self, bindings):
        return self.prepare_graph_execute(bindings)()

    def _binding_descriptions(self, bindings):
        specs = [(self.rays, (f32,), 8), (self.hits, (f32,), 4)]
        if self.hit_indices is not None:
            specs.append((self.hit_indices, (i32, u32), 4))
        descriptions = []
        for name, dtypes, width in specs:
            description, count = _ray_storage(bindings[name], width, dtypes, name)
            if count != self.ray_count:
                raise TaichiRuntimeError(
                    f"OptiX {name} binding has the wrong ray count"
                )
            descriptions.append(description)
        return descriptions

    def validate_graph_bindings(self, bindings):
        self._binding_descriptions(bindings)

    def prepare_graph_execute(self, bindings):
        validate_exact_bindings(self, bindings, "OptiX ray query")
        self.validate_graph_lifetime()
        descriptions = self._binding_descriptions(bindings)
        values = tuple(bindings[name] for name in self.binding_names)
        storage, owners = _prepare_storage(
            self.scene, values, descriptions, (False, *([True] * (len(values) - 1)))
        )
        api = self.scene.provider._loaded.api
        if not int(api.info.features) & _WORD_ALIGNED_QUERY_STORAGE:
            if any(pointer % 16 for pointer in storage.pointers):
                raise TaichiRuntimeError(
                    "This OptiX adapter requires 16-byte aligned ray/hit storage; "
                    "use a newer Forge adapter for word-aligned dense views"
                )
        descriptor_type = _TraceDesc if self.hit_indices is None else _TypedTraceDesc
        function = api.trace if self.hit_indices is None else api.trace_typed
        desc = descriptor_type(
            ctypes.sizeof(descriptor_type), self.ray_count, *storage.pointers, 0
        )
        return _PreparedOptixCall(
            self.scene,
            storage,
            partial(
                _invoke_checked, api, function, self.scene._scene, ctypes.byref(desc)
            ),
            owners,
        )

    def validate_graph_lifetime(self):
        self.scene._validate_lifetime()

    def memory_report(self):
        return self.scene.memory_report()

    def _as_graph_native_node(self):
        return native_recording_node(
            self,
            lifetime_leases=lambda item: (item.scene, item.scene.provider),
            debug_info=lambda item: {
                "kind": "optix_triangle_ray_query",
                "ray_count": item.ray_count,
                "provider_abi": PROVIDER_ABI_NAME,
                "hit_layout": "legacy_float4" if item.hit_indices is None else "typed",
            },
            publish_time_binding_validation_stable=True,
        )


@instrument_hardware_recording("ray.as_refit.optix")
class OptixRayRefitRecording(BackendCommandRecording):
    """One fixed-topology OptiX GAS update."""

    def __init__(self, scene, *, vertices="vertices"):
        if not isinstance(scene, OptixTriangleScene):
            raise TypeError("OptiX refit recording requires an OptixTriangleScene")
        if not isinstance(vertices, str) or not vertices:
            raise ValueError("OptiX refit binding must be a nonempty string")
        super().__init__(
            backend="cuda",
            binding_names=(vertices,),
            command_count=1,
            queue="compute",
            stream_binding="runtime_ordered",
            barrier_policy="internal",
            workspace_ownership="provider_generation",
            replay_mode="rerecord",
            no_host_readback=True,
        )
        object.__setattr__(self, "scene", scene)
        object.__setattr__(self, "vertices", vertices)

    @property
    def resource_effects(self):
        return (
            ResourceEffect(self.vertices, GraphAccess.READ),
            static_resource_effect(self.scene._effect_name, GraphAccess.WRITE),
        )

    def execute(self, bindings):
        return self.prepare_graph_execute(bindings)()

    def _binding_descriptions(self, bindings):
        description, count = _ray_storage(
            bindings[self.vertices], 3, (f32,), self.vertices
        )
        if count != self.scene.vertex_count:
            raise TaichiRuntimeError("OptiX refit must preserve the vertex count")
        return description, self.scene._indices_description

    def validate_graph_bindings(self, bindings):
        self._binding_descriptions(bindings)

    def prepare_graph_execute(self, bindings):
        validate_exact_bindings(self, bindings, "OptiX ray refit")
        self.validate_graph_lifetime()
        descriptions = self._binding_descriptions(bindings)
        values = (bindings[self.vertices], self.scene._indices)
        storage, owners = _prepare_storage(
            self.scene, values, descriptions, (False, False)
        )
        desc = _TriangleSceneDesc(
            ctypes.sizeof(_TriangleSceneDesc),
            self.scene.vertex_count,
            self.scene.triangle_count,
            1,
            *storage.pointers,
            0,
        )
        api = self.scene.provider._loaded.api
        return _PreparedOptixCall(
            self.scene,
            storage,
            partial(
                _invoke_checked,
                api,
                api.update_triangle_scene,
                self.scene._scene,
                ctypes.byref(desc),
            ),
            owners,
        )

    def validate_graph_lifetime(self):
        self.scene._validate_lifetime()

    def memory_report(self):
        return self.scene.memory_report()

    def _as_graph_native_node(self):
        return native_recording_node(
            self,
            lifetime_leases=lambda item: (item.scene, item.scene.provider),
            debug_info=lambda item: {
                "kind": "optix_triangle_gas_update",
                "vertex_count": item.scene.vertex_count,
            },
            publish_time_binding_validation_stable=True,
        )


class OptixTriangleScene:
    """Fixed-topology triangle GAS plus identity IAS owned by a provider."""

    def __init__(self, provider, vertices, indices, *, allow_update=True):
        if not isinstance(provider, OptixProvider):
            raise TypeError("provider must be an OptixProvider")
        if not isinstance(allow_update, bool):
            raise TypeError("allow_update must be a bool")
        provider._validate_lifetime()
        vertex_description, vertex_count = _ray_storage(vertices, 3, (f32,), "vertices")
        index_description, triangle_count = _ray_storage(indices, 3, (i32,), "indices")
        storage, owners = _prepare_storage(
            provider,
            (vertices, indices),
            (vertex_description, index_description),
            (False, False),
        )
        scene = ctypes.c_void_p()
        desc = _TriangleSceneDesc(
            ctypes.sizeof(_TriangleSceneDesc),
            vertex_count,
            triangle_count,
            int(allow_update),
            *storage.pointers,
            0,
        )
        api = provider._loaded.api
        with hardware_failure_phase("provider_plan_failure"):
            provider._runtime_prog._invoke_external_cuda_prepared(
                storage,
                partial(
                    _invoke_checked,
                    api,
                    api.create_triangle_scene,
                    provider._context,
                    ctypes.byref(desc),
                    ctypes.byref(scene),
                ),
            )
            if not scene.value:
                raise TaichiRuntimeError("OptiX provider returned an empty scene")
        self.provider = provider
        self._scene = scene
        self._runtime_prog = provider._runtime_prog
        self._runtime_generation = provider._runtime_generation
        self.vertex_count = vertex_count
        self.triangle_count = triangle_count
        self.allow_update = allow_update
        self._indices = indices
        self._indices_description = index_description
        self._index_owner = indices.arr if isinstance(indices, Ndarray) else indices
        self._effect_name = (
            f"optix-ray-scene:{self._runtime_generation}:{int(scene.value)}"
        )
        memory = _SceneMemory()
        memory.struct_size = ctypes.sizeof(_SceneMemory)
        result = int(provider._loaded.api.get_scene_memory(scene, ctypes.byref(memory)))
        if result != _SUCCESS:
            provider._loaded.api.destroy_triangle_scene(scene)
            self._scene = None
            raise TaichiRuntimeError(_provider_error(provider._loaded.api))
        self._memory = memory
        provider._shared_pipeline_sbt_bytes = int(memory.shared_pipeline_sbt_bytes)
        provider._scenes.add(self)

    @property
    def closed(self):
        return self._scene is None

    def record(self, ray_count, *, rays="rays", hits="hits"):
        self._validate_lifetime()
        return OptixRayQueryRecording(self, ray_count, rays=rays, hits=hits)

    def trace(self, rays, hits):
        ray_count = _ray_storage(rays, 8, (f32,), "rays")[1]
        self.record(ray_count).execute({"rays": rays, "hits": hits})
        return hits

    def record_typed(
        self, ray_count, *, rays="rays", hits="hits", hit_indices="hit_indices"
    ):
        """Record f32 (t,u,v,0) and i32/u32 (primitive,instance,custom,hit).

        Misses write (-1,0,0,0) and (-1,-1,-1,0), with UINT32_MAX for u32.
        This single-instance scene has instance ordinal and custom ID zero.
        t is the ray parameter (distance only for unit directions); triangle
        weights are (1-u-v,u,v). Scalar (N,4) and AOS vector-4 layouts are accepted.
        """
        self._validate_lifetime()
        return OptixRayQueryRecording(
            self, ray_count, rays=rays, hits=hits, hit_indices=hit_indices
        )

    def trace_typed(self, rays, hits, hit_indices):
        self.record_typed(_ray_storage(rays, 8, (f32,), "rays")[1]).execute(
            {"rays": rays, "hits": hits, "hit_indices": hit_indices}
        )
        return hits, hit_indices

    def record_refit(self, *, vertices="vertices"):
        self._validate_lifetime()
        if not self.allow_update:
            raise TaichiRuntimeError("OptiX scene was not created for updates")
        return OptixRayRefitRecording(self, vertices=vertices)

    def refit(self, vertices):
        self.record_refit().execute({"vertices": vertices})
        return self

    def _validate_lifetime(self):
        if self._scene is None:
            raise TaichiRuntimeError("OptixTriangleScene has been closed")
        self.provider._validate_lifetime()
        validate_runtime_generation(
            self,
            "OptixTriangleScene belongs to a previous Taichi runtime generation",
        )

    def memory_report(self):
        resident = self._scene is not None and runtime_generation_matches(self)
        memory = self._memory
        components = (
            HardwareMemoryComponent(
                "gas_storage",
                int(memory.gas_bytes),
                True,
                "provider_generation",
                "provider",
                resident=resident,
            ),
            HardwareMemoryComponent(
                "identity_ias_storage",
                int(memory.ias_bytes),
                True,
                "provider_generation",
                "provider",
                resident=resident,
            ),
            HardwareMemoryComponent(
                "build_update_scratch",
                int(memory.build_update_scratch_bytes),
                True,
                "provider_generation",
                "provider",
                resident=resident,
            ),
            HardwareMemoryComponent(
                "instance_and_launch_params",
                int(memory.instance_bytes + memory.launch_params_bytes),
                True,
                "provider_generation",
                "provider",
                resident=resident,
            ),
            HardwareMemoryComponent(
                "shared_pipeline_sbt",
                self.provider._shared_pipeline_sbt_bytes,
                True,
                "runtime",
                "provider",
                resident=resident,
            ),
            HardwareMemoryComponent(
                "optix_driver_context_state",
                None,
                False,
                "runtime",
                "driver",
                resident=resident,
            ),
        )
        return make_memory_report(
            "optix_triangle_ray",
            "cuda",
            components,
            lifecycle_state="ready" if resident else "closed",
            ownership_scope="provider_context_and_scene_generation",
        )

    def _graph_provider_memory_report(self):
        return self.memory_report()

    def close(self):
        if self._scene is None:
            return None
        scene = self._scene
        self._scene = None
        if runtime_generation_matches(self):
            self._runtime_prog.synchronize()
            result = int(self.provider._loaded.api.destroy_triangle_scene(scene))
            if result != _SUCCESS:
                self._scene = scene
                raise TaichiRuntimeError(_provider_error(self.provider._loaded.api))
        return None

    destroy = close

    def __enter__(self):
        self._validate_lifetime()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def load_provider(library_path=None, *, validation=False):
    return OptixProvider(library_path, validation=validation)


def is_loaded():
    return bool(passive_status()["library_loaded"])


__all__ = [
    "OptixProvider",
    "OptixRayQueryRecording",
    "OptixRayRefitRecording",
    "OptixTriangleScene",
    "PROVIDER_ABI_NAME",
    "PROVIDER_ABI_VERSION",
    "SUPPORTED_OPTIX_ABIS",
    "is_loaded",
    "load_provider",
    "passive_status",
    "probe_provider",
]
