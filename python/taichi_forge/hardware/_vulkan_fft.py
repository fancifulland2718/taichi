"""Explicit, fixed-storage Vulkan FFT plans with cold-only JIT and discovery."""

import ctypes
from functools import partial
import os
from pathlib import Path
import weakref

from taichi_forge.graph._ir import GraphAccess, ResourceEffect
from taichi_forge.graph._native import BackendCommandRecording
from taichi_forge.graph._recipes.definition import _digest
from taichi_forge.hardware._bundled_runtime_provider import (
    _binary_sha256,
    _runtime_package_roots,
)
from taichi_forge.hardware._memory import HardwareMemoryComponent, make_memory_report
from taichi_forge.hardware._native_adapter import (
    native_recording_node,
    runtime_generation_matches,
)
from taichi_forge.hardware._runtime import active_backend
from taichi_forge.lang import impl
from taichi_forge.lang._ndarray import ScalarNdarray
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.types.primitive_types import f32


_ABI = "taichi-forge-vkfft-provider-c-abi1"
_PLANS = weakref.WeakSet()
_ADAPTER_STEM = "taichi_forge_vkfft_provider_abi1_vkfft134"


def _adapter_path(value):
    value = value if value is not None else os.environ.get("TI_VKFFT_LIBRARY_PATH")
    if value is None:
        filename = (
            f"{_ADAPTER_STEM}.dll" if os.name == "nt" else f"lib{_ADAPTER_STEM}.so"
        )
        for root in _runtime_package_roots():
            candidate = root / "_lib" / "hardware_providers" / filename
            if candidate.is_file():
                return str(candidate.resolve())
    if not value:
        raise ValueError(
            "the optional Vulkan FFT adapter is not bundled; provide adapter_path "
            "or TI_VKFFT_LIBRARY_PATH, or install a runtime wheel containing it"
        )
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Vulkan FFT adapter does not exist: {path}")
    return str(path.resolve())


class _Api(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint32)
        for name in (
            "struct_size",
            "abi_version",
            "vkfft_version",
            "glslang_major",
            "glslang_minor",
            "glslang_patch",
        )
    ] + [
        (name, ctypes.c_void_p)
        for name in ("create", "append", "memory", "destroy", "last_error")
    ]


def probe_provider(library_path=None):
    """Inspect the optional adapter ABI; do not initialize a device or FFT plan."""
    result = {
        "provider_id": "vkfft",
        "external_component_probed": False,
        "discovery": "missing",
        "unavailable_reason": "external_library_not_found",
        "provider_abi": _ABI,
        "provider_version": None,
        "last_error": None,
        "failure_scope": None,
        "native_facts": {
            "probe_policy": "explicit_transient_adapter_query",
            "provider_enablement_changed": False,
            "provider_selection_changed": False,
            "execution_qualified": False,
            "probe_created_execution_resource": False,
        },
    }
    library = None
    try:
        path = _adapter_path(library_path)
        result["external_component_probed"] = True
        library = ctypes.CDLL(path)
        query = library.taichi_forge_vkfft_provider_query
        query.argtypes = (ctypes.c_uint32, ctypes.c_size_t, ctypes.POINTER(_Api))
        query.restype = ctypes.c_int
        api = _Api()
        if (
            query(1, ctypes.sizeof(api), ctypes.byref(api)) != 0
            or api.struct_size != ctypes.sizeof(api)
            or api.abi_version != 1
            or api.vkfft_version != 10304
            or not all(
                (api.create, api.append, api.memory, api.destroy, api.last_error)
            )
        ):
            raise RuntimeError("Vulkan FFT adapter ABI/version is incompatible")
        result.update(
            discovery="available", unavailable_reason="none", provider_version="1.3.4"
        )
        result["native_facts"].update(
            library_candidate=path,
            glslang_version=(api.glslang_major, api.glslang_minor, api.glslang_patch),
            execution_api_available=True,
        )
    except (OSError, ValueError, TypeError, AttributeError, RuntimeError) as exc:
        result.update(last_error=str(exc), failure_scope="provider")
        if result["external_component_probed"]:
            result.update(
                discovery="incompatible", unavailable_reason="provider_query_failed"
            )
    finally:
        if library is not None:
            # No query pointer or adapter resource escapes this transient probe.
            import _ctypes  # pylint: disable=import-outside-toplevel

            release = _ctypes.FreeLibrary if os.name == "nt" else _ctypes.dlclose
            release(library._handle)  # pylint: disable=protected-access
    return result


def passive_status():
    plans = tuple(
        plan for plan in _PLANS if not plan.closed and runtime_generation_matches(plan)
    )
    return {
        "provider_id": "vkfft",
        "library_loaded": bool(plans),
        "provider_abi": _ABI,
        "provider_version": "1.3.4" if plans else None,
        "native_facts": {
            "status_policy": "passive_open_plan_registry",
            "external_component_probed": False,
            "provider_enablement_changed": False,
            "provider_selection_changed": False,
            "open_plan_count": len(plans),
            "pending_command_retention_observed": False,
            "status_scope": "public open plans; not an OS module or pending-command census",
        },
    }


def _positive_int(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 < value <= 0x7FFFFFFF
    ):
        raise ValueError(f"Vulkan FFT {name} must be an integer in [1, INT_MAX]")
    return value


class VulkanFftPlan:
    """In-place compact scalar-f32 complex FFT on a frozen Vulkan ndarray.

    ``dimensions`` are row-major, rank 1--3, with prime factors at most 13.
    Storage shape is ``dimensions + (2,)``, prefixed by ``batch_count`` only
    when it exceeds one. ``normalization='inverse'`` divides an inverse by
    the transform volume; the default leaves both directions unnormalized.

    The optional VkFFT 1.3.4 adapter is resolved from the runtime package only
    on explicit use. Override it with ``adapter_path`` or
    ``TI_VKFFT_LIBRARY_PATH``. Creation can compile shaders and initialize
    lookup tables. Execution submits no readback or host wait. Root Graphs
    retain a host call per FFT action, executing a pre-recorded GPU sequence;
    this default path is not enclosing Graph capture. Explicitly add
    ``VulkanFftRecipeProvider`` to complete-recipe search for batch workspace
    strategies and immutable whole-Graph secondary recording.
    """

    graph_runtime_lifetime_check_required = False

    def __init__(
        self,
        data,
        dimensions,
        *,
        batch_count=1,
        direction="forward",
        normalization="none",
        adapter_path=None,
        _batch_tile=None,
    ):
        self._closed = True
        self._dimensions = tuple(dimensions)
        if not 1 <= len(self._dimensions) <= 3:
            raise ValueError("Vulkan FFT requires rank 1--3")
        for value in self._dimensions:
            remaining = _positive_int(value, "dimension")
            for radix in (2, 3, 5, 7, 11, 13):
                while remaining % radix == 0:
                    remaining //= radix
            if remaining != 1:
                raise ValueError(
                    "Vulkan FFT dimensions must have prime factors at most 13"
                )
        self._batch_count = _positive_int(batch_count, "batch_count")
        if _batch_tile is not None:
            _positive_int(_batch_tile, "private batch tile")
            if _batch_tile > batch_count:
                raise ValueError("Vulkan FFT private batch tile exceeds batch_count")
        if direction not in ("forward", "inverse"):
            raise ValueError("Vulkan FFT direction must be 'forward' or 'inverse'")
        if normalization not in ("none", "inverse"):
            raise ValueError("Vulkan FFT normalization must be 'none' or 'inverse'")
        self._direction, self._normalization = direction, normalization
        self._shape = (
            ((batch_count,) if batch_count > 1 else ()) + self.dimensions + (2,)
        )
        program = impl.get_runtime().prog
        if program is None or active_backend() != "vulkan":
            raise TaichiRuntimeError(
                "Vulkan FFT requires an initialized Vulkan backend"
            )
        if (
            not isinstance(data, ScalarNdarray)
            or data.dtype != f32
            or tuple(data.shape) != self.shape
        ):
            raise ValueError(
                f"Vulkan FFT data must be a scalar f32 ndarray of shape {self.shape}"
            )
        self._runtime_prog = program
        self._runtime_generation = int(impl.runtime_generation())
        self._data = data
        self._adapter_path = _adapter_path(adapter_path)
        self._adapter_sha256 = _binary_sha256(self._adapter_path)
        create_name = (
            "_create_vulkan_fft_plan"
            if _batch_tile is None
            else "_create_vulkan_fft_recipe_plan"
        )
        create_plan = getattr(program, create_name, None)
        if create_plan is None:
            raise TaichiRuntimeError(
                "Vulkan FFT native bridge is unavailable in this runtime build"
            )
        self._handle = create_plan(
            self._adapter_path,
            data.arr,
            self.dimensions,
            batch_count,
            -1 if direction == "forward" else 1,
            normalization == "inverse",
            *((_batch_tile,) if _batch_tile is not None else ()),
        )
        self._closed = False
        try:
            self._statistics = dict(program._vulkan_fft_plan_statistics(self._handle))
        except Exception:
            self.close()
            raise
        # Keep the original Program and handle: reset clears its plan table.
        # Never route an old plan through a new Program with a colliding handle.
        self._submit = partial(program._vulkan_fft_execute, self._handle)
        self._semantic_id = _digest(
            (
                "vulkan-compact-inplace-c2c-f32-v1",
                self.dimensions,
                batch_count,
                direction,
                normalization,
            )
        )
        self._physical_id = _digest(
            (self._semantic_id, self._adapter_sha256, self._statistics)
        )
        _PLANS.add(self)

    dimensions = property(lambda self: self._dimensions)
    batch_count = property(lambda self: self._batch_count)
    direction = property(lambda self: self._direction)
    normalization = property(lambda self: self._normalization)
    shape = property(lambda self: self._shape)
    closed = property(lambda self: self._closed)

    def run(self):
        """Enqueue the fixed in-place transform on the runtime-ordered queue."""
        self._submit()

    def validate_graph_lifetime(self):
        if self.closed or not runtime_generation_matches(self):
            raise TaichiRuntimeError(
                "Vulkan FFT plan is closed or belongs to a previous runtime generation"
            )

    def record(self, *, data="data"):
        """Return a root-Graph node; bind ``data`` to this plan's original array."""
        self.validate_graph_lifetime()
        if not isinstance(data, str) or not data:
            raise ValueError("Vulkan FFT binding name must be a nonempty string")
        return _Recording(self, data).as_node()

    def statistics(self):
        """Cold-plan requested allocations and build/device facts, not device peak."""
        return {
            **self._statistics,
            "adapter_path": self._adapter_path,
            "adapter_sha256": self._adapter_sha256,
            "physical_plan_id": self._physical_id,
            "graph_integration": "root_ordered",
            "gpu_sequence": "retained_secondary_commands",
        }

    def memory_report(self):
        valid = runtime_generation_matches(self)
        ready = valid and not self.closed
        return make_memory_report(
            "vkfft",
            "vulkan",
            (
                HardwareMemoryComponent(
                    "plan_requested_device_allocations",
                    self._statistics["persistent_allocation_bytes"],
                    True,
                    "provider_generation",
                    "provider",
                    resident=ready,
                ),
                HardwareMemoryComponent(
                    "driver_objects_and_pending_command_retention",
                    None,
                    False,
                    "provider_generation",
                    "driver",
                    resident=valid,
                ),
            ),
            lifecycle_state="closed"
            if self.closed
            else "ready"
            if valid
            else "runtime_invalid",
            ownership_scope="plan capacity excluding caller storage; close does not observe command retirement",
        )

    _graph_provider_memory_report = memory_report

    def _graph_provider_memory_identity(self):
        return ("vulkan_fft", self._runtime_generation, self._handle)

    def close(self):
        """Invalidate future calls; enqueued command buffers keep their resources."""
        if not self.closed:
            self._runtime_prog._destroy_vulkan_fft_plan(self._handle)
            self._closed = True

    def __enter__(self):
        self.validate_graph_lifetime()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        if not getattr(self, "_closed", True):
            self.close()


class _Recording(BackendCommandRecording):
    def __init__(self, plan, data):
        super().__init__(
            backend="vulkan",
            binding_names=(data,),
            command_count=1,
            workspace_ownership="provider_generation",
            replay_mode="native_replay",
        )
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "_graph_semantic_fingerprint", plan._semantic_id)
        object.__setattr__(self, "_graph_physical_plan_id", plan._physical_id)

    @property
    def resource_effects(self):
        return (ResourceEffect(self.data, GraphAccess.READ_WRITE),)

    def validate_graph_bindings(self, bindings):
        if bindings[self.data] is not self.plan._data:
            raise TaichiRuntimeError(
                "Vulkan FFT Graph binding must retain the plan's original data ndarray"
            )

    def execute(self, bindings):
        # This private recording is only published as a native node. Its fixed
        # storage identity is validated at binding publication, not at replay.
        self.plan.run()

    def as_node(self):
        return native_recording_node(
            self,
            lifetime_leases=(self.plan,),
            debug_info={
                "kind": "vulkan_fft",
                "dimensions": self.plan.dimensions,
                "batch_count": self.plan.batch_count,
                "direction": self.plan.direction,
                "normalization": self.plan.normalization,
                "graph_integration": "root_ordered",
                "gpu_sequence": "retained_secondary_commands",
            },
            publish_time_binding_validation_stable=True,
        )
