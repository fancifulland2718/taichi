"""Explicit Vulkan acceleration-structure and batch ray-query provider."""

from dataclasses import dataclass
from functools import partial
import math

from taichi_forge._lib import core as _ti_core
from taichi_forge.graph._ir import GraphAccess, ResourceEffect
from taichi_forge.graph._native import BackendCommandRecording
from taichi_forge._hardware_telemetry import (
    hardware_failure_phase,
    instrument_hardware_recording,
)
from taichi_forge.hardware._memory import HardwareMemoryComponent, make_memory_report
from taichi_forge.hardware._native_adapter import (
    native_recording_node,
    runtime_generation_matches,
    static_resource_effect,
    validate_exact_bindings,
    validate_runtime_generation,
    graph_bindings_are_validated,
)
from taichi_forge.hardware._runtime import active_backend
from taichi_forge.lang import impl
from taichi_forge.lang._ndarray import Ndarray
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.types.primitive_types import f32, i32, u32


def _item_count(value, width, dtype, name):
    if not isinstance(value, Ndarray):
        raise TaichiRuntimeError(f"Vulkan ray {name} must be a Taichi ndarray")
    shape = tuple(value.shape)
    element_shape = tuple(value.element_shape)
    if value.dtype != dtype:
        raise TaichiRuntimeError(f"Vulkan ray {name} must use dtype {dtype}")
    if element_shape == () and len(shape) == 2 and shape[1] == width:
        count = shape[0]
    elif element_shape == (width,) and len(shape) == 1:
        count = shape[0]
    else:
        raise TaichiRuntimeError(
            f"Vulkan ray {name} must have scalar shape (N, {width}) or "
            f"AOS vector-{width} shape (N,)"
        )
    if count <= 0:
        raise TaichiRuntimeError(f"Vulkan ray {name} must not be empty")
    return count


@dataclass(frozen=True)
class _PreparedRayQuery:
    command: object
    # Keep the native ndarray wrappers themselves, not just mutable Python shells.
    owners: tuple


@instrument_hardware_recording("ray.query.batch.vulkan")
class VulkanRayQueryRecording(BackendCommandRecording):
    """One batch query against a fixed scene or TLAS generation."""

    def __init__(self, scene, ray_count, *, rays="rays", hits="hits", hit_indices=None):
        if not isinstance(scene, (TriangleScene, InstanceTLAS)):
            raise TypeError(
                "Vulkan ray query recording requires a TriangleScene or " "InstanceTLAS"
            )
        if (
            isinstance(ray_count, bool)
            or not isinstance(ray_count, int)
            or ray_count <= 0
            or ray_count > 0xFFFFFFFF
        ):
            raise ValueError("Vulkan ray query count must be in [1, UINT32_MAX]")
        names = (rays, hits) if hit_indices is None else (rays, hits, hit_indices)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Vulkan ray query binding names must be nonempty strings")
        if len(set(names)) != len(names):
            raise ValueError("Vulkan ray query binding names must be unique")
        super().__init__(
            backend="vulkan",
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
            scene._validate_lifetime()
            scene._runtime_prog._prepare_vulkan_typed_ray_query(
                scene._handle, isinstance(scene, InstanceTLAS)
            )

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
        if isinstance(bindings, _PreparedRayQuery):
            with hardware_failure_phase("provider_execution_failure"):
                return self.scene._runtime_prog._execute_vulkan_ray_query(
                    bindings.command
                )
        if not graph_bindings_are_validated(bindings):
            validate_exact_bindings(self, bindings, "Vulkan ray query")
            self.validate_graph_bindings(bindings)
        self.validate_graph_lifetime()
        with hardware_failure_phase("provider_execution_failure"):
            self.scene._execute_query(
                bindings[self.rays],
                bindings[self.hits],
                self.ray_count,
                None if self.hit_indices is None else bindings[self.hit_indices],
            )

    def prepare_graph_execute(self, bindings):
        """Prepare fixed native bindings without tracing rays or synchronizing."""
        validate_exact_bindings(self, bindings, "Vulkan ray query")
        self.validate_graph_bindings(bindings)
        self.validate_graph_lifetime()
        owners = tuple(bindings[name].arr for name in self.binding_names)
        command = self.scene._runtime_prog._prepare_vulkan_ray_query(
            self.scene._handle,
            isinstance(self.scene, InstanceTLAS),
            owners[0],
            owners[1],
            self.ray_count,
            None if self.hit_indices is None else owners[2],
        )
        return partial(self.execute, _PreparedRayQuery(command, owners))

    def validate_graph_bindings(self, bindings):
        rays = bindings[self.rays]
        hits = bindings[self.hits]
        if _item_count(rays, 8, f32, self.rays) != self.ray_count:
            raise TaichiRuntimeError(
                f"Vulkan ray binding {self.rays!r} has the wrong ray count"
            )
        if _item_count(hits, 4, f32, self.hits) != self.ray_count:
            raise TaichiRuntimeError(
                f"Vulkan ray binding {self.hits!r} has the wrong ray count"
            )
        if self.hit_indices is not None:
            indices = bindings[self.hit_indices]
            dtype = getattr(indices, "dtype", None)
            if dtype not in (i32, u32):
                raise TaichiRuntimeError(
                    "Vulkan ray hit_indices must use dtype i32 or u32"
                )
            if _item_count(indices, 4, dtype, self.hit_indices) != self.ray_count:
                raise TaichiRuntimeError(
                    "Vulkan ray hit_indices has the wrong ray count"
                )

    def validate_graph_lifetime(self):
        self.scene._validate_lifetime()

    def memory_report(self):
        return self.scene.memory_report()

    def _as_graph_native_node(self):
        return native_recording_node(
            self,
            lifetime_leases=lambda item: (item.scene,),
            debug_info=lambda item: {
                "kind": "vulkan_triangle_ray_query",
                "ray_count": item.ray_count,
                "scene_kind": item.scene._scene_kind,
                "hit_layout": (
                    "typed" if item.hit_indices is not None else "legacy_float4"
                ),
            },
            publish_time_binding_validation_stable=True,
        )


@instrument_hardware_recording("ray.as_refit.vulkan")
class VulkanRayRefitRecording(BackendCommandRecording):
    """One vertex-only BLAS update for a :class:`TriangleScene`."""

    def __init__(self, scene, *, vertices="vertices"):
        if not isinstance(scene, TriangleScene):
            raise TypeError("Vulkan ray refit recording requires a TriangleScene")
        if not isinstance(vertices, str) or not vertices:
            raise ValueError("Vulkan ray refit binding name must be a nonempty string")
        super().__init__(
            backend="vulkan",
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
        validate_exact_bindings(self, bindings, "Vulkan ray refit")
        self.validate_graph_lifetime()
        vertices = bindings[self.vertices]
        if _item_count(vertices, 3, f32, self.vertices) != self.scene.vertex_count:
            raise TaichiRuntimeError(
                f"Vulkan ray binding {self.vertices!r} has the wrong vertex count"
            )
        with hardware_failure_phase("provider_execution_failure"):
            self.scene._execute_refit(vertices)

    def validate_graph_lifetime(self):
        self.scene._validate_lifetime()

    def memory_report(self):
        return self.scene.memory_report()

    def _as_graph_native_node(self):
        return native_recording_node(
            self,
            lifetime_leases=lambda item: (item.scene,),
            debug_info=lambda item: {
                "kind": "vulkan_triangle_ray_refit",
                "vertex_count": item.scene.vertex_count,
                "scene_kind": "updatable_triangle_blas_tlas",
            },
        )


class _TypedRayScene:
    # Every native scene operation resolves a live handle under the Program's
    # submission gate. A second Python lifetime check cannot extend that lease.
    graph_runtime_lifetime_check_required = False

    def record_typed(
        self, ray_count, *, rays="rays", hits="hits", hit_indices="hit_indices"
    ):
        """Record typed triangle hits without an intermediate conversion kernel.

        ``hits`` is f32 ``(N, 4)``: (t, u, v, reserved=0).
        ``hit_indices`` is i32/u32 ``(N, 4)``: (primitive, instance, custom, hit).
        Instance is the zero-based TLAS ordinal, not the application custom ID.
        Misses write (-1, 0, 0, 0) and (-1, -1, -1, 0), respectively;
        u32 uses UINT32_MAX for absent indices. u32 preserves the full unsigned
        index range; i32 interprets those same 32 bits as signed integers.
        Vector-4 AOS arrays are also supported. ``t`` is the ray parameter;
        it is a metric distance only for a unit direction. Triangle weights are
        (1-u-v, u, v). The legacy :meth:`record` output is unchanged.
        """
        self._validate_lifetime()
        return VulkanRayQueryRecording(
            self, ray_count, rays=rays, hits=hits, hit_indices=hit_indices
        )

    def trace_typed(self, rays, hits, hit_indices):
        """Execute :meth:`record_typed` into caller-owned device arrays."""
        recording = self.record_typed(_item_count(rays, 8, f32, "rays"))
        recording.execute({"rays": rays, "hits": hits, "hit_indices": hit_indices})
        return hits, hit_indices


class TriangleScene(_TypedRayScene):
    """One updatable triangle BLAS and one identity-instance TLAS.

    ``vertices`` and ``indices`` accept scalar ``(N, 3)`` ndarrays or AOS
    vector-3 ndarrays with shape ``(N,)``. Indices are signed i32 for parity
    with Forge mesh storage but must all be nonnegative and in range; this
    low-level provider does not perform a host readback to validate them.

    :meth:`refit` updates vertex positions in hardware without rebuilding the
    topology. The vertex count and indices are fixed for the scene lifetime.
    """

    def __init__(self, vertices, indices):
        program = impl.get_runtime().prog
        if program is None:
            raise TaichiRuntimeError(
                "TriangleScene requires an initialized Taichi runtime"
            )
        if active_backend() != "vulkan":
            raise TaichiRuntimeError(
                "TriangleScene requires the Vulkan backend; the active backend is "
                f"{active_backend()}"
            )
        if not program.vulkan_ray_query_available():
            raise TaichiRuntimeError(
                "TriangleScene requires VK_KHR_acceleration_structure and "
                "VK_KHR_ray_query"
            )
        vertex_count = _item_count(vertices, 3, f32, "vertices")
        triangle_count = _item_count(indices, 3, i32, "indices")
        self._runtime_prog = program
        self._runtime_generation = int(impl.runtime_generation())
        self._handle = int(
            program._create_vulkan_triangle_ray_scene(
                vertices.arr,
                indices.arr,
                vertex_count,
                triangle_count,
            )
        )
        self.vertex_count = vertex_count
        self.triangle_count = triangle_count
        self._effect_name = (
            f"vulkan-ray-scene:{self._runtime_generation}:{self._handle}"
        )
        self._scene_kind = "updatable_triangle_blas_tlas"
        self._memory_stats = dict(
            program._vulkan_triangle_ray_scene_memory_stats(self._handle)
        )

    @property
    def closed(self):
        return self._handle is None

    def record(self, ray_count, *, rays="rays", hits="hits"):
        self._validate_lifetime()
        return VulkanRayQueryRecording(
            self, ray_count, rays=rays, hits=hits
        )

    def trace(self, rays, hits):
        ray_count = _item_count(rays, 8, f32, "rays")
        recording = self.record(ray_count)
        recording.execute({"rays": rays, "hits": hits})
        return hits

    def record_refit(self, *, vertices="vertices"):
        self._validate_lifetime()
        return VulkanRayRefitRecording(self, vertices=vertices)

    def refit(self, vertices):
        recording = self.record_refit()
        recording.execute({"vertices": vertices})
        return self

    def _execute_query(self, rays, hits, ray_count, hit_indices=None):
        self._validate_lifetime()
        self._runtime_prog._vulkan_triangle_ray_query(
            self._handle,
            rays.arr,
            hits.arr,
            ray_count,
            None if hit_indices is None else hit_indices.arr,
        )

    def _execute_refit(self, vertices):
        self._validate_lifetime()
        self._runtime_prog._vulkan_triangle_ray_refit(
            self._handle, vertices.arr, self.vertex_count
        )

    def _validate_lifetime(self):
        if self._handle is None:
            raise TaichiRuntimeError("TriangleScene has been closed")
        validate_runtime_generation(
            self,
            "TriangleScene belongs to a previous Taichi runtime generation",
        )

    def validate_graph_lifetime(self):
        self._validate_lifetime()

    def memory_report(self):
        """Return exact requested buffers and explicitly opaque driver state."""

        handle_present = self._handle is not None
        runtime_valid = handle_present and runtime_generation_matches(self)
        resident = runtime_valid
        stats = self._memory_stats
        components = (
            HardwareMemoryComponent(
                "geometry_build_inputs",
                int(stats["geometry_input_requested_bytes"]),
                True,
                "provider_generation",
                "provider",
                resident=resident,
            ),
            HardwareMemoryComponent(
                "blas_tlas_storage",
                int(stats["acceleration_structure_requested_bytes"]),
                True,
                "provider_generation",
                "provider",
                resident=resident,
            ),
            HardwareMemoryComponent(
                "build_refit_scratch",
                int(stats["build_scratch_requested_bytes"]),
                True,
                "provider_generation",
                "provider",
                resident=resident,
            ),
            HardwareMemoryComponent(
                "pipeline_descriptors_and_driver_state",
                None,
                False,
                "provider_generation",
                "driver",
                resident=resident,
            ),
        )
        return make_memory_report(
            "vulkan_triangle_ray",
            "vulkan",
            components,
            lifecycle_state=(
                "ready"
                if runtime_valid
                else "closed"
                if not handle_present
                else "runtime_invalid"
            ),
            ownership_scope="scene_generation",
        )

    def _graph_provider_memory_report(self):
        return self.memory_report()

    def close(self):
        if self._handle is None:
            return None
        handle = self._handle
        self._handle = None
        if runtime_generation_matches(self):
            self._runtime_prog._destroy_vulkan_triangle_ray_scene(handle)
        return None

    destroy = close

    def __enter__(self):
        self._validate_lifetime()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


@instrument_hardware_recording("ray.as_build.vulkan")
class VulkanBLASBuildRecording(BackendCommandRecording):
    """One explicit triangle BLAS rebuild with fixed allocation shape."""

    def __init__(self, blas, *, vertices="vertices", indices="indices"):
        if not isinstance(blas, TriangleBLAS):
            raise TypeError("Vulkan BLAS build recording requires a TriangleBLAS")
        if any(
            not isinstance(name, str) or not name
            for name in (vertices, indices)
        ):
            raise ValueError("Vulkan BLAS binding names must be nonempty strings")
        if vertices == indices:
            raise ValueError("Vulkan BLAS binding names must be unique")
        super().__init__(
            backend="vulkan",
            binding_names=(vertices, indices),
            command_count=1,
            queue="compute",
            stream_binding="runtime_ordered",
            barrier_policy="internal",
            workspace_ownership="provider_generation",
            replay_mode="rerecord",
            no_host_readback=True,
        )
        object.__setattr__(self, "blas", blas)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "indices", indices)

    @property
    def resource_effects(self):
        return (
            ResourceEffect(self.vertices, GraphAccess.READ),
            ResourceEffect(self.indices, GraphAccess.READ),
            static_resource_effect(self.blas._effect_name, GraphAccess.WRITE),
        )

    @property
    def lifetime_leases(self):
        return (self.blas,)

    @property
    def debug_info(self):
        return {
            "kind": "vulkan_triangle_blas_build",
            "vertex_count": self.blas.vertex_count,
            "triangle_count": self.blas.triangle_count,
        }

    def execute(self, bindings):
        validate_exact_bindings(self, bindings, "Vulkan BLAS build")
        self.validate_graph_lifetime()
        vertices = bindings[self.vertices]
        indices = bindings[self.indices]
        if _item_count(vertices, 3, f32, self.vertices) != self.blas.vertex_count:
            raise TaichiRuntimeError(
                f"Vulkan ray binding {self.vertices!r} has the wrong vertex count"
            )
        if _item_count(indices, 3, i32, self.indices) != self.blas.triangle_count:
            raise TaichiRuntimeError(
                f"Vulkan ray binding {self.indices!r} has the wrong triangle count"
            )
        with hardware_failure_phase("provider_execution_failure"):
            self.blas._execute_build(vertices, indices, update=False)

    def validate_graph_lifetime(self):
        self.blas._validate_lifetime()

    def memory_report(self):
        return self.blas.memory_report()

    def _as_graph_native_node(self):
        return native_recording_node(
            self,
            lifetime_leases=lambda item: item.lifetime_leases,
            debug_info=lambda item: item.debug_info,
        )


@instrument_hardware_recording("ray.as_refit.vulkan")
class VulkanBLASRefitRecording(BackendCommandRecording):
    """One vertex-only triangle BLAS update with fixed topology."""

    def __init__(self, blas, *, vertices="vertices"):
        if not isinstance(blas, TriangleBLAS):
            raise TypeError("Vulkan BLAS refit recording requires a TriangleBLAS")
        if not isinstance(vertices, str) or not vertices:
            raise ValueError("Vulkan BLAS binding name must be a nonempty string")
        super().__init__(
            backend="vulkan",
            binding_names=(vertices,),
            command_count=1,
            queue="compute",
            stream_binding="runtime_ordered",
            barrier_policy="internal",
            workspace_ownership="provider_generation",
            replay_mode="rerecord",
            no_host_readback=True,
        )
        object.__setattr__(self, "blas", blas)
        object.__setattr__(self, "vertices", vertices)

    @property
    def resource_effects(self):
        return (
            ResourceEffect(self.vertices, GraphAccess.READ),
            static_resource_effect(self.blas._effect_name, GraphAccess.WRITE),
        )

    @property
    def lifetime_leases(self):
        return (self.blas,)

    @property
    def debug_info(self):
        return {
            "kind": "vulkan_triangle_blas_refit",
            "vertex_count": self.blas.vertex_count,
        }

    def execute(self, bindings):
        validate_exact_bindings(self, bindings, "Vulkan BLAS refit")
        self.validate_graph_lifetime()
        vertices = bindings[self.vertices]
        if _item_count(vertices, 3, f32, self.vertices) != self.blas.vertex_count:
            raise TaichiRuntimeError(
                f"Vulkan ray binding {self.vertices!r} has the wrong vertex count"
            )
        with hardware_failure_phase("provider_execution_failure"):
            self.blas._execute_build(vertices, None, update=True)

    def validate_graph_lifetime(self):
        self.blas._validate_lifetime()

    def memory_report(self):
        return self.blas.memory_report()

    def _as_graph_native_node(self):
        return native_recording_node(
            self,
            lifetime_leases=lambda item: item.lifetime_leases,
            debug_info=lambda item: item.debug_info,
        )


class TriangleBLAS:
    """Independent fixed-topology Vulkan triangle BLAS resource."""

    def __init__(self, vertices, indices):
        program = _require_vulkan_ray_runtime("TriangleBLAS")
        vertex_count = _item_count(vertices, 3, f32, "vertices")
        triangle_count = _item_count(indices, 3, i32, "indices")
        self._runtime_prog = program
        self._runtime_generation = int(impl.runtime_generation())
        self.vertex_count = vertex_count
        self.triangle_count = triangle_count
        self._handle = int(
            program._create_vulkan_triangle_blas_resource(
                vertex_count, triangle_count
            )
        )
        self._effect_name = (
            f"vulkan-ray-blas:{self._runtime_generation}:{self._handle}"
        )
        try:
            self._memory_stats = dict(
                program._vulkan_ray_resource_memory_stats(self._handle)
            )
            self.build(vertices, indices)
        except Exception:
            self.close()
            raise

    @property
    def closed(self):
        return self._handle is None

    def record_build(self, *, vertices="vertices", indices="indices"):
        self._validate_lifetime()
        return VulkanBLASBuildRecording(
            self, vertices=vertices, indices=indices
        )

    def build(self, vertices, indices):
        self.record_build().execute({"vertices": vertices, "indices": indices})
        return self

    def record_refit(self, *, vertices="vertices"):
        self._validate_lifetime()
        return VulkanBLASRefitRecording(self, vertices=vertices)

    def refit(self, vertices):
        self.record_refit().execute({"vertices": vertices})
        return self

    def _execute_build(self, vertices, indices, *, update):
        self._validate_lifetime()
        self._runtime_prog._vulkan_triangle_blas_build(
            self._handle,
            vertices.arr,
            None if indices is None else indices.arr,
            self.vertex_count,
            self.triangle_count,
            bool(update),
        )

    def _validate_runtime_identity(self):
        validate_runtime_generation(
            self,
            "TriangleBLAS belongs to a previous Taichi runtime generation",
        )

    def _validate_lifetime(self):
        if self._handle is None:
            raise TaichiRuntimeError("TriangleBLAS has been closed")
        self._validate_runtime_identity()

    def validate_graph_lifetime(self):
        self._validate_lifetime()

    def memory_report(self):
        return _independent_ray_memory_report(
            self,
            provider="vulkan_triangle_blas",
            geometry_name="geometry_build_inputs",
            storage_name="blas_storage",
        )

    def _graph_provider_memory_report(self):
        return self.memory_report()

    def close(self):
        if self._handle is None:
            return None
        handle = self._handle
        self._handle = None
        if runtime_generation_matches(self):
            self._runtime_prog._destroy_vulkan_ray_resource(handle)
        return None

    destroy = close

    def __enter__(self):
        self._validate_lifetime()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


_IDENTITY_TRANSFORM_3X4 = (
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
)


def _normalize_transform(transform):
    try:
        values = tuple(transform)
    except TypeError as exc:
        raise TypeError("Vulkan ray instance transform must be iterable") from exc
    if len(values) == 3 and all(hasattr(row, "__iter__") for row in values):
        values = tuple(value for row in values for value in row)
    if len(values) != 12:
        raise ValueError(
            "Vulkan ray instance transform must be a row-major 3x4 matrix"
        )
    try:
        values = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise TypeError("Vulkan ray instance transform must be numeric") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Vulkan ray instance transform must be finite")
    return values


@dataclass(frozen=True)
class RayInstance:
    """One low-level TLAS instance descriptor with a fixed BLAS reference."""

    blas: TriangleBLAS
    transform: tuple = _IDENTITY_TRANSFORM_3X4
    mask: int = 0xFF
    custom_index: int = 0

    def __post_init__(self):
        if not isinstance(self.blas, TriangleBLAS):
            raise TypeError("Vulkan ray instance blas must be a TriangleBLAS")
        if isinstance(self.mask, bool) or not isinstance(self.mask, int):
            raise TypeError("Vulkan ray instance mask must be an integer")
        if self.mask < 0 or self.mask > 0xFF:
            raise ValueError("Vulkan ray instance mask must be in [0, 255]")
        if isinstance(self.custom_index, bool) or not isinstance(
            self.custom_index, int
        ):
            raise TypeError("Vulkan ray instance custom_index must be an integer")
        if self.custom_index < 0 or self.custom_index > 0xFFFFFF:
            raise ValueError(
                "Vulkan ray instance custom_index must be in [0, 16777215]"
            )
        object.__setattr__(self, "transform", _normalize_transform(self.transform))

    def _to_core(self):
        result = _ti_core.VulkanRayInstanceInfo()
        result.transform = self.transform
        result.mask = self.mask
        result.custom_index = self.custom_index
        return result


class _VulkanTLASRecording(BackendCommandRecording):
    def __init__(self, tlas, instances, *, update):
        if not isinstance(tlas, InstanceTLAS):
            raise TypeError("Vulkan TLAS recording requires an InstanceTLAS")
        normalized = tlas._normalize_instances(instances, require_live=False)
        super().__init__(
            backend="vulkan",
            binding_names=(),
            command_count=1,
            queue="compute",
            stream_binding="runtime_ordered",
            barrier_policy="internal",
            workspace_ownership="provider_generation",
            replay_mode="rerecord",
            no_host_readback=True,
        )
        object.__setattr__(self, "tlas", tlas)
        object.__setattr__(self, "instances", normalized)
        object.__setattr__(self, "update", bool(update))

    @property
    def resource_effects(self):
        effects = [
            static_resource_effect(instance.blas._effect_name, GraphAccess.READ)
            for instance in self.instances
        ]
        effects.append(
            static_resource_effect(self.tlas._effect_name, GraphAccess.WRITE)
        )
        return tuple(effects)

    @property
    def lifetime_leases(self):
        return (self.tlas,)

    @property
    def debug_info(self):
        return {
            "kind": (
                "vulkan_instance_tlas_refit"
                if self.update
                else "vulkan_instance_tlas_build"
            ),
            "instance_count": len(self.instances),
            "topology_fixed": True,
        }

    def execute(self, bindings):
        validate_exact_bindings(self, bindings, "Vulkan TLAS build")
        self.validate_graph_lifetime()
        with hardware_failure_phase("provider_execution_failure"):
            self.tlas._execute_build(self.instances, update=self.update)

    def validate_graph_lifetime(self):
        self.tlas._validate_lifetime()
        self.tlas._validate_topology(self.instances)

    def memory_report(self):
        return self.tlas.memory_report()

    def _as_graph_native_node(self):
        return native_recording_node(
            self,
            lifetime_leases=lambda item: item.lifetime_leases,
            debug_info=lambda item: item.debug_info,
        )


@instrument_hardware_recording("ray.as_build.vulkan")
class VulkanTLASBuildRecording(_VulkanTLASRecording):
    """One explicit TLAS rebuild with captured instance descriptors."""

    def __init__(self, tlas, instances):
        super().__init__(tlas, instances, update=False)


@instrument_hardware_recording("ray.as_refit.vulkan")
class VulkanTLASRefitRecording(_VulkanTLASRecording):
    """One TLAS update with fixed BLAS order and captured descriptors."""

    def __init__(self, tlas, instances):
        super().__init__(tlas, instances, update=True)


@dataclass(frozen=True)
class _KernelAccelerationStructureDescriptor:
    """Generation-qualified internal contract for a kernel-visible TLAS.

    This descriptor deliberately carries no query operations and remains a
    private resource/lifetime binding contract.  Kernel code sees only the
    typed, non-escaping acceleration-structure accessor.
    """

    owner: object
    runtime_generation: int
    handle: int
    instance_count: int
    effect_name: str

    @property
    def resource_effects(self):
        return (static_resource_effect(self.effect_name, GraphAccess.READ),)

    def validate_lifetime(self):
        self.owner._validate_lifetime()
        if (
            int(self.owner._runtime_generation) != self.runtime_generation
            or int(self.owner._handle) != self.handle
        ):
            raise TaichiRuntimeError(
                "Vulkan kernel acceleration-structure descriptor is stale"
            )


class InstanceTLAS(_TypedRayScene):
    """Independent Vulkan TLAS with fixed BLAS topology and mutable metadata."""

    def __init__(self, instances):
        program = _require_vulkan_ray_runtime("InstanceTLAS")
        self._runtime_prog = program
        self._runtime_generation = int(impl.runtime_generation())
        normalized = self._normalize_instances(instances, require_live=True)
        self._topology = tuple(instance.blas for instance in normalized)
        self._instances = normalized
        self._handle = int(
            program._create_vulkan_instance_tlas_resource(
                [blas._handle for blas in self._topology]
            )
        )
        self._effect_name = (
            f"vulkan-ray-tlas:{self._runtime_generation}:{self._handle}"
        )
        self._scene_kind = "independent_instance_tlas"
        try:
            self._memory_stats = dict(
                program._vulkan_ray_resource_memory_stats(self._handle)
            )
            self.build(normalized)
        except Exception:
            self.close()
            raise

    @property
    def closed(self):
        return self._handle is None

    @property
    def instance_count(self):
        return len(self._topology)

    def _normalize_instances(self, instances, *, require_live):
        try:
            normalized = tuple(instances)
        except TypeError as exc:
            raise TypeError("Vulkan TLAS instances must be iterable") from exc
        if not normalized:
            raise ValueError("Vulkan TLAS requires at least one RayInstance")
        for instance in normalized:
            if not isinstance(instance, RayInstance):
                raise TypeError("Vulkan TLAS entries must be RayInstance objects")
            if require_live:
                instance.blas._validate_lifetime()
            else:
                instance.blas._validate_runtime_identity()
            if instance.blas._runtime_prog is not self._runtime_prog:
                raise TaichiRuntimeError(
                    "Vulkan TLAS and all BLAS resources must share one runtime"
                )
        return normalized

    def _validate_topology(self, instances):
        if len(instances) != len(self._topology) or any(
            instance.blas is not expected
            for instance, expected in zip(instances, self._topology)
        ):
            raise TaichiRuntimeError(
                "Vulkan TLAS build/refit must preserve BLAS count and order"
            )

    def record_build(self, instances=None):
        self._validate_lifetime()
        return VulkanTLASBuildRecording(
            self, self._instances if instances is None else instances
        )

    def build(self, instances=None):
        recording = self.record_build(instances)
        recording.execute({})
        self._instances = recording.instances
        return self

    def record_refit(self, instances=None):
        self._validate_lifetime()
        return VulkanTLASRefitRecording(
            self, self._instances if instances is None else instances
        )

    def refit(self, instances=None):
        recording = self.record_refit(instances)
        recording.execute({})
        self._instances = recording.instances
        return self

    def record(self, ray_count, *, rays="rays", hits="hits"):
        self._validate_lifetime()
        return VulkanRayQueryRecording(
            self, ray_count, rays=rays, hits=hits
        )

    def trace(self, rays, hits):
        ray_count = _item_count(rays, 8, f32, "rays")
        recording = self.record(ray_count)
        recording.execute({"rays": rays, "hits": hits})
        return hits

    def _kernel_resource_descriptor(self):
        self._validate_lifetime()
        properties = dict(
            self._runtime_prog._vulkan_ray_kernel_resource_properties(
                self._handle
            )
        )
        if not (
            properties.get("top_level") == 1
            and properties.get("read_only") == 1
            and properties.get("exact_generation") == 1
            and properties.get("instance_count") == self.instance_count
        ):
            raise TaichiRuntimeError(
                "Vulkan kernel acceleration-structure contract is inconsistent"
            )
        return _KernelAccelerationStructureDescriptor(
            owner=self,
            runtime_generation=self._runtime_generation,
            handle=self._handle,
            instance_count=self.instance_count,
            effect_name=self._effect_name,
        )

    def _execute_build(self, instances, *, update):
        self._validate_lifetime()
        self._validate_topology(instances)
        self._runtime_prog._vulkan_instance_tlas_build(
            self._handle,
            [instance._to_core() for instance in instances],
            bool(update),
        )

    def _execute_query(self, rays, hits, ray_count, hit_indices=None):
        self._validate_lifetime()
        self._runtime_prog._vulkan_instance_tlas_query(
            self._handle,
            rays.arr,
            hits.arr,
            ray_count,
            None if hit_indices is None else hit_indices.arr,
        )

    def _validate_lifetime(self):
        if self._handle is None:
            raise TaichiRuntimeError("InstanceTLAS has been closed")
        validate_runtime_generation(
            self,
            "InstanceTLAS belongs to a previous Taichi runtime generation",
        )

    def validate_graph_lifetime(self):
        self._validate_lifetime()

    def memory_report(self):
        return _independent_ray_memory_report(
            self,
            provider="vulkan_instance_tlas",
            geometry_name="instance_build_inputs",
            storage_name="tlas_storage",
        )

    def _graph_provider_memory_report(self):
        return self.memory_report()

    def close(self):
        if self._handle is None:
            return None
        handle = self._handle
        self._handle = None
        if runtime_generation_matches(self):
            self._runtime_prog._destroy_vulkan_ray_resource(handle)
        return None

    destroy = close

    def __enter__(self):
        self._validate_lifetime()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def _require_vulkan_ray_runtime(resource_name):
    program = impl.get_runtime().prog
    if program is None:
        raise TaichiRuntimeError(
            f"{resource_name} requires an initialized Taichi runtime"
        )
    if active_backend() != "vulkan":
        raise TaichiRuntimeError(
            f"{resource_name} requires the Vulkan backend; the active backend "
            f"is {active_backend()}"
        )
    if not program.vulkan_ray_query_available():
        raise TaichiRuntimeError(
            f"{resource_name} requires VK_KHR_acceleration_structure and "
            "VK_KHR_ray_query"
        )
    return program


def _independent_ray_memory_report(
    resource, *, provider, geometry_name, storage_name
):
    handle_present = resource._handle is not None
    runtime_valid = handle_present and runtime_generation_matches(resource)
    stats = resource._memory_stats
    components = (
        HardwareMemoryComponent(
            geometry_name,
            int(stats["geometry_input_requested_bytes"]),
            True,
            "provider_generation",
            "provider",
            resident=runtime_valid,
        ),
        HardwareMemoryComponent(
            storage_name,
            int(stats["acceleration_structure_requested_bytes"]),
            True,
            "provider_generation",
            "provider",
            resident=runtime_valid,
        ),
        HardwareMemoryComponent(
            "build_refit_scratch",
            int(stats["build_scratch_requested_bytes"]),
            True,
            "provider_generation",
            "provider",
            resident=runtime_valid,
        ),
        HardwareMemoryComponent(
            "pipeline_descriptors_and_driver_state",
            None,
            False,
            "provider_generation",
            "driver",
            resident=runtime_valid,
        ),
    )
    return make_memory_report(
        provider,
        "vulkan",
        components,
        lifecycle_state=(
            "ready"
            if runtime_valid
            else "closed"
            if not handle_present
            else "runtime_invalid"
        ),
        ownership_scope="resource_generation",
    )


def is_available():
    """Return whether the active runtime supports the complete Vulkan slice."""

    program = impl.get_runtime().prog
    return bool(
        program is not None
        and active_backend() == "vulkan"
        and program.vulkan_ray_query_available()
    )


__all__ = [
    "InstanceTLAS",
    "RayInstance",
    "TriangleBLAS",
    "TriangleScene",
    "VulkanBLASBuildRecording",
    "VulkanBLASRefitRecording",
    "VulkanRayQueryRecording",
    "VulkanRayRefitRecording",
    "VulkanTLASBuildRecording",
    "VulkanTLASRefitRecording",
    "is_available",
]
