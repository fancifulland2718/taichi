import itertools
import math
import os
import threading
import time
import warnings
import weakref
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from taichi_forge._lib import core as _ti_core
from taichi_forge._contracts import (
    DYNAMIC_WORK_SCHEMA_VERSION,
    GRAPH_PIPELINE_SCHEMA_VERSION,
    STRUCTURED_CONTROL_SCHEMA_VERSION,
)
from taichi_forge.aot.utils import produce_injected_args_for_graph
from taichi_forge.lang import enums, impl, kernel_impl, ops
from taichi_forge.lang._ndarray import Ndarray, ScalarNdarray
from taichi_forge.lang._storage_view import (
    DenseNdarrayView,
    StorageDescription,
    analyze_storage_alias,
    describe_storage,
    ndarray_view,
    validate_storage_owner,
)
from taichi_forge.lang._texture import Texture
from taichi_forge.lang.exception import TaichiCompilationError, TaichiRuntimeError
from taichi_forge.lang.field import ScalarField
from taichi_forge.lang.util import to_numpy_type
from taichi_forge.lang.matrix import Matrix, MatrixField, MatrixType
from taichi_forge.types._argument_descriptor import (
    describe_element_type,
)
from taichi_forge.types import ndarray_type
from taichi_forge.types.annotations import template
from taichi_forge.types.primitive_types import f32, i32, u32
from taichi_forge.types.texture_type import FORMAT2TY_CH, TY_CH2FORMAT
from taichi_forge.graph._native import (
    BackendCommandGraphAction,
    BoundedPublicationTarget,
    GraphTemporaryBuffer,
    NativeActionManifest,
    PreparedGraphBindings,
    ProviderOwnedNdarrayBinding,
    RecordableGraphAction,
    VulkanBufferCommand,
    VulkanBufferCommandRecording,
    _CudaGraphCaptureRecipe,
    compile_native_graph_node,
    native_action_manifest,
)
from taichi_forge.graph._ir import (
    BoundedDomain,
    DispatchNode,
    IfRegion,
    GraphAccess,
    InternalNdarrayRequirement,
    NativeCallNode,
    ObservationNode,
    ParallelBranchSummary,
    ParallelEffectDependency,
    ResourceEffect,
    RuntimeBinding,
    SequentialRegion,
    SwitchRegion,
    WhileRegion,
    analyze_elementwise_fusion,
    analyze_graph_ir,
    analyze_parallel_candidate,
    graph_ir_to_dict,
    plan_temporary_memory,
)
from taichi_forge.graph._submission import (
    SubmissionPacer,
    _new_submission_lane,
    _reserve_paced_submission,
)
from taichi_forge.graph._optimization import (
    _CUDA_CONDITIONAL_CONTROL_RECIPE_ID,
    _CUDA_CONTROL_RECIPE_IDS,
    _CUDA_MASKED_CONTROL_RECIPE_ID,
    _CUDA_NESTED_CONTROL_RECIPE_IDS,
    _CUDA_NESTED_DEVICE_UPDATE_CONTROL_RECIPE_ID,
    _CUDA_NESTED_MASKED_CONTROL_RECIPE_ID,
    _GraphBoundedExecutionRecipeManifest,
    _GraphFusionQualificationCache,
    _GraphMemoryRecipeManifest,
    _GraphOffloadFusionRecipeManifest,
    _GraphSparseTraversalRecipeManifest,
    _build_executable_optimization_space,
)
from taichi_forge.graph._recipes import GraphDefinition

ArgKind = _ti_core.ArgKind

_INTERNAL_FUSION_QUALIFICATION_ENV = "TAICHI_FORGE_INTERNAL_GRAPH_FUSION_QUALIFICATION"
_INTERNAL_FUSION_EXPECTED_REPLAYS_ENV = (
    "TAICHI_FORGE_INTERNAL_GRAPH_FUSION_EXPECTED_REPLAYS"
)
_CUDA_NESTED_DEVICE_UPDATE_ROUTE = "cuda_device_node_update"
_CUDA_NESTED_MASKED_ROUTE = "cuda_masked_bounded_graph"


@kernel_impl.kernel
def _prepare_bounded_dispatch_packet(
    extent_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    packet: ndarray_type.ndarray(dtype=u32, ndim=1),
    capacity: i32,
    block_dim: i32,
):
    # Keep the extent load inside the range task.  A scalar statement before
    # the loop would introduce an unnecessary serial offload.
    for _ in range(1):
        raw_count = extent_state[0]
        count = raw_count
        if count < 0:
            count = 0
            extent_state[1] = 1
        elif count > capacity:
            count = capacity
            extent_state[1] = 1
        extent_state[0] = count
        packet[0] = ops.cast(count // block_dim, u32) + ops.cast(
            count % block_dim != 0, u32
        )
        packet[1] = 1
        packet[2] = 1


def _vulkan_max_compute_work_group_count_x():
    program = impl.get_runtime().prog
    query = getattr(program, "_vulkan_max_compute_work_group_count_x", None)
    if not callable(query):
        raise TaichiRuntimeError(
            "Vulkan indirect dispatch cannot prove the active device compute "
            "work-group limit"
        )
    try:
        maximum = int(query())
    except Exception as exc:
        raise TaichiRuntimeError(
            "Vulkan indirect dispatch could not query the active device compute "
            "work-group limit"
        ) from exc
    if maximum <= 0:
        raise TaichiRuntimeError(
            "Vulkan indirect dispatch received an invalid active-device "
            "compute work-group limit"
        )
    return maximum


def _validate_vulkan_indirect_grid_capacity(capacity, block_dim):
    grid_x = capacity // block_dim + int(capacity % block_dim != 0)
    maximum = _vulkan_max_compute_work_group_count_x()
    if grid_x > maximum:
        raise TaichiRuntimeError(
            "Vulkan indirect dispatch capacity requires "
            f"{grid_x} work groups on x, exceeding the active device limit "
            f"of {maximum}"
        )


@kernel_impl.func
def _prepare_ordered_segment_state_body(
    offsets: ndarray_type.ndarray(dtype=i32, ndim=1),
    extent_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    segment_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    segment_index: i32,
    segment_count: i32,
    capacity: i32,
):
    if segment_index == 0:
        segment_state[4] = 0
    raw_count = extent_state[0]
    count = raw_count
    if count < 0:
        count = 0
        extent_state[1] = 1
    elif count > capacity:
        count = capacity
        extent_state[1] = 1
    extent_state[0] = count

    raw_begin = offsets[segment_index]
    raw_end = offsets[segment_index + 1]
    begin = raw_begin
    end = raw_end
    invalid = raw_begin < 0 or raw_end < raw_begin or raw_end > count
    if segment_index == 0 and raw_begin != 0:
        invalid = True
    if segment_index + 1 == segment_count and raw_end != count:
        invalid = True
    if begin < 0:
        begin = 0
    if begin > count:
        begin = count
    if end < begin:
        end = begin
    if end > count:
        end = count
    if invalid:
        segment_state[4] = 1

    segment_state[0] = begin
    segment_state[1] = end
    segment_state[2] = segment_index
    segment_state[3] = segment_count


@kernel_impl.kernel
def _prepare_ordered_segment_dispatch(
    offsets: ndarray_type.ndarray(dtype=i32, ndim=1),
    extent_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    packet: ndarray_type.ndarray(dtype=u32, ndim=1),
    segment_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    segment_index: i32,
    segment_count: i32,
    capacity: i32,
    block_dim: i32,
):
    for _ in range(1):
        _prepare_ordered_segment_state_body(
            offsets,
            extent_state,
            segment_state,
            segment_index,
            segment_count,
            capacity,
        )
        begin = segment_state[0]
        end = segment_state[1]
        segment_length = end - begin
        packet[0] = ops.cast(segment_length // block_dim, u32) + ops.cast(
            segment_length % block_dim != 0, u32
        )
        packet[1] = 1
        packet[2] = 1


@kernel_impl.kernel
def _prepare_ordered_segment_state(
    offsets: ndarray_type.ndarray(dtype=i32, ndim=1),
    extent_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    segment_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    segment_index: i32,
    segment_count: i32,
    capacity: i32,
):
    for _ in range(1):
        _prepare_ordered_segment_state_body(
            offsets,
            extent_state,
            segment_state,
            segment_index,
            segment_count,
            capacity,
        )


@kernel_impl.func
def segmented_dispatch_begin(state: template()):
    """Return the current ordered segment's inclusive begin index."""

    return state[0]


@kernel_impl.func
def segmented_dispatch_end(state: template()):
    """Return the current ordered segment's exclusive end index."""

    return state[1]


@kernel_impl.func
def segmented_dispatch_index(state: template()):
    """Return the current ordered segment index."""

    return state[2]


@kernel_impl.func
def segmented_dispatch_count(state: template()):
    """Return the current ordered segment's bounded item count."""

    return state[1] - state[0]


def _decode_exact_map_partition(value):
    prefix = "exact-v1:"
    if not value.startswith(prefix):
        return None
    if len(value) > 4096:
        raise TaichiRuntimeError("Exact map partition payload is too large")
    payload = value[len(prefix) :]
    if not payload:
        raise TaichiRuntimeError("Exact map partition requires source groups")
    groups = []
    previous_end = -1
    for encoded_group in payload.split(";"):
        fields = encoded_group.split(",")
        if (
            len(fields) < 2
            or len(fields) > 4
            or any(not field.isdigit() for field in fields)
        ):
            raise TaichiRuntimeError(
                "Exact map partition groups require two to four logical IDs"
            )
        group = tuple(int(field) for field in fields)
        if tuple(range(group[0], group[0] + len(group))) != group:
            raise TaichiRuntimeError("Exact map partition groups must be contiguous")
        if group[0] <= previous_end:
            raise TaichiRuntimeError(
                "Exact map partition groups must be ordered and disjoint"
            )
        previous_end = group[-1]
        groups.append(group)
    canonical = prefix + ";".join(
        ",".join(str(item) for item in group) for group in groups
    )
    if canonical != value:
        raise TaichiRuntimeError("Exact map partition encoding is not canonical")
    return tuple(groups)


def _new_runtime_graph_builder(explicit_source_groups=None, map_recipe=None):
    builder = _ti_core.GraphBuilder()
    if explicit_source_groups is not None:
        source_groups = tuple(tuple(group) for group in explicit_source_groups)
        if source_groups:
            builder._set_map_composer_max_group_size(
                max(len(group) for group in source_groups)
            )
            builder._set_map_composer_allowed_groups(source_groups)
        return builder

    internal_recipe = map_recipe
    if internal_recipe is not None:
        internal_recipe = internal_recipe.strip().lower()
        exact_groups = _decode_exact_map_partition(internal_recipe)
        if exact_groups is not None:
            builder._set_map_composer_max_group_size(
                max(len(group) for group in exact_groups)
            )
            builder._set_map_composer_allowed_groups(exact_groups)
            return builder
        recipe_sizes = {
            "baseline": 1,
            "pair": 2,
            "map2": 2,
            "map3": 3,
            "map4": 4,
        }
        if internal_recipe not in recipe_sizes:
            raise TaichiRuntimeError(
                "internal map recipe must be baseline, pair, map3, "
                "map4, or a canonical exact-v1 partition"
            )
        max_group_size = recipe_sizes[internal_recipe]
        if max_group_size == 2:
            builder._enable_two_map_composer()
        elif max_group_size > 2:
            builder._set_map_composer_max_group_size(max_group_size)
        return builder
    composer_setting = os.environ.get("TI_GRAPH_TWO_MAP_COMPOSER")
    composer_enabled = (
        composer_setting != "0"
        if composer_setting is not None
        else impl.current_cfg().arch != _ti_core.Arch.vulkan
    )
    if composer_enabled:
        builder._enable_two_map_composer()
    return builder


def _align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


_GraphInternalNdarraySpec = InternalNdarrayRequirement


class GraphOwnedNdarray(InternalNdarrayRequirement):
    """Declarative storage materialized separately by each Graph instance.

    Use this value in a recordable action's ``fixed_bindings``. The binding is
    private to the compiled Graph and therefore never appears in its public
    runtime argument schema. Mutable storage should retain the default
    exclusive-submission contract so completion fences or workspace lanes
    prevent overlapping replay from aliasing the same allocation.
    """

    def __init__(self, dtype, shape, *, exclusive_submission=True):
        if isinstance(shape, int):
            shape = (shape,)
        super().__init__(
            dtype,
            tuple(shape),
            _ti_core.data_type_size(dtype),
            bool(exclusive_submission),
        )


def _materialize_graph_internal_bindings(bindings, allocators=None):
    allocators = allocators or {}
    materialized = {}
    storage_by_spec = {}
    owned = []
    for name, value in bindings.items():
        if isinstance(value, _GraphInternalNdarraySpec):
            key = id(value)
            storage = storage_by_spec.get(key)
            if storage is None:
                storage = allocators.get(name, ScalarNdarray)(value.dtype, value.shape)
                storage_by_spec[key] = storage
                owned.append(storage)
            materialized[name] = storage
        else:
            materialized[name] = value
    return materialized, tuple(owned)


def _graph_internal_storage_bytes(bindings):
    seen = set()
    total = 0
    for value in bindings.values():
        if not isinstance(value, _GraphInternalNdarraySpec):
            continue
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        total += value.storage_bytes
    return total


class _GraphTemporaryArenaLease:
    def __init__(self, arena, slot):
        self._arena = arena
        self._slot = slot
        self.bindings = slot["bindings"]

    def attach(self, completion):
        if self._slot is None:
            return
        self._slot["completion"] = completion if completion.has_backend_work else None
        self._slot = None

    def cancel(self):
        if self._slot is not None:
            self._slot["completion"] = None
            self._slot = None


class _GraphExclusiveInternalStorageLease:
    def __init__(self, instance):
        self._instance = instance

    def attach(self, completion):
        if self._instance is not None:
            self._instance._attach_exclusive_internal_storage(completion)
            self._instance = None

    def cancel(self):
        if self._instance is not None:
            self._instance._cancel_exclusive_internal_storage()
            self._instance = None


class _GraphTemporaryArena:
    """Runtime-owned byte arenas with a bounded async submission ring."""

    _BASE_ALIGNMENT = 16
    _WORD_BYTES = 4

    def __init__(self, plan, capacity=None):
        self.plan = plan
        self.capacity = int(
            capacity
            if capacity is not None
            else os.environ.get("TI_GRAPH_TEMPORARY_ARENA_SLOTS", "4")
        )
        if self.capacity < 1 or self.capacity > 64:
            raise TaichiRuntimeError(
                "TI_GRAPH_TEMPORARY_ARENA_SLOTS must be between 1 and 64"
            )
        self._slots = []
        self._allocations = 0
        self._reuses = 0
        self._waits = 0
        self._typed_slot_bytes = {}
        for allocation in plan.allocations:
            if allocation.storage_kind == "f32":
                self._typed_slot_bytes[allocation.slot] = max(
                    self._typed_slot_bytes.get(allocation.slot, 0),
                    allocation.bytes,
                )
        self._raw_storage_bytes = max(
            (
                allocation.offset + allocation.bytes
                for allocation in plan.allocations
                if allocation.storage_kind == "raw_i32"
            ),
            default=0,
        )
        self._raw_storage_bytes = _align_up(self._raw_storage_bytes, self._WORD_BYTES)
        # Typed slots are separate allocations, not slices in the raw arena.
        # Report actual requested storage rather than padding in the logical
        # combined liveness plan. Driver pool backing pages are separate data.
        self._storage_bytes = self._raw_storage_bytes + sum(self._typed_slot_bytes.values())
        self._available = bool(plan.allocations) and not (
            plan.conflicting_requirements
            or any(
                allocation.alignment > self._BASE_ALIGNMENT
                for allocation in plan.allocations
            )
        )

    def _new_slot(self, allocator=ScalarNdarray):
        raw_storage = (
            None
            if self._raw_storage_bytes == 0
            else allocator(i32, (self._raw_storage_bytes // self._WORD_BYTES,))
        )
        typed_storage = {
            slot: allocator(f32, (byte_count // self._WORD_BYTES,))
            for slot, byte_count in self._typed_slot_bytes.items()
        }
        bindings = {
            allocation.name: GraphTemporaryBuffer(
                storage=(
                    typed_storage[allocation.slot]
                    if allocation.storage_kind == "f32"
                    else raw_storage
                ),
                offset=(0 if allocation.storage_kind == "f32" else allocation.offset),
                bytes=allocation.bytes,
                alignment=allocation.alignment,
                slot=allocation.slot,
            )
            for allocation in self.plan.allocations
        }
        slot = {
            "storage": (raw_storage, typed_storage),
            "bindings": bindings,
            "completion": None,
        }
        self._slots.append(slot)
        self._allocations += 1
        return slot

    def prepare_storage(self, allocator):
        """Materialize the bounded ring at a recipe's cold setup boundary."""
        if not self._available:
            raise TaichiRuntimeError("Graph temporary arena cannot materialize this storage plan")
        while len(self._slots) < self.capacity:
            self._new_slot(allocator)

    def _reclaim(self):
        for slot in self._slots:
            completion = slot["completion"]
            if completion is not None and completion.done():
                slot["completion"] = None

    def acquire(self):
        if not self.plan.allocations:
            return None
        if not self._available:
            raise TaichiRuntimeError(
                "Graph temporary requirements conflict or exceed the portable "
                "16-byte arena alignment contract"
            )
        self._reclaim()
        for slot in self._slots:
            if slot["completion"] is None:
                self._reuses += 1
                return _GraphTemporaryArenaLease(self, slot)
        if len(self._slots) < self.capacity:
            return _GraphTemporaryArenaLease(self, self._new_slot())
        slot = self._slots[0]
        slot["completion"].wait()
        slot["completion"] = None
        self._waits += 1
        self._reuses += 1
        return _GraphTemporaryArenaLease(self, slot)

    @property
    def stats(self):
        return {
            "materialized": bool(self._slots),
            "capacity": self.capacity if self.plan.allocations else 0,
            "slots": len(self._slots),
            "reserved_bytes": len(self._slots) * self._storage_bytes,
            "allocations": self._allocations,
            "reuses": self._reuses,
            "waits": self._waits,
        }


def _copy_observation_result(result):
    return {batch: dict(values) for batch, values in result.items()}


def _observation_readback_mode():
    if os.environ.get("TI_GRAPH_COMPLETION_ATTACHED_OBSERVATION", "1") == "0":
        return "deferred_device_copy"
    arch = impl.current_cfg().arch
    if arch == _ti_core.Arch.cuda:
        return "completion_attached_pinned_copy"
    if arch in (_ti_core.Arch.x64, _ti_core.Arch.arm64, _ti_core.Arch.vulkan):
        return "completion_attached_host_visible"
    return "deferred_device_copy"


class _GraphObservationState:
    def __init__(self, arena, slot, sequence):
        self._arena = arena
        self._slot = slot
        self._sequence = sequence
        self._completion = None
        self._attached = False
        self._tail_readback_attached = False
        self._result = None
        self._discarded = False
        self._released = False
        self._lock = threading.Lock()

    @property
    def sequence(self):
        return self._sequence

    @property
    def bindings(self):
        return self._slot["bindings"]

    def attach(self, completion):
        with self._lock:
            if self._released:
                raise TaichiRuntimeError(
                    "Cannot attach a completion to a released observation slot"
                )
            self._attached = True
            self._completion = completion if completion.has_backend_work else None
        return self

    def enqueue_tail_readback(self):
        with self._lock:
            if self._released:
                raise TaichiRuntimeError(
                    "Cannot enqueue readback for a released observation slot"
                )
            if self._tail_readback_attached:
                raise TaichiRuntimeError(
                    "Graph observation readback was already enqueued"
                )
            if self._arena.readback_mode == "completion_attached_pinned_copy":
                self._arena._enqueue_slot(self._slot)
                self._tail_readback_attached = True

    def _wait_locked(self):
        if not self._attached:
            if impl.current_cfg().arch in (
                _ti_core.Arch.cuda,
                _ti_core.Arch.vulkan,
            ):
                self._arena._record_wait()
                impl.get_runtime().sync()
            return
        completion = self._completion
        if completion is None:
            return
        if not completion.done():
            self._arena._record_wait()
            completion.wait()

    def materialize(self):
        release = False
        with self._lock:
            if self._result is not None:
                return _copy_observation_result(self._result)
            if self._released:
                raise TaichiRuntimeError("Graph observation snapshot was discarded")
            self._wait_locked()
            result = self._arena._read_slot(
                self._slot,
                tail_readback_attached=self._tail_readback_attached,
            )
            self._result = result
            self._released = True
            release = True
        if release:
            self._arena._release(self._slot, self)
        return _copy_observation_result(result)

    def discard(self):
        release = False
        with self._lock:
            if self._released:
                return
            completion = self._completion
            if completion is None or completion.done():
                self._released = True
                release = True
            else:
                self._discarded = True
        if release:
            self._arena._release(self._slot, self)

    def make_reusable(self):
        release = False
        with self._lock:
            if self._released:
                return
            self._wait_locked()
            if not self._discarded:
                self._result = self._arena._read_slot(
                    self._slot,
                    tail_readback_attached=self._tail_readback_attached,
                )
            self._released = True
            release = True
        if release:
            self._arena._release(self._slot, self)


class _GraphObservationArenaLease:
    def __init__(self, state):
        self._state = state
        self.bindings = state.bindings

    def attach(self, completion):
        if self._state is None:
            return None
        state = self._state.attach(completion)
        self._state = None
        return state

    def enqueue_tail_readback(self):
        if self._state is not None:
            self._state.enqueue_tail_readback()

    def materialize(self):
        if self._state is None:
            return {}
        state = self._state
        self._state = None
        return state.materialize()

    def cancel(self):
        if self._state is not None:
            self._state.discard()
            self._state = None


class _GraphObservationArena:
    """Bounded device snapshot slots with deferred packed host readback."""

    def __init__(self, nodes, capacity=None):
        self.nodes = tuple(nodes)
        self.readback_mode = _observation_readback_mode()
        if not self.nodes:
            self.capacity = 0
            self._slots = []
            self._next_sequence = 1
            self._allocations = 0
            self._reuses = 0
            self._waits = 0
            self._materializations = 0
            self._host_readback_bytes = 0
            self._lock = threading.Lock()
            return
        configured = (
            capacity
            if capacity is not None
            else os.environ.get("TI_GRAPH_OBSERVATION_SLOTS", "4")
        )
        self.capacity = int(configured)
        if self.capacity < 1 or self.capacity > 64:
            raise TaichiRuntimeError(
                "TI_GRAPH_OBSERVATION_SLOTS must be between 1 and 64"
            )
        self._slots = []
        self._next_sequence = 1
        self._allocations = 0
        self._reuses = 0
        self._waits = 0
        self._materializations = 0
        self._host_readback_bytes = 0
        self._lock = threading.Lock()

    def _new_slot(self):
        bindings = {}
        payload_bytes = 0
        for node in self.nodes:
            node_bindings, node_bytes = node.allocate_snapshot_buffers(
                completion_attached=(
                    self.readback_mode == "completion_attached_host_visible"
                )
            )
            bindings[node.name] = node_bindings
            payload_bytes += node_bytes
        slot = {
            "bindings": bindings,
            "payload_bytes": payload_bytes,
            "state": None,
            "cuda_readback": (
                impl.get_runtime().prog._create_cuda_graph_observation_readback(
                    payload_bytes
                )
                if self.readback_mode == "completion_attached_pinned_copy"
                else None
            ),
        }
        self._slots.append(slot)
        self._allocations += 1
        return slot

    def acquire(self):
        if not self.nodes:
            return None
        while True:
            with self._lock:
                allocated = False
                slot = next(
                    (item for item in self._slots if item["state"] is None),
                    None,
                )
                if slot is None and len(self._slots) < self.capacity:
                    slot = self._new_slot()
                    allocated = True
                if slot is not None:
                    if not allocated:
                        self._reuses += 1
                    state = _GraphObservationState(self, slot, self._next_sequence)
                    self._next_sequence += 1
                    slot["state"] = state
                    return _GraphObservationArenaLease(state)
                oldest = min(
                    (item["state"] for item in self._slots),
                    key=lambda state: state.sequence,
                )
            oldest.make_reusable()

    def _slot_sources(self, slot):
        return [
            storage.arr
            for node in self.nodes
            for storage in slot["bindings"][node.name].values()
        ]

    def _slot_hosts(self, slot):
        sources = []
        hosts = []
        host_groups = {}
        for node in self.nodes:
            node_hosts = {}
            for key, storage in slot["bindings"][node.name].items():
                host = np.empty(
                    shape=storage.arr.total_shape(),
                    dtype=to_numpy_type(storage.dtype),
                )
                sources.append(storage.arr)
                hosts.append(host)
                node_hosts[key] = host
            host_groups[node.name] = node_hosts
        return sources, hosts, host_groups

    def _enqueue_slot(self, slot):
        impl.get_runtime().prog._enqueue_cuda_graph_observation_readback(
            slot["cuda_readback"], self._slot_sources(slot)
        )

    def _read_slot(self, slot, *, tail_readback_attached=False):
        sources, hosts, host_groups = self._slot_hosts(slot)
        program = impl.get_runtime().prog
        if self.readback_mode == "completion_attached_host_visible":
            program._copy_host_readable_graph_observations_to_host(sources, hosts)
        elif (
            self.readback_mode == "completion_attached_pinned_copy"
            and tail_readback_attached
        ):
            program._copy_cuda_graph_observation_readback_to_host(
                slot["cuda_readback"], hosts
            )
        else:
            program.copy_graph_observations_to_host(sources, hosts)
        result = {
            node.name: node.decode_snapshot(host_groups[node.name])
            for node in self.nodes
        }
        byte_count = sum(host.nbytes for host in hosts)
        with self._lock:
            self._materializations += 1
            self._host_readback_bytes += byte_count
        return result

    def _record_wait(self):
        with self._lock:
            self._waits += 1

    def _release(self, slot, state):
        with self._lock:
            if slot["state"] is state:
                slot["state"] = None

    @property
    def stats(self):
        with self._lock:
            return {
                "materialized": bool(self._slots),
                "capacity": self.capacity if self.nodes else 0,
                "slots": len(self._slots),
                "reserved_bytes": sum(slot["payload_bytes"] for slot in self._slots),
                "allocations": self._allocations,
                "reuses": self._reuses,
                "waits": self._waits,
                "materializations": self._materializations,
                "host_readback_bytes": self._host_readback_bytes,
                "readback_mode": self.readback_mode,
            }


@kernel_impl.kernel
def _pack_structured_submission_telemetry(
    predicate: ndarray_type.ndarray(dtype=i32, ndim=0),
    counter: ndarray_type.ndarray(dtype=i32, ndim=0),
    status: ndarray_type.ndarray(dtype=i32, ndim=0),
    destination: ndarray_type.ndarray(dtype=i32, ndim=1),
    offset: i32,
):
    destination[offset] = predicate[None]
    destination[offset + 1] = counter[None]
    destination[offset + 2] = status[None]


@kernel_impl.kernel
def _pack_bounded_pipeline_telemetry(
    extent_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    destination: ndarray_type.ndarray(dtype=i32, ndim=1),
    offset: i32,
):
    # This is an opt-in ticket tail snapshot. Keep the two loads in one range
    # task so CUDA/Vulkan do not acquire a host-visible scalar or add separate
    # serial offloads to the payload Graph.
    for _ in range(1):
        destination[offset] = extent_state[0]
        destination[offset + 1] = extent_state[1]


def _queue_submission_snapshot():
    if impl.current_cfg().arch != _ti_core.Arch.vulkan:
        return None
    result = impl.get_runtime().prog._debug_vulkan_queue_submission_stats()
    if not result.get("supported", False):
        return None
    return {
        name: int(result[name])
        for name in (
            "queue_submit_calls",
            "submitted_command_buffers",
            "batched_queue_submit_calls",
            "batched_command_buffers",
        )
    }


def _queue_submission_delta(before, after):
    if before is None or after is None:
        return GraphSubmissionQueueTelemetry(
            available=False,
            scope="unavailable",
            exact=False,
            queue_submit_calls=0,
            submitted_command_buffers=0,
            batched_queue_submit_calls=0,
            batched_command_buffers=0,
        )
    return GraphSubmissionQueueTelemetry(
        available=True,
        scope="vulkan_device_transaction_window",
        # These counters are device-wide. The Graph transaction serializes the
        # runtime compute stream, but external graphics/interop producers may
        # still submit in the same host window, so do not overclaim attribution.
        exact=False,
        **{name: max(0, int(after[name]) - int(before[name])) for name in before},
    )


def _normalize_submission_telemetry_mode(telemetry, owner="Graph.submit()"):
    """Normalize the backwards-compatible submission telemetry contract."""

    if telemetry is False:
        return False
    if telemetry is True or telemetry == "timestamps":
        return "timestamps"
    if telemetry == "summary":
        return "summary"
    raise TaichiRuntimeError(
        f"{owner} telemetry must be False, True, 'summary', or 'timestamps'"
    )


def _structured_submission_metadata(node):
    arch = impl.current_cfg().arch
    backend = _backend_name(_ti_core.arch_name(arch))
    if arch == _ti_core.Arch.vulkan:
        chunk_sizes = tuple(
            min(node.compound_chunk_limit, node.max_iterations - offset)
            for offset in range(0, node.max_iterations, node.compound_chunk_limit)
        )
        strategy_codes = (
            _vulkan_compound_strategy_codes(
                len(chunk_sizes), node.vulkan_first_chunk_strategy
            )
            if chunk_sizes
            else ()
        )
        strategy_names = {
            1: "compact",
            2: "chained",
            3: "conditional",
            4: "coarse_conditional",
        }
        chunk_strategies = tuple(strategy_names[int(code)] for code in strategy_codes)
        lowering = "vulkan_compound_masked"
    elif arch == _ti_core.Arch.cuda:
        chunk_sizes = (node.max_iterations,) if node.max_iterations else ()
        lowering = node._cuda_control_lowering
        chunk_strategies = (lowering,) if node.max_iterations else ()
    else:
        chunk_sizes = (node.max_iterations,) if node.max_iterations else ()
        chunk_strategies = ("portable",) if node.max_iterations else ()
        lowering = "portable"
    return {
        "name": node.name,
        "path_id": node.region_path,
        "control_depth": node.control_depth,
        "backend": backend,
        "lowering": lowering,
        "max_iterations": node.max_iterations,
        "chunk_sizes": chunk_sizes,
        "chunk_strategies": chunk_strategies,
        "has_status": node.status is not None,
    }


def _submission_telemetry_region_nodes(node):
    result = []

    def visit(current):
        if isinstance(current, _CompiledWhileGraphNode):
            result.append(current)
        for _, children in getattr(current, "_definition_children", ()):
            for child in children:
                visit(child)

    visit(node)
    return tuple(result)


class _GraphStructuredTelemetryState:
    def __init__(self, arena, slot, sequence, mode):
        self._arena = arena
        self._slot = slot
        self._sequence = sequence
        self._completion = None
        self._queue = None
        self._submission_statistics = None
        self._host_submit_ns = 0
        self._mode = mode
        self._result = None
        self._discarded = False
        self._released = False
        self._lock = threading.Lock()

    @property
    def sequence(self):
        return self._sequence

    @property
    def recorder(self):
        return self._slot["recorder"]

    def attach(self, completion, queue, host_submit_ns, submission_statistics):
        with self._lock:
            if self._released:
                raise TaichiRuntimeError(
                    "Cannot attach a completion to a released Graph telemetry slot"
                )
            # Keep the completed token as well: short CUDA submissions may be
            # retired by Program's opportunistic collection before Python
            # attaches this slot, but the token still owns the timing sample.
            self._completion = completion
            self._queue = queue
            self._submission_statistics = dict(submission_statistics)
            self._host_submit_ns = int(host_submit_ns)
            self._slot["completion_sequence"] = int(completion.sequence)
        return self

    def _wait_locked(self):
        completion = self._completion
        if completion is None:
            return
        if not completion.done():
            self._arena._record_wait()
            completion.wait()

    def materialize(self):
        release = False
        with self._lock:
            if self._result is not None:
                return self._result
            if self._released:
                raise TaichiRuntimeError("Graph submission telemetry was discarded")
            self._wait_locked()
            result = self._arena._read_slot(
                self._slot,
                self._queue,
                self._host_submit_ns,
                self._completion,
                self._submission_statistics,
                self._mode,
            )
            self._result = result
            self._released = True
            release = True
        if release:
            self._arena._release(self._slot, self)
        return result

    def discard(self):
        release = False
        with self._lock:
            if self._released:
                return
            completion = self._completion
            if completion is None or completion.done():
                self._released = True
                release = True
            else:
                self._discarded = True
        if release:
            self._arena._release(self._slot, self)

    def make_reusable(self):
        release = False
        with self._lock:
            if self._released:
                return
            self._wait_locked()
            if not self._discarded:
                self._result = self._arena._read_slot(
                    self._slot,
                    self._queue,
                    self._host_submit_ns,
                    self._completion,
                    self._submission_statistics,
                    self._mode,
                )
            self._released = True
            release = True
        if release:
            self._arena._release(self._slot, self)


class _GraphStructuredTelemetryRecorder:
    _VALUES_PER_PHASE = 3
    _VALUES_PER_REGION = 6

    def __init__(self, nodes, storage, bounded_sources=()):
        self._nodes = tuple(nodes)
        self._storage = storage
        self._bounded_sources = tuple(bounded_sources)
        self._metadata = [None] * len(self._nodes)
        self._host_started_ns = [0] * len(self._nodes)
        self._host_enqueue_ns = [0] * len(self._nodes)
        self._host_bounded_snapshots = {}
        self._gpu_timing_transaction = None

    @property
    def metadata(self):
        return tuple(self._metadata)

    @property
    def host_enqueue_ns(self):
        return tuple(self._host_enqueue_ns)

    def reset(self):
        self._metadata[:] = [None] * len(self._nodes)
        self._host_started_ns[:] = [0] * len(self._nodes)
        self._host_enqueue_ns[:] = [0] * len(self._nodes)
        self._host_bounded_snapshots.clear()
        self._gpu_timing_transaction = None

    def attach_gpu_timing(self, transaction):
        if self._gpu_timing_transaction is not None:
            raise TaichiRuntimeError(
                "Graph submission telemetry timing is already attached"
            )
        self._gpu_timing_transaction = transaction

    def detach_gpu_timing(self):
        self._gpu_timing_transaction = None

    @staticmethod
    def _control_value(context, name, role):
        value = context.runtime_args()[name]
        if (
            not isinstance(value, Ndarray)
            or value.shape != ()
            or str(value.dtype) != str(i32)
        ):
            raise TaichiRuntimeError(
                "Graph submission telemetry requires scalar i32 ndarray "
                f"{role} resources"
            )
        return value

    def _capture(self, index, node, context, phase):
        predicate = self._control_value(context, node.predicate, "predicate")
        counter = self._control_value(context, node.counter, "counter")
        status = (
            self._control_value(context, node.status, "status")
            if node.status is not None
            else counter
        )
        offset = index * self._VALUES_PER_REGION + phase * self._VALUES_PER_PHASE
        _pack_structured_submission_telemetry(
            predicate, counter, status, self._storage, offset
        )

    def begin_region(self, index, node, context):
        self._host_started_ns[index] = time.perf_counter_ns()
        self._metadata[index] = _structured_submission_metadata(node)
        self._capture(index, node, context, 0)
        if self._gpu_timing_transaction is not None:
            self._gpu_timing_transaction._begin_gpu_region_timing(
                self._metadata[index]["path_id"]
            )

    def end_region(self, index, node, context):
        if self._gpu_timing_transaction is not None:
            self._gpu_timing_transaction._end_gpu_region_timing(
                self._metadata[index]["path_id"]
            )
        self._capture(index, node, context, 1)
        self._host_enqueue_ns[index] = (
            time.perf_counter_ns() - self._host_started_ns[index]
        )

    def capture_bounded(self, prepared_args, *, public_args=None):
        from taichi_forge.lang.device_extent import DeviceExtent

        device_index = 0
        base_offset = len(self._nodes) * self._VALUES_PER_REGION
        for source in self._bounded_sources:
            key = source["snapshot_key"]
            count_name = source["count_name"]
            # Public host counts retain their pre-clamp observation semantics.
            # Private, temporary, and derived bindings only exist in the
            # prepared invocation used by the executable.
            if public_args is not None and count_name in public_args:
                value = public_args[count_name]
            else:
                value = prepared_args[count_name]
            if source["count_source"] == "host_scalar":
                raw = HostBoundedDispatchHandle._host_count(value)
                self._host_bounded_snapshots[key] = {
                    "source_count": raw,
                    "overflow": raw < 0 or raw > source["capacity"],
                    "snapshot_status": "host_argument",
                }
                continue
            if not isinstance(value, DeviceExtent):
                raise TaichiRuntimeError(
                    "Graph pipeline telemetry requires DeviceExtent values for "
                    "device-count bounded dispatches"
                )
            value._validate_current()
            if value.capacity != source["capacity"]:
                raise TaichiRuntimeError(
                    "Graph pipeline telemetry extent capacity does not match "
                    "the compiled bounded dispatch"
                )
            _pack_bounded_pipeline_telemetry(
                value.state,
                self._storage,
                base_offset + device_index * 2,
            )
            device_index += 1

    def bounded_snapshots(self, values):
        result = dict(self._host_bounded_snapshots)
        device_index = 0
        base_offset = len(self._nodes) * self._VALUES_PER_REGION
        for source in self._bounded_sources:
            if source["count_source"] != "device_extent":
                continue
            offset = base_offset + device_index * 2
            result[source["snapshot_key"]] = {
                "source_count": int(values[offset]),
                "overflow": bool(values[offset + 1]),
                "snapshot_status": "ticket_device_snapshot",
            }
            device_index += 1
        return result


class _GraphStructuredTelemetryLease:
    def __init__(self, state):
        self._state = state
        self.recorder = state.recorder

    def attach(self, completion, queue, host_submit_ns, submission_statistics):
        if self._state is None:
            return None
        state = self._state.attach(
            completion, queue, host_submit_ns, submission_statistics
        )
        self._state = None
        return state

    def cancel(self):
        if self._state is not None:
            self._state.discard()
            self._state = None


class _GraphStructuredTelemetryArena:
    """Bounded opt-in device snapshots for asynchronous Graph submissions."""

    def __init__(self, nodes, pipeline_definition, capacity=None):
        self.nodes = tuple(nodes)
        self._pipeline_definition = pipeline_definition
        self._pipeline_definition_cache = None
        self._bounded_sources_cache = None
        self.capacity = int(
            capacity
            if capacity is not None
            else os.environ.get("TI_GRAPH_TELEMETRY_SLOTS", "4")
        )
        if self.capacity < 1 or self.capacity > 64:
            raise TaichiRuntimeError(
                "TI_GRAPH_TELEMETRY_SLOTS must be between 1 and 64"
            )
        self._slots = []
        self._next_sequence = 1
        self._allocations = 0
        self._reuses = 0
        self._waits = 0
        self._materializations = 0
        self._host_readback_bytes = 0
        self._lock = threading.Lock()

    def _resolve_pipeline_definition(self):
        if self._pipeline_definition_cache is None:
            self._pipeline_definition_cache = tuple(self._pipeline_definition())
        return self._pipeline_definition_cache

    def _bounded_sources(self):
        if self._bounded_sources_cache is None:
            sources = []
            seen = set()
            for stage in self._resolve_pipeline_definition():
                for item in stage["bounded_dispatches"]:
                    key = item["snapshot_key"]
                    if key in seen:
                        continue
                    seen.add(key)
                    sources.append(
                        {
                            "snapshot_key": key,
                            "count_source": item["count_source"],
                            "count_name": item["count_name"],
                            "capacity": item["capacity"],
                        }
                    )
            self._bounded_sources_cache = tuple(sources)
        return self._bounded_sources_cache

    def _new_slot(self):
        bounded_sources = self._bounded_sources()
        device_bounded_count = sum(
            item["count_source"] == "device_extent" for item in bounded_sources
        )
        value_count = (
            len(self.nodes) * _GraphStructuredTelemetryRecorder._VALUES_PER_REGION
            + device_bounded_count * 2
        )
        storage = ScalarNdarray(i32, (value_count,)) if value_count else None
        slot = {
            "storage": storage,
            "recorder": _GraphStructuredTelemetryRecorder(
                self.nodes, storage, bounded_sources
            ),
            "payload_bytes": value_count * np.dtype(np.int32).itemsize,
            "state": None,
            "completion_sequence": 0,
        }
        self._slots.append(slot)
        self._allocations += 1
        return slot

    def acquire(self, mode):
        while True:
            with self._lock:
                allocated = False
                slot = next(
                    (item for item in self._slots if item["state"] is None),
                    None,
                )
                if slot is None and len(self._slots) < self.capacity:
                    slot = self._new_slot()
                    allocated = True
                if slot is not None:
                    if not allocated:
                        self._reuses += 1
                    slot["recorder"].reset()
                    state = _GraphStructuredTelemetryState(
                        self, slot, self._next_sequence, mode
                    )
                    self._next_sequence += 1
                    slot["state"] = state
                    return _GraphStructuredTelemetryLease(state)
                oldest = min(
                    (item["state"] for item in self._slots),
                    key=lambda state: state.sequence,
                )
            oldest.make_reusable()

    def prepare(self, slots=1):
        """Materialize bounded storage and compile snapshot kernels only."""

        if isinstance(slots, bool) or not isinstance(slots, (int, np.integer)):
            raise TaichiRuntimeError(
                "Graph.prepare_telemetry() slots must be an integer"
            )
        slots = int(slots)
        if slots < 1 or slots > self.capacity:
            raise TaichiRuntimeError(
                "Graph.prepare_telemetry() slots must be between 1 and "
                f"{self.capacity}"
            )
        with self._lock:
            while len(self._slots) < slots:
                self._new_slot()
            destination = next(
                (
                    slot["storage"]
                    for slot in self._slots
                    if slot["storage"] is not None
                ),
                None,
            )
            bounded_sources = self._bounded_sources()

        # Compile the two opt-in snapshot kernels without launching them. The
        # canonical ndarray specialization is shared by later telemetry slots;
        # the temporary source only supplies the exact scalar/vector ABI.
        if destination is not None and self.nodes:
            source = ScalarNdarray(i32, ())
            _pack_structured_submission_telemetry._primal.ensure_compiled(
                source, source, source, destination, 0
            )
        if destination is not None and any(
            item["count_source"] == "device_extent" for item in bounded_sources
        ):
            extent_state = ScalarNdarray(i32, (2,))
            _pack_bounded_pipeline_telemetry._primal.ensure_compiled(
                extent_state, destination, 0
            )
        return self

    def _read_slot(
        self,
        slot,
        queue,
        host_submit_ns,
        completion,
        submission_statistics,
        mode,
    ):
        storage = slot["storage"]
        if storage is None:
            host = np.empty(shape=(0,), dtype=np.int32)
        else:
            host = np.empty(
                shape=storage.arr.total_shape(),
                dtype=to_numpy_type(storage.dtype),
            )
            impl.get_runtime().prog.copy_graph_observations_to_host(
                [storage.arr], [host]
            )
        values = host.reshape(-1)
        recorder = slot["recorder"]
        timing_requested = mode == "timestamps"
        gpu_region_timings = (
            completion._gpu_region_timings()
            if timing_requested and completion is not None
            else []
        )
        gpu_region_timings_by_path = {}
        for timing in gpu_region_timings:
            path_id = str(timing["path_id"])
            if path_id in gpu_region_timings_by_path:
                raise TaichiRuntimeError(
                    "Graph submission telemetry observed duplicate GPU region paths"
                )
            gpu_region_timings_by_path[path_id] = timing
        regions = []
        for index, metadata in enumerate(recorder.metadata):
            if metadata is None:
                raise TaichiRuntimeError(
                    "Graph submission telemetry region was not recorded"
                )
            offset = index * _GraphStructuredTelemetryRecorder._VALUES_PER_REGION
            initial_predicate, initial_counter, initial_status = (
                int(value) for value in values[offset : offset + 3]
            )
            final_predicate, final_counter, final_status = (
                int(value) for value in values[offset + 3 : offset + 6]
            )
            del initial_predicate
            logical = final_counter - initial_counter
            max_iterations = int(metadata["max_iterations"])
            if logical < 0 or logical > max_iterations:
                raise TaichiRuntimeError(
                    "Graph submission telemetry observed an iteration counter "
                    "outside the region budget"
                )
            chunk_sizes = tuple(int(value) for value in metadata["chunk_sizes"])
            chunk_strategies = tuple(metadata["chunk_strategies"])
            if metadata["lowering"] == "cuda_conditional_graph":
                chunk_sizes = (logical,) if logical else ()
                chunk_strategies = ("cuda_conditional_graph",) if logical else ()
            offset_iterations = 0
            active_chunks = 0
            skipped_chunks = 0
            for chunk_size, strategy in zip(chunk_sizes, chunk_strategies):
                if logical > offset_iterations:
                    active_chunks += 1
                elif strategy == "coarse_conditional":
                    skipped_chunks += 1
                offset_iterations += chunk_size
            gpu_region = gpu_region_timings_by_path.get(
                metadata["path_id"],
                {
                    "available": False,
                    "duration_ns": 0,
                    "exact": False,
                    "measurement_path_changed": False,
                    "stream_id": 0,
                    "status": (
                        "unavailable" if timing_requested else "disabled_by_mode"
                    ),
                },
            )
            gpu_region_available = bool(gpu_region["available"])
            control_depth = int(metadata["control_depth"])
            logical_invocations = 1
            if control_depth > 1:
                parent_regions = tuple(
                    region
                    for region in regions
                    if metadata["path_id"].startswith(region.path_id + "/")
                    and region.control_depth == control_depth - 1
                )
                if len(parent_regions) != 1:
                    raise TaichiRuntimeError(
                        "Graph submission telemetry could not identify the "
                        "unique parent structured region"
                    )
                logical_invocations = parent_regions[0].logical_iterations
            regions.append(
                GraphSubmissionRegionTelemetry(
                    name=metadata["name"],
                    path_id=metadata["path_id"],
                    backend=metadata["backend"],
                    lowering=metadata["lowering"],
                    control_depth=control_depth,
                    max_iterations=max_iterations,
                    logical_invocations=logical_invocations,
                    logical_iterations=logical,
                    encoded_iterations=sum(chunk_sizes),
                    masked_iterations=sum(chunk_sizes) - logical,
                    chunk_sizes=chunk_sizes,
                    chunk_strategies=chunk_strategies,
                    active_chunk_count=active_chunks,
                    coarse_skipped_chunk_count=skipped_chunks,
                    initial_counter=initial_counter,
                    final_counter=final_counter,
                    terminal_predicate=final_predicate,
                    initial_status=(initial_status if metadata["has_status"] else None),
                    final_status=(final_status if metadata["has_status"] else None),
                    host_enqueue_ns=int(recorder.host_enqueue_ns[index]),
                    gpu_duration_ns=(
                        int(gpu_region["duration_ns"]) if gpu_region_available else None
                    ),
                    gpu_timestamp_exact=bool(gpu_region["exact"]),
                    gpu_measurement_path_changed=bool(
                        gpu_region["measurement_path_changed"]
                    ),
                    gpu_queue_or_stream_id=(
                        f"{metadata['backend']}:{int(gpu_region['stream_id'])}"
                    ),
                    gpu_timestamp_status=str(gpu_region["status"]),
                )
            )
        with self._lock:
            self._materializations += 1
            self._host_readback_bytes += int(host.nbytes)
        backend = (
            regions[0].backend
            if regions
            else _backend_name(_ti_core.arch_name(impl.current_cfg().arch))
        )
        gpu_timing = (
            completion._gpu_timing()
            if timing_requested and completion is not None
            else {
                "available": False,
                "duration_ns": 0,
                "exact": False,
                "measurement_path_changed": False,
                "stream_id": 0,
                "driver_owned_bytes": 0,
                "driver_owned_bytes_known": False,
                "status": ("unavailable" if timing_requested else "disabled_by_mode"),
            }
        )
        pipeline = _materialize_graph_pipeline_report(
            self._resolve_pipeline_definition(),
            backend=backend,
            sequence=int(slot["completion_sequence"]),
            host_submit_ns=int(host_submit_ns),
            gpu_timing=gpu_timing,
            gpu_region_timings=gpu_region_timings,
            bounded_snapshots=recorder.bounded_snapshots(values),
        )
        gpu_available = bool(gpu_timing["available"])
        execution = _materialize_graph_submission_execution_telemetry(
            backend=backend,
            regions=regions,
            queue=queue,
            submission_statistics=submission_statistics,
        )
        return GraphSubmissionTelemetry(
            schema_version=5,
            backend=backend,
            sequence=int(slot["completion_sequence"]),
            regions=tuple(regions),
            queue=queue,
            execution=execution,
            host_submit_ns=int(host_submit_ns),
            device_snapshot_bytes=int(host.nbytes),
            host_readback_bytes=int(host.nbytes),
            gpu_duration_ns=(int(gpu_timing["duration_ns"]) if gpu_available else None),
            gpu_timestamp_scope=("whole_ticket" if timing_requested else "unavailable"),
            gpu_timestamp_exact=bool(gpu_timing["exact"]),
            gpu_measurement_path_changed=bool(gpu_timing["measurement_path_changed"]),
            gpu_queue_or_stream_id=f"{backend}:{int(gpu_timing['stream_id'])}",
            gpu_timestamp_resource_bytes=int(gpu_timing["driver_owned_bytes"]),
            gpu_timestamp_resource_bytes_known=bool(
                gpu_timing["driver_owned_bytes_known"]
            ),
            gpu_timestamp_status=str(gpu_timing["status"]),
            pipeline=pipeline,
        )

    def _record_wait(self):
        with self._lock:
            self._waits += 1

    def _release(self, slot, state):
        with self._lock:
            if slot["state"] is state:
                slot["state"] = None

    @property
    def stats(self):
        with self._lock:
            return {
                "materialized": bool(self._slots),
                "capacity": self.capacity if self._slots else 0,
                "slots": len(self._slots),
                "reserved_bytes": sum(slot["payload_bytes"] for slot in self._slots),
                "allocations": self._allocations,
                "reuses": self._reuses,
                "waits": self._waits,
                "materializations": self._materializations,
                "host_readback_bytes": self._host_readback_bytes,
            }


@dataclass(frozen=True)
class GraphExecutionCounters:
    """Detailed backend counters captured after diagnostics are enabled."""

    attempts: int
    ordinary_fallbacks: int
    capture_attempts: int
    captures: int
    exact_replays: int
    patched_replays: int
    masked_captures: int
    masked_replays: int
    masked_patched_replays: int
    recaptures: int
    records: int
    replays: int
    structural_fallbacks: int
    transient_failures: int
    retry_backoff_fallbacks: int
    replay_slot_saturation_fallbacks: int
    capture_exceptions: int
    zero_arg_captures: int
    asynchronous_control_updates: int
    deferred_replay_waits: int
    peak_deferred_replay_batches: int


@dataclass(frozen=True)
class GraphReplayAttribution:
    """Opt-in host-side costs for one cached CGraph segment.

    Calling :meth:`Graph.execution_stats` enables collection for later
    replays. Nanosecond counters are cumulative. GPU payload execution is
    asynchronous and excluded; CPU ``backend_ns`` includes synchronous kernel
    execution.
    """

    enabled: bool
    calls: int
    total_ns: int
    snode_guard_wait_ns: int
    resource_guard_wait_ns: int
    cuda_submission_wait_ns: int
    cache_wait_ns: int
    binding_plan_ns: int
    resource_retain_ns: int
    snode_validation_ns: int
    backend_ns: int
    signature_ns: int
    binding_plan_hits: int
    binding_plan_misses: int
    signature_fast_hits: int
    signature_fast_misses: int
    snode_guard_acquisitions: int
    snode_guard_elisions: int


_REPLAY_ATTRIBUTION_FIELDS = (
    "calls",
    "total_ns",
    "snode_guard_wait_ns",
    "resource_guard_wait_ns",
    "cuda_submission_wait_ns",
    "cache_wait_ns",
    "binding_plan_ns",
    "resource_retain_ns",
    "snode_validation_ns",
    "backend_ns",
    "signature_ns",
    "binding_plan_hits",
    "binding_plan_misses",
    "signature_fast_hits",
    "signature_fast_misses",
    "snode_guard_acquisitions",
    "snode_guard_elisions",
)


def _replay_attribution(stats, *, enabled=False):
    return GraphReplayAttribution(
        enabled=bool(stats.get("replay_attribution_enabled", enabled)),
        **{
            name: int(stats.get(f"replay_{name}", 0))
            for name in _REPLAY_ATTRIBUTION_FIELDS
        },
    )


@dataclass(frozen=True)
class GraphExecutionSegmentReport:
    """Read-only execution snapshot for one CGraph or native segment."""

    node_index: int
    kind: str
    dispatch_count: int
    compiled_task_count: Optional[int]
    runtime_arg_count: int
    static_dependency_count: int
    static_layout_fingerprint: str
    backend: str
    last_path: str
    fallback_reason: str
    backend_graph_path: bool
    backend_replay_path: bool
    zero_arg_eligible: bool
    persistent_argument_bytes: int
    persistent_bounded_control_bytes: int
    bounded_update_groups: int
    bounded_updater_dispatches: int
    bounded_grouped_payloads: int
    bounded_producer_fused_groups: int
    bounded_payloads: int
    bounded_last_useful_lanes: int
    bounded_last_physical_blocks: int
    bounded_last_physical_threads: int
    bounded_last_baseline_blocks: int
    bounded_last_zero_payloads: int
    bounded_physical_observation_available: bool
    bounded_update_replays: int
    bounded_update_state_changes: int
    bounded_update_cache_hits: int
    bounded_node_api_calls: int
    bounded_max_group_size: int
    last_driver_error: int
    retry_backoff_remaining: int
    consecutive_transient_failures: int
    counters_complete: bool
    counters: GraphExecutionCounters
    replay_attribution: GraphReplayAttribution


@dataclass(frozen=True)
class GraphMemoryReport:
    """Known Graph-owned memory; driver-internal memory remains unknown."""

    persistent_argument_bytes: int
    persistent_bounded_control_bytes: int
    persistent_observation_bytes: int
    persistent_temporary_bytes: int
    persistent_bytes: int
    transient_temporary_bytes: int
    planned_temporary_bytes: int
    temporary_reuse_bytes: int
    opaque_temporary_bytes: int
    temporary_plan_materialized: bool
    temporary_arena_capacity: int
    temporary_arena_slots: int
    temporary_arena_allocations: int
    temporary_arena_reuses: int
    temporary_arena_waits: int
    observation_arena_capacity: int
    observation_arena_slots: int
    observation_arena_allocations: int
    observation_arena_reuses: int
    observation_arena_waits: int
    observation_materializations: int
    observation_host_readback_bytes: int
    observation_readback_mode: str
    observation_completion_attached: bool
    persistent_telemetry_bytes: int
    telemetry_arena_capacity: int
    telemetry_arena_slots: int
    telemetry_arena_allocations: int
    telemetry_arena_reuses: int
    telemetry_arena_waits: int
    telemetry_materializations: int
    telemetry_host_readback_bytes: int
    persistent_internal_storage_bytes: int
    internal_storage_exclusive: bool
    internal_storage_waits: int
    internal_storage_reuses: int
    workspace_lane_capacity: int
    workspace_lanes_materialized: int
    workspace_lanes_busy: int
    workspace_lane_acquisitions: int
    workspace_lane_waits: int
    workspace_lane_saturation_errors: int
    workspace_lane_saturation_policy: str
    provider_generation_report_count: int
    provider_generation_known_resident_requested_bytes: int
    provider_generation_known_capacity_requested_bytes: int
    provider_generation_requested_bytes_complete: bool
    provider_generation_opaque_component_count: int
    opaque_driver_bytes: Optional[int]


@dataclass(frozen=True)
class GraphExecutionReport:
    """Stable, immutable snapshot returned by Graph.execution_stats().

    Inspection is side-effect free: it never enables clocks, counters, labels,
    or backend instrumentation for later production replay. Detailed dynamic
    measurements belong to explicit submission telemetry; this report exposes
    the execution plan, cold-path state that is already available, and the
    known resource footprint.
    """

    schema_version: int
    arch: str
    lifecycle_state: str
    node_count: int
    cgraph_segment_count: int
    native_node_count: int
    observation_node_count: int
    dispatch_count: int
    compiled_task_count: Optional[int]
    runtime_arg_count: int
    static_dependency_count: int
    static_layout_fingerprint: str
    execution_path: str
    fallback_reason: str
    backend_graph_segments: int
    backend_replay_segments: int
    ordinary_fallback_segments: int
    counters_complete: bool
    segments: Tuple[GraphExecutionSegmentReport, ...]
    memory: GraphMemoryReport
    provider_memory: Tuple[object, ...]
    storage_pools: Tuple[object, ...] = ()


@dataclass(frozen=True)
class BoundedDispatchCapabilities:
    """Backend-honest lowering guarantees for one bounded dispatch."""

    schema_version: int
    backend: str
    requested_route: str
    route: str
    minimum_driver_api_version: Optional[int]
    driver_api_version: Optional[int]
    driver_version_eligible: bool
    required_symbols_loaded: bool
    device_update_ptx_linked: bool
    setup_probe_passed: bool
    device_known_count: bool
    no_host_readback: bool
    logical_iteration_exact: bool
    physical_launch_kind: str
    exact_grid: bool
    execution_semantics: str
    range_mapping: str
    masked_capacity: bool
    zero_count_command_skip: bool
    ordered_segments: bool
    global_segment_order: bool
    producer_owned_launch_state: bool
    producer_owned_launch_state_supported: bool
    preparation_dispatches: int
    baseline_capacity_grid: bool
    capacity: int
    block_dim: Optional[int]
    fallback_reason: str
    reason: str
    publication_contract: str = "device_extent"
    publication_reuse: str = "backend_owned"
    static_admission: str = "explicit_route"
    static_admission_reason: str = "none"
    physical_observation: str = "execution_stats_opt_in"
    physical_grid_policy: str = "auto"


@dataclass(frozen=True)
class BoundedDispatchSnapshot:
    """Explicit host observation of one bounded dispatch."""

    capabilities: BoundedDispatchCapabilities
    useful_count: int
    capacity: int
    executed_count: int
    skipped_count: int
    encoded_lanes: int
    overflow: bool


@dataclass
class _ActiveBoundedPublication:
    """One builder-local physical publication and its consumer ownership.

    This is deliberately not part of the public Graph ABI.  It binds the
    backend packet to the semantic extent/block contract that produced it and
    prevents a packet from being reused after an intervening action has made
    the reaching publication ambiguous.
    """

    key: tuple
    packet_arg: object
    packet: object
    packet_claimed: bool = False


@dataclass(frozen=True)
class OrderedSegmentDispatchSnapshot:
    """One ordered segment from an opt-in dispatch observation."""

    segment: int
    begin: int
    end: int
    useful_count: int
    executed_count: int
    skipped_count: int
    encoded_lanes: int
    invalid_offsets: bool


@dataclass(frozen=True)
class OrderedSegmentedDispatchSnapshot:
    """Explicit host observation of an ordered segmented dispatch."""

    capabilities: BoundedDispatchCapabilities
    useful_count: int
    capacity: int
    executed_count: int
    skipped_count: int
    encoded_lanes: int
    overflow: bool
    segments: Tuple[OrderedSegmentDispatchSnapshot, ...]


_bounded_dispatch_ids = itertools.count(1)


def _normalize_bounded_physical_grid_policy(value):
    if not isinstance(value, str):
        raise TypeError("bounded dispatch physical_grid must be a string")
    policy = value.strip().lower()
    aliases = {
        "auto": "auto",
        "extent": "extent",
        "capacity": "capacity",
    }
    if policy not in aliases:
        raise ValueError(
            "bounded dispatch physical_grid must be one of "
            f"{'|'.join(aliases)}, got {value!r}"
        )
    return aliases[policy]


def _bounded_route_request(backend, physical_grid="auto", requested=None):
    physical_grid = _normalize_bounded_physical_grid_policy(physical_grid)
    if physical_grid == "extent":
        if backend == "cuda":
            return "device_update"
        if backend == "cpu":
            return "exact_scheduler"
        return "not_applicable"
    if physical_grid == "capacity":
        return "masked_capacity"
    if backend == "cuda":
        env_name = "TI_CUDA_BOUNDED_DISPATCH_MODE"
        default = "auto"
        aliases = {
            "auto": "auto",
            "device_update": "device_update",
            "masked_capacity": "masked_capacity",
        }
    elif backend == "cpu":
        env_name = "TI_CPU_BOUNDED_DISPATCH_MODE"
        default = "auto"
        aliases = {
            "auto": "auto",
            "exact_scheduler": "exact_scheduler",
            "masked_capacity": "masked_capacity",
        }
    else:
        return "not_applicable"
    if requested is None:
        requested = os.environ.get(env_name, default).strip().lower()
    elif not isinstance(requested, str):
        raise TypeError("bounded route request must be a string")
    else:
        requested = requested.strip().lower()

    if requested not in aliases:
        choices = "|".join(aliases)
        raise TaichiRuntimeError(
            f"{env_name} must be one of {choices}, got {requested!r}"
        )
    return aliases[requested]


def _cuda_bounded_update_policy(requested=None):
    if requested is None:
        requested = (
            os.environ.get("TI_GRAPH_CUDA_BOUNDED_UPDATE_POLICY", "auto")
            .strip()
            .lower()
        )
    elif not isinstance(requested, str):
        raise TypeError("CUDA bounded update policy must be a string")
    else:
        requested = requested.strip().lower()

    aliases = {
        "auto": "grouped_stateful",
        "per_node": "per_node",
        "grouped_stateful": "grouped_stateful",
    }
    if requested not in aliases:
        raise TaichiRuntimeError(
            "TI_GRAPH_CUDA_BOUNDED_UPDATE_POLICY must be one of "
            f"{'|'.join(aliases)}, got {requested!r}"
        )
    return requested, aliases[requested]


def _cuda_nested_device_update_available():
    capabilities = dict(_ti_core.cuda_bounded_dispatch_probe())
    return bool(capabilities.get("exact_device_grid_available", False))


def _vulkan_bounded_packet_policy():
    requested = (
        os.environ.get("TI_GRAPH_VULKAN_BOUNDED_PACKET_POLICY", "auto").strip().lower()
    )
    aliases = {
        "auto": "reuse_consecutive",
        "reuse_consecutive": "reuse_consecutive",
        "per_consumer": "per_consumer",
    }
    if requested not in aliases:
        raise TaichiRuntimeError(
            "TI_GRAPH_VULKAN_BOUNDED_PACKET_POLICY must be one of "
            f"{'|'.join(aliases)}, got {requested!r}"
        )
    return requested, aliases[requested]


def _bounded_route(
    backend,
    ordered,
    physical_grid="auto",
    *,
    requested_route=None,
    requested_update_policy=None,
):
    physical_grid = _normalize_bounded_physical_grid_policy(physical_grid)
    requested_route = _bounded_route_request(
        backend,
        physical_grid,
        requested=requested_route,
    )
    if backend == "vulkan":
        if requested_route == "masked_capacity":
            return BoundedDispatchCapabilities(
                schema_version=5,
                backend=backend,
                requested_route=requested_route,
                route="masked_capacity",
                minimum_driver_api_version=None,
                driver_api_version=None,
                driver_version_eligible=True,
                required_symbols_loaded=True,
                device_update_ptx_linked=False,
                setup_probe_passed=True,
                device_known_count=True,
                no_host_readback=True,
                logical_iteration_exact=False,
                physical_launch_kind="fixed_capacity_grid_stride",
                exact_grid=False,
                execution_semantics="masked_capacity",
                range_mapping="grid_stride",
                masked_capacity=True,
                zero_count_command_skip=False,
                ordered_segments=ordered,
                global_segment_order=ordered,
                producer_owned_launch_state=False,
                producer_owned_launch_state_supported=False,
                preparation_dispatches=0,
                baseline_capacity_grid=True,
                capacity=0,
                block_dim=None,
                fallback_reason="forced_capacity_policy",
                reason=(
                    "Vulkan uses a fixed-capacity grid as explicitly requested "
                    "by this dispatch"
                ),
                physical_grid_policy=physical_grid,
            )
        return BoundedDispatchCapabilities(
            schema_version=5,
            backend=backend,
            requested_route=requested_route,
            route="exact_indirect",
            minimum_driver_api_version=None,
            driver_api_version=None,
            driver_version_eligible=True,
            required_symbols_loaded=True,
            device_update_ptx_linked=False,
            setup_probe_passed=True,
            device_known_count=True,
            no_host_readback=True,
            logical_iteration_exact=True,
            physical_launch_kind="indirect_one_to_one",
            exact_grid=True,
            execution_semantics="exact_device_grid",
            range_mapping="one_to_one",
            masked_capacity=False,
            zero_count_command_skip=True,
            ordered_segments=ordered,
            global_segment_order=ordered,
            producer_owned_launch_state=False,
            producer_owned_launch_state_supported=True,
            preparation_dispatches=1,
            baseline_capacity_grid=False,
            capacity=0,
            block_dim=None,
            fallback_reason="none",
            reason="Vulkan dispatchIndirect consumes a device-written grid packet",
            physical_grid_policy=physical_grid,
        )
    if backend == "cuda":
        update_policy = None
        if requested_route == "device_update":
            _, update_policy = _cuda_bounded_update_policy(requested_update_policy)
        cuda_capabilities = dict(
            (
                _ti_core.cuda_bounded_dispatch_probe()
                if requested_route == "device_update"
                else _ti_core.cuda_bounded_dispatch_capabilities()
            )
        )
        driver_api_version = cuda_capabilities["driver_api_version"]
        driver_version_eligible = bool(cuda_capabilities["driver_version_eligible"])
        required_symbols_loaded = bool(cuda_capabilities["required_symbols_loaded"])
        device_update_ptx_linked = bool(cuda_capabilities["device_update_ptx_linked"])
        setup_probe_passed = bool(cuda_capabilities["setup_probe_passed"])
        if requested_route == "device_update":
            if not cuda_capabilities["exact_device_grid_available"]:
                raise TaichiRuntimeError(
                    "CUDA bounded device_update is unavailable: "
                    f"{cuda_capabilities['unavailable_reason']}"
                )
            return BoundedDispatchCapabilities(
                schema_version=5,
                backend=backend,
                requested_route=requested_route,
                route="adaptive_device_grid_update",
                minimum_driver_api_version=12040,
                driver_api_version=driver_api_version,
                driver_version_eligible=driver_version_eligible,
                required_symbols_loaded=required_symbols_loaded,
                device_update_ptx_linked=device_update_ptx_linked,
                setup_probe_passed=setup_probe_passed,
                device_known_count=True,
                no_host_readback=True,
                logical_iteration_exact=True,
                physical_launch_kind="adaptive_saturated_grid_stride",
                exact_grid=False,
                execution_semantics="exact_device_range",
                range_mapping="device_bounded_grid_stride",
                masked_capacity=False,
                zero_count_command_skip=True,
                ordered_segments=False,
                global_segment_order=False,
                producer_owned_launch_state=False,
                producer_owned_launch_state_supported=False,
                preparation_dispatches=1,
                baseline_capacity_grid=True,
                capacity=0,
                block_dim=None,
                fallback_reason="none",
                reason=(
                    "CUDA loads the logical range end from DeviceExtent and "
                    "uses a saturation-capped device update only to reduce "
                    "physical launch work; the selected update policy is "
                    f"{update_policy}"
                ),
                physical_grid_policy=physical_grid,
            )
        if requested_route == "auto" and not ordered:
            return BoundedDispatchCapabilities(
                schema_version=5,
                backend=backend,
                requested_route=requested_route,
                route="device_bounded_grid_stride",
                minimum_driver_api_version=None,
                driver_api_version=driver_api_version,
                driver_version_eligible=True,
                required_symbols_loaded=True,
                device_update_ptx_linked=device_update_ptx_linked,
                setup_probe_passed=True,
                device_known_count=True,
                no_host_readback=True,
                logical_iteration_exact=True,
                physical_launch_kind="saturated_grid_stride",
                exact_grid=False,
                execution_semantics="exact_device_range",
                range_mapping="device_bounded_grid_stride",
                masked_capacity=False,
                zero_count_command_skip=False,
                ordered_segments=False,
                global_segment_order=False,
                producer_owned_launch_state=False,
                producer_owned_launch_state_supported=False,
                preparation_dispatches=0,
                baseline_capacity_grid=True,
                capacity=0,
                block_dim=None,
                fallback_reason="none",
                reason=(
                    "CUDA uses its ordinary saturation-capped grid-stride "
                    "scheduler and loads the exact logical range end from "
                    "DeviceExtent without host readback"
                ),
                physical_grid_policy=physical_grid,
            )
        return BoundedDispatchCapabilities(
            schema_version=5,
            backend=backend,
            requested_route=requested_route,
            route="masked_capacity",
            minimum_driver_api_version=None,
            driver_api_version=driver_api_version,
            driver_version_eligible=True,
            required_symbols_loaded=True,
            device_update_ptx_linked=device_update_ptx_linked,
            setup_probe_passed=False,
            device_known_count=True,
            no_host_readback=True,
            logical_iteration_exact=False,
            physical_launch_kind="fixed_capacity_grid_stride",
            exact_grid=False,
            execution_semantics="masked_capacity",
            range_mapping="grid_stride",
            masked_capacity=True,
            zero_count_command_skip=False,
            ordered_segments=False,
            global_segment_order=False,
            producer_owned_launch_state=False,
            producer_owned_launch_state_supported=False,
            preparation_dispatches=0,
            baseline_capacity_grid=True,
            capacity=0,
            block_dim=None,
            fallback_reason=(
                "forced_masked_capacity"
                if requested_route == "masked_capacity"
                else "ordered_segments_device_bounded_range_unavailable"
            ),
            reason=(
                "CUDA uses the fixed-capacity masked route for ordered "
                "segments or as an explicit diagnostic and performance "
                "baseline"
            ),
            physical_grid_policy=physical_grid,
        )
    else:
        driver_api_version = None
        driver_version_eligible = True
        required_symbols_loaded = False
        device_update_ptx_linked = False
        setup_probe_passed = False
        if requested_route == "exact_scheduler" or (
            requested_route == "auto" and not ordered
        ):
            return BoundedDispatchCapabilities(
                schema_version=5,
                backend=backend,
                requested_route=requested_route,
                route="exact_cpu_scheduler",
                minimum_driver_api_version=None,
                driver_api_version=None,
                driver_version_eligible=True,
                required_symbols_loaded=True,
                device_update_ptx_linked=False,
                setup_probe_passed=True,
                device_known_count=True,
                no_host_readback=True,
                logical_iteration_exact=True,
                physical_launch_kind="cpu_dynamic_chunks",
                exact_grid=True,
                execution_semantics="exact_cpu_scheduler",
                range_mapping="cpu_scheduler",
                masked_capacity=False,
                zero_count_command_skip=True,
                ordered_segments=False,
                global_segment_order=False,
                producer_owned_launch_state=False,
                producer_owned_launch_state_supported=False,
                preparation_dispatches=0,
                baseline_capacity_grid=False,
                capacity=0,
                block_dim=None,
                fallback_reason="none",
                reason=(
                    "CPU reads DeviceExtent from the Graph argument buffer "
                    "and submits only the clamped range as adaptive contiguous "
                    "chunks independent of GPU block geometry"
                ),
                physical_grid_policy=physical_grid,
            )
        reason = (
            "CPU uses the cached fixed-capacity range task and masks payload "
            "work from the extent"
        )
        fallback_reason = (
            "forced_masked_capacity"
            if requested_route == "masked_capacity"
            else "ordered_segments_exact_cpu_scheduler_unavailable"
        )
        range_mapping = "cpu_scheduler"
        minimum_driver_api_version = None
    return BoundedDispatchCapabilities(
        schema_version=5,
        backend=backend,
        requested_route=requested_route,
        route="masked_capacity",
        minimum_driver_api_version=minimum_driver_api_version,
        driver_api_version=driver_api_version,
        driver_version_eligible=driver_version_eligible,
        required_symbols_loaded=required_symbols_loaded,
        device_update_ptx_linked=device_update_ptx_linked,
        setup_probe_passed=setup_probe_passed,
        device_known_count=True,
        no_host_readback=True,
        logical_iteration_exact=False,
        physical_launch_kind="fixed_capacity_scheduler",
        exact_grid=False,
        execution_semantics="masked_capacity",
        range_mapping=range_mapping,
        masked_capacity=True,
        zero_count_command_skip=False,
        ordered_segments=ordered,
        global_segment_order=ordered,
        producer_owned_launch_state=False,
        producer_owned_launch_state_supported=False,
        preparation_dispatches=0,
        baseline_capacity_grid=True,
        capacity=0,
        block_dim=None,
        fallback_reason=fallback_reason,
        reason=reason,
        physical_grid_policy=physical_grid,
    )


class _DeviceExtentGraphContract:
    """One deduplicated runtime binding contract per symbolic extent.

    The contract owns no device storage. It replaces per-consumer observation
    handles in Graph lifetime validation and deliberately has no
    ``bind_graph_arguments`` hook, so a replay validates the extent once.
    """

    # DeviceExtent capacity, runtime generation, allocation identity, and an
    # optional producer-owned identity are immutable for one published
    # BindingVersion.  Publication validates all of them and the version keeps
    # the DeviceExtent plus its flattened runtime allocation alive.  Graph and
    # BindingVersion generation checks reject ti.reset() before replay, so a
    # second Python lifetime scan would only repeat fixed work.  Raw mappings
    # still enter validate_graph_bindings() for every invocation.
    graph_runtime_lifetime_check_required = False
    graph_publish_time_binding_validation_stable = True

    def __init__(self, extent_name, capacity):
        self.extent_name = extent_name
        self.capacity = int(capacity)
        self._expected_extent = None
        self._runtime_generation = int(impl.runtime_generation())
        self._runtime_program = impl.get_runtime().prog

    def require_identity(self, extent):
        from taichi_forge.lang.device_extent import DeviceExtent

        if not isinstance(extent, DeviceExtent):
            raise TypeError("Bounded extent identity must be a DeviceExtent")
        extent._validate_current()
        if extent.capacity != self.capacity:
            raise ValueError("Bounded extent identity capacity mismatch")
        if self._expected_extent is None:
            self._expected_extent = extent
        elif self._expected_extent is not extent:
            raise TaichiRuntimeError(
                "One symbolic bounded extent cannot require multiple owner identities"
            )

    def validate_graph_lifetime(self):
        if (
            impl.runtime_generation() != self._runtime_generation
            or impl.get_runtime().prog is not self._runtime_program
        ):
            raise TaichiRuntimeError(
                "Bounded extent contract belongs to a stale Taichi runtime"
            )
        if self._expected_extent is not None:
            self._expected_extent._validate_current()

    def validate_graph_bindings(self, args):
        from taichi_forge.lang.device_extent import DeviceExtent

        self.validate_graph_lifetime()
        value = args[self.extent_name]
        if not isinstance(value, DeviceExtent):
            raise TaichiRuntimeError(
                "Bounded dispatch extent must be the DeviceExtent whose capacity "
                "was used to compile this Graph"
            )
        value._validate_current()
        if value.capacity != self.capacity:
            raise TaichiRuntimeError(
                "Bounded dispatch DeviceExtent capacity does not match "
                f"the compiled capacity {self.capacity}"
            )
        if self._expected_extent is not None and value is not self._expected_extent:
            raise TaichiRuntimeError(
                "Producer-owned bounded state requires its bound DeviceExtent"
            )


class _OrderedOffsetsGraphContract:
    """Deduplicated shape/type contract for ordered-segment offsets."""

    # A BindingVersion snapshots the exact offsets ndarray. Its allocation,
    # dtype, and shape cannot change before that version is replaced, while
    # the offset contents intentionally remain dynamic device data. Validate
    # the immutable ABI once when publishing the version instead of scanning
    # it again before every replay.
    graph_publish_time_binding_validation_stable = True

    def __init__(self, offsets_name, segment_count):
        self.offsets_name = offsets_name
        self.segment_count = int(segment_count)

    def validate_graph_bindings(self, args):
        offsets = args[self.offsets_name]
        if not isinstance(offsets, ScalarNdarray):
            raise TaichiRuntimeError(
                "Ordered segmented dispatch offsets must be a scalar i32 ndarray"
            )
        if offsets.dtype != i32 or tuple(offsets.shape) != (self.segment_count + 1,):
            raise TaichiRuntimeError(
                "Ordered segmented dispatch offsets must contain "
                f"{self.segment_count + 1} i32 values"
            )


class BoundedDispatchHandle:
    """Definition and opt-in observation handle returned by GraphBuilder."""

    _SEGMENT_STATE_SIZE = 5

    def __init__(
        self,
        *,
        extent_name,
        capacity,
        block_dim,
        backend,
        ordered=False,
        offsets_name=None,
        segment_count=0,
        packet=None,
        segment_state=None,
        launch_state=None,
        preparation_dispatches=None,
        packet_allocation_owner=True,
        capabilities=None,
        publication_reuse=None,
    ):
        self.extent_name = extent_name
        self.offsets_name = offsets_name
        self.capacity = int(capacity)
        self.block_dim = None if block_dim is None else int(block_dim)
        self.segment_count = int(segment_count)
        self._ordered = bool(ordered)
        self._packet = packet
        self._segment_state = segment_state
        self._launch_state = launch_state
        self._packet_allocation_owner = bool(packet_allocation_owner)
        self._runtime_generation = int(impl.runtime_generation())
        self._runtime_program = impl.get_runtime().prog
        base = (
            _bounded_route(backend, self._ordered)
            if capabilities is None
            else capabilities
        )
        if preparation_dispatches is None:
            preparation_dispatches = (
                0
                if launch_state is not None and backend == "vulkan"
                else base.preparation_dispatches
            )
        preparation_dispatches = int(preparation_dispatches)
        if preparation_dispatches < 0:
            raise ValueError("preparation dispatch count must be nonnegative")
        if publication_reuse is None:
            publication_reuse = (
                "consecutive_packet"
                if backend == "vulkan" and preparation_dispatches == 0
                else (
                    "grouped_stateful"
                    if backend == "cuda"
                    and base.route == "adaptive_device_grid_update"
                    and _cuda_bounded_update_policy()[1] == "grouped_stateful"
                    else "per_consumer"
                )
            )

        if publication_reuse not in (
            "consecutive_packet",
            "grouped_stateful",
            "per_consumer",
        ):
            raise ValueError("bounded publication reuse policy is invalid")

        self._capabilities = replace(
            base,
            capacity=self.capacity,
            block_dim=self.block_dim,
            producer_owned_launch_state=launch_state is not None,
            preparation_dispatches=preparation_dispatches,
            publication_reuse=publication_reuse,
            static_admission=(
                "conservative_saturated"
                if backend == "cuda" and base.requested_route == "auto"
                else "explicit_or_backend_native"
            ),
            static_admission_reason=(
                "static topology does not prove sparse updater amortization"
                if backend == "cuda" and base.requested_route == "auto"
                else "none"
            ),
            physical_observation=(
                "execution_stats_opt_in"
                if backend == "cuda"
                else "handle_snapshot_opt_in"
            ),
        )

    @property
    def capabilities(self):
        return self._capabilities

    @property
    def workspace_bytes(self):
        packet_bytes = (
            0
            if (
                self._packet is None
                or self._launch_state is not None
                or not self._packet_allocation_owner
            )
            else getattr(self._packet, "storage_bytes", 3 * 4)
        )
        segment_bytes = (
            0
            if self._segment_state is None
            else getattr(
                self._segment_state,
                "storage_bytes",
                self._SEGMENT_STATE_SIZE * 4,
            )
        )
        return packet_bytes + segment_bytes

    @property
    def workspace_allocation_count(self):
        owns_packet = (
            self._packet is not None
            and self._launch_state is None
            and self._packet_allocation_owner
        )
        return int(owns_packet) + int(self._segment_state is not None)

    def validate_graph_lifetime(self):
        if self._launch_state is not None:
            self._launch_state._validate_current()
        if (
            impl.runtime_generation() != self._runtime_generation
            or impl.get_runtime().prog is not self._runtime_program
        ):
            raise TaichiRuntimeError(
                "Bounded dispatch belongs to a stale Taichi runtime"
            )
        for storage in (self._packet, self._segment_state):
            if (
                storage is not None
                and not isinstance(storage, _GraphInternalNdarraySpec)
                and storage.arr is None
            ):
                raise TaichiRuntimeError(
                    "Bounded dispatch internal storage is no longer available"
                )

    def _validate_extent_value(self, value):
        from taichi_forge.lang.device_extent import DeviceExtent

        if isinstance(value, DeviceExtent):
            value._validate_current()
            if value.capacity != self.capacity:
                raise TaichiRuntimeError(
                    "Bounded dispatch DeviceExtent capacity does not match "
                    f"the compiled capacity {self.capacity}"
                )
            if self._launch_state is not None:
                self._launch_state.validate_extent(value, require_identity=True)
            return value.state
        raise TaichiRuntimeError(
            "Bounded dispatch extent must be the DeviceExtent whose capacity "
            "was used to compile this Graph"
        )

    def validate_graph_bindings(self, args):
        self.validate_graph_lifetime()
        self._validate_extent_value(args[self.extent_name])
        if not self._ordered:
            return
        offsets = args[self.offsets_name]
        if not isinstance(offsets, ScalarNdarray):
            raise TaichiRuntimeError(
                "Ordered segmented dispatch offsets must be a scalar i32 ndarray"
            )
        if offsets.dtype != i32 or tuple(offsets.shape) != (self.segment_count + 1,):
            raise TaichiRuntimeError(
                "Ordered segmented dispatch offsets must contain "
                f"{self.segment_count + 1} i32 values"
            )

    def bind_graph_arguments(self, args):
        # _GraphRunContext flattens DeviceExtent directly to its stable state.
        # Avoid cloning the argument dict on every replay.
        self._validate_extent_value(args[self.extent_name])
        return {}

    def _execution_counts(self, useful):
        if self._capabilities.logical_iteration_exact:
            if self._capabilities.physical_launch_kind != "indirect_one_to_one":
                # CPU dynamic chunks and CUDA device-bounded grid-stride both
                # enter the payload body exactly once for [0, useful). Their
                # worker/thread envelope is deliberately separate from this
                # logical execution accounting.
                return useful, useful
            if self.block_dim is None:
                return useful, useful
            encoded = (
                0
                if useful == 0
                else ((useful + self.block_dim - 1) // self.block_dim) * self.block_dim
            )
            executed = min(self.capacity, encoded)
        else:
            encoded = self.capacity
            executed = self.capacity
        return executed, encoded

    def snapshot(self, extent, offsets=None):
        """Synchronize and materialize useful/executed/masked work."""

        from taichi_forge.lang.device_extent import DeviceExtent

        if not isinstance(extent, DeviceExtent):
            raise TypeError("BoundedDispatchHandle.snapshot() expects DeviceExtent")
        self._validate_extent_value(extent)
        extent_snapshot = extent.snapshot()
        if not self._ordered:
            if offsets is not None:
                raise TypeError("Non-segmented bounded dispatch has no offsets")
            executed, encoded = self._execution_counts(extent_snapshot.count)
            return BoundedDispatchSnapshot(
                capabilities=self._capabilities,
                useful_count=extent_snapshot.count,
                capacity=self.capacity,
                executed_count=executed,
                skipped_count=max(0, executed - extent_snapshot.count),
                encoded_lanes=encoded,
                overflow=extent_snapshot.overflow,
            )

        if not isinstance(offsets, ScalarNdarray):
            raise TypeError(
                "Ordered segmented dispatch snapshot requires its offsets ndarray"
            )
        self.validate_graph_bindings(
            {self.extent_name: extent, self.offsets_name: offsets}
        )
        values = offsets.to_numpy().astype(np.int64, copy=False)
        active = extent_snapshot.count
        segments = []
        total_executed = 0
        total_encoded = 0
        invalid_any = False
        for segment in range(self.segment_count):
            raw_begin = int(values[segment])
            raw_end = int(values[segment + 1])
            invalid = raw_begin < 0 or raw_end < raw_begin or raw_end > active
            if segment == 0 and raw_begin != 0:
                invalid = True
            if segment + 1 == self.segment_count and raw_end != active:
                invalid = True
            begin = min(max(raw_begin, 0), active)
            end = min(max(raw_end, begin), active)
            useful = end - begin
            executed, encoded = self._execution_counts(useful)
            segments.append(
                OrderedSegmentDispatchSnapshot(
                    segment=segment,
                    begin=begin,
                    end=end,
                    useful_count=useful,
                    executed_count=executed,
                    skipped_count=max(0, executed - useful),
                    encoded_lanes=encoded,
                    invalid_offsets=invalid,
                )
            )
            total_executed += executed
            total_encoded += encoded
            invalid_any = invalid_any or invalid
        total_useful = sum(segment.useful_count for segment in segments)
        return OrderedSegmentedDispatchSnapshot(
            capabilities=self._capabilities,
            useful_count=total_useful,
            capacity=self.capacity,
            executed_count=total_executed,
            skipped_count=max(0, total_executed - total_useful),
            encoded_lanes=total_encoded,
            overflow=extent_snapshot.overflow or invalid_any,
            segments=tuple(segments),
        )


class HostBoundedDispatchHandle:
    """Host-known exact range binding with capacity clamping."""

    def __init__(self, *, count_name, capacity, block_dim, backend):
        self.count_name = count_name
        self.capacity = int(capacity)
        self.block_dim = None if block_dim is None else int(block_dim)
        self._runtime_generation = int(impl.runtime_generation())
        self._runtime_program = impl.get_runtime().prog
        self._capabilities = BoundedDispatchCapabilities(
            schema_version=5,
            backend=backend,
            requested_route="host_known",
            route="exact_host_range",
            minimum_driver_api_version=None,
            driver_api_version=(
                _ti_core.cuda_driver_api_version() if backend == "cuda" else None
            ),
            driver_version_eligible=True,
            required_symbols_loaded=True,
            device_update_ptx_linked=False,
            setup_probe_passed=True,
            device_known_count=False,
            no_host_readback=True,
            logical_iteration_exact=True,
            physical_launch_kind=(
                "cpu_dynamic_range" if backend == "cpu" else "host_sized_grid_stride"
            ),
            exact_grid=True,
            execution_semantics="exact_host_range",
            range_mapping=("cpu_scheduler" if backend == "cpu" else "grid_stride"),
            masked_capacity=False,
            zero_count_command_skip=False,
            ordered_segments=False,
            global_segment_order=False,
            producer_owned_launch_state=False,
            producer_owned_launch_state_supported=False,
            preparation_dispatches=0,
            baseline_capacity_grid=False,
            capacity=self.capacity,
            block_dim=self.block_dim,
            fallback_reason="none",
            reason=(
                "the compiler proved that the payload range is driven by the "
                "host scalar count argument; the backend may retain scalar-range "
                "setup work when the bounded count is zero"
            ),
        )

    @property
    def capabilities(self):
        return self._capabilities

    @property
    def workspace_bytes(self):
        return 0

    @property
    def workspace_allocation_count(self):
        return 0

    def validate_graph_lifetime(self):
        if (
            impl.runtime_generation() != self._runtime_generation
            or impl.get_runtime().prog is not self._runtime_program
        ):
            raise TaichiRuntimeError(
                "Host bounded dispatch belongs to a stale Taichi runtime"
            )

    @staticmethod
    def _host_count(value):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TaichiRuntimeError(
                "host-known bounded dispatch count must be an integer"
            )
        value = int(value)
        if not -0x80000000 <= value <= 0x7FFFFFFF:
            raise TaichiRuntimeError(
                "host-known bounded dispatch count must fit signed i32"
            )
        return value

    def validate_graph_bindings(self, args):
        self.validate_graph_lifetime()
        self._host_count(args[self.count_name])

    def bind_graph_arguments(self, args):
        raw = self._host_count(args[self.count_name])
        bounded = min(max(raw, 0), self.capacity)
        if bounded == raw:
            return {}
        return {self.count_name: bounded}

    def snapshot(self, count):
        """Return a host-only report; no synchronization is required."""

        raw = self._host_count(count)
        useful = min(max(raw, 0), self.capacity)
        encoded = useful
        if self.block_dim is not None and useful:
            encoded = ((useful + self.block_dim - 1) // self.block_dim) * self.block_dim
        return BoundedDispatchSnapshot(
            capabilities=self._capabilities,
            useful_count=useful,
            capacity=self.capacity,
            executed_count=useful,
            skipped_count=0,
            encoded_lanes=encoded,
            overflow=raw != useful,
        )


@dataclass(frozen=True)
class GraphWhileReport:
    """Last execution of one structured Graph while region."""

    name: str
    backend: str
    lowering: str
    max_iterations: int
    logical_iterations: int
    executed_iterations: int
    overshoot_iterations: int
    observation_boundaries: Tuple[int, ...]
    predicate_values: Tuple[int, ...]
    counter_values: Tuple[int, ...]
    status_resource: Optional[str]
    status_values: Tuple[int, ...]
    chunk_sizes: Tuple[int, ...]
    observation_batches: int
    observation_scalar_count: int
    device_to_host_bytes: int
    initial_counter: Optional[int]
    final_counter: Optional[int]
    initial_status: Optional[int]
    final_status: Optional[int]
    native_upgrade_eligible: bool
    native_upgrade_reason: str
    persistent_staging_bytes: int
    staging_allocations: int
    staging_reuses: int
    packed_observation_batches: int
    direct_observation_batches: int
    staging_fallback_batches: int
    packed_observation_bytes: int
    condition_dispatch_count: int
    body_dispatch_count: int
    control_inputs: Tuple[str, ...]
    carried_state: Tuple[str, ...]
    indirect_dispatch_count: int = 0
    controller_dispatch_count: int = 0
    controller_invocation_count: int = 0
    logical_body_dispatch_count: int = 0
    zero_dispatch_count: int = 0
    control_arena_bytes: int = 0
    region_path: str = ""
    structured_depth: int = 1
    nested_region_path: str = ""
    nested_logical_iterations: Tuple[int, ...] = ()
    nested_encoded_iterations: Tuple[int, ...] = ()
    encoded_iterations: int = 0
    masked_iterations: int = 0


@dataclass(frozen=True)
class GraphBranchReport:
    """Last execution of one structured Graph if/switch region."""

    name: str
    backend: str
    kind: str
    lowering: str
    selector_resource: str
    selector_value: int
    selected_branch: str
    observation_scalar_count: int
    device_to_host_bytes: int
    condition_dispatch_count: int
    branch_dispatch_count: int
    control_inputs: Tuple[str, ...]
    region_path: str = ""
    structured_depth: int = 1
    encoded_dispatch_count: int = 0
    masked_dispatch_count: int = 0


@dataclass(frozen=True)
class GraphControlFlowInvocation:
    """One dynamically addressed structured-control invocation."""

    sequence: int
    definition_path: str
    invocation_path: str
    parent_iteration: Optional[int]
    report: Union[GraphWhileReport, GraphBranchReport]


@dataclass(frozen=True)
class GraphControlFlowTrace:
    """Immutable opt-in trace returned by ``Graph.run(trace=True)``."""

    schema_version: int
    invocations: Tuple[GraphControlFlowInvocation, ...]


class _ControlFlowTraceFrame:
    __slots__ = (
        "definition_path",
        "entry_index",
        "invocation_path",
        "node",
        "parent_iteration",
        "role_invocations",
        "iteration",
    )

    def __init__(
        self,
        node,
        *,
        definition_path,
        invocation_path,
        parent_iteration,
        entry_index,
    ):
        self.node = node
        self.definition_path = definition_path
        self.invocation_path = invocation_path
        self.parent_iteration = parent_iteration
        self.entry_index = entry_index
        self.iteration = None
        self.role_invocations = {}


class _ControlFlowTraceRecorder:
    """Run-local host trace; no instance is created for default Graph.run()."""

    schema_version = 1

    def __init__(self):
        self._frames = []
        self._entries = []

    @staticmethod
    def _relative_definition_path(parent, node):
        prefix = f"{parent.definition_path}/"
        if node.region_path.startswith(prefix):
            return node.region_path[len(prefix) :]
        return node.region_path

    def begin(self, node):
        definition_path = node.region_path
        parent_iteration = None
        if not self._frames:
            invocation_path = definition_path
        else:
            parent = self._frames[-1]
            relative = self._relative_definition_path(parent, node)
            role, separator, remainder = relative.partition("/")
            if isinstance(parent.node, _CompiledWhileGraphNode):
                if role == "body" and parent.iteration is not None:
                    parent_iteration = parent.iteration
                    prefix = f"{parent.invocation_path}[{parent.iteration}]/body"
                    invocation_path = f"{prefix}/{remainder}" if separator else prefix
                elif role == "condition":
                    call_index = parent.role_invocations.get(relative, 0)
                    parent.role_invocations[relative] = call_index + 1
                    prefix = f"{parent.invocation_path}/condition[{call_index}]"
                    invocation_path = f"{prefix}/{remainder}" if separator else prefix
                else:
                    invocation_path = f"{parent.invocation_path}/{relative}"
            else:
                invocation_path = f"{parent.invocation_path}/{relative}"
        entry_index = len(self._entries)
        self._entries.append(None)
        frame = _ControlFlowTraceFrame(
            node,
            definition_path=definition_path,
            invocation_path=invocation_path,
            parent_iteration=parent_iteration,
            entry_index=entry_index,
        )
        self._frames.append(frame)
        return frame

    def set_iteration(self, node, iteration):
        if not self._frames or self._frames[-1].node is not node:
            raise TaichiRuntimeError(
                "Graph control-flow trace iteration frame is inconsistent"
            )
        self._frames[-1].iteration = iteration

    def end(self, frame, report):
        if not self._frames or self._frames[-1] is not frame:
            raise TaichiRuntimeError("Graph control-flow trace frame is inconsistent")
        self._frames.pop()
        if report is None:
            raise TaichiRuntimeError(
                "Graph control-flow trace invocation did not produce a report"
            )
        dynamic_report = replace(report, region_path=frame.invocation_path)
        self._entries[frame.entry_index] = GraphControlFlowInvocation(
            sequence=frame.entry_index,
            definition_path=frame.definition_path,
            invocation_path=frame.invocation_path,
            parent_iteration=frame.parent_iteration,
            report=dynamic_report,
        )

    def abort(self, frame):
        if self._frames and self._frames[-1] is frame:
            self._frames.pop()

    def finish(self):
        if self._frames or any(entry is None for entry in self._entries):
            raise TaichiRuntimeError("Graph control-flow trace is incomplete")
        return GraphControlFlowTrace(
            schema_version=self.schema_version,
            invocations=tuple(self._entries),
        )


@dataclass(frozen=True)
class ParallelStorageFact:
    resource: str
    supported: bool
    owner_status: str
    failure_reason: str
    source_kind: str
    owner_kind: str
    program_domain: Optional[int]
    resource_identity: Optional[Tuple[object, ...]]
    tree_identity: Optional[Tuple[object, ...]]
    byte_offset: Optional[int]
    byte_begin: Optional[int]
    byte_end: Optional[int]
    compact_contiguous: Optional[bool]
    index_shape: Tuple[int, ...]
    index_strides_bytes: Tuple[int, ...]
    element_shape: Tuple[int, ...]
    scalar_count: Optional[int]
    record_stride: Optional[int]

    def to_dict(self):
        return self.__dict__.copy()


@dataclass(frozen=True)
class ParallelRuntimeAliasFact:
    left_branch: int
    right_branch: int
    left_resource: str
    right_resource: str
    result: str
    dependencies: Tuple[str, ...]

    def to_dict(self):
        return self.__dict__.copy()


@dataclass(frozen=True)
class ParallelCandidateReport:
    """Read-only safety and memory report for a possible fork/join region."""

    schema_version: int
    analysis_only: bool
    execution_changed: bool
    selection_domain: str
    branch_node_indices: Tuple[Tuple[int, ...], ...]
    decision: str
    safe: Optional[bool]
    runtime_binding_provided: bool
    runtime_generation: Optional[int]
    backend: Optional[str]
    branches: Tuple[ParallelBranchSummary, ...]
    conflicts: Tuple[ParallelEffectDependency, ...]
    unresolved_aliases: Tuple[ParallelEffectDependency, ...]
    runtime_aliases: Tuple[ParallelRuntimeAliasFact, ...]
    storage: Tuple[ParallelStorageFact, ...]
    blockers: Tuple[str, ...]
    sequential_fallback_peak_bytes: int
    parallel_branch_temporary_bytes: int
    parallel_peak_bytes: int
    memory_overhead_vs_sequential: int
    partial_output_bytes: int

    def to_dict(self):
        return {
            **self.__dict__,
            "branches": tuple(item.to_dict() for item in self.branches),
            "conflicts": tuple(item.to_dict() for item in self.conflicts),
            "unresolved_aliases": tuple(
                item.to_dict() for item in self.unresolved_aliases
            ),
            "runtime_aliases": tuple(item.to_dict() for item in self.runtime_aliases),
            "storage": tuple(item.to_dict() for item in self.storage),
        }


@dataclass(frozen=True)
class GraphSubmissionRegionTelemetry:
    """One structured while region from an opt-in asynchronous submission."""

    name: str
    path_id: str
    backend: str
    lowering: str
    control_depth: int
    max_iterations: int
    logical_invocations: int
    logical_iterations: int
    encoded_iterations: int
    masked_iterations: int
    chunk_sizes: Tuple[int, ...]
    chunk_strategies: Tuple[str, ...]
    active_chunk_count: int
    coarse_skipped_chunk_count: int
    initial_counter: int
    final_counter: int
    terminal_predicate: int
    initial_status: Optional[int]
    final_status: Optional[int]
    host_enqueue_ns: int
    gpu_duration_ns: Optional[int]
    gpu_timestamp_exact: bool
    gpu_measurement_path_changed: bool
    gpu_queue_or_stream_id: str
    gpu_timestamp_status: str


@dataclass(frozen=True)
class GraphSubmissionQueueTelemetry:
    """Queue counters attributed to the host transaction window."""

    available: bool
    scope: str
    exact: bool
    queue_submit_calls: int
    submitted_command_buffers: int
    batched_queue_submit_calls: int
    batched_command_buffers: int


@dataclass(frozen=True)
class GraphSubmissionExecutionTelemetry:
    """Submission taxonomy for one ticket, without backend overclaiming."""

    logical_graph_invocations: int
    logical_region_definitions: int
    logical_region_invocations: int
    kernel_submissions: int
    native_submissions: int
    backend_graph_launches: int
    backend_graph_launches_exact: bool
    stream_graph_enqueue_calls: Optional[int]
    stream_graph_enqueue_exact: bool
    physical_queue_submissions: Optional[int]
    physical_queue_submissions_exact: bool
    physical_queue_scope: str


def _materialize_graph_submission_execution_telemetry(
    *, backend, regions, queue, submission_statistics
):
    statistics = submission_statistics or {}
    submission_statistics_exact = bool(statistics.get("_exact", True))
    backend_launches = int(statistics.get("backend_graph_launches", 0))
    logical_graph_invocations = int(statistics.get("graph_submissions", 0))
    # A region snapshot represents one invocation of a top-level structured
    # region. Nested child invocation multiplicity is populated by the nested
    # lowering when it expands child snapshots.
    logical_region_invocations = sum(
        int(getattr(region, "logical_invocations", 1)) for region in regions
    )
    if backend == "cuda":
        stream_graph_enqueue_calls = backend_launches
        stream_graph_enqueue_exact = True
    else:
        stream_graph_enqueue_calls = None
        stream_graph_enqueue_exact = False
    physical_queue_submissions = (
        int(queue.queue_submit_calls) if queue.available else None
    )
    return GraphSubmissionExecutionTelemetry(
        logical_graph_invocations=logical_graph_invocations,
        logical_region_definitions=len(regions),
        logical_region_invocations=logical_region_invocations,
        kernel_submissions=int(statistics.get("kernel_submissions", 0)),
        native_submissions=int(statistics.get("native_submissions", 0)),
        backend_graph_launches=backend_launches,
        backend_graph_launches_exact=submission_statistics_exact,
        stream_graph_enqueue_calls=stream_graph_enqueue_calls,
        stream_graph_enqueue_exact=(
            stream_graph_enqueue_exact and submission_statistics_exact
        ),
        physical_queue_submissions=physical_queue_submissions,
        physical_queue_submissions_exact=bool(queue.available and queue.exact),
        physical_queue_scope=str(queue.scope),
    )


@dataclass(frozen=True)
class GraphPipelineBoundedDispatchReport:
    """Ticket-owned work and launch contract for one bounded dispatch."""

    logical_dispatch_index: int
    physical_dispatch_index: Optional[int]
    label: str
    count_source: str
    count_name: str
    capacity: int
    block_dim: Optional[int]
    ordered: bool
    segment_index: Optional[int]
    segment_count: int
    source_count: Optional[int]
    useful_count: Optional[int]
    executed_count: Optional[int]
    skipped_count: Optional[int]
    encoded_lanes: Optional[int]
    overflow: Optional[bool]
    selected_route: str
    execution_semantics: str
    physical_launch_kind: str
    logical_iteration_exact: bool
    snapshot_status: str


@dataclass(frozen=True)
class GraphPipelineStageReport:
    """One post-optimization execution stage in a ticket-owned report."""

    stage_index: int
    path_id: str
    name: str
    kind: str
    region_kind: str
    dispatch_count: int
    physical_dispatch_count: int
    runtime_arg_names: Tuple[str, ...]
    source_native_count: int
    native_action_count: int
    recordable_native_action_count: int
    opaque_native_action_count: int
    native_backend_eligible: Optional[bool]
    effect_count: int
    declared_temporary_bytes: int
    synchronization: bool
    opaque: bool
    native_actions: Tuple[NativeActionManifest, ...]
    task_mapping_status: str
    bounded_mapping_status: str
    tasks: Tuple["GraphTaskManifest", ...]
    bounded_dispatches: Tuple[GraphPipelineBoundedDispatchReport, ...]
    gpu_duration_ns: Optional[int]
    gpu_timestamp_scope: str
    gpu_timestamp_exact: bool
    gpu_measurement_path_changed: bool
    gpu_queue_or_stream_id: str
    gpu_timestamp_status: str


@dataclass(frozen=True)
class GraphPipelineReport:
    """Immutable post-optimization pipeline snapshot owned by one ticket."""

    schema_version: int
    selection_domain: str
    backend: str
    sequence: int
    stage_count: int
    dispatch_count: int
    physical_dispatch_count: int
    task_count: int
    bounded_dispatch_count: int
    native_action_count: int
    recordable_native_action_count: int
    opaque_native_action_count: int
    declared_temporary_bytes: int
    host_submit_ns: int
    gpu_duration_ns: Optional[int]
    gpu_timestamp_scope: str
    gpu_timestamp_exact: bool
    gpu_measurement_path_changed: bool
    gpu_queue_or_stream_id: str
    gpu_timestamp_status: str
    stages: Tuple[GraphPipelineStageReport, ...]


@dataclass(frozen=True)
class GraphSubmissionTelemetry:
    """Immutable ticket-level structured submission telemetry."""

    schema_version: int
    backend: str
    sequence: int
    regions: Tuple[GraphSubmissionRegionTelemetry, ...]
    queue: GraphSubmissionQueueTelemetry
    execution: GraphSubmissionExecutionTelemetry
    host_submit_ns: int
    device_snapshot_bytes: int
    host_readback_bytes: int
    gpu_duration_ns: Optional[int]
    gpu_timestamp_scope: str
    gpu_timestamp_exact: bool
    gpu_measurement_path_changed: bool
    gpu_queue_or_stream_id: str
    gpu_timestamp_resource_bytes: int
    gpu_timestamp_resource_bytes_known: bool
    gpu_timestamp_status: str
    pipeline: GraphPipelineReport


def _bounded_pipeline_route(domain, backend):
    if domain.count_source == "host_scalar":
        return (
            "exact_host_range",
            "exact_host_range",
            "cpu_dynamic_range" if backend == "cpu" else "host_sized_grid_stride",
            True,
        )
    requirement = domain.physical_grid_requirement
    if backend == "vulkan" and requirement != "fixed_capacity":
        return (
            "exact_indirect",
            "exact_device_grid",
            "indirect_one_to_one",
            True,
        )
    if backend == "cuda":
        if requirement == "adaptive_grid":
            return (
                "adaptive_device_grid_update",
                "exact_device_range",
                "adaptive_saturated_grid_stride",
                True,
            )
        if requirement == "logical_exact":
            return (
                "device_bounded_grid_stride",
                "exact_device_range",
                "saturated_grid_stride",
                True,
            )
        return (
            "masked_capacity",
            "masked_capacity",
            "fixed_capacity_grid_stride",
            False,
        )
    if requirement == "require_exact":
        return (
            "exact_cpu_scheduler",
            "exact_cpu_scheduler",
            "cpu_dynamic_chunks",
            True,
        )
    return (
        "masked_capacity",
        "masked_capacity",
        "fixed_capacity_scheduler",
        False,
    )


def _materialize_bounded_pipeline_dispatch(item, *, backend, snapshot):
    domain = item["domain"]
    route, semantics, launch_kind, logical_exact = _bounded_pipeline_route(
        domain, backend
    )
    source_count = None if snapshot is None else int(snapshot["source_count"])
    overflow = None if snapshot is None else bool(snapshot["overflow"])
    useful = executed = skipped = encoded = None
    snapshot_status = "unavailable" if snapshot is None else snapshot["snapshot_status"]
    if snapshot is not None and domain.ordered:
        # The extent is a reliable aggregate source count, but per-segment
        # useful work also depends on offsets. Do not manufacture a segment
        # value from the aggregate ticket snapshot.
        snapshot_status = "ordered_extent_only"
    elif snapshot is not None:
        useful = min(max(source_count, 0), domain.capacity)
        overflow = bool(overflow or source_count != useful)
        if domain.count_source == "host_scalar":
            executed = useful
            encoded = useful
            if domain.block_dim is not None and useful:
                encoded = (
                    (useful + domain.block_dim - 1) // domain.block_dim
                ) * domain.block_dim
        elif logical_exact and launch_kind == "indirect_one_to_one":
            encoded = useful
            if domain.block_dim is not None and useful:
                encoded = (
                    (useful + domain.block_dim - 1) // domain.block_dim
                ) * domain.block_dim
            executed = min(domain.capacity, encoded)
        elif logical_exact:
            executed = useful
            encoded = useful
        else:
            executed = domain.capacity
            encoded = domain.capacity
        skipped = max(0, executed - useful)
    return GraphPipelineBoundedDispatchReport(
        logical_dispatch_index=int(item["logical_dispatch_index"]),
        physical_dispatch_index=item["physical_dispatch_index"],
        label=str(item["label"]),
        count_source=domain.count_source,
        count_name=domain.extent,
        capacity=domain.capacity,
        block_dim=domain.block_dim,
        ordered=domain.ordered,
        segment_index=domain.segment_index,
        segment_count=domain.segment_count,
        source_count=source_count,
        useful_count=useful,
        executed_count=executed,
        skipped_count=skipped,
        encoded_lanes=encoded,
        overflow=overflow,
        selected_route=route,
        execution_semantics=semantics,
        physical_launch_kind=launch_kind,
        logical_iteration_exact=logical_exact,
        snapshot_status=snapshot_status,
    )


def _materialize_graph_pipeline_report(
    definition,
    *,
    backend,
    sequence,
    host_submit_ns,
    gpu_timing,
    gpu_region_timings,
    bounded_snapshots,
):
    regions = {str(item["path_id"]): item for item in gpu_region_timings}
    stages = []
    for item in definition:
        actions = tuple(item["native_actions"])
        tasks = tuple(item["tasks"])
        bounded_dispatches = tuple(
            _materialize_bounded_pipeline_dispatch(
                dispatch,
                backend=backend,
                snapshot=bounded_snapshots.get(dispatch["snapshot_key"]),
            )
            for dispatch in item["bounded_dispatches"]
        )
        region = regions.get(str(item["path_id"]))
        region_available = bool(region is not None and region["available"])
        declared_temporary_bytes = sum(
            temporary.bytes for action in actions for temporary in action.temporaries
        )
        stages.append(
            GraphPipelineStageReport(
                stage_index=int(item["stage_index"]),
                path_id=str(item["path_id"]),
                name=str(item["name"]),
                kind=str(item["kind"]),
                region_kind=str(item["region_kind"]),
                dispatch_count=int(item["dispatch_count"]),
                physical_dispatch_count=int(item["physical_dispatch_count"]),
                runtime_arg_names=tuple(item["runtime_arg_names"]),
                source_native_count=int(item["source_native_count"]),
                native_action_count=len(actions),
                recordable_native_action_count=sum(
                    bool(action.recordable) for action in actions
                ),
                opaque_native_action_count=sum(
                    bool(action.opaque) for action in actions
                ),
                native_backend_eligible=(
                    all(
                        action.recordable and backend in action.backends
                        for action in actions
                    )
                    if actions
                    else None
                ),
                effect_count=sum(len(action.effects) for action in actions),
                declared_temporary_bytes=declared_temporary_bytes,
                synchronization=bool(item["synchronization"]),
                opaque=bool(item["opaque"]),
                native_actions=actions,
                task_mapping_status=str(item["task_mapping_status"]),
                bounded_mapping_status=str(item["bounded_mapping_status"]),
                tasks=tasks,
                bounded_dispatches=bounded_dispatches,
                gpu_duration_ns=(
                    int(region["duration_ns"]) if region_available else None
                ),
                gpu_timestamp_scope=(
                    "structured_region" if region_available else "unavailable"
                ),
                gpu_timestamp_exact=(
                    bool(region["exact"]) if region is not None else False
                ),
                gpu_measurement_path_changed=(
                    bool(region["measurement_path_changed"])
                    if region is not None
                    else False
                ),
                gpu_queue_or_stream_id=(
                    f"{backend}:{int(region['stream_id'])}"
                    if region is not None
                    else f"{backend}:0"
                ),
                gpu_timestamp_status=(
                    str(region["status"]) if region is not None else "unavailable"
                ),
            )
        )
    gpu_available = bool(gpu_timing["available"])
    return GraphPipelineReport(
        schema_version=GRAPH_PIPELINE_SCHEMA_VERSION,
        selection_domain="post_optimization_execution_root",
        backend=backend,
        sequence=int(sequence),
        stage_count=len(stages),
        dispatch_count=sum(stage.dispatch_count for stage in stages),
        physical_dispatch_count=sum(stage.physical_dispatch_count for stage in stages),
        task_count=sum(len(stage.tasks) for stage in stages),
        bounded_dispatch_count=sum(len(stage.bounded_dispatches) for stage in stages),
        native_action_count=sum(stage.native_action_count for stage in stages),
        recordable_native_action_count=sum(
            stage.recordable_native_action_count for stage in stages
        ),
        opaque_native_action_count=sum(
            stage.opaque_native_action_count for stage in stages
        ),
        declared_temporary_bytes=sum(
            stage.declared_temporary_bytes for stage in stages
        ),
        host_submit_ns=int(host_submit_ns),
        gpu_duration_ns=(int(gpu_timing["duration_ns"]) if gpu_available else None),
        gpu_timestamp_scope=(
            "unavailable"
            if gpu_timing["status"] == "disabled_by_mode"
            else "whole_ticket"
        ),
        gpu_timestamp_exact=bool(gpu_timing["exact"]),
        gpu_measurement_path_changed=bool(gpu_timing["measurement_path_changed"]),
        gpu_queue_or_stream_id=f"{backend}:{int(gpu_timing['stream_id'])}",
        gpu_timestamp_status=str(gpu_timing["status"]),
        stages=tuple(stages),
    )


_COUNTER_FIELDS = (
    "attempts",
    "ordinary_fallbacks",
    "capture_attempts",
    "captures",
    "exact_replays",
    "patched_replays",
    "masked_captures",
    "masked_replays",
    "masked_patched_replays",
    "recaptures",
    "records",
    "replays",
    "structural_fallbacks",
    "transient_failures",
    "retry_backoff_fallbacks",
    "replay_slot_saturation_fallbacks",
    "capture_exceptions",
    "zero_arg_captures",
    "asynchronous_control_updates",
    "deferred_replay_waits",
    "peak_deferred_replay_batches",
)
_BACKEND_GRAPH_PATHS = frozenset(
    (
        "cuda_prepared_binding_plan",
        "cuda_capture",
        "cuda_exact_replay",
        "cuda_patched_replay",
        "cuda_masked_capture",
        "cuda_masked_replay",
        "cuda_masked_patched_replay",
        "cuda_device_update_nested_capture",
        "cuda_device_update_nested_replay",
        "cuda_device_update_nested_patched_replay",
        "vulkan_record",
        "vulkan_replay",
        "vulkan_patched_replay",
    )
)
_BACKEND_REPLAY_PATHS = frozenset(
    (
        "cuda_prepared_binding_plan",
        "cuda_exact_replay",
        "cuda_patched_replay",
        "cuda_masked_replay",
        "cuda_masked_patched_replay",
        "cuda_device_update_nested_replay",
        "cuda_device_update_nested_patched_replay",
        "vulkan_replay",
        "vulkan_patched_replay",
    )
)


def _combine_layout_fingerprints(fingerprints):
    # FNV-1a over sorted fixed-width values. Tree identity and addresses are
    # deliberately absent, so equal layouts produce equal report values.
    value = 14695981039346656037
    prime = 1099511628211
    ordered = sorted(int(item) for item in fingerprints)
    for item in (len(ordered), *ordered):
        for shift in range(0, 64, 8):
            value ^= (item >> shift) & 0xFF
            value = (value * prime) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def _empty_backend_stats():
    stats = {name: 0 for name in _COUNTER_FIELDS}
    stats.update(
        {
            "backend": "none",
            "last_path": "none",
            "last_fallback_reason": "none",
            "zero_arg_eligible": False,
            "known_persistent_argument_bytes": 0,
            "known_bounded_control_bytes": 0,
            "known_bounded_update_groups": 0,
            "known_bounded_updater_dispatches": 0,
            "known_bounded_grouped_payloads": 0,
            "known_bounded_producer_fused_groups": 0,
            "known_bounded_payloads": 0,
            "last_bounded_useful_lanes": 0,
            "last_bounded_physical_blocks": 0,
            "last_bounded_physical_threads": 0,
            "last_bounded_baseline_blocks": 0,
            "last_bounded_zero_payloads": 0,
            "bounded_physical_observation_available": False,
            "known_compiled_tasks": 0,
            "known_compiled_dispatches": 0,
            "last_driver_error": 0,
            "retry_backoff_remaining": 0,
            "consecutive_transient_failures": 0,
            "diagnostics_previously_enabled": False,
            "diagnostics_counters_complete": True,
        }
    )
    return stats


def _flatten_backend_stats(stats):
    for value in stats:
        if isinstance(value, Mapping):
            yield value
        elif isinstance(value, (tuple, list)):
            yield from _flatten_backend_stats(value)


def _backend_name(arch):
    if arch in ("x64", "arm64"):
        return "cpu"
    return arch


def _cuda_structured_control_routes(capabilities=None):
    """Return available CUDA control routes in ordinary auto-policy order."""

    if capabilities is None:
        capabilities = dict(_ti_core.cuda_conditional_graph_capabilities())
    exact = bool(
        capabilities.get(
            "general_graph_exact_control_available",
            capabilities.get("driver_version_eligible", False)
            and capabilities.get("conditional_graph_symbols_loaded", False)
            and capabilities.get("general_device_setter_lowering_compiled", False),
        )
    )
    masked = bool(capabilities.get("internal_masked_graph_available", False))
    return (
        *(("cuda_conditional_graph",) if exact else ()),
        *(("cuda_masked_bounded_graph",) if masked else ()),
    )


def _cuda_structured_control_lowering(capabilities=None, requested_recipe=None):
    """Select one stable CUDA control route for a compiled Graph node."""

    routes = _cuda_structured_control_routes(capabilities)
    requested = "auto" if requested_recipe is None else requested_recipe
    if requested in (
        "cuda_nested_device_update",
        "cuda_nested_masked_bounded",
    ):
        requested = "auto"
    if requested != "auto":
        if requested not in routes:
            raise TaichiRuntimeError(
                f"requested internal structured-control recipe {requested!r} "
                "is unavailable on this CUDA runtime"
            )
        return requested
    if (
        os.environ.get("TI_GRAPH_CUDA_FORCE_MASKED_CONTROL", "0") == "1"
        and "cuda_masked_bounded_graph" in routes
    ):
        return "cuda_masked_bounded_graph"
    if routes:
        return routes[0]
    return None


def _cuda_nested_structured_control_routes():
    """Return the two independently deployable depth-2 CUDA routes."""

    device_update = _cuda_nested_device_update_available()
    flat_capabilities = dict(_ti_core.cuda_conditional_graph_capabilities())
    masked = bool(flat_capabilities.get("internal_masked_graph_available", False))
    return (
        *((_CUDA_NESTED_DEVICE_UPDATE_ROUTE,) if device_update else ()),
        *((_CUDA_NESTED_MASKED_ROUTE,) if masked else ()),
    )


def _cuda_nested_structured_control_lowering(requested_recipe=None):
    """Freeze one depth-2 CUDA physical route during Graph construction."""

    routes = _cuda_nested_structured_control_routes()
    requested = "auto" if requested_recipe is None else requested_recipe
    explicit = {
        "cuda_nested_device_update": _CUDA_NESTED_DEVICE_UPDATE_ROUTE,
        "cuda_nested_masked_bounded": _CUDA_NESTED_MASKED_ROUTE,
        # Preserve the older internal masked override for nested Graphs built
        # under the flat-control worker overlay.
        "cuda_masked_bounded_graph": _CUDA_NESTED_MASKED_ROUTE,
    }.get(requested)
    if explicit is not None:
        if explicit not in routes:
            raise TaichiRuntimeError(
                f"requested internal nested structured-control recipe "
                f"{requested!r} is unavailable on this CUDA runtime"
            )
        return explicit
    if (
        os.environ.get("TI_GRAPH_CUDA_FORCE_MASKED_CONTROL", "0") == "1"
        and _CUDA_NESTED_MASKED_ROUTE in routes
    ):
        return _CUDA_NESTED_MASKED_ROUTE
    if routes:
        return routes[0]
    # The legacy nested runtime always attempted its masked fallback when the
    # setup probe could not establish exact device update. Keep that runtime
    # behavior for ordinary auto Graphs; strict CompileIQ eligibility below
    # still requires both independently reported routes.
    return _CUDA_NESTED_MASKED_ROUTE


def bounded_dispatch_capabilities(physical_grid="auto"):
    """Return one physical-grid policy's device-known dispatch contract."""

    physical_grid = _normalize_bounded_physical_grid_policy(physical_grid)
    backend = _backend_name(_ti_core.arch_name(impl.current_cfg().arch))
    if backend not in ("cpu", "cuda", "vulkan"):
        return {
            "schema_version": DYNAMIC_WORK_SCHEMA_VERSION,
            "backend": backend,
            "available": False,
            "requested_route": "not_applicable",
            "selected_route": "unsupported",
            "route": "unsupported",
            "host_known_route": "unsupported",
            "minimum_driver_api_version": None,
            "driver_api_version": None,
            "driver_version_eligible": False,
            "required_symbols_loaded": False,
            "device_update_ptx_linked": False,
            "setup_probe_passed": False,
            "device_known_count": False,
            "no_host_readback": False,
            "logical_iteration_exact": False,
            "physical_launch_kind": "unsupported",
            "exact_grid": False,
            "exact_physical_grid": False,
            "execution_semantics": "unsupported",
            "range_mapping": "unsupported",
            "masked_capacity": False,
            "zero_count_command_skip": False,
            "ordered_segments": False,
            "global_segment_order": False,
            "producer_owned_launch_state_supported": False,
            "producer_packet_consumed": False,
            "producer_update_policy_requested": "not_applicable",
            "producer_update_policy": "not_applicable",
            "grouped_updates_supported": False,
            "forge_producer_fusion_supported": False,
            "default_preparation_dispatches": 0,
            "updater_dispatches": 0,
            "baseline_capacity_grid": False,
            "fallback_reason": "backend_not_qualified",
            "reason": "backend is not qualified for bounded dispatch",
            "publication_contract": "unavailable",
            "publication_reuse": "unavailable",
            "static_admission": "unsupported",
            "static_admission_reason": "backend_not_qualified",
            "physical_observation": "unavailable",
            "physical_grid_policy": physical_grid,
            "supported_physical_grid_policies": (),
        }
    # Report the default single bounded-dispatch route. Ordered segmented
    # dispatch has its own per-operation lowering and may conservatively fall
    # back when that selected exact route cannot preserve global segment order.
    capabilities = _bounded_route(backend, False, physical_grid=physical_grid)
    if backend == "cuda":
        update_policy_requested, update_policy = _cuda_bounded_update_policy()
    else:
        update_policy_requested = "not_applicable"
        update_policy = "not_applicable"
    if backend == "cuda" and capabilities.requested_route == "auto":
        static_admission = "conservative_saturated"
        static_admission_reason = (
            "static topology does not prove sparse updater amortization"
        )
    else:
        static_admission = "explicit_or_backend_native"
        static_admission_reason = "none"
    publication_reuse = (
        "consecutive_packet"
        if backend == "vulkan"
        else (
            update_policy
            if backend == "cuda" and capabilities.route == "adaptive_device_grid_update"
            else "per_consumer"
        )
    )
    return {
        "schema_version": capabilities.schema_version,
        "backend": backend,
        "available": True,
        "requested_route": capabilities.requested_route,
        "selected_route": capabilities.route,
        "route": capabilities.route,
        "host_known_route": "exact_host_range",
        "minimum_driver_api_version": capabilities.minimum_driver_api_version,
        "driver_api_version": capabilities.driver_api_version,
        "driver_version_eligible": capabilities.driver_version_eligible,
        "required_symbols_loaded": capabilities.required_symbols_loaded,
        "device_update_ptx_linked": capabilities.device_update_ptx_linked,
        "setup_probe_passed": capabilities.setup_probe_passed,
        "device_known_count": capabilities.device_known_count,
        "no_host_readback": capabilities.no_host_readback,
        "logical_iteration_exact": capabilities.logical_iteration_exact,
        "physical_launch_kind": capabilities.physical_launch_kind,
        "exact_grid": capabilities.exact_grid,
        "exact_physical_grid": capabilities.exact_grid,
        "execution_semantics": capabilities.execution_semantics,
        "range_mapping": capabilities.range_mapping,
        "masked_capacity": capabilities.masked_capacity,
        "zero_count_command_skip": capabilities.zero_count_command_skip,
        "ordered_segments": capabilities.ordered_segments,
        "global_segment_order": capabilities.global_segment_order,
        "producer_owned_launch_state_supported": (
            capabilities.producer_owned_launch_state_supported
        ),
        "producer_packet_consumed": backend == "vulkan",
        "producer_update_policy_requested": update_policy_requested,
        "producer_update_policy": update_policy,
        "grouped_updates_supported": (
            backend == "cuda" and capabilities.route == "adaptive_device_grid_update"
        ),
        "forge_producer_fusion_supported": backend == "vulkan",
        "default_preparation_dispatches": capabilities.preparation_dispatches,
        "updater_dispatches": capabilities.preparation_dispatches,
        "baseline_capacity_grid": capabilities.baseline_capacity_grid,
        "fallback_reason": capabilities.fallback_reason,
        "reason": capabilities.reason,
        "publication_contract": "device_extent",
        "publication_reuse": publication_reuse,
        "static_admission": static_admission,
        "static_admission_reason": static_admission_reason,
        "physical_observation": (
            "execution_stats_opt_in" if backend == "cuda" else "handle_snapshot_opt_in"
        ),
        "physical_grid_policy": capabilities.physical_grid_policy,
        "supported_physical_grid_policies": (
            "auto",
            "extent",
            "capacity",
        ),
    }


def structured_control_capabilities():
    """Return the qualified structured-control lowering for this runtime.

    Schema v4 separates compilation, backend qualification, and the complete
    Graph runtime path. Vulkan bounded while regions expose chained and compact
    indirect masking plus bounded asynchronous chunk replay. Automatic
    lowering uses compact masking for the first chunk and, when qualified,
    coarse conditional rendering for later chunks. Vulkan native-required
    while sequences support one asynchronous compound transaction with
    bounded pre-enqueued chunks; branch regions and exact dynamic termination
    remain separate capabilities.
    """
    arch = impl.current_cfg().arch
    backend = _backend_name(_ti_core.arch_name(arch))
    cuda = None
    native = False
    branch_native = False
    structured_submit = False
    primitive = "none"
    rhi_primitive_compiled = False
    rhi_primitive_qualified = False
    runtime_path_compiled = False
    runtime_path_qualified = False
    skip_strategy = "none"
    stops_command_issue_after_exit = False
    exact_dynamic_termination = False
    max_encoded_dispatches = 0
    chunked_runtime_qualified = False
    chunk_iteration_limit = 0
    replay_slot_count = 0
    structured_submit_reason = "native_structured_submission_unavailable"
    compound_structured_submit = False
    compound_max_chunks_per_region = 0
    compound_max_iterations_per_region = 0
    compound_terminal_observation = "unavailable"
    queue_submit_coalescing = False
    compound_per_region_chunk_size = False
    compound_chunk_size_limit = 0
    compound_first_chunk_strategies = ()
    compound_default_first_chunk_strategy = "unavailable"
    submission_ticket_region_telemetry = False
    submission_ticket_queue_telemetry = "unavailable"
    submission_ticket_gpu_timestamps = "unavailable"
    conditional_rendering_available = False
    conditional_rendering_qualified = False
    coarse_conditional_available = False
    coarse_conditional_qualified = False
    compound_tail_strategy = "none"
    queue_submit_policy = "backend_default"
    parallel_indirect_dispatch = False
    parallel_indirect_dispatch_reason = "native_vulkan_graph_replay_unavailable"
    compound_single_preparation = False
    structured_barrier_policy = "unavailable"
    nested_native_compiled = bool(
        (
            arch == _ti_core.Arch.cuda
            and hasattr(
                _ti_core.CompiledGraph,
                "jit_submit_bounded_cuda_nested_sequence_cached",
            )
        )
        or (
            arch == _ti_core.Arch.vulkan
            and hasattr(
                _ti_core.CompiledGraph,
                "jit_run_bounded_vulkan_nested_sequence_cached",
            )
        )
    )
    nested_native = False
    nested_cuda_device_update_candidate = False
    nested_cuda_device_update_qualified = False
    nested_cuda_device_update_forced_off = False
    nested_async_route = "unavailable"
    if arch == _ti_core.Arch.cuda:
        cuda = dict(_ti_core.cuda_conditional_graph_capabilities())
        exact_compiled = bool(
            cuda.get("general_device_setter_lowering_compiled", False)
        )
        masked_compiled = bool(cuda.get("internal_masked_latch_compiled", False))
        exact_native = bool(
            cuda.get(
                "general_graph_exact_control_available",
                cuda.get("driver_version_eligible", False)
                and cuda.get("conditional_graph_symbols_loaded", False)
                and exact_compiled,
            )
        )
        masked_native = bool(cuda.get("internal_masked_graph_available", False))
        native = exact_native or masked_native
        primitive = (
            _cuda_structured_control_lowering(cuda) or "none" if native else "none"
        )
        using_exact = primitive == "cuda_conditional_graph"
        rhi_primitive_compiled = exact_compiled or masked_compiled
        rhi_primitive_qualified = native
        runtime_path_compiled = rhi_primitive_compiled
        runtime_path_qualified = native
        branch_native = native
        structured_submit = native
        compound_structured_submit = native
        compound_terminal_observation = "submission_ticket" if native else "unavailable"
        submission_ticket_region_telemetry = native
        submission_ticket_gpu_timestamps = (
            "opt_in_whole_ticket_and_region" if native else "unavailable"
        )
        structured_submit_reason = (
            "none" if native else "cuda_device_control_unavailable"
        )
        skip_strategy = primitive
        stops_command_issue_after_exit = using_exact
        exact_dynamic_termination = using_exact
        max_encoded_dispatches = 4096 if primitive == "cuda_masked_bounded_graph" else 0
        nested_native = bool(native and nested_native_compiled)
        nested_cuda_device_update_forced_off = (
            os.environ.get("TI_GRAPH_CUDA_FORCE_MASKED_CONTROL", "0") == "1"
        )
        nested_update = dict(_ti_core.cuda_bounded_dispatch_capabilities())
        nested_cuda_device_update_candidate = bool(
            not nested_cuda_device_update_forced_off
            and nested_update.get("driver_version_eligible", False)
            and nested_update.get("required_symbols_loaded", False)
            and nested_update.get("device_update_ptx_compiled", False)
        )
        nested_cuda_device_update_qualified = bool(
            nested_cuda_device_update_candidate
            and nested_update.get("setup_probe_passed", False)
        )
        if nested_native:
            if nested_cuda_device_update_qualified:
                nested_async_route = "cuda_device_node_update"
            elif nested_cuda_device_update_candidate:
                nested_async_route = (
                    "cuda_device_node_update_probe_then_masked_fallback"
                )
            else:
                nested_async_route = "cuda_masked_bounded_graph"
        if native:
            reason = "none"
        elif not exact_compiled and not masked_compiled:
            reason = "cuda_device_control_lowering_not_compiled"
        elif not cuda.get("ordinary_graph_symbols_loaded", False):
            reason = "cuda_graph_capture_symbols_not_loaded"
        else:
            reason = "cuda_device_control_unavailable"
    elif arch == _ti_core.Arch.vulkan:
        conditional_rendering_available = bool(
            impl.get_runtime().prog._vulkan_conditional_rendering_available()
        )
        coarse_conditional_available = conditional_rendering_available
        vulkan_runtime_mode_qualified = bool(
            not impl.current_cfg().kernel_profiler
            and not impl.current_cfg().vulkan_dispatch_cache
        )
        native = vulkan_runtime_mode_qualified
        primitive = "vulkan_dispatch_indirect"
        rhi_primitive_compiled = True
        rhi_primitive_qualified = True
        runtime_path_compiled = True
        runtime_path_qualified = vulkan_runtime_mode_qualified
        coarse_conditional_qualified = bool(native and coarse_conditional_available)
        skip_strategy = (
            "auto_compact_with_coarse_conditional_tail"
            if coarse_conditional_qualified
            else "compact_indirect"
        )
        compound_tail_strategy = (
            "coarse_conditional" if coarse_conditional_qualified else "compact_indirect"
        )
        max_encoded_dispatches = 4096
        chunked_runtime_qualified = vulkan_runtime_mode_qualified
        chunk_iteration_limit = 64
        replay_slot_count = 8
        structured_submit = native
        compound_structured_submit = native
        compound_max_chunks_per_region = 8
        compound_max_iterations_per_region = 512
        compound_terminal_observation = "submission_ticket" if native else "unavailable"
        submission_ticket_region_telemetry = native
        submission_ticket_gpu_timestamps = (
            "opt_in_whole_ticket_and_region_runtime_probe" if native else "unavailable"
        )
        submission_ticket_queue_telemetry = (
            "device_transaction_window" if native else "unavailable"
        )
        queue_submit_coalescing = native
        compound_per_region_chunk_size = native
        compound_chunk_size_limit = 64
        compound_first_chunk_strategies = (
            "compact",
            *(("coarse_conditional",) if coarse_conditional_qualified else ()),
        )
        compound_default_first_chunk_strategy = "compact"
        queue_submit_policy = (
            "transaction_batch_plus_completion_fence" if native else "backend_default"
        )
        structured_submit_reason = (
            "none" if native else "vulkan_runtime_mode_disables_graph_replay"
        )
        parallel_indirect_dispatch = native
        parallel_indirect_dispatch_reason = (
            "none" if native else "vulkan_runtime_mode_disables_graph_replay"
        )
        compound_single_preparation = native
        structured_barrier_policy = (
            "effect_planned_with_controller_boundaries" if native else "unavailable"
        )
        nested_native = bool(
            native and conditional_rendering_available and nested_native_compiled
        )
        if nested_native:
            nested_async_route = "vulkan_conditional_replay"
        reason = (
            "vulkan_if_switch_runtime_not_compiled"
            if vulkan_runtime_mode_qualified
            else "vulkan_runtime_mode_disables_graph_replay"
        )
    else:
        reason = "device_control_is_gpu_only"
    portable_while = (
        "cpu_exact_host_loop" if backend == "cpu" else "portable_exact_or_masked_replay"
    )
    return {
        "schema_version": STRUCTURED_CONTROL_SCHEMA_VERSION,
        "backend": backend,
        "portable": {
            "while": portable_while,
            "if": "host_selected_exact_branch",
            "switch": "host_selected_exact_branch",
            "nested_structured_control": True,
            "max_structured_depth": 2,
            "nested_lowering": "portable_parent_with_qualified_native_leaf_upgrade",
        },
        "nested_structured_control": True,
        "max_structured_depth": 2,
        "nested_native_lowering": nested_native,
        "device_control": {
            "while": native,
            "if": branch_native,
            "switch": branch_native,
            "nested_structured_control": True,
            "max_structured_depth": 2,
            "nested_exact_portable": True,
            "nested_native_lowering": nested_native,
            "nested_native_lowering_compiled": nested_native_compiled,
            "nested_native_kinds": (("while_while",) if nested_native else ()),
            "nested_native_outer_iteration_limit": (64 if nested_native else 0),
            "nested_native_inner_iteration_limit": (64 if nested_native else 0),
            "nested_native_max_ordered_inner_regions": (8 if nested_native else 0),
            "nested_native_max_encoded_actions": (4096 if nested_native else 0),
            "nested_native_stop_telemetry": nested_native,
            "nested_async_route": (
                nested_async_route if nested_native else "unavailable"
            ),
            "nested_cuda_device_update_candidate": (
                nested_cuda_device_update_candidate
            ),
            "nested_cuda_device_update_qualified": (
                nested_cuda_device_update_qualified
            ),
            "nested_cuda_device_update_forced_off": (
                nested_cuda_device_update_forced_off
            ),
            "nested_cuda_fallback_route": (
                "cuda_masked_bounded_graph"
                if nested_native and arch == _ti_core.Arch.cuda
                else "unavailable"
            ),
            "nested_exact_dynamic_termination": False,
            "nested_no_host_readback": nested_native,
            "nested_submit_stop_observation": (
                "device_terminal_packet_or_outer_suffix_trace"
                if nested_native
                else "unavailable"
            ),
            "nested_trace_uses_portable_execution": True,
            "nested_leaf_native_upgrade": native,
            "nested_leaf_native_kinds": (
                ("while", "if", "switch")
                if arch == _ti_core.Arch.cuda and native
                else (("while",) if arch == _ti_core.Arch.vulkan and native else ())
            ),
            "native_max_structured_depth": (
                2 if nested_native else (1 if native else 0)
            ),
            "nested_async_submit": nested_native,
            "structured_submit": structured_submit,
            "structured_submit_reason": structured_submit_reason,
            "parallel_indirect_dispatch": parallel_indirect_dispatch,
            "parallel_indirect_dispatch_reason": (parallel_indirect_dispatch_reason),
            "compound_single_preparation": compound_single_preparation,
            "structured_barrier_policy": structured_barrier_policy,
            "compound_structured_submit": compound_structured_submit,
            "compound_max_chunks_per_region": (compound_max_chunks_per_region),
            "compound_max_iterations_per_region": (compound_max_iterations_per_region),
            "compound_terminal_observation": (compound_terminal_observation),
            "queue_submit_coalescing": queue_submit_coalescing,
            "queue_submit_policy": queue_submit_policy,
            "compound_tail_strategy": compound_tail_strategy,
            "compound_per_region_chunk_size": compound_per_region_chunk_size,
            "compound_chunk_size_limit": compound_chunk_size_limit,
            "compound_first_chunk_strategies": compound_first_chunk_strategies,
            "compound_default_first_chunk_strategy": (
                compound_default_first_chunk_strategy
            ),
            "submission_ticket_region_telemetry": (submission_ticket_region_telemetry),
            "submission_ticket_queue_telemetry": (submission_ticket_queue_telemetry),
            "submission_ticket_gpu_timestamps": (submission_ticket_gpu_timestamps),
            "logical_termination_exact": native,
            "device_controlled_masking": native,
            "per_iteration_host_observation": False,
            "stops_command_issue_after_exit": stops_command_issue_after_exit,
            "exact_dynamic_termination": exact_dynamic_termination,
            "exact_conditional_graph": (
                arch == _ti_core.Arch.cuda and primitive == "cuda_conditional_graph"
            ),
            "bounded_masked_graph": (
                arch == _ti_core.Arch.cuda and primitive == "cuda_masked_bounded_graph"
            ),
            "primitive": primitive,
            "skip_strategy": skip_strategy,
            "rhi_primitive_compiled": rhi_primitive_compiled,
            "rhi_primitive_qualified": rhi_primitive_qualified,
            "runtime_path_compiled": runtime_path_compiled,
            "runtime_path_qualified": runtime_path_qualified,
            "conditional_rendering_available": conditional_rendering_available,
            "conditional_rendering_qualified": conditional_rendering_qualified,
            "coarse_conditional_available": coarse_conditional_available,
            "coarse_conditional_qualified": coarse_conditional_qualified,
            "max_encoded_dispatches": max_encoded_dispatches,
            "chunked_runtime_qualified": chunked_runtime_qualified,
            "chunk_iteration_limit": chunk_iteration_limit,
            "replay_slot_count": replay_slot_count,
            "available_strategies": (
                (
                    "chained_indirect",
                    "compact_indirect",
                    *(("conditional",) if conditional_rendering_available else ()),
                    *(("coarse_conditional",) if coarse_conditional_available else ()),
                )
                if arch == _ti_core.Arch.vulkan
                else ()
            ),
            "qualified_strategies": (
                (
                    "compact_indirect",
                    *(("coarse_conditional",) if coarse_conditional_qualified else ()),
                )
                if arch == _ti_core.Arch.vulkan and native
                else ()
            ),
            "chained_runtime_qualified": False,
            "chained_max_encoded_dispatches": (
                256 if arch == _ti_core.Arch.vulkan else 0
            ),
            "max_control_bytes_per_slot": (
                64 * 1024 if arch == _ti_core.Arch.vulkan else 0
            ),
            "unsupported_reason": reason,
        },
        "cuda_conditional_graph": cuda,
    }


def dynamic_work_capabilities():
    """Return one backend-honest view of dynamic launch and iteration.

    Device-count dispatch and structured iteration are deliberately separate
    axes.  CUDA conditional Graph can stop an iterative command stream exactly
    without providing CUDA indirect grid launch; conversely Vulkan can consume
    an exact indirect grid while a bounded structured loop still encodes a
    masked command budget.
    """

    bounded = bounded_dispatch_capabilities()
    structured = structured_control_capabilities()
    control = structured["device_control"]
    if control["exact_dynamic_termination"]:
        iteration_semantics = "exact_dynamic_termination"
    elif control["runtime_path_qualified"]:
        iteration_semantics = "bounded_masked_encoding"
    else:
        iteration_semantics = "portable_host_control"
    ticket_observation = structured["backend"] in ("cpu", "cuda", "vulkan")
    observation_readback_mode = _observation_readback_mode()
    completion_attached = observation_readback_mode.startswith("completion_attached_")
    return {
        "schema_version": DYNAMIC_WORK_SCHEMA_VERSION,
        "backend": structured["backend"],
        "count_contract": {
            "owner": "DeviceExtent",
            "state_words": 2,
            "fixed_capacity": True,
            "device_published_count": True,
            "sticky_overflow": True,
            "runtime_generation_qualified": True,
            "replay_host_readback": False,
        },
        "bounded_dispatch": {
            "available": bounded["available"],
            "requested_route": bounded["requested_route"],
            "selected_route": bounded["selected_route"],
            "route": bounded["route"],
            "minimum_driver_api_version": bounded["minimum_driver_api_version"],
            "driver_api_version": bounded["driver_api_version"],
            "driver_version_eligible": bounded["driver_version_eligible"],
            "required_symbols_loaded": bounded["required_symbols_loaded"],
            "device_update_ptx_linked": bounded["device_update_ptx_linked"],
            "setup_probe_passed": bounded["setup_probe_passed"],
            "execution_semantics": bounded["execution_semantics"],
            "range_mapping": bounded["range_mapping"],
            "device_known_count": bounded["device_known_count"],
            "no_host_readback": bounded["no_host_readback"],
            "logical_iteration_exact": bounded["logical_iteration_exact"],
            "physical_launch_kind": bounded["physical_launch_kind"],
            "exact_physical_grid": bounded["exact_grid"],
            "masked_capacity": bounded["masked_capacity"],
            "zero_count_command_skip": bounded["zero_count_command_skip"],
            "producer_owned_launch_state": bounded[
                "producer_owned_launch_state_supported"
            ],
            "producer_packet_consumed": bounded["producer_packet_consumed"],
            "producer_update_policy_requested": bounded[
                "producer_update_policy_requested"
            ],
            "producer_update_policy": bounded["producer_update_policy"],
            "grouped_updates_supported": bounded["grouped_updates_supported"],
            "forge_producer_fusion_supported": bounded[
                "forge_producer_fusion_supported"
            ],
            "default_preparation_dispatches": bounded["default_preparation_dispatches"],
            "updater_dispatches": bounded["updater_dispatches"],
            "baseline_capacity_grid": bounded["baseline_capacity_grid"],
            "fallback_reason": bounded["fallback_reason"],
            "publication_contract": bounded["publication_contract"],
            "publication_reuse": bounded["publication_reuse"],
            "static_admission": bounded["static_admission"],
            "static_admission_reason": bounded["static_admission_reason"],
            "physical_observation": bounded["physical_observation"],
            "accounting_fields": (
                "useful_count",
                "capacity",
                "executed_count",
                "skipped_count",
                "encoded_lanes",
                "overflow",
                "physical_blocks",
                "physical_threads",
            ),
        },
        "structured_iteration": {
            "available": control["runtime_path_qualified"],
            "route": control["primitive"],
            "execution_semantics": iteration_semantics,
            "logical_termination_exact": control["logical_termination_exact"],
            "command_termination_exact": control["exact_dynamic_termination"],
            "stops_command_issue_after_exit": control["stops_command_issue_after_exit"],
            "bounded_masked_encoding": control["bounded_masked_graph"]
            or (
                structured["backend"] == "vulkan" and control["runtime_path_qualified"]
            ),
            "max_encoded_dispatches": control["max_encoded_dispatches"],
        },
        "worklist": {
            "available": bounded["available"],
            "fixed_capacity": True,
            "front_back_storage": True,
            "device_atomic_append": True,
            "atomic_append_finalize_required": True,
            "capacity_mismatch_fail_closed": True,
            "atomic_append_order": "unspecified",
            "concurrent_producer_contract": "single_writer_per_transition",
            "stable_selection": True,
            "deterministic_keyed_claim": True,
            "claim_order": "key_priority_ordinal",
            "claim_parallelism": "one_winner_scan_per_key",
            "recordable_sequence": True,
            "no_replay_allocation": True,
            "no_replay_host_readback": True,
            "performance_crossover": "workload_dependent",
            "physical_launch_semantics": bounded["execution_semantics"],
            "counter_fields": (
                "generated",
                "accepted",
                "rejected",
                "conflicts",
                "winners",
                "overflow",
            ),
        },
        "observation": {
            "submission_ticket": ticket_observation,
            "completion_attached": completion_attached,
            "readback_mode": observation_readback_mode,
            "per_iteration_host_observation": False,
            "worklist_counters": ticket_observation,
            "fallback_reason": (
                "none" if completion_attached else "disabled_or_backend_unqualified"
            ),
        },
    }


def _execution_report(
    definition,
    arch,
    lifecycle_state,
    instance_kind,
    backend_stats,
    observation_staging_bytes=0,
    temporary_memory_plan=None,
    temporary_arena_stats=None,
    observation_arena_stats=None,
    telemetry_arena_stats=None,
    internal_storage_stats=None,
    provider_memory=(),
    storage_pools=(),
):
    flat_backend_stats = tuple(_flatten_backend_stats(backend_stats))
    segments = []
    stats_cursor = 0
    for index, node in enumerate(definition["nodes"]):
        kind = node["kind"]
        if kind != "cgraph":
            path = (
                "unavailable"
                if lifecycle_state != "ready"
                else (
                    "asynchronous_snapshot"
                    if kind == "observation"
                    else (
                        "portable_host_control"
                        if kind == "structured_sequence"
                        else (
                            "native_replay"
                            if instance_kind
                            in ("cuda_native_replay", "cpu_native_replay")
                            else "native_dispatch"
                        )
                    )
                )
            )
            segments.append(
                GraphExecutionSegmentReport(
                    node_index=index,
                    kind=kind,
                    dispatch_count=node["dispatch_count"],
                    compiled_task_count=None,
                    runtime_arg_count=node["runtime_arg_count"],
                    static_dependency_count=0,
                    static_layout_fingerprint=_combine_layout_fingerprints(()),
                    backend=_backend_name(arch),
                    last_path=path,
                    fallback_reason=(
                        lifecycle_state if lifecycle_state != "ready" else "none"
                    ),
                    backend_graph_path=False,
                    backend_replay_path=False,
                    zero_arg_eligible=False,
                    persistent_argument_bytes=0,
                    persistent_bounded_control_bytes=0,
                    bounded_update_groups=0,
                    bounded_updater_dispatches=0,
                    bounded_grouped_payloads=0,
                    bounded_producer_fused_groups=0,
                    bounded_payloads=0,
                    bounded_last_useful_lanes=0,
                    bounded_last_physical_blocks=0,
                    bounded_last_physical_threads=0,
                    bounded_last_baseline_blocks=0,
                    bounded_last_zero_payloads=0,
                    bounded_physical_observation_available=False,
                    bounded_update_replays=0,
                    bounded_update_state_changes=0,
                    bounded_update_cache_hits=0,
                    bounded_node_api_calls=0,
                    bounded_max_group_size=0,
                    last_driver_error=0,
                    retry_backoff_remaining=0,
                    consecutive_transient_failures=0,
                    counters_complete=True,
                    counters=GraphExecutionCounters(
                        **{name: 0 for name in _COUNTER_FIELDS}
                    ),
                    replay_attribution=_replay_attribution({}, enabled=False),
                )
            )
            continue

        stats = (
            flat_backend_stats[stats_cursor]
            if stats_cursor < len(flat_backend_stats)
            else _empty_backend_stats()
        )
        stats_cursor += 1
        known_dispatches = int(stats.get("known_compiled_dispatches", 0))
        tasks = (
            int(stats.get("known_compiled_tasks", 0))
            if known_dispatches
            == node.get("physical_dispatch_count", node["dispatch_count"])
            else None
        )
        backend = stats.get("backend", "none")
        if backend == "none":
            backend = _backend_name(arch)
        path = stats.get("last_path", "none")
        if lifecycle_state != "ready":
            path = "unavailable"
        elif path == "none" and tasks is not None:
            path = "ordinary"
        elif path == "none":
            path = "not_run"
        fallback_reason = stats.get("last_fallback_reason", "none")
        if lifecycle_state != "ready":
            fallback_reason = lifecycle_state
        gpu_backend = backend in ("cuda", "vulkan")
        counters_complete = not gpu_backend or bool(
            stats.get("diagnostics_counters_complete", True)
        )
        segments.append(
            GraphExecutionSegmentReport(
                node_index=index,
                kind=kind,
                dispatch_count=node["dispatch_count"],
                compiled_task_count=tasks,
                runtime_arg_count=node["runtime_arg_count"],
                static_dependency_count=len(node["dependency_info"]),
                static_layout_fingerprint=_combine_layout_fingerprints(
                    dependency[2] for dependency in node["dependency_info"]
                ),
                backend=backend,
                last_path=path,
                fallback_reason=fallback_reason,
                backend_graph_path=path in _BACKEND_GRAPH_PATHS,
                backend_replay_path=path in _BACKEND_REPLAY_PATHS,
                zero_arg_eligible=bool(stats.get("zero_arg_eligible", False)),
                persistent_argument_bytes=int(
                    stats.get("known_persistent_argument_bytes", 0)
                ),
                persistent_bounded_control_bytes=int(
                    stats.get("known_bounded_control_bytes", 0)
                ),
                bounded_update_groups=int(stats.get("known_bounded_update_groups", 0)),
                bounded_updater_dispatches=int(
                    stats.get("known_bounded_updater_dispatches", 0)
                ),
                bounded_grouped_payloads=int(
                    stats.get("known_bounded_grouped_payloads", 0)
                ),
                bounded_producer_fused_groups=int(
                    stats.get("known_bounded_producer_fused_groups", 0)
                ),
                bounded_payloads=int(stats.get("known_bounded_payloads", 0)),
                bounded_last_useful_lanes=int(
                    stats.get("last_bounded_useful_lanes", 0)
                ),
                bounded_last_physical_blocks=int(
                    stats.get("last_bounded_physical_blocks", 0)
                ),
                bounded_last_physical_threads=int(
                    stats.get("last_bounded_physical_threads", 0)
                ),
                bounded_last_baseline_blocks=int(
                    stats.get("last_bounded_baseline_blocks", 0)
                ),
                bounded_last_zero_payloads=int(
                    stats.get("last_bounded_zero_payloads", 0)
                ),
                bounded_physical_observation_available=bool(
                    stats.get("bounded_physical_observation_available", False)
                ),
                bounded_update_replays=int(stats.get("bounded_update_replays", 0)),
                bounded_update_state_changes=int(
                    stats.get("bounded_update_state_changes", 0)
                ),
                bounded_update_cache_hits=int(
                    stats.get("bounded_update_cache_hits", 0)
                ),
                bounded_node_api_calls=int(stats.get("bounded_node_api_calls", 0)),
                bounded_max_group_size=int(
                    stats.get("known_bounded_max_group_size", 0)
                ),
                last_driver_error=int(stats.get("last_driver_error", 0)),
                retry_backoff_remaining=int(stats.get("retry_backoff_remaining", 0)),
                consecutive_transient_failures=int(
                    stats.get("consecutive_transient_failures", 0)
                ),
                counters_complete=counters_complete,
                counters=GraphExecutionCounters(
                    **{name: int(stats.get(name, 0)) for name in _COUNTER_FIELDS}
                ),
                replay_attribution=_replay_attribution(stats),
            )
        )

    cgraph_segments = tuple(s for s in segments if s.kind == "cgraph")
    task_counts = tuple(s.compiled_task_count for s in cgraph_segments)
    compiled_task_count = (
        sum(task_counts)
        if task_counts and all(value is not None for value in task_counts)
        else None
    )
    path_segments = cgraph_segments or tuple(segments)
    paths = {segment.last_path for segment in path_segments}
    execution_path = (
        lifecycle_state
        if lifecycle_state != "ready"
        else (next(iter(paths)) if len(paths) == 1 else "mixed") if paths else "not_run"
    )
    reasons = {
        segment.fallback_reason
        for segment in segments
        if segment.fallback_reason != "none"
    }
    fallback_reason = (
        "none" if not reasons else next(iter(reasons)) if len(reasons) == 1 else "mixed"
    )
    dependency_info = definition["dependency_info"]
    # Structured nodes own condition/body JIT caches and, on qualified
    # backends, an additional native replay cache. They are intentionally not
    # expanded into public CGraph segments, but their stable argument,
    # control-arena, and observation allocations are still Graph-owned
    # persistent memory. Count every leaf cache once instead of restricting
    # memory accounting to top-level CGraph segments.
    internal_storage_stats = internal_storage_stats or {}
    persistent_argument_bytes = sum(
        int(stats.get("known_persistent_argument_bytes", 0))
        for stats in flat_backend_stats
    ) + int(
        internal_storage_stats.get(
            "reserved_bytes", definition.get("internal_storage_bytes", 0)
        )
    )
    persistent_bounded_control_bytes = sum(
        int(stats.get("known_bounded_control_bytes", 0)) for stats in flat_backend_stats
    )
    temporary_memory_plan = temporary_memory_plan or {}
    temporary_arena_stats = temporary_arena_stats or {}
    observation_arena_stats = observation_arena_stats or {}
    telemetry_arena_stats = telemetry_arena_stats or {}
    temporary_plan_materialized = bool(temporary_arena_stats.get("materialized", False))
    planned_temporary_bytes = int(temporary_memory_plan.get("planned_peak_bytes", 0))
    persistent_temporary_bytes = int(temporary_arena_stats.get("reserved_bytes", 0))
    persistent_observation_bytes = int(observation_staging_bytes) + int(
        observation_arena_stats.get("reserved_bytes", 0)
    )
    persistent_telemetry_bytes = int(telemetry_arena_stats.get("reserved_bytes", 0))
    provider_memory = tuple(provider_memory)
    memory = GraphMemoryReport(
        persistent_argument_bytes=persistent_argument_bytes,
        persistent_bounded_control_bytes=persistent_bounded_control_bytes,
        persistent_observation_bytes=persistent_observation_bytes,
        persistent_temporary_bytes=persistent_temporary_bytes,
        persistent_bytes=(
            persistent_argument_bytes
            + persistent_observation_bytes
            + persistent_temporary_bytes
            + persistent_telemetry_bytes
        ),
        transient_temporary_bytes=(
            planned_temporary_bytes if temporary_plan_materialized else 0
        ),
        planned_temporary_bytes=planned_temporary_bytes,
        temporary_reuse_bytes=int(temporary_memory_plan.get("reused_bytes", 0)),
        opaque_temporary_bytes=int(temporary_memory_plan.get("opaque_bytes", 0)),
        temporary_plan_materialized=temporary_plan_materialized,
        temporary_arena_capacity=int(temporary_arena_stats.get("capacity", 0)),
        temporary_arena_slots=int(temporary_arena_stats.get("slots", 0)),
        temporary_arena_allocations=int(temporary_arena_stats.get("allocations", 0)),
        temporary_arena_reuses=int(temporary_arena_stats.get("reuses", 0)),
        temporary_arena_waits=int(temporary_arena_stats.get("waits", 0)),
        observation_arena_capacity=int(observation_arena_stats.get("capacity", 0)),
        observation_arena_slots=int(observation_arena_stats.get("slots", 0)),
        observation_arena_allocations=int(
            observation_arena_stats.get("allocations", 0)
        ),
        observation_arena_reuses=int(observation_arena_stats.get("reuses", 0)),
        observation_arena_waits=int(observation_arena_stats.get("waits", 0)),
        observation_materializations=int(
            observation_arena_stats.get("materializations", 0)
        ),
        observation_host_readback_bytes=int(
            observation_arena_stats.get("host_readback_bytes", 0)
        ),
        observation_readback_mode=str(
            observation_arena_stats.get("readback_mode", "unavailable")
        ),
        observation_completion_attached=(
            str(observation_arena_stats.get("readback_mode", "")).startswith(
                "completion_attached_"
            )
        ),
        persistent_telemetry_bytes=persistent_telemetry_bytes,
        telemetry_arena_capacity=int(telemetry_arena_stats.get("capacity", 0)),
        telemetry_arena_slots=int(telemetry_arena_stats.get("slots", 0)),
        telemetry_arena_allocations=int(telemetry_arena_stats.get("allocations", 0)),
        telemetry_arena_reuses=int(telemetry_arena_stats.get("reuses", 0)),
        telemetry_arena_waits=int(telemetry_arena_stats.get("waits", 0)),
        telemetry_materializations=int(
            telemetry_arena_stats.get("materializations", 0)
        ),
        telemetry_host_readback_bytes=int(
            telemetry_arena_stats.get("host_readback_bytes", 0)
        ),
        persistent_internal_storage_bytes=int(
            internal_storage_stats.get("reserved_bytes", 0)
        ),
        internal_storage_exclusive=bool(internal_storage_stats.get("exclusive", False)),
        internal_storage_waits=int(internal_storage_stats.get("waits", 0)),
        internal_storage_reuses=int(internal_storage_stats.get("reuses", 0)),
        workspace_lane_capacity=int(internal_storage_stats.get("lane_capacity", 1)),
        workspace_lanes_materialized=int(
            internal_storage_stats.get("lanes_materialized", 1)
        ),
        workspace_lanes_busy=int(internal_storage_stats.get("lanes_busy", 0)),
        workspace_lane_acquisitions=int(
            internal_storage_stats.get("lane_acquisitions", 0)
        ),
        workspace_lane_waits=int(internal_storage_stats.get("lane_waits", 0)),
        workspace_lane_saturation_errors=int(
            internal_storage_stats.get("lane_saturation_errors", 0)
        ),
        workspace_lane_saturation_policy=str(
            internal_storage_stats.get("lane_saturation_policy", "wait")
        ),
        provider_generation_report_count=len(provider_memory),
        provider_generation_known_resident_requested_bytes=sum(
            report.known_resident_requested_bytes for report in provider_memory
        ),
        provider_generation_known_capacity_requested_bytes=sum(
            report.known_capacity_requested_bytes for report in provider_memory
        ),
        provider_generation_requested_bytes_complete=all(
            report.resident_requested_bytes_complete for report in provider_memory
        ),
        provider_generation_opaque_component_count=sum(
            report.opaque_component_count for report in provider_memory
        ),
        opaque_driver_bytes=None,
    )
    return GraphExecutionReport(
        schema_version=7,
        arch=arch,
        lifecycle_state=lifecycle_state,
        node_count=len(segments),
        cgraph_segment_count=len(cgraph_segments),
        native_node_count=definition["native_count"],
        observation_node_count=definition.get("observation_count", 0),
        dispatch_count=definition["dispatch_count"],
        compiled_task_count=compiled_task_count,
        runtime_arg_count=definition["runtime_arg_count"],
        static_dependency_count=len(dependency_info),
        static_layout_fingerprint=_combine_layout_fingerprints(
            dependency[2] for dependency in dependency_info
        ),
        execution_path=execution_path,
        fallback_reason=fallback_reason,
        backend_graph_segments=sum(
            segment.backend_graph_path for segment in cgraph_segments
        ),
        backend_replay_segments=sum(
            segment.backend_replay_path for segment in cgraph_segments
        ),
        ordinary_fallback_segments=sum(
            segment.last_path in ("ordinary", "ordinary_fallback")
            for segment in cgraph_segments
        ),
        counters_complete=all(segment.counters_complete for segment in segments),
        segments=tuple(segments),
        memory=memory,
        provider_memory=provider_memory,
        storage_pools=tuple(storage_pools),
    )


class _NativeReplayExecutable:
    def __init__(self, nodes):
        self._nodes = tuple(nodes)

    def prewarm(self):
        for node in self._nodes:
            node.executable.prewarm()
        return self

    def run(self, context, temporaries=None):
        for node in self._nodes:
            node.run(context, temporaries)


class _CGraphJITExecutable:
    def __init__(self, compiled_graph):
        self.compiled_graph = compiled_graph
        self._jit_cache = _ti_core.CompiledGraphJITCache()

    def prewarm(self):
        return self

    def run(self, context, temporaries=None):
        self.compiled_graph.jit_run_cached(
            context.compile_config(), context.flattened_args(), self._jit_cache
        )

    def run_cuda_concurrent_batch(self, context):
        self.compiled_graph.jit_run_cached_cuda_concurrent_batch(
            context.compile_config(), context.flattened_args(), self._jit_cache
        )

    def invalidate_runtime(self, preserve_executables=False):
        if preserve_executables:
            self._jit_cache.retire_snode_tree_runtime_state()
        else:
            self._jit_cache.clear_runtime_state()

    @property
    def debug_graph_stats(self):
        return self._jit_cache._debug_graph_stats()

    @property
    def snapshot_graph_stats(self):
        return self._jit_cache._debug_graph_stats(False)


class _GraphRunContext:
    _empty_args = {}

    def __init__(self):
        self._args = None
        self._flattened_args = None
        self._compile_config = None
        self._last_arg_signature = None
        self._last_flattened = None
        self._trace_recorder = None

    def begin(
        self,
        args,
        fixed_args=None,
        trace_recorder=None,
        *,
        flattened_args=None,
    ):
        if flattened_args is not None and fixed_args:
            raise TaichiRuntimeError(
                "A preflattened Graph binding frame cannot be merged with fixed bindings"
            )
        if fixed_args:
            overlap = fixed_args.keys() & args.keys()
            if overlap:
                raise TaichiRuntimeError(
                    "Graph runtime arguments collide with provider-owned "
                    "fixed bindings: " + ", ".join(sorted(overlap))
                )
            merged = dict(fixed_args)
            merged.update(args)
            self._args = merged
        else:
            self._args = args
        self._flattened_args = flattened_args
        self._trace_recorder = trace_recorder

    def end(self):
        # Runtime resource completion is owned by the native Program registry.
        # Keeping the Python argument dict here after submission only delays
        # wrapper retirement and can pin arbitrarily large user object graphs.
        # The generation-qualified flattened fast cache remains reusable.
        self._args = None
        self._flattened_args = None
        self._trace_recorder = None

    def begin_control_trace(self, node):
        recorder = self._trace_recorder
        if recorder is None:
            return None
        return recorder.begin(node)

    def set_control_trace_iteration(self, node, iteration):
        recorder = self._trace_recorder
        if recorder is not None:
            recorder.set_iteration(node, iteration)

    def end_control_trace(self, frame, report):
        if frame is not None:
            self._trace_recorder.end(frame, report)

    def abort_control_trace(self, frame):
        if frame is not None:
            self._trace_recorder.abort(frame)

    def control_trace_enabled(self):
        return self._trace_recorder is not None

    def runtime_args(self):
        return self._args

    def compile_config(self):
        if self._compile_config is None:
            self._compile_config = impl.get_runtime().prog.config()
        return self._compile_config

    def flattened_args(self, arg_names=None):
        if self._flattened_args is None:
            self._flattened_args = self._flatten_runtime_args(self._args)
        if arg_names is not None and not arg_names.issubset(self._flattened_args):
            raise TaichiRuntimeError(
                "CGraph segment requested undeclared runtime arguments"
            )
        # The CompiledGraph Python binding iterates the compiled segment's own
        # declarations and constructs a node-local C++ IValue map. Passing the
        # shared flattened dict here is therefore a zero-copy filtered view;
        # slicing it again in Python adds two dict copies to every mixed run.
        return self._flattened_args

    def _flatten_runtime_args(self, args):
        if not args:
            return self._empty_args

        from taichi_forge.lang.device_extent import DeviceExtent

        signature = []
        dynamic_items = []
        arch = self.compile_config().arch
        runtime_storage_backend = arch in (
            _ti_core.Arch.x64,
            _ti_core.Arch.arm64,
            _ti_core.Arch.cuda,
            _ti_core.Arch.vulkan,
        )
        if arch == _ti_core.Arch.cuda:
            ndarray_consumer = "graph_capture"
            ndarray_mode = "capture"
        else:
            ndarray_consumer = "graph_replay"
            ndarray_mode = "replay"
        for k, v in args.items():
            if isinstance(v, DeviceExtent):
                v._validate_current()
                signature.append((k, "device_extent", v.binding.allocation_identity))
            elif isinstance(v, (Ndarray, ProviderOwnedNdarrayBinding)):
                if v.arr is None:
                    raise TaichiRuntimeError(
                        "Cannot submit an Ndarray to Graph.run() after its Taichi runtime has been reset"
                    )
                signature.append((k, "ndarray", v._runtime_allocation_identity))
            elif isinstance(v, (DenseNdarrayView, ScalarField, MatrixField)):
                signature.append((k, "dense_storage", id(v)))
            elif isinstance(v, Texture):
                if v.tex is None:
                    raise TaichiRuntimeError(
                        "Cannot submit a Texture to Graph.run() after its Taichi runtime has been reset"
                    )
                signature.append((k, "texture", id(v), id(v.tex)))
            elif isinstance(v, Matrix):
                signature.append((k, "matrix"))
                dynamic_items.append((k, v.entries))
            elif isinstance(v, (int, float)):
                signature.append((k, "scalar", type(v)))
                dynamic_items.append((k, v))
            else:
                raise TaichiRuntimeError(
                    "Only Python scalars, ti.Matrix, ti.Ndarray, DeviceExtent, "
                    "canonical dense Field, and DenseNdarrayView are supported "
                    "as "
                    f"runtime arguments but got {type(v)}"
                )

        signature = tuple(signature)
        if signature == self._last_arg_signature:
            flattened = self._last_flattened
        else:
            flattened = {}
            for k, v in args.items():
                if isinstance(v, DeviceExtent):
                    state = v.state
                    if runtime_storage_backend:
                        flattened[k] = (
                            state.arr,
                            state._runtime_storage_argument(
                                ndarray_consumer, ndarray_mode
                            ),
                        )
                    else:
                        flattened[k] = state.arr
                elif isinstance(v, (Ndarray, ProviderOwnedNdarrayBinding)):
                    if runtime_storage_backend:
                        flattened[k] = (
                            v.arr,
                            v._runtime_storage_argument(ndarray_consumer, ndarray_mode),
                        )
                    else:
                        flattened[k] = v.arr
                elif isinstance(v, (DenseNdarrayView, ScalarField, MatrixField)):
                    if not runtime_storage_backend:
                        raise TaichiRuntimeError(
                            "Dense storage Graph runtime arguments are supported "
                            "on CPU, CUDA, and Vulkan"
                        )
                    view = v if isinstance(v, DenseNdarrayView) else ndarray_view(v)
                    try:
                        runtime_argument = view._runtime_storage_argument(
                            ndarray_consumer, ndarray_mode
                        )
                    except ValueError:
                        if arch != _ti_core.Arch.cuda:
                            raise
                        runtime_argument = view._runtime_storage_argument(
                            "graph_replay", "replay"
                        )
                    flattened[k] = (view, runtime_argument)
                elif isinstance(v, Texture):
                    flattened[k] = v.tex
            self._last_arg_signature = signature
            self._last_flattened = flattened
        for k, v in dynamic_items:
            flattened[k] = v
        return flattened


class _CompiledCGraphNode:
    needs_runtime_args = True

    def __init__(
        self,
        compiled_graph,
        dispatch_count,
        runtime_arg_names=(),
        ir_node=None,
        recording_dispatches=(),
        lifetime_leases=(),
        source_native_count=0,
        region_kind="cgraph",
        fixed_runtime_args=None,
        temporary_actions=(),
        native_action_manifests=(),
        recipe_operations=(),
        parallel_dispatch_groups=(),
    ):
        self.compiled_graph = compiled_graph
        self.dispatch_count = dispatch_count
        composer_stats = dict(getattr(compiled_graph, "_composer_stats", {}))
        self.physical_dispatch_count = int(
            composer_stats.get("physical_dispatches", dispatch_count)
        )
        self.composer_applied_groups = int(composer_stats.get("applied_groups", 0))
        self.composer_source_groups = tuple(
            tuple(None if item is None else f"dispatch:{int(item)}" for item in group)
            for group in composer_stats.get("source_groups", ())
        )
        self.composer_lowering_available = bool(
            composer_stats.get("lowering_available", False)
        )
        self.recording_runtime_arg_names = frozenset(runtime_arg_names)
        self.fixed_runtime_args = dict(
            {} if fixed_runtime_args is None else fixed_runtime_args
        )
        self.temporary_actions = tuple(temporary_actions)
        self.native_action_manifests = tuple(native_action_manifests)
        self.recipe_operations = tuple(recipe_operations)
        self.parallel_dispatch_groups = tuple(
            tuple(int(index) for index in group) for group in parallel_dispatch_groups
        )

        if not all(
            isinstance(manifest, NativeActionManifest)
            for manifest in self.native_action_manifests
        ):
            raise TaichiRuntimeError(
                "CGraph native action manifests must contain "
                "NativeActionManifest values"
            )
        self.temporary_runtime_arg_names = frozenset().union(
            *(frozenset(action.temporary_bindings) for action in self.temporary_actions)
        )
        self.derived_runtime_arg_names = frozenset(
            binding.name
            for manifest in self.native_action_manifests
            for binding in manifest.derived_runtime_bindings
        )
        if not self.fixed_runtime_args.keys() <= self.recording_runtime_arg_names:
            raise TaichiRuntimeError(
                "CGraph fixed bindings must be declared runtime arguments"
            )
        if not self.temporary_runtime_arg_names <= self.recording_runtime_arg_names:
            raise TaichiRuntimeError(
                "CGraph temporary bindings must be declared runtime arguments"
            )
        if not self.derived_runtime_arg_names <= self.recording_runtime_arg_names:
            raise TaichiRuntimeError(
                "CGraph derived bindings must be declared runtime arguments"
            )
        if self.fixed_runtime_args.keys() & self.temporary_runtime_arg_names:
            raise TaichiRuntimeError(
                "CGraph fixed and temporary bindings must be disjoint"
            )
        self.runtime_arg_names = self.recording_runtime_arg_names.difference(
            (
                *self.fixed_runtime_args,
                *self.temporary_runtime_arg_names,
                *self.derived_runtime_arg_names,
            )
        )
        self.recording_dispatches = tuple(
            _normalize_recording_dispatch(dispatch) for dispatch in recording_dispatches
        )
        self.lifetime_leases = tuple(lifetime_leases)
        self.source_native_count = int(source_native_count)
        if self.source_native_count != len(self.native_action_manifests):
            raise TaichiRuntimeError(
                "CGraph source_native_count must match its native action manifests"
            )
        self.region_kind = region_kind
        self.ir_node = ir_node or SequentialRegion(
            tuple(
                DispatchNode(name=f"dispatch_{index}")
                for index in range(dispatch_count)
            ),
            name="cgraph",
        )
        dependency_info = getattr(compiled_graph, "_snode_tree_dependency_info", None)
        if dependency_info is None:
            dependency_info = (
                (*dependency, 0)
                for dependency in getattr(
                    compiled_graph, "_snode_tree_dependencies", ()
                )
            )
        self.snode_tree_dependency_info = frozenset(
            tuple(dependency) for dependency in dependency_info
        )
        self.snode_tree_dependencies = frozenset(
            dependency[:2] for dependency in self.snode_tree_dependency_info
        )
        self._jit_cache = _ti_core.CompiledGraphJITCache()

    def run(self, context, temporaries=None):
        self.compiled_graph.jit_run_cached(
            context.compile_config(),
            context.flattened_args(self.recording_runtime_arg_names),
            self._jit_cache,
        )

    def invalidate_runtime(self, preserve_executables=False):
        if preserve_executables:
            self._jit_cache.retire_snode_tree_runtime_state()
        else:
            self._jit_cache.clear_runtime_state()

    @property
    def debug_graph_stats(self):
        return self._jit_cache._debug_graph_stats()

    @property
    def snapshot_graph_stats(self):
        return self._jit_cache._debug_graph_stats(False)

    @property
    def debug_info(self):
        info = {"kind": self.region_kind, "dispatch_count": self.dispatch_count}
        if self.physical_dispatch_count != self.dispatch_count:
            info["physical_dispatch_count"] = self.physical_dispatch_count
        if self.composer_applied_groups:
            info["composed_two_map_groups"] = self.composer_applied_groups
        if self.source_native_count:
            info["lowered_native_count"] = self.source_native_count
        if self.parallel_dispatch_groups:
            info["cuda_parallel_dispatch_groups"] = self.parallel_dispatch_groups
        return info


_DEFAULT_NATIVE_RECORDABLE_ACTION = object()


class _CompiledNativeGraphNode:
    snode_tree_dependencies = frozenset()
    snode_tree_dependency_info = frozenset()
    dispatch_count = 0
    source_native_count = 1
    region_kind = "native"

    def __init__(
        self,
        executable,
        recordable_action=_DEFAULT_NATIVE_RECORDABLE_ACTION,
    ):
        self.executable = executable
        self.recordable_action = (
            executable.recordable_action
            if recordable_action is _DEFAULT_NATIVE_RECORDABLE_ACTION
            else recordable_action
        )
        self.action_manifest = native_action_manifest(
            executable, self.recordable_action
        )
        self.native_action_manifests = (self.action_manifest,)
        self.ir_node = executable.graph_ir_node
        schema = self.action_manifest.runtime_bindings
        derived_schema = self.action_manifest.derived_runtime_bindings
        if any(not binding.required for binding in schema):
            raise TaichiRuntimeError(
                "Optional native Graph runtime arguments are not supported"
            )
        if any(not binding.required for binding in derived_schema):
            raise TaichiRuntimeError(
                "Optional native Graph derived arguments are not supported"
            )
        public_runtime_arg_names = frozenset(binding.name for binding in schema)
        self.derived_runtime_arg_names = frozenset(
            binding.name for binding in derived_schema
        )
        self.temporary_names = frozenset(
            requirement.name for requirement in self.action_manifest.temporaries
        )
        self.fixed_runtime_args = (
            {}
            if self.recordable_action is None
            else dict(self.recordable_action.fixed_bindings)
        )
        temporary_binding_map = (
            {}
            if self.recordable_action is None
            else dict(self.recordable_action.temporary_bindings)
        )
        if any(
            not isinstance(symbol, str)
            or not symbol
            or not isinstance(requirement, str)
            or not requirement
            for symbol, requirement in temporary_binding_map.items()
        ):
            raise TaichiRuntimeError(
                "Recordable action temporary bindings must map nonempty "
                "symbol names to nonempty requirement names"
            )
        if (
            self.recordable_action is not None
            and set(temporary_binding_map.values()) != self.temporary_names
        ):
            raise TaichiRuntimeError(
                "Recordable action temporary bindings must cover exactly its "
                "Graph temporary requirements"
            )
        self.temporary_binding_map = temporary_binding_map
        self.temporary_runtime_arg_names = frozenset(temporary_binding_map)
        self.temporary_actions = (
            () if not temporary_binding_map else (self.recordable_action,)
        )
        overlap = public_runtime_arg_names & self.fixed_runtime_args.keys()
        if overlap:
            raise TaichiRuntimeError(
                "Recordable action fixed bindings overlap public runtime "
                "arguments: " + ", ".join(sorted(overlap))
            )
        private_overlap = self.temporary_runtime_arg_names & (
            public_runtime_arg_names
            | self.derived_runtime_arg_names
            | self.fixed_runtime_args.keys()
        )
        if private_overlap:
            raise TaichiRuntimeError(
                "Recordable action temporary symbols overlap public or fixed "
                "arguments: " + ", ".join(sorted(private_overlap))
            )
        derived_overlap = self.derived_runtime_arg_names & (
            public_runtime_arg_names | self.fixed_runtime_args.keys()
        )
        if derived_overlap:
            raise TaichiRuntimeError(
                "Recordable action derived bindings overlap public or fixed "
                "arguments: " + ", ".join(sorted(derived_overlap))
            )
        self.recording_runtime_arg_names = frozenset(
            (
                *public_runtime_arg_names,
                *self.derived_runtime_arg_names,
                *self.fixed_runtime_args.keys(),
                *self.temporary_runtime_arg_names,
            )
        )
        self.runtime_arg_names = public_runtime_arg_names
        self.needs_runtime_args = bool(self.recording_runtime_arg_names)
        self.lifetime_leases = (
            executable,
            *tuple(executable.lifetime_leases),
        )
        if self.recordable_action is not None:
            recording = self.recordable_action.backend_command_recording
            if recording is None:
                recorder_names = frozenset().union(
                    *(
                        _runtime_arg_names(args)
                        for _, args in self.recordable_action.dispatches
                    )
                )
            else:
                recorder_names = frozenset(recording.binding_names)
            required_private_names = frozenset(
                (
                    *self.fixed_runtime_args,
                    *self.temporary_runtime_arg_names,
                )
            )
            complete = recorder_names == self.recording_runtime_arg_names
            valid_subset = (
                self.recordable_action.allows_unused_public_bindings
                and recorder_names <= self.recording_runtime_arg_names
                and required_private_names <= recorder_names
            )
            if not complete and not valid_subset:
                raise TaichiRuntimeError(
                    "Recordable action bindings must match its public, "
                    "derived, temporary, and fixed bindings"
                )

    def run(self, context, temporaries=None):
        if isinstance(self.recordable_action, BackendCommandGraphAction):
            all_args = context.runtime_args()
            names = self.recordable_action.backend_command_recording.binding_names
            bindings = {name: all_args[name] for name in names}
            return self.recordable_action.execute_graph_validated(bindings)
        runtime_args = None
        if self.needs_runtime_args:
            all_args = context.runtime_args()
            runtime_args = {name: all_args[name] for name in self.runtime_arg_names}
        if not self.temporary_names:
            if runtime_args is None:
                return self.executable.run()
            return self.executable.run(runtime_args)
        if temporaries is None or not self.temporary_names.issubset(temporaries):
            raise TaichiRuntimeError(
                "Native Graph temporary requirements were not materialized"
            )
        bindings = {name: temporaries[name] for name in self.temporary_names}
        return self.executable.run_with_graph_temporaries(bindings, runtime_args)

    @property
    def debug_info(self):
        info = dict(self.executable.debug_info)
        if self.recordable_action is not None:
            info["recordable_action"] = self.recordable_action.capabilities.to_dict()
            info["fixed_binding_count"] = len(self.fixed_runtime_args)
        return info


_OBSERVATION_PACK_KERNELS = {}


def _observation_pack_kernel(dtype):
    key = str(dtype)
    kernel = _OBSERVATION_PACK_KERNELS.get(key)
    if kernel is not None:
        return kernel

    @kernel_impl.kernel
    def pack_scalar_snapshot(
        source: ndarray_type.ndarray(dtype=dtype, ndim=0),
        destination: ndarray_type.ndarray(dtype=dtype, ndim=1),
        index: i32,
    ):
        destination[index] = source[None]

    _OBSERVATION_PACK_KERNELS[key] = pack_scalar_snapshot
    return pack_scalar_snapshot


class _CompiledObservationGraphNode:
    needs_runtime_args = True
    snode_tree_dependencies = frozenset()
    snode_tree_dependency_info = frozenset()
    source_native_count = 0
    region_kind = "observation"

    def __init__(self, values, name):
        if not isinstance(name, str) or not name:
            raise TaichiRuntimeError(
                "Graph observation name must be a non-empty string"
            )
        values = tuple(values)
        if not values:
            raise TaichiRuntimeError("Graph observation requires at least one value")
        groups = {}
        entries = []
        for value in values:
            if getattr(value, "tag", None) != ArgKind.NDARRAY:
                raise TaichiRuntimeError(
                    "Graph observation values must be symbolic ndarray arguments"
                )
            descriptor = describe_element_type(value.dtype())
            if (
                value.field_dim != 0
                or value.element_shape
                or descriptor.category != "scalar"
            ):
                raise TaichiRuntimeError(
                    "Graph observation values must be scalar ndarrays with ndim=0"
                )
            dtype = value.dtype()
            key = str(dtype)
            group = groups.setdefault(key, {"dtype": dtype, "names": []})
            index = len(group["names"])
            group["names"].append(value.name)
            entries.append((value.name, key, index, dtype))
        names = tuple(entry[0] for entry in entries)
        if len(set(names)) != len(names):
            raise TaichiRuntimeError(
                "Graph observation values must have unique argument names"
            )
        self.name = name
        self._groups = tuple(
            (key, group["dtype"], tuple(group["names"]))
            for key, group in groups.items()
        )
        self._entries = tuple(entries)
        self._kernels = {
            key: _observation_pack_kernel(dtype) for key, dtype, _ in self._groups
        }
        self._active_buffers = None
        self.runtime_arg_names = frozenset(names)
        self.dispatch_count = len(entries)
        self.physical_dispatch_count = self.dispatch_count
        self.ir_node = ObservationNode(
            name=name,
            effects=tuple(
                ResourceEffect(arg_name, GraphAccess.READ) for arg_name in names
            ),
            bindings=tuple(RuntimeBinding(arg_name, "ndarray") for arg_name in names),
            batch=name,
            synchronization=False,
            opaque=False,
        )

    def allocate_snapshot_buffers(self, *, completion_attached=False):
        buffers = {}
        byte_count = 0
        for key, dtype, names in self._groups:
            if completion_attached:
                buffers[key] = ScalarNdarray._graph_observation_storage(
                    dtype, (len(names),)
                )
            else:
                buffers[key] = ScalarNdarray(dtype, (len(names),))
            byte_count += np.dtype(to_numpy_type(dtype)).itemsize * len(names)
        return buffers, byte_count

    def bind_snapshot_buffers(self, buffers):
        self._active_buffers = buffers

    def clear_snapshot_buffers(self):
        self._active_buffers = None

    def run(self, context, temporaries=None):
        if self._active_buffers is None:
            raise TaichiRuntimeError("Graph observation snapshot slot was not bound")
        runtime_args = context.runtime_args()
        for arg_name, key, index, dtype in self._entries:
            value = runtime_args[arg_name]
            if (
                not isinstance(value, Ndarray)
                or value.shape != ()
                or str(value.dtype) != str(dtype)
            ):
                raise TaichiRuntimeError(
                    f"Graph observation {arg_name} requires a scalar ndarray "
                    f"with dtype {dtype}"
                )
            self._kernels[key](value, self._active_buffers[key], index)

    def decode_snapshot(self, hosts):
        result = {}
        for key, _, names in self._groups:
            values = hosts[key].reshape(-1)
            result.update(
                (name, values[index].item()) for index, name in enumerate(names)
            )
        return result

    @property
    def debug_info(self):
        return {
            "kind": "observation",
            "name": self.name,
            "value_count": len(self._entries),
            "packed_group_count": len(self._groups),
            "asynchronous": True,
        }


def _control_scalar_values(values, names, *, use_transfer_planner):
    # The final prepared-frame preflight proves every source is one canonical
    # scalar i32 ndarray before any condition dispatch. Portable observation
    # therefore only performs the dynamic device read; repeating owner, dtype,
    # and shape validation in every loop/branch observation is redundant.
    sources = [value.arr for value in values]
    hosts = [np.empty(shape=(), dtype=np.int32) for _ in values]
    program = impl.get_runtime().prog
    if use_transfer_planner:
        program.copy_graph_observations_to_host(sources, hosts)
    else:
        program.copy_ndarrays_to_host(sources, hosts)
    return (
        tuple(int(host.reshape(-1)[0]) for host in hosts),
        sum(host.nbytes for host in hosts),
    )


def _structured_chunk_limit(arch, requested, masked_execution):
    cpu_arches = (_ti_core.Arch.x64, _ti_core.Arch.arm64)
    if arch in cpu_arches or not masked_execution:
        return 1
    configured = (
        requested
        if requested is not None
        else os.environ.get("TI_GRAPH_WHILE_CHUNK_SIZE", "4")
    )
    if isinstance(configured, bool) or not isinstance(
        configured, (int, np.integer, str)
    ):
        raise TaichiRuntimeError("Graph while chunk_size must be an integer")
    try:
        configured = int(configured)
    except ValueError as error:
        raise TaichiRuntimeError("Graph while chunk_size must be an integer") from error
    if configured <= 0:
        raise TaichiRuntimeError("Graph while chunk_size must be positive")
    return min(configured, 64)


def _vulkan_structured_strategy():
    strategy = (
        os.environ.get("TI_GRAPH_VULKAN_STRUCTURED_STRATEGY", "auto").strip().lower()
    )
    strategies = {
        "auto": 0,
        "compact": 1,
        "chained": 2,
        "conditional": 3,
        "coarse_conditional": 4,
    }
    if strategy not in strategies:
        raise TaichiRuntimeError(
            "TI_GRAPH_VULKAN_STRUCTURED_STRATEGY must be auto, compact, "
            "chained, conditional, or coarse_conditional"
        )
    return strategies[strategy]


def _vulkan_first_chunk_strategy(value):
    if not isinstance(value, str):
        raise TaichiRuntimeError(
            "Graph while vulkan_first_chunk_strategy must be a string"
        )
    normalized = value.strip().lower()
    strategies = {
        "auto": 0,
        "compact": 1,
        "coarse_conditional": 4,
    }
    if normalized not in strategies:
        raise TaichiRuntimeError(
            "Graph while vulkan_first_chunk_strategy must be auto, compact, "
            "or coarse_conditional"
        )
    return normalized, strategies[normalized]


def _vulkan_compound_strategy_codes(chunk_count, first_chunk_strategy):
    requested = _vulkan_structured_strategy()
    first_name, first_requested = _vulkan_first_chunk_strategy(first_chunk_strategy)
    if requested != 0 and first_name == "auto":
        return (requested,) * chunk_count
    conditional_available = bool(
        impl.get_runtime().prog._vulkan_conditional_rendering_available()
    )
    if first_requested == 4 and not conditional_available:
        raise TaichiRuntimeError(
            "Graph while requested a coarse-conditional first Vulkan chunk, "
            "but VK_EXT_conditional_rendering is unavailable"
        )
    first = first_requested or 1
    if chunk_count <= 1:
        return (first,)
    if requested != 0:
        return (first, *((requested,) * (chunk_count - 1)))
    if not conditional_available:
        return (first, *((1,) * (chunk_count - 1)))
    # The first chunk is likely active and uses the lowest fixed-cost compact
    # path. Later chunks add one coarse predicate gate so a converged region
    # skips its complete encoded tail without invoking per-iteration
    # controllers or payload kernels.
    return (first,) + (4,) * (chunk_count - 1)


def _while_upgrade_status(arch, mode, requested_recipe=None):
    if mode not in ("auto", "portable", "native_required"):
        raise TaichiRuntimeError(
            "Graph while lowering_mode must be auto, portable, or " "native_required"
        )
    if mode == "portable":
        return False, "forced_portable"
    if arch == _ti_core.Arch.vulkan:
        return True, "eligible"
    if arch != _ti_core.Arch.cuda:
        if mode == "native_required":
            raise TaichiRuntimeError(
                "Graph while native_required mode needs CUDA or Vulkan"
            )
        return False, "not_gpu_structured_runtime"
    capabilities = dict(_ti_core.cuda_conditional_graph_capabilities())
    lowering = _cuda_structured_control_lowering(
        capabilities,
        requested_recipe=requested_recipe,
    )
    if lowering is not None:
        return True, (
            "eligible"
            if lowering == "cuda_conditional_graph"
            else "eligible_masked_bounded"
        )
    if not capabilities.get("ordinary_graph_symbols_loaded", False):
        reason = "cuda_graph_capture_symbols_not_loaded"
    else:
        reason = "cuda_device_control_lowering_unavailable"
    if mode == "native_required":
        raise TaichiRuntimeError(
            f"Graph while native CUDA lowering unavailable: {reason}"
        )
    return False, reason


def _cuda_branch_upgrade_status(arch, mode, kind, requested_recipe=None):
    if mode not in ("auto", "portable", "native_required"):
        raise TaichiRuntimeError(
            f"Graph {kind} lowering_mode must be auto, portable, or " "native_required"
        )
    if mode == "portable":
        return False, "forced_portable"
    if arch != _ti_core.Arch.cuda:
        if mode == "native_required":
            raise TaichiRuntimeError(f"Graph {kind} native_required mode needs CUDA")
        return False, "not_cuda"
    capabilities = dict(_ti_core.cuda_conditional_graph_capabilities())
    lowering = _cuda_structured_control_lowering(
        capabilities,
        requested_recipe=requested_recipe,
    )
    if lowering is not None:
        return True, (
            "eligible"
            if lowering == "cuda_conditional_graph"
            else "eligible_masked_bounded"
        )
    if not capabilities.get("ordinary_graph_symbols_loaded", False):
        reason = "cuda_graph_capture_symbols_not_loaded"
    else:
        reason = "cuda_device_control_lowering_unavailable"
    if mode == "native_required":
        raise TaichiRuntimeError(
            f"Graph {kind} native CUDA lowering unavailable: {reason}"
        )
    return False, reason


def _compile_plain_sequential_runtime_node(
    sequences,
    *,
    repetitions=1,
    name,
    region_kind="sequential",
    region_kinds=None,
):
    sequences = tuple(sequences)
    if not sequences or any(
        not isinstance(sequence, Sequential) for sequence in sequences
    ):
        raise TaichiRuntimeError("Structured Graph regions require Sequential values")
    if region_kinds is None:
        region_kinds = (region_kind,) * len(sequences)
    else:
        region_kinds = tuple(region_kinds)
        if len(region_kinds) != len(sequences):
            raise TaichiRuntimeError(
                "Structured Graph region kinds must match its Sequential values"
            )
    builder = _new_runtime_graph_builder(())
    ir_nodes = []
    dispatch_count = 0
    runtime_arg_names = set()
    recording_dispatches = []
    fixed_runtime_args = {}
    lifetime_leases = []
    source_native_count = 0
    native_action_manifests = []
    temporary_actions = []
    for _ in range(repetitions):
        for sequence, sequence_region_kind in zip(sequences, region_kinds):
            recording_dispatches.extend(
                sequence._dispatch_to(builder, region_kind=sequence_region_kind)
            )
            ir_nodes.extend(sequence._ir_nodes)
            dispatch_count += sequence._dispatch_count
            runtime_arg_names.update(sequence._recording_runtime_arg_names)
            for binding_name, value in sequence._fixed_runtime_args.items():
                existing = fixed_runtime_args.get(binding_name)
                if existing is not None and existing is not value:
                    if not (
                        isinstance(existing, (int, float))
                        and isinstance(value, (int, float))
                        and existing == value
                    ):
                        raise TaichiRuntimeError(
                            "Structured Graph regions provide conflicting "
                            f"fixed binding {binding_name!r}"
                        )
                fixed_runtime_args[binding_name] = value
            lifetime_leases.extend(sequence._lifetime_leases)
            source_native_count += sequence._source_native_count
            native_action_manifests.extend(sequence._native_action_manifests)
            temporary_actions.extend(sequence._temporary_actions)
    return _CompiledCGraphNode(
        builder.compile(),
        dispatch_count,
        runtime_arg_names,
        SequentialRegion(tuple(ir_nodes), name=name),
        recording_dispatches=recording_dispatches,
        lifetime_leases=lifetime_leases,
        source_native_count=source_native_count,
        region_kind=region_kind,
        fixed_runtime_args=fixed_runtime_args,
        temporary_actions=_deduplicate_temporary_actions(temporary_actions),
        native_action_manifests=native_action_manifests,
    )


def _is_structured_control_node(node):
    return isinstance(
        node,
        (
            _CompiledWhileGraphNode,
            _CompiledIfGraphNode,
            _CompiledSwitchGraphNode,
        ),
    )


def _sequence_structured_nodes(sequence):
    return tuple(item[1] for item in sequence._items if item[0] == "structured")


class _CompiledSequentialRegionNode:
    """Ordered host-level sequence used when a region contains control nodes.

    Plain action runs remain coalesced into CGraph segments. Structured
    children stay explicit, so an empty action prefix/suffix never creates a
    meaningless CGraph.
    """

    needs_runtime_args = True
    region_kind = "nested_sequential"

    def __init__(
        self,
        nodes,
        *,
        name,
        ir_node,
        definition_sequences=(),
    ):
        self.nodes = tuple(nodes)
        if not self.nodes:
            raise TaichiRuntimeError(
                "Structured Graph sequence must contain at least one action"
            )
        self.name = name
        self.ir_node = ir_node
        self.definition_sequence_snapshots = tuple(
            _GraphSequentialRecipeSnapshot(sequence)
            for sequence in definition_sequences
        )

        self.definition_children = tuple(
            _sequence_structured_nodes(sequence) for sequence in definition_sequences
        )
        self.dispatch_count = sum(
            getattr(node, "dispatch_count", 0) for node in self.nodes
        )
        self.physical_dispatch_count = sum(
            getattr(
                node,
                "physical_dispatch_count",
                getattr(node, "dispatch_count", 0),
            )
            for node in self.nodes
        )
        self.source_native_count = sum(
            getattr(node, "source_native_count", 0) for node in self.nodes
        )
        self.native_action_manifests = tuple(
            manifest
            for node in self.nodes
            for manifest in _native_action_manifests_for_node(node)
        )
        if self.source_native_count != len(self.native_action_manifests):
            raise TaichiRuntimeError(
                "Structured sequence native count must match its action manifests"
            )
        self.composer_applied_groups = sum(
            getattr(node, "composer_applied_groups", 0) for node in self.nodes
        )
        self.composer_lowering_available = any(
            getattr(node, "composer_lowering_available", False) for node in self.nodes
        )
        self.temporary_actions = _merge_temporary_actions(self.nodes)
        self.temporary_runtime_arg_names = frozenset().union(
            *(
                getattr(node, "temporary_runtime_arg_names", frozenset())
                for node in self.nodes
            )
        )
        fixed_runtime_args = dict(_merge_fixed_runtime_args(self.nodes))
        for sequence in definition_sequences:
            for binding_name, value in sequence._fixed_runtime_args.items():
                existing = fixed_runtime_args.get(binding_name)
                if existing is not None and existing is not value:
                    if not (
                        isinstance(existing, (int, float))
                        and isinstance(value, (int, float))
                        and existing == value
                    ):
                        raise TaichiRuntimeError(
                            "Structured sequence provides conflicting fixed "
                            f"binding {binding_name!r}"
                        )
                fixed_runtime_args[binding_name] = value
        self.fixed_runtime_args = fixed_runtime_args
        self.recording_runtime_arg_names = frozenset().union(
            *(
                getattr(
                    node,
                    "recording_runtime_arg_names",
                    node.runtime_arg_names,
                )
                for node in self.nodes
            )
        )
        self.derived_runtime_arg_names = _merge_derived_runtime_arg_names(self.nodes)
        self.runtime_arg_names = self.recording_runtime_arg_names.difference(
            (
                *self.fixed_runtime_args,
                *self.temporary_runtime_arg_names,
                *self.derived_runtime_arg_names,
            )
        )
        lifetime_leases = []
        seen_lifetime_leases = set()
        for owner in (*self.nodes, *definition_sequences):
            leases = getattr(
                owner,
                "lifetime_leases",
                getattr(owner, "_lifetime_leases", ()),
            )
            for lease in leases:
                identity = id(lease)
                if identity not in seen_lifetime_leases:
                    seen_lifetime_leases.add(identity)
                    lifetime_leases.append(lease)
        self.lifetime_leases = tuple(lifetime_leases)
        self.snode_tree_dependencies = frozenset().union(
            *(node.snode_tree_dependencies for node in self.nodes)
        )
        self.snode_tree_dependency_info = frozenset().union(
            *(node.snode_tree_dependency_info for node in self.nodes)
        )

    @property
    def control_nodes(self):
        result = []
        seen = set()
        for node in self.nodes:
            candidates = (
                (node,)
                if _is_structured_control_node(node)
                else getattr(node, "control_nodes", ())
            )
            for candidate in candidates:
                identity = id(candidate)
                if identity not in seen:
                    seen.add(identity)
                    result.append(candidate)
        return tuple(result)

    def run(self, context, temporaries=None):
        for node in self.nodes:
            node.run(context, temporaries)

    @property
    def supports_native_submission(self):
        return all(node.supports_native_submission for node in self.control_nodes)

    def run_for_submission(self, context, temporaries=None):
        if not self.supports_native_submission:
            raise TaichiRuntimeError(
                "Structured sequence submission requires every control "
                "region to provide submission-capable native lowering"
            )
        for node in self.nodes:
            if _is_structured_control_node(node):
                node.run_for_submission(context, temporaries)
            elif isinstance(node, _CompiledSequentialRegionNode):
                node.run_for_submission(context, temporaries)
            else:
                node.run(context, temporaries)

    def materialize_pending_report(self):
        for node in self.control_nodes:
            materialize = getattr(node, "materialize_pending_report", None)
            if materialize is not None:
                materialize()

    def invalidate_runtime(self, preserve_executables=False):
        seen = set()
        for node in self.nodes:
            identity = id(node)
            if identity in seen:
                continue
            seen.add(identity)
            invalidate = getattr(node, "invalidate_runtime", None)
            if invalidate is not None:
                invalidate(preserve_executables=preserve_executables)

    @property
    def debug_graph_stats(self):
        return tuple(node.debug_graph_stats for node in self.nodes)

    @property
    def snapshot_graph_stats(self):
        return tuple(node.snapshot_graph_stats for node in self.nodes)

    @property
    def debug_info(self):
        return {
            "kind": self.region_kind,
            "name": self.name,
            "dispatch_count": self.dispatch_count,
            "segment_count": len(self.nodes),
            "structured_control_count": len(self.control_nodes),
        }


def _compile_sequential_runtime_node(
    sequences,
    *,
    repetitions=1,
    name,
    region_kind="sequential",
    region_kinds=None,
):
    sequences = tuple(sequences)
    if not sequences or any(
        not isinstance(sequence, Sequential) for sequence in sequences
    ):
        raise TaichiRuntimeError("Structured Graph regions require Sequential values")
    if region_kinds is None:
        region_kinds = (region_kind,) * len(sequences)
    else:
        region_kinds = tuple(region_kinds)
        if len(region_kinds) != len(sequences):
            raise TaichiRuntimeError(
                "Structured Graph region kinds must match its Sequential values"
            )
    if not any(sequence._structured_depth for sequence in sequences):
        return _compile_plain_sequential_runtime_node(
            sequences,
            repetitions=repetitions,
            name=name,
            region_kind=region_kind,
            region_kinds=region_kinds,
        )

    nodes = []
    ir_nodes = []
    segment_index = 0

    def flush_plain(sequence, item_pairs, sequence_region_kind):
        nonlocal segment_index
        if not item_pairs:
            return
        view = sequence._plain_view(item_pairs)
        nodes.append(
            _compile_plain_sequential_runtime_node(
                (view,),
                name=f"{name}_actions_{segment_index}",
                region_kind=sequence_region_kind,
            )
        )
        segment_index += 1

    for _ in range(repetitions):
        for sequence, sequence_region_kind in zip(sequences, region_kinds):
            pending = []
            for item, ir_node in zip(sequence._items, sequence._ir_nodes):
                ir_nodes.append(ir_node)
                if item[0] != "structured":
                    pending.append((item, ir_node))
                    continue
                flush_plain(sequence, pending, sequence_region_kind)
                pending = []
                nodes.append(item[1])
            flush_plain(sequence, pending, sequence_region_kind)

    return _CompiledSequentialRegionNode(
        nodes,
        name=name,
        ir_node=SequentialRegion(tuple(ir_nodes), name=name),
        definition_sequences=sequences,
    )


def _control_transfer_uses_planner(arch):
    return (
        arch == _ti_core.Arch.vulkan
        and os.environ.get("TI_GRAPH_OBSERVATION_TRANSFER_PLANNER", "1") != "0"
    )


def _structured_host_lowering(arch):
    if arch in (_ti_core.Arch.x64, _ti_core.Arch.arm64):
        return "cpu_host_control"
    return "portable_host_control"


def _graph_control_name(value, role):
    name = value if isinstance(value, str) else getattr(value, "name", None)
    if not isinstance(name, str) or not name:
        raise TaichiRuntimeError(
            f"Graph {role} must be a symbolic argument or resource name"
        )
    return name


def _graph_control_names(values, role):
    names = tuple(_graph_control_name(value, role) for value in values)
    if len(names) != len(set(names)):
        raise TaichiRuntimeError(f"Graph {role} must not contain duplicate resources")
    return names


def _prepare_structured_definition(kind, name, lowering_mode, definition_regions):
    sequences = tuple(sequence for _, sequence in definition_regions if sequence)
    child_depth = max(
        (sequence._structured_depth for sequence in sequences),
        default=0,
    )
    structured_depth = 1 + child_depth
    if structured_depth > 2:
        raise TaichiRuntimeError(
            f"Graph {kind} {name!r} exceeds the maximum structured-control "
            "depth of 2"
        )
    nested = child_depth != 0
    if nested:
        for sequence in sequences:
            for node in _sequence_structured_nodes(sequence):
                node._mark_nested_portable()
    return structured_depth, nested


def _set_control_region_path(node, path, depth):
    if depth > 2:
        raise TaichiRuntimeError(
            "Graph structured-control definition exceeds the maximum depth of 2"
        )
    node.region_path = path
    node.control_depth = depth
    if depth > 1:
        node._mark_nested_portable()
    for role, children in node._definition_children:
        name_counts = {}
        for child in children:
            occurrence = name_counts.get(child.name, 0)
            name_counts[child.name] = occurrence + 1
            suffix = "" if occurrence == 0 else f"[{occurrence}]"
            _set_control_region_path(
                child,
                f"{path}/{role}/{child.name}{suffix}",
                depth + 1,
            )


def _compile_native_nested_while_runtime_node(
    condition,
    body,
    *,
    outer_name,
    outer_predicate,
    outer_counter,
    outer_status,
    outer_max_iterations,
):
    """Build the strict depth-2 while -> while backend replay program.

    The returned CGraph contains one copy of every static dispatch. Native
    CUDA/Vulkan lowering owns bounded command encoding; Python never expands
    the outer x inner Cartesian product.
    """

    def unavailable(reason):
        return None, None, None, reason

    arch = impl.current_cfg().arch
    if arch not in (_ti_core.Arch.cuda, _ti_core.Arch.vulkan):
        return unavailable("backend_is_not_cuda_or_vulkan")
    binding = (
        "jit_submit_bounded_cuda_nested_sequence_cached"
        if arch == _ti_core.Arch.cuda
        else "jit_run_bounded_vulkan_nested_sequence_cached"
    )
    if not hasattr(_ti_core.CompiledGraph, binding):
        return unavailable("nested_runtime_binding_unavailable")
    config = impl.current_cfg()
    if arch == _ti_core.Arch.vulkan and config.kernel_profiler:
        return unavailable("kernel_profiler_enabled")
    if arch == _ti_core.Arch.vulkan and config.vulkan_dispatch_cache:
        return unavailable("vulkan_dispatch_cache_enabled")
    if outer_counter is None:
        return unavailable("outer_counter_required")
    if outer_max_iterations <= 0 or outer_max_iterations > 64:
        return unavailable("outer_iteration_budget_out_of_range")
    if (
        arch == _ti_core.Arch.vulkan
        and not impl.get_runtime().prog._vulkan_conditional_rendering_available()
    ):
        return unavailable("vulkan_conditional_rendering_unavailable")

    condition_pairs = tuple(zip(condition._items, condition._ir_nodes))
    body_pairs = tuple(zip(body._items, body._ir_nodes))
    if any(item[0] not in ("dispatch", "native") for item, _ in condition_pairs):
        return unavailable("outer_condition_must_be_plain")
    structured_indices = tuple(
        index for index, (item, _) in enumerate(body_pairs) if item[0] == "structured"
    )
    if not structured_indices:
        return unavailable("inner_region_required")
    if len(structured_indices) > 8:
        return unavailable("inner_region_count_exceeds_eight")

    outer_controls = {outer_predicate, outer_counter}
    if outer_status is not None:
        outer_controls.add(outer_status)
    all_controls = set(outer_controls)
    inners = []
    inner_sequences = []
    for inner_slot, inner_index in enumerate(structured_indices):
        inner = body_pairs[inner_index][0][1]
        if not isinstance(inner, _CompiledWhileGraphNode):
            return unavailable("inner_region_must_be_while")
        if inner._has_nested_control:
            return unavailable("inner_region_must_be_leaf")
        if inner.counter is None:
            return unavailable("inner_counter_required")
        if inner.max_iterations <= 0 or inner.max_iterations > 64:
            return unavailable("inner_iteration_budget_out_of_range")
        if inner.compound_chunk_limit <= 0 or inner.compound_chunk_limit > 64:
            return unavailable("inner_chunk_size_out_of_range")
        if not inner._native_upgrade_eligible:
            return unavailable(
                f"inner_native_unavailable:{inner._native_upgrade_reason}"
            )
        controls = {inner.predicate, inner.counter}
        if inner.status is not None:
            controls.add(inner.status)
        if all_controls & controls:
            return unavailable("nested_control_resources_must_be_pairwise_independent")
        all_controls.update(controls)
        regions = dict(inner._definition_regions)
        inner_condition = regions["condition"]
        inner_body = regions["body"]
        for sequence, role in (
            (inner_condition, f"inner_{inner_slot}_condition"),
            (inner_body, f"inner_{inner_slot}_body"),
        ):
            if any(item[0] not in ("dispatch", "native") for item in sequence._items):
                return unavailable(f"{role}_must_be_plain")
        inners.append(inner)
        inner_sequences.append((inner_condition, inner_body))

    plain_segments = []
    cursor = 0
    for inner_index in structured_indices:
        pairs = body_pairs[cursor:inner_index]
        if any(item[0] not in ("dispatch", "native") for item, _ in pairs):
            return unavailable("outer_between_inner_actions_must_be_plain")
        plain_segments.append(body._plain_view(pairs))
        cursor = inner_index + 1
    trailing_pairs = body_pairs[cursor:]
    if any(item[0] not in ("dispatch", "native") for item, _ in trailing_pairs):
        return unavailable("outer_suffix_must_be_plain")
    plain_segments.append(body._plain_view(trailing_pairs))

    outer_condition_count = condition._dispatch_count
    dispatch_cursor = outer_condition_count
    descriptors = []
    flattened_sequences = [condition]
    region_kinds = ["while_condition"]
    repeated_dispatches = 0
    for segment, inner, (inner_condition, inner_body) in zip(
        plain_segments, inners, inner_sequences
    ):
        flattened_sequences.extend(
            (segment, inner_condition, inner_body, inner_condition)
        )
        region_kinds.extend(
            ("while_body", "while_condition", "while_body", "while_condition")
        )
        dispatch_cursor += segment._dispatch_count
        condition_begin = dispatch_cursor
        body_begin = condition_begin + inner_condition._dispatch_count
        end = body_begin + inner_body._dispatch_count + inner_condition._dispatch_count
        descriptors.append((condition_begin, body_begin, end))
        repeated_dispatches += inner.max_iterations * (end - body_begin)
        dispatch_cursor = end
    trailing = plain_segments[-1]
    flattened_sequences.extend((trailing, condition))
    region_kinds.extend(("while_body", "while_condition"))
    flattened_dispatch_count = (
        dispatch_cursor + trailing._dispatch_count + condition._dispatch_count
    )
    single_copy_repeated = sum(end - body for _, body, end in descriptors)
    outer_static_dispatches = (
        flattened_dispatch_count - outer_condition_count - single_copy_repeated
    )
    encoded_action_count = outer_condition_count + outer_max_iterations * (
        outer_static_dispatches + repeated_dispatches
    )
    if flattened_dispatch_count <= 0 or encoded_action_count > 4096:
        return unavailable("encoded_action_budget_exceeded")

    compiled = _compile_plain_sequential_runtime_node(
        tuple(flattened_sequences),
        name=f"{outer_name}_{_backend_name(_ti_core.arch_name(arch))}_nested",
        region_kind="while_nested",
        region_kinds=tuple(region_kinds),
    )
    boundaries = (outer_condition_count, tuple(descriptors))
    return compiled, tuple(inners), boundaries, "eligible"


class _CompiledWhileGraphNode:
    needs_runtime_args = True
    source_native_count = 0
    region_kind = "structured_while"

    def __init__(
        self,
        condition,
        body,
        *,
        predicate,
        control_inputs,
        carried_state,
        max_iterations,
        counter,
        status,
        chunk_size,
        vulkan_first_chunk_strategy,
        masked_execution,
        lowering_mode,
        name,
        _control_recipe=None,
    ):
        if not isinstance(condition, Sequential) or condition._dispatch_count == 0:
            raise TaichiRuntimeError(
                "Graph while condition must be a non-empty Sequential"
            )
        if not isinstance(body, Sequential) or body._dispatch_count == 0:
            raise TaichiRuntimeError("Graph while body must be a non-empty Sequential")
        if isinstance(max_iterations, bool) or not isinstance(
            max_iterations, (int, np.integer)
        ):
            raise TaichiRuntimeError("Graph while max_iterations must be an integer")
        if max_iterations < 0:
            raise TaichiRuntimeError("Graph while max_iterations must be non-negative")
        self.name = name
        self.predicate = predicate
        self.control_inputs = tuple(control_inputs)
        self.carried_state = tuple(carried_state)
        self.counter = counter
        self.status = status
        self.max_iterations = int(max_iterations)
        self.masked_execution = bool(masked_execution)
        self.lowering_mode = lowering_mode
        self.region_path = name
        self.control_depth = 1
        self._nested_subregion = False
        self._definition_regions = (
            ("condition", condition),
            ("body", body),
        )
        self._definition_children = tuple(
            (role, _sequence_structured_nodes(region))
            for role, region in self._definition_regions
        )
        self.structured_depth, self._has_nested_control = (
            _prepare_structured_definition(
                "while",
                name,
                lowering_mode,
                self._definition_regions,
            )
        )
        self._portable_exact_nested = self._has_nested_control
        (
            self.vulkan_first_chunk_strategy,
            self._vulkan_first_chunk_strategy_code,
        ) = _vulkan_first_chunk_strategy(vulkan_first_chunk_strategy)
        self.condition_dispatch_count = condition._dispatch_count
        self.body_dispatch_count = body._dispatch_count
        self.dispatch_count = self.condition_dispatch_count + self.body_dispatch_count
        required_condition = {predicate, *self.control_inputs}
        if status is not None:
            required_condition.add(status)
        missing_condition = sorted(
            required_condition.difference(condition._runtime_arg_names)
        )
        if missing_condition:
            raise TaichiRuntimeError(
                "Graph while condition does not declare control resources: "
                + ", ".join(missing_condition)
            )
        required_body = set(self.carried_state)
        if counter is not None:
            required_body.add(counter)
        missing_body = sorted(required_body.difference(body._runtime_arg_names))
        if missing_body:
            raise TaichiRuntimeError(
                "Graph while body does not declare carried resources: "
                + ", ".join(missing_body)
            )

        arch = impl.current_cfg().arch
        self._native_submission_eligible = not self._has_nested_control and (
            arch == _ti_core.Arch.cuda
            or (
                arch == _ti_core.Arch.vulkan
                and not impl.current_cfg().kernel_profiler
                and not impl.current_cfg().vulkan_dispatch_cache
                and self.max_iterations <= 512
            )
        )
        requested_compound_chunk_limit = _structured_chunk_limit(arch, chunk_size, True)
        self.chunk_limit = (
            1
            if self._has_nested_control
            else min(
                self.max_iterations or 1,
                _structured_chunk_limit(arch, chunk_size, self.masked_execution),
            )
        )
        self.compound_chunk_limit = (
            1
            if self._has_nested_control
            else min(
                self.max_iterations or 1,
                64 if chunk_size is None else requested_compound_chunk_limit,
            )
        )
        self.compound_chunk_count = (
            0
            if self.max_iterations == 0
            else (self.max_iterations + self.compound_chunk_limit - 1)
            // self.compound_chunk_limit
        )
        if (
            arch == _ti_core.Arch.vulkan
            and lowering_mode == "native_required"
            and self._native_submission_eligible
            and self.compound_chunk_count > 8
        ):
            raise TaichiRuntimeError(
                "Native Vulkan Graph while compound submission requires at "
                "most eight chunks; increase chunk_size or reduce "
                "max_iterations"
            )
        if (
            arch == _ti_core.Arch.vulkan
            and self._vulkan_first_chunk_strategy_code == 4
            and not impl.get_runtime().prog._vulkan_conditional_rendering_available()
        ):
            raise TaichiRuntimeError(
                "Graph while requested a coarse-conditional first Vulkan "
                "chunk, but VK_EXT_conditional_rendering is unavailable"
            )
        if self.chunk_limit > 1 and self.counter is None:
            raise TaichiRuntimeError(
                "Chunked Graph while regions require a device counter"
            )
        self._condition = _compile_sequential_runtime_node(
            (condition,),
            name=f"{name}_condition",
            region_kind="while_condition",
        )
        self._chunks = {}
        chunk = 1
        while chunk <= self.chunk_limit and chunk <= self.max_iterations:
            self._chunks[chunk] = _compile_sequential_runtime_node(
                (body, condition),
                repetitions=chunk,
                name=f"{name}_body_condition_{chunk}",
                region_kinds=("while_body", "while_condition"),
            )
            chunk *= 2
        self._vulkan_structured = (
            _compile_sequential_runtime_node(
                (condition, body, condition),
                name=f"{name}_vulkan_structured",
                region_kinds=(
                    "while_condition",
                    "while_body",
                    "while_condition",
                ),
            )
            if arch == _ti_core.Arch.vulkan and not self._has_nested_control
            else None
        )
        self._vulkan_nested = None
        self._vulkan_nested_inner = None
        self._vulkan_nested_boundaries = None
        self._vulkan_nested_reason = "not_nested"
        self._cuda_nested = None
        self._cuda_nested_inner = None
        self._cuda_nested_boundaries = None
        self._cuda_nested_reason = "not_nested"
        self._cuda_nested_control_lowering = None
        if self._has_nested_control and lowering_mode != "portable":
            nested = _compile_native_nested_while_runtime_node(
                condition,
                body,
                outer_name=name,
                outer_predicate=predicate,
                outer_counter=counter,
                outer_status=status,
                outer_max_iterations=self.max_iterations,
            )
            if arch == _ti_core.Arch.vulkan:
                (
                    self._vulkan_nested,
                    self._vulkan_nested_inner,
                    self._vulkan_nested_boundaries,
                    self._vulkan_nested_reason,
                ) = nested
            elif arch == _ti_core.Arch.cuda:
                (
                    self._cuda_nested,
                    self._cuda_nested_inner,
                    self._cuda_nested_boundaries,
                    self._cuda_nested_reason,
                ) = nested
                if self._cuda_nested is not None:
                    self._cuda_nested_control_lowering = (
                        _cuda_nested_structured_control_lowering(_control_recipe)
                    )
        elif self._has_nested_control:
            self._vulkan_nested_reason = "outer_portable_lowering_requested"
            self._cuda_nested_reason = "outer_portable_lowering_requested"
        dependency_nodes = (
            self._condition,
            *self._chunks.values(),
            *(
                (self._vulkan_structured,)
                if self._vulkan_structured is not None
                else ()
            ),
            *((self._vulkan_nested,) if self._vulkan_nested is not None else ()),
            *((self._cuda_nested,) if self._cuda_nested is not None else ()),
        )
        self.temporary_actions = _merge_temporary_actions(dependency_nodes)
        self.temporary_runtime_arg_names = frozenset().union(
            *(node.temporary_runtime_arg_names for node in dependency_nodes)
        )
        self.fixed_runtime_args = _merge_fixed_runtime_args(dependency_nodes)
        self.recording_runtime_arg_names = frozenset().union(
            *(node.recording_runtime_arg_names for node in dependency_nodes)
        )
        self.derived_runtime_arg_names = _merge_derived_runtime_arg_names(
            dependency_nodes
        )
        self.runtime_arg_names = self.recording_runtime_arg_names.difference(
            (
                *self.fixed_runtime_args,
                *self.temporary_runtime_arg_names,
                *self.derived_runtime_arg_names,
            )
        )
        active_sequences = (
            (condition,) if self.max_iterations == 0 else (condition, body)
        )
        self.lifetime_leases = tuple(
            lease
            for sequence in active_sequences
            for lease in sequence._lifetime_leases
        )
        self.source_native_count = sum(
            sequence._source_native_count for sequence in active_sequences
        )
        self.snode_tree_dependencies = frozenset().union(
            *(node.snode_tree_dependencies for node in dependency_nodes)
        )
        self.snode_tree_dependency_info = frozenset().union(
            *(node.snode_tree_dependency_info for node in dependency_nodes)
        )
        self.ir_node = WhileRegion(
            predicate=predicate,
            max_iterations=self.max_iterations,
            condition=SequentialRegion(
                tuple(condition._ir_nodes), name=f"{name}_condition"
            ),
            body=SequentialRegion(tuple(body._ir_nodes), name=f"{name}_body"),
            control_inputs=self.control_inputs,
            carried_state=self.carried_state,
            counter=counter,
            status=status,
            chunk_size=self.chunk_limit,
            compound_chunk_size=self.compound_chunk_limit,
            vulkan_first_chunk_strategy=self.vulkan_first_chunk_strategy,
            masked_execution=self.masked_execution,
            lowering_mode=lowering_mode,
            name=name,
        )
        self._last_report = None
        self._native_jit_cache = _ti_core.CompiledGraphJITCache()
        self._vulkan_chunk_limits = {}
        self._native_upgrade_eligible, self._native_upgrade_reason = (
            _while_upgrade_status(
                arch,
                lowering_mode,
                requested_recipe=_control_recipe,
            )
        )
        self._cuda_control_lowering = (
            _cuda_structured_control_lowering(requested_recipe=_control_recipe)
            if arch == _ti_core.Arch.cuda
            else None
        )
        if self._has_nested_control:
            nested_compiled = (
                self._vulkan_nested is not None or self._cuda_nested is not None
            )
            self._native_upgrade_eligible = nested_compiled
            self._native_upgrade_reason = (
                "eligible_nested_bounded"
                if nested_compiled
                else (
                    self._vulkan_nested_reason
                    if arch == _ti_core.Arch.vulkan
                    else self._cuda_nested_reason
                )
            )
            self._native_submission_eligible = nested_compiled
            if lowering_mode == "native_required" and not nested_compiled:
                raise TaichiRuntimeError(
                    f"Graph while {name!r} native nested lowering unavailable: "
                    f"{self._native_upgrade_reason}"
                )
        if self._native_upgrade_eligible and self.counter is None:
            if lowering_mode == "native_required":
                raise TaichiRuntimeError(
                    "Native Graph while lowering requires an exact " "iteration counter"
                )
            self._native_upgrade_eligible = False
            self._native_upgrade_reason = "exact_counter_required"

        # Everything below describes the materialized control topology. A
        # Graph is invalidated across runtime reinitialization, so none of
        # these facts can change during replay. Keep the replay path focused
        # on the predicate/counter/status resources that are genuinely
        # invocation-dependent.
        self._supports_native_submission = (
            self.lowering_mode in ("auto", "portable")
            if arch in (_ti_core.Arch.x64, _ti_core.Arch.arm64)
            else (
                self.lowering_mode in ("auto", "native_required")
                and self._native_upgrade_eligible
                and self.counter is not None
                and self._native_submission_eligible
            )
        )
        self._portable_max_chunk = max(self._chunks, default=1)
        nested_inners = self._vulkan_nested_inner or self._cuda_nested_inner or ()
        self._nested_inner_max_iterations = tuple(
            inner.max_iterations for inner in nested_inners
        )
        self._nested_inner_compound_chunk_limits = tuple(
            inner.compound_chunk_limit for inner in nested_inners
        )
        # Only the native flat Vulkan route consumes this tuple, and that
        # route already has a <=512-iteration eligibility bound. Keeping it
        # empty elsewhere avoids turning an intentionally large portable or
        # CUDA iteration budget into persistent host memory at Graph build.
        self._vulkan_chunk_iterations = (
            tuple(
                min(
                    self.compound_chunk_limit,
                    self.max_iterations - offset,
                )
                for offset in range(
                    0,
                    self.max_iterations,
                    self.compound_chunk_limit,
                )
            )
            if arch == _ti_core.Arch.vulkan
            and self._supports_native_submission
            and not self._has_nested_control
            else ()
        )

    def _select_chunk(self, remaining):
        if self._portable_exact_nested:
            return 1
        bounded = min(int(remaining), self._portable_max_chunk)
        return 1 << (bounded.bit_length() - 1)

    def _mark_nested_portable(self):
        # The enclosing control node owns native depth-2 lowering. Mark the
        # child so portable synchronous execution may still reuse its flat
        # backend replay when the parent itself is not submitted.
        self._nested_subregion = True

    @property
    def supports_native_submission(self):
        return self._supports_native_submission

    def run_for_submission(self, context, temporaries=None):
        if not self.supports_native_submission:
            raise TaichiRuntimeError(
                "Graph while submission requires a submission-capable "
                "native_required backend lowering"
            )
        if impl.current_cfg().arch in (_ti_core.Arch.x64, _ti_core.Arch.arm64):
            self.run(context, temporaries)
            return
        self._last_report = None
        if self._has_nested_control:
            runtime_args = context.runtime_args()
            nested = (
                self._vulkan_nested
                if impl.current_cfg().arch == _ti_core.Arch.vulkan
                else self._cuda_nested
            )
            inners = (
                self._vulkan_nested_inner
                if impl.current_cfg().arch == _ti_core.Arch.vulkan
                else self._cuda_nested_inner
            )
            boundaries = (
                self._vulkan_nested_boundaries
                if impl.current_cfg().arch == _ti_core.Arch.vulkan
                else self._cuda_nested_boundaries
            )
            if nested is None or inners is None or boundaries is None:
                raise TaichiRuntimeError(
                    "Native nested Graph while submission became unavailable"
                )

            def control_ndarray(name):
                if name is None:
                    return None
                return getattr(runtime_args[name], "arr", None)

            outer_predicate = control_ndarray(self.predicate)
            outer_counter = control_ndarray(self.counter)
            outer_status = control_ndarray(self.status)
            inner_predicates = tuple(
                control_ndarray(inner.predicate) for inner in inners
            )
            inner_counters = tuple(control_ndarray(inner.counter) for inner in inners)
            inner_statuses = tuple(control_ndarray(inner.status) for inner in inners)
            if any(
                value is None
                for value in (
                    outer_predicate,
                    outer_counter,
                    *inner_predicates,
                    *inner_counters,
                )
            ):
                raise TaichiRuntimeError(
                    "Native nested Graph while submission requires ndarray "
                    "predicate and counter resources"
                )
            if (self.status is not None and outer_status is None) or any(
                inner.status is not None and inner_status is None
                for inner, inner_status in zip(inners, inner_statuses)
            ):
                raise TaichiRuntimeError(
                    "Native nested Graph while submission requires ndarray "
                    "status resources"
                )
            outer_condition_count, inner_boundaries = boundaries
            flattened_args = context.flattened_args(nested.recording_runtime_arg_names)
            if impl.current_cfg().arch == _ti_core.Arch.vulkan:
                result = dict(
                    nested.compiled_graph.jit_run_bounded_vulkan_nested_sequence_cached(
                        context.compile_config(),
                        flattened_args,
                        self._native_jit_cache,
                        outer_predicate,
                        outer_counter,
                        inner_predicates,
                        inner_counters,
                        outer_condition_count,
                        inner_boundaries,
                        self.max_iterations,
                        self._nested_inner_max_iterations,
                        self._nested_inner_compound_chunk_limits,
                        outer_status,
                        inner_statuses,
                        False,
                    )
                )
                submitted = bool(result.get("submitted", False))
            else:
                submitted = bool(
                    nested.compiled_graph.jit_submit_bounded_cuda_nested_sequence_cached(
                        context.compile_config(),
                        flattened_args,
                        self._native_jit_cache,
                        outer_predicate,
                        outer_counter,
                        inner_predicates,
                        inner_counters,
                        outer_condition_count,
                        inner_boundaries,
                        self.max_iterations,
                        self._nested_inner_max_iterations,
                        outer_status,
                        inner_statuses,
                        self._cuda_nested_control_lowering
                        == _CUDA_NESTED_DEVICE_UPDATE_ROUTE,
                    )
                )
            if not submitted:
                raise TaichiRuntimeError(
                    "Native nested Graph while submission became unavailable; "
                    "synchronous fallback is disabled"
                )
            return
        if impl.current_cfg().arch == _ti_core.Arch.vulkan:
            if self.max_iterations == 0:
                self._condition.run(context)
                return
            runtime_args = context.runtime_args()
            predicate_object = runtime_args[self.predicate]
            counter_object = runtime_args[self.counter]
            status_object = (
                runtime_args[self.status] if self.status is not None else None
            )
            predicate_ndarray = getattr(predicate_object, "arr", None)
            counter_ndarray = getattr(counter_object, "arr", None)
            status_ndarray = (
                getattr(status_object, "arr", None)
                if status_object is not None
                else None
            )
            if predicate_ndarray is None or counter_ndarray is None:
                raise TaichiRuntimeError(
                    "Native Vulkan Graph while submission requires ndarray "
                    "predicate and counter resources"
                )
            if status_object is not None and status_ndarray is None:
                raise TaichiRuntimeError(
                    "Native Vulkan Graph while submission requires an "
                    "ndarray status resource"
                )

            chunk_count = self.compound_chunk_count
            if chunk_count > 8:
                raise TaichiRuntimeError(
                    "Native Vulkan Graph while submission exceeds the fixed "
                    "eight-slot compound chunk budget"
                )
            flattened_args = context.flattened_args(
                self._vulkan_structured.recording_runtime_arg_names
            )
            submitted = self._vulkan_structured.compiled_graph.jit_submit_bounded_vulkan_compound_cached(
                context.compile_config(),
                flattened_args,
                self._native_jit_cache,
                predicate_ndarray,
                counter_ndarray,
                self.condition_dispatch_count,
                self._vulkan_chunk_iterations,
                status_ndarray,
                _vulkan_compound_strategy_codes(
                    chunk_count, self.vulkan_first_chunk_strategy
                ),
            )
            if not submitted:
                raise TaichiRuntimeError(
                    "Native Vulkan Graph while compound submission "
                    "became unavailable; synchronous fallback is disabled"
                )
            return

        self._condition.run(context)
        if self.max_iterations == 0:
            return
        predicate_object = context.runtime_args()[self.predicate]
        predicate_ndarray = getattr(predicate_object, "arr", None)
        if predicate_ndarray is None:
            raise TaichiRuntimeError(
                "Native CUDA Graph while submission requires an ndarray predicate"
            )
        native_run = (
            self._chunks[1].compiled_graph.jit_run_bounded_cuda_masked_cached
            if self._cuda_control_lowering == "cuda_masked_bounded_graph"
            else self._chunks[1].compiled_graph.jit_run_bounded_cuda_cached
        )
        submitted = native_run(
            context.compile_config(),
            context.flattened_args(self.recording_runtime_arg_names),
            self._native_jit_cache,
            predicate_ndarray,
            self.max_iterations,
            True,
        )
        if not submitted:
            raise TaichiRuntimeError(
                "Native CUDA Graph while submission became unavailable; "
                "synchronous fallback is disabled"
            )

    def _run_vulkan_nested_structured(
        self,
        context,
        runtime_args,
        transfer_before,
    ):
        if (
            impl.current_cfg().arch != _ti_core.Arch.vulkan
            or self._vulkan_nested is None
            or self._vulkan_nested_inner is None
            or self._vulkan_nested_boundaries is None
            or len(self._vulkan_nested_inner) != 1
            or context.control_trace_enabled()
        ):
            return False
        native_run = getattr(
            self._vulkan_nested.compiled_graph,
            "jit_run_bounded_vulkan_nested_cached",
            None,
        )
        if native_run is None:
            return False

        inner = self._vulkan_nested_inner[0]

        def control_ndarray(name):
            if name is None:
                return None
            return getattr(runtime_args[name], "arr", None)

        outer_predicate = control_ndarray(self.predicate)
        outer_counter = control_ndarray(self.counter)
        outer_status = control_ndarray(self.status)
        inner_predicate = control_ndarray(inner.predicate)
        inner_counter = control_ndarray(inner.counter)
        inner_status = control_ndarray(inner.status)
        if any(
            value is None
            for value in (
                outer_predicate,
                outer_counter,
                inner_predicate,
                inner_counter,
            )
        ):
            return False
        if self.status is not None and outer_status is None:
            return False
        if inner.status is not None and inner_status is None:
            return False

        outer_condition_count, inner_boundaries = self._vulkan_nested_boundaries
        inner_condition_begin, inner_body_begin, outer_suffix_begin = inner_boundaries[
            0
        ]
        result = dict(
            native_run(
                context.compile_config(),
                context.flattened_args(self._vulkan_nested.recording_runtime_arg_names),
                self._native_jit_cache,
                outer_predicate,
                outer_counter,
                inner_predicate,
                inner_counter,
                outer_condition_count,
                inner_condition_begin,
                inner_body_begin,
                outer_suffix_begin,
                self.max_iterations,
                inner.max_iterations,
                inner.compound_chunk_limit,
                outer_status,
                inner_status,
            )
        )
        if not result.get("submitted", False):
            return False

        outer_logical = int(result["outer_logical_iterations"])
        outer_encoded = int(result["outer_encoded_iterations"])
        outer_initial_counter = int(result["outer_initial_counter"])
        outer_final_counter = int(result["outer_final_counter"])
        if (
            outer_logical < 0
            or outer_logical > outer_encoded
            or outer_encoded > self.max_iterations
            or outer_final_counter - outer_initial_counter != outer_logical
        ):
            raise TaichiRuntimeError(
                "Vulkan nested Graph while returned invalid outer iteration counts"
            )

        inner_logical = tuple(
            int(value) for value in result["inner_logical_iterations"]
        )
        inner_encoded = tuple(
            int(value)
            for value in result.get(
                "inner_encoded_iterations",
                (inner.max_iterations,) * len(inner_logical),
            )
        )
        inner_initial_counters = tuple(
            int(value) for value in result["inner_initial_counters"]
        )
        inner_final_counters = tuple(
            int(value) for value in result["inner_final_counters"]
        )
        inner_final_predicates = tuple(
            int(value) for value in result["inner_final_predicates"]
        )
        expected_inner_count = outer_logical
        inner_vectors = (
            inner_logical,
            inner_encoded,
            inner_initial_counters,
            inner_final_counters,
            inner_final_predicates,
        )
        if any(len(values) != expected_inner_count for values in inner_vectors):
            raise TaichiRuntimeError(
                "Vulkan nested Graph while returned incomplete inner telemetry"
            )
        if any(
            logical < 0
            or logical > encoded
            or encoded > inner.max_iterations
            or final_counter - initial_counter != logical
            for logical, encoded, initial_counter, final_counter in zip(
                inner_logical,
                inner_encoded,
                inner_initial_counters,
                inner_final_counters,
            )
        ):
            raise TaichiRuntimeError(
                "Vulkan nested Graph while returned invalid inner iteration counts"
            )

        inner_initial_statuses = tuple(
            int(value) for value in result.get("inner_initial_statuses", ())
        )
        inner_final_statuses = tuple(
            int(value) for value in result.get("inner_final_statuses", ())
        )
        if inner.status is not None and (
            len(inner_initial_statuses) != expected_inner_count
            or len(inner_final_statuses) != expected_inner_count
        ):
            raise TaichiRuntimeError(
                "Vulkan nested Graph while returned incomplete inner status telemetry"
            )

        observation_bytes = int(result["observation_bytes"])
        control_bytes = int(result["control_bytes"])
        indirect_dispatches = int(result["indirect_dispatches"])
        controller_dispatches = int(result["controller_dispatches"])
        controller_invocations = int(result["controller_invocations"])
        zero_dispatches = int(result["zero_dispatches"])
        if expected_inner_count:
            last = expected_inner_count - 1
            last_encoded = inner_encoded[last]
            last_initial_status = (
                inner_initial_statuses[last] if inner.status is not None else None
            )
            last_final_status = (
                inner_final_statuses[last] if inner.status is not None else None
            )
            inner._last_report = GraphWhileReport(
                name=inner.name,
                region_path=inner.region_path,
                structured_depth=inner.control_depth,
                backend="vulkan",
                lowering="vulkan_nested_compact_indirect",
                max_iterations=inner.max_iterations,
                logical_iterations=inner_logical[last],
                executed_iterations=last_encoded,
                overshoot_iterations=last_encoded - inner_logical[last],
                observation_boundaries=(0, last_encoded),
                predicate_values=(inner_final_predicates[last],),
                counter_values=(
                    inner_initial_counters[last],
                    inner_final_counters[last],
                ),
                status_resource=inner.status,
                status_values=(
                    (last_initial_status, last_final_status)
                    if inner.status is not None
                    else ()
                ),
                chunk_sizes=((last_encoded,) if last_encoded else ()),
                observation_batches=1,
                observation_scalar_count=(4 if inner.status is not None else 3),
                device_to_host_bytes=0,
                initial_counter=inner_initial_counters[last],
                final_counter=inner_final_counters[last],
                initial_status=last_initial_status,
                final_status=last_final_status,
                native_upgrade_eligible=True,
                native_upgrade_reason="selected_nested_tree",
                persistent_staging_bytes=0,
                staging_allocations=0,
                staging_reuses=0,
                packed_observation_batches=1,
                direct_observation_batches=0,
                staging_fallback_batches=0,
                packed_observation_bytes=0,
                condition_dispatch_count=inner.condition_dispatch_count,
                body_dispatch_count=inner.body_dispatch_count,
                control_inputs=inner.control_inputs,
                carried_state=inner.carried_state,
                logical_body_dispatch_count=(
                    inner_logical[last] * inner.body_dispatch_count
                ),
                control_arena_bytes=control_bytes,
                encoded_iterations=last_encoded,
                masked_iterations=last_encoded - inner_logical[last],
            )

        outer_initial_status = (
            int(result["outer_initial_status"]) if self.status is not None else None
        )
        outer_final_status = (
            int(result["outer_final_status"]) if self.status is not None else None
        )
        outer_initial_predicate = int(result["outer_initial_predicate"])
        outer_final_predicate = int(result["outer_final_predicate"])
        transfer_after = impl.get_runtime().prog._graph_observation_staging_stats()

        def transfer_delta(name):
            return int(transfer_after[name]) - int(transfer_before[name])

        self._last_report = GraphWhileReport(
            name=self.name,
            region_path=self.region_path,
            structured_depth=self.control_depth,
            backend="vulkan",
            lowering="vulkan_nested_conditional_compact_indirect",
            max_iterations=self.max_iterations,
            logical_iterations=outer_logical,
            executed_iterations=outer_encoded,
            overshoot_iterations=outer_encoded - outer_logical,
            observation_boundaries=(0, outer_encoded),
            predicate_values=(outer_initial_predicate, outer_final_predicate),
            counter_values=(outer_initial_counter, outer_final_counter),
            status_resource=self.status,
            status_values=(
                (outer_initial_status, outer_final_status)
                if self.status is not None
                else ()
            ),
            chunk_sizes=((outer_encoded,) if outer_encoded else ()),
            observation_batches=1,
            observation_scalar_count=(
                4 + len(inner_logical) * (5 if inner.status is not None else 3)
            ),
            device_to_host_bytes=observation_bytes,
            initial_counter=outer_initial_counter,
            final_counter=outer_final_counter,
            initial_status=outer_initial_status,
            final_status=outer_final_status,
            native_upgrade_eligible=True,
            native_upgrade_reason="selected_nested_tree",
            persistent_staging_bytes=0,
            staging_allocations=transfer_delta("allocations"),
            staging_reuses=transfer_delta("reuses"),
            packed_observation_batches=1,
            direct_observation_batches=0,
            staging_fallback_batches=0,
            packed_observation_bytes=observation_bytes,
            condition_dispatch_count=self.condition_dispatch_count,
            body_dispatch_count=self.body_dispatch_count,
            control_inputs=self.control_inputs,
            carried_state=self.carried_state,
            indirect_dispatch_count=indirect_dispatches,
            controller_dispatch_count=controller_dispatches,
            controller_invocation_count=controller_invocations,
            logical_body_dispatch_count=outer_logical * self.body_dispatch_count,
            zero_dispatch_count=zero_dispatches,
            control_arena_bytes=control_bytes,
            nested_region_path=inner.region_path,
            nested_logical_iterations=inner_logical,
            nested_encoded_iterations=inner_encoded,
            encoded_iterations=outer_encoded,
            masked_iterations=outer_encoded - outer_logical,
        )
        return True

    def _run_vulkan_structured(self, context, runtime_args, transfer_before):
        if (
            impl.current_cfg().arch != _ti_core.Arch.vulkan
            or not self._native_upgrade_eligible
            or self._vulkan_structured is None
        ):
            return False
        predicate_object = runtime_args[self.predicate]
        counter_object = runtime_args[self.counter]
        status_object = runtime_args[self.status] if self.status is not None else None
        predicate_ndarray = getattr(predicate_object, "arr", None)
        counter_ndarray = getattr(counter_object, "arr", None)
        status_ndarray = (
            getattr(status_object, "arr", None) if status_object is not None else None
        )
        unavailable = None
        if predicate_ndarray is None:
            unavailable = "predicate_ndarray_required"
        elif counter_ndarray is None:
            unavailable = "counter_ndarray_required"
        elif status_object is not None and status_ndarray is None:
            unavailable = "status_ndarray_required"
        if unavailable is not None:
            if self.lowering_mode == "native_required":
                raise TaichiRuntimeError(
                    "Graph while native Vulkan lowering failed: " f"{unavailable}"
                )
            return False

        flattened_args = context.flattened_args(
            self._vulkan_structured.recording_runtime_arg_names
        )
        strategy_code = _vulkan_structured_strategy()

        def run_chunk(iterations, *, execute_initial_dispatches):
            return dict(
                self._vulkan_structured.compiled_graph.jit_run_bounded_vulkan_cached(
                    context.compile_config(),
                    flattened_args,
                    self._native_jit_cache,
                    predicate_ndarray,
                    counter_ndarray,
                    self.condition_dispatch_count,
                    iterations,
                    status_ndarray,
                    execute_initial_dispatches,
                    strategy_code,
                )
            )

        requested_chunk_limit = min(
            self.max_iterations,
            (
                self.compound_chunk_limit
                if self._nested_subregion
                else self.max_iterations
            ),
        )
        chunk_limit = self._vulkan_chunk_limits.get(
            (strategy_code, requested_chunk_limit)
        )
        first_result = None
        if chunk_limit is None:
            first_result = run_chunk(
                requested_chunk_limit,
                execute_initial_dispatches=True,
            )
            if first_result["submitted"]:
                chunk_limit = requested_chunk_limit
            elif strategy_code in (0, 1) and requested_chunk_limit > 1:
                chunk_limit = min(requested_chunk_limit, 64)
                while chunk_limit >= 1:
                    first_result = run_chunk(
                        chunk_limit,
                        execute_initial_dispatches=True,
                    )
                    if first_result["submitted"]:
                        break
                    chunk_limit //= 2
            if not first_result["submitted"]:
                if self.lowering_mode == "native_required":
                    raise TaichiRuntimeError(
                        "Graph while native Vulkan structured lowering "
                        "became unavailable"
                    )
                return False
            self._vulkan_chunk_limits[(strategy_code, requested_chunk_limit)] = (
                chunk_limit
            )

        results = []
        remaining = self.max_iterations
        execute_initial_dispatches = True
        while True:
            iterations = min(remaining, chunk_limit)
            result = (
                first_result
                if first_result is not None and not results
                else run_chunk(
                    iterations,
                    execute_initial_dispatches=execute_initial_dispatches,
                )
            )
            first_result = None
            if not result["submitted"]:
                if self.lowering_mode == "native_required":
                    raise TaichiRuntimeError(
                        "Graph while native Vulkan structured lowering "
                        "became unavailable during chunk replay"
                    )
                return False
            results.append(result)
            encoded_chunk = int(result["encoded_iterations"])
            logical_chunk = int(result["logical_iterations"])
            if (
                encoded_chunk != iterations
                or logical_chunk < 0
                or logical_chunk > encoded_chunk
            ):
                raise TaichiRuntimeError(
                    "Vulkan Graph while runtime returned an invalid chunk result"
                )
            remaining -= encoded_chunk
            if remaining == 0 or int(result["predicate"]) == 0 or encoded_chunk == 0:
                break
            execute_initial_dispatches = False

        strategy_names = {
            1: "compact_indirect",
            2: "chained_indirect",
            3: "conditional",
            4: "coarse_conditional",
        }
        strategies = {int(result["strategy"]) for result in results}
        if len(strategies) != 1:
            raise TaichiRuntimeError(
                "Vulkan Graph while changed control strategy during chunk replay"
            )
        strategy = strategy_names.get(strategies.pop())
        if strategy is None:
            raise TaichiRuntimeError(
                "Vulkan Graph while runtime returned an invalid control strategy"
            )

        logical = sum(int(result["logical_iterations"]) for result in results)
        encoded = sum(int(result["encoded_iterations"]) for result in results)
        if logical < 0 or logical > encoded or encoded > self.max_iterations:
            raise TaichiRuntimeError(
                "Vulkan Graph while runtime returned an invalid logical "
                "iteration count"
            )
        final_predicate = int(results[-1]["predicate"])
        final_counter = int(results[-1]["counter"])
        final_status = int(results[-1]["status"]) if status_object is not None else None
        initial_status = (
            int(results[0]["initial_status"]) if status_object is not None else None
        )
        initial_counter = final_counter - logical
        observation_bytes = sum(int(result["observation_bytes"]) for result in results)
        chunk_sizes = tuple(
            int(result["encoded_iterations"])
            for result in results
            if int(result["encoded_iterations"]) != 0
        )
        boundaries = []
        boundary = 0
        for chunk in chunk_sizes:
            boundary += chunk
            boundaries.append(boundary)
        chunked = chunk_limit < self.max_iterations
        transfer_after = impl.get_runtime().prog._graph_observation_staging_stats()

        def transfer_delta(name):
            return int(transfer_after[name]) - int(transfer_before[name])

        self._last_report = GraphWhileReport(
            name=self.name,
            region_path=self.region_path,
            structured_depth=self.control_depth,
            backend="vulkan",
            lowering=(
                f"vulkan_chunked_{strategy}" if chunked else f"vulkan_{strategy}"
            ),
            max_iterations=self.max_iterations,
            logical_iterations=logical,
            executed_iterations=encoded,
            overshoot_iterations=encoded - logical,
            observation_boundaries=tuple(boundaries),
            predicate_values=tuple(int(result["predicate"]) for result in results),
            counter_values=tuple(int(result["counter"]) for result in results),
            status_resource=self.status,
            status_values=(
                (
                    initial_status,
                    *(int(result["status"]) for result in results),
                )
                if final_status is not None
                else ()
            ),
            chunk_sizes=chunk_sizes,
            observation_batches=len(results),
            observation_scalar_count=5 * len(results),
            device_to_host_bytes=observation_bytes,
            initial_counter=initial_counter,
            final_counter=final_counter,
            initial_status=initial_status,
            final_status=final_status,
            native_upgrade_eligible=True,
            native_upgrade_reason=("selected_chunked" if chunked else "selected"),
            # Terminal buffers belong to the backend replay slots and are
            # included in backend persistent bytes, not in the Program's
            # portable observation-staging arena.
            persistent_staging_bytes=0,
            staging_allocations=0,
            staging_reuses=0,
            packed_observation_batches=len(results),
            direct_observation_batches=0,
            staging_fallback_batches=0,
            packed_observation_bytes=observation_bytes,
            condition_dispatch_count=self.condition_dispatch_count,
            body_dispatch_count=self.body_dispatch_count,
            control_inputs=self.control_inputs,
            carried_state=self.carried_state,
            indirect_dispatch_count=sum(
                int(result["indirect_dispatches"]) for result in results
            ),
            controller_dispatch_count=sum(
                int(result["controller_dispatches"]) for result in results
            ),
            controller_invocation_count=sum(
                int(result["controller_invocations"]) for result in results
            ),
            logical_body_dispatch_count=logical * self.body_dispatch_count,
            zero_dispatch_count=sum(
                int(result["zero_dispatches"]) for result in results
            ),
            control_arena_bytes=max(int(result["control_bytes"]) for result in results),
            encoded_iterations=encoded,
            masked_iterations=encoded - logical,
        )
        return True

    def run(self, context, temporaries=None):
        trace_frame = context.begin_control_trace(self)
        try:
            self._run(context, temporaries)
            context.end_control_trace(trace_frame, self._last_report)
        except BaseException:
            context.abort_control_trace(trace_frame)
            raise

    def _run(self, context, temporaries=None):
        runtime_args = context.runtime_args()
        predicate_object = runtime_args[self.predicate]
        counter_object = (
            runtime_args[self.counter] if self.counter is not None else None
        )
        status_object = runtime_args[self.status] if self.status is not None else None
        observations = []
        predicate_values = []
        counter_values = []
        status_values = []
        chunks = []
        executed = 0
        encoded_iterations = 0
        masked_iterations = 0
        observation_batches = 0
        observation_scalar_count = 0
        device_to_host_bytes = 0
        program = impl.get_runtime().prog
        arch = impl.current_cfg().arch
        use_transfer_planner = _control_transfer_uses_planner(arch)
        transfer_before = program._graph_observation_staging_stats()
        vulkan_nested_attempted = (
            arch == _ti_core.Arch.vulkan
            and self._vulkan_nested is not None
            and self._vulkan_nested_inner is not None
            and self._vulkan_nested_boundaries is not None
            and not context.control_trace_enabled()
        )
        if self._run_vulkan_nested_structured(
            context,
            runtime_args,
            transfer_before,
        ):
            return
        vulkan_native_attempted = (
            arch == _ti_core.Arch.vulkan
            and self._native_upgrade_eligible
            and self._vulkan_structured is not None
        )
        if self._run_vulkan_structured(context, runtime_args, transfer_before):
            return

        def observe_control(boundary):
            nonlocal observation_batches
            nonlocal observation_scalar_count
            nonlocal device_to_host_bytes
            values = [predicate_object]
            names = [self.predicate]
            if counter_object is not None:
                values.append(counter_object)
                names.append(self.counter)
            if status_object is not None:
                values.append(status_object)
                names.append(self.status)
            observed, byte_count = _control_scalar_values(
                values,
                names,
                use_transfer_planner=use_transfer_planner,
            )
            observation_batches += 1
            observation_scalar_count += len(observed)
            device_to_host_bytes += byte_count
            observations.append(boundary)
            predicate_values.append(observed[0])
            next_index = 1
            if counter_object is not None:
                counter_values.append(observed[next_index])
                next_index += 1
            if status_object is not None:
                status_values.append(observed[next_index])
            return observed[0]

        self._condition.run(context)
        predicate_value = observe_control(0)
        initial_counter = counter_values[-1] if counter_object is not None else None
        initial_status = status_values[-1] if status_object is not None else None
        active = predicate_value != 0
        native_selected = False
        native_reason = (
            "vulkan_nested_runtime_fallback"
            if vulkan_nested_attempted
            else (
                "vulkan_structured_runtime_fallback"
                if vulkan_native_attempted
                else self._native_upgrade_reason
            )
        )
        if (
            arch == _ti_core.Arch.cuda
            and active
            and self.max_iterations > 0
            and self._native_upgrade_eligible
            and not self._has_nested_control
        ):
            predicate_ndarray = getattr(predicate_object, "arr", None)
            if predicate_ndarray is None:
                native_reason = "predicate_ndarray_required"
            else:
                native_run = (
                    self._chunks[1].compiled_graph.jit_run_bounded_cuda_masked_cached
                    if self._cuda_control_lowering == "cuda_masked_bounded_graph"
                    else self._chunks[1].compiled_graph.jit_run_bounded_cuda_cached
                )
                native_selected = native_run(
                    context.compile_config(),
                    context.flattened_args(self.recording_runtime_arg_names),
                    self._native_jit_cache,
                    predicate_ndarray,
                    self.max_iterations,
                    True,
                )
                native_reason = (
                    f"selected_{self._cuda_control_lowering}"
                    if native_selected
                    else "conditional_capture_fallback"
                )
            if not native_selected and self.lowering_mode == "native_required":
                raise TaichiRuntimeError(
                    "Graph while native CUDA lowering failed: " f"{native_reason}"
                )
        if native_selected:
            predicate_value = observe_control(-1)
            logical_native = counter_values[-1] - initial_counter
            if logical_native < 0 or logical_native > self.max_iterations:
                raise TaichiRuntimeError(
                    "Graph while counter left its iteration budget"
                )
            active = predicate_value != 0
            if self._cuda_control_lowering == "cuda_masked_bounded_graph":
                executed = logical_native
                encoded_iterations = self.max_iterations
                masked_iterations = self.max_iterations - logical_native
            else:
                executed = self.max_iterations if active else logical_native
                encoded_iterations = executed
            observations[-1] = executed
            if encoded_iterations:
                chunks.append(encoded_iterations)

        while not native_selected and active and executed < self.max_iterations:
            chunk = self._select_chunk(self.max_iterations - executed)
            context.set_control_trace_iteration(self, executed)
            self._chunks[chunk].run(context)
            executed += chunk
            chunks.append(chunk)
            predicate_value = observe_control(executed)
            active = predicate_value != 0

        if not observations or observations[-1] != executed:
            observe_control(executed)
        final_counter = counter_values[-1] if counter_object is not None else None
        final_status = status_values[-1] if status_object is not None else None
        if final_counter is not None:
            logical = final_counter - initial_counter
            if logical < 0 or logical > executed:
                raise TaichiRuntimeError(
                    "Graph while counter must increase by no more than the "
                    "executed iteration count"
                )
        else:
            logical = executed
        lowering = (
            self._cuda_control_lowering
            if native_selected
            else (
                "cpu_host_loop"
                if arch in (_ti_core.Arch.x64, _ti_core.Arch.arm64)
                else (
                    "portable_chunk_replay"
                    if self.chunk_limit > 1
                    else "portable_exact_replay"
                )
            )
        )
        transfer_after = program._graph_observation_staging_stats()

        def transfer_delta(name):
            return int(transfer_after[name]) - int(transfer_before[name])

        self._last_report = GraphWhileReport(
            name=self.name,
            region_path=self.region_path,
            structured_depth=self.control_depth,
            backend=_backend_name(_ti_core.arch_name(arch)),
            lowering=lowering,
            max_iterations=self.max_iterations,
            logical_iterations=logical,
            executed_iterations=executed,
            overshoot_iterations=executed - logical,
            observation_boundaries=tuple(observations),
            predicate_values=tuple(predicate_values),
            counter_values=tuple(counter_values),
            status_resource=self.status,
            status_values=tuple(status_values),
            chunk_sizes=tuple(chunks),
            observation_batches=observation_batches,
            observation_scalar_count=observation_scalar_count,
            device_to_host_bytes=device_to_host_bytes,
            initial_counter=initial_counter,
            final_counter=final_counter,
            initial_status=initial_status,
            final_status=final_status,
            native_upgrade_eligible=self._native_upgrade_eligible,
            native_upgrade_reason=native_reason,
            persistent_staging_bytes=int(transfer_after["persistent_bytes"]),
            staging_allocations=transfer_delta("allocations"),
            staging_reuses=transfer_delta("reuses"),
            packed_observation_batches=transfer_delta("packed_batches"),
            direct_observation_batches=(
                transfer_delta("direct_batches")
                if use_transfer_planner
                else observation_batches
            ),
            staging_fallback_batches=transfer_delta("fallback_batches"),
            packed_observation_bytes=transfer_delta("packed_payload_bytes"),
            condition_dispatch_count=self.condition_dispatch_count,
            body_dispatch_count=self.body_dispatch_count,
            control_inputs=self.control_inputs,
            carried_state=self.carried_state,
            controller_dispatch_count=(
                self.max_iterations
                if native_selected
                and self._cuda_control_lowering == "cuda_masked_bounded_graph"
                else 0
            ),
            controller_invocation_count=(
                self.max_iterations
                if native_selected
                and self._cuda_control_lowering == "cuda_masked_bounded_graph"
                else 0
            ),
            logical_body_dispatch_count=logical * self.body_dispatch_count,
            encoded_iterations=(encoded_iterations if native_selected else executed),
            masked_iterations=masked_iterations,
        )

    def invalidate_runtime(self, preserve_executables=False):
        if preserve_executables:
            self._native_jit_cache.retire_snode_tree_runtime_state()
        else:
            self._native_jit_cache.clear_runtime_state()
        self._condition.invalidate_runtime(preserve_executables=preserve_executables)
        for node in self._chunks.values():
            node.invalidate_runtime(preserve_executables=preserve_executables)
        if self._vulkan_structured is not None:
            self._vulkan_structured.invalidate_runtime(
                preserve_executables=preserve_executables
            )
        if self._vulkan_nested is not None:
            self._vulkan_nested.invalidate_runtime(
                preserve_executables=preserve_executables
            )
        if self._cuda_nested is not None:
            self._cuda_nested.invalidate_runtime(
                preserve_executables=preserve_executables
            )

    @property
    def debug_graph_stats(self):
        chunk_stats = tuple(node.debug_graph_stats for node in self._chunks.values())
        condition_stats = (self._condition.debug_graph_stats,)
        if self._vulkan_nested is not None or self._cuda_nested is not None:
            return (
                self._native_jit_cache._debug_graph_stats(),
                *condition_stats,
                *chunk_stats,
            )
        if not self._native_upgrade_eligible:
            return (*condition_stats, *chunk_stats)
        if self._vulkan_structured is not None:
            return (
                self._native_jit_cache._debug_graph_stats(),
                *condition_stats,
                *chunk_stats,
            )
        return (
            self._native_jit_cache._debug_graph_stats(),
            *condition_stats,
            *chunk_stats,
        )

    @property
    def snapshot_graph_stats(self):
        chunk_stats = tuple(node.snapshot_graph_stats for node in self._chunks.values())
        condition_stats = (self._condition.snapshot_graph_stats,)
        if self._vulkan_nested is not None or self._cuda_nested is not None:
            return (
                self._native_jit_cache._debug_graph_stats(False),
                *condition_stats,
                *chunk_stats,
            )
        if not self._native_upgrade_eligible:
            return (*condition_stats, *chunk_stats)
        return (
            self._native_jit_cache._debug_graph_stats(False),
            *condition_stats,
            *chunk_stats,
        )

    @property
    def last_report(self):
        return self._last_report

    @property
    def debug_info(self):
        return {
            "kind": "structured_while",
            "name": self.name,
            "region_path": self.region_path,
            "structured_depth": self.control_depth,
            "max_nested_depth": self.structured_depth,
            "nested_portable_exact": self._portable_exact_nested,
            "nested_subregion": self._nested_subregion,
            "nested_leaf_native_upgrade_eligible": (
                self._nested_subregion and self._native_upgrade_eligible
            ),
            "nested_native_upgrade_eligible": (
                self._vulkan_nested is not None or self._cuda_nested is not None
            ),
            "nested_native_upgrade_reason": (
                self._vulkan_nested_reason
                if impl.current_cfg().arch == _ti_core.Arch.vulkan
                else self._cuda_nested_reason
            ),
            "cuda_nested_control_lowering": (self._cuda_nested_control_lowering),
            "condition_dispatch_count": self.condition_dispatch_count,
            "body_dispatch_count": self.body_dispatch_count,
            "max_iterations": self.max_iterations,
            "chunk_limit": self.chunk_limit,
            "compound_chunk_limit": self.compound_chunk_limit,
            "compound_chunk_count": self.compound_chunk_count,
            "vulkan_first_chunk_strategy": self.vulkan_first_chunk_strategy,
            "control_input_count": len(self.control_inputs),
            "carried_state_count": len(self.carried_state),
            "has_status": self.status is not None,
            "masked_execution": self.masked_execution,
            "lowering_mode": self.lowering_mode,
            "native_upgrade_eligible": self._native_upgrade_eligible,
        }


class _CompiledIfGraphNode:
    needs_runtime_args = True
    source_native_count = 0
    region_kind = "structured_if"

    def __init__(
        self,
        condition,
        then_region,
        else_region,
        *,
        predicate,
        control_inputs,
        lowering_mode,
        name,
        _control_recipe=None,
    ):
        if not isinstance(condition, Sequential) or condition._dispatch_count == 0:
            raise TaichiRuntimeError(
                "Graph if condition must be a non-empty Sequential"
            )
        if not isinstance(then_region, Sequential) or then_region._dispatch_count == 0:
            raise TaichiRuntimeError(
                "Graph if then_region must be a non-empty Sequential"
            )
        if else_region is not None and (
            not isinstance(else_region, Sequential) or else_region._dispatch_count == 0
        ):
            raise TaichiRuntimeError(
                "Graph if else_region must be a non-empty Sequential"
            )
        self.name = name
        self.predicate = predicate
        self.control_inputs = tuple(control_inputs)
        self.lowering_mode = lowering_mode
        self.region_path = name
        self.control_depth = 1
        self._nested_subregion = False
        self._definition_regions = tuple(
            (role, region)
            for role, region in (
                ("condition", condition),
                ("then", then_region),
                ("else", else_region),
            )
            if region is not None
        )
        self._definition_children = tuple(
            (role, _sequence_structured_nodes(region))
            for role, region in self._definition_regions
        )
        self.structured_depth, self._has_nested_control = (
            _prepare_structured_definition(
                "if",
                name,
                lowering_mode,
                self._definition_regions,
            )
        )
        required = {predicate, *self.control_inputs}
        missing = sorted(required.difference(condition._runtime_arg_names))
        if missing:
            raise TaichiRuntimeError(
                "Graph if condition does not declare control resources: "
                + ", ".join(missing)
            )
        self._condition = _compile_sequential_runtime_node(
            (condition,),
            name=f"{name}_condition",
            region_kind="if_condition",
        )
        self._then = _compile_sequential_runtime_node(
            (then_region,), name=f"{name}_then", region_kind="if_branch"
        )
        self._else = (
            None
            if else_region is None
            else _compile_sequential_runtime_node(
                (else_region,), name=f"{name}_else", region_kind="if_branch"
            )
        )
        branch_sequences = (
            (then_region,) if else_region is None else (then_region, else_region)
        )
        self._native_branches = (
            None
            if self._has_nested_control
            else _compile_sequential_runtime_node(
                branch_sequences,
                name=f"{name}_native_branches",
                region_kind="if_branch",
                region_kinds=("if_branch",) * len(branch_sequences),
            )
        )
        self._native_branch_dispatch_counts = tuple(
            region._dispatch_count for region in branch_sequences
        )
        arch = impl.current_cfg().arch
        self._native_upgrade_eligible, self._native_upgrade_reason = (
            _cuda_branch_upgrade_status(
                arch,
                lowering_mode,
                "if",
                requested_recipe=_control_recipe,
            )
        )
        self._cuda_control_lowering = (
            _cuda_structured_control_lowering(requested_recipe=_control_recipe)
            if arch == _ti_core.Arch.cuda
            else None
        )
        if self._has_nested_control:
            self._native_upgrade_eligible = False
            self._native_upgrade_reason = "nested_structured_portable_exact"
        self._native_jit_cache = _ti_core.CompiledGraphJITCache()
        nodes = tuple(
            node for node in (self._condition, self._then, self._else) if node
        )
        self.temporary_actions = _merge_temporary_actions(nodes)
        self.temporary_runtime_arg_names = frozenset().union(
            *(node.temporary_runtime_arg_names for node in nodes)
        )
        self.fixed_runtime_args = _merge_fixed_runtime_args(nodes)
        self.recording_runtime_arg_names = frozenset().union(
            *(node.recording_runtime_arg_names for node in nodes)
        )
        self.derived_runtime_arg_names = _merge_derived_runtime_arg_names(nodes)
        self.runtime_arg_names = self.recording_runtime_arg_names.difference(
            (
                *self.fixed_runtime_args,
                *self.temporary_runtime_arg_names,
                *self.derived_runtime_arg_names,
            )
        )
        sequences = tuple(
            region
            for region in (condition, then_region, else_region)
            if region is not None
        )
        self.lifetime_leases = tuple(
            lease for region in sequences for lease in region._lifetime_leases
        )
        self.source_native_count = sum(
            region._source_native_count for region in sequences
        )
        self.snode_tree_dependencies = frozenset().union(
            *(node.snode_tree_dependencies for node in nodes)
        )
        self.snode_tree_dependency_info = frozenset().union(
            *(node.snode_tree_dependency_info for node in nodes)
        )
        self.condition_dispatch_count = condition._dispatch_count
        self.then_dispatch_count = then_region._dispatch_count
        self.else_dispatch_count = (
            0 if else_region is None else else_region._dispatch_count
        )
        self.dispatch_count = (
            self.condition_dispatch_count
            + self.then_dispatch_count
            + self.else_dispatch_count
        )
        self.ir_node = IfRegion(
            predicate=predicate,
            condition=SequentialRegion(
                tuple(condition._ir_nodes), name=f"{name}_condition"
            ),
            then_region=SequentialRegion(
                tuple(then_region._ir_nodes), name=f"{name}_then"
            ),
            else_region=(
                None
                if else_region is None
                else SequentialRegion(tuple(else_region._ir_nodes), name=f"{name}_else")
            ),
            control_inputs=self.control_inputs,
            name=name,
        )
        self._last_report = None
        self._pending_report = None

    @property
    def supports_native_submission(self):
        return self.lowering_mode == "native_required" and self._native_upgrade_eligible

    def _run_native_branch(self, context):
        if self._native_branches is None:
            return False
        predicate_object = context.runtime_args()[self.predicate]
        predicate_ndarray = getattr(predicate_object, "arr", None)
        if predicate_ndarray is None:
            return False
        native_run = (
            self._native_branches.compiled_graph.jit_run_conditional_cuda_masked_cached
            if self._cuda_control_lowering == "cuda_masked_bounded_graph"
            else self._native_branches.compiled_graph.jit_run_conditional_cuda_cached
        )
        return native_run(
            context.compile_config(),
            context.flattened_args(self._native_branches.recording_runtime_arg_names),
            self._native_jit_cache,
            predicate_ndarray,
            self._native_branch_dispatch_counts,
            0,
            -1,
        )

    def _mark_nested_portable(self):
        if self.lowering_mode == "native_required":
            raise TaichiRuntimeError(
                f"Graph if {self.name!r} is nested at {self.region_path!r}, "
                "but nested native_required lowering is unavailable"
            )
        self._nested_subregion = True

    def run_for_submission(self, context, temporaries=None):
        if not self.supports_native_submission:
            raise TaichiRuntimeError(
                "Graph if submission requires native_required CUDA lowering"
            )
        self._last_report = None
        self._pending_report = None
        self._condition.run(context)
        if not self._run_native_branch(context):
            raise TaichiRuntimeError(
                "Native CUDA Graph if submission became unavailable; "
                "synchronous fallback is disabled"
            )

    def run(self, context, temporaries=None):
        trace_frame = context.begin_control_trace(self)
        try:
            self._run(context, temporaries)
            if trace_frame is not None and self._last_report is None:
                self.materialize_pending_report()
            context.end_control_trace(trace_frame, self._last_report)
        except BaseException:
            context.abort_control_trace(trace_frame)
            raise

    def _run(self, context, temporaries=None):
        self._condition.run(context)
        runtime_args = context.runtime_args()
        arch = impl.current_cfg().arch
        native_selected = False
        native_reason = self._native_upgrade_reason
        if self._native_upgrade_eligible:
            native_selected = self._run_native_branch(context)
            native_reason = (
                "selected" if native_selected else "conditional_capture_fallback"
            )
            if not native_selected and self.lowering_mode == "native_required":
                raise TaichiRuntimeError(
                    f"Graph if native CUDA lowering failed: {native_reason}"
                )
        if native_selected:
            self._last_report = None
            self._pending_report = (
                runtime_args[self.predicate],
                arch,
                self._cuda_control_lowering,
            )
            return
        self._pending_report = None
        observed, byte_count = _control_scalar_values(
            [runtime_args[self.predicate]],
            [self.predicate],
            use_transfer_planner=_control_transfer_uses_planner(arch),
        )
        predicate_value = observed[0]
        if predicate_value != 0:
            selected = "then"
            selected_dispatches = self.then_dispatch_count
            if not native_selected:
                self._then.run(context)
        elif self._else is not None:
            selected = "else"
            selected_dispatches = self.else_dispatch_count
            if not native_selected:
                self._else.run(context)
        else:
            selected = "none"
            selected_dispatches = 0
        self._last_report = GraphBranchReport(
            name=self.name,
            region_path=self.region_path,
            structured_depth=self.control_depth,
            backend=_backend_name(_ti_core.arch_name(arch)),
            kind="if",
            lowering=(
                "cuda_conditional_graph"
                if native_selected
                else _structured_host_lowering(arch)
            ),
            selector_resource=self.predicate,
            selector_value=predicate_value,
            selected_branch=selected,
            observation_scalar_count=1,
            device_to_host_bytes=byte_count,
            condition_dispatch_count=self.condition_dispatch_count,
            branch_dispatch_count=selected_dispatches,
            control_inputs=self.control_inputs,
        )

    def materialize_pending_report(self):
        if self._pending_report is None:
            return
        predicate_object, arch, lowering = self._pending_report
        observed, byte_count = _control_scalar_values(
            [predicate_object],
            [self.predicate],
            use_transfer_planner=_control_transfer_uses_planner(arch),
        )
        predicate_value = observed[0]
        if predicate_value != 0:
            selected = "then"
            selected_dispatches = self.then_dispatch_count
        elif self._else is not None:
            selected = "else"
            selected_dispatches = self.else_dispatch_count
        else:
            selected = "none"
            selected_dispatches = 0
        self._last_report = GraphBranchReport(
            name=self.name,
            region_path=self.region_path,
            structured_depth=self.control_depth,
            backend=_backend_name(_ti_core.arch_name(arch)),
            kind="if",
            lowering=lowering,
            selector_resource=self.predicate,
            selector_value=predicate_value,
            selected_branch=selected,
            observation_scalar_count=1,
            device_to_host_bytes=byte_count,
            condition_dispatch_count=self.condition_dispatch_count,
            branch_dispatch_count=selected_dispatches,
            control_inputs=self.control_inputs,
            encoded_dispatch_count=(
                sum(self._native_branch_dispatch_counts)
                if lowering == "cuda_masked_bounded_graph"
                else selected_dispatches
            ),
            masked_dispatch_count=(
                sum(self._native_branch_dispatch_counts) - selected_dispatches
                if lowering == "cuda_masked_bounded_graph"
                else 0
            ),
        )
        self._pending_report = None

    def invalidate_runtime(self, preserve_executables=False):
        self._pending_report = None
        if preserve_executables:
            self._native_jit_cache.retire_snode_tree_runtime_state()
        else:
            self._native_jit_cache.clear_runtime_state()
        self._condition.invalidate_runtime(preserve_executables=preserve_executables)
        self._then.invalidate_runtime(preserve_executables=preserve_executables)
        if self._else is not None:
            self._else.invalidate_runtime(preserve_executables=preserve_executables)

    @property
    def debug_graph_stats(self):
        nodes = tuple(
            node for node in (self._condition, self._then, self._else) if node
        )
        child_stats = tuple(node.debug_graph_stats for node in nodes)
        if not self._native_upgrade_eligible:
            return child_stats
        return (self._native_jit_cache._debug_graph_stats(), *child_stats)

    @property
    def snapshot_graph_stats(self):
        nodes = tuple(
            node for node in (self._condition, self._then, self._else) if node
        )
        child_stats = tuple(node.snapshot_graph_stats for node in nodes)
        if not self._native_upgrade_eligible:
            return child_stats
        return (
            self._native_jit_cache._debug_graph_stats(False),
            *child_stats,
        )

    @property
    def last_report(self):
        return self._last_report

    @property
    def debug_info(self):
        return {
            "kind": "structured_if",
            "name": self.name,
            "region_path": self.region_path,
            "structured_depth": self.control_depth,
            "max_nested_depth": self.structured_depth,
            "nested_portable_exact": self._has_nested_control
            or (self.control_depth > 1 and not self._native_upgrade_eligible),
            "nested_subregion": self._nested_subregion,
            "nested_leaf_native_upgrade_eligible": (
                self._nested_subregion and self._native_upgrade_eligible
            ),
            "condition_dispatch_count": self.condition_dispatch_count,
            "then_dispatch_count": self.then_dispatch_count,
            "else_dispatch_count": self.else_dispatch_count,
            "control_input_count": len(self.control_inputs),
            "lowering_mode": self.lowering_mode,
            "native_upgrade_eligible": self._native_upgrade_eligible,
        }


class _CompiledSwitchGraphNode:
    needs_runtime_args = True
    source_native_count = 0
    region_kind = "structured_switch"

    def __init__(
        self,
        condition,
        branches,
        default_region,
        *,
        selector,
        control_inputs,
        lowering_mode,
        name,
        _control_recipe=None,
    ):
        if not isinstance(condition, Sequential) or condition._dispatch_count == 0:
            raise TaichiRuntimeError(
                "Graph switch condition must be a non-empty Sequential"
            )
        branches = tuple(branches)
        if not branches or any(
            not isinstance(branch, Sequential) or branch._dispatch_count == 0
            for branch in branches
        ):
            raise TaichiRuntimeError(
                "Graph switch requires non-empty Sequential branches"
            )
        if default_region is not None and (
            not isinstance(default_region, Sequential)
            or default_region._dispatch_count == 0
        ):
            raise TaichiRuntimeError(
                "Graph switch default_region must be a non-empty Sequential"
            )
        self.name = name
        self.selector = selector
        self.control_inputs = tuple(control_inputs)
        self.lowering_mode = lowering_mode
        self.region_path = name
        self.control_depth = 1
        self._nested_subregion = False
        definition_regions = [("condition", condition)]
        definition_regions.extend(
            (f"case_{index}", branch) for index, branch in enumerate(branches)
        )
        if default_region is not None:
            definition_regions.append(("default", default_region))
        self._definition_regions = tuple(definition_regions)
        self._definition_children = tuple(
            (role, _sequence_structured_nodes(region))
            for role, region in self._definition_regions
        )
        self.structured_depth, self._has_nested_control = (
            _prepare_structured_definition(
                "switch",
                name,
                lowering_mode,
                self._definition_regions,
            )
        )
        required = {selector, *self.control_inputs}
        missing = sorted(required.difference(condition._runtime_arg_names))
        if missing:
            raise TaichiRuntimeError(
                "Graph switch condition does not declare control resources: "
                + ", ".join(missing)
            )
        self._condition = _compile_sequential_runtime_node(
            (condition,),
            name=f"{name}_condition",
            region_kind="switch_condition",
        )
        self._branches = tuple(
            _compile_sequential_runtime_node(
                (branch,),
                name=f"{name}_case_{index}",
                region_kind="switch_branch",
            )
            for index, branch in enumerate(branches)
        )
        self._default = (
            None
            if default_region is None
            else _compile_sequential_runtime_node(
                (default_region,),
                name=f"{name}_default",
                region_kind="switch_branch",
            )
        )
        branch_sequences = branches
        if default_region is not None:
            branch_sequences = (*branch_sequences, default_region)
        self._native_branches = (
            None
            if self._has_nested_control
            else _compile_sequential_runtime_node(
                branch_sequences,
                name=f"{name}_native_branches",
                region_kind="switch_branch",
                region_kinds=("switch_branch",) * len(branch_sequences),
            )
        )
        self._native_branch_dispatch_counts = tuple(
            region._dispatch_count for region in branch_sequences
        )
        self._native_default_branch = -1 if default_region is None else len(branches)
        arch = impl.current_cfg().arch
        self._native_upgrade_eligible, self._native_upgrade_reason = (
            _cuda_branch_upgrade_status(
                arch,
                lowering_mode,
                "switch",
                requested_recipe=_control_recipe,
            )
        )
        self._cuda_control_lowering = (
            _cuda_structured_control_lowering(requested_recipe=_control_recipe)
            if arch == _ti_core.Arch.cuda
            else None
        )
        if self._has_nested_control:
            self._native_upgrade_eligible = False
            self._native_upgrade_reason = "nested_structured_portable_exact"
        self._native_jit_cache = _ti_core.CompiledGraphJITCache()
        nodes = (self._condition, *self._branches)
        if self._default is not None:
            nodes = (*nodes, self._default)
        self.temporary_actions = _merge_temporary_actions(nodes)
        self.temporary_runtime_arg_names = frozenset().union(
            *(node.temporary_runtime_arg_names for node in nodes)
        )
        self.fixed_runtime_args = _merge_fixed_runtime_args(nodes)
        self.recording_runtime_arg_names = frozenset().union(
            *(node.recording_runtime_arg_names for node in nodes)
        )
        self.derived_runtime_arg_names = _merge_derived_runtime_arg_names(nodes)
        self.runtime_arg_names = self.recording_runtime_arg_names.difference(
            (
                *self.fixed_runtime_args,
                *self.temporary_runtime_arg_names,
                *self.derived_runtime_arg_names,
            )
        )
        sequences = (condition, *branches)
        if default_region is not None:
            sequences = (*sequences, default_region)
        self.lifetime_leases = tuple(
            lease for region in sequences for lease in region._lifetime_leases
        )
        self.source_native_count = sum(
            region._source_native_count for region in sequences
        )
        self.snode_tree_dependencies = frozenset().union(
            *(node.snode_tree_dependencies for node in nodes)
        )
        self.snode_tree_dependency_info = frozenset().union(
            *(node.snode_tree_dependency_info for node in nodes)
        )
        self.condition_dispatch_count = condition._dispatch_count
        self.branch_dispatch_counts = tuple(
            branch._dispatch_count for branch in branches
        )
        self.default_dispatch_count = (
            0 if default_region is None else default_region._dispatch_count
        )
        self.dispatch_count = (
            self.condition_dispatch_count
            + sum(self.branch_dispatch_counts)
            + self.default_dispatch_count
        )
        self.ir_node = SwitchRegion(
            selector=selector,
            condition=SequentialRegion(
                tuple(condition._ir_nodes), name=f"{name}_condition"
            ),
            branches=tuple(
                SequentialRegion(tuple(branch._ir_nodes), name=f"{name}_case_{index}")
                for index, branch in enumerate(branches)
            ),
            default_region=(
                None
                if default_region is None
                else SequentialRegion(
                    tuple(default_region._ir_nodes), name=f"{name}_default"
                )
            ),
            control_inputs=self.control_inputs,
            name=name,
        )
        self._last_report = None
        self._pending_report = None

    @property
    def supports_native_submission(self):
        return self.lowering_mode == "native_required" and self._native_upgrade_eligible

    def _run_native_branch(self, context):
        if self._native_branches is None:
            return False
        selector_object = context.runtime_args()[self.selector]
        selector_ndarray = getattr(selector_object, "arr", None)
        if selector_ndarray is None:
            return False
        native_run = (
            self._native_branches.compiled_graph.jit_run_conditional_cuda_masked_cached
            if self._cuda_control_lowering == "cuda_masked_bounded_graph"
            else self._native_branches.compiled_graph.jit_run_conditional_cuda_cached
        )
        return native_run(
            context.compile_config(),
            context.flattened_args(self._native_branches.recording_runtime_arg_names),
            self._native_jit_cache,
            selector_ndarray,
            self._native_branch_dispatch_counts,
            2,
            self._native_default_branch,
        )

    def _mark_nested_portable(self):
        if self.lowering_mode == "native_required":
            raise TaichiRuntimeError(
                f"Graph switch {self.name!r} is nested at {self.region_path!r}, "
                "but nested native_required lowering is unavailable"
            )
        self._nested_subregion = True

    def run_for_submission(self, context, temporaries=None):
        if not self.supports_native_submission:
            raise TaichiRuntimeError(
                "Graph switch submission requires native_required CUDA lowering"
            )
        self._last_report = None
        self._pending_report = None
        self._condition.run(context)
        if not self._run_native_branch(context):
            raise TaichiRuntimeError(
                "Native CUDA Graph switch submission became unavailable; "
                "synchronous fallback is disabled"
            )

    def run(self, context, temporaries=None):
        trace_frame = context.begin_control_trace(self)
        try:
            self._run(context, temporaries)
            if trace_frame is not None and self._last_report is None:
                self.materialize_pending_report()
            context.end_control_trace(trace_frame, self._last_report)
        except BaseException:
            context.abort_control_trace(trace_frame)
            raise

    def _run(self, context, temporaries=None):
        self._condition.run(context)
        runtime_args = context.runtime_args()
        arch = impl.current_cfg().arch
        native_selected = False
        native_reason = self._native_upgrade_reason
        if self._native_upgrade_eligible:
            native_selected = self._run_native_branch(context)
            native_reason = (
                "selected" if native_selected else "conditional_capture_fallback"
            )
            if not native_selected and self.lowering_mode == "native_required":
                raise TaichiRuntimeError(
                    f"Graph switch native CUDA lowering failed: {native_reason}"
                )
        if native_selected:
            self._last_report = None
            self._pending_report = (
                runtime_args[self.selector],
                arch,
                self._cuda_control_lowering,
            )
            return
        self._pending_report = None
        observed, byte_count = _control_scalar_values(
            [runtime_args[self.selector]],
            [self.selector],
            use_transfer_planner=_control_transfer_uses_planner(arch),
        )
        selector_value = int(observed[0])
        if 0 <= selector_value < len(self._branches):
            selected = f"case_{selector_value}"
            selected_dispatches = self.branch_dispatch_counts[selector_value]
            if not native_selected:
                self._branches[selector_value].run(context)
        elif self._default is not None:
            selected = "default"
            selected_dispatches = self.default_dispatch_count
            if not native_selected:
                self._default.run(context)
        else:
            selected = "none"
            selected_dispatches = 0
        self._last_report = GraphBranchReport(
            name=self.name,
            region_path=self.region_path,
            structured_depth=self.control_depth,
            backend=_backend_name(_ti_core.arch_name(arch)),
            kind="switch",
            lowering=(
                "cuda_conditional_graph"
                if native_selected
                else _structured_host_lowering(arch)
            ),
            selector_resource=self.selector,
            selector_value=selector_value,
            selected_branch=selected,
            observation_scalar_count=1,
            device_to_host_bytes=byte_count,
            condition_dispatch_count=self.condition_dispatch_count,
            branch_dispatch_count=selected_dispatches,
            control_inputs=self.control_inputs,
        )

    def materialize_pending_report(self):
        if self._pending_report is None:
            return
        selector_object, arch, lowering = self._pending_report
        observed, byte_count = _control_scalar_values(
            [selector_object],
            [self.selector],
            use_transfer_planner=_control_transfer_uses_planner(arch),
        )
        selector_value = int(observed[0])
        if 0 <= selector_value < len(self._branches):
            selected = f"case_{selector_value}"
            selected_dispatches = self.branch_dispatch_counts[selector_value]
        elif self._default is not None:
            selected = "default"
            selected_dispatches = self.default_dispatch_count
        else:
            selected = "none"
            selected_dispatches = 0
        self._last_report = GraphBranchReport(
            name=self.name,
            region_path=self.region_path,
            structured_depth=self.control_depth,
            backend=_backend_name(_ti_core.arch_name(arch)),
            kind="switch",
            lowering=lowering,
            selector_resource=self.selector,
            selector_value=selector_value,
            selected_branch=selected,
            observation_scalar_count=1,
            device_to_host_bytes=byte_count,
            condition_dispatch_count=self.condition_dispatch_count,
            branch_dispatch_count=selected_dispatches,
            control_inputs=self.control_inputs,
            encoded_dispatch_count=(
                sum(self._native_branch_dispatch_counts)
                if lowering == "cuda_masked_bounded_graph"
                else selected_dispatches
            ),
            masked_dispatch_count=(
                sum(self._native_branch_dispatch_counts) - selected_dispatches
                if lowering == "cuda_masked_bounded_graph"
                else 0
            ),
        )
        self._pending_report = None

    def invalidate_runtime(self, preserve_executables=False):
        self._pending_report = None
        if preserve_executables:
            self._native_jit_cache.retire_snode_tree_runtime_state()
        else:
            self._native_jit_cache.clear_runtime_state()
        self._condition.invalidate_runtime(preserve_executables=preserve_executables)
        for branch in self._branches:
            branch.invalidate_runtime(preserve_executables=preserve_executables)
        if self._default is not None:
            self._default.invalidate_runtime(preserve_executables=preserve_executables)

    @property
    def debug_graph_stats(self):
        nodes = (self._condition, *self._branches)
        if self._default is not None:
            nodes = (*nodes, self._default)
        child_stats = tuple(node.debug_graph_stats for node in nodes)
        if not self._native_upgrade_eligible:
            return child_stats
        return (self._native_jit_cache._debug_graph_stats(), *child_stats)

    @property
    def snapshot_graph_stats(self):
        nodes = (self._condition, *self._branches)
        if self._default is not None:
            nodes = (*nodes, self._default)
        child_stats = tuple(node.snapshot_graph_stats for node in nodes)
        if not self._native_upgrade_eligible:
            return child_stats
        return (
            self._native_jit_cache._debug_graph_stats(False),
            *child_stats,
        )

    @property
    def last_report(self):
        return self._last_report

    @property
    def debug_info(self):
        return {
            "kind": "structured_switch",
            "name": self.name,
            "region_path": self.region_path,
            "structured_depth": self.control_depth,
            "max_nested_depth": self.structured_depth,
            "nested_portable_exact": self._has_nested_control
            or (self.control_depth > 1 and not self._native_upgrade_eligible),
            "nested_subregion": self._nested_subregion,
            "nested_leaf_native_upgrade_eligible": (
                self._nested_subregion and self._native_upgrade_eligible
            ),
            "condition_dispatch_count": self.condition_dispatch_count,
            "branch_dispatch_counts": self.branch_dispatch_counts,
            "default_dispatch_count": self.default_dispatch_count,
            "control_input_count": len(self.control_inputs),
            "lowering_mode": self.lowering_mode,
            "native_upgrade_eligible": self._native_upgrade_eligible,
        }


def _structured_root_call_sites(node):
    if _is_structured_control_node(node):
        yield node
        return
    if isinstance(node, _CompiledSequentialRegionNode):
        for child in node.nodes:
            yield from _structured_root_call_sites(child)


def _prepare_structured_definition_tree(nodes):
    """Validate a tree-shaped, single-owner structured definition.

    Compiled control nodes contain mutable reports, paths, and backend caches.
    Sharing one node across call sites or Graph objects would therefore bypass
    the per-Graph lifecycle lock. Structured definitions are intentionally
    single-consumer until compiled nodes can be cloned with independent state.
    """

    result = []
    seen_paths = {}
    active = set()
    root_name_counts = {}

    def visit(node, path, depth):
        identity = id(node)
        if identity in active:
            raise TaichiRuntimeError(
                f"Graph structured-control definition contains a cycle at {path!r}"
            )
        previous_path = seen_paths.get(identity)
        if previous_path is not None:
            raise TaichiRuntimeError(
                "Graph structured-control node is reused at multiple call "
                f"sites: {previous_path!r} and {path!r}. Build a fresh "
                "structured node for each call site."
            )
        if getattr(node, "_graph_owner_token", None) is not None:
            raise TaichiRuntimeError(
                f"Graph structured-control node at {path!r} already belongs "
                "to a compiled Graph. Rebuild the structured definition "
                "before compiling another Graph."
            )
        if depth > 2:
            raise TaichiRuntimeError(
                "Graph structured-control definition exceeds the maximum depth of 2"
            )
        seen_paths[identity] = path
        result.append((node, path, depth))
        active.add(identity)
        try:
            for role, children in node._definition_children:
                name_counts = {}
                for child in children:
                    occurrence = name_counts.get(child.name, 0)
                    name_counts[child.name] = occurrence + 1
                    suffix = "" if occurrence == 0 else f"[{occurrence}]"
                    visit(
                        child,
                        f"{path}/{role}/{child.name}{suffix}",
                        depth + 1,
                    )
        finally:
            active.remove(identity)

    for source_node in nodes:
        roots = tuple(_structured_root_call_sites(source_node))
        for root in roots:
            occurrence = root_name_counts.get(root.name, 0)
            root_name_counts[root.name] = occurrence + 1
            suffix = "" if occurrence == 0 else f"[{occurrence}]"
            visit(root, f"{root.name}{suffix}", 1)

    for node, path, depth in result:
        node.region_path = path
        node.control_depth = depth
        if depth > 1:
            node._mark_nested_portable()
    return tuple(node for node, _, _ in result)


def _reset_control_flow_reports(control_nodes):
    for node in control_nodes:
        node._last_report = None
        if hasattr(node, "_pending_report"):
            node._pending_report = None


@dataclass(frozen=True)
class _RecordingDispatch:
    kernel: object
    args: tuple
    dispatch_packet: object = None


def _normalize_recording_dispatch(dispatch):
    if isinstance(dispatch, _RecordingDispatch):
        return _RecordingDispatch(
            dispatch.kernel,
            tuple(dispatch.args),
            dispatch.dispatch_packet,
        )
    if len(dispatch) == 2:
        kernel, args = dispatch
        packet = None
    elif len(dispatch) == 3:
        kernel, args, packet = dispatch
    else:
        raise TaichiRuntimeError(
            "A Graph recording dispatch must contain kernel/args and an "
            "optional indirect packet"
        )
    return _RecordingDispatch(kernel, tuple(args), packet)


def _recordable_backend_dispatches(node, backend):
    if isinstance(node, _CompiledCGraphNode):
        if (
            node.dispatch_count == 0
            or len(node.recording_dispatches) != node.dispatch_count
        ):
            return None
        return node.recording_dispatches
    if not isinstance(node, _CompiledNativeGraphNode):
        return None
    recorder = node.recordable_action
    if recorder is None or not recorder.supports_backend(backend):
        return None
    dispatches = tuple(
        _normalize_recording_dispatch(dispatch) for dispatch in recorder.dispatches
    )
    return dispatches or None


def _recording_dispatch_ir_nodes(node, dispatch_count):
    ir_node = getattr(node, "ir_node", None)
    if isinstance(ir_node, SequentialRegion):
        children = tuple(ir_node.children)
        if len(children) == dispatch_count:
            return children
    if dispatch_count == 1 and isinstance(ir_node, DispatchNode):
        return (ir_node,)
    return (None,) * dispatch_count


def _record_backend_dispatch(builder, backend, dispatch, ir_node):
    dispatch = _normalize_recording_dispatch(dispatch)
    kernel = dispatch.kernel
    args = dispatch.args
    label = ir_node.dispatch_label if isinstance(ir_node, DispatchNode) else ""
    if dispatch.dispatch_packet is not None:
        builder.dispatch_indirect(
            kernel,
            args,
            dispatch.dispatch_packet,
            label,
        )
        return
    domain = ir_node.bounded_domain if isinstance(ir_node, DispatchNode) else None
    if domain is None:
        builder.dispatch(kernel, args, label)
        return
    requirement = domain.physical_grid_requirement
    if backend == "cuda" and requirement in (
        "logical_exact",
        "adaptive_grid",
        "require_exact",
    ):
        extent = next(
            (arg for arg in args if getattr(arg, "name", None) == domain.extent),
            None,
        )
        if extent is None:
            raise TaichiRuntimeError(
                "CUDA bounded Graph lowering lost its symbolic extent binding"
            )
        if domain.block_dim is None:
            raise TaichiRuntimeError(
                "CUDA bounded Graph lowering requires a block dimension"
            )
        builder.dispatch_cuda_bounded(
            kernel,
            args,
            extent,
            domain.capacity,
            domain.block_dim,
            requirement in ("adaptive_grid", "require_exact"),
            (
                requirement in ("adaptive_grid", "require_exact")
                and domain.update_policy == "grouped_stateful"
            ),
            label,
        )
        return
    if requirement != "require_exact":
        builder.dispatch(kernel, args, label)
        return
    extent = next(
        (arg for arg in args if getattr(arg, "name", None) == domain.extent),
        None,
    )
    if extent is None:
        raise TaichiRuntimeError(
            "Exact bounded Graph lowering lost its symbolic extent binding"
        )
    if backend == "cpu":
        builder.dispatch_cpu_bounded(
            kernel,
            args,
            extent,
            domain.capacity,
            label,
        )
        return
    raise TaichiRuntimeError(
        "Exact bounded Graph lowering cannot reconstruct the backend launch "
        f"recipe for {backend}"
    )


def _merge_fixed_runtime_args(nodes):
    merged = {}
    for node in nodes:
        for name, value in getattr(node, "fixed_runtime_args", {}).items():
            existing = merged.get(name)
            if existing is not None and existing is not value:
                if not (
                    isinstance(existing, (int, float))
                    and isinstance(value, (int, float))
                    and existing == value
                ):
                    raise TaichiRuntimeError(
                        "Recordable actions provide conflicting fixed binding "
                        f"{name!r}"
                    )
            merged[name] = value
    return merged


def _deduplicate_temporary_actions(actions):
    merged = []
    seen = set()
    for action in actions:
        identity = id(action)
        if identity in seen:
            continue
        seen.add(identity)
        merged.append(action)
    return tuple(merged)


def _merge_temporary_actions(nodes):
    return _deduplicate_temporary_actions(
        action for node in nodes for action in getattr(node, "temporary_actions", ())
    )


def _native_action_manifests_for_node(node):
    direct = getattr(node, "native_action_manifests", None)
    if direct is not None:
        manifests = tuple(direct)
    else:
        manifests = tuple(
            manifest
            for _, sequence in getattr(node, "_definition_regions", ())
            for manifest in getattr(sequence, "_native_action_manifests", ())
        )
    if not all(isinstance(item, NativeActionManifest) for item in manifests):
        raise TaichiRuntimeError(
            "Graph native action manifests must contain NativeActionManifest values"
        )
    return manifests


def _gpu_plan_logical_order(ir_node):
    if isinstance(ir_node, SequentialRegion):
        return tuple(
            kind
            for child in ir_node.children
            for kind in _gpu_plan_logical_order(child)
        )
    if isinstance(ir_node, DispatchNode):
        return ("dispatch",)
    if isinstance(ir_node, NativeCallNode):
        return ("native",)
    return ("structured",)


def _ir_contains_flag(node, flag):
    return bool(getattr(node, flag, False)) or any(
        _ir_contains_flag(child, flag) for child in getattr(node, "children", ())
    )


def _pipeline_task_manifests(node):
    if not isinstance(node, _CompiledCGraphNode):
        return (), ()
    frozen = getattr(node, "_frozen_pipeline_task_manifests", None)
    if frozen is not None:
        return frozen
    from taichi_forge.lang.task_manifest import GraphTaskManifest

    program = impl.get_runtime().prog
    raw = (
        program._graph_task_manifest(node.compiled_graph, include_external_commands=True)
        if node.source_native_count
        else program._graph_task_manifest(node.compiled_graph)
    )
    return (
        tuple(GraphTaskManifest._from_core(item) for item in raw if "external_command" not in item),
        tuple(dict(item) for item in raw if "external_command" in item),
    )


def _pipeline_mapping_status(node, external_commands=()):
    if isinstance(node, _CompiledCGraphNode):
        if external_commands:
            return "partial_external_commands", "partial_external_commands"
        return "available", "available"
    if _is_structured_control_node(node):
        return "structured_runtime_dependent", "structured_runtime_dependent"
    return "not_applicable", "not_applicable"


def _pipeline_physical_dispatch_map(tasks, logical_count):
    by_index = {}
    for task in tasks:
        by_index.setdefault(int(task.dispatch_index), []).append(task)
    mapping = []
    for physical_index in sorted(by_index):
        source_count = max(
            max(1, int(task.source_dispatch_count)) for task in by_index[physical_index]
        )
        mapping.extend((physical_index,) * source_count)
    if len(mapping) != logical_count:
        return (None,) * logical_count
    return tuple(mapping)


def _pipeline_bounded_dispatches(node, tasks, dispatch_count):
    return _pipeline_bounded_dispatches_with_publications(
        node, tasks, dispatch_count, {}
    )


def _effect_publishes_resource(effect):
    return effect.access in (
        GraphAccess.WRITE,
        GraphAccess.READ_WRITE,
        GraphAccess.ATOMIC,
        GraphAccess.OPAQUE,
    )


def _advance_publication_epochs(ir_node, publication_epochs, ignored_resources=()):
    ignored_resources = frozenset(ignored_resources)
    children = tuple(getattr(ir_node, "children", ()))
    if children:
        for child in children:
            _advance_publication_epochs(child, publication_epochs, ignored_resources)
        return
    for effect in getattr(ir_node, "effects", ()):
        if (
            effect.runtime_bound
            and effect.resource not in ignored_resources
            and _effect_publishes_resource(effect)
        ):
            publication_epochs[effect.resource] = (
                publication_epochs.get(effect.resource, 0) + 1
            )


def _pipeline_bounded_dispatches_with_publications(
    node, tasks, dispatch_count, publication_epochs
):
    ir_nodes = _recording_dispatch_ir_nodes(node, dispatch_count)
    physical_map = _pipeline_physical_dispatch_map(tasks, dispatch_count)
    label_indices = {}
    for task in tasks:
        if task.dispatch_label:
            label_indices.setdefault(task.dispatch_label, set()).add(
                int(task.dispatch_index)
            )
    result = []
    for logical_index, ir_node in enumerate(ir_nodes):
        if not isinstance(ir_node, DispatchNode):
            continue
        domain = ir_node.bounded_domain
        if domain is None:
            _advance_publication_epochs(ir_node, publication_epochs)
            continue
        publication_epoch = publication_epochs.get(domain.extent, 0)
        domain = replace(domain, publication_epoch=publication_epoch)
        physical_index = physical_map[logical_index]
        label_matches = label_indices.get(ir_node.dispatch_label, set())
        if ir_node.dispatch_label and len(label_matches) == 1:
            physical_index = next(iter(label_matches))
        snapshot_key = (
            domain.count_source,
            domain.extent,
            int(domain.capacity),
        )
        publication_key = (
            domain.count_source,
            domain.extent,
            int(domain.capacity),
            domain.block_dim,
            publication_epoch,
        )
        result.append(
            {
                "logical_dispatch_index": logical_index,
                "physical_dispatch_index": physical_index,
                "label": ir_node.dispatch_label,
                "domain": domain,
                "count_source": domain.count_source,
                "count_name": domain.extent,
                "capacity": int(domain.capacity),
                "snapshot_key": snapshot_key,
                "publication_key": publication_key,
            }
        )
        # A bounded payload is contractually a consumer of its extent.  Older
        # dispatch metadata conservatively labels every ndarray read_write;
        # allowing that fallback label to manufacture a new publication would
        # defeat reuse between adjacent consumers.
        _advance_publication_epochs(ir_node, publication_epochs, (domain.extent,))
    return tuple(result)


def _merge_derived_runtime_arg_names(nodes):
    return frozenset().union(
        *(getattr(node, "derived_runtime_arg_names", frozenset()) for node in nodes)
    )


def _runtime_node_physical_plan_id(node):
    parallel_groups = tuple(getattr(node, "parallel_dispatch_groups", ()))
    if parallel_groups:
        encoded = ";".join(
            ",".join(str(index) for index in group) for group in parallel_groups
        )
        return "cuda-fork-join:" + encoded
    controls = (
        (node,)
        if _is_structured_control_node(node)
        else tuple(getattr(node, "control_nodes", ()))
    )
    routes = []
    for control in controls:
        route = getattr(control, "_cuda_nested_control_lowering", None)
        if not route:
            route = getattr(control, "_cuda_control_lowering", None)
        if route:
            routes.append((control.region_path, route))
    if not routes:
        return ""
    return "structured-control:" + ";".join(f"{path}={route}" for path, route in routes)


def _graph_pipeline_definition(nodes):
    stages = []
    publication_epochs = {}
    for index, node in enumerate(nodes):
        manifests = _native_action_manifests_for_node(node)
        source_native_count = int(getattr(node, "source_native_count", 0))
        if source_native_count != len(manifests):
            raise TaichiRuntimeError(
                "Graph pipeline native count must match its action manifests"
            )
        ir_node = node.ir_node
        if isinstance(node, _CompiledWhileGraphNode):
            kind = "while"
        elif isinstance(node, _CompiledIfGraphNode):
            kind = "if"
        elif isinstance(node, _CompiledSwitchGraphNode):
            kind = "switch"
        elif isinstance(node, _CompiledCGraphNode):
            kind = "cgraph"
        elif isinstance(node, _CompiledNativeGraphNode):
            kind = "native"
        elif isinstance(node, _CompiledObservationGraphNode):
            kind = "observation"
        else:
            kind = str(getattr(ir_node, "kind", type(node).__name__))
        path_id = (
            str(node.region_path)
            if _is_structured_control_node(node)
            else f"root/{index}"
        )
        name = str(getattr(node, "name", getattr(ir_node, "name", f"stage_{index}")))
        dispatch_count = int(getattr(node, "dispatch_count", 0))
        tasks, external_commands = _pipeline_task_manifests(node)
        task_mapping_status, bounded_mapping_status = _pipeline_mapping_status(node, external_commands)
        bounded_dispatches = _pipeline_bounded_dispatches_with_publications(
            node, tasks, dispatch_count, publication_epochs
        )
        if dispatch_count == 0:
            _advance_publication_epochs(ir_node, publication_epochs)
        stages.append(
            {
                "stage_index": index,
                "path_id": path_id,
                "name": name,
                "kind": kind,
                "region_kind": str(getattr(node, "region_kind", kind)),
                "physical_plan_id": _runtime_node_physical_plan_id(node),
                "parallel_dispatch_groups": tuple(
                    getattr(node, "parallel_dispatch_groups", ())
                ),
                "dispatch_count": dispatch_count,
                "physical_dispatch_count": int(
                    getattr(node, "physical_dispatch_count", dispatch_count)
                ),
                "runtime_arg_names": tuple(
                    sorted(getattr(node, "runtime_arg_names", ()))
                ),
                "source_native_count": source_native_count,
                "native_actions": manifests,
                "task_mapping_status": task_mapping_status,
                "bounded_mapping_status": bounded_mapping_status,
                "tasks": tasks,
                "external_commands": external_commands,
                "bounded_dispatches": bounded_dispatches,
                "synchronization": _ir_contains_flag(ir_node, "synchronization"),
                "opaque": _ir_contains_flag(ir_node, "opaque"),
            }
        )
    return tuple(stages)


def _lower_mixed_backend_regions(nodes):
    backend = _backend_name(_ti_core.arch_name(impl.current_cfg().arch))
    nodes = tuple(nodes)
    lowered = []
    mixed_region_count = 0
    lowered_native_count = 0
    cursor = 0
    while cursor < len(nodes):
        if _recordable_backend_dispatches(nodes[cursor], backend) is None:
            lowered.append(nodes[cursor])
            cursor += 1
            continue

        end = cursor
        region = []
        while end < len(nodes):
            dispatches = _recordable_backend_dispatches(nodes[end], backend)
            if dispatches is None:
                break
            region.append((nodes[end], dispatches))
            end += 1

        has_cgraph = any(isinstance(node, _CompiledCGraphNode) for node, _ in region)
        has_native = any(
            isinstance(node, _CompiledNativeGraphNode) for node, _ in region
        )
        if not has_native:
            lowered.extend(node for node, _ in region)
            cursor = end
            continue

        builder = _new_runtime_graph_builder()
        recording_dispatches = []
        runtime_arg_names = set()
        fixed_runtime_args = _merge_fixed_runtime_args(node for node, _ in region)
        ir_children = []
        lifetime_leases = []
        region_native_count = 0
        native_action_manifests = []
        temporary_actions = []
        for node, dispatches in region:
            dispatch_ir_nodes = _recording_dispatch_ir_nodes(node, len(dispatches))
            for dispatch, dispatch_ir_node in zip(dispatches, dispatch_ir_nodes):
                _record_backend_dispatch(
                    builder,
                    backend,
                    dispatch,
                    dispatch_ir_node,
                )
                recording_dispatches.append(dispatch)
            runtime_arg_names.update(
                getattr(
                    node,
                    "recording_runtime_arg_names",
                    node.runtime_arg_names,
                )
            )
            if isinstance(node.ir_node, SequentialRegion):
                ir_children.extend(node.ir_node.children)
            else:
                ir_children.append(node.ir_node)
            lifetime_leases.extend(getattr(node, "lifetime_leases", ()))
            temporary_actions.extend(getattr(node, "temporary_actions", ()))
            region_native_count += getattr(node, "source_native_count", 0)
            native_action_manifests.extend(_native_action_manifests_for_node(node))

        lowered.append(
            _CompiledCGraphNode(
                builder.compile(),
                len(recording_dispatches),
                runtime_arg_names,
                SequentialRegion(
                    tuple(ir_children),
                    name=(
                        "mixed_backend_region"
                        if has_cgraph
                        else "recordable_provider_region"
                    ),
                ),
                recording_dispatches=recording_dispatches,
                lifetime_leases=lifetime_leases,
                source_native_count=region_native_count,
                region_kind=(
                    "mixed_cgraph_native" if has_cgraph else "recordable_provider"
                ),
                fixed_runtime_args=fixed_runtime_args,
                temporary_actions=_deduplicate_temporary_actions(temporary_actions),
                native_action_manifests=native_action_manifests,
            )
        )
        mixed_region_count += 1
        lowered_native_count += region_native_count
        cursor = end

    backend_command_nodes = sum(
        isinstance(node, _CompiledNativeGraphNode)
        and node.recordable_action is not None
        and node.recordable_action.backend_command_recording is not None
        for node in nodes
    )
    opaque_native_nodes = sum(
        isinstance(node, _CompiledNativeGraphNode) and node.recordable_action is None
        for node in nodes
    )
    statistics = {
        "backend": backend,
        "input_segments": len(nodes),
        "output_segments": len(lowered),
        "mixed_backend_regions": mixed_region_count,
        "lowered_native_nodes": lowered_native_count,
        "opaque_native_nodes": opaque_native_nodes,
    }
    if backend_command_nodes:
        statistics["backend_command_nodes"] = backend_command_nodes
    return tuple(lowered), statistics


def _structured_control_roles(node):
    if isinstance(node, _CompiledWhileGraphNode):
        return tuple(
            (role, name)
            for role, name in (
                ("predicate", node.predicate),
                ("counter", node.counter),
                ("status", node.status),
            )
            if name is not None
        )
    if isinstance(node, _CompiledIfGraphNode):
        return (("predicate", node.predicate),)
    if isinstance(node, _CompiledSwitchGraphNode):
        return (("selector", node.selector),)
    return ()


@dataclass(frozen=True)
class _StructuredControlBindingPlan:
    """Static control slots and the allocation identities that must differ."""

    bindings: tuple
    distinct_groups: tuple
    names: tuple


def _structured_control_binding_plan(control_nodes):
    bindings = tuple(
        (node.region_path, role, name)
        for node in control_nodes
        for role, name in _structured_control_roles(node)
    )
    root_subtrees = []
    current = []
    for node in control_nodes:
        if node.control_depth == 1:
            if current:
                root_subtrees.append(tuple(current))
            current = [node]
        else:
            current.append(node)
    if current:
        root_subtrees.append(tuple(current))

    distinct_groups = []
    for subtree in root_subtrees:
        root = subtree[0]
        if len(subtree) > 1:
            group_bindings = tuple(
                (node.region_path, role, name)
                for node in subtree
                for role, name in _structured_control_roles(node)
            )
            scope = f"nested root {root.region_path!r}"
        elif isinstance(root, _CompiledWhileGraphNode):
            group_bindings = tuple(
                (root.region_path, role, name)
                for role, name in _structured_control_roles(root)
            )
            scope = f"while region {root.region_path!r}"
        else:
            continue

        symbolic_owners = {}
        for path, role, name in group_bindings:
            previous = symbolic_owners.get(name)
            if previous is not None:
                previous_path, previous_role = previous
                raise TaichiRuntimeError(
                    "Structured Graph control resources must be independent "
                    f"within {scope}; {name!r} is used by "
                    f"{previous_path!r} {previous_role} and {path!r} {role}"
                )
            symbolic_owners[name] = (path, role)
        if len(group_bindings) > 1:
            distinct_groups.append(
                (
                    scope,
                    tuple((path, role, name) for path, role, name in group_bindings),
                )
            )

    return _StructuredControlBindingPlan(
        bindings=bindings,
        distinct_groups=tuple(distinct_groups),
        names=tuple(sorted({name for _, _, name in bindings})),
    )


def _normalize_parallel_branch_indices(branches, child_count):
    try:
        normalized = tuple(tuple(branch) for branch in branches)
    except TypeError as exc:
        raise TaichiRuntimeError(
            "parallel candidate branches must be an iterable of index iterables"
        ) from exc
    if not 2 <= len(normalized) <= 4:
        raise TaichiRuntimeError(
            "parallel candidate analysis requires between 2 and 4 branches"
        )
    flat = []
    for branch_index, indices in enumerate(normalized):
        if not indices:
            raise TaichiRuntimeError(
                f"parallel candidate branch {branch_index} must not be empty"
            )
        if not all(
            isinstance(index, int) and not isinstance(index, bool) for index in indices
        ):
            raise TaichiRuntimeError("parallel candidate node indices must be integers")
        if (
            tuple(sorted(indices)) != indices
            or tuple(range(indices[0], indices[-1] + 1)) != indices
        ):
            raise TaichiRuntimeError(
                f"parallel candidate branch {branch_index} must select one "
                "ordered contiguous node range"
            )
        if indices[0] < 0 or indices[-1] >= child_count:
            raise TaichiRuntimeError(
                f"parallel candidate branch {branch_index} selects a node "
                "outside the Graph root"
            )
        flat.extend(indices)
    if len(flat) != len(set(flat)):
        raise TaichiRuntimeError(
            "parallel candidate branches must not select the same node twice"
        )
    if tuple(flat) != tuple(range(flat[0], flat[-1] + 1)):
        raise TaichiRuntimeError(
            "parallel candidate branches must form one ordered contiguous root range"
        )
    return normalized


def _parallel_logical_root_children(root):
    children = []
    for child in root.children:
        if isinstance(child, SequentialRegion):
            children.extend(_parallel_logical_root_children(child))
        else:
            children.append(child)
    return tuple(children)


@dataclass(frozen=True)
class _GraphBranchJoinRecipeSource:
    """One compiler-proven coarse CUDA fork/join candidate."""

    source_key: str
    node_index: int
    branch_groups: tuple
    join_index: int
    disjoint_pairs: tuple
    sequential_temporary_bytes: int
    parallel_temporary_bytes: int

    @property
    def recipe_id(self):
        groups = ";".join(
            ",".join(str(index) for index in group) for group in self.branch_groups
        )
        return f"cuda-fork-join-v1:{self.node_index}:{groups}:{self.join_index}"


@dataclass(frozen=True)
class _GraphRecordingPartitionRecipeSource:
    """One bounded CUDA recording cut at a public binding frontier."""

    source_key: str
    node_index: int
    cut_index: int
    dispatch_count: int
    isolated_bindings: tuple
    stable_dispatch_count: int

    @property
    def recipe_id(self):
        return (
            "cuda-binding-frontier-v1:"
            f"{self.node_index}:{self.cut_index}:{self.dispatch_count}"
        )


def _effect_reads(access):
    return access in (
        GraphAccess.READ,
        GraphAccess.READ_WRITE,
        GraphAccess.ATOMIC,
    )


def _effect_writes(access):
    return access in (
        GraphAccess.WRITE,
        GraphAccess.READ_WRITE,
        GraphAccess.ATOMIC,
        GraphAccess.OPAQUE,
    )


def _branch_has_internal_dependency(nodes):
    for left_index, left in enumerate(nodes):
        for right in nodes[left_index + 1 :]:
            for left_effect in left.effects:
                if not _effect_writes(left_effect.access):
                    continue
                for right_effect in right.effects:
                    if left_effect.resource == right_effect.resource and (
                        _effect_reads(right_effect.access)
                        or _effect_writes(right_effect.access)
                    ):
                        return True
    return False


def _join_consumes_each_branch(branches, join):
    join_reads = {
        effect.resource for effect in join.effects if _effect_reads(effect.access)
    }
    if not join_reads:
        return False
    for branch in branches:
        written = {
            effect.resource
            for node in branch
            for effect in node.effects
            if _effect_writes(effect.access)
        }
        if not written.intersection(join_reads):
            return False
    return True


def _discover_cuda_branch_join_sources(source_nodes, backend):
    """Find exact two-chain/final-join shapes without timing heuristics."""

    if backend != "cuda" or len(source_nodes) != 1:
        return ()
    node = source_nodes[0]
    if not isinstance(node, _CompiledCGraphNode):
        return ()
    children = tuple(node.ir_node.children)
    operations = tuple(node.recipe_operations)
    if (
        len(children) < 5
        or len(operations) != len(children)
        or any(operation[0] != "dispatch" for operation in operations)
        or any(operation[3] for operation in operations)
        or any(child.bounded_domain is not None for child in children)
    ):
        return ()
    join_index = len(children) - 1
    join = children[join_index]
    if (
        not isinstance(join, DispatchNode)
        or join.opaque
        or join.synchronization
        or join.side_effects
    ):
        return ()

    sources = []
    for split in range(2, join_index - 1):
        branches = (children[:split], children[split:join_index])
        if any(len(branch) < 2 for branch in branches):
            continue
        if not all(_branch_has_internal_dependency(branch) for branch in branches):
            continue
        if not _join_consumes_each_branch(branches, join):
            continue
        plan = analyze_parallel_candidate(
            tuple(
                SequentialRegion(
                    tuple(branch),
                    name=f"branch_join_candidate_{branch_index}",
                )
                for branch_index, branch in enumerate(branches)
            )
        )
        if plan.blockers or plan.conflicts:
            continue
        disjoint_pairs = tuple(
            sorted(
                {
                    tuple(sorted((item.left_resource, item.right_resource)))
                    for item in plan.unresolved_aliases
                }
            )
        )
        sources.append(
            _GraphBranchJoinRecipeSource(
                source_key=f"branch-join:{split}:{join_index}",
                node_index=0,
                branch_groups=(
                    tuple(range(0, split)),
                    tuple(range(split, join_index)),
                ),
                join_index=join_index,
                disjoint_pairs=disjoint_pairs,
                sequential_temporary_bytes=(plan.sequential_fallback_peak_bytes),
                parallel_temporary_bytes=plan.parallel_peak_bytes,
            )
        )
    return tuple(sources)


def _discover_cuda_recording_partition_sources(source_nodes, backend):
    """Find bounded cuts that can isolate binding churn from a stable region.

    This is definition-time topology analysis only.  It does not guess which
    binding will change at runtime and it never exposes a raw cut index as a
    public optimization option.  Complete recipes carry the physical cut as
    an opaque Forge-owned choice for workload evaluation.
    """

    if backend != "cuda" or len(source_nodes) != 1:
        return ()
    node = source_nodes[0]
    children = tuple(getattr(node.ir_node, "children", ()))
    operations = tuple(getattr(node, "recipe_operations", ()))
    dispatches = tuple(getattr(node, "recording_dispatches", ()))
    if (
        not isinstance(node, _CompiledCGraphNode)
        or len(children) < 8
        or len(children) != len(operations)
        or len(children) != len(dispatches)
        or node.source_native_count
        or node.temporary_actions
        or node.parallel_dispatch_groups
        or any(not isinstance(child, DispatchNode) for child in children)
        or any(child.bounded_domain is not None for child in children)
        or any(operation[0] != "dispatch" for operation in operations)
    ):
        return ()

    # DispatchNode bindings preserve kernel-parameter roles (for example many
    # kernels may each have a local parameter called ``source``).  Recording
    # identity is keyed by the enclosing Graph's symbolic arguments instead,
    # so derive frontiers from the frozen operation arguments.
    public_bindings = frozenset(node.runtime_arg_names)
    binding_sets = tuple(
        frozenset(_runtime_arg_names(operation[2])).intersection(public_bindings)
        for operation in operations
    )
    prefix = []
    accumulated = frozenset()
    for bindings in binding_sets:
        accumulated = accumulated.union(bindings)
        prefix.append(accumulated)
    suffix = [frozenset()] * len(binding_sets)
    accumulated = frozenset()
    for index in range(len(binding_sets) - 1, -1, -1):
        accumulated = accumulated.union(binding_sets[index])
        suffix[index] = accumulated

    ranked = []
    for cut_index in range(1, len(children)):
        left_only = prefix[cut_index - 1].difference(suffix[cut_index])
        right_only = suffix[cut_index].difference(prefix[cut_index - 1])
        stable_dispatch_count = max(
            len(children) - cut_index if left_only else 0,
            cut_index if right_only else 0,
        )
        if stable_dispatch_count < 4:
            continue
        isolated_bindings = tuple(sorted(left_only.union(right_only)))
        if not isolated_bindings:
            continue
        ranked.append(
            (
                -stable_dispatch_count,
                min(cut_index, len(children) - cut_index),
                cut_index,
                isolated_bindings,
            )
        )

    # Four physical alternatives are enough to represent both edge frontiers
    # and the strongest interior frontiers without creating a raw partition
    # combinatorics problem for the complete-recipe catalog.
    result = []
    for _, _, cut_index, isolated_bindings in sorted(ranked)[:4]:
        stable_dispatch_count = max(
            (
                len(children) - cut_index
                if prefix[cut_index - 1].difference(suffix[cut_index])
                else 0
            ),
            cut_index if suffix[cut_index].difference(prefix[cut_index - 1]) else 0,
        )
        result.append(
            _GraphRecordingPartitionRecipeSource(
                source_key=f"recording-partition:0:{cut_index}",
                node_index=0,
                cut_index=cut_index,
                dispatch_count=len(children),
                isolated_bindings=isolated_bindings,
                stable_dispatch_count=stable_dispatch_count,
            )
        )
    return tuple(result)


def _graph_memory_disjoint_pairs(root):
    pairs = set()

    def visit(node):
        if isinstance(node, DispatchNode):
            pairs.update(tuple(pair) for pair in node.memory_disjoint_pairs)
        for child in node.children:
            visit(child)

    visit(root)
    return tuple(sorted(pairs))


def _graph_memory_layout_requirements(root):
    requirements = set()

    def visit(node):
        if isinstance(node, DispatchNode):
            requirements.update(tuple(item) for item in node.memory_layout_requirements)
        for child in node.children:
            visit(child)

    visit(root)
    return tuple(sorted(requirements))


def _graph_memory_semantic_root(root):
    """Remove only physical binding requirements from the first memory slice."""

    dispatch_count = 0

    def strip(node):
        nonlocal dispatch_count
        if isinstance(node, DispatchNode):
            dispatch_count += 1
            return replace(
                node,
                memory_disjoint_pairs=(),
                memory_layout_requirements=(),
            )
        if isinstance(node, SequentialRegion):
            return replace(
                node, children=tuple(strip(child) for child in node.children)
            )
        raise ValueError(
            "the initial GraphMemory recipe accepts sequential dispatch regions only"
        )

    semantic = strip(root)
    if dispatch_count != 1:
        raise ValueError("the initial GraphMemory recipe requires one logical dispatch")
    return semantic


def _graph_bounded_semantic_root(root):
    """Strip physical bounded choices while retaining the semantic domain."""

    def strip(node):
        if isinstance(node, DispatchNode):
            domain = node.bounded_domain
            if domain is None:
                return node
            return replace(
                node,
                logical_kernel_identity=domain.semantic_kernel_identity,
                bounded_domain=replace(
                    domain,
                    physical_grid_requirement="auto",
                    update_policy="",
                ),
            )
        if isinstance(node, SequentialRegion):
            return replace(
                node,
                children=tuple(strip(child) for child in node.children),
            )
        raise ValueError(
            "the initial GraphBounded recipe accepts sequential dispatch regions only"
        )

    return strip(root)


def _graph_definition_semantic_root(root):
    """Remove recipe-owned physical choices from a frozen semantic Graph.

    This normalization is definition-time only. Runtime binding generations,
    resource leases, and replay state remain owned by the compiled Graph.
    """

    def strip(node):
        if isinstance(node, DispatchNode):
            bounded = node.bounded_domain
            logical_kernel_identity = node.logical_kernel_identity
            if bounded is not None:
                logical_kernel_identity = (
                    bounded.semantic_kernel_identity or logical_kernel_identity
                )
                bounded = replace(
                    bounded,
                    block_dim=None,
                    block_mode="auto",
                    physical_grid_policy="auto",
                    physical_grid_requirement="auto",
                    update_policy="",
                    publication_epoch=None,
                )
            name = node.dispatch_label
            if not name:
                name = "dispatch" if logical_kernel_identity else node.name
            return replace(
                node,
                name=name,
                bounded_domain=bounded,
                logical_kernel_identity=logical_kernel_identity,
                fusion_blocker="",
                memory_disjoint_pairs=(),
                memory_layout_requirements=(),
            )
        if isinstance(node, SequentialRegion):
            return replace(
                node,
                children=tuple(strip(child) for child in node.children),
            )
        if isinstance(node, WhileRegion):
            return replace(
                node,
                condition=strip(node.condition),
                body=strip(node.body),
                chunk_size=1,
                compound_chunk_size=1,
                vulkan_first_chunk_strategy="auto",
                masked_execution=False,
                lowering_mode="auto",
            )
        if isinstance(node, IfRegion):
            return replace(
                node,
                condition=strip(node.condition),
                then_region=strip(node.then_region),
                else_region=(
                    None if node.else_region is None else strip(node.else_region)
                ),
            )
        if isinstance(node, SwitchRegion):
            return replace(
                node,
                condition=strip(node.condition),
                branches=tuple(strip(branch) for branch in node.branches),
                default_region=(
                    None if node.default_region is None else strip(node.default_region)
                ),
            )
        return node

    return strip(root)


def _graph_bounded_recipe_scope(pipeline):
    bounded = tuple(
        dispatch for stage in pipeline for dispatch in stage["bounded_dispatches"]
    )
    if not bounded:
        return (), "", "no_bounded_dispatch"
    domains = tuple(dispatch["domain"] for dispatch in bounded)
    if any(
        domain.count_source != "device_extent"
        or domain.ordered
        or domain.physical_grid_policy != "auto"
        or not domain.semantic_kernel_identity
        for domain in domains
    ):
        return (), "", "source_policy_out_of_scope"

    requirements = {domain.physical_grid_requirement for domain in domains}
    update_policies = {domain.update_policy for domain in domains}
    if requirements == {"logical_exact"} and update_policies == {""}:
        selected_strategy = "logical_exact"
    elif requirements == {"adaptive_grid"} and update_policies == {"per_node"}:
        selected_strategy = "adaptive_per_node"
    elif requirements == {"adaptive_grid"} and update_policies == {"grouped_stateful"}:
        selected_strategy = "adaptive_grouped"
    elif requirements == {"auto"} and update_policies == {""}:
        selected_strategy = "masked_capacity"
    else:
        return (), "", "mixed_or_unknown_materialization"

    grouped = {}
    for dispatch in bounded:
        key = dispatch["publication_key"]
        item = grouped.setdefault(
            key,
            {
                "count_name": dispatch["count_name"],
                "capacity": dispatch["capacity"],
                "block_dim": dispatch["domain"].block_dim,
                "publication_epoch": dispatch["domain"].publication_epoch,
                "consumer_count": 0,
            },
        )
        item["consumer_count"] += 1
    groups = tuple(grouped.values())
    strategies = ["logical_exact"]
    if _cuda_nested_device_update_available():
        strategies.append("adaptive_per_node")
        if any(group["consumer_count"] >= 2 for group in groups):
            strategies.append("adaptive_grouped")
    strategies.append("masked_capacity")
    manifests = tuple(
        _GraphBoundedExecutionRecipeManifest.from_payload(
            {
                "strategy": strategy,
                "source_physical_grid_policy": "auto",
                "bounded_dispatch_count": len(bounded),
                "publication_groups": groups,
            }
        )
        for strategy in strategies
    )
    selected = next(
        (
            manifest.recipe_id
            for manifest in manifests
            if manifest.strategy == selected_strategy
        ),
        "",
    )
    if not selected:
        return (), "", "materialization_outside_recipe_domain"
    return manifests, selected, "complete_recipe_domain"


def _parallel_identity_tuple(value):
    if value is None:
        return None
    result = []
    for item in tuple(value):
        if isinstance(item, (bool, int, str)):
            result.append(item)
        else:
            result.append(str(item))
    return tuple(result)


def _unsupported_parallel_storage_fact(resource, failure_reason, owner_status):
    return ParallelStorageFact(
        resource=resource,
        supported=False,
        owner_status=owner_status,
        failure_reason=failure_reason,
        source_kind="",
        owner_kind="",
        program_domain=None,
        resource_identity=None,
        tree_identity=None,
        byte_offset=None,
        byte_begin=None,
        byte_end=None,
        compact_contiguous=None,
        index_shape=(),
        index_strides_bytes=(),
        element_shape=(),
        scalar_count=None,
        record_stride=None,
    )


def _parallel_storage_fact(resource, description, owner_status):
    descriptor = description.descriptor
    if descriptor is None:
        return _unsupported_parallel_storage_fact(
            resource,
            description.failure_reason,
            description.failure_reason,
        )
    properties = description.properties
    return ParallelStorageFact(
        resource=resource,
        supported=True,
        owner_status=owner_status,
        failure_reason="kNone",
        source_kind=str(descriptor.source_kind),
        owner_kind=str(descriptor.owner_kind),
        program_domain=int(descriptor.program_domain),
        resource_identity=_parallel_identity_tuple(descriptor.resource_identity),
        tree_identity=_parallel_identity_tuple(descriptor.tree_identity),
        byte_offset=int(descriptor.byte_offset),
        byte_begin=int(properties.get("reachable_begin", descriptor.byte_offset)),
        byte_end=int(properties.get("reachable_end", descriptor.byte_offset)),
        compact_contiguous=bool(properties.get("compact_contiguous", False)),
        index_shape=tuple(int(value) for value in descriptor.index_shape),
        index_strides_bytes=tuple(
            int(value) for value in descriptor.index_strides_bytes
        ),
        element_shape=tuple(int(value) for value in descriptor.element_shape),
        scalar_count=int(properties.get("scalar_count", 0)),
        record_stride=int(properties.get("record_stride", 0)),
    )


def _describe_parallel_storage(value):
    if isinstance(value, ProviderOwnedNdarrayBinding):
        return StorageDescription(
            _ti_core._describe_ndarray_storage(value.arr, "readwrite")
        )
    return describe_storage(value, access="readwrite")


def _parallel_storage_description(resource, value):
    try:
        description = _describe_parallel_storage(value)
    except Exception as exc:
        return (
            _unsupported_parallel_storage_fact(
                resource,
                f"{type(exc).__name__}:{exc}",
                "kUnknown",
            ),
            None,
        )
    try:
        owner_status = validate_storage_owner(description)
    except Exception as exc:
        owner_status = f"{type(exc).__name__}:{exc}"
    return _parallel_storage_fact(resource, description, owner_status), description


def _graph_public_resource_accesses(root, public_names):
    public_names = frozenset(public_names)
    accesses = {}

    def visit(node):
        for effect in getattr(node, "effects", ()):
            if not effect.runtime_bound or effect.resource not in public_names:
                continue
            reads, writes = accesses.get(effect.resource, (False, False))
            reads = reads or effect.access in (
                GraphAccess.READ,
                GraphAccess.READ_WRITE,
                GraphAccess.ATOMIC,
                GraphAccess.OPAQUE,
            )
            writes = writes or effect.access in (
                GraphAccess.WRITE,
                GraphAccess.READ_WRITE,
                GraphAccess.ATOMIC,
                GraphAccess.OPAQUE,
            )
            accesses[effect.resource] = (reads, writes)
        for child in getattr(node, "children", ()):
            visit(child)

    visit(root)
    return MappingProxyType(accesses)


def _certify_graph_invocation_pair(spec, versions):
    accesses = _graph_public_resource_accesses(
        spec.pre_optimization_ir_root,
        spec.runtime_arg_names,
    )
    descriptions = [{}, {}]

    def storage(invocation, resource):
        cached = descriptions[invocation].get(resource)
        if cached is not None:
            return cached
        value = versions[invocation].arguments[resource]
        if isinstance(
            value,
            (int, float, np.integer, np.floating, Matrix),
        ) and not isinstance(value, (bool, np.bool_)):
            descriptions[invocation][resource] = False
            return False
        fact, description = _parallel_storage_description(resource, value)
        if description is None or not fact.supported or fact.owner_status != "kNone":
            raise TaichiRuntimeError(
                "Graph.bind_batch() requires certified device storage for "
                f"{resource!r} in invocation {invocation}"
            )
        descriptions[invocation][resource] = description
        return description

    disjoint_pairs = []
    for left, (_, left_writes) in accesses.items():
        for right, (_, right_writes) in accesses.items():
            if not left_writes and not right_writes:
                continue
            left_storage = storage(0, left)
            right_storage = storage(1, right)
            if left_storage is False or right_storage is False:
                continue
            try:
                alias = analyze_storage_alias(left_storage, right_storage)
            except Exception:
                alias = "kUnknown"
            if alias != "kProvenDisjoint":
                raise TaichiRuntimeError(
                    "Graph.bind_batch() requires proven disjoint cross-invocation "
                    f"storage for {left!r} and {right!r}; got {alias}"
                )
            disjoint_pairs.append((left, right))
    return tuple(disjoint_pairs)


def _workspace_concurrency_spec_eligible(spec, backend):
    if backend != "cuda":
        return False
    if (
        spec.native_count
        or spec.structured_control_count
        or spec.observation_count
        or spec.temporary_actions
        or spec.lifetime_leases
        or spec.exclusive_provider_submission
        or len(spec.nodes) != 1
    ):
        return False
    node = spec.nodes[0]
    if (
        not isinstance(node, _CompiledCGraphNode)
        or node.dispatch_count < 2
        or node.source_native_count
        or node.parallel_dispatch_groups
        or any(item.dispatch_packet is not None for item in node.recording_dispatches)
        or bool(getattr(node.compiled_graph, "_has_dispatch_labels", False))
    ):
        return False
    if not spec.fixed_runtime_args or not all(
        isinstance(value, _GraphInternalNdarraySpec) and value.exclusive_submission
        for value in spec.fixed_runtime_args.values()
    ):
        return False
    return bool(
        _graph_public_resource_accesses(
            spec.pre_optimization_ir_root,
            spec.runtime_arg_names,
        )
    )


@dataclass(frozen=True)
class _PreparedGraphInvocation:
    arguments: object
    submission_owners: tuple
    flattened_args: object = None
    binding_version: object = None


@dataclass(frozen=True)
class _GraphBindingPlan:
    """Immutable Python slot plan compiled with one Graph definition."""

    public_names: tuple
    public_name_set: frozenset
    slot_by_name: object
    fixed_names: tuple
    derived_names: tuple
    temporary_names: tuple
    control_names: tuple
    control_publish_frame_stable: bool
    memory_recipe_names: tuple
    memory_recipe_publish_frame_stable: bool
    static_fast_path_blockers: tuple

    def to_dict(self):
        return MappingProxyType(
            {
                "slot_order": self.public_names,
                "slot_by_name": self.slot_by_name,
                "fixed_names": self.fixed_names,
                "derived_names": self.derived_names,
                "temporary_names": self.temporary_names,
                "control_names": self.control_names,
                "control_publish_certificate_required": bool(self.control_names),
                "control_publish_frame_stable": self.control_publish_frame_stable,
                "memory_recipe_names": self.memory_recipe_names,
                "memory_recipe_publish_certificate_required": bool(
                    self.memory_recipe_names
                ),
                "memory_recipe_publish_frame_stable": (
                    self.memory_recipe_publish_frame_stable
                ),
                "static_fast_path_qualified": not self.static_fast_path_blockers,
                "static_fast_path_blockers": self.static_fast_path_blockers,
            }
        )


@dataclass(frozen=True)
class _GraphMemoryRecipeCertificate:
    """Publish-time proof for one immutable set of storage bindings."""

    runtime_generation: int
    bindings: tuple
    layout_requirements: tuple
    disjoint_pairs: tuple
    runtime_bindings: tuple = ()


@dataclass(frozen=True)
class _GraphControlBindingCertificate:
    """Publish-time proof for immutable structured-control resources."""

    runtime_generation: int
    bindings: tuple
    distinct_groups: tuple


@dataclass(frozen=True, eq=False)
class _GraphBindingVersion:
    """One immutable, generation-qualified public binding snapshot."""

    revision: int
    runtime_generation: int
    slot_values: tuple
    arguments: object
    execution_arguments: object
    flattened_args: object
    memory_recipe_certificate: object
    control_binding_certificate: object
    fixed_bindings_flattened: bool
    fast_path_qualified: bool
    volatile_reasons: tuple
    execution_frame: object = None


@dataclass(frozen=True)
class GraphInvocationBatch:
    """Two immutable binding frames for one complete Graph pair workload.

    Construct through :meth:`Graph.bind_batch`.  The object pins exact
    publication-time storage identities and is accepted only by the Graph
    that created it.  Cross-invocation write hazards are discharged once at
    publication, never in the replay path.
    """

    _graph: object
    runtime_generation: int
    versions: tuple
    disjoint_pairs: tuple

    @property
    def size(self):
        return len(self.versions)


def _snapshot_graph_binding_value(value):
    # Scalars and matrices are invocation values, not resource identities.
    # Snapshot them at publish time so mutating a caller-owned Matrix cannot
    # silently change an already published BindingVersion. Device allocations
    # remain shared intentionally: their contents are dynamic kernel inputs.
    if isinstance(value, Matrix) and not value.is_host_access:
        return Matrix(np.array(value.entries, copy=True))
    return value


class GraphBindingSet:
    """Versioned runtime arguments for repeat Graph invocation.

    Construct through :meth:`Graph.bind`. ``update()`` atomically publishes a
    copy-on-write version. Python scalars and matrices are snapshotted by value;
    device resource objects are retained by identity, so their device contents
    remain dynamic. A Graph only takes the version snapshot after pacer waits.
    """

    def __init__(self, graph, arguments):
        self._graph = graph
        self._lock = threading.RLock()
        self._retired_versions = {}
        self._version = None
        graph._initialize_binding_set(self, arguments)

    @property
    def revision(self):
        with self._lock:
            return self._version.revision

    @property
    def fast_path_qualified(self):
        with self._lock:
            return self._version.fast_path_qualified

    def snapshot(self):
        """Return the current immutable public argument mapping."""

        with self._lock:
            return MappingProxyType(
                {
                    name: _snapshot_graph_binding_value(value)
                    for name, value in self._version.arguments.items()
                }
            )

    def update(self, values=None, /, **changes):
        """Atomically publish a partial copy-on-write binding update."""

        patch = {}
        if values is not None:
            if not isinstance(values, Mapping):
                raise TypeError("GraphBindingSet.update() values must be a mapping")
            patch.update(values)
        overlap = patch.keys() & changes.keys()
        if overlap:
            raise TypeError(
                "GraphBindingSet.update() received duplicate bindings: "
                + ", ".join(sorted(overlap))
            )
        patch.update(changes)
        if not patch:
            return self
        self._graph._update_binding_set(self, patch, replace_all=False)
        return self

    def replace(self, arguments):
        """Atomically replace every public binding."""

        self._graph._update_binding_set(self, arguments, replace_all=True)
        return self

    def statistics(self):
        with self._lock:
            version = self._version
            memory_recipe_certificate = version.memory_recipe_certificate
            control_binding_certificate = version.control_binding_certificate
            return MappingProxyType(
                {
                    "revision": version.revision,
                    "slot_count": len(version.slot_values),
                    "fast_path_qualified": version.fast_path_qualified,
                    "fixed_bindings_flattened": (version.fixed_bindings_flattened),
                    "volatile_reasons": version.volatile_reasons,
                    "memory_recipe_publish_validated": (
                        memory_recipe_certificate is not None
                    ),
                    "memory_recipe_certified": bool(
                        memory_recipe_certificate is not None
                        and memory_recipe_certificate.runtime_bindings
                    ),
                    "memory_recipe_names": (
                        self._graph._spec.memory_recipe_binding_names
                    ),
                    "control_publish_validated": (
                        control_binding_certificate is not None
                    ),
                    "control_names": self._graph._spec.control_binding_plan.names,
                    "live_retired_versions": len(self._retired_versions),
                }
            )


def _cuda_structured_control_recipe_domain(source_nodes, control_nodes, backend):
    """Describe one exact CUDA control domain without changing Graph IR."""

    source_nodes = tuple(source_nodes)
    control_nodes = tuple(control_nodes)
    if backend != "cuda":
        return (), ""
    if any(
        isinstance(source, (_CompiledNativeGraphNode, _CompiledObservationGraphNode))
        or getattr(source, "source_native_count", 0)
        for source in source_nodes
    ):
        return (), ""

    if len(control_nodes) == 1:
        node = control_nodes[0]
        if (
            not isinstance(node, _CompiledWhileGraphNode)
            or node.control_depth != 1
            or node.structured_depth != 1
            or node._has_nested_control
            or node.lowering_mode != "auto"
            or node.counter is None
            or not node._native_upgrade_eligible
            or not node._native_submission_eligible
        ):
            return (), ""
        routes = _cuda_structured_control_routes()
        if routes != (
            "cuda_conditional_graph",
            "cuda_masked_bounded_graph",
        ):
            return (), ""
        selected_by_route = {
            "cuda_conditional_graph": _CUDA_CONDITIONAL_CONTROL_RECIPE_ID,
            "cuda_masked_bounded_graph": _CUDA_MASKED_CONTROL_RECIPE_ID,
        }
        selected = selected_by_route.get(node._cuda_control_lowering, "")
        if not selected:
            return (), ""
        return _CUDA_CONTROL_RECIPE_IDS, selected

    roots = tuple(node for node in control_nodes if node.control_depth == 1)
    if len(roots) != 1 or not 2 <= len(control_nodes) <= 9:
        return (), ""
    outer = roots[0]
    if (
        not isinstance(outer, _CompiledWhileGraphNode)
        or outer.structured_depth != 2
        or not outer._has_nested_control
        or outer.lowering_mode != "auto"
        or outer.counter is None
        or outer._cuda_nested is None
        or outer._cuda_nested_reason != "eligible"
        or not outer._native_upgrade_eligible
        or not outer._native_submission_eligible
    ):
        return (), ""
    body_children = ()
    for role, children in outer._definition_children:
        if role == "body":
            body_children = tuple(children)
        elif children:
            return (), ""
    inners = tuple(control_nodes[1:])
    if body_children != inners or not 1 <= len(inners) <= 8:
        return (), ""
    if any(
        not isinstance(inner, _CompiledWhileGraphNode)
        or inner.control_depth != 2
        or inner.structured_depth != 1
        or inner._has_nested_control
        or inner.lowering_mode != "auto"
        or inner.counter is None
        or not inner._native_upgrade_eligible
        or not inner._native_submission_eligible
        for inner in inners
    ):
        return (), ""

    routes = _cuda_nested_structured_control_routes()
    if routes != (
        _CUDA_NESTED_DEVICE_UPDATE_ROUTE,
        _CUDA_NESTED_MASKED_ROUTE,
    ):
        return (), ""
    selected_by_route = {
        _CUDA_NESTED_DEVICE_UPDATE_ROUTE: (
            _CUDA_NESTED_DEVICE_UPDATE_CONTROL_RECIPE_ID
        ),
        _CUDA_NESTED_MASKED_ROUTE: _CUDA_NESTED_MASKED_CONTROL_RECIPE_ID,
    }
    selected = selected_by_route.get(outer._cuda_nested_control_lowering, "")
    if not selected:
        return (), ""
    return _CUDA_NESTED_CONTROL_RECIPE_IDS, selected


def _record_frozen_recipe_dispatch(builder, node, kernel_cpp, args, label):
    # Reconstruct only bindings still consumed by this dispatch. Copying the
    # entire old fixed frame would also retain workspaces of replaced recipes.
    # The frozen declaration object preserves private aliases and lifetime.
    for symbolic in args:
        name = getattr(symbolic, "name", None)
        if name in node.fixed_runtime_args:
            builder._bind_internal_runtime_arg(symbolic, node.fixed_runtime_args[name])
    builder._record_dispatch(kernel_cpp, list(args), label)


def _replay_recipe_cgraph_node(
    node,
    assembly,
    map_source_groups,
    *,
    parallel_dispatch_groups=(),
    parallel_disjoint_pairs=(),
):
    if not node.recipe_operations:
        raise TaichiRuntimeError(
            "selected complete Graph recipe has no frozen CGraph replay source"
        )
    builder = GraphBuilder(
        _capture_recipe_sources=False,
        _explicit_map_source_groups=tuple(map_source_groups),
    )
    for operation in node.recipe_operations:
        kind = operation[0]
        if kind == "dispatch":
            (
                _,
                kernel_cpp,
                args,
                label,
                memory_source,
                offload_source,
                sparse_source,
            ) = operation
            contracts = ()
            layout_requirements = ()
            rewriters = tuple(
                rewriter
                for rewriter in (
                    assembly.dispatch_rewriter(memory_source),
                    assembly.dispatch_rewriter(offload_source),
                    assembly.dispatch_rewriter(sparse_source),
                )
                if rewriter is not None
            )
            if len(rewriters) > 1:
                raise TaichiRuntimeError(
                    "one Graph dispatch cannot select multiple physical "
                    "kernel recipe families simultaneously"
                )
            if rewriters:
                kernel_cpp, contracts, layout_requirements = rewriters[0]()
            _record_frozen_recipe_dispatch(builder, node, kernel_cpp, args, label)
            if contracts or layout_requirements:
                builder._pending_ir_nodes[-1] = replace(
                    builder._pending_ir_nodes[-1],
                    memory_disjoint_pairs=contracts,
                    memory_layout_requirements=layout_requirements,
                )
            continue
        if kind == "sequential":
            builder.append(operation[1].thaw())
            continue
        if kind == "native":
            _, executable, admission = operation
            rewriter = assembly.operation_rewriter(executable)
            if rewriter is None:
                from taichi_forge.graph._recipes.deferred import FrozenNativeRecipeSource

                if isinstance(executable, FrozenNativeRecipeSource):
                    executable.append_to_graph(builder, admission=admission)
                    continue
                builder._append_native_executable(executable, admission=admission)
            else:
                rewriter(builder, operation)
            continue
        if kind == "bounded":
            source = operation[1]
            rewriter = assembly.operation_rewriter(source)
            if rewriter is None:
                source.materialize(builder, source.selected_strategy)
            else:
                rewriter(builder, operation)
            continue
        if kind == "graph_reduction":
            _, source, label = operation
            rewriter = assembly.operation_rewriter(source)
            if rewriter is None:
                source.materialize(
                    builder,
                    None,
                    label=label,
                    record_selection=False,
                )
            else:
                rewriter(builder, operation)
            continue
        raise TaichiRuntimeError(f"unknown frozen Graph recipe operation {kind!r}")
    if parallel_dispatch_groups:
        branch_start = parallel_dispatch_groups[0][0]
        current = builder._pending_ir_nodes[branch_start]
        builder._pending_ir_nodes[branch_start] = replace(
            current,
            memory_disjoint_pairs=tuple(
                sorted(
                    set(current.memory_disjoint_pairs).union(parallel_disjoint_pairs)
                )
            ),
        )
        builder._set_cuda_parallel_dispatch_groups(parallel_dispatch_groups)
    builder._flush_graph_builder()
    if len(builder._nodes) != 1 or not isinstance(
        builder._nodes[0], _CompiledCGraphNode
    ):
        raise TaichiRuntimeError(
            "complete Graph CGraph replay did not produce one physical segment"
        )
    return builder._nodes[0]


def _replay_recording_partition_nodes(node, source):
    """Rebuild one frozen CGraph as two ordered backend Graph segments."""

    operations = tuple(node.recipe_operations)
    children = tuple(getattr(node.ir_node, "children", ()))
    if (
        source.dispatch_count != len(operations)
        or len(children) != len(operations)
        or not 0 < source.cut_index < len(operations)
        or any(operation[0] != "dispatch" for operation in operations)
        or any(not isinstance(child, DispatchNode) for child in children)
    ):
        raise TaichiRuntimeError(
            "recording-partition recipe no longer matches its frozen CGraph source"
        )

    builder = GraphBuilder(
        _capture_recipe_sources=False,
        _explicit_map_source_groups=(),
    )
    for index, (operation, child) in enumerate(zip(operations, children), start=1):
        _, kernel_cpp, args, label, *_ = operation
        _record_frozen_recipe_dispatch(builder, node, kernel_cpp, args, label)
        # Keep the frozen semantic/effect contract verbatim.  Recompiling the
        # dispatch recovers the executable packet; it must not silently erase
        # memory, alias, or logical identity metadata owned by the definition.
        builder._pending_ir_nodes[-1] = child
        if index == source.cut_index:
            builder._flush_graph_builder()
    builder._flush_graph_builder()
    nodes = tuple(builder._nodes)
    if (
        len(nodes) != 2
        or any(not isinstance(item, _CompiledCGraphNode) for item in nodes)
        or tuple(item.dispatch_count for item in nodes)
        != (source.cut_index, source.dispatch_count - source.cut_index)
    ):
        raise TaichiRuntimeError(
            "recording-partition recipe did not produce its declared CUDA segments"
        )
    return nodes


def _dispatch_leaf_count(node):
    if isinstance(node, DispatchNode):
        return 1
    return sum(_dispatch_leaf_count(child) for child in node.children)


def _clone_control_recipe_runtime_node(node, control_recipe):
    if _is_structured_control_node(node):
        return _clone_structured_recipe_node(node, control_recipe)
    if isinstance(node, _CompiledSequentialRegionNode):
        sequences = tuple(
            _clone_recipe_sequential(sequence, control_recipe)
            for sequence in node.definition_sequence_snapshots
        )
        if not sequences:
            raise TaichiRuntimeError(
                "structured sequential recipe has no frozen definition source"
            )
        return _compile_sequential_runtime_node(
            sequences,
            name=node.name,
            region_kind=node.region_kind,
        )
    raise TaichiRuntimeError("unknown structured-control recipe container")


def _clone_structured_recipe_node(node, control_recipe):
    regions = {
        role: _clone_recipe_sequential(region, control_recipe)
        for role, region in node._definition_regions
    }
    if isinstance(node, _CompiledWhileGraphNode):
        return _CompiledWhileGraphNode(
            regions["condition"],
            regions["body"],
            predicate=node.predicate,
            control_inputs=node.control_inputs,
            carried_state=node.carried_state,
            max_iterations=node.max_iterations,
            counter=node.counter,
            status=node.status,
            chunk_size=node.chunk_limit,
            vulkan_first_chunk_strategy=node.vulkan_first_chunk_strategy,
            masked_execution=node.masked_execution,
            lowering_mode=node.lowering_mode,
            name=node.name,
            _control_recipe=control_recipe,
        )
    if isinstance(node, _CompiledIfGraphNode):
        return _CompiledIfGraphNode(
            regions["condition"],
            regions["then"],
            regions.get("else"),
            predicate=node.predicate,
            control_inputs=node.control_inputs,
            lowering_mode=node.lowering_mode,
            name=node.name,
            _control_recipe=control_recipe,
        )
    if isinstance(node, _CompiledSwitchGraphNode):
        branches = tuple(
            region for role, region in regions.items() if role.startswith("case_")
        )
        return _CompiledSwitchGraphNode(
            regions["condition"],
            branches,
            regions.get("default"),
            selector=node.selector,
            control_inputs=node.control_inputs,
            lowering_mode=node.lowering_mode,
            name=node.name,
            _control_recipe=control_recipe,
        )
    raise TaichiRuntimeError("unknown structured-control recipe source")


def _clone_recipe_sequential(sequence, control_recipe):
    clone = (
        sequence.thaw()
        if isinstance(sequence, _GraphSequentialRecipeSnapshot)
        else _GraphSequentialRecipeSnapshot(sequence).thaw()
    )
    for index, item in enumerate(tuple(clone._items)):
        if item[0] != "structured":
            continue
        child = _clone_structured_recipe_node(item[1], control_recipe)
        clone._items[index] = ("structured", child)
        clone._ir_nodes[index] = child.ir_node
    return clone


_CONTROL_RECIPE_ROUTES = {
    _CUDA_CONDITIONAL_CONTROL_RECIPE_ID: "cuda_conditional_graph",
    _CUDA_MASKED_CONTROL_RECIPE_ID: "cuda_masked_bounded_graph",
    _CUDA_NESTED_DEVICE_UPDATE_CONTROL_RECIPE_ID: "cuda_nested_device_update",
    _CUDA_NESTED_MASKED_CONTROL_RECIPE_ID: "cuda_nested_masked_bounded",
}


class _GraphSpec:
    def _with_recipe_nodes(self, nodes):
        return _GraphSpec(
            nodes,
            graph_memory_sources=self._graph_memory_sources,
            graph_offload_fusion_sources=self._graph_offload_fusion_sources,
            graph_sparse_traversal_sources=self._graph_sparse_traversal_sources,
            graph_bounded_sources=self._graph_bounded_sources,
            graph_reduction_sources=self._graph_reduction_sources,
            graph_native_algorithm_sources=self._graph_native_algorithm_sources,
        )

    def freeze_native_recipe_sources(self):
        """Detach opt-in native owners without mutating any live Graph/builder."""
        from copy import copy
        from taichi_forge.graph._recipes.deferred import FrozenNativeRecipeSource

        nodes = []
        changed = False
        for node in self.nodes:
            if not isinstance(node, _CompiledCGraphNode):
                nodes.append(node)
                continue
            operations = []
            removed_leases, retained_leases = set(), set()
            detached = False
            for operation in node.recipe_operations:
                if operation[0] == "native":
                    executable = operation[1]
                    freeze = getattr(executable, "_freeze_graph_recipe_source", None)
                    source = None if freeze is None else freeze()
                    if source is not None:
                        if not isinstance(source, FrozenNativeRecipeSource):
                            raise TypeError("native recipe freeze requires a cold reconstruction source")
                        action = executable.recordable_action
                        if action.fixed_bindings or action.temporary_bindings:
                            raise ValueError("detachable native recipes cannot retain private Graph bindings")
                        removed_leases.update(id(lease) for lease in (executable, *executable.lifetime_leases))
                        operation = ("native", source, operation[2])
                        detached = True
                    elif not isinstance(executable, FrozenNativeRecipeSource):
                        retained_leases.update(id(lease) for lease in (executable, *executable.lifetime_leases))
                operations.append(operation)
            if detached:
                node = copy(node)
                node._frozen_pipeline_task_manifests = _pipeline_task_manifests(node)
                node.compiled_graph = None
                node._jit_cache = None
                node.recipe_operations = tuple(operations)
                node.lifetime_leases = tuple(
                    lease
                    for lease in node.lifetime_leases
                    if id(lease) not in removed_leases or id(lease) in retained_leases
                )
                node._requires_recipe_materialization = True
                changed = True
            nodes.append(node)
        return self._with_recipe_nodes(nodes) if changed else self

    def materialize_baseline_sources(self, definition):
        """Reconstruct detached segments only at the explicit compile boundary."""
        if not any(getattr(node, "_requires_recipe_materialization", False) for node in self.nodes):
            return self
        from taichi_forge.graph._recipes.runtime_assembly import GraphRuntimeRecipeAssembly

        assembly = GraphRuntimeRecipeAssembly(definition)
        nodes = []
        for node in self.nodes:
            if getattr(node, "_requires_recipe_materialization", False):
                groups = tuple(
                    tuple(int(item.split(":")[1]) for item in group)
                    for group in node.composer_source_groups
                    if len(group) > 1 and all(item is not None for item in group)
                )
                node = _replay_recipe_cgraph_node(node, assembly, groups)
            nodes.append(node)
        variant = self._with_recipe_nodes(nodes)
        variant._definition_source_spec = self
        return variant

    def __init__(
        self,
        nodes,
        aot_graph_builder=None,
        aot_compiled_graph=None,
        graph_memory_sources=(),
        graph_offload_fusion_sources=(),
        graph_sparse_traversal_sources=(),
        graph_bounded_sources=(),
        graph_reduction_sources=(),
        graph_native_algorithm_sources=(),
    ):
        source_nodes = tuple(nodes)
        structured_control_nodes = _prepare_structured_definition_tree(source_nodes)
        structured_owner_token = object()
        self.pre_optimization_ir_root = SequentialRegion(
            tuple(node.ir_node for node in source_nodes), name="graph"
        )
        self.pre_optimization_ir_analysis = analyze_graph_ir(
            self.pre_optimization_ir_root
        )
        self.memory_disjoint_pairs = _graph_memory_disjoint_pairs(
            self.pre_optimization_ir_root
        )
        self.memory_layout_requirements = _graph_memory_layout_requirements(
            self.pre_optimization_ir_root
        )
        self.memory_recipe_binding_names = tuple(
            sorted(
                {
                    *(item[0] for item in self.memory_layout_requirements),
                    *(name for pair in self.memory_disjoint_pairs for name in pair),
                }
            )
        )
        self.temporary_memory_plan = plan_temporary_memory(
            self.pre_optimization_ir_root
        )
        self.nodes, self.optimization = _lower_mixed_backend_regions(source_nodes)
        self._pipeline_definition_cache = None
        applied_groups = sum(
            getattr(node, "composer_applied_groups", 0) for node in self.nodes
        )
        applied_source_groups = tuple(
            tuple(f"graph/{source_index}:{node.ir_node.name}/{item}" for item in group)
            for source_index, node in enumerate(source_nodes)
            for group in getattr(node, "composer_source_groups", ())
            if all(item is not None for item in group)
        )
        lowering_available = any(
            getattr(node, "composer_lowering_available", False) for node in self.nodes
        )
        self.fusion_plan = analyze_elementwise_fusion(
            self.pre_optimization_ir_root,
            applied_groups=applied_groups,
            applied_source_groups=applied_source_groups,
            lowering_available=lowering_available,
        )
        backend = _backend_name(_ti_core.arch_name(impl.current_cfg().arch))
        self._graph_branch_join_sources = _discover_cuda_branch_join_sources(
            source_nodes,
            backend,
        )
        self._graph_recording_partition_sources = (
            _discover_cuda_recording_partition_sources(source_nodes, backend)
        )
        control_recipe_domains = []
        for node_index, source_node in enumerate(source_nodes):
            roots = tuple(_structured_root_call_sites(source_node))
            if not roots:
                continue
            controls = tuple(
                node
                for node in structured_control_nodes
                if any(
                    node.region_path == root.region_path
                    or node.region_path.startswith(root.region_path + "/")
                    for root in roots
                )
            )
            recipe_ids, selected_recipe_id = _cuda_structured_control_recipe_domain(
                (source_node,),
                controls,
                backend,
            )
            if recipe_ids:
                control_recipe_domains.append(
                    (node_index, tuple(recipe_ids), selected_recipe_id)
                )
        self.control_recipe_domains = tuple(control_recipe_domains)
        # Retain the selected single-domain metadata for the complete-recipe
        # provider. Qualified runtime fusion has its own map-only space.
        if len(self.control_recipe_domains) == 1:
            _, control_recipe_ids, selected_control_recipe_id = (
                self.control_recipe_domains[0]
            )
        else:
            control_recipe_ids, selected_control_recipe_id = (), ""
        self.control_recipe_ids = tuple(control_recipe_ids)
        self.selected_control_recipe_id = selected_control_recipe_id

        self._optimization_backend = backend
        self._executable_optimization_space = None
        self._graph_memory_sources = tuple(graph_memory_sources)
        self._graph_offload_fusion_sources = tuple(graph_offload_fusion_sources)
        self._graph_sparse_traversal_sources = tuple(graph_sparse_traversal_sources)
        self._graph_bounded_sources = tuple(graph_bounded_sources)

        self._graph_reduction_sources = tuple(graph_reduction_sources)
        self._graph_native_algorithm_sources = tuple(graph_native_algorithm_sources)
        self._aot_graph_builder = aot_graph_builder
        self._aot_compiled_graph = aot_compiled_graph
        self.needs_runtime_args = any(n.needs_runtime_args for n in self.nodes)
        self.dispatch_count = sum(getattr(n, "dispatch_count", 0) for n in self.nodes)
        self.native_count = sum(
            getattr(n, "source_native_count", 0) for n in self.nodes
        )
        self.structured_control_nodes = structured_control_nodes
        self.structured_control_roots = tuple(
            node for node in structured_control_nodes if node.control_depth == 1
        )
        self.structured_while_roots = tuple(
            node
            for node in self.structured_control_roots
            if isinstance(node, _CompiledWhileGraphNode)
        )
        self.supports_native_structured_submission = bool(
            self.structured_control_roots
        ) and all(
            node.supports_native_submission for node in self.structured_control_roots
        )
        self.control_binding_plan = _structured_control_binding_plan(
            self.structured_control_nodes
        )
        self.structured_control_count = len(self.structured_control_nodes)
        self.structured_while_count = sum(
            isinstance(n, _CompiledWhileGraphNode)
            for n in self.structured_control_nodes
        )
        self.max_structured_depth = max(
            (node.control_depth for node in self.structured_control_nodes),
            default=0,
        )
        self.observation_count = sum(
            isinstance(n, _CompiledObservationGraphNode) for n in self.nodes
        )
        self.temporary_actions = _merge_temporary_actions(self.nodes)
        self._temporary_binding_cache = {}
        # Mutable compatibility dictionaries still need a fresh owner check on
        # every replay. The cache skips only the exhaustive layout/alias proof
        # after collision-free descriptor equality. Fast-qualified
        # GraphBindingSet versions retain the returned certificate and do not
        # revisit this cache.
        self._memory_recipe_binding_cache = {}
        self.temporary_runtime_arg_names = frozenset().union(
            *(frozenset(action.temporary_bindings) for action in self.temporary_actions)
        )
        self.fixed_runtime_args = _merge_fixed_runtime_args(self.nodes)
        # These values are immutable scalars or retained ndarray allocations
        # whose identity cannot change before runtime teardown, just like a
        # public ndarray binding. Internal requirements resolve per instance.
        # A single workspace lane may therefore flatten them once together
        # with a published BindingVersion. Provider-controlled or otherwise
        # replaceable fixed values remain on the validated replay path.
        self._stable_fixed_binding_frame = bool(self.fixed_runtime_args) and all(
            isinstance(value, (InternalNdarrayRequirement, Ndarray, int, float))
            for value in self.fixed_runtime_args.values()
        )
        self.internal_storage_bytes = _graph_internal_storage_bytes(
            self.fixed_runtime_args
        )
        lifetime_leases = []
        seen_lifetime_leases = set()
        for node in self.nodes:
            for lease in getattr(node, "lifetime_leases", ()):
                identity = id(lease)
                if identity not in seen_lifetime_leases:
                    seen_lifetime_leases.add(identity)
                    lifetime_leases.append(lease)
        self.lifetime_leases = tuple(lifetime_leases)
        # Some providers already fail stale handles closed under the native
        # submission/resource lock. They still remain strong lifetime owners,
        # but do not need a Python validation call before every replay.
        self.runtime_lifetime_leases = tuple(
            lease
            for lease in self.lifetime_leases
            if getattr(lease, "validate_graph_lifetime", None) is not None
            and getattr(lease, "graph_runtime_lifetime_check_required", True)
        )
        self.exclusive_provider_submission = any(
            bool(getattr(lease, "exclusive_graph_submission", False))
            for lease in self.lifetime_leases
        )
        self.native_execution_observer_leases = tuple(
            lease
            for lease in self.lifetime_leases
            if getattr(lease, "_record_synchronous_graph_execution", None) is not None
            or getattr(lease, "_begin_graph_submission_observation", None) is not None
        )
        self.derived_runtime_arg_names = frozenset().union(
            *(
                getattr(node, "derived_runtime_arg_names", frozenset())
                for node in self.nodes
            )
        )
        all_runtime_arg_names = frozenset().union(
            *(
                getattr(
                    node,
                    "recording_runtime_arg_names",
                    node.runtime_arg_names,
                )
                for node in self.nodes
            )
        )
        self.runtime_arg_names = all_runtime_arg_names.difference(
            (
                *self.fixed_runtime_args,
                *self.temporary_runtime_arg_names,
                *self.derived_runtime_arg_names,
            )
        )
        self.snode_tree_dependencies = frozenset().union(
            *(n.snode_tree_dependencies for n in self.nodes)
        )
        self.snode_tree_dependency_info = frozenset().union(
            *(n.snode_tree_dependency_info for n in self.nodes)
        )
        public_names = tuple(sorted(self.runtime_arg_names))
        static_fast_path_blockers = []
        if self.runtime_lifetime_leases:
            static_fast_path_blockers.append("volatile_lifetime_provider")
        has_uncertified_binding_validation = any(
            callable(getattr(lease, "validate_graph_bindings", None))
            and getattr(
                lease,
                "graph_publish_time_binding_validation_stable",
                False,
            )
            is not True
            for lease in self.lifetime_leases
        )
        has_dynamic_argument_binding = any(
            callable(getattr(lease, "bind_graph_arguments", None))
            for lease in self.lifetime_leases
        )
        has_dynamic_submission_owners = any(
            callable(getattr(lease, "graph_submission_owners", None))
            for lease in self.lifetime_leases
        )
        has_dynamic_binding_hook = (
            has_dynamic_argument_binding or has_dynamic_submission_owners
        )
        if has_uncertified_binding_validation or has_dynamic_binding_hook:
            # A certified type/shape check can be discharged when publishing
            # an immutable BindingVersion. Resource replacement and exact
            # per-submission owners are dynamic by definition and therefore
            # always keep replay on the validated slow path.
            static_fast_path_blockers.append("volatile_runtime_provider")
        if self.fixed_runtime_args:
            static_fast_path_blockers.append("lane_fixed_bindings")
        if self.derived_runtime_arg_names:
            static_fast_path_blockers.append("provider_derived_bindings")
        if self.temporary_actions:
            static_fast_path_blockers.append("lane_temporary_bindings")
        dynamic_overlay_names = frozenset(
            (
                *self.fixed_runtime_args,
                *self.derived_runtime_arg_names,
                *self.temporary_runtime_arg_names,
            )
        )
        self.control_publish_frame_stable = (
            not has_dynamic_argument_binding
            and not dynamic_overlay_names.intersection(self.control_binding_plan.names)
        )
        self.memory_recipe_publish_frame_stable = (
            not has_dynamic_argument_binding
            and not dynamic_overlay_names.intersection(self.memory_recipe_binding_names)
        )
        self.binding_plan = _GraphBindingPlan(
            public_names=public_names,
            public_name_set=self.runtime_arg_names,
            slot_by_name=MappingProxyType(
                {name: index for index, name in enumerate(public_names)}
            ),
            fixed_names=tuple(sorted(self.fixed_runtime_args)),
            derived_names=tuple(sorted(self.derived_runtime_arg_names)),
            temporary_names=tuple(sorted(self.temporary_runtime_arg_names)),
            control_names=self.control_binding_plan.names,
            control_publish_frame_stable=self.control_publish_frame_stable,
            memory_recipe_names=self.memory_recipe_binding_names,
            memory_recipe_publish_frame_stable=(
                self.memory_recipe_publish_frame_stable
            ),
            static_fast_path_blockers=tuple(static_fast_path_blockers),
        )
        self._binding_statistics = {
            "version_builds": 0,
            "flattened_frame_builds": 0,
            "raw_replay_validations": 0,
            "version_fast_replays": 0,
            "version_volatile_replays": 0,
            "control_publish_validations": 0,
            "control_replay_validations": 0,
        }
        self.repeat_count = 0
        self.ir_root = SequentialRegion(
            tuple(node.ir_node for node in self.nodes), name="graph"
        )
        self.ir_analysis = analyze_graph_ir(self.ir_root)
        for node in self.structured_control_nodes:
            node._graph_owner_token = structured_owner_token

    @property
    def definition_semantic_root(self):
        """Return the physical-choice-free input for GraphDefinition."""

        native_algorithm_definition = bool(
            len(self._graph_native_algorithm_sources) == 1
            and not self._graph_reduction_sources
            and not self._graph_memory_sources
            and self.native_count == 1
            and self.dispatch_count == 0
            and self.structured_control_count == 0
            and self.observation_count == 0
            and len(self.nodes) == 1
        )
        if native_algorithm_definition:
            return self._graph_native_algorithm_sources[0].semantic_root

        reduction_definition = bool(
            len(self._graph_reduction_sources) == 1
            and not self._graph_memory_sources
            and self.native_count == 0
            and self.structured_control_count == 0
            and self.observation_count == 0
            and len(self.nodes) == 1
            and self.dispatch_count
            == self._graph_reduction_sources[0].selected_physical_dispatches
        )
        if reduction_definition:
            return self._graph_reduction_sources[0].semantic_root

        return _graph_definition_semantic_root(self.pre_optimization_ir_root)

    @property
    def definition_semantic_sources(self):
        """Return provider source facts without materializing any candidate."""

        sources = []
        for ordinal, source in enumerate(self._graph_memory_sources):
            sources.append(
                {
                    "provider": "graph_memory",
                    "ordinal": ordinal,
                    "label": source.label,
                    "symbolic_abi": _graph_memory_symbolic_abi(source.args),
                }
            )
        for ordinal, source in enumerate(self._graph_offload_fusion_sources):
            sources.append(
                {
                    "provider": "offload_phase_fusion",
                    "ordinal": ordinal,
                    "label": source.label,
                    "symbolic_abi": _graph_memory_symbolic_abi(source.args),
                }
            )
        for ordinal, source in enumerate(self._graph_reduction_sources):
            sources.append(
                {
                    "provider": "graph_reduction",
                    "ordinal": ordinal,
                    "semantics": source.semantics.to_dict(),
                }
            )
        for ordinal, source in enumerate(self._graph_native_algorithm_sources):
            sources.append(
                {
                    "provider": "graph_native_algorithm",
                    "ordinal": ordinal,
                    "semantics": source.semantics,
                }
            )
        return tuple(sources)

    def materialize_complete_recipe(
        self,
        definition,
        recipe,
        assembly,
        *,
        workspace_lanes,
        workspace_saturation,
    ):
        """Build one explicit variant from provider-installed runtime hooks."""

        from taichi_forge.graph._recipes.runtime_assembly import (
            GraphRuntimeRecipeAssembly,
        )

        if not isinstance(assembly, GraphRuntimeRecipeAssembly):
            raise TypeError(
                "complete Graph materialization requires a runtime assembly plan"
            )
        if assembly.definition is not definition or assembly.spec is not self:
            raise ValueError("runtime Graph assembly belongs to another definition")

        map_groups = assembly.map_source_groups

        nodes = []
        dispatch_offset = 0
        consumed_map_groups = set()
        for node_index, node in enumerate(self.nodes):
            node_expander = assembly.node_expander(node_index)
            if node_expander is not None:
                nodes.extend(tuple(node_expander(node)))
                if isinstance(node, _CompiledCGraphNode):
                    dispatch_offset += _dispatch_leaf_count(node.ir_node)
                continue
            if isinstance(node, _CompiledCGraphNode):
                dispatch_count = _dispatch_leaf_count(node.ir_node)
                local_groups = []
                for group_index, group in enumerate(map_groups):
                    inside = all(
                        dispatch_offset <= index < dispatch_offset + dispatch_count
                        for index in group
                    )
                    intersects = any(
                        dispatch_offset <= index < dispatch_offset + dispatch_count
                        for index in group
                    )
                    if intersects and not inside:
                        raise TaichiRuntimeError(
                            "map-fusion fragment crosses a materialization segment"
                        )
                    if inside:
                        local_groups.append(
                            tuple(index - dispatch_offset for index in group)
                        )
                        consumed_map_groups.add(group_index)
                selected_operation = any(
                    assembly.operation_has_selection(operation)
                    for operation in node.recipe_operations
                )
                parallel_groups, parallel_disjoint_pairs = assembly.parallel_schedule(
                    node_index
                )
                # A complete recipe describes the exact map partition for the
                # whole frozen Graph.  Uncovered dispatches are singleton
                # baseline regions, not permission to inherit the ordinary
                # GraphBuilder's process-wide auto composer.  Replay every
                # fusion-capable CGraph segment even when its exact group list
                # is empty so baseline and partial recipes remain physically
                # faithful to their declared coverage.
                if (
                    self.fusion_plan.candidate_recipes
                    or getattr(node, "_requires_recipe_materialization", False)
                    or local_groups
                    or selected_operation
                    or parallel_groups
                ):
                    node = _replay_recipe_cgraph_node(
                        node,
                        assembly,
                        local_groups,
                        parallel_dispatch_groups=parallel_groups,
                        parallel_disjoint_pairs=parallel_disjoint_pairs,
                    )
                dispatch_offset += dispatch_count
            node_rewriter = assembly.node_rewriter(node_index)
            if node_rewriter is not None:
                node = node_rewriter(node)
            nodes.append(node)

        if len(consumed_map_groups) != len(map_groups):
            raise TaichiRuntimeError(
                "map-fusion fragment is outside the frozen Graph source topology"
            )
        variant = _GraphSpec(
            nodes,
            graph_memory_sources=self._graph_memory_sources,
            graph_offload_fusion_sources=self._graph_offload_fusion_sources,
            graph_sparse_traversal_sources=(self._graph_sparse_traversal_sources),
            graph_bounded_sources=self._graph_bounded_sources,
            graph_reduction_sources=self._graph_reduction_sources,
            graph_native_algorithm_sources=self._graph_native_algorithm_sources,
        )
        variant._definition_source_spec = self
        variant._complete_recipe_id = recipe.recipe_id
        variant._disable_qualified_fusion_selector = True
        variant._binding_executor_factory = assembly.binding_executor_factory
        variant._storage_plans = assembly.storage_plans
        concurrent_workspace_pair = assembly.workspace_pair
        if variant._binding_executor_factory is not None and (concurrent_workspace_pair or workspace_lanes != 1):
            raise TaichiRuntimeError("This prepared-binding submission recipe requires one workspace lane")
        if concurrent_workspace_pair:
            workspace_lanes = 2
            workspace_saturation = "wait"
        return Graph(
            variant,
            workspace_lanes=workspace_lanes,
            workspace_saturation=workspace_saturation,
            definition=definition,
            _cuda_concurrent_workspace_pair=concurrent_workspace_pair,
        )

    def terminal_control_report(self, logical_iterations):
        """Synthesize the control report for one terminal-only submission."""

        roots = self.structured_while_roots
        if len(roots) != 1 or not self.supports_native_structured_submission:
            raise TaichiRuntimeError(
                "Terminal-only Graph observation requires exactly one "
                "submission-capable root while region"
            )
        node = roots[0]
        logical_iterations = int(logical_iterations)
        if logical_iterations < 0 or logical_iterations > node.max_iterations:
            raise TaichiRuntimeError(
                "Terminal-only Graph observation returned an invalid "
                "logical iteration count"
            )
        arch = impl.current_cfg().arch
        if arch == _ti_core.Arch.vulkan:
            encoded_iterations = node.max_iterations
            executed_iterations = encoded_iterations
            lowering = "vulkan_compact_indirect"
            boundaries = (encoded_iterations,) if encoded_iterations else ()
        elif arch == _ti_core.Arch.cuda:
            lowering = node._cuda_control_lowering or "cuda_conditional_graph"
            encoded_iterations = (
                node.max_iterations
                if lowering == "cuda_masked_bounded_graph"
                else logical_iterations
            )
            executed_iterations = logical_iterations
            boundaries = (0, executed_iterations)
        else:
            raise TaichiRuntimeError(
                "Terminal-only Graph observation requires CUDA or Vulkan"
            )
        return _GraphTerminalControlReport(
            logical_iterations=logical_iterations,
            executed_iterations=executed_iterations,
            observation_batches=0,
            observation_boundaries=boundaries,
            lowering=lowering,
            encoded_iterations=encoded_iterations,
            masked_iterations=encoded_iterations - logical_iterations,
            chunk_sizes=((encoded_iterations,) if encoded_iterations else ()),
        )

    @property
    def pipeline_definition(self):
        if self._pipeline_definition_cache is None:
            self._pipeline_definition_cache = _graph_pipeline_definition(self.nodes)
        return self._pipeline_definition_cache

    def validate_lifetime_leases(self):
        for lease in self.runtime_lifetime_leases:
            validate = getattr(lease, "validate_graph_lifetime", None)
            if validate is not None:
                validate()

    def provider_memory_reports(self):
        reports = []
        seen = set()
        for lease in self.lifetime_leases:
            observe = getattr(lease, "_graph_provider_memory_report", None)
            if observe is None:
                continue
            identity = getattr(lease, "_graph_provider_memory_identity", None)
            identity = identity() if identity is not None else ("lease", id(lease))
            if identity in seen:
                continue
            seen.add(identity)
            reports.append(observe())
        return tuple(reports)

    def graph_submission_owners(self):
        owners = []
        seen = set()
        for lease in self.lifetime_leases:
            acquire = getattr(lease, "graph_submission_owners", None)
            if acquire is None:
                continue
            for owner in tuple(acquire()):
                identity = id(owner)
                if identity not in seen:
                    seen.add(identity)
                    owners.append(owner)
        return tuple(owners)

    def record_synchronous_native_execution(self):
        """Notify opt-in native leases after one successful synchronous run."""

        for lease in self.native_execution_observer_leases:
            record = getattr(lease, "_record_synchronous_graph_execution", None)
            if record is not None:
                record()

    def begin_native_submission_observations(self, completion):
        """Create ticket-owned observations for opt-in native leases."""

        observations = []
        for lease in self.native_execution_observer_leases:
            begin = getattr(lease, "_begin_graph_submission_observation", None)
            if begin is not None:
                observation = begin()
                if observation is not None:
                    observations.append(observation)
        if not completion.has_backend_work:
            for observation in observations:
                observation._observe_graph_completion()
            return ()
        return tuple(observations)

    def _binding_value_volatile_reason(self, name, value):
        from taichi_forge.lang.device_extent import DeviceExtent

        if isinstance(value, DeviceExtent):
            certified = any(
                isinstance(lease, _DeviceExtentGraphContract)
                and lease.extent_name == name
                and lease.graph_publish_time_binding_validation_stable is True
                for lease in self.lifetime_leases
            )
            if certified:
                # Like an Ndarray, the certified extent has dynamic device
                # contents but a stable allocation identity. Graph.bind()
                # snapshots its owner, flattens that allocation once, and
                # qualifies the runtime generation after the bounded contract
                # has proved its exact capacity and optional owner identity.
                return None
            return f"volatile_device_extent:{name}"
        if isinstance(value, ProviderOwnedNdarrayBinding):
            return f"volatile_provider_binding:{name}"
        if isinstance(value, DenseNdarrayView) and (
            name in self.binding_plan.memory_recipe_names
            and self.binding_plan.memory_recipe_publish_frame_stable
        ):
            # A GraphMemory BindingVersion retains the canonical owner and an
            # exact generation-qualified RuntimeStorageArgument. Rank, pitch,
            # offset, and owner liveness are therefore publish-time facts, not
            # reasons to reconstruct a descriptor during every replay.
            return None
        if isinstance(value, (DenseNdarrayView, ScalarField, MatrixField)):
            return f"volatile_dense_storage:{name}"
        if isinstance(value, Matrix) and value.is_host_access:
            return f"volatile_host_matrix:{name}"
        if isinstance(value, (int, float, Matrix, Ndarray, Texture)):
            return None
        raise TaichiRuntimeError(
            "Only Python scalars, ti.Matrix, ti.Ndarray, DeviceExtent, "
            "canonical dense Field, DenseNdarrayView, and Texture are supported "
            "as Graph runtime arguments but got "
            f"{type(value)} for {name!r}"
        )

    @staticmethod
    def _bind_memory_recipe_certificate(certificate, flattened_args):
        runtime_bindings = []
        for name, description in certificate.bindings:
            flattened = flattened_args.get(name)
            if not isinstance(flattened, tuple) or len(flattened) != 2:
                raise TaichiRuntimeError(
                    "Graph memory-recipe fast binding requires a canonical "
                    f"runtime storage argument for {name!r}"
                )
            runtime_argument = flattened[1]
            runtime_descriptor = getattr(runtime_argument, "descriptor", None)
            if runtime_descriptor is None or not runtime_descriptor.exactly_matches(
                description.descriptor
            ):
                raise TaichiRuntimeError(
                    "Graph memory-recipe runtime storage changed while its "
                    f"BindingVersion was being published for {name!r}"
                )
            runtime_bindings.append((name, runtime_argument))
        return replace(certificate, runtime_bindings=tuple(runtime_bindings))

    def build_binding_version(
        self,
        args,
        revision,
        *,
        fixed_runtime_args=None,
        allow_fixed_binding_fast_path=False,
        allow_fast_path=True,
        entrypoint="Graph.bind",
    ):
        self._validate_runtime_arg_names(args, entrypoint)
        slot_values = tuple(
            _snapshot_graph_binding_value(args[name])
            for name in self.binding_plan.public_names
        )
        snapshot = MappingProxyType(
            dict(zip(self.binding_plan.public_names, slot_values))
        )
        blockers = list(self.binding_plan.static_fast_path_blockers)
        if not allow_fast_path:
            blockers.append("qualified_fusion_selector")
        fixed_binding_fast_path = bool(
            fixed_runtime_args
            and allow_fixed_binding_fast_path
            and self._stable_fixed_binding_frame
            and frozenset(fixed_runtime_args) == frozenset(self.fixed_runtime_args)
        )
        if fixed_binding_fast_path:
            blockers.remove("lane_fixed_bindings")
        for name, value in zip(self.binding_plan.public_names, slot_values):
            reason = self._binding_value_volatile_reason(name, value)
            if reason is not None:
                blockers.append(reason)

        validation_args = snapshot
        if fixed_runtime_args:
            validation_args = dict(fixed_runtime_args)
            validation_args.update(snapshot)
            validation_args = MappingProxyType(validation_args)
        # Validate at publication whenever the relevant bindings already form
        # the final frame, even if an unrelated blocker keeps replay on the
        # slow path. A provider that can replace resources defers its proof
        # until the prepared invocation frame exists.
        control_binding_certificate = None
        if self.control_binding_plan.bindings and self.control_publish_frame_stable:
            self._binding_statistics["control_publish_validations"] += 1
            control_binding_certificate = self._validate_structured_control_bindings(
                validation_args, build_certificate=True
            )
        memory_recipe_certificate = self._validate_bound_runtime_args(
            validation_args,
            validate_memory_recipe=self.memory_recipe_publish_frame_stable,
            validate_control_bindings=False,
        )
        if (
            not blockers
            and self.memory_recipe_binding_names
            and memory_recipe_certificate is None
        ):
            blockers.append("uncertified_memory_recipe")
        if (
            not blockers
            and self.control_binding_plan.bindings
            and control_binding_certificate is None
        ):
            blockers.append("uncertified_control_bindings")

        flattened = None
        if not blockers:
            context = _GraphRunContext()
            context.begin(validation_args if fixed_binding_fast_path else snapshot)
            try:
                # This private dict is never exposed for mutation. It owns the
                # exact scalar/matrix snapshot and generation-qualified runtime
                # storage arguments consumed by the compiled CGraph.
                flattened = dict(context.flattened_args())
            finally:
                context.end()
            if memory_recipe_certificate is not None:
                memory_recipe_certificate = self._bind_memory_recipe_certificate(
                    memory_recipe_certificate, flattened
                )
            self._binding_statistics["flattened_frame_builds"] += 1
        self._binding_statistics["version_builds"] += 1
        return _GraphBindingVersion(
            revision=int(revision),
            runtime_generation=int(impl.runtime_generation()),
            slot_values=slot_values,
            arguments=snapshot,
            execution_arguments=validation_args,
            flattened_args=flattened,
            memory_recipe_certificate=memory_recipe_certificate,
            control_binding_certificate=control_binding_certificate,
            fixed_bindings_flattened=bool(
                fixed_binding_fast_path and flattened is not None
            ),
            fast_path_qualified=not blockers,
            volatile_reasons=tuple(blockers),
        )

    def prepare_invocation(
        self,
        args,
        temporaries=None,
        fixed_runtime_args=None,
        *,
        entrypoint,
        binding_version=None,
    ):
        if binding_version is not None and binding_version.fast_path_qualified:
            if temporaries or (
                fixed_runtime_args and not binding_version.fixed_bindings_flattened
            ):
                raise TaichiRuntimeError(
                    "Fast Graph BindingVersion unexpectedly requires a lane overlay"
                )
            self._binding_statistics["version_fast_replays"] += 1
            return _PreparedGraphInvocation(
                binding_version.execution_arguments,
                (binding_version,),
                binding_version.flattened_args,
                binding_version,
            )

        if binding_version is None:
            self._binding_statistics["raw_replay_validations"] += 1
        else:
            self._binding_statistics["version_volatile_replays"] += 1
            args = binding_version.arguments
        self._validate_runtime_arg_names(args, entrypoint)
        prepared = self.prepare_runtime_args(args, temporaries, fixed_runtime_args)
        if self.control_binding_plan.bindings:
            self._binding_statistics["control_replay_validations"] += 1
        self._validate_bound_runtime_args(prepared.arguments)
        if binding_version is None:
            return prepared
        return _PreparedGraphInvocation(
            prepared.arguments,
            (*prepared.submission_owners, binding_version),
            None,
            binding_version,
        )

    def binding_statistics(self):
        return MappingProxyType(
            {
                "schema_version": 1,
                **self._binding_statistics,
                "slot_order": self.binding_plan.public_names,
                "static_fast_path_qualified": (
                    not self.binding_plan.static_fast_path_blockers
                ),
                "static_fast_path_blockers": (
                    self.binding_plan.static_fast_path_blockers
                ),
            }
        )

    def prepare_runtime_args(self, args, temporaries=None, fixed_runtime_args=None):
        if isinstance(args, _PreparedGraphInvocation):
            return args
        if fixed_runtime_args:
            overlap = fixed_runtime_args.keys() & args.keys()
            if overlap:
                raise TaichiRuntimeError(
                    "Graph runtime arguments collide with provider-owned "
                    "fixed bindings: " + ", ".join(sorted(overlap))
                )
            bound = dict(fixed_runtime_args)
            bound.update(args)
        else:
            bound = args
        provider_args = bound
        binding_owners = {}
        submission_owners = []
        submission_owner_ids = set()
        for lease in self.lifetime_leases:
            bind = getattr(lease, "bind_graph_arguments", None)
            if bind is None:
                continue
            prepared = bind(provider_args)
            if isinstance(prepared, PreparedGraphBindings):
                replacements = prepared.replacements
                lease_submission_owners = prepared.submission_owners
            else:
                replacements = prepared
                acquire = getattr(lease, "graph_submission_owners", None)
                lease_submission_owners = () if acquire is None else tuple(acquire())
            if replacements is None:
                replacements = {}
            elif not isinstance(replacements, Mapping):
                raise TaichiRuntimeError(
                    "Graph provider argument bindings must be a mapping"
                )
            else:
                # Bindings and owners describe one provider generation.
                # Snapshot the mapping at this boundary so a mutable/custom
                # Mapping cannot splice values from another generation into
                # the prepared frame.
                replacements = dict(replacements)
            for owner in lease_submission_owners:
                identity = id(owner)
                if identity not in submission_owner_ids:
                    submission_owner_ids.add(identity)
                    submission_owners.append(owner)
            if not replacements:
                continue
            if bound is args:
                bound = dict(args)
            for name, value in replacements.items():
                if (
                    name not in self.runtime_arg_names
                    and name not in self.derived_runtime_arg_names
                    and name not in (fixed_runtime_args or {})
                ):
                    raise TaichiRuntimeError(
                        f"Graph provider attempted to bind unknown argument {name!r}"
                    )
                previous = binding_owners.get(name)
                if previous is not None and bound[name] is not value:
                    previous_descriptor = getattr(bound[name], "descriptor", None)
                    value_descriptor = getattr(value, "descriptor", None)
                    equivalent = (
                        previous_descriptor is not None
                        and value_descriptor is not None
                        and int(previous_descriptor.fingerprint)
                        == int(value_descriptor.fingerprint)
                    )
                    if not equivalent:
                        raise TaichiRuntimeError(
                            "Graph providers produced conflicting argument "
                            f"bindings for {name!r}"
                        )
                    value = bound[name]
                bound[name] = value
                binding_owners[name] = lease
        missing_derived = self.derived_runtime_arg_names.difference(bound)
        if missing_derived:
            raise TaichiRuntimeError(
                "Graph provider did not bind derived arguments: "
                + ", ".join(sorted(missing_derived))
            )
        temporary_args = self.bind_temporary_args(temporaries)
        if temporary_args:
            if bound is args:
                bound = dict(args)
            for name, value in temporary_args.items():
                if name in bound:
                    raise TaichiRuntimeError(
                        "Graph temporary binding collides with runtime argument "
                        f"{name!r}"
                    )
                bound[name] = value
        return _PreparedGraphInvocation(bound, tuple(submission_owners))

    def bind_runtime_args(self, args, temporaries=None, fixed_runtime_args=None):
        return self.prepare_runtime_args(
            args, temporaries, fixed_runtime_args
        ).arguments

    def bind_temporary_args(self, temporaries):
        if not self.temporary_actions:
            return {}
        if temporaries is None:
            raise TaichiRuntimeError(
                "Recordable Graph temporaries were not materialized"
            )
        cache_key = tuple(
            sorted(
                (
                    name,
                    id(binding.storage),
                    binding.offset,
                    binding.bytes,
                    binding.alignment,
                    binding.slot,
                )
                for name, binding in temporaries.items()
            )
        )
        cached = self._temporary_binding_cache.get(cache_key)
        if cached is not None:
            return cached
        resolved = {}
        owners = {}
        for action in self.temporary_actions:
            declarations = dict(action.temporary_bindings)
            required = set(declarations.values())
            missing = sorted(required.difference(temporaries))
            if missing:
                raise TaichiRuntimeError(
                    "Recordable action temporary requirements were not "
                    "materialized: " + ", ".join(missing)
                )
            provided = action.bind_graph_temporaries(
                {name: temporaries[name] for name in required}
            )
            if provided is None:
                raise TaichiRuntimeError(
                    "Recordable action rejected the active Graph temporary slot"
                )
            if not isinstance(provided, Mapping):
                raise TaichiRuntimeError(
                    "Recordable action temporary bindings must be a mapping"
                )
            provided = dict(provided)
            if provided.keys() != declarations.keys():
                raise TaichiRuntimeError(
                    "Recordable action returned unexpected temporary symbols"
                )
            for name, value in provided.items():
                if value is None:
                    raise TaichiRuntimeError(
                        f"Recordable action returned no storage for {name!r}"
                    )
                previous = owners.get(name)
                if previous is not None and resolved[name] is not value:
                    raise TaichiRuntimeError(
                        "Recordable actions produced conflicting temporary "
                        f"binding {name!r}"
                    )
                resolved[name] = value
                owners[name] = action
        if resolved.keys() != self.temporary_runtime_arg_names:
            raise TaichiRuntimeError(
                "Recordable action temporary binding coverage changed"
            )
        self._temporary_binding_cache[cache_key] = resolved
        return resolved

    def _validate_runtime_arg_names(self, args, entrypoint):
        if not isinstance(args, (dict, MappingProxyType)):
            raise TaichiRuntimeError(
                f"{entrypoint}() expects a dict of runtime arguments, got {type(args)}"
            )
        if args.keys() == self.runtime_arg_names:
            return

        missing = sorted(self.runtime_arg_names.difference(args.keys()))
        unexpected = sorted(args.keys() - self.runtime_arg_names)
        details = []
        if missing:
            details.append(f"Missing graph runtime arguments: {', '.join(missing)}")
        if unexpected:
            details.append(
                f"Unexpected graph runtime arguments: {', '.join(unexpected)}"
            )
        raise TaichiRuntimeError("; ".join(details))

    def _validate_bound_runtime_args(
        self,
        validation_args,
        *,
        validate_memory_recipe=True,
        validate_control_bindings=True,
    ):
        if validate_control_bindings:
            self._validate_structured_control_bindings(validation_args)
        memory_recipe_certificate = (
            self._validate_memory_recipe_contracts(validation_args)
            if validate_memory_recipe
            else None
        )
        for lease in self.lifetime_leases:
            validate = getattr(lease, "validate_graph_bindings", None)
            if validate is not None:
                validate(validation_args)
        return memory_recipe_certificate

    def validate_runtime_args(
        self,
        args,
        entrypoint="Graph.run",
        fixed_runtime_args=None,
    ):
        self._validate_runtime_arg_names(args, entrypoint)
        validation_args = args
        if fixed_runtime_args:
            validation_args = dict(fixed_runtime_args)
            validation_args.update(args)
        return self._validate_bound_runtime_args(validation_args)

    def parallel_candidate_report(self, branches, args=None):
        root_children = _parallel_logical_root_children(self.pre_optimization_ir_root)
        branch_indices = _normalize_parallel_branch_indices(
            branches, len(root_children)
        )
        branch_regions = tuple(
            SequentialRegion(
                tuple(root_children[index] for index in indices),
                name=f"parallel_candidate_branch_{branch_index}",
            )
            for branch_index, indices in enumerate(branch_indices)
        )
        plan = analyze_parallel_candidate(branch_regions)
        blockers = list(plan.blockers)
        conflicts = list(plan.conflicts)
        unresolved = list(plan.unresolved_aliases)
        runtime_aliases = []
        storage_facts = []
        runtime_generation = None
        backend = None

        if args is not None:
            self.validate_runtime_args(args, "Graph._parallel_candidate_report")
            runtime_generation = int(impl.runtime_generation())
            backend = _backend_name(_ti_core.arch_name(impl.current_cfg().arch))
            values = dict(self.fixed_runtime_args)
            values.update(args)
            descriptions = {}

            def storage(resource):
                cached = descriptions.get(resource)
                if cached is not None:
                    return cached
                if resource not in values:
                    fact = ParallelStorageFact(
                        resource=resource,
                        supported=False,
                        owner_status="kMissingRuntimeBinding",
                        failure_reason="kMissingRuntimeBinding",
                        source_kind="",
                        owner_kind="",
                        program_domain=None,
                        resource_identity=None,
                        tree_identity=None,
                        byte_offset=None,
                        byte_begin=None,
                        byte_end=None,
                        compact_contiguous=None,
                        index_shape=(),
                        index_strides_bytes=(),
                        element_shape=(),
                        scalar_count=None,
                        record_stride=None,
                    )
                    result = (fact, None)
                else:
                    result = _parallel_storage_description(resource, values[resource])
                descriptions[resource] = result
                storage_facts.append(result[0])
                return result

            unresolved_after_binding = []
            for dependency in unresolved:
                left_fact, left_description = storage(dependency.left_resource)
                right_fact, right_description = storage(dependency.right_resource)
                alias = "kUnknown"
                if (
                    left_description is not None
                    and right_description is not None
                    and left_fact.supported
                    and right_fact.supported
                    and left_fact.owner_status == "kNone"
                    and right_fact.owner_status == "kNone"
                ):
                    try:
                        alias = analyze_storage_alias(
                            left_description, right_description
                        )
                    except Exception:
                        alias = "kUnknown"
                runtime_aliases.append(
                    ParallelRuntimeAliasFact(
                        left_branch=dependency.left_branch,
                        right_branch=dependency.right_branch,
                        left_resource=dependency.left_resource,
                        right_resource=dependency.right_resource,
                        result=alias,
                        dependencies=dependency.dependencies,
                    )
                )
                if alias == "kProvenOverlap":
                    conflicts.append(replace(dependency, alias="proven_overlap"))
                elif alias != "kProvenDisjoint":
                    unresolved_after_binding.append(
                        replace(dependency, alias="unknown")
                    )
                    blockers.append(
                        "runtime_alias_not_proven:"
                        f"{dependency.left_resource}:"
                        f"{dependency.right_resource}"
                    )
            unresolved = unresolved_after_binding

        if blockers or conflicts:
            decision = "rejected"
            safe = False
        elif unresolved:
            decision = "runtime_binding_required"
            safe = None
        else:
            decision = "safe"
            safe = True
        return ParallelCandidateReport(
            schema_version=1,
            analysis_only=True,
            execution_changed=False,
            selection_domain="pre_optimization_logical_root",
            branch_node_indices=branch_indices,
            decision=decision,
            safe=safe,
            runtime_binding_provided=args is not None,
            runtime_generation=runtime_generation,
            backend=backend,
            branches=plan.branches,
            conflicts=tuple(conflicts),
            unresolved_aliases=tuple(unresolved),
            runtime_aliases=tuple(runtime_aliases),
            storage=tuple(sorted(storage_facts, key=lambda fact: fact.resource)),
            blockers=tuple(sorted(set(blockers))),
            sequential_fallback_peak_bytes=(plan.sequential_fallback_peak_bytes),
            parallel_branch_temporary_bytes=(plan.parallel_branch_temporary_bytes),
            parallel_peak_bytes=plan.parallel_peak_bytes,
            memory_overhead_vs_sequential=(plan.memory_overhead_vs_sequential),
            partial_output_bytes=0,
        )

    def _validate_structured_control_bindings(self, args, *, build_certificate=False):
        plan = self.control_binding_plan
        if not plan.bindings:
            return None

        descriptions = {}
        allocation_keys = {}
        for path, role, name in plan.bindings:
            if name in descriptions:
                continue
            value = args.get(name)
            if not isinstance(value, (Ndarray, ProviderOwnedNdarrayBinding)):
                raise TaichiRuntimeError(
                    f"Structured Graph control {name!r} at {path!r} must be "
                    "a canonical device ndarray scalar"
                )
            if value.arr is None:
                raise TaichiRuntimeError(
                    f"Structured Graph control {name!r} at {path!r} belongs "
                    "to a reset or retired runtime"
                )
            try:
                description = _describe_parallel_storage(value)
                owner_status = validate_storage_owner(description)
            except Exception as exc:
                raise TaichiRuntimeError(
                    f"Structured Graph control {name!r} at {path!r} could not "
                    "be validated for the current Program"
                ) from exc
            descriptor = description.descriptor
            if descriptor is None:
                raise TaichiRuntimeError(
                    f"Structured Graph control {name!r} at {path!r} is not "
                    f"describable: {description.failure_reason}"
                )
            if owner_status != "kNone":
                raise TaichiRuntimeError(
                    f"Structured Graph control {name!r} at {path!r} does not "
                    f"belong to the current Program: {owner_status}"
                )
            if (
                descriptor.scalar_type != i32
                or tuple(descriptor.element_shape)
                or any(int(extent) != 1 for extent in descriptor.index_shape)
            ):
                raise TaichiRuntimeError(
                    f"Structured Graph control {name!r} at {path!r} must "
                    "contain exactly one scalar i32 value"
                )
            resource_identity = descriptor.resource_identity
            if resource_identity is None:
                raise TaichiRuntimeError(
                    f"Structured Graph control {name!r} at {path!r} has no "
                    "stable device allocation identity"
                )
            descriptions[name] = description
            allocation_keys[name] = (
                int(descriptor.program_domain),
                tuple(resource_identity),
                int(descriptor.byte_offset),
            )

        certified_groups = [] if build_certificate else None
        for scope, group_bindings in plan.distinct_groups:
            owners = {}
            certified = [] if build_certificate else None
            for path, role, name in group_bindings:
                key = allocation_keys[name]
                previous = owners.get(key)
                if previous is not None:
                    previous_path, previous_role, previous_name = previous
                    raise TaichiRuntimeError(
                        "Structured Graph engine-control resources must not "
                        f"alias within {scope}; {previous_name!r} "
                        f"({previous_path!r} {previous_role}) and {name!r} "
                        f"({path!r} {role}) resolve to the same allocation "
                        "and byte offset"
                    )
                owners[key] = (path, role, name)
                if build_certificate:
                    certified.append((path, role, name, key))
            if build_certificate:
                certified_groups.append((scope, tuple(certified)))

        if not build_certificate:
            return None

        return _GraphControlBindingCertificate(
            runtime_generation=int(impl.runtime_generation()),
            bindings=tuple(
                (name, descriptions[name], allocation_keys[name]) for name in plan.names
            ),
            distinct_groups=tuple(certified_groups),
        )

    def _validate_memory_recipe_contracts(self, args):
        if not self.memory_layout_requirements and not self.memory_disjoint_pairs:
            return None

        # Descriptor fingerprints cover owner generation, allocation identity,
        # layout, shape, strides, and byte offset. Mutable raw dictionaries
        # reconstruct these facts and revalidate owner liveness on every call.
        # A fast-qualified immutable GraphBindingVersion retains the successful
        # certificate, its canonical Ndarray owners, and the native
        # RuntimeStorageArguments; runtime-generation invalidation then makes
        # replay-time reconstruction redundant.
        # Keep the compatibility cache-hit path deliberately thin: descriptor
        # construction, owner liveness, fingerprint lookup, and collision-proof
        # exact equality only. Layout properties and ParallelStorageFact objects
        # are materialized only for a new descriptor tuple.
        descriptions = {}
        owner_statuses = {}
        failure_reasons = {}
        cacheable = True
        for name in self.memory_recipe_binding_names:
            try:
                description = _describe_parallel_storage(args[name])
            except Exception as exc:
                descriptions[name] = None
                owner_statuses[name] = "kUnknown"
                failure_reasons[name] = f"{type(exc).__name__}:{exc}"
                cacheable = False
                continue
            descriptions[name] = description
            descriptor = description.descriptor
            if descriptor is None:
                owner_statuses[name] = description.failure_reason
                failure_reasons[name] = description.failure_reason
                cacheable = False
                continue
            try:
                owner_status = validate_storage_owner(description)
            except Exception as exc:
                owner_status = f"{type(exc).__name__}:{exc}"
            owner_statuses[name] = owner_status
            failure_reasons[name] = "kNone"
            if owner_status != "kNone":
                cacheable = False

        cache_key = None
        if cacheable:
            binding_fingerprints = tuple(
                (name, int(descriptions[name].descriptor.fingerprint))
                for name in self.memory_recipe_binding_names
            )
            cache_key = (
                int(impl.runtime_generation()),
                binding_fingerprints,
            )
            cached_certificate = self._memory_recipe_binding_cache.get(cache_key)
            if cached_certificate is not None and all(
                descriptions[name].descriptor.exactly_matches(
                    cached_description.descriptor
                )
                for name, cached_description in cached_certificate.bindings
            ):
                return cached_certificate

        storage = {}
        for name in self.memory_recipe_binding_names:
            description = descriptions[name]
            if description is None:
                fact = _unsupported_parallel_storage_fact(
                    name,
                    failure_reasons[name],
                    owner_statuses[name],
                )
            else:
                fact = _parallel_storage_fact(
                    name,
                    description,
                    owner_statuses[name],
                )
            storage[name] = (fact, description)

        for requirement in self.memory_layout_requirements:
            expected_index_shape = ()
            if len(requirement) == 4:
                name, minimum_records, record_stride, alignment = requirement
                expected_element_shape = ()
                expected_layout = "scalar"
            elif len(requirement) == 6:
                (
                    name,
                    minimum_extent,
                    record_stride,
                    alignment,
                    expected_element_shape,
                    expected_layout,
                ) = requirement
                expected_element_shape = tuple(expected_element_shape)
                if isinstance(minimum_extent, (tuple, list)):
                    expected_index_shape = tuple(int(value) for value in minimum_extent)
                    minimum_records = None
                else:
                    minimum_records = int(minimum_extent)
            else:
                raise TaichiRuntimeError(
                    "Graph memory recipe has an invalid layout certificate"
                )
            fact, _ = storage[name]
            if expected_index_shape:
                layout_valid = (
                    expected_layout == "row_major_2d"
                    and len(expected_index_shape) == 2
                    and len(fact.index_shape) == 2
                    and len(fact.index_strides_bytes) == 2
                    and all(
                        actual >= minimum
                        for actual, minimum in zip(
                            fact.index_shape, expected_index_shape
                        )
                    )
                    and fact.index_strides_bytes[0]
                    >= fact.index_shape[1] * record_stride
                    and fact.index_strides_bytes[1] == record_stride
                )
                minimum_description = (
                    f"index shape at least {expected_index_shape}, positive "
                    f"row pitch, and packed {record_stride}-byte inner records"
                )
            else:
                layout_valid = (
                    bool(fact.compact_contiguous)
                    and len(fact.index_shape) == 1
                    and fact.index_shape[0] >= minimum_records
                    and expected_layout in ("scalar", "aos")
                )
                minimum_description = (
                    f"at least {minimum_records} scalar elements"
                    if not expected_element_shape
                    else f"at least {minimum_records} records and element shape "
                    f"{expected_element_shape}"
                )
            if (
                not fact.supported
                or fact.owner_status != "kNone"
                or not layout_valid
                or fact.element_shape != expected_element_shape
                or fact.record_stride != record_stride
                or fact.byte_offset is None
                or fact.byte_offset % alignment != 0
            ):
                raise TaichiRuntimeError(
                    "Graph memory recipe requires a certified layout for "
                    f"{name!r}, aligned to {alignment} bytes with "
                    f"{minimum_description}"
                )
        for left, right in self.memory_disjoint_pairs:
            left_fact, left_description = storage[left]
            right_fact, right_description = storage[right]
            alias = "kUnknown"
            if (
                left_description is not None
                and right_description is not None
                and left_fact.supported
                and right_fact.supported
                and left_fact.owner_status == "kNone"
                and right_fact.owner_status == "kNone"
            ):
                try:
                    alias = analyze_storage_alias(left_description, right_description)
                except Exception:
                    alias = "kUnknown"
            if alias != "kProvenDisjoint":
                raise TaichiRuntimeError(
                    "Graph memory recipe requires proven "
                    f"disjoint storage for {left!r} and {right!r}; got {alias}"
                )
        certificate = _GraphMemoryRecipeCertificate(
            runtime_generation=int(impl.runtime_generation()),
            bindings=tuple(
                (name, storage[name][1]) for name in self.memory_recipe_binding_names
            ),
            layout_requirements=self.memory_layout_requirements,
            disjoint_pairs=self.memory_disjoint_pairs,
        )
        if cache_key is not None:
            self._memory_recipe_binding_cache[cache_key] = certificate
            while len(self._memory_recipe_binding_cache) > 16:
                del self._memory_recipe_binding_cache[
                    next(iter(self._memory_recipe_binding_cache))
                ]
        return certificate

    def instantiate(self, key=None):
        if key is None:
            key = self.instance_key()
        return _GraphInstance(self, key)

    def invalidate_runtime(self, preserve_executables=False):
        self._temporary_binding_cache.clear()
        self._memory_recipe_binding_cache.clear()
        for node in self.nodes:
            invalidate = getattr(node, "invalidate_runtime", None)
            if invalidate is not None:
                invalidate(preserve_executables=preserve_executables)

    def instance_key(self):
        runtime = impl.get_runtime()
        return (impl.runtime_generation(), impl.current_cfg().arch, id(runtime.prog))

    def compiled_graph(self):
        if self.native_count or self.structured_control_count or self.observation_count:
            raise TaichiRuntimeError(
                "Graphs containing native, observation, or structured-control nodes cannot "
                "be serialized as AOT CGraph yet"
            )
        if self._aot_compiled_graph is None:
            if self._aot_graph_builder is None:
                raise TaichiRuntimeError("This graph does not have an AOT CGraph")
            self._aot_compiled_graph = self._aot_graph_builder.compile()
        return self._aot_compiled_graph

    @property
    def debug_info(self):
        info = {
            "node_count": len(self.nodes),
            "dispatch_count": self.dispatch_count,
            "native_count": self.native_count,
            "observation_count": self.observation_count,
            "structured_control_count": self.structured_control_count,
            "max_structured_depth": self.max_structured_depth,
            "repeat_count": self.repeat_count,
            "nodes": [n.debug_info for n in self.nodes],
            "optimization": dict(self.optimization),
        }
        if hasattr(self._aot_graph_builder, "item_count"):
            info["aot_item_count"] = self._aot_graph_builder.item_count
        return info

    @property
    def executable_optimization_space(self):
        """Build the qualified map-fusion space only for its real consumers."""

        if self._executable_optimization_space is None:
            self._executable_optimization_space = _build_executable_optimization_space(
                self.pre_optimization_ir_root,
                self.fusion_plan,
                self._optimization_backend,
            )
        return self._executable_optimization_space

    @property
    def ir_debug_info(self):
        return {
            "metadata_version": 1,
            "analysis_only": not bool(self.optimization["mixed_backend_regions"]),
            "analysis": self.ir_analysis.to_dict(),
            "root": graph_ir_to_dict(self.ir_root),
            "pre_optimization_analysis": (self.pre_optimization_ir_analysis.to_dict()),
            "pre_optimization_root": graph_ir_to_dict(self.pre_optimization_ir_root),
            "optimization": dict(self.optimization),
            "fusion_plan": self.fusion_plan.to_dict(),
            "executable_optimization": (self.executable_optimization_space.to_dict()),
            "temporary_memory_plan": self.temporary_memory_plan.to_dict(),
        }

    @property
    def execution_definition(self):
        nodes = []
        for node in self.nodes:
            nodes.append(
                {
                    "kind": (
                        "cgraph"
                        if isinstance(node, _CompiledCGraphNode)
                        else (
                            "structured_sequence"
                            if isinstance(node, _CompiledSequentialRegionNode)
                            else (
                                "while"
                                if isinstance(node, _CompiledWhileGraphNode)
                                else (
                                    "if"
                                    if isinstance(node, _CompiledIfGraphNode)
                                    else (
                                        "switch"
                                        if isinstance(node, _CompiledSwitchGraphNode)
                                        else (
                                            "observation"
                                            if isinstance(
                                                node, _CompiledObservationGraphNode
                                            )
                                            else "native"
                                        )
                                    )
                                )
                            )
                        )
                    ),
                    "dispatch_count": getattr(node, "dispatch_count", 0),
                    "physical_dispatch_count": getattr(
                        node,
                        "physical_dispatch_count",
                        getattr(node, "dispatch_count", 0),
                    ),
                    "runtime_arg_count": len(node.runtime_arg_names),
                    "region_kind": getattr(node, "region_kind", "opaque"),
                    "source_native_count": getattr(node, "source_native_count", 0),
                    "dependency_info": tuple(sorted(node.snode_tree_dependency_info)),
                }
            )
        return {
            "nodes": tuple(nodes),
            "dispatch_count": self.dispatch_count,
            "native_count": self.native_count,
            "observation_count": self.observation_count,
            "structured_control_count": self.structured_control_count,
            "max_structured_depth": self.max_structured_depth,
            "runtime_arg_count": len(self.runtime_arg_names),
            "fixed_runtime_arg_count": len(self.fixed_runtime_args),
            "internal_storage_bytes": self.internal_storage_bytes,
            "dependency_info": tuple(sorted(self.snode_tree_dependency_info)),
            "temporary_memory_plan": self.temporary_memory_plan.to_dict(),
        }


class _GraphExecutable:
    def __init__(self, spec, fixed_runtime_args=None):
        self.spec = spec
        self.fixed_runtime_args = (
            spec.fixed_runtime_args
            if fixed_runtime_args is None
            else fixed_runtime_args
        )
        self._context = _GraphRunContext() if self.spec.needs_runtime_args else None
        self._telemetry_region_indices = {
            node.region_path: index
            for index, node in enumerate(self.spec.structured_control_nodes)
            if isinstance(node, _CompiledWhileGraphNode)
        }
        self._submission_steps = self._build_submission_steps(self.spec.nodes)

    def _build_submission_steps(self, nodes):
        steps = []

        def append(node):
            if isinstance(node, _CompiledSequentialRegionNode):
                if not node.supports_native_submission:
                    raise TaichiRuntimeError(
                        "Structured sequence submission requires every control "
                        "region to provide submission-capable native lowering"
                    )
                for child in node.nodes:
                    append(child)
                return

            if isinstance(
                node,
                (
                    _CompiledWhileGraphNode,
                    _CompiledIfGraphNode,
                    _CompiledSwitchGraphNode,
                ),
            ):
                telemetry_entries = ()
                if isinstance(node, _CompiledWhileGraphNode):
                    try:
                        telemetry_entries = tuple(
                            (
                                self._telemetry_region_indices[item.region_path],
                                item,
                            )
                            for item in _submission_telemetry_region_nodes(node)
                        )
                    except KeyError as exc:
                        raise TaichiRuntimeError(
                            "Graph submission telemetry region is absent from the "
                            "structured definition"
                        ) from exc
                steps.append(
                    (
                        node.run_for_submission,
                        telemetry_entries,
                        tuple(reversed(telemetry_entries)),
                    )
                )
                return

            steps.append((node.run, (), ()))

        for node in nodes:
            append(node)
        return tuple(steps)

    def run(self, args, temporaries=None, trace_recorder=None):
        # Graph.run() holds a per-Graph lock, so this context can safely reuse
        # flattened runtime arguments and resource signatures across invocations.
        _reset_control_flow_reports(self.spec.structured_control_nodes)
        context = self._context
        if context is not None:
            prepared = self.spec.prepare_runtime_args(
                args, temporaries, self.fixed_runtime_args
            )
            context.begin(
                prepared.arguments,
                None,
                trace_recorder,
                flattened_args=prepared.flattened_args,
            )
        try:
            for node in self.spec.nodes:
                node.run(context, temporaries)
        finally:
            if context is not None:
                context.end()

    def run_for_submission(self, args, temporaries=None, telemetry=None):
        context = self._context
        if context is not None:
            prepared = self.spec.prepare_runtime_args(
                args, temporaries, self.fixed_runtime_args
            )
            context.begin(
                prepared.arguments,
                flattened_args=prepared.flattened_args,
            )
        try:
            for (
                run,
                telemetry_entries,
                reversed_telemetry_entries,
            ) in self._submission_steps:
                if telemetry is None or not telemetry_entries:
                    run(context, temporaries)
                    continue
                for index, telemetry_node in telemetry_entries:
                    telemetry.begin_region(index, telemetry_node, context)
                try:
                    run(context, temporaries)
                finally:
                    for index, telemetry_node in reversed_telemetry_entries:
                        telemetry.end_region(index, telemetry_node, context)
        finally:
            if context is not None:
                context.end()


class _GraphInstance:
    def __init__(self, spec, key):
        self._storage_owners = ()
        self._backend_executable = None
        try:
            self._initialize(spec, key)
        except BaseException as construction_error:
            try:
                self.invalidate_runtime()
            except BaseException as cleanup_error:
                raise cleanup_error from construction_error
            raise

    def _initialize(self, spec, key):
        self.spec = spec
        self.key = key
        storage_plans = getattr(spec, "_storage_plans", ())
        allocators, arena_allocator, arena_capacity = {}, None, None
        if storage_plans:
            from taichi_forge.graph._recipes.runtime_storage import create_storage_owners, validate_storage_plans

            validate_storage_plans(spec, storage_plans)
            allocators, arena_allocator, arena_capacity = create_storage_owners(self, storage_plans)
        (
            self._fixed_runtime_args,
            self._internal_storages,
        ) = _materialize_graph_internal_bindings(spec.fixed_runtime_args, allocators)
        self._exclusive_internal_storage = (
            any(
                isinstance(value, _GraphInternalNdarraySpec) and value.exclusive_submission
                for value in spec.fixed_runtime_args.values()
            )
            or spec.exclusive_provider_submission
        )
        self._exclusive_internal_completion = None
        self._exclusive_internal_reserved = False
        self._exclusive_internal_waits = 0
        self._exclusive_internal_reuses = 0
        self._executable = None
        self._native_nodes = None
        self._run_context = None
        self._temporary_arena = _GraphTemporaryArena(spec.temporary_memory_plan, capacity=arena_capacity)
        if arena_allocator is not None:
            self._temporary_arena.prepare_storage(arena_allocator)
        self._temporary_bindings = None
        self._observation_nodes = tuple(
            node
            for node in spec.nodes
            if isinstance(node, _CompiledObservationGraphNode)
        )
        self._observation_arena = _GraphObservationArena(self._observation_nodes)
        self._structured_telemetry_nodes = tuple(
            node
            for node in spec.structured_control_nodes
            if isinstance(node, _CompiledWhileGraphNode)
        )
        self._structured_telemetry_arena = _GraphStructuredTelemetryArena(
            self._structured_telemetry_nodes,
            lambda: spec.pipeline_definition,
        )

        binding_factory = getattr(spec, "_binding_executor_factory", None)
        if binding_factory is not None:
            executable = binding_factory(self)
            self._backend_executable = executable
            self._kind = executable.execution_kind
            self._set_run_impl(self._run_prepared_backend)
        elif len(spec.nodes) == 1 and isinstance(spec.nodes[0], _CompiledCGraphNode):
            node = spec.nodes[0]
            if spec.needs_runtime_args:
                self._run_context = _GraphRunContext()
            kind = (
                "mixed_backend_region" if node.source_native_count else "single_cgraph"
            )
            self._install_backend_executable(
                _CGraphJITExecutable(node.compiled_graph), kind
            )
        elif not spec.needs_runtime_args:
            self._native_nodes = spec.nodes
            self._kind = "native_only"
            self._set_run_impl(self._run_native_only)
        else:
            self._executable = _GraphExecutable(spec, self._fixed_runtime_args)
            self._kind = "dispatch_loop"
            self._set_run_impl(self._run_general)

        self._maybe_install_native_replay()

    @property
    def run_impl(self):
        return self.run

    def _set_run_impl(self, run_impl):
        # Store an unbound class function. Keeping a bound method here creates
        # a self-cycle (instance -> method -> instance), which can defer a JIT
        # cache and its backend leases until after Program teardown.
        self._run_impl = run_impl.__func__

    def run(self, args):
        self._run_impl(self, args, self._temporary_bindings)

    def run_cuda_concurrent_batch(self, args):
        if not isinstance(self._backend_executable, _CGraphJITExecutable):
            raise TaichiRuntimeError(
                "Concurrent workspace pair requires one JIT CGraph"
            )
        context = self._run_context
        if context is None:
            raise TaichiRuntimeError(
                "Concurrent workspace pair requires runtime bindings"
            )
        prepared = self.spec.prepare_runtime_args(args, None, self._fixed_runtime_args)
        context.begin(
            prepared.arguments,
            flattened_args=prepared.flattened_args,
        )
        try:
            self._backend_executable.run_cuda_concurrent_batch(context)
        finally:
            context.end()

    def acquire_exclusive_internal_storage(self):
        if not self._exclusive_internal_storage:
            return None
        if self._exclusive_internal_reserved:
            raise TaichiRuntimeError("Graph internal workspace is already reserved")
        completion = self._exclusive_internal_completion
        if completion is not None:
            if not completion.done():
                completion.wait()
                self._exclusive_internal_waits += 1
            self._exclusive_internal_completion = None
            self._exclusive_internal_reuses += 1
        self._exclusive_internal_reserved = True
        return _GraphExclusiveInternalStorageLease(self)

    @property
    def exclusive_internal_storage_available(self):
        if not self._exclusive_internal_storage:
            return True
        if self._exclusive_internal_reserved:
            return False
        completion = self._exclusive_internal_completion
        return completion is None or completion.done()

    def _attach_exclusive_internal_storage(self, completion):
        if not self._exclusive_internal_reserved:
            raise TaichiRuntimeError(
                "Graph internal workspace attachment lost its reservation"
            )
        self._exclusive_internal_reserved = False
        self._exclusive_internal_completion = (
            completion if completion.has_backend_work else None
        )

    def _cancel_exclusive_internal_storage(self):
        self._exclusive_internal_reserved = False

    @property
    def internal_storage_stats(self):
        return {
            "reserved_bytes": int(self.spec.internal_storage_bytes),
            "exclusive": self._exclusive_internal_storage,
            "waits": self._exclusive_internal_waits,
            "reuses": self._exclusive_internal_reuses,
        }

    def run_traced(self, args, trace_recorder):
        if not self.spec.structured_control_nodes:
            self.run(args)
            return
        if self._executable is None:
            self._executable = _GraphExecutable(self.spec, self._fixed_runtime_args)
        self._executable.run(
            args,
            self._temporary_bindings,
            trace_recorder,
        )

    def run_for_submission(self, args, telemetry=None):
        if not self.spec.supports_native_structured_submission:
            return self.run(args)
        if self._executable is None:
            self._executable = _GraphExecutable(self.spec, self._fixed_runtime_args)
        self._executable.run_for_submission(args, self._temporary_bindings, telemetry)

    def bind_temporary_buffers(self, bindings):
        self._temporary_bindings = bindings

    def clear_temporary_buffers(self):
        self._temporary_bindings = None

    def acquire_temporary_lease(self):
        return self._temporary_arena.acquire()

    @property
    def temporary_arena_stats(self):
        return self._temporary_arena.stats

    def acquire_observation_lease(self):
        return self._observation_arena.acquire()

    def bind_observation_buffers(self, bindings):
        if bindings is None:
            return
        for node in self._observation_nodes:
            node.bind_snapshot_buffers(bindings[node.name])

    def clear_observation_buffers(self):
        for node in self._observation_nodes:
            node.clear_snapshot_buffers()

    def acquire_structured_telemetry_lease(self, mode):
        return self._structured_telemetry_arena.acquire(mode)

    def prepare_structured_telemetry(self, slots=1):
        self._structured_telemetry_arena.prepare(slots)
        return self

    @property
    def structured_telemetry_arena_stats(self):
        return self._structured_telemetry_arena.stats

    @property
    def observation_arena_stats(self):
        return self._observation_arena.stats

    def _maybe_install_native_replay(self):
        arch = impl.current_cfg().arch
        if arch not in (_ti_core.Arch.cuda, _ti_core.Arch.x64, _ti_core.Arch.arm64):
            return
        if not all(
            isinstance(node, _CompiledNativeGraphNode) for node in self.spec.nodes
        ):
            return
        if self.spec.needs_runtime_args:
            return
        kind = (
            "cuda_native_replay" if arch == _ti_core.Arch.cuda else "cpu_native_replay"
        )
        self._install_backend_executable(
            _NativeReplayExecutable(self.spec.nodes),
            kind,
        )

    def _install_backend_executable(self, executable, kind):
        self._backend_executable = executable
        self._kind = kind
        self._set_run_impl(self._run_backend)
        return self

    def invalidate_runtime(self, preserve_executables=False):
        try:
            if self._backend_executable is not None:
                invalidate = getattr(self._backend_executable, "invalidate_runtime", None)
                if invalidate is not None:
                    invalidate(preserve_executables=preserve_executables)
        finally:
            if self._storage_owners:
                from taichi_forge.graph._recipes.runtime_storage import retire_storage_owners

                retire_storage_owners(self)

    def prewarm(self):
        if self._backend_executable is not None:
            prewarm = getattr(self._backend_executable, "prewarm", None)
            if prewarm is not None:
                prewarm()
            return self

        for node in self.spec.nodes:
            if isinstance(node, _CompiledNativeGraphNode):
                node.executable.prewarm()
        return self

    def _run_backend(self, args, temporaries=None):
        if self._backend_executable is None:
            return self._run_general(args, temporaries)
        context = self._run_context
        if context is not None:
            prepared = self.spec.prepare_runtime_args(
                args, temporaries, self._fixed_runtime_args
            )
            context.begin(
                prepared.arguments,
                flattened_args=prepared.flattened_args,
            )
        try:
            self._backend_executable.run(context, temporaries)
        finally:
            if context is not None:
                context.end()

    def _run_native_only(self, args, temporaries=None):
        for node in self._native_nodes:
            node.run(None, temporaries)

    def prepare_binding_version(self, version):
        prepare = getattr(self._backend_executable, "prepare_binding_version", None)
        return version if prepare is None else prepare(version)

    def _run_prepared_backend(self, prepared, temporaries=None):
        self._backend_executable.run_prepared(prepared)

    @property
    def physical_submission_mode(self):
        return getattr(self._backend_executable, "physical_submission_mode", "runtime_managed")

    def _run_general(self, args, temporaries=None):
        self._executable.run(args, temporaries)

    @property
    def debug_info(self):
        return {"kind": self._kind}

    @property
    def debug_graph_stats(self):
        backend_stats = getattr(self._backend_executable, "debug_graph_stats", None)
        if backend_stats is not None:
            return [backend_stats]
        result = []
        for node in self.spec.nodes:
            if isinstance(node, _CompiledCGraphNode):
                result.append(node.debug_graph_stats)
            elif isinstance(
                node,
                (
                    _CompiledWhileGraphNode,
                    _CompiledIfGraphNode,
                    _CompiledSwitchGraphNode,
                ),
            ):
                result.extend(node.debug_graph_stats)
            elif isinstance(node, _CompiledSequentialRegionNode):
                result.extend(node.debug_graph_stats)
        return result

    @property
    def snapshot_graph_stats(self):
        backend_stats = getattr(self._backend_executable, "snapshot_graph_stats", None)
        if backend_stats is not None:
            return [backend_stats]
        result = []
        for node in self.spec.nodes:
            if isinstance(node, _CompiledCGraphNode):
                result.append(node.snapshot_graph_stats)
            elif isinstance(
                node,
                (
                    _CompiledWhileGraphNode,
                    _CompiledIfGraphNode,
                    _CompiledSwitchGraphNode,
                ),
            ):
                result.extend(node.snapshot_graph_stats)
            elif isinstance(node, _CompiledSequentialRegionNode):
                result.extend(node.snapshot_graph_stats)
        return result


class _GraphWorkspaceLanePool:
    def __init__(self, spec, key, capacity, saturation):
        self._spec = spec
        self._key = key
        self._capacity = capacity
        self._saturation = saturation
        self._instances = [None] * capacity
        self._instances[0] = spec.instantiate((*key, 0))
        self._next_lane = 0
        self._acquisitions = 0
        self._waits = 0
        self._saturation_errors = 0

    @property
    def primary(self):
        return self._instances[0]

    @property
    def instances(self):
        return tuple(instance for instance in self._instances if instance is not None)

    def _materialize(self, index):
        instance = self._instances[index]
        if instance is None:
            instance = self._spec.instantiate((*self._key, index))
            self._instances[index] = instance
        return instance

    def materialize_pair(self):
        if not self.primary._exclusive_internal_storage:
            raise TaichiRuntimeError(
                "Graph invocation pairs require exclusive Graph-owned storage"
            )
        if self._capacity == 1:
            return (self.primary, self.primary)
        if self._capacity != 2:
            raise TaichiRuntimeError(
                "Complete workspace concurrency recipes require exactly two lanes"
            )
        return (self._materialize(0), self._materialize(1))

    def acquire(self, requested_lane=None):
        if not self.primary._exclusive_internal_storage:
            if requested_lane not in (None, 0):
                raise TaichiRuntimeError(
                    "Graph workspace lanes require exclusive Graph-owned "
                    "internal storage"
                )
            self._acquisitions += 1
            return 0, self.primary
        if requested_lane is not None:
            if isinstance(requested_lane, bool) or not isinstance(
                requested_lane, (int, np.integer)
            ):
                raise TaichiRuntimeError(
                    "Graph.submit() workspace_lane must be an integer or None"
                )
            requested_lane = int(requested_lane)
            if requested_lane < 0 or requested_lane >= self._capacity:
                raise TaichiRuntimeError(
                    "Graph.submit() workspace_lane is outside the configured "
                    f"range [0, {self._capacity})"
                )
            candidates = (requested_lane,)
        else:
            candidates = tuple(
                (self._next_lane + offset) % self._capacity
                for offset in range(self._capacity)
            )

        for index in candidates:
            instance = self._instances[index]
            if instance is None or instance.exclusive_internal_storage_available:
                self._next_lane = (index + 1) % self._capacity
                self._acquisitions += 1
                return index, self._materialize(index)

        if self._saturation == "raise":
            self._saturation_errors += 1
            raise TaichiRuntimeError(
                "All Graph workspace lanes are occupied; wait for a prior "
                "SubmissionTicket or compile with workspace_saturation='wait'"
            )

        index = candidates[0]
        self._waits += 1
        self._next_lane = (index + 1) % self._capacity
        self._acquisitions += 1
        return index, self._materialize(index)

    def invalidate_runtime(self, preserve_executables=False):
        for instance in self.instances:
            instance.invalidate_runtime(preserve_executables=preserve_executables)

    @staticmethod
    def _sum_stats(instances, attribute):
        result = {}
        for instance in instances:
            for name, value in getattr(instance, attribute).items():
                if isinstance(value, bool):
                    result[name] = bool(result.get(name, False) or value)
                elif isinstance(value, (int, np.integer)):
                    result[name] = int(result.get(name, 0)) + int(value)
                elif name not in result:
                    result[name] = value
        return result

    @property
    def temporary_arena_stats(self):
        return self._sum_stats(self.instances, "temporary_arena_stats")

    @property
    def observation_arena_stats(self):
        return self._sum_stats(self.instances, "observation_arena_stats")

    @property
    def structured_telemetry_arena_stats(self):
        return self._sum_stats(self.instances, "structured_telemetry_arena_stats")

    @property
    def internal_storage_stats(self):
        stats = self._sum_stats(self.instances, "internal_storage_stats")
        stats.update(
            {
                "lane_capacity": self._capacity,
                "lanes_materialized": len(self.instances),
                "lanes_busy": sum(
                    not instance.exclusive_internal_storage_available
                    for instance in self.instances
                ),
                "lane_acquisitions": self._acquisitions,
                "lane_waits": self._waits,
                "lane_saturation_errors": self._saturation_errors,
                "lane_saturation_policy": self._saturation,
            }
        )
        return stats


class _AOTGraphBuilderPlan:
    def __init__(self):
        self._items = []
        self._runtime_arg_names = set()
        self._has_indirect_dispatch = False
        self._has_internal_fixed_bindings = False

    def dispatch(self, kernel_cpp, args, label=""):
        runtime_arg_names = frozenset(_runtime_arg_names(args))
        self._items.append(("dispatch", kernel_cpp, args, label, runtime_arg_names))
        self._runtime_arg_names.update(runtime_arg_names)

    def dispatch_indirect(self, kernel_cpp, args, dispatch_packet, label=""):
        runtime_arg_names = frozenset((*_runtime_arg_names(args), dispatch_packet.name))
        self._items.append(
            (
                "indirect",
                kernel_cpp,
                args,
                dispatch_packet,
                label,
                runtime_arg_names,
            )
        )
        self._runtime_arg_names.update(runtime_arg_names)
        self._has_indirect_dispatch = True

    @property
    def runtime_arg_names(self):
        """Return symbolic arguments recorded by the durable AOT plan.

        Low-level graph adapters historically dispatched a precompiled kernel
        directly to both ``_aot_graph_plan`` and the native graph builder.  In
        that path ``GraphBuilder.dispatch()`` cannot update its fast-path name
        cache, but the AOT plan still owns the complete symbolic argument list.
        Recovering names here keeps strict runtime validation compatible with
        those adapters without accepting genuinely unknown arguments.
        """
        return frozenset(self._runtime_arg_names)

    def mark_internal_fixed_bindings(self):
        self._has_internal_fixed_bindings = True

    def runtime_arg_names_since(self, cursor):
        if cursor < 0 or cursor > len(self._items):
            raise TaichiRuntimeError(f"Invalid AOT graph plan cursor {cursor}")
        return frozenset().union(*(item[-1] for item in self._items[cursor:]))

    def append(self, node):
        # Freeze each append at the point where the runtime builder consumes it.
        # Reusing and then mutating one Sequential between appends must not make
        # the lazily compiled AOT plan observe only its final definition.
        runtime_arg_names = frozenset(node._runtime_arg_names)
        self._items.append(
            (
                "append",
                _AOTSequentialSnapshot(node._dispatches, node._dispatch_labels),
                1,
                runtime_arg_names,
            )
        )
        self._runtime_arg_names.update(runtime_arg_names)
        self._has_indirect_dispatch = (
            self._has_indirect_dispatch or node._has_indirect_dispatch
        )

    def snapshot(self):
        items = []
        for item in self._items:
            if item[0] == "dispatch":
                _, kernel_cpp, args, label, runtime_arg_names = item
                items.append(
                    (
                        "dispatch",
                        kernel_cpp,
                        tuple(args),
                        label,
                        runtime_arg_names,
                    )
                )
            elif item[0] == "indirect":
                (
                    _,
                    kernel_cpp,
                    args,
                    dispatch_packet,
                    label,
                    runtime_arg_names,
                ) = item
                items.append(
                    (
                        "indirect",
                        kernel_cpp,
                        tuple(args),
                        dispatch_packet,
                        label,
                        runtime_arg_names,
                    )
                )
            elif item[0] == "append":
                _, node, count, runtime_arg_names = item
                items.append(
                    (
                        "append",
                        _AOTSequentialSnapshot(node._dispatches, node._dispatch_labels),
                        count,
                        runtime_arg_names,
                    )
                )
            else:
                raise TaichiRuntimeError(f"Unknown AOT graph item kind {item[0]}")

        snapshot = _AOTGraphBuilderPlan()
        snapshot._items = tuple(items)
        snapshot._runtime_arg_names = set(self._runtime_arg_names)
        snapshot._has_indirect_dispatch = self._has_indirect_dispatch
        snapshot._has_internal_fixed_bindings = self._has_internal_fixed_bindings
        return snapshot

    def _compile(self, map_composer_max_group_size, allowed_source_groups):
        if self._has_internal_fixed_bindings:
            raise TaichiRuntimeError(
                "Graph bounded dispatch uses JIT-only internal fixed bindings "
                "and cannot be added to an AOT module"
            )
        if self._has_indirect_dispatch:
            raise TaichiRuntimeError(
                "Graph indirect dispatch is currently JIT-only and cannot "
                "be added to an AOT module"
            )
        if (
            isinstance(map_composer_max_group_size, bool)
            or not isinstance(map_composer_max_group_size, int)
            or map_composer_max_group_size < 1
            or map_composer_max_group_size > 4
        ):
            raise TaichiRuntimeError(
                "AOT Graph map composer group size must be in [1, 4]"
            )
        builder = _ti_core.GraphBuilder()
        if map_composer_max_group_size > 1:
            builder._set_map_composer_max_group_size(map_composer_max_group_size)
        if allowed_source_groups:
            builder._set_map_composer_allowed_groups(allowed_source_groups)
        for item in self._items:
            if item[0] == "dispatch":
                _, kernel_cpp, args, label, _ = item
                builder.dispatch(kernel_cpp, args, label)
            elif item[0] == "append":
                _, node, count, _ = item
                seq = builder.create_sequential()
                node._dispatch_to(seq)
                for _ in range(count):
                    builder.seq().append(seq)
            else:
                raise TaichiRuntimeError(f"Unknown AOT graph item kind {item[0]}")
        return builder.compile()

    def compile(self, *, map_composer_max_group_size=1):
        return self._compile(map_composer_max_group_size, ())

    def _compile_map_recipes(self, source_groups):
        source_groups = tuple(
            tuple(int(item) for item in group) for group in source_groups
        )
        if not source_groups:
            raise TaichiRuntimeError("Map recipe compilation requires source groups")
        claimed = set()
        for group in source_groups:
            if len(group) < 2 or len(group) > 4 or len(set(group)) != len(group):
                raise TaichiRuntimeError(
                    "Map recipe source groups must contain two to four unique IDs"
                )
            if claimed.intersection(group):
                raise TaichiRuntimeError("Map recipe source groups must be disjoint")
            claimed.update(group)
        return self._compile(max(len(group) for group in source_groups), source_groups)

    @property
    def item_count(self):
        return len(self._items)


def gen_cpp_kernel(
    kernel_fn,
    args,
    *,
    template_args=None,
    task_launch_policy=None,
    range_one_to_one=False,
    allow_graph_memory_recipe=False,
):
    execution_plan = None
    if isinstance(kernel_fn, kernel_impl._OffloadExecutionPlanBinding):
        if kernel_fn._bound_args:
            raise TaichiCompilationError(
                "Graph task-indexed execution plans do not support bound "
                "class-kernel instances"
            )
        kernel = kernel_fn._kernel
        execution_plan = kernel_fn.plan
        if execution_plan.requires_graph_memory and not allow_graph_memory_recipe:
            raise TaichiCompilationError(
                "shared-staged execution plans require the private "
                "Graph-owned memory recipe materializer"
            )
    else:
        kernel = (
            kernel_fn
            if isinstance(kernel_fn, kernel_impl.Kernel)
            else getattr(kernel_fn, "_primal", None)
        )
    if not isinstance(kernel, kernel_impl.Kernel):
        raise TaichiCompilationError(
            "Graph dispatch expects a decorated Taichi kernel or an explicit "
            "kernel.grad object. Python callables and ti.func objects cannot "
            "be submitted as Graph nodes."
        )
    injected_args = produce_injected_args_for_graph(
        kernel, symbolic_args=args, template_args=template_args
    )
    if execution_plan is not None:
        if task_launch_policy is not None or range_one_to_one:
            raise TaichiCompilationError(
                "Graph task-indexed execution plans cannot be combined with "
                "legacy launch-policy or one-to-one lowering controls"
            )
        key = kernel._ensure_compiled_with_offload_execution_plan(
            execution_plan, *injected_args
        )
        kernel._validate_offload_execution_plan_specialization(key, execution_plan)
    elif (task_launch_policy is None or task_launch_policy.mode == "auto") and not (
        range_one_to_one
    ):
        key = kernel.ensure_compiled(*injected_args)
    else:
        if task_launch_policy is None:
            from taichi_forge.lang.task_launch import TaskLaunchPolicy

            task_launch_policy = TaskLaunchPolicy.auto()
        key = kernel._ensure_compiled_with_task_launch_policy(
            task_launch_policy,
            *injected_args,
            range_one_to_one=range_one_to_one,
        )
        kernel._validate_task_launch_policy_specialization(key, task_launch_policy)
    return kernel.compiled_kernels[key]


def _bounded_semantic_kernel_identity(kernel_cpp):
    raw = impl.get_runtime().prog._kernel_gpu_semantics_snapshot(kernel_cpp)
    identity = str(
        raw.get("logical_kernel_identity") or raw.get("kernel_identity") or ""
    )
    if not identity:
        raise TaichiRuntimeError(
            "Graph bounded recipe source has no stable compiler identity"
        )
    return identity


def _graph_shared_staged_2d_contract(raw, task, args):
    if (
        task.get("range_mapping") != "shared_tiled_2d_one_to_one"
        or isinstance(task.get("selected_block_size"), bool)
        or not isinstance(task.get("selected_block_size"), int)
        or task["selected_block_size"] <= 0
        or not isinstance(task.get("static_shared_bytes"), int)
        or task["static_shared_bytes"] <= 0
        or task["static_shared_bytes"] > 32 * 1024
    ):
        raise TaichiRuntimeError(
            "Graph two-dimensional shared-staged task did not materialize "
            "its exact BLS mapping"
        )
    iteration_shape = tuple(task.get("staged_iteration_shape", ()))
    iteration_origin = tuple(task.get("staged_iteration_origin", ()))
    tile_shape = tuple(task.get("staged_tile_shape", ()))
    if (
        len(iteration_shape) != 2
        or len(iteration_origin) != 2
        or len(tile_shape) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (*iteration_shape, *iteration_origin, *tile_shape)
        )
        or any(value <= 0 for value in (*iteration_shape, *tile_shape))
        or tile_shape[0] * tile_shape[1] != task["selected_block_size"]
    ):
        raise TaichiRuntimeError(
            "Graph two-dimensional shared-staged task has no exact domain "
            "and tile policy"
        )
    staged_indices = tuple(task.get("staged_external_arg_indices", ()))
    halo_lows = tuple(tuple(axis) for axis in task.get("staged_halo_lows_nd", ()))
    halo_highs = tuple(tuple(axis) for axis in task.get("staged_halo_highs_nd", ()))
    access_offsets = tuple(
        tuple(tuple(offset) for offset in source)
        for source in task.get("staged_access_offsets", ())
    )
    byte_offsets = tuple(task.get("staged_byte_offsets", ()))
    element_bytes = tuple(task.get("staged_element_bytes", ()))
    scalar_bytes = tuple(task.get("staged_scalar_bytes", ()))
    element_shapes = tuple(
        tuple(shape) for shape in task.get("staged_element_shapes", ())
    )
    if (
        not 1 <= len(staged_indices) <= 2
        or any(
            len(values) != len(staged_indices)
            for values in (
                halo_lows,
                halo_highs,
                access_offsets,
                byte_offsets,
                element_bytes,
                scalar_bytes,
                element_shapes,
            )
        )
        or staged_indices != tuple(sorted(set(staged_indices)))
        or any(not 0 <= index < len(args) for index in staged_indices)
        or any(len(bounds) != 2 for bounds in (*halo_lows, *halo_highs))
        or any(
            len(offset) != 2
            for source_offsets in access_offsets
            for offset in source_offsets
        )
    ):
        raise TaichiRuntimeError(
            "Graph two-dimensional shared-staged task has incomplete ordered "
            "tile metadata"
        )

    metadata = raw["graph_metadata"]
    domain = metadata.get("iteration_domain", {})
    if (
        not metadata.get("available", False)
        or metadata.get("opaque", True)
        or metadata.get("blocker")
        or domain.get("kind") != "constant_range"
        or tuple(domain.get("logical_shape", ())) != iteration_shape
        or tuple(domain.get("logical_origin", ())) != iteration_origin
        or domain.get("begin") != 0
        or domain.get("end") != iteration_shape[0] * iteration_shape[1]
        or task.get("constant_range_size") != domain.get("end")
    ):
        raise TaichiRuntimeError(
            "Graph two-dimensional shared staging requires one compiler-proven "
            "canonical flattened ndrange"
        )

    effects = {}
    for effect in metadata.get("effects", ()):
        path = tuple(int(index) for index in effect.get("arg_id", ()))
        if effect.get("resource_kind") != "argument" or len(path) != 1:
            raise TaichiRuntimeError(
                "Graph two-dimensional shared staging supports top-level "
                "ndarray effects only"
            )
        index = path[0]
        if not 0 <= index < len(args):
            raise TaichiRuntimeError(
                "Graph shared-staged effect argument is outside the symbolic ABI"
            )
        symbolic = args[index]
        symbolic_element_shape = tuple(symbolic.element_shape)
        if (
            getattr(symbolic, "tag", None) != ArgKind.NDARRAY
            or symbolic.field_dim != 2
            or len(symbolic_element_shape) > 2
            or any(extent <= 0 for extent in symbolic_element_shape)
        ):
            raise TaichiRuntimeError(
                "Graph two-dimensional shared-staged effects require rank-two "
                "scalar, vector, or matrix ndarrays"
            )
        effects[index] = effect

    logical_output_count = iteration_shape[0] * iteration_shape[1]

    def staged_record_count(halo_low, halo_high):
        total = 0
        for outer in range(0, iteration_shape[0], tile_shape[0]):
            outer_records = (
                min(tile_shape[0], iteration_shape[0] - outer)
                + halo_high[0]
                - halo_low[0]
            )
            for inner in range(0, iteration_shape[1], tile_shape[1]):
                inner_records = (
                    min(tile_shape[1], iteration_shape[1] - inner)
                    + halo_high[1]
                    - halo_low[1]
                )
                total += outer_records * inner_records
        return total

    staged_sources = []
    source_names = []
    tile_end = 0
    for (
        index,
        halo_low,
        halo_high,
        offsets,
        byte_offset,
        reported_element_bytes,
        reported_scalar_bytes,
        reported_element_shape,
    ) in zip(
        staged_indices,
        halo_lows,
        halo_highs,
        access_offsets,
        byte_offsets,
        element_bytes,
        scalar_bytes,
        element_shapes,
    ):
        staged_effect = effects.get(index)
        footprint = (
            None if staged_effect is None else staged_effect.get("footprint", {})
        )
        proven_offsets = tuple(
            tuple(int(value) for value in offset)
            for offset in (
                () if footprint is None else footprint.get("affine_index_offsets", ())
            )
        )
        proven_halo = tuple(
            tuple(int(value) for value in axis)
            for axis in (() if footprint is None else footprint.get("halo", ()))
        )
        if (
            staged_effect is None
            or staged_effect.get("access") != "read"
            or footprint.get("pattern") != "stencil"
            or footprint.get("iteration_rank") != 2
            or tuple(offsets) != tuple(sorted(set(offsets)))
            or tuple(offsets) != proven_offsets
            or proven_halo != tuple(zip(halo_low, halo_high))
            or len(offsets) < 2
            or any(iteration_origin[axis] + halo_low[axis] < 0 for axis in range(2))
        ):
            raise TaichiRuntimeError(
                "Graph two-dimensional shared-staged input does not match its "
                "compiler-proven stencil footprint"
            )
        symbolic = args[index]
        source_name = symbolic.name
        source_dtype = symbolic.dtype()
        scalar_stride = _ti_core.data_type_size(source_dtype)
        alignment = _ti_core.data_type_alignment(source_dtype)
        source_element_shape = tuple(symbolic.element_shape)
        lane_count = math.prod(source_element_shape) if source_element_shape else 1
        record_stride = scalar_stride * lane_count
        tile_extents = tuple(
            tile_shape[axis] + halo_high[axis] - halo_low[axis] for axis in range(2)
        )
        tile_elements = tile_extents[0] * tile_extents[1]
        tile_bytes = tile_elements * record_stride
        expected_offset = (tile_end + alignment - 1) // alignment * alignment
        if (
            scalar_stride not in (2, 4, 8)
            or lane_count > 16
            or reported_scalar_bytes != scalar_stride
            or reported_element_shape != source_element_shape
            or reported_element_bytes != record_stride
            or byte_offset != expected_offset
            or byte_offset % alignment
            or tile_elements <= 0
            or byte_offset + tile_bytes > task["static_shared_bytes"]
        ):
            raise TaichiRuntimeError(
                "Graph two-dimensional shared-staged ordered tile layout is "
                "not exact"
            )
        direct_input_records = logical_output_count * len(offsets)
        staged_input_records = staged_record_count(halo_low, halo_high)
        tile_end = byte_offset + tile_bytes
        source_names.append(source_name)
        staged_sources.append(
            {
                "arg_index": index,
                "arg_name": source_name,
                "iteration_shape": iteration_shape,
                "iteration_origin": iteration_origin,
                "tile_shape": tile_shape,
                "halo_low": tuple(halo_low),
                "halo_high": tuple(halo_high),
                "element_bytes": record_stride,
                "scalar_bytes": scalar_stride,
                "element_shape": source_element_shape,
                "lane_count": lane_count,
                "alignment": alignment,
                "byte_offset": byte_offset,
                "tile_extents": tile_extents,
                "tile_elements": tile_elements,
                "tile_bytes": tile_bytes,
                "access_offsets": tuple(offsets),
                "logical_output_count": logical_output_count,
                "direct_input_records": direct_input_records,
                "staged_input_records": staged_input_records,
                "direct_input_bytes": direct_input_records * record_stride,
                "staged_input_bytes": staged_input_records * record_stride,
            }
        )
    if tile_end != task["static_shared_bytes"]:
        raise TaichiRuntimeError(
            "Graph two-dimensional shared-staged aggregate shared-memory "
            "budget is not exact"
        )

    def layout_requirement(name, minimum_shape, stride, alignment, shape):
        return (
            name,
            tuple(minimum_shape),
            stride,
            alignment,
            tuple(shape),
            "row_major_2d",
        )

    layout_requirements = [
        layout_requirement(
            source["arg_name"],
            tuple(
                iteration_origin[axis]
                + iteration_shape[axis]
                + source["halo_high"][axis]
                for axis in range(2)
            ),
            source["element_bytes"],
            source["alignment"],
            source["element_shape"],
        )
        for source in staged_sources
    ]
    output_names = []
    for index, effect in effects.items():
        if index in staged_indices:
            continue
        footprint = effect.get("footprint", {})
        if effect.get("access") == "read" and footprint.get("pattern") == "stencil":
            continue
        offsets = tuple(
            tuple(value for value in offset)
            for offset in footprint.get("affine_index_offsets", ())
        )
        if (
            effect.get("access") != "write"
            or footprint.get("pattern") != "exact_pointwise"
            or offsets != ((0, 0),)
        ):
            raise TaichiRuntimeError(
                "Graph two-dimensional shared-staged outputs must be proven "
                "write-only and pointwise"
            )
        symbolic = args[index]
        output_name = symbolic.name
        output_names.append(output_name)
        output_scalar_stride = _ti_core.data_type_size(symbolic.dtype())
        output_alignment = _ti_core.data_type_alignment(symbolic.dtype())
        output_element_shape = tuple(symbolic.element_shape)
        output_lane_count = (
            math.prod(output_element_shape) if output_element_shape else 1
        )
        output_stride = output_scalar_stride * output_lane_count
        if output_scalar_stride not in (2, 4, 8) or output_lane_count > 16:
            raise TaichiRuntimeError(
                "Graph two-dimensional shared-staged outputs use an unsupported "
                "record ABI"
            )
        layout_requirements.append(
            layout_requirement(
                output_name,
                tuple(
                    iteration_origin[axis] + iteration_shape[axis] for axis in range(2)
                ),
                output_stride,
                output_alignment,
                output_element_shape,
            )
        )
    if not output_names or set(source_names) & set(output_names):
        raise TaichiRuntimeError(
            "Graph two-dimensional shared-staged recipe requires distinct "
            "write-only outputs"
        )
    return (
        tuple(
            sorted(
                (source_name, output_name)
                for source_name in source_names
                for output_name in output_names
            )
        ),
        tuple(sorted(layout_requirements)),
        tuple(staged_sources),
    )


def _graph_shared_staged_contract(kernel_cpp, args):
    raw = impl.get_runtime().prog._kernel_gpu_semantics_snapshot(kernel_cpp)
    if _backend_name(raw["backend"]) != "cuda":
        raise TaichiRuntimeError(
            "Graph shared-staged memory recipes require the CUDA backend"
        )
    staged_tasks = tuple(
        task
        for task in raw["tasks"]
        if task.get("requested_memory_strategy")
        in ("shared_staged_1d", "shared_staged_2d")
    )
    if len(staged_tasks) != 1:
        raise TaichiRuntimeError(
            "Graph shared-staged recipe must materialize exactly one staged task"
        )
    task = staged_tasks[0]
    if task.get("requested_memory_strategy") == "shared_staged_2d":
        return _graph_shared_staged_2d_contract(raw, task, args)
    if (
        task.get("task_type") != "range_for"
        or task.get("range_mapping") != "shared_tiled_one_to_one"
        or isinstance(task.get("selected_block_size"), bool)
        or not isinstance(task.get("selected_block_size"), int)
        or task["selected_block_size"] <= 0
        or not isinstance(task.get("static_shared_bytes"), int)
        or task["static_shared_bytes"] <= 0
        or task["static_shared_bytes"] > 32 * 1024
    ):
        raise TaichiRuntimeError(
            "Graph shared-staged task did not materialize its exact BLS mapping"
        )

    staged_indices = tuple(task.get("staged_external_arg_indices", ()))
    halo_lows = tuple(task.get("staged_halo_lows", ()))
    halo_highs = tuple(task.get("staged_halo_highs", ()))
    byte_offsets = tuple(task.get("staged_byte_offsets", ()))
    element_bytes = tuple(task.get("staged_element_bytes", ()))
    scalar_bytes = tuple(task.get("staged_scalar_bytes", ()))
    element_shapes = tuple(
        tuple(shape) for shape in task.get("staged_element_shapes", ())
    )
    if (
        not 1 <= len(staged_indices) <= 2
        or any(
            len(values) != len(staged_indices)
            for values in (
                halo_lows,
                halo_highs,
                byte_offsets,
                element_bytes,
                scalar_bytes,
                element_shapes,
            )
        )
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for values in (
                staged_indices,
                halo_lows,
                halo_highs,
                byte_offsets,
                element_bytes,
                scalar_bytes,
            )
            for value in values
        )
        or any(
            len(shape) > 2
            or any(
                isinstance(extent, bool) or not isinstance(extent, int) or extent <= 0
                for extent in shape
            )
            for shape in element_shapes
        )
        or staged_indices != tuple(sorted(set(staged_indices)))
        or any(not 0 <= index < len(args) for index in staged_indices)
        or any(low >= high for low, high in zip(halo_lows, halo_highs))
    ):
        raise TaichiRuntimeError(
            "Graph shared-staged task has incomplete ordered tile metadata"
        )
    if (
        task.get("staged_external_arg_index") != staged_indices[0]
        or task.get("staged_halo_low") != halo_lows[0]
        or task.get("staged_halo_high") != halo_highs[0]
    ):
        raise TaichiRuntimeError(
            "Graph shared-staged compatibility metadata disagrees with its layout"
        )

    metadata = raw["graph_metadata"]
    if (
        not metadata.get("available", False)
        or metadata.get("opaque", True)
        or metadata.get("blocker")
    ):
        raise TaichiRuntimeError(
            "Graph shared-staged recipe requires proven pre-offload effects"
        )
    domain = metadata.get("iteration_domain", {})
    domain_begin = domain.get("begin")
    domain_end = domain.get("end")
    if (
        domain.get("kind") != "constant_range"
        or isinstance(domain_begin, bool)
        or not isinstance(domain_begin, int)
        or isinstance(domain_end, bool)
        or not isinstance(domain_end, int)
        or domain_begin < 0
        or domain_begin >= domain_end
        or task.get("constant_range_size") != domain_end - domain_begin
    ):
        raise TaichiRuntimeError(
            "Graph shared-staged recipe requires one exact non-empty constant domain"
        )
    effects = {}
    for effect in metadata.get("effects", ()):
        path = tuple(int(index) for index in effect.get("arg_id", ()))
        if effect.get("resource_kind") != "argument" or len(path) != 1:
            raise TaichiRuntimeError(
                "Graph shared-staged recipe supports top-level ndarray effects only"
            )
        index = path[0]
        if not 0 <= index < len(args):
            raise TaichiRuntimeError(
                "Graph shared-staged effect argument is outside the symbolic ABI"
            )
        symbolic = args[index]
        symbolic_element_shape = tuple(symbolic.element_shape)
        if (
            getattr(symbolic, "tag", None) != ArgKind.NDARRAY
            or symbolic.field_dim != 1
            or len(symbolic_element_shape) > 2
            or any(extent <= 0 for extent in symbolic_element_shape)
        ):
            raise TaichiRuntimeError(
                "Graph shared-staged effects require one-dimensional scalar, "
                "vector, or matrix ndarrays"
            )
        effects[index] = effect

    staged_sources = []
    source_names = []
    tile_end = 0
    for (
        index,
        halo_low,
        halo_high,
        byte_offset,
        reported_element_bytes,
        reported_scalar_bytes,
        reported_element_shape,
    ) in zip(
        staged_indices,
        halo_lows,
        halo_highs,
        byte_offsets,
        element_bytes,
        scalar_bytes,
        element_shapes,
    ):
        staged_effect = effects.get(index)
        footprint = (
            None if staged_effect is None else staged_effect.get("footprint", {})
        )
        try:
            halo = tuple(
                tuple(int(value) for value in axis)
                for axis in (() if footprint is None else footprint.get("halo", ()))
            )
        except (TypeError, ValueError):
            halo = ()
        try:
            access_offsets = tuple(
                int(value)
                for value in (
                    () if footprint is None else footprint.get("affine_offsets", ())
                )
            )
        except (TypeError, ValueError):
            access_offsets = ()
        if (
            staged_effect is None
            or staged_effect.get("access") != "read"
            or footprint.get("pattern") != "stencil"
            or halo != ((halo_low, halo_high),)
            or access_offsets != tuple(sorted(set(access_offsets)))
            or len(access_offsets) < 2
            or access_offsets[0] != halo_low
            or access_offsets[-1] != halo_high
            or domain_begin + halo_low < 0
        ):
            raise TaichiRuntimeError(
                "Graph shared-staged input does not match its proven stencil "
                "footprint"
            )

        symbolic = args[index]
        source_name = symbolic.name
        source_dtype = symbolic.dtype()
        scalar_stride = _ti_core.data_type_size(source_dtype)
        alignment = _ti_core.data_type_alignment(source_dtype)
        source_element_shape = tuple(symbolic.element_shape)
        lane_count = 1
        for extent in source_element_shape:
            lane_count *= extent
        record_stride = scalar_stride * lane_count
        if (
            scalar_stride not in (2, 4, 8)
            or lane_count > 16
            or reported_scalar_bytes != scalar_stride
            or reported_element_shape != source_element_shape
            or reported_element_bytes != record_stride
        ):
            raise TaichiRuntimeError(
                "Graph shared-staged input must publish its exact primitive-lane "
                "and packed-record layout"
            )
        expected_offset = (tile_end + alignment - 1) // alignment * alignment
        tile_elements = task["selected_block_size"] + halo_high - halo_low
        tile_bytes = tile_elements * record_stride
        logical_output_count = domain_end - domain_begin
        block_count = (logical_output_count + task["selected_block_size"] - 1) // task[
            "selected_block_size"
        ]
        direct_input_records = logical_output_count * len(access_offsets)
        staged_input_records = logical_output_count + block_count * (
            halo_high - halo_low
        )
        if (
            byte_offset != expected_offset
            or byte_offset % alignment
            or tile_elements <= 0
            or byte_offset + tile_bytes > task["static_shared_bytes"]
        ):
            raise TaichiRuntimeError(
                "Graph shared-staged ordered tile layout is not exact"
            )
        tile_end = byte_offset + tile_bytes
        source_names.append(source_name)
        staged_sources.append(
            {
                "arg_index": index,
                "arg_name": source_name,
                "halo_low": halo_low,
                "halo_high": halo_high,
                "element_bytes": record_stride,
                "scalar_bytes": scalar_stride,
                "element_shape": source_element_shape,
                "lane_count": lane_count,
                "alignment": alignment,
                "byte_offset": byte_offset,
                "tile_elements": tile_elements,
                "tile_bytes": tile_bytes,
                "access_offsets": access_offsets,
                "logical_output_count": logical_output_count,
                "direct_input_records": direct_input_records,
                "staged_input_records": staged_input_records,
                "direct_input_bytes": direct_input_records * record_stride,
                "staged_input_bytes": staged_input_records * record_stride,
            }
        )
    if tile_end != task["static_shared_bytes"]:
        raise TaichiRuntimeError(
            "Graph shared-staged aggregate shared-memory budget is not exact"
        )

    output_names = []

    def layout_requirement(name, minimum_records, record_stride, alignment, shape):
        if shape:
            return (
                name,
                minimum_records,
                record_stride,
                alignment,
                tuple(shape),
                "aos",
            )
        return (name, minimum_records, record_stride, alignment)

    layout_requirements = [
        layout_requirement(
            source["arg_name"],
            domain_end + source["halo_high"],
            source["element_bytes"],
            source["alignment"],
            source["element_shape"],
        )
        for source in staged_sources
    ]
    for index, effect in effects.items():
        if index in staged_indices:
            continue
        input_footprint = effect.get("footprint", {})
        if (
            effect.get("access") == "read"
            and input_footprint.get("pattern") == "stencil"
        ):
            # A generated GraphMemory recipe may stage only a proven subset
            # of independent stencil inputs. The remaining inputs retain
            # their ordinary global-load path and therefore need no staged
            # layout or alias contract.
            continue
        output_footprint = effect.get("footprint", {})
        if (
            effect.get("access") != "write"
            or output_footprint.get("pattern") != "exact_pointwise"
            or tuple(output_footprint.get("affine_offsets", ())) != (0,)
        ):
            raise TaichiRuntimeError(
                "Graph shared-staged outputs must be proven write-only and pointwise"
            )
        output_name = args[index].name
        output_names.append(output_name)
        output_dtype = args[index].dtype()
        output_scalar_stride = _ti_core.data_type_size(output_dtype)
        output_alignment = _ti_core.data_type_alignment(output_dtype)
        output_element_shape = tuple(args[index].element_shape)
        output_lane_count = 1
        for extent in output_element_shape:
            output_lane_count *= extent
        output_stride = output_scalar_stride * output_lane_count
        if output_scalar_stride not in (2, 4, 8) or output_lane_count > 16:
            raise TaichiRuntimeError(
                "Graph shared-staged outputs must use bounded two-, four-, or "
                "eight-byte primitive-lane records"
            )
        layout_requirements.append(
            layout_requirement(
                output_name,
                domain_end,
                output_stride,
                output_alignment,
                output_element_shape,
            )
        )
    if not output_names or set(source_names) & set(output_names):
        raise TaichiRuntimeError(
            "Graph shared-staged recipe requires distinct write-only outputs"
        )
    return (
        tuple(
            sorted(
                (source_name, output_name)
                for source_name in source_names
                for output_name in output_names
            )
        ),
        tuple(sorted(layout_requirements)),
        tuple(staged_sources),
    )


def _kernel_task_manifests(kernel_cpp):
    from taichi_forge.lang.task_manifest import OffloadedTaskManifest

    raw = impl.get_runtime().prog._kernel_task_manifest(kernel_cpp)
    return tuple(OffloadedTaskManifest._from_core(item) for item in raw)


def _graph_memory_symbolic_abi(args):
    return tuple(
        {
            "name": str(arg.name),
            "kind": str(getattr(arg, "tag", "")),
            "dtype": str(arg.dtype()),
            "rank": int(getattr(arg, "field_dim", 0)),
            "element_shape": tuple(getattr(arg, "element_shape", ())),
        }
        for arg in args
    )


def _graph_memory_candidate_preflight(kernel_cpp, args):
    """Reject impossible shared-stage shapes before native specialization."""

    raw = impl.get_runtime().prog._kernel_gpu_semantics_snapshot(kernel_cpp)
    metadata = raw.get("graph_metadata", {})
    if (
        _backend_name(raw.get("backend", "")) != "cuda"
        or not metadata.get("available", False)
        or metadata.get("opaque", True)
        or metadata.get("blocker")
    ):
        return False, "candidate has no proven CUDA pre-offload effects", (), (), ()
    domain = metadata.get("iteration_domain", {})
    begin = domain.get("begin")
    end = domain.get("end")
    if (
        domain.get("kind") != "constant_range"
        or isinstance(begin, bool)
        or not isinstance(begin, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or begin < 0
        or begin >= end
    ):
        return False, "candidate has no exact non-empty constant range", (), (), ()

    logical_shape = tuple(domain.get("logical_shape", ()))
    logical_origin = tuple(domain.get("logical_origin", ()))
    if logical_shape or logical_origin:
        if (
            len(logical_shape) != 2
            or len(logical_origin) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (*logical_shape, *logical_origin)
            )
            or any(value <= 0 for value in logical_shape)
            or logical_shape[0] * logical_shape[1] != end - begin
        ):
            return False, "candidate has invalid canonical ndrange metadata", (), (), ()
        expected_rank = 2
    else:
        expected_rank = 1

    staged_sources = set()
    outputs = set()
    for effect in metadata.get("effects", ()):
        path = tuple(effect.get("arg_id", ()))
        if (
            effect.get("resource_kind") != "argument"
            or len(path) != 1
            or isinstance(path[0], bool)
            or not isinstance(path[0], int)
            or not 0 <= path[0] < len(args)
        ):
            return (
                False,
                "candidate effects are not top-level symbolic arguments",
                (),
                (),
                (),
            )
        index = path[0]
        symbolic = args[index]
        if (
            getattr(symbolic, "tag", None) != ArgKind.NDARRAY
            or symbolic.field_dim != expected_rank
        ):
            return (
                False,
                "candidate effects do not match the compiler-proven ndarray rank",
                (),
                (),
                (),
            )
        footprint = effect.get("footprint", {})
        access = effect.get("access")
        pattern = footprint.get("pattern")
        if expected_rank == 1:
            offsets = tuple(footprint.get("affine_offsets", ()))
        else:
            offsets = tuple(
                tuple(offset) for offset in footprint.get("affine_index_offsets", ())
            )
        if access == "read" and pattern == "stencil":
            halo = tuple(tuple(axis) for axis in footprint.get("halo", ()))
            if (
                footprint.get("iteration_rank") != expected_rank
                or len(set(offsets)) < 2
                or len(halo) != expected_rank
                or any(len(axis) != 2 for axis in halo)
            ):
                return False, "staged input has incomplete affine offsets", (), (), ()
            staged_sources.add(index)
            continue
        zero = (0,) if expected_rank == 1 else ((0, 0),)
        if access == "write" and pattern == "exact_pointwise" and offsets == zero:
            outputs.add(index)
            continue
        return (
            False,
            "candidate effects do not form a stencil-to-pointwise route",
            (),
            (),
            (),
        )
    if not 1 <= len(staged_sources) <= 2:
        return False, "candidate requires one or two proven stencil inputs", (), (), ()
    if not outputs or staged_sources.intersection(outputs):
        return False, "candidate requires distinct write-only outputs", (), (), ()
    return (
        True,
        "",
        tuple(sorted(staged_sources)),
        logical_shape,
        logical_origin,
    )


def _graph_memory_recipe_manifest(
    plan,
    manifests,
    args,
    label,
    *,
    strategy,
    memory_disjoint_pairs=(),
    memory_layout_requirements=(),
    staged_sources=(),
):
    return _GraphMemoryRecipeManifest.from_payload(
        {
            "strategy": strategy,
            "dispatch_label": label,
            "symbolic_abi": _graph_memory_symbolic_abi(args),
            "semantic_kernel_identity": plan.semantic_kernel_identity,
            "offload_plan_identity": plan.identity,
            "offload_compilation_identity": plan.compilation_identity,
            "offload_plan": plan.stable_payload,
            "materialized_tasks": tuple(asdict(task) for task in manifests),
            "memory_disjoint_pairs": tuple(memory_disjoint_pairs),
            "memory_layout_requirements": tuple(memory_layout_requirements),
            "staged_sources": tuple(staged_sources),
        }
    )


class _GraphMemoryRecipeSource:
    """Frozen Graph-definition lineage with a lazy physical candidate cache."""

    def __init__(self, kernel_fn, args, label, baseline_kernel_cpp):
        self.kernel_fn = kernel_fn
        self.args = tuple(args)
        self.label = label
        self.baseline_kernel_cpp = baseline_kernel_cpp
        self.baseline_manifests = None
        self.baseline_plan = None
        self.direct_manifest = None
        self.selected_recipe_id = ""
        self._definition_lock = threading.Lock()
        self._candidates = None
        self._candidate_failures = {}
        self._candidate_lock = threading.Lock()

    @classmethod
    def try_create(cls, kernel_fn, args, label, baseline_kernel_cpp, template_args):
        if impl.current_cfg().arch != _ti_core.Arch.cuda or template_args:
            return None
        if isinstance(kernel_fn, kernel_impl._OffloadExecutionPlanBinding):
            return None
        if not hasattr(kernel_fn, "with_launch_policy"):
            return None
        ndarray_names = {
            arg.name for arg in args if getattr(arg, "tag", None) == ArgKind.NDARRAY
        }

        if len(ndarray_names) < 2:
            # A shared-staged recipe requires at least one read-only source and
            # one distinct write-only output. Reject the impossible shape from
            # frozen symbolic ABI facts before asking the compiler to build a
            # shared-memory specialization.
            return None

        return cls(kernel_fn, args, label, baseline_kernel_cpp)

    def _prepare_definition(self):
        if self.direct_manifest is not None:
            return
        with self._definition_lock:
            if self.direct_manifest is not None:
                return
            from taichi_forge.lang._offload_execution_plan import (
                _OffloadExecutionPlan,
            )

            manifests = _kernel_task_manifests(self.baseline_kernel_cpp)
            plan = _OffloadExecutionPlan.from_task_manifests(manifests)
            ranges = tuple(task for task in plan.tasks if task.task_kind == "range_for")
            if len(ranges) != 1:
                raise ValueError("GraphMemory requires exactly one range-for task")
            direct_manifest = _graph_memory_recipe_manifest(
                plan,
                manifests,
                self.args,
                self.label,
                strategy="direct",
            )
            self.baseline_manifests = manifests
            self.baseline_plan = plan
            self.direct_manifest = direct_manifest
            if not self.selected_recipe_id:
                self.selected_recipe_id = direct_manifest.recipe_id

    def _build_candidate(
        self,
        block_dim,
        source_indices,
        *,
        domain_shape=(),
        domain_origin=(),
        tile_shape=(),
    ):
        from taichi_forge.lang._offload_execution_plan import (
            _bind_offload_execution_plan,
        )

        self._prepare_definition()
        range_task = next(
            task for task in self.baseline_plan.tasks if task.task_kind == "range_for"
        )
        strategy = "shared_staged_2d" if tile_shape else "shared_staged_1d"
        staged_plan = self.baseline_plan.replace_task(
            range_task.task_index,
            workgroup_size=block_dim,
            memory_strategy=strategy,
            memory_source_arg_indices=tuple(source_indices),
            memory_domain_shape=tuple(domain_shape),
            memory_domain_origin=tuple(domain_origin),
            memory_tile_shape=tuple(tile_shape),
        )
        staged_binding = _bind_offload_execution_plan(self.kernel_fn, staged_plan)
        staged_kernel_cpp = gen_cpp_kernel(
            staged_binding,
            self.args,
            allow_graph_memory_recipe=True,
        )
        contracts, layout_requirements, staged_sources = _graph_shared_staged_contract(
            staged_kernel_cpp,
            self.args,
        )
        manifests = _kernel_task_manifests(staged_kernel_cpp)
        staged_plan.validate_materialization(manifests)
        manifest = _graph_memory_recipe_manifest(
            staged_plan,
            manifests,
            self.args,
            self.label,
            strategy=strategy,
            memory_disjoint_pairs=contracts,
            memory_layout_requirements=layout_requirements,
            staged_sources=staged_sources,
        )
        return (
            manifest,
            staged_kernel_cpp,
            contracts,
            layout_requirements,
        )

    def candidates(self):
        if self._candidates is not None:
            return self._candidates
        with self._candidate_lock:
            if self._candidates is not None:
                return self._candidates
            candidates = []
            try:
                self._prepare_definition()
                (
                    eligible,
                    reason,
                    source_indices,
                    domain_shape,
                    domain_origin,
                ) = _graph_memory_candidate_preflight(
                    self.baseline_kernel_cpp, self.args
                )
                if eligible:
                    subsets = tuple(
                        subset
                        for size in range(1, len(source_indices) + 1)
                        for subset in itertools.combinations(source_indices, size)
                    )
                    if domain_shape:
                        configurations = tuple(
                            (
                                tile_shape[0] * tile_shape[1],
                                subset,
                                tile_shape,
                            )
                            for tile_shape in ((8, 8), (8, 16), (16, 16))
                            for subset in subsets
                        )
                    else:
                        # Preserve the former 128-thread/all-source candidate
                        # as the first generated neighbor for compatibility.
                        configurations = ((128, source_indices, ()),) + tuple(
                            (block_dim, subset, ())
                            for block_dim in (64, 128, 256)
                            for subset in subsets
                            if (block_dim, subset) != (128, source_indices)
                        )
                    for block_dim, subset, tile_shape in configurations:
                        try:
                            candidates.append(
                                self._build_candidate(
                                    block_dim,
                                    subset,
                                    domain_shape=domain_shape,
                                    domain_origin=domain_origin,
                                    tile_shape=tile_shape,
                                )
                            )
                        except (
                            ValueError,
                            RuntimeError,
                            TaichiCompilationError,
                            TaichiRuntimeError,
                        ) as error:
                            self._candidate_failures[
                                (block_dim, subset, tuple(tile_shape))
                            ] = (str(error).strip() or type(error).__name__)
                else:
                    self._candidate_failures[()] = reason
            except (
                ValueError,
                RuntimeError,
                TaichiCompilationError,
                TaichiRuntimeError,
            ) as error:
                self._candidate_failures[()] = (
                    str(error).strip() or type(error).__name__
                )
            self._candidates = tuple(candidates)
            return self._candidates

    def candidate(self):
        candidates = self.candidates()
        return None if not candidates else candidates[0]

    @property
    def candidate_failure(self):
        self.candidates()
        return "; ".join(
            f"{key or 'domain'}: {value}"
            for key, value in self._candidate_failures.items()
        )

    def manifests(self):
        candidates = self.candidates()
        if not candidates:
            return ()
        return (self.direct_manifest, *(candidate[0] for candidate in candidates))

    def materialize(self, requested_recipe_id, *, record_selection=True):
        try:
            self._prepare_definition()
        except (
            TypeError,
            ValueError,
            RuntimeError,
            TaichiCompilationError,
            TaichiRuntimeError,
        ) as error:
            raise TaichiRuntimeError(
                "requested GraphMemory recipe cannot prepare its source plan: "
                + (str(error).strip() or type(error).__name__)
            ) from None
        if requested_recipe_id == self.direct_manifest.recipe_id:
            if record_selection:
                self.selected_recipe_id = requested_recipe_id
            return self.baseline_kernel_cpp, (), ()
        candidates = self.candidates()
        if not candidates:
            reason = self.candidate_failure or "candidate is unsupported"
            raise TaichiRuntimeError(
                "requested GraphMemory recipe cannot be materialized: " + reason
            )
        candidate = next(
            (item for item in candidates if item[0].recipe_id == requested_recipe_id),
            None,
        )
        if candidate is None:
            raise TaichiRuntimeError(
                "requested GraphMemory recipe is absent from this Graph definition"
            )
        manifest, kernel_cpp, contracts, layout_requirements = candidate
        if record_selection:
            self.selected_recipe_id = requested_recipe_id

        return kernel_cpp, contracts, layout_requirements


def _graph_sparse_traversal_recipe_manifest(
    plan,
    manifests,
    args,
    label,
    strategy,
):
    listgen_tasks = tuple(
        {
            "task_index": task.task_index,
            "task_kind": task.task_type,
            "logical_task_id": task.logical_task_id,
            "snode_id": task.sparse_list_snode_id,
            "parent_snode_id": task.sparse_list_parent_snode_id,
            "parent_grid_bound": task.sparse_list_parent_grid_bound,
            "selected_grid": task.selected_grid_size,
            "actual_grid": task.actual_grid_size,
            "policy": task.requested_sparse_list_policy,
        }
        for task in manifests
        if task.sparse_list_op == 2
    )
    physical_stages = tuple(
        {
            "name": f"offload_{index}_{task.task_type}",
            "kind": task.task_type,
            "source_task_logical_ids": plan.materialized_task_lineage[index],
            "manifest": asdict(task),
        }
        for index, task in enumerate(manifests)
    )
    return _GraphSparseTraversalRecipeManifest.from_payload(
        {
            "strategy": strategy,
            "dispatch_label": label,
            "symbolic_abi": _graph_memory_symbolic_abi(args),
            "semantic_kernel_identity": plan.semantic_kernel_identity,
            "offload_plan_identity": plan.identity,
            "offload_compilation_identity": plan.compilation_identity,
            "materialized_tasks": tuple(asdict(task) for task in manifests),
            "source_task_lineage": plan.materialized_task_lineage,
            "listgen_tasks": listgen_tasks,
            "physical_stages": physical_stages,
            "native_graph_capture": False,
        }
    )


class _GraphSparseTraversalRecipeSource:
    """Graph-owned complete policy for compiled sparse listgen tasks."""

    def __init__(self, kernel_fn, args, label, baseline_kernel_cpp, manifests):
        self.kernel_fn = kernel_fn
        self.args = tuple(args)
        self.label = label
        self.baseline_kernel_cpp = baseline_kernel_cpp
        self.baseline_manifests = tuple(manifests)
        self.baseline_plan = None
        self.direct_manifest = None
        self.selected_recipe_id = ""
        self._recipe_source_key = ""
        self._definition_lock = threading.Lock()
        self._candidate = None
        self._candidate_prepared = False
        self._candidate_failure = ""
        self._candidate_lock = threading.Lock()

    @classmethod
    def try_create(cls, kernel_fn, args, label, baseline_kernel_cpp, template_args):
        if impl.current_cfg().arch != _ti_core.Arch.cuda or template_args:
            return None
        if isinstance(kernel_fn, kernel_impl._OffloadExecutionPlanBinding):
            return None
        if not hasattr(kernel_fn, "with_launch_policy"):
            return None
        try:
            manifests = _kernel_task_manifests(baseline_kernel_cpp)
        except (RuntimeError, TaichiRuntimeError):
            return None
        listgens = tuple(task for task in manifests if task.task_type == "listgen")
        if (
            not listgens
            or any(task.may_mutate_sparse_topology for task in manifests)
            or any(
                task.sparse_list_op != 2
                or task.sparse_list_snode_id < 0
                or task.sparse_list_parent_snode_id < 0
                or task.sparse_list_parent_grid_bound is None
                or task.sparse_list_parent_grid_bound <= 0
                or task.selected_grid_size is None
                or task.selected_grid_size <= 0
                for task in listgens
            )
            or not any(
                task.sparse_list_parent_grid_bound < task.selected_grid_size
                for task in listgens
            )
        ):
            return None
        return cls(kernel_fn, args, label, baseline_kernel_cpp, manifests)

    def _prepare_definition(self):
        if self.direct_manifest is not None:
            return
        with self._definition_lock:
            if self.direct_manifest is not None:
                return
            from taichi_forge.lang._offload_execution_plan import (
                _OffloadExecutionPlan,
            )

            plan = _OffloadExecutionPlan.from_task_manifests(self.baseline_manifests)
            direct_manifest = _graph_sparse_traversal_recipe_manifest(
                plan,
                self.baseline_manifests,
                self.args,
                self.label,
                "saturating",
            )
            self.baseline_plan = plan
            self.direct_manifest = direct_manifest
            if not self.selected_recipe_id:
                self.selected_recipe_id = direct_manifest.recipe_id

    def candidate(self):
        if self._candidate_prepared:
            return self._candidate
        with self._candidate_lock:
            if self._candidate_prepared:
                return self._candidate
            try:
                from taichi_forge.lang._offload_execution_plan import (
                    _bind_offload_execution_plan,
                )

                self._prepare_definition()
                plan = self.baseline_plan
                for task in self.baseline_manifests:
                    if task.sparse_list_op == 2:
                        plan = plan.replace_task(
                            task.task_index,
                            sparse_list_policy="parent_capacity_bound",
                        )
                binding = _bind_offload_execution_plan(self.kernel_fn, plan)
                kernel_cpp = gen_cpp_kernel(binding, self.args)
                manifests = _kernel_task_manifests(kernel_cpp)
                plan.validate_materialization(manifests)
                manifest = _graph_sparse_traversal_recipe_manifest(
                    plan,
                    manifests,
                    self.args,
                    self.label,
                    "parent_capacity_bound",
                )
                self._candidate = (manifest, kernel_cpp)
            except (
                TypeError,
                ValueError,
                RuntimeError,
                TaichiCompilationError,
                TaichiRuntimeError,
            ) as error:
                self._candidate_failure = str(error).strip() or type(error).__name__
            self._candidate_prepared = True
            return self._candidate

    @property
    def candidate_failure(self):
        self.candidate()
        return self._candidate_failure

    def manifests(self):
        self._prepare_definition()
        candidate = self.candidate()
        return (
            (self.direct_manifest,)
            if candidate is None
            else (self.direct_manifest, candidate[0])
        )

    def materialize(self, requested_recipe_id, *, record_selection=True):
        self._prepare_definition()
        if requested_recipe_id == self.direct_manifest.recipe_id:
            if record_selection:
                self.selected_recipe_id = requested_recipe_id
            return self.baseline_kernel_cpp
        candidate = self.candidate()
        if candidate is None or requested_recipe_id != candidate[0].recipe_id:
            reason = self.candidate_failure or "recipe is absent from this definition"
            raise TaichiRuntimeError(
                "requested sparse traversal recipe cannot be materialized: " + reason
            )
        if record_selection:
            self.selected_recipe_id = requested_recipe_id
        return candidate[1]


def _offload_fusion_group_candidates(tasks, *, limit=64):
    """Enumerate complete, disjoint phase-fusion choices in useful order."""

    tasks = tuple(tasks)
    result = []
    seen = set()

    def visit(index, groups):
        if len(result) >= limit:
            return
        if index == len(tasks):
            if groups and groups not in seen:
                seen.add(groups)
                result.append(groups)
            return
        for size in (4, 3, 2):
            end = index + size
            if end <= len(tasks) and all(
                task.task_kind == "range_for" for task in tasks[index:end]
            ):
                visit(end, groups + (tuple(range(index, end)),))
        visit(index + 1, groups)

    visit(0, ())
    return tuple(result)


def _graph_offload_disjoint_pairs(args):
    names = tuple(
        sorted(
            {arg.name for arg in args if getattr(arg, "tag", None) == ArgKind.NDARRAY}
        )
    )
    return tuple(itertools.combinations(names, 2))


def _graph_offload_fusion_recipe_manifest(
    plan,
    manifests,
    args,
    label,
    strategy,
    *,
    memory_disjoint_pairs=(),
):
    lineage = plan.materialized_task_lineage
    physical_stages = tuple(
        {
            "name": f"offload_{index}_{manifest.task_type}",
            "kind": manifest.task_type,
            "source_task_logical_ids": lineage[index],
            "manifest": asdict(manifest),
        }
        for index, manifest in enumerate(manifests)
    )
    return _GraphOffloadFusionRecipeManifest.from_payload(
        {
            "strategy": strategy,
            "dispatch_label": label,
            "symbolic_abi": _graph_memory_symbolic_abi(args),
            "semantic_kernel_identity": plan.semantic_kernel_identity,
            "offload_plan_identity": plan.identity,
            "offload_compilation_identity": plan.compilation_identity,
            "offload_plan": plan.stable_payload,
            "materialized_tasks": tuple(asdict(task) for task in manifests),
            "source_task_lineage": lineage,
            "fusion_groups": plan.fusion_groups,
            "memory_disjoint_pairs": tuple(memory_disjoint_pairs),
            "physical_stages": physical_stages,
        }
    )


class _GraphOffloadFusionRecipeSource:
    """Graph-owned compiler-IR topology domain for one semantic dispatch."""

    def __init__(self, kernel_fn, args, label, baseline_kernel_cpp, manifests):
        self.kernel_fn = kernel_fn
        self.args = tuple(args)
        self.label = label
        self.baseline_kernel_cpp = baseline_kernel_cpp
        self.baseline_manifests = tuple(manifests)
        self.baseline_plan = None
        self.direct_manifest = None
        self.selected_recipe_id = ""
        self._recipe_source_key = ""
        self._definition_lock = threading.Lock()
        self._candidates = None
        self._candidate_failures = {}
        self._candidate_lock = threading.Lock()

    @classmethod
    def try_create(cls, kernel_fn, args, label, baseline_kernel_cpp, template_args):
        if impl.current_cfg().arch != _ti_core.Arch.cuda or template_args:
            return None
        if isinstance(kernel_fn, kernel_impl._OffloadExecutionPlanBinding):
            return None
        if not hasattr(kernel_fn, "with_launch_policy"):
            return None
        try:
            manifests = _kernel_task_manifests(baseline_kernel_cpp)
        except (RuntimeError, TaichiRuntimeError):
            return None
        return cls(kernel_fn, args, label, baseline_kernel_cpp, manifests)

    def _prepare_definition(self):
        if self.direct_manifest is not None:
            return
        with self._definition_lock:
            if self.direct_manifest is not None:
                return
            from taichi_forge.lang._offload_execution_plan import (
                _OffloadExecutionPlan,
            )

            plan = _OffloadExecutionPlan.from_task_manifests(self.baseline_manifests)
            direct_manifest = _graph_offload_fusion_recipe_manifest(
                plan,
                self.baseline_manifests,
                self.args,
                self.label,
                "direct",
            )
            self.baseline_plan = plan
            self.direct_manifest = direct_manifest
            if not self.selected_recipe_id:
                self.selected_recipe_id = direct_manifest.recipe_id

    def _build_candidate(self, groups):
        from taichi_forge.lang._offload_execution_plan import (
            _bind_offload_execution_plan,
        )

        self._prepare_definition()
        plan = self.baseline_plan.with_fused_task_groups(*groups)
        binding = _bind_offload_execution_plan(self.kernel_fn, plan)
        kernel_cpp = gen_cpp_kernel(binding, self.args)
        manifests = _kernel_task_manifests(kernel_cpp)
        plan.validate_materialization(manifests)
        manifest = _graph_offload_fusion_recipe_manifest(
            plan,
            manifests,
            self.args,
            self.label,
            "exact_pointwise_phase_fusion",
            memory_disjoint_pairs=_graph_offload_disjoint_pairs(self.args),
        )
        return (
            manifest,
            kernel_cpp,
            tuple(tuple(pair) for pair in manifest.to_dict()["memory_disjoint_pairs"]),
        )

    def candidates(self):
        if self._candidates is not None:
            return self._candidates
        with self._candidate_lock:
            if self._candidates is not None:
                return self._candidates
            candidates = []
            try:
                self._prepare_definition()
                groups_domain = _offload_fusion_group_candidates(
                    self.baseline_plan.tasks
                )
                for groups in groups_domain:
                    try:
                        candidates.append(self._build_candidate(groups))
                    except (
                        ValueError,
                        RuntimeError,
                        TaichiCompilationError,
                        TaichiRuntimeError,
                    ) as error:
                        self._candidate_failures[groups] = (
                            str(error).strip() or type(error).__name__
                        )
            except (
                ValueError,
                RuntimeError,
                TaichiCompilationError,
                TaichiRuntimeError,
            ) as error:
                self._candidate_failures[()] = (
                    str(error).strip() or type(error).__name__
                )
            self._candidates = tuple(candidates)
            return self._candidates

    @property
    def candidate_failure(self):
        self.candidates()
        return "; ".join(
            f"{key or 'domain'}: {value}"
            for key, value in self._candidate_failures.items()
        )

    def manifests(self):
        candidates = self.candidates()
        if not candidates:
            return ()
        return (self.direct_manifest, *(candidate[0] for candidate in candidates))

    def materialize(self, requested_recipe_id, *, record_selection=True):
        self._prepare_definition()
        if requested_recipe_id == self.direct_manifest.recipe_id:
            if record_selection:
                self.selected_recipe_id = requested_recipe_id
            return self.baseline_kernel_cpp, ()
        candidate = next(
            (
                item
                for item in self.candidates()
                if item[0].recipe_id == requested_recipe_id
            ),
            None,
        )
        if candidate is None:
            reason = self.candidate_failure or "candidate is unsupported"
            raise TaichiRuntimeError(
                "requested Graph offload-fusion recipe cannot be materialized: "
                + reason
            )
        manifest, kernel_cpp, memory_disjoint_pairs = candidate
        if record_selection:
            self.selected_recipe_id = manifest.recipe_id
        return kernel_cpp, memory_disjoint_pairs


class _GraphBoundedRecipeSource:
    """Replayable device-bounded source with explicit physical selection."""

    def __init__(
        self,
        kernel_fn,
        args,
        *,
        extent,
        capacity,
        block_dim,
        block_mode,
        launch_state,
        template_args,
        label,
        selected_strategy,
        selected_block,
    ):
        self.kernel_fn = kernel_fn
        self.args = tuple(args)
        self.extent = extent
        self.capacity = int(capacity)
        self.block_dim = block_dim
        self.block_mode = block_mode
        self.launch_state = launch_state
        self.template_args = template_args
        self.label = label
        self.selected_strategy = selected_strategy
        self.selected_block = None if selected_block is None else int(selected_block)
        self._recipe_group_key = ""

    def materialize(self, builder, strategy):
        try:
            route, update_policy = {
                "logical_exact": ("auto", "auto"),
                "adaptive_per_node": ("device_update", "per_node"),
                "adaptive_grouped": ("device_update", "grouped_stateful"),
                "masked_capacity": ("masked_capacity", "auto"),
            }[strategy]
        except KeyError as error:
            raise TaichiRuntimeError(
                f"unknown complete Graph bounded strategy {strategy!r}"
            ) from error
        return builder.dispatch_bounded(
            self.kernel_fn,
            *self.args,
            extent=self.extent,
            capacity=self.capacity,
            block_dim=self.block_dim,
            block_mode=self.block_mode,
            physical_grid="auto",
            launch_state=self.launch_state,
            template_args=self.template_args,
            label=self.label,
            _recipe_route=route,
            _recipe_update_policy=update_policy,
            _register_recipe_source=False,
        )


def _require_bounded_symbolic_ndarray(value, role, dtype):
    if getattr(value, "tag", None) != ArgKind.NDARRAY:
        raise TaichiRuntimeError(
            f"Graph bounded dispatch {role} must be a symbolic ndarray argument"
        )
    if value.dtype() != dtype or value.field_dim != 1 or value.element_shape:
        raise TaichiRuntimeError(
            f"Graph bounded dispatch {role} must be a one-dimensional scalar "
            f"{dtype} ndarray argument"
        )
    return value


def _bounded_kernel_geometry(
    kernel_cpp,
    backend,
    *,
    allow_range_setup=False,
    require_one_to_one=False,
):
    raw = tuple(impl.get_runtime().prog._kernel_task_manifest(kernel_cpp))
    range_tasks = tuple(item for item in raw if item["task_type"] == "range_for")
    setup_tasks = tuple(item for item in raw if item["task_type"] == "serial")
    valid_setup = allow_range_setup and len(setup_tasks) == len(raw) - 1
    if len(range_tasks) != 1 or not (len(raw) == 1 or valid_setup):
        expected = (
            "range task with only scalar-range setup offloads"
            if allow_range_setup
            else "range task without serial offloads"
        )
        raise TaichiRuntimeError(
            "Graph bounded dispatch payload must compile to one parallel "
            + expected
            + "; got "
            + ", ".join(str(item["task_type"]) for item in raw)
        )
    selected = range_tasks[0]["selected_block_size"]
    if require_one_to_one and not allow_range_setup:
        required_mappings = (
            ("one_to_one", "device_bounded_grid_stride")
            if backend == "cuda"
            else ("one_to_one",)
        )
        if range_tasks[0].get("range_mapping") not in required_mappings:
            raise TaichiRuntimeError(
                f"{backend.upper()} bounded dispatch payload did not compile with "
                "the required backend-specialized range mapping"
            )
    if backend in ("cuda", "vulkan"):
        if selected is None or int(selected) <= 0:
            raise TaichiRuntimeError(
                "Graph bounded dispatch backend did not expose a selected block size"
            )
        return int(selected)
    return None


def _verify_bounded_host_range(kernel_cpp, args, count_arg):
    probe = _ti_core.GraphBuilder()
    probe.dispatch(kernel_cpp, args, "")
    compiled = probe.compile()
    fallback = _dispatch_ir_node(kernel_cpp, args)
    nodes = _compiled_dispatch_ir_nodes(compiled, (fallback,))
    expected = f"scalar_argument:{count_arg.name}"
    if len(nodes) != 1 or nodes[0].iteration_domain != expected:
        raise TaichiRuntimeError(
            "host-known bounded dispatch requires the payload's sole range "
            f"domain to be the scalar argument {count_arg.name!r}"
        )


def flatten_args(args):
    """Normalize symbolic args while accepting the pre-native Matrix adapter."""

    normalized_args = []
    for arg in args:
        if isinstance(arg, list):
            if not all(isinstance(row, (list, tuple)) for row in arg):
                raise TaichiRuntimeError(
                    "Legacy Matrix Graph arguments must be a nested list of "
                    "symbolic scalar arguments"
                )
            for row in arg:
                normalized_args.extend(row)
        else:
            normalized_args.append(arg)
    return normalized_args


def _runtime_arg_names(args):
    return {arg.name for arg in args}


def _normalize_dispatch_label(label):
    if label is None:
        return ""
    if not isinstance(label, str):
        raise TypeError("Graph dispatch label must be a string or None")
    return label


def _dispatch_ir_name(kernel_cpp):
    for attribute in ("name", "get_name"):
        value = getattr(kernel_cpp, attribute, None)
        if value is None:
            continue
        if callable(value):
            value = value()
        if value:
            return str(value)
    return type(kernel_cpp).__name__


def _dispatch_ir_node(
    kernel_cpp,
    args,
    *,
    dispatch_packet=None,
    dispatch_label="",
    bounded_domain=None,
):
    effects = []
    bindings = []
    for arg in args:
        tag = getattr(arg, "tag", None)
        kind = str(tag)
        access = (
            GraphAccess.READ
            if tag in (ArgKind.SCALAR, ArgKind.MATRIX, ArgKind.TEXTURE)
            else GraphAccess.READ_WRITE
        )
        effects.append(ResourceEffect(arg.name, access))
        bindings.append(RuntimeBinding(arg.name, kind))
    if dispatch_packet is not None:
        effects.append(ResourceEffect(dispatch_packet.name, GraphAccess.READ))
        bindings.append(RuntimeBinding(dispatch_packet.name, "indirect_dispatch"))
    # Until backend kernel access metadata is attached to this JIT record,
    # ndarray writes remain conservative and the node is not rewriteable.
    return DispatchNode(
        name=dispatch_label or _dispatch_ir_name(kernel_cpp),
        effects=tuple(effects),
        bindings=tuple(bindings),
        opaque=True,
        bounded_domain=bounded_domain,
        dispatch_label=dispatch_label,
    )


def _metadata_symbolic_arg(record, arg_id):
    try:
        path = tuple(int(index) for index in arg_id)
        symbolic_args = tuple(record["symbolic_args"])
    except (KeyError, TypeError, ValueError):
        return None
    if not path or path[0] < 0 or path[0] >= len(symbolic_args):
        return None
    name = symbolic_args[path[0]].get("name")
    return str(name) if name else None


def _metadata_iteration_domain(record):
    domain = record.get("iteration_domain", {})
    kind = domain.get("kind")
    if kind == "constant_range":
        return f"range:{int(domain['begin'])}:{int(domain['end'])}"
    if kind in ("external_tensor", "scalar_argument"):
        name = _metadata_symbolic_arg(record, domain.get("arg_id", ()))
        if name is None:
            return None
        if kind == "external_tensor":
            axis = int(domain.get("axis", -1))
            return f"external_tensor:{name}:axis:{axis}"
        return f"scalar_argument:{name}"
    return None


def _metadata_resource_effect(effect, record):
    kind = effect.get("resource_kind")
    if kind == "argument":
        resource = _metadata_symbolic_arg(record, effect.get("arg_id", ()))
        if resource is None:
            return None
        runtime_bound = True
    elif kind == "snode":
        tree_id = int(effect.get("snode_tree_id", -1))
        snode_id = int(effect.get("snode_id", -1))
        if tree_id < 0 or snode_id < 0:
            return None
        grad = ":grad" if bool(effect.get("is_grad", False)) else ""
        resource = f"snode:{tree_id}:{snode_id}{grad}"
        runtime_bound = False
    else:
        return None
    try:
        access = GraphAccess(str(effect["access"]))
    except (KeyError, ValueError):
        return None
    return ResourceEffect(resource, access, runtime_bound=runtime_bound)


def _compiled_dispatch_ir_nodes(compiled_graph, fallback_nodes):
    fallback_nodes = tuple(fallback_nodes)
    records = getattr(compiled_graph, "_dispatch_metadata", None)
    if records is None or len(records) != len(fallback_nodes):
        return fallback_nodes
    result = []
    for record, fallback in zip(records, fallback_nodes):
        if not isinstance(fallback, DispatchNode):
            # JIT-only provider commands carry their own NativeCallNode IR.
            # Core task metadata is kernel-shaped and must not erase or
            # reinterpret that provider effect/lifetime contract.
            result.append(fallback)
            continue
        side_effects = tuple(str(item) for item in record.get("side_effects", ()))
        raw_logical_dispatch_id = record.get("logical_dispatch_id")
        logical_dispatch_id = (
            fallback.logical_dispatch_id
            if raw_logical_dispatch_id is None
            else f"dispatch:{int(raw_logical_dispatch_id)}"
        )
        logical_kernel_identity = str(
            record.get("logical_kernel_identity") or fallback.logical_kernel_identity
        )
        iteration_domain = _metadata_iteration_domain(record)
        effects = tuple(
            _metadata_resource_effect(effect, record)
            for effect in record.get("effects", ())
        )
        proven = (
            bool(record.get("available", False))
            and not bool(record.get("opaque", True))
            and iteration_domain is not None
            and all(effect is not None for effect in effects)
        )
        if not proven:
            result.append(
                DispatchNode(
                    name=fallback.name,
                    effects=fallback.effects,
                    bindings=fallback.bindings,
                    synchronization=bool(record.get("synchronization", False)),
                    opaque=True,
                    elementwise=False,
                    side_effects=side_effects,
                    bounded_domain=fallback.bounded_domain,
                    dispatch_label=fallback.dispatch_label,
                    logical_dispatch_id=logical_dispatch_id,
                    logical_kernel_identity=logical_kernel_identity,
                    fusion_blocker=str(record.get("blocker") or "metadata_unavailable"),
                    memory_disjoint_pairs=fallback.memory_disjoint_pairs,
                    memory_layout_requirements=fallback.memory_layout_requirements,
                )
            )
            continue
        result.append(
            DispatchNode(
                name=fallback.name,
                effects=effects,
                bindings=fallback.bindings,
                iteration_domain=iteration_domain,
                synchronization=bool(record.get("synchronization", False)),
                opaque=False,
                elementwise=bool(record.get("elementwise", False)),
                side_effects=side_effects,
                bounded_domain=fallback.bounded_domain,
                dispatch_label=fallback.dispatch_label,
                logical_dispatch_id=logical_dispatch_id,
                logical_kernel_identity=logical_kernel_identity,
                memory_disjoint_pairs=fallback.memory_disjoint_pairs,
                memory_layout_requirements=fallback.memory_layout_requirements,
            )
        )
    return tuple(result)


class _AOTSequentialSnapshot:
    def __init__(self, dispatches, dispatch_labels=()):
        if dispatch_labels and len(dispatch_labels) != len(dispatches):
            raise TaichiRuntimeError(
                "Sequential dispatch labels do not match dispatches"
            )
        if not dispatch_labels:
            dispatch_labels = ("",) * len(dispatches)
        self._dispatches = tuple(
            (kernel_cpp, tuple(args)) for kernel_cpp, args in dispatches
        )
        self._dispatch_labels = tuple(dispatch_labels)

    def _dispatch_to(self, builder):
        for (kernel_cpp, args), label in zip(self._dispatches, self._dispatch_labels):
            builder.dispatch(kernel_cpp, args, label)


class _GraphSequentialRecipeSnapshot:
    """Frozen replay input for one non-structured Sequential append."""

    def __init__(self, sequence):
        self._items = tuple(sequence._items)
        self._ir_nodes = tuple(sequence._ir_nodes)
        self._dispatches = tuple(
            (kernel_cpp, tuple(args)) for kernel_cpp, args in sequence._dispatches
        )
        self._dispatch_labels = tuple(sequence._dispatch_labels)
        self._runtime_arg_names = frozenset(sequence._runtime_arg_names)
        self._recording_runtime_arg_names = frozenset(
            sequence._recording_runtime_arg_names
        )
        self._derived_runtime_arg_names = frozenset(sequence._derived_runtime_arg_names)
        self._fixed_runtime_args = dict(sequence._fixed_runtime_args)
        self._lifetime_leases = tuple(sequence._lifetime_leases)
        self._source_native_count = int(sequence._source_native_count)
        self._native_action_manifests = tuple(sequence._native_action_manifests)
        self._temporary_actions = tuple(sequence._temporary_actions)
        self._dispatch_count = int(sequence._dispatch_count)
        self._has_indirect_dispatch = bool(sequence._has_indirect_dispatch)
        self._structured_depth = int(sequence._structured_depth)

    def thaw(self):
        sequence = Sequential()
        sequence._items = list(self._items)
        sequence._ir_nodes = list(self._ir_nodes)
        sequence._dispatches = [
            (kernel_cpp, list(args)) for kernel_cpp, args in self._dispatches
        ]
        sequence._dispatch_labels = list(self._dispatch_labels)
        sequence._runtime_arg_names = set(self._runtime_arg_names)
        sequence._recording_runtime_arg_names = set(self._recording_runtime_arg_names)
        sequence._derived_runtime_arg_names = set(self._derived_runtime_arg_names)
        sequence._fixed_runtime_args = dict(self._fixed_runtime_args)
        sequence._lifetime_leases = list(self._lifetime_leases)
        sequence._source_native_count = self._source_native_count
        sequence._native_action_manifests = list(self._native_action_manifests)
        sequence._temporary_actions = list(self._temporary_actions)
        sequence._dispatch_count = self._dispatch_count
        sequence._has_indirect_dispatch = self._has_indirect_dispatch
        sequence._structured_depth = self._structured_depth
        return sequence


class Sequential:
    def __init__(self):
        self._dispatch_count = 0
        self._dispatches = []
        self._dispatch_labels = []
        self._items = []
        self._ir_nodes = []
        self._runtime_arg_names = set()
        self._recording_runtime_arg_names = set()
        self._derived_runtime_arg_names = set()
        self._fixed_runtime_args = {}
        self._lifetime_leases = []
        self._source_native_count = 0
        self._native_action_manifests = []
        self._temporary_actions = []
        self._has_indirect_dispatch = False
        self._structured_depth = 0

    def _bind_internal_ndarray(
        self,
        name,
        dtype,
        shape,
        *,
        exclusive_submission=False,
    ):
        """Reserve private address-stable storage for a provider sequence."""

        if not isinstance(name, str) or not name:
            raise TaichiRuntimeError(
                "Graph internal ndarray binding name must be nonempty"
            )
        if name in self._fixed_runtime_args:
            raise TaichiRuntimeError(
                f"Graph internal binding {name!r} is already defined"
            )
        shape = tuple(int(value) for value in shape)
        self._fixed_runtime_args[name] = InternalNdarrayRequirement(
            dtype,
            shape,
            _ti_core.data_type_size(dtype),
            bool(exclusive_submission),
        )
        return Arg(ArgKind.NDARRAY, name, dtype, ndim=len(shape))

    def private_ndarray(
        self,
        name,
        dtype,
        shape,
        *,
        exclusive_submission=True,
    ):
        """Declare private address-stable storage for this recorded region."""

        if isinstance(shape, int):
            shape = (shape,)
        return self._bind_internal_ndarray(
            name,
            dtype,
            shape,
            exclusive_submission=exclusive_submission,
        )

    def _bind_internal_scalar(self, name, dtype, value):
        """Bind one private immutable scalar for a provider sequence."""

        if not isinstance(name, str) or not name:
            raise TaichiRuntimeError(
                "Graph internal scalar binding name must be nonempty"
            )
        if name in self._fixed_runtime_args:
            raise TaichiRuntimeError(
                f"Graph internal binding {name!r} is already defined"
            )
        self._fixed_runtime_args[name] = value
        return Arg(ArgKind.SCALAR, name, dtype)

    def dispatch(self, kernel_fn, *args, template_args=None, label=None):
        label = _normalize_dispatch_label(label)
        kernel_cpp = gen_cpp_kernel(kernel_fn, args, template_args=template_args)
        unzipped_args = flatten_args(args)
        ir_node = _dispatch_ir_node(kernel_cpp, unzipped_args, dispatch_label=label)
        self._dispatches.append((kernel_cpp, unzipped_args))
        self._dispatch_labels.append(label)
        self._items.append(("dispatch", kernel_cpp, unzipped_args, label))
        self._ir_nodes.append(ir_node)
        names = _runtime_arg_names(unzipped_args)
        self._runtime_arg_names.update(names)
        self._recording_runtime_arg_names.update(names)
        self._dispatch_count += 1
        return self

    def _dispatch_bounded(
        self,
        kernel_fn,
        *args,
        extent,
        capacity,
        block_dim=128,
        template_args=None,
        label=None,
    ):
        """Append an internal device-bounded CUDA payload to this region.

        Structured Vulkan replay does not yet carry a reusable indirect packet
        inside a conditional body, so this deliberately fails closed outside
        CUDA.  The private entry point keeps that backend qualification from
        becoming a broader public Graph promise before the Vulkan ownership
        model is complete.
        """

        backend = _backend_name(_ti_core.arch_name(impl.current_cfg().arch))
        if backend != "cuda":
            raise TaichiRuntimeError(
                "structured bounded dispatch is currently qualified only for CUDA"
            )
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("bounded dispatch capacity must be an integer")
        if capacity <= 0 or capacity > 0x7FFFFFFF:
            raise ValueError("bounded dispatch capacity must be in [1, 2^31-1]")
        extent = _require_bounded_symbolic_ndarray(extent, "extent", i32)
        unzipped_args = flatten_args(args)
        if extent.name not in _runtime_arg_names(unzipped_args):
            raise TaichiRuntimeError(
                "bounded dispatch payload arguments must include the extent argument"
            )
        route = _bounded_route(backend, False)
        if not (route.device_known_count and route.no_host_readback):
            raise TaichiRuntimeError(
                "CUDA structured bounded dispatch requires a device-known, "
                "no-readback route"
            )
        policy = GraphBuilder._bounded_launch_policy(block_dim, "require", backend)
        kernel_cpp = gen_cpp_kernel(
            kernel_fn,
            args,
            template_args=template_args,
            task_launch_policy=policy,
            range_one_to_one=False,
        )
        selected_block = _bounded_kernel_geometry(kernel_cpp, backend)
        label = _normalize_dispatch_label(label)
        ir_node = replace(
            _dispatch_ir_node(kernel_cpp, unzipped_args, dispatch_label=label),
            bounded_domain=BoundedDomain(
                extent=extent.name,
                capacity=capacity,
                block_dim=selected_block,
                block_mode=policy.mode,
                # CUDA's standalone bounded node cannot currently be nested
                # safely in a conditional Graph. Keep the capacity grid and
                # use the compact prefix as the semantic mask; capability
                # reporting must not call this exact physical dispatch.
                physical_grid_requirement="auto",
            ),
        )
        self._dispatches.append((kernel_cpp, unzipped_args))
        self._dispatch_labels.append(label)
        self._items.append(("dispatch", kernel_cpp, unzipped_args, label))
        self._ir_nodes.append(ir_node)
        names = _runtime_arg_names(unzipped_args)
        self._runtime_arg_names.update(names)
        self._recording_runtime_arg_names.update(names)
        self._dispatch_count += 1
        return self

    def dispatch_indirect(
        self,
        kernel_fn,
        *args,
        dispatch_packet,
        template_args=None,
        label=None,
    ):
        label = _normalize_dispatch_label(label)
        kernel_cpp = gen_cpp_kernel(kernel_fn, args, template_args=template_args)
        unzipped_args = flatten_args(args)
        self._items.append(
            (
                "indirect",
                kernel_cpp,
                unzipped_args,
                dispatch_packet,
                label,
            )
        )
        self._ir_nodes.append(
            _dispatch_ir_node(
                kernel_cpp,
                unzipped_args,
                dispatch_packet=dispatch_packet,
                dispatch_label=label,
            )
        )
        names = _runtime_arg_names(unzipped_args)
        names.add(dispatch_packet.name)
        self._runtime_arg_names.update(names)
        self._recording_runtime_arg_names.update(names)
        self._dispatch_count += 1
        self._has_indirect_dispatch = True
        return self

    def append_native(self, node, *, prewarm=False):
        executable = compile_native_graph_node(node)
        if prewarm:
            executable.prewarm()
        structured = executable.recordable_sequence
        if structured is not None:
            return self._append_recordable_sequence(structured, executable)
        compiled = _CompiledNativeGraphNode(executable)
        action = compiled.recordable_action
        if action is None:
            raise TaichiRuntimeError(
                "Structured Graph Sequential.append_native requires a "
                "recordable provider action"
            )
        backend = _backend_name(_ti_core.arch_name(impl.current_cfg().arch))
        if not action.supports_backend(backend):
            raise TaichiRuntimeError(
                f"Recordable action does not support the active {backend} backend"
            )
        dispatches = tuple(action.dispatches)
        if action.backend_command_recording is not None:
            raise TaichiRuntimeError(
                "Backend command actions are not yet qualified inside a "
                "structured Graph Sequential"
            )
        if not dispatches:
            raise TaichiRuntimeError(
                "Recordable action must provide at least one dispatch"
            )
        for name, value in compiled.fixed_runtime_args.items():
            existing = self._fixed_runtime_args.get(name)
            if existing is not None and existing is not value:
                if not (
                    isinstance(existing, (int, float))
                    and isinstance(value, (int, float))
                    and existing == value
                ):
                    raise TaichiRuntimeError(
                        "Sequential recordable actions provide conflicting "
                        f"fixed binding {name!r}"
                    )
            self._fixed_runtime_args[name] = value
        self._items.append(("native", compiled))
        self._ir_nodes.append(compiled.ir_node)
        self._runtime_arg_names.update(compiled.runtime_arg_names)
        self._recording_runtime_arg_names.update(compiled.recording_runtime_arg_names)
        self._derived_runtime_arg_names.update(compiled.derived_runtime_arg_names)
        self._lifetime_leases.extend(compiled.lifetime_leases)
        self._temporary_actions.extend(compiled.temporary_actions)
        self._source_native_count += compiled.source_native_count
        self._native_action_manifests.extend(compiled.native_action_manifests)
        self._dispatch_count += len(dispatches)
        return self

    def _append_recordable_sequence(self, sequence, executable):
        if not isinstance(sequence, Sequential):
            raise TaichiRuntimeError(
                "Native recordable_sequence must be a Graph Sequential"
            )
        if not sequence._items:
            raise TaichiRuntimeError(
                "Native recordable_sequence must contain at least one action"
            )
        schema = tuple(executable.runtime_arg_schema)
        if any(not binding.required for binding in schema):
            raise TaichiRuntimeError(
                "Optional structured native Graph arguments are not supported"
            )
        schema_names = frozenset(binding.name for binding in schema)
        if len(schema_names) != len(schema):
            raise TaichiRuntimeError(
                "Structured native Graph runtime bindings must be unique"
            )
        public_names = sequence._recording_runtime_arg_names.difference(
            (
                *sequence._fixed_runtime_args,
                *sequence._derived_runtime_arg_names,
            )
        )
        if public_names != schema_names:
            missing = sorted(schema_names.difference(public_names))
            unexpected = sorted(public_names.difference(schema_names))
            details = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if unexpected:
                details.append("unexpected " + ", ".join(unexpected))
            raise TaichiRuntimeError(
                "Structured native Graph sequence bindings do not match its "
                "public schema: " + "; ".join(details)
            )
        for name, value in sequence._fixed_runtime_args.items():
            existing = self._fixed_runtime_args.get(name)
            if existing is not None and existing is not value:
                if not (
                    isinstance(existing, (int, float))
                    and isinstance(value, (int, float))
                    and existing == value
                ):
                    raise TaichiRuntimeError(
                        "Structured native actions provide conflicting fixed "
                        f"binding {name!r}"
                    )
            self._fixed_runtime_args[name] = value
        self._items.extend(sequence._items)
        self._ir_nodes.extend(sequence._ir_nodes)
        self._runtime_arg_names.update(sequence._runtime_arg_names)
        self._recording_runtime_arg_names.update(sequence._recording_runtime_arg_names)
        self._derived_runtime_arg_names.update(sequence._derived_runtime_arg_names)
        self._lifetime_leases.extend(sequence._lifetime_leases)
        self._lifetime_leases.append(executable)
        self._temporary_actions.extend(sequence._temporary_actions)
        self._source_native_count += sequence._source_native_count
        self._native_action_manifests.extend(sequence._native_action_manifests)
        self._dispatch_count += sequence._dispatch_count
        self._has_indirect_dispatch |= sequence._has_indirect_dispatch
        self._structured_depth = max(self._structured_depth, sequence._structured_depth)
        return self

    def _plain_view(self, item_pairs):
        view = Sequential()
        for item, ir_node in item_pairs:
            kind = item[0]
            if kind == "structured":
                raise TaichiRuntimeError(
                    "Internal Graph plain sequence view cannot contain "
                    "structured control"
                )
            view._items.append(item)
            view._ir_nodes.append(ir_node)
            if kind == "dispatch":
                _, kernel_cpp, args, label = item
                view._dispatches.append((kernel_cpp, args))
                view._dispatch_labels.append(label)
                names = _runtime_arg_names(args)
                view._runtime_arg_names.update(names)
                view._recording_runtime_arg_names.update(names)
                view._dispatch_count += 1
            elif kind == "indirect":
                _, _, args, dispatch_packet, _ = item
                names = _runtime_arg_names(args)
                names.add(dispatch_packet.name)
                view._runtime_arg_names.update(names)
                view._recording_runtime_arg_names.update(names)
                view._dispatch_count += 1
                view._has_indirect_dispatch = True
            else:
                _, compiled = item
                view._fixed_runtime_args.update(compiled.fixed_runtime_args)
                view._runtime_arg_names.update(compiled.runtime_arg_names)
                view._recording_runtime_arg_names.update(
                    compiled.recording_runtime_arg_names
                )
                view._derived_runtime_arg_names.update(
                    compiled.derived_runtime_arg_names
                )
                view._lifetime_leases.extend(compiled.lifetime_leases)
                view._temporary_actions.extend(compiled.temporary_actions)
                view._source_native_count += compiled.source_native_count
                view._native_action_manifests.extend(compiled.native_action_manifests)
                view._dispatch_count += len(compiled.recordable_action.dispatches)
        return view

    def _append_structured(self, node):
        if not _is_structured_control_node(node):
            raise TaichiRuntimeError(
                "Sequential structured action must be while, if, or switch"
            )
        for binding_name, value in node.fixed_runtime_args.items():
            existing = self._fixed_runtime_args.get(binding_name)
            if existing is not None and existing is not value:
                if not (
                    isinstance(existing, (int, float))
                    and isinstance(value, (int, float))
                    and existing == value
                ):
                    raise TaichiRuntimeError(
                        "Sequential structured actions provide conflicting "
                        f"fixed binding {binding_name!r}"
                    )
            self._fixed_runtime_args[binding_name] = value
        self._items.append(("structured", node))
        self._ir_nodes.append(node.ir_node)
        self._runtime_arg_names.update(node.runtime_arg_names)
        self._recording_runtime_arg_names.update(node.recording_runtime_arg_names)
        self._derived_runtime_arg_names.update(node.derived_runtime_arg_names)
        self._lifetime_leases.extend(node.lifetime_leases)
        self._temporary_actions.extend(node.temporary_actions)
        self._source_native_count += node.source_native_count
        self._native_action_manifests.extend(_native_action_manifests_for_node(node))
        self._dispatch_count += node.dispatch_count
        self._structured_depth = max(self._structured_depth, node.structured_depth)
        return self

    def while_loop(
        self,
        condition,
        body,
        *,
        predicate,
        max_iterations,
        control_inputs=(),
        carried_state=(),
        counter=None,
        status=None,
        chunk_size=None,
        vulkan_first_chunk_strategy="auto",
        masked_execution=False,
        lowering_mode="auto",
        name="while",
    ):
        """Append a depth-bounded structured while child to this region."""
        return self._append_structured(
            _CompiledWhileGraphNode(
                condition,
                body,
                predicate=_graph_control_name(predicate, "while predicate"),
                control_inputs=_graph_control_names(
                    control_inputs, "while control_inputs"
                ),
                carried_state=_graph_control_names(
                    carried_state, "while carried_state"
                ),
                max_iterations=max_iterations,
                counter=(
                    None
                    if counter is None
                    else _graph_control_name(counter, "while counter")
                ),
                status=(
                    None
                    if status is None
                    else _graph_control_name(status, "while status")
                ),
                chunk_size=chunk_size,
                vulkan_first_chunk_strategy=vulkan_first_chunk_strategy,
                masked_execution=masked_execution,
                lowering_mode=lowering_mode,
                name=name,
            )
        )

    def if_then_else(
        self,
        condition,
        then_region,
        *,
        predicate,
        control_inputs=(),
        else_region=None,
        lowering_mode="auto",
        name="if",
    ):
        """Append a depth-bounded structured conditional child."""
        return self._append_structured(
            _CompiledIfGraphNode(
                condition,
                then_region,
                else_region,
                predicate=_graph_control_name(predicate, "if predicate"),
                control_inputs=_graph_control_names(
                    control_inputs, "if control_inputs"
                ),
                lowering_mode=lowering_mode,
                name=name,
            )
        )

    def switch(
        self,
        condition,
        branches,
        *,
        selector,
        control_inputs=(),
        default_region=None,
        lowering_mode="auto",
        name="switch",
    ):
        """Append a depth-bounded zero-based structured switch child."""
        return self._append_structured(
            _CompiledSwitchGraphNode(
                condition,
                tuple(branches),
                default_region,
                selector=_graph_control_name(selector, "switch selector"),
                control_inputs=_graph_control_names(
                    control_inputs, "switch control_inputs"
                ),
                lowering_mode=lowering_mode,
                name=name,
            )
        )

    def _dispatch_to(self, builder, *, region_kind="sequential"):
        backend = _backend_name(_ti_core.arch_name(impl.current_cfg().arch))
        recording_dispatches = []
        for item, ir_node in zip(self._items, self._ir_nodes):
            if item[0] == "dispatch":
                _, kernel_cpp, args, label = item
                _record_backend_dispatch(
                    builder,
                    backend,
                    _RecordingDispatch(kernel_cpp, tuple(args)),
                    ir_node,
                )
                recording_dispatches.append(_RecordingDispatch(kernel_cpp, tuple(args)))
                continue
            elif item[0] == "indirect":
                _, kernel_cpp, args, dispatch_packet, label = item
                builder.dispatch_indirect(kernel_cpp, args, dispatch_packet, label)
                recording_dispatches.append(
                    _RecordingDispatch(
                        kernel_cpp,
                        tuple(args),
                        dispatch_packet,
                    )
                )
                continue
            elif item[0] == "native":
                _, node = item
                label = ""
                action = node.recordable_action
                if not action.supports_backend(backend):
                    raise TaichiRuntimeError(
                        f"Recordable action does not support the active {backend} backend"
                    )
                if not action.supports_region(region_kind):
                    raise TaichiRuntimeError(
                        "Recordable action is not qualified for Graph region "
                        f"{region_kind!r}"
                    )
                dispatches = tuple(action.dispatches)
            else:
                raise TaichiRuntimeError(
                    "Nested structured control cannot be flattened into a "
                    "single CGraph segment"
                )
            for kernel_cpp, args in dispatches:
                builder.dispatch(kernel_cpp, args, label)
                recording_dispatches.append(_RecordingDispatch(kernel_cpp, tuple(args)))
        return tuple(recording_dispatches)


class GraphBuilder:
    def __init__(
        self,
        *,
        _capture_recipe_sources=True,
        _explicit_map_source_groups=None,
        _map_recipe=None,
    ):
        self._aot_graph_plan = _AOTGraphBuilderPlan()
        self._aot_plan_cursor = 0
        self._explicit_map_source_groups = _explicit_map_source_groups

        self._runtime_graph_builder = _new_runtime_graph_builder(
            _explicit_map_source_groups,
            _map_recipe,
        )
        self._capture_recipe_sources = bool(_capture_recipe_sources)

        self._runtime_graph_recipe_operations = []

        self._dispatch_count = 0
        self._runtime_graph_arg_names = set()
        self._runtime_graph_dispatches = []
        self._nodes = []
        self._pending_ir_nodes = []
        self._observation_names = set()
        self._runtime_graph_fixed_args = {}
        self._runtime_graph_lifetime_leases = []
        self._bounded_extent_contracts = {}
        self._ordered_offsets_contracts = {}
        self._runtime_graph_source_native_count = 0
        self._runtime_graph_native_action_manifests = []
        self._active_bounded_publication = None
        self._declared_private_bindings = {}
        self._graph_memory_sources = []
        self._graph_offload_fusion_sources = []
        self._graph_sparse_traversal_sources = []
        self._graph_bounded_sources = []
        self._cuda_parallel_dispatch_groups = ()

        self._graph_reduction_sources = []
        self._graph_native_algorithm_sources = []

    def private_ndarray(
        self,
        name,
        dtype,
        shape,
        *,
        exclusive_submission=True,
    ):
        """Declare one Graph-instance-owned private ndarray argument."""

        if not isinstance(name, str) or not name:
            raise ValueError("Graph private ndarray name must be nonempty")
        if name in self._declared_private_bindings:
            raise TaichiRuntimeError(
                f"Graph private binding {name!r} is already defined"
            )
        requirement = GraphOwnedNdarray(
            dtype,
            shape,
            exclusive_submission=exclusive_submission,
        )
        symbolic = Arg(
            ArgKind.NDARRAY,
            name,
            dtype,
            ndim=len(requirement.shape),
        )
        self._declared_private_bindings[name] = (symbolic, requirement)
        return symbolic

    def _bind_declared_private_args(self, args):
        for symbolic in args:
            declared = self._declared_private_bindings.get(
                getattr(symbolic, "name", None)
            )
            if declared is None:
                continue
            expected, requirement = declared
            if symbolic is not expected:
                raise TaichiRuntimeError(
                    f"Graph private binding {expected.name!r} must use the "
                    "symbol returned by private_ndarray()"
                )
            self._bind_internal_runtime_arg(symbolic, requirement)

    def dispatch(self, kernel_fn, *args, template_args=None, label=None):
        label = _normalize_dispatch_label(label)
        kernel_cpp = gen_cpp_kernel(kernel_fn, args, template_args=template_args)
        unzipped_args = flatten_args(args)
        source = _GraphMemoryRecipeSource.try_create(
            kernel_fn,
            unzipped_args,
            label,
            kernel_cpp,
            template_args,
        )
        offload_source = _GraphOffloadFusionRecipeSource.try_create(
            kernel_fn,
            unzipped_args,
            label,
            kernel_cpp,
            template_args,
        )
        sparse_source = _GraphSparseTraversalRecipeSource.try_create(
            kernel_fn,
            unzipped_args,
            label,
            kernel_cpp,
            template_args,
        )
        contracts, layout_requirements = (), ()
        if source is not None:
            source._recipe_source_key = f"memory:{len(self._graph_memory_sources)}"
        if offload_source is not None:
            offload_source._recipe_source_key = (
                f"offload-fusion:{len(self._graph_offload_fusion_sources)}"
            )
        if sparse_source is not None:
            sparse_source._recipe_source_key = (
                f"sparse-traversal:{len(self._graph_sparse_traversal_sources)}"
            )
        self._record_dispatch(
            kernel_cpp,
            unzipped_args,
            label,
            _recipe_source=source,
            _offload_recipe_source=offload_source,
            _sparse_recipe_source=sparse_source,
        )
        if contracts or layout_requirements:
            self._pending_ir_nodes[-1] = replace(
                self._pending_ir_nodes[-1],
                memory_disjoint_pairs=contracts,
                memory_layout_requirements=layout_requirements,
            )
        if source is not None:
            self._graph_memory_sources.append(source)
        if offload_source is not None:
            self._graph_offload_fusion_sources.append(offload_source)
        if sparse_source is not None:
            self._graph_sparse_traversal_sources.append(sparse_source)

    def reduce(
        self,
        values,
        output,
        *,
        count,
        op="sum",
        absolute_tolerance=None,
        relative_tolerance=None,
        label=None,
    ):
        """Append one typed, complete Graph reduction phase.

        The semantic API exposes operation, dtype, exact item count, and the
        floating-point equivalence contract.  Physical direct/TLS versus
        block-partial/finalize selection remains a private complete recipe
        chosen only through the reviewed CompileIQ reconstruction path.
        """

        from taichi_forge.graph._reduction import (
            append_typed_graph_reduction,
        )

        operation_start = len(self._runtime_graph_recipe_operations)

        source = append_typed_graph_reduction(
            self,
            values,
            output,
            count=count,
            op=op,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            label=label,
            requested_recipe_id=None,
        )
        source._recipe_source_key = (
            f"graph_reduction:{len(self._graph_reduction_sources)}"
        )

        if self._capture_recipe_sources:
            del self._runtime_graph_recipe_operations[operation_start:]
            self._runtime_graph_recipe_operations.append(
                ("graph_reduction", source, label)
            )

        self._graph_reduction_sources.append(source)
        return self

    def segmented_scan(
        self,
        values,
        layout,
        output,
        *,
        inclusive=True,
        op="sum",
    ):
        """Append one typed Graph-native segmented-scan action.

        Forge freezes the immutable layout and fixed resource ABI at Graph
        construction. Modified CompileIQ can then select between the complete
        segment-local and global-scan/correction strategies during explicit
        reconstruction; no provider or launch-parameter axis is exposed.
        """

        from taichi_forge.graph._native_algorithm import (
            append_graph_segmented_scan,
        )

        source = append_graph_segmented_scan(
            self,
            values,
            layout,
            output,
            inclusive=inclusive,
            op=op,
            requested_recipe_id=None,
        )
        source._recipe_source_key = (
            f"native_algorithm:{len(self._graph_native_algorithm_sources)}"
        )

        source._recipe_node_index = len(self._nodes) - 1

        self._graph_native_algorithm_sources.append(source)
        return self

    def segmented_reduce(self, values, layout, output, *, op="sum"):
        """Append a fixed-layout CUDA i32/u32 segmented sum.

        Empty segments produce zero; arithmetic is modulo 2**32. Complete
        recipes own segment-local or chunk/partial/finalize execution and
        workspace. Launch dimensions are provider internals, not API axes.
        Ordinary experimental_segmented_reduce() remains unchanged.
        """
        from taichi_forge.graph._segmented_reduce import append_graph_segmented_reduce

        source = append_graph_segmented_reduce(self, values, layout, output, op=op)
        source._recipe_source_key = (
            f"native_algorithm:{len(self._graph_native_algorithm_sources)}"
        )
        source._recipe_node_index = len(self._nodes) - 1
        self._graph_native_algorithm_sources.append(source)
        return self

    def keyed_reduce(self, keys, values, output, *, op="sum"):
        """Append one typed, fixed-resource keyed aggregation.

        The public i32 semantic operation intentionally has no provider
        ``method`` parameter. Forge generates complete global-atomic,
        and block-shared dense physical recipes when their
        context applies, and the whole-Graph CompileIQ session selects among
        them as opaque recipe tokens.
        """

        from taichi_forge.graph._aggregation import (
            append_graph_keyed_aggregation,
        )

        source = append_graph_keyed_aggregation(
            self,
            keys,
            values,
            output,
            op=op,
        )
        source._recipe_source_key = (
            f"native_algorithm:{len(self._graph_native_algorithm_sources)}"
        )
        source._recipe_node_index = len(self._nodes) - 1
        self._graph_native_algorithm_sources.append(source)
        return self

    def _dispatch_shared_staged_1d(
        self, kernel_fn, *args, template_args=None, label=None
    ):
        """Materialize one private Graph-owned external shared-stage recipe."""

        if not isinstance(kernel_fn, kernel_impl._OffloadExecutionPlanBinding):
            raise TaichiRuntimeError(
                "Graph shared-staged dispatch requires an exact offload plan binding"
            )
        if (
            sum(
                task.memory_strategy in ("shared_staged_1d", "shared_staged_2d")
                for task in kernel_fn.plan.tasks
            )
            != 1
        ):
            raise TaichiRuntimeError(
                "Graph shared-staged dispatch requires exactly one staged task"
            )
        label = _normalize_dispatch_label(label)
        kernel_cpp = gen_cpp_kernel(
            kernel_fn,
            args,
            template_args=template_args,
            allow_graph_memory_recipe=True,
        )
        unzipped_args = flatten_args(args)
        contracts, layout_requirements, _ = _graph_shared_staged_contract(
            kernel_cpp, unzipped_args
        )
        self._record_dispatch(kernel_cpp, unzipped_args, label)
        self._pending_ir_nodes[-1] = replace(
            self._pending_ir_nodes[-1],
            memory_disjoint_pairs=contracts,
            memory_layout_requirements=layout_requirements,
        )

    def dispatch_indirect(
        self,
        kernel_fn,
        *args,
        dispatch_packet,
        template_args=None,
        label=None,
    ):
        label = _normalize_dispatch_label(label)
        kernel_cpp = gen_cpp_kernel(kernel_fn, args, template_args=template_args)
        unzipped_args = flatten_args(args)
        self._record_indirect_dispatch(
            kernel_cpp, unzipped_args, dispatch_packet, label
        )

    def _record_indirect_dispatch(
        self,
        kernel_cpp,
        unzipped_args,
        dispatch_packet,
        label="",
        *,
        preserve_bounded_publication=False,
    ):
        self._bind_declared_private_args((*unzipped_args, dispatch_packet))
        if not preserve_bounded_publication:
            self._active_bounded_publication = None
        self._aot_graph_plan.dispatch_indirect(
            kernel_cpp, unzipped_args, dispatch_packet, label
        )
        self._ensure_runtime_graph_builder().dispatch_indirect(
            kernel_cpp, unzipped_args, dispatch_packet, label
        )
        self._runtime_graph_arg_names.update(_runtime_arg_names(unzipped_args))
        self._runtime_graph_arg_names.add(dispatch_packet.name)
        self._pending_ir_nodes.append(
            _dispatch_ir_node(
                kernel_cpp,
                unzipped_args,
                dispatch_packet=dispatch_packet,
                dispatch_label=label,
            )
        )
        self._runtime_graph_dispatches.append(
            _RecordingDispatch(
                kernel_cpp,
                tuple(unzipped_args),
                dispatch_packet,
            )
        )
        self._dispatch_count += 1

    def _bind_internal_runtime_arg(self, symbolic, value):
        name = symbolic.name
        previous = self._runtime_graph_fixed_args.get(name)
        if (
            previous is not None
            and previous is not value
            and not (
                isinstance(previous, (int, float))
                and isinstance(value, (int, float))
                and previous == value
            )
        ):
            raise TaichiRuntimeError(
                f"Graph internal fixed binding {name!r} is already defined"
            )
        self._runtime_graph_fixed_args[name] = value
        self._runtime_graph_arg_names.add(name)
        self._aot_graph_plan.mark_internal_fixed_bindings()

    def _retain_runtime_graph_lease(self, lease):
        if all(
            existing is not lease for existing in self._runtime_graph_lifetime_leases
        ):
            self._runtime_graph_lifetime_leases.append(lease)
        self._aot_graph_plan.mark_internal_fixed_bindings()

    def _vulkan_bounded_publication(self, extent, capacity, block_dim):
        if _vulkan_bounded_packet_policy()[1] == "per_consumer":
            self._active_bounded_publication = None
            return None
        key = (extent.name, int(capacity), int(block_dim))
        active = self._active_bounded_publication
        if active is not None and active.key == key:
            owns_packet = not active.packet_claimed
            active.packet_claimed = True
            return (
                active.packet_arg,
                active.packet,
                owns_packet,
            )

        self._active_bounded_publication = None
        if self._dispatch_count != 0 or not self._nodes:
            return None
        producer = self._nodes[-1]
        if not isinstance(producer, _CompiledNativeGraphNode):
            return None

        unique = next(_bounded_dispatch_ids)
        packet = _GraphInternalNdarraySpec(u32, (4,), 4)
        packet_arg = Arg(
            ArgKind.NDARRAY,
            f"__ti_bounded_publication_{unique}",
            u32,
            ndim=1,
        )
        target = BoundedPublicationTarget(
            backend="vulkan",
            extent_name=extent.name,
            capacity=capacity,
            block_dim=block_dim,
            packet_binding=packet_arg,
            packet_storage=packet,
        )
        action = producer.executable.recordable_bounded_publication(target)
        if action is None:
            return None
        if not action.supports_backend("vulkan"):
            raise TaichiRuntimeError(
                "Bounded publication specialization does not support Vulkan"
            )
        self._nodes[-1] = _CompiledNativeGraphNode(
            producer.executable,
            recordable_action=action,
        )
        self._active_bounded_publication = _ActiveBoundedPublication(
            key=key,
            packet_arg=packet_arg,
            packet=packet,
            packet_claimed=True,
        )
        return packet_arg, packet, True

    def _bounded_extent_contract(self, extent_name, capacity, *, expected_extent=None):
        contract = self._bounded_extent_contracts.get(extent_name)
        if contract is None:
            contract = _DeviceExtentGraphContract(extent_name, capacity)
            self._bounded_extent_contracts[extent_name] = contract
        elif contract.capacity != capacity:
            raise TaichiRuntimeError(
                f"Bounded extent {extent_name!r} changes capacity within one Graph"
            )
        if expected_extent is not None:
            contract.require_identity(expected_extent)
        self._retain_runtime_graph_lease(contract)
        return contract

    def _ordered_offsets_contract(self, offsets_name, segment_count):
        contract = self._ordered_offsets_contracts.get(offsets_name)
        if contract is None:
            contract = _OrderedOffsetsGraphContract(offsets_name, segment_count)
            self._ordered_offsets_contracts[offsets_name] = contract
        elif contract.segment_count != segment_count:
            raise TaichiRuntimeError(
                f"Ordered offsets {offsets_name!r} change segment count within one Graph"
            )
        self._retain_runtime_graph_lease(contract)
        return contract

    @staticmethod
    def _bounded_launch_policy(block_dim, block_mode, backend):
        from taichi_forge.lang.task_launch import TaskLaunchPolicy

        if block_mode not in ("hint", "require"):
            raise ValueError("bounded dispatch block_mode must be hint or require")
        if block_dim is None or backend == "cpu":
            return TaskLaunchPolicy.auto()
        return TaskLaunchPolicy.block(block_dim, mode=block_mode)

    def dispatch_bounded(
        self,
        kernel_fn,
        *args,
        extent=None,
        count=None,
        capacity,
        block_dim=None,
        block_mode="require",
        physical_grid="auto",
        launch_state=None,
        template_args=None,
        label=None,
        _recipe_route=None,
        _recipe_update_policy=None,
        _register_recipe_source=True,
    ):
        """Append one device-count-driven bounded payload dispatch.

        Pass either a device ``extent`` ndarray or a host scalar ``count``.
        The device payload must mask its semantic body with
        ``device_extent_count(extent)`` as a defensive guard. Device-known
        routes execute only the clamped logical prefix; backends may use CPU
        scheduler chunks, a CUDA device-bounded grid-stride range, or Vulkan
        indirect dispatch. A host count is accepted only when compiler
        metadata proves it is the payload's sole range domain, and is clamped
        before launch. ``physical_grid='extent'`` requires a no-readback
        backend route whose physical work tracks the clamped device extent;
        ``'capacity'`` is the explicit fixed-grid baseline, and ``'auto'``
        keeps the backend's conservative admission policy.
        """

        operation_start = len(self._runtime_graph_recipe_operations)

        ir_start = len(self._pending_ir_nodes)

        physical_grid = _normalize_bounded_physical_grid_policy(physical_grid)
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("bounded dispatch capacity must be an integer")
        if capacity <= 0 or capacity > 0x7FFFFFFF:
            raise ValueError("bounded dispatch capacity must be in [1, 2^31-1]")
        if (extent is None) == (count is None):
            raise ValueError("bounded dispatch requires exactly one of extent or count")
        if count is not None and physical_grid != "auto":
            raise ValueError(
                "host-known bounded dispatch already uses an exact launch; "
                "physical_grid must remain 'auto'"
            )
        if launch_state is not None:
            from taichi_forge.lang.device_extent import DeviceDispatchState

            if count is not None:
                raise ValueError(
                    "producer-owned launch_state is valid only for device extents"
                )
            if physical_grid == "capacity":
                raise ValueError(
                    "producer-owned launch_state cannot be combined with "
                    "physical_grid='capacity'"
                )
            if not isinstance(launch_state, DeviceDispatchState):
                raise TypeError(
                    "bounded dispatch launch_state must be a DeviceDispatchState"
                )
            launch_state._validate_current()
            if launch_state.capacity != capacity:
                raise ValueError(
                    "bounded dispatch launch_state capacity does not match capacity"
                )
            if block_dim is None:
                block_dim = launch_state.block_dim
            elif block_dim != launch_state.block_dim:
                raise ValueError(
                    "bounded dispatch block_dim must match launch_state.block_dim"
                )
        unzipped_args = flatten_args(args)
        backend = _backend_name(_ti_core.arch_name(impl.current_cfg().arch))
        if backend not in ("cpu", "cuda", "vulkan"):
            raise TaichiRuntimeError(
                f"bounded dispatch is unavailable on backend {backend}"
            )
        if launch_state is not None and backend == "cuda":
            raise TaichiRuntimeError(
                "CUDA bounded dispatch does not consume producer-owned "
                "DeviceDispatchState packets. Pass extent= without "
                "launch_state; CUDA will select its qualified logical/physical "
                "route from the DeviceExtent."
            )
        selected_route = (
            None
            if count is not None
            else _bounded_route(
                backend,
                False,
                physical_grid=physical_grid,
                requested_route=_recipe_route,
                requested_update_policy=_recipe_update_policy,
            )
        )
        exact_device_grid = bool(
            selected_route is not None and selected_route.exact_grid
        )
        logical_exact = bool(
            selected_route is not None and selected_route.logical_iteration_exact
        )
        cuda_bounded_range = backend == "cuda" and logical_exact
        cuda_adaptive_grid = bool(
            cuda_bounded_range and selected_route.route == "adaptive_device_grid_update"
        )
        cuda_update_policy = (
            _cuda_bounded_update_policy(_recipe_update_policy)[1]
            if cuda_adaptive_grid
            else ""
        )
        cuda_grouped_update = cuda_update_policy == "grouped_stateful"
        specialized_range = exact_device_grid or cuda_bounded_range
        policy = self._bounded_launch_policy(block_dim, block_mode, backend)
        kernel_cpp = gen_cpp_kernel(
            kernel_fn,
            args,
            template_args=template_args,
            task_launch_policy=policy,
            range_one_to_one=specialized_range,
        )
        bounded_semantic_kernel_identity = ""
        if backend == "cuda" and physical_grid == "auto":
            semantic_kernel_cpp = kernel_cpp
            if not specialized_range:
                semantic_kernel_cpp = gen_cpp_kernel(
                    kernel_fn,
                    args,
                    template_args=template_args,
                    task_launch_policy=policy,
                    range_one_to_one=True,
                )
            bounded_semantic_kernel_identity = _bounded_semantic_kernel_identity(
                semantic_kernel_cpp
            )
        label = _normalize_dispatch_label(label)

        if count is not None:
            if getattr(count, "tag", None) != ArgKind.SCALAR or count.dtype() != i32:
                raise TaichiRuntimeError(
                    "host-known bounded dispatch count must be a symbolic i32 scalar"
                )
            if count.name not in _runtime_arg_names(unzipped_args):
                raise TaichiRuntimeError(
                    "bounded dispatch payload arguments must include the count argument"
                )
            selected_block = _bounded_kernel_geometry(
                kernel_cpp, backend, allow_range_setup=True
            )
            _verify_bounded_host_range(kernel_cpp, unzipped_args, count)
            self._record_dispatch(kernel_cpp, unzipped_args, label)
            self._pending_ir_nodes[-1] = replace(
                self._pending_ir_nodes[-1],
                bounded_domain=BoundedDomain(
                    extent=count.name,
                    capacity=capacity,
                    block_dim=selected_block,
                    block_mode=policy.mode,
                    physical_grid_requirement="logical_exact",
                    count_source="host_scalar",
                ),
            )
            handle = HostBoundedDispatchHandle(
                count_name=count.name,
                capacity=capacity,
                block_dim=selected_block,
                backend=backend,
            )
            self._retain_runtime_graph_lease(handle)
            return handle

        selected_block = _bounded_kernel_geometry(
            kernel_cpp,
            backend,
            require_one_to_one=specialized_range,
        )
        if (
            launch_state is not None
            and backend in ("cuda", "vulkan")
            and selected_block != launch_state.block_dim
        ):
            raise TaichiRuntimeError(
                "bounded dispatch selected block dimension does not match "
                "the producer-owned launch state"
            )
        effective_launch_state = launch_state if backend == "vulkan" else None
        extent = _require_bounded_symbolic_ndarray(extent, "extent", i32)
        if extent.name not in _runtime_arg_names(unzipped_args):
            raise TaichiRuntimeError(
                "bounded dispatch payload arguments must include the extent argument"
            )
        packet = None
        packet_allocation_owner = True
        preparation_dispatches = None
        preserve_vulkan_packet = False
        vulkan_indirect = bool(
            backend == "vulkan" and selected_route.route == "exact_indirect"
        )
        if vulkan_indirect:
            _validate_vulkan_indirect_grid_capacity(capacity, selected_block)
            publication = (
                None
                if effective_launch_state is not None
                else self._vulkan_bounded_publication(extent, capacity, selected_block)
            )
            if publication is not None:
                packet_arg, packet, packet_allocation_owner = publication
                preparation_dispatches = 0
                preserve_vulkan_packet = True
            else:
                unique = next(_bounded_dispatch_ids)
                packet = (
                    effective_launch_state.packet
                    if effective_launch_state is not None
                    else _GraphInternalNdarraySpec(u32, (3,), 4)
                )
                packet_arg = Arg(
                    ArgKind.NDARRAY,
                    f"__ti_bounded_packet_{unique}",
                    u32,
                    ndim=1,
                )
            if effective_launch_state is None and publication is None:
                capacity_arg = Arg(
                    ArgKind.SCALAR, f"__ti_bounded_capacity_{unique}", i32
                )
                block_arg = Arg(ArgKind.SCALAR, f"__ti_bounded_block_{unique}", i32)
                prepare_args = (extent, packet_arg, capacity_arg, block_arg)
                prepare_cpp = gen_cpp_kernel(
                    _prepare_bounded_dispatch_packet, prepare_args
                )
                self._record_dispatch(prepare_cpp, list(prepare_args))
                self._bind_internal_runtime_arg(capacity_arg, capacity)
                self._bind_internal_runtime_arg(block_arg, selected_block)
                # The packet remains valid for consecutive consumers of the
                # same extent contract. Any intervening action clears this
                # conservative builder-local publication state.
                if _vulkan_bounded_packet_policy()[1] == "reuse_consecutive":
                    self._active_bounded_publication = _ActiveBoundedPublication(
                        key=(
                            extent.name,
                            int(capacity),
                            int(selected_block),
                        ),
                        packet_arg=packet_arg,
                        packet=packet,
                        packet_claimed=True,
                    )
                    preserve_vulkan_packet = True
            self._record_indirect_dispatch(
                kernel_cpp,
                unzipped_args,
                packet_arg,
                label,
                preserve_bounded_publication=preserve_vulkan_packet,
            )
            self._bind_internal_runtime_arg(packet_arg, packet)
        elif cuda_bounded_range:
            self._record_cuda_bounded_dispatch(
                kernel_cpp,
                unzipped_args,
                extent,
                capacity,
                selected_block,
                cuda_adaptive_grid,
                cuda_grouped_update,
                label=label,
            )
        elif backend == "cpu" and exact_device_grid:
            self._record_cpu_bounded_dispatch(
                kernel_cpp,
                unzipped_args,
                extent,
                capacity,
                label,
            )
        else:
            self._record_dispatch(kernel_cpp, unzipped_args, label)

        bounded_domain = BoundedDomain(
            extent=extent.name,
            capacity=capacity,
            block_dim=selected_block,
            block_mode=policy.mode,
            physical_grid_policy=physical_grid,
            physical_grid_requirement=(
                "adaptive_grid"
                if cuda_adaptive_grid
                else (
                    "logical_exact"
                    if cuda_bounded_range
                    else (
                        "require_exact"
                        if exact_device_grid
                        else (
                            "fixed_capacity" if physical_grid == "capacity" else "auto"
                        )
                    )
                )
            ),
            update_policy=cuda_update_policy,
            semantic_kernel_identity=bounded_semantic_kernel_identity,
        )
        self._pending_ir_nodes[-1] = replace(
            self._pending_ir_nodes[-1], bounded_domain=bounded_domain
        )
        self._bounded_extent_contract(
            extent.name,
            capacity,
            expected_extent=(
                None
                if effective_launch_state is None
                else effective_launch_state.extent
            ),
        )
        handle = BoundedDispatchHandle(
            extent_name=extent.name,
            capacity=capacity,
            block_dim=selected_block,
            backend=backend,
            packet=packet,
            launch_state=effective_launch_state,
            preparation_dispatches=preparation_dispatches,
            packet_allocation_owner=packet_allocation_owner,
            capabilities=selected_route,
            publication_reuse=(
                "grouped_stateful"
                if cuda_grouped_update
                else "per_consumer" if backend == "cuda" else None
            ),
        )
        if _register_recipe_source:
            source = _GraphBoundedRecipeSource(
                kernel_fn,
                args,
                extent=extent,
                capacity=capacity,
                block_dim=block_dim,
                block_mode=block_mode,
                launch_state=launch_state,
                template_args=template_args,
                label=label,
                selected_strategy=(
                    "adaptive_grouped"
                    if cuda_grouped_update
                    else (
                        "adaptive_per_node"
                        if cuda_adaptive_grid
                        else (
                            "logical_exact" if cuda_bounded_range else "masked_capacity"
                        )
                    )
                ),
                selected_block=selected_block,
            )
            source._recipe_source_key = f"bounded:{len(self._graph_bounded_sources)}"
            source._recipe_physical_dispatches = len(self._pending_ir_nodes) - ir_start
            self._graph_bounded_sources.append(source)
            if self._capture_recipe_sources:
                del self._runtime_graph_recipe_operations[operation_start:]
                self._runtime_graph_recipe_operations.append(("bounded", source))

        return handle

    def dispatch_ordered_segments(
        self,
        kernel_fn,
        *args,
        offsets,
        extent,
        capacity,
        segment_count,
        block_dim=None,
        block_mode="require",
        template_args=None,
        label=None,
    ):
        """Append globally ordered ranges using one reusable payload kernel.

        ``segment_state`` is injected as the payload kernel's final ndarray
        argument. Read it with ``segmented_dispatch_begin/end/index/count``
        and mask local indices against ``segmented_dispatch_count``. No offset
        or count is read back while the Graph executes.
        """

        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("ordered dispatch capacity must be an integer")
        if capacity <= 0 or capacity > 0x7FFFFFFF:
            raise ValueError("ordered dispatch capacity must be in [1, 2^31-1]")
        if isinstance(segment_count, bool) or not isinstance(segment_count, int):
            raise TypeError("ordered dispatch segment_count must be an integer")
        if not 1 <= segment_count <= 4096:
            raise ValueError("ordered dispatch segment_count must be in [1, 4096]")
        offsets = _require_bounded_symbolic_ndarray(offsets, "offsets", i32)
        extent = _require_bounded_symbolic_ndarray(extent, "extent", i32)
        public_args = flatten_args(args)
        public_names = _runtime_arg_names(public_args)
        for symbolic, role in ((offsets, "offsets"), (extent, "extent")):
            if symbolic.name not in public_names:
                raise TaichiRuntimeError(
                    f"ordered dispatch payload arguments must include {role}"
                )
        backend = _backend_name(_ti_core.arch_name(impl.current_cfg().arch))
        if backend not in ("cpu", "cuda", "vulkan"):
            raise TaichiRuntimeError(
                f"ordered segmented dispatch is unavailable on backend {backend}"
            )
        ordered_route = _bounded_route(backend, True)
        if backend in ("cpu", "cuda") and ordered_route.logical_iteration_exact:
            raise TaichiRuntimeError(
                f"{backend.upper()} exact bounded dispatch does not yet support ordered "
                "segmented dispatch; select masked_capacity for this graph"
            )
        unique = next(_bounded_dispatch_ids)
        packet = (
            _GraphInternalNdarraySpec(u32, (3,), 4) if backend == "vulkan" else None
        )
        segment_state = _GraphInternalNdarraySpec(
            i32, (BoundedDispatchHandle._SEGMENT_STATE_SIZE,), 4
        )
        state_arg = Arg(ArgKind.NDARRAY, f"__ti_segment_state_{unique}", i32, ndim=1)
        capacity_arg = Arg(ArgKind.SCALAR, f"__ti_segment_capacity_{unique}", i32)
        count_arg = Arg(ArgKind.SCALAR, f"__ti_segment_count_{unique}", i32)
        packet_arg = None
        block_arg = None
        if backend == "vulkan":
            packet_arg = Arg(
                ArgKind.NDARRAY, f"__ti_segment_packet_{unique}", u32, ndim=1
            )
            block_arg = Arg(ArgKind.SCALAR, f"__ti_segment_block_{unique}", i32)
        payload_args = (*args, state_arg)
        policy = self._bounded_launch_policy(block_dim, block_mode, backend)
        payload_cpp = gen_cpp_kernel(
            kernel_fn,
            payload_args,
            template_args=template_args,
            task_launch_policy=policy,
            range_one_to_one=backend == "vulkan",
        )
        selected_block = _bounded_kernel_geometry(payload_cpp, backend)
        if backend == "vulkan":
            _validate_vulkan_indirect_grid_capacity(capacity, selected_block)
        payload_flattened = flatten_args(payload_args)
        base_label = _normalize_dispatch_label(label)

        self._bind_internal_runtime_arg(state_arg, segment_state)
        self._bind_internal_runtime_arg(capacity_arg, capacity)
        self._bind_internal_runtime_arg(count_arg, segment_count)
        if backend == "vulkan":
            self._bind_internal_runtime_arg(packet_arg, packet)
            self._bind_internal_runtime_arg(block_arg, selected_block)
        for segment in range(segment_count):
            index_arg = Arg(
                ArgKind.SCALAR,
                f"__ti_segment_index_{unique}_{segment}",
                i32,
            )
            if backend == "vulkan":
                prepare_args = (
                    offsets,
                    extent,
                    packet_arg,
                    state_arg,
                    index_arg,
                    count_arg,
                    capacity_arg,
                    block_arg,
                )
                prepare_kernel = _prepare_ordered_segment_dispatch
            else:
                prepare_args = (
                    offsets,
                    extent,
                    state_arg,
                    index_arg,
                    count_arg,
                    capacity_arg,
                )
                prepare_kernel = _prepare_ordered_segment_state
            prepare_cpp = gen_cpp_kernel(prepare_kernel, prepare_args)
            prepare_label = f"{base_label}/prepare:{segment}" if base_label else ""
            payload_label = f"{base_label}/segment:{segment}" if base_label else ""
            self._record_dispatch(prepare_cpp, list(prepare_args), prepare_label)
            if backend == "vulkan":
                self._record_indirect_dispatch(
                    payload_cpp,
                    payload_flattened,
                    packet_arg,
                    payload_label,
                )
            else:
                self._record_dispatch(payload_cpp, payload_flattened, payload_label)
            self._pending_ir_nodes[-1] = replace(
                self._pending_ir_nodes[-1],
                bounded_domain=BoundedDomain(
                    extent=extent.name,
                    capacity=capacity,
                    block_dim=selected_block,
                    block_mode=policy.mode,
                    ordered=True,
                    segment_index=segment,
                    segment_count=segment_count,
                ),
            )
            self._bind_internal_runtime_arg(index_arg, segment)

        self._bounded_extent_contract(extent.name, capacity)
        self._ordered_offsets_contract(offsets.name, segment_count)
        handle = BoundedDispatchHandle(
            extent_name=extent.name,
            offsets_name=offsets.name,
            capacity=capacity,
            block_dim=selected_block,
            backend=backend,
            ordered=True,
            segment_count=segment_count,
            packet=packet,
            segment_state=segment_state,
        )
        return handle

    def _record_dispatch(
        self,
        kernel_cpp,
        unzipped_args,
        label="",
        _recipe_source=None,
        _offload_recipe_source=None,
        _sparse_recipe_source=None,
    ):
        self._bind_declared_private_args(unzipped_args)
        self._active_bounded_publication = None
        self._aot_graph_plan.dispatch(kernel_cpp, unzipped_args, label)
        self._ensure_runtime_graph_builder().dispatch(kernel_cpp, unzipped_args, label)
        self._runtime_graph_dispatches.append(
            _RecordingDispatch(kernel_cpp, tuple(unzipped_args))
        )
        self._runtime_graph_arg_names.update(_runtime_arg_names(unzipped_args))
        self._pending_ir_nodes.append(
            _dispatch_ir_node(kernel_cpp, unzipped_args, dispatch_label=label)
        )
        if self._capture_recipe_sources:
            self._runtime_graph_recipe_operations.append(
                (
                    "dispatch",
                    kernel_cpp,
                    tuple(unzipped_args),
                    label,
                    _recipe_source,
                    _offload_recipe_source,
                    _sparse_recipe_source,
                )
            )

        self._dispatch_count += 1

    def _record_cuda_bounded_dispatch(
        self,
        kernel_cpp,
        unzipped_args,
        extent,
        capacity,
        block_dim,
        adaptive_grid,
        grouped_update,
        label="",
    ):
        self._bind_declared_private_args(unzipped_args)
        self._active_bounded_publication = None
        # The public AOT graph schema stays backend-neutral. The exact CUDA
        # node is a JIT-only specialization carried by the runtime builder.
        self._aot_graph_plan.dispatch(kernel_cpp, unzipped_args, label)
        self._ensure_runtime_graph_builder().dispatch_cuda_bounded(
            kernel_cpp,
            unzipped_args,
            extent,
            capacity,
            block_dim,
            adaptive_grid,
            grouped_update,
            label,
        )
        self._runtime_graph_dispatches.append(
            _RecordingDispatch(kernel_cpp, tuple(unzipped_args))
        )
        self._runtime_graph_arg_names.update(_runtime_arg_names(unzipped_args))
        self._pending_ir_nodes.append(
            _dispatch_ir_node(kernel_cpp, unzipped_args, dispatch_label=label)
        )
        self._dispatch_count += 1

    def _record_cpu_bounded_dispatch(
        self,
        kernel_cpp,
        unzipped_args,
        extent,
        capacity,
        label="",
    ):
        self._bind_declared_private_args(unzipped_args)
        self._active_bounded_publication = None
        self._aot_graph_plan.dispatch(kernel_cpp, unzipped_args, label)
        self._ensure_runtime_graph_builder().dispatch_cpu_bounded(
            kernel_cpp,
            unzipped_args,
            extent,
            capacity,
            label,
        )
        self._runtime_graph_dispatches.append(
            _RecordingDispatch(kernel_cpp, tuple(unzipped_args))
        )
        self._runtime_graph_arg_names.update(_runtime_arg_names(unzipped_args))
        self._pending_ir_nodes.append(
            _dispatch_ir_node(kernel_cpp, unzipped_args, dispatch_label=label)
        )
        self._dispatch_count += 1

    def create_sequential(self):
        return Sequential()

    def append(self, node):
        # TODO: support appending dispatch node as well.
        assert isinstance(node, Sequential)
        self._active_bounded_publication = None
        if node._source_native_count or node._structured_depth:
            self._flush_graph_builder()
            self._nodes.append(
                _compile_sequential_runtime_node(
                    (node,), name="sequential", region_kind="sequential"
                )
            )
            return self
        self._aot_graph_plan.append(node)
        recording_dispatches = node._dispatch_to(self._runtime_graph_builder)
        for binding_name, value in node._fixed_runtime_args.items():
            existing = self._runtime_graph_fixed_args.get(binding_name)
            if existing is not None and existing is not value:
                if not (
                    isinstance(existing, (int, float))
                    and isinstance(value, (int, float))
                    and existing == value
                ):
                    raise TaichiRuntimeError(
                        "Appended Graph sequences provide conflicting fixed "
                        f"binding {binding_name!r}"
                    )
            self._runtime_graph_fixed_args[binding_name] = value
        for lease in node._lifetime_leases:
            self._retain_runtime_graph_lease(lease)
        if node._fixed_runtime_args:
            self._aot_graph_plan.mark_internal_fixed_bindings()
        self._runtime_graph_dispatches.extend(recording_dispatches)
        self._runtime_graph_arg_names.update(node._runtime_arg_names)
        self._dispatch_count += node._dispatch_count
        self._pending_ir_nodes.extend(node._ir_nodes)
        if self._capture_recipe_sources:
            self._runtime_graph_recipe_operations.append(
                ("sequential", _GraphSequentialRecipeSnapshot(node))
            )

        return self

    def _ensure_runtime_graph_builder(self):
        return self._runtime_graph_builder

    def _set_cuda_parallel_dispatch_groups(self, groups):
        groups = tuple(tuple(int(index) for index in group) for group in groups)
        if self._dispatch_count == 0 or len(groups) < 2:
            raise TaichiRuntimeError(
                "CUDA parallel Graph schedule requires an active CGraph segment"
            )
        self._runtime_graph_builder._set_cuda_parallel_dispatch_groups(groups)
        self._cuda_parallel_dispatch_groups = groups

    def _flush_graph_builder(self):
        if self._dispatch_count == 0:
            return
        # ``_aot_graph_plan`` is the durable source of truth for dispatches.
        # Only recover items added since the previous flush: legacy low-level
        # adapters can bypass ``_record_dispatch()``, but names from an older
        # segment must not leak across a native node boundary.
        self._runtime_graph_arg_names.update(
            self._aot_graph_plan.runtime_arg_names_since(self._aot_plan_cursor)
        )
        self._aot_plan_cursor = self._aot_graph_plan.item_count
        compiled_graph = self._runtime_graph_builder.compile()
        ir_nodes = _compiled_dispatch_ir_nodes(compiled_graph, self._pending_ir_nodes)
        self._nodes.append(
            _CompiledCGraphNode(
                compiled_graph,
                self._dispatch_count,
                self._runtime_graph_arg_names,
                SequentialRegion(ir_nodes, name="cgraph"),
                recording_dispatches=self._runtime_graph_dispatches,
                fixed_runtime_args=self._runtime_graph_fixed_args,
                lifetime_leases=self._runtime_graph_lifetime_leases,
                source_native_count=self._runtime_graph_source_native_count,
                native_action_manifests=(self._runtime_graph_native_action_manifests),
                recipe_operations=self._runtime_graph_recipe_operations,
                parallel_dispatch_groups=self._cuda_parallel_dispatch_groups,
            )
        )
        self._runtime_graph_builder = _new_runtime_graph_builder(
            self._explicit_map_source_groups
        )
        self._dispatch_count = 0
        self._runtime_graph_arg_names = set()
        self._runtime_graph_dispatches = []
        self._pending_ir_nodes = []
        self._runtime_graph_fixed_args = {}
        self._runtime_graph_lifetime_leases = []
        self._runtime_graph_source_native_count = 0
        self._runtime_graph_native_action_manifests = []
        self._runtime_graph_recipe_operations = []
        self._cuda_parallel_dispatch_groups = ()

        self._active_bounded_publication = None

    def _append_native(self, node, *, prewarm=False, admission="explicit"):
        if admission not in ("explicit", "auto"):
            raise TaichiRuntimeError(
                "Graph native admission must be 'explicit' or 'auto'"
            )
        describe = None if prewarm else getattr(node, "_graph_recipe_description", None)
        executable = None if describe is None else describe()
        if executable is None:
            executable = compile_native_graph_node(node)
        if prewarm:
            executable.prewarm()
        return self._append_native_executable(executable, admission=admission)

    def _append_native_executable(self, executable, *, admission):
        """Record a frozen native executable during build, never during replay."""
        structured = executable.recordable_sequence
        if structured is not None:
            sequence = Sequential()
            sequence._append_recordable_sequence(structured, executable)
            return self.append(sequence)
        action = executable.recordable_action
        if action is not None and action.backend_command_recording is not None:
            backend = _backend_name(_ti_core.arch_name(impl.current_cfg().arch))
            if not action.supports_backend(backend):
                raise TaichiRuntimeError(
                    "Backend command action is compiled for "
                    f"{action.backend_command_recording.backend}, not the active "
                    f"{backend} backend"
                )
            recording = action.backend_command_recording
            capture_recipe = getattr(recording, "_cuda_capture_recipe", None)
            if (
                backend == "cuda"
                and recording.replay_mode == "stream_capture"
                and isinstance(capture_recipe, _CudaGraphCaptureRecipe)
            ):
                compiled = _CompiledNativeGraphNode(executable)
                # Retained addon workspace and other private resources belong
                # to the CGraph binding frame too. Validate conflicts before
                # mutating the native builder, at this cold build boundary.
                for name, value in compiled.fixed_runtime_args.items():
                    existing = self._runtime_graph_fixed_args.get(name)
                    if existing is not None and existing is not value:
                        raise TaichiRuntimeError(
                            "CUDA capture actions provide conflicting fixed "
                            f"binding {name!r}"
                        )
                capture_recipe.append_to_graph(
                    self._ensure_runtime_graph_builder(),
                    impl.get_runtime().prog,
                )
                self._dispatch_count += 1
                self._runtime_graph_arg_names.update(
                    compiled.recording_runtime_arg_names
                )
                self._runtime_graph_fixed_args.update(compiled.fixed_runtime_args)
                self._runtime_graph_lifetime_leases.extend(compiled.lifetime_leases)
                self._runtime_graph_source_native_count += compiled.source_native_count
                self._runtime_graph_native_action_manifests.extend(
                    compiled.native_action_manifests
                )
                self._pending_ir_nodes.append(compiled.ir_node)
                if self._capture_recipe_sources:
                    self._runtime_graph_recipe_operations.append(
                        ("native", executable, admission)
                    )
                # Provider pointers are JIT-only and must never leak into the
                # backend-neutral serialized Graph schema.
                self._aot_graph_plan.mark_internal_fixed_bindings()
                return self
        if admission == "auto":
            backend = _backend_name(_ti_core.arch_name(impl.current_cfg().arch))
            action = executable.recordable_action
            if action is None or not action.supports_backend(backend):
                manifest = native_action_manifest(executable, action)
                raise TaichiRuntimeError(
                    "Automatic native Graph admission rejected a fragmented "
                    f"provider plan: {manifest.name}:"
                    f"{manifest.fragmentation_reason}. Use admission='explicit' "
                    "only for diagnostic segmented execution."
                )
        self._active_bounded_publication = None
        self._flush_graph_builder()
        self._nodes.append(_CompiledNativeGraphNode(executable))
        return self

    @staticmethod
    def _control_name(value, role):
        return _graph_control_name(value, role)

    @classmethod
    def _control_names(cls, values, role):
        return _graph_control_names(values, role)

    def while_loop(
        self,
        condition,
        body,
        *,
        predicate,
        max_iterations,
        control_inputs=(),
        carried_state=(),
        counter=None,
        status=None,
        chunk_size=None,
        vulkan_first_chunk_strategy="auto",
        masked_execution=False,
        lowering_mode="auto",
        name="while",
    ):
        """Append a structured, capped while region.

        ``condition`` computes ``predicate`` from any number of declared
        ``control_inputs``. Nonzero means continue. ``body`` updates the fixed
        ``carried_state`` and optional exact iteration ``counter``. An optional
        user-defined ``status`` resource records why iteration terminated; it
        is observed with the predicate but never interpreted by Graph. Backend
        lowering is selected automatically unless ``lowering_mode`` requests
        a portable or required-native path. Vulkan compound submission honors
        an explicit per-region ``chunk_size`` and can select ``compact`` or
        ``coarse_conditional`` for its first chunk through
        ``vulkan_first_chunk_strategy``.
        """
        self._flush_graph_builder()
        predicate_name = self._control_name(predicate, "while predicate")
        counter_name = (
            None if counter is None else self._control_name(counter, "while counter")
        )
        status_name = (
            None if status is None else self._control_name(status, "while status")
        )
        self._nodes.append(
            _CompiledWhileGraphNode(
                condition,
                body,
                predicate=predicate_name,
                control_inputs=self._control_names(
                    control_inputs, "while control_inputs"
                ),
                carried_state=self._control_names(carried_state, "while carried_state"),
                max_iterations=max_iterations,
                counter=counter_name,
                status=status_name,
                chunk_size=chunk_size,
                vulkan_first_chunk_strategy=vulkan_first_chunk_strategy,
                masked_execution=masked_execution,
                lowering_mode=lowering_mode,
                name=name,
            )
        )
        return self

    def if_then_else(
        self,
        condition,
        then_region,
        *,
        predicate,
        control_inputs=(),
        else_region=None,
        lowering_mode="auto",
        name="if",
    ):
        """Append a structured conditional region."""
        self._flush_graph_builder()
        self._nodes.append(
            _CompiledIfGraphNode(
                condition,
                then_region,
                else_region,
                predicate=self._control_name(predicate, "if predicate"),
                control_inputs=self._control_names(control_inputs, "if control_inputs"),
                lowering_mode=lowering_mode,
                name=name,
            )
        )
        return self

    def switch(
        self,
        condition,
        branches,
        *,
        selector,
        control_inputs=(),
        default_region=None,
        lowering_mode="auto",
        name="switch",
    ):
        """Append a zero-based structured switch region."""
        self._flush_graph_builder()
        self._nodes.append(
            _CompiledSwitchGraphNode(
                condition,
                tuple(branches),
                default_region,
                selector=self._control_name(selector, "switch selector"),
                control_inputs=self._control_names(
                    control_inputs, "switch control_inputs"
                ),
                lowering_mode=lowering_mode,
                name=name,
            )
        )
        return self

    def observe(self, *values, name="observation"):
        """Append a deferred packed snapshot of scalar ndarray arguments.

        ``Graph.submit()`` captures values on device and returns before host
        readback. Consume the immutable snapshot through
        ``SubmissionTicket.observations()``.
        """
        if name in self._observation_names:
            raise TaichiRuntimeError(
                f"Graph observation name {name!r} is already defined"
            )
        node = _CompiledObservationGraphNode(values, name)
        self._flush_graph_builder()
        self._nodes.append(node)
        self._observation_names.add(name)
        return self

    def append_native(self, node, *, prewarm=False, admission="explicit"):
        """Append a native provider under explicit or fail-closed admission.

        ``admission='auto'`` accepts only a provider that can be integrated
        into the enclosing backend Graph.  ``'explicit'`` preserves the
        diagnostic segmented route for providers whose backend command plan
        is not yet integrated.
        """
        return self._append_native(node, prewarm=prewarm, admission=admission)

    def freeze(self):
        """Freeze this builder into an immutable semantic Graph definition."""

        self._flush_graph_builder()
        if not self._nodes:
            compiled_graph = self._ensure_runtime_graph_builder().compile()
            spec = _GraphSpec(
                [
                    _CompiledCGraphNode(
                        compiled_graph,
                        0,
                        (),
                        SequentialRegion((), name="cgraph"),
                    )
                ],
                aot_compiled_graph=compiled_graph,
            )
        else:
            spec = _GraphSpec(
                self._nodes,
                aot_graph_builder=self._aot_graph_plan.snapshot(),
                graph_memory_sources=tuple(self._graph_memory_sources),
                graph_offload_fusion_sources=tuple(self._graph_offload_fusion_sources),
                graph_sparse_traversal_sources=tuple(
                    self._graph_sparse_traversal_sources
                ),
                graph_bounded_sources=tuple(self._graph_bounded_sources),
                graph_reduction_sources=tuple(self._graph_reduction_sources),
                graph_native_algorithm_sources=tuple(
                    self._graph_native_algorithm_sources
                ),
            )
        backend = _backend_name(_ti_core.arch_name(impl.current_cfg().arch))
        return GraphDefinition._from_graph_spec(
            spec,
            backend,
            core_commit=str(_ti_core.get_commit_hash()).lower(),
        )

    def compile(self, *, workspace_lanes=1, workspace_saturation="wait"):
        return self.freeze().compile(
            workspace_lanes=workspace_lanes,
            workspace_saturation=workspace_saturation,
        )


class SubmissionTicket:
    """Completion for one opt-in ``Graph.submit()`` invocation.

    Construct tickets through ``Graph.submit()``. The runtime keeps native
    Graph ownership valid until backend completion, even when the user drops
    the ticket without waiting.
    """

    __slots__ = (
        "_admission",
        "_completion",
        "_completion_observations",
        "_graph_token",
        "_observation",
        "_runtime_owner_retained",
        "_runtime",
        "_submission_owners",
        "_telemetry",
        "_workspace_lane",
        "_workspace_lanes",
    )

    def __init__(
        self,
        completion,
        runtime,
        admission=None,
        observation=None,
        telemetry=None,
        workspace_lane=0,
        workspace_lanes=None,
        submission_owners=(),
        completion_observations=(),
        runtime_owner_retained=False,
        graph_token=None,
    ):
        self._admission = admission
        self._completion = completion
        self._completion_observations = tuple(completion_observations)
        self._graph_token = graph_token
        self._observation = observation
        self._runtime_owner_retained = bool(runtime_owner_retained)
        self._runtime = runtime
        self._submission_owners = tuple(submission_owners)
        self._telemetry = telemetry
        self._workspace_lane = int(workspace_lane)
        self._workspace_lanes = tuple(
            (self._workspace_lane,)
            if workspace_lanes is None
            else (int(value) for value in workspace_lanes)
        )

    def _observe_completion(self):
        observations = self._completion_observations
        if not observations:
            return
        self._completion_observations = ()
        for observation in observations:
            observation._observe_graph_completion()

    def done(self):
        if self._admission is None:
            ready = self._completion.done()
        else:
            ready = self._admission._completion_done(self._completion)
        if ready:
            self._observe_completion()
            if self._runtime_owner_retained:
                self._runtime.release_runtime_submission_owner(self._completion)
                self._runtime_owner_retained = False
            self._submission_owners = ()
        return ready

    def wait(self):
        if self._admission is None:
            self._completion.wait()
        else:
            self._admission._completion_wait(self._completion)
        self._observe_completion()
        if self._runtime_owner_retained:
            self._runtime.release_runtime_submission_owner(self._completion)
            self._runtime_owner_retained = False
        self._submission_owners = ()

    def observations(self):
        """Wait if needed, then materialize this submission's snapshot."""
        if self._observation is None:
            return {}
        self.wait()
        return self._observation.materialize()

    def telemetry(self):
        """Wait if needed, then return opt-in per-submission telemetry."""
        if self._telemetry is None:
            return None
        self.wait()
        return self._telemetry.materialize()

    def pipeline_report(self):
        """Return this ticket's opt-in immutable execution-pipeline report."""
        telemetry = self.telemetry()
        return None if telemetry is None else telemetry.pipeline

    @property
    def backend(self):
        return self._completion.backend

    @property
    def sequence(self):
        return self._completion.sequence

    @property
    def workspace_lane(self):
        return self._workspace_lane

    @property
    def workspace_lanes(self):
        return self._workspace_lanes

    @property
    def _has_backend_work(self):
        return self._completion.has_backend_work

    def __del__(self):
        observations = getattr(self, "_completion_observations", ())
        if observations:
            try:
                if self._completion.done():
                    self._observe_completion()
            except Exception:
                pass

        observation = getattr(self, "_observation", None)
        if observation is not None:
            try:
                observation.discard()
            except Exception:
                pass
        telemetry = getattr(self, "_telemetry", None)
        if telemetry is not None:
            try:
                telemetry.discard()
            except Exception:
                pass


@dataclass(frozen=True)
class _GraphTerminalControlReport:
    logical_iterations: int
    executed_iterations: int
    observation_batches: int
    observation_boundaries: tuple
    lowering: str
    encoded_iterations: int
    masked_iterations: int
    chunk_sizes: tuple


@dataclass(frozen=True)
class _GraphTerminalObservation:
    value: object
    control_report: _GraphTerminalControlReport
    backend: str
    sequence: object
    workspace_lane: int


def _workspace_lane_configuration(workspace_lanes, workspace_saturation):
    if isinstance(workspace_lanes, bool) or not isinstance(
        workspace_lanes, (int, np.integer)
    ):
        raise TaichiRuntimeError("Graph workspace_lanes must be an integer")
    workspace_lanes = int(workspace_lanes)
    if workspace_lanes < 1 or workspace_lanes > 64:
        raise TaichiRuntimeError("Graph workspace_lanes must be between 1 and 64")
    if workspace_saturation not in ("wait", "raise"):
        raise TaichiRuntimeError("Graph workspace_saturation must be 'wait' or 'raise'")
    return workspace_lanes, workspace_saturation


def _graph_fusion_runtime_scope(backend):
    scope = {"core_commit": str(_ti_core.get_commit_hash()).lower()}
    if backend != "cuda":
        return scope
    try:
        from taichi_forge.interop import current_cuda_device_uuid

        scope.update(
            {
                "cuda_compute_capability": int(impl.get_cuda_compute_capability()),
                "cuda_device_uuid": current_cuda_device_uuid().hex(),
                "cuda_driver_api_version": int(_ti_core.cuda_driver_api_version()),
            }
        )
    except (AttributeError, RuntimeError, ValueError):
        return None
    return scope


def _graph_fusion_binding_descriptor(value):
    if isinstance(value, Ndarray):
        if value.arr is None or value.shape is None or value.dtype is None:
            return None
        try:
            shape = tuple(int(extent) for extent in value.shape)
            element_shape = tuple(int(extent) for extent in value.element_shape)
        except (AttributeError, TypeError, ValueError):
            return None
        return {
            "kind": "ndarray",
            "dtype": str(value.dtype),
            "rank": len(shape),
            "shape": shape,
            "element_shape": element_shape,
        }
    if isinstance(value, DenseNdarrayView):
        try:
            shape = tuple(int(extent) for extent in value.shape)
            element_shape = tuple(
                int(extent) for extent in value.descriptor.element_shape
            )
            dtype = str(value.descriptor.scalar_type)
        except (AttributeError, TypeError, ValueError):
            return None
        return {
            "kind": "ndarray",
            "dtype": dtype,
            "rank": len(shape),
            "shape": shape,
            "element_shape": element_shape,
        }
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    ):
        scalar = value.item() if hasattr(value, "item") else value
        return {"kind": "scalar", "value": scalar}
    return None


def _graph_fusion_binding_descriptors(args):
    if not isinstance(args, Mapping):
        return {}
    result = {}
    for name, value in args.items():
        descriptor = _graph_fusion_binding_descriptor(value)
        if descriptor is not None:
            result[str(name)] = descriptor
    return result


def _graph_fusion_group_size(spec):
    if not spec.fusion_recipe_ids:
        return 1
    sizes = []
    for recipe_id in spec.fusion_recipe_ids:
        fields = recipe_id.split(":")
        if (
            len(fields) != 3
            or fields[0] != "fusion"
            or not fields[1].startswith("map")
            or len(fields[2]) != 24
            or any(character not in "0123456789abcdef" for character in fields[2])
        ):
            raise TaichiRuntimeError(
                f"Unsupported qualified fusion recipe {recipe_id!r}"
            )
        try:
            size = int(fields[1][3:])
        except ValueError as exc:
            raise TaichiRuntimeError(
                f"Unsupported qualified fusion recipe {recipe_id!r}"
            ) from exc
        if size < 2 or size > 4:
            raise TaichiRuntimeError(
                f"Unsupported qualified fusion recipe {recipe_id!r}"
            )
        sizes.append(size)
    return max(sizes)


def _graph_fusion_source_groups(spec, fusion_plan):
    recipes = {recipe.recipe_id: recipe for recipe in fusion_plan.candidate_recipes}
    groups = []
    claimed = set()
    for recipe_id in spec.fusion_recipe_ids:
        recipe = recipes.get(recipe_id)
        if recipe is None:
            raise TaichiRuntimeError(
                f"Qualified fusion recipe {recipe_id!r} is not in the semantic plan"
            )
        group = []
        for source_id in recipe.source_dispatch_ids:
            marker = source_id.rsplit("/dispatch:", 1)
            if len(marker) != 2 or not marker[1].isdigit():
                raise TaichiRuntimeError(
                    "Qualified fusion recipe has no exact logical dispatch ID"
                )
            logical_id = int(marker[1])
            if logical_id in claimed:
                raise TaichiRuntimeError(
                    "Qualified fusion recipes overlap one logical dispatch"
                )
            claimed.add(logical_id)
            group.append(logical_id)
        groups.append(tuple(group))
    return tuple(groups)


class _QualifiedFusionRuntimeSelector:
    def __init__(self, cache, expected_replays, space, runtime_scope):
        self._cache = cache
        self._expected_replays = expected_replays
        self._space = space
        self._runtime_scope = dict(runtime_scope)
        self._source_commit = str(_ti_core.get_commit_hash()).lower()
        self._specs = {
            spec.spec_id: spec
            for spec in (self._space.baseline, *self._space.candidates)
        }
        self._variants = {}
        self._failed_entries = set()
        self._attempts = 0
        self._qualified_selections = 0
        self._baseline_fallbacks = 0
        self._materializations = 0
        self._last_reason = "not_invoked"
        self._last_entry_id = None

    @classmethod
    def from_environment(cls, spec):
        path = os.environ.get(_INTERNAL_FUSION_QUALIFICATION_ENV, "").strip()
        if not path:
            return None
        space = spec.executable_optimization_space
        # Runtime auto-selection currently has CUDA performance evidence only.
        # CPU/Vulkan may still materialize recipes explicitly for correctness
        # qualification, but must not consume CUDA admission records.
        if space.baseline.backend != "cuda":
            return None
        raw_replays = os.environ.get(_INTERNAL_FUSION_EXPECTED_REPLAYS_ENV, "").strip()
        try:
            expected_replays = int(raw_replays)
        except ValueError as exc:
            raise TaichiRuntimeError(
                f"{_INTERNAL_FUSION_EXPECTED_REPLAYS_ENV} must be a positive integer"
            ) from exc
        if expected_replays < 1:
            raise TaichiRuntimeError(
                f"{_INTERNAL_FUSION_EXPECTED_REPLAYS_ENV} must be a positive integer"
            )
        try:
            cache = _GraphFusionQualificationCache.load(path)
        except ValueError as exc:
            raise TaichiRuntimeError(str(exc)) from exc
        runtime_scope = _graph_fusion_runtime_scope(space.baseline.backend)
        if runtime_scope is None:
            return None
        return cls(cache, expected_replays, space, runtime_scope)

    def select(self, graph, args):
        self._attempts += 1
        entry, reason = self._cache.select(
            semantic_plan_id=self._space.semantic_plan_id,
            backend=self._space.baseline.backend,
            source_commit=self._source_commit,
            runtime_scope=self._runtime_scope,
            bindings=_graph_fusion_binding_descriptors(args),
            expected_replays=self._expected_replays,
        )
        if entry is None:
            self._baseline_fallbacks += 1
            self._last_reason = reason
            self._last_entry_id = None
            return graph._instance
        self._last_entry_id = entry.identity
        if entry.identity in self._failed_entries:
            self._baseline_fallbacks += 1
            self._last_reason = "materialization_previously_failed"
            return graph._instance
        spec = self._specs.get(entry.selected_spec_id)
        if (
            spec is None
            or spec is self._space.baseline
            or spec.execution_identity != entry.execution_identity
            or self._space.baseline.execution_identity
            != entry.baseline_execution_identity
            or self._space.selected_spec_id != self._space.baseline.spec_id
        ):
            self._baseline_fallbacks += 1
            self._last_reason = "executable_scope_mismatch"
            return graph._instance
        instance = self._variants.get(entry.identity)
        if instance is None:
            try:
                instance = graph._materialize_qualified_fusion_instance(spec)
            except (RuntimeError, ValueError) as exc:
                self._failed_entries.add(entry.identity)
                self._baseline_fallbacks += 1
                self._last_reason = f"materialization_failed:{type(exc).__name__}"
                return graph._instance
            self._variants[entry.identity] = instance
            self._materializations += 1
        self._qualified_selections += 1
        self._last_reason = "qualified"
        return instance

    def invalidate_runtime(self, preserve_executables=False):
        for instance in self._variants.values():
            instance.invalidate_runtime(preserve_executables=preserve_executables)
        if not preserve_executables:
            self._variants.clear()

    @property
    def stats(self):
        return MappingProxyType(
            {
                "configured": True,
                "cache_path": self._cache.source_path,
                "expected_replays": self._expected_replays,
                "attempts": self._attempts,
                "qualified_selections": self._qualified_selections,
                "baseline_fallbacks": self._baseline_fallbacks,
                "materializations": self._materializations,
                "retained_variants": len(self._variants),
                "last_reason": self._last_reason,
                "last_entry_id": self._last_entry_id,
            }
        )


class Graph:
    def __init__(
        self,
        compiled_graph,
        *,
        workspace_lanes=1,
        workspace_saturation="wait",
        definition=None,
        _cuda_concurrent_workspace_pair=False,
    ) -> None:
        self._lifecycle_lock = threading.Lock()
        self._terminal_observation_token = object()
        self._stale_snode_tree_dependencies = set()
        (
            self._workspace_lane_capacity,
            self._workspace_saturation,
        ) = _workspace_lane_configuration(workspace_lanes, workspace_saturation)
        if isinstance(compiled_graph, _GraphSpec):
            self._spec = compiled_graph
        elif isinstance(compiled_graph, _CompiledCGraphNode):
            self._spec = _GraphSpec(
                [compiled_graph], aot_compiled_graph=compiled_graph.compiled_graph
            )
        else:
            node = _CompiledCGraphNode(compiled_graph, 0, ())
            self._spec = _GraphSpec([node], aot_compiled_graph=compiled_graph)
        if definition is None:
            backend = _backend_name(_ti_core.arch_name(impl.current_cfg().arch))
            definition = GraphDefinition._from_graph_spec(
                self._spec,
                backend,
                core_commit=str(_ti_core.get_commit_hash()).lower(),
            )
        elif (
            definition._runtime_spec is not self._spec
            and getattr(
                self._spec,
                "_definition_source_spec",
                None,
            )
            is not definition._runtime_spec
        ):
            raise TaichiRuntimeError(
                "GraphDefinition baseline spec does not match this Graph"
            )
        self._definition = definition
        self._cuda_concurrent_workspace_pair = bool(_cuda_concurrent_workspace_pair)
        if self._cuda_concurrent_workspace_pair and self._workspace_lane_capacity != 2:
            raise TaichiRuntimeError(
                "CUDA concurrent workspace pair recipes require exactly two lanes"
            )
        if (
            self._spec.exclusive_provider_submission
            and self._workspace_lane_capacity != 1
        ):
            raise TaichiRuntimeError(
                "Graphs with exclusive provider-owned fixed storage require "
                "workspace_lanes=1; use independent providers for concurrency"
            )
        self._contains_native_nodes_value = self._spec.native_count > 0
        self._contains_structured_control_value = (
            self._spec.structured_control_count > 0
        )
        self._contains_structured_while_value = self._spec.structured_while_count > 0
        self._contains_observations_value = self._spec.observation_count > 0
        self._has_native_execution_observers = bool(
            self._spec.native_execution_observer_leases
        )
        self._last_observations = {}
        self._latest_control_flow_was_async = False
        self._submission_lane = _new_submission_lane("graph")
        self._execution_definition = self._spec.execution_definition
        self._execution_arch = _ti_core.arch_name(impl.current_cfg().arch)
        self._qualified_fusion_selector = (
            None
            if getattr(self._spec, "_disable_qualified_fusion_selector", False)
            else _QualifiedFusionRuntimeSelector.from_environment(
                self._spec
            )
        )
        self._instances = {}
        self._workspace_pool = self._workspace_pool_for_current_runtime()
        self._instance = self._workspace_pool.primary
        if self._cuda_concurrent_workspace_pair:
            self._workspace_pool.materialize_pair()
        self._latest_instance = self._instance
        self._prepared_telemetry_modes = set()
        self._runtime_valid = True
        self._run_impl = self._instance.run_impl
        impl.get_runtime().register_runtime_object(self)

    def bind(self, arguments):
        """Create a versioned runtime binding source for repeat invocation.

        Raw dictionaries remain supported and are fully validated on every
        replay. A returned :class:`GraphBindingSet` can reuse an unchanged,
        preflattened frame only when every resource has a provably stable
        identity; ``statistics()`` reports conservative qualification blockers.
        """

        return GraphBindingSet(self, arguments)

    def bind_batch(self, invocations):
        """Publish exactly two independent binding frames for pair execution.

        This is the workload boundary used by the complete workspace-
        concurrency recipe.  It accepts no lane or saturation parameter: the
        recipe owns a fixed two-lane physical executor, while CompileIQ chooses
        between that executor and the serial complete-Graph baseline.
        """

        if not isinstance(invocations, (tuple, list)) or len(invocations) != 2:
            raise TaichiRuntimeError(
                "Graph.bind_batch() requires exactly two invocation mappings"
            )
        if not all(isinstance(item, Mapping) for item in invocations):
            raise TypeError("Graph.bind_batch() invocations must be mappings")
        with self._lifecycle_lock:
            self._check_runtime_valid()
            if not _workspace_concurrency_spec_eligible(
                self._spec, self._execution_arch
            ):
                raise TaichiRuntimeError(
                    "Graph.bind_batch() requires an eligible CUDA JIT Graph "
                    "with exclusive private workspace"
                )
            instances = (
                self._workspace_pool.materialize_pair()
                if self._cuda_concurrent_workspace_pair
                else (self._instance, self._instance)
            )
            versions = tuple(
                self._spec.build_binding_version(
                    arguments,
                    index + 1,
                    fixed_runtime_args=instance._fixed_runtime_args,
                    allow_fixed_binding_fast_path=True,
                    allow_fast_path=self._qualified_fusion_selector is None,
                    entrypoint="Graph.bind_batch",
                )
                for index, (arguments, instance) in enumerate(
                    zip(invocations, instances)
                )
            )
            if not all(version.fast_path_qualified for version in versions):
                blockers = sorted(
                    {
                        blocker
                        for version in versions
                        for blocker in version.volatile_reasons
                    }
                )
                raise TaichiRuntimeError(
                    "Graph.bind_batch() requires immutable fast-qualified "
                    "bindings; blockers: " + ", ".join(blockers)
                )
            disjoint_pairs = _certify_graph_invocation_pair(self._spec, versions)
            return GraphInvocationBatch(
                self,
                int(impl.runtime_generation()),
                versions,
                disjoint_pairs,
            )

    @property
    def definition(self):
        """Return the immutable semantic definition used by this Graph."""

        return self._definition

    def binding_plan(self):
        """Return the immutable compiled public binding slot plan."""

        return self._spec.binding_plan.to_dict()

    def binding_statistics(self):
        """Return host binding preparation counters for this Graph."""

        with self._lifecycle_lock:
            return self._spec.binding_statistics()

    def _initialize_binding_set(self, binding_set, arguments):
        with self._lifecycle_lock:
            self._check_runtime_valid()
            version = self._spec.build_binding_version(
                arguments,
                1,
                fixed_runtime_args=self._instance._fixed_runtime_args,
                allow_fixed_binding_fast_path=(self._workspace_lane_capacity == 1),
                allow_fast_path=self._qualified_fusion_selector is None,
            )
            with binding_set._lock:
                binding_set._version = self._instance.prepare_binding_version(version)

    def _update_binding_set(self, binding_set, values, *, replace_all):
        if not isinstance(values, Mapping):
            raise TypeError("GraphBindingSet bindings must be a mapping")
        with self._lifecycle_lock:
            self._check_runtime_valid()
            if binding_set._graph is not self:
                raise TaichiRuntimeError(
                    "GraphBindingSet belongs to a different compiled Graph"
                )
            with binding_set._lock:
                current = binding_set._version
                if current.runtime_generation != int(impl.runtime_generation()):
                    raise TaichiRuntimeError(
                        "GraphBindingSet was published before ti.reset() or a "
                        "runtime reinitialization"
                    )
                if replace_all:
                    candidate = dict(values)
                else:
                    unexpected = values.keys() - self._spec.runtime_arg_names
                    if unexpected:
                        raise TaichiRuntimeError(
                            "Unexpected graph runtime arguments: "
                            + ", ".join(sorted(unexpected))
                        )
                    candidate = dict(current.arguments)
                    candidate.update(values)
                revision = current.revision + 1
                if revision > 0x7FFFFFFFFFFFFFFF:
                    raise TaichiRuntimeError(
                        "GraphBindingSet revision space is exhausted"
                    )
                version = self._spec.build_binding_version(
                    candidate,
                    revision,
                    fixed_runtime_args=self._instance._fixed_runtime_args,
                    allow_fixed_binding_fast_path=(self._workspace_lane_capacity == 1),
                    allow_fast_path=self._qualified_fusion_selector is None,
                    entrypoint="GraphBindingSet.update",
                )
                version = self._instance.prepare_binding_version(version)
                binding_set_ref = weakref.ref(binding_set)
                retired_revision = current.revision

                def discard_retired_version(retired_ref):
                    owner = binding_set_ref()
                    if owner is None:
                        return
                    # Cyclic GC can invoke this callback on the thread already
                    # publishing an update, hence the binding lock is
                    # re-entrant. The monotonic revision plus identity check
                    # keeps removal exact even after a container replacement.
                    with owner._lock:
                        retired_versions = owner._retired_versions
                        if retired_versions.get(retired_revision) is retired_ref:
                            retired_versions.pop(retired_revision, None)
                            if not retired_versions:
                                # Drop the potentially high-water dict table
                                # after the last ticket/in-flight owner exits.
                                owner._retired_versions = {}

                retired_ref = weakref.ref(current, discard_retired_version)
                binding_set._retired_versions[retired_revision] = retired_ref
                binding_set._version = version

    def _snapshot_binding_source(self, source):
        if not isinstance(source, GraphBindingSet):
            if not isinstance(source, (dict, MappingProxyType)):
                return source, None
            # Compatibility mappings have no revision notification. A
            # MappingProxyType can still wrap a caller-owned mutable dict, so
            # copy both accepted mapping forms after admission. Later caller
            # mutation then cannot change the frame between validation and a
            # pybind launch that releases the GIL. Use a GraphBindingSet when
            # rebinding must be atomic across threads.
            return {
                name: _snapshot_graph_binding_value(value)
                for name, value in source.items()
            }, None
        if source._graph is not self:
            raise TaichiRuntimeError(
                "GraphBindingSet belongs to a different compiled Graph"
            )
        with source._lock:
            version = source._version
        if version.runtime_generation != int(impl.runtime_generation()):
            raise TaichiRuntimeError(
                "GraphBindingSet was published before ti.reset() or a runtime "
                "reinitialization"
            )
        return version.arguments, version

    def _materialize_qualified_fusion_instance(self, selected_spec):
        if (
            self._workspace_lane_capacity != 1
            or self._spec.native_count
            or self._spec.structured_control_count
            or self._spec.observation_count
            or self._spec.fixed_runtime_args
            or self._spec.temporary_actions
            or self._spec.lifetime_leases
            or len(self._spec.nodes) != 1
            or not isinstance(self._spec.nodes[0], _CompiledCGraphNode)
            or self._spec._aot_graph_builder is None
        ):
            raise TaichiRuntimeError(
                "Qualified fusion materialization requires one ordinary JIT "
                "CGraph without provider, structured, temporary, or fixed state"
            )
        source = self._spec.nodes[0]
        if not isinstance(source.ir_node, SequentialRegion):
            raise TaichiRuntimeError(
                "Qualified fusion materialization lost its logical source region"
            )
        _graph_fusion_group_size(selected_spec)
        source_groups = _graph_fusion_source_groups(
            selected_spec, self._spec.fusion_plan
        )
        compiled_graph = self._spec._aot_graph_builder._compile_map_recipes(
            source_groups
        )
        ir_nodes = _compiled_dispatch_ir_nodes(compiled_graph, source.ir_node.children)
        variant_node = _CompiledCGraphNode(
            compiled_graph,
            source.dispatch_count,
            source.recording_runtime_arg_names,
            SequentialRegion(ir_nodes, name=source.ir_node.name),
            recording_dispatches=source.recording_dispatches,
            lifetime_leases=source.lifetime_leases,
            source_native_count=source.source_native_count,
            region_kind=source.region_kind,
            fixed_runtime_args=source.fixed_runtime_args,
            temporary_actions=source.temporary_actions,
            native_action_manifests=source.native_action_manifests,
        )
        variant_spec = _GraphSpec(
            [variant_node],
            aot_graph_builder=self._spec._aot_graph_builder,
        )
        variant_space = variant_spec.executable_optimization_space
        if (
            variant_space.semantic_plan_id
            != self._spec.executable_optimization_space.semantic_plan_id
            or variant_space.selected_spec_id != selected_spec.spec_id
            or variant_space.selected is None
            or variant_space.selected.execution_identity
            != selected_spec.execution_identity
        ):
            raise TaichiRuntimeError(
                "Qualified fusion recipe did not materialize its exact "
                "semantic and physical executable identity"
            )
        return variant_spec.instantiate()

    def _qualified_execution_instance(self, args):
        if self._qualified_fusion_selector is None:
            return self._instance
        return self._qualified_fusion_selector.select(self, args)

    def _supports_terminal_observation(self):
        with self._lifecycle_lock:
            self._check_runtime_valid()
            roots = tuple(
                node
                for node in self._spec.structured_control_nodes
                if node.control_depth == 1 and isinstance(node, _CompiledWhileGraphNode)
            )
            return bool(
                len(roots) == 1
                and self._spec.supports_native_structured_submission
                and impl.current_cfg().arch
                in (_ti_core.Arch.cuda, _ti_core.Arch.vulkan)
            )

    def _observe_terminal_submission(
        self,
        ticket,
        materialize,
        *,
        logical_iterations,
    ):
        """Wait, materialize one terminal value, and derive control metadata."""

        if (
            not isinstance(ticket, SubmissionTicket)
            or ticket._graph_token is not self._terminal_observation_token
        ):
            raise TaichiRuntimeError(
                "Terminal-only Graph observation requires this Graph's "
                "SubmissionTicket"
            )
        if not callable(materialize) or not callable(logical_iterations):
            raise TaichiRuntimeError(
                "Terminal-only Graph observation requires materialization "
                "and logical-iteration callbacks"
            )
        with self._lifecycle_lock:
            self._check_runtime_valid()
            if not self._spec.supports_native_structured_submission:
                raise TaichiRuntimeError(
                    "Terminal-only Graph observation is unavailable for this Graph"
                )
        ticket.wait()
        value = materialize()
        logical = logical_iterations(value)
        with self._lifecycle_lock:
            self._check_runtime_valid()
            report = self._spec.terminal_control_report(logical)
        return _GraphTerminalObservation(
            value=value,
            control_report=report,
            backend=ticket.backend,
            sequence=ticket.sequence,
            workspace_lane=ticket.workspace_lane,
        )

    def _parallel_candidate_report(self, branches, args=None):
        """Analyze a possible root-level fork/join without executing it.

        ``branches`` contains two to four ordered, contiguous groups of root
        node indices. Supplying ``args`` resolves storage aliases against the
        current runtime; unknown aliases remain fail-closed. This private
        contract is intentionally analysis-only while parallel lowering is
        being qualified.
        """
        with self._lifecycle_lock:
            self._check_runtime_valid()
            return self._spec.parallel_candidate_report(branches, args)

    def task_manifest(self):
        """Return immutable per-task metadata for this JIT CGraph.

        This observation may compile a cold specialization, but it does not
        launch the Graph or allocate device telemetry. Native actions and
        structured-control regions do not yet have a single serializable
        CGraph task list and therefore reject this query explicitly.
        """
        from taichi_forge.lang.task_manifest import GraphTaskManifest

        with self._lifecycle_lock:
            self._check_runtime_valid()
            nodes = self._spec.nodes
            if len(nodes) != 1 or not isinstance(nodes[0], _CompiledCGraphNode):
                raise TaichiRuntimeError(
                    "Graph.task_manifest() currently requires one JIT CGraph "
                    "segment without native, observation, or structured-control nodes"
                )
            compiled_graph = nodes[0].compiled_graph
            raw = impl.get_runtime().prog._graph_task_manifest(compiled_graph)
        return tuple(GraphTaskManifest._from_core(item) for item in raw)

    def _gpu_semantics_snapshot(self):
        """Build a lazy, value-only CUDA/Vulkan executable-plan snapshot."""

        from taichi_forge.lang._gpu_semantics_graph import (
            _build_gpu_executable_plan_semantics,
        )

        with self._lifecycle_lock:
            self._check_runtime_valid()
            program = impl.get_runtime().prog
            stages = []
            pipeline = self._spec.pipeline_definition
            for node, stage in zip(self._spec.nodes, pipeline):
                raw = (
                    program._graph_gpu_semantics_snapshot(node.compiled_graph)
                    if isinstance(node, _CompiledCGraphNode)
                    else {"backend": self._execution_arch, "segments": ()}
                )
                if raw["backend"] != self._execution_arch:
                    raise TaichiRuntimeError(
                        "Graph GPU semantics backend changed after compilation"
                    )
                logical_order = _gpu_plan_logical_order(node.ir_node)
                actions = tuple(action.to_dict() for action in stage["native_actions"])
                topology_static = "structured" not in logical_order
                if not topology_static:
                    logical_order = tuple(
                        "native" if kind == "structured" else kind
                        for kind in logical_order
                    )
                    actions = (
                        *actions,
                        {
                            "name": f"structured:{stage['path_id']}",
                            "recordable": bool(
                                getattr(node, "supports_native_submission", False)
                            ),
                            "backends": (self._execution_arch,),
                            "runtime_bindings": tuple(
                                {
                                    "name": name,
                                    "kind": "opaque",
                                    "required": True,
                                }
                                for name in sorted(node.runtime_arg_names)
                            ),
                            "derived_runtime_bindings": (),
                            "effects": (),
                            "fixed_binding_names": (),
                        },
                    )
                stages.append(
                    {
                        "stage_index": int(stage["stage_index"]),
                        "path_id": str(stage["path_id"]),
                        "kind": str(stage["kind"]),
                        "region_kind": str(stage["region_kind"]),
                        "logical_order": logical_order,
                        "topology_static": topology_static,
                        "raw": raw,
                        "native_actions": actions,
                    }
                )
            temporary = self._spec.temporary_memory_plan
            definition = {
                "backend": self._execution_arch,
                "stages": tuple(stages),
                "workspace_lane_capacity": self._workspace_lane_capacity,
                "fixed_internal_storage_bytes": self._spec.internal_storage_bytes,
                "temporary_peak_bytes": (
                    temporary.planned_peak_bytes + temporary.opaque_bytes
                ),
                "lifetime_lease_count": len(self._spec.lifetime_leases),
                "executable_optimization": (
                    self._spec.executable_optimization_space.to_dict()
                ),
            }
        return _build_gpu_executable_plan_semantics(definition)

    def physical_plan(self):
        """Return the immutable logical-to-physical Graph execution plan.

        This is a compile/materialization report, not per-submit telemetry.
        Queue submissions remain ``None`` until a ticket is submitted with
        ``telemetry=True``.
        """

        with self._lifecycle_lock:
            self._check_runtime_valid()
            backend = _backend_name(self._execution_arch)
            stages = []
            recordable_actions = 0
            opaque_actions = 0
            loose_actions = 0
            loose_helper_count = 0
            loose_helper_count_exact = True
            rejection_reasons = []
            publications = {}
            for stage in self._spec.pipeline_definition:
                actions = tuple(stage["native_actions"])
                stage_reasons = []
                for action in actions:
                    if action.recordable and backend in action.backends:
                        recordable_actions += 1
                    else:
                        loose_actions += 1
                        if not action.recordable or action.opaque:
                            opaque_actions += 1
                            stage_reasons.append(
                                f"{action.name}:{action.fragmentation_reason}"
                            )
                        else:
                            stage_reasons.append(
                                f"{action.name}:backend_not_recordable:{backend}"
                            )
                        if action.loose_helper_count_exact:
                            loose_helper_count += int(action.loose_helper_count)
                        else:
                            loose_helper_count_exact = False
                rejection_reasons.extend(stage_reasons)
                for bounded in stage["bounded_dispatches"]:
                    key = tuple(bounded["publication_key"])
                    publication = publications.setdefault(
                        key,
                        {
                            "count_source": str(bounded["count_source"]),
                            "extent": str(bounded["count_name"]),
                            "capacity": int(bounded["capacity"]),
                            "block_dim": bounded["domain"].block_dim,
                            "publication_generation": bounded[
                                "domain"
                            ].publication_epoch,
                            "consumer_count": 0,
                        },
                    )
                    publication["consumer_count"] += 1
                stages.append(
                    MappingProxyType(
                        {
                            "stage_index": int(stage["stage_index"]),
                            "path_id": str(stage["path_id"]),
                            "name": str(stage["name"]),
                            "kind": str(stage["kind"]),
                            "logical_dispatches": int(stage["dispatch_count"]),
                            "physical_dispatches": int(
                                stage["physical_dispatch_count"]
                            ),
                            "native_actions": len(actions),
                            "providers": tuple(action.name for action in actions),
                            "recordable_native_actions": sum(
                                action.recordable and backend in action.backends
                                for action in actions
                            ),
                            "loose_native_actions": len(stage_reasons),
                            "backend_command_count": sum(
                                int(action.backend_command_count or 0)
                                for action in actions
                                if action.backend_command_count_exact
                            ),
                            "backend_command_count_exact": all(
                                action.backend_command_count_exact for action in actions
                            ),
                            "host_observation": stage["kind"] == "observation",
                            "rejection_reasons": tuple(stage_reasons),
                        }
                    )
                )
            temporary_plan = self._spec.temporary_memory_plan
            publication_plan = tuple(
                MappingProxyType(dict(publication))
                for _, publication in sorted(
                    publications.items(), key=lambda item: repr(item[0])
                )
            )
            return MappingProxyType(
                {
                    "schema_version": 1,
                    "backend": backend,
                    "logical_submission_count": 1,
                    "logical_node_count": len(self._spec.nodes),
                    "logical_dispatch_count": int(self._spec.dispatch_count),
                    "physical_dispatch_count": sum(
                        stage["physical_dispatches"] for stage in stages
                    ),
                    "native_action_count": recordable_actions + loose_actions,
                    "recordable_native_action_count": recordable_actions,
                    "opaque_native_action_count": opaque_actions,
                    "loose_native_action_count": loose_actions,
                    # An opaque native action can contain one or more backend
                    # helpers.  Until that provider publishes a command plan,
                    # reporting the action count as an exact helper count
                    # would be misleading.
                    "loose_helper_count": (
                        loose_helper_count
                        if loose_actions and loose_helper_count_exact
                        else None
                    ),
                    "loose_helper_count_exact": bool(
                        loose_actions and loose_helper_count_exact
                    ),
                    "host_observation_boundary_count": int(
                        self._spec.observation_count
                    ),
                    "backend_recording_complete": loose_actions == 0,
                    "fragmented_native_plan": loose_actions > 0,
                    "backend_graph_launches": None,
                    "backend_graph_launches_exact": False,
                    "physical_queue_submissions": None,
                    "physical_queue_submission_source": ("SubmissionTicket.telemetry"),
                    "rejection_reasons": tuple(sorted(set(rejection_reasons))),
                    "workspace_topology": MappingProxyType(
                        {
                            "fixed_internal_storage_bytes": int(
                                self._spec.internal_storage_bytes
                            ),
                            "temporary_declared_bytes": int(
                                temporary_plan.declared_bytes
                            ),
                            "temporary_peak_bytes": int(
                                temporary_plan.planned_peak_bytes
                            ),
                            "temporary_slot_count": int(temporary_plan.slot_count),
                        }
                    ),
                    "dynamic_publication_count": len(publication_plan),
                    "dynamic_publication_reuse_count": sum(
                        max(0, publication["consumer_count"] - 1)
                        for publication in publication_plan
                    ),
                    "dynamic_publications": publication_plan,
                    "stages": tuple(stages),
                }
            )

    def run(self, args, *, trace=False):
        """Run synchronously, optionally returning every control invocation.

        ``trace=False`` preserves the allocation-free default for dynamic path
        snapshots. ``trace=True`` returns an immutable
        :class:`GraphControlFlowTrace`; reports in
        :meth:`control_flow_stats` remain one last invocation per static
        definition. On CUDA/Vulkan, a production depth-2 structured Graph uses
        the same single-submission native lowering as :meth:`submit` and waits
        only at the terminal completion boundary. Flat structured regions
        already use their native synchronous lowering and retain their existing
        report contract. The depth-2 fast path intentionally does not collect
        host control-flow reports; request ``trace=True`` or explicit submission
        telemetry when diagnostic observations are required.
        """
        if not isinstance(trace, bool):
            raise TaichiRuntimeError("Graph.run() trace must be a bool")
        if (
            not trace
            and self._contains_structured_control_value
            and self._spec.max_structured_depth > 1
            and self._spec.supports_native_structured_submission
            and impl.current_cfg().arch in (_ti_core.Arch.cuda, _ti_core.Arch.vulkan)
        ):
            # Production depth-2 execution is a single backend transaction.
            # Reusing submit's qualified native path avoids the portable
            # per-iteration host observations without enabling telemetry or
            # mutating replay diagnostics. The returned ticket is an internal
            # completion owner only; Graph.run() waits exactly once.
            self.submit(args).wait()
            return None
        trace_recorder = _ControlFlowTraceRecorder() if trace else None
        # A graph invocation is one host-side transaction, including mixed
        # CGraph/native sequences. The lock is per Graph and does not wait for
        # GPU completion, so independent graphs remain independently submitable.
        with self._lifecycle_lock:
            self._check_runtime_valid()
            runtime_args, binding_version = self._snapshot_binding_source(args)
            runtime = impl.pytaichi
            submission_state = runtime._active_graph_submissions
            if submission_state < 0:
                raise TaichiRuntimeError(
                    "Graph.run() is primal-only and cannot start while another "
                    "Python thread is entering ti.ad.Tape() or ti.ad.FwdMode(). "
                    "Wait for automatic AD setup to finish."
                )
            if runtime.target_tape is not None:
                raise TaichiRuntimeError(
                    "Graph.run() is primal-only and cannot execute inside an "
                    "active ti.ad.Tape(). Graph dispatches are opaque to Tape "
                    "and would omit gradients. Run the Graph outside automatic "
                    "AD, or manually run an explicit grad-kernel Graph outside "
                    "the Tape context."
                )
            if runtime.fwd_mode_manager is not None:
                raise TaichiRuntimeError(
                    "Graph.run() is primal-only and cannot execute inside an "
                    "active ti.ad.FwdMode(). Graph dispatches are opaque to "
                    "forward AD and would omit dual propagation. Run the Graph "
                    "outside automatic AD."
                )
            if self._qualified_fusion_selector is None:
                execution_instance = self._instance
                execution_run = self._run_impl
            else:
                execution_instance = self._qualified_execution_instance(runtime_args)
                execution_run = execution_instance.run_impl
            # Runtime AD state is process-global rather than thread-local. The
            # signed state closes the window where a native call releases the
            # GIL and another Python thread enters Tape/FwdMode. Publish the
            # increment before the first native call: otherwise two independent
            # Graphs can both snapshot zero, overwrite the count with one, and
            # drive it negative when their paired finally blocks run.
            internal_storage_lease = (
                execution_instance.acquire_exclusive_internal_storage()
            )
            temporary_lease = execution_instance.acquire_temporary_lease()
            observation_lease = None
            try:
                observation_lease = execution_instance.acquire_observation_lease()
                temporary_bindings = (
                    temporary_lease.bindings if temporary_lease is not None else None
                )
                runtime._active_graph_submissions = submission_state + 1
                try:
                    execution_instance.bind_temporary_buffers(temporary_bindings)
                    execution_instance.bind_observation_buffers(
                        observation_lease.bindings
                        if observation_lease is not None
                        else None
                    )
                    prepared = self._spec.prepare_invocation(
                        runtime_args,
                        temporary_bindings,
                        execution_instance._fixed_runtime_args,
                        entrypoint="Graph.run",
                        binding_version=binding_version,
                    )
                    runtime.prog._record_runtime_graph_submission()
                    if trace_recorder is None:
                        execution_run(prepared)
                    else:
                        execution_instance.run_traced(prepared, trace_recorder)
                    if self._has_native_execution_observers:
                        self._spec.record_synchronous_native_execution()
                    self._latest_control_flow_was_async = False
                    if observation_lease is not None:
                        self._last_observations = observation_lease.materialize()
                    self._latest_instance = execution_instance
                finally:
                    execution_instance.clear_observation_buffers()
                    execution_instance.clear_temporary_buffers()
                    runtime._active_graph_submissions -= 1
            finally:
                if observation_lease is not None:
                    observation_lease.cancel()
                if temporary_lease is not None:
                    temporary_lease.cancel()
                if internal_storage_lease is not None:
                    internal_storage_lease.cancel()
        if trace_recorder is not None:
            return trace_recorder.finish()
        return None

    def submit(
        self,
        args,
        *,
        pacer=None,
        lane=None,
        on_saturation="wait",
        telemetry=False,
        workspace_lane=None,
    ):
        """Submit one Graph invocation and return a ``SubmissionTicket``.

        Submission is asynchronous on CUDA/Vulkan when backend work remains;
        CPU tickets are already complete. The runtime argument, lifecycle,
        concurrency, and automatic-differentiation rules are identical to
        ``run()``. A shared ``SubmissionPacer`` can bound backend backlog and
        fairly arbitrate complete host submissions before they enqueue work.
        ``telemetry=True`` and ``telemetry="timestamps"`` add ticket-owned GPU
        timing. ``telemetry="summary"`` retains structured device snapshots,
        queue/submission accounting, and pipeline metadata without inserting
        backend timestamp markers. All results are exposed through
        ``SubmissionTicket.telemetry()``.
        Graphs compiled with multiple workspace lanes select a ready lane
        automatically. ``workspace_lane`` pins a submission to one lane.
        """
        telemetry = _normalize_submission_telemetry_mode(telemetry)
        telemetry_enabled = telemetry is not False
        timestamp_telemetry = telemetry == "timestamps"
        if (
            self._contains_structured_control_value
            and not self._spec.supports_native_structured_submission
        ):
            raise TaichiRuntimeError(
                "Graph.submit() supports structured control only when every "
                "region has a submission-capable backend lowering"
            )
        runtime = impl.pytaichi
        # Admission may wait. Runtime arguments are intentionally snapshotted
        # and validated only after that wait: the post-admission transaction is
        # the safety boundary for reset, rebind, and alias changes. A cheap
        # pacer generation check still wakes/rejects reset waiters.
        admission = _reserve_paced_submission(
            pacer,
            runtime,
            self._submission_lane,
            lane=lane,
            on_saturation=on_saturation,
        )
        temporary_lease = None
        internal_storage_lease = None
        observation_lease = None
        observation_state = None
        telemetry_lease = None
        telemetry_state = None
        submission_owners = ()
        completion_observations = ()
        runtime_owner_retained = False
        submission_instance = self._instance
        workspace_lane_index = 0
        try:
            with self._lifecycle_lock:
                self._check_runtime_valid()
                if runtime is not impl.pytaichi:
                    raise TaichiRuntimeError(
                        "This graph was compiled before ti.reset() or a "
                        "runtime reinitialization. Please rebuild the graph "
                        "after ti.init()."
                    )
                runtime_args, binding_version = self._snapshot_binding_source(args)
                submission_state = runtime._active_graph_submissions
                if submission_state < 0:
                    raise TaichiRuntimeError(
                        "Graph submission cannot start while another Python "
                        "thread is entering ti.ad.Tape() or ti.ad.FwdMode()."
                    )
                if runtime.target_tape is not None:
                    raise TaichiRuntimeError(
                        "Graph submission is primal-only and cannot execute "
                        "inside an active ti.ad.Tape()."
                    )
                if runtime.fwd_mode_manager is not None:
                    raise TaichiRuntimeError(
                        "Graph submission is primal-only and cannot execute "
                        "inside an active ti.ad.FwdMode()."
                    )

                # Publish the AD exclusion count before transaction creation:
                # the pybind call may release the GIL while waiting for another
                # runtime submission reader/writer boundary.
                qualified_instance = (
                    self._instance
                    if self._qualified_fusion_selector is None
                    else self._qualified_execution_instance(runtime_args)
                )
                if qualified_instance is self._instance:
                    workspace_lane_index, submission_instance = (
                        self._workspace_pool.acquire(workspace_lane)
                    )
                else:
                    # Qualified materialization excludes Graph-owned mutable
                    # storage, so the ordinary lane pool would also return its
                    # primary instance. Preserve that exact lane contract.
                    if workspace_lane not in (None, 0):
                        raise TaichiRuntimeError(
                            "Qualified fusion variants require workspace_lane 0"
                        )
                    workspace_lane_index = 0
                    submission_instance = qualified_instance
                internal_storage_lease = (
                    submission_instance.acquire_exclusive_internal_storage()
                )
                temporary_lease = submission_instance.acquire_temporary_lease()
                observation_lease = submission_instance.acquire_observation_lease()
                telemetry_lease = (
                    submission_instance.acquire_structured_telemetry_lease(telemetry)
                    if telemetry_enabled
                    else None
                )
                temporary_bindings = (
                    temporary_lease.bindings if temporary_lease is not None else None
                )
                runtime._active_graph_submissions = submission_state + 1
                try:
                    submission_instance.bind_temporary_buffers(temporary_bindings)
                    submission_instance.bind_observation_buffers(
                        observation_lease.bindings
                        if observation_lease is not None
                        else None
                    )
                    queue_before = (
                        _queue_submission_snapshot() if telemetry_enabled else None
                    )
                    host_submit_start_ns = time.perf_counter_ns()
                    transaction = runtime.prog._begin_runtime_submission_transaction(
                        timestamp_telemetry
                    )
                    runtime.prog._record_runtime_graph_submission()
                    telemetry_recorder = (
                        telemetry_lease.recorder
                        if telemetry_lease is not None
                        else None
                    )
                    prepared = self._spec.prepare_invocation(
                        runtime_args,
                        temporary_bindings,
                        submission_instance._fixed_runtime_args,
                        entrypoint="Graph.submit",
                        binding_version=binding_version,
                    )
                    if self._contains_structured_control_value:
                        if telemetry_recorder is not None:
                            if timestamp_telemetry:
                                telemetry_recorder.attach_gpu_timing(transaction)
                        try:
                            submission_instance.run_for_submission(
                                prepared,
                                telemetry_recorder,
                            )
                        finally:
                            if telemetry_recorder is not None and timestamp_telemetry:
                                telemetry_recorder.detach_gpu_timing()
                        self._latest_control_flow_was_async = True
                    else:
                        submission_instance.run(prepared)
                    submission_owners = prepared.submission_owners
                    # CGraph/kernel paths publish work themselves. Native plans
                    # use Program methods outside that launch path, so publish
                    # once for the whole native portion without changing run().
                    if self._contains_native_nodes_value:
                        transaction._mark_submission()
                    if telemetry_recorder is not None:
                        telemetry_recorder.capture_bounded(
                            prepared.arguments, public_args=runtime_args
                        )
                    if observation_lease is not None:
                        observation_lease.enqueue_tail_readback()
                    completion = transaction._finish()
                    submission_statistics = (
                        transaction._submission_statistics()
                        if telemetry_enabled
                        else None
                    )
                    if telemetry == "summary":
                        # The public Graph invocation is exact without backend
                        # instrumentation. Per-kind backend counters require
                        # the timestamp transaction's scoped native recorder;
                        # keep their zero placeholders explicitly inexact
                        # instead of sampling process-global counters that may
                        # include concurrent Graphs.
                        submission_statistics = dict(submission_statistics)
                        submission_statistics["graph_submissions"] = 1
                        submission_statistics["_exact"] = False
                    host_submit_ns = time.perf_counter_ns() - host_submit_start_ns
                    queue_after = (
                        _queue_submission_snapshot() if telemetry_enabled else None
                    )
                    if temporary_lease is not None:
                        temporary_lease.attach(completion)
                        temporary_lease = None
                    if observation_lease is not None:
                        observation_state = observation_lease.attach(completion)
                        observation_lease = None
                    if telemetry_lease is not None:
                        telemetry_state = telemetry_lease.attach(
                            completion,
                            _queue_submission_delta(queue_before, queue_after),
                            host_submit_ns,
                            submission_statistics,
                        )
                        telemetry_lease = None
                    if internal_storage_lease is not None:
                        internal_storage_lease.attach(completion)
                        internal_storage_lease = None
                    self._latest_instance = submission_instance
                finally:
                    submission_instance.clear_observation_buffers()
                    submission_instance.clear_temporary_buffers()
                    runtime._active_graph_submissions -= 1

                if (
                    self._contains_native_nodes_value or submission_owners
                ) and completion.has_backend_work:
                    runtime.retain_runtime_submission_owner(
                        completion, (self, *submission_owners)
                    )
                    runtime_owner_retained = True
        except BaseException:
            if observation_lease is not None:
                observation_lease.cancel()
            if telemetry_lease is not None:
                telemetry_lease.cancel()
            if temporary_lease is not None:
                temporary_lease.cancel()
            if internal_storage_lease is not None:
                internal_storage_lease.cancel()
            if admission is not None:
                admission._cancel()
            raise
        if admission is not None:
            admission._attach(completion)
        if self._has_native_execution_observers:
            completion_observations = self._spec.begin_native_submission_observations(
                completion
            )
        return SubmissionTicket(
            completion,
            runtime,
            admission,
            observation=observation_state,
            telemetry=telemetry_state,
            workspace_lane=workspace_lane_index,
            submission_owners=submission_owners,
            completion_observations=completion_observations,
            runtime_owner_retained=runtime_owner_retained,
            graph_token=self._terminal_observation_token,
        )

    def submit_batch(self, batch, *, telemetry=False):
        """Submit one certified two-invocation workload as one transaction.

        The serial baseline replays both binding frames on the default stream
        and one private workspace.  A materialized CUDA concurrency recipe
        replays the complete Graph once on each private lane, then joins both
        streams back to the default stream before publishing one completion.
        """

        telemetry = _normalize_submission_telemetry_mode(
            telemetry, "Graph.submit_batch()"
        )
        telemetry_enabled = telemetry is not False
        timestamp_telemetry = telemetry == "timestamps"
        if not isinstance(batch, GraphInvocationBatch) or batch._graph is not self:
            raise TaichiRuntimeError(
                "Graph.submit_batch() requires this Graph's GraphInvocationBatch"
            )
        runtime = impl.pytaichi
        internal_storage_leases = []
        telemetry_lease = None
        telemetry_state = None
        transaction = None
        submission_owners = ()
        runtime_owner_retained = False
        completion = None
        instances = ()
        lane_indices = ()
        try:
            with self._lifecycle_lock:
                self._check_runtime_valid()
                if runtime is not impl.pytaichi or batch.runtime_generation != int(
                    impl.runtime_generation()
                ):
                    raise TaichiRuntimeError(
                        "GraphInvocationBatch was published before ti.reset() "
                        "or a runtime reinitialization"
                    )
                if not _workspace_concurrency_spec_eligible(
                    self._spec, self._execution_arch
                ):
                    raise TaichiRuntimeError(
                        "Graph.submit_batch() requires an eligible CUDA JIT Graph"
                    )
                submission_state = runtime._active_graph_submissions
                if submission_state < 0:
                    raise TaichiRuntimeError(
                        "Graph batch submission cannot start while another Python "
                        "thread is entering automatic differentiation"
                    )
                if (
                    runtime.target_tape is not None
                    or runtime.fwd_mode_manager is not None
                ):
                    raise TaichiRuntimeError(
                        "Graph batch submission is primal-only and cannot execute "
                        "inside automatic differentiation"
                    )
                if self._cuda_concurrent_workspace_pair:
                    instances = self._workspace_pool.materialize_pair()
                    lane_indices = (0, 1)
                else:
                    instances = (self._instance, self._instance)
                    lane_indices = (0, 0)
                for instance in dict.fromkeys(instances):
                    lease = instance.acquire_exclusive_internal_storage()
                    if lease is not None:
                        internal_storage_leases.append(lease)
                telemetry_lease = (
                    instances[0].acquire_structured_telemetry_lease(telemetry)
                    if telemetry_enabled
                    else None
                )
                runtime._active_graph_submissions = submission_state + 1
                try:
                    queue_before = (
                        _queue_submission_snapshot() if telemetry_enabled else None
                    )
                    host_submit_start_ns = time.perf_counter_ns()
                    transaction = runtime.prog._begin_runtime_submission_transaction(
                        timestamp_telemetry,
                        self._cuda_concurrent_workspace_pair,
                    )
                    owners = []
                    for instance, version in zip(instances, batch.versions):
                        runtime.prog._record_runtime_graph_submission()
                        prepared = self._spec.prepare_invocation(
                            version.arguments,
                            None,
                            None,
                            entrypoint="Graph.submit_batch",
                            binding_version=version,
                        )
                        if self._cuda_concurrent_workspace_pair:
                            instance.run_cuda_concurrent_batch(prepared)
                        else:
                            instance.run(prepared)
                        owners.extend(prepared.submission_owners)
                    completion = transaction._finish()
                    submission_statistics = (
                        transaction._submission_statistics()
                        if telemetry_enabled
                        else None
                    )
                    transaction = None
                    if telemetry == "summary":
                        submission_statistics = dict(submission_statistics)
                        submission_statistics["graph_submissions"] = 2
                        submission_statistics["_exact"] = bool(
                            self._cuda_concurrent_workspace_pair
                        )
                    host_submit_ns = time.perf_counter_ns() - host_submit_start_ns
                    queue_after = (
                        _queue_submission_snapshot() if telemetry_enabled else None
                    )
                    if telemetry_lease is not None:
                        telemetry_state = telemetry_lease.attach(
                            completion,
                            _queue_submission_delta(queue_before, queue_after),
                            host_submit_ns,
                            submission_statistics,
                        )
                        telemetry_lease = None
                    for lease in internal_storage_leases:
                        lease.attach(completion)
                    internal_storage_leases.clear()
                    submission_owners = tuple(owners)
                    self._latest_instance = instances[-1]
                finally:
                    runtime._active_graph_submissions -= 1

                if submission_owners and completion.has_backend_work:
                    runtime.retain_runtime_submission_owner(
                        completion, (self, *submission_owners)
                    )
                    runtime_owner_retained = True
        except BaseException:
            if transaction is not None:
                del transaction
            if telemetry_lease is not None:
                telemetry_lease.cancel()
            for lease in internal_storage_leases:
                lease.cancel()
            raise
        return SubmissionTicket(
            completion,
            runtime,
            telemetry=telemetry_state,
            workspace_lane=lane_indices[0],
            workspace_lanes=lane_indices,
            submission_owners=submission_owners,
            runtime_owner_retained=runtime_owner_retained,
            graph_token=self._terminal_observation_token,
        )

    def prepare_telemetry(self, mode, *, slots=1):
        """Move opt-in telemetry allocation and JIT setup off the first sample.

        This method never executes the Graph or reads user resources. It
        prepares ``slots`` telemetry records for every currently materialized
        workspace lane. Timestamp mode also performs one empty instrumented
        runtime transaction so backend event/query initialization occurs at
        this explicit boundary rather than inside the first measured submit.
        """

        mode = _normalize_submission_telemetry_mode(mode, "Graph.prepare_telemetry()")
        if mode is False:
            return self
        with self._lifecycle_lock:
            self._check_runtime_valid()
            for instance in self._workspace_pool.instances:
                instance.prepare_structured_telemetry(slots)

            if (
                mode == "timestamps"
                and "timestamps" not in self._prepared_telemetry_modes
            ):
                transaction = (
                    impl.get_runtime().prog._begin_runtime_submission_transaction(True)
                )
                for node in self._instance._structured_telemetry_nodes:
                    transaction._begin_gpu_region_timing(node.region_path)
                    transaction._end_gpu_region_timing(node.region_path)
                completion = transaction._finish()
                completion.wait()
                self._prepared_telemetry_modes.add("timestamps")
            self._prepared_telemetry_modes.add("summary")
        return self

    def _workspace_pool_for_current_runtime(self):
        key = self._spec.instance_key()
        pool = self._instances.get(key)
        if pool is None:
            pool = _GraphWorkspaceLanePool(
                self._spec,
                key,
                self._workspace_lane_capacity,
                self._workspace_saturation,
            )
            self._instances[key] = pool
        return pool

    def _prewarm(self):
        with self._lifecycle_lock:
            self._check_runtime_valid()
            self._instance.prewarm()
        return self

    def _check_runtime_valid(self):
        if not self._runtime_valid:
            raise TaichiRuntimeError(
                "This graph was compiled before ti.reset() or a runtime "
                "reinitialization. Please rebuild the graph after ti.init()."
            )
        self._spec.validate_lifetime_leases()
        if self._stale_snode_tree_dependencies:
            dependencies = ", ".join(
                f"id={tree_id} generation={generation}"
                for tree_id, generation in sorted(self._stale_snode_tree_dependencies)
            )
            raise TaichiRuntimeError(
                "This graph references a destroyed SNodeTree "
                f"({dependencies}); rebuild the Graph."
            )

    def _retire_snode_tree(self, dependency):
        dependency = tuple(dependency)
        with self._lifecycle_lock:
            if (
                self._spec is None
                or dependency not in self._spec.snode_tree_dependencies
            ):
                return False
            if dependency in self._stale_snode_tree_dependencies:
                return True
            self._stale_snode_tree_dependencies.add(dependency)
            for pool in self._instances.values():
                pool.invalidate_runtime(preserve_executables=True)
            if self._qualified_fusion_selector is not None:
                self._qualified_fusion_selector.invalidate_runtime(
                    preserve_executables=True
                )
            self._spec.invalidate_runtime(preserve_executables=True)
            return True

    def _cancel_snode_tree_retirement(self, dependency):
        with self._lifecycle_lock:
            self._stale_snode_tree_dependencies.discard(tuple(dependency))

    def _invalidate_runtime(self):
        with self._lifecycle_lock:
            self._runtime_valid = False
            self._run_impl = None
            for pool in self._instances.values():
                pool.invalidate_runtime()
            if self._qualified_fusion_selector is not None:
                self._qualified_fusion_selector.invalidate_runtime()
            if self._spec is not None:
                self._spec.invalidate_runtime()
            self._instance = None
            self._latest_instance = None
            self._instances.clear()
            self._qualified_fusion_selector = None
            # Definition nodes currently own mixed-graph JIT caches and native
            # executables. Release them before Program/backend teardown so
            # backend allocation leases cannot outlive their Device registry.
            self._spec = None
            self._workspace_pool = None

    @property
    def _debug_info(self):
        with self._lifecycle_lock:
            self._check_runtime_valid()
            return self._spec.debug_info

    @property
    def _ir_debug_info(self):
        with self._lifecycle_lock:
            self._check_runtime_valid()
            return self._spec.ir_debug_info

    @property
    def _executable_optimization_space(self):
        with self._lifecycle_lock:
            self._check_runtime_valid()
            return self._spec.executable_optimization_space

    @property
    def _qualified_fusion_stats(self):
        with self._lifecycle_lock:
            self._check_runtime_valid()
            if self._qualified_fusion_selector is None:
                return MappingProxyType(
                    {
                        "configured": False,
                        "attempts": 0,
                        "qualified_selections": 0,
                        "baseline_fallbacks": 0,
                        "materializations": 0,
                        "retained_variants": 0,
                        "last_reason": "not_configured",
                        "last_entry_id": None,
                    }
                )
            return self._qualified_fusion_selector.stats

    @property
    def _instance_debug_info(self):
        with self._lifecycle_lock:
            self._check_runtime_valid()
            return self._latest_instance.debug_info

    @property
    def _graph_stats(self):
        with self._lifecycle_lock:
            self._check_runtime_valid()
            return self._latest_instance.debug_graph_stats

    def control_flow_stats(self):
        """Return the last invocation of each definition in the latest run.

        Repeated nested definitions overwrite their prior report. Use
        ``Graph.run(..., trace=True)`` to retain every invocation. Production
        depth-2 native structured execution deliberately does not capture
        these diagnostics.
        """
        with self._lifecycle_lock:
            self._check_runtime_valid()
            if self._latest_control_flow_was_async:
                raise TaichiRuntimeError(
                    "Control-flow reports are unavailable after asynchronous "
                    "submission or production native structured execution. "
                    "Use Graph.run(..., trace=True) for a diagnostic trace, "
                    "or submit with explicit telemetry for ticket-owned "
                    "terminal observations."
                )
            control_nodes = self._spec.structured_control_nodes
            for node in control_nodes:
                materialize = getattr(node, "materialize_pending_report", None)
                if materialize is not None:
                    materialize()
            return tuple(
                node.last_report
                for node in control_nodes
                if node.last_report is not None
            )

    def latest_observations(self):
        """Return the most recent synchronous ``run()`` snapshot."""
        with self._lifecycle_lock:
            self._check_runtime_valid()
            return _copy_observation_result(self._last_observations)

    def execution_stats(self):
        """Return an immutable execution-path and static-Field report.

        Calling this method never changes subsequent Graph execution. In
        particular, production replay remains free of timing attribution and
        per-run diagnostic counters. Use explicit submission telemetry when a
        measured diagnostic execution is required.
        """
        with self._lifecycle_lock:
            if not self._runtime_valid:
                lifecycle_state = "runtime_invalid"
                instance_kind = "unavailable"
                backend_stats = ()
            elif self._stale_snode_tree_dependencies:
                lifecycle_state = "stale_field_dependency"
                instance_kind = self._latest_instance.debug_info["kind"]
                backend_stats = ()
            else:
                lifecycle_state = "ready"
                instance_kind = self._latest_instance.debug_info["kind"]
                backend_stats = self._latest_instance.snapshot_graph_stats
            temporary_arena_stats = (
                self._workspace_pool.temporary_arena_stats
                if lifecycle_state == "ready"
                else {}
            )
            observation_arena_stats = (
                self._workspace_pool.observation_arena_stats
                if lifecycle_state == "ready"
                else {}
            )
            telemetry_arena_stats = (
                self._workspace_pool.structured_telemetry_arena_stats
                if lifecycle_state == "ready"
                else {}
            )
            internal_storage_stats = (
                self._workspace_pool.internal_storage_stats
                if lifecycle_state == "ready"
                else {}
            )
            observation_staging_bytes = 0
            if lifecycle_state == "ready" and (
                self._contains_structured_control_value
                or self._contains_observations_value
            ):
                observation_staging_bytes = int(
                    impl.get_runtime().prog._graph_observation_staging_stats()[
                        "persistent_bytes"
                    ]
                )
            provider_memory = (
                self._spec.provider_memory_reports() if self._spec is not None else ()
            )
            from taichi_forge.graph._recipes.runtime_storage import storage_pool_reports

            storage_pools = storage_pool_reports(self._workspace_pool.instances) if lifecycle_state == "ready" else ()
            return _execution_report(
                self._execution_definition,
                self._execution_arch,
                lifecycle_state,
                instance_kind,
                backend_stats,
                observation_staging_bytes=observation_staging_bytes,
                temporary_memory_plan=self._execution_definition[
                    "temporary_memory_plan"
                ],
                temporary_arena_stats=temporary_arena_stats,
                observation_arena_stats=observation_arena_stats,
                telemetry_arena_stats=telemetry_arena_stats,
                internal_storage_stats=internal_storage_stats,
                provider_memory=provider_memory,
                storage_pools=storage_pools,
            )

    @property
    def _compiled_graph(self):
        with self._lifecycle_lock:
            self._check_runtime_valid()
            return self._spec.compiled_graph()

    @property
    def _contains_native_nodes(self):
        return self._contains_native_nodes_value


def _deprecate_arg_args(kwargs: Dict[str, Any]):
    if "field_dim" in kwargs:
        warnings.warn(
            "The field_dim argument for ndarray will be deprecated in v1.6.0, use ndim instead.",
            DeprecationWarning,
        )
        if "ndim" in kwargs:
            raise TaichiRuntimeError(
                "field_dim is deprecated, please do not specify field_dim and ndim at the same time."
            )
        kwargs["ndim"] = kwargs["field_dim"]
        del kwargs["field_dim"]
    tag = kwargs["tag"]

    if tag == ArgKind.SCALAR:
        if "element_shape" in kwargs:
            raise TaichiRuntimeError(
                "The element_shape argument for scalar is deprecated in v1.6.0, and is removed in v1.7.0. "
                "Please remove them."
            )

    if tag == ArgKind.NDARRAY:
        if "element_shape" in kwargs:
            raise TaichiRuntimeError(
                "The element_shape argument for ndarray is deprecated in v1.6.0, and it is removed in v1.7.0. "
                "Please use vector or matrix data type instead."
            )

    if tag == ArgKind.RWTEXTURE or tag == ArgKind.TEXTURE:
        if "dtype" in kwargs:
            warnings.warn(
                "The dtype argument for texture will be deprecated in v1.6.0, use format instead.",
                DeprecationWarning,
            )
            del kwargs["dtype"]

        if "shape" in kwargs:
            raise TaichiRuntimeError(
                "The shape argument for texture is deprecated in v1.6.0, and it is removed in v1.7.0. "
                "Please use ndim instead. (Note that you no longer need the exact texture size.)"
            )

        if "channel_format" in kwargs or "num_channels" in kwargs:
            if "fmt" in kwargs:
                raise TaichiRuntimeError(
                    "channel_format and num_channels are deprecated, please do not specify channel_format/num_channels and fmt at the same time."
                )
            if tag == ArgKind.RWTEXTURE:
                fmt = TY_CH2FORMAT[(kwargs["channel_format"], kwargs["num_channels"])]
                kwargs["fmt"] = fmt
                raise TaichiRuntimeError(
                    "The channel_format and num_channels arguments for texture are deprecated in v1.6.0, "
                    "and they are removed in v1.7.0. Please use fmt instead."
                )
            else:
                raise TaichiRuntimeError(
                    "The channel_format and num_channels arguments are no longer required for non-RW textures "
                    "since v1.6.0, and they are removed in v1.7.0. Please remove them."
                )


def _check_args(kwargs: Dict[str, Any], allowed_kwargs: List[str]):
    for k, v in kwargs.items():
        if k not in allowed_kwargs:
            raise TaichiRuntimeError(
                f"Invalid argument: {k}, you can only create a graph argument with: {allowed_kwargs}"
            )
        if k == "tag":
            if not isinstance(v, ArgKind):
                raise TaichiRuntimeError(
                    f"tag must be a ArgKind variant, but found {type(v)}."
                )
        if k == "name":
            if not isinstance(v, str):
                raise TaichiRuntimeError(f"name must be a string, but found {type(v)}.")


def _make_arg_scalar(kwargs: Dict[str, Any]):
    allowed_kwargs = [
        "tag",
        "name",
        "dtype",
    ]
    _check_args(kwargs, allowed_kwargs)
    name = kwargs["name"]
    dtype = kwargs["dtype"]
    descriptor = describe_element_type(dtype)
    if descriptor.category != "scalar":
        raise TaichiRuntimeError(
            f"Tag ArgKind.SCALAR must specify a scalar type, but found {type(dtype)}."
        )
    return _ti_core.Arg(ArgKind.SCALAR, name, descriptor.logical_type, 0, [])


def _make_arg_ndarray(kwargs: Dict[str, Any]):
    allowed_kwargs = [
        "tag",
        "name",
        "dtype",
        "ndim",
    ]
    _check_args(kwargs, allowed_kwargs)
    name = kwargs["name"]
    ndim = kwargs["ndim"]
    dtype = kwargs["dtype"]
    descriptor = describe_element_type(dtype)
    if not descriptor.is_complete:
        raise TaichiRuntimeError(
            f"Tag ArgKind.NDARRAY requires a concrete scalar, vector, matrix, "
            f"or struct element type, but found {dtype}."
        )
    if descriptor.category == "tensor" and len(descriptor.shape) not in (1, 2):
        raise TaichiRuntimeError(
            "Graph ndarray tensor elements support vector (rank 1) and "
            f"matrix (rank 2) types only, but got shape {descriptor.shape}."
        )
    if descriptor.category == "struct":
        raise TaichiRuntimeError(
            "Graph StructNdarray arguments are not supported by the current "
            "serialized Graph schema."
        )
    return _ti_core.Arg(ArgKind.NDARRAY, name, descriptor.logical_type, ndim, [])


def _make_arg_matrix(kwargs: Dict[str, Any]):
    allowed_kwargs = [
        "tag",
        "name",
        "dtype",
    ]
    _check_args(kwargs, allowed_kwargs)
    name = kwargs["name"]
    dtype = kwargs["dtype"]
    descriptor = describe_element_type(dtype)
    if not isinstance(dtype, MatrixType) or descriptor.category != "tensor":
        raise TaichiRuntimeError(
            f"Tag ArgKind.MATRIX must specify matrix type, but got {dtype}."
        )
    return _ti_core.Arg(ArgKind.MATRIX, f"{name}", descriptor.logical_type, 0, [])


def _make_arg_texture(kwargs: Dict[str, Any]):
    allowed_kwargs = [
        "tag",
        "name",
        "ndim",
    ]
    _check_args(kwargs, allowed_kwargs)
    name = kwargs["name"]
    ndim = kwargs["ndim"]
    return _ti_core.Arg(ArgKind.TEXTURE, name, impl.f32, 4, [2] * ndim)


def _make_arg_rwtexture(kwargs: Dict[str, Any]):
    allowed_kwargs = [
        "tag",
        "name",
        "ndim",
        "fmt",
    ]
    _check_args(kwargs, allowed_kwargs)
    name = kwargs["name"]
    ndim = kwargs["ndim"]
    fmt = kwargs["fmt"]
    if fmt == enums.Format.unknown:
        raise TaichiRuntimeError(
            f"Tag ArgKind.RWTEXTURE must specify a valid color format, but found {fmt}."
        )
    channel_format, num_channels = FORMAT2TY_CH[fmt]
    return _ti_core.Arg(
        ArgKind.RWTEXTURE, name, channel_format, num_channels, [2] * ndim
    )


def _make_arg(kwargs: Dict[str, Any]):
    assert "tag" in kwargs
    _deprecate_arg_args(kwargs)
    proc = {
        ArgKind.SCALAR: _make_arg_scalar,
        ArgKind.NDARRAY: _make_arg_ndarray,
        ArgKind.MATRIX: _make_arg_matrix,
        ArgKind.TEXTURE: _make_arg_texture,
        ArgKind.RWTEXTURE: _make_arg_rwtexture,
    }
    tag = kwargs["tag"]
    return proc[tag](kwargs)


def _kwarg_rewriter(args, kwargs):
    for i, arg in enumerate(args):
        rewrite_map = {
            0: "tag",
            1: "name",
            2: "dtype",
            3: "ndim",
            4: "field_dim",
            5: "element_shape",
            6: "channel_format",
            7: "shape",
            8: "num_channels",
        }
        if i in rewrite_map:
            kwargs[rewrite_map[i]] = arg
        else:
            raise TaichiRuntimeError(f"Unexpected {i}th positional argument")


def Arg(*args, **kwargs):
    _kwarg_rewriter(args, kwargs)
    return _make_arg(kwargs)


__all__ = [
    "GraphBuilder",
    "Graph",
    "GraphDefinition",
    "GraphBindingSet",
    "GraphInvocationBatch",
    "SubmissionTicket",
    "SubmissionPacer",
    "NativeActionManifest",
    "VulkanBufferCommand",
    "VulkanBufferCommandRecording",
    "GraphExecutionCounters",
    "GraphReplayAttribution",
    "GraphExecutionSegmentReport",
    "GraphExecutionReport",
    "BoundedDispatchCapabilities",
    "GraphOwnedNdarray",
    "BoundedDispatchHandle",
    "HostBoundedDispatchHandle",
    "BoundedDispatchSnapshot",
    "OrderedSegmentDispatchSnapshot",
    "OrderedSegmentedDispatchSnapshot",
    "GraphWhileReport",
    "GraphBranchReport",
    "GraphControlFlowInvocation",
    "GraphControlFlowTrace",
    "GraphSubmissionRegionTelemetry",
    "GraphSubmissionQueueTelemetry",
    "GraphSubmissionExecutionTelemetry",
    "GraphPipelineBoundedDispatchReport",
    "GraphPipelineStageReport",
    "GraphPipelineReport",
    "GraphSubmissionTelemetry",
    "bounded_dispatch_capabilities",
    "dynamic_work_capabilities",
    "segmented_dispatch_begin",
    "segmented_dispatch_end",
    "segmented_dispatch_index",
    "segmented_dispatch_count",
    "structured_control_capabilities",
    "Arg",
    "ArgKind",
]
