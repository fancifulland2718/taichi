"""Prepared consecutive unique: a head kernel and shared-prefix compaction."""

import hashlib
import json

from taichi_forge._kernels import rle_mark_boundaries_ndarray
from taichi_forge._lib import core as _ti_core
from taichi_forge.algorithms._prepared_compact import PreparedCompactPlan
from taichi_forge.graph._ir import GraphAccess, ResourceEffect
from taichi_forge.graph._native import DispatchGraphAction, NativeGraphExecutable, NativeGraphNode
from taichi_forge.lang import impl
from taichi_forge.lang._ndarray import ScalarNdarray
from taichi_forge.lang._storage_view import describe_storage, ndarray_view
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.types.primitive_types import i32, i64, u32, u64


class _UniqueHeadExecutable(NativeGraphExecutable):
    def __init__(self, keys, flags, size, capacity, namespace):
        from taichi_forge.graph._graph import Arg, ArgKind, _GraphRunContext

        key_view = ndarray_view(keys)
        args = (
            Arg(ArgKind.NDARRAY, namespace + "_keys", key_view.descriptor.scalar_type, ndim=1),
            Arg(ArgKind.NDARRAY, namespace + "_heads", i32, ndim=1),
            Arg(ArgKind.SCALAR, namespace + "_size", i32),
            Arg(ArgKind.SCALAR, namespace + "_capacity", i32),
        )
        self._bindings = dict(zip((arg.name for arg in args), (key_view, flags, size, capacity)))
        # Real fixed bindings provide specialization facts. Symbolic Graph
        # injection would allocate/free dummy ndarrays and can wait on Vulkan.
        kernel = rle_mark_boundaries_ndarray._primal
        specialization = kernel.ensure_compiled(key_view, flags, size, capacity)
        compiled = kernel.compiled_kernels[specialization]
        self._action = DispatchGraphAction(
            ((compiled, args),),
            backends=("cuda", "vulkan"),
            fixed_bindings=self._bindings,
            update_policy="immutable",
        )
        # Build the ordinary native fallback without executing the kernel. A
        # root Graph may instead lower the same DispatchGraphAction into a
        # retained segment. Both consume the same prepared storage contract.
        builder = _ti_core.GraphBuilder()
        builder.dispatch(compiled, args, "")
        self._compiled = builder.compile()
        context = _GraphRunContext()
        context.begin(self._bindings)
        try:
            self._flattened = dict(context.flattened_args())
        finally:
            context.end()
        self._effects = (
            ResourceEffect(keys, GraphAccess.READ, runtime_bound=False),
            ResourceEffect(flags, GraphAccess.WRITE, runtime_bound=False),
        )

    def run(self):
        self._compiled.run(self._flattened)

    @property
    def recordable_action(self):
        return self._action

    @property
    def resource_effects(self):
        return self._effects

    @property
    def lifetime_leases(self):
        return tuple(self._bindings.values())

    @property
    def debug_info(self):
        return {"kind": "consecutive_unique_heads", "dispatch_count": 1}


class _UniqueExecutable(NativeGraphExecutable):
    def __init__(self, plan):
        self._plan = plan

    def recordable_root_actions(self, namespace):
        plan = self._plan
        plan._validate_lifetime()
        return (
            _UniqueHeadExecutable(plan._keys, plan._flags, plan._size, plan._capacity, namespace),
            plan._compact.record()._as_graph_native_node().compile(),
        )


class _UniqueNode(NativeGraphNode):
    def __init__(self, plan):
        self._plan = plan

    def compile(self):
        self._plan._validate_lifetime()
        return _UniqueExecutable(self._plan)


class PreparedUniquePlan:
    """Keep the first item/payload of each consecutive integer-key run.

    This does not sort. Keys are scalar i32/u32/i64/u64 dense ranges; optional
    payloads follow PreparedCompactPlan's scalar/vector/matrix record contract.
    Output capacity covers the key capacity; only the device-written count
    prefix is defined. ``size`` is a fixed integer in [0, capacity], not a host
    readback. Contents may change in place between runs.

    Preparation compiles a head kernel and allocates private i32 flags without
    running mathematics. One native prefix is shared by key and payload output.
    ``record()`` expands at root Graph compilation into ordinary recordable
    actions, not nested Graph execution. Native compaction remains a recording
    boundary; this is not a fully capture-safe unique operation.
    """

    def __init__(self, keys, output, count, *, size=None, values=None, unique_values=None):
        from taichi_forge.graph._graph import GraphBuilder

        description = describe_storage(keys, access="read")
        descriptor = description.descriptor
        if (
            descriptor is None
            or len(descriptor.index_shape) != 1
            or descriptor.element_shape
            or descriptor.scalar_type not in (i32, u32, i64, u64)
        ):
            raise TaichiRuntimeError("Prepared unique requires 1D scalar i32/u32/i64/u64 keys")
        capacity = int(descriptor.index_shape[0])
        if not 0 < capacity <= (1 << 31) - 1:
            raise ValueError("Prepared unique capacity must be positive and at most INT_MAX")
        if size is None:
            size = capacity
        if isinstance(size, bool) or not isinstance(size, int) or not 0 <= size <= capacity:
            raise ValueError("Prepared unique size must be an integer in [0, capacity]")
        if (values is None) != (unique_values is None):
            raise ValueError("Prepared unique requires both payload input and output")
        self._program = impl.get_runtime().prog
        self._keys, self._capacity, self._size = keys, capacity, size
        self._graph = None
        self._binding = None
        self._compact = None
        # The head kernel writes every flag, including all i >= size. Avoid
        # Vulkan's synchronized zero-fill of scratch that will be overwritten.
        self._flags = ScalarNdarray._private_scratch_storage(i32, (capacity,))
        columns = ((keys, output),) if values is None else ((keys, output), (values, unique_values))
        try:
            self._compact = PreparedCompactPlan._from_columns(columns, self._flags, count)
            builder = GraphBuilder()
            builder.append_native(self.record())
            self._graph = builder.compile()
            self._binding = self._graph.bind({})
        except BaseException:
            self.close()
            raise
        self._description = {
            "kind": "consecutive_unique",
            "backend": self._compact._backend,
            "capacity": capacity,
            "size": size,
            "columns": self._compact._description["columns"],
            "prefix_evaluations": 1,
            "private_flag_bytes": capacity * 4,
            "count_location": "device",
            "device_capture": False,
            "root_expansion": ("head_dispatch", "shared_prefix_compact"),
        }
        canonical = json.dumps(self._description, sort_keys=True, separators=(",", ":"))
        self._physical_id = "prepared-unique-v1:" + hashlib.sha256(canonical.encode()).hexdigest()

    def _validate_lifetime(self):
        if self._compact is None:
            raise TaichiRuntimeError("PreparedUniquePlan is closed")
        self._compact._validate_lifetime()

    def run(self):
        self._validate_lifetime()
        self._graph.run(self._binding)

    def record(self):
        """Return a root-Graph expansion retaining the same fixed resources."""
        self._validate_lifetime()
        return _UniqueNode(self)

    def report(self):
        return {
            **self._description,
            "physical_plan_id": self._physical_id,
            "closed": self._compact is None,
            "workspace_bytes_last_observed": None if self._compact is None else self._compact._workspace_bytes,
            "workspace_owner": "program_arena_and_plan_flags",
        }

    def close(self):
        if self._graph is not None:
            self._graph.close()
        if self._compact is not None:
            self._compact.close()
        self._graph = self._binding = self._compact = self._flags = self._keys = self._program = None

    def __enter__(self):
        self._validate_lifetime()
        return self

    def __exit__(self, *_):
        self.close()


def prepare_unique(values, output, count, *, size=None):
    return PreparedUniquePlan(values, output, count, size=size)


def prepare_unique_by_key(keys, values, unique_keys, unique_values, count, *, size=None):
    return PreparedUniquePlan(keys, unique_keys, count, size=size, values=values, unique_values=unique_values)


__all__ = ["PreparedUniquePlan", "prepare_unique", "prepare_unique_by_key"]
