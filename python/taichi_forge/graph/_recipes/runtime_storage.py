"""Cold, provider-owned storage contributions to a runtime Graph instance."""

from contextlib import ExitStack
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class GraphStoragePoolReport:
    """Explicit cold-boundary measurements, separate from requested storage."""

    allocator: str
    allocation_members: tuple[str, ...]
    allocation_count: int
    requested_bytes: int
    used_current_bytes: int | None
    reserved_current_bytes: int | None
    used_high_bytes: int | None
    reserved_high_bytes: int | None
    release_threshold_bytes: int | None
    closed: bool
    instance_index: int = 0


@dataclass(frozen=True)
class GraphRuntimeStoragePlan:
    """An allocation owner, not a replay hook or an external-pointer escape hatch.

    The factory returns an owner with ``allocate(dtype, shape)`` and ``close()``.
    Allocations must be ordinary Program-registered ScalarNdarrays. Closing the
    factory must preserve storage retained by bindings or in-flight execution.
    """

    plan_id: str
    binding_names: tuple[str, ...]
    temporary_arena: bool
    factory: object = field(repr=False, compare=False)
    temporary_capacity: int | None = None
    ordered_temporary_reuse: bool = False

    def __post_init__(self):
        if not isinstance(self.plan_id, str) or not self.plan_id:
            raise ValueError("Graph storage plan requires a stable identity")
        names = tuple(self.binding_names)
        if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Graph storage bindings must be unique nonempty names")
        if not names and not self.temporary_arena:
            raise ValueError("Graph storage plan must own private bindings or a temporary arena")
        if not callable(self.factory):
            raise TypeError("Graph storage owner factory must be callable")
        if self.temporary_capacity is not None and (
            not self.temporary_arena
            or isinstance(self.temporary_capacity, bool)
            or not isinstance(self.temporary_capacity, int)
            or not 1 <= self.temporary_capacity <= 64
        ):
            raise ValueError("Graph storage temporary capacity requires an arena and a bounded slot count")
        if not isinstance(self.ordered_temporary_reuse, bool) or (
            self.ordered_temporary_reuse
            and (not self.temporary_arena or self.temporary_capacity != 1)
        ):
            raise ValueError(
                "Ordered temporary storage requires one eagerly owned slot"
            )
        object.__setattr__(self, "binding_names", names)


def ordered_temporary_reuse_eligible(spec):
    """Cold proof over the final executor, not a provider's semantic promise.

    A single CGraph's dispatches enqueue on the CUDA runtime stream under the
    submission lock. No host action or unjoined parallel lane may consume this
    generation's scratch. Temporary callbacks are resolved separately at setup.
    """
    from taichi_forge._lib import core
    from taichi_forge.graph._graph import _CompiledCGraphNode
    from taichi_forge.lang import impl

    if (
        impl.current_cfg().arch != core.Arch.cuda
        or len(spec.nodes) != 1
        or getattr(spec, "_binding_executor_factory", None) is not None
        or spec.native_execution_observer_leases
    ):
        return False
    node = spec.nodes[0]
    if (
        not isinstance(node, _CompiledCGraphNode)
        or not node.temporary_actions
        or node.parallel_dispatch_groups
        or len(node.recording_dispatches) != node.dispatch_count
        or spec.snode_tree_dependency_info
        or spec.runtime_lifetime_leases
    ):
        return False
    # Native algorithms must already be fully lowered to ordinary dispatches;
    # a capture command's stream/host behavior is not implied by this proof.
    if any(
        action.backend_command_recording is not None
        for action in node.temporary_actions
    ):
        return False
    if any(
        manifest.execution_kind != "kernel_dispatch"
        for manifest in node.native_action_manifests
    ):
        return False
    if any(
        callable(getattr(lease, hook, None))
        for lease in spec.lifetime_leases
        for hook in (
            "bind_graph_arguments",
            "graph_submission_owners",
            "validate_graph_bindings",
        )
    ):
        return False
    plan = spec.temporary_memory_plan
    return (
        bool(plan.allocations)
        and not plan.conflicting_requirements
        and all(
            allocation.offset == 0 and allocation.alignment <= 16
            for allocation in plan.allocations
        )
    )


def validate_storage_plans(spec, plans):
    """Resolve ownership once, before creating any allocation or executable."""
    from taichi_forge.graph._graph import _GraphInternalNdarraySpec

    by_name = {}
    identities = set()
    arena_owner = None
    aliases = {}
    for name, value in spec.fixed_runtime_args.items():
        if isinstance(value, _GraphInternalNdarraySpec):
            aliases.setdefault(id(value), set()).add(name)
    private_names = set().union(*aliases.values()) if aliases else set()
    for plan in plans:
        if not isinstance(plan, GraphRuntimeStoragePlan):
            raise TypeError("runtime Graph storage contribution must be a GraphRuntimeStoragePlan")
        if plan.plan_id in identities:
            raise ValueError("runtime Graph storage plan is selected twice")
        identities.add(plan.plan_id)
        selected = set(plan.binding_names)
        if not selected <= private_names:
            raise ValueError("Graph storage plans can allocate only declared private bindings")
        if selected.intersection(by_name):
            raise ValueError("Graph private binding has multiple allocation owners")
        if any(selected.intersection(group) and not group <= selected for group in aliases.values()):
            raise ValueError("Graph private binding aliases must share one allocation owner")
        by_name.update((name, plan) for name in selected)
        if plan.temporary_arena:
            if arena_owner is not None:
                raise ValueError("Graph temporary arena has multiple allocation owners")
            if not spec.temporary_memory_plan.allocations:
                raise ValueError("Graph storage plan selected an absent temporary arena")
            if plan.ordered_temporary_reuse and not ordered_temporary_reuse_eligible(
                spec
            ):
                raise ValueError(
                    "Ordered temporary storage requires a fully lowered single-stream CUDA Graph"
                )
            arena_owner = plan


def create_storage_owners(instance, plans):
    """Publish each owner immediately so partial construction can retire it."""
    allocators = {}
    arena_allocator = None
    arena_capacity = None
    ordered_reuse = False
    for plan in plans:
        owner = plan.factory()
        instance._storage_owners += (owner,)
        allocate = owner.allocate
        allocators.update((name, allocate) for name in plan.binding_names)
        if plan.temporary_arena:
            arena_allocator = allocate
            arena_capacity = plan.temporary_capacity
            ordered_reuse = plan.ordered_temporary_reuse
    return allocators, arena_allocator, arena_capacity, ordered_reuse


def storage_pool_reports(instances):
    """Only called by explicit execution_stats, never by replay or acquisition."""
    return tuple(
        replace(observe(), instance_index=index)
        for index, instance in enumerate(instances)
        for owner in instance._storage_owners
        if (observe := getattr(owner, "storage_pool_report", None)) is not None
    )


def retire_storage_owners(instance):
    owners, instance._storage_owners = instance._storage_owners, ()
    # Retire every factory even if a provider reports a cleanup failure. The
    # existing allocation leases, not this callback stack, protect GPU uses.
    with ExitStack() as cleanup:
        for owner in owners:
            cleanup.callback(owner.close)


__all__ = ["GraphRuntimeStoragePlan"]
