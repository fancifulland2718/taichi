"""Generation-owned CUDA storage over complete semantic usage regions."""

from dataclasses import dataclass, replace
from functools import lru_cache, partial
from math import prod

from taichi_forge.graph._recipes.definition import _digest
from taichi_forge.graph._recipes.families import (
    GraphRuntimeFragmentProvider,
    _fragment,
    runtime_family_provider_descriptor,
)
from taichi_forge.graph._recipes.fragments import GraphFragmentResourceRequirement, GraphFragmentTask
from taichi_forge.graph._recipes.runtime_storage import GraphRuntimeStoragePlan, GraphStoragePoolReport


@lru_cache(maxsize=4)
def _pool_available(runtime_generation):
    from taichi_forge._lib import core

    pool = getattr(core, "_CudaGraphMemoryPool", None)
    return pool is not None and pool.available()


def _nodes_by_path(root):
    result = {}

    def visit(node, path):
        result[path] = node
        for index, child in enumerate(node.children):
            visit(child, f"{path}/{index}:{child.kind}")

    visit(root, "graph")
    return result


@dataclass(frozen=True)
class _ResourceGroup:
    first: int
    last: int
    bindings: tuple[str, ...]
    private_bytes: int
    arena_capacity: int
    coverage: tuple[str, ...]
    temporary_bytes: int = 0
    ordered_temporary_reuse: bool = False

    @property
    def key(self):
        return _digest(
            (
                self.bindings,
                self.arena_capacity,
                self.coverage,
                self.ordered_temporary_reuse,
            )
        )[:32]

    @property
    def choice(self):
        return (
            "queue-ordered-storage"
            if self.ordered_temporary_reuse
            else "generation-owned-storage"
        )

    @property
    def members(self):
        return tuple(f"private:{name}" for name in self.bindings) + (
            ("temporary_arena",) if self.arena_capacity else ()
        )


def _covered_group(definition, first, last, bindings, private_bytes, arena_capacity, temporary_bytes=0):
    sources = definition.sources[first : last + 1]
    if len({source.path.rsplit("/", 1)[0] for source in sources}) == 1:
        coverage = tuple(source.region_id for source in sources)
    else:
        common = []
        for components in zip(*(source.path.split("/") for source in sources)):
            if len(set(components)) != 1:
                break
            common.append(components[0])
        ancestor = "/".join(common)
        coverage = tuple(
            region.region_id
            for region in definition.regions
            if region.path == ancestor or region.path.startswith(ancestor + "/")
        )
        indices = [index for index, source in enumerate(definition.sources) if source.region_id in coverage]
        first, last = indices[0], indices[-1]
    return _ResourceGroup(
        first, last, tuple(sorted(bindings)), private_bytes, arena_capacity, coverage, temporary_bytes
    )


def _combine(definition, groups):
    return _covered_group(
        definition,
        min(group.first for group in groups),
        max(group.last for group in groups),
        tuple(name for group in groups for name in group.bindings),
        sum(group.private_bytes for group in groups),
        max(group.arena_capacity for group in groups),
        sum(group.temporary_bytes for group in groups),
    )


def _resource_groups(definition, nodes):
    from taichi_forge.graph._graph import _GraphInternalNdarraySpec, _GraphTemporaryArena
    from taichi_forge.types.primitive_types import all_types

    spec = definition._runtime_spec
    aliases = {}
    for name, requirement in spec.fixed_runtime_args.items():
        if isinstance(requirement, _GraphInternalNdarraySpec):
            aliases.setdefault(id(requirement), (requirement, []))[1].append(name)
    groups = []
    for requirement, names in aliases.values():
        if requirement.dtype not in all_types or not requirement.shape or any(size <= 0 for size in requirement.shape):
            continue
        uses = [
            index
            for index, source in enumerate(definition.sources)
            if set(names).intersection(
                {binding.name for binding in nodes[source.path].bindings}
                | {effect.resource for effect in nodes[source.path].effects if isinstance(effect.resource, str)}
            )
        ]
        if uses:
            groups.append(_covered_group(definition, uses[0], uses[-1], names, requirement.storage_bytes, 0))
    plan = spec.temporary_memory_plan
    if plan.allocations:
        arena = _GraphTemporaryArena(plan)
        declared = {allocation.name for allocation in plan.allocations}
        uses = [
            index
            for index, source in enumerate(definition.sources)
            if declared.intersection(temporary.name for temporary in nodes[source.path].temporaries)
        ]
        observed = {temporary.name for source in definition.sources for temporary in nodes[source.path].temporaries}
        if arena._available and arena._storage_bytes > 0 and uses and declared <= observed:
            groups.append(
                _covered_group(
                    definition, uses[0], uses[-1], (), 0, arena.capacity, arena._storage_bytes * arena.capacity
                )
            )
    # Structural closure may widen a span. Merge to a fixed point, never skip
    # a source or steal only part of a control subtree to obtain composition.
    while True:
        merged = []
        for group in sorted(groups, key=lambda item: (item.first, item.last)):
            if merged and group.first <= merged[-1].last:
                merged[-1] = _combine(definition, (merged[-1], group))
            else:
                merged.append(group)
        if merged == groups:
            break
        groups = merged
    if len(groups) > 1:
        groups.append(_combine(definition, groups))
    return tuple(groups)


class _CudaGraphStorageOwner:
    def __init__(self, group):
        from taichi_forge._lib import core
        from taichi_forge.lang import impl

        self._group = group
        # This is generation ownership, not a search over threshold values.
        # Live arrays retain backing pages; completed retired generations do
        # not intentionally keep a second, indefinitely retained shared pool.
        self._pool = core._CudaGraphMemoryPool(impl.get_runtime().prog, 0)
        self._allocation_count = 0
        self._requested_bytes = 0

    def allocate(self, dtype, shape):
        from taichi_forge._lib import core
        from taichi_forge.lang._ndarray import ScalarNdarray

        storage = ScalarNdarray._graph_pool_storage(dtype, shape, self._pool)
        self._allocation_count += 1
        self._requested_bytes += prod(shape) * core.data_type_size(dtype)
        return storage

    def close(self):
        self._pool.close()

    def storage_pool_report(self):
        snapshot = self._pool.snapshot()
        return GraphStoragePoolReport(
            allocator="cuda_generation_pool",
            allocation_members=self._group.members,
            allocation_count=self._allocation_count,
            requested_bytes=self._requested_bytes,
            used_current_bytes=snapshot.get("used_current_bytes"),
            reserved_current_bytes=snapshot.get("reserved_current_bytes"),
            used_high_bytes=snapshot.get("used_high_bytes"),
            reserved_high_bytes=snapshot.get("reserved_high_bytes"),
            release_threshold_bytes=snapshot.get("release_threshold_bytes"),
            closed=bool(snapshot["closed"]),
        )


class GraphResourceLifetimeRecipeProvider(GraphRuntimeFragmentProvider):
    descriptor = runtime_family_provider_descriptor(
        "resource_lifetime",
        domain_version="graph-owned-storage-domain-v2",
        semantic_fingerprint="private-usage-components-and-ordered-arena-v2",
        capabilities=(
            "generation-owned-storage",
            "usage-region-composition",
            "typed-runtime-fragment",
        ),
    )

    def _groups(self, definition):
        from taichi_forge._lib import core
        from taichi_forge.lang import impl

        if definition.backend != "cuda" or impl.current_cfg().arch != core.Arch.cuda:
            return (), {}
        if not _pool_available(impl.runtime_generation()):
            return (), {}
        spec = definition._runtime_spec
        root = getattr(spec, "definition_semantic_root", spec.pre_optimization_ir_root)
        nodes = _nodes_by_path(root)
        groups = _resource_groups(definition, nodes)
        from taichi_forge.graph._recipes.runtime_storage import (
            ordered_temporary_reuse_eligible,
        )

        if ordered_temporary_reuse_eligible(spec):
            groups += tuple(
                replace(
                    group,
                    temporary_bytes=group.temporary_bytes // group.arena_capacity,
                    arena_capacity=1,
                    ordered_temporary_reuse=True,
                )
                for group in groups
                if group.arena_capacity
            )
        return groups, nodes

    def fragments(self, definition):
        groups, nodes = self._groups(definition)
        return tuple(self._build_fragment(definition, group, nodes) for group in groups)

    def resolve(self, definition, fragment_key):
        # Refresh cold capability/lifetime facts, but do not rebuild every other
        # fragment for each selected group. No process-global resolution cache.
        groups, nodes = self._groups(definition)
        for group in groups:
            if fragment_key == f"resource_lifetime:{group.key}:{group.choice}":
                return self._build_fragment(definition, group, nodes)
        raise KeyError(f"{self.descriptor.namespace} fragment is unavailable: {fragment_key}")

    def _build_fragment(self, definition, group, nodes):
        tasks = []
        for index, source in enumerate(definition.sources[group.first : group.last + 1]):
            node = nodes[source.path]
            tasks.append(
                GraphFragmentTask.create(
                    f"{group.key}:source:{index}",
                    source.kind,
                    depends_on=() if not tasks else (tasks[-1].task_id,),
                    effects=node.effects,
                    bindings=node.bindings,
                    temporaries=node.temporaries,
                    physical={
                        "execution": "unchanged",
                        "storage_allocator": "cuda_generation_pool",
                        "allocation_members": group.members,
                        "temporary_ring_capacity": group.arena_capacity,
                        "temporary_reuse": (
                            "runtime_stream_order"
                            if group.ordered_temporary_reuse
                            else "completion_ring"
                        ),
                    },
                )
            )
        return _fragment(
            definition,
            family="resource_lifetime",
            source_key=group.key,
            choice_id=group.choice,
            coverage=group.coverage,
            tasks=tasks,
            provider_descriptor=self.descriptor,
            resources=(
                GraphFragmentResourceRequirement(
                    name=f"storage-group:{group.key}",
                    kind="cuda_generation_pool",
                    bytes=group.private_bytes + group.temporary_bytes,
                    alignment=1,
                    ownership="graph_instance",
                    lifetime="graph",
                ),
            ),
        )

    def contribute_runtime(self, assembly, selection):
        groups, _ = self._groups(assembly.definition)
        matches = tuple(group for group in groups if group.key == selection.source_key)
        if len(matches) != 1 or selection.choice_id != matches[0].choice:
            raise ValueError("Graph generation-owned resource selection is unavailable")
        group = matches[0]
        assembly.select_storage(
            GraphRuntimeStoragePlan(
                group.key,
                group.bindings,
                bool(group.arena_capacity),
                partial(_CudaGraphStorageOwner, group),
                temporary_capacity=group.arena_capacity or None,
                ordered_temporary_reuse=group.ordered_temporary_reuse,
            )
        )

    def describe(self, definition, fragment_key):
        metadata = super().describe(definition, fragment_key)
        groups, _ = self._groups(definition)
        group = next(group for group in groups if group.key == metadata["family_selection"]["source_key"])
        return {
            **metadata,
            "display_name": "Graph-generation-owned storage",
            "allocation_members": group.members,
            "private_requested_bytes": group.private_bytes,
            "temporary_requested_bytes": group.temporary_bytes,
            "temporary_ring_capacity": group.arena_capacity,
            "temporary_reuse": (
                "runtime_stream_order"
                if group.ordered_temporary_reuse
                else "completion_ring"
            ),
            "changes": (
                *(
                    ("isolate private storage by complete usage region",)
                    if group.bindings
                    else ()
                ),
                *(
                    (
                        (
                            "reuse one cold-prepared scratch by proven CUDA stream order without arena polling"
                            if group.ordered_temporary_reuse
                            else "prepare the selected bounded temporary ring at setup"
                        ),
                    )
                    if group.arena_capacity
                    else ()
                ),
                "retire pool ownership after the generation and in-flight allocation leases finish",
            ),
            "limitations": (
                "pool pages and eager ring storage can increase resident VRAM",
                "no ownership of vendor-private workspaces",
                "ordered reuse requires one fully lowered CUDA CGraph, no alternate executor or parallel lanes, and whole owned temporary bindings",
                "overlapping execution fragments remain mutually exclusive in the exact-cover composer",
                "closed factory is not proof of immediate device-memory reclamation",
                "performance and generation break-even require workload measurements",
            ),
        }


__all__ = ["GraphResourceLifetimeRecipeProvider"]
