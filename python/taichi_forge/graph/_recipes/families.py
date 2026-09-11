"""Migrate established Forge Graph families into complete recipe fragments.

This module is the only boundary between older family-specific manifests and
the complete-Graph recipe system. Providers describe region replacements;
materialization happens once at the whole-Graph boundary and never reconstructs
a choice from process environment state.
"""

from dataclasses import dataclass

from taichi_forge.graph._recipes.fragments import (
    GraphFragmentBindingRequirement,
    GraphFragmentResourceRequirement,
    GraphFragmentSubmissionRequirement,
    GraphFragmentTask,
    GraphRecipeFragment,
)
from taichi_forge.graph._recipes.materialize import (
    GraphMaterializationProduct,
    GraphMaterializedFragment,
)
from taichi_forge.graph._recipes.physical import observe_graph_physical_manifest
from taichi_forge.graph._recipes.providers import (
    GraphRecipeProviderDescriptor,
    RUNTIME_GRAPH_ASSEMBLY_V1,
)

_PROVIDER_VERSION = "complete-graph-family-v1"
_RUNTIME_ASSEMBLY_PROVIDER_DESCRIPTOR = GraphRecipeProviderDescriptor(
    namespace="taichi_forge.graph.runtime_assembly",
    provider_version=_PROVIDER_VERSION,
    domain_version="runtime-graph-assembly-v1",
    semantic_fingerprint="runtime-graph-assembly-delegation-v1",
    assembly_protocols=(RUNTIME_GRAPH_ASSEMBLY_V1,),
    capabilities=("typed-runtime-graph-assembly",),
    fragment_key_schema="assembly-owner-no-fragments.v1",
)


def runtime_family_provider_descriptor(
    family,
    *,
    capabilities,
    domain_version="existing-family-domain-v1",
    semantic_fingerprint="existing-family-fragment-generation-v1",
):
    """Create one independently owned built-in runtime-fragment contract."""

    return GraphRecipeProviderDescriptor(
        namespace=f"taichi_forge.graph.{family}",
        provider_version=_PROVIDER_VERSION,
        domain_version=domain_version,
        semantic_fingerprint=semantic_fingerprint,
        assembly_protocols=(RUNTIME_GRAPH_ASSEMBLY_V1,),
        capabilities=tuple(capabilities),
        fragment_key_schema=f"{family}:source:choice.v1",
    )


@dataclass(frozen=True)
class GraphFamilySelection:
    """One explicit provider choice consumed by the whole-Graph assembler."""

    family: str
    source_key: str
    choice_id: str
    materialization_choice: str
    coverage_region_ids: tuple[str, ...]

    def to_dict(self):
        return {
            "family": self.family,
            "source_key": self.source_key,
            "choice_id": self.choice_id,
            "materialization_choice": self.materialization_choice,
            "coverage_region_ids": self.coverage_region_ids,
        }

    @classmethod
    def from_fragment(cls, fragment):
        payload = fragment.provider_metadata.get("family_selection")
        if not isinstance(payload, dict):
            raise ValueError("Graph family fragment has no stable selection metadata")
        return cls(
            family=str(payload["family"]),
            source_key=str(payload["source_key"]),
            choice_id=str(payload["choice_id"]),
            materialization_choice=str(payload["materialization_choice"]),
            coverage_region_ids=tuple(payload["coverage_region_ids"]),
        )


def _subtree_regions(definition, root_region_id):
    root = definition.region(root_region_id)
    return tuple(
        region.region_id
        for region in definition.regions
        if region.path == root.path or region.path.startswith(root.path + "/")
    )


def _binding_requirements(definition, coverage):
    coverage = frozenset(coverage)
    return tuple(
        GraphFragmentBindingRequirement(
            item.name,
            kinds=item.kinds,
            required=item.required,
            scope=item.scope,
        )
        for item in definition.binding_abi
        if not item.region_ids or coverage.intersection(item.region_ids)
    )


def _workspace_resource(source_key, manifest):
    payload = manifest.to_dict()
    workspace = dict(payload.get("workspace", {}))
    size = int(workspace.get("bytes", workspace.get("action_owned_bytes", 0)) or 0)
    ownership = workspace.get("ownership", "none")
    if size <= 0 or ownership == "none":
        return ()
    normalized_ownership = {
        "graph_instance": "graph_instance",
        "graph_native_action": "graph_instance",
        "fragment": "fragment",
        "shared": "shared",
    }.get(ownership, "fragment")
    return (
        GraphFragmentResourceRequirement(
            name=f"{source_key}:workspace",
            kind="workspace",
            bytes=size,
            alignment=1,
            ownership=normalized_ownership,
            lifetime="graph",
            exclusive_submission=bool(
                workspace.get(
                    "exclusive_submission",
                    normalized_ownership == "graph_instance",
                )
            ),
        ),
    )


def _manifest_tasks(family, source_key, manifest):
    payload = manifest.to_dict()
    stages = tuple(payload.get("physical_stages", ()))
    if not stages:
        stages = (
            {
                "name": payload.get("strategy", family),
                "kind": family,
                "manifest": payload,
            },
        )
    tasks = []
    for index, stage in enumerate(stages):
        stage = dict(stage)
        tasks.append(
            GraphFragmentTask.create(
                f"{source_key}:task:{index}",
                str(stage.get("name") or stage.get("kind") or family),
                depends_on=(() if index == 0 else (tasks[-1].task_id,)),
                physical={"family": family, "stage": stage},
            )
        )
    return tuple(tasks)


def _fragment(
    definition,
    *,
    family,
    source_key,
    choice_id,
    materialization_choice=None,
    coverage,
    tasks,
    resources=(),
    exclusive_submission=False,
    provider_descriptor=None,
    executor_kind="",
    compatible_executor_kinds=(),
):
    provider_descriptor = (
        _RUNTIME_ASSEMBLY_PROVIDER_DESCRIPTOR
        if provider_descriptor is None
        else provider_descriptor
    )
    coverage = tuple(coverage)
    selection = GraphFamilySelection(
        family=family,
        source_key=source_key,
        choice_id=choice_id,
        materialization_choice=(
            choice_id if materialization_choice is None else materialization_choice
        ),
        coverage_region_ids=coverage,
    )
    return GraphRecipeFragment.create(
        definition,
        provider_namespace=f"taichi_forge.graph.{family}",
        provider_version=provider_descriptor.provider_version,
        provider_domain_version=provider_descriptor.domain_version,
        fragment_key=f"{family}:{source_key}:{choice_id}",
        coverage_region_ids=coverage,
        tasks=tasks,
        binding_requirements=_binding_requirements(definition, coverage),
        resources=resources,
        submission=GraphFragmentSubmissionRequirement(
            recording_scope="whole_graph",
            exclusive_submission=bool(exclusive_submission),
            executor_kind=executor_kind,
            compatible_executor_kinds=compatible_executor_kinds,
        ),
        backend_requirements=(definition.backend,),
        assembly_protocol=RUNTIME_GRAPH_ASSEMBLY_V1,
        assembly_provider_namespace=(_RUNTIME_ASSEMBLY_PROVIDER_DESCRIPTOR.namespace),
        provider_metadata={"family_selection": selection.to_dict()},
    )


def _choice_fragment(
    definition,
    family,
    source_key,
    coverage,
    manifest,
    *,
    materialization_choice=None,
    provider_descriptor=None,
):
    payload = manifest.to_dict()
    exclusive = bool(
        payload.get("submission", {}).get("exclusive", False)
        or payload.get("workspace", {}).get("exclusive_submission", False)
    )
    resources = _workspace_resource(source_key, manifest)
    return _fragment(
        definition,
        family=family,
        source_key=source_key,
        choice_id=manifest.recipe_id,
        materialization_choice=materialization_choice,
        coverage=coverage,
        tasks=_manifest_tasks(family, source_key, manifest),
        resources=resources,
        exclusive_submission=exclusive
        or any(item.exclusive_submission for item in resources),
        provider_descriptor=provider_descriptor,
    )


def _native_source_coverage(definition, source):
    prefix = f"graph/{source._recipe_node_index}:"
    root = next(
        (
            region
            for region in definition.regions
            if region.parent_region_id == definition.regions[0].region_id
            and region.path.startswith(prefix)
        ),
        None,
    )
    return () if root is None else _subtree_regions(definition, root.region_id)


def _recipe_operation_dispatch_count(operation):
    kind = operation[0]
    if kind in ("dispatch", "native"):
        return 1
    if kind == "sequential":
        return operation[1]._dispatch_count
    if kind == "bounded":
        return operation[1]._recipe_physical_dispatches
    if kind == "graph_reduction":
        return operation[1].selected_physical_dispatches
    raise ValueError(f"unknown frozen Graph recipe operation {kind!r}")


def _operation_source_coverage(
    definition,
    spec,
    operation_kind,
    requested_source,
):
    regions_by_path = {region.path: region for region in definition.regions}
    for node_index, node in enumerate(spec.nodes):
        operations = getattr(node, "recipe_operations", ())
        if not operations:
            continue
        node_path = f"graph/{node_index}:{node.ir_node.kind}"
        dispatch_index = 0
        for operation in operations:
            count = _recipe_operation_dispatch_count(operation)
            if operation[0] == operation_kind and operation[1] is requested_source:
                coverage = []
                for index in range(dispatch_index, dispatch_index + count):
                    region = regions_by_path.get(f"{node_path}/{index}:dispatch")
                    if region is None:
                        return ()
                    coverage.extend(_subtree_regions(definition, region.region_id))
                return tuple(coverage)
            dispatch_index += count
    return ()


def _semantic_source_fragments(
    definition,
    spec,
    sources,
    family,
    *,
    provider_descriptor=None,
):
    if len(sources) == 1 and len(definition.sources) == 1:
        coverage_by_source = (
            _subtree_regions(definition, definition.sources[0].region_id),
        )
    elif family == "native_algorithm":
        coverage_by_source = tuple(
            _native_source_coverage(definition, source) for source in sources
        )
    elif family == "graph_reduction":
        coverage_by_source = tuple(
            _operation_source_coverage(
                definition,
                spec,
                "graph_reduction",
                source,
            )
            for source in sources
        )
    else:
        return ()

    result = []
    for source, coverage in zip(sources, coverage_by_source):
        if not coverage:
            continue
        for manifest in source.manifests():
            if manifest.recipe_id == source.selected_recipe_id:
                continue
            result.append(
                _choice_fragment(
                    definition,
                    family,
                    source._recipe_source_key,
                    coverage,
                    manifest,
                    provider_descriptor=provider_descriptor,
                )
            )
    return tuple(result)


class GraphRuntimeFragmentProvider:
    """Shared stable-key mechanics for one independently owned family."""

    descriptor = None

    def discover(self, definition):
        return self.fragments(definition)

    def resolve(self, definition, fragment_key):
        matches = tuple(
            fragment
            for fragment in self.fragments(definition)
            if fragment.fragment_key == fragment_key
        )
        if len(matches) != 1:
            raise KeyError(
                f"{self.descriptor.namespace} fragment is unavailable: "
                f"{fragment_key}"
            )
        return matches[0]

    def expand(self, definition, fragment_key):
        self.resolve(definition, fragment_key)
        return ()

    def materialize(self, scope, fragment):
        selection = GraphFamilySelection.from_fragment(fragment)
        if fragment.coverage_region_ids != selection.coverage_region_ids:
            raise ValueError("Graph family fragment coverage changed before build")
        return GraphMaterializedFragment.create(fragment, selection)

    def describe(self, definition, fragment_key):
        return self.resolve(definition, fragment_key).provider_metadata


class GraphRuntimeAssemblyProvider:
    """Assembly-only owner for independently provided runtime fragments."""

    descriptor = _RUNTIME_ASSEMBLY_PROVIDER_DESCRIPTOR

    def discover(self, definition):
        return ()

    def fragments(self, definition):
        return ()

    def resolve(self, definition, fragment_key):
        matches = tuple(
            fragment
            for fragment in self.fragments(definition)
            if fragment.fragment_key == fragment_key
        )
        if len(matches) != 1:
            raise KeyError(
                f"runtime assembly provider owns no fragments: {fragment_key}"
            )
        return matches[0]

    def expand(self, definition, fragment_key):
        self.resolve(definition, fragment_key)
        return ()

    def materialize(self, scope, fragment):
        raise TypeError("runtime assembly provider owns no fragments")

    def assemble(self, scope, definition, recipe, materialized_fragments):
        return assemble_runtime_graph_recipe(
            scope,
            definition,
            recipe,
            materialized_fragments,
        )

    def describe(self, definition, fragment_key):
        return self.resolve(definition, fragment_key).provider_metadata


def default_graph_recipe_providers():
    """Return the built-in providers required by the public recipe path."""

    from taichi_forge.graph._recipes.binding_frames import GraphBindingFrameRecipeProvider
    from taichi_forge.graph._recipes.branch_join import (
        GraphBranchJoinRecipeProvider,
    )
    from taichi_forge.graph._recipes.dispatch_families import (
        GraphOffloadPhaseFusionRecipeProvider,
        GraphSparseTraversalRecipeProvider,
    )
    from taichi_forge.graph._recipes.graph_memory import GraphMemoryRecipeProvider
    from taichi_forge.graph._recipes.map_fusion import GraphMapFusionRecipeProvider
    from taichi_forge.graph._recipes.resource_lifetime import GraphResourceLifetimeRecipeProvider
    from taichi_forge.graph._recipes.value_fusion import GraphValueFusionRecipeProvider
    from taichi_forge.graph._recipes.semantic_families import (
        GraphBoundedExecutionRecipeProvider,
        GraphNativeAlgorithmRecipeProvider,
        GraphReductionRecipeProvider,
        GraphStructuredControlRecipeProvider,
    )
    from taichi_forge.graph._recipes.submission_families import (
        GraphRecordingPartitionRecipeProvider,
        GraphWorkspaceConcurrencyRecipeProvider,
    )

    return (
        GraphRuntimeAssemblyProvider(),
        GraphMapFusionRecipeProvider(),
        GraphMemoryRecipeProvider(),
        GraphOffloadPhaseFusionRecipeProvider(),
        GraphSparseTraversalRecipeProvider(),
        GraphBranchJoinRecipeProvider(),
        GraphRecordingPartitionRecipeProvider(),
        GraphBindingFrameRecipeProvider(),
        GraphResourceLifetimeRecipeProvider(),
        GraphWorkspaceConcurrencyRecipeProvider(),
        GraphBoundedExecutionRecipeProvider(),
        GraphStructuredControlRecipeProvider(),
        GraphReductionRecipeProvider(),
        GraphNativeAlgorithmRecipeProvider(),
        GraphValueFusionRecipeProvider(),
    )


def assemble_runtime_graph_recipe(
    scope,
    definition,
    recipe,
    materialized_fragments,
):
    """Delegate every fragment contribution, then build one runtime Graph."""

    from taichi_forge.graph._recipes.runtime_assembly import (
        GraphRuntimeRecipeAssembly,
    )

    if len(materialized_fragments) != len(recipe.fragments):
        raise ValueError("runtime Graph assembler received an incomplete fragment set")
    assembly = GraphRuntimeRecipeAssembly(definition)
    provider_set = scope._context.provider_set
    for fragment, materialized in zip(recipe.fragments, materialized_fragments):
        if materialized.fragment_id != fragment.fragment_id:
            raise ValueError("runtime Graph materialization order changed")
        selection = materialized.payload
        if not isinstance(selection, GraphFamilySelection):
            raise TypeError("runtime Graph assembler received an unknown payload")
        provider = provider_set.provider_for_fragment_namespace(
            fragment.provider_namespace
        )
        contribute = getattr(provider, "contribute_runtime", None)
        if not callable(contribute):
            raise TypeError(
                "runtime Graph fragment provider has no assembly contribution"
            )
        contribute(assembly, selection)
    graph = definition._runtime_spec.materialize_complete_recipe(
        definition,
        recipe,
        assembly,
        workspace_lanes=scope._context.workspace_lanes,
        workspace_saturation=scope._context.workspace_saturation,
    )
    release = assembly.executor_release
    try:
        manifest = observe_graph_physical_manifest(definition, recipe, graph)
    except BaseException:
        if release is not None:
            release(graph)
        raise
    return GraphMaterializationProduct(graph, manifest, release=release)


def materialize_runtime_graph_baseline(scope, definition, recipe):
    """Materialize the exact all-baseline whole-Graph recipe.

    Reusing ``GraphDefinition.compile()`` here would also reuse any map fusion
    that the ordinary GraphBuilder selected while the definition was frozen.
    Complete recipes instead treat every uncovered map region as a singleton,
    so route the empty selection through the same exact whole-Graph assembler
    as non-baseline recipes.
    """

    if definition._runtime_spec.fusion_plan.candidate_recipes:
        from taichi_forge.graph._recipes.runtime_assembly import (
            GraphRuntimeRecipeAssembly,
        )

        graph = definition._runtime_spec.materialize_complete_recipe(
            definition,
            recipe,
            GraphRuntimeRecipeAssembly(definition),
            workspace_lanes=scope._context.workspace_lanes,
            workspace_saturation=scope._context.workspace_saturation,
        )
    else:
        # Without a map axis, compile preserves the exact baseline partition
        # while instantiating independent mutable control/provider state.
        graph = definition.compile(
            workspace_lanes=scope._context.workspace_lanes,
            workspace_saturation=scope._context.workspace_saturation,
        )
    manifest = observe_graph_physical_manifest(definition, recipe, graph)
    return GraphMaterializationProduct(graph, manifest)


__all__ = [
    "GraphRuntimeAssemblyProvider",
    "GraphRuntimeFragmentProvider",
    "GraphFamilySelection",
    "assemble_runtime_graph_recipe",
    "default_graph_recipe_providers",
    "materialize_runtime_graph_baseline",
    "runtime_family_provider_descriptor",
]
