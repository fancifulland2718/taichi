import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import taichi_forge as ti
from taichi_forge.graph._ir import (
    DispatchNode,
    GraphAccess,
    ResourceEffect,
    RuntimeBinding,
    SequentialRegion,
)
from taichi_forge.graph._recipes import (
    CompiledGraphPhysicalManifest,
    GraphDefinition,
    GraphFragmentResourceRequirement,
    GraphFragmentTask,
    GraphMaterializationContext,
    GraphMaterializationError,
    GraphMaterializationProduct,
    GraphMaterializedAllocation,
    GraphMaterializedFragment,
    GraphPhysicalBindingManifest,
    GraphPhysicalCommandManifest,
    GraphPhysicalKernelManifest,
    GraphPhysicalManifestError,
    GraphPhysicalResourceManifest,
    GraphPhysicalSubmissionManifest,
    GraphPhysicalTaskManifest,
    GraphRecipeComposer,
    GraphRecipeFragment,
    GraphRecipeProviderDescriptor,
    GraphRecipeProviderSet,
    PROVIDER_OWNED_WHOLE_GRAPH_V1,
    RUNTIME_GRAPH_ASSEMBLY_V1,
)

from tests import test_utils


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_external_compiled_graph_observation_and_post_evaluator_memory():
    import numpy as np
    from taichi_forge.examples.graph.complete_recipe_provider import AffineProvider, make_builder

    definition = make_builder().freeze()
    provider = AffineProvider(definition)
    source, output = ti.ndarray(ti.i32, 257), ti.ndarray(ti.i32, 257)
    expected = np.arange(257, dtype=np.int32) * 2 + 1
    observed = []

    def evaluate(graph, recipe):
        source.from_numpy(np.arange(257, dtype=np.int32))
        graph.run(graph.bind({"source": source, "output": output}))
        np.testing.assert_array_equal(output.to_numpy(), expected)
        observed.append(recipe.recipe_id)
        return {"dispatches": float(graph.physical_plan()["physical_dispatch_count"])}

    result = definition.search_recipes(
        providers=(provider,),
        budget=ti.graph.GraphSearchBudget(evaluation_limit=8),
        target=ti.graph.GraphOptimizationTarget(objectives=(("dispatches", "min"),)),
    ).run(evaluate)
    assert result.status == "selected"
    assert len(set(observed)) == 2  # Both actually execute, not merely admitted.
    assert not result.selection.manifest.is_baseline
    with definition.materialize(result.selection) as materialized:
        assert materialized.manifest.tasks  # One DSL dispatch may contain several native tasks.
        assert materialized.manifest.provenance["semantic_mapping"] == "provider_declared_whole_graph"
        assert materialized.executor.definition is not definition
        assert evaluate(materialized.executor, result.selection) == {"dispatches": 1.0}
        with pytest.raises(GraphPhysicalManifestError, match="whole-Graph"):
            CompiledGraphPhysicalManifest.from_graph(
                definition,
                definition.recipe_catalog(providers=()).baseline.recipe,
                materialized.executor,
            )


def _definition():
    nodes = (
        DispatchNode(
            name="load",
            effects=(
                ResourceEffect("values", GraphAccess.READ),
                ResourceEffect("middle", GraphAccess.WRITE),
            ),
            bindings=(RuntimeBinding("values", "ndarray"),),
            logical_kernel_identity="kernel:load",
        ),
        DispatchNode(
            name="map",
            effects=(ResourceEffect("middle", GraphAccess.READ_WRITE),),
            logical_kernel_identity="kernel:map",
        ),
        DispatchNode(
            name="store",
            effects=(
                ResourceEffect("middle", GraphAccess.READ),
                ResourceEffect("output", GraphAccess.WRITE),
            ),
            bindings=(RuntimeBinding("output", "ndarray"),),
            logical_kernel_identity="kernel:store",
        ),
    )
    root = SequentialRegion(nodes, name="graph")
    spec = SimpleNamespace(
        pre_optimization_ir_root=root,
        definition_semantic_root=root,
        definition_semantic_sources=(),
        ir_root=root,
        runtime_arg_names=frozenset(("values", "output")),
        fixed_runtime_args={},
        temporary_runtime_arg_names=frozenset(),
        derived_runtime_arg_names=frozenset(),
        execution_definition={
            "nodes": (),
            "dispatch_count": 3,
            "native_count": 0,
            "observation_count": 0,
            "structured_control_count": 0,
            "max_structured_depth": 0,
            "runtime_arg_count": 2,
            "fixed_runtime_arg_count": 0,
            "internal_storage_bytes": 0,
            "temporary_memory_plan": {},
        },
    )
    return GraphDefinition._from_graph_spec(spec, "cuda", core_commit="test-core")


def _bindings(definition):
    return tuple(
        GraphPhysicalBindingManifest(
            binding_index=index,
            name=item.name,
            kinds=item.kinds,
            required=item.required,
            scope=item.scope,
        )
        for index, item in enumerate(definition.binding_abi)
    )


def _manifest(
    definition,
    recipe,
    route,
    *,
    resources=(),
    regions=None,
    bindings=None,
    provenance=None,
    exact=True,
):
    regions = (
        tuple(
            dict.fromkeys(
                region_id
                for step in recipe.execution_steps
                for region_id in step.region_ids
            )
        )
        if regions is None
        else tuple(regions)
    )
    kernels = (
        GraphPhysicalKernelManifest.create(
            0,
            "kernel",
            f"artifact:{route}",
            pipeline_identity=f"pipeline:{route}",
            source_identities=(f"source:{route}",),
            actual_grid_size=8,
            actual_block_size=128,
            static_shared_bytes=256,
            abi_binding_names=tuple(item.name for item in definition.binding_abi),
        ),
    )
    tasks = (
        GraphPhysicalTaskManifest.create(
            0,
            "kernel",
            region_ids=regions,
            kernel_indices=(0,),
            queue="compute",
            binding_names=tuple(item.name for item in definition.binding_abi),
            properties={"actual_route": route},
        ),
    )
    commands = (
        GraphPhysicalCommandManifest.create(
            0,
            "dispatch",
            task_indices=(0,),
            queue="compute",
            recording_scope="whole_graph",
        ),
    )
    submissions = (
        GraphPhysicalSubmissionManifest(
            submission_index=0,
            depends_on=(),
            command_indices=(0,),
            queues=("compute",),
            recording_scope="whole_graph",
            replay_mode="exact_replay",
        ),
    )
    return CompiledGraphPhysicalManifest.create(
        definition,
        recipe,
        backend=definition.backend,
        kernels=kernels,
        tasks=tasks,
        commands=commands,
        submissions=submissions,
        resources=resources,
        binding_abi=_bindings(definition) if bindings is None else tuple(bindings),
        task_topology_exact=exact,
        command_topology_exact=exact,
        allocation_topology_exact=exact,
        provenance=provenance,
    )


_TEST_PROVIDER_DESCRIPTOR = GraphRecipeProviderDescriptor(
    namespace="test.materialization",
    provider_version="1",
    domain_version="test-materialization-v1",
    semantic_fingerprint="test-materialization-semantics-v1",
)
_TEST_MATERIALIZERS = {}


class _TestMaterializationProvider:
    descriptor = _TEST_PROVIDER_DESCRIPTOR

    def __init__(self, fragments, materializers, assembler=None):
        self._fragments = tuple(fragments)
        self._by_key = {
            fragment.fragment_key: fragment for fragment in self._fragments
        }
        self._materializers = dict(materializers)
        self._assembler = assembler

    def discover(self, definition):
        assert all(
            fragment.semantic_graph_id == definition.semantic_graph_id
            for fragment in self._fragments
        )
        return self._fragments

    def resolve(self, _definition, fragment_key):
        return self._by_key[fragment_key]

    def expand(self, _definition, _fragment_key):
        return ()

    def materialize(self, scope, fragment):
        return self._materializers[fragment.fragment_key](scope, fragment)

    def assemble(self, scope, definition, recipe, fragments):
        if self._assembler is None:
            raise AssertionError("test provider has no assembler")
        return self._assembler(scope, definition, recipe, fragments)

    def describe(self, _definition, fragment_key):
        return {"fragment_key": fragment_key}


def _provider_set(definition, fragments, *, assembler=None):
    fragments = tuple(fragments)
    provider = _TestMaterializationProvider(
        fragments,
        {
            fragment.fragment_key: _TEST_MATERIALIZERS[fragment.fragment_id]
            for fragment in fragments
        },
        assembler=assembler,
    )
    return GraphRecipeProviderSet(definition, (provider,))


def _fragment(
    definition,
    source_index,
    planned_route,
    materializer,
    *,
    resources=(),
):
    task = GraphFragmentTask.create(
        f"task:{planned_route}",
        "kernel",
        physical={"planned_route": planned_route},
    )
    fragment = GraphRecipeFragment.create(
        definition,
        provider_namespace=_TEST_PROVIDER_DESCRIPTOR.namespace,
        provider_version="1",
        provider_domain_version=_TEST_PROVIDER_DESCRIPTOR.domain_version,
        fragment_key=f"fragment:{planned_route}",
        coverage_region_ids=(definition.sources[source_index].region_id,),
        tasks=(task,),
        resources=resources,
        provider_metadata={"planned_route": planned_route},
    )
    _TEST_MATERIALIZERS[fragment.fragment_id] = materializer
    return fragment


def _constant_runtime():
    return ("test-runtime", 7)


def test_complete_recipe_publishes_atomically_and_retires_owners_in_reverse_order():
    definition = _definition()
    events = []
    context = None

    def materializer(name):
        def materialize(scope, fragment):
            assert context.statistics()["publications"] == 0
            scope.own(
                {"owner": name},
                release=lambda _value: events.append(f"release:{name}"),
                label=name,
            )
            events.append(f"materialized:{name}")
            return GraphMaterializedFragment.create(fragment, {"segment": name})

        return materialize

    first = _fragment(definition, 0, "first", materializer("first"))
    third = _fragment(definition, 2, "third", materializer("third"))
    recipe = GraphRecipeComposer(definition).compose((first, third))

    def assemble(scope, requested_definition, requested_recipe, fragments):
        assert requested_definition is definition
        assert tuple(item.payload["segment"] for item in fragments) == (
            "first",
            "third",
        )
        assert context.statistics()["publications"] == 0
        scope.own(
            {"owner": "assembly"},
            release=lambda _value: events.append("release:assembly"),
            label="assembly",
        )
        executor = {"route": "first+baseline+third"}
        return GraphMaterializationProduct(
            executor,
            _manifest(definition, requested_recipe, "composed"),
            release=lambda _value: events.append("release:executor"),
        )

    context = GraphMaterializationContext(
        definition,
        provider_set=_provider_set(
            definition,
            (first, third),
            assembler=assemble,
        ),
        runtime_identity_provider=_constant_runtime,
    )
    result = context.materialize(recipe)
    cached = context.materialize(recipe)

    assert result.executor == {"route": "first+baseline+third"}
    assert cached.executor is result.executor
    assert cached.cache_hit
    assert result.manifest.identity_complete
    assert result.manifest.recipe_id == recipe.recipe_id
    json.dumps(result.materialization_report(), sort_keys=True, allow_nan=False)
    assert context.statistics() == {
        "attempts": 1,
        "publications": 1,
        "failures": 0,
        "recipe_cache_hits": 1,
        "materialized_physical_deduplications": 0,
        "rollbacks": 0,
        "rollback_failures": 0,
        "releases": 0,
        "state": "open",
        "active_transactions": 0,
        "live_recipe_ids": 1,
        "live_physical_materializations": 1,
        "live_handles": 2,
        "live_owned_resources": 4,
    }
    result.close()
    assert events == ["materialized:first", "materialized:third"]
    cached.close()
    assert events == [
        "materialized:first",
        "materialized:third",
        "release:executor",
        "release:assembly",
        "release:third",
        "release:first",
    ]
    assert context.statistics()["live_owned_resources"] == 0


def test_failed_candidate_rolls_back_without_disturbing_baseline_or_next_candidate():
    definition = _definition()
    releases = []

    def baseline(scope, requested_definition, recipe):
        assert requested_definition is definition
        return GraphMaterializationProduct(
            {"route": "baseline"},
            _manifest(definition, recipe, "baseline"),
            release=lambda _value: releases.append("baseline"),
        )

    def failed(scope, fragment):
        scope.own(
            object(),
            release=lambda _value: releases.append("failed-candidate"),
            label="failed candidate",
        )
        raise RuntimeError("provider compilation failed")

    good = lambda scope, fragment: GraphMaterializedFragment.create(
        fragment,
        {"route": "good"},
    )
    failed_fragment = _fragment(definition, 1, "failed", failed)
    good_fragment = _fragment(definition, 1, "good", good)
    failed_recipe = GraphRecipeComposer(definition).compose((failed_fragment,))
    good_recipe = GraphRecipeComposer(definition).compose((good_fragment,))

    def assemble(scope, requested_definition, recipe, fragments):
        assert len(fragments) == 1
        return GraphMaterializationProduct(
            {"route": fragments[0].payload["route"]},
            _manifest(definition, recipe, fragments[0].payload["route"]),
            release=lambda value: releases.append(value["route"]),
        )

    context = GraphMaterializationContext(
        definition,
        baseline_materializer=baseline,
        provider_set=_provider_set(
            definition,
            (failed_fragment, good_fragment),
            assembler=assemble,
        ),
        runtime_identity_provider=_constant_runtime,
    )
    baseline_result = context.materialize()
    with pytest.raises(GraphMaterializationError) as failure:
        context.materialize(failed_recipe)
    assert failure.value.phase == "fragment"
    assert failure.value.cleanup_complete
    assert releases == ["failed-candidate"]
    assert baseline_result.executor == {"route": "baseline"}

    good_result = context.materialize(good_recipe)
    assert good_result.executor == {"route": "good"}
    assert context.statistics()["failures"] == 1
    assert context.statistics()["publications"] == 2
    good_result.close()
    baseline_result.close()
    assert releases == ["failed-candidate", "good", "baseline"]


@pytest.mark.parametrize("shareable", [False, True])
def test_actual_physical_identity_only_shares_explicitly_shareable_executors(shareable):
    definition = _definition()
    releases = []

    def materializer(name):
        def materialize(scope, fragment):
            scope.own(
                {"compile": name},
                release=lambda _value: releases.append(f"compile:{name}"),
                label=f"compile {name}",
            )
            return GraphMaterializedFragment.create(fragment, {"name": name})

        return materialize

    first = _fragment(definition, 1, "planned-a", materializer("a"))
    second = _fragment(definition, 1, "planned-b", materializer("b"))
    third = _fragment(definition, 1, "planned-c", materializer("c"))
    composer = GraphRecipeComposer(definition)
    first_recipe = composer.compose((first,))
    second_recipe = composer.compose((second,))
    third_recipe = composer.compose((third,))
    assert first_recipe.planned_physical_id != second_recipe.planned_physical_id

    def assemble(scope, requested_definition, recipe, fragments):
        name = fragments[0].payload["name"]
        resources = (
            (
                GraphPhysicalResourceManifest(
                    resource_id="actual-workspace",
                    kind="device_buffer",
                    requested_bytes=64,
                    allocated_bytes=256,
                    alignment=256,
                    ownership="graph_instance",
                    lifetime="graph",
                ),
            )
            if name == "c"
            else ()
        )
        return GraphMaterializationProduct(
            {"executor": name},
            _manifest(
                definition,
                recipe,
                "actual-shared-route",
                resources=resources,
                provenance={"candidate": name},
            ),
            release=lambda _value: releases.append(f"executor:{name}"),
            shareable_executor=shareable,
        )

    context = GraphMaterializationContext(
        definition,
        provider_set=_provider_set(
            definition,
            (first, second, third),
            assembler=assemble,
        ),
        runtime_identity_provider=_constant_runtime,
    )
    first_result = context.materialize(first_recipe)
    second_result = context.materialize(second_recipe)
    third_result = context.materialize(third_recipe)

    assert first_result.materialized_physical_id == (
        second_result.materialized_physical_id
    )
    assert second_result.deduplicated == shareable
    assert (second_result.executor is first_result.executor) == shareable
    assert (second_result.resource_instance_id == first_result.resource_instance_id) == shareable
    assert releases == (["executor:b", "compile:b"] if shareable else [])
    assert context.statistics()["materialized_physical_deduplications"] == int(shareable)
    assert third_result.materialized_physical_id != (
        first_result.materialized_physical_id
    )
    assert not third_result.deduplicated
    assert third_result.manifest.persistent_allocated_bytes == 256

    first_result.close()
    assert releases == (["executor:b", "compile:b"] if shareable else ["executor:a", "compile:a"])
    second_result.close()
    order = ("b", "a") if shareable else ("a", "b")
    assert releases == [label for name in order for label in (f"executor:{name}", f"compile:{name}")]
    third_result.close()
    assert releases[-2:] == ["executor:c", "compile:c"]


def test_physical_identity_uses_allocation_policy_not_backing_page_size():
    definition = _definition()
    recipe = GraphRecipeComposer(definition).compose()
    resource = GraphPhysicalResourceManifest(
        resource_id="workspace",
        kind="device_buffer",
        requested_bytes=64,
        allocated_bytes=256,
        alignment=256,
        ownership="graph_instance",
        lifetime="graph",
        allocation_members=("temporary",),
        allocation_count=1,
    )
    baseline = _manifest(definition, recipe, "same-plan", resources=(resource,))
    larger_backing = _manifest(
        definition,
        recipe,
        "same-plan",
        resources=(replace(resource, allocated_bytes=512),),
    )
    assert baseline.materialized_physical_id == larger_backing.materialized_physical_id
    assert baseline.persistent_allocated_bytes == 256
    assert larger_backing.persistent_allocated_bytes == 512
    assert "allocated_bytes" not in baseline.to_dict()["resource_plan"][0]
    for changed in (
        replace(resource, requested_bytes=128),
        replace(resource, allocation_count=2),
        replace(resource, allocation_members=("different-temporary",)),
        replace(resource, lifetime="submission"),
    ):
        assert _manifest(
            definition, recipe, "same-plan", resources=(changed,)
        ).materialized_physical_id != (baseline.materialized_physical_id)


def test_allocated_resources_must_match_requirements_and_appear_in_manifest():
    definition = _definition()
    releases = []
    include_resource = {"value": False}
    requirement = GraphFragmentResourceRequirement(
        "workspace",
        "device_buffer",
        bytes=4096,
        alignment=256,
        ownership="graph_instance",
        lifetime="graph",
        exclusive_submission=True,
    )
    physical = GraphPhysicalResourceManifest(
        resource_id="workspace",
        kind="device_buffer",
        requested_bytes=4096,
        allocated_bytes=4352,
        alignment=256,
        ownership="graph_instance",
        lifetime="graph",
        exclusive_submission=True,
    )

    def allocator(requested):
        assert requested is requirement
        value = {"allocation": len(releases)}
        return GraphMaterializedAllocation(
            value,
            physical,
            release=lambda _value: releases.append("workspace"),
        )

    def materialize(scope, fragment):
        allocation = scope.allocate(requirement)
        return GraphMaterializedFragment.create(fragment, allocation)

    fragment = _fragment(
        definition,
        1,
        "workspace",
        materialize,
        resources=(requirement,),
    )
    recipe = GraphRecipeComposer(definition).compose((fragment,))

    def assemble(scope, requested_definition, requested_recipe, fragments):
        resources = (physical,) if include_resource["value"] else ()
        return GraphMaterializationProduct(
            {"allocation": fragments[0].payload},
            _manifest(
                definition,
                requested_recipe,
                "workspace-route",
                resources=resources,
            ),
            release=lambda _value: releases.append("executor"),
        )

    context = GraphMaterializationContext(
        definition,
        provider_set=_provider_set(
            definition,
            (fragment,),
            assembler=assemble,
        ),
        resource_allocator=allocator,
        runtime_identity_provider=_constant_runtime,
    )
    with pytest.raises(GraphMaterializationError, match="omits") as failure:
        context.materialize(recipe)
    assert failure.value.cleanup_complete
    assert releases == ["executor", "workspace"]
    assert context.statistics()["live_physical_materializations"] == 0

    include_resource["value"] = True
    result = context.materialize(recipe)
    assert result.manifest.persistent_requested_bytes == 4096
    assert result.manifest.persistent_allocated_bytes == 4352
    result.close()
    assert releases == ["executor", "workspace", "executor", "workspace"]


def test_runtime_generation_change_aborts_publish_and_rolls_back_the_candidate():
    definition = _definition()
    runtime = {"generation": 4}
    releases = []

    def materialize(scope, fragment):
        scope.own(
            object(),
            release=lambda _value: releases.append("compiled-fragment"),
            label="compiled fragment",
        )
        return GraphMaterializedFragment.create(fragment, {"route": "candidate"})

    fragment = _fragment(definition, 1, "runtime-drift", materialize)
    recipe = GraphRecipeComposer(definition).compose((fragment,))

    def assemble(scope, requested_definition, requested_recipe, fragments):
        runtime["generation"] += 1
        return GraphMaterializationProduct(
            {"route": fragments[0].payload["route"]},
            _manifest(definition, requested_recipe, "runtime-drift"),
            release=lambda _value: releases.append("executor"),
        )

    context = GraphMaterializationContext(
        definition,
        provider_set=_provider_set(
            definition,
            (fragment,),
            assembler=assemble,
        ),
        runtime_identity_provider=lambda: ("runtime", runtime["generation"]),
    )
    with pytest.raises(GraphMaterializationError, match="runtime changed") as failure:
        context.materialize(recipe)
    assert failure.value.cleanup_complete
    assert releases == ["executor", "compiled-fragment"]
    assert context.statistics()["publications"] == 0
    assert context.statistics()["live_owned_resources"] == 0
    context.close()


def test_incomplete_cleanup_poisoning_and_manifest_integrity_fail_closed():
    definition = _definition()
    baseline = GraphRecipeComposer(definition).compose()
    source_regions = tuple(source.region_id for source in definition.sources)
    with pytest.raises(GraphPhysicalManifestError, match="complete executable"):
        _manifest(
            definition,
            baseline,
            "missing-region",
            regions=source_regions[:-1],
        )
    with pytest.raises(GraphPhysicalManifestError, match="public binding ABI"):
        _manifest(
            definition,
            baseline,
            "missing-binding",
            bindings=_bindings(definition)[:-1],
        )
    forged = replace(baseline, planned_physical_id="planned-physical:forged")
    admission_context = GraphMaterializationContext(
        definition,
        runtime_identity_provider=_constant_runtime,
    )
    with pytest.raises(GraphMaterializationError, match="canonical composition"):
        admission_context.materialize(forged)
    admission_context.close()

    def failed(scope, fragment):
        def cannot_release(_value):
            raise RuntimeError("allocator lost the device owner")

        scope.own(object(), release=cannot_release, label="poisoned allocation")
        raise RuntimeError("materialization failed after allocation")

    fragment = _fragment(definition, 1, "poison", failed)
    recipe = GraphRecipeComposer(definition).compose((fragment,))
    context = GraphMaterializationContext(
        definition,
        provider_set=_provider_set(definition, (fragment,)),
        runtime_identity_provider=_constant_runtime,
    )
    with pytest.raises(GraphMaterializationError) as failure:
        context.materialize(recipe)
    assert not failure.value.cleanup_complete
    assert len(failure.value.rollback_errors) == 1
    assert context.statistics()["state"] == "poisoned"
    with pytest.raises(GraphMaterializationError, match="poisoned"):
        context.materialize(baseline)
    context.close()


def test_external_provider_materializes_both_complete_recipe_assembly_protocols():
    definition = _definition()

    class ExternalProvider:
        descriptor = ti.graph.GraphRecipeProviderDescriptor(
            namespace="external.complete_graph",
            provider_version="2.1",
            domain_version="external-domain-v3",
            semantic_fingerprint="external-semantics-v1",
            assembly_protocols=(
                ti.graph.RUNTIME_GRAPH_ASSEMBLY_V1,
                ti.graph.PROVIDER_OWNED_WHOLE_GRAPH_V1,
            ),
            fragment_key_schema="route-name.v1",
        )

        def __init__(self):
            common = {
                "provider_namespace": self.descriptor.namespace,
                "provider_version": self.descriptor.provider_version,
                "provider_domain_version": self.descriptor.domain_version,
                "assembly_provider_namespace": self.descriptor.namespace,
            }
            self.runtime_fragment = ti.graph.GraphRecipeFragment.create(
                definition,
                fragment_key="runtime-map",
                coverage_region_ids=(definition.sources[1].region_id,),
                tasks=(
                    ti.graph.GraphFragmentTask.create(
                        "runtime-map",
                        "runtime_graph_fragment",
                        physical={"route": "runtime-map"},
                    ),
                ),
                provider_metadata={"route": "runtime-map"},
                **common,
            )
            self.whole_fragment = ti.graph.GraphRecipeFragment.create(
                definition,
                fragment_key="owned-whole-graph",
                coverage_region_ids=tuple(
                    region.region_id for region in definition.regions
                ),
                tasks=(
                    ti.graph.GraphFragmentTask.create(
                        "owned-whole-graph",
                        "provider_owned_graph",
                        physical={"route": "owned-whole-graph"},
                    ),
                ),
                assembly_protocol=ti.graph.PROVIDER_OWNED_WHOLE_GRAPH_V1,
                provider_metadata={"route": "owned-whole-graph"},
                **common,
            )
            self._by_key = {
                fragment.fragment_key: fragment
                for fragment in (self.runtime_fragment, self.whole_fragment)
            }
            self.resolved_keys = []

        def discover(self, requested_definition):
            assert requested_definition is definition
            return tuple(self._by_key.values())

        def resolve(self, requested_definition, fragment_key):
            assert requested_definition is definition
            self.resolved_keys.append(fragment_key)
            return self._by_key[fragment_key]

        def expand(self, requested_definition, fragment_key):
            self.resolve(requested_definition, fragment_key)
            return ()

        def materialize(self, scope, fragment):
            return ti.graph.GraphMaterializedFragment.create(
                fragment,
                {"route": fragment.provider_metadata["route"]},
            )

        def assemble(self, scope, requested_definition, recipe, fragments):
            assert requested_definition is definition
            route = fragments[0].payload["route"]
            return ti.graph.GraphMaterializationProduct(
                {"route": route, "protocol": recipe.assembly_protocol},
                _manifest(definition, recipe, route),
            )

        def describe(self, requested_definition, fragment_key):
            assert requested_definition is definition
            return {"route": fragment_key}

    provider = ExternalProvider()
    catalog = definition.recipe_catalog(providers=(provider,))
    recipes = tuple(
        entry.recipe
        for entry in catalog.entries(stage="single-region")
    )
    assert len(recipes) == 2
    by_protocol = {recipe.assembly_protocol: recipe for recipe in recipes}
    assert set(by_protocol) == {
        RUNTIME_GRAPH_ASSEMBLY_V1,
        PROVIDER_OWNED_WHOLE_GRAPH_V1,
    }
    assert by_protocol[PROVIDER_OWNED_WHOLE_GRAPH_V1].baseline_coverage_region_ids == ()
    assert all(catalog.resolve(recipe.recipe_id) == recipe for recipe in recipes)

    with definition.materialization_context(
        provider_set=catalog.provider_set,
        runtime_identity_provider=_constant_runtime,
    ) as context:
        runtime = context.materialize(by_protocol[RUNTIME_GRAPH_ASSEMBLY_V1])
        whole = context.materialize(by_protocol[PROVIDER_OWNED_WHOLE_GRAPH_V1])
        assert runtime.executor == {
            "route": "runtime-map",
            "protocol": RUNTIME_GRAPH_ASSEMBLY_V1,
        }
        assert whole.executor == {
            "route": "owned-whole-graph",
            "protocol": PROVIDER_OWNED_WHOLE_GRAPH_V1,
        }
        assert runtime.materialized_physical_id != whole.materialized_physical_id
        runtime.close()
        whole.close()

    assert provider.resolved_keys == [
        "runtime-map",
        "owned-whole-graph",
        "runtime-map",
        "owned-whole-graph",
    ]
    assert ti.graph.GraphRecipeProviderDescriptor is GraphRecipeProviderDescriptor


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_native_task_observation_produces_stable_complete_baseline_manifest():
    @ti.kernel
    def add_one(
        values: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in values:
            output[i] = values[i] + 1

    values_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY,
        "values",
        ti.i32,
        ndim=1,
    )
    output_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY,
        "output",
        ti.i32,
        ndim=1,
    )
    builder = ti.graph.GraphBuilder()
    builder.dispatch(add_one, values_arg, output_arg)
    definition = builder.freeze()

    first_context = definition.materialization_context()
    second_context = definition.materialization_context()
    first = first_context.materialize()
    second = second_context.materialize()
    try:
        first_report = first.materialization_report()
        assert first.manifest.task_topology_exact
        assert first.manifest.tasks
        assert first.manifest.kernels
        assert all(kernel.artifact_identity for kernel in first.manifest.kernels)
        assert tuple(item["name"] for item in first_report["binding_abi"]) == (
            "output",
            "values",
        )
        assert first.materialized_physical_id == second.materialized_physical_id
        assert first.resource_instance_id != second.resource_instance_id

        values = ti.ndarray(ti.i32, shape=8)
        output = ti.ndarray(ti.i32, shape=8)
        values.from_numpy(__import__("numpy").arange(8, dtype="int32"))
        retained_executor = first.executor
        first.executor.run({"values": values, "output": output})
        assert output.to_numpy().tolist() == list(range(1, 9))
        warmed = CompiledGraphPhysicalManifest.from_graph(
            definition,
            definition.recipe_catalog(providers=()).baseline.recipe,
            first.executor,
        )
        assert warmed.materialized_physical_id == first.materialized_physical_id
        assert (
            warmed.persistent_allocated_bytes
            >= first.manifest.persistent_allocated_bytes
        )
        first.close()
        first.close()
        second.executor.run({"values": values, "output": output})
        assert output.to_numpy().tolist() == list(range(1, 9))
    finally:
        first.close()
        second.close()

    one_shot = definition.materialize()
    assert one_shot.materialized_physical_id == first_report["materialized_physical_id"]
    one_shot.close()
    with pytest.raises(GraphMaterializationError, match="closed"):
        _ = one_shot.executor
    with pytest.raises(RuntimeError, match="closed"):
        retained_executor.run({"values": values, "output": output})


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_materialization_failure_and_reset_retire_retained_executors(monkeypatch):
    from taichi_forge.examples.graph.complete_recipe_provider import make_builder
    from taichi_forge.graph._recipes import families

    definition = make_builder().freeze()
    failed_executors = []

    def fail_observation(definition, recipe, graph):
        failed_executors.append(graph)
        raise ValueError("injected baseline observation failure")

    context = definition.materialization_context()
    with monkeypatch.context() as patch:
        patch.setattr(families, "observe_graph_physical_manifest", fail_observation)
        with pytest.raises(GraphMaterializationError, match="injected baseline"):
            context.materialize()
    assert not failed_executors[0]._runtime_valid
    assert context.statistics()["live_owned_resources"] == 0

    handle = context.materialize()
    executor = handle.executor
    source, output = ti.ndarray(ti.i32, 257), ti.ndarray(ti.i32, 257)
    source.fill(3)
    executor.run(executor.bind({"source": source, "output": output}))
    ti.reset()
    assert context.statistics()["state"] == "closed"
    assert not executor._runtime_valid
    with pytest.raises(GraphMaterializationError, match="closed"):
        _ = handle.executor
    handle.close()
    context.close()
