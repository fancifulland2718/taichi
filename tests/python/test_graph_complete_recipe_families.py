from itertools import pairwise

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge._lib import core as ti_core
from taichi_forge.graph._recipes.physical import observe_graph_physical_manifest
from taichi_forge.lang.exception import TaichiRuntimeError

from tests import test_utils

_MATERIALIZATION_ENVIRONMENTS = (
    "TI_CUDA_BOUNDED_DISPATCH_MODE",
    "TI_GRAPH_CUDA_BOUNDED_UPDATE_POLICY",
)


def _family_fragments(catalog, family):
    namespace = f"taichi_forge.graph.{family}"
    return tuple(
        fragment
        for fragment in catalog.fragments
        if fragment.provider_namespace == namespace
    )


def _selection(fragment):
    from taichi_forge.graph._recipes import GraphFamilySelection

    return GraphFamilySelection.from_fragment(fragment)


def test_default_runtime_recipe_families_have_independent_provider_ownership():
    from taichi_forge.graph._recipes import default_graph_recipe_providers

    providers = default_graph_recipe_providers()
    descriptors = {
        provider.descriptor.namespace: provider.descriptor for provider in providers
    }
    assembly = descriptors.pop("taichi_forge.graph.runtime_assembly")
    assert assembly.owned_fragment_namespaces == ()
    assert "legacy-family-adapter" not in assembly.capabilities
    assert set(descriptors) == {
        "taichi_forge.graph.binding_frames",
        "taichi_forge.graph.bounded_execution",
        "taichi_forge.graph.branch_join_schedule",
        "taichi_forge.graph.graph_memory",
        "taichi_forge.graph.graph_reduction",
        "taichi_forge.graph.map_fusion",
        "taichi_forge.graph.native_algorithm",
        "taichi_forge.graph.offload_phase_fusion",
        "taichi_forge.graph.recording_partition",
        "taichi_forge.graph.resource_lifetime",
        "taichi_forge.graph.sparse_traversal",
        "taichi_forge.graph.structured_control",
        "taichi_forge.graph.workspace_concurrency",
    }
    assert all(
        descriptor.owned_fragment_namespaces == ()
        for descriptor in descriptors.values()
    )


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_external_runtime_family_materializes_without_central_family_routing():
    from taichi_forge.graph._recipes import (
        GraphFamilySelection,
        GraphFragmentBindingRequirement,
        GraphFragmentTask,
        GraphRecipeFragment,
        GraphRecipeProviderDescriptor,
        GraphRuntimeAssemblyProvider,
        GraphRuntimeFragmentProvider,
        RUNTIME_GRAPH_ASSEMBLY_V1,
    )

    count = 257

    @ti.kernel
    def scale(
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        temporary: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(count):
            temporary[index] = source[index] * 2.0

    @ti.kernel
    def bias(
        temporary: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(count):
            output[index] = temporary[index] + 1.0

    symbolic = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)
        for name in ("source", "temporary", "output")
    }
    builder = ti.graph.GraphBuilder(_explicit_map_source_groups=())
    builder.dispatch(scale, symbolic["source"], symbolic["temporary"])
    builder.dispatch(bias, symbolic["temporary"], symbolic["output"])
    definition = builder.freeze()

    class ExternalMapProvider(GraphRuntimeFragmentProvider):
        descriptor = GraphRecipeProviderDescriptor(
            namespace="tests.external_runtime_map",
            provider_version="1.0",
            domain_version="external-runtime-map-v1",
            semantic_fingerprint="external-runtime-map-fusion-v1",
            assembly_protocols=(RUNTIME_GRAPH_ASSEMBLY_V1,),
            capabilities=("external-map-fusion",),
            fragment_key_schema="external-map.v1",
        )

        def fragments(self, requested_definition):
            fusion = next(
                recipe
                for recipe in requested_definition._runtime_spec.fusion_plan.candidate_recipes
                if len(recipe.source_dispatch_ids) == 2
            )
            coverage = tuple(
                source.region_id
                for source in requested_definition.sources
                if source.kind == "dispatch"
            )
            selection = GraphFamilySelection(
                family="external_runtime_map",
                source_key="external-map:0,1",
                choice_id=fusion.recipe_id,
                materialization_choice=fusion.recipe_id,
                coverage_region_ids=coverage,
            )
            return (
                GraphRecipeFragment.create(
                    requested_definition,
                    provider_namespace=self.descriptor.namespace,
                    provider_version=self.descriptor.provider_version,
                    provider_domain_version=self.descriptor.domain_version,
                    fragment_key="external-map:0,1",
                    coverage_region_ids=coverage,
                    tasks=(
                        GraphFragmentTask.create(
                            "external-map:0,1",
                            "synthetic_fused_kernel",
                            physical={"source_group": (0, 1)},
                        ),
                    ),
                    binding_requirements=tuple(
                        GraphFragmentBindingRequirement(
                            item.name,
                            kinds=item.kinds,
                            required=item.required,
                            scope=item.scope,
                        )
                        for item in requested_definition.binding_abi
                    ),
                    backend_requirements=(requested_definition.backend,),
                    assembly_protocol=RUNTIME_GRAPH_ASSEMBLY_V1,
                    assembly_provider_namespace=(
                        GraphRuntimeAssemblyProvider.descriptor.namespace
                    ),
                    provider_metadata={"family_selection": selection.to_dict()},
                ),
            )

        def contribute_runtime(self, assembly, selection):
            assert selection.family == "external_runtime_map"
            assembly.add_map_source_group((0, 1))

    providers = (GraphRuntimeAssemblyProvider(), ExternalMapProvider())
    catalog = definition.recipe_catalog(providers=providers)
    recipe = catalog.entries(stage="single-region")[0].recipe
    assert catalog.resolve(recipe.recipe_id) == recipe

    source = ti.ndarray(ti.f32, shape=count)
    temporary = ti.ndarray(ti.f32, shape=count)
    output = ti.ndarray(ti.f32, shape=count)
    host = np.linspace(-1.0, 1.0, count, dtype=np.float32)
    source.from_numpy(host)
    with definition.materialization_context(
        provider_set=catalog.provider_set
    ) as context:
        with context.materialize(catalog.baseline.recipe) as baseline:
            baseline_id = baseline.materialized_physical_id
            assert len(baseline.manifest.tasks) == 2
        with context.materialize(recipe) as optimized:
            assert optimized.materialized_physical_id != baseline_id
            assert len(optimized.manifest.tasks) == 1
            optimized.executor.run(
                {"source": source, "temporary": temporary, "output": output}
            )
            ti.sync()
    np.testing.assert_allclose(output.to_numpy(), host * 2.0 + 1.0)


@test_utils.test(
    arch=ti.cuda,
    offline_cache=False,
    cuda_sparse_pool_auto_size=True,
    cuda_sparse_per_snode_pool=True,
    listgen_static_grid_dim=False,
)
def test_complete_recipe_owns_sparse_listgen_grid_for_migrating_active_set():
    block_count = 33
    block_size = 8
    domain_size = block_count * block_size
    values = ti.field(ti.i32)
    fields = ti.FieldsBuilder()
    pointer = fields.pointer(
        ti.i,
        block_count,
        vk_max_active=block_count,
    )
    pointer.bitmasked(ti.i, block_size).place(values)
    tree = fields.finalize()
    output = ti.ndarray(ti.i32, shape=2)

    @ti.kernel
    def deactivate_all():
        for block in range(block_count):
            ti.deactivate(pointer, block)

    @ti.kernel
    def activate_phase(phase: ti.i32):
        for index in range(domain_size):
            block = index // block_size
            local = index % block_size
            if block % 3 == phase and (local + phase) % 4 == 1:
                values[index] = 1000 * (phase + 1) + index

    @ti.kernel
    def clear(result: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        result[0] = 0
        result[1] = 0

    @ti.kernel
    def reduce_active(result: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for index in values:
            ti.atomic_add(result[0], values[index])
            ti.atomic_add(result[1], 1)

    result_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY,
        "result",
        ti.i32,
        ndim=1,
    )
    builder = ti.graph.GraphBuilder()
    builder.dispatch(clear, result_arg)
    builder.dispatch(reduce_active, result_arg)
    definition = builder.freeze()

    sources = definition._runtime_spec._graph_sparse_traversal_sources
    assert len(sources) == 1
    manifests = sources[0].manifests()
    assert tuple(manifest.strategy for manifest in manifests) == (
        "saturating",
        "parent_capacity_bound",
    )
    bounded_manifest = manifests[1]
    listgens = bounded_manifest.to_dict()["listgen_tasks"]
    assert len(listgens) == 2
    assert all(task["policy"] == "parent_capacity_bound" for task in listgens)
    assert all(
        task["actual_grid"] == min(task["selected_grid"], task["parent_grid_bound"])
        for task in listgens
    )
    assert any(task["actual_grid"] < task["selected_grid"] for task in listgens)

    catalog = definition.recipe_catalog()
    sparse_fragments = _family_fragments(catalog, "sparse_traversal")
    assert len(sparse_fragments) == 1
    fragment = sparse_fragments[0]
    entry = catalog.compose(
        (fragment.fragment_id,),
        stage="sparse-traversal",
        parent_recipe_ids=(catalog.baseline.recipe.recipe_id,),
    )
    session = definition.search_recipes(
        engine="compileiq",
        target=ti.graph.GraphOptimizationTarget(
            objectives=(("device_time_ns", "min"),)
        ),
        budget=ti.graph.GraphSearchBudget(evaluation_limit=2),
    )
    assert len(session.recipes) == 2
    assert any(
        handle.manifest.families == ("sparse_traversal",) for handle in session.recipes
    )

    prog = ti.lang.impl.get_runtime().prog
    with definition.materialization_context() as context:
        product = context.materialize(entry.recipe)
        task_manifests = product.executor.task_manifest()
        physical_listgens = tuple(
            task for task in task_manifests if task.sparse_list_op == 2
        )
        assert len(physical_listgens) == 2
        assert all(
            task.requested_sparse_list_policy == "parent_capacity_bound"
            and task.actual_grid_size
            == min(task.selected_grid_size, task.sparse_list_parent_grid_bound)
            for task in physical_listgens
        )
        bindings = product.executor.bind({"result": output})
        prog._debug_reset_sparse_listgen_stats()
        for phase in (0, 1):
            deactivate_all()
            activate_phase(phase)
            expected_values = [
                1000 * (phase + 1) + index
                for index in range(domain_size)
                if index // block_size % 3 == phase
                and (index % block_size + phase) % 4 == 1
            ]
            for _ in range(4):
                product.executor.run(bindings)
            ti.sync()
            np.testing.assert_array_equal(
                output.to_numpy(),
                np.asarray((sum(expected_values), len(expected_values)), np.int32),
            )

        stats = dict(prog._debug_sparse_snode_tree_stats(tree.id))["listgen"]
        totals = dict(stats["totals"])
        assert totals["rebuilds"] >= 4
        assert totals["reuse_hits"] >= 8
        assert totals["candidate_slots_dispatched"] == 2 * sum(
            task.actual_grid_size * task.selected_block_size
            for task in physical_listgens
        )
        graph_stats = product.executor._graph_stats[0]
        assert graph_stats["captures"] == 0
        assert graph_stats["last_path"] == "ordinary_fallback"

    tree.destroy()


@test_utils.test(arch=ti.cuda, offline_cache=False, kernel_profiler=True)
def test_complete_recipe_searches_and_materializes_offload_phase_fusion():
    count = (1 << 17) + 19

    @ti.kernel
    def three_phase(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        first: ti.types.ndarray(dtype=ti.i32, ndim=1),
        second: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for index in range(count):
            first[index] = source[index] * 2 + 1
        for index in range(count):
            second[index] = first[index] * 3 - 4
        for index in range(count):
            output[index] = second[index] ^ 0x55AA

    symbolic = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=1)
        for name in ("source", "first", "second", "output")
    }
    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        three_phase,
        symbolic["source"],
        symbolic["first"],
        symbolic["second"],
        symbolic["output"],
    )
    definition = builder.freeze()
    source = definition._runtime_spec._graph_offload_fusion_sources[0]
    manifests = source.manifests()
    fused_manifest = next(
        manifest
        for manifest in manifests
        if manifest.strategy == "exact_pointwise_phase_fusion"
        and len(manifest.to_dict()["materialized_tasks"]) == 1
    )
    payload = fused_manifest.to_dict()
    assert len(payload["source_task_lineage"]) == 1
    assert len(payload["source_task_lineage"][0]) == 3
    assert payload["fusion_groups"] == [[0, 1, 2]]

    catalog = definition.recipe_catalog()
    fragment = next(
        item
        for item in _family_fragments(catalog, "offload_phase_fusion")
        if _selection(item).choice_id == fused_manifest.recipe_id
    )
    entry = catalog.compose(
        (fragment.fragment_id,),
        stage="compiler-ir-topology",
        parent_recipe_ids=(catalog.baseline.recipe.recipe_id,),
    )
    session = definition.search_recipes(
        engine="compileiq",
        target=ti.graph.GraphOptimizationTarget(
            objectives=(("device_time_ns", "min"),)
        ),
        budget=ti.graph.GraphSearchBudget(evaluation_limit=16),
    )
    assert any(
        handle.manifest.families == ("offload_phase_fusion",)
        for handle in session.recipes
    )

    with definition.materialization_context() as context:
        product = context.materialize(entry.recipe)
        task_manifest = tuple(
            task
            for task in product.executor.task_manifest()
            if task.task_type == "range_for"
        )
        assert len(task_manifest) == 1
        assert (
            task_manifest[0].optimization_spec_id
            == payload["offload_compilation_identity"]
        )

        arrays = {name: ti.ndarray(ti.i32, shape=count) for name in symbolic}
        host = np.arange(count, dtype=np.int32) - 31
        arrays["source"].from_numpy(host)
        arrays["output"].fill(0)
        bindings = product.executor.bind(arrays)
        published = product.executor.binding_statistics()
        for _ in range(4):
            product.executor.run(bindings)
        ti.sync()
        replayed = product.executor.binding_statistics()
        assert replayed["version_builds"] == published["version_builds"]
        assert replayed["raw_replay_validations"] == published["raw_replay_validations"]
        assert replayed["version_fast_replays"] == (
            published["version_fast_replays"] + 4
        )
        expected = ((host * 2 + 1) * 3 - 4) ^ 0x55AA
        np.testing.assert_array_equal(arrays["output"].to_numpy(), expected)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_complete_recipe_excludes_cross_lane_offload_phase_fusion():
    count = 4096

    @ti.kernel
    def shifted_read(
        data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for index in range(count):
            data[index] = index + 1
        for index in range(count):
            output[index] = data[(index + 1) % count]

    data = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "data", ti.i32, ndim=1)
    output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(shifted_read, data, output)
    definition = builder.freeze()
    source = definition._runtime_spec._graph_offload_fusion_sources[0]

    assert source.manifests() == ()
    assert "non-pointwise external access" in source.candidate_failure
    assert not _family_fragments(definition.recipe_catalog(), "offload_phase_fusion")


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_complete_recipe_composes_fusion_and_memory_without_environment(monkeypatch):
    count = 1027

    @ti.kernel
    def scale(
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        temporary: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(count):
            temporary[index] = source[index] * 2.0

    @ti.kernel
    def bias(
        temporary: ti.types.ndarray(dtype=ti.f32, ndim=1),
        staged_input: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(count):
            staged_input[index] = temporary[index] + 1.0

    @ti.kernel
    def stencil(
        staged_input: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(1, count - 1):
            output[index] = (
                staged_input[index - 1] + staged_input[index] + staged_input[index + 1]
            )

    symbolic = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)
        for name in ("source", "temporary", "staged_input", "output")
    }
    builder = ti.graph.GraphBuilder(_map_recipe="baseline")
    builder.dispatch(scale, symbolic["source"], symbolic["temporary"])
    builder.dispatch(bias, symbolic["temporary"], symbolic["staged_input"])
    builder.dispatch(stencil, symbolic["staged_input"], symbolic["output"])
    definition = builder.freeze()
    catalog = definition.recipe_catalog()

    fusion = next(
        fragment
        for fragment in _family_fragments(catalog, "map_fusion")
        if len(fragment.coverage_region_ids) == 2
    )
    memory_source = definition._runtime_spec._graph_memory_sources[2]
    staged_id = next(
        manifest.recipe_id
        for manifest in memory_source.manifests()
        if manifest.strategy == "shared_staged_1d"
    )
    memory = next(
        fragment
        for fragment in _family_fragments(catalog, "graph_memory")
        if _selection(fragment).choice_id == staged_id
    )
    from taichi_forge.graph._recipes import (
        GraphMemoryRecipeProvider,
        GraphRuntimeAssemblyProvider,
    )

    assert isinstance(
        catalog.provider_set.provider_for_fragment_namespace(memory.provider_namespace),
        GraphMemoryRecipeProvider,
    )
    assert memory.provider_domain_version == "graph-memory-domain-v1"
    assert memory.assembly_provider_namespace == (
        GraphRuntimeAssemblyProvider.descriptor.namespace
    )
    assert fusion.assembly_provider_namespace == memory.assembly_provider_namespace
    assert not set(fusion.coverage_region_ids).intersection(memory.coverage_region_ids)
    entry = catalog.compose(
        (fusion.fragment_id, memory.fragment_id),
        stage="compatible-composition",
        parent_recipe_ids=(catalog.baseline.recipe.recipe_id,),
    )
    public_session = definition.search_recipes(
        engine="compileiq",
        target=ti.graph.GraphOptimizationTarget(
            objectives=(("device_time_ns", "min"),)
        ),
        budget=ti.graph.GraphSearchBudget(evaluation_limit=64),
    )
    assert any(
        set(handle.manifest.families) == {"map_fusion", "graph_memory"}
        and handle.manifest.selected_fragment_count == 2
        for handle in public_session.recipes
    )

    for name in _MATERIALIZATION_ENVIRONMENTS:
        monkeypatch.setenv(name, "intentionally-invalid-after-freeze")
    with definition.materialization_context() as context:
        materialized = context.materialize(entry.recipe)
        graph = materialized.executor
        assert graph.definition is definition
        assert materialized.manifest.recipe_id == entry.recipe.recipe_id
        task_manifest = graph.task_manifest()
        assert any(task.source_dispatch_count == 2 for task in task_manifest)
        assert any(
            task.requested_memory_strategy == "shared_staged_1d"
            for task in task_manifest
        )

        source = ti.ndarray(ti.f32, shape=count)
        temporary = ti.ndarray(ti.f32, shape=count)
        staged_input = ti.ndarray(ti.f32, shape=count)
        output = ti.ndarray(ti.f32, shape=count)
        values = np.arange(count, dtype=np.float32) * 0.25
        source.from_numpy(values)
        output.fill(0)
        graph.run(
            graph.bind(
                {
                    "source": source,
                    "temporary": temporary,
                    "staged_input": staged_input,
                    "output": output,
                }
            )
        )
        ti.sync()
        intermediate = values * 2.0 + 1.0
        expected = np.zeros(count, dtype=np.float32)
        expected[1:-1] = intermediate[:-2] + intermediate[1:-1] + intermediate[2:]
        np.testing.assert_allclose(output.to_numpy(), expected, rtol=0, atol=0)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_complete_reduction_recipe_owns_workspace_and_survives_bad_environment(
    monkeypatch,
):
    count = 4097
    values_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.reduce(values_arg, output_arg, count=count)
    definition = builder.freeze()
    catalog = definition.recipe_catalog()
    source = definition._runtime_spec._graph_reduction_sources[0]
    phased_id = next(
        manifest.recipe_id
        for manifest in source.manifests()
        if manifest.strategy == "block_partial_finalize"
    )
    fragment = next(
        item
        for item in _family_fragments(catalog, "graph_reduction")
        if _selection(item).choice_id == phased_id
    )
    recipe = catalog.compose(
        (fragment.fragment_id,),
        stage="single-region",
        parent_recipe_ids=(catalog.baseline.recipe.recipe_id,),
    ).recipe
    assert recipe.declared_persistent_resource_bytes == ((count + 1023) // 1024) * 4

    for name in _MATERIALIZATION_ENVIRONMENTS:
        monkeypatch.setenv(name, "intentionally-invalid-after-freeze")
    with definition.materialization_context() as context:
        materialized = context.materialize(recipe)
        graph = materialized.executor
        assert graph.execution_stats().memory.persistent_bytes >= (
            recipe.declared_persistent_resource_bytes
        )
        values = ti.ndarray(ti.i32, shape=count)
        output = ti.ndarray(ti.i32, shape=1)
        host = ((np.arange(count, dtype=np.int32) % 23) - 11).astype(np.int32)
        values.from_numpy(host)
        graph.run(graph.bind({"values": values, "output": output}))
        ti.sync()
        expected = np.asarray(host.sum(dtype=np.int64), dtype=np.int32).item()
        assert output.to_numpy()[0] == expected


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_complete_native_algorithm_recipe_materializes_all_physical_routes(
    monkeypatch,
):
    capacity = 8192
    offsets = np.arange(0, capacity + 1, 2048, dtype=np.int32)
    layout = ti.algorithms.SegmentedLayout.from_offsets(
        offsets,
        capacity=capacity,
    )
    values = ti.ndarray(ti.i32, shape=capacity)
    output = ti.ndarray(ti.i32, shape=capacity)
    builder = ti.graph.GraphBuilder()
    builder.segmented_scan(values, layout, output)
    definition = builder.freeze()
    catalog = definition.recipe_catalog()
    fragments = _family_fragments(catalog, "native_algorithm")
    assert len(fragments) == 5  # Includes register-shuffle and hierarchical carry.

    host = ((np.arange(capacity, dtype=np.int64) % 7) + 1).astype(np.int32)
    expected = np.empty_like(host)
    for begin, end in pairwise(offsets):
        expected[begin:end] = np.cumsum(host[begin:end], dtype=np.int32)

    for name in _MATERIALIZATION_ENVIRONMENTS:
        monkeypatch.setenv(name, "intentionally-invalid-after-freeze")
    physical_identities = set()
    persistent_bytes = set()
    with definition.materialization_context() as context:
        recipes = [catalog.baseline.recipe]
        recipes.extend(
            catalog.compose(
                (fragment.fragment_id,),
                stage="single-region",
                parent_recipe_ids=(catalog.baseline.recipe.recipe_id,),
            ).recipe
            for fragment in fragments
        )
        for recipe in recipes:
            materialized = context.materialize(recipe)
            values.from_numpy(host)
            output.fill(0)
            materialized.executor.run({})
            ti.sync()
            np.testing.assert_array_equal(output.to_numpy(), expected)
            physical_identities.add(materialized.manifest.materialized_physical_id)
            persistent_bytes.add(materialized.manifest.persistent_requested_bytes)
    assert len(physical_identities) == 1 + len(fragments)
    # The retained global recipe owns exact hierarchical scratch too, instead
    # of borrowing the mutable Program scan arena.
    assert persistent_bytes == {0, layout.num_segments * 4 + (capacity // 1024) * 4}


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_provider_owned_whole_graph_rebuilds_native_and_synthetic_fused_routes(
    monkeypatch,
):
    from taichi_forge.graph._recipes import (
        GraphFamilySelection,
        GraphMaterializationProduct,
        GraphMaterializedFragment,
        GraphMapFusionRecipeProvider,
        GraphRecipeProviderDescriptor,
        GraphRuntimeRecipeAssembly,
        PROVIDER_OWNED_WHOLE_GRAPH_V1,
    )

    count = 1024

    @ti.kernel
    def scale(
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        temporary: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(count):
            temporary[index] = source[index] * 2.0

    @ti.kernel
    def bias(
        temporary: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(count):
            output[index] = temporary[index] + 1.0

    @ti.kernel
    def stencil(
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
        final: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(1, count - 1):
            final[index] = output[index - 1] + output[index] + output[index + 1]

    symbolic = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)
        for name in ("source", "temporary", "output", "final")
    }
    builder = ti.graph.GraphBuilder(_map_recipe="baseline")
    builder.dispatch(scale, symbolic["source"], symbolic["temporary"])
    builder.dispatch(bias, symbolic["temporary"], symbolic["output"])
    builder.dispatch(stencil, symbolic["output"], symbolic["final"])
    definition = builder.freeze()

    class WholeGraphProvider:
        descriptor = GraphRecipeProviderDescriptor(
            namespace="tests.whole_graph_native_fusion",
            provider_version="1.0",
            domain_version="native-fusion-domain-v1",
            semantic_fingerprint="native-fusion-reconstruction-v1",
            assembly_protocols=(PROVIDER_OWNED_WHOLE_GRAPH_V1,),
            capabilities=("native-kernels", "synthetic-fused-kernel"),
            fragment_key_schema="whole-graph-route.v1",
        )

        def __init__(self):
            self.retired_routes = []

        def _fragment(self, requested_definition, route):
            all_regions = tuple(
                region.region_id for region in requested_definition.regions
            )
            if route == "native-and-direct-kernels":
                physical_tasks = (
                    ("direct-scale", "kernel", ()),
                    ("direct-bias", "kernel", ("direct-scale",)),
                    ("direct-stencil", "kernel", ("direct-bias",)),
                )
            elif route == "native-and-synthetic-fusion":
                physical_tasks = (
                    ("synthetic-map", "synthetic_fused_kernel", ()),
                    ("direct-stencil", "kernel", ("synthetic-map",)),
                )
            elif route == "assembly-failure":
                physical_tasks = (("failed-build", "provider_build", ()),)
            else:
                raise KeyError(route)
            tasks = tuple(
                ti.graph.GraphFragmentTask.create(
                    task_id,
                    kind,
                    depends_on=depends_on,
                    physical={"route": route, "kind": kind},
                )
                for task_id, kind, depends_on in physical_tasks
            )
            bindings = tuple(
                ti.graph.GraphFragmentBindingRequirement(
                    item.name,
                    kinds=item.kinds,
                    required=item.required,
                    scope=item.scope,
                )
                for item in requested_definition.binding_abi
            )
            return ti.graph.GraphRecipeFragment.create(
                requested_definition,
                provider_namespace=self.descriptor.namespace,
                provider_version=self.descriptor.provider_version,
                provider_domain_version=self.descriptor.domain_version,
                fragment_key=route,
                coverage_region_ids=all_regions,
                tasks=tasks,
                binding_requirements=bindings,
                backend_requirements=(requested_definition.backend,),
                assembly_protocol=PROVIDER_OWNED_WHOLE_GRAPH_V1,
                provider_metadata={"route": route},
            )

        def discover(self, requested_definition):
            return tuple(
                self._fragment(requested_definition, route)
                for route in (
                    "native-and-direct-kernels",
                    "native-and-synthetic-fusion",
                    "assembly-failure",
                )
            )

        def resolve(self, requested_definition, fragment_key):
            return self._fragment(requested_definition, fragment_key)

        def expand(self, requested_definition, fragment_key):
            self.resolve(requested_definition, fragment_key)
            return ()

        def materialize(self, scope, fragment):
            route = fragment.provider_metadata["route"]
            scope.own(
                route,
                release=self.retired_routes.append,
                label=f"provider route {route}",
            )
            return GraphMaterializedFragment.create(fragment, route)

        @staticmethod
        def _optimized_assembly(requested_definition):
            spec = requested_definition._runtime_spec
            fusion = next(
                recipe
                for recipe in spec.fusion_plan.candidate_recipes
                if len(recipe.source_dispatch_ids) == 2
            )
            dispatch_indices = tuple(
                int(source_id.rsplit("/dispatch:", 1)[1])
                for source_id in fusion.source_dispatch_ids
            )
            dispatch_regions = tuple(
                source.region_id
                for source in requested_definition.sources
                if source.kind == "dispatch"
            )
            fusion_selection = GraphFamilySelection(
                family="map_fusion",
                source_key="dispatches:" + ",".join(map(str, dispatch_indices)),
                choice_id=fusion.recipe_id,
                materialization_choice=fusion.recipe_id,
                coverage_region_ids=tuple(
                    dispatch_regions[index] for index in dispatch_indices
                ),
            )
            assembly = GraphRuntimeRecipeAssembly(requested_definition)
            GraphMapFusionRecipeProvider().contribute_runtime(
                assembly,
                fusion_selection,
            )
            return assembly

        def assemble(self, scope, requested_definition, recipe, fragments):
            assert len(fragments) == 1
            route = fragments[0].payload
            if route == "assembly-failure":
                raise RuntimeError("injected whole-Graph assembly failure")
            assembly = (
                GraphRuntimeRecipeAssembly(requested_definition)
                if route == "native-and-direct-kernels"
                else self._optimized_assembly(requested_definition)
            )
            graph = requested_definition._runtime_spec.materialize_complete_recipe(
                requested_definition,
                recipe,
                assembly,
                workspace_lanes=scope._context.workspace_lanes,
                workspace_saturation=scope._context.workspace_saturation,
            )
            manifest = observe_graph_physical_manifest(
                requested_definition,
                recipe,
                graph,
            )
            return GraphMaterializationProduct(graph, manifest)

        def describe(self, requested_definition, fragment_key):
            self.resolve(requested_definition, fragment_key)
            return {"route": fragment_key}

    provider = WholeGraphProvider()
    catalog = definition.recipe_catalog(providers=(provider,))
    by_route = {
        entry.recipe.fragments[0].provider_metadata["route"]: entry.recipe
        for entry in catalog.entries(stage="single-region")
    }
    assert set(by_route) == {
        "native-and-direct-kernels",
        "native-and-synthetic-fusion",
        "assembly-failure",
    }
    assert all(not recipe.baseline_coverage_region_ids for recipe in by_route.values())
    assert (
        len(
            {
                by_route[route].planned_physical_id
                for route in (
                    "native-and-direct-kernels",
                    "native-and-synthetic-fusion",
                )
            }
        )
        == 2
    )

    source = ti.ndarray(ti.f32, shape=count)
    temporary = ti.ndarray(ti.f32, shape=count)
    output = ti.ndarray(ti.f32, shape=count)
    final = ti.ndarray(ti.f32, shape=count)
    map_values = np.arange(count, dtype=np.float32) * 0.25
    map_intermediate = map_values * 2.0 + 1.0
    map_expected = np.zeros(count, dtype=np.float32)
    map_expected[1:-1] = (
        map_intermediate[:-2] + map_intermediate[1:-1] + map_intermediate[2:]
    )

    materialized_ids = set()
    memory_facts = {}
    with definition.materialization_context(
        provider_set=catalog.provider_set
    ) as context:
        with pytest.raises(RuntimeError, match="injected whole-Graph"):
            context.materialize(by_route["assembly-failure"])
        assert provider.retired_routes == ["assembly-failure"]
        assert context.statistics()["state"] == "open"

        for route in (
            "native-and-direct-kernels",
            "native-and-synthetic-fusion",
        ):
            source.from_numpy(map_values)
            temporary.fill(0)
            output.fill(0)
            final.fill(0)
            with context.materialize(by_route[route]) as materialized:
                bindings = materialized.executor.bind(
                    {
                        "source": source,
                        "temporary": temporary,
                        "output": output,
                        "final": final,
                    }
                )
                before = materialized.executor.binding_statistics()
                for _ in range(4):
                    materialized.executor.run(bindings)
                ti.sync()
                after = materialized.executor.binding_statistics()
                assert after["raw_replay_validations"] == (
                    before["raw_replay_validations"]
                )
                assert after["version_fast_replays"] == (
                    before["version_fast_replays"] + 4
                )
                np.testing.assert_array_equal(temporary.to_numpy(), map_values * 2.0)
                np.testing.assert_array_equal(output.to_numpy(), map_values * 2.0 + 1.0)
                np.testing.assert_array_equal(final.to_numpy(), map_expected)
                materialized_ids.add(materialized.materialized_physical_id)
                memory_facts[route] = (
                    materialized.manifest.persistent_requested_bytes,
                    materialized.manifest.transient_requested_bytes,
                )
                if route == "native-and-synthetic-fusion":
                    assert any(
                        task.properties.get("source_dispatch_count") == 2
                        for task in materialized.manifest.tasks
                    )

    assert len(materialized_ids) == 2
    assert set(memory_facts) == {
        "native-and-direct-kernels",
        "native-and-synthetic-fusion",
    }
    assert set(provider.retired_routes) == {
        "assembly-failure",
        "native-and-direct-kernels",
        "native-and-synthetic-fusion",
    }


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_complete_bounded_recipe_replays_all_scope_strategies_without_environment(
    monkeypatch,
):
    probe = dict(ti_core.cuda_bounded_dispatch_probe())
    if not probe["exact_device_grid_available"]:
        pytest.skip(probe["unavailable_reason"])

    capacity = 257
    block_dim = 32

    @ti.kernel
    def publish(
        requested: ti.i32,
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.device_extent_publish(extent, capacity, requested)

    @ti.kernel
    def consume(
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
        observed: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.loop_config(block_dim=block_dim)
        for index in range(capacity):
            if index < ti.device_extent_count(extent):
                ti.atomic_add(observed[0], index + 1)

    requested_arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "requested", ti.i32)
    extent_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "extent", ti.i32, ndim=1)
    first_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "first", ti.i32, ndim=1)
    second_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "second", ti.i32, ndim=1)

    monkeypatch.setenv("TI_CUDA_BOUNDED_DISPATCH_MODE", "auto")
    monkeypatch.setenv("TI_GRAPH_CUDA_BOUNDED_UPDATE_POLICY", "auto")
    builder = ti.graph.GraphBuilder()
    builder.dispatch(publish, requested_arg, extent_arg)
    for output_arg in (first_arg, second_arg):
        builder.dispatch_bounded(
            consume,
            extent_arg,
            output_arg,
            extent=extent_arg,
            capacity=capacity,
            block_dim=block_dim,
        )
    definition = builder.freeze()
    catalog = definition.recipe_catalog()
    fragments = _family_fragments(catalog, "bounded_execution")
    expected_strategies = {
        "logical_exact",
        "adaptive_per_node",
        "adaptive_grouped",
        "masked_capacity",
    }
    assert {_selection(item).materialization_choice for item in fragments} == (
        expected_strategies - {"logical_exact"}
    )

    for name in _MATERIALIZATION_ENVIRONMENTS:
        monkeypatch.setenv(name, "intentionally-invalid-after-freeze")
    extent = ti.DeviceExtent(capacity)
    first = ti.ndarray(ti.i32, shape=1)
    second = ti.ndarray(ti.i32, shape=1)
    arguments = {
        "requested": 17,
        "extent": extent,
        "first": first,
        "second": second,
    }
    physical_identities = {}
    persistent_control_bytes = {}
    manifest_persistent_bytes = {}
    execution_persistent_bytes = {}
    bounded_resource_bytes = {}
    with definition.materialization_context() as context:
        recipes = [("logical_exact", catalog.baseline.recipe)]
        recipes.extend(
            (
                _selection(fragment).materialization_choice,
                catalog.compose(
                    (fragment.fragment_id,),
                    stage="single-region",
                    parent_recipe_ids=(catalog.baseline.recipe.recipe_id,),
                ).recipe,
            )
            for fragment in fragments
        )
        for strategy, recipe in recipes:
            materialized = context.materialize(recipe)
            first.fill(0)
            second.fill(0)
            materialized.executor.run(arguments)
            ti.sync()
            observed_manifest = observe_graph_physical_manifest(
                definition,
                recipe,
                materialized.executor,
            )
            expected = arguments["requested"] * (arguments["requested"] + 1) // 2
            assert int(first.to_numpy()[0]) == expected
            assert int(second.to_numpy()[0]) == expected
            physical_identities[strategy] = observed_manifest.materialized_physical_id
            memory = materialized.executor.execution_stats().memory
            persistent_control_bytes[strategy] = memory.persistent_bounded_control_bytes
            execution_persistent_bytes[strategy] = memory.persistent_bytes
            manifest_persistent_bytes[strategy] = (
                observed_manifest.persistent_requested_bytes
            )
            bounded_resource_bytes[strategy] = sum(
                resource.requested_bytes
                for resource in observed_manifest.resources
                if resource.kind == "bounded_control_state"
            )

    assert len(set(physical_identities.values())) == len(
        expected_strategies
    ), physical_identities
    assert persistent_control_bytes["logical_exact"] == 0
    assert persistent_control_bytes["masked_capacity"] == 0
    assert persistent_control_bytes["adaptive_per_node"] > 0
    assert persistent_control_bytes["adaptive_grouped"] > 0
    assert manifest_persistent_bytes == execution_persistent_bytes
    assert bounded_resource_bytes == persistent_control_bytes


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_complete_bounded_recipe_composes_independent_publication_groups(monkeypatch):
    probe = dict(ti_core.cuda_bounded_dispatch_probe())
    if not probe["exact_device_grid_available"]:
        pytest.skip(probe["unavailable_reason"])

    capacity = 257

    @ti.kernel
    def publish(
        requested: ti.i32,
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.device_extent_publish(extent, capacity, requested)

    @ti.kernel
    def consume(
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
        observed: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.loop_config(block_dim=32)
        for index in range(capacity):
            if index < ti.device_extent_count(extent):
                ti.atomic_add(observed[0], index + 1)

    def ndarray_arg(name):
        return ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=1)

    requested_a = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "requested_a", ti.i32)
    requested_b = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "requested_b", ti.i32)
    extent_a_arg = ndarray_arg("extent_a")
    extent_b_arg = ndarray_arg("extent_b")
    output_a_arg = ndarray_arg("output_a")
    output_b_arg = ndarray_arg("output_b")
    monkeypatch.setenv("TI_CUDA_BOUNDED_DISPATCH_MODE", "auto")
    monkeypatch.setenv("TI_GRAPH_CUDA_BOUNDED_UPDATE_POLICY", "auto")
    builder = ti.graph.GraphBuilder()
    builder.dispatch(publish, requested_a, extent_a_arg)
    builder.dispatch_bounded(
        consume,
        extent_a_arg,
        output_a_arg,
        extent=extent_a_arg,
        capacity=capacity,
        block_dim=32,
    )
    builder.dispatch(publish, requested_b, extent_b_arg)
    builder.dispatch_bounded(
        consume,
        extent_b_arg,
        output_b_arg,
        extent=extent_b_arg,
        capacity=capacity,
        block_dim=32,
    )
    definition = builder.freeze()
    catalog = definition.recipe_catalog()
    fragments = _family_fragments(catalog, "bounded_execution")
    by_source = {}
    for fragment in fragments:
        selection = _selection(fragment)
        by_source.setdefault(selection.source_key, {})[
            selection.materialization_choice
        ] = fragment
    assert len(by_source) == 2
    assert all("masked_capacity" in choices for choices in by_source.values())
    masked = tuple(
        choices["masked_capacity"].fragment_id for choices in by_source.values()
    )
    entry = catalog.compose(
        masked,
        stage="compatible-composition",
        parent_recipe_ids=(catalog.baseline.recipe.recipe_id,),
    )
    assert len(entry.recipe.fragments) == 2

    extent_a = ti.DeviceExtent(capacity)
    extent_b = ti.DeviceExtent(capacity)
    output_a = ti.ndarray(ti.i32, shape=1)
    output_b = ti.ndarray(ti.i32, shape=1)
    arguments = {
        "requested_a": 17,
        "requested_b": 29,
        "extent_a": extent_a,
        "extent_b": extent_b,
        "output_a": output_a,
        "output_b": output_b,
    }
    for output in (output_a, output_b):
        output.fill(0)
    with definition.materialization_context() as context:
        with context.materialize(entry.recipe) as materialized:
            materialized.executor.run(arguments)
            ti.sync()
            assert int(output_a.to_numpy()[0]) == 17 * 18 // 2
            assert int(output_b.to_numpy()[0]) == 29 * 30 // 2
            bounded = materialized.executor.execution_stats().memory
            assert bounded.persistent_bounded_control_bytes == 0


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_mixed_map_control_recipes_own_independent_control_state(monkeypatch):
    capabilities = dict(ti_core.cuda_conditional_graph_capabilities())
    if not capabilities.get("general_graph_exact_control_available", False):
        pytest.skip("general CUDA conditional Graph is unavailable")

    @ti.kernel
    def advance(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in range(33):
            output[i] = source[i] + 1

    @ti.kernel
    def condition(
        state: ti.types.ndarray(dtype=ti.i32, ndim=0),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        target: ti.i32,
    ):
        predicate[None] = int(state[None] < target)

    @ti.kernel
    def step(
        state: ti.types.ndarray(dtype=ti.i32, ndim=0),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        if predicate[None] != 0:
            state[None] += 1
            counter[None] += 1

    arrays = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=1)
        for name in ("source", "temporary", "output")
    }
    scalars = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=0)
        for name in ("state", "predicate", "counter")
    }
    target = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "target", ti.i32)
    monkeypatch.delenv("TI_GRAPH_CUDA_FORCE_MASKED_CONTROL", raising=False)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(advance, arrays["source"], arrays["temporary"])
    builder.dispatch(advance, arrays["temporary"], arrays["output"])
    condition_region = builder.create_sequential()
    condition_region.dispatch(condition, scalars["state"], scalars["predicate"], target)
    body = builder.create_sequential()
    body.dispatch(step, scalars["state"], scalars["predicate"], scalars["counter"])
    builder.while_loop(
        condition_region,
        body,
        predicate=scalars["predicate"],
        control_inputs=(scalars["state"], target),
        carried_state=(scalars["state"],),
        counter=scalars["counter"],
        max_iterations=8,
        name="mixed_control",
    )
    definition = builder.freeze()
    catalog = definition.recipe_catalog()
    maps = _family_fragments(catalog, "map_fusion")
    assert maps, "the regression requires a real map-fusion axis"
    fused = catalog.compose(
        (maps[0].fragment_id,),
        stage="single-region",
        parent_recipe_ids=(catalog.baseline.recipe.recipe_id,),
    ).recipe
    recipes = [
        (catalog.baseline.recipe, "cuda_conditional_graph"),
        (fused, "cuda_conditional_graph"),
    ]
    controls = _family_fragments(catalog, "structured_control")
    if controls:
        combined = catalog.compose(
            (maps[0].fragment_id, controls[0].fragment_id),
            stage="compatible-composition",
            parent_recipe_ids=(fused.recipe_id,),
        ).recipe
        recipes.append((combined, "cuda_masked_bounded_graph"))
    # A recipe's frozen control route must not inherit later process overrides.
    monkeypatch.setenv("TI_GRAPH_CUDA_FORCE_MASKED_CONTROL", "1")
    owners = []
    with definition.materialization_context() as left, definition.materialization_context() as right:
        materializations = [
            (context.materialize(recipe), route)
            for context in (left, right)
            for recipe, route in recipes
        ]
        for index, (handle, expected_route) in enumerate(materializations):
            arguments = {name: ti.ndarray(ti.i32, shape=33) for name in arrays}
            arguments.update({name: ti.ndarray(ti.i32, shape=()) for name in scalars})
            arguments["source"].from_numpy(np.arange(33, dtype=np.int32) + index)
            for name in scalars:
                arguments[name].fill(0)
            arguments["target"] = index + 2
            handle.executor.run(arguments)
            report = handle.executor.control_flow_stats()[0]
            assert report.logical_iterations == index + 2
            assert report.lowering == expected_route
            np.testing.assert_array_equal(
                arguments["output"].to_numpy(),
                np.arange(33, dtype=np.int32) + index + 2,
            )
            assert arguments["state"].to_numpy()[()] == index + 2
            owners.append(handle.executor._spec.structured_control_nodes[0])
        assert len({id(owner) for owner in owners}) == len(materializations)
        assert [owner._last_report.logical_iterations for owner in owners] == list(
            range(2, len(materializations) + 2)
        )


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_complete_structured_control_recipe_rebuilds_both_routes_without_environment(
    monkeypatch,
):
    capabilities = dict(ti_core.cuda_conditional_graph_capabilities())
    if not capabilities.get("general_graph_exact_control_available", False):
        pytest.skip("general CUDA conditional Graph is unavailable")
    if not capabilities.get("internal_masked_graph_available", False):
        pytest.skip("internal masked CUDA Graph control is unavailable")

    @ti.kernel
    def initialize(
        state: ti.types.ndarray(dtype=ti.i32, ndim=0),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        state[None] = 0
        predicate[None] = 0
        counter[None] = 0

    @ti.kernel
    def condition(
        state: ti.types.ndarray(dtype=ti.i32, ndim=0),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        target: ti.i32,
    ):
        predicate[None] = int(state[None] < target)

    @ti.kernel
    def step(
        state: ti.types.ndarray(dtype=ti.i32, ndim=0),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        if predicate[None] != 0:
            state[None] += 1
            counter[None] += 1

    def scalar(name):
        return ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=0)

    state = scalar("state")
    predicate = scalar("predicate")
    counter = scalar("counter")
    target = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "target", ti.i32)
    monkeypatch.delenv("TI_GRAPH_CUDA_FORCE_MASKED_CONTROL", raising=False)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(initialize, state, predicate, counter)
    condition_region = builder.create_sequential()
    condition_region.dispatch(condition, state, predicate, target)
    body = builder.create_sequential()
    body.dispatch(step, state, predicate, counter)
    builder.while_loop(
        condition_region,
        body,
        predicate=predicate,
        control_inputs=(state, target),
        carried_state=(state,),
        counter=counter,
        max_iterations=8,
        name="complete_recipe_control",
    )
    definition = builder.freeze()
    catalog = definition.recipe_catalog()
    fragments = _family_fragments(catalog, "structured_control")
    assert len(fragments) == 1, tuple(
        (region.kind, region.path, region.parent_region_id)
        for region in definition.regions
    )

    for name in _MATERIALIZATION_ENVIRONMENTS:
        monkeypatch.setenv(name, "intentionally-invalid-after-freeze")
    physical_identities = set()
    routes = set()
    with definition.materialization_context() as context:
        recipes = [catalog.baseline.recipe]
        recipes.extend(
            catalog.compose(
                (fragment.fragment_id,),
                stage="single-region",
                parent_recipe_ids=(catalog.baseline.recipe.recipe_id,),
            ).recipe
            for fragment in fragments
        )
        for recipe in recipes:
            materialized = context.materialize(recipe)
            arguments = {
                "state": ti.ndarray(ti.i32, shape=()),
                "predicate": ti.ndarray(ti.i32, shape=()),
                "counter": ti.ndarray(ti.i32, shape=()),
                "target": 5,
            }
            materialized.executor.run(arguments)
            report = materialized.executor.control_flow_stats()[0]
            assert report.logical_iterations == 5
            assert arguments["state"].to_numpy()[()] == 5
            assert arguments["counter"].to_numpy()[()] == 5
            routes.add(report.lowering)
            physical_identities.add(materialized.manifest.materialized_physical_id)

    assert routes == {"cuda_conditional_graph", "cuda_masked_bounded_graph"}
    assert len(physical_identities) == 2


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_complete_structured_control_composes_independent_top_level_domains(
    monkeypatch,
):
    capabilities = dict(ti_core.cuda_conditional_graph_capabilities())
    if not capabilities.get("general_graph_exact_control_available", False):
        pytest.skip("general CUDA conditional Graph is unavailable")
    if not capabilities.get("internal_masked_graph_available", False):
        pytest.skip("internal masked CUDA Graph control is unavailable")

    @ti.kernel
    def condition(
        state: ti.types.ndarray(dtype=ti.i32, ndim=0),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        target: ti.i32,
    ):
        predicate[None] = int(state[None] < target)

    @ti.kernel
    def step(
        state: ti.types.ndarray(dtype=ti.i32, ndim=0),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        if predicate[None] != 0:
            state[None] += 1
            counter[None] += 1

    def scalar(name):
        return ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=0)

    monkeypatch.delenv("TI_GRAPH_CUDA_FORCE_MASKED_CONTROL", raising=False)
    builder = ti.graph.GraphBuilder()
    symbols = []
    for suffix in ("a", "b"):
        state = scalar(f"state_{suffix}")
        predicate = scalar(f"predicate_{suffix}")
        counter = scalar(f"counter_{suffix}")
        target = ti.graph.Arg(
            ti.graph.ArgKind.SCALAR,
            f"target_{suffix}",
            ti.i32,
        )
        condition_region = builder.create_sequential()
        condition_region.dispatch(condition, state, predicate, target)
        body = builder.create_sequential()
        body.dispatch(step, state, predicate, counter)
        builder.while_loop(
            condition_region,
            body,
            predicate=predicate,
            control_inputs=(state, target),
            carried_state=(state,),
            counter=counter,
            max_iterations=8,
            name=f"complete_recipe_control_{suffix}",
        )
        symbols.append((suffix, state, predicate, counter, target))

    definition = builder.freeze()
    catalog = definition.recipe_catalog()
    fragments = _family_fragments(catalog, "structured_control")
    by_source = {}
    for fragment in fragments:
        selection = _selection(fragment)
        by_source.setdefault(selection.source_key, {})[selection.choice_id] = fragment
    assert len(by_source) == 2
    assert all(len(choices) == 1 for choices in by_source.values())
    masked_fragment_ids = tuple(
        next(iter(choices.values())).fragment_id for choices in by_source.values()
    )
    masked_entry = catalog.compose(
        masked_fragment_ids,
        stage="compatible-composition",
        parent_recipe_ids=(catalog.baseline.recipe.recipe_id,),
    )

    physical_ids = set()
    with definition.materialization_context() as context:
        for recipe, expected_route in (
            (catalog.baseline.recipe, "cuda_conditional_graph"),
            (masked_entry.recipe, "cuda_masked_bounded_graph"),
        ):
            arguments = {}
            for index, (suffix, *_symbols) in enumerate(symbols):
                state = ti.ndarray(ti.i32, shape=())
                predicate = ti.ndarray(ti.i32, shape=())
                counter = ti.ndarray(ti.i32, shape=())
                state.fill(0)
                predicate.fill(0)
                counter.fill(0)
                arguments.update(
                    {
                        f"state_{suffix}": state,
                        f"predicate_{suffix}": predicate,
                        f"counter_{suffix}": counter,
                        f"target_{suffix}": 3 + index * 2,
                    }
                )
            with context.materialize(recipe) as materialized:
                materialized.executor.run(arguments)
                reports = materialized.executor.control_flow_stats()
                assert len(reports) == 2
                assert {report.lowering for report in reports} == {expected_route}
                assert [report.logical_iterations for report in reports] == [3, 5]
                assert arguments["state_a"].to_numpy()[()] == 3
                assert arguments["state_b"].to_numpy()[()] == 5
                physical_ids.add(materialized.materialized_physical_id)
    assert len(physical_ids) == 2


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_complete_recipe_materializes_coarse_cuda_branch_join_dag():
    count = 257

    @ti.kernel
    def advance_a(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        target: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for index in range(count):
            target[index] = source[index] + 1

    @ti.kernel
    def advance_b(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        target: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for index in range(count):
            target[index] = source[index] - 2

    @ti.kernel
    def join(
        lhs: ti.types.ndarray(dtype=ti.i32, ndim=1),
        rhs: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for index in range(count):
            output[index] = lhs[index] + rhs[index]

    symbols = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=1)
        for name in ("a0", "a1", "b0", "b1", "output")
    }
    builder = ti.graph.GraphBuilder()
    for stage in range(4):
        builder.dispatch(
            advance_a,
            symbols["a0" if stage % 2 == 0 else "a1"],
            symbols["a1" if stage % 2 == 0 else "a0"],
        )
    for stage in range(4):
        builder.dispatch(
            advance_b,
            symbols["b0" if stage % 2 == 0 else "b1"],
            symbols["b1" if stage % 2 == 0 else "b0"],
        )
    builder.dispatch(join, symbols["a0"], symbols["b0"], symbols["output"])
    definition = builder.freeze()

    sources = definition._runtime_spec._graph_branch_join_sources
    assert len(sources) == 1
    assert sources[0].branch_groups == ((0, 1, 2, 3), (4, 5, 6, 7))
    assert sources[0].join_index == 8
    assert sources[0].parallel_temporary_bytes == (
        sources[0].sequential_temporary_bytes
    )

    catalog = definition.recipe_catalog()
    fragments = _family_fragments(catalog, "branch_join_schedule")
    assert len(fragments) == 1
    from taichi_forge.graph._recipes import (
        GraphBranchJoinRecipeProvider,
        GraphRuntimeAssemblyProvider,
    )

    assert isinstance(
        catalog.provider_set.provider_for_fragment_namespace(
            fragments[0].provider_namespace
        ),
        GraphBranchJoinRecipeProvider,
    )
    assert fragments[0].provider_domain_version == "branch-join-domain-v1"
    assert fragments[0].assembly_provider_namespace == (
        GraphRuntimeAssemblyProvider.descriptor.namespace
    )
    assert tuple(task.physical["queue"] for task in fragments[0].tasks) == (
        *("cuda_branch:0",) * 4,
        *("cuda_branch:1",) * 4,
        "default",
    )
    entry = catalog.compose(
        (fragments[0].fragment_id,),
        stage="coarse-cuda-branch-join",
        parent_recipe_ids=(catalog.baseline.recipe.recipe_id,),
    )

    with definition.materialization_context() as context:
        with context.materialize(catalog.baseline.recipe) as baseline:
            baseline_physical_id = baseline.materialized_physical_id
        with context.materialize(entry.recipe) as product:
            assert product.materialized_physical_id != baseline_physical_id
            tasks = product.manifest.tasks
            assert tuple(task.queue for task in tasks) == (
                *("cuda_branch:0",) * 4,
                *("cuda_branch:1",) * 4,
                "default",
            )
            assert tuple(task.depends_on for task in tasks) == (
                (),
                (0,),
                (1,),
                (2,),
                (),
                (4,),
                (5,),
                (6,),
                (3, 7),
            )

            values = {name: ti.ndarray(ti.i32, shape=count) for name in symbols}
            initial = np.arange(count, dtype=np.int32)
            values["a0"].from_numpy(initial)
            values["a1"].fill(0)
            values["b0"].from_numpy(initial * 3)
            values["b1"].fill(0)
            values["output"].fill(0)
            bindings = product.executor.bind(values)
            assert bindings.fast_path_qualified
            published = product.executor.binding_statistics()
            for _ in range(32):
                product.executor.run(bindings)
            ti.sync()
            replayed = product.executor.binding_statistics()
            assert replayed["version_builds"] == published["version_builds"]
            assert replayed["raw_replay_validations"] == (
                published["raw_replay_validations"]
            )
            assert replayed["version_fast_replays"] == (
                published["version_fast_replays"] + 32
            )
            assert product.executor.execution_stats().execution_path == (
                "cuda_exact_replay"
            )
            np.testing.assert_array_equal(
                values["output"].to_numpy(),
                initial * 4 - 128,
            )

            aliased = dict(values)
            aliased["b0"] = aliased["a0"]
            with pytest.raises(
                TaichiRuntimeError,
                match="requires proven disjoint storage",
            ):
                product.executor.bind(aliased)

    short = ti.graph.GraphBuilder()
    short.dispatch(advance_a, symbols["a0"], symbols["a1"])
    short.dispatch(advance_b, symbols["b0"], symbols["b1"])
    short.dispatch(join, symbols["a1"], symbols["b1"], symbols["output"])
    assert not short.freeze()._runtime_spec._graph_branch_join_sources


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_complete_recipe_executes_independent_workspace_pair_as_one_submission():
    count = 257

    @ti.kernel
    def initialize(
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        scratch: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in source:
            scratch[index] = source[index] * 0.75 + 1.0

    @ti.kernel
    def relax(
        scratch: ti.types.ndarray(dtype=ti.f32, ndim=1),
        scale: ti.f32,
    ):
        for index in scratch:
            value = scratch[index]
            for _ in range(8):
                value = value * scale + 0.03125
            scratch[index] = value

    @ti.kernel
    def finalize(
        scratch: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in output:
            output[index] = scratch[index] * 1.25

    source_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "pair_source", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "pair_output", ti.f32, ndim=1)
    scale_arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "pair_scale", ti.f32)
    builder = ti.graph.GraphBuilder()
    scratch = builder.private_ndarray("pair_scratch", ti.f32, count)
    builder.dispatch(initialize, source_arg, scratch)
    for _ in range(6):
        builder.dispatch(relax, scratch, scale_arg)
    builder.dispatch(finalize, scratch, output_arg)
    definition = builder.freeze()

    catalog = definition.recipe_catalog()
    fragments = _family_fragments(catalog, "workspace_concurrency")
    assert len(fragments) == 1
    assert _selection(fragments[0]).source_key == "whole-graph-pair"
    entry = catalog.compose(
        (fragments[0].fragment_id,),
        stage="cuda-complete-workspace-pair",
        parent_recipe_ids=(catalog.baseline.recipe.recipe_id,),
    )

    sources = tuple(ti.ndarray(ti.f32, shape=count) for _ in range(2))
    outputs = tuple(ti.ndarray(ti.f32, shape=count) for _ in range(2))
    host_sources = (
        np.linspace(0.0, 1.0, count, dtype=np.float32),
        np.linspace(1.0, 2.0, count, dtype=np.float32),
    )
    for source, host in zip(sources, host_sources):
        source.from_numpy(host)
    frames = tuple(
        {
            "pair_source": source,
            "pair_output": output,
            "pair_scale": 0.999,
        }
        for source, output in zip(sources, outputs)
    )

    with definition.materialization_context() as context:
        with context.materialize(catalog.baseline.recipe) as baseline_product:
            baseline = baseline_product.executor
            baseline_batch = baseline.bind_batch(frames)
            baseline_ticket = baseline.submit_batch(baseline_batch)
            baseline_ticket.wait()
            assert baseline_ticket.workspace_lanes == (0, 0)
            baseline_id = baseline_product.materialized_physical_id
            baseline_bytes = (
                baseline.execution_stats().memory.persistent_internal_storage_bytes
            )

        with context.materialize(entry.recipe) as candidate_product:
            candidate = candidate_product.executor
            candidate.prepare_telemetry("timestamps")
            candidate_batch = candidate.bind_batch(frames)
            published = candidate.binding_statistics()
            ticket = candidate.submit_batch(
                candidate_batch,
                telemetry="timestamps",
            )
            telemetry = ticket.telemetry()
            replayed = candidate.binding_statistics()

            assert ticket.workspace_lanes == (0, 1)
            assert telemetry.gpu_duration_ns > 0
            assert telemetry.host_submit_ns >= 0
            assert replayed["raw_replay_validations"] == (
                published["raw_replay_validations"]
            )
            assert replayed["version_fast_replays"] == (
                published["version_fast_replays"] + 2
            )
            assert candidate_product.materialized_physical_id != baseline_id
            assert candidate_product.manifest.submissions[0].replay_mode == (
                "cuda_concurrent_complete_graph_pair"
            )
            assert tuple(candidate_product.manifest.submissions[0].queues) == (
                "cuda_workspace:0",
                "cuda_workspace:1",
                "default",
            )
            assert (
                candidate.execution_stats().memory.persistent_internal_storage_bytes
                == baseline_bytes * 2
            )

            aliased_frames = (frames[0], {**frames[1], "pair_output": outputs[0]})
            with pytest.raises(
                TaichiRuntimeError,
                match="requires proven disjoint cross-invocation storage",
            ):
                candidate.bind_batch(aliased_frames)

    expected = []
    for host in host_sources:
        value = host * np.float32(0.75) + np.float32(1.0)
        for _ in range(6):
            for _ in range(8):
                value = value * np.float32(0.999) + np.float32(0.03125)
        expected.append(value * np.float32(1.25))
    for output, reference in zip(outputs, expected):
        np.testing.assert_allclose(output.to_numpy(), reference, rtol=2e-5)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_complete_recipe_partitions_binding_churn_from_stable_cuda_recording():
    count = 257
    stages = 8

    @ti.kernel
    def initialize(
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        scratch_a: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(count):
            scratch_a[index] = source[index] * 0.875 + 0.125

    @ti.kernel
    def advance(
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        target: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(count):
            target[index] = source[index] * 0.999 + 0.03125

    @ti.kernel
    def publish(
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(count):
            output[index] = source[index] * 1.25

    symbols = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)
        for name in ("source", "scratch_a", "scratch_b", "output")
    }
    builder = ti.graph.GraphBuilder()
    builder.dispatch(initialize, symbols["source"], symbols["scratch_a"])
    current = "scratch_a"
    for _ in range(stages):
        target = "scratch_b" if current == "scratch_a" else "scratch_a"
        builder.dispatch(advance, symbols[current], symbols[target])
        current = target
    builder.dispatch(publish, symbols[current], symbols["output"])
    definition = builder.freeze()

    catalog = definition.recipe_catalog()
    fragments = _family_fragments(catalog, "recording_partition")
    total_dispatches = stages + 2
    tail = next(
        fragment
        for fragment in fragments
        if _selection(fragment).source_key
        == f"recording-partition:0:{total_dispatches - 1}"
    )
    assert "output" in tail.tasks[1].physical["isolated_bindings"]
    entry = catalog.compose(
        (tail.fragment_id,),
        stage="cuda-binding-frontier-partition",
        parent_recipe_ids=(catalog.baseline.recipe.recipe_id,),
    )

    source = ti.ndarray(ti.f32, shape=count)
    scratch_a = ti.ndarray(ti.f32, shape=count)
    scratch_b = ti.ndarray(ti.f32, shape=count)
    outputs = tuple(ti.ndarray(ti.f32, shape=count) for _ in range(3))
    host_source = np.linspace(-0.5, 1.5, count, dtype=np.float32)
    source.from_numpy(host_source)
    frames = tuple(
        {
            "source": source,
            "scratch_a": scratch_a,
            "scratch_b": scratch_b,
            "output": output,
        }
        for output in outputs
    )

    with definition.materialization_context() as context:
        with context.materialize(catalog.baseline.recipe) as baseline:
            baseline_id = baseline.materialized_physical_id
            assert len(baseline.executor.execution_stats().segments) == 1
        with context.materialize(entry.recipe) as candidate:
            graph = candidate.executor
            bindings = tuple(graph.bind(frame) for frame in frames)
            published = graph.binding_statistics()
            for replay in range(12):
                graph.run(bindings[replay % len(bindings)])
            ti.sync()
            replayed = graph.binding_statistics()
            stats = graph.execution_stats()

            assert candidate.materialized_physical_id != baseline_id
            assert tuple(segment.dispatch_count for segment in stats.segments) == (
                total_dispatches - 1,
                1,
            )
            assert all(segment.backend_graph_path for segment in stats.segments)
            assert replayed["raw_replay_validations"] == (
                published["raw_replay_validations"]
            )
            assert replayed["version_fast_replays"] == (
                published["version_fast_replays"] + 12
            )

    expected = host_source * np.float32(0.875) + np.float32(0.125)
    for _ in range(stages):
        expected = expected * np.float32(0.999) + np.float32(0.03125)
    expected *= np.float32(1.25)
    for output in outputs:
        np.testing.assert_allclose(output.to_numpy(), expected, rtol=1e-5, atol=1e-6)
