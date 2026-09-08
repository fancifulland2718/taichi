import gc
import weakref

import numpy as np
import pytest
import taichi_forge as ti

from tests import test_utils


def _definition(dtype=ti.i32, lengths=(0, 1, 17, 129, 2049, 8193)):
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int32)
    capacity = max(1, int(offsets[-1]) + 11)
    layout = ti.algorithms.SegmentedLayout.from_offsets(offsets, capacity=capacity)
    values = ti.ndarray(dtype, shape=capacity)
    output = ti.ndarray(dtype, shape=len(lengths))
    raw = (np.arange(capacity, dtype=np.uint64) * 0x60000001 + 0xF0000001).astype(
        np.uint32
    )
    host = raw if dtype == ti.u32 else raw.view(np.int32)
    expected = np.asarray(
        [
            int(raw[a:b].sum(dtype=np.uint64)) & 0xFFFFFFFF
            for a, b in zip(offsets, offsets[1:])
        ],
        dtype=np.uint32,
    )
    if dtype == ti.i32:
        expected = expected.view(np.int32)
    values.from_numpy(host)
    output.fill(123)
    builder = ti.graph.GraphBuilder()
    builder.segmented_reduce(values, layout, output)
    return builder.freeze(), values, output, host, expected


def _recipes(definition):
    catalog = definition.recipe_catalog()
    return [catalog.baseline.recipe] + [
        catalog.compose(
            (fragment.fragment_id,),
            stage="single-region",
            parent_recipe_ids=(catalog.baseline.recipe.recipe_id,),
        ).recipe
        for fragment in catalog.fragments
        if fragment.provider_namespace == "taichi_forge.graph.native_algorithm"
    ]


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("dtype", (ti.i32, ti.u32))
def test_segmented_reduce_complete_recipes_preserve_values_and_retire_scratch(
    dtype, monkeypatch
):
    import taichi_forge.graph._segmented_reduce as implementation

    definition, values, output, host, expected = _definition(dtype)
    recipes = _recipes(definition)
    assert len(recipes) == 4
    # Discovery/materialization must not perform the user's reduction.
    np.testing.assert_array_equal(
        output.to_numpy(), np.full(len(expected), 123, dtype=expected.dtype)
    )
    identities = set()
    scratch_refs = []
    for recipe in recipes:
        with definition.materialization_context() as context:
            materialized = context.materialize(recipe)
            executor = materialized.executor._spec.nodes[0].executable
            owned_bytes = sum(int(array.shape[0]) * 4 for array in executor._owned)
            assert owned_bytes == recipe.declared_persistent_resource_bytes
            scratch_refs.extend(weakref.ref(array) for array in executor._owned)
            assert materialized.manifest.persistent_requested_bytes == owned_bytes
            materialized.executor.run({})
            ti.sync()
            np.testing.assert_array_equal(output.to_numpy(), expected)

            def reject_cold_work(*_args, **_kwargs):
                raise AssertionError(
                    "replay repeated reduction validation or generation"
                )

            with monkeypatch.context() as replay:
                replay.setattr(
                    implementation, "_check_segmented_request", reject_cold_work
                )
                replay.setattr(implementation, "reduction_kernel", reject_cold_work)
                # Queue multiple calls without host readback/sync in between.
                for _ in range(3):
                    materialized.executor.run({})
            ti.sync()
            np.testing.assert_array_equal(output.to_numpy(), expected)
            np.testing.assert_array_equal(
                values.to_numpy(), host
            )  # Includes unused capacity.
            identities.add(materialized.manifest.materialized_physical_id)
        del executor, materialized, context
    gc.collect()
    assert len(identities) == 4
    assert scratch_refs and all(reference() is None for reference in scratch_refs)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_segmented_reduce_empty_domain_and_cold_semantic_rejections():
    definition, values, output, _, expected = _definition(lengths=(0, 0, 0))
    assert len(_recipes(definition)) == 1
    with definition.materialization_context() as context:
        context.materialize(_recipes(definition)[0]).executor.run({})
        ti.sync()
        np.testing.assert_array_equal(output.to_numpy(), expected)
    layout = ti.algorithms.SegmentedLayout.from_offsets(
        np.asarray((0, 1, 2, 3), np.int32), capacity=3
    )
    aliased = ti.ndarray(ti.i32, shape=3)
    with pytest.raises(ValueError, match="alias"):
        ti.graph.GraphBuilder().segmented_reduce(aliased, layout, aliased)
    with pytest.raises(ValueError, match="op='sum'"):
        ti.graph.GraphBuilder().segmented_reduce(values, layout, output, op="min")
    floats = ti.ndarray(ti.f32, shape=3)
    with pytest.raises(ti.TaichiRuntimeError, match="i32/u32"):
        ti.graph.GraphBuilder().segmented_reduce(
            floats, layout, ti.ndarray(ti.f32, shape=3)
        )


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_segmented_reduce_public_search_and_equivalent_definition_restore():
    definition, values, output, _, expected = _definition()
    observed = []

    def evaluate(graph, recipe):
        graph.run({})
        ti.sync()
        np.testing.assert_array_equal(output.to_numpy(), expected)
        observed.append(recipe.recipe_id)
        # A structural test objective deliberately selects the two-stage
        # candidate. These are actual bytes, not simulated performance times.
        return {
            "owned_bytes": float(
                graph._spec.nodes[0].executable.debug_info["action_owned_bytes"]
            )
        }

    session = definition.search_recipes(
        engine="compileiq",
        target=ti.graph.GraphOptimizationTarget(objectives=(("owned_bytes", "max"),)),
        budget=ti.graph.GraphSearchBudget(evaluation_limit=4),
        workload_context=ti.graph.GraphWorkloadContext(
            {"fixture": "segmented-reduce-ownership"}
        ),
        evaluation_contract=ti.graph.GraphEvaluationContract(
            {"metric": "actual-requested-bytes-not-performance"}
        ),
        backend_environment=ti.graph.GraphBackendEnvironment(
            {"fixture": "current-cuda"}
        ),
    )
    decision = session.run(evaluate)
    assert len(set(observed)) == 4
    assert not decision.selection.manifest.is_baseline
    second, second_values, second_output, _, second_expected = _definition()
    assert second.semantic_graph_id == definition.semantic_graph_id
    resolved = second.resolve_recipe(decision.selection_artifact)
    with second.materialize(resolved) as materialized:
        materialized.executor.run({})
        ti.sync()
        np.testing.assert_array_equal(second_output.to_numpy(), second_expected)
    report = ti.graph.GraphOptimizationReportV2.from_json(decision.report.to_json())
    assert report.to_dict() == decision.report.to_dict()
    assert "graph-segmented-reduce:" in report.to_json()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_segmented_reduce_recipes_compose_with_an_ordinary_consumer():
    definition, values, output, _, expected = _definition()
    layout = definition._runtime_spec._graph_native_algorithm_sources[0].layout
    consumed = ti.ndarray(ti.i32, shape=output.shape)

    @ti.kernel
    def consume(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        target: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for index in source:
            target[index] = source[index] ^ 7

    builder = ti.graph.GraphBuilder()
    builder.segmented_reduce(values, layout, output)
    builder.dispatch(
        consume,
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "consumed", ti.i32, ndim=1),
    )
    mixed = builder.freeze()
    recipes = _recipes(mixed)
    assert len(recipes) == 4
    with mixed.materialization_context() as context:
        for recipe in recipes:
            with context.materialize(recipe) as materialized:
                materialized.executor.run({"output": output, "consumed": consumed})
                ti.sync()
                np.testing.assert_array_equal(consumed.to_numpy(), expected ^ 7)
