"""Actual native value substitution across reduction iteration domains."""

import gc

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge._lib import core
from taichi_forge.graph._graph import (
    Graph,
    _CompiledCGraphNode,
    _GraphSpec,
    gen_cpp_kernel,
)
from taichi_forge.graph._segmented_reduce_kernels import reduction_kernel
from taichi_forge.lang import impl
from tests import test_utils


@ti.kernel
def _produce(
    source: ti.types.ndarray(),
    weights: ti.types.ndarray(),
    values: ti.types.ndarray(),
    count: ti.i32,
    factor: ti.u32,
):
    for i in range(count):
        bits = ti.cast(source[i], ti.u32) * factor + ti.u32(0xF0000001)
        values[i] = (bits ^ ti.cast(weights[i], ti.u32)) - ti.cast(i, ti.u32)


@ti.kernel
def _consume(
    reduced: ti.types.ndarray(), bias: ti.types.ndarray(), result: ti.types.ndarray()
):
    for i in reduced:
        bits = ti.cast(reduced[i], ti.u32) ^ ti.cast(bias[i], ti.u32)
        result[i] = (bits << 3) | (bits >> 29)


def _array(name, dtype):
    return ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, dtype, ndim=1)


def _fixture(dtype):
    bounds = np.asarray((0, 0, 1, 18, 147, 2196, 10389), np.int32)
    count = int(bounds[-1])
    size = count + 11
    raw = (np.arange(size, dtype=np.uint64) * 0x60000001 + 0xF0000001).astype(np.uint32)
    weights = raw[::-1].copy()
    bias = raw[: len(bounds) - 1].copy()
    arrays = {}
    for name, value in (("source", raw), ("weights", weights), ("bias", bias)):
        arrays[name] = ti.ndarray(dtype, shape=len(value))
        arrays[name].from_numpy(value if dtype == ti.u32 else value.view(np.int32))
    arrays["offsets"] = ti.ndarray(ti.i32, shape=len(bounds))
    arrays["offsets"].from_numpy(bounds)
    for name, size_ in (
        ("values", size),
        ("reduced", len(bounds) - 1),
        ("result", len(bounds) - 1),
    ):
        arrays[name] = ti.ndarray(dtype, shape=size_)
        arrays[name].fill(777)
    factor = 0x60000001
    mapped = (
        (raw.astype(np.uint64) * factor + 0xF0000001) & 0xFFFFFFFF
    ) ^ weights.astype(np.uint64)
    mapped = ((mapped - np.arange(size, dtype=np.uint64)) & 0xFFFFFFFF).astype(
        np.uint32
    )
    expected_values = np.full(size, 777, np.uint32)
    expected_values[:count] = mapped[:count]
    sums = np.asarray(
        [
            int(mapped[a:b].sum(dtype=np.uint64)) & 0xFFFFFFFF
            for a, b in zip(bounds, bounds[1:])
        ],
        np.uint32,
    )
    bits = sums ^ bias
    final = (bits << 3) | (bits >> 29)
    arrays.update(count=count, factor=factor)
    return arrays, bounds, (expected_values, sums, final)


def _sources(dtype):
    producer_args = [_array(name, dtype) for name in ("source", "weights", "values")]
    producer_args += [
        ti.graph.Arg(ti.graph.ArgKind.SCALAR, "count", ti.i32),
        ti.graph.Arg(ti.graph.ArgKind.SCALAR, "factor", ti.u32),
    ]
    consumer_args = [_array(name, dtype) for name in ("reduced", "bias", "result")]
    return (gen_cpp_kernel(_produce, producer_args), producer_args), (
        gen_cpp_kernel(_consume, consumer_args),
        consumer_args,
    )


def _fused_node(
    dtype,
    segments,
    block_dim,
    producer,
    consumer,
    *,
    values="values",
    offsets="offsets",
    output="reduced"
):
    reduction_args = [
        _array(values, dtype),
        _array(offsets, ti.i32),
        _array(output, dtype),
    ]
    reduction = gen_cpp_kernel(
        reduction_kernel(dtype, segments, block_dim=block_dim), reduction_args
    )
    compiled = core._compile_graph_segmented_reduce_values(
        impl.get_runtime().prog,
        reduction,
        reduction_args,
        *(producer if producer is not None else (None, ())),
        *(consumer if consumer is not None else (None, ())),
        0 if consumer is not None else -1,
    )
    names = tuple(
        dict.fromkeys(
            argument.name
            for source in (
                reduction_args,
                producer[1] if producer else (),
                consumer[1] if consumer else (),
            )
            for argument in source
        )
    )
    assert compiled._composer_stats["compiled_tasks"] == 1
    recordings = compiled._owned_jit_dispatch_sources
    assert len(recordings) == 1
    assert {argument.name for argument in recordings[0][1]} == set(names)
    # Re-record through the same raw-dispatch bridge used when Graph regions
    # are concatenated. Only the recording handles now retain the original
    # graph that owns the synthetic kernel; the new CGraph has no such owner.
    builder = core.GraphBuilder()
    for kernel_cpp, arguments in recordings:
        builder.dispatch(kernel_cpp, arguments)
    rerecorded = builder.compile()
    return _CompiledCGraphNode(rerecorded, 1, names, recording_dispatches=recordings)


def _assert_outputs(bindings, expected):
    for name, values in zip(("values", "reduced", "result"), expected):
        np.testing.assert_array_equal(bindings[name].to_numpy().view(np.uint32), values)


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("dtype", (ti.i32, ti.u32))
def test_native_value_fusion_preserves_stores_bits_empty_segments_and_tail(
    dtype, monkeypatch
):
    bindings, bounds, expected = _fixture(dtype)
    producer, consumer = _sources(dtype)
    # Compare with the actual three-dispatch baseline before discarding source
    # handles. The reference is independent and includes unused capacity.
    _produce(
        bindings["source"],
        bindings["weights"],
        bindings["values"],
        bindings["count"],
        bindings["factor"],
    )
    reduction_kernel(dtype, len(bounds) - 1)(
        bindings["values"], bindings["offsets"], bindings["reduced"]
    )
    _consume(bindings["reduced"], bindings["bias"], bindings["result"])
    _assert_outputs(bindings, expected)
    for block_dim in (0, 32, 128):
        for name in ("values", "reduced", "result"):
            bindings[name].fill(777)
        node = _fused_node(dtype, len(bounds) - 1, block_dim, producer, consumer)
        graph = Graph(node)
        bound = graph.bind(bindings)

        # Native compilation owns the synthetic kernel; no source execution or
        # value inspection is permitted once the Graph is bound.
        def forbidden(*args, **kwargs):
            raise AssertionError("replay invoked cold value compilation")

        with monkeypatch.context() as replay:
            replay.setattr(core, "_compile_graph_segmented_reduce_values", forbidden)
            replay.setattr(core, "_graph_pointwise_value_program", forbidden)
            gc.collect()
            for _ in range(3):
                graph.run(bound)
        _assert_outputs(bindings, expected)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_native_partial_fusion_maps_only_input_and_final_stage():
    bindings, bounds, expected = _fixture(ti.u32)
    producer, consumer = _sources(ti.u32)
    tiles, ends = [0], [0]
    for a, b in zip(bounds, bounds[1:]):
        tiles.extend(min(int(i) + 1024, int(b)) for i in range(int(a), int(b), 1024))
        ends.append(len(tiles) - 1)
    for name, data in (("tile_offsets", tiles), ("final_offsets", ends)):
        bindings[name] = ti.ndarray(ti.i32, shape=len(data))
        bindings[name].from_numpy(np.asarray(data, np.int32))
    bindings["partials"] = ti.ndarray(ti.u32, shape=len(tiles) - 1)
    first = _fused_node(
        ti.u32,
        len(tiles) - 1,
        128,
        producer,
        None,
        offsets="tile_offsets",
        output="partials",
    )
    final = _fused_node(
        ti.u32,
        len(bounds) - 1,
        32,
        None,
        consumer,
        values="partials",
        offsets="final_offsets",
    )
    bindings.pop("offsets")
    graph = Graph(_GraphSpec([first, final]))
    graph.run(graph.bind(bindings))
    _assert_outputs(bindings, expected)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_native_value_fusion_rejects_symbolic_abi_mismatch_before_execution():
    bindings, bounds, _ = _fixture(ti.u32)
    producer, consumer = _sources(ti.u32)
    wrong = list(producer[1])
    wrong[2] = _array("different_values", ti.u32)
    with pytest.raises(RuntimeError, match="producer output.*values symbol"):
        _fused_node(ti.u32, len(bounds) - 1, 32, (producer[0], wrong), consumer)
    for name in ("values", "reduced", "result"):
        np.testing.assert_array_equal(bindings[name].to_numpy(), 777)


@ti.kernel
def _copy_add(source: ti.types.ndarray(), target: ti.types.ndarray()):
    for i in target:
        target[i] = source[i] + 9


def _value_definition(*, extra_dispatches=False):
    bindings, bounds, expected = _fixture(ti.u32)
    producer, consumer = _sources(ti.u32)
    layout = ti.algorithms.SegmentedLayout.from_offsets(
        bounds, capacity=bindings["values"].shape[0]
    )
    builder = ti.graph.GraphBuilder()
    if extra_dispatches:
        bindings["prefix"] = ti.ndarray(ti.u32, shape=bindings["source"].shape)
        bindings["suffix"] = ti.ndarray(ti.u32, shape=bindings["result"].shape)
        builder.dispatch(_copy_add, _array("source", ti.u32), _array("prefix", ti.u32))
    builder.dispatch(_produce, *producer[1])
    builder.segmented_reduce(bindings["values"], layout, bindings["reduced"])
    builder.dispatch(_consume, *consumer[1])
    if extra_dispatches:
        builder.dispatch(_copy_add, _array("result", ti.u32), _array("suffix", ti.u32))
    bindings.pop("offsets")
    return builder.freeze(), bindings, expected


def _value_recipes(definition):
    catalog = definition.recipe_catalog()
    result = []
    for fragment in catalog.fragments:
        if fragment.provider_namespace != "taichi_forge.graph.value_fusion":
            continue
        facts = fragment.tasks[0].physical
        recipe = catalog.compose(
            (fragment.fragment_id,),
            stage="single-region",
            parent_recipe_ids=(catalog.baseline.recipe.recipe_id,),
        ).recipe
        result.append((recipe, facts))
    return result


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_public_value_recipes_preserve_all_stores_and_cold_binding_proofs(monkeypatch):
    from taichi_forge.graph._recipes.value_fusion import _BindingProof

    definition, bindings, expected = _value_definition()
    recipes = [
        (recipe, facts)
        for recipe, facts in _value_recipes(definition)
        if facts["producer"] and facts["consumer"] and facts["consumer_input"] == 0
    ]
    assert len(recipes) == 8  # Four reduction topologies, two submission plans.
    physical = set()
    for recipe, facts in recipes:
        for name in ("values", "reduced", "result"):
            bindings[name].fill(777)
        with definition.materialization_context() as context:
            materialized = context.materialize(recipe)
            graph = materialized.executor
            # Discovery/compilation have not run any part of the user's DAG.
            np.testing.assert_array_equal(bindings["result"].to_numpy(), 777)
            bound = graph.bind(bindings)

            def forbidden(*args, **kwargs):
                raise AssertionError(
                    "published replay performed cold value-fusion work"
                )

            with monkeypatch.context() as replay:
                replay.setattr(_BindingProof, "validate_graph_bindings", forbidden)
                replay.setattr(core, "_graph_pointwise_value_program", forbidden)
                replay.setattr(
                    core, "_compile_graph_segmented_reduce_values", forbidden
                )
                for _ in range(3):
                    graph.run(bound)
            _assert_outputs(bindings, expected)
            assert len(graph._spec.nodes) == 1
            assert graph._instance.physical_submission_mode == (
                "cuda_immutable_argument_frames_exec_reuse"
                if facts["submission"] == "immutable_frames"
                else "runtime_managed"
            )
            expected_tasks = (
                2 if facts["reduction"]["strategy"] == "chunk_partial_finalize" else 1
            )
            assert graph._spec.nodes[0].physical_dispatch_count == expected_tasks
            assert (
                materialized.manifest.persistent_requested_bytes
                == recipe.declared_persistent_resource_bytes
            )
            physical.add(materialized.manifest.materialized_physical_id)
    assert len(physical) == 8


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_value_recipe_binding_rejects_false_dataflow_tail_and_cross_phase_alias():
    from taichi_forge.graph._recipes.materialize import GraphMaterializationError

    definition, bindings, _ = _value_definition()
    recipe, _ = next(
        (recipe, facts)
        for recipe, facts in _value_recipes(definition)
        if facts["producer"]
        and facts["consumer"]
        and facts["consumer_input"] == 0
        and facts["submission"] == "immutable_frames"
    )
    with definition.materialization_context(workspace_lanes=2) as context:
        with pytest.raises(
            GraphMaterializationError, match="one ordered workspace lane"
        ):
            context.materialize(recipe)
    with definition.materialization_context() as context:
        materialized = context.materialize(recipe)
        graph = materialized.executor
        changed = dict(
            bindings, values=ti.ndarray(ti.u32, shape=bindings["values"].shape)
        )
        with pytest.raises(ValueError, match="fixed reduction values"):
            graph.bind(changed)
        with pytest.raises(ValueError, match="exact iteration coverage"):
            graph.bind(dict(bindings, count=bindings["count"] + 1))
        with pytest.raises(ValueError, match="storage alias"):
            graph.bind(dict(bindings, source=bindings["values"]))
        with pytest.raises(ValueError, match="storage alias"):
            graph.bind(dict(bindings, bias=bindings["reduced"]))
        # Failed publication leaves all original outputs untouched and does not
        # invalidate the legal baseline or a subsequent correct publication.
        for name in ("values", "reduced", "result"):
            np.testing.assert_array_equal(bindings[name].to_numpy(), 777)
        graph.run(graph.bind(bindings))


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_public_value_search_records_binding_failures_and_restores_selection():
    definition, bindings, expected = _value_definition()
    observed = []

    def evaluate(graph, recipe):
        bound = graph.bind(bindings)
        graph.run(bound)
        _assert_outputs(bindings, expected)
        # Count actual compiled dispatches plus root native calls, not speed.
        count = sum(
            node.physical_dispatch_count + int(node.source_native_count)
            for node in graph._spec.nodes
        )
        observed.append((recipe.recipe_id, count))
        return {"execution_units": float(count)}

    session = definition.search_recipes(
        engine="compileiq",
        target=ti.graph.GraphOptimizationTarget(
            objectives=(("execution_units", "min"),)
        ),
        budget=ti.graph.GraphSearchBudget(evaluation_limit=40),
        workload_context=ti.graph.GraphWorkloadContext(
            {"fixture": "certified-value-region"}
        ),
        evaluation_contract=ti.graph.GraphEvaluationContract(
            {"metric": "actual-execution-units-not-timing"}
        ),
        backend_environment=ti.graph.GraphBackendEnvironment(
            {"fixture": "current-cuda"}
        ),
    )
    decision = session.run(evaluate)
    assert any(count == 1 for _, count in observed)
    assert not decision.selection.manifest.is_baseline
    second, second_bindings, second_expected = _value_definition()
    assert second.semantic_graph_id == definition.semantic_graph_id
    resolved = second.resolve_recipe(decision.selection_artifact)
    with second.materialize(resolved) as materialized:
        materialized.executor.run(materialized.executor.bind(second_bindings))
        _assert_outputs(second_bindings, second_expected)
    report = decision.report.to_json()
    assert "fixed reduction output" in report  # Invalid role is a recorded failure.
    assert "segmented-values:" in report


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_value_fusion_retains_unaffected_prefix_suffix_and_source_lineage():
    definition, bindings, expected = _value_definition(extra_dispatches=True)
    recipe, _ = next(
        (recipe, facts)
        for recipe, facts in _value_recipes(definition)
        if facts["producer"]
        and facts["consumer"]
        and facts["consumer_input"] == 0
        and facts["submission"] == "immutable_frames"
        and facts["reduction"]["strategy"] == "chunk_partial_finalize"
    )
    with definition.materialization_context() as context:
        materialized = context.materialize(recipe)
        graph = materialized.executor
        graph.run(graph.bind(bindings))
        _assert_outputs(bindings, expected)
        np.testing.assert_array_equal(
            bindings["prefix"].to_numpy(), bindings["source"].to_numpy() + np.uint32(9)
        )
        np.testing.assert_array_equal(
            bindings["suffix"].to_numpy(), expected[2] + np.uint32(9)
        )
        assert graph._spec.nodes[0].physical_dispatch_count == 4
        assert {
            region for task in materialized.manifest.tasks for region in task.region_ids
        } == {source.region_id for source in definition.sources}
