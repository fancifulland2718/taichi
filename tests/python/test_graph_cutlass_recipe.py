"""Optional source-addon integration, not application performance qualification."""

import json
import os

import numpy as np
import pytest
import taichi_forge as ti

from tests import test_utils


def _provider(**kwargs):
    from taichi_forge.hardware.source_providers import CutlassMatmulRecipeProvider

    path = os.environ.get("TI_FORGE_TEST_CUTLASS_MANIFEST")
    cublas = os.environ.get("TI_FORGE_TEST_CUBLASLT_LIBRARY_PATH")
    if not path or not cublas:
        pytest.skip("explicit CUTLASS addon and cuBLASLt baseline are required")
    os.environ["TI_CUBLASLT_LIBRARY_PATH"] = cublas
    return CutlassMatmulRecipeProvider(path, **kwargs)


def _definition(
    *,
    shape=(65, 97, 513),
    transpose=(False, False),
    activation="relu",
    preparation=None
):
    operation = ti.linalg.record_matmul(
        *shape,
        transpose_a=transpose[0],
        transpose_b=transpose[1],
        alpha=0.75,
        beta=0.25,
        activation=activation,
        absolute_tolerance=5e-5,
        relative_tolerance=5e-5,
        preparation=preparation,
    )
    if preparation is None:
        operation.prepare(heuristic_limit=1)
    builder = ti.graph.GraphBuilder()
    builder.append_native(operation)
    return builder.freeze(), operation


def _inputs(operation):
    from taichi_forge.linalg._matmul import _shapes

    rng = np.random.default_rng(910)
    host = [
        rng.uniform(-0.5, 0.5, shape).astype(np.float32)
        for shape in _shapes(operation.semantics)
    ]
    arrays = [ti.ndarray(ti.f32, shape=x.shape) for x in host]
    for arr, x in zip(arrays, host):
        arr.from_numpy(x)
    return dict(zip(("a", "b", "output"), arrays)), host


def _recipes(definition, provider):
    providers = (*ti.graph.default_recipe_providers(), provider)
    catalog = definition.recipe_catalog(providers=providers)
    fragments = [
        f
        for f in catalog.fragments
        if f.provider_namespace == provider.descriptor.namespace
    ]
    return providers, catalog, fragments


def _fail(*args, **kwargs):
    raise AssertionError("CUTLASS steady replay repeated cold work")


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize(
    "transpose,activation",
    [
        ((False, False), "identity"),
        ((False, True), "relu"),
        ((True, False), "relu"),
        ((True, True), "identity"),
    ],
)
def test_cutlass_complete_regions_feedback_layout_and_cold_boundaries(
    transpose, activation, monkeypatch
):
    from taichi_forge.hardware.source_providers._cutlass_matmul import (
        _Library,
        _Recording,
    )

    provider = _provider()
    definition, operation = _definition(transpose=transpose, activation=activation)
    bindings, host = _inputs(operation)
    providers, catalog, fragments = _recipes(definition, provider)
    assert len(fragments) == 3
    assert sorted(sum(r.bytes for r in f.resources) for f in fragments) == [
        0,
        65 * 97 * 4 * 16,
        65 * 97 * 4 * 128,
    ]
    product = 0.75 * (
        (host[0].T if transpose[0] else host[0]).astype(np.float64)
        @ (host[1].T if transpose[1] else host[1]).astype(np.float64)
    )
    identities = set()
    for fragment in fragments:
        bindings["output"].from_numpy(host[2])
        recipe = catalog.compose((fragment.fragment_id,), stage="test").recipe
        with definition.materialization_context(providers=providers) as context:
            with context.materialize(recipe) as materialized:
                graph = materialized.executor
                assert (
                    graph.definition.semantic_graph_id == definition.semantic_graph_id
                )
                bound = graph.bind(bindings)
                # Neither preparation nor capture is allowed to advance beta feedback.
                np.testing.assert_array_equal(bindings["output"].to_numpy(), host[2])
                expected = host[2].astype(np.float64)
                with monkeypatch.context() as hot:
                    hot.setattr(_Library, "workspace", _fail)
                    hot.setattr(_Recording, "validate_graph_bindings", _fail)
                    hot.setattr(provider._library, "query", _fail)
                    for _ in range(3):
                        graph.run(bound)
                        expected = product + 0.25 * expected
                        if activation == "relu":
                            expected = np.maximum(expected, 0)
                np.testing.assert_allclose(
                    bindings["output"].to_numpy(), expected, rtol=5e-5, atol=5e-5
                )
                assert graph._graph_stats[0]["last_path"] == "cuda_exact_replay"
                # Live input refresh, not a retained-value cache.
                bindings["a"].from_numpy(host[0] * -0.5)
                graph.run(bound)
                expected = -0.5 * product + 0.25 * expected
                if activation == "relu":
                    expected = np.maximum(expected, 0)
                np.testing.assert_allclose(
                    bindings["output"].to_numpy(), expected, rtol=5e-5, atol=5e-5
                )
                bindings["a"].from_numpy(host[0])
                identities.add(materialized.materialized_physical_id)
                assert materialized.manifest.persistent_requested_bytes == sum(
                    r.bytes for r in fragment.resources
                )
    assert len(identities) == 3


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cutlass_search_report_resolve_and_workspace_budget():
    provider = _provider()
    definition, operation = _definition(shape=(32, 48, 257))
    bindings, host = _inputs(operation)
    providers, catalog, fragments = _recipes(definition, provider)
    assert len(fragments) == 3
    limited = _provider(workspace_limit_bytes=0)
    assert len(_recipes(definition, limited)[2]) == 1
    observed = set()
    session = definition.search_recipes(
        providers=providers,
        target=ti.graph.GraphOptimizationTarget(
            objectives=(("structural_addon_selection", "max"),)
        ),
        budget=ti.graph.GraphSearchBudget(evaluation_limit=16, repeat_count=1),
        strategy=ti.graph.GraphRecipeSearchStrategy(mode="exact_if_bounded"),
    )

    def evaluate(graph, recipe):
        bindings["output"].from_numpy(host[2])
        graph.run(graph.bind(bindings))
        expected = np.maximum(
            0.75 * (host[0].astype(np.float64) @ host[1].astype(np.float64))
            + 0.25 * host[2],
            0,
        )
        np.testing.assert_allclose(
            bindings["output"].to_numpy(), expected, rtol=5e-5, atol=5e-5
        )
        selected = any(
            f.provider_namespace == provider.descriptor.namespace
            for f in catalog.entry(recipe.recipe_id).recipe.fragments
        )
        observed.add(recipe.recipe_id)
        return {"structural_addon_selection": float(selected)}

    decision = session.run(evaluate)
    assert decision.status == "selected", decision.report.results
    assert decision.report.search_complete
    assert len(observed) == len(session.recipes)
    assert len(observed) >= 3
    assert "cutlass" in json.dumps(decision.report.recipe_annotations)
    restored = definition.resolve_recipe(
        decision.selection_artifact, providers=providers
    )
    with definition.materialize(restored) as materialized:
        evaluate(materialized.executor, restored)
    fresh, _ = _definition(
        shape=(32, 48, 257), preparation=operation.preparation_artifact()
    )
    assert fresh.semantic_graph_id == definition.semantic_graph_id
    resolved = fresh.resolve_recipe(decision.selection_artifact, providers=providers)
    with fresh.materialize(resolved) as materialized:
        evaluate(materialized.executor, resolved)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cutlass_rejects_same_count_wrong_shape_at_binding():
    provider = _provider()
    definition, operation = _definition(shape=(32, 48, 257))
    bindings, _ = _inputs(operation)
    providers, catalog, fragments = _recipes(definition, provider)
    recipe = catalog.compose((fragments[0].fragment_id,), stage="test").recipe
    with definition.materialization_context(providers=providers) as context:
        with context.materialize(recipe) as materialized:
            bad = {**bindings, "a": ti.ndarray(ti.f32, (257, 32))}
            with pytest.raises(ti.TaichiRuntimeError, match="shape"):
                materialized.executor.bind(bad)
