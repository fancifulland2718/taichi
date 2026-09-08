import gc
import json
import os
import weakref

import numpy as np
import pytest
import taichi_forge as ti

from tests import test_utils


def _operation(monkeypatch, *, preparation=None, batch=2, size=(65, 49, 97)):
    library = os.environ.get("TI_FORGE_TEST_CUBLASLT_LIBRARY_PATH")
    if not library:
        pytest.skip("a compatible user-provided cuBLASLt is required")
    monkeypatch.setenv("TI_CUBLASLT_LIBRARY_PATH", library)
    return ti.linalg.record_matmul(
        *size,
        batch_count=batch,
        transpose_a=True,
        transpose_b=True,
        alpha=0.75,
        beta=0.25,
        activation="relu",
        absolute_tolerance=2e-5,
        relative_tolerance=2e-5,
        preparation=preparation,
    )


@ti.kernel
def _consume(
    source: ti.types.ndarray(dtype=ti.f32), target: ti.types.ndarray(dtype=ti.f32)
):
    for index in ti.grouped(source):
        target[index] = 0.5 * source[index] + 1.0


def _freeze(operation, *, consumer=False):
    builder = ti.graph.GraphBuilder()
    builder.append_native(operation)
    if consumer:
        rank = 3 if operation.semantics["batch_count"] > 1 else 2
        builder.dispatch(
            _consume,
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=rank),
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "consumed", ti.f32, ndim=rank),
        )
    return builder.freeze()


def _providers():
    return (
        *ti.graph.default_recipe_providers(),
        ti.hardware.linalg.MatmulRecipeProvider(),
    )


def _recipes(definition):
    catalog = definition.recipe_catalog(providers=_providers())
    return [(catalog.baseline.recipe, None)] + [
        (
            catalog.compose((fragment.fragment_id,), stage="single-region").recipe,
            fragment,
        )
        for fragment in catalog.fragments
        if fragment.provider_namespace == "taichi_forge.graph.matmul"
    ]


def _inputs(operation):
    facts = operation.semantics
    m, n, k, batch = (facts[key] for key in ("m", "n", "k", "batch_count"))
    shapes = ((k, m), (n, k), (m, n))
    if batch > 1:
        shapes = tuple((batch, *shape) for shape in shapes)
    rng = np.random.default_rng(197)
    host = [rng.standard_normal(shape).astype(np.float32) for shape in shapes]
    arrays = [ti.ndarray(ti.f32, shape=shape) for shape in shapes]
    for array, value in zip(arrays, host):
        array.from_numpy(value)
    return dict(zip(("a", "b", "output"), arrays)), host


def _product(host):
    return 0.75 * (
        np.swapaxes(host[0].astype(np.float64), -1, -2)
        @ np.swapaxes(host[1].astype(np.float64), -1, -2)
    )


def _no_cold_work(*_args, **_kwargs):
    raise AssertionError("immutable Graph replay repeated cold provider work")


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("batch", (1, 2))
def test_matmul_complete_strategies_refresh_inputs_and_retire_storage(
    batch, monkeypatch
):
    from taichi_forge.hardware._cublaslt_algorithms import _AlgorithmApi
    from taichi_forge.linalg._matmul import _MatmulRecording, _storage_bytes

    operation = _operation(
        monkeypatch, batch=batch, size=(513, 257, 385) if batch == 1 else (65, 49, 97)
    )
    artifact = operation.prepare(heuristic_limit=2)
    owner = operation._provider_owner
    assert not tuple(
        owner._plans
    )  # Preparation owns neither trial plans nor GPU scratch.
    definition = _freeze(operation, consumer=True)
    bindings, host = _inputs(operation)
    bindings["consumed"] = ti.ndarray(ti.f32, shape=host[2].shape)
    recipes = _recipes(definition)
    configs = artifact["choices"]
    # Cover distinct packing/epilogue strategies, not a vendor-version-specific
    # heuristic count or an exhaustive algorithm Cartesian product.
    selected = {}
    for recipe, fragment in recipes:
        key = (
            artifact["baseline"]
            if fragment is None
            else fragment.provider_metadata["family_selection"][
                "materialization_choice"
            ]
        )
        config = configs[key]
        selected.setdefault(
            (tuple(config["packed_inputs"]), config["epilogue"]), (recipe, config)
        )
    assert ((), "separate") in selected and ((), "fused") in selected
    assert (("a", "b"), "separate") in selected and (("a", "b"), "fused") in selected
    identities, resources = set(), []
    operation.close()  # Frozen semantics, not the source operation, own reconstruction.
    for recipe, config in selected.values():
        for array, value in zip(bindings.values(), host):
            array.from_numpy(value)
        with definition.materialization_context(providers=_providers()) as context:
            with context.materialize(recipe) as materialized:
                graph = materialized.executor
                frame = graph.bind(bindings)
                plans = tuple(owner._plans)
                assert len(plans) == 1
                resources.extend(weakref.ref(plan) for plan in plans)
                resources.extend(
                    weakref.ref(plan.workspace)
                    for plan in plans
                    if plan.workspace is not None
                )
                expected_bytes = (
                    _storage_bytes(operation.semantics, config)
                    + config["algorithm"]["workspace_bytes"]
                )
                assert (
                    materialized.manifest.persistent_requested_bytes == expected_bytes
                )
                assert (
                    not graph.execution_stats().memory.provider_generation_requested_bytes_complete
                )
                expected = host[2].astype(np.float64)
                with monkeypatch.context() as replay:
                    replay.setattr(_AlgorithmApi, "shortlist", _no_cold_work)
                    replay.setattr(_AlgorithmApi, "restore", _no_cold_work)
                    replay.setattr(
                        _MatmulRecording, "validate_graph_bindings", _no_cold_work
                    )
                    for _ in range(3):
                        graph.run(frame)
                        expected = np.maximum(_product(host) + 0.25 * expected, 0)
                np.testing.assert_allclose(
                    bindings["output"].to_numpy(), expected, rtol=2e-5, atol=2e-5
                )
                changed = [host[0] * -0.5, host[1], host[2]]
                bindings["a"].from_numpy(changed[0])
                graph.run(frame)
                final = np.maximum(_product(changed) + 0.25 * expected, 0)
                np.testing.assert_allclose(
                    bindings["output"].to_numpy(), final, rtol=2e-5, atol=2e-5
                )
                np.testing.assert_allclose(
                    bindings["consumed"].to_numpy(),
                    0.5 * final + 1.0,
                    rtol=2e-5,
                    atol=2e-5,
                )
                identities.add(materialized.manifest.materialized_physical_id)
                assert graph._graph_stats[0]["last_path"] == "cuda_exact_replay"
        del plans, frame, graph, materialized, context
        gc.collect()
        assert not tuple(owner._plans)
    assert len(identities) == 4
    assert all(reference() is None for reference in resources)
    owner.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_matmul_public_search_resolves_frozen_choices_without_heuristics(monkeypatch):
    from taichi_forge.hardware._cublaslt_algorithms import _AlgorithmApi

    operation = _operation(monkeypatch)
    artifact = operation.prepare(heuristic_limit=2)
    definition = _freeze(operation)
    bindings, host = _inputs(operation)
    expected = np.maximum(_product(host) + 0.25 * host[2], 0)
    observed = []

    def evaluate(graph, recipe):
        bindings["output"].from_numpy(host[2])
        graph.run(graph.bind(bindings))
        np.testing.assert_allclose(
            bindings["output"].to_numpy(), expected, rtol=2e-5, atol=2e-5
        )
        observed.append(recipe.recipe_id)
        # Actual structural objective, not fabricated device-time evidence.
        return {"dispatches": float(graph.execution_stats().dispatch_count)}

    with monkeypatch.context() as frozen:
        frozen.setattr(_AlgorithmApi, "shortlist", _no_cold_work)
        decision = definition.search_recipes(
            engine="compileiq",
            providers=_providers(),
            target=ti.graph.GraphOptimizationTarget(
                objectives=(("dispatches", "min"),)
            ),
            budget=ti.graph.GraphSearchBudget(
                evaluation_limit=2 * len(artifact["choices"])
            ),
            workload_context=ti.graph.GraphWorkloadContext(
                {"fixture": "matmul-mutable-batch"}
            ),
            evaluation_contract=ti.graph.GraphEvaluationContract(
                {"metric": "actual-dispatches-not-performance"}
            ),
            backend_environment=ti.graph.GraphBackendEnvironment(
                {"fixture": "current-cuda"}
            ),
        ).run(evaluate)
        assert len(set(observed)) >= len(artifact["choices"])
        assert len(observed) == decision.report.to_dict()["search"]["evaluation_count"]
        assert decision.selection is not None, decision.report.to_dict()["search"]
        assert not decision.selection.manifest.is_baseline
        restored_operation = _operation(
            monkeypatch, preparation=json.loads(json.dumps(artifact))
        )
        assert (
            restored_operation.preparation_artifact()["preparation"]["origin"]
            == "imported_preparation_not_current_measurement"
        )
        restored = _freeze(restored_operation)
        assert restored.semantic_graph_id == definition.semantic_graph_id
        resolved = restored.resolve_recipe(
            decision.selection_artifact, providers=_providers()
        )
        with restored.materialize(resolved, providers=_providers()) as materialized:
            bindings["output"].from_numpy(host[2])
            materialized.executor.run(bindings)
            np.testing.assert_allclose(
                bindings["output"].to_numpy(), expected, rtol=2e-5, atol=2e-5
            )
    report = ti.graph.GraphOptimizationReportV2.from_json(decision.report.to_json())
    assert report.to_dict() == decision.report.to_dict()
    assert "frozen_config" in report.to_json()
    assert "matmul-physical:" in report.to_json()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_matmul_rejects_drift_and_public_alias_before_replay(monkeypatch):
    operation = _operation(monkeypatch, size=(8, 8, 8), batch=1)
    artifact = operation.prepare(heuristic_limit=1)
    for field in ("semantics", "device", "component"):
        corrupt = json.loads(json.dumps(artifact))
        corrupt[field] = {}
        with pytest.raises(ValueError, match="drifted"):
            _operation(monkeypatch, size=(8, 8, 8), batch=1, preparation=corrupt)
    corrupt = json.loads(json.dumps(artifact))
    corrupt["choices"][artifact["baseline"]]["algorithm"]["workspace_bytes"] += 256
    with pytest.raises(ValueError, match="identity drifted"):
        _operation(monkeypatch, size=(8, 8, 8), batch=1, preparation=corrupt)
    definition = _freeze(operation)
    bindings, _ = _inputs(operation)
    for recipe, _ in _recipes(definition):
        with definition.materialization_context(providers=_providers()) as context:
            with context.materialize(recipe) as materialized:
                with pytest.raises(RuntimeError, match="alias"):
                    materialized.executor.bind({**bindings, "output": bindings["a"]})
                with pytest.raises(RuntimeError):
                    materialized.executor.bind(
                        {**bindings, "a": ti.ndarray(ti.f32, shape=(4, 16))}
                    )


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_matmul_composes_with_immutable_frames_and_retains_feedback(monkeypatch):
    from taichi_forge.hardware._cublaslt_capture import _BINDING_FRAMES_SUPPORTED
    from taichi_forge.hardware._cublaslt_algorithms import _AlgorithmApi
    from taichi_forge.linalg._matmul import _MatmulRecording

    if not _BINDING_FRAMES_SUPPORTED:
        pytest.skip("native runtime omits matmul immutable binding-frame support")
    operation = _operation(monkeypatch)
    artifact = operation.prepare(heuristic_limit=1)
    definition = _freeze(operation)
    catalog = definition.recipe_catalog(providers=_providers())
    frame_fragment = next(
        f for f in catalog.fragments if f.provider_namespace.endswith(".binding_frames")
    )
    matmul_fragment = next(
        f
        for f in catalog.fragments
        if f.provider_namespace.endswith(".matmul")
        and artifact["choices"][
            f.provider_metadata["family_selection"]["materialization_choice"]
        ]["packed_inputs"]
        and artifact["choices"][
            f.provider_metadata["family_selection"]["materialization_choice"]
        ]["epilogue"]
        == "fused"
    )
    recipe = catalog.compose(
        (frame_fragment.fragment_id, matmul_fragment.fragment_id), stage="composed"
    ).recipe
    pairs = [_inputs(operation) for _ in range(2)]
    with definition.materialization_context(providers=_providers()) as context:
        with context.materialize(recipe) as materialized:
            graph = materialized.executor
            frames = [graph.bind(bindings) for bindings, _ in pairs]
            operation.close()
            for bindings, host in pairs:
                np.testing.assert_array_equal(bindings["output"].to_numpy(), host[2])
            expected = [host[2].astype(np.float64) for _, host in pairs]
            with monkeypatch.context() as replay:
                replay.setattr(_AlgorithmApi, "restore", _no_cold_work)
                replay.setattr(
                    _MatmulRecording, "validate_graph_bindings", _no_cold_work
                )
                for index in (0, 1, 0, 0, 1):
                    graph.run(frames[index])
                    expected[index] = np.maximum(
                        _product(pairs[index][1]) + 0.25 * expected[index], 0
                    )
            for (bindings, _), value in zip(pairs, expected):
                np.testing.assert_allclose(
                    bindings["output"].to_numpy(), value, rtol=2e-5, atol=2e-5
                )
            assert graph._graph_stats[0]["last_path"] == "cuda_prepared_binding_plan"
