"""Complete contraction dataflow, search and reuse contracts on CUDA."""

import gc
import json
import os
import weakref

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


def _operation(monkeypatch, *, batch=False, preparation=None):
    path = os.environ.get("TI_FORGE_TEST_CUTENSOR_LIBRARY_PATH")
    if not path:
        pytest.skip(
            "set TI_FORGE_TEST_CUTENSOR_LIBRARY_PATH for contraction recipe tests"
        )
    monkeypatch.setenv("TI_CUTENSOR_LIBRARY_PATH", path)
    shapes = ((2, 19, 5, 7), (2, 11, 19, 3)) if batch else ((19, 5, 7), (3, 19, 11))
    modes = ("bkmi", "bnkj", "bmijn") if batch else ("kmi", "jkn", "imjn")
    return ti.linalg.record_contraction(
        shapes[0],
        modes[0],
        shapes[1],
        modes[1],
        modes[2],
        alpha=0.75,
        beta=0.25,
        activation="identity" if batch else "relu",
        compute="tf32" if batch else "f32",
        absolute_tolerance=3e-4 if batch else 3e-5,
        relative_tolerance=3e-3 if batch else 3e-5,
        preparation=preparation,
    )


def _providers():
    return (
        *ti.graph.default_recipe_providers(),
        ti.hardware.tensor.ContractionRecipeProvider(),
    )


def _freeze(operation):
    builder = ti.graph.GraphBuilder()
    builder.append_native(operation)
    return builder.freeze()


def _inputs(operation):
    semantics = operation.semantics
    random = np.random.default_rng(318)
    host = [
        random.standard_normal(semantics[name]["shape"]).astype(np.float32) * 0.1
        for name in ("a_tensor", "b_tensor", "out")
    ]
    arrays = [ti.ndarray(ti.f32, shape=x.shape) for x in host]
    for array, x in zip(arrays, host):
        array.from_numpy(x)
    expression = "bkmi,bnkj->bmijn" if len(host[0].shape) == 4 else "kmi,jkn->imjn"
    product = 0.75 * np.einsum(
        expression, host[0].astype(np.float64), host[1].astype(np.float64)
    )
    return dict(a=arrays[0], b=arrays[1], c=arrays[2], output=arrays[2]), host, product


def _no_cold_work(*args, **kwargs):
    raise AssertionError("replay/restoration repeated forbidden preparation work")


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("batch", (False, True))
def test_contraction_complete_dataflows_refresh_feedback_and_release(
    batch, monkeypatch
):
    from taichi_forge.hardware import _cutensor
    from taichi_forge.linalg._contraction import _ContractionRecording

    operation = _operation(monkeypatch, batch=batch)
    with monkeypatch.context() as preparation:
        preparation.setattr(_cutensor, "ScalarNdarray", _no_cold_work)
        artifact = operation.prepare()
    assert len(artifact["choices"]) == 8
    owner = operation._provider_owner
    assert not tuple(owner._plans)
    definition = _freeze(operation)
    catalog = definition.recipe_catalog(providers=_providers())
    frame_fragment = next(
        f for f in catalog.fragments if f.provider_namespace.endswith(".binding_frames")
    )
    choices = [(catalog.baseline.recipe, artifact["baseline"])]
    choices.extend(
        (
            catalog.compose(
                (f.fragment_id, frame_fragment.fragment_id), stage="test"
            ).recipe,
            f.provider_metadata["family_selection"]["materialization_choice"],
        )
        for f in catalog.fragments
        if f.provider_namespace.endswith(".contraction")
    )
    inputs, host, product = _inputs(operation)
    numeric = operation.semantics["numerical_contract"]
    tolerance = dict(
        atol=numeric["absolute_tolerance"], rtol=numeric["relative_tolerance"]
    )
    activate = (lambda x: x) if batch else (lambda x: np.maximum(x, 0))
    physical_ids, resources = set(), []
    operation.close()
    for recipe, key in choices:
        inputs["a"].from_numpy(host[0])
        inputs["output"].from_numpy(host[2])
        with definition.materialization_context(providers=_providers()) as context:
            with context.materialize(recipe) as materialized:
                graph = materialized.executor
                frame = graph.bind(inputs)
                np.testing.assert_array_equal(inputs["output"].to_numpy(), host[2])
                physical_ids.add(materialized.manifest.materialized_physical_id)
                plans = tuple(owner._plans)
                assert len(plans) == 1
                resources.extend(weakref.ref(p) for p in plans)
                assert (
                    plans[0].workspace_required_bytes
                    == artifact["choices"][key]["workspace_bytes"]
                )
                with monkeypatch.context() as replay:
                    replay.setattr(
                        _ContractionRecording, "validate_graph_bindings", _no_cold_work
                    )
                    replay.setattr(type(plans[0]), "execute", _no_cold_work)
                    for _ in range(2):
                        graph.run(frame)
                expected = activate(product + 0.25 * host[2])
                expected = activate(product + 0.25 * expected)
                np.testing.assert_allclose(
                    inputs["output"].to_numpy(), expected, **tolerance
                )
                inputs["a"].from_numpy(host[0] * 0.5)
                graph.run(frame)
                np.testing.assert_allclose(
                    inputs["output"].to_numpy(),
                    activate(product * 0.5 + 0.25 * expected),
                    **tolerance,
                )
                assert graph._graph_stats[0]["last_fallback_reason"] == "none"
        del graph, frame, plans, materialized, context
        gc.collect()
    assert len(physical_ids) == 8
    assert all(ref() is None or ref().closed for ref in resources)
    owner.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_contraction_search_resolves_without_rediscovering_dataflows(monkeypatch):
    import taichi_forge.linalg._contraction as module

    operation = _operation(monkeypatch)
    artifact = operation.prepare()
    definition = _freeze(operation)
    inputs, host, product = _inputs(operation)
    observed = []

    def evaluate(graph, recipe):
        inputs["output"].from_numpy(host[2])
        graph.run(graph.bind(inputs))
        np.testing.assert_allclose(
            inputs["output"].to_numpy(),
            np.maximum(product + 0.25 * host[2], 0),
            atol=3e-5,
            rtol=3e-5,
        )
        observed.append(recipe.recipe_id)
        return {"dispatches": float(graph.execution_stats().dispatch_count)}

    original = module._make_plan

    def selected_only(*args, **kwargs):
        assert not kwargs.get(
            "description", False
        ), "search/resume must not prepare an entire new catalog"
        return original(*args, **kwargs)

    with monkeypatch.context() as reconstruction:
        reconstruction.setattr(module, "_make_plan", selected_only)
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
                {"fixture": "contraction-feedback"}
            ),
            evaluation_contract=ti.graph.GraphEvaluationContract(
                {"metric": "actual-dispatches-not-device-time"}
            ),
            backend_environment=ti.graph.GraphBackendEnvironment(
                {"fixture": "current-cuda"}
            ),
        ).run(evaluate)
        assert decision.selection is not None, decision.report.to_dict()["search"]
        assert len(observed) == decision.report.to_dict()["search"]["evaluation_count"]
        assert len(set(observed)) >= len(artifact["choices"])
        restored_operation = _operation(
            monkeypatch, preparation=json.loads(json.dumps(artifact))
        )
        restored = _freeze(restored_operation)
        assert restored.semantic_graph_id == definition.semantic_graph_id
        resolved = restored.resolve_recipe(
            decision.selection_artifact, providers=_providers()
        )
        with restored.materialize(resolved, providers=_providers()) as materialized:
            evaluate(materialized.executor, resolved)
    report = ti.graph.GraphOptimizationReportV2.from_json(decision.report.to_json())
    assert report.to_dict() == decision.report.to_dict()
    assert "frozen_dataflow" in report.to_json() and "baselines" in report.to_json()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_contraction_rejects_public_alias_and_preparation_drift(monkeypatch):
    from taichi_forge.hardware import _cutensor
    from taichi_forge.linalg._contraction import _make_plan

    operation = _operation(monkeypatch)
    artifact = operation.prepare()
    for field in ("semantics", "device", "component"):
        corrupt = json.loads(json.dumps(artifact))
        corrupt[field] = {}
        with pytest.raises(ValueError, match="drifted"):
            _operation(monkeypatch, preparation=corrupt)
    corrupt = json.loads(json.dumps(artifact))
    corrupt["choices"][artifact["baseline"]]["workspace_bytes"] += 1
    with pytest.raises(ValueError, match="drifted"):
        _operation(monkeypatch, preparation=corrupt)
    changed = dict(artifact["choices"][artifact["baseline"]])
    changed["workspace_bytes"] -= 1
    with monkeypatch.context() as allocation:
        allocation.setattr(_cutensor, "ScalarNdarray", _no_cold_work)
        with pytest.raises(RuntimeError, match="workspace drifted"):
            _make_plan(operation._provider_owner, operation.semantics, changed)
    assert not tuple(operation._provider_owner._plans)
    with pytest.raises(ValueError, match="extents disagree"):
        ti.linalg.record_contraction(
            (2, 3),
            "ik",
            (4, 5),
            "kj",
            "ij",
            absolute_tolerance=1e-5,
            relative_tolerance=1e-5,
        )
    definition = _freeze(operation)
    inputs, _, _ = _inputs(operation)
    catalog = definition.recipe_catalog(providers=_providers())
    with definition.materialize(
        catalog.baseline.recipe, providers=_providers()
    ) as materialized:
        with pytest.raises(RuntimeError):
            materialized.executor.bind({**inputs, "output": inputs["a"]})
