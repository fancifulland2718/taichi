"""Shared-current-A sparse matmul composition, reuse and recording contracts."""

import gc
import json
import os

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


def _operation(monkeypatch, *, products=2, preparation=None, shape=(32, 32, 64)):
    path = os.environ.get("TI_FORGE_TEST_CUSPARSELT_LIBRARY_PATH")
    if not path:
        pytest.skip(
            "set TI_FORGE_TEST_CUSPARSELT_LIBRARY_PATH for sparse matmul recipe tests"
        )
    monkeypatch.setenv("TI_CUSPARSELT_LIBRARY_PATH", path)
    return ti.linalg.record_sparse_matmul(
        *shape,
        products=tuple((f"b{i}", f"c{i}", f"d{i}") for i in range(products)),
        alpha=0.75,
        beta=0.25,
        activation="relu",
        absolute_tolerance=5e-4,
        relative_tolerance=3e-3,
        preparation=preparation,
    )


def _providers():
    return (
        *ti.graph.default_recipe_providers(),
        ti.hardware.tensor.SparseMatmulRecipeProvider(),
    )


def _freeze(operation):
    builder = ti.graph.GraphBuilder()
    builder.append_native(operation)
    return builder.freeze()


def _inputs(operation, seed=442):
    semantics = operation.semantics
    m, n, k = (semantics[x] for x in ("m", "n", "k"))
    random = np.random.default_rng(seed)
    host = dict(a=(random.standard_normal((m, k)) * 0.125).astype(np.float16))
    host["a"][:, 2::4] = host["a"][:, 3::4] = 0
    for b, c, _ in semantics["products"]:
        host[b] = (random.standard_normal((n, k)) * 0.125).astype(np.float16)
        host[c] = (random.standard_normal((m, n)) * 0.125).astype(np.float16)
    arrays = {
        name: ti.ndarray(ti.f16, shape=value.shape) for name, value in host.items()
    }
    for name, array in arrays.items():
        array.from_numpy(host[name])
    for _, c, d in semantics["products"]:
        arrays[d] = arrays[c]
    return arrays, host


def _expected(semantics, host, previous=None):
    return {
        d: np.maximum(
            0.75 * host["a"].astype(np.float64) @ host[b].astype(np.float64).T
            + 0.25 * (host[c] if previous is None else previous[d]).astype(np.float64),
            0,
        ).astype(np.float16)
        for b, c, d in semantics["products"]
    }


def _assert_outputs(semantics, bindings, expected):
    for _, _, d in semantics["products"]:
        np.testing.assert_allclose(
            bindings[d].to_numpy(), expected[d], atol=5e-4, rtol=3e-3
        )


def _forbidden(*args, **kwargs):
    raise AssertionError("replay/restoration performed forbidden preparation work")


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("products", (1, 2))
def test_sparse_matmul_complete_recipes_refresh_frames_and_memory(
    products, monkeypatch
):
    from taichi_forge.hardware import _cusparselt

    operation = _operation(monkeypatch, products=products)
    with monkeypatch.context() as preparation:
        preparation.setattr(_cusparselt, "ScalarNdarray", _forbidden)
        artifact = operation.prepare(max_algorithms=2)
    owner = operation._provider_owner
    definition = _freeze(operation)
    catalog = definition.recipe_catalog(providers=_providers())
    frame_fragment = next(
        f for f in catalog.fragments if f.provider_namespace.endswith(".binding_frames")
    )
    recipes = [(catalog.baseline.recipe, artifact["baseline"])]
    recipes.extend(
        (
            catalog.compose(
                (f.fragment_id, frame_fragment.fragment_id), stage="test"
            ).recipe,
            f.provider_metadata["family_selection"]["materialization_choice"],
        )
        for f in catalog.fragments
        if f.provider_namespace.endswith(".sparse_matmul")
    )
    assert len(recipes) == len(artifact["choices"])
    semantics = operation.semantics
    inputs = [_inputs(operation, seed) for seed in (41, 97)]
    identities = set()
    operation.close()
    for recipe, key in recipes:
        for bindings, host in inputs:
            for name, value in host.items():
                bindings[name].from_numpy(value)
        with definition.materialize(recipe, providers=_providers()) as materialized:
            graph = materialized.executor
            frames = [graph.bind(bindings) for bindings, _ in inputs]
            plans = tuple(owner._plans)
            assert len(plans) == 1  # Not one retained plan per product.
            assert plans[0]._capture_leases == 1
            assert [
                plans[0].compressed_bytes,
                plans[0].compression_buffer_bytes,
                plans[0].workspace_bytes,
            ] == artifact["choices"][key]["resources"]
            reports = graph._spec.provider_memory_reports()
            assert len(reports) == 1
            known = [
                x["requested_bytes"]
                for x in reports[0].to_dict()["components"]
                if x["requested_bytes"] is not None
            ]
            assert sum(known) == sum(artifact["choices"][key]["resources"])
            for bindings, host in inputs:
                for _, c, d in semantics["products"]:
                    np.testing.assert_array_equal(bindings[d].to_numpy(), host[c])
            with monkeypatch.context() as replay:
                replay.setattr(type(plans[0]), "compress", _forbidden)
                replay.setattr(type(plans[0]), "execute", _forbidden)
                replay.setattr(type(plans[0]), "_validate_lifetime", _forbidden)
                for index in (0, 1, 0, 1):
                    graph.run(frames[index])
            for frame_index, (bindings, host) in enumerate(inputs):
                expected = _expected(semantics, host, _expected(semantics, host))
                _assert_outputs(semantics, bindings, expected)
                changed = {**host, "a": np.roll(host["a"], 2, axis=1) * np.float16(0.5)}
                bindings["a"].from_numpy(changed["a"])
                graph.run(frames[frame_index])
                _assert_outputs(
                    semantics, bindings, _expected(semantics, changed, expected)
                )
            identities.add(materialized.manifest.materialized_physical_id)
            assert graph._graph_stats[0]["last_fallback_reason"] == "none"
        del graph, frames, plans, materialized
        gc.collect()
        assert not tuple(owner._plans)
    assert len(identities) == len(artifact["choices"])
    owner.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_sparse_matmul_search_and_frozen_selection_restore(monkeypatch):
    import taichi_forge.linalg._sparse_matmul as module

    operation = _operation(monkeypatch)
    artifact = operation.prepare(max_algorithms=2)
    definition = _freeze(operation)
    inputs, host = _inputs(operation)
    observed = []

    def evaluate(graph, recipe):
        for _, c, d in operation.semantics["products"]:
            inputs[d].from_numpy(host[c])
        graph.run(graph.bind(inputs))
        _assert_outputs(
            operation.semantics, inputs, _expected(operation.semantics, host)
        )
        observed.append(recipe.recipe_id)
        return {"dispatches": float(graph.execution_stats().dispatch_count)}

    original = module._plan

    def selected_only(*args, **kwargs):
        assert not kwargs.get("description", False)
        return original(*args, **kwargs)

    with monkeypatch.context() as restoration:
        restoration.setattr(module, "_plan", selected_only)
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
                {"fixture": "shared-current-A"}
            ),
            evaluation_contract=ti.graph.GraphEvaluationContract(
                {"metric": "actual-dispatches-not-device-time"}
            ),
            backend_environment=ti.graph.GraphBackendEnvironment(
                {"fixture": "current-cuda"}
            ),
        ).run(evaluate)
        assert decision.selection is not None, decision.report.to_dict()["search"]
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
def test_sparse_matmul_rejects_cross_product_alias_and_contract_drift(monkeypatch):
    operation = _operation(monkeypatch, shape=(32, 32, 32))
    artifact = operation.prepare(max_algorithms=1)
    for field in ("semantics", "component", "device"):
        invalid = json.loads(json.dumps(artifact))
        invalid[field] = {}
        with pytest.raises(ValueError, match="drifted"):
            _operation(monkeypatch, shape=(32, 32, 32), preparation=invalid)
    invalid = json.loads(json.dumps(artifact))
    invalid["choices"][artifact["baseline"]]["resources"][0] += 1
    with pytest.raises(ValueError, match="drifted"):
        _operation(monkeypatch, shape=(32, 32, 32), preparation=invalid)
    definition = _freeze(operation)
    inputs, host = _inputs(operation)
    baseline = definition.recipe_catalog(providers=_providers()).baseline.recipe
    with definition.materialize(baseline, providers=_providers()) as materialized:
        graph = materialized.executor
        for alias in ("a", "b0", "b1", "c1", "d1"):
            with pytest.raises(RuntimeError):
                # Native validation is at first capture, before any math; bind
                # alone need not instantiate a CUDA Graph executable.
                graph.run(graph.bind({**inputs, "d0": inputs[alias]}))
            for name, value in host.items():
                np.testing.assert_array_equal(inputs[name].to_numpy(), value)
        graph.run(graph.bind(inputs))
        _assert_outputs(
            operation.semantics, inputs, _expected(operation.semantics, host)
        )
