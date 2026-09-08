"""Sparse-solve semantic lifecycles, shared factors, search and recovery."""

import gc
import json
import os

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


def _problem(kind="spd", n=9):
    library = os.environ.get("TI_CUDSS_TEST_LIBRARY")
    if not library:
        pytest.skip("set TI_CUDSS_TEST_LIBRARY to enable sparse-solve recipe contracts")
    dense = np.diag(np.linspace(4, 6, n, dtype=np.float32))
    for i in range(n - 1):
        dense[i, i + 1] = -0.6 if kind == "general" else -1
        dense[i + 1, i] = -1
    rows, columns, numbers = [0], [], []
    for row in dense:
        indices = np.flatnonzero(row)
        columns.extend(indices)
        numbers.extend(row[indices])
        rows.append(len(columns))
    row, col, values = (
        ti.ndarray(ti.i32, n + 1),
        ti.ndarray(ti.i32, len(columns)),
        ti.ndarray(ti.f32, len(numbers)),
    )
    row.from_numpy(np.array(rows, np.int32))
    col.from_numpy(np.array(columns, np.int32))
    numbers = np.array(numbers, np.float32)
    values.from_numpy(numbers)
    return ti.linalg.SparsePattern.csr(n, n, row, col), values, dense, numbers, library


def _operation(problem, kind="spd", fixed=False, preparation=None):
    pattern, values, _, _, library = problem
    return ti.linalg.record_sparse_solve(
        pattern,
        values,
        values=None if fixed else "values",
        rhs_pairs=(("b0", "x0"), ("b1", "x1"), ("b2", "x2")),
        matrix_type=kind,
        matrix_view="full",
        absolute_tolerance=2e-5,
        relative_tolerance=2e-5,
        library_path=library,
        preparation=preparation,
    )


def _providers():
    return (
        *ti.graph.default_recipe_providers(),
        ti.hardware.linalg.SparseSolveRecipeProvider(),
    )


@ti.kernel
def _produce_values(
    source: ti.types.ndarray(ti.f32, ndim=1), values: ti.types.ndarray(ti.f32, ndim=1)
):
    for i in source:
        values[i] = source[i]


def _freeze(operation, producer=False):
    builder = ti.graph.GraphBuilder()
    if producer:
        arg = lambda n: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, n, ti.f32, ndim=1)
        builder.dispatch(_produce_values, arg("source"), arg("values"))
    builder.append_native(operation)
    return builder.freeze()


def _inputs(problem, scale=1.0, fixed=False, producer=False):
    _, _, dense, numbers, _ = problem
    n = dense.shape[0]
    bindings, host = {}, {}
    for i in range(3):
        host[f"b{i}"] = np.arange(1, n + 1, dtype=np.float32) * (i + 1) * scale
        bindings[f"b{i}"] = ti.ndarray(ti.f32, n)
        bindings[f"b{i}"].from_numpy(host[f"b{i}"])
        bindings[f"x{i}"] = ti.ndarray(ti.f32, n)
        bindings[f"x{i}"].fill(-7)
    if not fixed:
        bindings["values"] = ti.ndarray(ti.f32, len(numbers))
        bindings["values"].from_numpy(numbers * scale)
        if producer:
            bindings["source"] = ti.ndarray(ti.f32, len(numbers))
            bindings["source"].from_numpy(numbers * scale)
            bindings["values"].fill(0)
    return bindings, host


def _assert(problem, bindings, host, scale):
    dense = problem[2] * scale
    for i in range(3):
        residual = np.linalg.norm(
            dense @ bindings[f"x{i}"].to_numpy() - host[f"b{i}"], ord=np.inf
        )
        assert residual <= 2e-5 + 2e-5 * np.linalg.norm(host[f"b{i}"], ord=np.inf)


def _forbidden(*args, **kwargs):
    raise AssertionError("steady replay called cold solver preparation")


@pytest.mark.parametrize(
    "kind,fixed", (("spd", False), ("general", False), ("spd", True))
)
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_sparse_solve_lifecycles_shared_rhs_and_binding_frames(
    kind, fixed, monkeypatch
):
    from taichi_forge.hardware._linalg import CudssPlan

    problem = _problem(kind)
    operation = _operation(problem, kind, fixed)
    artifact = operation.prepare(max_plans=2)
    for observation in artifact["preparation"]["observations"]:
        statistics = observation["preparation_factor_statistics"]
        assert (
            statistics["scope"]
            == "initial_private_graph_snapshot_not_current_replay_values"
        )
        assert statistics["source"] == "vendor_query_not_gpu_counters"
        if statistics["adapter_abi"] == 1:
            assert statistics["collection_count"] == 1
            assert statistics["status"] == "available"
            assert statistics["lu_nonzeros"]["value"] >= problem[2].shape[0]
            assert statistics["superpanels"]["written_bytes"] == 4
            assert statistics["factor_flops"]["written_bytes"] == 8
        else:
            assert statistics["status"] == "unavailable"
            assert statistics["collection_count"] == 0
    definition = _freeze(operation, producer=not fixed)
    catalog = definition.recipe_catalog(providers=_providers())
    frame_fragment = next(
        f for f in catalog.fragments if f.provider_namespace.endswith(".binding_frames")
    )
    recipes = [(catalog.baseline.recipe, artifact["baseline"])]
    recipes += [
        (
            catalog.compose(
                (f.fragment_id, frame_fragment.fragment_id), stage="contract"
            ).recipe,
            f.provider_metadata["family_selection"]["materialization_choice"],
        )
        for f in catalog.fragments
        if f.provider_namespace.endswith(".sparse_solve")
    ]
    assert len(recipes) == (2 if fixed else 4)
    # Both the caller's initial values and the operation shell may disappear.
    # Frozen sources retain a distinct numeric snapshot, not mutable inputs.
    problem[1].fill(0)
    operation.close()
    physical_ids = set()
    for recipe, key in recipes:
        inputs = [_inputs(problem, scale, fixed, not fixed) for scale in (0.75, 1.8)]
        with definition.materialize(recipe, providers=_providers()) as materialized:
            graph = materialized.executor
            frames = [graph.bind(b) for b, _ in inputs]
            resident = (
                artifact["choices"][key]["capture_parameter_storage"]
                == "device_resident_per_binding"
            )
            before_stats = graph.execution_stats().memory.persistent_bytes
            for b, _ in inputs:
                for i in range(3):
                    np.testing.assert_array_equal(b[f"x{i}"].to_numpy(), np.full(9, -7))
            with monkeypatch.context() as replay:
                for method in (
                    "_ensure_open",
                    "_configuration_report",
                    "solve",
                    "refactor_solve",
                ):
                    replay.setattr(CudssPlan, method, _forbidden)
                for index in (0, 1, 0, 1):
                    graph.run(frames[index])
            for scale, (b, host) in zip((0.75, 1.8), inputs):
                _assert(problem, b, host, 1 if fixed else scale)
            if resident:
                after_stats = graph.execution_stats().memory.persistent_bytes
                assert after_stats > 0
                # Ordinary baseline captures lazily on its first run; immutable
                # frame recipes already own all images after bind.
                if before_stats:
                    assert after_stats == before_stats
            if not fixed:
                inputs[0][0]["source"].from_numpy(problem[3] * 1.25)
                graph.run(frames[0])
                _assert(problem, *inputs[0], 1.25)
            reports = graph._spec.provider_memory_reports()
            assert len(reports) == 1
            known = sum(
                c.requested_bytes
                for c in reports[0].components
                if c.requested_bytes is not None
            )
            assert known == sum(
                artifact["choices"][key]["resources"].values()
            ) + 4 * len(problem[3])
            assert any(c.requested_bytes is None for c in reports[0].components)
            physical_ids.add(materialized.manifest.materialized_physical_id)
        del graph, frames, materialized
        gc.collect()
    assert len(physical_ids) == len(recipes)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_sparse_solve_public_search_and_selection_restore(monkeypatch):
    import taichi_forge.linalg._sparse_solve as module

    problem = _problem()
    operation = _operation(problem)
    artifact = operation.prepare(max_plans=2)
    definition, inputs = _freeze(operation), _inputs(problem)
    observed = set()

    def evaluate(graph, recipe):
        graph.run(graph.bind(inputs[0]))
        _assert(problem, *inputs, 1)
        observed.add(recipe.recipe_id)
        reports = graph._spec.provider_memory_reports()
        return {
            "requested_device_bytes": float(
                sum(c.requested_bytes or 0 for r in reports for c in r.components)
            )
        }

    decision = definition.search_recipes(
        engine="compileiq",
        providers=_providers(),
        target=ti.graph.GraphOptimizationTarget(
            objectives=(("requested_device_bytes", "min"),)
        ),
        budget=ti.graph.GraphSearchBudget(
            evaluation_limit=2 * len(artifact["choices"])
        ),
        workload_context=ti.graph.GraphWorkloadContext(
            {"fixture": "shared-pattern-three-rhs"}
        ),
        evaluation_contract=ti.graph.GraphEvaluationContract(
            {"metric": "known-requested-payload-not-resident-vram"}
        ),
        backend_environment=ti.graph.GraphBackendEnvironment(
            {"fixture": "current-cuda"}
        ),
    ).run(evaluate)
    assert decision.selection is not None, decision.report.to_dict()["search"]
    assert len(observed) >= len(artifact["choices"])
    with monkeypatch.context() as restoration:
        restoration.setattr(module, "_plan", _forbidden)
        restored_problem = _problem()
        restored_operation = _operation(
            restored_problem, preparation=json.loads(json.dumps(artifact))
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
    assert (
        "frozen_dataflow" in report.to_json()
        and "numerical_contract" in report.to_json()
    )


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_sparse_solve_semantic_drift_alias_and_reset():
    problem = _problem()
    operation = _operation(problem)
    artifact = operation.prepare(max_plans=1)
    for field in ("semantics", "device", "component", "initial_values_sha256"):
        invalid = json.loads(json.dumps(artifact))
        invalid[field] = None
        with pytest.raises(ValueError, match="drifted"):
            _operation(problem, preparation=invalid)
    invalid = json.loads(json.dumps(artifact))
    invalid["choices"][artifact["baseline"]]["numeric_phase"] = "solve"
    with pytest.raises(ValueError, match="drifted"):
        _operation(problem, preparation=invalid)
    definition = _freeze(operation)
    baseline = definition.recipe_catalog(providers=_providers()).baseline.recipe
    bindings, host = _inputs(problem)
    with definition.materialize(baseline, providers=_providers()) as materialized:
        graph = materialized.executor
        for name in ("b0", "b1", "x1"):
            with pytest.raises(RuntimeError):
                graph.run(graph.bind({**bindings, "x0": bindings[name]}))
        graph.run(graph.bind(bindings))
        _assert(problem, bindings, host, 1)
        ti.reset()
        with pytest.raises(RuntimeError):
            graph.run(bindings)
