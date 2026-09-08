"""Graph-owned cuDSS snapshots, numerical phases, and immutable binding frames."""

import gc
import weakref

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.hardware._cudss_capture import CudssCaptureRecording
from taichi_forge.hardware._cudss_config import configured_plan
from tests import test_utils
from tests.python.test_cudss_config import _problem


def _forbidden(*_args, **_kwargs):
    raise AssertionError("cold cuDSS provider work reached Graph replay")


def _owner():
    matrix, values, dense, library = _problem()
    plan = configured_plan(
        matrix,
        {"version": 1, "reordering": "amd", "solve": "general"},
        matrix_type="spd",
        matrix_view="full",
        library_path=library,
        _graph_owned=True,
    )
    return plan, values, dense


def _bindings(scale=1.0):
    values, rhs, solution = (
        ti.ndarray(ti.f32, 10),
        ti.ndarray(ti.f32, 4),
        ti.ndarray(ti.f32, 4),
    )
    values_np = np.array([4, -1, -1, 4, -1, -1, 4, -1, -1, 3], np.float32) * scale
    rhs_np = np.arange(1, 5, dtype=np.float32) * scale
    values.from_numpy(values_np)
    rhs.from_numpy(rhs_np)
    solution.fill(1)
    return dict(matrix_values=values, rhs=rhs, solution=solution), values_np, rhs_np


@ti.kernel
def _produce_values(
    source: ti.types.ndarray(dtype=ti.f32, ndim=1),
    matrix_values: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    for i in range(10):
        matrix_values[i] = source[i]


@ti.kernel
def _feedback(
    rhs: ti.types.ndarray(dtype=ti.f32, ndim=1),
    solution: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    for i in range(4):
        rhs[i] += solution[i] * 0.25


@pytest.mark.parametrize("phase", ["solve", "factor_solve", "refactor_solve"])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cudss_capture_snapshot_feedback_and_retained_owner(phase, monkeypatch):
    plan, template_values, dense = _owner()
    # Numerical input used to materialize the owner is a private snapshot.
    template_values.fill(0)
    recording = CudssCaptureRecording(plan, phase=phase)
    bindings, values_np, rhs_np = _bindings(1.5)
    builder = ti.graph.GraphBuilder()
    arg = lambda name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)
    if phase != "solve":
        source = ti.ndarray(ti.f32, 10)
        source.from_numpy(values_np)
        bindings["source"] = source
        bindings["matrix_values"].fill(0)  # Produced inside the first Graph.
        builder.dispatch(_produce_values, arg("source"), arg("matrix_values"))
    else:
        del bindings["matrix_values"]
    builder.dispatch(_feedback, arg("rhs"), arg("solution"))
    builder.append_native(recording)
    graph = builder.compile()
    frame = graph.bind(bindings)
    np.testing.assert_array_equal(bindings["rhs"].to_numpy(), rhs_np)
    np.testing.assert_array_equal(bindings["solution"].to_numpy(), np.ones(4))
    with monkeypatch.context() as replay:
        for name in (
            "solve",
            "refactor_solve",
            "analyze",
            "_ensure_open",
            "_configuration_report",
        ):
            replay.setattr(type(plan), name, _forbidden)
        replay.setattr(type(recording), "execute", _forbidden)
        for _ in range(3):
            graph.run(frame)
    expected = np.ones(4, np.float32)
    for _ in range(3):
        rhs_np = rhs_np + expected * 0.25
        expected = np.linalg.solve(dense * (1 if phase == "solve" else 1.5), rhs_np)
    np.testing.assert_allclose(bindings["solution"].to_numpy(), expected, rtol=2e-5)
    assert graph._graph_stats[0]["last_fallback_reason"] == "none"
    allocation = plan._configuration_report()["graph_allocator"]
    assert allocation["live_bytes"] >= allocation["snapshot_bytes"] > 0
    assert allocation["sealed_allocation_rejections"] == 0
    with pytest.raises(RuntimeError, match="retained by a Graph"):
        plan.close()
    reference = weakref.ref(plan)
    del plan, recording, builder
    gc.collect()
    assert reference() is not None
    graph.run(frame)
    ti.sync()
    plan = reference()
    del graph, frame
    gc.collect()
    assert plan._capture_leases == 0
    plan.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cudss_capture_two_binding_frames_and_memory_owner(monkeypatch):
    plan, _template_values, dense = _owner()
    recording = CudssCaptureRecording(plan, phase="refactor_solve")
    builder = ti.graph.GraphBuilder()
    builder.append_native(recording)
    definition = builder.freeze()
    catalog = definition.recipe_catalog()
    fragment = next(
        f for f in catalog.fragments if f.provider_namespace.endswith(".binding_frames")
    )
    recipe = catalog.compose((fragment.fragment_id,), stage="test").recipe
    inputs = [_bindings(scale) for scale in (0.75, 2.0)]
    with definition.materialize(recipe) as materialized:
        graph = materialized.executor
        frames = [graph.bind(bindings) for bindings, _, _ in inputs]
        with monkeypatch.context() as replay:
            replay.setattr(type(plan), "_ensure_open", _forbidden)
            replay.setattr(type(plan), "refactor_solve", _forbidden)
            replay.setattr(type(recording), "execute", _forbidden)
            for index in (0, 1, 0, 1, 0):
                graph.run(frames[index])
        for scale, (bindings, _, rhs_np) in zip((0.75, 2.0), inputs):
            np.testing.assert_allclose(
                dense * scale @ bindings["solution"].to_numpy(), rhs_np, rtol=2e-5
            )
        inputs[0][0]["matrix_values"].from_numpy(inputs[0][1] * 2)
        graph.run(frames[0])
        np.testing.assert_allclose(
            dense * 1.5 @ inputs[0][0]["solution"].to_numpy(), inputs[0][2], rtol=2e-5
        )
        assert graph._graph_stats[0]["last_fallback_reason"] == "none"
        memory = recording._graph_provider_memory_report().to_dict()
        assert [c["name"] for c in memory["components"]] == [
            "private_snapshot",
            "vendor_requested_payload",
            "driver_pool_backing_and_opaque_state",
        ]
        assert memory["components"][-1]["requested_bytes"] is None


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cudss_capture_binding_phase_isolation_and_reset():
    plan, _values, dense = _owner()
    builder = ti.graph.GraphBuilder()
    builder.append_native(CudssCaptureRecording(plan, phase="refactor_solve"))
    graph = builder.compile()
    bindings, _, rhs_np = _bindings()
    for bad in (
        dict(bindings, solution=bindings["rhs"]),
        dict(bindings, matrix_values=ti.ndarray(ti.f32, 9)),
    ):
        with pytest.raises(RuntimeError):
            graph.run(bad)
    np.testing.assert_array_equal(bindings["solution"].to_numpy(), np.ones(4))
    other = ti.graph.GraphBuilder()
    with pytest.raises(RuntimeError, match="cannot mix"):
        other.append_native(CudssCaptureRecording(plan, phase="solve"))
    # Graph ownership is disjoint from the old mutable native entry points.
    with pytest.raises(RuntimeError, match="stale or closed"):
        plan.compute()
    graph.run(bindings)
    np.testing.assert_allclose(
        dense @ bindings["solution"].to_numpy(), rhs_np, rtol=2e-5
    )
    ti.reset()
    with pytest.raises(RuntimeError):
        graph.run(bindings)
