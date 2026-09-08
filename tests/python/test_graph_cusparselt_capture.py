"""cuSPARSELt compression epochs and retained Graph capture contracts."""

import gc
import os
import weakref

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


def _provider():
    path = os.environ.get("TI_FORGE_TEST_CUSPARSELT_LIBRARY_PATH")
    if not path:
        pytest.skip(
            "set TI_FORGE_TEST_CUSPARSELT_LIBRARY_PATH for real cuSPARSELt tests"
        )
    return ti.hardware.tensor.CusparseLtProvider(path)


def _inputs(m, n, k, seed=182):
    random = np.random.default_rng(seed)
    host = [
        random.standard_normal(shape).astype(np.float16) * np.float16(0.125)
        for shape in ((m, k), (n, k), (m, n))
    ]
    host[0][:, 2::4] = 0
    host[0][:, 3::4] = 0
    arrays = [ti.ndarray(ti.f16, shape=x.shape) for x in host]
    for array, value in zip(arrays, host):
        array.from_numpy(value)
    return arrays, host


def _step(a, b, c):
    return (
        0.75 * (a.astype(np.float64) @ b.astype(np.float64).T)
        + 0.25 * c.astype(np.float64)
    ).astype(np.float16)


def _forbidden(*args, **kwargs):
    raise AssertionError("Graph replay repeated provider preparation or validation")


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize(
    "refresh,shape", ((False, (32, 48, 64)), (True, (128, 96, 256)))
)
def test_cusparselt_capture_snapshot_refresh_feedback_and_ownership(
    refresh, shape, monkeypatch
):
    from taichi_forge.hardware._cusparselt_capture import _MatmulCaptureRecipe

    provider = _provider()
    plan = provider.matmul_plan(*shape)
    (a, b, output), host = _inputs(*shape)
    if not refresh:
        with pytest.raises(RuntimeError, match=r"compress\(A\) first"):
            plan.record()
        plan.compress(a)
    recording = plan.record(a="a" if refresh else None, alpha=0.75, beta=0.25)
    builder = ti.graph.GraphBuilder()
    builder.append_native(recording)
    graph = builder.compile()
    bindings = dict(b=b, c=output, d=output)
    if refresh:
        bindings["a"] = a
    frame = graph.bind(bindings)
    np.testing.assert_array_equal(output.to_numpy(), host[2])
    with monkeypatch.context() as replay:
        replay.setattr(type(plan), "compress", _forbidden)
        replay.setattr(type(plan), "execute", _forbidden)
        replay.setattr(type(plan), "_validate_lifetime", _forbidden)
        replay.setattr(type(recording), "execute", _forbidden)
        replay.setattr(_MatmulCaptureRecipe, "append_to_graph", _forbidden)
        for _ in range(3):
            graph.run(frame)
    expected = host[2]
    for _ in range(3):
        expected = _step(host[0], host[1], expected)
    np.testing.assert_allclose(output.to_numpy(), expected, rtol=3e-3, atol=5e-4)
    # A remains a snapshot when it is not a Graph binding. B is always live.
    changed_a = np.roll(host[0], 2, axis=1) * np.float16(0.5)
    changed_b = host[1] * np.float16(0.5)
    a.from_numpy(changed_a)
    b.from_numpy(changed_b)
    graph.run(frame)
    expected = _step(changed_a if refresh else host[0], changed_b, expected)
    np.testing.assert_allclose(output.to_numpy(), expected, rtol=3e-3, atol=5e-4)
    assert graph._graph_stats[0]["last_fallback_reason"] == "none"
    report = recording._graph_provider_memory_report().to_dict()
    assert [x["requested_bytes"] for x in report["components"]] == [
        plan.compressed_bytes,
        plan.compression_buffer_bytes,
        plan.workspace_bytes,
        None,
    ]
    with pytest.raises(RuntimeError, match="capture leases"):
        plan.compress(a)
    with pytest.raises(RuntimeError, match="capture leases"):
        plan.close()
    with pytest.raises(RuntimeError, match="plans are live"):
        provider.close()
    with pytest.raises(RuntimeError, match="cannot mix"):
        plan.record(a=None if refresh else "a")
    ref = weakref.ref(plan)
    del plan, recording, builder
    gc.collect()
    assert ref() is not None
    graph.run(frame)
    plan = ref()
    del graph, frame
    gc.collect()
    assert plan._capture_leases == 0
    # The old Graph is retired; explicit re-compression establishes a new epoch.
    plan.compress(a).execute(b, output, output, alpha=0.75, beta=0.25)
    expected = _step(changed_a if refresh else host[0], changed_b, expected)
    expected = _step(changed_a, changed_b, expected)
    np.testing.assert_allclose(output.to_numpy(), expected, rtol=3e-3, atol=5e-4)
    plan.close()
    provider.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cusparselt_refresh_two_frames_do_not_share_input_snapshots(monkeypatch):
    provider = _provider()
    plan = provider.matmul_plan(32, 32, 64)
    recording = plan.record(a="a", alpha=0.75, beta=0.25)
    builder = ti.graph.GraphBuilder()
    builder.append_native(recording)
    definition = builder.freeze()
    catalog = definition.recipe_catalog()
    fragment = next(
        f for f in catalog.fragments if f.provider_namespace.endswith(".binding_frames")
    )
    recipe = catalog.compose((fragment.fragment_id,), stage="test").recipe
    inputs = [_inputs(32, 32, 64, seed) for seed in (6, 13)]
    with definition.materialize(recipe) as materialized:
        graph = materialized.executor
        frames = [
            graph.bind(dict(a=values[0], b=values[1], c=values[2], d=values[2]))
            for values, _ in inputs
        ]
        for values, host in inputs:
            np.testing.assert_array_equal(values[2].to_numpy(), host[2])
        with monkeypatch.context() as replay:
            replay.setattr(type(plan), "_validate_lifetime", _forbidden)
            replay.setattr(type(plan), "compress", _forbidden)
            replay.setattr(type(plan), "execute", _forbidden)
            for i in (0, 1, 0, 1):
                graph.run(frames[i])
        for values, host in inputs:
            expected = _step(host[0], host[1], _step(*host))
            np.testing.assert_allclose(
                values[2].to_numpy(), expected, rtol=3e-3, atol=5e-4
            )
        assert graph._graph_stats[0]["last_fallback_reason"] == "none"


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cusparselt_capture_binding_rejections_and_reset():
    provider = _provider()
    plan = provider.matmul_plan(32, 32, 32)
    builder = ti.graph.GraphBuilder()
    builder.append_native(plan.record(a="a"))
    graph = builder.compile()
    (a, b, output), host = _inputs(32, 32, 32)
    for bad in (
        dict(a=a, b=b, c=output, d=a),
        dict(a=a, b=b, c=output, d=b),
        dict(a=a, b=ti.ndarray(ti.f16, (16, 64)), c=output, d=output),
    ):
        with pytest.raises(RuntimeError):
            graph.run(bad)
    graph.run(dict(a=a, b=b, c=output, d=output))
    np.testing.assert_allclose(
        output.to_numpy(),
        host[0].astype(np.float32) @ host[1].astype(np.float32).T,
        rtol=3e-3,
        atol=5e-4,
    )
    ti.reset()
    assert plan.closed and provider.closed
    with pytest.raises(RuntimeError):
        graph.run(dict(a=a, b=b, c=output, d=output))
