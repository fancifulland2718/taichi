"""cuTENSOR capture contracts; requires an explicitly supplied vendor runtime."""

import gc
import os
import weakref

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


def _provider_or_skip():
    path = os.environ.get("TI_FORGE_TEST_CUTENSOR_LIBRARY_PATH")
    if not path:
        pytest.skip("set TI_FORGE_TEST_CUTENSOR_LIBRARY_PATH for real cuTENSOR tests")
    # A supplied but broken deployment must fail, not turn into a passing skip.
    return ti.hardware.tensor.CutensorProvider(path)


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("dimensions", ((5, 7, 19, 3, 11), (4, 64, 96, 5, 32)))
def test_cutensor_capture_feedback_and_retained_workspace(dimensions, monkeypatch):
    from taichi_forge.hardware._cutensor_capture import _ContractionCaptureRecipe

    i, m, k, j, n = dimensions
    shapes = ((i, m, k), (k, j, n), (i, m, j, n))
    provider = _provider_or_skip()
    plan = provider.contraction_plan(
        shapes[0], "imk", shapes[1], "kjn", shapes[2], "imjn", shapes[2], "imjn"
    )
    random = np.random.default_rng(173)
    host = [random.standard_normal(shape).astype(np.float32) for shape in shapes]
    a, b, output = [ti.ndarray(ti.f32, shape=shape) for shape in shapes]
    for array, value in zip((a, b, output), host):
        array.from_numpy(value)
    recording = plan.record(alpha=0.75, beta=0.25)
    builder = ti.graph.GraphBuilder()
    builder.append_native(recording)
    graph = builder.compile()
    binding = graph.bind(dict(a=a, b=b, c=output, d=output))
    # Neither graph creation nor cold capture may advance C/D feedback.
    np.testing.assert_array_equal(output.to_numpy(), host[2])

    def no_python_provider_work(*_args, **_kwargs):
        raise AssertionError(
            "retained capture replay must not call the Python provider"
        )

    with monkeypatch.context() as replay:
        replay.setattr(type(plan), "execute", no_python_provider_work)
        replay.setattr(type(plan), "_validate_lifetime", no_python_provider_work)
        replay.setattr(type(recording), "execute", no_python_provider_work)
        replay.setattr(
            _ContractionCaptureRecipe, "append_to_graph", no_python_provider_work
        )
        for _ in range(4):
            graph.run(binding)
    expected = host[2].astype(np.float64)
    product = 0.75 * np.einsum(
        "imk,kjn->imjn", host[0].astype(np.float64), host[1].astype(np.float64)
    )
    for _ in range(4):
        expected = product + 0.25 * expected
    np.testing.assert_allclose(output.to_numpy(), expected, rtol=3e-5, atol=3e-5)
    stats = graph._graph_stats[0]
    assert stats["last_path"] == "cuda_exact_replay"
    assert stats["last_fallback_reason"] == "none"
    memory = recording._graph_provider_memory_report().to_dict()
    assert memory["components"][0]["requested_bytes"] == plan.workspace_required_bytes
    assert memory["components"][1]["requested_bytes"] is None
    with pytest.raises(RuntimeError, match="capture leases"):
        plan.close()
    with pytest.raises(RuntimeError, match="plans are live"):
        provider.close()

    retained = weakref.ref(plan)
    del plan, recording, builder
    gc.collect()
    assert retained() is not None
    # Operands remain live values, not a cache of the first input contents.
    a.from_numpy(host[0] * 0.5)
    graph.run(binding)
    np.testing.assert_allclose(
        output.to_numpy(), product * 0.5 + 0.25 * expected, rtol=3e-5, atol=3e-5
    )
    plan = retained()
    del graph, binding
    gc.collect()
    assert plan._capture_leases == 0
    plan.close()
    provider.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cutensor_capture_binding_rejection_and_reset():
    provider = _provider_or_skip()
    # The output mode order differs from A's free mode order.
    plan = provider.contraction_plan(
        (8, 8), "ik", (8, 8), "kj", (8, 8), "ji", (8, 8), "ji"
    )
    recording = plan.record(beta=0.25)
    builder = ti.graph.GraphBuilder()
    builder.append_native(recording)
    graph = builder.compile()
    a, b, c, d = [ti.ndarray(ti.f32, (8, 8)) for _ in range(4)]
    values = np.arange(64, dtype=np.float32).reshape(8, 8)
    for array, value in zip(
        (a, b, c, d),
        (values, np.eye(8, dtype=np.float32), values, np.zeros_like(values)),
    ):
        array.from_numpy(value)
    for bad in (
        dict(a=a, b=b, c=c, d=a),
        dict(a=ti.ndarray(ti.f32, (4, 16)), b=b, c=c, d=d),
    ):
        with pytest.raises(RuntimeError):
            graph.run(bad)
    graph.run(dict(a=a, b=b, c=c, d=d))
    np.testing.assert_allclose(d.to_numpy(), values.T + 0.25 * values)
    ti.reset()
    assert plan.closed and provider.closed
    with pytest.raises(RuntimeError):
        graph.run(dict(a=a, b=b, c=c, d=d))


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cutensor_capture_composes_with_two_immutable_frames(monkeypatch):
    provider = _provider_or_skip()
    plan = provider.contraction_plan(
        (8, 8), "ik", (8, 8), "kj", (8, 8), "ij", (8, 8), "ij"
    )
    recording = plan.record(beta=0.5)
    builder = ti.graph.GraphBuilder()
    builder.append_native(recording)
    definition = builder.freeze()
    catalog = definition.recipe_catalog()
    fragment = next(
        f for f in catalog.fragments if f.provider_namespace.endswith(".binding_frames")
    )
    recipe = catalog.compose((fragment.fragment_id,), stage="test").recipe
    pairs = []
    for factor in (1.0, 2.0):
        a, b, output = [ti.ndarray(ti.f32, (8, 8)) for _ in range(3)]
        a.from_numpy(np.eye(8, dtype=np.float32) * factor)
        b.from_numpy(np.eye(8, dtype=np.float32))
        output.fill(0)
        pairs.append(dict(a=a, b=b, c=output, d=output))
    with definition.materialization_context() as context:
        with context.materialize(recipe) as materialized:
            graph = materialized.executor
            frames = [graph.bind(pair) for pair in pairs]
            expected = [0.0, 0.0]

            def no_replay_validation(*_args):
                raise AssertionError(
                    "immutable frame replay must not validate a provider"
                )

            with monkeypatch.context() as replay:
                replay.setattr(type(plan), "_validate_lifetime", no_replay_validation)
                for index in (0, 1, 0, 0, 1):
                    graph.run(frames[index])
                    expected[index] = (index + 1) + 0.5 * expected[index]
            for pair, factor in zip(pairs, expected):
                np.testing.assert_array_equal(
                    pair["d"].to_numpy(), np.eye(8, dtype=np.float32) * factor
                )
            assert graph._graph_stats[0]["last_path"] == "cuda_prepared_binding_plan"
    del (
        context,
        materialized,
        graph,
        frames,
        catalog,
        fragment,
        recipe,
        definition,
        builder,
        recording,
    )
    gc.collect()
    assert plan._capture_leases == 0
    plan.close()
    provider.close()
