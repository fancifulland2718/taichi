"""Private cuSPARSELt materializer configuration and restoration contracts."""

import gc
import json
from types import SimpleNamespace

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils
from tests.python.test_graph_cusparselt_capture import (
    _forbidden,
    _inputs,
    _provider,
    _step,
)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cusparselt_preparation_restore_and_failed_restore_cleanup(monkeypatch):
    from taichi_forge.hardware import _cusparselt
    from taichi_forge.hardware._cusparselt_config import configured_plan

    provider = _provider()
    execution = provider._execution_api
    destroyed = []

    def destroy(handle):
        destroyed.append(handle.value)
        return execution.destroy_matmul_plan(handle)

    monkeypatch.setattr(
        provider, "_execution_api", SimpleNamespace(destroy_matmul_plan=destroy)
    )
    # Describing and rejecting candidates must not allocate ndarray buffers.
    with monkeypatch.context() as preparation:
        preparation.setattr(_cusparselt, "ScalarNdarray", _forbidden)
        with configured_plan(provider, 32, 48, 64, preparation_only=True) as plan:
            facts = json.loads(json.dumps(plan._configuration))
            sizes = (
                plan.compressed_bytes,
                plan.compression_buffer_bytes,
                plan.workspace_bytes,
            )
            assert 0 <= facts["algorithm_id"] < plan._algorithm_count
            assert (
                plan._compressed_a
                is plan._compression_buffer
                is plan._workspace
                is None
            )
            for operation in (plan.compress, plan.execute, plan.record):
                with pytest.raises(RuntimeError, match="cannot execute"):
                    operation()
        assert len(destroyed) == 1
        for overrides, message in (
            (
                {"expected_configuration": dict(facts, activation="relu")},
                "configuration drifted",
            ),
            ({"expected_resources": (sizes[0] + 1, *sizes[1:])}, "resources drifted"),
        ):
            with pytest.raises(RuntimeError, match=message):
                configured_plan(provider, 32, 48, 64, configuration=facts, **overrides)
        assert len(destroyed) == 3
        with configured_plan(
            provider,
            32,
            48,
            64,
            configuration=facts,
            preparation_only=True,
            expected_configuration=facts,
            expected_resources=sizes,
        ) as restored:
            assert restored._configuration == facts
        assert len(destroyed) == 4
    assert not any(not plan.closed for plan in provider._plans)
    provider.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cusparselt_frozen_algorithms_relu_capture_and_legacy_execution(monkeypatch):
    from taichi_forge.hardware._cusparselt_config import _api, configured_plan

    provider = _provider()
    (a, b, output), host = _inputs(32, 48, 64)
    # ABI 1 remains executable with the new adapter, without querying ABI 2.
    with provider.matmul_plan(32, 48, 64) as legacy:
        legacy.compress(a).execute(b, output, output, alpha=0.75, beta=0.25)
        np.testing.assert_allclose(
            output.to_numpy(), _step(*host), rtol=3e-3, atol=5e-4
        )
    with configured_plan(provider, 32, 48, 64, preparation_only=True) as description:
        count = description._algorithm_count

    plans, graphs, frames, outputs = [], [], [], []
    for algorithm_id in range(min(count, 2)):
        with configured_plan(
            provider,
            32,
            48,
            64,
            preparation_only=True,
            configuration=dict(algorithm_id=algorithm_id, activation="relu"),
        ) as description:
            facts = json.loads(json.dumps(description._configuration))
            sizes = (
                description.compressed_bytes,
                description.compression_buffer_bytes,
                description.workspace_bytes,
            )
        plan = configured_plan(
            provider,
            32,
            48,
            64,
            configuration=facts,
            expected_configuration=facts,
            expected_resources=sizes,
        )
        target = ti.ndarray(ti.f16, (32, 48))
        target.from_numpy(host[2])
        builder = ti.graph.GraphBuilder()
        builder.append_native(plan.record(a="a", alpha=0.75, beta=0.25))
        graph = builder.compile()
        frame = graph.bind(dict(a=a, b=b, c=target, d=target))
        np.testing.assert_array_equal(target.to_numpy(), host[2])
        plans.append(plan)
        graphs.append(graph)
        frames.append(frame)
        outputs.append(target)
    assert len({id(plan._compressed_a) for plan in plans}) == len(plans)
    assert len({plan._configuration["algorithm_id"] for plan in plans}) == len(plans)
    # Config queries and Python submissions belong outside steady Graph replay.
    with monkeypatch.context() as replay:
        replay.setattr(
            provider,
            "_configured_execution_api",
            SimpleNamespace(get_configuration=_forbidden),
        )
        replay.setattr(type(plans[0]), "compress", _forbidden)
        replay.setattr(type(plans[0]), "execute", _forbidden)
        replay.setattr(type(plans[0]), "_validate_lifetime", _forbidden)
        for graph, frame in zip(graphs, frames):
            graph.run(frame)
        changed_a = np.roll(host[0], 2, axis=1) * np.float16(0.5)
        a.from_numpy(changed_a)
        for graph, frame in zip(graphs, frames):
            graph.run(frame)
    expected = np.maximum(_step(*host), np.float16(0))
    expected = np.maximum(_step(changed_a, host[1], expected), np.float16(0))
    for target in outputs:
        np.testing.assert_allclose(target.to_numpy(), expected, rtol=3e-3, atol=5e-4)
    assert _api(provider).execution_abi_version == 2
    del builder, graph, frame, graphs, frames
    gc.collect()
    for plan in plans:
        assert plan._capture_leases == 0
        plan.close()
    provider.close()
