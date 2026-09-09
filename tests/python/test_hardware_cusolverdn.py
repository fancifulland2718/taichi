"""Bounded device/lifetime contracts, with optional unmodified vendor execution."""

import os
from types import SimpleNamespace
import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


@pytest.mark.skipif(
    not os.environ.get("TI_FORGE_TEST_CUSOLVERDN_LIBRARY_PATH"),
    reason="real cuSOLVER required",
)
@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("dtype,alias", [(ti.f32, False), (ti.f64, True)])
def test_cusolverdn_capture_live_inputs_root_order_and_retirement(
    dtype, alias, monkeypatch
):
    n = 65
    a, rhs = ti.ndarray(dtype, (n, n)), ti.ndarray(dtype, (3, n))
    output = rhs if alias else ti.ndarray(dtype, (3, n))
    with ti.hardware.linalg.CusolverDnProvider(
        os.environ["TI_FORGE_TEST_CUSOLVERDN_LIBRARY_PATH"]
    ) as provider:
        with provider.cholesky_plan(n, rhs_count=3, dtype=dtype) as plan:
            binding = plan.bind(a, rhs, output)
            with pytest.raises(ti.TaichiRuntimeError, match="factor"):
                binding.capture()
            action = binding.capture(mode="factor_and_solve")
            np.testing.assert_array_equal(plan.info.to_numpy(), [-1, -1])
            builder = ti.graph.GraphBuilder()
            builder.append_native(action.record())
            graph = builder.compile()
            published = graph.bind({"a": a, "rhs": rhs, "solution": output})
            with pytest.raises(ti.TaichiRuntimeError, match="original"):
                graph.bind({"a": a, "rhs": rhs, "solution": ti.ndarray(dtype, (3, n))})

            def cold_only(*args):
                raise AssertionError("vendor or capture called during replay")

            monkeypatch.setattr(binding, "_invoke", cold_only)
            for diagonal in (2, 4, 8):
                host_a = (
                    np.eye(n, dtype=np.float32 if dtype == ti.f32 else np.float64)
                    * diagonal
                )
                a.from_numpy(host_a)
                rhs.fill(diagonal * 3)
                graph.run(published)
                np.testing.assert_allclose(output.to_numpy(), 3, rtol=1e-5)
                np.testing.assert_array_equal(a.to_numpy(), host_a)
                assert plan.status()["solve_ok"]
            action.close()
            with pytest.raises(ti.TaichiRuntimeError, match="closed"):
                action.run()

    # Saved entry points must be invalidated before resetting owned CUDA context.
    provider = ti.hardware.linalg.CusolverDnProvider(
        os.environ["TI_FORGE_TEST_CUSOLVERDN_LIBRARY_PATH"]
    )
    plan = provider.cholesky_plan(n, rhs_count=3, dtype=dtype)
    action = plan.bind(a, rhs, output).capture(mode="factor_and_solve")
    saved = action.run
    ti.reset()
    with pytest.raises(ti.TaichiRuntimeError, match="closed"):
        saved()


@test_utils.test(arch=ti.cpu)
def test_cusolverdn_public_probe_preserves_facts_and_does_not_enable(monkeypatch):
    from taichi_forge.hardware import _cusolverdn_abi as abi

    monkeypatch.setattr(abi, "_LIBRARIES", {})
    unloaded = []
    marker = object()
    monkeypatch.setattr(abi, "_unload", unloaded.append)
    monkeypatch.setattr(
        abi,
        "DenseLibrary",
        lambda path: SimpleNamespace(library=marker, path=path, version="12.1.0"),
    )

    def probe():
        return next(
            o
            for o in ti.hardware.probe(
                "cusolverdn", library_path="vendor.dll"
            ).to_dict()["operations"]
            if o["provider_id"] == "cusolverdn"
        )

    result = probe()
    assert result["discovery"] == "available"
    assert result["provider_version"] == "12.1.0"
    assert (
        result["enablement"] == "disabled" and result["selection"] == "not_considered"
    )
    assert result["native_facts"]["external_component_probed"]
    assert not result["native_facts"]["execution_qualified"]
    assert unloaded == [marker] and not abi.passive_status()["library_loaded"]

    def missing(path):
        raise OSError("missing vendor dependency")

    monkeypatch.setattr(abi, "DenseLibrary", missing)
    failure = probe()
    assert failure["unavailable_reason"] == "vendor_runtime_probe_failed"
    assert "missing vendor dependency" in failure["last_error"]
    assert not abi.passive_status()["library_loaded"]


@test_utils.test(arch=ti.cpu)
def test_cusolverdn_is_explicit_and_requires_cuda():
    with pytest.raises(ti.TaichiRuntimeError, match="CUDA"):
        ti.hardware.linalg.CusolverDnProvider()
    assert (
        ti.hardware.capability("linalg.cholesky.cusolverdn").to_dict()[
            "graph_integration"
        ]
        == "root_ordered"
    )


@pytest.mark.skipif(
    not os.environ.get("TI_FORGE_TEST_CUSOLVERDN_LIBRARY_PATH"),
    reason="explicit real cuSOLVER library required",
)
@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("dtype,rhs_count", [(ti.f32, 1), (ti.f32, 4), (ti.f64, 4)])
def test_cusolverdn_live_device_data_reuse_failure_and_retirement(
    dtype, rhs_count, monkeypatch
):
    from taichi_forge.hardware import _cusolverdn

    n = 64
    a = ti.ndarray(dtype, (n, n))
    shape = (n,) if rhs_count == 1 else (rhs_count, n)
    rhs, output = ti.ndarray(dtype, shape), ti.ndarray(dtype, shape)
    consumed = ti.ndarray(dtype, shape)
    expected = np.arange(n, dtype=np.float64) / 64 + 1
    expected = expected if rhs_count == 1 else np.broadcast_to(expected, shape)

    @ti.kernel
    def produce(a: ti.types.ndarray(), diagonal: ti.f64):
        for i, j in a:
            a[i, j] = 0
            if i == j:
                a[i, j] = diagonal
            if i == j + 1:
                a[i, j] = -0.25
            if i < j:
                a[i, j] = 1000  # ignored upper triangle must not affect the solve

    @ti.kernel
    def consume(source: ti.types.ndarray(), target: ti.types.ndarray()):
        for index in ti.grouped(source):
            target[index] = 2 * source[index]

    with ti.hardware.linalg.CusolverDnProvider(
        os.environ["TI_FORGE_TEST_CUSOLVERDN_LIBRARY_PATH"]
    ) as provider:
        with pytest.raises(ValueError):
            provider.cholesky_plan(True)
        with pytest.raises(TypeError):
            provider.cholesky_plan(n, dtype=ti.i32)
        with provider.cholesky_plan(n, rhs_count=rhs_count, dtype=dtype) as plan:
            binding = plan.bind(a, rhs, output)
            stale_call = binding.factor_and_solve
            with pytest.raises(ti.TaichiRuntimeError, match="factor"):
                binding.solve()
            with pytest.raises(ti.TaichiRuntimeError, match="plans"):
                provider.close()
            with pytest.raises(ti.TaichiRuntimeError, match="one immutable"):
                plan.bind(a, rhs, output)

            def cold_only(*args):
                raise AssertionError("repeated cold pointer validation")

            monkeypatch.setattr(_cusolverdn, "_pointer", cold_only)
            for iteration in range(3):
                diagonal = 2 + iteration
                produce(a, diagonal)
                matrix = (
                    np.diag(np.full(n, diagonal))
                    + np.diag(np.full(n - 1, -0.25), 1)
                    + np.diag(np.full(n - 1, -0.25), -1)
                )
                rhs.from_numpy(
                    (expected @ matrix).astype(
                        np.float32 if dtype == ti.f32 else np.float64
                    )
                )
                if iteration == 1:
                    binding.factor()
                    assert plan.status()["solve_info"] == -1
                    binding.solve()
                else:
                    binding.factor_and_solve()
                consume(output, consumed)  # ordered, no explicit host wait
                tol = 2e-5 if dtype == ti.f32 else 1e-12
                np.testing.assert_allclose(
                    consumed.to_numpy(), 2 * expected, rtol=tol, atol=tol
                )
                assert plan.status()["solve_ok"]
                # Fresh RHS, unchanged factors: reuse must not refactor or copy A.
                rhs.from_numpy(
                    (3 * expected @ matrix).astype(
                        np.float32 if dtype == ti.f32 else np.float64
                    )
                )
                binding.solve()
                np.testing.assert_allclose(
                    output.to_numpy(), 3 * expected, rtol=tol, atol=tol
                )
            produce(a, -1)
            binding.factor()
            assert plan.status()["factor_info"] > 0
            assert not plan.status()["solve_ok"]
            assert (
                plan.memory_report().known_resident_requested_bytes
                >= n * n * (4 if dtype == ti.f32 else 8) + 8
            )
        with pytest.raises(ti.TaichiRuntimeError, match="closed"):
            stale_call()
        assert plan.memory_report().known_resident_requested_bytes == 0


@pytest.mark.skipif(
    not os.environ.get("TI_FORGE_TEST_CUSOLVERDN_LIBRARY_PATH"),
    reason="explicit real cuSOLVER library required",
)
def test_cusolverdn_reset_invalidates_cached_actions_and_alias_binding():
    ti.init(arch=ti.cuda, offline_cache=False)
    provider = ti.hardware.linalg.CusolverDnProvider(
        os.environ["TI_FORGE_TEST_CUSOLVERDN_LIBRARY_PATH"]
    )
    plan = provider.cholesky_plan(32)
    a = ti.ndarray(ti.f32, (32, 32))
    rhs = ti.ndarray(ti.f32, 32)
    a.from_numpy(np.eye(32, dtype=np.float32) * 2)
    rhs.fill(3)
    binding = plan.bind(a, rhs, rhs)
    binding.factor_and_solve()
    np.testing.assert_allclose(rhs.to_numpy(), 1.5)
    action = binding.solve
    binding.solve()  # remains queued when reset starts
    ti.reset()
    assert provider.closed and plan.closed
    with pytest.raises(ti.TaichiRuntimeError, match="closed"):
        action()
