"""Bounded device/lifetime contracts, with optional unmodified vendor execution."""

import os
import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=ti.cpu)
def test_cusolverdn_is_explicit_and_requires_cuda():
    with pytest.raises(ti.TaichiRuntimeError, match="CUDA"):
        ti.hardware.linalg.CusolverDnProvider()
    assert (
        ti.hardware.capability("linalg.cholesky.cusolverdn").to_dict()[
            "graph_integration"
        ]
        == "unsupported"
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
