"""FidelityFX sort: exact capacity, stability, queue order and plan ownership."""

import os

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.hardware import _parallel_sort
from tests import test_utils


def _compiler():
    path = os.environ.get("TI_TEST_PARALLEL_SORT_DXC")
    if not path:
        pytest.skip("configure TI_TEST_PARALLEL_SORT_DXC for the optional shader JIT")
    return path


@pytest.mark.parametrize(
    "with_payload,fused", [(False, True), (True, True), (True, False)]
)
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_parallel_sort_exact_capacity_and_stable_duplicate_payload(with_payload, fused):
    compiler = _compiler()
    # Both sides of the shader block boundary and a middle-scale partial block.
    for n in (1, 511, 513, 131072, 131073, 262147):
        keys = ti.ndarray(ti.u32, shape=n)
        values = ti.ndarray(ti.u32, shape=n) if with_payload else None
        with ti.hardware.sort.VulkanParallelSortPlan(
            keys, values, compiler_path=compiler, fuse_prefix=fused
        ) as plan:
            rng = np.random.default_rng(956)
            for source in (
                rng.integers(0, 16, n, dtype=np.uint32),
                rng.integers(0, 2**32, n, dtype=np.uint32),
            ):
                source[0] = 0xFFFFFFFF
                keys.from_numpy(source)
                if with_payload:
                    values.from_numpy(np.arange(n, dtype=np.uint32))
                plan.run()
                order = np.argsort(source, kind="stable")
                np.testing.assert_array_equal(keys.to_numpy(), source[order])
                if with_payload:
                    np.testing.assert_array_equal(
                        values.to_numpy(), order.astype(np.uint32)
                    )
            stats = plan.statistics()
            assert stats["dispatch_count"] == (24 if stats.get("fused_prefix") else 40)
            assert stats["barrier_count"] == stats["dispatch_count"] + 1
            assert stats["device_copy_count"] == 0
            assert (
                plan.memory_report().known_capacity_requested_bytes
                == stats["workspace_bytes"]
            )
            assert not plan.memory_report().resident_requested_bytes_complete
        with pytest.raises(RuntimeError, match="closed"):
            plan.run()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_parallel_sort_graph_publication_queue_order_and_pending_close(monkeypatch):
    n = 1048579
    keys = ti.ndarray(ti.u32, shape=n)
    values = ti.ndarray(ti.u32, shape=n)
    other = ti.ndarray(ti.u32, shape=n)
    plan = ti.hardware.sort.VulkanParallelSortPlan(
        keys, values, compiler_path=_compiler(), fuse_prefix=True
    )

    @ti.kernel
    def before(k: ti.types.ndarray(ti.u32), v: ti.types.ndarray(ti.u32)):
        for i in k:
            k[i] = ti.cast((n - 1 - i) % 257, ti.u32)
            v[i] = ti.cast(i, ti.u32)

    @ti.kernel
    def after(k: ti.types.ndarray(ti.u32)):
        for i in k:
            k[i] += 1

    builder = ti.graph.GraphBuilder()
    k = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "keys", ti.u32, ndim=1)
    v = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", ti.u32, ndim=1)
    builder.dispatch(before, k, v)
    builder.append_native(plan.record())
    builder.dispatch(after, k)
    graph = builder.compile()
    with pytest.raises(RuntimeError, match="original ndarray"):
        graph.bind({"keys": other, "values": values})
    bindings = graph.bind({"keys": keys, "values": values})

    def unexpected(*args, **kwargs):
        raise AssertionError("cold JIT/binding/lifetime/report work reached replay")

    monkeypatch.setattr(_parallel_sort, "compile_shaders", unexpected)
    monkeypatch.setattr(
        _parallel_sort._Recording, "validate_graph_bindings", unexpected
    )
    monkeypatch.setattr(plan, "validate_graph_lifetime", unexpected)
    monkeypatch.setattr(plan, "statistics", unexpected)
    monkeypatch.setattr(plan, "memory_report", unexpected)
    for _ in range(3):
        graph.run(bindings)
    plan.close()  # Queue work still owns shader pipelines and workspace.
    expected = ((n - 1 - np.arange(n)) % 257).astype(np.uint32)
    order = np.argsort(expected, kind="stable")
    np.testing.assert_array_equal(keys.to_numpy(), expected[order] + 1)
    np.testing.assert_array_equal(values.to_numpy(), order.astype(np.uint32))
    with pytest.raises(RuntimeError, match="closed"):
        graph.run(bindings)


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_parallel_sort_cold_rejections_and_runtime_reset():
    capability = ti.hardware.capability("sort.radix.fidelityfx")
    assert capability.public_api == "ti.hardware.sort.VulkanParallelSortPlan"
    assert capability.recipe_provider_api is None
    assert capability.update_policy == "immutable"
    compiler = _compiler()
    keys = ti.ndarray(ti.u32, shape=513)
    with pytest.raises(ValueError, match="distinct"):
        ti.hardware.sort.VulkanParallelSortPlan(keys, keys, compiler_path=compiler)
    with pytest.raises(ValueError, match="scalar 1D u32"):
        ti.hardware.sort.VulkanParallelSortPlan(
            ti.ndarray(ti.f32, shape=8), compiler_path=compiler
        )
    with pytest.raises(FileNotFoundError):
        ti.hardware.sort.VulkanParallelSortPlan(
            keys, compiler_path=compiler + ".missing"
        )
    old = ti.hardware.sort.VulkanParallelSortPlan(keys, compiler_path=compiler)
    with pytest.raises(ValueError, match="binding names"):
        old.record(keys="")
    ti.reset()
    ti.init(arch=ti.vulkan, offline_cache=False)
    fresh = ti.ndarray(ti.u32, shape=513)
    with ti.hardware.sort.VulkanParallelSortPlan(
        fresh, compiler_path=compiler
    ) as current:
        with pytest.raises(RuntimeError, match="closed|finalized|destroyed|runtime"):
            old.run()
        old.close()
        source = np.arange(513, dtype=np.uint32)[::-1].copy()
        fresh.from_numpy(source)
        current.run()
        np.testing.assert_array_equal(fresh.to_numpy(), source[::-1])
