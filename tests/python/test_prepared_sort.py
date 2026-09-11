"""Prepared native sort contracts, not another primitive algorithm suite."""

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("storage", ("ndarray", "field", "view"))
@test_utils.test(arch=[ti.cuda, ti.vulkan], vulkan_sparse_experimental=False)
def test_prepared_sort_cold_binding_stability_and_graph(storage, monkeypatch):
    n = 4097
    if storage == "field":
        # Separate trees give both scalar types their natural alignment. Vulkan
        # mixed-width root siblings can otherwise place u64 at a 4-byte offset;
        # that existing layout is not qualified for native bulk operations.
        keys, values = ti.field(ti.u64), ti.field(ti.i32)
        key_builder, value_builder = ti.FieldsBuilder(), ti.FieldsBuilder()
        key_builder.dense(ti.i, n + 2).place(keys)
        value_builder.dense(ti.i, n + 2).place(values)
        key_tree, value_tree = key_builder.finalize(), value_builder.finalize()
    else:
        keys = ti.ndarray(ti.u64, shape=n + 2)
        values = ti.ndarray(ti.i32, shape=n + 2)
    keys_np = (np.arange(n + 2, dtype=np.uint64) * 397 % 61) + np.uint64(1 << 40)
    values_np = np.arange(n + 2, dtype=np.int32)
    keys.from_numpy(keys_np)
    values.from_numpy(values_np)
    if storage == "view":
        key_binding = ti.experimental.ndarray_view(keys, slices=(slice(1, n + 1),))
        value_binding = ti.experimental.ndarray_view(values, slices=(slice(1, n + 1),))
        selection = slice(1, n + 1)
    else:
        key_binding, value_binding = keys, values
        selection = slice(None)
    ti.sync()
    program = ti.lang.impl.get_runtime().prog
    before = program._runtime_statistics_snapshot()
    plan = ti.algorithms.prepare_sort(key_binding, value_binding)
    builder = ti.graph.GraphBuilder()
    builder.append_native(plan.record())
    graph = builder.compile()
    after = program._runtime_statistics_snapshot()
    assert after["submission"] == before["submission"]
    assert after["synchronization"] == before["synchronization"]
    np.testing.assert_array_equal(keys.to_numpy(), keys_np)
    np.testing.assert_array_equal(values.to_numpy(), values_np)
    assert plan.report()["workspace_bytes_last_observed"] is None
    physical_id = plan.report()["physical_plan_id"]

    # Execution must not rebuild Python storage descriptors or call public sort.
    import taichi_forge.algorithms._prepared_primitive as prepared

    def unexpected(*args, **kwargs):
        raise AssertionError("cold preparation repeated in steady execution")

    monkeypatch.setattr(prepared, "describe_storage", unexpected)
    monkeypatch.setattr(ti.algorithms, "sort", unexpected)
    for execute in (plan.run, lambda: graph.run({}), plan.run):
        keys.from_numpy(keys_np)
        values.from_numpy(values_np)
        execute()
        expected_order = np.argsort(keys_np[selection], kind="stable")
        expected_keys, expected_values = keys_np.copy(), values_np.copy()
        expected_keys[selection] = keys_np[selection][expected_order]
        expected_values[selection] = values_np[selection][expected_order]
        np.testing.assert_array_equal(keys.to_numpy(), expected_keys)
        np.testing.assert_array_equal(values.to_numpy(), expected_values)
        keys_np = keys_np[::-1].copy()
    assert plan.report()["physical_plan_id"] == physical_id
    assert plan.report()["workspace_bytes_last_observed"] > 0
    assert not plan.report()["device_capture"]
    plan.close()
    plan.close()
    with pytest.raises(RuntimeError, match="closed"):
        graph.run({})
    graph.close()


@test_utils.test(arch=[ti.cuda, ti.vulkan], vulkan_sparse_experimental=False)
def test_prepared_sort_lifetime_and_failed_preparation():
    builder = ti.FieldsBuilder()
    keys = ti.field(ti.i32)
    builder.dense(ti.i, 33).place(keys)
    tree = builder.finalize()
    keys.from_numpy(np.arange(33, dtype=np.int32)[::-1].copy())
    plan = ti.algorithms.prepare_sort(keys)
    plan.run()
    # Retirement, not a repeated full field-layout check, invalidates a packet.
    tree.destroy()
    with pytest.raises(RuntimeError, match="retired|destroyed|generation|stale|inactive"):
        plan.run()
    plan.close()

    values = ti.ndarray(ti.i32, shape=33)
    initial = np.arange(33, dtype=np.int32)[::-1].copy()
    values.from_numpy(initial)
    with pytest.raises(RuntimeError, match="overlap"):
        ti.algorithms.prepare_sort(values, values)
    with pytest.raises(RuntimeError, match="compact"):
        ti.algorithms.prepare_sort(ti.experimental.ndarray_view(values, slices=(slice(None, None, 2),)))
    np.testing.assert_array_equal(values.to_numpy(), initial)
    fresh = ti.algorithms.prepare_sort(values)
    fresh.run()
    np.testing.assert_array_equal(values.to_numpy(), np.sort(initial))
    ti.reset()
    with pytest.raises(RuntimeError, match="another runtime"):
        fresh.run()
    fresh.close()


@test_utils.test(arch=[ti.cuda, ti.vulkan], vulkan_sparse_experimental=False)
def test_prepared_sort_float_nan_and_tiny_ranges():
    keys = ti.ndarray(ti.f32, shape=7)
    payload = ti.ndarray(ti.i32, shape=7)
    initial = np.array([np.nan, -3, 1, np.inf, -np.inf, 1, np.nan], dtype=np.float32)
    initial.view(np.uint32)[0] = 0xFFC00001
    for policy in (("last", "bitwise") if ti.lang.impl.current_cfg().arch == ti.cuda else ("last",)):
        keys.from_numpy(initial)
        payload.from_numpy(np.arange(7, dtype=np.int32))
        with ti.algorithms.prepare_sort(keys, payload, nan_policy=policy) as plan:
            plan.run()
            bits = initial.view(np.uint32)
            sortable = np.where(bits >> 31, ~bits, bits ^ np.uint32(1 << 31))
            expected = [4, 1, 2, 5, 3, 0, 6] if policy == "last" else np.argsort(sortable, kind="stable")
            np.testing.assert_array_equal(payload.to_numpy(), expected)
            np.testing.assert_array_equal(keys.to_numpy(), initial[expected])
    one = ti.ndarray(ti.i32, shape=1)
    one.from_numpy(np.array([19], dtype=np.int32))
    with ti.algorithms.prepare_sort(one) as plan:
        plan.run()
        assert plan.report()["workspace_bytes_last_observed"] == 0
    np.testing.assert_array_equal(one.to_numpy(), [19])


@test_utils.test(arch=[ti.cuda, ti.vulkan], vulkan_sparse_experimental=False)
def test_prepared_sort_between_producer_and_consumer_and_async_close():
    n = 1025
    keys, payload, output = (ti.ndarray(ti.i32, n) for _ in range(3))

    @ti.kernel
    def produce(k: ti.types.ndarray(), v: ti.types.ndarray(), seed: ti.i32):
        for i in k:
            k[i] = (i * 17 + seed) % 31
            v[i] = i

    @ti.kernel
    def consume(k: ti.types.ndarray(), v: ti.types.ndarray(), out: ti.types.ndarray()):
        for i in out:
            out[i] = k[i] * 10000 + v[i]

    plan = ti.algorithms.prepare_sort(keys, payload)
    builder = ti.graph.GraphBuilder()
    k, v, out = (ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=1) for name in ("k", "v", "out"))
    builder.dispatch(produce, k, v, ti.graph.Arg(ti.graph.ArgKind.SCALAR, "seed", ti.i32))
    builder.append_native(plan.record())
    builder.dispatch(consume, k, v, out)
    graph = builder.compile()
    binding = graph.bind({"k": keys, "v": payload, "out": output, "seed": 7})
    graph.run(binding)
    binding.update(seed=11)
    ticket = graph.submit(binding)
    graph.close()
    plan.close()
    ticket.wait()
    original = (np.arange(n) * 17 + 11) % 31
    order = np.argsort(original, kind="stable")
    np.testing.assert_array_equal(output.to_numpy(), original[order] * 10000 + order)
