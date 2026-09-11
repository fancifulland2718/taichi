"""Prepared predicate pipelines: storage, shared prefixes and root composition."""

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False, vulkan_sparse_experimental=False)
def test_prepared_compact_vector_subranges_and_changing_predicates(monkeypatch):
    n = 4097
    values = ti.Vector.ndarray(3, ti.i32, n + 4)
    flags = ti.ndarray(ti.i32, n + 4)
    output = ti.Vector.ndarray(3, ti.i32, n + 6)
    count = ti.field(ti.i32, shape=())
    original = np.arange((n + 4) * 3, dtype=np.int32).reshape(n + 4, 3)
    values.from_numpy(original)
    view = ti.experimental.ndarray_view
    from taichi_forge.lang._storage_view import describe_storage

    writable = view(values)
    assert describe_storage(writable, access="read").supported
    assert describe_storage(writable, access="write").supported
    with pytest.raises(ValueError, match="access"):
        describe_storage(writable, access="invalid")
    plan = ti.algorithms.prepare_compact(
        view(values, slices=slice(2, n + 2)),
        view(flags, slices=slice(1, n + 1)),
        view(output, slices=slice(3, n + 3)),
        count,
    )
    import taichi_forge.algorithms._prepared_compact as module

    def cold_path(*args, **kwargs):
        raise AssertionError("storage description entered prepared replay")

    monkeypatch.setattr(module, "describe_storage", cold_path)
    for mask in (np.zeros(n, np.int32), np.ones(n, np.int32), (np.arange(n) % 3 - 1).astype(np.int32)):
        flag_data = np.zeros(n + 4, np.int32)
        flag_data[1 : n + 1] = mask
        flags.from_numpy(flag_data)
        output.fill(-91)
        plan.run()
        selected = original[2 : n + 2][mask != 0]
        assert count[None] == len(selected)
        actual = output.to_numpy()
        np.testing.assert_array_equal(actual[3 : 3 + len(selected)], selected)
        np.testing.assert_array_equal(actual[:3], -91)
        np.testing.assert_array_equal(actual[3 + len(selected) :], -91)
    np.testing.assert_array_equal(values.to_numpy(), original)
    assert plan.report()["prefix_evaluations"] == 1
    plan.close()
    with pytest.raises(RuntimeError, match="closed"):
        plan.run()


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False, vulkan_sparse_experimental=False)
def test_prepared_unique_payloads_share_prefix_and_compose_at_graph_root():
    n = 65537
    plans, bindings, expected = [], [], []
    for divisor in (3, 7):
        keys = ti.ndarray(ti.u64, n)
        values = ti.Vector.ndarray(2, ti.i32, n)
        unique_keys = ti.ndarray(ti.u64, n)
        unique_values = ti.Vector.ndarray(2, ti.i32, n)
        count = ti.ndarray(ti.i32, 1)
        original = ((np.arange(n, dtype=np.uint64) // divisor) % 257) + np.uint64(1 << 42)
        payload = np.column_stack((np.arange(n), -np.arange(n))).astype(np.int32)
        keys.from_numpy(original)
        values.from_numpy(payload)
        count.fill(-13)
        ti.sync()
        program = ti.lang.impl.get_runtime().prog
        before = program._runtime_statistics_snapshot()
        plan = ti.algorithms.prepare_unique_by_key(keys, values, unique_keys, unique_values, count)
        after = program._runtime_statistics_snapshot()
        assert after["submission"] == before["submission"]
        # Cold JIT may acquire/sample a CPU driver lock; it must not wait for
        # device work or issue a runtime synchronization.
        for counter in ("program_syncs", "backend_waits", "completion_waits"):
            assert after["synchronization"][counter] == before["synchronization"][counter]
        assert count.to_numpy()[0] == -13
        np.testing.assert_array_equal(keys.to_numpy(), original)
        plans.append(plan)
        bindings.append((unique_keys, unique_values, count))
        heads = np.r_[True, original[1:] != original[:-1]]
        expected.append((original[heads], payload[heads]))
    builder = ti.graph.GraphBuilder()
    for plan in plans:
        builder.append_native(plan.record())
    graph = builder.compile()
    fixed = graph.bind({})
    for _ in range(3):
        graph.run(fixed)
        for plan, (keys, values, count), (key_ref, value_ref) in zip(plans, bindings, expected):
            actual_count = int(count.to_numpy()[0])
            assert actual_count == len(key_ref)
            np.testing.assert_array_equal(keys.to_numpy()[:actual_count], key_ref)
            np.testing.assert_array_equal(values.to_numpy()[:actual_count], value_ref)
            assert plan.report()["prefix_evaluations"] == 1
    ticket = graph.submit(fixed)
    graph.close()
    for plan in plans:
        plan.close()
    ticket.wait()


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False, vulkan_sparse_experimental=False)
def test_prepared_unique_field_zero_size_and_tree_retirement():
    n = 127
    fields = ti.FieldsBuilder()
    keys, values, unique_keys, unique_values = (ti.field(ti.i32) for _ in range(4))
    count = ti.field(ti.i32)
    for field in (keys, values, unique_keys, unique_values):
        fields.dense(ti.i, n).place(field)
    fields.place(count)
    tree = fields.finalize()
    original = ((np.arange(n) // 3) % 7).astype(np.int32)
    keys.from_numpy(original)
    values.from_numpy(np.arange(n, dtype=np.int32))
    unique_keys.fill(-19)
    with ti.algorithms.prepare_unique(keys, unique_keys, count, size=0) as empty:
        empty.run()
        assert count[None] == 0
        np.testing.assert_array_equal(unique_keys.to_numpy(), -19)
    plan = ti.algorithms.prepare_unique_by_key(keys, values, unique_keys, unique_values, count)
    plan.run()
    heads = np.r_[True, original[1:] != original[:-1]]
    length = int(count[None])
    assert length == np.count_nonzero(heads)
    np.testing.assert_array_equal(unique_keys.to_numpy()[:length], original[heads])
    np.testing.assert_array_equal(unique_values.to_numpy()[:length], np.arange(n)[heads])
    tree.destroy()
    with pytest.raises(RuntimeError, match="retired|destroyed|generation|stale|inactive"):
        plan.run()
    plan.close()


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False, vulkan_sparse_experimental=False)
def test_prepared_unique_failures_reset_and_structured_boundary(monkeypatch):
    keys, output = ti.ndarray(ti.i32, 17), ti.ndarray(ti.i32, 17)
    count = ti.ndarray(ti.i32, 1)
    keys.from_numpy(np.arange(17, dtype=np.int32))
    count.fill(-1)
    with pytest.raises(RuntimeError, match="overlap"):
        ti.algorithms.prepare_unique(keys, keys, count)
    from taichi_forge._kernels import rle_mark_boundaries_ndarray

    def fail_compile(*args, **kwargs):
        raise RuntimeError("injected head compilation failure")

    with monkeypatch.context() as patch:
        patch.setattr(rle_mark_boundaries_ndarray._primal, "ensure_compiled", fail_compile)
        with pytest.raises(RuntimeError, match="injected"):
            ti.algorithms.prepare_unique(keys, output, count)
    assert count.to_numpy()[0] == -1
    np.testing.assert_array_equal(keys.to_numpy(), np.arange(17))
    plan = ti.algorithms.prepare_unique(keys, output, count)
    # Root expansion must not silently permit native compaction inside while/if.
    with pytest.raises(RuntimeError, match="recordable"):
        ti.graph.GraphBuilder().create_sequential().append_native(plan.record())
    plan.run()
    np.testing.assert_array_equal(output.to_numpy(), np.arange(17))
    ti.reset()
    with pytest.raises(RuntimeError, match="another runtime"):
        plan.run()
    plan.close()
