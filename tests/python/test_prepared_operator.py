"""Pinned operator generation, composition ownership and fixed dense binding."""

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils
from tests.python.test_linear_operator_graph_action import _diagonal_operator


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_prepared_operator_pins_numeric_but_reads_live_dense_contents(monkeypatch):
    n = 257
    operator = _diagonal_operator(np.arange(n, dtype=np.float32) + 1)
    x, y = ti.field(ti.f32, shape=n), ti.field(ti.f32, shape=n)
    x.fill(2)
    y.fill(-17)
    ti.sync()
    program = ti.lang.impl.get_runtime().prog
    before = program._runtime_statistics_snapshot()
    plan = operator.prepare_apply(x, y)
    after = program._runtime_statistics_snapshot()
    assert after["submission"] == before["submission"]
    np.testing.assert_array_equal(y.to_numpy(), -17)
    plan.run()
    np.testing.assert_array_equal(y.to_numpy(), 2 * (np.arange(n) + 1))
    new_numeric = ti.ndarray(ti.f32, n)
    new_numeric.fill(3)
    operator.update_numeric(new_numeric, expected_topology_version=1, expected_numeric_version=1)
    current = operator.prepare_apply(x, y)
    import taichi_forge.linalg._runtime as module

    def no_cold_path(*args, **kwargs):
        raise AssertionError("prepared replay repeated operator generation/binding work")

    monkeypatch.setattr(module._LinearOperatorGraphExecutable, "_current_compatible_record", no_cold_path)
    monkeypatch.setattr(module._LinearOperatorGraphExecutable, "bind_graph_arguments", no_cold_path)
    monkeypatch.setattr(module, "describe_storage", no_cold_path)
    x.fill(4)
    plan.run()
    np.testing.assert_array_equal(y.to_numpy(), 4 * (np.arange(n) + 1))
    current.run()
    np.testing.assert_array_equal(y.to_numpy(), 12)
    assert plan.report()["generation_policy"] == "pinned_until_reprepare"
    assert plan.report()["binding"]["fast_path_qualified"]
    current.close()
    plan.close()
    with pytest.raises(RuntimeError, match="closed"):
        plan.run()


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_prepared_operator_composition_records_with_independent_workspace():
    n = 1025
    left, right = _diagonal_operator(np.full(n, 2)), _diagonal_operator(np.full(n, 3))
    operator = left + right
    x, y = ti.ndarray(ti.f32, n), ti.ndarray(ti.f32, n)
    x.fill(7)
    plan = operator.prepare_apply(x, y)
    assert plan.report()["binding"]["fast_path_qualified"]
    builder = ti.graph.GraphBuilder()
    builder.append_native(plan.record())
    builder.append_native(plan.record())
    graph = builder.compile()
    bound = graph.bind({})
    from taichi_forge.lang._ndarray import ScalarNdarray

    def scratch_allocations(compiled):
        return {
            value._runtime_allocation_identity
            for value in compiled._instance._fixed_runtime_args.values()
            if isinstance(value, ScalarNdarray) and value is not x and value is not y
        }

    own_scratch, outer_scratch = scratch_allocations(plan._graph), scratch_allocations(graph)
    assert len(own_scratch) == len(outer_scratch) == 1
    assert own_scratch.isdisjoint(outer_scratch)
    plan.run()
    np.testing.assert_array_equal(y.to_numpy(), 35)
    plan.close()
    # record owns an independent reference to the pinned snapshot; closing the
    # source plan does not destroy a separately compiled Graph's workspace.
    x.fill(11)
    graph.run(bound)
    np.testing.assert_array_equal(y.to_numpy(), 55)
    ticket = graph.submit(bound)
    graph.close()
    ticket.wait()


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_prepared_operator_alias_failure_and_retired_field():
    n = 127
    operator = _diagonal_operator(np.full(n, 2))
    fields = ti.FieldsBuilder()
    x, y = ti.field(ti.f32), ti.field(ti.f32)
    fields.dense(ti.i, n).place(x)
    fields.dense(ti.i, n).place(y)
    tree = fields.finalize()
    x.fill(3)
    with pytest.raises(RuntimeError, match="disjoint"):
        operator.prepare_apply(x, x)
    plan = operator.prepare_apply(x, y)
    plan.run()
    np.testing.assert_array_equal(y.to_numpy(), 6)
    tree.destroy()
    with pytest.raises(RuntimeError, match="destroyed|retired|stale|generation|inactive"):
        plan.run()
    plan.close()
    a, b = ti.ndarray(ti.f32, n), ti.ndarray(ti.f32, n)
    reset_plan = operator.prepare_apply(a, b)
    recorded = reset_plan.record()
    arch = ti.lang.impl.current_cfg().arch
    ti.reset()
    with pytest.raises(RuntimeError, match="closed|retired|runtime"):
        reset_plan.run()
    reset_plan.close()
    ti.init(arch=arch, offline_cache=False)
    with pytest.raises(RuntimeError, match="another runtime"):
        ti.graph.GraphBuilder().append_native(recorded)


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_prepared_graph_operator_keeps_live_state_and_rejects_retirement():
    n = 513
    fields = ti.FieldsBuilder()
    weight = ti.field(ti.f32)
    fields.dense(ti.i, n).place(weight)
    tree = fields.finalize()
    topology = ti.ndarray(ti.i32, n)
    topology.from_numpy(np.arange(n, dtype=np.int32))
    x, y = ti.ndarray(ti.f32, n), ti.ndarray(ti.f32, n)
    x.fill(4)
    weight.fill(2)

    @ti.kernel
    def apply_live(
        topo: ti.types.ndarray(ti.i32, ndim=1),
        input: ti.types.ndarray(ti.f32, ndim=1),
        output: ti.types.ndarray(ti.f32, ndim=1),
    ):
        for i in input:
            output[i] = weight[i] * input[topo[i]]

    arg = lambda name, dtype: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, dtype, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(apply_live, arg("topology", ti.i32), arg("input", ti.f32), arg("output", ti.f32))
    source = builder.compile()
    operator = ti.linalg.LinearOperator.from_graph(source, n, topology={"topology": topology}, state={"weight": weight})
    plan = operator.prepare_apply(x, y)
    for value in (2, 7):
        weight.fill(value)
        plan.run()
        np.testing.assert_array_equal(y.to_numpy(), 4 * value)
    assert plan.report()["binding"]["fast_path_qualified"]
    tree.destroy()
    with pytest.raises(RuntimeError, match="destroyed|retired|stale|generation|inactive"):
        plan.run()
    plan.close()
    source.close()
