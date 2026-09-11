import pytest

import taichi_forge as ti
from taichi_forge.graph import _graph as graph_impl
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_observation_dense_scalar_bindings_are_cold_and_snapshots_are_independent(
    monkeypatch,
):
    monkeypatch.setenv("TI_GRAPH_OBSERVATION_SLOTS", "2")
    padding = ti.field(ti.i32)
    counter, residual = ti.field(ti.i32), ti.field(ti.f32)
    fields = ti.FieldsBuilder()
    fields.dense(ti.i, 7).place(padding)
    fields.place(counter, residual)
    tree = fields.finalize()
    padding.fill(919)
    counter[None], residual[None] = 4, 8.0
    residual_view = ti.experimental.ndarray_view(residual)
    assert residual_view.descriptor.byte_offset != 0

    @ti.kernel
    def advance(
        c: ti.types.ndarray(ti.i32, ndim=0), r: ti.types.ndarray(ti.f32, ndim=0)
    ):
        c[None] += 1
        r[None] *= 0.5

    c = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "counter", ti.i32, ndim=0)
    r = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "residual", ti.f32, ndim=0)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(advance, c, r)
    builder.observe(c, r, name="tail")
    graph = builder.compile()
    binding = graph.bind({"counter": counter, "residual": residual_view})
    assert binding.fast_path_qualified, binding.statistics()
    assert counter[None] == 4 and residual[None] == 8.0
    before = ti.lang.impl.get_runtime().prog._graph_observation_staging_stats()
    first = graph.submit(binding)
    second = graph.submit(binding)
    # Completion/snapshot storage is existing machinery: no host materialization
    # is needed to submit two independent snapshots of the same in-place state.
    assert graph.execution_stats().memory.observation_host_readback_bytes == 0
    assert second.observations() == {"tail": {"counter": 6, "residual": 2.0}}
    assert first.observations() == {"tail": {"counter": 5, "residual": 4.0}}
    after = ti.lang.impl.get_runtime().prog._graph_observation_staging_stats()
    for name in ("packed_batches", "direct_batches", "fallback_batches"):
        assert after[name] == before[name]

    def unexpected_validation(*args, **kwargs):
        raise AssertionError("Observation storage validation entered fixed replay")

    with monkeypatch.context() as patched:
        patched.setattr(
            graph_impl._CompiledObservationGraphNode,
            "validate_bindings",
            unexpected_validation,
        )
        assert graph.submit(binding).observations() == {
            "tail": {"counter": 7, "residual": 1.0}
        }
    revision = binding.revision
    wrong = ti.ndarray(ti.i32, (1,))
    with pytest.raises(RuntimeError, match="scalar ndarray"):
        binding.update(counter=wrong)
    assert binding.revision == revision
    assert graph.submit(binding).observations() == {
        "tail": {"counter": 8, "residual": 0.5}
    }
    assert all(padding[i] == 919 for i in range(7))
    tree.destroy()
    with pytest.raises(RuntimeError, match="retired|destroyed|generation"):
        graph.submit(binding)
    graph.close()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_observation_rejects_bad_storage_before_producer_submission():
    state = ti.ndarray(ti.i32, ())
    state.fill(17)

    @ti.kernel
    def mutate(value: ti.types.ndarray(ti.i32, ndim=0)):
        value[None] += 10

    arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "state", ti.i32, ndim=0)
    observed = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "observed", ti.f32, ndim=0)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(mutate, arg)
    builder.observe(observed)
    graph = builder.compile()
    wrong_dtype = ti.field(ti.i32, shape=())
    with pytest.raises(RuntimeError, match="dtype"):
        graph.run({"state": state, "observed": wrong_dtype})
    assert state.to_numpy()[()] == 17
    graph.close()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_observation_recipe_instances_own_their_execution_cache():
    # Mixed types, a full four-value batch and one tail; integer IDs must not
    # pass through a floating-point packing representation.
    dtypes = (ti.i32, ti.u32, ti.f32, ti.i32, ti.u32)
    fields = {
        f"value_{index}": ti.field(dtype, shape=())
        for index, dtype in enumerate(dtypes)
    }
    builder = ti.graph.GraphBuilder()
    builder.observe(
        *(
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, dtype, ndim=0)
            for name, dtype in zip(fields, dtypes)
        )
    )
    definition = builder.freeze()
    first, second = definition.materialize(), definition.materialize()
    first_node = first.executor._spec.observation_nodes[0]
    second_node = second.executor._spec.observation_nodes[0]
    assert first_node is not second_node
    assert first_node._jit_cache is not second_node._jit_cache
    first_binding = first.executor.bind(fields)
    second_binding = second.executor.bind(
        {name: ti.experimental.ndarray_view(value) for name, value in fields.items()}
    )
    expected = dict(zip(fields, (11, 0xFFFFFFFF, 0.5, -16777217, 7)))
    for name, value in fields.items():
        value[None] = expected[name]
    a = first.executor.submit(first_binding)
    fields["value_0"][None] = 29
    b = second.executor.submit(second_binding)
    assert a.observations() == {"observation": expected}
    assert b.observations() == {"observation": {**expected, "value_0": 29}}
    first.close()
    fields["value_0"][None] = 37
    assert second.executor.submit(second_binding).observations() == {
        "observation": {**expected, "value_0": 37}
    }
    second.close()
