"""Actual native value substitution across reduction iteration domains."""

import gc

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge._lib import core
from taichi_forge.graph._graph import (
    Graph,
    _CompiledCGraphNode,
    _GraphSpec,
    gen_cpp_kernel,
)
from taichi_forge.graph._segmented_reduce_kernels import reduction_kernel
from taichi_forge.lang import impl
from tests import test_utils


@ti.kernel
def _produce(
    source: ti.types.ndarray(),
    weights: ti.types.ndarray(),
    values: ti.types.ndarray(),
    count: ti.i32,
    factor: ti.u32,
):
    for i in range(count):
        bits = ti.cast(source[i], ti.u32) * factor + ti.u32(0xF0000001)
        values[i] = (bits ^ ti.cast(weights[i], ti.u32)) - ti.cast(i, ti.u32)


@ti.kernel
def _consume(
    reduced: ti.types.ndarray(), bias: ti.types.ndarray(), result: ti.types.ndarray()
):
    for i in reduced:
        bits = ti.cast(reduced[i], ti.u32) ^ ti.cast(bias[i], ti.u32)
        result[i] = (bits << 3) | (bits >> 29)


def _array(name, dtype):
    return ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, dtype, ndim=1)


def _fixture(dtype):
    bounds = np.asarray((0, 0, 1, 18, 147, 2196, 10389), np.int32)
    count = int(bounds[-1])
    size = count + 11
    raw = (np.arange(size, dtype=np.uint64) * 0x60000001 + 0xF0000001).astype(np.uint32)
    weights = raw[::-1].copy()
    bias = raw[: len(bounds) - 1].copy()
    arrays = {}
    for name, value in (("source", raw), ("weights", weights), ("bias", bias)):
        arrays[name] = ti.ndarray(dtype, shape=len(value))
        arrays[name].from_numpy(value if dtype == ti.u32 else value.view(np.int32))
    arrays["offsets"] = ti.ndarray(ti.i32, shape=len(bounds))
    arrays["offsets"].from_numpy(bounds)
    for name, size_ in (
        ("values", size),
        ("reduced", len(bounds) - 1),
        ("result", len(bounds) - 1),
    ):
        arrays[name] = ti.ndarray(dtype, shape=size_)
        arrays[name].fill(777)
    factor = 0x60000001
    mapped = (
        (raw.astype(np.uint64) * factor + 0xF0000001) & 0xFFFFFFFF
    ) ^ weights.astype(np.uint64)
    mapped = ((mapped - np.arange(size, dtype=np.uint64)) & 0xFFFFFFFF).astype(
        np.uint32
    )
    expected_values = np.full(size, 777, np.uint32)
    expected_values[:count] = mapped[:count]
    sums = np.asarray(
        [
            int(mapped[a:b].sum(dtype=np.uint64)) & 0xFFFFFFFF
            for a, b in zip(bounds, bounds[1:])
        ],
        np.uint32,
    )
    bits = sums ^ bias
    final = (bits << 3) | (bits >> 29)
    arrays.update(count=count, factor=factor)
    return arrays, bounds, (expected_values, sums, final)


def _sources(dtype):
    producer_args = [_array(name, dtype) for name in ("source", "weights", "values")]
    producer_args += [
        ti.graph.Arg(ti.graph.ArgKind.SCALAR, "count", ti.i32),
        ti.graph.Arg(ti.graph.ArgKind.SCALAR, "factor", ti.u32),
    ]
    consumer_args = [_array(name, dtype) for name in ("reduced", "bias", "result")]
    return (gen_cpp_kernel(_produce, producer_args), producer_args), (
        gen_cpp_kernel(_consume, consumer_args),
        consumer_args,
    )


def _fused_node(
    dtype,
    segments,
    block_dim,
    producer,
    consumer,
    *,
    values="values",
    offsets="offsets",
    output="reduced"
):
    reduction_args = [
        _array(values, dtype),
        _array(offsets, ti.i32),
        _array(output, dtype),
    ]
    reduction = gen_cpp_kernel(
        reduction_kernel(dtype, segments, block_dim=block_dim), reduction_args
    )
    compiled = core._compile_graph_segmented_reduce_values(
        impl.get_runtime().prog,
        reduction,
        reduction_args,
        *(producer if producer is not None else (None, ())),
        *(consumer if consumer is not None else (None, ())),
        0 if consumer is not None else -1,
    )
    names = tuple(
        dict.fromkeys(
            argument.name
            for source in (
                reduction_args,
                producer[1] if producer else (),
                consumer[1] if consumer else (),
            )
            for argument in source
        )
    )
    assert compiled._composer_stats["compiled_tasks"] == 1
    recordings = compiled._owned_jit_dispatch_sources
    assert len(recordings) == 1
    assert {argument.name for argument in recordings[0][1]} == set(names)
    # Re-record through the same raw-dispatch bridge used when Graph regions
    # are concatenated. Only the recording handles now retain the original
    # graph that owns the synthetic kernel; the new CGraph has no such owner.
    builder = core.GraphBuilder()
    for kernel_cpp, arguments in recordings:
        builder.dispatch(kernel_cpp, arguments)
    rerecorded = builder.compile()
    return _CompiledCGraphNode(rerecorded, 1, names, recording_dispatches=recordings)


def _assert_outputs(bindings, expected):
    for name, values in zip(("values", "reduced", "result"), expected):
        np.testing.assert_array_equal(bindings[name].to_numpy().view(np.uint32), values)


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("dtype", (ti.i32, ti.u32))
def test_native_value_fusion_preserves_stores_bits_empty_segments_and_tail(
    dtype, monkeypatch
):
    bindings, bounds, expected = _fixture(dtype)
    producer, consumer = _sources(dtype)
    # Compare with the actual three-dispatch baseline before discarding source
    # handles. The reference is independent and includes unused capacity.
    _produce(
        bindings["source"],
        bindings["weights"],
        bindings["values"],
        bindings["count"],
        bindings["factor"],
    )
    reduction_kernel(dtype, len(bounds) - 1)(
        bindings["values"], bindings["offsets"], bindings["reduced"]
    )
    _consume(bindings["reduced"], bindings["bias"], bindings["result"])
    _assert_outputs(bindings, expected)
    for block_dim in (0, 32, 128):
        for name in ("values", "reduced", "result"):
            bindings[name].fill(777)
        node = _fused_node(dtype, len(bounds) - 1, block_dim, producer, consumer)
        graph = Graph(node)
        bound = graph.bind(bindings)

        # Native compilation owns the synthetic kernel; no source execution or
        # value inspection is permitted once the Graph is bound.
        def forbidden(*args, **kwargs):
            raise AssertionError("replay invoked cold value compilation")

        with monkeypatch.context() as replay:
            replay.setattr(core, "_compile_graph_segmented_reduce_values", forbidden)
            replay.setattr(core, "_graph_pointwise_value_program", forbidden)
            gc.collect()
            for _ in range(3):
                graph.run(bound)
        _assert_outputs(bindings, expected)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_native_partial_fusion_maps_only_input_and_final_stage():
    bindings, bounds, expected = _fixture(ti.u32)
    producer, consumer = _sources(ti.u32)
    tiles, ends = [0], [0]
    for a, b in zip(bounds, bounds[1:]):
        tiles.extend(min(int(i) + 1024, int(b)) for i in range(int(a), int(b), 1024))
        ends.append(len(tiles) - 1)
    for name, data in (("tile_offsets", tiles), ("final_offsets", ends)):
        bindings[name] = ti.ndarray(ti.i32, shape=len(data))
        bindings[name].from_numpy(np.asarray(data, np.int32))
    bindings["partials"] = ti.ndarray(ti.u32, shape=len(tiles) - 1)
    first = _fused_node(
        ti.u32,
        len(tiles) - 1,
        128,
        producer,
        None,
        offsets="tile_offsets",
        output="partials",
    )
    final = _fused_node(
        ti.u32,
        len(bounds) - 1,
        32,
        None,
        consumer,
        values="partials",
        offsets="final_offsets",
    )
    bindings.pop("offsets")
    graph = Graph(_GraphSpec([first, final]))
    graph.run(graph.bind(bindings))
    _assert_outputs(bindings, expected)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_native_value_fusion_rejects_symbolic_abi_mismatch_before_execution():
    bindings, bounds, _ = _fixture(ti.u32)
    producer, consumer = _sources(ti.u32)
    wrong = list(producer[1])
    wrong[2] = _array("different_values", ti.u32)
    with pytest.raises(RuntimeError, match="producer output.*values symbol"):
        _fused_node(ti.u32, len(bounds) - 1, 32, (producer[0], wrong), consumer)
    for name in ("values", "reduced", "result"):
        np.testing.assert_array_equal(bindings[name].to_numpy(), 777)
