"""Synthetic kernels must not alias each other's compilation cache entries."""

import numpy as np

import taichi_forge as ti
from taichi_forge._lib import core
from taichi_forge.graph._graph import Graph, _CompiledCGraphNode, gen_cpp_kernel
from taichi_forge.graph._segmented_reduce_kernels import reduction_kernel
from taichi_forge.lang import impl
from tests import test_utils


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_value_variants_keep_distinct_cache_keys_and_live_reduction_source():
    @ti.kernel
    def double(source: ti.types.ndarray(), values: ti.types.ndarray()):
        for i in source:
            values[i] = source[i] * 2

    @ti.kernel
    def triple(source: ti.types.ndarray(), values: ti.types.ndarray()):
        for i in source:
            values[i] = source[i] * 3

    def argument(name):
        return ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=1)

    reduction_args = [argument(name) for name in ("values", "offsets", "output")]
    producer_args = [argument(name) for name in ("source", "values")]
    reduction = reduction_kernel(ti.i32, 3, block_dim=128)
    # This freshly generated source has not entered the compilation manager.
    reduction_cpp = gen_cpp_kernel(reduction, reduction_args)
    bindings = {
        name: ti.ndarray(ti.i32, shape=size)
        for name, size in (
            ("source", 257),
            ("values", 257),
            ("offsets", 4),
            ("output", 3),
        )
    }
    raw = np.arange(257, dtype=np.int32)
    bounds = np.asarray((0, 1, 17, 257), np.int32)
    bindings["source"].from_numpy(raw)
    bindings["offsets"].from_numpy(bounds)
    sums = np.asarray([raw[a:b].sum() for a, b in zip(bounds, bounds[1:])], np.int32)
    for producer, factor in ((double, 2), (triple, 3), (double, 2)):
        compiled = core._compile_graph_segmented_reduce_values(
            impl.get_runtime().prog,
            reduction_cpp,
            reduction_args,
            gen_cpp_kernel(producer, producer_args),
            producer_args,
        )
        graph = Graph(_CompiledCGraphNode(compiled, 1, bindings))
        graph.run(graph.bind(bindings))
        np.testing.assert_array_equal(bindings["values"].to_numpy(), raw * factor)
        np.testing.assert_array_equal(bindings["output"].to_numpy(), sums * factor)
        # Cache-key generation and ordinary compilation must still see the
        # source AST, not stale lowered statements from a discarded clone.
        reduction(bindings["source"], bindings["offsets"], bindings["output"])
        np.testing.assert_array_equal(bindings["output"].to_numpy(), sums)
