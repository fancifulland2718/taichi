"""Cold fixed-shape lowering: no device task to discover known loop bounds."""

from functools import lru_cache

from taichi_forge.lang import ops
from taichi_forge.lang._ndrange import ndrange
from taichi_forge.lang.kernel_impl import kernel
from taichi_forge.linalg._packing_kernels import tiled_permutation_kernel
from taichi_forge.types import ndarray
from taichi_forge.types.primitive_types import f32


@lru_cache(maxsize=128)
def packing_kernel(shape, *, tiled=False):
    if tiled:
        permutation = (1, 0) if len(shape) == 2 else (0, 2, 1)
        return tiled_permutation_kernel(shape, permutation)
    if len(shape) == 2:
        rows, columns = shape

        @kernel
        def pack(
            source: ndarray(dtype=f32, ndim=2), destination: ndarray(dtype=f32, ndim=2)
        ):
            for i, j in ndrange(rows, columns):
                destination[i, j] = source[j, i]

    else:
        batches, rows, columns = shape

        @kernel
        def pack(
            source: ndarray(dtype=f32, ndim=3), destination: ndarray(dtype=f32, ndim=3)
        ):
            for batch, i, j in ndrange(batches, rows, columns):
                destination[batch, i, j] = source[batch, j, i]

    return pack


@lru_cache(maxsize=128)
def relu_kernel(shape):
    if len(shape) == 2:
        rows, columns = shape

        @kernel
        def relu(output: ndarray(dtype=f32, ndim=2)):
            for i, j in ndrange(rows, columns):
                output[i, j] = ops.max(output[i, j], 0.0)

    else:
        batches, rows, columns = shape

        @kernel
        def relu(output: ndarray(dtype=f32, ndim=3)):
            for batch, i, j in ndrange(batches, rows, columns):
                output[batch, i, j] = ops.max(output[batch, i, j], 0.0)

    return relu
