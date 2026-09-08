"""Fixed-shape permutation and pointwise contraction epilogues, built cold."""

from functools import lru_cache
from math import prod

from taichi_forge.lang import ops
from taichi_forge.lang.kernel_impl import kernel
from taichi_forge.lang.matrix import Vector
from taichi_forge.lang.impl import static
from taichi_forge.linalg._packing_kernels import tiled_permutation_kernel
from taichi_forge.types import ndarray
from taichi_forge.types.primitive_types import f32, i32


@lru_cache(maxsize=128)
def permutation_kernel(shape, permutation, *, tiled=False):
    if tiled:
        return tiled_permutation_kernel(shape, permutation)
    rank, count = len(shape), prod(shape)

    @kernel
    def permute(
        source: ndarray(dtype=f32, ndim=rank),
        destination: ndarray(dtype=f32, ndim=rank),
    ):
        for linear in range(count):
            index = Vector.zero(i32, rank)
            original = Vector.zero(i32, rank)
            remainder = linear
            for axis in static(range(rank - 1, -1, -1)):
                index[axis] = remainder % shape[axis]
                remainder = remainder // shape[axis]
            for axis in static(range(rank)):
                original[permutation[axis]] = index[axis]
            destination[index] = source[original]

    return permute


@lru_cache(maxsize=128)
def epilogue_kernel(shape, alpha, beta, relu):
    rank, count = len(shape), prod(shape)

    @kernel
    def epilogue(
        product: ndarray(dtype=f32, ndim=rank),
        c: ndarray(dtype=f32, ndim=rank),
        output: ndarray(dtype=f32, ndim=rank),
    ):
        for linear in range(count):
            index = Vector.zero(i32, rank)
            remainder = linear
            for axis in static(range(rank - 1, -1, -1)):
                index[axis] = remainder % shape[axis]
                remainder = remainder // shape[axis]
            value = alpha * product[index]
            if static(beta != 0.0):
                value += beta * c[index]
            if static(relu):
                value = ops.max(value, 0.0)
            output[index] = value

    return epilogue
