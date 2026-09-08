"""Physical operand packing and a bounded, separate matmul epilogue."""

from taichi_forge.lang import ops
from taichi_forge.lang.kernel_impl import kernel
from taichi_forge.types import ndarray
from taichi_forge.types.primitive_types import f32


@kernel
def pack_transpose_2d(
    source: ndarray(dtype=f32, ndim=2), destination: ndarray(dtype=f32, ndim=2)
):
    for i, j in destination:
        destination[i, j] = source[j, i]


@kernel
def pack_transpose_3d(
    source: ndarray(dtype=f32, ndim=3), destination: ndarray(dtype=f32, ndim=3)
):
    for batch, i, j in destination:
        destination[batch, i, j] = source[batch, j, i]


@kernel
def relu_2d(output: ndarray(dtype=f32, ndim=2)):
    for i, j in output:
        output[i, j] = ops.max(output[i, j], 0.0)


@kernel
def relu_3d(output: ndarray(dtype=f32, ndim=3)):
    for batch, i, j in output:
        output[batch, i, j] = ops.max(output[batch, i, j], 0.0)
