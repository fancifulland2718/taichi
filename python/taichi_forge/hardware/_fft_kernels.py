"""Provider-owned fixed-shape FFT postprocessing, built at materialization."""

from functools import lru_cache

from taichi_forge.lang._ndrange import ndrange
from taichi_forge.lang.kernel_impl import kernel
from taichi_forge.types import ndarray
from taichi_forge.types.primitive_types import f32


@lru_cache(maxsize=64)
def output_scale_kernel(dimensions, batch_count, scale):
    height, width = dimensions
    if batch_count == 1:

        @kernel
        def apply(output: ndarray(dtype=f32, ndim=3)):
            for h, w, component in ndrange(height, width, 2):
                output[h, w, component] = output[h, w, component] * scale

    else:

        @kernel
        def apply(output: ndarray(dtype=f32, ndim=4)):
            for batch, h, w, component in ndrange(batch_count, height, width, 2):
                output[batch, h, w, component] = output[batch, h, w, component] * scale

    return apply
