"""Provider-owned fixed-shape FFT postprocessing, built at materialization."""

from functools import lru_cache

from taichi_forge.lang._ndrange import ndrange
from taichi_forge.lang.kernel_impl import kernel
from taichi_forge.types import ndarray
from taichi_forge.types.primitive_types import f32


@lru_cache(maxsize=64)
def output_scale_kernel(dimensions, batch_count, scale, *, transform="c2c"):
    height, width = dimensions
    if transform == "r2c":
        width = width // 2 + 1
    if transform == "c2r":
        if batch_count == 1:

            @kernel
            def apply(output: ndarray(dtype=f32, ndim=2)):
                for h, w in ndrange(height, width):
                    output[h, w] = output[h, w] * scale

        else:

            @kernel
            def apply(output: ndarray(dtype=f32, ndim=3)):
                for batch, h, w in ndrange(batch_count, height, width):
                    output[batch, h, w] = output[batch, h, w] * scale

        return apply
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
