"""Provider-owned pointwise helpers for complete sparse matmul dataflows."""

from functools import lru_cache

import taichi_forge as ti


@lru_cache(maxsize=32)
def relu_kernel(m, n):
    @ti.kernel
    def relu(output: ti.types.ndarray(dtype=ti.f16, ndim=2)):
        for i, j in ti.ndrange(m, n):
            output[i, j] = ti.cast(ti.max(output[i, j], 0.0), ti.f16)

    return relu
