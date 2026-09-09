"""Cold provider-owned packing, including ragged tiles and grid-stride reuse."""

import numpy as np
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_tiled_packing_covers_narrow_ragged_batched_and_grid_stride_shapes():
    from taichi_forge.linalg._contraction_kernels import permutation_kernel

    # One focused contract covers both tile sizes, outer-axis mappings and a
    # domain large enough to reuse each block's shared storage across tiles.
    for shape, permutation in (
        ((7, 65), (1, 0)),
        ((65, 97), (1, 0)),
        ((3, 65, 97), (0, 2, 1)),
        ((65, 3, 97), (2, 1, 0)),
        ((1025, 2049), (1, 0)),
    ):
        source_shape = tuple(
            shape[permutation.index(axis)] for axis in range(len(shape))
        )
        values = (np.arange(np.prod(source_shape), dtype=np.float32) % 4096).reshape(
            source_shape
        )
        source = ti.ndarray(ti.f32, source_shape)
        destination = ti.ndarray(ti.f32, shape)
        pack = permutation_kernel(shape, permutation, tiled=True)
        for shift in (0, 64):
            source.from_numpy(values + shift)
            pack(source, destination)
            np.testing.assert_array_equal(
                destination.to_numpy(), np.transpose(values + shift, permutation)
            )
