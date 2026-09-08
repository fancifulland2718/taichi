"""Cold fixed-shape, coalesced FP32 operand movement for CUDA recipes."""

from functools import lru_cache
from math import prod

from taichi_forge.lang import impl, simt
from taichi_forge.lang.kernel_impl import kernel
from taichi_forge.lang.matrix import Vector
from taichi_forge.lang.misc import loop_config
from taichi_forge.types import ndarray
from taichi_forge.types.primitive_types import f32, i32


TILED_PACKING_IMPLEMENTATION = "f32-permutation-tile16x16-padded-v1"


@lru_cache(maxsize=128)
def tiled_permutation_kernel(shape, permutation):
    """Exchange contiguous axes; all other axes retain their fixed coordinates.

    ``shape`` is the destination shape. Callers choose this lowering only for
    their existing rank-2/3 CUDA operand permutations. No input is retained
    between invocations and no runtime shape discovery is required.
    """
    rank = len(shape)
    row_axis = permutation.index(rank - 1)
    column_axis = rank - 1
    assert rank in (2, 3) and row_axis != column_axis
    rows, columns = shape[row_axis], shape[column_axis]
    outer_axes = tuple(i for i in range(rank) if i not in (row_axis, column_axis))
    row_tiles, column_tiles = (rows + 15) // 16, (columns + 15) // 16
    workers = prod(shape[i] for i in outer_axes) * row_tiles * column_tiles * 128

    @kernel
    def pack_tiled(
        source: ndarray(dtype=f32, ndim=rank),
        destination: ndarray(dtype=f32, ndim=rank),
    ):
        loop_config(block_dim=128)
        for worker in range(workers):
            tile = simt.block.SharedArray((16, 17), f32)
            lane = worker % 128
            x, y = lane % 16, lane // 16
            tile_index = worker // 128
            column = (tile_index % column_tiles) * 16
            row = ((tile_index // column_tiles) % row_tiles) * 16
            outer = tile_index // (column_tiles * row_tiles)
            index = Vector.zero(i32, rank)
            for axis in impl.static(outer_axes[::-1]):
                index[axis] = outer % shape[axis]
                outer = outer // shape[axis]
            for offset in impl.static((0, 8)):
                # Adjacent lanes read the source's contiguous axis.
                index[row_axis] = row + x
                index[column_axis] = column + y + offset
                original = Vector.zero(i32, rank)
                for axis in impl.static(range(rank)):
                    original[permutation[axis]] = index[axis]
                if row + x < rows and column + y + offset < columns:
                    tile[y + offset, x] = source[original]
            simt.block.sync()
            for offset in impl.static((0, 8)):
                index[row_axis] = row + y + offset
                index[column_axis] = column + x
                if row + y + offset < rows and column + x < columns:
                    destination[index] = tile[x, y + offset]
            # The range loop can grid-stride to another tile. Every thread
            # finishes reading before the block reuses its shared storage.
            simt.block.sync()

    return pack_tiled
