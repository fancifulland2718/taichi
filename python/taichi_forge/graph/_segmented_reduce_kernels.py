"""Device implementations for complete fixed-layout segmented reductions."""

from taichi_forge.lang import impl, ops, simt
from taichi_forge.lang.kernel_impl import kernel
from taichi_forge.lang.misc import loop_config
from taichi_forge.types import ndarray_type
from taichi_forge.types.primitive_types import i32, u32


_KERNELS = {}


def reduction_kernel(dtype, segment_count, *, block_dim=0):
    """Reduce modulo 2**32, including signed storage, without atomics.

    Serial, warp-owned and block-owned segments use the same numerical
    contract. All lanes participate in shuffles/barriers, including empty
    segments and partial tails. Blocks write distinct segment results.
    """
    if dtype not in (i32, u32) or block_dim not in (0, 32, 128):
        raise ValueError("unsupported segmented reduction kernel topology")
    key = (dtype, int(segment_count), block_dim)
    if key in _KERNELS:
        return _KERNELS[key]

    if block_dim == 0:

        @kernel
        def reduce(
            values: ndarray_type.ndarray(dtype=dtype, ndim=1),
            offsets: ndarray_type.ndarray(dtype=i32, ndim=1),
            output: ndarray_type.ndarray(dtype=dtype, ndim=1),
        ):
            for segment in range(segment_count):
                total = ops.cast(0, u32)
                for index in range(offsets[segment], offsets[segment + 1]):
                    total += ops.cast(values[index], u32)
                output[segment] = ops.cast(total, dtype)

    else:
        worker_count = int(segment_count) * block_dim
        warp_count = block_dim // 32

        @kernel
        def reduce(
            values: ndarray_type.ndarray(dtype=dtype, ndim=1),
            offsets: ndarray_type.ndarray(dtype=i32, ndim=1),
            output: ndarray_type.ndarray(dtype=dtype, ndim=1),
        ):
            loop_config(block_dim=block_dim)
            for worker in range(worker_count):
                segment = worker // block_dim
                lane = worker % block_dim
                warp_lane = lane % 32
                total = ops.cast(0, u32)
                index = offsets[segment] + lane
                while index < offsets[segment + 1]:
                    total += ops.cast(values[index], u32)
                    index += block_dim
                for shift in impl.static((16, 8, 4, 2, 1)):
                    total += ops.bit_cast(
                        simt.warp.shfl_down_i32(
                            ops.cast(-1, u32), ops.bit_cast(total, i32), shift
                        ),
                        u32,
                    )
                if impl.static(warp_count == 1):
                    if lane == 0:
                        output[segment] = ops.cast(total, dtype)
                else:
                    totals = simt.block.SharedArray((warp_count,), u32)
                    if warp_lane == 0:
                        totals[lane // 32] = total
                    simt.block.sync()
                    if lane == 0:
                        block_total = ops.cast(0, u32)
                        for warp in impl.static(range(warp_count)):
                            block_total += totals[warp]
                        output[segment] = ops.cast(block_total, dtype)

    _KERNELS[key] = reduce
    return reduce
