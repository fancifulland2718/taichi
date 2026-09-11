import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


def _offset_field():
    padding = ti.field(ti.i32)
    values = ti.field(ti.i32)
    builder = ti.FieldsBuilder()
    builder.dense(ti.i, 7).place(padding)
    builder.dense(ti.i, 64).place(values)
    tree = builder.finalize()
    return values, padding, tree


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_buffer_field_subranges_compose_with_kernels_and_reuse_fixed_bindings(
    monkeypatch,
):
    values, padding, tree = _offset_field()
    source = ti.experimental.ndarray_view(values, slices=slice(0, 16))
    destination = ti.experimental.ndarray_view(values, slices=slice(32, 48))
    assert source.descriptor.byte_offset != 0
    values.from_numpy(np.arange(64, dtype=np.int32))
    padding.fill(999)

    @ti.kernel
    def produce(seed: ti.i32):
        for i in range(16):
            values[i] = 3 * i + seed

    @ti.kernel
    def consume(output: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in output:
            output[i] += 1

    command = ti.graph.VulkanBufferCommand
    recording = ti.graph.VulkanBufferCommandRecording(
        (
            command.fill_u32("destination", 64, 0xDEADBEEF),
            command.buffer_barrier("destination"),
            command.copy("destination", "source", 64),
            command.memory_barrier(),
        )
    )
    builder = ti.graph.GraphBuilder()
    builder.dispatch(produce, ti.graph.Arg(ti.graph.ArgKind.SCALAR, "seed", ti.i32))
    builder.append_native(recording, admission="auto")
    builder.dispatch(
        consume, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "destination", ti.i32, ndim=1)
    )
    graph = builder.compile()
    binding = graph.bind({"source": source, "destination": destination, "seed": 7})
    assert binding.fast_path_qualified, binding.statistics()
    np.testing.assert_array_equal(values.to_numpy(), np.arange(64, dtype=np.int32))

    def unexpected_prepare(*args, **kwargs):
        raise AssertionError("Native layout/command preparation entered replay")

    with monkeypatch.context() as patched:
        patched.setattr(
            ti.graph.VulkanBufferCommandRecording, "_prepare_packet", unexpected_prepare
        )
        for _ in range(3):
            graph.run(binding)
    expected = np.arange(64, dtype=np.int32)
    expected[:16] = 3 * np.arange(16) + 7
    expected[32:48] = expected[:16] + 1
    np.testing.assert_array_equal(values.to_numpy(), expected)
    np.testing.assert_array_equal(padding.to_numpy(), np.full(7, 999, np.int32))

    # Structure unchanged, values and scalar binding may change independently.
    binding.update(seed=-8)
    graph.run(binding)
    expected[:16] = 3 * np.arange(16) - 8
    expected[32:48] = expected[:16] + 1
    np.testing.assert_array_equal(values.to_numpy(), expected)
    graph.close()
    tree.destroy()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_buffer_field_binding_checks_ranges_at_prepare_and_retires_with_tree():
    values, padding, tree = _offset_field()
    values.from_numpy(np.arange(64, dtype=np.int32))
    source = ti.experimental.ndarray_view(values, slices=slice(0, 16))
    destination = ti.experimental.ndarray_view(values, slices=slice(32, 48))
    command = ti.graph.VulkanBufferCommand
    recording = ti.graph.VulkanBufferCommandRecording(
        (command.copy("dst", "src", 64), command.memory_barrier())
    )
    builder = ti.graph.GraphBuilder()
    builder.append_native(recording, admission="auto")
    graph = builder.compile()
    binding = graph.bind({"src": source, "dst": destination})
    graph.run(binding)
    expected = np.arange(64, dtype=np.int32)
    expected[32:48] = expected[:16]
    np.testing.assert_array_equal(values.to_numpy(), expected)

    revision = binding.revision
    with pytest.raises(RuntimeError, match="overlap"):
        binding.update(dst=ti.experimental.ndarray_view(values, slices=slice(8, 24)))
    assert binding.revision == revision
    with pytest.raises(RuntimeError, match="exceeds 32 bytes"):
        binding.update(src=ti.experimental.ndarray_view(values, slices=slice(0, 8)))
    with pytest.raises(RuntimeError, match="compact"):
        binding.update(src=ti.experimental.ndarray_view(values, slices=slice(0, 32, 2)))
    np.testing.assert_array_equal(values.to_numpy(), expected)
    tree.destroy()
    with pytest.raises(RuntimeError, match="destroyed|retired|generation"):
        graph.run(binding)

    replacement, replacement_padding, replacement_tree = _offset_field()
    replacement.fill(17)
    replacement_dst = ti.ndarray(ti.i32, shape=64)
    # Direct field input is accepted too; only the specified byte range is copied.
    recording.execute({"src": replacement, "dst": replacement_dst})
    np.testing.assert_array_equal(
        replacement_dst.to_numpy()[:16], np.full(16, 17, np.int32)
    )
    binding.replace({"src": replacement, "dst": replacement_dst})
    graph.run(binding)
    np.testing.assert_array_equal(
        replacement_dst.to_numpy()[:16], np.full(16, 17, np.int32)
    )
    graph.close()
    replacement_tree.destroy()
