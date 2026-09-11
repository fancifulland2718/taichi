import gc
import threading
import weakref

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge._lib import core as ti_core
from taichi_forge.lang import impl
from tests import test_utils


def _graph():
    @ti.kernel
    def stage(
        scale: ti.i32,
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        scratch: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in source:
            scratch[i] = source[i] * scale

    @ti.kernel
    def finish(
        slot: ti.i32,
        scratch: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        output[slot] += scratch[slot] + 7

    def array(name):
        return ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=1)

    scale = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "scale", ti.i32)
    slot = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "slot", ti.i32)
    source, scratch, output = (array(name) for name in ("source", "scratch", "output"))
    builder = ti.graph.GraphBuilder()
    builder.dispatch(stage, scale, source, scratch)
    builder.dispatch(finish, slot, scratch, output)
    return builder.compile()


def _executor(graph):
    if not ti_core._CudaGraphBindingExecutor.available():
        pytest.skip("CUDA driver/device does not support immutable binding-frame execution")
    return ti_core._CudaGraphBindingExecutor(graph._compiled_graph, impl.current_cfg(), impl.get_runtime().prog)


def _prepare(executor, source, scratch, output, scale, slot=0):
    return executor.prepare(
        dict(
            source=source.arr,
            scratch=scratch.arr,
            output=output.arr,
            scale=scale,
            slot=slot,
        )
    )


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_prepared_frames_are_pure_and_queued_reuse_preserves_each_binding():
    graph = _graph()
    executor = _executor(graph)
    size = 513
    source = ti.ndarray(ti.i32, size)
    scratch = ti.ndarray(ti.i32, size)
    output = ti.ndarray(ti.i32, size)
    values = np.arange(size, dtype=np.int32) + 3
    source.from_numpy(values)
    scratch.fill(-91)
    output.fill(0)
    bindings = [graph.bind(dict(source=source, scratch=scratch, output=output, scale=i + 1, slot=i)) for i in range(7)]
    frames = [executor.prepare(dict(binding._version.flattened_args)) for binding in bindings]
    ti.sync()
    np.testing.assert_array_equal(scratch.to_numpy(), np.full(size, -91, np.int32))
    np.testing.assert_array_equal(output.to_numpy(), np.zeros(size, np.int32))
    before = executor.snapshot()
    assert before["frames"] == 7
    # A published frame owns one aligned parameter image for all dispatches.
    assert before["preparation_upload_calls"] == 7
    assert before["preparation_upload_bytes"] == before["argument_bytes"]
    assert before["executables"] == 1
    assert before["kernel_nodes"] >= 14
    expected = np.zeros(size, np.int32)
    try:
        for step in range(224):
            index = (step * 3) % len(frames)
            executor.run(frames[index])
            expected[index] += values[index] * (index + 1) + 7
        ti.sync()
        np.testing.assert_array_equal(output.to_numpy(), expected)
        # Argument images stay immutable, but pointed-to device data does not.
        source.from_numpy(values + 5)
        for _ in range(11):
            executor.run(frames[0])
        expected[0] += 11 * (values[0] + 5 + 7)
        np.testing.assert_array_equal(output.to_numpy(), expected)
        after = executor.snapshot()
        assert after["preparation_upload_calls"] == before["preparation_upload_calls"]
        assert after["preparation_upload_bytes"] == before["preparation_upload_bytes"]
        assert after["argument_bytes"] == before["argument_bytes"]
        assert after["executables"] == 1
        assert after["pending_frame_leases"] == 0
    finally:
        executor.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_binding_event_backlog_reuses_observed_peak_and_releases_on_close():
    # A serial device dependency establishes a real backlog before the short
    # frame submissions, without a host sleep or a replay synchronization.
    @ti.kernel
    def queue_dependency(value: ti.types.ndarray(dtype=ti.u32, ndim=1)):
        x = value[0]
        ti.loop_config(serialize=True)
        for _ in range(10000000):
            x ^= x << 13
            x ^= x >> 17
            x ^= x << 5
        value[0] = x

    dependency = ti.ndarray(ti.u32, 1)
    dependency.fill(123)
    queue_dependency(dependency)
    ti.sync()
    graph = _graph()
    executor = _executor(graph)
    source, scratch, output = (ti.ndarray(ti.i32, 513) for _ in range(3))
    source.fill(3)
    output.fill(0)
    frames = [_prepare(executor, source, scratch, output, i + 1, i) for i in range(4)]
    ti.sync()
    before = executor.snapshot()
    created = None
    try:
        for batch in range(3):
            queue_dependency(dependency)
            for index in range(64):
                executor.run(frames[index % 4])
            pending = executor.snapshot()
            ti.sync()
            finished = executor.snapshot()
            assert finished["pending_frame_leases"] == 0
            assert finished["completion_events_destroyed"] == 0
            assert finished["completion_events_abandoned"] == 0
            assert finished["completion_events_cached"] == finished["completion_events_created"]
            if batch == 0:
                # This is a resource-path coverage assertion, not a speed gate.
                assert pending["pending_frame_leases"] > 16
                created = finished["completion_events_created"]
            else:
                assert finished["completion_events_created"] == created
            assert finished["argument_bytes"] == before["argument_bytes"]
            assert finished["preparation_upload_calls"] == before["preparation_upload_calls"]
        expected = np.zeros(513, np.int32)
        expected[:4] = 3 * 16 * (3 * np.arange(1, 5) + 7)
        np.testing.assert_array_equal(output.to_numpy(), expected)
        assert finished["completion_events_reused"] >= 128
    finally:
        executor.close()
    closed = executor.snapshot()
    assert closed["completion_events_cached"] == 0
    assert closed["completion_events_created"] == closed["completion_events_destroyed"]
    assert closed["argument_bytes"] == 0


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_prepared_frame_pins_device_allocations_without_pinning_python_arrays():
    graph = _graph()
    executor = _executor(graph)
    source = ti.ndarray(ti.i32, 37)
    scratch = ti.ndarray(ti.i32, 37)
    output = ti.ndarray(ti.i32, 37)
    source.fill(9)
    output.fill(0)
    frame = _prepare(executor, source, scratch, output, 13)
    references = weakref.ref(source), weakref.ref(scratch)
    del source, scratch
    gc.collect()
    assert all(ref() is None for ref in references)
    try:
        for _ in range(31):
            executor.run(frame)
        assert output.to_numpy()[0] == 31 * (9 * 13 + 7)
    finally:
        executor.close()
    assert executor.snapshot()["argument_bytes"] == 0
    with pytest.raises(RuntimeError, match="closed"):
        executor.run(frame)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_binding_frame_churn_retires_history_and_rejects_foreign_owner():
    graph = _graph()
    executor = _executor(graph)
    other = _executor(graph)
    source = ti.ndarray(ti.i32, 17)
    scratch = ti.ndarray(ti.i32, 17)
    output = ti.ndarray(ti.i32, 17)
    source.fill(2)
    output.fill(0)
    try:
        for scale in range(1, 97):
            frame = _prepare(executor, source, scratch, output, scale)
            executor.run(frame)
        ti.sync()
        assert output.to_numpy()[0] == sum(2 * scale + 7 for scale in range(1, 97))
        gc.collect()
        snapshot = executor.snapshot()
        assert snapshot["frames"] == 1
        assert snapshot["pending_frame_leases"] == 0
        assert snapshot["argument_bytes"] > 0
        with pytest.raises(RuntimeError, match="another executor"):
            other.run(frame)
        # Failed preparation must leave the published frame usable.
        wrong_dtype = ti.ndarray(ti.f32, 17)
        with pytest.raises(RuntimeError):
            _prepare(executor, source, scratch, wrong_dtype, 2)
        with pytest.raises(RuntimeError):
            executor.prepare(dict(source=source.arr, scratch=scratch.arr, output=output.arr, scale=2))
        executor.run(frame)
        assert output.to_numpy()[0] == sum(2 * scale + 7 for scale in range(1, 97)) + 199
        # A successful prepare can replace the executor's current
        # configuration without executing it or changing the old frame.
        unlaunched = _prepare(executor, source, scratch, output, 500, slot=3)
        executor.run(frame)
        result = output.to_numpy()
        assert result[0] == sum(2 * scale + 7 for scale in range(1, 97)) + 398
        assert result[3] == 0
        del unlaunched
    finally:
        executor.close()
        other.close()
    assert executor.snapshot()["frames"] == 0


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_binding_frame_update_and_launch_are_one_native_thread_transaction():
    graph = _graph()
    executor = _executor(graph)
    source = ti.ndarray(ti.i32, 8)
    scratch = ti.ndarray(ti.i32, 8)
    output = ti.ndarray(ti.i32, 8)
    source.fill(3)
    output.fill(0)
    frames = [_prepare(executor, source, scratch, output, i + 1, i) for i in range(4)]
    failures = []

    def run(index):
        try:
            for _ in range(80):
                executor.run(frames[index])
        except BaseException as error:
            failures.append(error)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(4)]
    try:
        impl.get_runtime().prog._record_runtime_completion().wait()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)
        assert all(not thread.is_alive() for thread in threads)
        assert not failures
        np.testing.assert_array_equal(output.to_numpy()[:4], [80 * (3 * (i + 1) + 7) for i in range(4)])
    finally:
        executor.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_device_reset_invalidates_frames_that_outlive_their_program():
    graph = _graph()
    executor = _executor(graph)
    source = ti.ndarray(ti.i32, 8)
    scratch = ti.ndarray(ti.i32, 8)
    output = ti.ndarray(ti.i32, 8)
    source.fill(2)
    frame = _prepare(executor, source, scratch, output, 5)
    executor.run(frame)
    ti.reset()
    assert executor.snapshot()["closed"] == 1
    assert executor.snapshot()["argument_bytes"] == 0
    with pytest.raises(RuntimeError, match="closed"):
        executor.run(frame)
    executor.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_binding_frames_reject_snode_lifetimes_before_capture():
    field = ti.field(ti.i32, shape=16)

    @ti.kernel
    def increment():
        for i in field:
            field[i] += 1

    builder = ti.graph.GraphBuilder()
    builder.dispatch(increment)
    graph = builder.compile()
    with pytest.raises(RuntimeError, match="SNode dependencies"):
        _executor(graph)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_binding_frames_copy_matrix_arguments_and_survive_executor_close():
    matrix_type = ti.types.matrix(2, 2, ti.i32)

    @ti.kernel
    def append(value: matrix_type, output: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(4):
            output[i] += value[i // 2, i % 2]

    value = ti.graph.Arg(ti.graph.ArgKind.MATRIX, "value", matrix_type)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(append, value, output_arg)
    graph = builder.compile()
    executor = _executor(graph)
    output = ti.ndarray(ti.i32, 4)
    output.fill(0)
    host = np.array([[3, -11], [17, 31]], dtype=np.int32)
    first = executor.prepare(dict(value=host, output=output.arr))
    host *= 2
    second = executor.prepare(dict(value=host, output=output.arr))
    host.fill(-999)
    for _ in range(37):
        executor.run(first)
        executor.run(second)
    executor.close()
    np.testing.assert_array_equal(output.to_numpy(), np.array([3, -11, 17, 31]) * 111)
    assert executor.snapshot()["argument_bytes"] == 0


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_packed_frames_preserve_mixed_scalar_layouts_and_async_host_images():
    @ti.kernel
    def integer_stage(bias: ti.i64, output: ti.types.ndarray(dtype=ti.i64, ndim=1)):
        for i in output:
            output[i] = bias + i

    @ti.kernel
    def real_stage(gain: ti.f64, bias: ti.i32, output: ti.types.ndarray(dtype=ti.f64, ndim=1)):
        for i in output:
            output[i] += gain * i + bias

    bias = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "bias", ti.i64)
    gain = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "gain", ti.f64)
    shift = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "shift", ti.i32)
    integers = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "integers", ti.i64, ndim=1)
    reals = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "reals", ti.f64, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(integer_stage, bias, integers)
    builder.dispatch(real_stage, gain, shift, reals)
    builder.dispatch(integer_stage, bias, integers)
    graph = builder.compile()
    executor = _executor(graph)
    integer_output = ti.ndarray(ti.i64, 257)
    real_output = ti.ndarray(ti.f64, 257)
    real_output.fill(0)
    try:
        # Prepare and immediately queue different images without a host wait.
        for i in range(19):
            frame = executor.prepare(
                dict(bias=2**42 + i, gain=0.125, shift=i, integers=integer_output.arr, reals=real_output.arr)
            )
            executor.run(frame)
        np.testing.assert_array_equal(integer_output.to_numpy(), 2**42 + 18 + np.arange(257, dtype=np.int64))
        np.testing.assert_array_equal(real_output.to_numpy(), 19 * 0.125 * np.arange(257) + sum(range(19)))
        assert executor.snapshot()["preparation_upload_calls"] == 19
    finally:
        executor.close()
    assert executor.snapshot()["argument_bytes"] == 0
