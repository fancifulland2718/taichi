import gc
import os

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.graph._ir import GraphAccess
from taichi_forge.hardware._retained import retained_execution_contract
from tests import test_utils
from tests.python.hardware_provider_lifecycle_qualification import (
    stress_iterations,
)
from tests.python.hardware_process_memory import ProcessMemoryPlateau


def _callback_compiler_paths():
    nvrtc = os.environ.get("TI_TEST_NVRTC_LIBRARY_PATH")
    linker = os.environ.get("TI_TEST_NVJITLINK_LIBRARY_PATH")
    if not nvrtc or not linker:
        pytest.skip("explicit optional NVRTC/nvJitLink test paths are not configured")
    return {"nvrtc_library": nvrtc, "nvjitlink_library": linker}


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cufft_lto_scale_cache_identity_and_capture_ownership():
    from taichi_forge._lib import core
    from taichi_forge.hardware._fft_lto import _CufftStoreScalePlan, _StoreScaleCallback
    from taichi_forge.lang import impl

    paths = _callback_compiler_paths()
    plain = ti.hardware.fft.CufftPlanND((32, 64), batch_count=2)
    callback = _StoreScaleCallback(0.25, **paths)
    scaled = _CufftStoreScalePlan((32, 64), 2, callback)
    same = _CufftStoreScalePlan((32, 64), 2, callback)
    other = _CufftStoreScalePlan((32, 64), 2, _StoreScaleCallback(-0.125, **paths))
    assert ti.hardware.fft.cache_statistics().live_plans == 3
    assert (
        scaled._graph_provider_memory_identity()
        != plain._graph_provider_memory_identity()
    )
    assert (
        scaled._graph_provider_memory_identity()
        != other._graph_provider_memory_identity()
    )
    arrays = {name: ti.ndarray(ti.f32, (2, 32, 64, 2)) for name in ("input", "output")}
    rng = np.random.default_rng(763)
    values = rng.uniform(-0.25, 0.25, arrays["input"].shape).astype(np.float32)
    arrays["input"].from_numpy(values)
    builder = ti.graph.GraphBuilder()
    builder.append_native(scaled.record())
    graph = builder.compile()
    native = core._CudaGraphBindingExecutor(
        graph._spec.nodes[0].compiled_graph, impl.current_cfg(), impl.get_runtime().prog
    )
    arguments = {name: array.arr for name, array in arrays.items()}
    frame = native.prepare(arguments)
    native.run(frame)
    expected = np.fft.fft2(values[..., 0] + 1j * values[..., 1], axes=(-2, -1)) * 0.25
    np.testing.assert_allclose(
        arrays["output"].to_numpy(),
        np.stack((expected.real, expected.imag), -1),
        atol=2e-5,
        rtol=2e-5,
    )
    same.close()
    other.close()
    plain.close()
    # Immutable frames retain the native plan even after its public handle is
    # closed; changed device values are not an implicit callback snapshot.
    scaled.close()
    arrays["input"].from_numpy(values * 2)
    try:
        for _ in range(5):
            native.run(frame)
        np.testing.assert_allclose(
            arrays["output"].to_numpy(),
            np.stack((expected.real, expected.imag), -1) * 2,
            atol=4e-5,
            rtol=2e-5,
        )
        with pytest.raises(RuntimeError, match="fixed plan"):
            native.prepare(arguments)
    finally:
        native.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cufft_lto_empty_artifact_fails_before_plan_creation():
    from taichi_forge.lang import impl

    before = ti.hardware.fft.cache_statistics()
    with pytest.raises(RuntimeError, match="LTO IR"):
        impl.get_runtime().prog._create_cuda_cufft_store_callback_plan(
            (32, 64), 2, b"", "store"
        )
    assert ti.hardware.fft.cache_statistics() == before


@test_utils.test(arch=ti.cpu)
def test_cufft_plan_rejects_non_cuda_runtime_and_bad_contracts():
    assert not ti.hardware.fft.is_available()
    cache = ti.hardware.fft.cache_statistics()
    assert cache.create_requests == 0
    assert cache.cache_hits == 0
    assert cache.cache_misses == 0
    assert cache.live_handles == 0
    assert cache.live_plans == 0
    assert cache.workspace_bytes_live == 0
    with pytest.raises(ValueError, match="length"):
        ti.hardware.fft.CufftPlan1D(0)
    with pytest.raises(ValueError, match="transform"):
        ti.hardware.fft.CufftPlan1D(8, transform="z2z")
    with pytest.raises(ValueError, match="rank 2 or 3"):
        ti.hardware.fft.CufftPlanND((8,))
    with pytest.raises(ValueError, match="cover"):
        ti.hardware.fft.CufftPlanND(
            (4, 8),
            input_layout=ti.hardware.fft.CufftLayout(embed=(4, 7)),
        )
    with pytest.raises(ValueError, match="overlap"):
        ti.hardware.fft.CufftPlanND(
            (4, 8),
            input_layout=ti.hardware.fft.CufftLayout(batch_distance=1),
        )
    with pytest.raises(RuntimeError, match="requires the CUDA backend"):
        ti.hardware.fft.CufftPlan1D(8)

    descriptor = ti.hardware.capability("fft.transform.cufft")
    assert descriptor.dependency_tier == "lazy_external"
    assert descriptor.implementation_status == "existing_public"
    assert descriptor.scopes == ("python", "graph")
    assert descriptor.execution_kind == "external_library"
    assert descriptor.graph_integration == "root_ordered"
    assert descriptor.workspace_ownership == "provider_owned"


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cufft_c2c_executes_directly_and_through_graph():
    if not ti.hardware.fft.is_available():
        pytest.skip("a compatible optional cuFFT shared library is unavailable")

    batch_count = 2
    length = 8
    rng = np.random.default_rng(20260823)
    complex_values = (
        rng.standard_normal((batch_count, length))
        + 1j * rng.standard_normal((batch_count, length))
    ).astype(np.complex64)
    packed_values = np.stack(
        (complex_values.real, complex_values.imag), axis=-1
    ).astype(np.float32)

    source = ti.ndarray(ti.f32, shape=(batch_count, length, 2))
    spectrum = ti.ndarray(ti.f32, shape=(batch_count, length, 2))
    recovered = ti.ndarray(ti.f32, shape=(batch_count, length, 2))
    source.from_numpy(packed_values)

    plan = ti.hardware.fft.CufftPlan1D(length, batch_count=batch_count)
    assert plan.shape == (batch_count, length, 2)
    plan.execute(source, spectrum, direction="forward")
    ti.sync()
    spectrum_values = spectrum.to_numpy()
    spectrum_complex = spectrum_values[..., 0] + 1j * spectrum_values[..., 1]
    np.testing.assert_allclose(
        spectrum_complex,
        np.fft.fft(complex_values, axis=-1),
        rtol=2e-5,
        atol=2e-5,
    )

    resolved = next(
        operation
        for operation in ti.hardware.report().operations
        if operation.descriptor.operation_id == "fft.transform.cufft"
    )
    assert resolved.discovery == "available"
    assert resolved.enablement == "enabled"
    assert resolved.selection == "eligible"
    assert resolved.provider_abi == "cufft-plan-many-dynamic-symbols-v3"

    recording = plan.record(direction="inverse", input="spectrum", output="signal")
    assert recording.replay_mode == "stream_capture"
    retained = retained_execution_contract(recording)
    assert retained is retained_execution_contract(plan.record())
    assert retained.identity.operation_id == "fft.transform.cufft"
    assert retained.identity.provider_id == "cufft"
    assert retained.automatic_selection_policy == "forbidden"
    assert not retained.identity.persistent_cache_safe
    assert retained.concurrency_policy == "runtime_ordered"
    assert retained.cost_model.scale_costs[0].dimensions == (
        "transform_elements",
        "batch_count",
    )
    assert tuple(
        (effect.resource, effect.access) for effect in recording.resource_effects
    ) == (("spectrum", GraphAccess.READ), ("signal", GraphAccess.WRITE))
    builder = ti.graph.GraphBuilder()
    builder.append_native(recording, admission="auto")
    graph = builder.compile()
    assert len(graph._graph_stats) == 1
    graph.run({"spectrum": spectrum, "signal": recovered})
    ti.sync()
    np.testing.assert_allclose(
        recovered.to_numpy(), packed_values * length, rtol=2e-5, atol=2e-5
    )
    optimization = graph._debug_info["optimization"]
    assert optimization["mixed_backend_regions"] == 0
    assert graph._debug_info["native_count"] == 1
    assert "backend_command_nodes" not in optimization
    graph_stats = graph._graph_stats[0]
    assert graph_stats["captures"] == 1, {
        key: graph_stats.get(key)
        for key in (
            "attempts",
            "captures",
            "last_path",
            "last_fallback_reason",
            "fallbacks",
        )
    }
    assert graph_stats["last_path"] == "cuda_capture"
    assert graph_stats["last_fallback_reason"] == "none"

    plan.close()
    with pytest.raises(RuntimeError, match="closed"):
        graph.run({"spectrum": spectrum, "signal": recovered})


@test_utils.test(arch=ti.cuda)
def test_cufft_recording_validation_is_fail_closed():
    if not ti.hardware.fft.is_available():
        pytest.skip("a compatible optional cuFFT shared library is unavailable")

    plan = ti.hardware.fft.CufftPlan1D(8)
    try:
        with pytest.raises(ValueError, match="direction"):
            plan.record(direction="sideways")
        with pytest.raises(ValueError, match="differ"):
            plan.record(input="data", output="data")

        wrong = ti.ndarray(ti.f32, shape=(8, 3))
        output = ti.ndarray(ti.f32, shape=(8, 2))
        with pytest.raises(RuntimeError, match="shape"):
            plan.execute(wrong, output)
        with pytest.raises(RuntimeError, match="distinct"):
            plan.execute(output, output)
    finally:
        plan.close()


@pytest.mark.parametrize("length", (7, 8))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cufft_real_transforms_preserve_hermitian_layout_and_scale(length):
    if not ti.hardware.fft.is_available():
        pytest.skip("a compatible optional cuFFT shared library is unavailable")

    batch_count = 2
    rng = np.random.default_rng(20260824 + length)
    values = rng.standard_normal((batch_count, length)).astype(np.float32)
    hermitian_length = length // 2 + 1

    signal = ti.ndarray(ti.f32, shape=(batch_count, length))
    spectrum = ti.ndarray(ti.f32, shape=(batch_count, hermitian_length, 2))
    recovered = ti.ndarray(ti.f32, shape=(batch_count, length))
    signal.from_numpy(values)

    forward = ti.hardware.fft.CufftPlan1D(
        length, batch_count=batch_count, transform="r2c"
    )
    inverse = ti.hardware.fft.CufftPlan1D(
        length, batch_count=batch_count, transform="c2r"
    )
    try:
        assert forward.input_shape == (batch_count, length)
        assert forward.output_shape == (batch_count, hermitian_length, 2)
        assert inverse.input_shape == forward.output_shape
        assert inverse.output_shape == forward.input_shape
        assert inverse.inverse_scale == pytest.approx(1.0 / length)

        forward.execute(signal, spectrum)
        ti.sync()
        packed = spectrum.to_numpy()
        actual_spectrum = packed[..., 0] + 1j * packed[..., 1]
        np.testing.assert_allclose(
            actual_spectrum,
            np.fft.rfft(values, axis=-1),
            rtol=2e-5,
            atol=2e-5,
        )

        recording = inverse.record(input="spectrum", output="signal")
        assert recording.direction == "inverse"
        assert recording.output_scale == pytest.approx(1.0 / length)
        builder = ti.graph.GraphBuilder()
        builder.append_native(recording, admission="auto")
        graph = builder.compile()
        graph.run({"spectrum": spectrum, "signal": recovered})
        ti.sync()
        np.testing.assert_allclose(
            recovered.to_numpy() * recording.output_scale,
            values,
            rtol=2e-5,
            atol=2e-5,
        )

        with pytest.raises(ValueError, match="forward"):
            forward.record(direction="inverse")
        with pytest.raises(ValueError, match="inverse"):
            inverse.record(direction="forward")
    finally:
        forward.close()
        inverse.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cufft_batched_nd_compact_transforms_match_numpy():
    if not ti.hardware.fft.is_available():
        pytest.skip("a compatible optional cuFFT shared library is unavailable")

    rng = np.random.default_rng(20260825)
    dimensions = (3, 5)
    batch_count = 2
    values = rng.standard_normal((batch_count, *dimensions)).astype(np.float32)
    signal = ti.ndarray(ti.f32, shape=values.shape)
    signal.from_numpy(values)
    forward = ti.hardware.fft.CufftPlanND(
        dimensions, batch_count=batch_count, transform="r2c"
    )
    inverse = ti.hardware.fft.CufftPlanND(
        dimensions, batch_count=batch_count, transform="c2r"
    )
    spectrum = ti.ndarray(ti.f32, shape=forward.output_shape)
    recovered = ti.ndarray(ti.f32, shape=inverse.output_shape)
    try:
        forward.execute(signal, spectrum)
        packed = spectrum.to_numpy()
        actual = packed[..., 0] + 1j * packed[..., 1]
        np.testing.assert_allclose(
            actual,
            np.fft.rfftn(values, axes=(-2, -1)),
            rtol=3e-5,
            atol=3e-5,
        )

        builder = ti.graph.GraphBuilder()
        builder.append_native(
            inverse.record(input="spectrum", output="signal"), admission="auto"
        )
        graph = builder.compile()
        graph.run({"spectrum": spectrum, "signal": recovered})
        ti.sync()
        np.testing.assert_allclose(
            recovered.to_numpy() * inverse.inverse_scale,
            values,
            rtol=3e-5,
            atol=3e-5,
        )
    finally:
        forward.close()
        inverse.close()

    dimensions = (2, 3, 4)
    complex_values = (
        rng.standard_normal((batch_count, *dimensions))
        + 1j * rng.standard_normal((batch_count, *dimensions))
    ).astype(np.complex64)
    packed_values = np.stack(
        (complex_values.real, complex_values.imag), axis=-1
    ).astype(np.float32)
    plan = ti.hardware.fft.CufftPlanND(
        dimensions, batch_count=batch_count, transform="c2c"
    )
    source = ti.ndarray(ti.f32, shape=plan.input_shape)
    output = ti.ndarray(ti.f32, shape=plan.output_shape)
    source.from_numpy(packed_values)
    try:
        plan.execute(source, output)
        ti.sync()
        packed = output.to_numpy()
        actual = packed[..., 0] + 1j * packed[..., 1]
        np.testing.assert_allclose(
            actual,
            np.fft.fftn(complex_values, axes=(-3, -2, -1)),
            rtol=3e-5,
            atol=3e-5,
        )
    finally:
        plan.close()


def _layout_offset(index, layout):
    offset = index[0]
    for axis in range(1, len(index)):
        offset = offset * layout.embed[axis] + index[axis]
    return offset * layout.stride


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cufft_nd_explicit_embed_stride_and_batch_distance():
    if not ti.hardware.fft.is_available():
        pytest.skip("a compatible optional cuFFT shared library is unavailable")

    dimensions = (3, 4)
    batch_count = 2
    input_layout = ti.hardware.fft.CufftLayout(
        embed=(3, 6), stride=2, batch_distance=36
    )
    output_layout = ti.hardware.fft.CufftLayout(
        embed=(3, 5), stride=1, batch_distance=15
    )
    plan = ti.hardware.fft.CufftPlanND(
        dimensions,
        batch_count=batch_count,
        input_layout=input_layout,
        output_layout=output_layout,
    )
    rng = np.random.default_rng(20260826)
    values = (
        rng.standard_normal((batch_count, *dimensions))
        + 1j * rng.standard_normal((batch_count, *dimensions))
    ).astype(np.complex64)
    input_storage = np.full(plan.input_shape, -123.0, dtype=np.float32)
    for batch in range(batch_count):
        for index in np.ndindex(dimensions):
            element = batch * plan.input_layout.batch_distance + _layout_offset(
                index, plan.input_layout
            )
            input_storage[2 * element] = values[(batch, *index)].real
            input_storage[2 * element + 1] = values[(batch, *index)].imag

    source = ti.ndarray(ti.f32, shape=plan.input_shape)
    output = ti.ndarray(ti.f32, shape=plan.output_shape)
    source.from_numpy(input_storage)
    try:
        plan.execute(source, output)
        ti.sync()
        output_storage = output.to_numpy()
        actual = np.empty_like(values)
        for batch in range(batch_count):
            for index in np.ndindex(dimensions):
                element = batch * plan.output_layout.batch_distance + _layout_offset(
                    index, plan.output_layout
                )
                actual[(batch, *index)] = (
                    output_storage[2 * element] + 1j * output_storage[2 * element + 1]
                )
        np.testing.assert_allclose(
            actual,
            np.fft.fft2(values, axes=(-2, -1)),
            rtol=3e-5,
            atol=3e-5,
        )
    finally:
        plan.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cufft_plan_cache_reuses_plan_and_reports_workspace():
    if not ti.hardware.fft.is_available():
        pytest.skip("a compatible optional cuFFT shared library is unavailable")

    before = ti.hardware.fft.cache_statistics()
    first = ti.hardware.fft.CufftPlanND((16, 16), batch_count=2)
    second = ti.hardware.fft.CufftPlanND((16, 16), batch_count=2)
    after_create = ti.hardware.fft.cache_statistics()
    assert after_create.create_requests == before.create_requests + 2
    assert after_create.cache_misses == before.cache_misses + 1
    assert after_create.cache_hits == before.cache_hits + 1
    assert after_create.live_handles == before.live_handles + 2
    assert after_create.live_plans == before.live_plans + 1

    report = first.memory_report()
    workspace = next(
        component
        for component in report.components
        if component.name == "automatic_workspace"
    )
    assert workspace.requested_bytes_exact
    assert workspace.requested_bytes == after_create.workspace_bytes_live
    assert report.opaque_component_count == 1

    program = ti.lang.impl.get_runtime().prog
    waits_before = program._runtime_statistics_snapshot()["synchronization"][
        "backend_waits"
    ]
    first.close()
    after_first_close = ti.hardware.fft.cache_statistics()
    assert after_first_close.live_handles == before.live_handles + 1
    assert after_first_close.live_plans == before.live_plans + 1
    assert (
        program._runtime_statistics_snapshot()["synchronization"]["backend_waits"]
        == waits_before
    )

    source = ti.ndarray(ti.f32, shape=second.input_shape)
    output = ti.ndarray(ti.f32, shape=second.output_shape)
    second.execute(source, output)
    ti.sync()
    telemetry = ti.hardware.telemetry().providers["cufft"]
    assert telemetry["plan_cache_hits"] == after_create.cache_hits
    assert telemetry["plan_live_plans"] == after_create.live_plans

    second.close()
    after_close = ti.hardware.fft.cache_statistics()
    assert after_close.live_handles == before.live_handles
    assert after_close.live_plans == before.live_plans
    assert after_close.workspace_bytes_live == before.workspace_bytes_live


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cufft_inflight_close_is_completion_retained_and_generation_safe():
    if not ti.hardware.fft.is_available():
        pytest.skip("a compatible optional cuFFT shared library is unavailable")

    program = ti.lang.impl.get_runtime().prog
    before = ti.hardware.fft.cache_statistics()
    plan = ti.hardware.fft.CufftPlan1D(4096)
    source = ti.ndarray(ti.f32, shape=plan.input_shape)
    output = ti.ndarray(ti.f32, shape=plan.output_shape)
    host = np.zeros(plan.input_shape, dtype=np.float32)
    host[:, 0] = np.arange(4096, dtype=np.float32) % 17
    source.from_numpy(host)
    ti.sync()
    memory_before = program._runtime_statistics_snapshot()["memory"]

    plan.execute(source, output)
    waits_before = program._runtime_statistics_snapshot()["synchronization"][
        "backend_waits"
    ]
    capacity = plan.memory_report().known_capacity_requested_bytes
    plan.close()

    assert (
        program._runtime_statistics_snapshot()["synchronization"]["backend_waits"]
        == waits_before
    )
    closed = plan.memory_report()
    assert closed.lifecycle_state == "closed"
    assert closed.known_resident_requested_bytes == 0
    assert closed.known_capacity_requested_bytes == capacity
    after_close = ti.hardware.fft.cache_statistics()
    assert after_close.live_handles == before.live_handles
    assert after_close.live_plans == before.live_plans
    assert (
        program._runtime_statistics_snapshot()["memory"]["inflight_resources"]
        >= memory_before["inflight_resources"] + 1
    )

    replacement = ti.hardware.fft.CufftPlan1D(4096)
    after_replacement = ti.hardware.fft.cache_statistics()
    assert after_replacement.cache_misses == before.cache_misses + 2
    assert after_replacement.cache_hits == before.cache_hits
    ti.sync()
    assert np.isfinite(output.to_numpy()).all()
    replacement.close()

    collected = ti.hardware.fft.CufftPlan1D(128)
    assert ti.hardware.fft.cache_statistics().live_handles == before.live_handles + 1
    del collected
    gc.collect()
    assert ti.hardware.fft.cache_statistics().live_handles == before.live_handles


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("dimensions,batch", (((7, 9), 3), ((32, 16), 1), ((31, 64), 2), ((7, 3), 17)))
@pytest.mark.parametrize("direction", ("forward", "inverse"))
@pytest.mark.parametrize("cross_batch", (False, True))
def test_cufft_separable_plan_preserves_batches_capture_and_cache_identity(dimensions, batch, direction, cross_batch):
    from taichi_forge.hardware._fft import _CufftPlanBase

    if not ti.hardware.fft.is_available():
        pytest.skip("the optional cuFFT runtime is unavailable")
    if cross_batch and not hasattr(ti.lang.impl.get_runtime().prog, "_create_cuda_cufft_cross_batch_plan"):
        pytest.skip("cross-batch FFT is unavailable")

    class SeparablePlan(_CufftPlanBase):
        def __init__(self):
            self._initialize(dimensions, batch_count=batch, transform="c2c", _separable=True, _cross_batch=cross_batch)

    initial = ti.hardware.fft.cache_statistics()
    regular = ti.hardware.fft.CufftPlanND(dimensions, batch_count=batch)
    separable, shared = SeparablePlan(), SeparablePlan()
    stats = ti.hardware.fft.cache_statistics()
    assert stats.live_plans == initial.live_plans + 2
    assert stats.cache_misses == initial.cache_misses + 2
    assert stats.cache_hits == initial.cache_hits + 1
    program = ti.lang.impl.get_runtime().prog
    assert not program._cuda_cufft_plan_memory_statistics(regular._handle)["separable"]
    assert program._cuda_cufft_plan_memory_statistics(separable._handle)["separable"]
    if cross_batch:
        assert program._cuda_cufft_plan_memory_statistics(separable._handle)["cross_batch"]
    assert separable._graph_provider_memory_identity() != regular._graph_provider_memory_identity()

    source = ti.ndarray(ti.f32, regular.input_shape)
    product = ti.ndarray(ti.f32, regular.output_shape)
    output = ti.ndarray(ti.f32, regular.output_shape)
    host = np.random.default_rng(81).uniform(-1, 1, regular.input_shape).astype(np.float32)
    source.from_numpy(host)
    values = host[..., 0] + 1j * host[..., 1]
    reference = np.fft.fft2(values) if direction == "forward" else np.fft.ifft2(values) * np.prod(dimensions)
    expected = np.stack((reference.real, reference.imag), axis=-1)

    @ti.kernel
    def finish(source: ti.types.ndarray(dtype=ti.f32), output: ti.types.ndarray(dtype=ti.f32)):
        for index in ti.grouped(source):
            output[index] = source[index] + 0.125

    try:
        shared.close()
        for plan in (regular, separable):
            plan.execute(source, product, direction=direction)
            np.testing.assert_allclose(product.to_numpy(), expected, rtol=2e-4, atol=2e-4)
            builder = ti.graph.GraphBuilder()
            builder.append_native(plan.record(direction=direction, output="product"), admission="auto")
            builder.dispatch(
                finish,
                *(
                    ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=len(source.shape))
                    for name in ("product", "output")
                ),
            )
            graph = builder.compile()
            bound = graph.bind({"input": source, "product": product, "output": output})
            for _ in range(3):
                graph.run(bound)
            np.testing.assert_allclose(output.to_numpy(), expected + 0.125, rtol=2e-4, atol=2e-4)
            np.testing.assert_array_equal(source.to_numpy(), host)
            assert graph._graph_stats[0]["last_path"] == "cuda_exact_replay", graph._graph_stats
        del bound, graph, builder
    finally:
        shared.close()
        separable.close()
        regular.close()
    ti.sync()
    final = ti.hardware.fft.cache_statistics()
    assert final.live_plans == initial.live_plans
    assert final.workspace_bytes_live == initial.workspace_bytes_live


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cufft_separable_plan_rejects_incompatible_layout_before_caching():
    if not ti.hardware.fft.is_available():
        pytest.skip("the optional cuFFT runtime is unavailable")
    program = ti.lang.impl.get_runtime().prog
    baseline = ti.hardware.fft.cache_statistics()
    for transform, embed, stride, distance in ((1, (8, 16), 1, 128), (0, (8, 20), 1, 160), (0, (8, 16), 2, 256)):
        with pytest.raises(RuntimeError, match="Separable cuFFT"):
            program._create_cuda_cufft_plan_many(
                (8, 16), embed, stride, distance, embed, stride, distance, 1, transform, True
            )
    after = ti.hardware.fft.cache_statistics()
    assert after.live_plans == baseline.live_plans
    assert after.live_handles == baseline.live_handles
    assert after.workspace_bytes_live == baseline.workspace_bytes_live


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cufft_cross_batch_rejects_invalid_descriptions_without_allocating_plans():
    program = ti.lang.impl.get_runtime().prog
    create = getattr(program, "_create_cuda_cufft_cross_batch_plan", None)
    if create is None:
        pytest.skip("cross-batch FFT is unavailable")
    before = ti.hardware.fft.cache_statistics()
    for dimensions, batch in (((), 1), ((7,), 1), ((7, 0), 2), ((2**30, 4), 1), ((7, 3), 0)):
        with pytest.raises(RuntimeError):
            create(dimensions, batch)
    after = ti.hardware.fft.cache_statistics()
    assert after.create_requests == before.create_requests
    assert after.live_plans == before.live_plans
    assert after.workspace_bytes_live == before.workspace_bytes_live
    assert after.live_handles == before.live_handles


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cufft_plan_and_graph_fail_closed_after_runtime_reset():
    if not ti.hardware.fft.is_available():
        pytest.skip("a compatible optional cuFFT shared library is unavailable")

    plan = ti.hardware.fft.CufftPlan1D(64)
    source = ti.ndarray(ti.f32, shape=plan.input_shape)
    output = ti.ndarray(ti.f32, shape=plan.output_shape)
    builder = ti.graph.GraphBuilder()
    builder.append_native(plan.record(), admission="auto")
    graph = builder.compile()
    capacity = plan.memory_report().known_capacity_requested_bytes

    ti.reset()

    invalid = plan.memory_report()
    assert invalid.lifecycle_state == "runtime_invalid"
    assert invalid.known_resident_requested_bytes == 0
    assert invalid.known_capacity_requested_bytes == capacity
    with pytest.raises(RuntimeError, match="previous Taichi runtime"):
        plan.record()
    with pytest.raises(RuntimeError, match="compiled before ti.reset"):
        graph.run({"input": source, "output": output})
    plan.close()
    assert plan.memory_report().lifecycle_state == "closed"


@pytest.mark.run_in_serial
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cufft_serial_churn_releases_all_generations():
    if not ti.hardware.fft.is_available():
        pytest.skip("a compatible optional cuFFT shared library is unavailable")

    iterations = stress_iterations(32)
    length = 16
    source = ti.ndarray(ti.f32, shape=(length, 2))
    output = ti.ndarray(ti.f32, shape=(length, 2))
    host = np.zeros((length, 2), dtype=np.float32)
    host[:, 0] = np.arange(length, dtype=np.float32)
    source.from_numpy(host)
    ti.sync()
    program = ti.lang.impl.get_runtime().prog
    baseline_cache = ti.hardware.fft.cache_statistics()
    baseline_memory = program._runtime_statistics_snapshot()["memory"]
    process_memory = ProcessMemoryPlateau("cuda-cufft-churn", ("cuda-cufft",))
    process_memory.capture("before")
    midpoint = None

    for iteration in range(iterations):
        plan = ti.hardware.fft.CufftPlan1D(length)
        plan.execute(source, output)
        plan.close()
        if (iteration + 1) % 64 == 0:
            ti.sync()
        if iteration + 1 == max(1, iterations // 2):
            ti.sync()
            midpoint = program._runtime_statistics_snapshot()["memory"]
            process_memory.capture("midpoint")
            assert ti.hardware.fft.cache_statistics().live_handles == (
                baseline_cache.live_handles
            )
    ti.sync()

    final_cache = ti.hardware.fft.cache_statistics()
    final_memory = program._runtime_statistics_snapshot()["memory"]
    process_memory.capture("after")
    process_memory.finish(iterations)
    assert final_cache.live_handles == baseline_cache.live_handles
    assert final_cache.live_plans == baseline_cache.live_plans
    assert final_cache.workspace_bytes_live == baseline_cache.workspace_bytes_live
    assert final_cache.cache_misses == baseline_cache.cache_misses + iterations
    for key in ("live_resources", "retiring_resources", "inflight_resources"):
        assert midpoint[key] == baseline_memory[key]
        assert final_memory[key] == baseline_memory[key]
    assert np.isfinite(output.to_numpy()).all()
