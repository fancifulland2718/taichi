"""Cold mixed kernel/provider recording and immutable argument ownership."""

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge._lib import core
from taichi_forge.hardware import _vulkan_fft
from taichi_forge.lang import impl
from tests import test_utils
from tests.python.test_hardware_vulkan_fft import _adapter, _input


def _scale_graph(name, ndim):
    @ti.kernel
    def scale(data: ti.types.ndarray(dtype=ti.f32), factor: ti.f32):
        for index in ti.grouped(data):
            data[index] *= factor

    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        scale,
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "data", ti.f32, ndim=ndim),
        ti.graph.Arg(ti.graph.ArgKind.SCALAR, name, ti.f32),
    )
    return builder.compile()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_fixed_graph_preparation_does_not_execute_and_frames_are_immutable(
    monkeypatch,
):
    data, values = _input((16, 8), 5)
    program = impl.get_runtime().prog
    kernel_graph = _scale_graph("factor", 4)
    native = kernel_graph._spec.nodes[0].compiled_graph
    forward = _vulkan_fft.VulkanFftPlan(
        data, (16, 8), batch_count=5, _batch_tile=2, adapter_path=_adapter()
    )
    inverse = _vulkan_fft.VulkanFftPlan(
        data,
        (16, 8),
        batch_count=5,
        direction="inverse",
        normalization="inverse",
        _batch_tile=2,
    )
    sources = [
        native,
        program._vulkan_fft_graph_command(forward._handle, "data"),
        program._vulkan_fft_graph_command(inverse._handle, "data"),
        native,
    ]
    bindings = {"data": data.arr, "factor": 2.0}
    frozen_plan_facts = forward.statistics()
    first = core._prepare_vulkan_graph_recording(program, sources, bindings)
    bindings["factor"] = 3.0
    second = core._prepare_vulkan_graph_recording(program, sources, bindings)
    np.testing.assert_array_equal(data.to_numpy(), values)
    assert first.argument_bytes() > 0
    assert first.argument_bytes() == second.argument_bytes()
    assert first.uses_secondary_commands()
    assert second.uses_secondary_commands()
    assert forward.statistics() == frozen_plan_facts
    argument_bytes = first.argument_bytes()
    # Recorded resources, not the mutable open-plan table or source Graph,
    # own execution. This is distinct from legacy plan.run()/root Graph.
    forward.close()
    inverse.close()
    with pytest.raises(RuntimeError, match="source command is closed"):
        core._prepare_vulkan_graph_recording(program, sources, bindings)
    sources.clear()
    bindings["factor"] = -100.0

    def unexpected(*args, **kwargs):
        raise AssertionError("Replay must not rebuild or call Python provider code")

    monkeypatch.setattr(_vulkan_fft.VulkanFftPlan, "run", unexpected)
    monkeypatch.setattr(_vulkan_fft.VulkanFftPlan, "statistics", unexpected)
    monkeypatch.setattr(core, "_prepare_vulkan_graph_recording", unexpected)
    first.run()
    # A primary kernel after the embedded segment must rebind its pipeline.
    kernel_graph.run({"data": data, "factor": 0.5})
    second.run()
    first.run()
    np.testing.assert_allclose(
        data.to_numpy(), values * 4 * 0.5 * 9 * 4, atol=0.003, rtol=5e-5
    )
    assert first.argument_bytes() == argument_bytes
    first.close()
    second.close()
    with pytest.raises(RuntimeError, match="closed"):
        first.run()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_fixed_graph_rejects_storage_mismatch_without_partial_execution():
    data, values = _input((16, 8), 2)
    other, _ = _input((16, 8), 2)
    program = impl.get_runtime().prog
    kernel_graph = _scale_graph("factor", 4)
    with _vulkan_fft.VulkanFftPlan(
        data, (16, 8), batch_count=2, adapter_path=_adapter()
    ) as plan:
        sources = [
            kernel_graph._spec.nodes[0].compiled_graph,
            program._vulkan_fft_graph_command(plan._handle, "data"),
        ]
        with pytest.raises(RuntimeError, match="original compact storage"):
            core._prepare_vulkan_graph_recording(
                program, sources, {"data": other.arr, "factor": 7.0}
            )
        np.testing.assert_array_equal(other.to_numpy(), values)
        np.testing.assert_array_equal(data.to_numpy(), values)
        # A failed materialization must not poison the source plan.
        recording = core._prepare_vulkan_graph_recording(
            program, sources, {"data": data.arr, "factor": 1.0}
        )
        recording.close()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_fixed_graph_close_retains_pending_commands_and_reset_invalidates():
    data, values = _input((64,), 2)
    program = impl.get_runtime().prog
    kernel_graph = _scale_graph("factor", 3)
    sources = [kernel_graph._spec.nodes[0].compiled_graph]
    recording = core._prepare_vulkan_graph_recording(
        program, sources, {"data": data.arr, "factor": 4.0}
    )
    assert recording.uses_secondary_commands()
    recording.run()
    recording.close()
    np.testing.assert_array_equal(data.to_numpy(), values * 4)
    plan = _vulkan_fft.VulkanFftPlan(
        data, (64,), batch_count=2, adapter_path=_adapter()
    )
    command = program._vulkan_fft_graph_command(plan._handle, "data")
    stale = core._prepare_vulkan_graph_recording(program, [command], {"data": data.arr})
    ti.reset()
    ti.init(arch=ti.vulkan, enable_fallback=False, offline_cache=False)
    with pytest.raises(RuntimeError, match="closed|finaliz|runtime"):
        stale.run()
    stale.close()
    current, _ = _input((64,), 2)
    with pytest.raises(RuntimeError, match="source command is closed"):
        core._prepare_vulkan_graph_recording(
            impl.get_runtime().prog, [command], {"data": current.arr}
        )
    # A surviving source token must not postpone VkFFT/Vulkan destruction
    # until after the old device has gone away.
    del command, plan, stale
