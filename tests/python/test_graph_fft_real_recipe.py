"""Real FFT math, destructive-input effects and frozen Graph reuse contracts."""

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=ti.cpu)
def test_real_fft_rejects_inconsistent_directions_before_provider_loading():
    for transform, direction in (("r2c", "inverse"), ("c2r", "forward")):
        with pytest.raises(ValueError, match="requires direction"):
            ti.linalg.record_fft(
                (16, 32), transform=transform, direction=direction, absolute_tolerance=1e-4, relative_tolerance=1e-4
            )


@pytest.mark.parametrize("transform", ("r2c", "c2r"))
@pytest.mark.parametrize("dimensions,batch", (((15, 21), 1), ((32, 64), 2)))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_real_fft_scaled_frames_search_effects_and_plan_free_restore(monkeypatch, transform, dimensions, batch):
    from taichi_forge.linalg import _fft
    from taichi_forge.graph._ir import GraphAccess

    scale = 0.25 if transform == "r2c" else 1 / np.prod(dimensions)

    def operation(preparation=None):
        return ti.linalg.record_fft(
            dimensions,
            batch_count=batch,
            transform=transform,
            output_scale=float(scale),
            preparation=preparation,
            absolute_tolerance=1e-4,
            relative_tolerance=1e-4,
        )

    op = operation()
    assert tuple(op.prepare()) == ("whole_transform",)
    with pytest.raises(ValueError, match="C2C LTO"):
        op.prepare(lto_callbacks=True)
    artifact = op.preparation_artifact()
    prepared = op._catalog._recording("whole_transform")
    expected_access = GraphAccess.READ_WRITE if transform == "c2r" else GraphAccess.READ
    assert prepared.resource_effects[0].access == expected_access
    assert _fft._FftDescriptionRecording(op._catalog).resource_effects[0].access == expected_access
    builder = ti.graph.GraphBuilder()
    builder.append_native(op)
    definition = builder.freeze()
    providers = (*ti.graph.default_recipe_providers(), ti.hardware.fft.FftRecipeProvider())
    catalog = definition.recipe_catalog(providers=providers)
    # No fictitious real-FFT algorithm candidate. Executor composition is real.
    assert not any(f.provider_namespace.endswith(".fft") for f in catalog.fragments)
    arrays = {
        name: ti.ndarray(ti.f32, shape) for name, shape in (("input", op._shape()), ("output", op._shape(output=True)))
    }
    real_shape = dimensions if batch == 1 else (batch, *dimensions)
    signal = np.random.default_rng(602).uniform(-0.125, 0.125, real_shape).astype(np.float32)
    spectrum = np.fft.rfft2(signal, axes=(-2, -1))
    initial = signal if transform == "r2c" else np.stack((spectrum.real, spectrum.imag), -1).astype(np.float32)
    expected = np.stack((spectrum.real, spectrum.imag), -1) * scale if transform == "r2c" else signal
    calls = []

    def evaluate(graph, recipe):
        bound = graph.bind(arrays)
        for multiplier in (1.0, 2.0):
            arrays["input"].from_numpy(initial * multiplier)  # C2R input must be replenished
            graph.run(bound)
            np.testing.assert_allclose(arrays["output"].to_numpy(), expected * multiplier, rtol=1e-4, atol=1e-4)
            if transform == "r2c":
                np.testing.assert_array_equal(arrays["input"].to_numpy(), initial * multiplier)
        framed = bound._version.execution_frame is not None
        calls.append(framed)
        return {"contract_choice": float(framed)}

    decision = definition.search_recipes(
        providers=providers,
        target=ti.graph.GraphOptimizationTarget(objectives=(("contract_choice", "max"),)),
        budget=ti.graph.GraphSearchBudget(evaluation_limit=8, repeat_count=1),
        strategy=ti.graph.GraphRecipeSearchStrategy(mode="exact_if_bounded"),
    ).run(evaluate)
    assert any(calls)
    assert decision.status == "selected"
    op.close()
    with monkeypatch.context() as cold:
        cold.setattr(_fft, "CufftPlanND", lambda *a, **k: pytest.fail("restore must not create plans"))
        restored = operation(artifact)
        restored_builder = ti.graph.GraphBuilder()
        restored_builder.append_native(restored)
        equivalent = restored_builder.freeze()
    assert equivalent.semantic_graph_id == definition.semantic_graph_id
    # Same stable selection is resolvable after plan-free reconstruction.
    selection = equivalent.resolve_recipe(decision.selection_artifact, providers=providers)
    with equivalent.materialize(selection) as result:
        evaluate(result.executor, selection)
