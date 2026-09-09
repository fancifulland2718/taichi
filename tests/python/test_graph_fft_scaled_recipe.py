"""Explicit FFT math, provider lowering and plan-free reuse, not timing gates."""

import copy

import numpy as np
import pytest
import taichi_forge as ti

from taichi_forge.hardware import _fft_lto
from tests import test_utils
from tests.python.test_hardware_fft import _callback_compiler_paths


def _definition(
    *,
    direction="forward",
    scale=0.25,
    preparation=None,
    callbacks=False,
    dimensions=(32, 64),
    batch_count=2
):
    operation = ti.linalg.record_fft(
        dimensions,
        batch_count=batch_count,
        direction=direction,
        output_scale=scale,
        absolute_tolerance=4e-5,
        relative_tolerance=3e-5,
        preparation=preparation,
    )
    if preparation is None:
        operation.prepare(
            lto_callbacks=callbacks, **(_callback_compiler_paths() if callbacks else {})
        )
    builder = ti.graph.GraphBuilder()
    builder.append_native(operation)
    definition = builder.freeze()
    providers = (
        *ti.graph.default_recipe_providers(),
        ti.hardware.fft.FftRecipeProvider(),
    )
    return definition, operation, providers


def _data(direction, scale, *, dimensions=(32, 64), batch_count=2):
    shape = (*dimensions, 2) if batch_count == 1 else (batch_count, *dimensions, 2)
    arrays = {name: ti.ndarray(ti.f32, shape) for name in ("input", "output")}
    values = (
        np.random.default_rng(637)
        .uniform(-0.125, 0.125, arrays["input"].shape)
        .astype(np.float32)
    )
    arrays["input"].from_numpy(values)
    signal = values[..., 0] + 1j * values[..., 1]
    expected = (
        np.fft.fft2(signal, axes=(-2, -1))
        if direction == "forward"
        else np.fft.ifft2(signal, axes=(-2, -1)) * np.prod(dimensions)
    )
    return arrays, values, np.stack((expected.real, expected.imag), -1) * scale


def _framed_recipes(definition, providers):
    catalog = definition.recipe_catalog(providers=providers)
    frame = next(
        f for f in catalog.fragments if f.provider_namespace.endswith(".binding_frames")
    )
    result = [
        (
            "whole_transform",
            catalog.compose((frame.fragment_id,), stage="fft-scale-contract").recipe,
        )
    ]
    result += [
        (
            f.provider_metadata["family_selection"]["materialization_choice"],
            catalog.compose(
                (frame.fragment_id, f.fragment_id), stage="fft-scale-contract"
            ).recipe,
        )
        for f in catalog.fragments
        if f.provider_namespace.endswith(".fft")
    ]
    return result


@test_utils.test(arch=ti.cpu)
def test_fft_scale_is_a_finite_semantic_value_not_a_provider_axis():
    for value in (True, "0.25", float("nan"), float("inf"), 1e100):
        with pytest.raises((ValueError, TypeError), match="output_scale"):
            ti.linalg.record_fft(
                (32, 64),
                output_scale=value,
                absolute_tolerance=1e-5,
                relative_tolerance=1e-5,
            )


@pytest.mark.parametrize("direction,scale", (("forward", 0.25), ("inverse", -0.03125)))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_scaled_fft_plain_recipes_do_not_require_a_jit_library(
    monkeypatch, direction, scale
):
    monkeypatch.setattr(
        _fft_lto,
        "_library",
        lambda *a, **k: pytest.fail("plain FFT must not load a JIT library"),
    )
    scope = {} if direction == "forward" else {"dimensions": (15, 21), "batch_count": 1}
    definition, operation, providers = _definition(
        direction=direction, scale=scale, **scope
    )
    prepared = operation.preparation_report()
    with monkeypatch.context() as cold_failure:

        def unavailable(*args, **kwargs):
            raise RuntimeError("optional callback preparation unavailable")

        cold_failure.setattr(_fft_lto, "_StoreScaleCallback", unavailable)
        with pytest.raises(RuntimeError, match="optional callback preparation"):
            operation.prepare(lto_callbacks=True)
    assert operation.preparation_report() == prepared
    arrays, values, expected = _data(direction, scale, **scope)
    recipes = _framed_recipes(definition, providers)
    assert len(recipes) == 3
    operation.close()
    for _, recipe in recipes:
        with definition.materialize(recipe, providers=providers) as result:
            graph = result.executor
            bound = graph.bind(arrays)
            assert bound._version.execution_frame is not None
            graph.run(bound)
            np.testing.assert_allclose(
                arrays["output"].to_numpy(), expected, rtol=3e-5, atol=4e-5
            )
            arrays["input"].from_numpy(values * 2)
            graph.run(bound)
            np.testing.assert_allclose(
                arrays["output"].to_numpy(), expected * 2, rtol=3e-5, atol=8e-5
            )
            arrays["input"].from_numpy(values)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_scaled_fft_callback_search_resolve_and_replay_are_provider_owned(monkeypatch):
    definition, operation, providers = _definition(callbacks=True)
    preparation = operation.preparation_artifact()
    # Returned nested compiler facts cannot mutate the frozen source.
    changed = operation.preparation_report()
    changed["whole_transform_store_scale"]["callback"]["compiler"][
        "sha256"
    ] = "not-the-source"
    assert operation.preparation_artifact() == preparation
    arrays, values, expected = _data("forward", 0.25)
    observed = set()

    def evaluate(graph, recipe):
        bound = graph.bind(arrays)
        graph.run(bound)
        np.testing.assert_allclose(
            arrays["output"].to_numpy(), expected, rtol=3e-5, atol=4e-5
        )
        record = next(
            lease
            for lease in graph._spec.lifetime_leases
            if hasattr(lease, "_graph_fft_strategy")
        )
        observed.add(record._graph_fft_strategy)
        return {
            "contract_choice": float(
                record._graph_fft_strategy == "whole_transform_store_scale"
            )
            + float(bound._version.execution_frame is not None)
        }

    decision = definition.search_recipes(
        providers=providers,
        target=ti.graph.GraphOptimizationTarget(
            objectives=(("contract_choice", "max"),)
        ),
        budget=ti.graph.GraphSearchBudget(evaluation_limit=16, repeat_count=1),
        strategy=ti.graph.GraphRecipeSearchStrategy(mode="exact_if_bounded"),
    ).run(evaluate)
    assert decision.status == "selected" and decision.report.search_complete
    assert observed == set(preparation["plans"])
    assert "cufft-store-scale-lto-v1" in decision.report.to_json()
    operation.close()
    before = ti.hardware.fft.cache_statistics().create_requests
    restored, owner, restored_providers = _definition(preparation=preparation)
    selection = restored.resolve_recipe(
        decision.selection_artifact, providers=restored_providers
    )
    assert ti.hardware.fft.cache_statistics().create_requests == before
    with restored.materialize(selection) as result:
        graph = result.executor
        bound = graph.bind(arrays)
        assert ti.hardware.fft.cache_statistics().create_requests == before + 1
        owner.close()
        monkeypatch.setattr(
            _fft_lto,
            "_StoreScaleCallback",
            lambda *a, **k: pytest.fail("callback compile on replay"),
        )
        arrays["input"].from_numpy(values * 3)
        for _ in range(9):
            graph.run(bound)
        np.testing.assert_allclose(
            arrays["output"].to_numpy(), expected * 3, rtol=3e-5, atol=1.2e-4
        )


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_fft_callback_drift_rejects_only_callback_materialization():
    _, operation, _ = _definition(callbacks=True)
    preparation = copy.deepcopy(operation.preparation_artifact())
    operation.close()
    preparation["plans"]["whole_transform_store_scale"]["callback"][
        "source_sha256"
    ] = "drift"
    definition, owner, providers = _definition(preparation=preparation)
    recipes = dict(_framed_recipes(definition, providers))
    before = ti.hardware.fft.cache_statistics().create_requests
    with pytest.raises(RuntimeError, match="contract drifted"):
        with definition.materialize(
            recipes["whole_transform_store_scale"], providers=providers
        ):
            pass
    assert ti.hardware.fft.cache_statistics().create_requests == before
    arrays, _, expected = _data("forward", 0.25)
    with definition.materialize(
        recipes["whole_transform"], providers=providers
    ) as result:
        bound = result.executor.bind(arrays)
        result.executor.run(bound)
        np.testing.assert_allclose(
            arrays["output"].to_numpy(), expected, rtol=3e-5, atol=4e-5
        )
    owner.close()
