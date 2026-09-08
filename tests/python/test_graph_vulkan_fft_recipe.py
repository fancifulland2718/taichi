"""Real mixed Graph batch/submission recipes and retained binding ownership."""

import json

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils
from tests.python.test_hardware_vulkan_fft import _adapter, _input


@ti.kernel
def _scale(data: ti.types.ndarray(dtype=ti.f32), factor: ti.f32):
    for index in ti.grouped(data):
        data[index] *= factor


def _definition(data, dimensions, batches):
    plans = [
        ti.hardware.fft.VulkanFftPlan(
            data, dimensions, batch_count=batches, adapter_path=_adapter()
        ),
        ti.hardware.fft.VulkanFftPlan(
            data,
            dimensions,
            batch_count=batches,
            direction="inverse",
            normalization="inverse",
        ),
    ]
    builder = ti.graph.GraphBuilder()
    arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "data", ti.f32, ndim=len(data.shape))
    factor = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "factor", ti.f32)
    builder.dispatch(_scale, arg, factor)
    for plan in plans:
        builder.append_native(plan.record())
    builder.dispatch(_scale, arg, factor)
    return builder.freeze(), plans


def _providers():
    return (
        *ti.graph.default_recipe_providers(),
        ti.hardware.fft.VulkanFftRecipeProvider(),
    )


def _forbidden(*args, **kwargs):
    raise AssertionError("Replay called cold plan/binding/provider work")


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_fft_complete_recipes_compose_real_partitions_and_immutable_frames(
    monkeypatch,
):
    from taichi_forge.graph._recipes.vulkan_binding_frames import (
        VulkanBindingFrameExecutor,
    )
    from taichi_forge.hardware._vulkan_fft import VulkanFftPlan, _Recording

    data, original = _input((32, 16), 5)
    definition, plans = _definition(data, (32, 16), 5)
    catalog = definition.recipe_catalog(providers=_providers())
    fragments = [
        f for f in catalog.fragments if f.provider_namespace.endswith(".vulkan_fft")
    ]
    frame = next(
        f
        for f in fragments
        if f.provider_metadata["family_selection"]["source_key"]
        == "whole-graph-bindings"
    )
    tiles = [
        f
        for f in fragments
        if f.provider_metadata["family_selection"]["choice_id"] == "batch-tile-2"
    ]
    assert len(tiles) == 2
    choices = [catalog.baseline.recipe]
    for selection in ((frame,), tuple(tiles), (*tiles, frame)):
        choices.append(
            catalog.compose(
                tuple(f.fragment_id for f in selection), stage="contract"
            ).recipe
        )
    physical_ids = set()
    for index, recipe in enumerate(choices):
        data.from_numpy(original)
        with definition.materialize(recipe, providers=_providers()) as materialized:
            graph = materialized.executor
            bindings = [
                graph.bind({"data": data, "factor": value}) for value in (-1.0, 2.0)
            ]
            np.testing.assert_array_equal(data.to_numpy(), original)
            embedded = index in (1, 3)
            assert (
                graph._instance.physical_submission_mode
                == "vulkan_secondary_immutable_argument_frames"
            ) == embedded
            before = graph.execution_stats().memory.persistent_bytes
            with monkeypatch.context() as replay:
                replay.setattr(VulkanFftPlan, "__init__", _forbidden)
                replay.setattr(VulkanFftPlan, "statistics", _forbidden)
                replay.setattr(_Recording, "validate_graph_bindings", _forbidden)
                if embedded:
                    replay.setattr(VulkanFftPlan, "run", _forbidden)
                    replay.setattr(VulkanFftPlan, "validate_graph_lifetime", _forbidden)
                    replay.setattr(VulkanBindingFrameExecutor, "_frame", _forbidden)
                for version in (0, 1, 0):
                    graph.run(bindings[version])
            np.testing.assert_allclose(
                data.to_numpy(), original * 4, atol=0.002, rtol=5e-5
            )
            if embedded:
                assert before > 0
                assert graph.execution_stats().memory.persistent_bytes == before
            physical_ids.add(materialized.manifest.materialized_physical_id)
    assert len(physical_ids) == len(choices)
    for plan in plans:
        plan.close()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_fft_binding_mismatch_is_cold_and_close_retires_pending_graph():
    data, original = _input((64,), 3)
    other, _ = _input((64,), 3)
    definition, plans = _definition(data, (64,), 3)
    catalog = definition.recipe_catalog(providers=_providers())
    frame = next(
        f for f in catalog.fragments if f.fragment_key.endswith(":secondary-frames")
    )
    recipe = catalog.compose((frame.fragment_id,), stage="contract").recipe
    materialized = definition.materialize(recipe, providers=_providers())
    graph = materialized.executor
    with pytest.raises(RuntimeError, match="original data"):
        graph.bind({"data": other, "factor": 3.0})
    np.testing.assert_array_equal(other.to_numpy(), original)
    bindings = graph.bind({"data": data, "factor": 2.0})
    graph.run(bindings)
    materialized.close()
    for plan in plans:
        plan.close()
    np.testing.assert_allclose(data.to_numpy(), original * 4, atol=0.002, rtol=5e-5)


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_fft_public_search_report_and_equivalent_graph_resolution():
    data, original = _input((262144,), 4)
    definition, plans = _definition(data, (262144,), 4)
    observed = set()

    def evaluate(graph, recipe):
        data.from_numpy(original)
        graph.run(graph.bind({"data": data, "factor": -1.0}))
        np.testing.assert_allclose(data.to_numpy(), original, atol=4e-5, rtol=4e-5)
        observed.add(recipe.recipe_id)
        reports = graph._spec.provider_memory_reports()
        return {
            "requested_plan_bytes": float(
                sum(report.known_resident_requested_bytes for report in reports)
            )
        }

    decision = definition.search_recipes(
        engine="compileiq",
        providers=_providers(),
        target=ti.graph.GraphOptimizationTarget(
            objectives=(("requested_plan_bytes", "min"),)
        ),
        budget=ti.graph.GraphSearchBudget(evaluation_limit=18),
        workload_context=ti.graph.GraphWorkloadContext(
            {"case": "four-long-transforms-roundtrip"}
        ),
        evaluation_contract=ti.graph.GraphEvaluationContract(
            {"metric": "requested-plan-bytes-not-driver-vram"}
        ),
        backend_environment=ti.graph.GraphBackendEnvironment(
            {"fixture": "current-vulkan"}
        ),
    ).run(evaluate)
    assert decision.selection is not None, decision.report.to_dict()["search"]
    assert len(observed) >= 6
    assert decision.selection.recipe_id != definition.baseline_recipe.recipe_id
    restored_data, _ = _input((262144,), 4)
    restored, restored_plans = _definition(restored_data, (262144,), 4)
    assert restored.semantic_graph_id == definition.semantic_graph_id
    artifact = json.loads(json.dumps(decision.selection_artifact.to_dict()))
    resolved = restored.resolve_recipe(artifact, providers=_providers())
    with restored.materialize(resolved, providers=_providers()) as materialized:
        materialized.executor.run(
            materialized.executor.bind({"data": restored_data, "factor": -1.0})
        )
        np.testing.assert_allclose(
            restored_data.to_numpy(), original, atol=4e-5, rtol=4e-5
        )
    wrong_data, _ = _input((131072,), 4)
    wrong, wrong_plans = _definition(wrong_data, (131072,), 4)
    with pytest.raises(ValueError):
        wrong.resolve_recipe(artifact, providers=_providers())
    report = ti.graph.GraphOptimizationReportV2.from_json(decision.report.to_json())
    assert report.to_dict() == decision.report.to_dict()
    assert "vulkan_fft" in report.to_json()
    for plan in (*plans, *restored_plans, *wrong_plans):
        plan.close()
