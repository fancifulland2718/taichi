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


def _definition(data, dimensions, batches, *, recipe_owned=False):
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
        builder.append_native(plan.record(recipe_owned=recipe_owned))
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
    definition, plans = _definition(data, (262144,), 4, recipe_owned=True)
    for plan in plans:
        plan.close()
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
    restored, restored_plans = _definition(
        restored_data, (262144,), 4, recipe_owned=True
    )
    for plan in restored_plans:
        plan.close()
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


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_fft_detached_sources_materialize_only_selected_plans(monkeypatch):
    import gc
    import weakref

    from taichi_forge.hardware import _vulkan_fft as fft

    data, original = _input((64,), 5)
    definition, plans = _definition(data, (64,), 5, recipe_owned=True)
    references = tuple(weakref.ref(plan) for plan in plans)
    for plan in plans:
        plan.close()
    del plan, plans
    gc.collect()
    assert all(reference() is None for reference in references)
    assert not definition._runtime_spec.provider_memory_reports()
    # Discovery and equivalent freezing remain descriptive after source close.
    with monkeypatch.context() as cold:
        cold.setattr(fft.VulkanFftPlan, "__init__", _forbidden)
        catalog = definition.recipe_catalog(providers=_providers())
    frame = next(
        f for f in catalog.fragments if f.fragment_key.endswith(":secondary-frames")
    )
    tiles = tuple(
        f for f in catalog.fragments if f.fragment_key.endswith(":batch-tile-2")
    )
    recipes = (
        catalog.baseline.recipe,
        catalog.compose((tiles[0].fragment_id,), stage="partial").recipe,
        catalog.compose(
            tuple(f.fragment_id for f in (*tiles, frame)), stage="complete"
        ).recipe,
    )
    create = fft.VulkanFftPlan.__init__
    created = []

    def counted(plan, *args, **kwargs):
        create(plan, *args, **kwargs)
        created.append(weakref.ref(plan))

    monkeypatch.setattr(fft.VulkanFftPlan, "__init__", counted)
    for recipe in recipes:
        data.from_numpy(original)
        created.clear()
        with definition.materialize(recipe, providers=_providers()) as selected:
            graph = selected.executor
            assert len(created) == 2  # Partial replacement must not build 3 plans.
            assert len(graph._spec.provider_memory_reports()) == 2
            binding = graph.bind({"data": data, "factor": -1.0})
            with monkeypatch.context() as replay:
                replay.setattr(fft, "_recreate_recording", _forbidden)
                replay.setattr(
                    fft._FrozenVulkanFftSource, "validate_graph_lifetime", _forbidden
                )
                graph.run(binding)
            # Closing before a readback must keep pending native command leases.
        np.testing.assert_allclose(data.to_numpy(), original, atol=2e-5, rtol=5e-5)
        del graph, binding, selected
        gc.collect()
        assert all(reference() is None for reference in created)
    # A baseline compile and a selected graph own independent plans. Releasing
    # one must not break the other or require source reconstruction on replay.
    baseline = definition.compile()
    with definition.materialize(recipes[-1], providers=_providers()) as selected:
        baseline.run(baseline.bind({"data": data, "factor": -1.0}))
        del baseline
        gc.collect()
        selected.executor.run(selected.executor.bind({"data": data, "factor": -1.0}))
        np.testing.assert_allclose(data.to_numpy(), original, atol=3e-5, rtol=5e-5)


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_fft_detached_source_drift_and_rollback_are_cold(monkeypatch):
    import gc
    import weakref

    from taichi_forge.hardware import _vulkan_fft as fft

    data, _ = _input((64,), 3)
    definition, plans = _definition(data, (64,), 3, recipe_owned=True)
    for plan in plans:
        plan.close()
    create = fft.VulkanFftPlan.__init__
    created = []

    def fail_second(plan, *args, **kwargs):
        if created:
            raise RuntimeError("injected second plan failure")
        create(plan, *args, **kwargs)
        created.append(weakref.ref(plan))

    with monkeypatch.context() as failure:
        failure.setattr(fft.VulkanFftPlan, "__init__", fail_second)
        with pytest.raises(RuntimeError, match="second plan failure"):
            definition.compile()
    gc.collect()
    assert created[0]() is None
    with monkeypatch.context() as drift:
        drift.setattr(fft, "_binary_sha256", lambda path: "different-adapter")
        with pytest.raises(ValueError, match="adapter changed"):
            definition.compile()

    def changed_facts(plan, *args, **kwargs):
        create(plan, *args, **kwargs)
        plan._physical_id = "different-shader-or-workspace-facts"

    with monkeypatch.context() as drift:
        drift.setattr(fft.VulkanFftPlan, "__init__", changed_facts)
        with pytest.raises(ValueError, match="frozen physical facts"):
            definition.compile()
    assert fft.passive_status()["native_facts"]["open_plan_count"] == 0
    # Neither a drift nor rollback poisons the source description.
    graph = definition.compile()
    graph.run(graph.bind({"data": data, "factor": 1.0}))
    ti.sync()
    ti.reset()
    ti.init(arch=ti.vulkan, offline_cache=False)
    with pytest.raises(RuntimeError, match="previous runtime"):
        definition.compile()
