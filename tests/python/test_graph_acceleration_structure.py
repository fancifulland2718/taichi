"""Typed JIT Graph AS bindings, immutable recording and resource retirement."""

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge._lib import core
from taichi_forge.graph._recipes.binding_frames import GraphBindingFrameRecipeProvider
from taichi_forge.graph._recipes.families import GraphRuntimeAssemblyProvider
from tests import test_utils


def _definition(*, label=""):
    @ti.kernel
    def query(
        scene: ti.types.acceleration_structure(),
        indices: ti.types.ndarray(dtype=ti.i32, ndim=2),
        values: ti.types.ndarray(dtype=ti.f32, ndim=2),
    ):
        for i in range(indices.shape[0]):
            hit = scene.trace_closest(
                ti.Vector([ti.cast(i % 2, ti.f32) * 4.0, 0.0, 2.0]),
                ti.Vector([0.0, 0.0, -1.0]),
            )
            indices[i, 0] = hit.hit
            indices[i, 1] = ti.cast(hit.primitive_index, ti.i32)
            indices[i, 2] = ti.cast(hit.instance_id, ti.i32)
            indices[i, 3] = ti.cast(hit.instance_custom_index, ti.i32)
            values[i, 0] = hit.t
            values[i, 1] = hit.barycentric_u
            values[i, 2] = hit.barycentric_v

    @ti.kernel
    def consume(
        indices: ti.types.ndarray(dtype=ti.i32, ndim=2),
        values: ti.types.ndarray(dtype=ti.f32, ndim=2),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in output:
            output[i] = values[i, 0] if indices[i, 0] else -17.0

    builder = ti.graph.GraphBuilder()
    scene = ti.graph.Arg(ti.graph.ArgKind.ACCELERATION_STRUCTURE, "scene")
    indices = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "indices", ti.i32, ndim=2)
    values = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", ti.f32, ndim=2)
    output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)
    builder.dispatch(query, scene, indices, values, label=label)
    builder.dispatch(consume, indices, values, output)
    return builder.freeze()


def _scene(blas, custom, z):
    return ti.hardware.ray.InstanceTLAS(
        [
            ti.hardware.ray.RayInstance(
                blas,
                transform=(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, z),
                custom_index=custom,
            )
        ]
    )


def _resources():
    if not ti.hardware.ray.is_available():
        pytest.skip("Vulkan ray query features are unavailable")
    vertices = ti.ndarray(ti.f32, (3, 3))
    triangles = ti.ndarray(ti.i32, (1, 3))
    vertices.from_numpy(np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], np.float32))
    triangles.from_numpy(np.array([[0, 1, 2]], np.int32))
    blas = ti.hardware.ray.TriangleBLAS(vertices, triangles)
    scenes = [_scene(blas, 0xFFFFFD, 0), _scene(blas, 123, -1)]
    arrays = dict(
        indices=ti.ndarray(ti.i32, (18, 4)),
        values=ti.ndarray(ti.f32, (18, 3)),
        output=ti.ndarray(ti.f32, 18),
    )
    arrays["output"].fill(-53)
    return blas, scenes, arrays


def _assert_result(arrays, custom, distance):
    indices = arrays["indices"].to_numpy()
    values = arrays["values"].to_numpy()
    np.testing.assert_array_equal(indices[::2], np.tile([1, 0, 0, custom], (9, 1)))
    np.testing.assert_array_equal(indices[1::2], np.tile([0, -1, -1, -1], (9, 1)))
    np.testing.assert_allclose(values[::2], np.tile([distance, 0.25, 0.5], (9, 1)))
    np.testing.assert_array_equal(arrays["output"].to_numpy(), [distance, -17] * 9)


@pytest.mark.parametrize("retirement", ["close", "reset"])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_as_binding_frames_rebind_refit_and_retain_native_resources(retirement, monkeypatch):
    blas, scenes, arrays = _resources()
    definition = _definition(label="inline-ray")
    metadata = definition._runtime_spec.nodes[0].compiled_graph._dispatch_metadata[0]
    assert any(
        effect["arg_id"] == [0] and effect["access"] == "read"
        for effect in metadata["effects"]
    )
    providers = (GraphRuntimeAssemblyProvider(), GraphBindingFrameRecipeProvider())
    catalog = definition.recipe_catalog(providers=providers)
    assert len(catalog.entries()) == 2
    candidate = next(entry.recipe for entry in catalog.entries() if entry.recipe.fragments)
    assert candidate.planned_physical_id != catalog.baseline.recipe.planned_physical_id
    assert any(
        fragment.provider_namespace.endswith(".binding_frames")
        for fragment in definition.recipe_catalog().fragments
    )
    with definition.materialization_context(provider_set=catalog.provider_set) as context:
        with context.materialize(catalog.baseline.recipe) as baseline:
            # Same-shaped AS rebinding must invalidate cached launch contexts.
            for index in (0, 1, 0):
                baseline.executor.run(dict(scene=scenes[index], **arrays))
                _assert_result(arrays, (0xFFFFFD, 123)[index], 2 + index)
        with context.materialize(candidate) as materialized:
            graph = materialized.executor
            arrays["output"].fill(-53)
            bindings = [graph.bind(dict(scene=scene, **arrays)) for scene in scenes]
            np.testing.assert_array_equal(arrays["output"].to_numpy(), [-53] * 18)
            frames = [binding._version.execution_frame for binding in bindings]
            assert all(frame.uses_secondary_commands() for frame in frames)
            argument_bytes = [frame.argument_bytes() for frame in frames]
            version = bindings[0]._version
            with pytest.raises((ValueError, RuntimeError, ti.TaichiRuntimeError)):
                bindings[0].update(scene=blas)
            assert bindings[0]._version is version

            def unexpected(*args, **kwargs):
                raise AssertionError("Replay must not rebuild or validate Python AS descriptors")

            with monkeypatch.context() as patch:
                patch.setattr(core, "_prepare_vulkan_graph_recording", unexpected)
                for scene in scenes:
                    patch.setattr(scene, "_kernel_resource_descriptor", unexpected)
                for index in (1, 0, 1):
                    graph.run(bindings[index])
                    _assert_result(arrays, (0xFFFFFD, 123)[index], 2 + index)
                # Existing AS build barriers order this update before the next
                # inline query; immutable resource identity does not freeze data.
                scenes[0].refit(
                    [
                        ti.hardware.ray.RayInstance(
                            blas,
                            custom_index=321,
                            transform=(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -2),
                        )
                    ]
                )
                graph.run(bindings[0])
                _assert_result(arrays, 321, 4)
            assert [frame.argument_bytes() for frame in frames] == argument_bytes
            for scene in scenes:
                scene.close()
            blas.close()
            with pytest.raises(RuntimeError, match="closed"):
                graph.bind(dict(scene=scenes[0], **arrays))
            # Command buffers retain the actual TLAS/BLAS and backing buffers,
            # not merely mutable Python wrappers or the open resource table.
            graph.run(bindings[0])
            _assert_result(arrays, 321, 4)
            if retirement == "reset":
                ti.reset()
            else:
                graph.close()
                ti.sync()
            assert all(frame.argument_bytes() == 0 for frame in frames)
            with pytest.raises(RuntimeError, match="closed|retired|finalized"):
                frames[0].run()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_as_recipe_fork_search_and_equivalent_definition_resolution():
    blas, scenes, arrays = _resources()
    definition = _definition()
    providers = (GraphRuntimeAssemblyProvider(), GraphBindingFrameRecipeProvider())
    session = definition.search_recipes(
        providers=providers,
        target=ti.graph.GraphOptimizationTarget(objectives=(("prepared_binding_candidate", "max"),)),
        budget=ti.graph.GraphSearchBudget(evaluation_limit=4, repeat_count=1),
        strategy=ti.graph.GraphRecipeSearchStrategy(mode="exact_if_bounded"),
    )
    observed = set()

    def evaluate(graph, request):
        binding = graph.bind(dict(scene=scenes[0], **arrays))
        graph.run(binding)
        _assert_result(arrays, 0xFFFFFD, 2)
        observed.add(request.recipe_id)
        # Tests reachability only, never a synthetic acceleration claim.
        return {"prepared_binding_candidate": float(binding._version.execution_frame is not None)}

    decision = session.run(evaluate)
    assert decision.status == "selected", decision.report.results
    assert decision.report.search_complete and len(observed) == 2
    fresh = _definition()
    assert fresh.semantic_graph_id == definition.semantic_graph_id
    selection = fresh.resolve_recipe(decision.selection_artifact, providers=providers)
    with fresh.materialize(selection) as materialized:
        evaluate(materialized.executor, selection)
    for scene in scenes:
        scene.close()
    blas.close()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_as_symbolic_graph_is_jit_only_and_discovery_does_not_allocate(monkeypatch):
    if not ti.hardware.ray.is_available():
        pytest.skip("Vulkan ray query features are unavailable")

    def unexpected(*args, **kwargs):
        raise AssertionError("Compiling a symbolic AS Graph must not allocate an AS")

    monkeypatch.setattr(ti.hardware.ray.InstanceTLAS, "__init__", unexpected)

    @ti.kernel
    def distances(scene: ti.types.acceleration_structure(), output: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in output:
            hit = scene.trace_closest(ti.Vector([0.0, 0.0, 1.0]), ti.Vector([0.0, 0.0, -1.0]))
            output[i] = hit.t

    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        distances,
        ti.graph.Arg(ti.graph.ArgKind.ACCELERATION_STRUCTURE, "scene"),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1),
    )
    metadata = builder.freeze()._runtime_spec.nodes[0].compiled_graph._dispatch_metadata[0]
    assert not metadata["elementwise"]
    # Existing intrinsic side-effect blocking remains intact; this change
    # completes the native resource facts, not ray fusion eligibility.
    assert metadata["blocker"] == "unsupported_side_effect"
    assert any(
        effect["arg_id"] == [0] and effect["access"] == "read"
        for effect in metadata["effects"]
    )
    definition = _definition()
    assert GraphBindingFrameRecipeProvider().fragments(definition)
    catalog = definition.recipe_catalog(providers=(GraphRuntimeAssemblyProvider(),))
    with definition.materialization_context(provider_set=catalog.provider_set) as context:
        with context.materialize(catalog.baseline.recipe) as materialized:
            module = ti.aot.Module(ti.vulkan)
            with pytest.raises(RuntimeError, match="JIT-only acceleration"):
                module.add_graph("ray", materialized.executor)
