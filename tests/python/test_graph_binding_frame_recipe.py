"""Complete recipe reachability, binding publication and physical ownership."""

import gc
import json

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge._lib import core
from taichi_forge.graph._recipes.binding_frames import GraphBindingFrameRecipeProvider
from taichi_forge.graph._recipes.families import GraphRuntimeAssemblyProvider
from tests import test_utils


def _definition(size=131):
    @ti.kernel
    def stage(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        scratch: ti.types.ndarray(dtype=ti.i32, ndim=1),
        scale: ti.i32,
    ):
        for i in source:
            scratch[i] = source[i] * scale

    @ti.kernel
    def finish(
        scratch: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in output:
            output[i] = scratch[i] + 7

    source = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.i32, ndim=1)
    output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1)
    scale = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "scale", ti.i32)
    builder = ti.graph.GraphBuilder()
    scratch = builder.private_ndarray("scratch", ti.i32, size)
    builder.dispatch(stage, source, scratch, scale)
    builder.dispatch(finish, scratch, output)
    return builder.freeze()


def _catalog(definition):
    if definition.backend == "cuda" and not core._CudaGraphBindingExecutor.available():
        pytest.skip("CUDA binding-frame execution is unavailable")
    return definition.recipe_catalog(providers=(GraphRuntimeAssemblyProvider(), GraphBindingFrameRecipeProvider()))


def _arrays(size=131):
    source = ti.ndarray(ti.i32, size)
    output = ti.ndarray(ti.i32, size)
    host = np.arange(size, dtype=np.int32) + 1
    source.from_numpy(host)
    output.fill(-53)
    return source, output, host


def _texture_definition():
    @ti.kernel
    def sample(
        image: ti.types.texture(num_dimensions=2),
        scratch: ti.types.ndarray(dtype=ti.f32, ndim=2),
    ):
        for i, j in scratch:
            scratch[i, j] = image.fetch(ti.Vector([i, j]), 0).x

    @ti.kernel
    def finish(
        scratch: ti.types.ndarray(dtype=ti.f32, ndim=2),
        output: ti.types.ndarray(dtype=ti.f32, ndim=2),
    ):
        for i, j in output:
            output[i, j] = scratch[i, j] * 2 + 1

    image = ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "image", ndim=2)
    output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=2)
    builder = ti.graph.GraphBuilder()
    scratch = builder.private_ndarray("scratch", ti.f32, (17, 23))
    builder.dispatch(sample, image, scratch, label="sample")
    builder.dispatch(finish, scratch, output, label="consume")
    return builder.freeze()


@pytest.mark.parametrize("retirement", ["close", "reset"])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_texture_binding_frames_own_resources_and_only_prepare_uploads(retirement):
    from taichi_forge.lang import impl

    definition = _texture_definition()
    catalog = _catalog(definition)
    recipe = next(entry.recipe for entry in catalog.entries() if entry.recipe.fragments)
    assert core._CudaGraphBindingExecutor.supports_sampled_texture_bindings()
    source = ti.ndarray(ti.f32, (17, 23))
    output = ti.ndarray(ti.f32, (17, 23))
    output.fill(-53)
    textures = [ti.Texture(ti.Format.r32f, (17, 23)) for _ in range(2)]
    for index, texture in enumerate(textures):
        source.fill(index + 3)
        texture.from_ndarray(source)
    program = impl.get_runtime().prog
    cold_stats = dict(program._debug_texture_resource_stats())
    native_textures = [texture.tex for texture in textures]
    with definition.materialization_context(provider_set=catalog.provider_set) as context:
        with context.materialize(recipe) as materialized:
            graph = materialized.executor
            native = graph._instance._backend_executable._native
            bindings = [graph.bind(dict(image=texture, output=output)) for texture in textures]
            np.testing.assert_array_equal(output.to_numpy(), np.full((17, 23), -53, np.float32))
            before = native.snapshot()
            for index in (0, 1, 0, 1, 1):
                graph.submit(bindings[index]).wait()
                np.testing.assert_array_equal(output.to_numpy(), np.full((17, 23), 7 + 2 * index, np.float32))
            after = native.snapshot()
            assert after["preparation_upload_calls"] == before["preparation_upload_calls"]
            assert after["argument_bytes"] == before["argument_bytes"]
            assert after["executables"] == 1
            source.fill(9)
            textures[0].from_ndarray(source)
            graph.run(bindings[0])
            np.testing.assert_array_equal(output.to_numpy(), np.full((17, 23), 19, np.float32))
            version = bindings[0]._version
            with pytest.raises((ValueError, RuntimeError, ti.TaichiRuntimeError)):
                bindings[0].update(image=output)
            assert bindings[0]._version is version

            # A published immutable frame owns its resources, not the source
            # wrapper. New preparation from a retired wrapper must still fail.
            for texture in textures:
                texture._delete_runtime_texture()
            with pytest.raises((ValueError, RuntimeError, ti.TaichiRuntimeError)):
                graph.bind(dict(image=textures[0], output=output))
            graph.run(bindings[0])
            np.testing.assert_array_equal(output.to_numpy(), np.full((17, 23), 19, np.float32))
            if retirement == "reset":
                ti.reset()
            else:
                graph.close()
                ti.sync()
                stats = dict(program._debug_texture_resource_stats())
                assert stats["released_total"] >= cold_stats["released_total"] + len(textures)
                assert stats["retiring"] == stats["inflight"] == stats["release_errors"] == 0
            assert native.snapshot()["closed"] == 1
            assert native.snapshot()["argument_bytes"] == 0
            with pytest.raises(RuntimeError, match="closed|executor"):
                native.run(version.execution_frame)
            assert native_textures


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_binding_recipe_publication_replay_update_and_failure_are_transactional():
    definition = _definition()
    catalog = _catalog(definition)
    assert len(catalog.entries()) == 2
    recipe = next(entry.recipe for entry in catalog.entries() if entry.recipe.fragments)
    assert recipe.planned_physical_id != catalog.baseline.recipe.planned_physical_id
    default = definition.recipe_catalog()
    assert any(fragment.provider_namespace.endswith(".binding_frames") for fragment in default.fragments)
    source, output, host = _arrays()
    with definition.materialization_context(provider_set=catalog.provider_set) as context:
        with context.materialize(recipe) as materialized:
            graph = materialized.executor
            native = graph._instance._backend_executable._native
            assert materialized.manifest.submissions[0].replay_mode == "cuda_immutable_argument_frames_exec_reuse"
            bindings = graph.bind(dict(source=source, output=output, scale=3))
            version = bindings._version
            assert version.execution_frame is not None
            assert "scratch" not in version.arguments
            assert "scratch" in version.execution_arguments
            np.testing.assert_array_equal(output.to_numpy(), np.full(host.size, -53, np.int32))
            before = native.snapshot()
            for _ in range(29):
                graph.run(bindings)
            np.testing.assert_array_equal(output.to_numpy(), host * 3 + 7)
            after = native.snapshot()
            assert before["preparation_upload_calls"] == after["preparation_upload_calls"]
            assert before["argument_bytes"] == after["argument_bytes"]
            with pytest.raises((ValueError, RuntimeError)):
                bindings.update(output=ti.ndarray(ti.f32, host.size))
            assert bindings._version is version
            bindings.update(scale=-5)
            assert bindings._version.execution_frame is not version.execution_frame
            graph.submit(bindings).wait()
            np.testing.assert_array_equal(output.to_numpy(), host * -5 + 7)
            del version
            gc.collect()
            native.snapshot()
            memory = graph.execution_stats().memory
            assert (
                memory.persistent_argument_bytes - memory.persistent_internal_storage_bytes
                == native.snapshot()["argument_bytes"]
                > 0
            )
            assert graph._graph_stats[0]["diagnostics_counters_complete"] is False
            candidate_id = materialized.materialized_physical_id
        assert native.snapshot()["closed"] == 1
        assert native.snapshot()["argument_bytes"] == 0
        with context.materialize(catalog.baseline.recipe) as baseline:
            assert baseline.materialized_physical_id != candidate_id
            assert baseline.manifest.submissions[0].replay_mode == "runtime_managed"


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_binding_recipe_raw_mapping_calls_prepare_without_silent_ordinary_fallback():
    definition = _definition()
    catalog = _catalog(definition)
    recipe = next(entry.recipe for entry in catalog.entries() if entry.recipe.fragments)
    source, output, host = _arrays()
    with definition.materialization_context(provider_set=catalog.provider_set) as context:
        with context.materialize(recipe) as candidate:
            graph = candidate.executor
            native = graph._instance._backend_executable._native
            for scale in (2, -1, 7):
                before = native.snapshot()["preparation_upload_calls"]
                graph.run(dict(source=source, output=output, scale=scale))
                np.testing.assert_array_equal(output.to_numpy(), host * scale + 7)
                assert native.snapshot()["preparation_upload_calls"] > before
            assert native.snapshot()["executables"] == 1
            assert native.snapshot()["frames"] == 1


@pytest.mark.parametrize("sampled_texture", [False, True])
@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_binding_recipe_fork_search_report_and_fresh_definition_resolve(sampled_texture):
    factory = _texture_definition if sampled_texture else _definition
    definition = factory()
    catalog = _catalog(definition)
    providers = (GraphRuntimeAssemblyProvider(), GraphBindingFrameRecipeProvider())
    source, output, host = _arrays()
    if sampled_texture:
        source = ti.ndarray(ti.f32, (17, 23))
        source.fill(3)
        image = ti.Texture(ti.Format.r32f, (17, 23))
        image.from_ndarray(source)
        output = ti.ndarray(ti.f32, (17, 23))
        arguments = dict(image=image, output=output)
        expected = np.full((17, 23), 7, np.float32)
    else:
        arguments = dict(source=source, output=output, scale=11)
        expected = host * 11 + 7
    # Deterministic contract selection, not a performance or acceleration claim.
    session = definition.search_recipes(
        providers=providers,
        target=ti.graph.GraphOptimizationTarget(objectives=(("prepared_binding_candidate", "max"),)),
        budget=ti.graph.GraphSearchBudget(evaluation_limit=4, repeat_count=1),
        strategy=ti.graph.GraphRecipeSearchStrategy(mode="exact_if_bounded"),
    )
    observed = set()

    def evaluate(graph, request):
        bindings = graph.bind(arguments)
        graph.run(bindings)
        np.testing.assert_array_equal(output.to_numpy(), expected)
        selected = bindings._version.execution_frame is not None
        observed.add(request.recipe_id)
        return {"prepared_binding_candidate": float(selected)}

    decision = session.run(evaluate)
    assert decision.status == "selected", decision.report.results
    assert decision.report.search_complete
    assert len(observed) == 2
    assert "raw mapping calls" in json.dumps(decision.report.recipe_annotations)
    fresh = factory()
    assert fresh.semantic_graph_id == definition.semantic_graph_id
    selection = fresh.resolve_recipe(decision.selection_artifact, providers=providers)
    with fresh.materialize(selection) as materialized:
        evaluate(materialized.executor, selection)
        expected_mode = (
            "vulkan_secondary_immutable_argument_frames"
            if definition.backend == "vulkan"
            else "cuda_immutable_argument_frames_exec_reuse"
        )
        assert materialized.executor._instance.physical_submission_mode == expected_mode


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_image_frames_close_layout_cycles_across_upload_and_kernel_use(monkeypatch):
    from taichi_forge.lang import impl

    @ti.kernel
    def write(
        image: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=0),
        source: ti.types.ndarray(dtype=ti.f32, ndim=2),
    ):
        for i, j in source:
            image.store(ti.Vector([i, j]), ti.Vector([source[i, j], 0.0, 0.0, 1.0]))

    @ti.kernel
    def read(image: ti.types.texture(num_dimensions=2), output: ti.types.ndarray(dtype=ti.f32, ndim=2)):
        for i, j in output:
            output[i, j] = image.fetch(ti.Vector([i, j]), 0).x

    image = ti.Texture(ti.Format.r32f, (17, 23))
    source = ti.ndarray(ti.f32, (17, 23))
    output = ti.ndarray(ti.f32, (17, 23))
    host = np.arange(17 * 23, dtype=np.float32).reshape(17, 23) / 16
    source.from_numpy(host)
    output.fill(-53)
    reader = _texture_definition()
    with monkeypatch.context() as patch:
        patch.setattr(core, "_VulkanFixedGraphRecording", object)
        assert GraphBindingFrameRecipeProvider().fragments(reader) == ()
        assert GraphBindingFrameRecipeProvider().fragments(_definition())
    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        write,
        ti.graph.Arg(ti.graph.ArgKind.RWTEXTURE, "write_image", fmt=ti.Format.r32f, ndim=2),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.f32, ndim=2),
        label="write",
    )
    builder.dispatch(
        read,
        ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "read_image", ndim=2),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=2),
        label="read-after-write",
    )
    cycle = builder.freeze()
    plans = []
    for definition in (reader, cycle):
        catalog = _catalog(definition)
        recipe = next(entry.recipe for entry in catalog.entries() if entry.recipe.fragments)
        context = definition.materialization_context(provider_set=catalog.provider_set)
        plans.append((context, context.materialize(recipe)))
    read_graph, cycle_graph = [plan.executor for _, plan in plans]
    read_binding = read_graph.bind(dict(image=image, output=output))
    cycle_binding = cycle_graph.bind(dict(write_image=image, read_image=image, source=source, output=output))
    np.testing.assert_array_equal(output.to_numpy(), np.full((17, 23), -53, np.float32))
    frames = (read_binding._version.execution_frame, cycle_binding._version.execution_frame)
    assert all(frame.uses_secondary_commands() for frame in frames)
    argument_bytes = [frame.argument_bytes() for frame in frames]

    def unexpected(*args, **kwargs):
        raise AssertionError("Immutable replay must not prepare again")

    monkeypatch.setattr(core, "_prepare_vulkan_graph_recording", unexpected)
    for _ in range(3):
        cycle_graph.run(cycle_binding)
        np.testing.assert_array_equal(output.to_numpy(), host)
        read_graph.run(read_binding)
        np.testing.assert_array_equal(output.to_numpy(), host * 2 + 1)
        source.fill(9)
        image.from_ndarray(source)
        read_graph.run(read_binding)
        np.testing.assert_array_equal(output.to_numpy(), np.full((17, 23), 19, np.float32))
        source.from_numpy(host)
        write(image, source)
        read_graph.run(read_binding)
        np.testing.assert_array_equal(output.to_numpy(), host * 2 + 1)
    assert [frame.argument_bytes() for frame in frames] == argument_bytes
    version = cycle_binding._version
    wrong_format = ti.Texture(ti.Format.rgba32f, (17, 23))
    with pytest.raises((ValueError, RuntimeError, ti.TaichiRuntimeError)):
        cycle_binding.update(write_image=wrong_format)
    assert cycle_binding._version is version
    wrong_format._delete_runtime_texture()
    program = impl.get_runtime().prog
    before = dict(program._debug_texture_resource_stats())
    native_image = image.tex
    image._delete_runtime_texture()
    cycle_graph.run(cycle_binding)
    # Closing before submission must retain images until the parent commands
    # finish, not merely until the Python recording wrapper is closed.
    for context, plan in plans:
        plan.close()
        context.close()
    np.testing.assert_array_equal(output.to_numpy(), host)
    ti.sync()
    after = dict(program._debug_texture_resource_stats())
    assert after["released_total"] >= before["released_total"] + 1
    assert after["retiring"] == after["inflight"] == after["release_errors"] == 0
    assert all(frame.argument_bytes() == 0 for frame in frames)
    assert native_image


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_sampled_frame_consumes_graphics_output_and_retires_on_reset():
    from tests.python.test_hardware_graphics import _triangle_pipeline, _triangle_vertices

    @ti.kernel
    def read(image: ti.types.texture(num_dimensions=2), output: ti.types.ndarray(dtype=ti.f32, ndim=3)):
        for i, j in ti.ndrange(64, 64):
            value = image.fetch(ti.Vector([i, j]), 0)
            for c in ti.static(range(4)):
                output[i, j, c] = value[c]

    image = ti.Texture(ti.Format.rgba8, (64, 64))
    output = ti.ndarray(ti.f32, (64, 64, 4))
    reference = ti.ndarray(ti.f32, (64, 64, 4))
    vertices = _triangle_vertices()
    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        read,
        ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "image", ndim=2),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=3),
        label="postprocess",
    )
    definition = builder.freeze()
    catalog = _catalog(definition)
    recipe = next(entry.recipe for entry in catalog.entries() if entry.recipe.fragments)
    with definition.materialization_context(provider_set=catalog.provider_set) as context:
        with context.materialize(recipe) as materialized, _triangle_pipeline() as pipeline:
            graph = materialized.executor
            binding = graph.bind(dict(image=image, output=output))
            frame = binding._version.execution_frame
            for color in ((0.1, 0.2, 0.3, 1.0), (0.6, 0.4, 0.2, 1.0)):
                pipeline.draw(image, {0: vertices}, draw=ti.hardware.graphics.Draw(3), clear_color=color)
                graph.run(binding)
                # The direct consumer is an oracle, not a second recording.
                read(image, reference)
                np.testing.assert_array_equal(output.to_numpy(), reference.to_numpy())
                np.testing.assert_allclose(output.to_numpy()[0, 0], color, atol=1 / 255)
            pipeline.close()
            ti.reset()
            with pytest.raises(RuntimeError, match="closed|finaliz|runtime"):
                frame.run()
            frame.close()
            assert frame.argument_bytes() == 0


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_binding_recipe_scope_and_missing_capability_do_not_change_ordinary_execution(monkeypatch):
    definition = _definition()
    catalog = _catalog(definition)
    recipe = next(entry.recipe for entry in catalog.entries() if entry.recipe.fragments)
    with definition.materialization_context(provider_set=catalog.provider_set, workspace_lanes=2) as context:
        with pytest.raises((ValueError, RuntimeError), match="one workspace lane"):
            context.materialize(recipe)
    source, output, host = _arrays()
    texture_definition = _texture_definition()
    native_type = core._CudaGraphBindingExecutor

    class WithoutTextureCapability:
        available = native_type.available
        retains_completion_events_until_close = native_type.retains_completion_events_until_close
        supports_capture_commands = native_type.supports_capture_commands

    with monkeypatch.context() as patch:
        patch.setattr(core, "_CudaGraphBindingExecutor", WithoutTextureCapability)
        provider = GraphBindingFrameRecipeProvider()
        assert provider.fragments(definition)
        assert provider.fragments(texture_definition) == ()

    class MissingNativeCapability:
        @staticmethod
        def available():
            return False

    monkeypatch.setattr(core, "_CudaGraphBindingExecutor", MissingNativeCapability)
    provider = GraphBindingFrameRecipeProvider()
    assert provider.fragments(definition) == ()
    ordinary = definition.compile()
    try:
        ordinary.run(dict(source=source, output=output, scale=2))
        np.testing.assert_array_equal(output.to_numpy(), host * 2 + 7)
    finally:
        ordinary._invalidate_runtime()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_binding_recipe_manifest_failure_retires_unpublished_executor(monkeypatch):
    from taichi_forge.graph._recipes import families

    definition = _definition()
    catalog = _catalog(definition)
    recipe = next(entry.recipe for entry in catalog.entries() if entry.recipe.fragments)
    source, output, host = _arrays()
    unpublished = []

    def fail_manifest(definition, recipe, graph):
        # Keep a strong reference: rollback must close resources immediately,
        # not rely on destruction when the failed materialization loses scope.
        unpublished.append(graph._instance._backend_executable._native)
        graph.bind(dict(source=source, output=output, scale=3))
        assert unpublished[-1].snapshot()["argument_bytes"] > 0
        raise ValueError("injected physical observation failure")

    with definition.materialization_context(provider_set=catalog.provider_set) as context:
        with monkeypatch.context() as patch:
            patch.setattr(families, "observe_graph_physical_manifest", fail_manifest)
            with pytest.raises((ValueError, RuntimeError), match="injected physical observation failure"):
                context.materialize(recipe)
        assert unpublished[0].snapshot()["closed"] == 1
        assert unpublished[0].snapshot()["argument_bytes"] == 0
        with context.materialize(recipe) as recovered:
            bindings = recovered.executor.bind(dict(source=source, output=output, scale=3))
            recovered.executor.run(bindings)
            np.testing.assert_array_equal(output.to_numpy(), host * 3 + 7)
