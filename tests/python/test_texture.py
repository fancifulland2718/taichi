import gc
from io import BytesIO

import numpy as np
import pytest
import requests
from PIL import Image
from taichi_forge._lib import core as ti_core
from taichi_forge.lang import impl

import taichi_forge as ti
from tests import test_utils
from tests.python.hardware_provider_lifecycle_qualification import (
    stress_iterations,
)
from tests.python.hardware_process_memory import ProcessMemoryPlateau

supported_archs_texture = [ti.vulkan]
supported_archs_texture_excluding_load_store = [ti.vulkan, ti.opengl]

integer_storage_image_cases = [
    (ti.Format.r16u, ti.u32, 1, (1, 0, 0, 0)),
    (ti.Format.rg16u, ti.u32, 2, (1, 65535, 0, 0)),
    (ti.Format.rgba16u, ti.u32, 4, (1, 65535, 32768, 7)),
    (ti.Format.r16i, ti.i32, 1, (-32768, 0, 0, 0)),
    (ti.Format.rg16i, ti.i32, 2, (-32768, -1, 0, 0)),
    (ti.Format.rgba16i, ti.i32, 4, (-32768, -1, 1234, 32767)),
    (ti.Format.r32u, ti.u32, 1, (1, 0, 0, 0)),
    (ti.Format.rg32u, ti.u32, 2, (1, 0xFFFFFFFF, 0, 0)),
    (ti.Format.rgba32u, ti.u32, 4, (1, 0xFFFFFFFF, 0x80000000, 7)),
    (ti.Format.r32i, ti.i32, 1, (-0x80000000, 0, 0, 0)),
    (ti.Format.rg32i, ti.i32, 2, (-0x80000000, -1, 0, 0)),
    (ti.Format.rgba32i, ti.i32, 4, (-0x80000000, -1, 1234, 0x7FFFFFFF)),
]


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_texture_hardware_sampling_and_lifecycle():
    runtime = impl.get_runtime()
    source = ti.ndarray(dtype=ti.f32, shape=(2, 2))
    source.from_numpy(np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
    texture = ti.Texture(ti.Format.r32f, (2, 2))
    texture.from_ndarray(source)
    result = ti.ndarray(dtype=ti.f32, shape=5)

    @ti.kernel
    def sample(
        image: ti.types.texture(num_dimensions=2),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        output[0] = image.fetch(ti.Vector([0, 0]), 0).x
        output[1] = image.fetch(ti.Vector([1, 0]), 0).x
        output[2] = image.fetch(ti.Vector([0, 1]), 0).x
        output[3] = image.fetch(ti.Vector([1, 1]), 0).x
        output[4] = image.sample_lod(ti.Vector([0.5, 0.5]), 0.0).x

    sample(texture, result)
    identity = dict(runtime.prog._debug_texture_resource_identity(texture.tex))
    texture._delete_runtime_texture()
    ti.sync()
    np.testing.assert_allclose(
        result.to_numpy(),
        np.asarray([0.0, 2.0, 1.0, 3.0, 1.5], dtype=np.float32),
    )
    stats = dict(runtime.prog._debug_texture_resource_stats())
    assert identity["generation"] > 0
    assert stats["live"] == 0
    assert stats["inflight"] == 0
    assert stats["release_errors"] == 0

    resolved = next(
        operation
        for operation in ti.hardware.report().operations
        if operation.descriptor.operation_id == "sampling.texture.cuda"
    )
    assert resolved.selection == "not_considered"
    assert resolved.unavailable_reason == "operation_requirements_not_evaluated"
    assert resolved.descriptor.hardware_acceleration == "qualified"
    assert resolved.descriptor.graph_integration == "inline"


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_texture_rejects_incompatible_resource_contracts():
    with pytest.raises(RuntimeError, match="vec4-f32 sampling contract"):
        ti.Texture(ti.Format.r32u, (4, 4))

    sampler = ti.hardware.sampling.SamplerConfig(
        min_filter="nearest", mag_filter="linear"
    )
    with pytest.raises(RuntimeError, match="matching min and mag filters"):
        ti.Texture(ti.Format.r32f, (4, 4), sampler=sampler)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_texture_samples_1d_normalized_and_3d_float_resources():
    source_1d = ti.ndarray(dtype=ti.u8, shape=4)
    source_1d.from_numpy(np.asarray([0, 64, 128, 255], dtype=np.uint8))
    texture_1d = ti.Texture(ti.Format.r8, (4,))
    texture_1d.from_ndarray(source_1d)

    source_3d = ti.ndarray(dtype=ti.f32, shape=(2, 2, 2))
    source_3d.from_numpy(np.arange(8, dtype=np.float32).reshape((2, 2, 2)))
    texture_3d = ti.Texture(ti.Format.r32f, (2, 2, 2))
    texture_3d.from_ndarray(source_3d)
    result = ti.ndarray(dtype=ti.f32, shape=4)

    @ti.kernel
    def sample_dimensions(
        image_1d: ti.types.texture(num_dimensions=1),
        image_3d: ti.types.texture(num_dimensions=3),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        output[0] = image_1d.fetch(ti.Vector([2]), 0).x
        output[1] = image_1d.sample_lod(ti.Vector([0.875]), 0.0).x
        output[2] = image_3d.fetch(ti.Vector([1, 1, 1]), 0).x
        output[3] = image_3d.sample_lod(
            ti.Vector([0.5, 0.5, 0.5]), 0.0
        ).x

    sample_dimensions(texture_1d, texture_3d, result)
    ti.sync()
    np.testing.assert_allclose(
        result.to_numpy(),
        np.asarray([128.0 / 255.0, 1.0, 7.0, 3.5], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_texture_preserves_asymmetric_logical_axes():
    source_2d_host = np.arange(15, dtype=np.float32).reshape((3, 5))
    source_2d = ti.ndarray(dtype=ti.f32, shape=(3, 5))
    source_2d.from_numpy(source_2d_host)
    texture_2d = ti.Texture(ti.Format.r32f, (3, 5))
    texture_2d.from_ndarray(source_2d)

    source_3d_host = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
    source_3d = ti.ndarray(dtype=ti.f32, shape=(2, 3, 4))
    source_3d.from_numpy(source_3d_host)
    texture_3d = ti.Texture(ti.Format.r32f, (2, 3, 4))
    texture_3d.from_ndarray(source_3d)
    result = ti.ndarray(dtype=ti.f32, shape=4)

    @ti.kernel
    def fetch_asymmetric(
        image_2d: ti.types.texture(num_dimensions=2),
        image_3d: ti.types.texture(num_dimensions=3),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        output[0] = image_2d.fetch(ti.Vector([0, 4]), 0).x
        output[1] = image_2d.fetch(ti.Vector([2, 1]), 0).x
        output[2] = image_3d.fetch(ti.Vector([0, 2, 3]), 0).x
        output[3] = image_3d.fetch(ti.Vector([1, 0, 2]), 0).x

    fetch_asymmetric(texture_2d, texture_3d, result)
    ti.sync()
    np.testing.assert_array_equal(
        result.to_numpy(),
        np.asarray(
            [
                source_2d_host[0, 4],
                source_2d_host[2, 1],
                source_3d_host[0, 2, 3],
                source_3d_host[1, 0, 2],
            ],
            dtype=np.float32,
        ),
    )


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_texture_uploads_tightly_packed_dense_field():
    source = ti.field(dtype=ti.f32, shape=(3, 5))
    texture = ti.Texture(ti.Format.r32f, (3, 5))
    vector_source = ti.Vector.field(2, dtype=ti.f32, shape=(3, 5))
    vector_texture = ti.Texture(ti.Format.rg32f, (3, 5))
    result = ti.ndarray(dtype=ti.f32, shape=5)

    @ti.kernel
    def initialize():
        for i, j in source:
            source[i, j] = ti.cast(i * 10 + j, ti.f32)
            vector_source[i, j] = ti.Vector(
                [ti.cast(i * 10 + j, ti.f32), ti.cast(100 + i + j, ti.f32)]
            )

    @ti.kernel
    def fetch_field(
        image: ti.types.texture(num_dimensions=2),
        vector_image: ti.types.texture(num_dimensions=2),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        output[0] = image.fetch(ti.Vector([0, 4]), 0).x
        output[1] = image.fetch(ti.Vector([1, 3]), 0).x
        output[2] = image.fetch(ti.Vector([2, 0]), 0).x
        vector_value = vector_image.fetch(ti.Vector([2, 4]), 0)
        output[3] = vector_value.x
        output[4] = vector_value.y

    initialize()
    texture.from_field(source)
    vector_texture.from_field(vector_source)
    fetch_field(texture, vector_texture, result)
    ti.sync()
    np.testing.assert_array_equal(
        result.to_numpy(),
        np.asarray([4.0, 13.0, 20.0, 24.0, 106.0], dtype=np.float32),
    )


@pytest.mark.parametrize("fixed", [False, True])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_texture_graph_replay_retains_resources_and_rebinds(fixed):
    source = ti.ndarray(dtype=ti.f32, shape=(2, 2))
    source.from_numpy(np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    textures = [ti.Texture(ti.Format.r32f, (2, 2)) for _ in range(3)]
    for index, texture in enumerate(textures):
        source.fill(index + 1)
        texture.from_ndarray(source)
    output = ti.ndarray(dtype=ti.f32, shape=2)
    output.fill(0)

    @ti.kernel
    def fetch(
        image: ti.types.texture(num_dimensions=2),
        result: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        result[0] = image.fetch(ti.Vector([1, 1]), 0).x

    @ti.kernel
    def consume(result: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        result[1] += result[0]

    image_arg = ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "image", ndim=2)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(fetch, image_arg, output_arg)
    builder.dispatch(consume, output_arg)
    graph = builder.compile()
    # Graph compilation owns a dummy texture for kernel specialization.
    cold_stats = dict(impl.get_runtime().prog._debug_texture_resource_stats())
    bindings = graph.bind({"image": textures[0], "output": output}) if fixed else None
    np.testing.assert_array_equal(output.to_numpy(), [0, 0])

    total = 0
    # Exercise exact replay, both bounded cache slots, and eviction. Resource
    # identity changes must not patch a sampled object behind an old lease.
    for index in (0, 1, 0, 1, 2, 0):
        if fixed:
            bindings.update(image=textures[index])
        for _ in range(2):
            graph.run(
                bindings if fixed else {"image": textures[index], "output": output}
            )
            total += index + 1
        np.testing.assert_array_equal(output.to_numpy(), [index + 1, total])
        assert graph.execution_stats().segments[0].last_path == "cuda_exact_replay"

    # Content upload preserves the sampled object; the captured query sees it.
    source.fill(9)
    textures[0].from_ndarray(source)
    graph.run(bindings if fixed else {"image": textures[0], "output": output})
    total += 9
    np.testing.assert_array_equal(output.to_numpy(), [9, total])
    assert graph.execution_stats().segments[0].last_path == "cuda_exact_replay"

    if fixed:
        revision = bindings.revision
        with pytest.raises((ValueError, RuntimeError, ti.TaichiRuntimeError)):
            bindings.update(image=output)
        wrong_dimension = ti.Texture(ti.Format.r32f, (2,))
        with pytest.raises(ti.TaichiRuntimeError, match="dimensionality"):
            bindings.update(image=wrong_dimension)
        wrong_dimension._delete_runtime_texture()
        assert bindings.revision == revision
        graph.run(bindings)
        total += 9
        np.testing.assert_array_equal(output.to_numpy(), [9, total])

    # Keep native wrappers alive to distinguish registry retirement from Python
    # garbage collection. Closing a capture must drop its persistent leases.
    native_textures = [texture.tex for texture in textures]
    for texture in textures:
        texture._delete_runtime_texture()
    if fixed:
        with pytest.raises(
            (RuntimeError, ti.TaichiRuntimeError), match="Texture|texture|retired|stale"
        ):
            graph.run(bindings)
    graph.close()
    ti.sync()
    stats = dict(impl.get_runtime().prog._debug_texture_resource_stats())
    assert stats["live"] <= cold_stats["live"] - len(textures)
    assert stats["released_total"] >= cold_stats["released_total"] + len(textures)
    assert stats["retiring"] == stats["inflight"] == stats["release_errors"] == 0
    assert native_textures


@pytest.mark.parametrize("force_masked", [False, True])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_texture_control_capture_keeps_sampler_lifetime(monkeypatch, force_masked):
    capabilities = dict(ti_core.cuda_conditional_graph_capabilities())
    required = (
        "internal_masked_graph_available"
        if force_masked
        else "general_graph_exact_control_available"
    )
    if not capabilities.get(required, False):
        pytest.skip(f"CUDA control capture unavailable: {required}")
    if force_masked:
        monkeypatch.setenv("TI_GRAPH_CUDA_FORCE_MASKED_CONTROL", "1")

    @ti.kernel
    def condition(value: ti.i32, predicate: ti.types.ndarray(dtype=ti.i32, ndim=0)):
        predicate[None] = value

    @ti.kernel
    def body(
        image: ti.types.texture(num_dimensions=2),
        result: ti.types.ndarray(dtype=ti.f32, ndim=0),
    ):
        result[None] += image.sample_lod(ti.Vector([0.5, 0.5]), 0.0).x

    builder = ti.graph.GraphBuilder()
    value = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "value", ti.i32)
    predicate = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "predicate", ti.i32, ndim=0)
    image = ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "image", ndim=2)
    result = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "result", ti.f32, ndim=0)
    condition_region = builder.create_sequential().dispatch(condition, value, predicate)
    body_region = builder.create_sequential().dispatch(body, image, result)
    builder.if_then_else(
        condition_region,
        body_region,
        predicate=predicate,
        control_inputs=(value,),
        lowering_mode="native_required",
    )
    graph = builder.compile()
    source = ti.ndarray(ti.f32, shape=(2, 2))
    source.fill(2)
    texture = ti.Texture(ti.Format.r32f, (2, 2))
    texture.from_ndarray(source)
    output = ti.ndarray(ti.f32, shape=())
    output.fill(0)
    control = ti.ndarray(ti.i32, shape=())
    bindings = graph.bind(
        {"value": 1, "predicate": control, "image": texture, "result": output}
    )
    for enabled in (1, 0, 1):
        bindings.update(value=enabled)
        graph.run(bindings)
    assert output.to_numpy()[()] == 4
    expected = "cuda_masked_bounded_graph" if force_masked else "cuda_conditional_graph"
    assert graph.control_flow_stats()[0].lowering == expected
    revision = bindings.revision
    with pytest.raises(ti.TaichiRuntimeError, match="Texture"):
        bindings.update(image=source)
    assert bindings.revision == revision
    # Reset retires captured samplers before the CUDA context, even with live
    # Python Graph/binding wrappers retained here.
    ti.reset()
    assert graph.execution_stats().lifecycle_state == "runtime_invalid"


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_texture_hardware_sampling_qualification():
    tex = ti.Texture(ti.Format.r32f, (2, 2))
    out = ti.ndarray(dtype=ti.f32, shape=5)

    @ti.kernel
    def write(
        texture: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=0),
    ):
        texture.store(ti.Vector([0, 0]), ti.Vector([0.0, 0.0, 0.0, 0.0]))
        texture.store(ti.Vector([1, 0]), ti.Vector([1.0, 0.0, 0.0, 0.0]))
        texture.store(ti.Vector([0, 1]), ti.Vector([2.0, 0.0, 0.0, 0.0]))
        texture.store(ti.Vector([1, 1]), ti.Vector([3.0, 0.0, 0.0, 0.0]))

    @ti.kernel
    def sample(
        texture: ti.types.texture(num_dimensions=2),
        result: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        result[0] = texture.fetch(ti.Vector([0, 0]), 0).x
        result[1] = texture.fetch(ti.Vector([1, 0]), 0).x
        result[2] = texture.fetch(ti.Vector([0, 1]), 0).x
        result[3] = texture.fetch(ti.Vector([1, 1]), 0).x
        result[4] = texture.sample_lod(ti.Vector([0.5, 0.5]), 0.0).x

    write(tex)
    sample(tex, out)
    np.testing.assert_allclose(
        out.to_numpy(), np.asarray([0.0, 1.0, 2.0, 3.0, 1.5], dtype=np.float32)
    )

    capability = ti.hardware.capability("sampling.texture.vulkan")
    assert capability.hardware_acceleration == "qualified"
    assert capability.execution_kind == "kernel_intrinsic"
    assert capability.graph_integration == "inline"


@test_utils.test(arch=ti.vulkan, offline_cache=False, advanced_optimization=False)
def test_rgba32f_store_is_a_side_effect_without_a_temporary_result():
    image = ti.Texture(ti.Format.rgba32f, (3, 2))
    source = ti.ndarray(ti.f32, (2, 3, 4))
    output = ti.ndarray(ti.f32, (2, 3, 4))

    @ti.kernel
    def constant(texture: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.rgba32f, lod=0)):
        texture.store(ti.Vector([0, 0]), ti.Vector([0.25, 0.5, 0.75, 1.0]))

    @ti.kernel
    def upload(
        texture: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.rgba32f, lod=0),
        values: ti.types.ndarray(dtype=ti.f32, ndim=3),
    ):
        for y, x in ti.ndrange(2, 3):
            texture.store(ti.Vector([x, y]), ti.Vector([values[y, x, c] for c in ti.static(range(4))]))

    @ti.kernel
    def fetch(
        texture: ti.types.texture(num_dimensions=2),
        values: ti.types.ndarray(dtype=ti.f32, ndim=3),
    ):
        for y, x in ti.ndrange(2, 3):
            value = texture.fetch(ti.Vector([x, y]), 0)
            for c in ti.static(range(4)):
                values[y, x, c] = value[c]

    constant(image)
    fetch(image, output)
    np.testing.assert_array_equal(output.to_numpy()[0, 0], [0.25, 0.5, 0.75, 1.0])
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 32
    for expected in (values, 1 - values):
        source.from_numpy(expected)
        upload(image, source)
        fetch(image, output)
        np.testing.assert_array_equal(output.to_numpy(), expected)


@ti.func
def taichi_logo(pos: ti.template(), scale: float = 1 / 1.11):
    p = (pos - 0.5) / scale + 0.5
    ret = -1
    if not (p - 0.50).norm_sqr() <= 0.52**2:
        if ret == -1:
            ret = 0
    if not (p - 0.50).norm_sqr() <= 0.495**2:
        if ret == -1:
            ret = 1
    if (p - ti.Vector([0.50, 0.25])).norm_sqr() <= 0.08**2:
        if ret == -1:
            ret = 1
    if (p - ti.Vector([0.50, 0.75])).norm_sqr() <= 0.08**2:
        if ret == -1:
            ret = 0
    if (p - ti.Vector([0.50, 0.25])).norm_sqr() <= 0.25**2:
        if ret == -1:
            ret = 0
    if (p - ti.Vector([0.50, 0.75])).norm_sqr() <= 0.25**2:
        if ret == -1:
            ret = 1
    if p[0] < 0.5:
        if ret == -1:
            ret = 1
    else:
        if ret == -1:
            ret = 0
    return 1 - ret


@ti.kernel
def make_texture_2d_r32f(tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=0), n: ti.i32):
    for i, j in ti.ndrange(n, n):
        ret = ti.cast(taichi_logo(ti.Vector([i, j]) / n), ti.f32)
        tex.store(ti.Vector([i, j]), ti.Vector([ret, 0.0, 0.0, 0.0]))


@ti.kernel
def make_texture_2d_rgba8(tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.rgba8, lod=0), n: ti.i32):
    for i, j in ti.ndrange(n, n):
        ret = ti.cast(taichi_logo(ti.Vector([i, j]) / n), ti.f32)
        tex.store(ti.Vector([i, j]), ti.Vector([ret, 0.0, 0.0, 0.0]))


@ti.kernel
def make_texture_3d(tex: ti.types.rw_texture(num_dimensions=3, fmt=ti.Format.r32f, lod=0), n: ti.i32):
    for i, j, k in ti.ndrange(n, n, n):
        div = ti.cast(i / n, ti.f32)
        if div > 0.5:
            tex.store(ti.Vector([i, j, k]), ti.Vector([1.0, 0.0, 0.0, 0.0]))
        else:
            tex.store(ti.Vector([i, j, k]), ti.Vector([0.5, 0.0, 0.0, 0.0]))


@test_utils.test(arch=supported_archs_texture)
def test_texture_compiled_functions():
    res = (512, 512)
    pixels = ti.Vector.field(3, dtype=float, shape=res)

    @ti.kernel
    def paint(t: ti.f32, tex: ti.types.texture(num_dimensions=2), n: ti.i32):
        for i, j in pixels:
            uv = ti.Vector([i / res[0], j / res[1]])
            warp_uv = uv + ti.Vector([ti.cos(t + uv.x * 5.0), ti.sin(t + uv.y * 5.0)]) * 0.1
            c = ti.math.vec4(0.0)
            if uv.x > 0.5:
                c = tex.sample_lod(warp_uv, 0.0)
            else:
                c = tex.fetch(ti.cast(warp_uv * n, ti.i32), 0)
            pixels[i, j] = [c.r, c.r, c.r]

    n1 = 128
    texture1 = ti.Texture(ti.Format.r32f, (n1, n1))
    n2 = 256
    texture2 = ti.Texture(ti.Format.r32f, (n2, n2))
    texture3 = ti.Texture(ti.Format.rgba8, (n1, n1))

    make_texture_2d_r32f(texture1, n1)
    assert impl.get_runtime().get_num_compiled_functions() == 1

    make_texture_2d_r32f(texture2, n2)
    assert impl.get_runtime().get_num_compiled_functions() == 1

    make_texture_2d_rgba8(texture3, n1)
    assert impl.get_runtime().get_num_compiled_functions() == 2

    paint(0.1, texture1, n1)
    assert impl.get_runtime().get_num_compiled_functions() == 3

    paint(0.2, texture2, n2)
    assert impl.get_runtime().get_num_compiled_functions() == 3

    # (penguinliong) Remember that non-RW textures don't enforce a format so
    # it's the same as the first call to `paint`.
    paint(0.3, texture3, n1)
    assert impl.get_runtime().get_num_compiled_functions() == 3


@test_utils.test(arch=supported_archs_texture_excluding_load_store)
def test_texture_from_field():
    res = (128, 128)
    f = ti.Vector.field(2, ti.f32, res)
    tex = ti.Texture(ti.Format.r32f, res)

    @ti.kernel
    def init_taichi_logo_field():
        for i, j in f:
            f[i, j] = [taichi_logo(ti.Vector([i / res[0], j / res[1]])), 0]

    init_taichi_logo_field()
    tex.from_field(f)


@test_utils.test(arch=supported_archs_texture_excluding_load_store)
def test_texture_from_ndarray():
    res = (128, 128)
    f = ti.Vector.ndarray(2, ti.f32, res)
    tex = ti.Texture(ti.Format.r32f, res)

    @ti.kernel
    def init_taichi_logo_ndarray(f: ti.types.ndarray(ndim=2)):
        for i, j in f:
            f[i, j] = [taichi_logo(ti.Vector([i / res[0], j / res[1]])), 0]

    init_taichi_logo_ndarray(f)
    tex.from_ndarray(f)


@test_utils.test(arch=supported_archs_texture)
def test_texture_3d():
    res = (32, 32, 32)
    tex = ti.Texture(ti.Format.r32f, res)

    make_texture_3d(tex, res[0])


@test_utils.test(arch=supported_archs_texture)
def test_from_to_image():
    url = "https://github.com/taichi-dev/taichi/blob/master/misc/logo.png?raw=true"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    tex = ti.Texture(ti.Format.rgba8, img.size)

    tex.from_image(img)
    out = tex.to_image()

    assert (np.asarray(out) == np.asarray(img.convert("RGB"))).all()


@test_utils.test(arch=supported_archs_texture)
def test_rw_texture_2d_struct_for():
    res = (128, 128)
    tex = ti.Texture(ti.Format.r32f, res)
    arr = ti.ndarray(ti.f32, res)

    @ti.kernel
    def write(tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=0)):
        for i, j in tex:
            tex.store(ti.Vector([i, j]), ti.Vector([1.0, 0.0, 0.0, 0.0]))

    @ti.kernel
    def read(tex: ti.types.texture(num_dimensions=2), arr: ti.types.ndarray()):
        for i, j in arr:
            arr[i, j] = tex.fetch(ti.Vector([i, j]), 0).x

    write(tex)
    read(tex, arr)
    assert arr.to_numpy().sum() == 128 * 128


@pytest.mark.parametrize(
    "fmt,dtype,num_channels,values", integer_storage_image_cases
)
@test_utils.test(arch=supported_archs_texture)
def test_rw_texture_integer_sampled_types(fmt, dtype, num_channels, values):
    tex = ti.Texture(fmt, (1, 1))
    out = ti.ndarray(dtype=dtype, shape=4)

    @ti.kernel
    def write(
        tex: ti.types.rw_texture(num_dimensions=2, fmt=fmt, lod=0),
        v0: dtype,
        v1: dtype,
        v2: dtype,
        v3: dtype,
    ):
        tex.store(
            ti.Vector([0, 0]),
            ti.Vector([v0, v1, v2, v3]),
        )

    @ti.kernel
    def read(
        tex: ti.types.rw_texture(num_dimensions=2, fmt=fmt, lod=0),
        out: ti.types.ndarray(dtype=dtype, ndim=1),
    ):
        value = tex.load(ti.Vector([0, 0]))
        for i in ti.static(range(4)):
            out[i] = value[i]

    write(tex, *values)
    read(tex, out)
    np.testing.assert_array_equal(
        out.to_numpy()[:num_channels],
        np.asarray(
            values[:num_channels],
            dtype=np.uint32 if dtype == ti.u32 else np.int32,
        ),
    )


@test_utils.test(arch=supported_archs_texture)
def test_rw_texture_2d_struct_for_dim_check():
    tex = ti.Texture(ti.Format.r32f, (32, 32, 32))

    @ti.kernel
    def write(tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=0)):
        for i, j in tex:
            tex.store(ti.Vector([i, j]), ti.Vector([1.0, 0.0, 0.0, 0.0]))

    with pytest.raises(
        ti.TaichiRuntimeError,
        match="RWTextureType dimension mismatch for argument tex: expected 2, got 3",
    ) as e:
        write(tex)


@test_utils.test(arch=supported_archs_texture)
def test_rw_texture_wrong_fmt():
    tex = ti.Texture(ti.Format.rgba8, (32, 32))

    @ti.kernel
    def write(tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=0)):
        for i, j in tex:
            tex.store(ti.Vector([i, j]), ti.Vector([1.0, 0.0, 0.0, 0.0]))

    with pytest.raises(
        ti.TaichiRuntimeError,
        match="RWTextureType format mismatch for argument tex: expected Format.r32f, got Format.rgba8",
    ) as e:
        write(tex)


@test_utils.test(arch=supported_archs_texture)
def test_rw_texture_wrong_ndim():
    tex = ti.Texture(ti.Format.rgba8, (32, 32))

    @ti.kernel
    def write(tex: ti.types.rw_texture(num_dimensions=1, fmt=ti.Format.rgba8, lod=0)):
        for i, j in tex:
            tex.store(ti.Vector([i, j]), ti.Vector([1.0, 0.0, 0.0, 0.0]))

    with pytest.raises(
        ti.TaichiRuntimeError,
        match="RWTextureType dimension mismatch for argument tex: expected 1, got 2",
    ) as e:
        write(tex)


@test_utils.test(arch=supported_archs_texture)
def test_texture_registry_keeps_inflight_launch_alive_until_sync():
    @ti.kernel
    def write(tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=0)):
        tex.store(ti.Vector([0, 0]), ti.Vector([7.0, 0.0, 0.0, 0.0]))

    warmup = ti.Texture(ti.Format.r32f, (1, 1))
    write(warmup)
    ti.sync()
    del warmup
    gc.collect()
    ti.sync()

    prog = impl.get_runtime().prog
    baseline = prog._debug_texture_resource_stats()
    tex = ti.Texture(ti.Format.r32f, (1, 1))
    created = prog._debug_texture_resource_stats()
    assert created["created_total"] == baseline["created_total"] + 1
    assert created["live"] == baseline["live"] + 1
    assert created["views"] == baseline["views"] + 1
    assert created["leases"] == baseline["leases"] + 1

    write(tex)
    launched = prog._debug_texture_resource_stats()
    assert launched["inflight"] == baseline["inflight"] + 1
    assert launched["leases"] == baseline["leases"] + 2

    del tex
    gc.collect()
    retired = prog._debug_texture_resource_stats()
    assert retired["views"] == baseline["views"]
    assert retired["retiring"] == baseline["retiring"] + 1
    assert retired["retired_total"] == baseline["retired_total"] + 1

    ti.sync()
    completed = prog._debug_texture_resource_stats()
    for key in ("live", "retiring", "leases", "views", "inflight"):
        assert completed[key] == baseline[key]
    assert completed["created_total"] == baseline["created_total"] + 1
    assert completed["retired_total"] == baseline["retired_total"] + 1
    assert completed["released_total"] == baseline["released_total"] + 1


@test_utils.test(arch=supported_archs_texture)
def test_texture_launch_context_rejects_stale_resource_generation():
    @ti.kernel
    def write(tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=0)):
        tex.store(ti.Vector([0, 0]), ti.Vector([3.0, 0.0, 0.0, 0.0]))

    old = ti.Texture(ti.Format.r32f, (1, 1))
    primal = write._primal
    key = primal.ensure_compiled(old)
    kernel_cpp = primal.compiled_kernels[key]
    prog = impl.get_runtime().prog
    compiled = prog.compile_kernel(
        prog.config(), prog.get_device_caps(), kernel_cpp
    )
    old_identity = prog._debug_texture_resource_identity(old.tex)

    old_native = old.tex
    old._invalidate_runtime()
    prog.delete_texture(old_native)

    replacement = ti.Texture(ti.Format.r32f, (1, 1))
    replacement_identity = prog._debug_texture_resource_identity(
        replacement.tex
    )
    assert replacement_identity["domain"] == old_identity["domain"]
    assert replacement_identity["index"] == old_identity["index"]
    assert replacement_identity["generation"] != old_identity["generation"]

    launch_ctx = kernel_cpp.make_launch_context()
    launch_ctx.set_arg_rw_texture([0], replacement.tex)
    launch_ctx._debug_set_texture_resource_handle(
        [0],
        old_identity["domain"],
        old_identity["kind"],
        old_identity["index"],
        old_identity["generation"],
    )
    with pytest.raises(RuntimeError, match="stale or retired Texture"):
        prog.launch_kernel(compiled, launch_ctx)


@pytest.mark.run_in_serial
@test_utils.test(arch=[ti.vulkan, ti.cuda])
def test_texture_registry_resize_churn_conserves_resources():
    prog = impl.get_runtime().prog
    runtime = impl.get_runtime()
    baseline = prog._debug_texture_resource_stats()
    baseline_runtime_objects = len(runtime._runtime_object_refs)

    iterations = stress_iterations(256)
    backend = "cuda" if impl.current_cfg().arch == ti.cuda else "vulkan"
    # Vulkan image recordings retain these same texture registry generations.
    memory_providers = ("vulkan-texture", "vulkan-image") if backend == "vulkan" else ("cuda-texture",)
    process_memory = ProcessMemoryPlateau(
        f"{backend}-texture-resource-churn",
        memory_providers,
        enabled=True,
    )
    process_memory.capture("before")
    for i in range(iterations):
        extent = 1 << (i % 5)
        tex = ti.Texture(ti.Format.rgba8, (extent, extent))
        del tex
        if i % 32 == 0:
            gc.collect()
        if i + 1 == max(1, iterations // 2):
            gc.collect()
            process_memory.capture("midpoint")
    gc.collect()
    process_memory.capture("after")
    process_memory.finish(iterations)

    completed = prog._debug_texture_resource_stats()
    for key in ("live", "retiring", "leases", "views", "inflight"):
        assert completed[key] == baseline[key]
    assert completed["created_total"] - baseline["created_total"] == iterations
    assert completed["retired_total"] - baseline["retired_total"] == iterations
    assert completed["released_total"] - baseline["released_total"] == iterations
    assert len(runtime._runtime_object_refs) == baseline_runtime_objects
