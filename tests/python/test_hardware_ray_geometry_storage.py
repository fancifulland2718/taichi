import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=ti.vulkan, offline_cache=False)
@pytest.mark.parametrize("mode", ["scene_refit", "blas_refit", "blas_build"])
def test_ray_geometry_field_ranges_compose_with_producer_and_query(mode, monkeypatch):
    if not ti.hardware.ray.is_available():
        pytest.skip("Vulkan ray query is unavailable")
    padding, vertices, indices = ti.field(ti.i32), ti.field(ti.f32), ti.field(ti.i32)
    fields = ti.FieldsBuilder()
    fields.dense(ti.i, 7).place(padding)
    fields.dense(ti.ij, (7, 3)).place(vertices)
    fields.dense(ti.ij, (3, 3)).place(indices)
    tree = fields.finalize()
    padding.fill(917)
    positions = np.full((7, 3), 31, np.float32)
    positions[1:4] = [[0, 0, 0], [2, 0, 0], [0, 2, 0]]
    triangles = np.full((3, 3), 19, np.int32)
    triangles[1] = [0, 1, 2]
    vertices.from_numpy(positions)
    indices.from_numpy(triangles)
    vertex_view = ti.experimental.ndarray_view(
        vertices, slices=(slice(1, 4), slice(None))
    )
    index_view = ti.experimental.ndarray_view(
        indices, slices=(slice(1, 2), slice(None))
    )
    assert vertex_view.descriptor.byte_offset != 0
    assert index_view.descriptor.byte_offset != 0
    if mode == "scene_refit":
        owner = scene = ti.hardware.ray.TriangleScene(vertex_view, index_view)
    else:
        owner = ti.hardware.ray.TriangleBLAS(vertex_view, index_view)
        scene = ti.hardware.ray.InstanceTLAS([ti.hardware.ray.RayInstance(owner)])
    recording = owner.record_build() if mode == "blas_build" else owner.record_refit()
    shift = ti.ndarray(ti.f32, (1,))
    rays = ti.ndarray(ti.f32, (2, 8))
    hits = ti.ndarray(ti.f32, (2, 4))
    ids = ti.ndarray(ti.i32, (2, 4))
    result = ti.ndarray(ti.i32, (2,))
    rays.from_numpy(
        np.array(
            [[0.5, 0.5, 2, 0.001, 0, 0, -1, 100], [4.5, 0.5, 2, 0.001, 0, 0, -1, 100]],
            np.float32,
        )
    )
    shift.fill(4)

    @ti.kernel
    def move(v: ti.types.ndarray(ti.f32, ndim=2), s: ti.types.ndarray(ti.f32, ndim=1)):
        for i in range(3):
            v[i, 0] = s[0] + (2.0 if i == 1 else 0.0)
            v[i, 1] = 2.0 if i == 2 else 0.0
            v[i, 2] = 0.0

    @ti.kernel
    def consume(
        hit: ti.types.ndarray(ti.i32, ndim=2), out: ti.types.ndarray(ti.i32, ndim=1)
    ):
        for i in out:
            out[i] = hit[i, 3]

    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        move,
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "vertices", ti.f32, ndim=2),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "shift", ti.f32, ndim=1),
    )
    builder.append_native(recording, admission="auto")
    if scene is not owner:
        # Independent owners make the BLAS -> TLAS dependency explicit.
        builder.append_native(scene.record_refit(), admission="auto")
    builder.append_native(scene.record_typed(2), admission="auto")
    builder.dispatch(
        consume,
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "hit_indices", ti.i32, ndim=2),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "result", ti.i32, ndim=1),
    )
    graph = builder.compile()
    args = dict(
        vertices=vertex_view,
        shift=shift,
        rays=rays,
        hits=hits,
        hit_indices=ids,
        result=result,
    )
    if mode == "blas_build":
        args["indices"] = index_view
    binding = graph.bind(args)
    assert binding.fast_path_qualified, binding.statistics()
    np.testing.assert_array_equal(vertices.to_numpy(), positions)
    scene.trace_typed(rays, hits, ids)
    np.testing.assert_array_equal(ids.to_numpy()[:, 3], [1, 0])

    def unexpected_prepare(*args, **kwargs):
        raise AssertionError("Geometry layout validation entered fixed replay")

    with monkeypatch.context() as patched:
        patched.setattr(type(recording), "_prepare_packet", unexpected_prepare)
        for displacement in (4, 0, 4):
            shift.fill(displacement)
            graph.run(binding)
            np.testing.assert_array_equal(
                result.to_numpy(), [displacement == 0, displacement == 4]
            )
    np.testing.assert_array_equal(
        vertices.to_numpy()[[0, 4, 5, 6]], positions[[0, 4, 5, 6]]
    )
    np.testing.assert_array_equal(indices.to_numpy(), triangles)
    np.testing.assert_array_equal(padding.to_numpy(), np.full(7, 917))
    revision = binding.revision
    with pytest.raises(RuntimeError, match="compact"):
        binding.update(
            vertices=ti.experimental.ndarray_view(
                vertices, slices=(slice(0, 6, 2), slice(None))
            )
        )
    assert binding.revision == revision
    graph.run(binding)
    native_args = {"vertices": vertex_view}
    if mode == "blas_build":
        native_args["indices"] = index_view
    prepared_geometry = recording.prepare_graph_execute(native_args)
    tree.destroy()
    with pytest.raises(RuntimeError, match="retired|destroyed|generation"):
        prepared_geometry()
    with pytest.raises(RuntimeError, match="retired|destroyed|generation"):
        graph.run(binding)
    graph.close()
    scene.close()
    if owner is not scene:
        owner.close()
