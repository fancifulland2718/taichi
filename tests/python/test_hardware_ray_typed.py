import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=ti.vulkan, offline_cache=False)
@pytest.mark.parametrize("instanced", [False, True])
def test_typed_ray_hits_preserve_indices_barycentrics_and_graph_consumption(
    instanced, monkeypatch
):
    if not ti.hardware.ray.is_available():
        pytest.skip("Vulkan ray query is unavailable")
    vertices = ti.ndarray(ti.f32, shape=(3, 3))
    triangles = ti.ndarray(ti.i32, shape=(1, 3))
    vertices.from_numpy(np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], np.float32))
    triangles.from_numpy(np.array([[0, 1, 2]], np.int32))
    if instanced:
        blas = ti.hardware.ray.TriangleBLAS(vertices, triangles)
        scene = ti.hardware.ray.InstanceTLAS(
            [
                ti.hardware.ray.RayInstance(blas, custom_index=37),
                ti.hardware.ray.RayInstance(
                    blas,
                    transform=(1, 0, 0, 4, 0, 1, 0, 0, 0, 0, 1, 0),
                    custom_index=0xFFFFFD,
                ),
            ]
        )
    else:
        scene = ti.hardware.ray.TriangleScene(vertices, triangles)
    ray_values = np.array(
        [
            [0.5, 0.5, 2, 0.001, 0, 0, -2, 100],
            [4.5, 0.5, 2, 0.001, 0, 0, -2, 100],
            [20, 0.5, 2, 0.001, 0, 0, -2, 100],
        ],
        np.float32,
    )
    rays = ti.ndarray(ti.f32, shape=(3, 8))
    hits = ti.ndarray(ti.f32, shape=(3, 4))
    indices = ti.ndarray(ti.i32, shape=(3, 4))
    legacy = ti.ndarray(ti.f32, shape=(3, 4))
    rays.from_numpy(ray_values)
    scene.trace_typed(rays, hits, indices)
    expected_values = np.array(
        [[1, 0.25, 0.25, 0], [-1, 0, 0, 0], [-1, 0, 0, 0]], np.float32
    )
    expected_indices = np.array(
        [[0, 0, 37 if instanced else 0, 1], [-1, -1, -1, 0], [-1, -1, -1, 0]], np.int32
    )
    if instanced:
        expected_values[1] = expected_values[0]
        expected_indices[1] = [0, 1, 0xFFFFFD, 1]
    np.testing.assert_allclose(hits.to_numpy(), expected_values, atol=1e-6)
    np.testing.assert_array_equal(indices.to_numpy(), expected_indices)

    # Both pipelines may coexist on the same owner. Legacy remains float4.
    scene.trace(rays, legacy)
    legacy_expected = np.full((3, 4), -1, np.float32)
    legacy_expected[:, 3] = 0
    valid = expected_indices[:, 3] == 1
    legacy_expected[valid, 0] = expected_values[valid, 0]
    legacy_expected[valid, 1:3] = expected_indices[valid][:, (0, 2)]
    legacy_expected[valid, 3] = 1
    np.testing.assert_array_equal(legacy.to_numpy(), legacy_expected)

    result = ti.ndarray(ti.i32, shape=3)
    positions = ti.ndarray(ti.f32, shape=(3, 2))

    @ti.kernel
    def consume(
        geometry: ti.types.ndarray(ti.f32, ndim=2),
        ids: ti.types.ndarray(ti.i32, ndim=2),
        picked: ti.types.ndarray(ti.i32, ndim=1),
        points: ti.types.ndarray(ti.f32, ndim=2),
    ):
        for i in picked:
            picked[i] = -1
            points[i, 0] = -1
            points[i, 1] = -1
            if ids[i, 3] != 0:
                picked[i] = ids[i, 2]
                points[i, 0] = 2 * geometry[i, 1]
                points[i, 1] = 2 * geometry[i, 2]

    builder = ti.graph.GraphBuilder()
    builder.append_native(scene.record_typed(3), admission="auto")
    builder.dispatch(
        consume,
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "hits", ti.f32, ndim=2),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "hit_indices", ti.i32, ndim=2),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "result", ti.i32, ndim=1),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "positions", ti.f32, ndim=2),
    )
    graph = builder.compile()
    hits.fill(123)
    indices.fill(123)
    binding = graph.bind(
        {
            "rays": rays,
            "hits": hits,
            "hit_indices": indices,
            "result": result,
            "positions": positions,
        }
    )
    assert binding.fast_path_qualified, binding.statistics()
    # Preparing a binding must not execute the query or its consumer.
    np.testing.assert_array_equal(hits.to_numpy(), np.full((3, 4), 123, np.float32))
    np.testing.assert_array_equal(indices.to_numpy(), np.full((3, 4), 123, np.int32))
    # In-place ray changes reuse the binding. No host hit conversion is needed.
    for order in ([2, 0, 1], [1, 2, 0]):
        rays.from_numpy(ray_values[order])
        with monkeypatch.context() as patched:

            def unexpected_prepare(*args, **kwargs):
                raise AssertionError(
                    "Fixed ray bindings must not be revalidated or reassembled on replay"
                )

            patched.setattr(
                ti.hardware.ray.VulkanRayQueryRecording,
                "prepare_graph_execute",
                unexpected_prepare,
            )
            patched.setattr(
                ti.hardware.ray.VulkanRayQueryRecording,
                "validate_graph_bindings",
                unexpected_prepare,
            )
            graph.run(binding)
        np.testing.assert_array_equal(indices.to_numpy(), expected_indices[order])
        np.testing.assert_allclose(hits.to_numpy(), expected_values[order], atol=1e-6)
        np.testing.assert_array_equal(result.to_numpy(), expected_indices[order, 2])
        expected_points = np.where(expected_indices[order, 3:4] != 0, 0.5, -1.0)
        np.testing.assert_allclose(
            positions.to_numpy(), np.repeat(expected_points, 2, axis=1), atol=1e-6
        )
    with pytest.raises(RuntimeError, match="dtype"):
        scene.trace_typed(rays, hits, legacy)
    # Rebinding publishes a new native packet; the old output remains untouched.
    old_indices = indices.to_numpy()
    replacement_indices = ti.ndarray(ti.i32, shape=(3, 4))
    binding.update(hit_indices=replacement_indices)
    graph.run(binding)
    np.testing.assert_array_equal(indices.to_numpy(), old_indices)
    np.testing.assert_array_equal(
        replacement_indices.to_numpy(), expected_indices[order]
    )
    unsigned_indices = ti.ndarray(ti.u32, shape=(3, 4))
    scene.trace_typed(rays, hits, unsigned_indices)
    np.testing.assert_array_equal(
        unsigned_indices.to_numpy(), expected_indices[order].astype(np.uint32)
    )
    revision = binding.revision
    with pytest.raises(RuntimeError, match="dtype"):
        binding.update(hit_indices=legacy)
    assert binding.revision == revision
    graph.run(binding)
    scene.close()
    with pytest.raises(RuntimeError, match="closed"):
        graph.run(binding)
    graph.close()
    if instanced:
        blas.close()
