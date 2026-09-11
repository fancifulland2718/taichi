from dataclasses import replace

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils
from taichi_forge.lang import impl
from taichi_forge.lang._storage_view import (
    StorageRequirement,
    _flatten_storage_to_scalar_vector,
    analyze_storage_alias,
    describe_storage,
    qualify_storage,
    shadow_validate_dense_field_descriptor,
    shadow_validate_primitive_view,
    validate_storage_owner,
)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_storage_view_describes_existing_dense_storage_without_copy():
    scalar_array = ti.ndarray(ti.f32, shape=(4, 3))
    vector_array = ti.Vector.ndarray(3, ti.f32, shape=4)
    payload = ti.types.struct(
        scalar=ti.f32,
        vector=ti.types.vector(2, ti.f32),
    )
    struct_array = ti.ndarray(payload, shape=4)
    scalar_member = struct_array.field("scalar")
    vector_member = struct_array.field("vector")
    scalar_field = ti.field(ti.f32, shape=(4, 3))
    vector_field = ti.Vector.field(3, ti.f32, shape=4)

    program = impl.get_runtime().prog
    resources_before = program._debug_ndarray_resource_stats()
    scalar_array_view = describe_storage(scalar_array)
    vector_array_view = describe_storage(vector_array)
    scalar_member_view = describe_storage(scalar_member)
    vector_member_view = describe_storage(vector_member)
    scalar_field_view = describe_storage(scalar_field)
    vector_field_view = describe_storage(vector_field)
    resources_after = program._debug_ndarray_resource_stats()
    assert resources_after["live"] == resources_before["live"]
    assert resources_after["created_total"] == resources_before["created_total"]

    for description in (
        scalar_array_view,
        vector_array_view,
        scalar_member_view,
        vector_member_view,
        scalar_field_view,
        vector_field_view,
    ):
        assert description.supported
        assert validate_storage_owner(description) == "kNone"

    assert tuple(scalar_array_view.descriptor.index_shape) == (4, 3)
    assert tuple(scalar_array_view.descriptor.element_shape) == ()
    assert scalar_array_view.properties["compact_contiguous"]
    assert scalar_array_view.properties["ndarray_abi_compatible"]

    assert tuple(vector_array_view.descriptor.index_shape) == (4,)
    assert tuple(vector_array_view.descriptor.element_shape) == (3,)
    assert vector_array_view.properties["array_layout"] == "kAos"
    assert vector_array_view.properties["record_stride"] == 12

    assert not scalar_member_view.properties["compact_contiguous"]
    assert scalar_member_view.properties["single_record_stride_compatible"]
    assert scalar_member_view.properties["record_stride"] == 12
    assert scalar_member_view.descriptor.byte_offset == 0
    assert tuple(vector_member_view.descriptor.element_shape) == (2,)
    assert vector_member_view.descriptor.byte_offset == 4
    assert vector_member_view.properties["record_stride"] == 12

    assert tuple(scalar_field_view.descriptor.index_shape) == (4, 3)
    assert scalar_field_view.descriptor.source_kind == "kDenseScalarField"
    assert scalar_field_view.descriptor.tree_identity is not None
    assert tuple(vector_field_view.descriptor.element_shape) == (3,)
    assert vector_field_view.descriptor.source_kind == "kDensePackedField"
    assert vector_field_view.properties["array_layout"] == "kAos"

    flat_vector_field = _flatten_storage_to_scalar_vector(vector_field_view)
    assert flat_vector_field.supported
    assert tuple(flat_vector_field.descriptor.index_shape) == (12,)
    assert tuple(flat_vector_field.descriptor.element_shape) == ()
    assert flat_vector_field.descriptor.byte_offset == (vector_field_view.descriptor.byte_offset)
    assert flat_vector_field.descriptor.owner_kind == "kSNodePayload"
    assert flat_vector_field.properties["compact_contiguous"]

    direct_record = qualify_storage(
        scalar_member_view,
        StorageRequirement(
            accept_single_record_stride=True,
            require_unique_mapping=True,
        ),
    )
    assert direct_record["supported"]
    assert direct_record["execution_mode"] == "kDirectAffine"
    assert not direct_record["requires_materialization"]

    readonly = describe_storage(scalar_array, access="readonly")
    writable = qualify_storage(
        readonly,
        StorageRequirement(require_writable=True),
    )
    assert not writable["supported"]
    assert writable["reason"] == "kReadOnlySource"

    assert analyze_storage_alias(scalar_array_view, scalar_array_view) == "kProvenOverlap"
    assert analyze_storage_alias(scalar_array_view, vector_array_view) == "kProvenDisjoint"
    assert analyze_storage_alias(scalar_member_view, vector_member_view) == "kUnknown"


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_storage_view_rejects_sparse_field_without_materializing():
    sparse = ti.field(ti.f32)
    builder = ti.FieldsBuilder()
    builder.pointer(ti.i, 8).place(sparse)
    tree = builder.finalize()
    try:
        description = describe_storage(sparse)
        assert not description.supported
        assert description.failure_reason == "kUnsupportedLayout"
    finally:
        tree.destroy()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_storage_view_ndarray_owner_rejects_reused_registry_slot():
    old = ti.ndarray(ti.f32, shape=8)
    old_description = describe_storage(old)
    assert old_description.supported
    old_identity = old_description.descriptor.resource_identity

    program = impl.get_runtime().prog
    old_native = old.arr
    old._invalidate_runtime()
    program.delete_ndarray(old_native)
    assert validate_storage_owner(old_description) == "kStaleOwner"

    replacement = ti.ndarray(ti.f32, shape=8)
    replacement_description = describe_storage(replacement)
    replacement_identity = replacement_description.descriptor.resource_identity
    assert replacement_identity[:3] == old_identity[:3]
    assert replacement_identity[3] != old_identity[3]
    assert validate_storage_owner(old_description) == "kStaleOwner"
    assert validate_storage_owner(replacement_description) == "kNone"


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_storage_view_snode_owner_tracks_tree_generation():
    old_field = ti.field(ti.f32)
    old_builder = ti.FieldsBuilder()
    old_builder.dense(ti.i, 8).place(old_field)
    old_tree = old_builder.finalize()
    old_tree_id = int(old_tree.ptr.id())
    old_description = describe_storage(old_field)
    assert old_description.supported

    old_tree.destroy()
    assert validate_storage_owner(old_description) == "kStaleOwner"

    replacement_field = ti.field(ti.f32)
    replacement_builder = ti.FieldsBuilder()
    replacement_builder.dense(ti.i, 8).place(replacement_field)
    replacement_tree = replacement_builder.finalize()
    try:
        assert int(replacement_tree.ptr.id()) == old_tree_id
        replacement_description = describe_storage(replacement_field)
        assert replacement_description.supported
        assert validate_storage_owner(old_description) == "kRetiredGeneration"
        assert validate_storage_owner(replacement_description) == "kNone"
    finally:
        replacement_tree.destroy()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_storage_view_shadow_checks_existing_algorithm_descriptors(monkeypatch):
    from taichi_forge.algorithms import _algorithms
    from taichi_forge.lang import _storage_view
    from taichi_forge.linalg import _vector_io

    monkeypatch.setattr(_storage_view, "STORAGE_VIEW_SHADOW_MODE", "error")

    array = ti.Vector.ndarray(3, ti.f32, shape=4)
    primitive = _algorithms._primitive_view_legacy(array)
    description = shadow_validate_primitive_view(array, primitive)
    assert description.supported

    authoritative = _algorithms._primitive_view(array)
    assert authoritative.description is not None
    assert authoritative.description.descriptor.fingerprint == (describe_storage(array).descriptor.fingerprint)

    field = ti.Vector.field(3, ti.f32, shape=4)
    legacy_field = _vector_io._describe_value_field_legacy(field, "value")
    description = shadow_validate_dense_field_descriptor(field, legacy_field)
    assert description.supported

    wrong_shape = replace(legacy_field, index_shape=(5,))
    with pytest.raises(RuntimeError, match="index_shape"):
        shadow_validate_dense_field_descriptor(field, wrong_shape)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_native_plan_hot_replay_does_not_reparse_storage(monkeypatch):
    from taichi_forge.algorithms import _algorithms

    program = impl.get_runtime().prog
    if not program.cpu_transform_available():
        pytest.skip("CPU native transform is unavailable")

    source = ti.field(ti.i32, shape=32)
    output = ti.field(ti.i32, shape=32)
    source.fill(3)
    workspace = ti.algorithms.TransformWorkspace(max_items=32)
    ti.algorithms.experimental_transform(
        source,
        output,
        scale=2,
        bias=1,
        method="cpu_native",
        workspace=workspace,
    )

    descriptor_builds = 0
    original_describe_storage = _algorithms.describe_storage

    def count_descriptions(value):
        nonlocal descriptor_builds
        descriptor_builds += 1
        return original_describe_storage(value)

    monkeypatch.setattr(_algorithms, "describe_storage", count_descriptions)
    for _ in range(4):
        ti.algorithms.experimental_transform(
            source,
            output,
            scale=2,
            bias=1,
            method="cpu_native",
            workspace=workspace,
        )

    assert descriptor_builds == 0
    assert (output.to_numpy() == 7).all()


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
    offline_cache=False,
)
def test_ndarray_view_binds_field_and_ndarray_without_temporary_storage():
    @ti.kernel
    def add_bias(values: ti.types.ndarray(dtype=ti.f32, ndim=1), bias: ti.f32):
        for i in values:
            values[i] += bias

    field = ti.field(ti.f32, shape=16)
    array = ti.ndarray(ti.f32, shape=16)
    field.fill(2.0)
    array.fill(5.0)

    program = impl.get_runtime().prog
    before = program._debug_ndarray_resource_stats()
    binding_before = program._debug_dense_storage_binding_stats()
    field_view = ti.experimental.ndarray_view(field)
    array_view = ti.experimental.ndarray_view(array)
    for view in (field_view, array_view):
        qualification = view.runtime_argument.qualification
        assert qualification["describable"]
        assert qualification["bindable"]
        assert qualification["zero_copy_qualified"]
        assert qualification["reason"] == "kNone"
        assert view.runtime_argument.stable_signature != 0
    after_describe = program._debug_ndarray_resource_stats()
    assert after_describe["live"] == before["live"]
    assert after_describe["created_total"] == before["created_total"]

    add_bias(field_view, 3.0)
    add_bias(array_view, 7.0)
    assert (field.to_numpy() == 5.0).all()
    assert (array.to_numpy() == 12.0).all()

    after_launch = program._debug_ndarray_resource_stats()
    assert after_launch["live"] == before["live"]
    assert after_launch["created_total"] == before["created_total"]
    binding_after = program._debug_dense_storage_binding_stats()
    assert binding_after["direct_submissions"] == binding_before["direct_submissions"] + 2
    assert binding_after["resolved_bindings"] == binding_before["resolved_bindings"] + 2
    assert binding_after["field_bindings"] == binding_before["field_bindings"] + 1
    assert binding_after["ndarray_bindings"] == binding_before["ndarray_bindings"] + 1
    assert binding_after["temporary_allocations"] == 0
    assert binding_after["temporary_bytes"] == 0


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
    offline_cache=False,
)
def test_graph_ndarray_prepares_runtime_storage_without_temporary_allocations():
    @ti.kernel
    def increment(values: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in values:
            values[i] += i + 1

    values = ti.ndarray(ti.i32, shape=32)
    values.fill(0)
    symbolic = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(increment, symbolic)
    builder.dispatch(increment, symbolic)
    graph = builder.compile()
    graph.execution_stats()

    program = impl.get_runtime().prog
    before = program._debug_dense_storage_binding_stats()
    resources_before = program._debug_ndarray_resource_stats()
    graph.run({"values": values})
    graph.run({"values": values})
    ti.sync()

    np.testing.assert_array_equal(values.to_numpy(), (np.arange(32, dtype=np.int32) + 1) * 4)
    after = program._debug_dense_storage_binding_stats()
    resources_after = program._debug_ndarray_resource_stats()
    assert after["resolved_bindings"] >= before["resolved_bindings"] + 2
    assert after["ndarray_bindings"] >= before["ndarray_bindings"] + 2
    assert after["temporary_allocations"] == 0
    assert after["temporary_bytes"] == 0
    assert resources_after["created_total"] == resources_before["created_total"]
    assert resources_after["live"] <= resources_before["live"]

    arch = impl.current_cfg().arch
    if arch == ti.cuda:
        stats = graph._graph_stats[0]
        assert stats["captures"] == 1
        assert stats["exact_replays"] == 1
        assert stats["ordinary_fallbacks"] == 0
        assert stats["last_path"] == "cuda_exact_replay"
    elif arch == ti.vulkan:
        stats = graph._graph_stats[0]
        assert stats["records"] == 2
        assert stats["ordinary_fallbacks"] == 0
        assert stats["last_path"] == "vulkan_record"


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
    offline_cache=False,
)
def test_graph_automatically_normalizes_dense_field_and_view_without_copy():
    @ti.kernel
    def increment(values: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in values:
            values[i] += i + 1

    symbolic = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(increment, symbolic)
    builder.dispatch(increment, symbolic)
    graph = builder.compile()
    # Public reports are passive; replay counters require explicit test opt-in.
    _ = graph._graph_stats

    automatic = ti.field(ti.i32, shape=32)
    explicit_field = ti.field(ti.i32, shape=32)
    explicit = ti.experimental.ndarray_view(explicit_field)
    program = impl.get_runtime().prog
    bindings_before = program._debug_dense_storage_binding_stats()
    resources_before = program._debug_ndarray_resource_stats()

    for runtime_value, field in (
        (automatic, automatic),
        (explicit, explicit_field),
    ):
        for run_index in range(9):
            if run_index == 8:
                ti.sync()
            graph.run({"values": runtime_value})
        ti.sync()
        np.testing.assert_array_equal(field.to_numpy(), (np.arange(32, dtype=np.int32) + 1) * 18)

    bindings_after = program._debug_dense_storage_binding_stats()
    resources_after = program._debug_ndarray_resource_stats()
    assert bindings_after["field_bindings"] >= (bindings_before["field_bindings"] + 18)
    assert bindings_after["temporary_allocations"] == 0
    assert bindings_after["temporary_bytes"] == 0
    assert resources_after["created_total"] == resources_before["created_total"]
    assert resources_after["live"] == resources_before["live"]

    arch = impl.current_cfg().arch
    if arch == ti.cuda:
        stats = graph._graph_stats[0]
        assert stats["ordinary_fallbacks"] == 0
        assert stats["captures"] == 1
        # Each field owner has its own retained CUDA signature slot. These
        # counters describe the active slot, not the sum of both executables.
        assert stats["backend_replay_signature_slots"] == 2
        assert stats["patched_replays"] == 0
        assert stats["exact_replays"] == 8
        assert stats["recaptures"] == 0
        assert stats["last_path"] == "cuda_exact_replay"
        assert stats["last_fallback_reason"] == "none"
    elif arch == ti.vulkan:
        stats = graph._graph_stats[0]
        # Completion timing controls which reusable slot is free when the two
        # field owners alternate, so the record/replay split is deliberately
        # not a stable count. Every submission must still use one of those two
        # paths, and both paths must be exercised.
        assert stats["attempts"] == 18
        assert stats["records"] + stats["replays"] == 18
        assert stats["records"] > 0
        assert stats["replays"] > 0
        assert stats["ordinary_fallbacks"] == 0
        assert stats["last_path"] == "vulkan_replay"


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
    offline_cache=False,
)
def test_graph_automatically_normalizes_packed_dense_field():
    vec3 = ti.types.vector(3, ti.f32)

    @ti.kernel
    def update(values: ti.types.ndarray(dtype=vec3, ndim=1)):
        for i in values:
            values[i] += vec3(1.0, 2.0, 3.0)

    symbolic = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", vec3, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(update, symbolic)
    graph = builder.compile()
    _ = graph._graph_stats

    values = ti.Vector.field(3, ti.f32, shape=8)
    values.fill(4.0)
    graph.run({"values": values})
    graph.run({"values": values})
    ti.sync()
    result = values.to_numpy()
    np.testing.assert_array_equal(result[:, 0], np.full(8, 6.0))
    np.testing.assert_array_equal(result[:, 1], np.full(8, 8.0))
    np.testing.assert_array_equal(result[:, 2], np.full(8, 10.0))
    if impl.current_cfg().arch == ti.cuda:
        stats = graph._graph_stats[0]
        assert stats["captures"] == 1
        assert stats["exact_replays"] == 1
        assert stats["ordinary_fallbacks"] == 0
        assert stats["last_path"] == "cuda_exact_replay"


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_graph_dense_field_runtime_argument_validates_declared_type_and_rank():
    @ti.kernel
    def touch(values: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in values:
            values[i] += 1

    symbolic = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(touch, symbolic)
    graph = builder.compile()

    wrong_dtype = ti.field(ti.f32, shape=8)
    with pytest.raises(RuntimeError, match="dtype"):
        graph.run({"values": wrong_dtype})

    wrong_rank = ti.field(ti.i32, shape=(2, 4))
    with pytest.raises(RuntimeError, match="field_dim"):
        graph.run({"values": wrong_rank})

    wrong_element = ti.Vector.field(2, ti.i32, shape=8)
    with pytest.raises(RuntimeError, match="element rank"):
        graph.run({"values": wrong_element})


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_graph_dense_field_runtime_argument_rejects_retired_tree():
    @ti.kernel
    def touch(values: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in values:
            values[i] += 1.0

    symbolic = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", ti.f32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(touch, symbolic)
    graph = builder.compile()

    values = ti.field(ti.f32)
    fields_builder = ti.FieldsBuilder()
    fields_builder.dense(ti.i, 8).place(values)
    tree = fields_builder.finalize()
    graph.run({"values": values})
    tree.destroy()
    with pytest.raises(RuntimeError, match="retired SNodeTree|generation"):
        graph.run({"values": values})


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_graph_borrowed_field_rejects_retired_generation_and_recaptures():
    @ti.kernel
    def touch(values: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in values:
            values[i] += 1.0

    symbolic = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", ti.f32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(touch, symbolic)
    builder.dispatch(touch, symbolic)
    graph = builder.compile()
    _ = graph._graph_stats

    old_values = ti.field(ti.f32)
    old_builder = ti.FieldsBuilder()
    old_builder.dense(ti.i, 8).place(old_values)
    old_tree = old_builder.finalize()
    old_identity = (old_tree.id, old_tree.generation)
    graph.run({"values": old_values})
    graph.run({"values": old_values})
    ti.sync()
    np.testing.assert_array_equal(old_values.to_numpy(), np.full(8, 4.0, dtype=np.float32))
    first_stats = graph._graph_stats[0]
    assert first_stats["captures"] == 1
    assert first_stats["exact_replays"] == 1

    old_tree.destroy()
    with pytest.raises(RuntimeError, match="retired SNodeTree|generation"):
        graph.run({"values": old_values})

    new_values = ti.field(ti.f32)
    new_builder = ti.FieldsBuilder()
    new_builder.dense(ti.i, 8).place(new_values)
    new_tree = new_builder.finalize()
    assert new_tree.id == old_identity[0]
    assert new_tree.generation > old_identity[1]
    graph.run({"values": new_values})
    graph.run({"values": new_values})
    ti.sync()
    np.testing.assert_array_equal(new_values.to_numpy(), np.full(8, 4.0, dtype=np.float32))
    final_stats = graph._graph_stats[0]
    assert final_stats["captures"] == 2
    assert final_stats["recaptures"] == 1
    assert final_stats["exact_replays"] == 2
    assert final_stats["patched_replays"] == 0
    assert final_stats["ordinary_fallbacks"] == 0
    assert final_stats["last_path"] == "cuda_exact_replay"
    new_tree.destroy()


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
    offline_cache=False,
)
def test_ndarray_view_supports_canonical_packed_compound_fields():
    vec3 = ti.types.vector(3, ti.f32)
    mat2 = ti.types.matrix(2, 2, ti.f32)

    @ti.kernel
    def update_vector(values: ti.types.ndarray(dtype=vec3, ndim=1)):
        for i in values:
            values[i] = values[i] + vec3(1.0, 2.0, 3.0)

    @ti.kernel
    def update_matrix(values: ti.types.ndarray(dtype=mat2, ndim=1)):
        for i in values:
            values[i][0, 0] += 1.0
            values[i][0, 1] += 2.0
            values[i][1, 0] += 3.0
            values[i][1, 1] += 4.0

    field = ti.Vector.field(3, ti.f32, shape=8)
    field.fill(4.0)
    view = ti.experimental.ndarray_view(field)
    update_vector(view)
    result = field.to_numpy()
    assert (result[:, 0] == 5.0).all()
    assert (result[:, 1] == 6.0).all()
    assert (result[:, 2] == 7.0).all()

    matrix_field = ti.Matrix.field(2, 2, ti.f32, shape=8)
    matrix_field.fill(5.0)
    matrix_view = ti.experimental.ndarray_view(matrix_field)
    update_matrix(matrix_view)
    matrix_result = matrix_field.to_numpy()
    assert (matrix_result[:, 0, 0] == 6.0).all()
    assert (matrix_result[:, 0, 1] == 7.0).all()
    assert (matrix_result[:, 1, 0] == 8.0).all()
    assert (matrix_result[:, 1, 1] == 9.0).all()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_ndarray_view_rejects_stale_owners_before_launch():
    @ti.kernel
    def touch(values: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in values:
            values[i] += 1.0

    field = ti.field(ti.f32)
    builder = ti.FieldsBuilder()
    builder.dense(ti.i, 8).place(field)
    tree = builder.finalize()
    field_view = ti.experimental.ndarray_view(field)
    touch(field_view)
    tree.destroy()
    with pytest.raises(RuntimeError, match="retired SNodeTree|generation"):
        touch(field_view)

    array = ti.ndarray(ti.f32, shape=8)
    array_view = ti.experimental.ndarray_view(array, slices=slice(0, 8, 2))
    touch(array_view)
    program = impl.get_runtime().prog
    native = array.arr
    array._invalidate_runtime()
    program.delete_ndarray(native)
    with pytest.raises(RuntimeError, match="stale or retired Ndarray"):
        touch(array_view)


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
    offline_cache=False,
)
def test_ndarray_view_executes_positive_padded_dense_layout_without_copy():
    @ti.kernel
    def touch(values: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in values:
            values[i] += 3.0

    lhs = ti.field(ti.f32)
    rhs = ti.field(ti.f32)
    builder = ti.FieldsBuilder()
    builder.dense(ti.i, 8).place(lhs, rhs)
    tree = builder.finalize()
    try:
        lhs.fill(2.0)
        rhs.fill(17.0)
        view = ti.experimental.ndarray_view(lhs)
        assert view.description.properties["element_contiguous"]
        assert not view.description.properties["ndarray_abi_compatible"]
        touch(view)
        assert (lhs.to_numpy() == 5.0).all()
        assert (rhs.to_numpy() == 17.0).all()
    finally:
        tree.destroy()


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
    offline_cache=False,
)
def test_positive_multiaxis_affine_view_matches_ordinary_and_graph_addressing():
    @ti.kernel
    def write(values: ti.types.ndarray(dtype=ti.f32, ndim=2), bias: ti.f32):
        for i, j in values:
            values[i, j] = bias + i * 10.0 + j

    @ti.kernel
    def increment(values: ti.types.ndarray(dtype=ti.f32, ndim=2)):
        for i, j in values:
            values[i, j] += 1.0

    initial = np.full((7, 8), -7.0, dtype=np.float32)
    base = ti.ndarray(ti.f32, shape=(7, 8))
    base.from_numpy(initial)
    view = ti.experimental.ndarray_view(
        base,
        slices=(slice(1, 7, 2), slice(2, 8, 3)),
    )
    assert view.shape == (3, 2)
    assert tuple(view.descriptor.index_strides_bytes) == (64, 12)
    assert view.descriptor.byte_offset == 40
    assert view.description.properties["uniqueness"] == "kProvenUnique"
    assert view.description.properties["element_contiguous"]
    assert not view.description.properties["ndarray_abi_compatible"]

    write(view, 20.0)
    expected = initial.copy()
    expected[np.ix_([1, 3, 5], [2, 5])] = np.array([[20.0, 21.0], [30.0, 31.0], [40.0, 41.0]], dtype=np.float32)
    np.testing.assert_array_equal(base.to_numpy(), expected)

    symbolic_values = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", ti.f32, ndim=2)
    symbolic_bias = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "bias", ti.f32)
    graph_builder = ti.graph.GraphBuilder()
    graph_builder.dispatch(write, symbolic_values, symbolic_bias)
    graph_builder.dispatch(increment, symbolic_values)
    graph = graph_builder.compile()
    graph.execution_stats()
    for _ in range(9):
        graph.run({"values": view, "bias": 100.0})
    expected[np.ix_([1, 3, 5], [2, 5])] = np.array([[101.0, 102.0], [111.0, 112.0], [121.0, 122.0]], dtype=np.float32)
    np.testing.assert_array_equal(base.to_numpy(), expected)

    if impl.current_cfg().arch == ti.cuda:
        stats = graph._graph_stats[0]
        assert stats["captures"] == 0
        assert stats["ordinary_fallbacks"] == 9
        assert stats["last_path"] == "ordinary_fallback"
    elif impl.current_cfg().arch == ti.vulkan:
        stats = graph._graph_stats[0]
        # The replay cache may recycle a completed record before filling all
        # eight in-flight slots. Record/replay distribution is timing- and
        # driver-dependent; the contract is that every submission stays on
        # the Vulkan replay path after at least one recording.
        assert 1 <= stats["records"] <= 8
        assert stats["records"] + stats["replays"] == 9
        assert stats["ordinary_fallbacks"] == 0
        assert stats["last_path"] == "vulkan_replay"


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
    offline_cache=False,
)
def test_positive_affine_view_preserves_contiguous_vector_elements():
    vec3 = ti.types.vector(3, ti.f32)

    @ti.kernel
    def update(values: ti.types.ndarray(dtype=vec3, ndim=1)):
        for i in values:
            values[i][0] += 1.0
            values[i][1] += 2.0
            values[i][2] += 3.0

    base = ti.Vector.ndarray(3, ti.f32, shape=6)
    base.fill(4.0)
    view = ti.experimental.ndarray_view(base, slices=slice(1, 6, 2))
    assert view.shape == (3,)
    assert tuple(view.descriptor.element_shape) == (3,)
    assert tuple(view.descriptor.index_strides_bytes) == (24,)
    assert tuple(view.descriptor.element_strides_bytes) == (4,)
    assert view.descriptor.byte_offset == 12
    update(view)

    expected = np.full((6, 3), 4.0, dtype=np.float32)
    expected[[1, 3, 5], :] += np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.testing.assert_array_equal(base.to_numpy(), expected)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_affine_view_composition_and_a1_exclusions():
    base = ti.ndarray(ti.f32, shape=(8, 9))
    first = ti.experimental.ndarray_view(base, slices=(slice(1, 8, 2), slice(0, 9, 3)))
    second = ti.experimental.ndarray_view(first, slices=(slice(1, 4, 2), slice(1, 3)))
    assert second.shape == (2, 2)
    assert tuple(second.descriptor.index_strides_bytes) == (144, 12)
    assert second.descriptor.byte_offset == 120
    assert second.descriptor.resource_identity == first.descriptor.resource_identity

    empty = ti.experimental.ndarray_view(base, slices=(slice(8, 8), slice(None)))
    assert empty.shape == (0, 9)
    assert empty.description.properties["empty"]

    with pytest.raises(ValueError, match="strictly positive"):
        ti.experimental.ndarray_view(base, slices=(slice(None, None, -1), slice(None)))
    with pytest.raises(ValueError, match="strictly positive"):
        ti.experimental.ndarray_view(base, slices=(slice(None, None, 0), slice(None)))
    with pytest.raises(TypeError, match="bounds must be integers"):
        ti.experimental.ndarray_view(base, slices=(slice(None, None, 1.5), slice(None)))
    with pytest.raises(TypeError, match="one slice per index axis"):
        ti.experimental.ndarray_view(base, slices=(1, slice(None)))
    with pytest.raises(ValueError, match="exactly 2"):
        ti.experimental.ndarray_view(base, slices=(slice(None),))
