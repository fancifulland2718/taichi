from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge._lib import core as ti_core
from taichi_forge.hardware import _capabilities, _optix
from taichi_forge.hardware._external_providers import (
    external_provider_ids,
    passive_external_provider_status,
)
from tests import test_utils


_OPERATION_IDS = (
    "sort.radix.fidelityfx",
    "runtime.buffer_commands.vulkan",
    "image.copy.vulkan",
    "raster.draw.vulkan",
    "raster.adapter.ggui.vulkan",
    "raster.mesh_tasks.vulkan",
    "ray.as_build.vulkan",
    "ray.as_refit.vulkan",
    "ray.query.batch.vulkan",
    "ray.query.inline.vulkan",
    "sampling.texture.vulkan",
    "sampling.texture.cuda",
    "kernel.atomic.cuda",
    "kernel.atomic.vulkan",
    "kernel.simt.warp.cuda",
    "kernel.simt.subgroup.vulkan",
    "kernel.shared_memory.cuda_vulkan",
    "kernel.block_local.cuda",
    "internal.reduction.grouped.cuda_vulkan",
    "internal.listgen.subgroup_ballot.vulkan",
    "matrix.mma.cuda",
    "matrix.mma.vulkan",
    "interop.external_buffer.cuda_vulkan",
    "linalg.cholesky.cusolverdn",
    "linalg.gemm.cublas",
    "linalg.matmul.cublaslt_explicit",
    "linalg.spmv.cusparse",
    "linalg.spmv.cusparse_explicit",
    "linalg.spmm.cusparse_explicit",
    "linalg.spsv.cusparse_explicit",
    "linalg.spsm.cusparse_explicit",
    "fft.transform.vkfft",
    "fft.transform.cufft",
    "linalg.solve.cudss",
    "linalg.solve.cudss_auto",
    "linalg.refactor_solve.cudss",
    "tensor.matmul.cusparselt",
    "tensor.contract.cutensor",
    "linalg.solve.amgx",
    "runtime.probe.cusparselt",
    "runtime.probe.cutensor",
    "runtime.probe.amgx",
    "ray.as_build.optix",
    "ray.as_refit.optix",
    "ray.query.batch.optix",
    "algorithms.primitives.cub",
    "internal.tile.async.cuda",
)


def test_static_hardware_catalog_is_complete_immutable_and_schema_separate():
    ti.reset()
    operations = ti.hardware.operations()

    assert tuple(operation.operation_id for operation in operations) == _OPERATION_IDS
    assert all(
        operation.schema_version == ti.hardware.HARDWARE_CAPABILITY_SCHEMA_VERSION == 4 for operation in operations
    )
    assert ti.algorithms.PRIMITIVE_CAPABILITY_SCHEMA_VERSION == 2
    assert ti.hardware.DEPENDENCY_TIERS == (
        "core",
        "lazy_external",
        "build_external",
    )
    assert "wheel_variant" not in ti.hardware.DEPENDENCY_TIERS
    assert ti.hardware.LOAD_MODES == ("built_in", "runtime_lazy", "build_only")
    assert ti.hardware.ACTIVATION_MODES == (
        "explicit_hardware_api",
        "explicit_kernel_intrinsic",
        "domain_api_auto_provider",
        "compiler_automatic",
    )
    assert ti.hardware.HARDWARE_ROUTE_LEVELS == (
        "qualified",
        "implementation_defined",
        "none",
    )
    assert ti.hardware.PERFORMANCE_STATES == (
        "stable_positive",
        "stable_negative",
        "unstable",
        "not_measured",
    )
    assert ti.hardware.GRAPH_INTEGRATION_MODES == (
        "inline",
        "root_ordered",
        "backend_recorded",
        "stream_captured",
        "opaque",
        "unsupported",
    )
    assert all(operation.backends for operation in operations)
    assert all(operation.scopes for operation in operations)
    assert all(operation.provider_id for operation in operations)
    assert all(operation.graph_integration in ti.hardware.GRAPH_INTEGRATION_MODES for operation in operations)

    with pytest.raises(FrozenInstanceError):
        operations[0].dependency_tier = "wheel_variant"
    with pytest.raises(TypeError):
        operations[0].backends[0] = "cpu"
    with pytest.raises(ValueError, match="unqualified implementations"):
        replace(
            ti.hardware.capability("raster.mesh_tasks.vulkan"),
            hardware_acceleration="qualified",
            hardware_route="qualified",
        )
    with pytest.raises(ValueError, match="load_mode must match"):
        replace(ti.hardware.capability("linalg.gemm.cublas"), load_mode="built_in")
    with pytest.raises(ValueError, match="activation mode"):
        replace(
            ti.hardware.capability("linalg.gemm.cublas"),
            activation_mode="automatic_or_maybe_manual",
        )


def test_hardware_recipe_entry_points_are_static_narrow_and_do_not_load_vendor_runtimes(monkeypatch):
    monkeypatch.setattr(ti_core, "cuda_external_library_status", lambda *args: pytest.fail("no vendor probe"))
    spmm = ti.hardware.capability("linalg.spmm.cusparse_explicit").to_dict()["recipe_search"]
    fft = ti.hardware.capability("fft.transform.cufft").to_dict()["recipe_search"]
    assert spmm["semantic_api"] == "SparseMatrix.record_spmm"
    assert spmm["provider_api"] == "ti.hardware.linalg.SparseSpmmRecipeProvider"
    assert "CUDA f32 CSR SparsePattern" in spmm["scope"]
    assert fft["semantic_api"] == "ti.linalg.record_fft"
    assert "CUDA batched 2D C2C/R2C/C2R f32" in fft["scope"]
    vulkan_fft = ti.hardware.capability("fft.transform.vkfft").to_dict()["recipe_search"]
    assert vulkan_fft["semantic_api"] == "VulkanFftPlan.record"
    assert vulkan_fft["provider_api"] == "ti.hardware.fft.VulkanFftRecipeProvider"
    assert "Vulkan compact in-place complex-f32, rank 1--3" in vulkan_fft["scope"]
    matmul = ti.hardware.capability("linalg.matmul.cublaslt_explicit").to_dict()["recipe_search"]
    assert matmul["semantic_api"] == "ti.linalg.record_matmul"
    assert matmul["provider_api"] == "ti.hardware.linalg.MatmulRecipeProvider"
    assert spmm["qualification"] == "static_entry_points_not_workload_availability"


def test_hardware_activation_modes_make_automatic_and_manual_routes_explicit():
    by_id = {operation.operation_id: operation for operation in ti.hardware.operations()}
    expected = {
        "domain_api_auto_provider": {
            "linalg.spmv.cusparse",
            "linalg.solve.cudss_auto",
        },
        "compiler_automatic": {
            "internal.reduction.grouped.cuda_vulkan",
            "internal.listgen.subgroup_ballot.vulkan",
            "internal.tile.async.cuda",
        },
        "explicit_kernel_intrinsic": {
            "ray.query.inline.vulkan",
            "sampling.texture.vulkan",
            "sampling.texture.cuda",
            "kernel.atomic.cuda",
            "kernel.atomic.vulkan",
            "kernel.simt.warp.cuda",
            "kernel.simt.subgroup.vulkan",
            "kernel.shared_memory.cuda_vulkan",
            "kernel.block_local.cuda",
            "matrix.mma.vulkan",
        },
    }
    for mode, operation_ids in expected.items():
        assert {
            operation_id for operation_id, operation in by_id.items() if operation.activation_mode == mode
        } == operation_ids
    explicit = {
        operation_id
        for operation_id, operation in by_id.items()
        if operation.activation_mode == "explicit_hardware_api"
    }
    assert explicit == set(by_id).difference(set().union(*expected.values()))
    assert by_id["linalg.spmv.cusparse"].activation_mode == ("domain_api_auto_provider")
    assert by_id["linalg.spmv.cusparse_explicit"].activation_mode == ("explicit_hardware_api")


def test_hardware_route_and_scoped_performance_evidence_are_separate():
    operations = ti.hardware.operations()
    assert all(
        operation.hardware_route
        == (
            "qualified"
            if operation.hardware_acceleration in ("guaranteed", "qualified")
            else operation.hardware_acceleration
        )
        for operation in operations
    )

    report = ti.hardware.report()
    assert all(
        operation.performance_state == "not_measured" and not operation.performance_scope
        for operation in report.operations
    )
    payload = next(
        operation
        for operation in report.operations
        if operation.descriptor.operation_id == "runtime.buffer_commands.vulkan"
    ).to_dict()
    assert payload["hardware_route"] == "qualified"
    assert payload["hardware_acceleration"] == "qualified"
    assert payload["performance_state"] == "not_measured"
    assert payload["performance_scope"] == {}

    measured = replace(
        report.operations[0],
        performance_state="stable_positive",
        performance_scope={
            "workload": "synthetic",
            "device": "test-device",
            "revision": "test-revision",
            "baseline": "test-baseline",
        },
    )
    assert measured.to_dict()["performance_state"] == "stable_positive"
    with pytest.raises(ValueError, match="require a performance scope"):
        replace(report.operations[0], performance_state="stable_negative")
    with pytest.raises(ValueError, match="cannot carry a performance scope"):
        replace(report.operations[0], performance_scope={"workload": "stale"})
    with pytest.raises(ValueError, match="must match the legacy"):
        replace(operations[0], hardware_route="none")


def test_stable_hardware_status_layer_excludes_diagnostic_planner_fields():
    ti.reset()

    report = ti.hardware.execution_report()
    capability = report.capability("sampling.texture.vulkan")
    provider = report.provider("vulkan_texture")

    assert isinstance(report, ti.hardware.HardwareExecutionReport)
    assert isinstance(capability, ti.hardware.HardwareCapability)
    assert isinstance(provider, ti.hardware.HardwareProviderStatus)
    assert capability == ti.hardware.status("sampling.texture.vulkan")
    assert provider == ti.hardware.provider_status("vulkan_texture")
    assert capability.operation_id == "sampling.texture.vulkan"
    assert capability.backend is None
    assert capability.selected_provider is None
    assert capability.route == "qualified"
    assert not capability.available
    assert capability.reason != "none"
    assert not provider.available
    assert not provider.selected
    payload = report.to_dict()
    assert set(payload) == {
        "runtime_initialized",
        "backend",
        "capabilities",
        "providers",
    }
    assert "dependency_tier" not in capability.to_dict()
    assert "native_facts" not in capability.to_dict()
    with pytest.raises(KeyError, match="unknown hardware operation"):
        report.capability("missing.operation")
    with pytest.raises(KeyError, match="unknown hardware provider"):
        report.provider("missing_provider")


def test_hardware_catalog_keeps_dependency_and_provider_axes_orthogonal():
    by_id = {operation.operation_id: operation for operation in ti.hardware.operations()}

    optix = by_id["ray.query.batch.optix"]
    cufft = by_id["fft.transform.cufft"]
    cub = by_id["algorithms.primitives.cub"]
    interop = by_id["interop.external_buffer.cuda_vulkan"]

    assert optix.dependency_tier == cufft.dependency_tier == "lazy_external"
    assert optix.provider_class == "vendor_hardware_runtime"
    assert cufft.provider_class == "vendor_algorithm"
    assert cufft.implementation_status == "existing_public"
    assert cufft.graph_integration == "root_ordered"
    assert cufft.public_api == "ti.hardware.fft.CufftPlan1D / CufftPlanND"
    assert cub.dependency_tier == "build_external"
    assert cub.implementation_status == "reference_only"
    assert interop.dependency_tier == "core"
    assert interop.provider_class == "runtime_interop"

    for operation in by_id.values():
        expected_load_mode = {
            "core": "built_in",
            "lazy_external": "runtime_lazy",
            "build_external": "build_only",
        }[operation.dependency_tier]
        assert operation.load_mode == expected_load_mode
        assert (operation.dependency_name is None) == (operation.dependency_tier == "core")

    for operation in by_id.values():
        if operation.implementation_status in (
            "qualification_required",
            "planned",
            "reference_only",
        ):
            assert operation.hardware_acceleration not in ("guaranteed", "qualified")


def test_kernel_and_executable_scope_contracts_do_not_overlap_accidentally():
    for operation in ti.hardware.operations():
        if "kernel" in operation.scopes:
            assert operation.execution_kind == "kernel_intrinsic"
            assert operation.graph_integration == "inline"
        if operation.execution_kind == "external_library":
            assert "kernel" not in operation.scopes
        if operation.operation_id == "ray.query.batch.optix":
            assert operation.scopes == ("python", "graph")
        if operation.operation_id == "ray.query.inline.vulkan":
            assert operation.scopes == ("kernel",)


def test_capability_and_provider_queries_are_stable_and_fail_closed():
    buffer_commands = ti.hardware.capability("runtime.buffer_commands.vulkan")
    assert buffer_commands.implementation_status == "existing_public"
    assert buffer_commands.hardware_acceleration == "qualified"
    assert buffer_commands.scopes == ("python", "graph")
    assert buffer_commands.execution_kind == "native_command"
    assert buffer_commands.public_api == ("ti.graph.VulkanBufferCommandRecording")

    image_copy = ti.hardware.capability("image.copy.vulkan")
    assert image_copy.implementation_status == "existing_public"
    assert image_copy.hardware_acceleration == "implementation_defined"
    assert image_copy.scopes == ("python", "graph")
    assert image_copy.resource_effects == (
        "read:source_image_or_buffer",
        "write:destination_image_or_buffer",
    )

    texture = ti.hardware.capability("sampling.texture.vulkan")
    assert texture.implementation_status == "existing_public"
    assert texture.hardware_acceleration == "qualified"
    assert texture.scopes == ("kernel",)
    assert texture.execution_kind == "kernel_intrinsic"
    assert texture.graph_integration == "inline"
    assert texture.shapes_or_tiles == ("1D", "2D", "3D")
    assert texture.layouts == ("sampled_image", "storage_image")
    assert "SPIR-V OpImageSampleExplicitLod and OpImageFetch" in texture.requirements
    assert "SamplerConfig" in texture.public_api
    assert "fetch uses integer texel coordinates" in texture.notes[2]
    assert texture.deterministic is False

    cuda_texture = ti.hardware.capability("sampling.texture.cuda")
    assert cuda_texture.implementation_status == "existing_public"
    assert cuda_texture.hardware_acceleration == "qualified"
    assert cuda_texture.scopes == ("kernel",)
    assert cuda_texture.execution_kind == "kernel_intrinsic"
    assert cuda_texture.graph_integration == "inline"
    assert cuda_texture.shapes_or_tiles == ("1D", "2D", "3D")
    assert cuda_texture.layouts == ("cuda_array",)
    assert "Driver API symbols" in cuda_texture.notes[0]
    assert "CUDA RW textures remain unsupported" in cuda_texture.notes[1]
    assert "CUDA Graph capture remains unsupported" in cuda_texture.notes[4]
    assert "does not silently replace ndarray loads" in cuda_texture.notes[5]
    assert len(cuda_texture.requirements) == 4

    inline_ray = ti.hardware.capability("ray.query.inline.vulkan")
    assert inline_ray.implementation_status == "existing_public"
    assert inline_ray.hardware_acceleration == "qualified"
    assert "VK_KHR_ray_query" in inline_ray.requirements
    assert "trace_closest" in inline_ray.public_api
    assert "JIT Vulkan kernels only" in inline_ray.notes[2]
    assert "dense-field particle contact" in inline_ray.notes[4]
    assert "prepacked ray-buffer" in inline_ray.notes[5]

    cuda_atomic = ti.hardware.capability("kernel.atomic.cuda")
    vulkan_atomic = ti.hardware.capability("kernel.atomic.vulkan")
    assert cuda_atomic.implementation_status == "existing_public"
    assert vulkan_atomic.implementation_status == "existing_public"
    assert cuda_atomic.public_api == vulkan_atomic.public_api == "ti.atomic_*"
    assert cuda_atomic.scopes == vulkan_atomic.scopes == ("kernel",)
    assert cuda_atomic.graph_integration == vulkan_atomic.graph_integration == "inline"
    assert "atomic-CAS" in cuda_atomic.notes[1]

    warp = ti.hardware.capability("kernel.simt.warp.cuda")
    subgroup = ti.hardware.capability("kernel.simt.subgroup.vulkan")
    assert warp.public_api == "ti.simt.warp"
    assert subgroup.public_api == "ti.simt.subgroup"
    assert warp.backends == ("cuda",)
    assert subgroup.backends == ("vulkan",)
    assert "fail closed" in subgroup.notes[1]

    shared = ti.hardware.capability("kernel.shared_memory.cuda_vulkan")
    block_local = ti.hardware.capability("kernel.block_local.cuda")
    assert shared.backends == ("cuda", "vulkan")
    assert shared.public_api == "ti.simt.block.SharedArray"
    assert block_local.public_api == "ti.block_local"
    assert block_local.requirements[0] == "ti.extension.bls"
    assert "automatically retains" in block_local.notes[1]

    grouped_reduction = ti.hardware.capability("internal.reduction.grouped.cuda_vulkan")
    listgen = ti.hardware.capability("internal.listgen.subgroup_ballot.vulkan")
    assert grouped_reduction.implementation_status == "existing_internal"
    assert listgen.implementation_status == "existing_internal"
    assert grouped_reduction.scopes == listgen.scopes == ("internal",)
    assert grouped_reduction.public_api is None
    assert listgen.public_api is None
    assert "Automatic" in grouped_reduction.notes[0]
    assert "Automatic" in listgen.notes[0]

    raster = ti.hardware.capability("raster.draw.vulkan")
    assert raster.semantic_family == "raster.draw"
    assert raster.public_api == "ti.hardware.graphics.VulkanGraphicsPipeline"
    assert raster.implementation_status == "existing_public"
    assert raster.hardware_acceleration == "qualified"
    assert raster.scopes == ("python", "graph")
    assert raster.execution_kind == "native_command"
    assert raster.graph_integration == "root_ordered"
    assert "index:u32" in raster.dtypes
    assert all("index:i32" not in dtype for dtype in raster.dtypes)
    assert "kernel calls are impossible" in raster.notes[0]
    assert raster.workspace_ownership == "provider_owned"
    assert raster.layouts[0] == "declared vertex bindings and attributes"
    assert raster.deterministic is False
    assert ti.hardware.capability(raster.operation_id) is raster

    adapter = ti.hardware.capability("raster.adapter.ggui.vulkan")
    assert adapter.semantic_family == "raster.adapter"
    assert adapter.public_api == "ti.hardware.raster.RasterPass"
    assert adapter.graph_integration == "opaque"
    assert adapter.layouts == ("mesh", "mesh_instance", "particles", "lines")
    assert "qualification adapter only" in adapter.notes[0]

    matrix = ti.hardware.capability("matrix.mma.cuda")
    assert matrix.semantic_family == "matrix.mma"
    assert matrix.public_api == "ti.hardware.matrix.mma_f16_f32"
    assert matrix.implementation_status == "existing_public"
    assert matrix.hardware_acceleration == "qualified"
    assert matrix.scopes == ("python", "graph")
    assert matrix.execution_kind == "native_command"
    assert matrix.graph_integration == "root_ordered"
    assert matrix.stream_binding == "runtime_ordered"
    assert matrix.workspace_ownership == "none"
    assert matrix.shapes_or_tiles == ("m16n16k16", "compact batch")
    assert matrix.layouts == (
        "row_major_a",
        "row_major_b",
        "row_major_output",
    )
    assert "kernel calls remain unsupported" in matrix.notes[0]

    vulkan_matrix = ti.hardware.capability("matrix.mma.vulkan")
    assert vulkan_matrix.implementation_status == "existing_public"
    assert vulkan_matrix.hardware_acceleration == "qualified"
    assert vulkan_matrix.scopes == ("kernel",)
    assert vulkan_matrix.graph_integration == "inline"
    assert vulkan_matrix.dtypes == (
        "a:f16",
        "b:f16",
        "c:f32",
        "output:f32",
    )
    assert vulkan_matrix.numeric_contracts == (
        "output=A*B+C",
        "f16_inputs:f32_accumulate",
    )
    assert "tuple enumeration" in vulkan_matrix.requirements[1]
    assert "instead of copying" in vulkan_matrix.notes[1]
    assert "NVIDIA Vulkan" in vulkan_matrix.notes[2]

    optix = ti.hardware.capability("ray.query.batch.optix")
    assert optix.implementation_status == "internal_foundation"
    assert "OPTIX_ABI_VERSION 93, 105, or 118" in optix.requirements[1]
    assert "failure-isolated" in optix.notes[1]
    assert "No kernel-inline route" in optix.notes[2]

    optix_build = ti.hardware.capability("ray.as_build.optix")
    assert optix_build.graph_integration == "unsupported"
    assert optix_build.update_policy == "rebuild"
    assert "share one runtime wheel" in optix_build.notes[1]

    optix_refit = ti.hardware.capability("ray.as_refit.optix")
    assert optix_refit.graph_integration == "opaque"
    assert optix_refit.update_policy == "refit"
    assert optix_refit.scopes == ("python", "graph")

    cusparse = ti.hardware.capability("linalg.spmv.cusparse")
    assert cusparse.implementation_status == "existing_public"
    assert cusparse.scopes == ("python",)
    assert cusparse.graph_integration == "unsupported"
    assert cusparse.stream_binding == "runtime_ordered"
    assert cusparse.workspace_ownership == "provider_owned"
    assert cusparse.public_api == ("ti.linalg.SparseMatrix.spmv / set_provider_profile")
    assert "considers cuSPARSE automatically on CUDA" in cusparse.notes[0]
    assert "fails closed" in cusparse.notes[1]
    assert "handwritten timing fields" in cusparse.notes[2]
    assert "topology fingerprint" in cusparse.notes[3]

    explicit_cusparse = ti.hardware.capability("linalg.spmv.cusparse_explicit")
    assert explicit_cusparse.implementation_status == "existing_public"
    assert explicit_cusparse.scopes == ("python", "graph")
    assert explicit_cusparse.graph_integration == "root_ordered"
    assert explicit_cusparse.stream_binding == "runtime_ordered"
    assert explicit_cusparse.workspace_ownership == "provider_owned"
    assert explicit_cusparse.public_api == "ti.hardware.linalg.spmv_f32"
    assert "manual hardware interface" in explicit_cusparse.notes[2]

    spmm = ti.hardware.capability("linalg.spmm.cusparse_explicit")
    assert spmm.implementation_status == "existing_public"
    assert spmm.scopes == ("python", "graph")
    assert spmm.graph_integration == "root_ordered"
    assert spmm.stream_binding == "runtime_ordered"
    assert spmm.workspace_ownership == "provider_owned"
    assert spmm.public_api == "ti.hardware.linalg.spmm_f32"
    assert "fixed costs" in spmm.notes[2]
    assert "single-inflight" in spmm.notes[4]

    spsv = ti.hardware.capability("linalg.spsv.cusparse_explicit")
    assert spsv.implementation_status == "existing_public"
    assert spsv.scopes == ("python", "graph")
    assert spsv.graph_integration == "root_ordered"
    assert spsv.stream_binding == "runtime_ordered"
    assert spsv.workspace_ownership == "provider_owned"
    assert spsv.update_policy == "implementation_defined"
    assert spsv.public_api == "ti.hardware.linalg.spsv_f32"
    assert "fixed costs" in spsv.notes[2]
    assert "never selected automatically" in spsv.notes[0]
    assert "older providers require a new matrix generation" in spsv.notes[3]
    assert "no host postsolve check" in spsv.notes[5]

    spsm = ti.hardware.capability("linalg.spsm.cusparse_explicit")
    assert spsm.implementation_status == "existing_public"
    assert spsm.scopes == ("python", "graph")
    assert spsm.graph_integration == "root_ordered"
    assert spsm.stream_binding == "runtime_ordered"
    assert spsm.workspace_ownership == "provider_owned"
    assert spsm.update_policy == "implementation_defined"
    assert spsm.public_api == "ti.hardware.linalg.spsm_f32"
    assert "right-hand-side count" in spsm.notes[1]
    assert "older providers require a new matrix generation" in spsm.notes[2]
    assert "no automatic route" in spsm.notes[3]
    assert "no host postsolve check" in spsm.notes[4]

    cudss = ti.hardware.capability("linalg.solve.cudss")
    assert cudss.implementation_status == "existing_public"
    assert cudss.scopes == ("python", "graph")
    assert cudss.graph_integration == "root_ordered"
    assert cudss.stream_binding == "runtime_ordered"
    assert cudss.workspace_ownership == "provider_owned"
    assert cudss.public_api == ("ti.hardware.linalg.CudssPlan / CudssSolveRecording")
    assert cudss.requirements[0] == "CUDA driver API >= 12.0"
    assert cudss.requirements[1] == "bundled Forge cuDSS provider C ABI adapter"
    assert cudss.requirements[2] == "user-managed cuDSS 0.8.x shared library"
    assert "never rewritten" in cudss.notes[0]
    assert "no Forge wheel variant" in cudss.notes[1]
    assert cudss.recipe_semantic_api == "ti.linalg.record_sparse_solve"
    assert cudss.recipe_provider_api == "ti.hardware.linalg.SparseSolveRecipeProvider"
    assert "ordered vector RHS pairs" in cudss.recipe_scope[1]
    assert "Legacy CudssSolveRecording" in cudss.notes[3]

    automatic_cudss = ti.hardware.capability("linalg.solve.cudss_auto")
    assert automatic_cudss.activation_mode == "domain_api_auto_provider"
    assert automatic_cudss.scopes == ("python",)
    assert automatic_cudss.graph_integration == "unsupported"
    assert automatic_cudss.public_api == ("ti.linalg.SparseSolver(provider='auto', provider_profile=profile)")
    assert "exact topology" in automatic_cudss.notes[1]
    assert "does not probe or load" in automatic_cudss.notes[2]
    assert "never performance-gated" in automatic_cudss.notes[4]
    assert automatic_cudss.requirements[0] == "CUDA driver API >= 12.0"
    assert "retain cuSOLVERSp" in automatic_cudss.notes[3]

    transactional_cudss = ti.hardware.capability("linalg.refactor_solve.cudss")
    assert transactional_cudss.activation_mode == "explicit_hardware_api"
    assert transactional_cudss.scopes == ("python", "graph")
    assert transactional_cudss.graph_integration == "root_ordered"
    assert transactional_cudss.stream_binding == "runtime_ordered"
    assert transactional_cudss.lifetime_policy == "provider_plan"
    assert transactional_cudss.update_policy == "rebind"
    assert transactional_cudss.resource_effects == (
        "read:matrix_values",
        "read:rhs",
        "write:solution",
        "read_write:provider_factor_workspace",
    )
    assert "at most one" in transactional_cudss.notes[1]
    assert "no host numeric-version" in transactional_cudss.notes[2]

    async_tile = ti.hardware.capability("internal.tile.async.cuda")
    assert async_tile.implementation_status == "existing_internal"
    assert async_tile.hardware_acceleration == "qualified"
    assert async_tile.scopes == ("internal",)
    assert async_tile.execution_kind == "kernel_intrinsic"
    assert async_tile.public_api is None
    assert async_tile.shapes_or_tiles == ("compiler-generated struct-for BLS >= 8192 bytes",)
    assert "no public cp.async or TMA API" in async_tile.notes[0]

    mesh_shader = ti.hardware.capability("raster.mesh_tasks.vulkan")
    assert mesh_shader.implementation_status == "qualification_required"
    assert mesh_shader.hardware_acceleration == "implementation_defined"
    assert mesh_shader.activation_mode == "explicit_hardware_api"
    assert "meshShader feature" in mesh_shader.requirements[1]
    assert "ordinary kernel calls are impossible" in mesh_shader.notes[0]

    with pytest.raises(KeyError, match="unknown hardware operation"):
        ti.hardware.capability("missing.operation")
    with pytest.raises(TypeError, match="nonempty string"):
        ti.hardware.capability(None)

    providers = ti.hardware.providers()
    provider_ids = tuple(provider.provider_id for provider in providers)
    assert provider_ids == tuple(sorted(provider_ids))
    assert "cublas" in provider_ids
    assert "cublaslt" in provider_ids
    assert "cudss" in provider_ids
    assert "cub_reference" in provider_ids
    assert all(provider.operation_ids for provider in providers)
    assert next(provider for provider in providers if provider.provider_id == "cublas").to_dict() == {
        "schema_version": 4,
        "provider_id": "cublas",
        "dependency_tier": "lazy_external",
        "dependency_name": "cuBLAS",
        "load_mode": "runtime_lazy",
        "provider_class": "vendor_algorithm",
        "operation_ids": ("linalg.gemm.cublas",),
    }
    cublaslt = ti.hardware.capability("linalg.matmul.cublaslt_explicit")
    assert cublaslt.implementation_status == "existing_public"
    assert cublaslt.activation_mode == "explicit_hardware_api"
    assert cublaslt.scopes == ("python", "graph")
    assert cublaslt.workspace_ownership == "provider_owned"
    assert cublaslt.lifetime_policy == "provider_plan"
    assert cublaslt.update_policy == "rebind"
    assert cublaslt.public_api == "ti.linalg.record_matmul"
    assert "no automatic crossover" in cublaslt.notes[2]
    cub = ti.hardware.capability("algorithms.primitives.cub")
    assert cub.scopes == ("python", "graph")
    assert cub.stream_binding == "runtime_ordered"
    assert cub.workspace_ownership == "provider_owned"
    assert cub.lifetime_policy == "provider_plan"
    assert cub.update_policy == "rebind"
    assert cub.public_api is None
    assert "explicit only" in cub.notes[2]
    assert len({operation_id for provider in providers for operation_id in provider.operation_ids}) == len(
        ti.hardware.operations()
    )


def test_static_hardware_descriptor_serialization_is_plain_and_complete():
    descriptor = ti.hardware.capability("matrix.mma.vulkan")
    payload = descriptor.to_dict()

    assert payload["schema_version"] == 4
    assert payload["operation_id"] == descriptor.operation_id
    assert "tuple enumeration" in payload["requirements"][1]
    assert payload["hardware_acceleration"] == "qualified"
    assert payload["implementation_status"] == "existing_public"
    assert payload["activation_mode"] == "explicit_kernel_intrinsic"
    assert payload["graph_integration"] == "inline"
    assert "graph_support" not in payload
    assert payload["load_mode"] == "built_in"
    assert payload["resource_effects"] == (
        "read:a",
        "read:b",
        "read:c",
        "write:output",
    )
    assert payload["lifetime_policy"] == "runtime_generation"
    assert payload["update_policy"] == "immutable"
    assert payload["dtypes"] == (
        "a:f16",
        "b:f16",
        "c:f32",
        "output:f32",
    )
    assert payload["deterministic"] is False
    assert payload["fallback_provider"] is None
    assert payload["fallback_equivalent"] is None

    payload["requirements"] = ()
    assert "SPV_KHR_cooperative_matrix" in descriptor.requirements[3]


@pytest.mark.run_in_serial
def test_vulkan_cooperative_matrix_properties_are_exact_admission_tuples():
    ti.init(arch=ti.vulkan)
    program = ti.lang.impl.get_runtime().prog
    available = bool(program.vulkan_cooperative_matrix_available())
    properties = tuple(dict(item) for item in program._vulkan_cooperative_matrix_properties())

    assert available == bool(properties)
    required = {
        "m",
        "n",
        "k",
        "a_type",
        "b_type",
        "c_type",
        "result_type",
        "scope",
        "saturating_accumulation",
        "subgroup_size",
        "supported_stages",
    }
    for item in properties:
        assert set(item) == required
        assert item["m"] > 0 and item["n"] > 0 and item["k"] > 0
        assert item["scope"] in (1, 2, 3, 5)
        assert item["subgroup_size"] > 0
        assert item["supported_stages"] & 0x20  # VK_SHADER_STAGE_COMPUTE_BIT

    resolved = next(
        operation
        for operation in ti.hardware.report().operations
        if operation.descriptor.operation_id == "matrix.mma.vulkan"
    )
    assert resolved.native_facts["provider_available"] == available
    assert tuple(resolved.native_facts["supported_tuples"]) == properties
    executable = tuple(
        item
        for item in properties
        if (
            item["a_type"],
            item["b_type"],
            item["c_type"],
            item["result_type"],
        )
        == (0, 0, 1, 1)
        and item["scope"] == 3
        and not item["saturating_accumulation"]
    )
    assert resolved.native_facts["operation_requirements_evaluated"] is True
    assert resolved.native_facts["executable_f16_f32_tuple_count"] == len(executable)
    assert resolved.selection == ("eligible" if executable else "rejected")
    assert resolved.unavailable_reason == ("none" if executable else "hardware_requirement_not_met")


def test_passive_report_does_not_probe_or_enable_external_components(monkeypatch):
    def reject_implicit_probe(_provider_id):
        raise AssertionError("passive reports must not invoke a native D1 probe")

    monkeypatch.setattr(_capabilities, "_native_external_probe", reject_implicit_probe)
    monkeypatch.setattr(
        _capabilities,
        "_native_external_status",
        lambda provider_id: {
            "provider_id": provider_id,
            "library_loaded": False,
            "provider_abi": None,
            "provider_version": None,
            "native_facts": {
                "status_policy": "passive_existing_loader",
                "external_component_probed": False,
            },
        },
    )
    ti.reset()
    report = ti.hardware.report()

    assert report.schema_version == 4
    assert report.runtime_initialized is False
    assert report.backend is None
    assert report.external_components_probed is False
    assert set(report.compiled_backends) == {"cuda", "vulkan"}
    assert len(report.operations) == len(ti.hardware.operations())

    by_id = {operation.descriptor.operation_id: operation for operation in report.operations}
    cublas = by_id["linalg.gemm.cublas"]
    if cublas.unavailable_reason == "external_probe_not_requested":
        assert cublas.discovery is None
    assert cublas.enablement == "disabled"
    assert cublas.selection == "not_considered"
    assert cublas.unavailable_reason in (
        "external_probe_not_requested",
        "backend_not_compiled",
    )
    assert cublas.native_facts["external_component_probed"] is False
    assert cublas.provider_abi is None
    assert cublas.provider_version is None
    assert cublas.last_error is None
    assert cublas.failure_scope is None


def test_passive_report_observes_an_already_loaded_provider(monkeypatch):
    monkeypatch.setattr(
        _capabilities,
        "_runtime_facts",
        lambda: (True, "cuda", {"cuda": True, "vulkan": True}),
    )

    def provider_status(provider_id):
        return {
            "provider_id": provider_id,
            "library_loaded": provider_id == "cusparse",
            "provider_abi": f"{provider_id}-dynamic-symbols-v1",
            "provider_version": "12.6.3" if provider_id == "cusparse" else None,
            "native_facts": {
                "status_policy": "passive_existing_loader",
                "external_component_probed": False,
                "provider_enablement_changed": False,
                "provider_selection_changed": False,
                "generic_bsr_spmv_available": provider_id == "cusparse",
                "spmm_f32_available": provider_id == "cusparse",
                "spsv_f32_available": provider_id == "cusparse",
                "spsm_f32_available": provider_id == "cusparse",
                "spsv_value_update_available": False,
                "spsm_value_update_available": False,
            },
        }

    monkeypatch.setattr(_capabilities, "_native_external_status", provider_status)
    report = ti.hardware.report()
    cusparse = next(operation for operation in report.operations if operation.descriptor.provider_id == "cusparse")
    cublas = next(operation for operation in report.operations if operation.descriptor.provider_id == "cublas")
    spmm = next(
        operation
        for operation in report.operations
        if operation.descriptor.operation_id == "linalg.spmm.cusparse_explicit"
    )
    spsv = next(
        operation
        for operation in report.operations
        if operation.descriptor.operation_id == "linalg.spsv.cusparse_explicit"
    )
    spsm = next(
        operation
        for operation in report.operations
        if operation.descriptor.operation_id == "linalg.spsm.cusparse_explicit"
    )

    assert cusparse.discovery == "available"
    assert cusparse.enablement == "enabled"
    assert cusparse.selection == "eligible"
    assert cusparse.unavailable_reason == "none"
    assert cusparse.provider_abi == "cusparse-dynamic-symbols-v1"
    assert cusparse.provider_version == "12.6.3"
    assert cusparse.native_facts["generic_bsr_spmv_available"]
    assert not cusparse.native_facts["external_component_probed"]
    assert spmm.discovery == "available"
    assert spmm.selection == "eligible"
    assert spmm.native_facts["spmm_f32_available"]
    assert spsv.discovery == "available"
    assert spsv.selection == "eligible"
    assert spsv.native_facts["spsv_f32_available"]
    assert not spsv.native_facts["spsv_value_update_available"]
    assert spsm.discovery == "available"
    assert spsm.selection == "eligible"
    assert spsm.native_facts["spsm_f32_available"]
    assert not spsm.native_facts["spsm_value_update_available"]
    assert cublas.enablement == "disabled"
    assert cublas.selection == "not_considered"


def test_hardware_report_passive_status_does_not_load_external_libraries():
    before = {
        provider_id: passive_external_provider_status(provider_id)
        for provider_id in external_provider_ids()
    }
    report = ti.hardware.report()
    after = {
        provider_id: passive_external_provider_status(provider_id)
        for provider_id in external_provider_ids()
    }

    assert report.external_components_probed is False
    assert {provider_id: status["library_loaded"] for provider_id, status in after.items()} == {
        provider_id: status["library_loaded"] for provider_id, status in before.items()
    }
    assert all(status["native_facts"]["status_policy"] for status in after.values())
    assert all(not status["native_facts"]["external_component_probed"] for status in after.values())
    assert all(not status["native_facts"]["provider_enablement_changed"] for status in after.values())
    assert all(not status["native_facts"]["provider_selection_changed"] for status in after.values())


def test_multibackend_core_route_requires_every_backend(monkeypatch):
    monkeypatch.setattr(
        _capabilities,
        "_runtime_facts",
        lambda: (False, None, {"cuda": True, "vulkan": False}),
    )

    report = ti.hardware.report()
    interop = next(
        operation
        for operation in report.operations
        if operation.descriptor.operation_id == "interop.external_buffer.cuda_vulkan"
    )

    assert interop.discovery == "missing"
    assert interop.selection == "rejected"
    assert interop.unavailable_reason == "backend_not_compiled"
    assert interop.native_facts["provider_backends_compiled"] == ("cuda",)


def test_cuda_matrix_capability_rejects_an_unqualified_active_device(
    monkeypatch,
):
    monkeypatch.setattr(
        _capabilities,
        "_runtime_facts",
        lambda: (True, "cuda", {"cuda": True, "vulkan": True}),
    )
    monkeypatch.setattr(
        _capabilities,
        "_passive_core_statuses",
        lambda runtime_initialized, backend: {
            "matrix.mma.cuda": {
                "available": False,
                "native_facts": {"provider_available": False},
            }
        },
    )

    operation = next(
        operation
        for operation in ti.hardware.report().operations
        if operation.descriptor.operation_id == "matrix.mma.cuda"
    )
    assert operation.discovery == "incompatible"
    assert operation.enablement == "enabled"
    assert operation.selection == "rejected"
    assert operation.unavailable_reason == "hardware_requirement_not_met"
    assert operation.native_facts["provider_available"] is False


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_async_tile_is_automatic_workload_gated_and_reported():
    initial = next(
        operation
        for operation in ti.hardware.report().operations
        if operation.descriptor.operation_id == "internal.tile.async.cuda"
    )
    if not initial.native_facts.get("provider_available", False):
        pytest.skip("CUDA async tile requires sm_80 and PTX 7.0 or newer")

    assert initial.selection == "eligible"
    assert initial.native_facts["minimum_bls_bytes"] == 8192
    assert initial.native_facts["lowered_specializations"] == 0

    small_input = ti.field(ti.f32)
    small_output = ti.field(ti.f32)
    ti.root.pointer(ti.i, 1).dense(ti.i, 64).place(small_input, small_output)

    x0 = ti.field(ti.f32)
    x1 = ti.field(ti.f32)
    x2 = ti.field(ti.f32)
    x3 = ti.field(ti.f32)
    x4 = ti.field(ti.f32)
    x5 = ti.field(ti.f32)
    x6 = ti.field(ti.f32)
    x7 = ti.field(ti.f32)
    output = ti.field(ti.f32)
    block = ti.root.pointer(ti.ij, 1).dense(ti.ij, (16, 16))
    block.place(x0, x1, x2, x3, x4, x5, x6, x7, output)

    @ti.kernel
    def initialize():
        for i in small_input:
            small_input[i] = i
        for i, j in ti.ndrange(16, 16):
            base = i * 16 + j
            x0[i, j] = base
            x1[i, j] = base + 1
            x2[i, j] = base + 2
            x3[i, j] = base + 3
            x4[i, j] = base + 4
            x5[i, j] = base + 5
            x6[i, j] = base + 6
            x7[i, j] = base + 7

    @ti.kernel
    def small_copy():
        ti.loop_config(block_dim=64)
        ti.block_local(small_input)
        for i in small_input:
            small_output[i] = small_input[i]

    @ti.kernel
    def admitted_copy():
        ti.loop_config(block_dim=256)
        ti.block_local(x0, x1, x2, x3, x4, x5, x6, x7)
        for i, j in x0:
            output[i, j] = x0[i, j] + x1[i, j] + x2[i, j] + x3[i, j] + x4[i, j] + x5[i, j] + x6[i, j] + x7[i, j]

    initialize()
    small_copy()
    ti.sync()
    after_small = next(
        operation
        for operation in ti.hardware.report().operations
        if operation.descriptor.operation_id == "internal.tile.async.cuda"
    )
    assert after_small.selection == "eligible"
    assert after_small.native_facts["lowered_specializations"] == 0
    assert after_small.native_facts["candidates"] >= 1
    assert after_small.native_facts["admitted"] == 0
    assert after_small.native_facts["rejected"] >= 1
    assert after_small.native_facts["fallback"] >= 1
    assert after_small.native_facts["below_size"] >= 1

    admitted_copy()
    ti.sync()
    base = np.arange(256, dtype=np.float32).reshape(16, 16)
    np.testing.assert_array_equal(output.to_numpy(), base * 8 + 28)

    selected = next(
        operation
        for operation in ti.hardware.report().operations
        if operation.descriptor.operation_id == "internal.tile.async.cuda"
    )
    assert selected.discovery == "available"
    assert selected.enablement == "enabled"
    assert selected.selection == "selected"
    assert selected.unavailable_reason == "none"
    assert selected.native_facts["lowered_specializations"] >= 1
    assert selected.native_facts["copy_sites"] >= 8
    assert selected.native_facts["admitted"] >= 8
    assert selected.native_facts["lowered"] >= 8
    assert selected.native_facts["candidates"] == (
        selected.native_facts["admitted"] + selected.native_facts["rejected"]
    )
    assert (
        sum(
            selected.native_facts[name]
            for name in (
                "below_size",
                "read_write_bls",
                "unsupported_width",
                "non_direct_address",
                "alias_unknown",
                "shared_memory_pressure",
                "target_capability",
                "cost_gate",
            )
        )
        == selected.native_facts["rejected"]
    )


@test_utils.test(arch=ti.cuda, require=ti.extension.data64, offline_cache=False)
def test_cuda_async_tile_preserves_f64_block_local_values():
    initial = next(
        operation
        for operation in ti.hardware.report().operations
        if operation.descriptor.operation_id == "internal.tile.async.cuda"
    )
    if not initial.native_facts.get("provider_available", False):
        pytest.skip("CUDA async tile requires sm_80 and PTX 7.0 or newer")

    x0 = ti.field(ti.f64)
    x1 = ti.field(ti.f64)
    x2 = ti.field(ti.f64)
    x3 = ti.field(ti.f64)
    output = ti.field(ti.f64)
    block = ti.root.pointer(ti.ij, 1).dense(ti.ij, (16, 16))
    block.place(x0, x1, x2, x3, output)

    @ti.kernel
    def initialize():
        for i, j in ti.ndrange(16, 16):
            base = ti.cast(i * 16 + j, ti.f64) * 0.25
            x0[i, j] = base
            x1[i, j] = base + 1.0
            x2[i, j] = base + 2.0
            x3[i, j] = base + 3.0

    @ti.kernel
    def admitted_copy():
        ti.loop_config(block_dim=256)
        ti.block_local(x0, x1, x2, x3)
        for i, j in x0:
            output[i, j] = x0[i, j] + x1[i, j] + x2[i, j] + x3[i, j]

    initialize()
    admitted_copy()
    ti.sync()
    base = np.arange(256, dtype=np.float64).reshape(16, 16) * 0.25
    np.testing.assert_array_equal(output.to_numpy(), base * 4 + 6)

    selected = next(
        operation
        for operation in ti.hardware.report().operations
        if operation.descriptor.operation_id == "internal.tile.async.cuda"
    )
    assert selected.selection == "selected"
    assert selected.native_facts["copy_sites"] >= 4


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_async_tile_rejects_read_write_block_local_cache():
    initial = next(
        operation
        for operation in ti.hardware.report().operations
        if operation.descriptor.operation_id == "internal.tile.async.cuda"
    )
    if not initial.native_facts.get("provider_available", False):
        pytest.skip("CUDA async tile requires sm_80 and PTX 7.0 or newer")

    values = ti.field(ti.f32)
    ti.root.pointer(ti.ij, 1).dense(ti.ij, (64, 64)).place(values)

    @ti.kernel
    def initialize():
        for i, j in ti.ndrange(64, 64):
            values[i, j] = 1.0

    @ti.kernel
    def update():
        ti.loop_config(block_dim=256)
        ti.block_local(values)
        for i, j in values:
            values[i, j] += 1.0

    initialize()
    update()
    ti.sync()
    np.testing.assert_array_equal(values.to_numpy(), np.full((64, 64), 2.0, dtype=np.float32))

    after_update = next(
        operation
        for operation in ti.hardware.report().operations
        if operation.descriptor.operation_id == "internal.tile.async.cuda"
    )
    assert after_update.selection == "eligible"
    assert after_update.native_facts["lowered_specializations"] == 0
    assert after_update.native_facts["copy_sites"] == 0
    assert after_update.native_facts["candidates"] >= 1
    assert after_update.native_facts["admitted"] == 0
    assert after_update.native_facts["read_write_bls"] >= 1


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_sparse_block_local_scatter_falls_back_to_global_atomics():
    source = ti.field(ti.i32)
    destination = ti.field(ti.i32)
    ti.root.pointer(ti.i, 16).dense(ti.i, 8).place(source)
    ti.root.pointer(ti.i, 16).dense(ti.i, 8).place(destination)
    active_destination_elements = ti.field(ti.i32, shape=())

    @ti.kernel
    def initialize():
        for lane in range(8):
            source[lane] = lane + 1
            source[32 + lane] = 101 + lane

    @ti.kernel
    def scatter():
        ti.block_local(destination)
        for i in source:
            ti.atomic_add(destination[2 * i], source[i])

    @ti.kernel
    def count_active_destination_elements():
        for _ in destination:
            ti.atomic_add(active_destination_elements[None], 1)

    initialize()
    scatter()
    count_active_destination_elements()
    ti.sync()

    expected = np.zeros(128, dtype=np.int32)
    expected[2 * np.arange(8)] = np.arange(1, 9, dtype=np.int32)
    expected[64 + 2 * np.arange(8)] = np.arange(101, 109, dtype=np.int32)
    np.testing.assert_array_equal(destination.to_numpy(), expected)
    assert active_destination_elements[None] == 32

    async_tile = next(
        operation
        for operation in ti.hardware.report().operations
        if operation.descriptor.operation_id == "internal.tile.async.cuda"
    )
    assert async_tile.native_facts["lowered_specializations"] == 0
    assert async_tile.native_facts["copy_sites"] == 0


@test_utils.test(arch=ti.vulkan)
def test_vulkan_passive_routes_only_admit_evaluated_provider_requirements():
    report = ti.hardware.report()
    for operation_id in (
        "runtime.buffer_commands.vulkan",
        "image.copy.vulkan",
        "raster.draw.vulkan",
    ):
        operation = next(
            operation for operation in report.operations if operation.descriptor.operation_id == operation_id
        )

        assert operation.discovery == "available"
        assert operation.enablement == "enabled"
        assert operation.selection == "eligible"
        assert operation.unavailable_reason == "none"
        assert not operation.native_facts["external_component_probed"]

    raster = next(
        operation for operation in report.operations if operation.descriptor.operation_id == "raster.draw.vulkan"
    )
    assert raster.native_facts["indirect_fixed_count"]
    assert raster.native_facts["max_draw_indirect_count"] >= 1
    assert isinstance(raster.native_facts["indirect_count_buffer"], bool)

    sampling = next(
        operation for operation in report.operations if operation.descriptor.operation_id == "sampling.texture.vulkan"
    )
    assert sampling.discovery == "present"
    assert sampling.enablement == "enabled"
    assert sampling.selection == "not_considered"
    assert sampling.unavailable_reason == "operation_requirements_not_evaluated"
    assert not sampling.native_facts["operation_requirements_evaluated"]

    ray_routes = {
        operation.descriptor.operation_id: operation
        for operation in report.operations
        if operation.descriptor.operation_id in ("ray.as_build.vulkan", "ray.as_refit.vulkan", "ray.query.batch.vulkan")
    }
    assert len(ray_routes) == 3
    assert len({operation.discovery for operation in ray_routes.values()}) == 1
    assert len({operation.selection for operation in ray_routes.values()}) == 1
    assert all(
        operation.native_facts["capability_query"] == "active_vulkan_feature_chain" for operation in ray_routes.values()
    )


@test_utils.test(
    arch=ti.vulkan,
    offline_cache=False,
    spirv_listgen_subgroup_ballot=True,
)
def test_vulkan_subgroup_ballot_listgen_opt_in_preserves_sparse_iteration():
    values = ti.field(ti.i32)
    ti.root.pointer(ti.i, 64).dense(ti.i, 4).place(values)
    result = ti.field(ti.i32, shape=())

    @ti.kernel
    def populate():
        for i in range(32):
            values[(i * 29) % 256] = i + 1

    @ti.kernel
    def reduce_active_values():
        for i in values:
            ti.atomic_add(result[None], values[i])

    populate()
    reduce_active_values()

    assert result[None] == sum(range(1, 33))


def _native_probe_payload(discovery, unavailable_reason, **overrides):
    payload = {
        "provider_id": "cublas",
        "external_component_probed": True,
        "discovery": discovery,
        "unavailable_reason": unavailable_reason,
        "provider_abi": None,
        "provider_version": None,
        "last_error": None,
        "failure_scope": None,
        "native_facts": {
            "probe_policy": "explicit_transient_load",
            "provider_enablement_changed": False,
            "provider_selection_changed": False,
        },
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    (
        "native_payload",
        "expected_discovery",
        "expected_reason",
        "expected_failure_scope",
    ),
    [
        (
            _native_probe_payload(
                "available",
                "none",
                provider_abi="cublas-dynamic-symbols-v1",
            ),
            "available",
            "none",
            None,
        ),
        (
            _native_probe_payload("missing", "external_library_not_found"),
            "missing",
            "external_library_not_found",
            None,
        ),
        (
            _native_probe_payload(
                "incompatible",
                "required_provider_symbol_missing",
                provider_abi="cublas-dynamic-symbols-v1",
                last_error="required provider symbol missing: cublasCreate_v2",
                failure_scope="provider",
            ),
            "incompatible",
            "required_provider_symbol_missing",
            "provider",
        ),
    ],
)
@pytest.mark.parametrize("provider_id", ("cublas", "cusparse", "cufft"))
def test_explicit_external_probe_normalizes_native_facts_without_enabling(
    monkeypatch,
    native_payload,
    expected_discovery,
    expected_reason,
    expected_failure_scope,
    provider_id,
):
    monkeypatch.setattr(
        _capabilities,
        "_runtime_facts",
        lambda: (False, None, {"cuda": True, "vulkan": True}),
    )
    monkeypatch.setattr(
        _capabilities,
        "_native_external_probe",
        lambda requested_provider: {
            **native_payload,
            "provider_id": requested_provider,
        },
    )

    report = ti.hardware.probe(provider_id)
    operation = next(operation for operation in report.operations if operation.descriptor.provider_id == provider_id)

    assert operation.discovery == expected_discovery
    assert operation.unavailable_reason == expected_reason
    assert operation.failure_scope == expected_failure_scope
    assert operation.enablement == "disabled"
    assert operation.selection == "not_considered"
    assert operation.native_facts["external_component_probed"] is True
    assert operation.native_facts["provider_enablement_changed"] is False
    assert operation.native_facts["provider_selection_changed"] is False
    assert report.external_components_probed is True


def test_explicit_external_probe_failures_remain_provider_scoped(monkeypatch):
    monkeypatch.setattr(
        _capabilities,
        "_runtime_facts",
        lambda: (False, None, {"cuda": True, "vulkan": True}),
    )

    def fail_probe(_provider_id):
        raise RuntimeError("isolated loader failure")

    monkeypatch.setattr(_capabilities, "_native_external_probe", fail_probe)
    report = ti.hardware.probe("cusparse")
    operation = next(operation for operation in report.operations if operation.descriptor.provider_id == "cusparse")

    assert operation.discovery == "incompatible"
    assert operation.unavailable_reason == "native_probe_failed"
    assert operation.last_error == "isolated loader failure"
    assert operation.failure_scope == "provider"
    assert operation.enablement == "disabled"
    assert operation.selection == "not_considered"


def test_optional_optix_probe_and_invalid_tiers_fail_closed(monkeypatch):
    monkeypatch.setattr(_optix, "_bundled_provider_candidates", lambda: ())
    monkeypatch.setattr(
        _capabilities,
        "_runtime_facts",
        lambda: (False, None, {"cuda": True, "vulkan": True}),
    )

    report = ti.hardware.probe("optix")
    optix_operations = tuple(
        operation for operation in report.operations if operation.descriptor.provider_id == "optix"
    )
    assert len(optix_operations) == 3
    assert all(operation.discovery == "missing" for operation in optix_operations)
    assert all(
        operation.unavailable_reason == "bundled_provider_adapter_not_installed" for operation in optix_operations
    )
    assert all(operation.enablement == "disabled" for operation in optix_operations)
    assert all(operation.selection == "not_considered" for operation in optix_operations)
    assert report.external_components_probed is False

    with pytest.raises(ValueError, match="only lazy_external"):
        ti.hardware.probe("vulkan_texture")
    with pytest.raises(ValueError, match="only lazy_external"):
        ti.hardware.probe("cub_reference")
    with pytest.raises(KeyError, match="unknown hardware provider"):
        ti.hardware.probe("missing")
    with pytest.raises(KeyError, match="unknown hardware provider"):
        ti.hardware.probe("cusolver")
    with pytest.raises(KeyError, match="unknown hardware operation"):
        ti.hardware.capability("linalg.solve.cusolver")
    with pytest.raises(ValueError, match="unsupported CUDA external provider"):
        ti_core.probe_cuda_external_library("cusolver")
    with pytest.raises(ValueError, match="unsupported CUDA external provider"):
        ti_core.cuda_external_library_status("cusolver")
    with pytest.raises(TypeError, match="nonempty string"):
        ti.hardware.probe(None)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_passive_report_on_cpu_rejects_gpu_routes_without_loading_them():
    report = ti.hardware.report()

    assert report.runtime_initialized is True
    assert report.backend in ("x64", "arm64")
    for operation in report.operations:
        assert operation.selection != "selected"
        assert operation.native_facts["probe_policy"] == "passive"
        if operation.descriptor.dependency_tier == "lazy_external":
            if operation.enablement == "enabled":
                assert operation.discovery == "available"
                assert operation.selection == "rejected"
                assert operation.unavailable_reason == "backend_not_active"
            else:
                assert operation.enablement == "disabled"
                assert operation.selection == "not_considered"

    payload = report.to_dict()
    assert payload["external_components_probed"] is False
    assert len(payload["operations"]) == len(report.operations)
