# ./taichi/codegen/spirv/CMakeLists.txt

add_library(spirv_codegen)
target_sources(spirv_codegen
  PRIVATE
    kernel_utils.cpp
    snode_struct_compiler.cpp
    spirv_codegen.cpp
    spirv_ir_builder.cpp
    spirv_types.cpp
    compiled_kernel_data.cpp
    kernel_compiler.cpp
  )

target_include_directories(spirv_codegen
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/include
    ${PROJECT_SOURCE_DIR}/external/SPIRV-Headers/include
    ${PROJECT_SOURCE_DIR}/external/SPIRV-Reflect
    ${PROJECT_SOURCE_DIR}/external/FP16/include
    ${LLVM_INCLUDE_DIRS}
  )

target_link_libraries(spirv_codegen PRIVATE taichi_util)
# NOTE: SPIRV-Tools-opt must come before SPIRV-Tools
# https://github.com/KhronosGroup/SPIRV-Tools/issues/1569#issuecomment-390250792
target_link_libraries(spirv_codegen PRIVATE SPIRV-Tools-opt ${SPIRV_TOOLS})

# B-2.b (2026-05): the four pointer-SNode policy macros below have been
# promoted to runtime fields on `CompileConfig` (vulkan_pointer_freelist /
# _ambient_zone / _cas_marker / _pool_fraction). Defaults (true/true/true/
# 1.0) match the legacy CMake-ON behavior, so the build no longer carries
# any compile-time switch for them. Override at runtime via ti.init kwargs.
# Removed:
#   - TI_VULKAN_POINTER_CAS_MARKER  (G1.a, was default ON)
#   - TI_VULKAN_POINTER_FREELIST    (G1.b, was default ON)
#   - TI_VULKAN_POINTER_POOL_FRACTION (G2,  was default ON)
#   - TI_VULKAN_POINTER_AMBIENT_ZONE  (G10-P2, was default ON)

# G4 (2026-04-30): dynamic SNode support on Vulkan.
# Default ON: enables ti.append / ti.length / ti.deactivate / struct_for on
# dynamic SNodes via a flat-array protocol (no chunked allocator). Each
# dynamic container is laid out as [data][length u32]; length is an atomic
# counter capped at num_cells_per_container by user contract. struct_for
# walks [0, length) per container.
# Default OFF would only be used to fall back to the legacy "dynamic
# unsupported on SPIR-V" behavior.
option(TI_VULKAN_DYNAMIC
  "Enable dynamic-SNode codegen + listgen on Vulkan" ON)
if(TI_VULKAN_DYNAMIC)
  target_compile_definitions(spirv_codegen PRIVATE TI_VULKAN_DYNAMIC=1)
endif()

# (B-2.b) Removed compile-time options TI_VULKAN_POINTER_POOL_FRACTION and
# TI_VULKAN_POINTER_AMBIENT_ZONE — see header comment near the top of this
# file. Both are runtime CompileConfig fields now.
