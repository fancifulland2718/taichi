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

# G1.a (2026-04-30): pointer-SNode alloc protocol fix.
# Default ON: race-correct CAS-marker-first protocol replaces the older
# atomicIAdd-first protocol. The old protocol over-consumes pool watermark
# under concurrent activation of the same slot (see
# compile_doc/SNode_Vulkan_规划.md §4.6.3).
# Default OFF would only be used as an emergency fallback if the structured
# spin loop misbehaves on a specific Vulkan driver / GPU family.
option(TI_VULKAN_POINTER_CAS_MARKER
  "Use CAS-marker-first protocol for pointer-SNode allocation on Vulkan (race-correct)" ON)
if(TI_VULKAN_POINTER_CAS_MARKER)
  target_compile_definitions(spirv_codegen PRIVATE TI_VULKAN_POINTER_CAS_MARKER=1)
endif()

# G1.b (2026-04-30): pointer-SNode freelist + recycle on deactivate.
# Default ON: alloc tries freelist pop before bumping the watermark, and
# deactivate pushes the released slot onto the freelist. Eliminates the
# pool-monotonic-growth issue (§4.4 R3) and shrinks effective working set
# under repeated activate/deactivate cycles.
# Constraint: deactivate and activate MUST NOT run concurrently in the same
# kernel (ABA on free_head is not protected); typical usage is per-frame
# deactivate_all on host followed by per-frame activate, which is naturally
# separated. See compile_doc/SNode_Vulkan_规划.md §4.6.4.
option(TI_VULKAN_POINTER_FREELIST
  "Enable freelist + recycle for pointer-SNode on Vulkan (requires CAS-marker)" ON)
if(TI_VULKAN_POINTER_FREELIST)
  if(NOT TI_VULKAN_POINTER_CAS_MARKER)
    message(FATAL_ERROR "TI_VULKAN_POINTER_FREELIST requires TI_VULKAN_POINTER_CAS_MARKER=ON")
  endif()
  target_compile_definitions(spirv_codegen PRIVATE TI_VULKAN_POINTER_FREELIST=1)
endif()

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

# G2 (2026-04-30): allow shrinking pointer-pool capacity at materialization
# time via the TI_VULKAN_POOL_FRACTION env var (default 1.0 = legacy
# worst-case reservation). When the flag is OFF the env var is ignored and
# the pool always reserves total_num_cells_from_root cells per pointer
# SNode, byte-identical to the pre-G2 layout. Combined with G1.b
# freelist recycling, lower fractions trade memory for the assumption that
# steady-state activations stay below the worst case; out-of-capacity
# activations gracefully fall back to slot=0 (silent inactive) per the
# existing cap_v guard. Default ON because the env var alone defaults to
# 1.0; users opting in must set TI_VULKAN_POOL_FRACTION explicitly.
option(TI_VULKAN_POINTER_POOL_FRACTION
  "Enable TI_VULKAN_POOL_FRACTION env var to shrink pointer-pool capacity" ON)
if(TI_VULKAN_POINTER_POOL_FRACTION)
  target_compile_definitions(spirv_codegen PRIVATE TI_VULKAN_POINTER_POOL_FRACTION=1)
endif()
