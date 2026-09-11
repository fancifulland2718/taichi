option(USE_STDCPP "Use -stdlib=libc++" OFF)
option(TI_WITH_LLVM "Build with LLVM backends" ON)              # wheel-tag: llvm
option(TI_WITH_METAL "Build with the Metal backend" ON)         # wheel-tag: mtl
option(TI_WITH_CUDA "Build with the CUDA backend" ON)           # wheel-tag: cu
option(TI_WITH_CUDA_TOOLKIT
       "Legacy switch: enable the CUDA primitive reference provider" OFF)
option(TI_WITH_CUDA_TOOLKIT_PRIMITIVE_REFERENCE
       "Build the optional CUB/CUDART primitive reference provider"
       ${TI_WITH_CUDA_TOOLKIT})
option(TI_WITH_CUPTI
       "Build the optional CUPTI/NVPerf profiler"
       OFF)
if (WIN32)
    option(TI_CUDA_CUB_SORT_DYNAMIC_CUDART
        "Link CUDA CUB sort against dynamic cudart with delay-load" ON)
else()
    option(TI_CUDA_CUB_SORT_DYNAMIC_CUDART
        "Link CUDA CUB sort against dynamic cudart" OFF)
endif()
option(TI_WITH_AMDGPU "Build with the AMDGPU backend" OFF)      # wheel-tag: amd
option(TI_WITH_OPENGL "Build with the OpenGL backend" ON)       # wheel-tag: gl
option(TI_WITH_VULKAN "Build with the Vulkan backend" OFF)      # wheel-tag: vk
option(TI_WITH_DX11 "Build with the DX11 backend" OFF)          # wheel-tag: dx11
option(TI_WITH_DX12 "Build with the DX12 backend" OFF)          # wheel-tag: dx12
option(TI_WITH_GGUI "Build with GGUI" OFF)                      # wheel-tag: ggui
option(TI_WITH_SPLIT_PYTHON_RUNTIME
       "Build taichi_python as a small shim linked against a shared native runtime"
       OFF)
set(TI_PREBUILT_PYTHON_RUNTIME_DIR ""
    CACHE PATH "Directory containing a prebuilt split Python runtime library for shim-only builds")
set(TI_PYTHON_INSTALL_PACKAGE ""
    CACHE STRING "Python package directory that receives installed Taichi native libraries")
set(TI_WITH_PREBUILT_PYTHON_RUNTIME OFF)
if(TI_WITH_SPLIT_PYTHON_RUNTIME AND TI_PREBUILT_PYTHON_RUNTIME_DIR)
    set(TI_WITH_PREBUILT_PYTHON_RUNTIME ON)
endif()

# Force symbols to be 'hidden' by default so nothing is exported from the Taichi
# library including the third-party dependencies.
# As Taichi can be used by external projects, some of the internal dependencies
# such as Vulkan, ImGui, etc. could be in conflict with the dependencies of those
# projects.
#
# Split Python runtime builds are different: taichi_python is a CPython-specific
# shim and must resolve a generated package-private C++ ABI from a separate
# platform shared library. Keep definitions available while compiling the
# closure; the final runtime link restricts the dynamic export table with a
# PE/COFF .def file or an ELF version script.
if(TI_WITH_SPLIT_PYTHON_RUNTIME)
    set(CMAKE_CXX_VISIBILITY_PRESET default)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN OFF)
else()
    set(CMAKE_CXX_VISIBILITY_PRESET hidden)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
endif()
# Suppress warnings from submodules introduced by the above symbol visibility change
set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
# Pick the install layout based on which build front-end is driving us.
#
# scikit-build-core (PEP 517 backend, used by `pip install` / `python -m
# build`) sets `SKBUILD_PROJECT_NAME` and treats `install(DESTINATION ...)`
# as wheel-relative — files installed to `taichi_forge/_lib/...` end up at
# the wheel root as `taichi_forge/_lib/...`, alongside the Python source
# from `wheel.packages = ["python/taichi_forge"]`.
#
# Legacy scikit-build (still used by `python setup.py bdist_wheel` in some
# local scripts) only defines `SKBUILD` and expects an absolute install
# path under `${CMAKE_INSTALL_PREFIX}/python/taichi_forge`.
#
# IMPORTANT: scikit-build-core ALSO sets `SKBUILD=2`, so a naive
# `if(DEFINED SKBUILD)` check matches both backends and silently picks
# the wrong (legacy) layout for scikit-build-core builds. The result is
# that `taichi_python.pyd` lands at `python/taichi_forge/_lib/core/...`
# inside the wheel — separate from the Python `taichi_forge/` package —
# and the wheel ships without `taichi_forge._lib.core`, breaking
# `import taichi_forge`. Detect scikit-build-core via its uniquely
# defined `SKBUILD_PROJECT_NAME` variable instead.
if(TI_PYTHON_INSTALL_PACKAGE)
    set(INSTALL_LIB_DIR ${TI_PYTHON_INSTALL_PACKAGE}/_lib)
elseif(DEFINED SKBUILD_PROJECT_NAME)
    # scikit-build-core: relative path lands directly at wheel root.
    string(REPLACE "_" "-" _ti_skbuild_project_name "${SKBUILD_PROJECT_NAME}")
    if(_ti_skbuild_project_name STREQUAL "taichi-forge-runtime")
        set(INSTALL_LIB_DIR taichi_forge_runtime/_lib)
    else()
        set(INSTALL_LIB_DIR taichi_forge/_lib)
    endif()
elseif(DEFINED SKBUILD AND SKBUILD)
    # Legacy scikit-build: absolute staging path.
    set(INSTALL_LIB_DIR ${CMAKE_INSTALL_PREFIX}/python/taichi_forge/_lib)
else()
    set(INSTALL_LIB_DIR ${CMAKE_INSTALL_PREFIX}/python/taichi_forge/_lib)
endif()

if (TI_WITH_AMDGPU AND TI_WITH_CUDA)
    message(WARNING "Compiling CUDA and AMDGPU backends simultaneously")
endif()

if(UNIX AND NOT APPLE)
    # Handy helper for Linux
    # https://stackoverflow.com/a/32259072/12003165
    set(LINUX TRUE)
endif()

if (APPLE)
    if (TI_WITH_CUDA)
        set(TI_WITH_CUDA OFF)
        message(WARNING "CUDA backend not supported on OS X. Setting TI_WITH_CUDA to OFF.")
    endif()
    set(TI_WITH_CUDA_TOOLKIT OFF)
    set(TI_WITH_CUDA_TOOLKIT_PRIMITIVE_REFERENCE OFF)
    set(TI_WITH_CUPTI OFF)
    if (TI_WITH_OPENGL)
        set(TI_WITH_OPENGL OFF)
        message(WARNING "OpenGL backend not supported on OS X. Setting TI_WITH_OPENGL to OFF.")
    endif()
    if (TI_WITH_AMDGPU)
        set(TI_WITH_AMDGPU OFF)
        message(WARNING "AMDGPU backend not supported on OS X. Setting TI_WITH_AMDGPU to OFF.")
    endif()
else()
    if (TI_WITH_METAL)
        set(TI_WITH_METAL OFF)
        message(WARNING "Metal backend only supported on OS X. Setting TI_WITH_METAL to OFF.")
    endif()
endif()

if (WIN32)
    if (TI_WITH_AMDGPU)
        set(TI_WITH_AMDGPU OFF)
        message(WARNING "AMDGPU backend not supported on Windows. Setting TI_WITH_AMDGPU to OFF.")
    endif()
endif()

if(TI_WITH_VULKAN)
    set(TI_WITH_GGUI ON)
endif()

if (NOT TI_WITH_PREBUILT_PYTHON_RUNTIME AND
        NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/glad/src/gl.c")
    set(TI_WITH_OPENGL OFF)
    message(WARNING "external/glad submodule not detected. Settings TI_WITH_OPENGL to OFF.")
endif()

if(NOT TI_WITH_LLVM)
    set(TI_WITH_CUDA OFF)
    set(TI_WITH_CUDA_TOOLKIT OFF)
    set(TI_WITH_CUDA_TOOLKIT_PRIMITIVE_REFERENCE OFF)
    set(TI_WITH_CUPTI OFF)
    set(TI_WITH_DX12 OFF)
endif()

if(NOT TI_WITH_CUDA)
    set(TI_WITH_CUDA_TOOLKIT OFF)
    set(TI_WITH_CUDA_TOOLKIT_PRIMITIVE_REFERENCE OFF)
    set(TI_WITH_CUPTI OFF)
endif()

# The CUDA driver and toolkit version are independent from the slim libdevice
# bitcode shipped by this package. Derive the latter from the actual asset so a
# repackaged wheel cannot silently advertise or look up a hardcoded version.
# This also runs for a prebuilt-runtime shim: CMake configures version.h before
# compiling the shim, even though that wheel does not install runtime assets.
file(GLOB _ti_cuda_libdevice_files
    RELATIVE "${CMAKE_SOURCE_DIR}/external/cuda_libdevice"
    "${CMAKE_SOURCE_DIR}/external/cuda_libdevice/slim_libdevice.*.bc")
list(LENGTH _ti_cuda_libdevice_files _ti_cuda_libdevice_count)
if(NOT _ti_cuda_libdevice_count EQUAL 1)
    message(FATAL_ERROR
        "Expected exactly one slim_libdevice.<major>.bc asset, found: "
        "${_ti_cuda_libdevice_files}")
endif()
list(GET _ti_cuda_libdevice_files 0 _ti_cuda_libdevice_filename)
string(REGEX MATCH "^slim_libdevice\\.([0-9]+)\\.bc$"
    _ti_cuda_libdevice_match "${_ti_cuda_libdevice_filename}")
if(NOT _ti_cuda_libdevice_match)
    message(FATAL_ERROR
        "Unsupported slim libdevice filename: ${_ti_cuda_libdevice_filename}")
endif()
set(TI_CUDA_LIBDEVICE_MAJOR "${CMAKE_MATCH_1}")
# Retain the historical public string format while deriving it solely from the
# bundled major-versioned asset.
set(TI_CUDA_LIBDEVICE_VERSION "${TI_CUDA_LIBDEVICE_MAJOR}.0")

if(NOT TI_WITH_PREBUILT_PYTHON_RUNTIME)
file(GLOB TAICHI_CORE_SOURCE CONFIGURE_DEPENDS
    "taichi/analysis/*.cpp" "taichi/analysis/*.h"
    "taichi/ir/*"
    "taichi/jit/*"
    "taichi/math/*"
    "taichi/program/*"
    "taichi/struct/*"
    "taichi/system/*"
    "taichi/transforms/*"
    "taichi/aot/*.cpp" "taichi/aot/*.h"
    "taichi/platform/cuda/*" "taichi/platform/amdgpu/*"
    "taichi/platform/mac/*" "taichi/platform/windows/*"
    "taichi/codegen/*.cpp" "taichi/codegen/*.h"
    "taichi/runtime/*.h" "taichi/runtime/*.cpp"
)

if(TI_WITH_LLVM)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_WITH_LLVM")
endif()

if (TI_WITH_CUDA)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_WITH_CUDA")
  file(GLOB TAICHI_CUDA_RUNTIME_SOURCE "taichi/runtime/cuda/runtime.cpp")
  list(APPEND TAICHI_CORE_SOURCE ${TAICHI_CUDA_RUNTIME_SOURCE})
endif()

if (TI_WITH_AMDGPU)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_WITH_AMDGPU")
  file(GLOB TAICHI_AMDGPU_RUNTIME_SOURCE "taichi/runtime/amdgpu/runtime.cpp")
  list(APPEND TAIHI_CORE_SOURCE ${TAICHI_AMDGPU_RUNTIME_SOURCE})
endif()

if (TI_WITH_DX12)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_WITH_DX12")
endif()

if (TI_WITH_METAL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_WITH_METAL")
endif()

if (TI_WITH_OPENGL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_WITH_OPENGL")
endif()

if (TI_WITH_DX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_WITH_DX11")
endif()

if (TI_WITH_VULKAN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_WITH_VULKAN")
endif ()

add_subdirectory(taichi/rhi)

set(CORE_LIBRARY_NAME taichi_core)
add_library(${CORE_LIBRARY_NAME} OBJECT ${TAICHI_CORE_SOURCE})
include(cmake/TaichiNvtx.cmake)
if(TI_WITH_SPLIT_PYTHON_RUNTIME AND NOT TI_WITH_PREBUILT_PYTHON_RUNTIME)
    target_compile_definitions(${CORE_LIBRARY_NAME}
        PRIVATE TI_WITH_SPLIT_PYTHON_RUNTIME TI_BUILDING_PYTHON_RUNTIME)
endif()

if (TI_WITH_VULKAN)
    set(TI_VULKAN_SORT_SHADER_SOURCE_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/taichi/program/vulkan_sort_shaders")
    set(TI_VULKAN_SORT_GENERATED_INCLUDE_DIR
        "${CMAKE_CURRENT_BINARY_DIR}/generated/vulkan_sort")
    set(TI_VULKAN_SORT_GENERATED_SHADER_DIR
        "${TI_VULKAN_SORT_GENERATED_INCLUDE_DIR}/taichi/program/vulkan_sort_shaders")
    set(TI_VULKAN_SORT_GENERATED_SPV_HEADERS)
    set(TI_VULKAN_SORT_PREGENERATED_SPV_HEADERS)

    find_program(TI_GLSLC_EXECUTABLE
        NAMES glslc glslc.exe
        HINTS
            "$ENV{VULKAN_SDK}/bin"
            "$ENV{VULKAN_SDK}/Bin"
            "$ENV{VK_SDK_PATH}/bin"
            "$ENV{VK_SDK_PATH}/Bin"
            "$ENV{VK_LAYER_PATH}"
    )
    if (TI_GLSLC_EXECUTABLE)
        message(STATUS "Vulkan sort SPIR-V headers will be generated with ${TI_GLSLC_EXECUTABLE}")
    else()
        message(STATUS "glslc not found; checking checked-in Vulkan sort SPIR-V headers from ${TI_VULKAN_SORT_SHADER_SOURCE_DIR}")
    endif()

    macro(ti_vulkan_builtin_shader source output target_env)
        set(output_path "${TI_VULKAN_SORT_GENERATED_SHADER_DIR}/${output}")
        if (TI_GLSLC_EXECUTABLE)
            add_custom_command(
                OUTPUT "${output_path}"
                COMMAND ${CMAKE_COMMAND} -E make_directory
                        "${TI_VULKAN_SORT_GENERATED_SHADER_DIR}"
                COMMAND "${TI_GLSLC_EXECUTABLE}"
                        --target-env=${target_env}
                        ${ARGN}
                        -mfmt=c
                        "${TI_VULKAN_SORT_SHADER_SOURCE_DIR}/${source}"
                        -o "${output_path}"
                DEPENDS "${TI_VULKAN_SORT_SHADER_SOURCE_DIR}/${source}"
                VERBATIM
            )
            list(APPEND TI_VULKAN_SORT_GENERATED_SPV_HEADERS "${output_path}")
        else()
            set(pregenerated_path "${TI_VULKAN_SORT_SHADER_SOURCE_DIR}/${output}")
            if (NOT EXISTS "${pregenerated_path}")
                message(FATAL_ERROR
                    "TI_WITH_VULKAN=ON requires either glslc or the checked-in Vulkan sort SPIR-V header: ${pregenerated_path}")
            endif()
            list(APPEND TI_VULKAN_SORT_PREGENERATED_SPV_HEADERS "${pregenerated_path}")
        endif()
    endmacro()

    macro(ti_vulkan_sort_shader source output)
        ti_vulkan_builtin_shader(${source} ${output} vulkan1.1 ${ARGN})
    endmacro()

    ti_vulkan_sort_shader(init_i32.comp init_i32.comp.spv.h)
    ti_vulkan_sort_shader(copy_i32.comp copy_i32.comp.spv.h)
    ti_vulkan_sort_shader(sort_init_key_index.comp sort_init_u32_index.comp.spv.h "-DKEY_KIND=0")
    ti_vulkan_sort_shader(sort_init_key_index.comp sort_init_i32_index.comp.spv.h "-DKEY_KIND=1")
    ti_vulkan_sort_shader(sort_init_key_index.comp sort_init_f32_index.comp.spv.h "-DKEY_KIND=2")
    ti_vulkan_sort_shader(sort_init_key_index.comp sort_init_u64_index.comp.spv.h "-DKEY_KIND=3")
    ti_vulkan_sort_shader(sort_init_key_index.comp sort_init_i64_index.comp.spv.h "-DKEY_KIND=4")
    ti_vulkan_sort_shader(sort_init_key_index.comp sort_init_f64_index.comp.spv.h "-DKEY_KIND=5")
    ti_vulkan_sort_shader(gather_u32_by_u32.comp gather_u32_by_u32.comp.spv.h)
    ti_vulkan_sort_shader(prefix_block.comp prefix_block.comp.spv.h)
    ti_vulkan_sort_shader(prefix_chunks.comp prefix_chunks.comp.spv.h)
    ti_vulkan_sort_shader(prefix_single_chunk.comp prefix_single_chunk.comp.spv.h)
    ti_vulkan_sort_shader(scan_i32_block.comp scan_i32_block.comp.spv.h "-DVALUE_KIND=0")
    ti_vulkan_sort_shader(scan_i32_block.comp scan_i32_block_strided.comp.spv.h "-DVALUE_KIND=0" "-DSTRIDED_SOURCE")
    ti_vulkan_sort_shader(scan_i32_block.comp scan_i32_block_reverse.comp.spv.h "-DVALUE_KIND=0" "-DREVERSE_SCAN")
    ti_vulkan_sort_shader(scan_i32_block.comp scan_i32_block_strided_reverse.comp.spv.h "-DVALUE_KIND=0" "-DSTRIDED_SOURCE" "-DREVERSE_SCAN")
    ti_vulkan_sort_shader(scan_i32_block_subgroup.comp scan_i32_block_subgroup.comp.spv.h "-DVALUE_KIND=0")
    ti_vulkan_sort_shader(scan_i32_add.comp scan_i32_add.comp.spv.h "-DVALUE_KIND=0")
    ti_vulkan_sort_shader(scan_i32_add.comp scan_i32_add_strided.comp.spv.h "-DVALUE_KIND=0" "-DSTRIDED_SOURCE")
    ti_vulkan_sort_shader(scan_i32_add.comp scan_i32_add_reverse.comp.spv.h "-DVALUE_KIND=0" "-DREVERSE_SCAN")
    ti_vulkan_sort_shader(scan_i32_add.comp scan_i32_add_strided_reverse.comp.spv.h "-DVALUE_KIND=0" "-DSTRIDED_SOURCE" "-DREVERSE_SCAN")
    ti_vulkan_sort_shader(scan_i32_small_subgroup.comp scan_i32_small_subgroup.comp.spv.h "-DVALUE_KIND=0")
    ti_vulkan_sort_shader(scan_i32_small_subgroup.comp scan_i32_small_subgroup_strided.comp.spv.h "-DVALUE_KIND=0" "-DSTRIDED_SOURCE")
    ti_vulkan_sort_shader(scan_i32_small_subgroup.comp scan_i32_small_subgroup_reverse.comp.spv.h "-DVALUE_KIND=0" "-DREVERSE_SCAN")
    ti_vulkan_sort_shader(scan_i32_small_subgroup.comp scan_i32_small_subgroup_strided_reverse.comp.spv.h "-DVALUE_KIND=0" "-DSTRIDED_SOURCE" "-DREVERSE_SCAN")
    ti_vulkan_sort_shader(scan_i32_block.comp scan_f32_block.comp.spv.h "-DVALUE_KIND=1")
    ti_vulkan_sort_shader(scan_i32_block.comp scan_f32_block_strided.comp.spv.h "-DVALUE_KIND=1" "-DSTRIDED_SOURCE")
    ti_vulkan_sort_shader(scan_i32_block.comp scan_f32_block_reverse.comp.spv.h "-DVALUE_KIND=1" "-DREVERSE_SCAN")
    ti_vulkan_sort_shader(scan_i32_block.comp scan_f32_block_strided_reverse.comp.spv.h "-DVALUE_KIND=1" "-DSTRIDED_SOURCE" "-DREVERSE_SCAN")
    ti_vulkan_sort_shader(scan_i32_block_subgroup.comp scan_f32_block_subgroup.comp.spv.h "-DVALUE_KIND=1")
    ti_vulkan_sort_shader(scan_i32_add.comp scan_f32_add.comp.spv.h "-DVALUE_KIND=1")
    ti_vulkan_sort_shader(scan_i32_add.comp scan_f32_add_strided.comp.spv.h "-DVALUE_KIND=1" "-DSTRIDED_SOURCE")
    ti_vulkan_sort_shader(scan_i32_add.comp scan_f32_add_reverse.comp.spv.h "-DVALUE_KIND=1" "-DREVERSE_SCAN")
    ti_vulkan_sort_shader(scan_i32_add.comp scan_f32_add_strided_reverse.comp.spv.h "-DVALUE_KIND=1" "-DSTRIDED_SOURCE" "-DREVERSE_SCAN")
    ti_vulkan_sort_shader(scan_i32_small_subgroup.comp scan_f32_small_subgroup.comp.spv.h "-DVALUE_KIND=1")
    ti_vulkan_sort_shader(scan_i32_small_subgroup.comp scan_f32_small_subgroup_strided.comp.spv.h "-DVALUE_KIND=1" "-DSTRIDED_SOURCE")
    ti_vulkan_sort_shader(scan_i32_small_subgroup.comp scan_f32_small_subgroup_reverse.comp.spv.h "-DVALUE_KIND=1" "-DREVERSE_SCAN")
    ti_vulkan_sort_shader(scan_i32_small_subgroup.comp scan_f32_small_subgroup_strided_reverse.comp.spv.h "-DVALUE_KIND=1" "-DSTRIDED_SOURCE" "-DREVERSE_SCAN")
    ti_vulkan_sort_shader(scan_i32_block.comp scan_u32_block.comp.spv.h "-DVALUE_KIND=2")
    ti_vulkan_sort_shader(scan_i32_block.comp scan_u32_block_strided.comp.spv.h "-DVALUE_KIND=2" "-DSTRIDED_SOURCE")
    ti_vulkan_sort_shader(scan_i32_block.comp scan_u32_block_reverse.comp.spv.h "-DVALUE_KIND=2" "-DREVERSE_SCAN")
    ti_vulkan_sort_shader(scan_i32_block.comp scan_u32_block_strided_reverse.comp.spv.h "-DVALUE_KIND=2" "-DSTRIDED_SOURCE" "-DREVERSE_SCAN")
    ti_vulkan_sort_shader(scan_i32_block_subgroup.comp scan_u32_block_subgroup.comp.spv.h "-DVALUE_KIND=2")
    ti_vulkan_sort_shader(scan_i32_add.comp scan_u32_add.comp.spv.h "-DVALUE_KIND=2")
    ti_vulkan_sort_shader(scan_i32_add.comp scan_u32_add_strided.comp.spv.h "-DVALUE_KIND=2" "-DSTRIDED_SOURCE")
    ti_vulkan_sort_shader(scan_i32_add.comp scan_u32_add_reverse.comp.spv.h "-DVALUE_KIND=2" "-DREVERSE_SCAN")
    ti_vulkan_sort_shader(scan_i32_add.comp scan_u32_add_strided_reverse.comp.spv.h "-DVALUE_KIND=2" "-DSTRIDED_SOURCE" "-DREVERSE_SCAN")
    ti_vulkan_sort_shader(scan_i32_small_subgroup.comp scan_u32_small_subgroup.comp.spv.h "-DVALUE_KIND=2")
    ti_vulkan_sort_shader(scan_i32_small_subgroup.comp scan_u32_small_subgroup_strided.comp.spv.h "-DVALUE_KIND=2" "-DSTRIDED_SOURCE")
    ti_vulkan_sort_shader(scan_i32_small_subgroup.comp scan_u32_small_subgroup_reverse.comp.spv.h "-DVALUE_KIND=2" "-DREVERSE_SCAN")
    ti_vulkan_sort_shader(scan_i32_small_subgroup.comp scan_u32_small_subgroup_strided_reverse.comp.spv.h "-DVALUE_KIND=2" "-DSTRIDED_SOURCE" "-DREVERSE_SCAN")
    ti_vulkan_sort_shader(scan_i32_block.comp scan_u64_block.comp.spv.h "-DVALUE_KIND=3")
    ti_vulkan_sort_shader(scan_i32_block.comp scan_u64_block_strided.comp.spv.h "-DVALUE_KIND=3" "-DSTRIDED_SOURCE")
    ti_vulkan_sort_shader(scan_i32_block.comp scan_u64_block_reverse.comp.spv.h "-DVALUE_KIND=3" "-DREVERSE_SCAN")
    ti_vulkan_sort_shader(scan_i32_block.comp scan_u64_block_strided_reverse.comp.spv.h "-DVALUE_KIND=3" "-DSTRIDED_SOURCE" "-DREVERSE_SCAN")
    ti_vulkan_sort_shader(scan_i32_add.comp scan_u64_add.comp.spv.h "-DVALUE_KIND=3")
    ti_vulkan_sort_shader(scan_i32_add.comp scan_u64_add_strided.comp.spv.h "-DVALUE_KIND=3" "-DSTRIDED_SOURCE")
    ti_vulkan_sort_shader(scan_i32_add.comp scan_u64_add_reverse.comp.spv.h "-DVALUE_KIND=3" "-DREVERSE_SCAN")
    ti_vulkan_sort_shader(scan_i32_add.comp scan_u64_add_strided_reverse.comp.spv.h "-DVALUE_KIND=3" "-DSTRIDED_SOURCE" "-DREVERSE_SCAN")
    ti_vulkan_sort_shader(scan_i32_block.comp scan_i64_block.comp.spv.h "-DVALUE_KIND=4")
    ti_vulkan_sort_shader(scan_i32_block.comp scan_i64_block_strided.comp.spv.h "-DVALUE_KIND=4" "-DSTRIDED_SOURCE")
    ti_vulkan_sort_shader(scan_i32_block.comp scan_i64_block_reverse.comp.spv.h "-DVALUE_KIND=4" "-DREVERSE_SCAN")
    ti_vulkan_sort_shader(scan_i32_block.comp scan_i64_block_strided_reverse.comp.spv.h "-DVALUE_KIND=4" "-DSTRIDED_SOURCE" "-DREVERSE_SCAN")
    ti_vulkan_sort_shader(scan_i32_add.comp scan_i64_add.comp.spv.h "-DVALUE_KIND=4")
    ti_vulkan_sort_shader(scan_i32_add.comp scan_i64_add_strided.comp.spv.h "-DVALUE_KIND=4" "-DSTRIDED_SOURCE")
    ti_vulkan_sort_shader(scan_i32_add.comp scan_i64_add_reverse.comp.spv.h "-DVALUE_KIND=4" "-DREVERSE_SCAN")
    ti_vulkan_sort_shader(scan_i32_add.comp scan_i64_add_strided_reverse.comp.spv.h "-DVALUE_KIND=4" "-DSTRIDED_SOURCE" "-DREVERSE_SCAN")
    ti_vulkan_sort_shader(scan_i32_block.comp scan_f64_block.comp.spv.h "-DVALUE_KIND=5")
    ti_vulkan_sort_shader(scan_i32_block.comp scan_f64_block_strided.comp.spv.h "-DVALUE_KIND=5" "-DSTRIDED_SOURCE")
    ti_vulkan_sort_shader(scan_i32_block.comp scan_f64_block_reverse.comp.spv.h "-DVALUE_KIND=5" "-DREVERSE_SCAN")
    ti_vulkan_sort_shader(scan_i32_block.comp scan_f64_block_strided_reverse.comp.spv.h "-DVALUE_KIND=5" "-DSTRIDED_SOURCE" "-DREVERSE_SCAN")
    ti_vulkan_sort_shader(scan_i32_add.comp scan_f64_add.comp.spv.h "-DVALUE_KIND=5")
    ti_vulkan_sort_shader(scan_i32_add.comp scan_f64_add_strided.comp.spv.h "-DVALUE_KIND=5" "-DSTRIDED_SOURCE")
    ti_vulkan_sort_shader(scan_i32_add.comp scan_f64_add_reverse.comp.spv.h "-DVALUE_KIND=5" "-DREVERSE_SCAN")
    ti_vulkan_sort_shader(scan_i32_add.comp scan_f64_add_strided_reverse.comp.spv.h "-DVALUE_KIND=5" "-DSTRIDED_SOURCE" "-DREVERSE_SCAN")
    ti_vulkan_sort_shader(compact_i32_flags.comp compact_i32_flags.comp.spv.h)
    ti_vulkan_sort_shader(compact_i32_scatter.comp compact_i32_scatter.comp.spv.h)
    ti_vulkan_sort_shader(compact_i32_scatter.comp compact_i32_scatter_packet.comp.spv.h "-DDISPATCH_PACKET")
    ti_vulkan_sort_shader(histogram_i32_clear.comp histogram_i32_clear.comp.spv.h)
    ti_vulkan_sort_shader(histogram_i32_count_direct.comp histogram_i32_count_direct.comp.spv.h "-DVALUE_KIND=0")
    ti_vulkan_sort_shader(histogram_i32_count_private.comp histogram_i32_count_private.comp.spv.h "-DVALUE_KIND=0")
    ti_vulkan_sort_shader(histogram_i32_count_private_shared.comp histogram_i32_count_private_shared.comp.spv.h "-DVALUE_KIND=0")
    ti_vulkan_sort_shader(histogram_i32_reduce_private.comp histogram_i32_reduce_private.comp.spv.h)
    ti_vulkan_sort_shader(histogram_i32_single_shared.comp histogram_i32_single_shared.comp.spv.h "-DVALUE_KIND=0")
    ti_vulkan_sort_shader(histogram_i32_count_direct.comp histogram_u32_count_direct.comp.spv.h "-DVALUE_KIND=2")
    ti_vulkan_sort_shader(histogram_i32_count_private.comp histogram_u32_count_private.comp.spv.h "-DVALUE_KIND=2")
    ti_vulkan_sort_shader(histogram_i32_count_private_shared.comp histogram_u32_count_private_shared.comp.spv.h "-DVALUE_KIND=2")
    ti_vulkan_sort_shader(histogram_i32_single_shared.comp histogram_u32_single_shared.comp.spv.h "-DVALUE_KIND=2")
    ti_vulkan_sort_shader(histogram_i32_clear.comp histogram_i64_clear.comp.spv.h "-DBIN_KIND=4")
    ti_vulkan_sort_shader(histogram_i32_count_direct.comp histogram_i32_i64_count_direct.comp.spv.h "-DVALUE_KIND=0" "-DBIN_KIND=4")
    ti_vulkan_sort_shader(histogram_i32_count_private.comp histogram_i32_i64_count_private.comp.spv.h "-DVALUE_KIND=0" "-DBIN_KIND=4")
    ti_vulkan_sort_shader(histogram_i32_count_private_shared.comp histogram_i32_i64_count_private_shared.comp.spv.h "-DVALUE_KIND=0" "-DBIN_KIND=4")
    ti_vulkan_sort_shader(histogram_i32_reduce_private.comp histogram_i64_reduce_private.comp.spv.h "-DBIN_KIND=4")
    ti_vulkan_sort_shader(histogram_i32_count_direct.comp histogram_u32_i64_count_direct.comp.spv.h "-DVALUE_KIND=2" "-DBIN_KIND=4")
    ti_vulkan_sort_shader(histogram_i32_count_private.comp histogram_u32_i64_count_private.comp.spv.h "-DVALUE_KIND=2" "-DBIN_KIND=4")
    ti_vulkan_sort_shader(histogram_i32_count_private_shared.comp histogram_u32_i64_count_private_shared.comp.spv.h "-DVALUE_KIND=2" "-DBIN_KIND=4")
    ti_vulkan_sort_shader(transform_i32_affine.comp transform_i32_affine.comp.spv.h)
    ti_vulkan_sort_shader(transform_i32_affine_dense.comp
        transform_i32_affine_dense.comp.spv.h)
    ti_vulkan_sort_shader(transform_f32_affine.comp transform_f32_affine.comp.spv.h)
    ti_vulkan_sort_shader(transform_indexed_affine.comp transform_indexed_i32_affine.comp.spv.h "-DVALUE_KIND=0")
    ti_vulkan_sort_shader(transform_indexed_affine.comp transform_indexed_f32_affine.comp.spv.h "-DVALUE_KIND=1")
    ti_vulkan_sort_shader(transform_u64_affine.comp transform_u64_affine.comp.spv.h)
    ti_vulkan_sort_shader(transform_f64_affine.comp transform_f64_affine.comp.spv.h)
    ti_vulkan_sort_shader(check_count.comp check_count_i32.comp.spv.h "-DVALUE_KIND=0")
    ti_vulkan_sort_shader(check_count.comp check_count_f32.comp.spv.h "-DVALUE_KIND=1")
    ti_vulkan_sort_shader(check_count.comp check_count_u32.comp.spv.h "-DVALUE_KIND=2")
    ti_vulkan_sort_shader(check_count.comp check_count_u64.comp.spv.h "-DVALUE_KIND=3")
    ti_vulkan_sort_shader(check_count.comp check_count_i64.comp.spv.h "-DVALUE_KIND=4")
    ti_vulkan_sort_shader(check_count.comp check_count_f64.comp.spv.h "-DVALUE_KIND=5")
    ti_vulkan_sort_shader(metric_reduce_f32.comp metric_reduce_f32.comp.spv.h)
    ti_vulkan_sort_shader(csr_spmv_f32.comp csr_spmv_f32.comp.spv.h)
    ti_vulkan_sort_shader(bsr_spmv_f32.comp bsr_spmv_f32.comp.spv.h)
    ti_vulkan_sort_shader(sparse_axpy_f32.comp sparse_axpy_f32.comp.spv.h)
    ti_vulkan_sort_shader(sparse_diagonal_apply_f32.comp sparse_diagonal_apply_f32.comp.spv.h)
    ti_vulkan_sort_shader(sparse_diagonal_refresh_f32.comp sparse_diagonal_refresh_f32.comp.spv.h)
    ti_vulkan_sort_shader(sparse_block_cholesky_refresh_f32.comp sparse_block_cholesky_refresh_f32.comp.spv.h)
    ti_vulkan_sort_shader(sparse_block_cholesky_apply_f32.comp sparse_block_cholesky_apply_f32.comp.spv.h)
    ti_vulkan_sort_shader(sparse_dot_partial_f32.comp sparse_dot_partial_f32.comp.spv.h)
    ti_vulkan_sort_shader(sparse_sqrt_scalar_f32.comp sparse_sqrt_scalar_f32.comp.spv.h)
    ti_vulkan_sort_shader(sparse_scalar_divide_f32.comp sparse_scalar_divide_f32.comp.spv.h)
    ti_vulkan_sort_shader(sparse_cg_update_f32.comp sparse_cg_update_f32.comp.spv.h)
    ti_vulkan_sort_shader(sparse_cg_direction_f32.comp sparse_cg_direction_f32.comp.spv.h)
    ti_vulkan_sort_shader(sparse_convergence_f32.comp sparse_convergence_f32.comp.spv.h)
    ti_vulkan_sort_shader(sparse_minres_scalar_f32.comp sparse_minres_scalar_f32.comp.spv.h)
    ti_vulkan_sort_shader(sparse_minres_vector_state_f32.comp sparse_minres_vector_state_f32.comp.spv.h)
    ti_vulkan_sort_shader(sparse_minres_commit_f32.comp sparse_minres_commit_f32.comp.spv.h)
    ti_vulkan_sort_shader(sparse_bicgstab_scalar_f32.comp sparse_bicgstab_scalar_f32.comp.spv.h)
    ti_vulkan_sort_shader(sparse_bicgstab_direction_f32.comp sparse_bicgstab_direction_f32.comp.spv.h)
    ti_vulkan_sort_shader(sparse_bicgstab_intermediate_f32.comp sparse_bicgstab_intermediate_f32.comp.spv.h)
    ti_vulkan_sort_shader(sparse_bicgstab_commit_f32.comp sparse_bicgstab_commit_f32.comp.spv.h)
    ti_vulkan_sort_shader(sparse_bicgstab_reconcile_f32.comp sparse_bicgstab_reconcile_f32.comp.spv.h)
    ti_vulkan_sort_shader(sparse_gmres_multi_dot_partial_f32.comp sparse_gmres_multi_dot_partial_f32.comp.spv.h)
    ti_vulkan_sort_shader(sparse_gmres_multi_dot_final_f32.comp sparse_gmres_multi_dot_final_f32.comp.spv.h)
    ti_vulkan_sort_shader(sparse_gmres_projection_f32.comp sparse_gmres_projection_f32.comp.spv.h)
    ti_vulkan_sort_shader(sparse_gmres_basis_f32.comp sparse_gmres_basis_f32.comp.spv.h)
    ti_vulkan_sort_shader(sparse_gmres_combine_f32.comp sparse_gmres_combine_f32.comp.spv.h)
    ti_vulkan_sort_shader(sparse_gmres_scalar_f32.comp sparse_gmres_scalar_f32.comp.spv.h)
    ti_vulkan_sort_shader(sparse_triplet_pack_f32.comp sparse_triplet_pack_f32.comp.spv.h)
    ti_vulkan_sort_shader(sparse_triplet_pack_packed_f32.comp sparse_triplet_pack_packed_f32.comp.spv.h)
    ti_vulkan_sort_shader(sparse_segment_flags_u64.comp sparse_segment_flags_u64.comp.spv.h)
    ti_vulkan_sort_shader(sparse_segment_scatter_u64.comp sparse_segment_scatter_u64.comp.spv.h)
    ti_vulkan_sort_shader(sparse_segment_reduce_f32.comp sparse_segment_reduce_f32.comp.spv.h)
    ti_vulkan_sort_shader(sparse_csr_finalize_u64.comp sparse_csr_finalize_u64.comp.spv.h)
    ti_vulkan_sort_shader(sparse_assembly_finalize_control.comp sparse_assembly_finalize_control.comp.spv.h)
    ti_vulkan_sort_shader(add_merge_i32.comp add_merge_i32.comp.spv.h "-DVALUE_KIND=0")
    ti_vulkan_sort_shader(add_merge_i32.comp add_merge_f32.comp.spv.h "-DVALUE_KIND=1")
    ti_vulkan_sort_shader(add_merge_i32.comp add_merge_u32.comp.spv.h "-DVALUE_KIND=2")
    ti_vulkan_sort_shader(add_merge_i32.comp add_merge_u64.comp.spv.h "-DVALUE_KIND=3")
    ti_vulkan_sort_shader(add_merge_i32.comp add_merge_i64.comp.spv.h "-DVALUE_KIND=4")
    ti_vulkan_sort_shader(add_merge_i32.comp add_merge_f64.comp.spv.h "-DVALUE_KIND=5")
    ti_vulkan_sort_shader(gather_u32_by_i32.comp gather_u32_by_i32.comp.spv.h)
    ti_vulkan_sort_shader(scatter_u32_by_i32.comp scatter_u32_by_i32.comp.spv.h)
    ti_vulkan_sort_shader(indexed_copy_u32_by_i32.comp scatter_dense_u32_by_i32.comp.spv.h "-DSCATTER_OP=1")
    ti_vulkan_sort_shader(indexed_copy_strided_u32_by_i32.comp gather_strided_u32_by_i32.comp.spv.h)
    ti_vulkan_sort_shader(indexed_copy_strided_u32_by_i32.comp scatter_strided_u32_by_i32.comp.spv.h "-DSCATTER_OP=1")
    set(vulkan_native_value_names i32 f32 u32 u64 i64 f64)
    set(vulkan_native_value_kinds 0 1 2 3 4 5)
    foreach(value_index RANGE 0 5)
        list(GET vulkan_native_value_names ${value_index} value_name)
        list(GET vulkan_native_value_kinds ${value_index} value_kind)
        ti_vulkan_sort_shader(scatter_add_i32_by_i32.comp
            "scatter_add_${value_name}_by_i32.comp.spv.h" "-DVALUE_KIND=${value_kind}")
        ti_vulkan_sort_shader(scatter_add_i32_by_i32.comp
            "scatter_add_${value_name}_by_i32_strided.comp.spv.h"
            "-DVALUE_KIND=${value_kind}" "-DSTRIDED_SOURCE=1")
        ti_vulkan_sort_shader(scatter_add_i32_by_i32.comp
            "scatter_add_${value_name}_by_i32_packed.comp.spv.h"
            "-DVALUE_KIND=${value_kind}" "-DPACKED_SOURCE=1")
    endforeach()
    ti_vulkan_sort_shader(bucket_clear_i32.comp bucket_clear_i32.comp.spv.h)
    ti_vulkan_sort_shader(bucket_count_i32.comp bucket_count_i32.comp.spv.h)
    ti_vulkan_sort_shader(bucket_count_private_shared_i32.comp bucket_count_private_shared_i32.comp.spv.h)
    ti_vulkan_sort_shader(bucket_prefix_i32.comp bucket_prefix_i32.comp.spv.h)
    ti_vulkan_sort_shader(bucket_prefix_chunks_i32.comp bucket_prefix_chunks_i32.comp.spv.h)
    ti_vulkan_sort_shader(bucket_scatter_i32.comp bucket_scatter_i32.comp.spv.h "-DVALUE_KIND=0")
    ti_vulkan_sort_shader(bucket_scatter_i32.comp bucket_scatter_f32.comp.spv.h "-DVALUE_KIND=1")
    ti_vulkan_sort_shader(bucket_scatter_i32.comp bucket_scatter_u32.comp.spv.h "-DVALUE_KIND=2")
    ti_vulkan_sort_shader(bucket_scatter_i32.comp bucket_scatter_raw64.comp.spv.h "-DVALUE_KIND=6")
    ti_vulkan_sort_shader(bucket_scatter_i32.comp bucket_scatter_raw_words.comp.spv.h "-DVALUE_KIND=7")
    ti_vulkan_sort_shader(bucket_scatter_private_shared_i32.comp bucket_scatter_private_shared_i32.comp.spv.h "-DVALUE_KIND=0")
    ti_vulkan_sort_shader(bucket_scatter_private_shared_i32.comp bucket_scatter_private_shared_f32.comp.spv.h "-DVALUE_KIND=1")
    ti_vulkan_sort_shader(bucket_scatter_private_shared_i32.comp bucket_scatter_private_shared_u32.comp.spv.h "-DVALUE_KIND=2")
    ti_vulkan_sort_shader(bucket_scatter_private_shared_i32.comp bucket_scatter_private_shared_raw64.comp.spv.h "-DVALUE_KIND=6")
    ti_vulkan_sort_shader(bucket_scatter_private_shared_i32.comp bucket_scatter_private_shared_raw_words.comp.spv.h "-DVALUE_KIND=7")
    ti_vulkan_sort_shader(grouped_reduce_zero_i32.comp grouped_reduce_zero_i32.comp.spv.h "-DVALUE_KIND=0")
    ti_vulkan_sort_shader(grouped_reduce_zero_i32.comp grouped_reduce_zero_f32.comp.spv.h "-DVALUE_KIND=1")
    ti_vulkan_sort_shader(grouped_reduce_zero_i32.comp grouped_reduce_zero_u32.comp.spv.h "-DVALUE_KIND=2")
    ti_vulkan_sort_shader(grouped_reduce_zero_i32.comp grouped_reduce_zero_u64.comp.spv.h "-DVALUE_KIND=3")
    ti_vulkan_sort_shader(grouped_reduce_zero_i32.comp grouped_reduce_zero_i64.comp.spv.h "-DVALUE_KIND=4")
    ti_vulkan_sort_shader(grouped_reduce_zero_i32.comp grouped_reduce_zero_f64.comp.spv.h "-DVALUE_KIND=5")
    ti_vulkan_sort_shader(grouped_reduce_zero_i32.comp grouped_reduce_zero_i32_strided.comp.spv.h "-DVALUE_KIND=0" "-DSTRIDED_DEST=1")
    ti_vulkan_sort_shader(grouped_reduce_zero_i32.comp grouped_reduce_zero_f32_strided.comp.spv.h "-DVALUE_KIND=1" "-DSTRIDED_DEST=1")
    ti_vulkan_sort_shader(grouped_reduce_zero_i32.comp grouped_reduce_zero_u32_strided.comp.spv.h "-DVALUE_KIND=2" "-DSTRIDED_DEST=1")
    ti_vulkan_sort_shader(grouped_reduce_zero_i32.comp grouped_reduce_zero_u64_strided.comp.spv.h "-DVALUE_KIND=3" "-DSTRIDED_DEST=1")
    ti_vulkan_sort_shader(grouped_reduce_zero_i32.comp grouped_reduce_zero_i64_strided.comp.spv.h "-DVALUE_KIND=4" "-DSTRIDED_DEST=1")
    ti_vulkan_sort_shader(grouped_reduce_zero_i32.comp grouped_reduce_zero_f64_strided.comp.spv.h "-DVALUE_KIND=5" "-DSTRIDED_DEST=1")
    ti_vulkan_sort_shader(grouped_reduce_atomic_sum_i32.comp grouped_reduce_atomic_sum_i32.comp.spv.h "-DVALUE_KIND=0")
    ti_vulkan_sort_shader(grouped_reduce_atomic_sum_i32.comp grouped_reduce_atomic_sum_f32.comp.spv.h "-DVALUE_KIND=1")
    ti_vulkan_sort_shader(grouped_reduce_atomic_sum_i32.comp grouped_reduce_atomic_sum_u32.comp.spv.h "-DVALUE_KIND=2")
    ti_vulkan_sort_shader(grouped_reduce_atomic_sum_i32.comp grouped_reduce_atomic_sum_u64.comp.spv.h "-DVALUE_KIND=3")
    ti_vulkan_sort_shader(grouped_reduce_atomic_sum_i32.comp grouped_reduce_atomic_sum_i64.comp.spv.h "-DVALUE_KIND=4")
    ti_vulkan_sort_shader(grouped_reduce_atomic_sum_i32.comp grouped_reduce_atomic_sum_f64.comp.spv.h "-DVALUE_KIND=5")
    ti_vulkan_sort_shader(grouped_reduce_atomic_sum_i32.comp grouped_reduce_atomic_sum_i32_strided.comp.spv.h "-DVALUE_KIND=0" "-DSTRIDED_SOURCE=1")
    ti_vulkan_sort_shader(grouped_reduce_atomic_sum_i32.comp grouped_reduce_atomic_sum_f32_strided.comp.spv.h "-DVALUE_KIND=1" "-DSTRIDED_SOURCE=1")
    ti_vulkan_sort_shader(grouped_reduce_atomic_sum_i32.comp grouped_reduce_atomic_sum_u32_strided.comp.spv.h "-DVALUE_KIND=2" "-DSTRIDED_SOURCE=1")
    ti_vulkan_sort_shader(grouped_reduce_atomic_sum_i32.comp grouped_reduce_atomic_sum_u64_strided.comp.spv.h "-DVALUE_KIND=3" "-DSTRIDED_SOURCE=1")
    ti_vulkan_sort_shader(grouped_reduce_atomic_sum_i32.comp grouped_reduce_atomic_sum_i64_strided.comp.spv.h "-DVALUE_KIND=4" "-DSTRIDED_SOURCE=1")
    ti_vulkan_sort_shader(grouped_reduce_atomic_sum_i32.comp grouped_reduce_atomic_sum_f64_strided.comp.spv.h "-DVALUE_KIND=5" "-DSTRIDED_SOURCE=1")
    ti_vulkan_sort_shader(grouped_reduce_sum_i32.comp grouped_reduce_sum_i32.comp.spv.h "-DVALUE_KIND=0")
    ti_vulkan_sort_shader(grouped_reduce_sum_i32.comp grouped_reduce_sum_f32.comp.spv.h "-DVALUE_KIND=1")
    ti_vulkan_sort_shader(grouped_reduce_sum_i32.comp grouped_reduce_sum_u32.comp.spv.h "-DVALUE_KIND=2")
    ti_vulkan_sort_shader(grouped_reduce_sum_i32.comp grouped_reduce_sum_u64.comp.spv.h "-DVALUE_KIND=3")
    ti_vulkan_sort_shader(grouped_reduce_sum_i32.comp grouped_reduce_sum_i64.comp.spv.h "-DVALUE_KIND=4")
    ti_vulkan_sort_shader(grouped_reduce_sum_i32.comp grouped_reduce_sum_f64.comp.spv.h "-DVALUE_KIND=5")
    set(reduce_value_names i32 f32 u32 u64 i64 f64)
    set(reduce_value_kinds 0 1 2 3 4 5)
    foreach(value_index RANGE 0 5)
        list(GET reduce_value_names ${value_index} value_name)
        list(GET reduce_value_kinds ${value_index} value_kind)
        foreach(op_kind 0 1 2)
            if(op_kind EQUAL 0)
                set(op_name "sum")
            elseif(op_kind EQUAL 1)
                set(op_name "min")
            else()
                set(op_name "max")
            endif()
            ti_vulkan_sort_shader(reduce_i32_private.comp
                "reduce_${value_name}_${op_name}_private.comp.spv.h"
                "-DOP_KIND=${op_kind}" "-DVALUE_KIND=${value_kind}")
            ti_vulkan_sort_shader(reduce_i32_private.comp
                "reduce_${value_name}_${op_name}_private_strided.comp.spv.h"
                "-DOP_KIND=${op_kind}" "-DVALUE_KIND=${value_kind}" "-DSTRIDED_SOURCE=1")
            ti_vulkan_sort_shader(reduce_i32_final.comp
                "reduce_${value_name}_${op_name}_final.comp.spv.h"
                "-DOP_KIND=${op_kind}" "-DVALUE_KIND=${value_kind}")
            ti_vulkan_sort_shader(reduce_i32_single.comp
                "reduce_${value_name}_${op_name}_single.comp.spv.h"
                "-DOP_KIND=${op_kind}" "-DVALUE_KIND=${value_kind}")
            ti_vulkan_sort_shader(reduce_i32_single.comp
                "reduce_${value_name}_${op_name}_single_strided.comp.spv.h"
                "-DOP_KIND=${op_kind}" "-DVALUE_KIND=${value_kind}" "-DSTRIDED_SOURCE=1")
        endforeach()
    endforeach()
    ti_vulkan_sort_shader(reduce_i32_sum_atomic.comp
        reduce_i32_sum_atomic.comp.spv.h)
    ti_vulkan_sort_shader(radix8_spine.comp radix8_spine.comp.spv.h)
    # Ray queries require SPIR-V 1.4, which is core in the provider's Vulkan
    # 1.2 floor. The generated C array is linked into the runtime, so glslc is
    # a build-only tool and never becomes a wheel/runtime dependency.
    ti_vulkan_builtin_shader(ray_query_triangles.comp
        ray_query_triangles.comp.spv.h vulkan1.2)
    ti_vulkan_builtin_shader(ray_query_triangles.comp
        ray_query_triangles_typed.comp.spv.h vulkan1.2 "-DTYPED_HITS")

    foreach(shift 0 4 8 12 16 20 24 28)
        ti_vulkan_sort_shader(rank_hist.comp
            "rank_hist_shift${shift}.comp.spv.h" "-DSHIFT=${shift}")
        ti_vulkan_sort_shader(rank_hist_subgroup.comp
            "rank_hist_subgroup_shift${shift}.comp.spv.h" "-DSHIFT=${shift}")
        ti_vulkan_sort_shader(scatter_keys.comp
            "scatter_keys_shift${shift}.comp.spv.h" "-DSHIFT=${shift}")
        ti_vulkan_sort_shader(scatter_keys_inline_chunks.comp
            "scatter_keys_inline_chunks_shift${shift}.comp.spv.h" "-DSHIFT=${shift}")
        ti_vulkan_sort_shader(scatter_pairs.comp
            "scatter_pairs_shift${shift}.comp.spv.h" "-DSHIFT=${shift}")
        ti_vulkan_sort_shader(scatter_pairs.comp
            "scatter_pairs_raw64_shift${shift}.comp.spv.h" "-DSHIFT=${shift}" "-DVALUE_KIND=6")
        ti_vulkan_sort_shader(scatter_pairs_inline_chunks.comp
            "scatter_pairs_inline_chunks_shift${shift}.comp.spv.h" "-DSHIFT=${shift}")
        ti_vulkan_sort_shader(scatter_pairs_inline_chunks.comp
            "scatter_pairs_inline_chunks_raw64_shift${shift}.comp.spv.h" "-DSHIFT=${shift}" "-DVALUE_KIND=6")
    endforeach()

    ti_vulkan_sort_shader(radix8_upsweep.comp
        radix8_upsweep.comp.spv.h)
    ti_vulkan_sort_shader(radix8_downsweep_keys.comp
        radix8_downsweep_keys.comp.spv.h)
    ti_vulkan_sort_shader(radix8_downsweep_pairs.comp
        radix8_downsweep_pairs.comp.spv.h)
    ti_vulkan_sort_shader(radix8_downsweep_pairs.comp
        radix8_downsweep_pairs_raw64.comp.spv.h "-DVALUE_KIND=6")

    if (TI_GLSLC_EXECUTABLE)
        add_custom_target(vulkan_sort_spv_headers
            DEPENDS ${TI_VULKAN_SORT_GENERATED_SPV_HEADERS})
        add_dependencies(${CORE_LIBRARY_NAME} vulkan_sort_spv_headers)
        target_include_directories(${CORE_LIBRARY_NAME} BEFORE PRIVATE
            "${TI_VULKAN_SORT_GENERATED_INCLUDE_DIR}")
    else()
        list(LENGTH TI_VULKAN_SORT_PREGENERATED_SPV_HEADERS
            TI_VULKAN_SORT_PREGENERATED_SPV_HEADER_COUNT)
        message(STATUS
            "Using ${TI_VULKAN_SORT_PREGENERATED_SPV_HEADER_COUNT} checked-in Vulkan sort SPIR-V headers")
    endif()
endif()

target_include_directories(${CORE_LIBRARY_NAME} PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(${CORE_LIBRARY_NAME} PRIVATE external/include)
target_include_directories(${CORE_LIBRARY_NAME} PRIVATE external/SPIRV-Tools/include)
target_include_directories(${CORE_LIBRARY_NAME} PRIVATE external/PicoSHA2)
target_include_directories(${CORE_LIBRARY_NAME} PRIVATE external/eigen)
target_include_directories(${CORE_LIBRARY_NAME} PRIVATE external/FP16/include)

target_link_libraries(${CORE_LIBRARY_NAME} PUBLIC ti_device_api)

if(TI_WITH_LLVM)
    if(DEFINED ENV{LLVM_DIR})
        set(LLVM_DIR $ENV{LLVM_DIR})
        message("Getting LLVM_DIR=${LLVM_DIR} from the environment variable")
    endif()

    # http://llvm.org/docs/CMake.html#embedding-llvm-in-your-project
    find_package(LLVM REQUIRED CONFIG)
    message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
    if(${LLVM_PACKAGE_VERSION} VERSION_LESS "10.0")
        message(FATAL_ERROR "LLVM version < 10 is not supported")
    endif()
    message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
    target_include_directories(${CORE_LIBRARY_NAME} PUBLIC ${LLVM_INCLUDE_DIRS})

    message("LLVM include dirs ${LLVM_INCLUDE_DIRS}")
    message("LLVM library dirs ${LLVM_LIBRARY_DIRS}")
    add_definitions(${LLVM_DEFINITIONS})

    llvm_map_components_to_libnames(llvm_libs
            Core
            ExecutionEngine
            InstCombine
            OrcJIT
            RuntimeDyld
            TransformUtils
            BitReader
            BitWriter
            Object
            ScalarOpts
            Support
            native
            Linker
            Target
            MC
            Passes
            ipo
            Analysis
            )

    if (APPLE AND "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "arm64")
        llvm_map_components_to_libnames(llvm_aarch64_libs AArch64)
    endif()

    add_subdirectory(taichi/codegen/cpu)
    add_subdirectory(taichi/runtime/cpu)

    target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE cpu_codegen)
    target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE cpu_runtime)

    if (TI_WITH_CUDA)
        llvm_map_components_to_libnames(llvm_ptx_libs NVPTX)
        add_subdirectory(taichi/codegen/cuda)
        add_subdirectory(taichi/runtime/cuda)

        target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE cuda_codegen)
        target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE cuda_runtime)
    endif()

    if (TI_WITH_AMDGPU)
        llvm_map_components_to_libnames(llvm_amdgpu_libs AMDGPU)
        add_subdirectory(taichi/codegen/amdgpu)
        add_subdirectory(taichi/runtime/amdgpu)

        target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE amdgpu_codegen)
        target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE amdgpu_runtime)
    endif()

    if (TI_WITH_DX12)
        llvm_map_components_to_libnames(llvm_directx_libs DirectX)

        add_subdirectory(taichi/runtime/dx12)
        add_subdirectory(taichi/codegen/dx12)
        add_subdirectory(taichi/runtime/program_impls/dx12)

        target_include_directories(${CORE_LIBRARY_NAME} PRIVATE external/DirectX-Headers/include)
        target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE dx12_codegen)
        target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE dx12_runtime)
        target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE dx12_program_impl)
    endif()

    add_subdirectory(taichi/codegen/llvm)
    add_subdirectory(taichi/runtime/llvm)
    add_subdirectory(taichi/runtime/program_impls/llvm)

    target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE llvm_program_impl)
    target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE llvm_codegen)
    target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE llvm_runtime)

    if (LINUX)
        # Remove symbols from llvm static libs
        foreach(LETTER ${llvm_libs})
            target_link_options(${CORE_LIBRARY_NAME} PUBLIC -Wl,--exclude-libs=lib${LETTER}.a)
        endforeach()
    endif()
endif()

if (TI_WITH_METAL OR TI_WITH_OPENGL OR TI_WITH_DX11 OR TI_WITH_VULKAN)
    add_subdirectory(taichi/runtime/program_impls/gfx)
    target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE gfx_program_impl)
endif()

if (TI_WITH_METAL)
    add_subdirectory(taichi/runtime/program_impls/metal)
    target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE metal_program_impl)
endif()

if (TI_WITH_OPENGL)
    add_subdirectory(taichi/runtime/program_impls/opengl)
    target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE opengl_program_impl)
endif()

if (TI_WITH_DX11)
    add_subdirectory(taichi/runtime/program_impls/dx)
    target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE dx_program_impl)
endif()

if (TI_WITH_VULKAN)
    add_subdirectory(taichi/runtime/program_impls/vulkan)
    target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE vulkan_program_impl)
endif ()

add_subdirectory(taichi/util)
add_subdirectory(taichi/common)
add_subdirectory(taichi/compilation_manager)

target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE taichi_util)
target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE taichi_common)
target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE compilation_manager)

if (TI_WITH_CUDA AND
        (TI_WITH_CUDA_TOOLKIT_PRIMITIVE_REFERENCE OR TI_WITH_CUPTI))
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "Found CUDAToolkit ${CUDAToolkit_VERSION}")
endif()

if (TI_WITH_CUDA AND TI_WITH_CUPTI)
    include(CheckCXXSourceCompiles)
    set(_ti_saved_required_includes "${CMAKE_REQUIRED_INCLUDES}")
    set(CMAKE_REQUIRED_INCLUDES
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CUDAToolkit_CUPTI_INCLUDE_DIR})
    unset(TI_CUPTI_HAS_METRICS_CONTEXT_EVALUATE_API CACHE)
    check_cxx_source_compiles([=[
        #include <nvperf_host.h>
        int main() {
          NVPW_MetricsContext_EvaluateToGpuValues_Params params{};
          return params.structSize == 0;
        }
    ]=] TI_CUPTI_HAS_METRICS_CONTEXT_EVALUATE_API)
    set(CMAKE_REQUIRED_INCLUDES "${_ti_saved_required_includes}")
    unset(_ti_saved_required_includes)
    if (NOT TI_CUPTI_HAS_METRICS_CONTEXT_EVALUATE_API)
        message(FATAL_ERROR
            "TI_WITH_CUPTI=ON requires the legacy NVPerf metrics-context API "
            "used by Taichi's current CUPTI profiler, but the selected CUDA "
            "Toolkit does not provide it. Build with TI_WITH_CUPTI=OFF. This "
            "does not affect the CUDA backend or primitive providers.")
    endif()
    target_link_libraries(${CORE_LIBRARY_NAME} PUBLIC CUDA::cupti)
endif()

# SPIR-V codegen is always there, regardless of Vulkan
set(SPIRV_SKIP_EXECUTABLES true)
set(SPIRV-Headers_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/SPIRV-Headers)
set(ENABLE_SPIRV_TOOLS_INSTALL OFF)
add_subdirectory(external/SPIRV-Tools)
add_subdirectory(taichi/codegen/spirv)
add_subdirectory(taichi/runtime/gfx)

if (TI_WITH_OPENGL OR TI_WITH_VULKAN OR TI_WITH_DX11 OR TI_WITH_METAL)
  target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE spirv_codegen)
  target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE gfx_runtime)
endif()

if (TI_WITH_OPENGL OR TI_WITH_DX11 OR TI_WITH_METAL)
  set(SPIRV_CROSS_CLI false)
  add_subdirectory(${PROJECT_SOURCE_DIR}/external/SPIRV-Cross ${PROJECT_BINARY_DIR}/external/SPIRV-Cross)
endif()


# Optional dependencies
if (APPLE)
    set(APPLE_FRAMEWORKS "")
    find_library(Foundation NAMES Foundation REQUIRED)
    find_library(Metal NAMES Metal REQUIRED)
    list(APPEND APPLE_FRAMEWORKS ${Foundation} ${Metal})
    if (NOT IOS)
        find_library(ApplicationServices NAMES ApplicationServices REQUIRED)
        find_library(Cocoa NAMES Cocoa REQUIRED)
        list(APPEND APPLE_FRAMEWORKS ${ApplicationServices} ${Cocoa})
    endif()
    target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE ${APPLE_FRAMEWORKS})
endif ()

if (ANDROID)
    # Android has a custom toolchain so pthread is not available and should
    # link against other libraries as well for logcat and internal features.
    target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE android log)
elseif (LINUX)
    target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE X11 pthread)
    if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
        # Avoid glibc dependencies
        if (TI_WITH_VULKAN)
            target_link_options(${CORE_LIBRARY_NAME} PRIVATE -Wl,--wrap=log2f)
        else()
            # Enforce compatibility with manylinux2014
            target_link_options(${CORE_LIBRARY_NAME} PRIVATE -Wl,--wrap=log2f -Wl,--wrap=exp2 -Wl,--wrap=log2 -Wl,--wrap=logf -Wl,--wrap=powf -Wl,--wrap=exp -Wl,--wrap=log -Wl,--wrap=pow)
        endif()
    endif()
elseif (WIN32)
    target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE Winmm)
endif()



foreach (source IN LISTS TAICHI_CORE_SOURCE)
    file(RELATIVE_PATH source_rel ${CMAKE_CURRENT_LIST_DIR} ${source})
    get_filename_component(source_path "${source_rel}" PATH)
    string(REPLACE "/" "\\" source_path_msvc "${source_path}")
    source_group("${source_path_msvc}" FILES "${source}")
endforeach ()
endif()

if(TI_WITH_PYTHON)
    # TODO Use TI_WITH_UI to guard the compilation of this target.
    # This requires refactoring on the python/export_*.cpp as well as better
    # error message on the Python side.
    if(NOT TI_WITH_PREBUILT_PYTHON_RUNTIME)
        add_subdirectory(taichi/ui)
    endif()

    message("PYTHON_LIBRARIES: " ${PYTHON_LIBRARIES})
    set(CORE_WITH_PYBIND_LIBRARY_NAME taichi_python)
    set(_ti_pybind_compile_targets ${CORE_WITH_PYBIND_LIBRARY_NAME})
    if (NOT ANDROID)
        # NO_EXTRAS is required here to avoid llvm symbol error during build
        file(GLOB TAICHI_PYBIND_SOURCE
            "taichi/python/*.cpp"
            "taichi/python/*.h"
        )
        if(TI_WITH_SPLIT_PYTHON_RUNTIME AND
                NOT TI_WITH_PREBUILT_PYTHON_RUNTIME)
            set(_ti_pybind_object_target taichi_python_bindings)
            add_library(${_ti_pybind_object_target} OBJECT ${TAICHI_PYBIND_SOURCE})
            set_target_properties(${_ti_pybind_object_target} PROPERTIES
                POSITION_INDEPENDENT_CODE ON)
            target_link_libraries(${_ti_pybind_object_target}
                PRIVATE pybind11::module)
            pybind11_add_module(${CORE_WITH_PYBIND_LIBRARY_NAME} NO_EXTRAS
                $<TARGET_OBJECTS:${_ti_pybind_object_target}>)
            list(APPEND _ti_pybind_compile_targets ${_ti_pybind_object_target})
        else()
            pybind11_add_module(${CORE_WITH_PYBIND_LIBRARY_NAME} NO_EXTRAS
                ${TAICHI_PYBIND_SOURCE})
        endif()
    else()
        add_library(${CORE_WITH_PYBIND_LIBRARY_NAME} SHARED)
    endif ()

    # Remove symbols from static libs: https://stackoverflow.com/a/14863432/12003165
    if (LINUX)
        target_link_options(${CORE_WITH_PYBIND_LIBRARY_NAME} PUBLIC -Wl,--exclude-libs=ALL)
        if (TI_WITH_CUDA)
            # Explicit hardware provider probes use transient dlopen/dlsym
            # handles in the pybind shim and must not depend on private split
            # runtime C++ symbols.
            target_link_libraries(${CORE_WITH_PYBIND_LIBRARY_NAME}
                PRIVATE ${CMAKE_DL_LIBS})
        endif()
        if (NOT ANDROID)
            # Excluding Android
            # Android defaults to static linking with libc++, no tinkering needed.
            # The split runtime and pybind shim exchange C++ objects, RTTI,
            # inline singletons, and exception state. libtaichi_runtime.so
            # uses the platform libstdc++/libgcc_s, so statically linking a
            # second copy into the shim creates two incompatible C++ runtime
            # domains in one process and can corrupt pybind initialization.
            # Let the C++ linker driver add the same shared dependencies used
            # by libtaichi_runtime.so. Both libraries remain manylinux-policy
            # dependencies and are not copied into either wheel.
            if (NOT TI_WITH_SPLIT_PYTHON_RUNTIME)
                target_link_options(${CORE_WITH_PYBIND_LIBRARY_NAME} PUBLIC -static-libgcc -static-libstdc++)
            endif()
        endif()
    endif()

    if (TI_WITH_BACKTRACE AND NOT TI_WITH_PREBUILT_PYTHON_RUNTIME)
        # Defined by external/backward-cpp:
        # This will add libraries, definitions and include directories needed by backward
        # by setting each property on the target.
        foreach(_ti_pybind_target IN LISTS _ti_pybind_compile_targets)
            target_link_libraries(${_ti_pybind_target} PRIVATE ${BACKWARD_ENABLE})
        endforeach()
    endif()

    if(TI_WITH_GGUI)
        foreach(_ti_pybind_target IN LISTS _ti_pybind_compile_targets)
            target_compile_definitions(${_ti_pybind_target}
                PRIVATE -DTI_WITH_GGUI -DIMGUI_IMPL_VULKAN_NO_PROTOTYPES)
        endforeach()
    endif()
    if(TI_WITH_PREBUILT_PYTHON_RUNTIME)
        foreach(_ti_backend_define IN ITEMS
            TI_WITH_LLVM
            TI_WITH_CUDA
            TI_WITH_CUDA_TOOLKIT
            TI_WITH_AMDGPU
            TI_WITH_METAL
            TI_WITH_OPENGL
            TI_WITH_VULKAN
            TI_WITH_DX11
            TI_WITH_DX12)
            if(${_ti_backend_define})
                foreach(_ti_pybind_target IN LISTS _ti_pybind_compile_targets)
                    target_compile_definitions(${_ti_pybind_target}
                        PRIVATE -D${_ti_backend_define})
                endforeach()
            endif()
        endforeach()

        if(TI_WITH_LLVM)
            if(NOT LLVM_DIR AND DEFINED ENV{LLVM_DIR})
                set(LLVM_DIR $ENV{LLVM_DIR})
                message("Getting LLVM_DIR=${LLVM_DIR} from the environment variable")
            endif()
            if(NOT LLVM_DIR)
                message(FATAL_ERROR
                    "Prebuilt split Python runtime shim requires LLVM_DIR to "
                    "point to the LLVM 20 package downloaded from "
                    "LLVM20_WIN_URL/LLVM20_LINUX_URL in the publish workflow.")
            endif()

            # The prebuilt-runtime shim still compiles Python binding sources
            # that include Taichi LLVM/GFX headers.  The native LLVM libraries
            # stay in taichi_runtime; the shim only needs LLVM compile inputs.
            find_package(LLVM REQUIRED CONFIG)
            message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
            if("${LLVM_PACKAGE_VERSION}" VERSION_LESS "20.0" OR
                    NOT "${LLVM_PACKAGE_VERSION}" VERSION_LESS "21.0")
                message(FATAL_ERROR
                    "Prebuilt split Python runtime shim requires LLVM 20.x; "
                    "found LLVM ${LLVM_PACKAGE_VERSION}")
            endif()
            message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
            message("LLVM include dirs ${LLVM_INCLUDE_DIRS}")
            add_definitions(${LLVM_DEFINITIONS})

            if(LINUX)
                # The prebuilt Linux shim consumes LLVM header-only ADTs but
                # deliberately has no DT_NEEDED edge to libLLVM/LLVMSupport.
                # LLVM's generated abi-breaking.h explicitly supports this
                # mode; otherwise every binding translation unit retains an
                # unresolved Enable/DisableABIBreakingChecks sentinel. The
                # native runtime build remains fully ABI-checked.
                foreach(_ti_pybind_target IN LISTS _ti_pybind_compile_targets)
                    target_compile_definitions(${_ti_pybind_target}
                        PRIVATE LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1)
                endforeach()
            endif()
        endif()
    endif()

    if(TI_WITH_SPLIT_PYTHON_RUNTIME)
        foreach(_ti_pybind_target IN LISTS _ti_pybind_compile_targets)
            target_compile_definitions(${_ti_pybind_target}
                PRIVATE TI_WITH_SPLIT_PYTHON_RUNTIME)
        endforeach()
        set(CORE_PYTHON_RUNTIME_LIBRARY_NAME taichi_runtime)

        function(_ti_link_split_runtime_native_targets target)
            foreach(_ti_runtime_lib IN LISTS ARGN)
                if(TARGET ${_ti_runtime_lib})
                    get_target_property(_ti_runtime_lib_type
                        ${_ti_runtime_lib} TYPE)
                    if(_ti_runtime_lib_type STREQUAL "OBJECT_LIBRARY")
                        target_link_libraries(${target}
                            PRIVATE ${_ti_runtime_lib})
                    elseif(_ti_runtime_lib_type STREQUAL "STATIC_LIBRARY")
                        if(MSVC)
                            target_link_libraries(${target}
                                PRIVATE ${_ti_runtime_lib})
                            target_link_options(${target}
                                PRIVATE "/WHOLEARCHIVE:$<TARGET_FILE:${_ti_runtime_lib}>")
                        elseif(LINUX)
                            target_link_libraries(${target}
                                PRIVATE
                                    "-Wl,--whole-archive"
                                    ${_ti_runtime_lib}
                                    "-Wl,--no-whole-archive")
                        elseif(APPLE)
                            target_link_libraries(${target}
                                PRIVATE ${_ti_runtime_lib})
                            target_link_options(${target}
                                PRIVATE
                                    "-Wl,-force_load,$<TARGET_FILE:${_ti_runtime_lib}>")
                        else()
                            target_link_libraries(${target}
                                PRIVATE ${_ti_runtime_lib})
                        endif()
                    else()
                        target_link_libraries(${target}
                            PRIVATE ${_ti_runtime_lib})
                    endif()
                endif()
            endforeach()
        endfunction()

        function(_ti_collect_windows_runtime_objects output_var)
            set(_ti_runtime_objects)
            foreach(_ti_runtime_lib IN LISTS ARGN)
                if(TARGET ${_ti_runtime_lib})
                    get_target_property(_ti_runtime_lib_type
                        ${_ti_runtime_lib} TYPE)
                    if(_ti_runtime_lib_type STREQUAL "OBJECT_LIBRARY")
                        list(APPEND _ti_runtime_objects
                            "$<TARGET_OBJECTS:${_ti_runtime_lib}>")
                    endif()
                endif()
            endforeach()
            set(${output_var} ${_ti_runtime_objects} PARENT_SCOPE)
        endfunction()

        function(_ti_collect_runtime_symbol_inputs output_var)
            set(_ti_runtime_symbol_inputs)
            foreach(_ti_runtime_lib IN LISTS ARGN)
                if(TARGET ${_ti_runtime_lib})
                    get_target_property(_ti_runtime_lib_type
                        ${_ti_runtime_lib} TYPE)
                    if(_ti_runtime_lib_type STREQUAL "OBJECT_LIBRARY")
                        list(APPEND _ti_runtime_symbol_inputs
                            "$<TARGET_OBJECTS:${_ti_runtime_lib}>")
                    elseif(_ti_runtime_lib_type STREQUAL "STATIC_LIBRARY")
                        list(APPEND _ti_runtime_symbol_inputs
                            "$<TARGET_FILE:${_ti_runtime_lib}>")
                    endif()
                endif()
            endforeach()
            set(${output_var} ${_ti_runtime_symbol_inputs} PARENT_SCOPE)
        endfunction()

        if(TI_PREBUILT_PYTHON_RUNTIME_DIR)
            get_filename_component(TI_PREBUILT_PYTHON_RUNTIME_DIR
                "${TI_PREBUILT_PYTHON_RUNTIME_DIR}" ABSOLUTE)

            if(WIN32)
                set(_ti_prebuilt_runtime_location
                    "${TI_PREBUILT_PYTHON_RUNTIME_DIR}/taichi_runtime.dll")
                set(_ti_prebuilt_runtime_implib
                    "${TI_PREBUILT_PYTHON_RUNTIME_DIR}/taichi_runtime.lib")
            elseif(APPLE)
                set(_ti_prebuilt_runtime_location
                    "${TI_PREBUILT_PYTHON_RUNTIME_DIR}/libtaichi_runtime.dylib")
            else()
                set(_ti_prebuilt_runtime_location
                    "${TI_PREBUILT_PYTHON_RUNTIME_DIR}/libtaichi_runtime.so")
            endif()

            if(NOT EXISTS "${_ti_prebuilt_runtime_location}")
                message(FATAL_ERROR
                    "TI_PREBUILT_PYTHON_RUNTIME_DIR does not contain ${_ti_prebuilt_runtime_location}")
            endif()

            add_library(${CORE_PYTHON_RUNTIME_LIBRARY_NAME} SHARED IMPORTED GLOBAL)
            set_target_properties(${CORE_PYTHON_RUNTIME_LIBRARY_NAME} PROPERTIES
                IMPORTED_LOCATION "${_ti_prebuilt_runtime_location}")

            if(WIN32)
                if(NOT EXISTS "${_ti_prebuilt_runtime_implib}")
                    message(FATAL_ERROR
                        "TI_PREBUILT_PYTHON_RUNTIME_DIR does not contain ${_ti_prebuilt_runtime_implib}")
                endif()
                set_target_properties(${CORE_PYTHON_RUNTIME_LIBRARY_NAME} PROPERTIES
                    IMPORTED_IMPLIB "${_ti_prebuilt_runtime_implib}")
            endif()

            message(STATUS
                "Using prebuilt split Python runtime: ${_ti_prebuilt_runtime_location}")
        else()
            add_library(${CORE_PYTHON_RUNTIME_LIBRARY_NAME} SHARED)
            set(_ti_runtime_anchor_source
                "${CMAKE_CURRENT_BINARY_DIR}/${CORE_PYTHON_RUNTIME_LIBRARY_NAME}_anchor.cpp")
            file(WRITE "${_ti_runtime_anchor_source}"
                "extern \"C\" void taichi_runtime_anchor() {}\n")
            target_sources(${CORE_PYTHON_RUNTIME_LIBRARY_NAME}
                PRIVATE "${_ti_runtime_anchor_source}")
            target_compile_definitions(${CORE_PYTHON_RUNTIME_LIBRARY_NAME}
                PRIVATE
                    TI_WITH_SPLIT_PYTHON_RUNTIME
                    TI_BUILDING_PYTHON_RUNTIME)
            set(_ti_runtime_native_targets
                taichi_ui
                taichi_ui_ggui
                taichi_common
                taichi_util
                compilation_manager
                ti_device_api
                cpu_codegen
                cpu_runtime
                cuda_codegen
                cuda_runtime
                llvm_program_impl
                llvm_codegen
                llvm_runtime
                gfx_program_impl
                opengl_program_impl
                vulkan_program_impl
                spirv_codegen
                gfx_runtime
                common_rhi
                interop_rhi
                cpu_rhi
                cuda_rhi
                llvm_rhi
                opengl_rhi
                vulkan_rhi)

            target_link_libraries(${CORE_PYTHON_RUNTIME_LIBRARY_NAME} PRIVATE taichi_ui)
            target_link_libraries(${CORE_PYTHON_RUNTIME_LIBRARY_NAME} PRIVATE ${CORE_LIBRARY_NAME})
            _ti_link_split_runtime_native_targets(
                ${CORE_PYTHON_RUNTIME_LIBRARY_NAME}
                ${_ti_runtime_native_targets})

            if(MSVC)
                # The runtime export closure scans undefined references from
                # the pybind shim objects. File-level TARGET_OBJECTS entries in
                # the custom command do not create a cross-project build edge
                # with Visual Studio multi-config generators, so a direct
                # taichi_runtime build could run the scanner before those
                # objects existed.
                add_dependencies(${CORE_PYTHON_RUNTIME_LIBRARY_NAME}
                    ${_ti_pybind_object_target})
                _ti_collect_windows_runtime_objects(
                    _ti_windows_runtime_export_objects
                    ${CORE_LIBRARY_NAME}
                    ${_ti_runtime_native_targets})
                set(_ti_runtime_export_objlist
                    "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/taichi_runtime.$<CONFIG>.exports.def.objs")
                set(_ti_runtime_raw_exports
                    "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/taichi_runtime.$<CONFIG>.raw.exports.def")
                set(_ti_runtime_filtered_exports
                    "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/taichi_runtime.$<CONFIG>.exports.def")
                set(_ti_runtime_export_manifest
                    "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/taichi_runtime.$<CONFIG>.exports.json")
                set(_ti_pybind_export_objlist
                    "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/taichi_python.$<CONFIG>.imports.objs")

                file(GENERATE
                    OUTPUT "${_ti_runtime_export_objlist}"
                    CONTENT "$<JOIN:${_ti_windows_runtime_export_objects},\n>\n")
                file(GENERATE
                    OUTPUT "${_ti_pybind_export_objlist}"
                    CONTENT "$<JOIN:$<TARGET_OBJECTS:${_ti_pybind_object_target}>,\n>\n")
                add_custom_command(
                    OUTPUT
                        "${_ti_runtime_filtered_exports}"
                        "${_ti_runtime_export_manifest}"
                    COMMAND ${CMAKE_COMMAND} -E __create_def
                        "${_ti_runtime_raw_exports}"
                        "${_ti_runtime_export_objlist}"
                    COMMAND "${PYTHON_EXECUTABLE}"
                        "${PROJECT_SOURCE_DIR}/misc/generate_windows_runtime_export_closure.py"
                        "${_ti_runtime_raw_exports}"
                        "${_ti_pybind_export_objlist}"
                        "${_ti_runtime_filtered_exports}"
                        "${_ti_runtime_export_manifest}"
                    DEPENDS
                        ${_ti_windows_runtime_export_objects}
                        $<TARGET_OBJECTS:${_ti_pybind_object_target}>
                        "${PROJECT_SOURCE_DIR}/misc/generate_windows_runtime_export_closure.py"
                        "${PROJECT_SOURCE_DIR}/misc/runtime_export_closure.py"
                    VERBATIM)
                set_source_files_properties("${_ti_runtime_filtered_exports}"
                    PROPERTIES GENERATED TRUE)
                target_sources(${CORE_PYTHON_RUNTIME_LIBRARY_NAME}
                    PRIVATE "${_ti_runtime_filtered_exports}")
                set_target_properties(${CORE_PYTHON_RUNTIME_LIBRARY_NAME}
                    PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS OFF)
                add_custom_command(
                    TARGET ${CORE_PYTHON_RUNTIME_LIBRARY_NAME}
                    POST_BUILD
                    COMMAND "${PYTHON_EXECUTABLE}"
                        "${PROJECT_SOURCE_DIR}/misc/generate_windows_runtime_export_closure.py"
                        --audit-dll
                        "$<TARGET_FILE:${CORE_PYTHON_RUNTIME_LIBRARY_NAME}>"
                        "${_ti_runtime_export_manifest}"
                    VERBATIM)
            elseif(LINUX)
                _ti_collect_runtime_symbol_inputs(
                    _ti_runtime_export_symbol_inputs
                    ${CORE_LIBRARY_NAME}
                    ${_ti_runtime_native_targets})
                set(_ti_runtime_export_input_list
                    "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/taichi_runtime.exports.inputs")
                set(_ti_pybind_export_objlist
                    "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/taichi_python.imports.objs")
                set(_ti_runtime_filtered_exports
                    "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/taichi_runtime.exports.map")
                set(_ti_runtime_export_manifest
                    "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/taichi_runtime.exports.json")

                file(GENERATE
                    OUTPUT "${_ti_runtime_export_input_list}"
                    CONTENT "$<JOIN:${_ti_runtime_export_symbol_inputs},\n>\n")
                file(GENERATE
                    OUTPUT "${_ti_pybind_export_objlist}"
                    CONTENT "$<JOIN:$<TARGET_OBJECTS:${_ti_pybind_object_target}>,\n>\n")
                add_custom_command(
                    OUTPUT
                        "${_ti_runtime_filtered_exports}"
                        "${_ti_runtime_export_manifest}"
                    COMMAND "${PYTHON_EXECUTABLE}"
                        "${PROJECT_SOURCE_DIR}/misc/generate_elf_runtime_export_closure.py"
                        "${_ti_runtime_export_input_list}"
                        "${_ti_pybind_export_objlist}"
                        "${_ti_runtime_filtered_exports}"
                        "${_ti_runtime_export_manifest}"
                    DEPENDS
                        ${_ti_runtime_export_symbol_inputs}
                        $<TARGET_OBJECTS:${_ti_pybind_object_target}>
                        "${PROJECT_SOURCE_DIR}/misc/generate_elf_runtime_export_closure.py"
                        "${PROJECT_SOURCE_DIR}/misc/runtime_export_closure.py"
                    VERBATIM)
                add_custom_target(taichi_runtime_export_closure
                    DEPENDS
                        "${_ti_runtime_filtered_exports}"
                        "${_ti_runtime_export_manifest}")
                add_dependencies(${CORE_PYTHON_RUNTIME_LIBRARY_NAME}
                    taichi_runtime_export_closure)
                target_link_options(${CORE_PYTHON_RUNTIME_LIBRARY_NAME}
                    PRIVATE
                        "-Wl,--version-script=${_ti_runtime_filtered_exports}")
                set_property(TARGET ${CORE_PYTHON_RUNTIME_LIBRARY_NAME}
                    APPEND PROPERTY LINK_DEPENDS
                        "${_ti_runtime_filtered_exports}")
                add_custom_command(
                    TARGET ${CORE_PYTHON_RUNTIME_LIBRARY_NAME}
                    POST_BUILD
                    COMMAND "${PYTHON_EXECUTABLE}"
                        "${PROJECT_SOURCE_DIR}/misc/generate_elf_runtime_export_closure.py"
                        --audit-elf
                        "$<TARGET_FILE:${CORE_PYTHON_RUNTIME_LIBRARY_NAME}>"
                        "${_ti_runtime_export_manifest}"
                    VERBATIM)
            elseif(APPLE)
                _ti_collect_runtime_symbol_inputs(
                    _ti_runtime_export_symbol_inputs
                    ${CORE_LIBRARY_NAME}
                    ${_ti_runtime_native_targets})
                set(_ti_runtime_export_input_list
                    "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/taichi_runtime.exports.inputs")
                set(_ti_pybind_export_objlist
                    "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/taichi_python.imports.objs")
                set(_ti_runtime_filtered_exports
                    "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/taichi_runtime.exports.list")
                set(_ti_runtime_export_manifest
                    "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/taichi_runtime.exports.json")

                file(GENERATE
                    OUTPUT "${_ti_runtime_export_input_list}"
                    CONTENT "$<JOIN:${_ti_runtime_export_symbol_inputs},\n>\n")
                file(GENERATE
                    OUTPUT "${_ti_pybind_export_objlist}"
                    CONTENT "$<JOIN:$<TARGET_OBJECTS:${_ti_pybind_object_target}>,\n>\n")
                add_custom_command(
                    OUTPUT
                        "${_ti_runtime_filtered_exports}"
                        "${_ti_runtime_export_manifest}"
                    COMMAND "${PYTHON_EXECUTABLE}"
                        "${PROJECT_SOURCE_DIR}/misc/generate_macho_runtime_export_closure.py"
                        "${_ti_runtime_export_input_list}"
                        "${_ti_pybind_export_objlist}"
                        "${_ti_runtime_filtered_exports}"
                        "${_ti_runtime_export_manifest}"
                    DEPENDS
                        ${_ti_runtime_export_symbol_inputs}
                        $<TARGET_OBJECTS:${_ti_pybind_object_target}>
                        "${PROJECT_SOURCE_DIR}/misc/generate_macho_runtime_export_closure.py"
                        "${PROJECT_SOURCE_DIR}/misc/generate_elf_runtime_export_closure.py"
                        "${PROJECT_SOURCE_DIR}/misc/runtime_export_closure.py"
                    VERBATIM)
                add_custom_target(taichi_runtime_export_closure
                    DEPENDS
                        "${_ti_runtime_filtered_exports}"
                        "${_ti_runtime_export_manifest}")
                add_dependencies(${CORE_PYTHON_RUNTIME_LIBRARY_NAME}
                    taichi_runtime_export_closure)
                target_link_options(${CORE_PYTHON_RUNTIME_LIBRARY_NAME}
                    PRIVATE
                        "-Wl,-exported_symbols_list,${_ti_runtime_filtered_exports}")
                set_property(TARGET ${CORE_PYTHON_RUNTIME_LIBRARY_NAME}
                    APPEND PROPERTY LINK_DEPENDS
                        "${_ti_runtime_filtered_exports}")
                set_target_properties(${CORE_PYTHON_RUNTIME_LIBRARY_NAME}
                    PROPERTIES INSTALL_NAME_DIR "@rpath")
                add_custom_command(
                    TARGET ${CORE_PYTHON_RUNTIME_LIBRARY_NAME}
                    POST_BUILD
                    COMMAND "${PYTHON_EXECUTABLE}"
                        "${PROJECT_SOURCE_DIR}/misc/generate_macho_runtime_export_closure.py"
                        --audit-macho
                        "$<TARGET_FILE:${CORE_PYTHON_RUNTIME_LIBRARY_NAME}>"
                        "${_ti_runtime_export_manifest}"
                    VERBATIM)
            endif()
            target_enable_function_level_linking(${CORE_PYTHON_RUNTIME_LIBRARY_NAME})

            if(WIN32)
                set_target_properties(${CORE_PYTHON_RUNTIME_LIBRARY_NAME} PROPERTIES
                    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/runtimes"
                    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/runtimes"
                    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/runtimes"
                    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_CURRENT_SOURCE_DIR}/runtimes"
                    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_CURRENT_SOURCE_DIR}/runtimes"
                    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/runtimes"
                    ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/runtimes"
                    ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/runtimes"
                    ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_CURRENT_SOURCE_DIR}/runtimes"
                    ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_CURRENT_SOURCE_DIR}/runtimes")
                if(NOT MSVC)
                    set_target_properties(${CORE_PYTHON_RUNTIME_LIBRARY_NAME}
                        PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
                endif()
            endif()

            install(TARGETS ${CORE_PYTHON_RUNTIME_LIBRARY_NAME}
                    RUNTIME DESTINATION ${INSTALL_LIB_DIR}/runtime_native
                    LIBRARY DESTINATION ${INSTALL_LIB_DIR}/runtime_native
                    ARCHIVE DESTINATION ${INSTALL_LIB_DIR}/runtime_native
                    COMPONENT runtime)
            if(MSVC OR LINUX OR APPLE)
                install(FILES "${_ti_runtime_export_manifest}"
                    DESTINATION ${INSTALL_LIB_DIR}/runtime_native
                    RENAME taichi_runtime.exports.json
                    COMPONENT runtime)
            endif()
        endif()

        target_link_libraries(${CORE_WITH_PYBIND_LIBRARY_NAME}
            PRIVATE ${CORE_PYTHON_RUNTIME_LIBRARY_NAME})
        if(LINUX)
            # The runtime lives in the sibling taichi-forge-runtime
            # distribution. An explicit ELF dependency keeps its private C++
            # ABI in this module's local lookup scope and removes the need for
            # a process-wide RTLD_GLOBAL preload.
            set_target_properties(${CORE_WITH_PYBIND_LIBRARY_NAME} PROPERTIES
                INSTALL_RPATH
                    "$ORIGIN/../../../taichi_forge_runtime/_lib/runtime_native")
        elseif(APPLE)
            set_target_properties(${CORE_WITH_PYBIND_LIBRARY_NAME} PROPERTIES
                INSTALL_RPATH
                    "@loader_path/../../../taichi_forge_runtime/_lib/runtime_native")
        endif()
    else()
        target_link_libraries(${CORE_WITH_PYBIND_LIBRARY_NAME} PRIVATE taichi_ui)
        target_link_libraries(${CORE_WITH_PYBIND_LIBRARY_NAME} PRIVATE ${CORE_LIBRARY_NAME})
    endif()

    foreach(_ti_pybind_target IN LISTS _ti_pybind_compile_targets)
      target_include_directories(${_ti_pybind_target}
        PRIVATE
          ${PROJECT_SOURCE_DIR}
          ${PROJECT_SOURCE_DIR}/external/spdlog/include
          ${PROJECT_SOURCE_DIR}/external/eigen
          ${PROJECT_SOURCE_DIR}/external/volk
          ${PROJECT_SOURCE_DIR}/external/SPIRV-Tools/include
          ${PROJECT_SOURCE_DIR}/external/SPIRV-Headers/include
          ${PROJECT_SOURCE_DIR}/external/Vulkan-Headers/include
          ${PROJECT_SOURCE_DIR}/external/glm
          ${PROJECT_SOURCE_DIR}/external/imgui
          ${PROJECT_SOURCE_DIR}/external/imgui/backends
          ${PROJECT_SOURCE_DIR}/external/FP16/include
        )
      target_include_directories(${_ti_pybind_target} SYSTEM
        PRIVATE
          ${PROJECT_SOURCE_DIR}/external/VulkanMemoryAllocator/include
        )
    endforeach()

    if(TI_WITH_LLVM)
      foreach(_ti_pybind_target IN LISTS _ti_pybind_compile_targets)
        target_include_directories(${_ti_pybind_target} SYSTEM
          PRIVATE
            ${LLVM_INCLUDE_DIRS}
          )
      endforeach()
    endif()

    if (NOT ANDROID)
      foreach(_ti_pybind_target IN LISTS _ti_pybind_compile_targets)
        target_include_directories(${_ti_pybind_target}
          PRIVATE
            external/glfw/include
            external/glad/include
          )
      endforeach()
    endif()

    # These commands should apply to the DLL that is loaded from python, not the OBJECT library.
    if (MSVC)
        set_property(TARGET ${CORE_WITH_PYBIND_LIBRARY_NAME} APPEND PROPERTY LINK_FLAGS /DEBUG)
    endif ()

    if (WIN32)
        set_target_properties(${CORE_WITH_PYBIND_LIBRARY_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                "${CMAKE_CURRENT_SOURCE_DIR}/runtimes")
    endif ()

    install(TARGETS ${CORE_WITH_PYBIND_LIBRARY_NAME}
            RUNTIME DESTINATION ${INSTALL_LIB_DIR}/core
            LIBRARY DESTINATION ${INSTALL_LIB_DIR}/core
            COMPONENT python)
endif()

# Runtime bitcode belongs to taichi-forge-runtime. A shim build that links a
# prebuilt split runtime must not register install rules for those assets.
if (NOT TI_WITH_PREBUILT_PYTHON_RUNTIME)
    if (NOT APPLE)
        install(FILES ${CMAKE_SOURCE_DIR}/external/cuda_libdevice/${_ti_cuda_libdevice_filename}
                DESTINATION ${INSTALL_LIB_DIR}/runtime
                COMPONENT runtime)
    endif()

    if (TI_WITH_AMDGPU)
        file(GLOB AMDGPU_BC_FILES ${CMAKE_SOURCE_DIR}/external/amdgpu_libdevice/*.bc)
        install(FILES ${AMDGPU_BC_FILES}
                DESTINATION ${INSTALL_LIB_DIR}/runtime
                COMPONENT runtime)
    endif()
endif()
