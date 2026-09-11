option(TI_BUILD_BUNDLED_OPTIX_PROVIDERS
       "Build the pinned OptiX provider set for the runtime wheel" OFF)
option(TI_ALLOW_UNQUALIFIED_OPTIX_PTX_TOOLKIT
       "Allow non-release CUDA Toolkit versions for local adapter builds" OFF)
set(TI_OPTIX_ROOT_93 "" CACHE PATH
    "Optional OptiX ABI 93 header root for a bundled-provider build")
set(TI_OPTIX_ROOT_105 "" CACHE PATH
    "Optional OptiX ABI 105 header root for a bundled-provider build")
set(TI_OPTIX_ROOT_118 "" CACHE PATH
    "Optional OptiX ABI 118 header root for a bundled-provider build")

# scikit-build-core names this target explicitly on every platform. It remains
# empty where the NVIDIA CUDA/OptiX provider is not supported, so the option
# does not create a new wheel variant or disturb macOS builds.
add_custom_target(taichi_forge_optix_providers)

if(NOT TI_BUILD_BUNDLED_OPTIX_PROVIDERS)
    return()
endif()

if(APPLE OR NOT (WIN32 OR LINUX))
    message(STATUS "Bundled OptiX providers are not built on this platform")
    return()
endif()

find_package(CUDAToolkit REQUIRED)
if(NOT CUDAToolkit_NVCC_EXECUTABLE)
    message(FATAL_ERROR "The OptiX provider build requires nvcc for its PTX artifact")
endif()
set(_ti_optix_expected_ptx_version "8.5")
if(NOT TI_ALLOW_UNQUALIFIED_OPTIX_PTX_TOOLKIT AND
   (CUDAToolkit_VERSION VERSION_LESS "12.5" OR
    NOT CUDAToolkit_VERSION VERSION_LESS "12.6"))
    message(FATAL_ERROR
        "Bundled OptiX providers require CUDA Toolkit 12.5.x so their PTX 8.5 "
        "does not raise the qualified OptiX 8.1 / R555 runtime floor; found "
        "CUDA ${CUDAToolkit_VERSION}")
endif()
if(TI_ALLOW_UNQUALIFIED_OPTIX_PTX_TOOLKIT)
    set(_ti_optix_expected_ptx_version "")
    message(WARNING
        "Building unqualified OptiX PTX with CUDA ${CUDAToolkit_VERSION}; "
        "this output must not enter a release wheel")
endif()

set(_ti_optix_supported_abis 93 105 118)

function(_ti_validate_optix_root root expected_abi out_abi)
    if(NOT EXISTS "${root}/include/optix.h" OR
       NOT EXISTS "${root}/include/optix_function_table.h" OR
       NOT EXISTS "${root}/include/optix_stubs.h")
        message(FATAL_ERROR
            "OptiX ABI ${expected_abi} headers were not found under ${root}/include")
    endif()
    file(STRINGS "${root}/include/optix_function_table.h"
         _abi_line REGEX "^#define[ \t]+OPTIX_ABI_VERSION[ \t]+[0-9]+")
    string(REGEX MATCH "[0-9]+$" _header_abi "${_abi_line}")
    list(FIND _ti_optix_supported_abis "${_header_abi}" _supported_index)
    if(_supported_index EQUAL -1)
        message(FATAL_ERROR
            "Unsupported OptiX header ABI ${_header_abi}; Forge supports "
            "ABI 93 (OptiX 8.1), ABI 105 (OptiX 9.0), and ABI 118 (OptiX 9.1)")
    endif()
    if(NOT "${expected_abi}" STREQUAL "any" AND
       NOT "${_header_abi}" STREQUAL "${expected_abi}")
        message(FATAL_ERROR
            "Expected OptiX ABI ${expected_abi} headers at ${root}, found ABI ${_header_abi}")
    endif()
    set(${out_abi} "${_header_abi}" PARENT_SCOPE)
endfunction()

function(_ti_add_optix_provider target_name root)
    _ti_validate_optix_root("${root}" any _header_abi)
    set(_generated_dir
        "${CMAKE_CURRENT_BINARY_DIR}/generated/optix_provider_${_header_abi}")
    # This custom PTX command does not enable CMake's CUDA language, so honor
    # its explicit host-compiler setting ourselves. The PTX toolchain may use
    # an older supported MSVC than the native runtime/shim toolchain.
    set(_ptx_host_options)
    if(CMAKE_CUDA_HOST_COMPILER)
        list(APPEND _ptx_host_options
             --compiler-bindir "${CMAKE_CUDA_HOST_COMPILER}")
    endif()
    set(_ptx_headers)
    # Keep the legacy four-payload module independent from typed-hit payloads.
    # Both artifacts retain the same portable PTX version and target contract.
    foreach(_typed IN ITEMS 0 1)
        set(_ptx "${_generated_dir}/device_program_${_typed}.ptx")
        set(_ptx_header "${_generated_dir}/device_program_${_typed}_ptx.h")
        list(APPEND _ptx_headers "${_ptx_header}")
        add_custom_command(
            OUTPUT "${_ptx}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${_generated_dir}"
            COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}"
                    ${_ptx_host_options}
                    --ptx --std=c++17 --use_fast_math
                    --gpu-architecture=compute_75
                    -DTI_FORGE_OPTIX_TYPED=${_typed}
                    -I "${root}/include"
                    "${CMAKE_CURRENT_SOURCE_DIR}/taichi/optix/provider/device_program.cu"
                    -o "${_ptx}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/taichi/optix/provider/device_program.cu"
            VERBATIM)
        add_custom_command(
            OUTPUT "${_ptx_header}"
            COMMAND ${CMAKE_COMMAND}
                    "-DINPUT_FILE=${_ptx}"
                    "-DOUTPUT_FILE=${_ptx_header}"
                    "-DSYMBOL_NAME=ti_forge_optix_device_${_typed}_ptx"
                    "-DEXPECTED_PTX_VERSION=${_ti_optix_expected_ptx_version}"
                    "-DEXPECTED_PTX_TARGET=sm_75"
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedText.cmake"
            DEPENDS "${_ptx}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedText.cmake"
            VERBATIM)
    endforeach()

    add_library(${target_name} SHARED
        "${CMAKE_CURRENT_SOURCE_DIR}/taichi/optix/provider/provider.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/taichi/optix/forge_optix_provider.h"
        ${_ptx_headers})
    target_compile_features(${target_name} PRIVATE cxx_std_17)
    target_compile_definitions(${target_name}
        PRIVATE TI_FORGE_OPTIX_PROVIDER_BUILD OPTIX_ENABLE_SDK_MIXING)
    target_include_directories(${target_name} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${root}/include"
        "${_generated_dir}"
        "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(${target_name} PRIVATE CUDA::cuda_driver ${CMAKE_DL_LIBS})
    set_target_properties(${target_name} PROPERTIES
        OUTPUT_NAME "taichi_forge_optix_provider_abi1_optix${_header_abi}")
    add_dependencies(taichi_forge_optix_providers ${target_name})

    install(TARGETS ${target_name}
            RUNTIME DESTINATION ${INSTALL_LIB_DIR}/hardware_providers
                    COMPONENT runtime
            LIBRARY DESTINATION ${INSTALL_LIB_DIR}/hardware_providers
                    COMPONENT runtime)
endfunction()

include(FetchContent)

function(_ti_resolve_optix_headers abi version commit out_root)
    set(_configured_root "${TI_OPTIX_ROOT_${abi}}")
    if(_configured_root)
        _ti_validate_optix_root("${_configured_root}" "${abi}" _unused_abi)
        set(${out_root} "${_configured_root}" PARENT_SCOPE)
        return()
    endif()

    set(_dependency "ti_optix_${abi}_headers")
    FetchContent_Declare(${_dependency}
        GIT_REPOSITORY https://github.com/NVIDIA/optix-dev.git
        GIT_TAG "${commit}"
        GIT_SHALLOW FALSE
        SOURCE_SUBDIR _forge_headers_only)
    FetchContent_MakeAvailable(${_dependency})
    set(_source_dir "${${_dependency}_SOURCE_DIR}")
    _ti_validate_optix_root("${_source_dir}" "${abi}" _unused_abi)
    message(STATUS
        "Forge bundled OptiX ABI ${abi} uses optix-dev ${version} at ${commit}")
    set(${out_root} "${_source_dir}" PARENT_SCOPE)
endfunction()

# Full immutable commit IDs keep the release build reproducible while all
# three adapters remain files in the same platform runtime wheel.
_ti_resolve_optix_headers(
    93 8.1.0 50021ea0af6d41609a97777ceebbdf1e1d34efe7 _root_93)
_ti_resolve_optix_headers(
    105 9.0.0 fff65c2a7c592f1ea5f1661ad7d2381cf965f9bd _root_105)
_ti_resolve_optix_headers(
    118 9.1.0 f1f6dd803f3159992d248178f6e09421c6eb8b6d _root_118)

_ti_add_optix_provider(taichi_forge_optix_provider_93 "${_root_93}")
_ti_add_optix_provider(taichi_forge_optix_provider_105 "${_root_105}")
_ti_add_optix_provider(taichi_forge_optix_provider_118 "${_root_118}")
