# ./taichi/runtime/llvm/CMakeLists.txt

add_library(llvm_runtime)
target_sources(llvm_runtime
  PRIVATE
    llvm_runtime_executor.cpp
    llvm_offline_cache.cpp
    llvm_context.cpp
    llvm_opt_pipeline.cpp
    llvm_aot_module_loader.cpp
    llvm_aot_module_builder.cpp
    snode_tree_buffer_manager.cpp
    kernel_launcher.cpp
  )

target_include_directories(llvm_runtime
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${LLVM_INCLUDE_DIRS}
  )

target_link_libraries(llvm_runtime PRIVATE ${llvm_libs})

# llvm_context.cpp uses llvm::legacy::PassManager which is deprecated on
# LLVM 19 and scheduled for Phase 6 migration to the new PassBuilder.
target_silence_llvm_deprecated_warnings(llvm_runtime)


if (APPLE AND "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "arm64")
    target_link_libraries(llvm_runtime PRIVATE ${llvm_aarch64_libs})
endif()

if (TI_WITH_CUDA)
    target_link_libraries(llvm_runtime PRIVATE ${llvm_ptx_libs})
endif()

if (TI_WITH_AMDGPU)
    target_link_libraries(llvm_runtime PRIVATE ${llvm_amdgpu_libs})
endif()

if (TI_WITH_DX12)
    target_link_libraries(llvm_runtime PRIVATE ${llvm_directx_libs})
endif()

target_link_libraries(llvm_runtime PRIVATE ti_device_api)
