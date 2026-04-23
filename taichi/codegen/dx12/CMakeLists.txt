# ./taichi/codegen/dx12/CMakeLists.txt

add_library(dx12_codegen)
target_sources(dx12_codegen
  PRIVATE
    codegen_dx12.cpp
    dx12_global_optimize_module.cpp
    dx12_lower_intrinsic.cpp
    dx12_lower_runtime_context.cpp
  )

# dx12_global_optimize_module.cpp uses llvm::legacy::PassManager /
# PassManagerBuilder, deprecated on LLVM 19. Phase 6 migration.
target_silence_llvm_deprecated_warnings(dx12_codegen)

target_include_directories(dx12_codegen
  PRIVATE
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${PROJECT_SOURCE_DIR}
    ${LLVM_INCLUDE_DIRS}
  )

target_link_libraries(dx12_codegen PRIVATE taichi_util)
target_link_libraries(dx12_codegen PRIVATE dx12_runtime)
