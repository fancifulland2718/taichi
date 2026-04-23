# ./taichi/runtime/amdgpu/CMakeLists.txt

add_library(amdgpu_runtime)
target_sources(amdgpu_runtime
  PRIVATE
    jit_amdgpu.cpp
    kernel_launcher.cpp
  )

# jit_amdgpu.cpp uses llvm::legacy::PassManager / PassManagerBuilder,
# deprecated on LLVM 19. Migration is tracked in Phase 6.
target_silence_llvm_deprecated_warnings(amdgpu_runtime)

target_include_directories(amdgpu_runtime
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${LLVM_INCLUDE_DIRS}
  )
