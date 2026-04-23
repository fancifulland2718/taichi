# ./taichi/runtime/cuda/CMakeLists.txt

add_library(cuda_runtime)
target_sources(cuda_runtime
  PRIVATE
    jit_cuda.cpp
    kernel_launcher.cpp
  )

# jit_cuda.cpp uses llvm::legacy::PassManager / PassManagerBuilder,
# deprecated on LLVM 19. Migration is tracked in Phase 6.
target_silence_llvm_deprecated_warnings(cuda_runtime)

target_include_directories(cuda_runtime
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${LLVM_INCLUDE_DIRS}
  )
