# ./taichi/codegen/cpu/CMakeLists.txt

add_library(cpu_codegen)
target_sources(cpu_codegen
  PRIVATE
    codegen_cpu.cpp
  )

# codegen_cpu.cpp uses llvm::legacy::PassManager / PassManagerBuilder,
# deprecated on LLVM 19. Migration is tracked in Phase 6.
target_silence_llvm_deprecated_warnings(cpu_codegen)

target_include_directories(cpu_codegen
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${LLVM_INCLUDE_DIRS}
  )

target_link_libraries(cpu_codegen PRIVATE taichi_util)
target_link_libraries(cpu_codegen PRIVATE cpu_runtime)
