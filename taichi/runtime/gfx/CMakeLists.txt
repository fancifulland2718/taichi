# ./taichi/runtime/gfx/CMakeLists.txt

# Phase 2a — pointer-on-Vulkan device-side allocator. 路线 B B-1（2026-04-30）
# 默认翻 ON：DeviceNodeAllocator 抽象层接通后 vanilla 行为字节等价（见
# compile_doc/SNode_Vulkan_规划.md §10）。OFF 时 snode_allocator.cpp 不入编译，
# 同时 GfxRuntime 不持有 node_allocators_，作为 ABI fallback 保留。
option(TI_WITH_VULKAN_POINTER "Enable pointer-SNode-on-Vulkan allocator (Route B)" ON)

add_library(gfx_runtime)
target_sources(gfx_runtime
  PRIVATE
    runtime.cpp
    snode_tree_manager.cpp
    aot_module_builder_impl.cpp
    aot_module_loader_impl.cpp
    kernel_launcher.cpp
  )
if(TI_WITH_VULKAN_POINTER)
  target_sources(gfx_runtime PRIVATE snode_allocator.cpp)
  target_compile_definitions(gfx_runtime PUBLIC TI_WITH_VULKAN_POINTER=1)
endif()
#TODO 4832, some dependencies here should not be required as they
# are build requirements of other targets.
target_include_directories(gfx_runtime
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/SPIRV-Tools/include
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/FP16/include
  PRIVATE
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
        ${LLVM_INCLUDE_DIRS}
  )
target_link_libraries(gfx_runtime PRIVATE taichi_common)
