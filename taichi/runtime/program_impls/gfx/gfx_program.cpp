#include "gfx_program.h"

#include "taichi/analysis/offline_cache_util.h"
#include "taichi/codegen/spirv/kernel_compiler.h"
#include "taichi/rhi/opengl/opengl_api.h"
#include "taichi/runtime/gfx/aot_module_builder_impl.h"
#include "taichi/runtime/gfx/kernel_launcher.h"
#include "taichi/rhi/common/host_memory_pool.h"

namespace taichi::lang {

GfxProgramImpl::GfxProgramImpl(CompileConfig &config) : ProgramImpl(config) {
}

// 2026-05-02 §13.4: chunked allocator emits a SPIR-V descriptor array
// (kChunkedArrayPtr) at the NodeAllocatorPool binding regardless of
// max_chunks; that binding only exists when independent_pool=true. Force
// it on for any chunked policy and warn-once if the user explicitly
// disabled it, so ptr_to_chunk_idx_ entries always pair with a
// kChunkedArrayPtr buffer (invariant required by
// spirv_codegen.cpp::at_buffer).
static void coerce_pointer_policy_for_chunked(spirv::PointerLayoutPolicy &p) {
  if (p.allocator_kind == "chunked" && !p.independent_pool) {
    static bool warned = false;
    if (!warned) {
      TI_WARN(
          "vulkan_pointer_allocator_kind=chunked requires "
          "vulkan_pointer_independent_pool=true (max_chunks={}); coercing "
          "to true.",
          p.max_chunks);
      warned = true;
    }
    p.independent_pool = true;
  }
}

void GfxProgramImpl::compile_snode_tree_types(SNodeTree *tree) {
  // B-2.b: build pointer SNode allocator policy from CompileConfig.
  spirv::PointerLayoutPolicy policy;
  policy.freelist = config->vulkan_pointer_freelist;
  policy.ambient_zone = config->vulkan_pointer_ambient_zone;
  policy.cas_marker = config->vulkan_pointer_cas_marker;
  policy.pool_fraction = config->vulkan_pointer_pool_fraction;
  policy.independent_pool = config->vulkan_pointer_independent_pool;
  policy.allocator_kind = config->vulkan_pointer_allocator_kind;
  policy.max_chunks = config->vulkan_pointer_max_chunks;
  policy.deterministic_slot = config->vulkan_pointer_deterministic_slot;
  coerce_pointer_policy_for_chunked(policy);
  if (runtime_) {
    snode_tree_mgr_->materialize_snode_tree(tree, policy);
  } else {
    gfx::CompiledSNodeStructs compiled_structs =
        gfx::compile_snode_structs(*tree->root(), policy);
    aot_compiled_snode_structs_.push_back(compiled_structs);
  }
}

void GfxProgramImpl::materialize_snode_tree(SNodeTree *tree,
                                            uint64 *result_buffer) {
  spirv::PointerLayoutPolicy policy;
  policy.freelist = config->vulkan_pointer_freelist;
  policy.ambient_zone = config->vulkan_pointer_ambient_zone;
  policy.cas_marker = config->vulkan_pointer_cas_marker;
  policy.pool_fraction = config->vulkan_pointer_pool_fraction;
  policy.independent_pool = config->vulkan_pointer_independent_pool;
  policy.allocator_kind = config->vulkan_pointer_allocator_kind;
  policy.max_chunks = config->vulkan_pointer_max_chunks;
  policy.deterministic_slot = config->vulkan_pointer_deterministic_slot;
  coerce_pointer_policy_for_chunked(policy);
  snode_tree_mgr_->materialize_snode_tree(tree, policy);
}

std::unique_ptr<AotModuleBuilder> GfxProgramImpl::make_aot_module_builder(
    const DeviceCapabilityConfig &caps) {
  if (runtime_) {
    return std::make_unique<gfx::AotModuleBuilderImpl>(
        snode_tree_mgr_->get_compiled_structs(),
        get_kernel_compilation_manager(), *config, caps);
  } else {
    return std::make_unique<gfx::AotModuleBuilderImpl>(
        aot_compiled_snode_structs_, get_kernel_compilation_manager(), *config,
        caps);
  }
}

DeviceAllocation GfxProgramImpl::allocate_memory_on_device(
    std::size_t alloc_size,
    uint64 *result_buffer) {
  DeviceAllocation alloc;
  RhiResult res = get_compute_device()->allocate_memory(
      {alloc_size, /*host_write=*/false, /*host_read=*/false,
       /*export_sharing=*/false},
      &alloc);
  TI_ASSERT(res == RhiResult::success);
  return alloc;
}

DeviceAllocation GfxProgramImpl::allocate_texture(const ImageParams &params) {
  return runtime_->create_image(params);
}

void GfxProgramImpl::finalize() {
  runtime_.reset();
  device_.reset();
}

GfxProgramImpl::~GfxProgramImpl() {
  // Calling virtual methods in destructors is unsafe. Use static-binding here.
  GfxProgramImpl::finalize();
}

std::unique_ptr<KernelCompiler> GfxProgramImpl::make_kernel_compiler() {
  spirv::KernelCompiler::Config cfg;
  cfg.compiled_struct_data = runtime_ ? &snode_tree_mgr_->get_compiled_structs()
                                      : &aot_compiled_snode_structs_;
  return std::make_unique<spirv::KernelCompiler>(std::move(cfg));
}

std::unique_ptr<KernelLauncher> GfxProgramImpl::make_kernel_launcher() {
  gfx::KernelLauncher::Config cfg;
  cfg.gfx_runtime_ = runtime_.get();
  return std::make_unique<gfx::KernelLauncher>(std::move(cfg));
}

DeviceCapabilityConfig GfxProgramImpl::get_device_caps() {
  TI_ASSERT(runtime_);
  return runtime_->get_ti_device()->get_caps();
}

}  // namespace taichi::lang
