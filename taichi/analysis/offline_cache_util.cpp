#include "offline_cache_util.h"

#include "taichi/common/core.h"
#include "taichi/common/serialization.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/transforms.h"
#include "taichi/program/compile_config.h"
#include "taichi/program/kernel.h"
#include "taichi/rhi/device_capability.h"
#include "taichi/system/profiler.h"

#include "picosha2.h"

#include <algorithm>
#include <vector>

namespace taichi::lang {

static std::vector<std::uint8_t> get_offline_cache_key_of_parameter_list(
    const std::vector<CallableBase::Parameter> &parameter_list) {
  BinaryOutputSerializer serializer;
  serializer.initialize();
  serializer(parameter_list);
  serializer.finalize();
  return serializer.data;
}

static std::vector<std::uint8_t> get_offline_cache_key_of_rets(
    const std::vector<CallableBase::Ret> &ret_list) {
  BinaryOutputSerializer serializer;
  serializer.initialize();
  serializer(ret_list);
  serializer.finalize();
  return serializer.data;
}

static std::vector<std::uint8_t> get_offline_cache_key_of_compile_config(
    const CompileConfig &config) {
  BinaryOutputSerializer serializer;
  serializer.initialize();
  serializer(config.arch);
  serializer(config.debug);
  serializer(config.cfg_optimization);
  serializer(config.check_out_of_bound);
  serializer(config.opt_level);
  serializer(config.external_optimization_level);
  serializer(config.llvm_opt_level);
  serializer(config.compile_tier);
  serializer(config.use_fused_passes);
  serializer(config.move_loop_invariant_outside_if);
  serializer(config.demote_dense_struct_fors);
  serializer(config.advanced_optimization);
  serializer(config.constant_folding);
  serializer(config.kernel_profiler);
  serializer(config.fast_math);
  serializer(config.flatten_if);
  serializer(config.make_thread_local);
  serializer(config.make_block_local);
  serializer(config.detect_read_only);
  serializer(config.default_fp->to_string());
  serializer(config.default_ip.to_string());
  if (arch_is_cpu(config.arch)) {
    serializer(config.default_cpu_block_dim);
    serializer(config.cpu_max_num_threads);
  } else if (arch_is_gpu(config.arch)) {
    serializer(config.default_gpu_block_dim);
    serializer(config.gpu_max_reg);
    serializer(config.saturating_grid_dim);
    serializer(config.cpu_max_num_threads);
  }
  serializer(config.ad_stack_size);
  serializer(config.default_ad_stack_size);
  // NOTE: config.random_seed is intentionally NOT part of the offline cache
  // key.  It only affects the runtime PRNG seed (see
  // LlvmRuntimeExecutor::materialize_runtime); the generated IR / LLVM module /
  // SPIR-V are identical regardless of its value.  Including it here caused
  // spurious cache misses whenever the user changes ti.init(random_seed=...)
  // between sessions.  [P1.a cache-key trim]
  if (config.arch == Arch::opengl || config.arch == Arch::gles) {
    serializer(config.allow_nv_shader_extension);
  }
  serializer(config.make_mesh_block_local);
  serializer(config.optimize_mesh_reordered_mapping);
  serializer(config.mesh_localize_to_end_mapping);
  serializer(config.mesh_localize_from_end_mapping);
  serializer(config.mesh_localize_all_attr_mappings);
  serializer(config.demote_no_access_mesh_fors);
  serializer(config.experimental_auto_mesh_local);
  serializer(config.auto_mesh_local_default_occupacy);
  serializer(config.real_matrix_scalarize);
  // P9.A (F2/F3): auto_real_function gating + inline budget influence
  // FuncCallStmt presence and inliner behavior; both must invalidate cache.
  serializer(config.auto_real_function);
  serializer(config.auto_real_function_threshold_us);
  serializer(config.auto_real_function_inline_budget);
  serializer(config.force_scalarize_matrix);
  serializer(config.half2_vectorization);
  // B2 (2026-04-26): SPIR-V disabled-pass list affects emitted SPIR-V on
  // SPIR-V backends. Sort first so user-supplied list ordering doesn't
  // produce spurious cache misses. Empty list (default) hashes to a
  // stable empty entry, so legacy users see no cache invalidation.
  {
    std::vector<std::string> sorted_disabled = config.spirv_disabled_passes;
    std::sort(sorted_disabled.begin(), sorted_disabled.end());
    serializer(sorted_disabled);
  }
  // B-2.b (2026-05): the 4 vulkan_pointer_* runtime fields drive both
  // root-buffer layout and pointer-SNode SPIR-V codegen. They MUST be
  // part of the cache key, otherwise toggling vulkan_pointer_ambient_zone
  // / _freelist / _cas_marker / _pool_fraction silently reuses kernels
  // compiled under the previous flag value. Default values (True/True/
  // True/1.0) hash deterministically so legacy users see no invalidation.
  if (config.arch == Arch::vulkan) {
    serializer(config.vulkan_pointer_freelist);
    serializer(config.vulkan_pointer_ambient_zone);
    serializer(config.vulkan_pointer_cas_marker);
    serializer(config.vulkan_pointer_pool_fraction);
    // B-3.b (2026-05): independent_pool 影响 SpirvAllocatorContract.
    // pool_buffer_binding_id 与 SNodeTree allocator 申请独立 DeviceAllocation。
    // 即使 codegen 在 B-3.b 不读 binding_id，提前纳入 cache key 避免 B-3.c
    // 切换 codegen 后命中旧缓存。默认 false 哈希稳定。
    serializer(config.vulkan_pointer_independent_pool);
    // C-2.1 (2026-05): allocator_kind 选不同 allocator 实现 → SPIR-V kernel
    // 寻址（C-2.3 起）和 SNodeTree allocator 构造均不同；必须进入 cache key。
    // 默认 "bump" 字符串哈希稳定，旧用户无 invalidation。
    serializer(config.vulkan_pointer_allocator_kind);
    serializer(config.vulkan_pointer_max_chunks);
  }
  serializer.finalize();

  return serializer.data;
}

static std::vector<std::uint8_t> get_offline_cache_key_of_device_caps(
    const DeviceCapabilityConfig &caps) {
  BinaryOutputSerializer serializer;
  serializer.initialize();
  serializer(caps.devcaps);
  serializer.finalize();
  return serializer.data;
}

static void get_offline_cache_key_of_snode_impl(
    const SNode *snode,
    BinaryOutputSerializer &serializer,
    std::unordered_set<int> &visited) {
  if (auto iter = visited.find(snode->id); iter != visited.end()) {
    serializer(snode->id);  // Use snode->id as placeholder to identify a snode
    return;
  }

  visited.insert(snode->id);
  for (auto &c : snode->ch) {
    get_offline_cache_key_of_snode_impl(c.get(), serializer, visited);
  }
  for (int i = 0; i < taichi_max_num_indices; ++i) {
    auto &extractor = snode->extractors[i];
    serializer(extractor.num_elements_from_root);
    serializer(extractor.shape);
    serializer(extractor.acc_shape);
    serializer(extractor.active);
  }
  serializer(snode->index_offsets);
  serializer(snode->num_active_indices);
  serializer(snode->physical_index_position);
  serializer(snode->id);
  serializer(snode->depth);
  serializer(snode->name);
  serializer(snode->num_cells_per_container);
  // C-1 (2026-05): per-pointer-SNode 池容量 hint 影响 SPIR-V layout 偏移，
  // 必须纳入 cache key。-1 默认值哈希稳定，旧 SNodeTree 不受影响。
  serializer(snode->vk_max_active_hint);
  serializer(snode->chunk_size);
  serializer(snode->cell_size_bytes);
  serializer(snode->offset_bytes_in_parent_cell);
  serializer(snode->dt->to_string());
  serializer(snode->has_ambient);
  if (!snode->ambient_val.dt->is_primitive(PrimitiveTypeID::unknown)) {
    serializer(snode->ambient_val.stringify());
  }
  if (snode->grad_info && !snode->grad_info->is_primal()) {
    if (auto *adjoint_snode = snode->grad_info->adjoint_snode()) {
      get_offline_cache_key_of_snode_impl(adjoint_snode, serializer, visited);
    }
    if (auto *dual_snode = snode->grad_info->dual_snode()) {
      get_offline_cache_key_of_snode_impl(dual_snode, serializer, visited);
    }
  }
  if (snode->physical_type) {
    serializer(snode->physical_type->to_string());
  }
  serializer(snode->id_in_bit_struct);
  serializer(snode->is_bit_level);
  serializer(snode->is_path_all_dense);
  serializer(snode->node_type_name);
  serializer(snode->type);
  serializer(snode->_morton);
  serializer(snode->get_snode_tree_id());
}

std::string get_hashed_offline_cache_key_of_snode(const SNode *snode) {
  TI_ASSERT(snode);

  BinaryOutputSerializer serializer;
  serializer.initialize();
  {
    std::unordered_set<int> visited;
    get_offline_cache_key_of_snode_impl(snode, serializer, visited);
  }
  serializer.finalize();

  picosha2::hash256_one_by_one hasher;
  hasher.process(serializer.data.begin(), serializer.data.end());
  hasher.finish();

  return picosha2::get_hash_hex_string(hasher);
}

std::string get_hashed_offline_cache_key(const CompileConfig &config,
                                         const DeviceCapabilityConfig &caps,
                                         Kernel *kernel) {
  TI_AUTO_PROF;
  std::vector<std::uint8_t> kernel_params_string, kernel_rets_string;
  std::string kernel_body_string;
  if (kernel) {  // param_list, rets, body
    kernel_params_string =
        get_offline_cache_key_of_parameter_list(kernel->parameter_list);
    kernel_rets_string = get_offline_cache_key_of_rets(kernel->rets);
    std::ostringstream oss;
    {
      TI_PROFILER("gen_offline_cache_key.body");
      gen_offline_cache_key(kernel->ir.get(), &oss);
    }
    kernel_body_string = oss.str();
  }

  auto compile_config_key = get_offline_cache_key_of_compile_config(config);
  auto device_caps_key = get_offline_cache_key_of_device_caps(caps);
  std::string autodiff_mode =
      std::to_string(static_cast<std::size_t>(kernel->autodiff_mode));
  // [P-Compile-2-A] Mix in cache schema version so future key-algo changes
  // automatically invalidate old .tic files without bumping TI_VERSION_*.
  // v1 is intentionally hash-equivalent to the pre-P-Compile-2-A algorithm
  // so existing .tic caches keep hitting; only v>=2 actually injects a
  // schema tag. See offline_cache_util.h for rationale.
  picosha2::hash256_one_by_one hasher;
  if constexpr (kOfflineCacheSchemaVersion >= 2) {
    std::string schema_tag =
        "tcs:" + std::to_string(kOfflineCacheSchemaVersion);
    hasher.process(schema_tag.begin(), schema_tag.end());
  }
  hasher.process(compile_config_key.begin(), compile_config_key.end());
  hasher.process(device_caps_key.begin(), device_caps_key.end());
  hasher.process(kernel_params_string.begin(), kernel_params_string.end());
  hasher.process(kernel_rets_string.begin(), kernel_rets_string.end());
  hasher.process(kernel_body_string.begin(), kernel_body_string.end());
  hasher.process(autodiff_mode.begin(), autodiff_mode.end());
  hasher.finish();

  auto res = picosha2::get_hash_hex_string(hasher);
  res.insert(res.begin(), 'T');  // The key must start with a letter
  return res;
}

}  // namespace taichi::lang
