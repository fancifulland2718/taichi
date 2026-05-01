#pragma once
#include "taichi/util/lang_util.h"

#include <vector>
#include <chrono>

#include "taichi/rhi/device.h"
#include "taichi/codegen/spirv/snode_struct_compiler.h"
#include "taichi/codegen/spirv/kernel_utils.h"
#include "taichi/program/compile_config.h"
#include "taichi/struct/snode_tree.h"
#include "taichi/program/snode_expr_utils.h"
#include "taichi/program/program_impl.h"
#include "taichi/program/kernel_launcher.h"
#if defined(TI_WITH_VULKAN_POINTER)
#include "taichi/runtime/gfx/snode_allocator.h"
#endif

namespace taichi::lang {
namespace gfx {

using namespace taichi::lang::spirv;

using BufferType = TaskAttributes::BufferType;
using BufferInfo = TaskAttributes::BufferInfo;
using BufferBind = TaskAttributes::BufferBind;
using BufferInfoHasher = TaskAttributes::BufferInfoHasher;

using high_res_clock = std::chrono::high_resolution_clock;

// TODO: In the future this isn't necessarily a pointer, since DeviceAllocation
// is already a pretty cheap handle>
using InputBuffersMap =
    std::unordered_map<BufferInfo, DeviceAllocation *, BufferInfoHasher>;

class SNodeTreeManager;

class CompiledTaichiKernel {
 public:
  struct Params {
    const TaichiKernelAttributes *ti_kernel_attribs{nullptr};
    std::vector<std::vector<uint32_t>> spirv_bins;
    std::size_t num_snode_trees{0};

    Device *device{nullptr};
    std::vector<DeviceAllocation *> root_buffers;
    DeviceAllocation *global_tmps_buffer{nullptr};
    DeviceAllocation *listgen_buffer{nullptr};
#if defined(TI_WITH_VULKAN_POINTER)
    // B-3.b (2026-05): 路线 B 阶段开启 vulkan_pointer_independent_pool 后，
    // 每个走独立池的 pointer SNode 对应一块独立 DeviceAllocation；在这里打
    // 包传入，CompiledTaichiKernel 构造时按 sid 注册到 input_buffers_ 以供
    // descriptor set fallback 路径绑定。key = SNode id。默认 (OFF) 为空。
    std::vector<std::pair<int, DeviceAllocation *>> node_allocator_pool_buffers;
    // C-2.5 (2026-05): chunked allocator 的全部 chunk DeviceAllocation 列表。
    // key = SNode id；value = {chunk[0], chunk[1], ..., chunk[N-1]}。
    // 只在 max_chunks > 1 时填充；单 chunk allocator（含 Bump）走上面的
    // node_allocator_pool_buffers 单 buffer 路径。CompiledTaichiKernel 在
    // dispatch 时按 BufferBind.chunk_count>0 调 rw_buffer_array(N)。
    std::vector<std::pair<int, std::vector<DeviceAllocation>>>
        node_allocator_chunk_arrays;
#endif

    PipelineCache *backend_cache{nullptr};
  };

  explicit CompiledTaichiKernel(const Params &ti_params);

  const TaichiKernelAttributes &ti_kernel_attribs() const;

  size_t num_pipelines() const;

  size_t get_args_buffer_size() const;
  size_t get_ret_buffer_size() const;

  Pipeline *get_pipeline(int i);

  DeviceAllocation *get_buffer_bind(const BufferInfo &bind) {
    return input_buffers_[bind];
  }

  // C-2.5 (2026-05): retrieve all chunk DeviceAllocations for a chunked
  // NodeAllocatorPool binding. Returns nullptr if `bind` is not a chunked
  // pool. Key = sid stored in bind.root_id.
  const std::vector<DeviceAllocation> *get_chunk_array(
      const BufferInfo &bind) const {
    auto it = chunk_arrays_.find(bind);
    if (it == chunk_arrays_.end()) {
      return nullptr;
    }
    return &it->second;
  }

 private:
  TaichiKernelAttributes ti_kernel_attribs_;
  std::vector<TaskAttributes> tasks_attribs_;

  [[maybe_unused]] Device *device_;

  InputBuffersMap input_buffers_;
  // C-2.5 (2026-05): per-binding chunk DeviceAllocation lists for chunked
  // NodeAllocatorPool bindings. Empty when no SNode uses chunked allocator.
  std::unordered_map<BufferInfo, std::vector<DeviceAllocation>, BufferInfoHasher>
      chunk_arrays_;

  size_t args_buffer_size_{0};
  size_t ret_buffer_size_{0};
  std::vector<std::unique_ptr<Pipeline>> pipelines_;
};

class TI_DLL_EXPORT GfxRuntime {
 public:
  struct Params {
    Device *device{nullptr};
    KernelProfilerBase *profiler{nullptr};
    // R2.a: Vulkan launch args/ret buffer pool.
    bool enable_buffer_pool{false};
    int buffer_pool_capacity{64};
  };

  explicit GfxRuntime(const Params &params);
  // To make Pimpl + std::unique_ptr work
  ~GfxRuntime();

  using KernelHandle = KernelLauncher::Handle;

  struct RegisterParams {
    TaichiKernelAttributes kernel_attribs;
    std::vector<std::vector<uint32_t>> task_spirv_source_codes;
    std::size_t num_snode_trees{0};
  };

  KernelHandle register_taichi_kernel(RegisterParams params);

  void launch_kernel(KernelHandle handle, LaunchContextBuilder &host_ctx);

  void buffer_copy(DevicePtr dst, DevicePtr src, size_t size);
  void copy_image(DeviceAllocation dst,
                  DeviceAllocation src,
                  const ImageCopyParams &params);

  DeviceAllocation create_image(const ImageParams &params);
  void track_image(DeviceAllocation image, ImageLayout layout);
  void untrack_image(DeviceAllocation image);
  void transition_image(DeviceAllocation image, ImageLayout layout);

  void synchronize();

  StreamSemaphore flush();

  Device *get_ti_device() const;

  void add_root_buffer(size_t root_buffer_size);

  DeviceAllocation *get_root_buffer(int id) const;

  size_t get_root_buffer_size(int id) const;

  void enqueue_compute_op_lambda(
      std::function<void(Device *device, CommandList *cmdlist)> op,
      const std::vector<ComputeOpImageRef> &image_refs);

  bool used_in_kernel(DeviceAllocationId id) {
    return ndarrays_in_use_.count(id) > 0 || argpacks_in_use_.count(id) > 0;
  }

  static std::pair<const lang::StructType *, size_t>
  get_struct_type_with_data_layout(const lang::StructType *old_ty,
                                   const std::string &layout);

  static std::tuple<const lang::StructType *, size_t, size_t>
  get_struct_type_with_data_layout_impl(const lang::StructType *old_ty,
                                        const std::string &layout);

 private:
  friend class taichi::lang::gfx::SNodeTreeManager;

  void ensure_current_cmdlist();
  void submit_current_cmdlist_if_timeout();

  void init_nonroot_buffers();

  Device *device_{nullptr};
  KernelProfilerBase *profiler_;

  std::unique_ptr<PipelineCache> backend_cache_{nullptr};

  std::vector<std::unique_ptr<DeviceAllocationGuard>> root_buffers_;
#if defined(TI_WITH_VULKAN_POINTER)
  // 路线 B B-1（2026-04-30）：每棵 SNode tree 上每个 pointer SNode 对应
  // 一个 DeviceNodeAllocator。outer key = root_id（与 root_buffers_ 同步），
  // 内层 map key = SNode id。当前阶段池仍寄居 root_buffer 子区间，allocator
  // 只是「字节等价的间接层」；B-3 阶段池将迁出 root_buffer。
  // 注：用 unordered_map 作外层是为了规避 MSVC 上 vector 增长时对不含
  // noexcept move 的 map 的硬性 copy 需求。
  std::unordered_map<int,
                     std::unordered_map<int, std::unique_ptr<DeviceNodeAllocator>>>
      node_allocators_;
#endif
  std::unique_ptr<DeviceAllocationGuard> global_tmps_buffer_;
  // FIXME: Support proper multiple lists
  std::unique_ptr<DeviceAllocationGuard> listgen_buffer_;

  std::vector<std::unique_ptr<DeviceAllocationGuard>> ctx_buffers_;

  // R2.a: Vulkan launch args/ret buffer pool. Three-stage to be GPU-safe:
  //   pending_pool_   : in-flight on current_cmdlist_ (recording)
  //   submitted_pool_ : submitted to stream, GPU may still be using
  //   free_pool_      : safe to reuse (only filled after wait_idle)
  // Each entry is keyed by (size, usage). Pool cap drops oldest free entries.
  struct PooledBuffer {
    std::unique_ptr<DeviceAllocationGuard> guard;
    size_t size{0};
    AllocUsage usage{AllocUsage::None};
  };
  bool buffer_pool_enabled_{false};
  size_t buffer_pool_capacity_{64};
  std::vector<PooledBuffer> pending_pool_;
  std::vector<PooledBuffer> submitted_pool_;
  std::vector<PooledBuffer> free_pool_;
  size_t buffer_pool_hits_{0};
  size_t buffer_pool_misses_{0};

  // Try to take a buffer from free_pool_ matching (size, usage). Returns
  // nullptr if pool disabled or no match.
  std::unique_ptr<DeviceAllocationGuard> try_take_pooled_buffer(
      size_t size,
      AllocUsage usage);
  // Move pending_pool_ -> submitted_pool_ (called on flush).
  void flush_pending_pool_to_submitted();
  // Move submitted_pool_ + pending_pool_ -> free_pool_ (called after wait_idle).
  void recycle_pools_to_free();

  std::unique_ptr<CommandList> current_cmdlist_{nullptr};
  high_res_clock::time_point current_cmdlist_pending_since_;

  std::vector<std::unique_ptr<CompiledTaichiKernel>> ti_kernels_;

  std::unordered_map<DeviceAllocation *, size_t> root_buffers_size_map_;
  std::unordered_map<DeviceAllocationId, ImageLayout> last_image_layouts_;
  // [Note] Why do we need to track ndarrays that are in use?
  // Since we separate cmdlist is async, taichi needs a way to know whether
  // ndarrays are still used by pending kernels to be executed. So we use
  // ndarray_in_use_ to track this so that we can free memory allocated for
  // ndarray whenever it's safe to do so.
  std::unordered_set<DeviceAllocationId> ndarrays_in_use_;
  std::unordered_set<DeviceAllocationId> argpacks_in_use_;
};

GfxRuntime::RegisterParams run_codegen(
    Kernel *kernel,
    Arch arch,
    const DeviceCapabilityConfig &caps,
    const std::vector<CompiledSNodeStructs> &compiled_structs,
    const CompileConfig &compile_config);

}  // namespace gfx
}  // namespace taichi::lang
