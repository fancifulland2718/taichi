// =============================================================================
// Phase 2a — pointer-on-Vulkan device-side node allocator (BUMP-ONLY IMPL)
// =============================================================================
//
// 本文件实现 snode_allocator.h 声明的 BumpOnlyDeviceNodeAllocator。
//
// 路线锁定依据：tests/p4/phase2a_workload_analysis.py 与 phase2a_workload_mpm.py
//   实测两类典型负载（实时渲染 logo / MPM 物理仿真）帧间 cell 重合率 97-99%，
//   池占用率上界 == 静态 num_cells（worst case 100%）→ 选择 bump-only 池。
//
// 与 vanilla 行为关系：
//   - 受 CMake 选项 TI_WITH_VULKAN_POINTER 控制，默认 OFF → 本 .cpp 不参与编译；
//   - 即使 ON 编译进 gfx_runtime，也不会被任何 codegen 路径引用，
//     直到 Phase 2b 把 pointer SNode 接入 codegen + GfxRuntime hook。
//   - vanilla / 1d-B 行为完全不受影响。
// =============================================================================

#include "taichi/runtime/gfx/snode_allocator.h"

#include "taichi/rhi/public_device.h"

namespace taichi::lang {
namespace gfx {

// -----------------------------------------------------------------------------
// BumpOnlyDeviceNodeAllocator
// -----------------------------------------------------------------------------

BumpOnlyDeviceNodeAllocator::BumpOnlyDeviceNodeAllocator(const Params &p)
    : params_(p) {
  TI_ASSERT(p.device != nullptr);
  TI_ASSERT(p.pool_capacity > 0);
  TI_ASSERT(p.cell_payload_bytes > 0);
  // watermark 与 pool_data 必须 4-byte 对齐（vkCmdFillBuffer 要求；
  // 同时 SPIR-V u32 atomic 也要求 4B 对齐）。
  TI_ASSERT(p.watermark_offset % 4 == 0);
  TI_ASSERT(p.pool_data_offset % 4 == 0);
  TI_ASSERT(p.cell_payload_bytes % 4 == 0);
  // watermark 占 4 字节，必须严格在 pool_data 之前不重叠
  TI_ASSERT(p.pool_data_offset >= p.watermark_offset + 4);

  // B-3.b (2026-05): 可选申请独立 pool DeviceAllocation。B-3.b 阶段
  // codegen 仍读 root_buffer，该 buffer 只被 runtime 注入 descriptor set
  // （实际 dead allocation），B-3.c 才切 codegen 路径。
  if (p.use_independent_pool) {
    TI_ASSERT_INFO(
        p.independent_pool_size > 0,
        "BumpOnlyDeviceNodeAllocator: use_independent_pool requires "
        "independent_pool_size > 0 (snode_id={})",
        p.snode_id);
    Device::AllocParams alloc_params;
    alloc_params.size = p.independent_pool_size;
    alloc_params.host_write = false;
    alloc_params.host_read = false;
    alloc_params.export_sharing = false;
    alloc_params.usage = AllocUsage::Storage;
    auto [guard, res] = p.device->allocate_memory_unique(alloc_params);
    TI_ASSERT_INFO(
        res == RhiResult::success,
        "BumpOnlyDeviceNodeAllocator: failed to allocate independent pool "
        "buffer of {} bytes (snode_id={}, RhiResult={})",
        p.independent_pool_size, p.snode_id, static_cast<int>(res));
    // 零初始化独立 buffer（与 add_root_buffer 对称；保证 watermark / pool /
    // freelist / ambient 在 B-3.c 切 codegen 后语义不变）。
    Stream *stream = p.device->get_compute_stream();
    auto [cmdlist, cmd_res] = stream->new_command_list_unique();
    TI_ASSERT(cmd_res == RhiResult::success);
    cmdlist->buffer_fill(guard->get_ptr(0), kBufferSizeEntireSize, /*data=*/0);
    stream->submit_synced(cmdlist.get());
    independent_pool_guard_ = std::move(guard);
  }
}

BumpOnlyDeviceNodeAllocator::~BumpOnlyDeviceNodeAllocator() = default;

DeviceAllocation BumpOnlyDeviceNodeAllocator::pool_buffer() const {
  // 当前实现把池放在 root_buffer 子区间，所以「pool_buffer」就是 root_buffer
  // alloc 自身；codegen 端通过 spirv_contract() 拿到子区间偏移做 indexing。
  // B-3.b (2026-05): 开独立池后返回独立 DeviceAllocation；但 B-3.b 阶段
  // codegen 未切过来，实际返回哪个不影响运行为。
  if (independent_pool_guard_) {
    return *independent_pool_guard_;
  }
  return params_.root_buffer_alloc;
}

DeviceAllocation *BumpOnlyDeviceNodeAllocator::independent_pool_alloc() const {
  return independent_pool_guard_ ? independent_pool_guard_.get() : nullptr;
}

void BumpOnlyDeviceNodeAllocator::clear_all(CommandList *cmd) {
  TI_ASSERT(cmd != nullptr);
  // 1) watermark 置 0（4 字节）
  cmd->buffer_fill(params_.root_buffer_alloc.get_ptr(
                       params_.watermark_offset),
                   /*size=*/4, /*data=*/0u);
  // 2) pool 数据区清零（pool_capacity * cell_payload_bytes）
  //    data=0 是 buffer_fill 的快路径（参见 public_device.h:399 注释）。
  const std::size_t pool_bytes =
      params_.pool_capacity * params_.cell_payload_bytes;
  cmd->buffer_fill(params_.root_buffer_alloc.get_ptr(
                       params_.pool_data_offset),
                   pool_bytes, /*data=*/0u);
  // 注意：调用方负责前后 memory barrier；本接口只发射两条 fill。
}

SpirvAllocatorContract BumpOnlyDeviceNodeAllocator::spirv_contract() const {
  SpirvAllocatorContract c;
  c.watermark_offset = params_.watermark_offset;
  c.pool_data_offset = params_.pool_data_offset;
  c.pool_capacity = static_cast<uint32_t>(params_.pool_capacity);
  c.cell_stride_bytes = static_cast<uint32_t>(params_.cell_payload_bytes);
  c.snode_id = params_.snode_id;
  // 路线 B B-1：透传 freelist / ambient zone 元数据。
  c.has_freelist = params_.has_freelist;
  c.freelist_head_offset = params_.freelist_head_offset;
  c.freelist_links_offset = params_.freelist_links_offset;
  c.has_ambient_zone = params_.has_ambient_zone;
  c.ambient_offset = params_.ambient_offset;
  // B-2.b：透传 alloc 协议 / pool_fraction 给 codegen / 调试日志。
  c.alloc_protocol = params_.alloc_protocol;
  c.pool_fraction = params_.pool_fraction;
  return c;
}

// -----------------------------------------------------------------------------
// Factory
// -----------------------------------------------------------------------------

std::unique_ptr<DeviceNodeAllocator> create_device_node_allocator(
    const BumpOnlyDeviceNodeAllocator::Params &params) {
  // Phase 2a 只此一种实现；将来加 freelist 时在此处分支。
  return std::make_unique<BumpOnlyDeviceNodeAllocator>(params);
}

}  // namespace gfx
}  // namespace taichi::lang
