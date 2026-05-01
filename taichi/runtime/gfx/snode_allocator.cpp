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
  // C-2.4.a Commit B (2026-05): 独立池路径下整个 buffer 清零。Chunked
  // 路径把 pool_data 区扩为 max_chunks * chunk_size_cells * cell_bytes，
  // BumpOnly 只看到 pool_capacity 不够覆盖；用整 buffer fill 既兼容
  // Bump (max_chunks=1, footprint == watermark+pool_capacity*cell_bytes) 又
  // 兼容 Chunked (footprint > pool_capacity*cell_bytes)。同时也覆盖
  // freelist / ambient zone 区域，与原来的两段 fill 行为等价。
  if (independent_pool_guard_) {
    cmd->buffer_fill(independent_pool_guard_->get_ptr(0),
                     kBufferSizeEntireSize, /*data=*/0u);
    return;
  }
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
// ChunkedDeviceNodeAllocator —— C-2.2 skeleton
// -----------------------------------------------------------------------------
//
// 见 snode_allocator.h 顶部注释：当前 skeleton 仅 own 一个 BumpOnly 子分配器
// 并强制独立 pool（chunk[0] = 独立 DeviceAllocation）；shader 寻址完全不变。
// chunks() 暴露 1 元素 list 供 C-2.3 的 codegen 替换为 descriptor-array 寻址。
//
ChunkedDeviceNodeAllocator::ChunkedDeviceNodeAllocator(const Params &p) {
  // C-2.2 skeleton：byte-equivalent 路径必须有独立 pool buffer（即使
  // codegen 仍读 root_buffer，B-3.b 保证独立 buffer 在初始化时被清零；
  // 后续 C-2.3 把 codegen 切到 chunk[0] 时直接复用）。
  // 上层 policy 在 chunked 路径下应已传 use_independent_pool=true；这里
  // 无条件赋值是兜底，不打 WARN（实测当前路径不触发）。
  Params bump_params = p;
  bump_params.use_independent_pool = true;
  if (bump_params.independent_pool_size == 0) {
    // independent_pool_size 由 snode_struct_compiler 在 B-3.b 起填入；
    // 若上层未填（例如老路径未 opt-in B-3.b），按 contract 自计算下界：
    // watermark(4B) + freelist + ambient + pool_data。skeleton 用最简
    // 上界：pool_data_offset + pool_capacity * cell_payload_bytes。
    bump_params.independent_pool_size =
        static_cast<std::size_t>(p.pool_data_offset) +
        p.pool_capacity * p.cell_payload_bytes;
  }
  bump_ = std::make_unique<BumpOnlyDeviceNodeAllocator>(bump_params);

  // C-2.4.a Commit A (2026-05) → C-2.5 (2026-05): 静态预分配 max_chunks-1
  // 个额外 chunk。每个 chunk size = chunk_size_cells * cell_stride，其中
  // chunk_size_cells = 1 << chunk_log2_capacity（由 layout pass 计算并通过
  // contract → Params 透传）。chunk[0] 仍是 NodeAllocatorPool buffer（含
  // meta + chunk[0] 的 pool_data），chunk[k>0] 是 cell-only buffer，由
  // SPIR-V chunked descriptor array 经二步 OpAccessChain 寻址。
  if (p.max_chunks > 1u) {
    // C-2.4.c (2026-05): probe descriptor-array hard limit before any
    // physical chunk allocation. Vulkan core only guarantees
    // maxPerStageDescriptorStorageBuffers >= 4, so users who set
    // vulkan_pointer_max_chunks=N too aggressively must hit a clean
    // TI_ERROR rather than a late vkCreateDescriptorSetLayout failure or
    // device-lost. Backends that don't expose this limit (default
    // implementation in public_device.h) return UINT32_MAX and skip the
    // check.
    const uint32_t descriptor_cap =
        p.device->get_max_storage_buffer_descriptors_per_binding();
    TI_TRACE(
        "ChunkedDeviceNodeAllocator: snode_id={} max_chunks={} "
        "descriptor_cap={} (UINT32_MAX={})",
        p.snode_id, p.max_chunks, descriptor_cap, UINT32_MAX);
    TI_ERROR_IF(
        p.max_chunks > descriptor_cap,
        "ChunkedDeviceNodeAllocator: vulkan_pointer_max_chunks={} exceeds "
        "device limit maxPerStageDescriptorStorageBuffers={} (snode_id={}). "
        "Lower max_chunks or run on a device with a larger descriptor "
        "limit.",
        p.max_chunks, descriptor_cap, p.snode_id);
    TI_ASSERT_INFO(
        p.chunk_log2_capacity < 32u,
        "ChunkedDeviceNodeAllocator: chunk_log2_capacity={} out of range "
        "(snode_id={})",
        p.chunk_log2_capacity, p.snode_id);
    const uint64_t chunk_size_cells_64 = 1ull << p.chunk_log2_capacity;
    const uint64_t extra_chunk_bytes_64 =
        chunk_size_cells_64 * static_cast<uint64_t>(p.cell_payload_bytes);
    TI_ASSERT_INFO(
        extra_chunk_bytes_64 <= 0xffffffffull,
        "ChunkedDeviceNodeAllocator: extra chunk size {} exceeds u32 "
        "(snode_id={})",
        extra_chunk_bytes_64, p.snode_id);
    const std::size_t extra_chunk_bytes =
        static_cast<std::size_t>(extra_chunk_bytes_64);
    extra_chunks_.reserve(p.max_chunks - 1u);
    Stream *stream = p.device->get_compute_stream();
    for (uint32_t i = 1; i < p.max_chunks; ++i) {
      Device::AllocParams alloc_params;
      alloc_params.size = extra_chunk_bytes;
      alloc_params.host_write = false;
      alloc_params.host_read = false;
      alloc_params.export_sharing = false;
      alloc_params.usage = AllocUsage::Storage;
      auto [guard, res] = p.device->allocate_memory_unique(alloc_params);
      TI_ASSERT_INFO(
          res == RhiResult::success,
          "ChunkedDeviceNodeAllocator: failed to allocate extra chunk[{}] "
          "of {} bytes (snode_id={}, RhiResult={})",
          i, extra_chunk_bytes, p.snode_id, static_cast<int>(res));
      auto [cmdlist, cmd_res] = stream->new_command_list_unique();
      TI_ASSERT(cmd_res == RhiResult::success);
      cmdlist->buffer_fill(guard->get_ptr(0), kBufferSizeEntireSize,
                           /*data=*/0);
      stream->submit_synced(cmdlist.get());
      extra_chunks_.push_back(std::move(guard));
    }
    extra_chunks_total_bytes_ =
        extra_chunk_bytes * extra_chunks_.size();
    TI_TRACE(
        "ChunkedDeviceNodeAllocator: snode_id={} max_chunks={} "
        "extra_chunks={} chunk_size_cells={} extra_chunk_bytes={} "
        "total_extra_bytes={}",
        p.snode_id, p.max_chunks, extra_chunks_.size(), chunk_size_cells_64,
        extra_chunk_bytes,
        extra_chunk_bytes * extra_chunks_.size());
  }
}

ChunkedDeviceNodeAllocator::~ChunkedDeviceNodeAllocator() = default;

DeviceAllocation ChunkedDeviceNodeAllocator::pool_buffer() const {
  return bump_->pool_buffer();
}

std::size_t ChunkedDeviceNodeAllocator::pool_capacity() const {
  return bump_->pool_capacity();
}

std::size_t ChunkedDeviceNodeAllocator::cell_payload_bytes() const {
  return bump_->cell_payload_bytes();
}

void ChunkedDeviceNodeAllocator::clear_all(CommandList *cmd) {
  bump_->clear_all(cmd);
}

SpirvAllocatorContract ChunkedDeviceNodeAllocator::spirv_contract() const {
  // C-2.2 skeleton：contract 与 BumpOnly 完全一致（shader 寻址不变）。
  // C-2.3 起会在 contract 上把 allocator_kind=Chunked + chunk_* 字段
  // 进一步透传给 SPIR-V codegen 切换到 chunk-array indexing。
  return bump_->spirv_contract();
}

DeviceAllocation *ChunkedDeviceNodeAllocator::independent_pool_alloc() const {
  return bump_->independent_pool_alloc();
}

std::vector<DeviceAllocation> ChunkedDeviceNodeAllocator::chunks() const {
  // chunk[0] = bump_->pool_buffer()（独立 pool buffer，包含 watermark/freelist/
  // ambient/pool_data）。Commit A 起 chunk[1..N-1] 依顺序 append，为未来
  // Commit B 的跨 chunk 寻址提供全部 DeviceAllocation。
  std::vector<DeviceAllocation> result;
  result.reserve(1u + extra_chunks_.size());
  result.push_back(bump_->pool_buffer());
  for (const auto &g : extra_chunks_) {
    result.push_back(*g);
  }
  return result;
}

std::size_t ChunkedDeviceNodeAllocator::extra_chunks_total_bytes() const {
  return extra_chunks_total_bytes_;
}

// -----------------------------------------------------------------------------
// Factory
// -----------------------------------------------------------------------------

std::unique_ptr<DeviceNodeAllocator> create_device_node_allocator(
    const BumpOnlyDeviceNodeAllocator::Params &params) {
  // C-2.1 (2026-05): allocator_kind 工厂分支。
  // C-2.2 (2026-05): Chunked skeleton 上线，与 Bump 字节等价（1 chunk =
  //   整池 + shader 寻址不变）。默认仍为 Bump，C-2.3 完成 shader 切换并
  //   验证 grow 收益后再考虑 flip。任一路径都不得 silent 降级。
  using AllocatorKind =
      ::taichi::lang::spirv::SpirvAllocatorContract::AllocatorKind;
  if (params.allocator_kind == AllocatorKind::Chunked) {
    return std::make_unique<ChunkedDeviceNodeAllocator>(params);
  }
  return std::make_unique<BumpOnlyDeviceNodeAllocator>(params);
}

}  // namespace gfx
}  // namespace taichi::lang
