// =============================================================================
// SpirvAllocatorContract — shared POD between spirv codegen and gfx runtime
// =============================================================================
//
// 见 compile_doc/SNode_Vulkan_规划.md §10 路线 B / Stage B-1。
//
// codegen 端通过 contract 读 pointer SNode 的 pool / watermark / freelist /
// ambient zone 偏移，而不再直接读 SNodeDescriptor 的 pointer_* 字段。这样将来
// allocator 切换实现（bump-only / freelist / chunked / sparse-binding）时
// codegen 不动，仅替换 contract 字段含义和 GfxRuntime 端构造逻辑。
//
// 当前实现的语义：
//   - 所有 offset 都是该 contract 所属 base buffer 内的字节偏移：
//     * pool_buffer_binding_id == -1（OFF 默认）→ base buffer = root_buffer
//     * pool_buffer_binding_id >= 0（B-3.c-2 ON）→ base buffer =
//       NodeAllocatorPool[pool_buffer_binding_id]，且整个池（含 watermark /
//       freelist / pool_data / ambient）从该 buffer 内 offset 0 起算
//   - has_freelist=false / has_ambient_zone=false 时对应 offset 字段未定义
//   - pool_capacity = num_cells（编译期常量）
//
// 本头文件**只**定义 POD，不依赖任何 RHI / Device 接口，方便被 codegen
// （taichi::lang::spirv::）和 runtime（taichi::lang::gfx::）共同使用。
// =============================================================================
#pragma once

#include <cstdint>

namespace taichi::lang {
namespace spirv {

struct SpirvAllocatorContract {
  // u32 atomic 计数器在 base buffer 内的字节偏移（4B 对齐）
  uint32_t watermark_offset{0};
  // pool 数据起始字节偏移（cell payload 数组开头，4B 对齐）
  uint32_t pool_data_offset{0};
  // 池容量（cell 数）—— worst-case = SNode 的 total_num_cells_from_root
  uint32_t pool_capacity{0};
  // 单 cell payload 字节数（含子 SNode container 全部内容）
  uint32_t cell_stride_bytes{0};

  // G1.b freelist：has_freelist=true 时下面两个 offset 有效
  bool has_freelist{false};
  uint32_t freelist_head_offset{0};
  uint32_t freelist_links_offset{0};

  // G10-P2 ambient zone：has_ambient_zone=true 时 ambient_offset 有效
  bool has_ambient_zone{false};
  uint32_t ambient_offset{0};

  // 该 contract 服务的 SNode id，仅供调试 / runtime error 信息使用
  int32_t snode_id{-1};

  // 路线 B B-2.b（2026-05）：把 G1.a 的 alloc 协议选择从编译宏下放到 contract，
  // codegen 端按该字段在 race-correct CAS-marker-first 与 legacy atomicIAdd-first
  // 之间分支。默认 CasMarker 与历史 #if defined(TI_VULKAN_POINTER_CAS_MARKER) 等价。
  enum class AllocProtocol : uint8_t {
    Legacy = 0,     // atomicIAdd-first（vanilla 1.7.4 路径）
    CasMarker = 1,  // CAS-marker-first（G1.a 默认行为）
  };
  AllocProtocol alloc_protocol{AllocProtocol::CasMarker};

  // 路线 B B-2.b：池容量收缩比例（仅 layout 端使用，已 baked 进 pool_capacity；
  // 这里保留是为日志 / 离线缓存键 / 调试可见性，codegen 端不读取）。
  // 1.0 = 100% 容量（vanilla 1.7.4 等价）；(0,1) 区间收缩；其余值视为 1.0。
  double pool_fraction{1.0};

  // 路线 B B-3.c-2（2026-05）：>= 0 表示该 pointer SNode 的池整体（watermark /
  // freelist / pool_data / ambient）已迁出 root_buffer，落在独立的
  // NodeAllocatorPool buffer 中，binding 序号 = root_id[0] = pool_buffer_binding_id；
  // 偏移字段以该 buffer 内 offset 0 为基准。-1（默认）= 池仍寄居 root_buffer
  // 子区间，偏移字段以 root_buffer 内 offset 0 为基准（B-1/B-2/B-3.b 行为）。
  int32_t pool_buffer_binding_id{-1};

  // C-2.1 (2026-05) — 路线 C chunked allocator 骨架字段（append-only）。
  //   AllocatorKind::Bump（默认）：B 路线行为，下面三个字段为 0/-1，codegen 与
  //     runtime 完全忽略，byte-equivalent。
  //   AllocatorKind::Chunked：C-2.2 起的 chunked allocator 路径；C-2.1 阶段
  //     工厂端会在见到 Chunked 时直接 TI_NOT_IMPLEMENTED，不 silent fallback。
  // 详见 compile_doc/SNode_Vulkan_规划.md §12.2.1。
  enum class AllocatorKind : uint8_t {
    Bump = 0,
    Chunked = 1,
  };
  AllocatorKind allocator_kind{AllocatorKind::Bump};
  // chunk 容量取 log2 形式（codegen 端用 SHR 即可定位 chunk_idx）。
  // 0 = 未设置（Bump 路径默认）。
  uint32_t chunk_log2_capacity{0};
  // chunk 数上限（runtime 端 descriptor 数组长度）。0 = 未设置。
  uint32_t max_chunks{0};
  // chunk descriptor 数组首 binding；-1 = 未设置（Bump 路径默认）。
  int32_t chunk_descriptor_array_first_binding{-1};

  // C-9 (2026-05) — deterministic slot mapping。true 时 codegen 在
  // do_activate=true 路径用 `new_slot = idx_u32 + 1` 直接 CAS 发布到
  // slot_ptr，跳过 watermark / freelist / spin。仅当 layout 端确认
  // pool_capacity >= worst_capacity 且 allocator_kind == Bump 时才会置 true。
  // 详见 compile_doc/SNode_Vulkan_规划.md §14。
  bool deterministic_slot{false};
};

}  // namespace spirv
}  // namespace taichi::lang
