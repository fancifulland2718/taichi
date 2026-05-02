// Codegen for the hierarchical data structure
#pragma once

#include <unordered_map>

#include "taichi/ir/snode.h"

#include "spirv_types.h"
#include "spirv_allocator_contract.h"

namespace taichi::lang {
namespace spirv {

struct SNodeDescriptor {
  const SNode *snode = nullptr;
  // Stride (bytes) of a single cell.
  size_t cell_stride = 0;

  // Bytes of a single container.
  size_t container_stride = 0;

  // Total number of CELLS of this SNode
  // For example, for a layout of
  // ti.root
  //   .dense(ti.ij, (3, 2))  // S1
  //   .dense(ti.ij, (5, 3))  // S2
  // |total_num_cells_from_root| for S2 is 3x2x5x3 = 90. That is, S2 has a total
  // of 90 cells. Note that the number of S2 (container) itself is 3x2=6!
  size_t total_num_cells_from_root = 0;
  // An SNode can have multiple number of components, where each component
  // starts at a fixed offset in its parent cell's memory.
  size_t mem_offset_in_parent_cell = 0;

  // 路线 B B-4（2026-05）：pointer SNode 元数据已完全迁到
  // `CompiledSNodeStructs.pointer_contracts`（SpirvAllocatorContract POD），
  // SNodeDescriptor 不再存 pointer_pool_offset / pointer_watermark_offset /
  // pointer_pool_capacity / pointer_freelist_* / pointer_ambient_offset 字段。
  // codegen 端通过 contract 读全部 pointer 元信息；layout pass 直接 emplace
  // 到 result.pointer_contracts。

  // G4 (vulkan_sparse_experimental, dynamic SNode only,
  // gated by TI_VULKAN_DYNAMIC at codegen time):
  // Each dynamic container is laid out as
  //   [data: cell_stride * num_cells_per_container][length u32]
  // where length holds the number of currently appended elements (capped
  // at num_cells_per_container by user contract; runtime does NOT bounds-
  // check beyond clamping the resulting cell index in `is_active` reads
  // and listgen). length is zero-initialized by the root buffer memset.
  // dynamic_length_offset_in_container == cell_stride * num_cells_per_container
  // when dynamic is enabled, else 0 (legacy flat layout).
  size_t dynamic_length_offset_in_container = 0;

  SNode *get_child(int ch_i) const {
    return snode->ch[ch_i].get();
  }
};

using SNodeDescriptorsMap = std::unordered_map<int, SNodeDescriptor>;

struct CompiledSNodeStructs {
  // Root buffer size in bytes.
  size_t root_size{0};
  // Root SNode
  const SNode *root{nullptr};
  // Map from SNode ID to its descriptor.
  SNodeDescriptorsMap snode_descriptors;

  // 路线 B B-1（2026-04-30）：pointer SNode → allocator contract。
  // 由 snode_struct_compiler 在生成 layout 后从 SNodeDescriptor.pointer_*
  // 字段直接抄入；codegen 端通过 contract 而非 SNodeDescriptor 读 pointer
  // 元数据，为 B-2/B-3 阶段池迁移到独立 buffer / 运行时可调容量做准备。
  // key = SNode id（仅 type==pointer 的 SNode 入表）。
  std::unordered_map<int, SpirvAllocatorContract> pointer_contracts;

  // B-3.b（2026-05）：每个 pointer SNode 在 root_buffer 中的 layout footprint
  // （字节数 = watermark + freelist + pool_data + ambient + alignment 累计）。
  // SNodeTreeManager 在 contract.pool_buffer_binding_id >= 0 时按此值申请独立
  // DeviceAllocation。key = SNode id；空表示该 pointer 走 root_buffer 子区间。
  std::unordered_map<int, size_t> pool_buffer_sizes;

  // TODO: Use the new type compiler
  // tinyir::Block *type_factory;
  // const tinyir::Type *root_type;
};

// B-2.b（2026-05）：路线 B 阶段把 4 个 CMake 宏（TI_VULKAN_POINTER_*）下放为
// CompileConfig 可调的运行时字段。此 POD 在 SNode struct 编译实际生产
// pointer SNode layout 与 SpirvAllocatorContract 时读取。默认值与 CMake-ON 路径
// 字节等价，以保留 AOT / 历史调用者未传入 policy 时的行为。
struct PointerLayoutPolicy {
  bool freelist{true};       // G1.b
  bool ambient_zone{true};   // G10-P2
  bool cas_marker{true};     // G1.a
  double pool_fraction{1.0}; // G2
  // B-3.b（2026-05）：当且仅当此 flag=true && SNodeTree 内恰好 1 个 pointer SNode
  // 时，把该 pointer 的 pool 元数据切到独立 NodeAllocatorPool buffer。
  // 默认 false 与历史行为字节等价。多 pointer / 嵌套 pointer 仍走 root_buffer fallback。
  bool independent_pool{false};
  // C-2.1（2026-05）：pointer SNode device-side allocator 选择。合法值
  // {"bump", "chunked"}。默认 "bump" 与 B 路线 byte-equivalent；"chunked"
  // 在 C-2.2 前未实现，工厂端报错。
  std::string allocator_kind{"bump"};
  // C-2.4.a（2026-05）：chunked 路径上限。默认 1 与 C-2.3 等价（单
  // chunk = 整池）；step i 仅透传到 contract.max_chunks，不改 SPIR-V
  // 寻址。后续 step ii–v 会在 max_chunks > 1 时发射 descriptor array
  // of buffers + 两步 OpAccessChain（选 chunk 后再在 chunk 内寻址）。
  uint32_t max_chunks{1};
  // C-9（2026-05）：pointer alloc 协议切换为 deterministic slot mapping。
  // 详见 §14。layout 端额外 gating capacity >= worst_capacity && bump。
  bool deterministic_slot{true};
};

CompiledSNodeStructs compile_snode_structs(
    SNode &root,
    const PointerLayoutPolicy &policy = {});

}  // namespace spirv
}  // namespace taichi::lang
