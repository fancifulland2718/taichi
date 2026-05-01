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

  // Phase 2b (vulkan_sparse_experimental, pointer SNode only):
  // For pointer SNodes, the slot array lives at the regular SNode container
  // position in the parent cell (size = 4 * num_cells_per_container, each u32
  // slot stores 0 = inactive, otherwise (pool_index + 1)). The actual child
  // cells live in a per-pointer-SNode pool appended to the end of the root
  // buffer; the bump watermark is a single u32 also appended at the end.
  // These three fields are only meaningful when snode->type == pointer; for
  // all other SNode types they remain zero.
  size_t pointer_pool_offset_in_root = 0;
  size_t pointer_watermark_offset_in_root = 0;
  size_t pointer_pool_capacity = 0;
  // G1.b (vulkan_sparse_experimental, pointer SNode only,
  // gated by TI_VULKAN_POINTER_FREELIST at codegen time):
  //   freelist_head:  u32 sentinel; 0 = empty, otherwise (pool_index + 1).
  //   freelist_links: u32[pool_capacity]; per-pool-slot "next" pointer in
  //                   the same encoding (0 = tail, otherwise pool_index+1).
  // Both are zero-initialized by GfxRuntime::add_root_buffer (every root
  // buffer is memset(0) on construction). Layout is stable across
  // alloc/deactivate; activate either pops freelist (if non-empty) or
  // bumps watermark; deactivate pushes onto freelist.
  size_t pointer_freelist_head_offset_in_root = 0;
  size_t pointer_freelist_links_offset_in_root = 0;

  // G10-P2 (2026-04-30, vulkan_sparse_experimental, pointer SNode only,
  // gated by TI_VULKAN_POINTER_AMBIENT_ZONE at codegen time):
  // A single zero-initialized cell-sized region appended after the pool
  // data. When `pointer_lookup_or_activate(do_activate=false)` finds the
  // slot to be 0 (inactive cell), the returned cell_byte_offset is this
  // ambient zone instead of `pool[0]`. This matches LLVM's
  // `ambient_val_addr` semantics (inactive sparse reads return zero) and
  // fixes the user-reported "inactive read returns garbage" issue. The
  // zone is sized = cell_stride bytes and never written by any kernel
  // (do_activate=true OOC fallback still routes to pool[0] to preserve
  // the documented silent-loss behavior). Zero-init is provided by
  // GfxRuntime::add_root_buffer's memset(0) on root buffer creation.
  size_t pointer_ambient_offset_in_root = 0;

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
};

CompiledSNodeStructs compile_snode_structs(
    SNode &root,
    const PointerLayoutPolicy &policy = {});

}  // namespace spirv
}  // namespace taichi::lang
