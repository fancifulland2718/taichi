#include "taichi/codegen/spirv/snode_struct_compiler.h"

#include <algorithm>
#include <cstdlib>

namespace taichi::lang {
namespace spirv {
namespace {

#if defined(TI_VULKAN_POINTER_POOL_FRACTION)
// G2 (2026-04-30): allow shrinking the pool capacity reserved for pointer
// SNodes via the TI_VULKAN_POOL_FRACTION env var. Default = 1.0 (the legacy
// behaviour, byte-identical to the OFF path). Lower fractions trade memory
// for the assumption that the user's working set is far below the
// worst-case product of nested pointer dimensions; combined with G1.b
// freelist recycling, deactivated cells return to the pool so steady-state
// occupancy — not peak — drives the actual demand. Out-of-capacity
// activations fall back to slot=0 (silent inactive) per the existing
// cap_v guard in pointer_lookup_or_activate; this is the same safe
// degradation as `ti.activate` racing past the watermark.
//
// Read once on first call and cached. Invalid / unset / out-of-range
// values (<= 0 or > 1) are treated as 1.0.
static double get_pool_fraction() {
  static double cached = []() -> double {
    const char *env = std::getenv("TI_VULKAN_POOL_FRACTION");
    if (!env) {
      return 1.0;
    }
    char *end = nullptr;
    const double v = std::strtod(env, &end);
    if (end == env || v <= 0.0 || v > 1.0) {
      return 1.0;
    }
    return v;
  }();
  return cached;
}
#endif

class StructCompiler {
 public:
  CompiledSNodeStructs run(SNode &root) {
    TI_ASSERT(root.type == SNodeType::root);

    CompiledSNodeStructs result;
    result.root = &root;
    result.root_size = compute_snode_size(&root);

    // Phase 2b: append per-pointer-SNode pool + watermark to the end of the
    // root buffer. The slot array (sized num_cells * 4) already lives at the
    // container position via compute_snode_size(); here we only reserve the
    // pool data region and the u32 watermark counter. Both regions are part
    // of the root buffer and are zero-filled by GfxRuntime::add_root_buffer.
    for (auto &kv : snode_descriptors_) {
      auto &desc = kv.second;
      if (desc.snode->type != SNodeType::pointer) {
        continue;
      }
      // Pool capacity = total instances of this pointer SNode globally
      // (parent.total_num_cells * num_cells_per_container). Using
      // num_cells_per_container alone would only fit a single instance and
      // silently corrupt nested pointer trees (e.g. pointer.pointer.dense)
      // where the same descriptor is re-instantiated under every parent
      // cell.
      size_t capacity = desc.total_num_cells_from_root;
#if defined(TI_VULKAN_POINTER_POOL_FRACTION)
      // G2: shrink to user-specified fraction of the worst case. Lower
      // bound by num_cells_per_container so the smallest sane case
      // (single instance) is never under-provisioned.
      const double frac = get_pool_fraction();
      if (frac < 1.0) {
        const size_t scaled = static_cast<size_t>(
            static_cast<double>(capacity) * frac + 0.5);
        const size_t lower_bound =
            static_cast<size_t>(desc.snode->num_cells_per_container);
        capacity = std::max<size_t>(scaled, lower_bound);
      }
#endif
      const size_t cell_bytes = desc.cell_stride;
      // u32 alignment for atomic ops on the watermark (root_size is already a
      // multiple of 4 because every preceding container size is, but be
      // explicit for safety against future container types).
      result.root_size = (result.root_size + 3u) & ~size_t(3);
      desc.pointer_watermark_offset_in_root = result.root_size;
      result.root_size += 4;
#if defined(TI_VULKAN_POINTER_FREELIST)
      // G1.b: insert freelist header + links between watermark and the
      // pool data region. freelist_head is a single u32 sentinel; the
      // links table is u32[capacity]. Both zero-initialized by the
      // root buffer memset on construction (= empty freelist).
      desc.pointer_freelist_head_offset_in_root = result.root_size;
      result.root_size += 4;
      desc.pointer_freelist_links_offset_in_root = result.root_size;
      result.root_size += 4 * capacity;
#endif
      // Pool data starts on a 4-byte boundary; cell_bytes itself is a
      // multiple of 4 in practice (smallest primitive is 4-byte i32/f32),
      // but keep the rounding for defensiveness.
      result.root_size = (result.root_size + 3u) & ~size_t(3);
      desc.pointer_pool_offset_in_root = result.root_size;
      desc.pointer_pool_capacity = capacity;
      result.root_size += cell_bytes * capacity;
    }

    result.snode_descriptors = std::move(snode_descriptors_);
    /*
    result.type_factory = new tinyir::Block;
    result.root_type = construct(*result.type_factory, &root);
    */
    TI_TRACE("RootBuffer size={}", result.root_size);

    /*
    std::unique_ptr<tinyir::Block> b = ir_reduce_types(result.type_factory);

    TI_WARN("Original types:\n{}", ir_print_types(result.type_factory));

    TI_WARN("Reduced types:\n{}", ir_print_types(b.get()));
    */

    return result;
  }

 private:
  const tinyir::Type *construct(tinyir::Block &ir_module, SNode *sn) {
    const tinyir::Type *cell_type = nullptr;

    if (sn->is_place()) {
      // Each cell is a single Type
      cell_type = translate_ti_primitive(ir_module, sn->dt);
    } else {
      // Each cell is a struct
      std::vector<const tinyir::Type *> struct_elements;
      for (auto &ch : sn->ch) {
        const tinyir::Type *elem_type = construct(ir_module, ch.get());
        struct_elements.push_back(elem_type);
      }
      tinyir::Type *st = ir_module.emplace_back<StructType>(struct_elements);
      st->set_debug_name(
          fmt::format("{}_{}", snode_type_name(sn->type), sn->get_name()));
      cell_type = st;

      if (sn->type == SNodeType::pointer) {
        cell_type = ir_module.emplace_back<PhysicalPointerType>(cell_type);
      }
    }

    if (sn->num_cells_per_container == 1 || sn->is_scalar()) {
      return cell_type;
    } else {
      return ir_module.emplace_back<ArrayType>(cell_type,
                                               sn->num_cells_per_container);
    }
  }

  std::size_t compute_snode_size(SNode *sn) {
    const bool is_place = sn->is_place();

    SNodeDescriptor sn_desc;
    sn_desc.snode = sn;
    if (is_place) {
      sn_desc.cell_stride = data_type_size(sn->dt);
      sn_desc.container_stride = sn_desc.cell_stride;
    } else {
      // Sort by size, so that smaller subfields are placed first.
      // This accelerates Nvidia's GLSL compiler, as the compiler tries to
      // place all statically accessed fields
      std::vector<std::pair<size_t, int>> element_strides;
      int i = 0;
      for (auto &ch : sn->ch) {
        element_strides.push_back({compute_snode_size(ch.get()), i});
        i += 1;
      }
      std::sort(
          element_strides.begin(), element_strides.end(),
          [](const std::pair<size_t, int> &a, const std::pair<size_t, int> &b) {
            return a.first < b.first;
          });

      std::size_t cell_stride = 0;
      for (auto &[snode_size, i] : element_strides) {
        auto &ch = sn->ch[i];
        auto child_offset = cell_stride;
        auto *ch_snode = ch.get();
        cell_stride += snode_size;
        snode_descriptors_.find(ch_snode->id)
            ->second.mem_offset_in_parent_cell = child_offset;
        ch_snode->offset_bytes_in_parent_cell = child_offset;
      }
      sn_desc.cell_stride = cell_stride;

      if (sn->type == SNodeType::bitmasked) {
        size_t num_cells = sn_desc.snode->num_cells_per_container;
        size_t bitmask_num_words =
            num_cells % 32 == 0 ? (num_cells / 32) : (num_cells / 32 + 1);
        sn_desc.container_stride =
            cell_stride * num_cells + bitmask_num_words * 4;
      } else if (sn->type == SNodeType::pointer) {
        // Phase 2b: the pointer SNode container resident in the parent cell
        // holds only the slot array (4 bytes per cell). The actual child
        // cells (`cell_stride` bytes each, capacity = num_cells_per_container)
        // live in a separate pool that is appended to the end of the root
        // buffer in StructCompiler::run() once tree size is known. Per-cell
        // recycle is intentionally not done in 2a/2b -- whole-pool reset
        // happens via the device allocator's clear_all() (Phase 2c hook).
        sn_desc.container_stride =
            sn_desc.snode->num_cells_per_container * 4;
      } else if (sn->type == SNodeType::dynamic) {
#if defined(TI_VULKAN_DYNAMIC)
        // G4: append a u32 length counter at the end of each dynamic
        // container. Layout = [data: cell_stride * N][length u32]. The
        // length is zero-initialized by the root buffer memset.
        sn_desc.dynamic_length_offset_in_container =
            cell_stride * sn_desc.snode->num_cells_per_container;
        sn_desc.container_stride =
            sn_desc.dynamic_length_offset_in_container + 4;
#else
        sn_desc.container_stride =
            cell_stride * sn_desc.snode->num_cells_per_container;
#endif
      } else {
        sn_desc.container_stride =
            cell_stride * sn_desc.snode->num_cells_per_container;
      }
    }

    sn->cell_size_bytes = sn_desc.cell_stride;

    sn_desc.total_num_cells_from_root = 1;
    for (const auto &e : sn->extractors) {
      // Note that the extractors are set in two places:
      // 1. When a new SNode is first defined
      // 2. StructCompiler::infer_snode_properties()
      // The second step is the finalized result.
      sn_desc.total_num_cells_from_root *= e.num_elements_from_root;
    }

    TI_TRACE("SNodeDescriptor");
    TI_TRACE("* snode={}", sn_desc.snode->id);
    TI_TRACE("* type={} (is_place={})", sn_desc.snode->node_type_name,
             is_place);
    TI_TRACE("* cell_stride={}", sn_desc.cell_stride);
    TI_TRACE("* num_cells_per_container={}",
             sn_desc.snode->num_cells_per_container);
    TI_TRACE("* container_stride={}", sn_desc.container_stride);
    TI_TRACE("* total_num_cells_from_root={}",
             sn_desc.total_num_cells_from_root);
    TI_TRACE("");

    TI_ASSERT(snode_descriptors_.find(sn->id) == snode_descriptors_.end());
    snode_descriptors_[sn->id] = sn_desc;
    return sn_desc.container_stride;
  }

  SNodeDescriptorsMap snode_descriptors_;
};

}  // namespace

CompiledSNodeStructs compile_snode_structs(SNode &root) {
  StructCompiler compiler;
  return compiler.run(root);
}

}  // namespace spirv
}  // namespace taichi::lang
