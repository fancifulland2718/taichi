#include "taichi/codegen/spirv/snode_struct_compiler.h"
#include "taichi/ir/type_factory.h"

#include <algorithm>
#include <cstdlib>

namespace taichi::lang {
namespace spirv {
namespace {

// B-2.b（2026-05）：运行时 pool_fraction 读取仅下放到主路径 if-分支，不再
// 从 TI_VULKAN_POOL_FRACTION 环境变量读取。环境变量仍是全局 fallback：
// 未传入 policy.pool_fraction（-1.0 哨兵）且环境变量合法时，该环境变量被应用。
// C-1.b（2026-05）：仅当 policy_value 在 (0, 1) 严格开区间时启用 fraction
// 缩放分支；其它（哨兵 / 1.0 / 越界）一律返回 -1.0 表示 "未启用"。
// 调用方根据返回值是否 in (0, 1) 决定是否走 L2/L3 缩放路径。
static double resolve_pool_fraction(double policy_value) {
  if (policy_value > 0.0 && policy_value < 1.0) {
    return policy_value;
  }
  static double cached_env = []() -> double {
    const char *env = std::getenv("TI_VULKAN_POOL_FRACTION");
    if (!env) {
      return -1.0;
    }
    char *end = nullptr;
    const double v = std::strtod(env, &end);
    if (end == env || v <= 0.0 || v >= 1.0) {
      return -1.0;
    }
    return v;
  }();
  return cached_env;
}

class StructCompiler {
 public:
  explicit StructCompiler(const PointerLayoutPolicy &policy)
      : policy_(policy) {}

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
      const size_t worst_capacity = capacity;
      const size_t lower_bound =
          static_cast<size_t>(desc.snode->num_cells_per_container);
      // C-1 (2026-05): 4 级 fallback —— L1 vk_max_active_hint > L2 policy
      // pool_fraction > L3 env > L4 worst-case。L1 = per-SNode JIT-time 用户
      // 精确意图；L2/L3 是全局比例缩放；L4 是路线 B 之前的 worst-case。
      // L1 命中时绕过 fraction 完全独立决议；hint 异常值（< 下界 / > 上限）
      // 抬回有效区间并 TI_WARN 一次。
      bool used_hint = false;
      if (desc.snode->vk_max_active_hint > 0) {
        size_t requested =
            static_cast<size_t>(desc.snode->vk_max_active_hint);
        if (requested < lower_bound) {
          TI_WARN(
              "vk_max_active={} on SNode {} is below container lower bound "
              "{}; clamping up. Pointer pool capacity is at least one full "
              "container.",
              requested, desc.snode->get_node_type_name_hinted(),
              lower_bound);
          requested = lower_bound;
        }
        if (requested > worst_capacity) {
          // C-1.b (2026-05): hint > worst_case is no longer clamped down.
          // worst_case is just a worst-case estimate; the user knows their
          // domain. Honor the request and let hardware OOM happen naturally
          // if vkAllocateMemory cannot serve the buffer (consistent with
          // CPU/CUDA "allocate-on-demand" semantics).
          TI_WARN(
              "vk_max_active={} on SNode {} exceeds worst-case capacity {}; "
              "honoring user request. Physical buffer will be sized to fit "
              "{} cells; if vkAllocateMemory cannot satisfy this request the "
              "runtime will surface a hardware OOM (matches CPU/CUDA).",
              requested, desc.snode->get_node_type_name_hinted(),
              worst_capacity, requested);
        }
        capacity = requested;
        used_hint = true;
      }
      const char *capacity_source = "L4_worst_case";
      if (used_hint) {
        capacity_source = "L1_vk_max_active_hint";
      } else {
        // B-2.b: 原 #if TI_VULKAN_POINTER_POOL_FRACTION 分支下放为运行时
        // policy_.pool_fraction；resolve_pool_fraction 会优先读 policy，其次
        // fall back 到环境变量 TI_VULKAN_POOL_FRACTION。C-1.b: resolve 现在
        // 返回 -1.0 表示 "未启用"（默认情形 / 1.0 显式 / 越界），仅 (0,1) 严
        // 格开区间才启用缩放。
        const double frac = resolve_pool_fraction(policy_.pool_fraction);
        if (frac > 0.0 && frac < 1.0) {
          const size_t scaled = static_cast<size_t>(
              static_cast<double>(capacity) * frac + 0.5);
          const size_t new_capacity = std::max<size_t>(scaled, lower_bound);
          if (new_capacity != capacity) {
            capacity = new_capacity;
            // L2 (CompileConfig) and L3 (env) collapse to the same code
            // path because resolve_pool_fraction picks env only when the
            // policy field is at the sentinel default.
            capacity_source = (policy_.pool_fraction > 0.0 &&
                               policy_.pool_fraction < 1.0)
                                  ? "L2_pool_fraction"
                                  : "L3_env_TI_VULKAN_POOL_FRACTION";
          }
        } else if (worst_capacity > 1024) {
          // C-1.b (2026-05): silent worst-case fallback is unfriendly when
          // the SNode shape implies a large physical footprint. Surface
          // one TI_WARN so users notice and can opt into vk_max_active.
          // Threshold 1024 cells matches typical small-grid demos that we
          // do NOT want to spam a warning for.
          TI_WARN(
              "pointer SNode {} defaults to worst-case capacity {} cells "
              "(physical buffer ~ {} bytes). If actual peak activation is "
              "much smaller, set vk_max_active=<expected_peak> on the "
              "pointer() call to reduce VRAM footprint.",
              desc.snode->get_node_type_name_hinted(), worst_capacity,
              worst_capacity * desc.cell_stride);
        }
      }
      // C-1 (2026-05): single-line decision trace for end users debugging
      // pool sizing. Kept at TI_INFO so it does not clutter normal output;
      // user can flip TI_LOG_LEVEL=info to surface it. Reports tier that
      // won, final capacity, worst-case, lower bound and (if used) the raw
      // hint, so a one-liner identifies which fallback level applied.
      TI_INFO(
          "[C-1] pointer SNode {} pool capacity decided: {} cells (source="
          "{}, worst_case={}, lower_bound={}, raw_hint={})",
          desc.snode->get_node_type_name_hinted(), capacity, capacity_source,
          worst_capacity, lower_bound, desc.snode->vk_max_active_hint);
      const size_t cell_bytes = desc.cell_stride;
      // B-3.c-2（2026-05）：在独立池开启时，pointer 池**整体**（元数据 +
      // pool_data + ambient）迁出 root_buffer，落在每个 pointer 自己的
      // NodeAllocatorPool buffer 中；root_size 不再含池数据。`cursor`
      // 是该 pointer SNode 的池 footprint 的统一游标：indep=ON 时从 0
      // 起算（即"offset_in_pool"），OFF 时与 root_size 对齐共享游标
      // （即"offset_in_root"，旧行为）。codegen 通过 container_buffer_value
      // 决定具体寻址哪个 base buffer。
      // B-4（2026-05）：layout 直接写入 SpirvAllocatorContract，移除
      // SNodeDescriptor 的 6 个 pointer_* 中转字段，contract 字段名也
      // 同步去掉 _in_root 后缀（语义随 pool_buffer_binding_id 决定）。
      const bool indep_pool = policy_.independent_pool;
      size_t cursor;
      if (indep_pool) {
        cursor = 0;
      } else {
        result.root_size = (result.root_size + 3u) & ~size_t(3);
        cursor = result.root_size;
      }
      SpirvAllocatorContract c;
      c.snode_id = desc.snode->id;
      c.cell_stride_bytes = static_cast<uint32_t>(cell_bytes);
      c.pool_capacity = static_cast<uint32_t>(capacity);
      c.alloc_protocol =
          policy_.cas_marker
              ? SpirvAllocatorContract::AllocProtocol::CasMarker
              : SpirvAllocatorContract::AllocProtocol::Legacy;
      c.pool_fraction = resolve_pool_fraction(policy_.pool_fraction);
      // C-2.1 (2026-05): allocator_kind 透传。Bump 路径所有 chunk_* 字段保持
      // 默认 0/-1，codegen 与 runtime 都不读取，byte-equivalent。
      // C-2.3 隐患 2 修复：kind 字符串集合显式校验，写错时直接 TI_ERROR，
      // 不允许 silent fallback 到 bump（遵循 §12.2.0 第 2 条「不回退 silent
      // OOC」原则）。
      TI_ERROR_IF(
          policy_.allocator_kind != "bump" &&
              policy_.allocator_kind != "chunked",
          "vulkan_pointer_allocator_kind must be one of {{\"bump\","
          " \"chunked\"}}, got \"{}\". Refusing silent fallback.",
          policy_.allocator_kind);
      c.allocator_kind =
          (policy_.allocator_kind == "chunked")
              ? SpirvAllocatorContract::AllocatorKind::Chunked
              : SpirvAllocatorContract::AllocatorKind::Bump;
      // C-2.3 (2026-05): 选择 Chunked 时计算 chunk_log2_capacity，使单
      // chunk 容量不小于 pool capacity；这样 SPIR-V 侧拆分出的
      // chunk_idx 在有效 slot 范围内恒 0，local_slot == slot，字节等
      // 价于 Bump。max_chunks=1 表明当前阶段仅 chunk[0] 被实际使用，
      // C-2.4 引入多 chunk + descriptor array 后才会递增。
      // C-2.3 隐患 1 修复：chunk_size_bytes = chunk_size_cells *
      // cell_stride 必须落在 u32（codegen 端 SPIR-V 用 u32 mul），否
      // 则 (chunk_idx * chunk_size_bytes) 溢出会让寻址错位。极端值
      // 由用户 vk_max_active 驱动，校验在此处一次性兜住。
      uint32_t pool_cells = static_cast<uint32_t>(capacity);
      if (c.allocator_kind ==
          SpirvAllocatorContract::AllocatorKind::Chunked) {
        // C-2.5 (2026-05)：chunked allocator 限 max_chunks > 1 时只支持
        // 顶层 pointer SNode（parent == root 或 parent 是无 pool 的 dense/
        // bitmasked 等容器）。嵌套 chunked pointer（外层 pointer 也是
        // chunked）需要在 listgen / 多级 slot_ptr 寻址中跨 chunk，超出
        // C-2.5 范围。max_chunks == 1 路径 byte-equivalent，不需此限制。
        const uint32_t requested_max_chunks =
            std::max(policy_.max_chunks, 1u);
        if (requested_max_chunks > 1u) {
          for (auto *p = desc.snode->parent; p != nullptr; p = p->parent) {
            TI_ERROR_IF(
                p->type == SNodeType::pointer,
                "pointer SNode {}: nested chunked pointer (an ancestor "
                "pointer SNode {} is also a pointer) is not supported "
                "with vulkan_pointer_max_chunks > 1. Set "
                "vulkan_pointer_max_chunks=1 or restructure the SNode tree "
                "so only the leaf-most pointer is sparse.",
                desc.snode->get_node_type_name_hinted(),
                p->get_node_type_name_hinted());
          }
        }
        // C-2.5 (2026-05)：chunked allocator 把池切成 max_chunks 个独立
        // DeviceAllocation（chunk[0] 复用 NodeAllocatorPool buffer 含 meta
        // + pool_data；chunk[k>0] 是 cell-only 独立 buffer）。每 chunk
        // 容量 = 1 << chunk_log2_capacity，其中 chunk_log2 = ceil_log2(
        // ceil(capacity/max_chunks))；SPIR-V 端 chunk_idx = effective_slot
        // >> chunk_log2 取值 [0, max_chunks-1]，通过 descriptor array of
        // buffers 二步 OpAccessChain 寻址（详见 spirv_codegen.cpp）。
        //
        // pool_data 区只占 chunk[0] 内的 chunk_size_cells * cell_bytes；
        // max_chunks=1 时 chunk_size_cells = next_pow2(capacity)，与 Bump
        // 路径布局字节等价（capacity 是 2 幂时完全一致）。
        const uint32_t max_chunks = std::max(policy_.max_chunks, 1u);
        const uint32_t chunk_capacity =
            (static_cast<uint32_t>(capacity) + max_chunks - 1u) / max_chunks;
        uint32_t chunk_log2 = 0;
        while ((1u << chunk_log2) < chunk_capacity) {
          ++chunk_log2;
        }
        const uint32_t chunk_size_cells = 1u << chunk_log2;
        const uint64_t chunk_size_bytes_64 =
            static_cast<uint64_t>(chunk_size_cells) *
            static_cast<uint64_t>(cell_bytes);
        TI_ERROR_IF(
            chunk_log2 >= 32 || chunk_size_bytes_64 > 0xffffffffull,
            "pointer SNode {}: chunked allocator single-chunk size exceeds "
            "u32 SPIR-V address math (chunk_log2={}, chunk_size_bytes={}, "
            "max_chunks={}). Reduce vk_max_active, increase "
            "vulkan_pointer_max_chunks, or wait for u64 addressing.",
            desc.snode->get_node_type_name_hinted(), chunk_log2,
            chunk_size_bytes_64, max_chunks);
        c.chunk_log2_capacity = chunk_log2;
        c.max_chunks = max_chunks;
        // pool_data 区只占 chunk[0]：chunk_size_cells * cell_bytes。
        // chunk[k>0] 物理 buffer 在 ChunkedDeviceNodeAllocator 中独立
        // alloc，layout pass 不涉及。
        pool_cells = chunk_size_cells;
      }
      c.watermark_offset = static_cast<uint32_t>(cursor);
      cursor += 4;
      // B-2.b: 原 #if TI_VULKAN_POINTER_FREELIST 布局下放为运行时 policy_.freelist。
      if (policy_.freelist) {
        // G1.b: freelist_head + freelist_links[capacity]。零初始化由
        // 池 buffer 在 SNodeTreeManager 中通过 buffer_fill(0) 提供。
        c.has_freelist = true;
        c.freelist_head_offset = static_cast<uint32_t>(cursor);
        cursor += 4;
        c.freelist_links_offset = static_cast<uint32_t>(cursor);
        cursor += 4 * capacity;
      }
      cursor = (cursor + 3u) & ~size_t(3);
      c.pool_data_offset = static_cast<uint32_t>(cursor);
      // C-2.5 (2026-05)：Chunked 路径 pool_cells = chunk_size_cells（仅
      // chunk[0] 容量；chunk[k>0] 在独立 DeviceAllocation 不计入 cursor）。
      // Bump 路径 pool_cells = capacity。
      cursor += static_cast<size_t>(cell_bytes) * pool_cells;
      // B-2.b: 原 #if TI_VULKAN_POINTER_AMBIENT_ZONE 下放为运行时 policy_.ambient_zone。
      if (policy_.ambient_zone) {
        // G10-P2: 零初始化 cell-sized ambient zone。pointer_lookup_or_activate
        // (do_activate=false) 在 slot==0 时返回此偏移，使 inactive 读结果
        // 恒为 0（与 LLVM ambient_val_addr 语义一致）。从未被任何 kernel
        // 写入；零初始化由池 buffer 的 fill(0) 或 root buffer 的 memset(0)
        // 提供（按 indep_pool 决定）。
        cursor = (cursor + 3u) & ~size_t(3);
        c.has_ambient_zone = true;
        c.ambient_offset = static_cast<uint32_t>(cursor);
        cursor += cell_bytes;
      }
      cursor = (cursor + 3u) & ~size_t(3);
      if (indep_pool) {
        // 独立池：池整体 footprint = cursor；root_size 不动。
        // contract 偏移以独立 buffer 内 offset 0 为基准；binding_id = sid。
        c.pool_buffer_binding_id = desc.snode->id;
        result.pool_buffer_sizes[desc.snode->id] = cursor;
      } else {
        // OFF 默认：池整体仍追加在 root buffer 末尾，老行为字节等价。
        // contract 偏移以 root buffer 内 offset 0 为基准；binding_id = -1。
        result.root_size = cursor;
      }
      result.pointer_contracts.emplace(desc.snode->id, c);
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
      // G9.2 (2026-04-30): a `place` whose dt is a quant scalar
      // (QuantIntType / QuantFixedType / QuantFloatType) lives entirely
      // INSIDE the parent quant_array / bit_struct's physical word.  It
      // consumes no bytes of its own; differentiation is done by bit
      // shifts at GlobalLoadStmt / GlobalStoreStmt time.  Reporting size
      // 0 here lets the parent quant_array branch below override
      // cell_stride / container_stride to the physical_type's size.
      if (sn->dt->is<QuantIntType>() || sn->dt->is<QuantFixedType>() ||
          sn->dt->is<QuantFloatType>()) {
        sn_desc.cell_stride = 0;
        sn_desc.container_stride = 0;
      } else {
        sn_desc.cell_stride = data_type_size(sn->dt);
        sn_desc.container_stride = sn_desc.cell_stride;
      }
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
      } else if (sn->type == SNodeType::quant_array) {
        // G9.2 (2026-04-30): quant_array packs num_cells_per_container
        // user-visible cells into ONE physical word (i32 by default; the
        // LLVM contract requires element_num_bits * num_cells <=
        // physical_type bits).  Both cell_stride and container_stride
        // therefore equal the physical_type's size; the per-cell
        // differentiation is done by bit shifts at the GlobalLoadStmt /
        // GlobalStoreStmt visitor (see spirv_codegen.cpp).
        // Note: the children loop above set cell_stride to the sum of
        // the single quant child's cell_stride, which is 0 by the new
        // is_place branch.  We override it here unconditionally to the
        // physical word size.
        TI_ASSERT(sn->ch.size() == 1);
        TI_ERROR_IF(data_type_bits(sn->physical_type) < 32,
                    "quant_array physical type must be at least 32 bits on "
                    "Vulkan/SPIR-V backend.");
        // Mirror taichi/codegen/llvm/struct_llvm.cpp:97-100 -- assign the
        // QuantArrayType to sn->dt so downstream codegen
        // (SNodeLookupStmt / GetChStmt / GlobalLoadStmt / GlobalStoreStmt
        // bit-pointer paths in spirv_codegen.cpp) can recover
        // element_num_bits via `sn->dt->as<QuantArrayType>()`.
        sn->dt = TypeFactory::get_instance().get_quant_array_type(
            sn->physical_type, sn->ch[0]->dt,
            sn->num_cells_per_container);
        std::size_t phys_bytes = data_type_size(sn->physical_type);
        sn_desc.cell_stride = phys_bytes;
        sn_desc.container_stride = phys_bytes;
      } else if (sn->type == SNodeType::bit_struct) {
        // G9.3b (2026-05-01): a bit_struct (BitpackedFields) packs all
        // child quant scalars into ONE physical word.  Both
        // cell_stride and container_stride equal the physical word
        // size; child differentiation is via member_bit_offset at
        // BitStructStoreStmt / GlobalLoadStmt time.  sn->dt is already
        // a BitStructType assigned in SNode::bit_struct() at frontend
        // time (see taichi/ir/snode.cpp).
        TI_ERROR_IF(data_type_bits(sn->physical_type) < 32,
                    "bit_struct physical type must be at least 32 bits on "
                    "Vulkan/SPIR-V backend.");
        std::size_t phys_bytes = data_type_size(sn->physical_type);
        sn_desc.cell_stride = phys_bytes;
        sn_desc.container_stride = phys_bytes;
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
  PointerLayoutPolicy policy_;
};

}  // namespace

CompiledSNodeStructs compile_snode_structs(
    SNode &root,
    const PointerLayoutPolicy &policy) {
  StructCompiler compiler(policy);
  return compiler.run(root);
}

}  // namespace spirv
}  // namespace taichi::lang
