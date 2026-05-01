#include "taichi/codegen/spirv/spirv_codegen.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <variant>
#include <future>
#include <algorithm>

#include "taichi/codegen/codegen_utils.h"
#include "taichi/program/program.h"
#include "taichi/program/kernel.h"
#include "taichi/program/extension.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/ir.h"
#include "taichi/util/line_appender.h"
#include "taichi/codegen/spirv/kernel_utils.h"
#include "taichi/codegen/spirv/spirv_ir_builder.h"
#include "taichi/ir/transforms.h"
#include "taichi/math/arithmetic.h"

#include <spirv-tools/libspirv.hpp>
#include <spirv-tools/optimizer.hpp>
#include "fp16.h"

namespace taichi::lang {
namespace spirv {
namespace {

constexpr char kRootBufferName[] = "root_buffer";
constexpr char kGlobalTmpsBufferName[] = "global_tmps_buffer";
constexpr char kArgsBufferName[] = "args_buffer";
constexpr char kRetBufferName[] = "ret_buffer";
constexpr char kListgenBufferName[] = "listgen_buffer";
constexpr char kExtArrBufferName[] = "ext_arr_buffer";
constexpr char kArgPackBufferName[] = "argpack_buffer";

constexpr int kMaxNumThreadsGridStrideLoop = 65536 * 2;

using BufferType = TaskAttributes::BufferType;
using BufferInfo = TaskAttributes::BufferInfo;
using BufferBind = TaskAttributes::BufferBind;
using BufferInfoHasher = TaskAttributes::BufferInfoHasher;

using TextureBind = TaskAttributes::TextureBind;

std::string buffer_instance_name(BufferInfo b) {
  // https://www.khronos.org/opengl/wiki/Interface_Block_(GLSL)#Syntax
  switch (b.type) {
    case BufferType::Root:
      return std::string(kRootBufferName) + "_" +
             fmt::format("{}", fmt::join(b.root_id, "_"));
    case BufferType::GlobalTmps:
      return kGlobalTmpsBufferName;
    case BufferType::Args:
      return kArgsBufferName;
    case BufferType::Rets:
      return kRetBufferName;
    case BufferType::ListGen:
      return kListgenBufferName;
    case BufferType::ExtArr:
      return std::string(kExtArrBufferName) + "_" +
             fmt::format("{}", fmt::join(b.root_id, "_"));
    case BufferType::ArgPack:
      return std::string(kArgPackBufferName) + "_" +
             fmt::format("{}", fmt::join(b.root_id, "_"));
    case BufferType::NodeAllocatorPool:
      // B-3.a: stable name; the actual binding is unused until B-3.b.
      return std::string("node_pool_") +
             fmt::format("{}", fmt::join(b.root_id, "_"));
    default:
      TI_NOT_IMPLEMENTED;
      break;
  }
  return {};
}

class TaskCodegen : public IRVisitor {
 public:
  struct Params {
    OffloadedStmt *task_ir;
    Arch arch;
    DeviceCapabilityConfig *caps;
    std::vector<CompiledSNodeStructs> compiled_structs;
    const KernelContextAttributes *ctx_attribs;
    std::string ti_kernel_name;
    int task_id_in_kernel;
  };

  const bool use_64bit_pointers = false;

  explicit TaskCodegen(const Params &params)
      : arch_(params.arch),
        caps_(params.caps),
        task_ir_(params.task_ir),
        compiled_structs_(params.compiled_structs),
        ctx_attribs_(params.ctx_attribs),
        task_name_(fmt::format("{}_t{:02d}",
                               params.ti_kernel_name,
                               params.task_id_in_kernel)) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;

    fill_snode_to_root();
    ir_ = std::make_shared<spirv::IRBuilder>(arch_, caps_);
  }

  void fill_snode_to_root() {
    for (int root = 0; root < compiled_structs_.size(); ++root) {
      for (auto &[node_id, node] : compiled_structs_[root].snode_descriptors) {
        snode_to_root_[node_id] = root;
      }
    }
  }

  // Replace the wild '%' in the format string with "%%".
  std::string sanitize_format_string(std::string const &str) {
    std::string sanitized_str;
    for (char c : str) {
      if (c == '%') {
        sanitized_str += "%%";
      } else {
        sanitized_str += c;
      }
    }
    return sanitized_str;
  }

  struct Result {
    std::vector<uint32_t> spirv_code;
    TaskAttributes task_attribs;
    std::unordered_map<std::vector<int>,
                       irpass::ExternalPtrAccess,
                       hashing::Hasher<std::vector<int>>>
        arr_access;
  };

  Result run() {
    ir_->init_header();
    kernel_function_ = ir_->new_function();  // void main();
    ir_->debug_name(spv::OpName, kernel_function_, "main");

    if (task_ir_->task_type == OffloadedTaskType::serial) {
      generate_serial_kernel(task_ir_);
    } else if (task_ir_->task_type == OffloadedTaskType::range_for) {
      // struct_for is automatically lowered to ranged_for for dense snodes
      generate_range_for_kernel(task_ir_);
    } else if (task_ir_->task_type == OffloadedTaskType::struct_for) {
      generate_struct_for_kernel(task_ir_);
    } else if (task_ir_->task_type == OffloadedTaskType::listgen) {
      generate_listgen_kernel(task_ir_);
    } else if (task_ir_->task_type == OffloadedTaskType::gc) {
      // G1.b: pointer-SNode pool recycling is performed inline by the
      // CAS-marker / freelist deactivate path (see pointer_deactivate). The
      // separate gc offload that the LLVM backend uses for compaction has
      // no work to do on SPIR-V, so we emit an empty serial kernel.
      generate_gc_noop_kernel(task_ir_);
    } else {
      TI_ERROR("Unsupported offload type={} on SPIR-V codegen",
               task_ir_->task_name());
    }
    // Headers need global information, so it has to be delayed after visiting
    // the task IR.
    emit_headers();

    Result res;
    res.spirv_code = ir_->finalize();
    res.task_attribs = std::move(task_attribs_);
    res.arr_access = irpass::detect_external_ptr_access_in_task(task_ir_);

    return res;
  }

  void visit(OffloadedStmt *) override {
    TI_ERROR("This codegen is supposed to deal with one offloaded task");
  }

  void visit(Block *stmt) override {
    for (auto &s : stmt->statements) {
      if (offload_loop_motion_.find(s.get()) == offload_loop_motion_.end()) {
        s->accept(this);
      }
    }
  }

  void visit(PrintStmt *stmt) override {
    if (!caps_->get(DeviceCapability::spirv_has_non_semantic_info)) {
      return;
    }

    std::string formats;
    std::vector<Value> vals;

    for (auto i = 0; i < stmt->contents.size(); ++i) {
      auto const &content = stmt->contents[i];
      auto const &format = stmt->formats[i];
      if (std::holds_alternative<Stmt *>(content)) {
        auto arg_stmt = std::get<Stmt *>(content);
        TI_ASSERT(!arg_stmt->ret_type->is<TensorType>());

        auto value = ir_->query_value(arg_stmt->raw_name());
        vals.push_back(value);

        auto &&merged_format = merge_printf_specifier(
            format, data_type_format(arg_stmt->ret_type));
        // Vulkan doesn't support length, flags, or width specifier, except for
        // unsigned long.
        // https://vulkan.lunarg.com/doc/view/1.3.204.1/windows/debug_printf.html
        auto &&[format_flags, format_width, format_precision, format_length,
                format_conversion] = parse_printf_specifier(merged_format);
        if (!format_flags.empty()) {
          TI_WARN(
              "The printf flags '{}' are not supported in Vulkan, "
              "and will be discarded.",
              format_flags);
          format_flags.clear();
        }
        if (!format_width.empty()) {
          TI_WARN(
              "The printf width modifier '{}' is not supported in Vulkan, "
              "and will be discarded.",
              format_width);
          format_width.clear();
        }
        if (!format_length.empty() &&
            !(format_length == "l" &&
              (format_conversion == "u" || format_conversion == "x"))) {
          TI_WARN(
              "The printf length modifier '{}' is not supported in Vulkan, "
              "and will be discarded.",
              format_length);
          format_length.clear();
        }
        formats +=
            "%" +
            format_precision.append(format_length).append(format_conversion);
      } else {
        auto arg_str = std::get<std::string>(content);
        formats += sanitize_format_string(arg_str);
      }
    }
    ir_->call_debugprintf(formats, vals);
  }

  void visit(ConstStmt *const_stmt) override {
    auto get_const = [&](const TypedConstant &const_val) {
      auto dt = const_val.dt.ptr_removed();
      spirv::SType stype = ir_->get_primitive_type(dt);

      if (dt->is_primitive(PrimitiveTypeID::f32)) {
        return ir_->float_immediate_number(
            stype, static_cast<double>(const_val.val_f32), false);
      } else if (dt->is_primitive(PrimitiveTypeID::f16)) {
        // Ref: See taichi::lang::TypedConstant::TypedConstant()
        // FP16 is stored as FP32 on host side,
        // as some CPUs does not have native FP16 (and no libc support)
        return ir_->float_immediate_number(
            stype, static_cast<double>(const_val.val_f32), false);
      } else if (dt->is_primitive(PrimitiveTypeID::i32)) {
        return ir_->int_immediate_number(
            stype, static_cast<int64_t>(const_val.val_i32), false);
      } else if (dt->is_primitive(PrimitiveTypeID::i64)) {
        return ir_->int_immediate_number(
            stype, static_cast<int64_t>(const_val.val_i64), false);
      } else if (dt->is_primitive(PrimitiveTypeID::f64)) {
        return ir_->float_immediate_number(
            stype, static_cast<double>(const_val.val_f64), false);
      } else if (dt->is_primitive(PrimitiveTypeID::i8)) {
        return ir_->int_immediate_number(
            stype, static_cast<int64_t>(const_val.val_i8), false);
      } else if (dt->is_primitive(PrimitiveTypeID::i16)) {
        return ir_->int_immediate_number(
            stype, static_cast<int64_t>(const_val.val_i16), false);
      } else if (dt->is_primitive(PrimitiveTypeID::u1)) {
        return ir_->uint_immediate_number(
            stype, static_cast<uint64_t>(const_val.val_u1), false);
      } else if (dt->is_primitive(PrimitiveTypeID::u8)) {
        return ir_->uint_immediate_number(
            stype, static_cast<uint64_t>(const_val.val_u8), false);
      } else if (dt->is_primitive(PrimitiveTypeID::u16)) {
        return ir_->uint_immediate_number(
            stype, static_cast<uint64_t>(const_val.val_u16), false);
      } else if (dt->is_primitive(PrimitiveTypeID::u32)) {
        return ir_->uint_immediate_number(
            stype, static_cast<uint64_t>(const_val.val_u32), false);
      } else if (dt->is_primitive(PrimitiveTypeID::u64)) {
        return ir_->uint_immediate_number(
            stype, static_cast<uint64_t>(const_val.val_u64), false);
      } else {
        TI_P(data_type_name(dt));
        TI_NOT_IMPLEMENTED
        return spirv::Value();
      }
    };

    spirv::Value val = get_const(const_stmt->val);
    ir_->register_value(const_stmt->raw_name(), val);
  }

  void visit(AllocaStmt *alloca) override {
    spirv::Value ptr_val;
    auto alloca_type = alloca->ret_type.ptr_removed();
    if (auto tensor_type = alloca_type->cast<TensorType>()) {
      auto elem_num = tensor_type->get_num_elements();
      spirv::SType elem_type =
          ir_->get_primitive_type(tensor_type->get_element_type());
      spirv::SType arr_type = ir_->get_array_type(elem_type, elem_num);
      if (alloca->is_shared) {  // for shared memory / workgroup memory
        ptr_val = ir_->alloca_workgroup_array(arr_type);
        shared_array_binds_.push_back(ptr_val);
      } else {  // for function memory
        ptr_val = ir_->alloca_variable(arr_type);
      }
    } else {
      // Alloca for a single variable
      spirv::SType src_type = ir_->get_primitive_type(alloca_type);
      ptr_val = ir_->alloca_variable(src_type);
      ir_->store_variable(ptr_val, ir_->get_zero(src_type));
    }
    ir_->register_value(alloca->raw_name(), ptr_val);
  }

  void visit(MatrixPtrStmt *stmt) override {
    spirv::Value ptr_val;
    spirv::Value origin_val = ir_->query_value(stmt->origin->raw_name());
    spirv::Value offset_val = ir_->query_value(stmt->offset->raw_name());
    auto dt = stmt->element_type().ptr_removed();
    if (stmt->offset_used_as_index()) {
      if (stmt->origin->is<AllocaStmt>()) {
        spirv::SType ptr_type = ir_->get_pointer_type(
            ir_->get_primitive_type(dt), origin_val.stype.storage_class);
        ptr_val = ir_->make_value(spv::OpAccessChain, ptr_type, origin_val,
                                  offset_val);
        if (stmt->origin->as<AllocaStmt>()->is_shared) {
          ptr_to_buffers_[stmt] = ptr_to_buffers_[stmt->origin];
        }
      } else if (stmt->origin->is<GlobalTemporaryStmt>()) {
        spirv::Value dt_bytes = ir_->int_immediate_number(
            ir_->i32_type(), ir_->get_primitive_type_size(dt), false);
        spirv::Value offset_bytes = ir_->mul(dt_bytes, offset_val);
        ptr_val = ir_->add(origin_val, offset_bytes);
        ptr_to_buffers_[stmt] = ptr_to_buffers_[stmt->origin];
      } else {
        TI_NOT_IMPLEMENTED;
      }
    } else {  // offset used as bytes
      ptr_val = ir_->add(origin_val, ir_->cast(origin_val.stype, offset_val));
      ptr_to_buffers_[stmt] = ptr_to_buffers_[stmt->origin];
    }
    ir_->register_value(stmt->raw_name(), ptr_val);
  }

  void visit(LocalLoadStmt *stmt) override {
    auto ptr = stmt->src;
    spirv::Value ptr_val = ir_->query_value(ptr->raw_name());
    spirv::Value val = ir_->load_variable(
        ptr_val, ir_->get_primitive_type(stmt->element_type()));
    ir_->register_value(stmt->raw_name(), val);
  }

  void visit(LocalStoreStmt *stmt) override {
    spirv::Value ptr_val = ir_->query_value(stmt->dest->raw_name());
    spirv::Value val = ir_->query_value(stmt->val->raw_name());
    ir_->store_variable(ptr_val, val);
  }

  void visit(GetRootStmt *stmt) override {
    const int root_id = snode_to_root_.at(stmt->root()->id);
    root_stmts_[root_id] = stmt;
    // get_buffer_value({BufferType::Root, root_id}, PrimitiveType::u32);
    spirv::Value root_val = make_pointer(0);
    ir_->register_value(stmt->raw_name(), root_val);
  }

  void visit(ClearListStmt *stmt) override {
    // No-op on SPIR-V/GFX backend. The runtime fills the listgen buffer with
    // zeros at dispatch time of the corresponding listgen task
    // (see taichi/runtime/gfx/runtime.cpp `if (attribs.task_type ==
    // OffloadedTaskType::listgen) { ... buffer_fill(0) ... }`). The
    // ClearListStmt thus appears in a serial offload that emits an empty
    // kernel here.
    (void)stmt;
  }

  void visit(GetChStmt *stmt) override {
    // TODO: GetChStmt -> GetComponentStmt ?
    const int root = snode_to_root_.at(stmt->input_snode->id);

    const auto &snode_descs = compiled_structs_[root].snode_descriptors;
    auto *out_snode = stmt->output_snode;
    TI_ASSERT(snode_descs.at(stmt->input_snode->id).get_child(stmt->chid) ==
              out_snode);

    const auto &desc = snode_descs.at(out_snode->id);

    spirv::Value input_ptr_val = ir_->query_value(stmt->input_ptr->raw_name());

    // G9.2 (2026-04-30): quant_array forwards both the byte address and
    // the bit offset of its parent unchanged -- the only "child" of a
    // quant_array is the quant `place`, which logically lives at the
    // same packed bit position computed by SNodeLookupStmt above.
    // Mirrors codegen_llvm.cpp::TaskCodeGenLLVM::visit(GetChStmt) for
    // SNodeType::quant_array.
    if (stmt->input_snode->type == SNodeType::quant_array) {
      ir_->register_value(stmt->raw_name(), input_ptr_val);
      auto it = bit_ptr_bit_offsets_.find(stmt->input_ptr);
      if (it != bit_ptr_bit_offsets_.end()) {
        bit_ptr_bit_offsets_[stmt] = it->second;
      }
      if (out_snode->is_place()) {
        TI_ASSERT(ptr_to_buffers_.count(stmt) == 0);
        // B-3.c-2: leaves under a pointer ancestor live in that pointer's
        // independent pool buffer (when independent_pool=ON); otherwise
        // root_buffer (OFF default, byte-equal).
        ptr_to_buffers_[stmt] = container_buffer_info(out_snode, root);
      }
      return;
    }

    // G9.3b (2026-05-01): GetChStmt(bit_struct -> place) produces a
    // bit-pointer whose bit offset is the constant
    // BitStructType::get_member_bit_offset(child_id_in_bit_struct).
    // The byte address (parent's physical word) is forwarded
    // unchanged.  Mirrors codegen_llvm.cpp::TaskCodeGenLLVM::visit
    // (GetChStmt) for the is_bit_pointer branch.
    if (stmt->input_snode->type == SNodeType::bit_struct) {
      ir_->register_value(stmt->raw_name(), input_ptr_val);
      auto *bst = stmt->input_snode->dt->as<BitStructType>();
      const int member_bit_off =
          bst->get_member_bit_offset(out_snode->id_in_bit_struct);
      DataType phys = bst->get_physical_type();
      SType phys_uint_stype =
          ir_->get_primitive_type(ir_->get_taichi_uint_type(phys));
      bit_ptr_bit_offsets_[stmt] = ir_->uint_immediate_number(
          phys_uint_stype, static_cast<uint64_t>(member_bit_off));
      if (out_snode->is_place()) {
        TI_ASSERT(ptr_to_buffers_.count(stmt) == 0);
        // B-3.c-2: see container_buffer_info note above.
        ptr_to_buffers_[stmt] = container_buffer_info(out_snode, root);
      }
      return;
    }

    spirv::Value offset = make_pointer(desc.mem_offset_in_parent_cell);
    spirv::Value val = ir_->add(input_ptr_val, offset);
    ir_->register_value(stmt->raw_name(), val);

    if (out_snode->is_place()) {
      TI_ASSERT(ptr_to_buffers_.count(stmt) == 0);
      // B-3.c-2: see container_buffer_info note above.
      ptr_to_buffers_[stmt] = container_buffer_info(out_snode, root);
    }
    // G9.3b: BitStructStoreStmt's `ptr` is a GetChStmt(dense -> bit_struct)
    // (or its forwarding SNodeLookupStmt). The bit_struct itself is not a
    // place, so we must explicitly register the buffer mapping so that
    // at_buffer() can locate the root buffer in the BitStructStoreStmt
    // visitor.
    if (out_snode->type == SNodeType::bit_struct) {
      // B-3.c-2: bit_struct under a pointer ancestor uses that pool buffer.
      ptr_to_buffers_[stmt] = container_buffer_info(out_snode, root);
    }
  }

  enum class ActivationOp { activate, deactivate, query };

  // B-3.c-2（2026-05）：返回一个 byte_offset 用于寻址 `sn` 容器（slot 表 /
  // bitmask 字 / dynamic length / cell 数据）时所在的 buffer。当 sn 有最近
  // pointer 祖先且该祖先 pool_buffer_binding_id>=0 时，sn 的容器位于该
  // pointer 的独立 NodeAllocatorPool buffer；否则位于 root buffer。OFF
  // 默认所有 pointer.binding_id==-1，结果总是 (Root, root_id)，行为字节
  // 等价旧路径。
  BufferInfo container_buffer_info(const SNode *sn, int root_id) {
    for (auto *p = sn ? sn->parent : nullptr; p != nullptr; p = p->parent) {
      if (p->type == SNodeType::pointer) {
        const auto &cs = compiled_structs_[root_id];
        auto it = cs.pointer_contracts.find(p->id);
        if (it != cs.pointer_contracts.end() &&
            it->second.pool_buffer_binding_id >= 0) {
          return BufferInfo(BufferType::NodeAllocatorPool,
                            it->second.pool_buffer_binding_id);
        }
      }
    }
    return BufferInfo(BufferType::Root, root_id);
  }

  spirv::Value container_buffer_value(const SNode *sn, int root_id) {
    return get_buffer_value(container_buffer_info(sn, root_id),
                            PrimitiveType::u32);
  }

  spirv::Value bitmasked_activation(ActivationOp op,
                                    spirv::Value parent_ptr,
                                    int root_id,
                                    const SNode *sn,
                                    spirv::Value input_index) {
    spirv::SType ptr_dt = parent_ptr.stype;
    const auto &snode_descs = compiled_structs_[root_id].snode_descriptors;
    const auto &desc = snode_descs.at(sn->id);

    auto bitmask_word_index =
        ir_->make_value(spv::OpShiftRightLogical, ptr_dt, input_index,
                        ir_->uint_immediate_number(ptr_dt, 5));
    auto bitmask_bit_index =
        ir_->make_value(spv::OpBitwiseAnd, ptr_dt, input_index,
                        ir_->uint_immediate_number(ptr_dt, 31));
    auto bitmask_mask = ir_->make_value(spv::OpShiftLeftLogical, ptr_dt,
                                        ir_->const_i32_one_, bitmask_bit_index);

    // B-3.c-2: bitmasked container lives in the parent's container buffer
    // (root_buffer when no pointer ancestor; pointer ancestor's pool buffer
    // otherwise). OFF default has no pointer.binding_id>=0 → root buffer.
    auto buffer = container_buffer_value(sn, root_id);
    auto bitmask_word_ptr =
        ir_->make_value(spv::OpShiftLeftLogical, ptr_dt, bitmask_word_index,
                        ir_->uint_immediate_number(ir_->u32_type(), 2));
    bitmask_word_ptr = ir_->add(
        bitmask_word_ptr,
        make_pointer(desc.cell_stride * desc.snode->num_cells_per_container));
    bitmask_word_ptr = ir_->add(parent_ptr, bitmask_word_ptr);
    bitmask_word_ptr = ir_->make_value(
        spv::OpShiftRightLogical, ir_->u32_type(), bitmask_word_ptr,
        ir_->uint_immediate_number(ir_->u32_type(), 2));
    bitmask_word_ptr =
        ir_->struct_array_access(ir_->u32_type(), buffer, bitmask_word_ptr);

    if (op == ActivationOp::activate) {
      return ir_->make_value(spv::OpAtomicOr, ir_->u32_type(), bitmask_word_ptr,
                             /*scope=*/ir_->const_i32_one_,
                             /*semantics=*/ir_->const_i32_zero_, bitmask_mask);
    } else if (op == ActivationOp::deactivate) {
      bitmask_mask = ir_->make_value(spv::OpNot, ir_->u32_type(), bitmask_mask);
      return ir_->make_value(spv::OpAtomicAnd, ir_->u32_type(),
                             bitmask_word_ptr,
                             /*scope=*/ir_->const_i32_one_,
                             /*semantics=*/ir_->const_i32_zero_, bitmask_mask);
    } else {
      auto bitmask_val = ir_->load_variable(bitmask_word_ptr, ir_->u32_type());
      auto bit = ir_->make_value(spv::OpShiftRightLogical, ir_->u32_type(),
                                 bitmask_val, bitmask_bit_index);
      bit = ir_->make_value(spv::OpBitwiseAnd, ir_->u32_type(), bit,
                            ir_->uint_immediate_number(ir_->u32_type(), 1));
      return ir_->make_value(spv::OpUGreaterThan, ir_->bool_type(), bit,
                             ir_->uint_immediate_number(ir_->u32_type(), 0));
    }
  }

  // ----------------------------------------------------------------------
  // Phase 2b: pointer SNode helpers (vulkan_sparse_experimental).
  //
  // The container at `parent_ptr + 4*i` stores a u32 slot value:
  //   slot == 0      -> cell is inactive
  //   slot == k+1    -> cell currently maps to pool index k
  //
  // Pool data and the bump watermark live at fixed absolute offsets in the
  // root buffer (see snode_struct_compiler.cpp). On activate we bump the
  // watermark and CAS the slot; on read we just translate slot to pool
  // address. Per-cell deactivate writes 0 to the slot but never recycles
  // the pool entry -- whole-pool reset goes through the device allocator's
  // clear_all() (Phase 2c hook).
  // ----------------------------------------------------------------------

  // Translate a u32 byte offset into root buffer to a u32 slot pointer.
  spirv::Value pointer_slot_ptr(spirv::Value root_buffer_u32,
                                spirv::Value parent_byte_offset,
                                spirv::Value index_u32) {
    auto u32_t = ir_->u32_type();
    // word_offset = parent_byte_offset / 4 + index
    auto parent_word = ir_->make_value(
        spv::OpShiftRightLogical, u32_t, parent_byte_offset,
        ir_->uint_immediate_number(u32_t, 2));
    auto slot_word_idx = ir_->add(parent_word, index_u32);
    return ir_->struct_array_access(u32_t, root_buffer_u32, slot_word_idx);
  }

  // Returns the byte offset (in root buffer) of cell `index_u32` in the
  // pointer SNode `sn`. If `do_activate` is true, atomically allocates a
  // pool slot when the cell is currently inactive. If `do_activate` is
  // false and the cell is inactive, returns the pool's cell-0 address as a
  // safe fallback (matches LLVM's "ambient_elements" semantics for the
  // smoke-test purpose; reads of inactive cells are user error).
  spirv::Value pointer_lookup_or_activate(spirv::Value parent_byte_offset,
                                          int root_id,
                                          const SNode *sn,
                                          spirv::Value index_u32,
                                          bool do_activate) {
    const auto &snode_descs = compiled_structs_[root_id].snode_descriptors;
    const auto &desc = snode_descs.at(sn->id);
    // 路线 B B-1/B-4：通过 SpirvAllocatorContract 间接读 pointer SNode 的池
    // 元数据。SNodeDescriptor 不再存 pointer_* 字段（B-4 已删除），contract
    // 是 pool/watermark/freelist/ambient 的唯一来源。
    const auto &contract =
        compiled_structs_[root_id].pointer_contracts.at(sn->id);

    auto u32_t = ir_->u32_type();
    // B-3.c-2（2026-05）：parent_byte_offset 寻址在 sn 的容器 buffer 里
    // （= sn 最近 pointer 祖先的独立池，或 root）。OFF 默认无 pointer 祖
    // 先有 binding_id>=0，等价于 root buffer。变量名保留 root_buffer 以
    // 减小本次 diff，但实际语义为 "parent container buffer"。
    auto root_buffer = container_buffer_value(sn, root_id);
    // B-3.c-1（2026-05）：sn 自身池的元数据 + pool_data + ambient 在 ON
    // 时位于 sn 自身的独立 NodeAllocatorPool buffer（B-3.c-2 把
    // pool_data + ambient 一并移入）；OFF 时退化为 root buffer。
    auto pool_meta_buffer =
        contract.pool_buffer_binding_id >= 0
            ? get_buffer_value(
                  BufferInfo(BufferType::NodeAllocatorPool,
                             contract.pool_buffer_binding_id),
                  PrimitiveType::u32)
            : root_buffer;

    auto idx_u32 = ir_->cast(u32_t, index_u32);
    auto slot_ptr =
        pointer_slot_ptr(root_buffer, parent_byte_offset, idx_u32);

    spirv::Value final_slot;
    if (do_activate) {
      // B-2.b（2026-05）：原 #if TI_VULKAN_POINTER_CAS_MARKER 下放为运行时
      // contract.alloc_protocol 分支，与 layout 路径同源于 CompileConfig
      // 上的 vulkan_pointer_cas_marker。CasMarker 路径与原宏-ON 路径字节等价；
      // Legacy 路径与原宏-OFF (vanilla 1.7.4 atomicIAdd-first) 字节等价。
      if (contract.alloc_protocol ==
          SpirvAllocatorContract::AllocProtocol::CasMarker) {
      // ----------------------------------------------------------------
      // G1.a: race-correct alloc protocol (CAS-marker-first).
      //
      // Old protocol (kept as fallback when the macro is undefined):
      //   atomicIAdd(watermark, 1)  -- always consumes a pool slot per
      //                                 racing thread, even though only
      //                                 one thread can win the slot CAS.
      //   atomicCAS(slot, 0, new)   -- losers' new_slot is wasted.
      //
      // New protocol:
      //   1. atomicCAS(slot, 0, BUSY): races for the right to allocate.
      //      Winner sees prior == 0; losers see prior == BUSY (someone
      //      claimed but not finished) or prior >= 1 (someone finished).
      //   2. Winner: atomicIAdd(watermark, 1) ONCE -> compute new_slot
      //      (or 0 if pool full); atomicStore(slot, new_slot).
      //   3. Loser: structured spin-loop on atomicLoad(slot) until value
      //      is no longer BUSY; consume that value.
      //
      // NOTE: the legacy code also had an OUTER non-atomic load
      // ("cur_slot") to skip the CAS for already-active slots. With BUSY
      // now in the value space, a non-atomic read can return BUSY and a
      // raw "use" of that value yields BUSY-1 = 0xFFFFFFFE which causes
      // OOB pool writes. The CAS-marker protocol replaces both branches
      // with a single CAS that handles all 3 states (0 / BUSY / resolved).
      // ----------------------------------------------------------------
      auto busy_v = ir_->uint_immediate_number(u32_t, 0xFFFFFFFFu);
      auto zero_v = ir_->uint_immediate_number(u32_t, 0);
      auto cap_v = ir_->uint_immediate_number(
          u32_t, contract.pool_capacity);

      // Step 1: stake a claim with BUSY marker.
      auto cas_marker = ir_->make_value(
          spv::OpAtomicCompareExchange, u32_t, slot_ptr,
          /*scope=*/ir_->const_i32_one_,
          /*semantics_eq=*/ir_->const_i32_zero_,
          /*semantics_uneq=*/ir_->const_i32_zero_, busy_v, zero_v);
      auto we_won = ir_->make_value(spv::OpIEqual, ir_->bool_type(),
                                    cas_marker, zero_v);

      spirv::Label winner_label = ir_->new_label();
      spirv::Label waiter_label = ir_->new_label();
      spirv::Label alloc_done_label = ir_->new_label();
      ir_->make_inst(spv::OpSelectionMerge, alloc_done_label,
                     spv::SelectionControlMaskNone);
      ir_->make_inst(spv::OpBranchConditional, we_won, winner_label,
                     waiter_label);

      // ---- winner ----
      ir_->start_label(winner_label);
      auto wm_word_idx = ir_->uint_immediate_number(
          u32_t, contract.watermark_offset / 4u);
      auto wm_ptr =
          ir_->struct_array_access(u32_t, pool_meta_buffer, wm_word_idx);
      // Outputs of the winner block, joined into the alloc_done OpPhi:
      //   new_slot                  -- the slot value to publish
      //   winner_terminator_label   -- block whose terminator branches to
      //                                alloc_done (predecessor for OpPhi)
      // Both branches below (freelist-pop-then-bump vs direct bump) must
      // assign to these two variables before falling through.
      spirv::Value new_slot;
      spirv::Label winner_terminator_label;
      if (contract.has_freelist) {
        // G1.b (2026-04-30 → B-2.a 2026-05): try to pop a recycled slot
        // off the freelist before bumping the watermark. The freelist is a
        // singly-linked stack rooted at freelist_head (offset known at
        // codegen time). Encoding: 0 = empty, otherwise (pool_index + 1)
        // — same as a slot value, so a successful pop yields a directly-
        // usable new_slot.
        //
        //   loop:
        //     head = atomicLoad(free_head)
        //     if head == 0: goto bump_path
        //     next = atomicLoad(freelist_links[head - 1])
        //     cas  = atomicCAS(free_head, head, next)
        //     if cas == head: new_slot_pop = head; goto publish_path
        //     // ABA-safe under our "no concurrent push+pop" constraint;
        //     // we just retry on contention from another popping thread.
        //
        // bump_path: same as the no-freelist code below — atomicIAdd
        // watermark and capacity-clamp. publish_path stores new_slot to
        // slot_ptr (this also releases waiters out of the spin loop).
        // Gating moved from `#if TI_VULKAN_POINTER_FREELIST` to runtime
        // `contract.has_freelist` (B-2.a); layout-side `#if` in
        // snode_struct_compiler still drives contract.has_freelist for
        // now, so behavior is byte-equivalent.
        auto fhead_word_idx = ir_->uint_immediate_number(
            u32_t, contract.freelist_head_offset / 4u);
        auto fhead_ptr =
            ir_->struct_array_access(u32_t, pool_meta_buffer, fhead_word_idx);
        auto flinks_word_base = ir_->uint_immediate_number(
            u32_t, contract.freelist_links_offset / 4u);

        spirv::Label fl_head_lbl = ir_->new_label();
        spirv::Label fl_body_lbl = ir_->new_label();
        spirv::Label fl_continue_lbl = ir_->new_label();
        spirv::Label fl_merge_lbl = ir_->new_label();
        spirv::Label bump_lbl = ir_->new_label();
        spirv::Label publish_lbl = ir_->new_label();

        ir_->make_inst(spv::OpBranch, fl_head_lbl);

        // fl_head: load free_head, decide empty/non-empty.
        ir_->start_label(fl_head_lbl);
        auto fhead_cur =
            ir_->make_value(spv::OpAtomicLoad, u32_t, fhead_ptr,
                            /*scope=*/ir_->const_i32_one_,
                            /*semantics=*/ir_->const_i32_zero_);
        auto fhead_empty = ir_->make_value(spv::OpIEqual, ir_->bool_type(),
                                           fhead_cur, zero_v);
        ir_->make_inst(spv::OpLoopMerge, fl_merge_lbl, fl_continue_lbl,
                       spv::LoopControlMaskNone);
        ir_->make_inst(spv::OpBranchConditional, fhead_empty, bump_lbl,
                       fl_body_lbl);

        // fl_body: try to CAS free_head from head to next.
        ir_->start_label(fl_body_lbl);
        auto fhead_node =
            ir_->sub(fhead_cur, ir_->uint_immediate_number(u32_t, 1));
        auto link_word_idx = ir_->add(flinks_word_base, fhead_node);
        auto link_ptr =
            ir_->struct_array_access(u32_t, pool_meta_buffer, link_word_idx);
        auto fhead_next =
            ir_->make_value(spv::OpAtomicLoad, u32_t, link_ptr,
                            /*scope=*/ir_->const_i32_one_,
                            /*semantics=*/ir_->const_i32_zero_);
        auto fhead_cas = ir_->make_value(
            spv::OpAtomicCompareExchange, u32_t, fhead_ptr,
            /*scope=*/ir_->const_i32_one_,
            /*semantics_eq=*/ir_->const_i32_zero_,
            /*semantics_uneq=*/ir_->const_i32_zero_, fhead_next, fhead_cur);
        auto fl_won = ir_->make_value(spv::OpIEqual, ir_->bool_type(),
                                      fhead_cas, fhead_cur);
        ir_->make_inst(spv::OpBranchConditional, fl_won, fl_merge_lbl,
                       fl_continue_lbl);

        // fl_continue: lost the head CAS to another popper, retry.
        ir_->start_label(fl_continue_lbl);
        ir_->make_inst(spv::OpBranch, fl_head_lbl);

        // bump_path: freelist empty, fall back to watermark bump.
        ir_->start_label(bump_lbl);
        auto old_wm = ir_->make_value(spv::OpAtomicIAdd, u32_t, wm_ptr,
                                      /*scope=*/ir_->const_i32_one_,
                                      /*semantics=*/ir_->const_i32_zero_,
                                      ir_->uint_immediate_number(u32_t, 1));
        // C-1.b (2026-05): no in_cap clamp here. OOC writes (old_wm >= cap)
        // produce new_slot >= cap+1, leading to OOB pool_data access. The
        // SSBO range check in the driver surfaces this as device-lost
        // (matches CPU/CUDA assert(slot < cap) semantics; honest hardware
        // OOM rather than silent fallback to slot 0).
        (void)cap_v;
        auto new_slot_bump =
            ir_->add(old_wm, ir_->uint_immediate_number(u32_t, 1));
        ir_->make_inst(spv::OpBranch, publish_lbl);

        // fl_merge: freelist pop succeeded; new_slot = fhead_cur (the
        // popped head value, still in slot-value encoding).
        ir_->start_label(fl_merge_lbl);
        ir_->make_inst(spv::OpBranch, publish_lbl);

        // publish: phi(new_slot_bump from bump_lbl, fhead_cur from
        // fl_merge).
        ir_->start_label(publish_lbl);
        new_slot = ir_->make_value(spv::OpPhi, u32_t, new_slot_bump,
                                   bump_lbl, fhead_cur, fl_merge_lbl);
        // Publish the resolved slot. Loser's spin will see this on its
        // next atomicLoad iteration and exit.
        ir_->make_inst(spv::OpAtomicStore, slot_ptr,
                       /*scope=*/ir_->const_i32_one_,
                       /*semantics=*/ir_->const_i32_zero_, new_slot);
        ir_->make_inst(spv::OpBranch, alloc_done_label);
        // For OpPhi at alloc_done_label: winner block's terminator is in
        // publish_lbl, so the predecessor passed to phi must be
        // publish_lbl.
        winner_terminator_label = publish_lbl;
      } else {
        auto old_wm = ir_->make_value(spv::OpAtomicIAdd, u32_t, wm_ptr,
                                      /*scope=*/ir_->const_i32_one_,
                                      /*semantics=*/ir_->const_i32_zero_,
                                      ir_->uint_immediate_number(u32_t, 1));
        // C-1.b (2026-05): no OOC clamp; see bump_lbl branch above.
        (void)cap_v;
        new_slot = ir_->add(old_wm, ir_->uint_immediate_number(u32_t, 1));
        // Publish the resolved slot. Loser's spin will see this on its
        // next atomicLoad iteration and exit.
        ir_->make_inst(spv::OpAtomicStore, slot_ptr,
                       /*scope=*/ir_->const_i32_one_,
                       /*semantics=*/ir_->const_i32_zero_, new_slot);
        ir_->make_inst(spv::OpBranch, alloc_done_label);
        winner_terminator_label = winner_label;
      }

      // ---- waiter ----
      // Either cas_marker == BUSY (someone resolving) or cas_marker >= 1
      // (someone already resolved). Either way, spin reads the slot
      // until it is non-BUSY; the spin's first iteration completes
      // immediately in the already-resolved case.
      ir_->start_label(waiter_label);
      spirv::Label spin_head = ir_->new_label();
      spirv::Label spin_continue = ir_->new_label();
      spirv::Label spin_merge = ir_->new_label();
      ir_->make_inst(spv::OpBranch, spin_head);

      ir_->start_label(spin_head);
      auto spin_cur =
          ir_->make_value(spv::OpAtomicLoad, u32_t, slot_ptr,
                          /*scope=*/ir_->const_i32_one_,
                          /*semantics=*/ir_->const_i32_zero_);
      auto spin_busy = ir_->make_value(spv::OpIEqual, ir_->bool_type(),
                                       spin_cur, busy_v);
      ir_->make_inst(spv::OpLoopMerge, spin_merge, spin_continue,
                     spv::LoopControlMaskNone);
      ir_->make_inst(spv::OpBranchConditional, spin_busy, spin_continue,
                     spin_merge);

      ir_->start_label(spin_continue);
      ir_->make_inst(spv::OpBranch, spin_head);

      ir_->start_label(spin_merge);
      // spin_cur is dominated here (defined in spin_head, single
      // predecessor of spin_merge).
      ir_->make_inst(spv::OpBranch, alloc_done_label);

      // ---- alloc_done: phi the winner's new_slot vs waiter's spin_cur ----
      ir_->start_label(alloc_done_label);
      final_slot = ir_->make_value(spv::OpPhi, u32_t, new_slot,
                                   winner_terminator_label, spin_cur,
                                   spin_merge);
      } else {  // contract.alloc_protocol == Legacy
      // ---- legacy atomicIAdd-first protocol (race-inflates watermark) ----
      // Read current slot non-atomically. The CAS below is the source of
      // truth for "who allocated"; a stale 0-read just sends us through
      // the allocate path where the CAS will tell us the real owner.
      auto cur_slot = ir_->load_variable(slot_ptr, u32_t);
      auto cond_zero = ir_->make_value(spv::OpIEqual, ir_->bool_type(),
                                       cur_slot,
                                       ir_->uint_immediate_number(u32_t, 0));

      spirv::Label alloc_label = ir_->new_label();
      spirv::Label use_label = ir_->new_label();
      spirv::Label merge_label = ir_->new_label();
      ir_->make_inst(spv::OpSelectionMerge, merge_label,
                     spv::SelectionControlMaskNone);
      ir_->make_inst(spv::OpBranchConditional, cond_zero, alloc_label,
                     use_label);

      // alloc path
      ir_->start_label(alloc_label);
      auto wm_word_idx = ir_->uint_immediate_number(
          u32_t, contract.watermark_offset / 4u);
      auto wm_ptr =
          ir_->struct_array_access(u32_t, pool_meta_buffer, wm_word_idx);
      auto old_wm = ir_->make_value(spv::OpAtomicIAdd, u32_t, wm_ptr,
                                    /*scope=*/ir_->const_i32_one_,
                                    /*semantics=*/ir_->const_i32_zero_,
                                    ir_->uint_immediate_number(u32_t, 1));
      auto cap_v = ir_->uint_immediate_number(
          u32_t, contract.pool_capacity);
      // C-1.b (2026-05): no OOC clamp; see CasMarker branch comment.
      (void)cap_v;
      auto new_slot = ir_->add(old_wm, ir_->uint_immediate_number(u32_t, 1));
      auto cas_old = ir_->make_value(
          spv::OpAtomicCompareExchange, u32_t, slot_ptr,
          /*scope=*/ir_->const_i32_one_,
          /*semantics_eq=*/ir_->const_i32_zero_,
          /*semantics_uneq=*/ir_->const_i32_zero_, new_slot,
          ir_->uint_immediate_number(u32_t, 0));
      auto we_won = ir_->make_value(spv::OpIEqual, ir_->bool_type(), cas_old,
                                    ir_->uint_immediate_number(u32_t, 0));
      auto alloc_slot = ir_->make_value(spv::OpSelect, u32_t, we_won, new_slot,
                                        cas_old);
      ir_->make_inst(spv::OpBranch, merge_label);

      // use path
      ir_->start_label(use_label);
      ir_->make_inst(spv::OpBranch, merge_label);

      ir_->start_label(merge_label);
      // Phi(alloc_slot from alloc_label, cur_slot from use_label)
      final_slot = ir_->make_value(spv::OpPhi, u32_t, alloc_slot, alloc_label,
                                   cur_slot, use_label);
      }
    } else {
      final_slot = ir_->load_variable(slot_ptr, u32_t);
    }

    // effective_slot = (final_slot == 0) ? 0 : (final_slot - 1)
    auto is_zero = ir_->make_value(spv::OpIEqual, ir_->bool_type(), final_slot,
                                   ir_->uint_immediate_number(u32_t, 0));
    auto slot_minus_one =
        ir_->sub(final_slot, ir_->uint_immediate_number(u32_t, 1));
    auto effective_slot = ir_->make_value(
        spv::OpSelect, u32_t, is_zero,
        ir_->uint_immediate_number(u32_t, 0), slot_minus_one);

    auto cell_stride_v =
        ir_->uint_immediate_number(u32_t, contract.cell_stride_bytes);
    auto pool_offset_v = ir_->uint_immediate_number(
        u32_t, contract.pool_data_offset);
    spirv::Value cell_byte_offset;
    if (contract.allocator_kind ==
        SpirvAllocatorContract::AllocatorKind::Chunked) {
      // C-2.3 (2026-05): chunked-allocator addressing。把 effective_slot
      // 拆为 (chunk_idx, local_slot) 对：
      //   chunk_idx  = slot >> chunk_log2_capacity
      //   local_slot = slot &  ((1 << chunk_log2_capacity) - 1)
      //   cell_byte  = pool_data_offset
      //              + chunk_idx * (1 << chunk_log2_capacity) * cell_stride
      //              + local_slot * cell_stride
      // 当 1 << chunk_log2_capacity >= pool_capacity（由 snode_struct_compiler
      // 在 C-2.3 保证）时，slot 恒落在 chunk[0]，与 Bump 路径字节等价；
      // C-2.4 起才允许 chunk_idx > 0 并接 descriptor array 拼出多 chunk
      // physical buffer。
      const uint32_t chunk_log2 = contract.chunk_log2_capacity;
      TI_ASSERT(chunk_log2 < 32);
      const uint32_t chunk_size_cells = 1u << chunk_log2;
      const uint64_t chunk_size_bytes_64 =
          static_cast<uint64_t>(chunk_size_cells) *
          static_cast<uint64_t>(contract.cell_stride_bytes);
      TI_ASSERT(chunk_size_bytes_64 <= 0xffffffffull);
      auto chunk_log2_v = ir_->uint_immediate_number(u32_t, chunk_log2);
      auto chunk_mask_v =
          ir_->uint_immediate_number(u32_t, chunk_size_cells - 1u);
      auto chunk_size_bytes_v = ir_->uint_immediate_number(
          u32_t, static_cast<uint32_t>(chunk_size_bytes_64));
      auto chunk_idx = ir_->make_value(
          spv::OpShiftRightLogical, u32_t, effective_slot, chunk_log2_v);
      auto local_slot = ir_->make_value(
          spv::OpBitwiseAnd, u32_t, effective_slot, chunk_mask_v);
      auto chunk_byte_off = ir_->mul(chunk_idx, chunk_size_bytes_v);
      auto local_byte_off = ir_->mul(local_slot, cell_stride_v);
      cell_byte_offset = ir_->add(
          pool_offset_v, ir_->add(chunk_byte_off, local_byte_off));
    } else {
      cell_byte_offset = ir_->add(
          pool_offset_v, ir_->mul(effective_slot, cell_stride_v));
    }
    // G10-P2 (2026-04-30 → B-2.a 2026-05): for inactive reads
    // (do_activate=false), route the byte offset to the per-pointer-SNode
    // ambient zone instead of pool[0]. The ambient zone is cell_stride
    // bytes of zero-initialized memory at contract.ambient_offset,
    // never written by any kernel; this matches LLVM's ambient_val_addr
    // semantics so that x[inactive_idx] reads as 0. do_activate=true keeps
    // the pool[0] fallback for OOC writes (silent loss, documented).
    // Gating moved from compile-time `#if TI_VULKAN_POINTER_AMBIENT_ZONE`
    // to runtime `contract.has_ambient_zone` (B-2.a). Layout is still
    // gated by the same compile-time macro in snode_struct_compiler, so
    // contract.has_ambient_zone == (macro defined) at this stage; B-2.b
    // will make the gate fully runtime via CompileConfig.
    if (contract.has_ambient_zone && !do_activate) {
      auto ambient_offset_v = ir_->uint_immediate_number(
          u32_t, contract.ambient_offset);
      cell_byte_offset = ir_->make_value(
          spv::OpSelect, u32_t, is_zero, ambient_offset_v, cell_byte_offset);
    }
    return cell_byte_offset;
  }

  // is_active: returns u32 (1 if active, else 0) -- callers cast to ret_type.
  spirv::Value pointer_is_active(spirv::Value parent_byte_offset,
                                 int root_id,
                                 const SNode *sn,
                                 spirv::Value index_u32) {
    auto u32_t = ir_->u32_type();
    // B-3.c-2: parent's container buffer; OFF default = root.
    auto root_buffer = container_buffer_value(sn, root_id);
    auto idx_u32 = ir_->cast(u32_t, index_u32);
    auto slot_ptr =
        pointer_slot_ptr(root_buffer, parent_byte_offset, idx_u32);
    auto slot_value = ir_->load_variable(slot_ptr, u32_t);
    return ir_->make_value(spv::OpUGreaterThan, ir_->bool_type(), slot_value,
                           ir_->uint_immediate_number(u32_t, 0));
  }

  // deactivate: write 0 to slot. The pool entry is leaked until clear_all.
  void pointer_deactivate(spirv::Value parent_byte_offset,
                          int root_id,
                          const SNode *sn,
                          spirv::Value index_u32) {
    const auto &snode_descs = compiled_structs_[root_id].snode_descriptors;
    const auto &desc = snode_descs.at(sn->id);
    // 路线 B B-1：见 pointer_lookup_or_activate 同名注释。
    const auto &contract =
        compiled_structs_[root_id].pointer_contracts.at(sn->id);
    auto u32_t = ir_->u32_type();
    // B-3.c-2: parent's container buffer; OFF default = root.
    auto root_buffer = container_buffer_value(sn, root_id);
    // B-3.c-1/c-2：sn 自身池的元数据 (freelist) 在 ON 时走 sn 自己的
    // NodeAllocatorPool buffer；slot_ptr 走 parent 的容器 buffer。
    auto pool_meta_buffer =
        contract.pool_buffer_binding_id >= 0
            ? get_buffer_value(
                  BufferInfo(BufferType::NodeAllocatorPool,
                             contract.pool_buffer_binding_id),
                  PrimitiveType::u32)
            : root_buffer;
    auto idx_u32 = ir_->cast(u32_t, index_u32);
    auto slot_ptr =
        pointer_slot_ptr(root_buffer, parent_byte_offset, idx_u32);
    if (contract.has_freelist) {
      // G1.b (2026-04-30 → B-2.a 2026-05): push the slot onto the freelist
      // before clearing it, so a later activate can recycle the pool cell
      // instead of consuming a fresh watermark slot. The freelist is a
      // singly-linked stack rooted at freelist_head; both head and
      // freelist_links[i] use slot-value encoding (0 = empty/tail,
      // otherwise pool_index+1).
      //
      //   old_slot = atomicExchange(slot, 0)
      //   if old_slot != 0 and old_slot != BUSY:
      //     loop:
      //       head = atomicLoad(free_head)
      //       atomicStore(freelist_links[old_slot - 1], head)
      //       cas  = atomicCAS(free_head, head, old_slot)
      //       if cas == head: done
      //
      // Constraints (documented in §4.6.4):
      //   - deactivate and activate MUST NOT race on the same root buffer
      //     within a single dispatch (ABA on free_head is unprotected).
      //   - BUSY (0xFFFFFFFFu) means another invocation is mid-allocation;
      //     we leave it alone (do not exchange to 0) and skip the push, so
      //     that activate's spin loop converges normally.
      //
      // SPIR-V structured-control-flow note: the outer Selection (skip if
      // 0/BUSY) and the inner Loop must use DISTINCT merge blocks. We use
      // deact_done_lbl as the Selection merge and loop_done_lbl as the
      // Loop merge; loop_done_lbl branches into deact_done_lbl. Gating
      // moved from `#if TI_VULKAN_POINTER_FREELIST` to runtime
      // `contract.has_freelist` (B-2.a).
      auto zero_v = ir_->uint_immediate_number(u32_t, 0);
      auto busy_v = ir_->uint_immediate_number(u32_t, 0xFFFFFFFFu);
      auto fhead_word_idx = ir_->uint_immediate_number(
          u32_t, contract.freelist_head_offset / 4u);
      auto fhead_ptr =
          ir_->struct_array_access(u32_t, pool_meta_buffer, fhead_word_idx);
      auto flinks_word_base = ir_->uint_immediate_number(
          u32_t, contract.freelist_links_offset / 4u);

      // Atomically claim and clear the slot.
      auto old_slot = ir_->make_value(
          spv::OpAtomicExchange, u32_t, slot_ptr,
          /*scope=*/ir_->const_i32_one_,
          /*semantics=*/ir_->const_i32_zero_, zero_v);
      auto re_zero = ir_->make_value(spv::OpIEqual, ir_->bool_type(),
                                     old_slot, zero_v);
      auto re_busy = ir_->make_value(spv::OpIEqual, ir_->bool_type(),
                                     old_slot, busy_v);
      auto re_skip = ir_->make_value(spv::OpLogicalOr, ir_->bool_type(),
                                     re_zero, re_busy);

      spirv::Label push_loop_head = ir_->new_label();
      spirv::Label push_loop_body = ir_->new_label();
      spirv::Label push_loop_continue = ir_->new_label();
      spirv::Label loop_done_lbl = ir_->new_label();
      spirv::Label deact_done_lbl = ir_->new_label();

      ir_->make_inst(spv::OpSelectionMerge, deact_done_lbl,
                     spv::SelectionControlMaskNone);
      ir_->make_inst(spv::OpBranchConditional, re_skip, deact_done_lbl,
                     push_loop_head);

      // push loop: CAS free_head from head to old_slot until success.
      ir_->start_label(push_loop_head);
      auto head_cur =
          ir_->make_value(spv::OpAtomicLoad, u32_t, fhead_ptr,
                          /*scope=*/ir_->const_i32_one_,
                          /*semantics=*/ir_->const_i32_zero_);
      ir_->make_inst(spv::OpLoopMerge, loop_done_lbl, push_loop_continue,
                     spv::LoopControlMaskNone);
      ir_->make_inst(spv::OpBranch, push_loop_body);

      ir_->start_label(push_loop_body);
      auto old_node =
          ir_->sub(old_slot, ir_->uint_immediate_number(u32_t, 1));
      auto link_word_idx = ir_->add(flinks_word_base, old_node);
      auto link_ptr =
          ir_->struct_array_access(u32_t, pool_meta_buffer, link_word_idx);
      // Stage links[old_node] = head before the CAS publishes the new head.
      ir_->make_inst(spv::OpAtomicStore, link_ptr,
                     /*scope=*/ir_->const_i32_one_,
                     /*semantics=*/ir_->const_i32_zero_, head_cur);
      auto head_cas = ir_->make_value(
          spv::OpAtomicCompareExchange, u32_t, fhead_ptr,
          /*scope=*/ir_->const_i32_one_,
          /*semantics_eq=*/ir_->const_i32_zero_,
          /*semantics_uneq=*/ir_->const_i32_zero_, old_slot, head_cur);
      auto pushed = ir_->make_value(spv::OpIEqual, ir_->bool_type(), head_cas,
                                    head_cur);
      ir_->make_inst(spv::OpBranchConditional, pushed, loop_done_lbl,
                     push_loop_continue);

      ir_->start_label(push_loop_continue);
      ir_->make_inst(spv::OpBranch, push_loop_head);

      ir_->start_label(loop_done_lbl);
      ir_->make_inst(spv::OpBranch, deact_done_lbl);

      ir_->start_label(deact_done_lbl);
    } else {
      ir_->store_variable(slot_ptr, ir_->uint_immediate_number(u32_t, 0));
    }
  }

  void visit(SNodeOpStmt *stmt) override {
    const int root_id = snode_to_root_.at(stmt->snode->id);
    std::string parent = stmt->ptr->raw_name();
    spirv::Value parent_val = ir_->query_value(parent);

    if (stmt->snode->type == SNodeType::bitmasked) {
      spirv::Value input_index_val =
          ir_->cast(parent_val.stype, ir_->query_value(stmt->val->raw_name()));

      if (stmt->op_type == SNodeOpType::is_active) {
        auto is_active =
            bitmasked_activation(ActivationOp::query, parent_val, root_id,
                                 stmt->snode, input_index_val);
        is_active =
            ir_->cast(ir_->get_primitive_type(stmt->ret_type), is_active);
        is_active = ir_->make_value(spv::OpSNegate, is_active.stype, is_active);
        ir_->register_value(stmt->raw_name(), is_active);
      } else if (stmt->op_type == SNodeOpType::deactivate) {
        bitmasked_activation(ActivationOp::deactivate, parent_val, root_id,
                             stmt->snode, input_index_val);
      } else if (stmt->op_type == SNodeOpType::activate) {
        bitmasked_activation(ActivationOp::activate, parent_val, root_id,
                             stmt->snode, input_index_val);
      } else {
        TI_NOT_IMPLEMENTED;
      }
    } else if (stmt->snode->type == SNodeType::pointer) {
      // Phase 2b: pointer SNode ops on Vulkan. Pool-backed bump allocator.
      spirv::Value input_index_val = ir_->query_value(stmt->val->raw_name());

      if (stmt->op_type == SNodeOpType::is_active) {
        auto is_active = pointer_is_active(parent_val, root_id, stmt->snode,
                                           input_index_val);
        is_active =
            ir_->cast(ir_->get_primitive_type(stmt->ret_type), is_active);
        is_active = ir_->make_value(spv::OpSNegate, is_active.stype, is_active);
        ir_->register_value(stmt->raw_name(), is_active);
      } else if (stmt->op_type == SNodeOpType::deactivate) {
        pointer_deactivate(parent_val, root_id, stmt->snode, input_index_val);
      } else if (stmt->op_type == SNodeOpType::activate) {
        // Discard the address result; activate is a side-effecting op.
        (void)pointer_lookup_or_activate(parent_val, root_id, stmt->snode,
                                         input_index_val,
                                         /*do_activate=*/true);
      } else {
        TI_NOT_IMPLEMENTED;
      }
    } else if (stmt->snode->type == SNodeType::dynamic) {
#if defined(TI_VULKAN_DYNAMIC)
      // G4: dynamic SNode on Vulkan via flat layout + length suffix.
      //   container = [data: cell_stride * N][length u32]
      // length is zero-initialized by root buffer memset. activate uses
      // OpAtomicUMax(length, idx+1); allocate uses OpAtomicIAdd(length,1);
      // is_active reads length and compares idx < length; deactivate stores
      // 0 to length. The cell address for a given index i is
      //   parent_byte_offset + i * cell_stride
      // matching the flat container layout (no chunk indirection).
      const auto &snode_descs = compiled_structs_[root_id].snode_descriptors;
      const auto &desc = snode_descs.at(stmt->snode->id);
      auto u32_t = ir_->u32_type();
      // B-3.c-2: dynamic 容器（含 length 后缀）位于 parent's container buffer。
      auto root_buffer = container_buffer_value(stmt->snode, root_id);
      auto len_byte_off = ir_->add(
          parent_val,
          ir_->uint_immediate_number(
              u32_t, (uint32_t)desc.dynamic_length_offset_in_container));
      auto len_word_idx = ir_->make_value(
          spv::OpShiftRightLogical, u32_t, len_byte_off,
          ir_->uint_immediate_number(u32_t, 2));
      auto len_ptr =
          ir_->struct_array_access(u32_t, root_buffer, len_word_idx);

      if (stmt->op_type == SNodeOpType::length) {
        auto len_val = ir_->make_value(
            spv::OpAtomicLoad, u32_t, len_ptr,
            /*scope=*/ir_->const_i32_one_,
            /*semantics=*/ir_->const_i32_zero_);
        auto ret =
            ir_->cast(ir_->get_primitive_type(stmt->ret_type), len_val);
        ir_->register_value(stmt->raw_name(), ret);
      } else if (stmt->op_type == SNodeOpType::deactivate) {
        ir_->make_inst(spv::OpAtomicStore, len_ptr,
                       /*scope=*/ir_->const_i32_one_,
                       /*semantics=*/ir_->const_i32_zero_,
                       ir_->uint_immediate_number(u32_t, 0));
      } else if (stmt->op_type == SNodeOpType::is_active) {
        auto idx =
            ir_->cast(u32_t, ir_->query_value(stmt->val->raw_name()));
        auto len_val = ir_->make_value(
            spv::OpAtomicLoad, u32_t, len_ptr,
            /*scope=*/ir_->const_i32_one_,
            /*semantics=*/ir_->const_i32_zero_);
        auto active = ir_->make_value(spv::OpULessThan, ir_->bool_type(),
                                      idx, len_val);
        auto active_int =
            ir_->cast(ir_->get_primitive_type(stmt->ret_type), active);
        active_int = ir_->make_value(spv::OpSNegate, active_int.stype,
                                     active_int);
        ir_->register_value(stmt->raw_name(), active_int);
      } else if (stmt->op_type == SNodeOpType::activate) {
        auto idx =
            ir_->cast(u32_t, ir_->query_value(stmt->val->raw_name()));
        auto idx_plus_one =
            ir_->add(idx, ir_->uint_immediate_number(u32_t, 1));
        (void)ir_->make_value(spv::OpAtomicUMax, u32_t, len_ptr,
                              /*scope=*/ir_->const_i32_one_,
                              /*semantics=*/ir_->const_i32_zero_,
                              idx_plus_one);
      } else if (stmt->op_type == SNodeOpType::allocate) {
        // ti.append: i = atomicAdd(length, 1); store i to alloca; the
        // resulting cell address is parent_val + i * cell_stride.
        auto idx = ir_->make_value(
            spv::OpAtomicIAdd, u32_t, len_ptr,
            /*scope=*/ir_->const_i32_one_,
            /*semantics=*/ir_->const_i32_zero_,
            ir_->uint_immediate_number(u32_t, 1));
        auto alloca_ptr = ir_->query_value(stmt->val->raw_name());
        auto idx_i32 = ir_->cast(ir_->i32_type(), idx);
        ir_->store_variable(alloca_ptr, idx_i32);
        auto cell_stride = ir_->uint_immediate_number(
            u32_t, (uint32_t)desc.cell_stride);
        auto cell_off = ir_->mul(idx, cell_stride);
        auto cell_addr = ir_->add(parent_val, cell_off);
        ir_->register_value(stmt->raw_name(), cell_addr);
      } else {
        TI_NOT_IMPLEMENTED;
      }
#else
      TI_NOT_IMPLEMENTED;
#endif
    } else {
      TI_NOT_IMPLEMENTED;
    }
  }

  void visit(SNodeLookupStmt *stmt) override {
    // TODO: SNodeLookupStmt -> GetSNodeCellStmt ?
    bool is_root{false};  // Eliminate first root snode access
    const int root_id = snode_to_root_.at(stmt->snode->id);
    std::string parent;

    if (stmt->input_snode) {
      parent = stmt->input_snode->raw_name();
    } else {
      TI_ASSERT(root_stmts_.at(root_id) != nullptr);
      parent = root_stmts_.at(root_id)->raw_name();
    }
    const auto *sn = stmt->snode;

    spirv::Value parent_val = ir_->query_value(parent);

    if (stmt->activate) {
      if (sn->type == SNodeType::dense) {
        // Do nothing
      } else if (sn->type == SNodeType::bitmasked) {
        spirv::Value input_index_val =
            ir_->query_value(stmt->input_index->raw_name());
        bitmasked_activation(ActivationOp::activate, parent_val, root_id, sn,
                             input_index_val);
      } else if (sn->type == SNodeType::pointer) {
        // Pointer activation is folded into the lookup below: when activate
        // is requested we run pointer_lookup_or_activate(do_activate=true)
        // which both allocates (if needed) and produces the cell address.
      } else {
        TI_NOT_IMPLEMENTED;
      }
    }

    spirv::Value val;
    if (is_root) {
      val = parent_val;  // Assert Root[0] access at first time
    } else if (sn->type == SNodeType::pointer) {
      // Phase 2b: pointer indirection -> pool cell byte offset.
      spirv::Value input_index_val =
          ir_->query_value(stmt->input_index->raw_name());
      val = pointer_lookup_or_activate(parent_val, root_id, sn, input_index_val,
                                       /*do_activate=*/stmt->activate);
    } else if (sn->type == SNodeType::quant_array) {
      // G9.2 (2026-04-30): all user cells in a quant_array share ONE
      // physical word inside the parent cell, so the byte address does
      // NOT advance with input_index.  Instead, we record an SSA bit
      // offset of `element_num_bits * input_index` on the side table;
      // GlobalLoadStmt / GlobalStoreStmt below will combine it with the
      // physical word load.  Mirrors codegen_llvm.cpp::create_bit_ptr.
      auto *qat = sn->dt->as<QuantArrayType>();
      const int element_num_bits = qat->get_element_num_bits();
      spirv::Value input_index_val = ir_->cast(
          ir_->u32_type(), ir_->query_value(stmt->input_index->raw_name()));
      spirv::Value bit_off = ir_->mul(
          input_index_val,
          ir_->uint_immediate_number(ir_->u32_type(),
                                     static_cast<uint32_t>(element_num_bits)));
      val = parent_val;  // byte address unchanged
      bit_ptr_bit_offsets_[stmt] = bit_off;
    } else if (sn->type == SNodeType::bit_struct) {
      // G9.3b (2026-05-01): bit_struct has no axes -- the lookup just
      // forwards the parent's byte address (the dense->bit_struct
      // GetChStmt result).  Mirrors codegen_llvm.cpp where
      // `llvm_val[stmt] = parent`.  The per-member bit offset is added
      // later by GetChStmt(bit_struct -> place).
      val = parent_val;
      // Forward the parent's buffer mapping so at_buffer() works on the
      // BitStructStoreStmt's `ptr` (which may be this lookup stmt).
      auto it = ptr_to_buffers_.find(stmt->input_snode);
      if (it != ptr_to_buffers_.end()) {
        ptr_to_buffers_[stmt] = it->second;
      }
    } else {
      const auto &snode_descs = compiled_structs_[root_id].snode_descriptors;
      const auto &desc = snode_descs.at(sn->id);

      spirv::Value input_index_val = ir_->cast(
          parent_val.stype, ir_->query_value(stmt->input_index->raw_name()));
      spirv::Value stride = make_pointer(desc.cell_stride);
      spirv::Value offset = ir_->mul(input_index_val, stride);
      val = ir_->add(parent_val, offset);
    }
    ir_->register_value(stmt->raw_name(), val);
  }

  void visit(RandStmt *stmt) override {
    spirv::Value val;
    spirv::Value global_tmp =
        get_buffer_value(BufferType::GlobalTmps, PrimitiveType::u32);
    if (stmt->element_type()->is_primitive(PrimitiveTypeID::i32)) {
      val = ir_->rand_i32(global_tmp);
    } else if (stmt->element_type()->is_primitive(PrimitiveTypeID::u32)) {
      val = ir_->rand_u32(global_tmp);
    } else if (stmt->element_type()->is_primitive(PrimitiveTypeID::f32)) {
      val = ir_->rand_f32(global_tmp);
    } else if (stmt->element_type()->is_primitive(PrimitiveTypeID::f16)) {
      auto highp_val = ir_->rand_f32(global_tmp);
      val = ir_->cast(ir_->f16_type(), highp_val);
    } else {
      TI_ERROR("rand only support 32-bit type");
    }
    ir_->register_value(stmt->raw_name(), val);
  }

  void visit(LinearizeStmt *stmt) override {
    spirv::Value val = ir_->const_i32_zero_;
    for (size_t i = 0; i < stmt->inputs.size(); ++i) {
      spirv::Value strides_val =
          ir_->int_immediate_number(ir_->i32_type(), stmt->strides[i]);
      spirv::Value input_val = ir_->query_value(stmt->inputs[i]->raw_name());
      val = ir_->add(ir_->mul(val, strides_val), input_val);
    }
    ir_->register_value(stmt->raw_name(), val);
  }

  void visit(LoopIndexStmt *stmt) override {
    const auto stmt_name = stmt->raw_name();
    if (stmt->loop->is<OffloadedStmt>()) {
      const auto type = stmt->loop->as<OffloadedStmt>()->task_type;
      if (type == OffloadedTaskType::range_for) {
        TI_ASSERT(stmt->index == 0);
        spirv::Value loop_var = ir_->query_value("ii");
        // spirv::Value val = ir_->add(loop_var, ir_->const_i32_zero_);
        ir_->register_value(stmt_name, loop_var);
      } else if (type == OffloadedTaskType::struct_for) {
        // Phase 1b/1c: depth-1 bitmasked at root. The listgen kernel emits
        // flat cell indices into "ii" (u32). For 1D bitmasked the flat index
        // *is* the per-axis coordinate; for multi-axis bitmasked we decode
        // using the SNode extractors:
        //   axis_value = (flat / extractors[index].acc_shape)
        //                % extractors[index].shape
        // The result is cast to i32 to match LoopIndexStmt's return type so
        // downstream arithmetic on the loop index stays well-typed.
        spirv::Value flat = ir_->query_value("ii");
        const auto *loop_sn = stmt->loop->as<OffloadedStmt>()->snode;
        TI_ASSERT(loop_sn != nullptr);
        const int axis = stmt->index;
        // Phase 1d-A: the listgen emits a flat index relative to the entire
        // path from root down to the loop SNode, so the per-axis shape used
        // in decoding must be the cumulative `num_elements_from_root` rather
        // than the leaf SNode's local extractor.shape. For depth-1 layouts
        // (Phase 1b/1c) the two values are identical, so this remains
        // backward compatible.
        const int shape =
            (int)loop_sn->extractors[axis].num_elements_from_root;
        int acc_shape = 1;
        for (int j = taichi_max_num_indices - 1; j > axis; --j) {
          int64_t s = loop_sn->extractors[j].num_elements_from_root;
          if (s > 1)
            acc_shape *= (int)s;
        }
        spirv::Value val = flat;
        if (acc_shape > 1) {
          val = ir_->make_value(
              spv::OpUDiv, ir_->u32_type(), val,
              ir_->uint_immediate_number(ir_->u32_type(), acc_shape));
        }
        if (shape > 0 && shape != 1) {
          val = ir_->make_value(
              spv::OpUMod, ir_->u32_type(), val,
              ir_->uint_immediate_number(ir_->u32_type(), shape));
        }
        // Bring back to i32 to match the IR-level type of LoopIndexStmt.
        val = ir_->cast(ir_->i32_type(), val);
        ir_->register_value(stmt_name, val);
      } else {
        TI_NOT_IMPLEMENTED;
      }
    } else if (stmt->loop->is<RangeForStmt>()) {
      TI_ASSERT(stmt->index == 0);
      spirv::Value loop_var = ir_->query_value(stmt->loop->raw_name());
      spirv::Value val = ir_->add(loop_var, ir_->const_i32_zero_);
      ir_->register_value(stmt_name, val);
    } else {
      TI_NOT_IMPLEMENTED;
    }
  }

  void visit(GlobalStoreStmt *stmt) override {
    // G9.3a (2026-04-30): bit-pointer write path for quant_array.
    // For a `place` whose parent is a quant_array, the IR pass
    // optimize_bit_struct_stores does NOT rewrite this store (it only
    // fires for bit_struct parents, see
    // taichi/transforms/optimize_bit_struct_stores.cpp:30), so we get a
    // raw GlobalStoreStmt with `is_bit_pointer()` dest.  Implementation
    // is non-atomic read-modify-write of the single physical word.
    if (auto *pt = stmt->dest->ret_type->cast<PointerType>();
        pt && pt->is_bit_pointer()) {
      emit_bit_pointer_store(stmt);
      return;
    }
    spirv::Value val = ir_->query_value(stmt->val->raw_name());

    store_buffer(stmt->dest, val);
  }

  void visit(GlobalLoadStmt *stmt) override {
    // G9.2 (2026-04-30): bit-pointer read path for quant_array.
    // Mirrors codegen_llvm.cpp:1609-1648 (`is_bit_pointer()` branch in
    // GlobalLoadStmt) -- load physical word, shift / mask / sign-extend,
    // optionally cast+scale for QuantFixed.
    if (auto *pt = stmt->src->ret_type->cast<PointerType>();
        pt && pt->is_bit_pointer()) {
      emit_bit_pointer_load(stmt);
      return;
    }
    auto dt = stmt->element_type();

    auto val = load_buffer(stmt->src, dt);

    ir_->register_value(stmt->raw_name(), val);
  }

  void visit(ArgLoadStmt *stmt) override {
    const auto arg_id = stmt->arg_id;
    const std::vector<int> indices_l(stmt->arg_id.begin(),
                                     stmt->arg_id.begin() + stmt->arg_depth);
    const std::vector<int> indices_r(stmt->arg_id.begin() + stmt->arg_depth,
                                     stmt->arg_id.end());
    const auto arg_type =
        stmt->arg_depth == 0
            ? ctx_attribs_->args_type()->get_element_type(arg_id)
            : ctx_attribs_->argpack_type(indices_l)
                  ->as<lang::StructType>()
                  ->get_element_type(indices_r);
    if (arg_type->is<PointerType>() ||
        (arg_type->is<lang::StructType>() &&
         arg_type->as<lang::StructType>()->elements().size() >= 2 &&
         arg_type->as<lang::StructType>()
             ->get_element_type({1})
             ->is<PointerType>())) {
      // Do not shift! We are indexing the buffers at byte granularity.
      // spirv::Value val =
      //    ir_->int_immediate_number(ir_->i32_type(), offset_in_mem);
      // ir_->register_value(stmt->raw_name(), val);
    } else {
      spirv::Value buffer_val, buffer_value;
      bool is_bool = arg_type->is_primitive(PrimitiveTypeID::u1);
      // `val_type` must be assigned after `get_buffer_value` because
      // `args_struct_types_` needs to be initialized by `get_buffer_value`.
      SType val_type;
      if (stmt->arg_depth > 0) {
        // Inside argpacks, load value from argpack buffer
        buffer_value = get_buffer_value({BufferType::ArgPack, indices_l},
                                        PrimitiveType::i32);
        val_type = is_bool ? ir_->i32_type()
                           : argpack_struct_types_[indices_l][indices_r];
        buffer_val = ir_->make_access_chain(
            ir_->get_pointer_type(val_type, spv::StorageClassUniform),
            buffer_value, indices_r);
      } else {
        // Not in argpacks, load value from args buffer
        buffer_value = get_buffer_value(BufferType::Args, PrimitiveType::i32);
        val_type = is_bool ? ir_->i32_type() : args_struct_types_[arg_id];
        buffer_val = ir_->make_access_chain(
            ir_->get_pointer_type(val_type, spv::StorageClassUniform),
            buffer_value, arg_id);
      }
      buffer_val.flag = ValueKind::kVariablePtr;
      if (!stmt->create_load) {
        ir_->register_value(stmt->raw_name(), buffer_val);
        return;
      }
      spirv::Value val = ir_->load_variable(buffer_val, val_type);
      if (is_bool) {
        val = ir_->make_value(spv::OpINotEqual, ir_->bool_type(), val,
                              ir_->int_immediate_number(ir_->i32_type(), 0));
      }
      ir_->register_value(stmt->raw_name(), val);
    }
  }

  void visit(GetElementStmt *stmt) override {
    spirv::Value val = ir_->query_value(stmt->src->raw_name());
    const auto val_type = ir_->get_primitive_type(stmt->element_type());
    const auto val_type_ptr =
        ir_->get_pointer_type(val_type, spv::StorageClassUniform);
    val = ir_->make_access_chain(val_type_ptr, val, stmt->index);
    val = ir_->load_variable(val, val_type);
    ir_->register_value(stmt->raw_name(), val);
  }

  void visit(ReturnStmt *stmt) override {
    TI_ASSERT(ctx_attribs_->has_rets());
    // The `PrimitiveType::i32` in this function call is a placeholder.
    auto buffer_value = get_buffer_value(BufferType::Rets, PrimitiveType::i32);
    // Function to store variable using indices provided by
    // `calc_indices_and_store`.
    auto store_variable = [&](int index, const std::vector<int> &indices) {
      auto dt = stmt->element_types()[index];
      auto val_type = ir_->get_primitive_type(dt);
      // Extend u1 values to i32 to be passed to the host.
      if (dt->is_primitive(PrimitiveTypeID::u1))
        val_type = ir_->i32_type();
      spirv::Value buffer_val;
      // Accessing based on `indices` using OpAccessChain.
      buffer_val = ir_->make_access_chain(
          ir_->get_storage_pointer_type(val_type), buffer_value, indices);
      buffer_val.flag = ValueKind::kVariablePtr;
      spirv::Value val = ir_->query_value(stmt->values[index]->raw_name());
      // Extend u1 values to i32 to be passed to the host.
      if (dt->is_primitive(PrimitiveTypeID::u1))
        val = ir_->select(val, ir_->const_i32_one_, ir_->const_i32_zero_);
      ir_->store_variable(buffer_val, val);
    };
    // Function to traverse struct tree in depth-first order recursively to
    // calculate AccessChain indices.
    std::function<void(const taichi::lang::Type *, int &, std::vector<int> &)>
        calc_indices_and_store = [&](const taichi::lang::Type *type, int &index,
                                     std::vector<int> &indices) {
          if (auto struct_type = type->cast<taichi::lang::StructType>()) {
            for (int i = 0; i < struct_type->elements().size(); ++i) {
              indices.push_back(i);
              calc_indices_and_store(struct_type->elements()[i].type, index,
                                     indices);
              indices.pop_back();
            }
          } else if (auto tensor_type =
                         type->cast<taichi::lang::TensorType>()) {
            int num = tensor_type->get_num_elements();
            for (int i = 0; i < num; ++i) {
              indices.push_back(i);
              store_variable(index++, indices);
              indices.pop_back();
            }
          } else {
            store_variable(index++, indices);
          }
        };
    // Launch depth-first traversal using `calc_indices_and_store` on return
    // struct.
    std::vector<int> indices;
    int index = 0;
    for (int i = 0; i < ctx_attribs_->rets_type()->elements().size(); ++i) {
      indices.push_back(i);
      calc_indices_and_store(ctx_attribs_->rets_type()->elements()[i].type,
                             index, indices);
      indices.pop_back();
    }
  }

  void visit(GlobalTemporaryStmt *stmt) override {
    spirv::Value val = ir_->int_immediate_number(ir_->i32_type(), stmt->offset,
                                                 false);  // Named Constant
    ir_->register_value(stmt->raw_name(), val);
    ptr_to_buffers_[stmt] = BufferType::GlobalTmps;
  }

  void visit(ExternalTensorShapeAlongAxisStmt *stmt) override {
    const auto name = stmt->raw_name();
    const auto arg_id = stmt->arg_id;
    const auto axis = stmt->axis;

    spirv::Value var_ptr;
    TI_ASSERT(ctx_attribs_->args_type()
                  ->get_element_type({arg_id})
                  ->is<lang::StructType>());
    std::vector<int> indices = arg_id;
    indices.push_back(TypeFactory::SHAPE_POS_IN_NDARRAY);
    indices.push_back(axis);
    var_ptr = ir_->make_access_chain(
        ir_->get_pointer_type(ir_->i32_type(), spv::StorageClassUniform),
        get_buffer_value(BufferType::Args, PrimitiveType::i32), indices);
    spirv::Value var = ir_->load_variable(var_ptr, ir_->i32_type());

    ir_->register_value(name, var);
  }

  void visit(ExternalPtrStmt *stmt) override {
    // Used mostly for transferring data between host (e.g. numpy array) and
    // device.
    spirv::Value linear_offset = ir_->int_immediate_number(ir_->i32_type(), 0);
    const auto *argload = stmt->base_ptr->as<ArgLoadStmt>();
    const auto arg_id = argload->arg_id;
    {
      const int num_indices = stmt->indices.size();
      std::vector<std::string> size_var_names;
      const auto &element_shape = stmt->element_shape;
      const size_t element_shape_index_offset =
          num_indices - element_shape.size();
      for (int i = 0; i < num_indices - element_shape.size(); i++) {
        std::string var_name = fmt::format("{}_size{}_", stmt->raw_name(), i);
        std::vector<int> indices = arg_id;
        indices.push_back(TypeFactory::SHAPE_POS_IN_NDARRAY);
        indices.push_back(i);
        spirv::Value var_ptr = ir_->make_access_chain(
            ir_->get_pointer_type(ir_->i32_type(), spv::StorageClassUniform),
            get_buffer_value(BufferType::Args, PrimitiveType::i32), indices);
        spirv::Value var = ir_->load_variable(var_ptr, ir_->i32_type());
        ir_->register_value(var_name, var);
        size_var_names.push_back(std::move(var_name));
      }
      int size_var_names_idx = 0;
      for (int i = 0; i < num_indices; i++) {
        spirv::Value size_var;
        // Use immediate numbers to flatten index for element shapes.
        if (i >= element_shape_index_offset &&
            i < element_shape_index_offset + element_shape.size()) {
          size_var = ir_->uint_immediate_number(
              ir_->i32_type(), element_shape[i - element_shape_index_offset]);
        } else {
          size_var = ir_->query_value(size_var_names[size_var_names_idx++]);
        }
        spirv::Value indices = ir_->query_value(stmt->indices[i]->raw_name());
        linear_offset = ir_->mul(linear_offset, size_var);
        linear_offset = ir_->add(linear_offset, indices);
      }
      linear_offset = ir_->make_value(
          spv::OpShiftLeftLogical, ir_->i32_type(), linear_offset,
          ir_->int_immediate_number(ir_->i32_type(),
                                    log2int(ir_->get_primitive_type_size(
                                        stmt->ret_type.ptr_removed()))));
      if (caps_->get(DeviceCapability::spirv_has_no_integer_wrap_decoration)) {
        ir_->decorate(spv::OpDecorate, linear_offset,
                      spv::DecorationNoSignedWrap);
      }
    }
    if (caps_->get(DeviceCapability::spirv_has_physical_storage_buffer)) {
      std::vector<int> indices = arg_id;
      indices.push_back(1);
      spirv::Value addr_ptr = ir_->make_access_chain(
          ir_->get_pointer_type(ir_->u64_type(), spv::StorageClassUniform),
          get_buffer_value(BufferType::Args, PrimitiveType::i32), indices);
      spirv::Value addr = ir_->load_variable(addr_ptr, ir_->u64_type());
      addr = ir_->add(addr, ir_->make_value(spv::OpSConvert, ir_->u64_type(),
                                            linear_offset));
      ir_->register_value(stmt->raw_name(), addr);
    } else {
      ir_->register_value(stmt->raw_name(), linear_offset);
    }

    if (ctx_attribs_->arg_at(arg_id).is_array) {
      ptr_to_buffers_[stmt] = {BufferType::ExtArr, arg_id};
    } else {
      ptr_to_buffers_[stmt] = BufferType::Args;
    }
  }

  void visit(DecorationStmt *stmt) override {
  }

  void visit(UnaryOpStmt *stmt) override {
    const auto operand_name = stmt->operand->raw_name();

    const auto src_dt = stmt->operand->element_type();
    const auto dst_dt = stmt->element_type();
    spirv::SType src_type = ir_->get_primitive_type(src_dt);
    spirv::SType dst_type;
    if (dst_dt.is_pointer()) {
      auto stype = dst_dt.ptr_removed()->as<lang::StructType>();
      std::vector<std::tuple<SType, std::string, size_t>> components;
      for (int i = 0; i < stype->elements().size(); i++) {
        components.push_back(
            {ir_->get_primitive_type(stype->get_element_type({i})),
             fmt::format("element{}", i), stype->get_element_offset({i})});
      }
      dst_type = ir_->create_struct_type(components);
    } else {
      dst_type = ir_->get_primitive_type(dst_dt);
    }
    spirv::Value operand_val = ir_->query_value(operand_name);
    spirv::Value val = spirv::Value();

    if (stmt->op_type == UnaryOpType::logic_not) {
      spirv::Value zero =
          ir_->get_zero(src_type);  // Math zero type to left hand side
      if (is_integral(src_dt)) {
        if (is_signed(src_dt)) {
          zero = ir_->int_immediate_number(src_type, 0);
        } else {
          zero = ir_->uint_immediate_number(src_type, 0);
        }
      } else if (is_real(src_dt)) {
        zero = ir_->float_immediate_number(src_type, 0);
      } else {
        TI_NOT_IMPLEMENTED
      }
      val = ir_->cast(dst_type, ir_->eq(operand_val, zero));
    } else if (stmt->op_type == UnaryOpType::neg) {
      operand_val = ir_->cast(dst_type, operand_val);
      if (is_integral(dst_dt)) {
        if (is_signed(dst_dt)) {
          val = ir_->make_value(spv::OpSNegate, dst_type, operand_val);
        } else {
          TI_NOT_IMPLEMENTED
        }
      } else if (is_real(dst_dt)) {
        val = ir_->make_value(spv::OpFNegate, dst_type, operand_val);
      } else {
        TI_NOT_IMPLEMENTED
      }
    } else if (stmt->op_type == UnaryOpType::rsqrt) {
      const uint32_t InverseSqrt_id = 32;
      if (is_real(src_dt)) {
        val = ir_->call_glsl450(src_type, InverseSqrt_id, operand_val);
        val = ir_->cast(dst_type, val);
      } else {
        TI_NOT_IMPLEMENTED
      }
    } else if (stmt->op_type == UnaryOpType::sgn) {
      const uint32_t FSign_id = 6;
      const uint32_t SSign_id = 7;
      if (is_integral(src_dt)) {
        if (is_signed(src_dt)) {
          val = ir_->call_glsl450(src_type, SSign_id, operand_val);
        } else {
          TI_NOT_IMPLEMENTED
        }
      } else if (is_real(src_dt)) {
        val = ir_->call_glsl450(src_type, FSign_id, operand_val);
      } else {
        TI_NOT_IMPLEMENTED
      }
      val = ir_->cast(dst_type, val);
    } else if (stmt->op_type == UnaryOpType::bit_not) {
      operand_val = ir_->cast(dst_type, operand_val);
      if (is_integral(dst_dt)) {
        val = ir_->make_value(spv::OpNot, dst_type, operand_val);
      } else {
        TI_NOT_IMPLEMENTED
      }
    } else if (stmt->op_type == UnaryOpType::cast_value) {
      val = ir_->cast(dst_type, operand_val);
    } else if (stmt->op_type == UnaryOpType::cast_bits) {
      if (data_type_bits(src_dt) == data_type_bits(dst_dt)) {
        val = ir_->make_value(spv::OpBitcast, dst_type, operand_val);
      } else {
        TI_ERROR("bit_cast is only supported between data type with same size");
      }
    } else if (stmt->op_type == UnaryOpType::abs) {
      const uint32_t FAbs_id = 4;
      const uint32_t SAbs_id = 5;
      if (src_type.id == dst_type.id) {
        if (is_integral(src_dt)) {
          if (is_signed(src_dt)) {
            val = ir_->call_glsl450(src_type, SAbs_id, operand_val);
          } else {
            TI_NOT_IMPLEMENTED
          }
        } else if (is_real(src_dt)) {
          val = ir_->call_glsl450(src_type, FAbs_id, operand_val);
        } else {
          TI_NOT_IMPLEMENTED
        }
      } else {
        TI_NOT_IMPLEMENTED
      }
    } else if (stmt->op_type == UnaryOpType::inv) {
      if (is_real(dst_dt)) {
        val = ir_->div(ir_->float_immediate_number(dst_type, 1), operand_val);
      } else {
        TI_NOT_IMPLEMENTED
      }
    } else if (stmt->op_type == UnaryOpType::frexp) {
      // FrexpStruct is the same type of the first member.
      val = ir_->alloca_variable(dst_type);
      auto v = ir_->call_glsl450(dst_type, 52, operand_val);
      ir_->store_variable(val, v);
    } else if (stmt->op_type == UnaryOpType::popcnt) {
      val = ir_->popcnt(operand_val);
    } else if (stmt->op_type == UnaryOpType::clz) {
      uint32_t FindMSB_id = 74;
      spirv::Value msb = ir_->call_glsl450(dst_type, FindMSB_id, operand_val);
      spirv::Value bitcnt = ir_->int_immediate_number(ir_->i32_type(), 32);
      spirv::Value one = ir_->int_immediate_number(ir_->i32_type(), 1);
      val = ir_->sub(ir_->sub(bitcnt, msb), one);
    }
#define UNARY_OP_TO_SPIRV(op, instruction, instruction_id, max_bits)           \
  else if (stmt->op_type == UnaryOpType::op) {                                 \
    const uint32_t instruction = instruction_id;                               \
    if (is_real(src_dt)) {                                                     \
      if (data_type_bits(src_dt) > max_bits) {                                 \
        TI_ERROR("Instruction {}({}) does not {}bits operation", #instruction, \
                 instruction_id, data_type_bits(src_dt));                      \
      }                                                                        \
      val = ir_->call_glsl450(src_type, instruction, operand_val);             \
    } else {                                                                   \
      TI_NOT_IMPLEMENTED                                                       \
    }                                                                          \
  }
    UNARY_OP_TO_SPIRV(round, Round, 1, 64)
    UNARY_OP_TO_SPIRV(floor, Floor, 8, 64)
    UNARY_OP_TO_SPIRV(ceil, Ceil, 9, 64)
    UNARY_OP_TO_SPIRV(sin, Sin, 13, 32)
    UNARY_OP_TO_SPIRV(asin, Asin, 16, 32)
    UNARY_OP_TO_SPIRV(cos, Cos, 14, 32)
    UNARY_OP_TO_SPIRV(acos, Acos, 17, 32)
    UNARY_OP_TO_SPIRV(tan, Tan, 15, 32)
    UNARY_OP_TO_SPIRV(tanh, Tanh, 21, 32)
    UNARY_OP_TO_SPIRV(exp, Exp, 27, 32)
    UNARY_OP_TO_SPIRV(log, Log, 28, 32)
    UNARY_OP_TO_SPIRV(sqrt, Sqrt, 31, 64)
#undef UNARY_OP_TO_SPIRV
    else {TI_NOT_IMPLEMENTED} ir_->register_value(stmt->raw_name(), val);
  }

  void generate_overflow_branch(const spirv::Value &cond_v,
                                const std::string &op,
                                const std::string &tb) {
    spirv::Value cond =
        ir_->ne(cond_v, ir_->cast(cond_v.stype, ir_->const_i32_zero_));
    spirv::Label then_label = ir_->new_label();
    spirv::Label merge_label = ir_->new_label();
    ir_->make_inst(spv::OpSelectionMerge, merge_label,
                   spv::SelectionControlMaskNone);
    ir_->make_inst(spv::OpBranchConditional, cond, then_label, merge_label);
    // then block
    ir_->start_label(then_label);
    ir_->call_debugprintf(
        op + " overflow detected in " + sanitize_format_string(tb), {});
    ir_->make_inst(spv::OpBranch, merge_label);
    // merge label
    ir_->start_label(merge_label);
  }

  spirv::Value generate_uadd_overflow(const spirv::Value &a,
                                      const spirv::Value &b,
                                      const std::string &tb) {
    std::vector<std::tuple<spirv::SType, std::string, size_t>>
        struct_components_;
    struct_components_.emplace_back(a.stype, "result", 0);
    struct_components_.emplace_back(a.stype, "carry",
                                    ir_->get_primitive_type_size(a.stype.dt));
    auto struct_type = ir_->create_struct_type(struct_components_);
    auto add_carry = ir_->make_value(spv::OpIAddCarry, struct_type, a, b);
    auto result =
        ir_->make_value(spv::OpCompositeExtract, a.stype, add_carry, 0);
    auto carry =
        ir_->make_value(spv::OpCompositeExtract, a.stype, add_carry, 1);
    generate_overflow_branch(carry, "Addition", tb);
    return result;
  }

  spirv::Value generate_usub_overflow(const spirv::Value &a,
                                      const spirv::Value &b,
                                      const std::string &tb) {
    std::vector<std::tuple<spirv::SType, std::string, size_t>>
        struct_components_;
    struct_components_.emplace_back(a.stype, "result", 0);
    struct_components_.emplace_back(a.stype, "borrow",
                                    ir_->get_primitive_type_size(a.stype.dt));
    auto struct_type = ir_->create_struct_type(struct_components_);
    auto add_carry = ir_->make_value(spv::OpISubBorrow, struct_type, a, b);
    auto result =
        ir_->make_value(spv::OpCompositeExtract, a.stype, add_carry, 0);
    auto borrow =
        ir_->make_value(spv::OpCompositeExtract, a.stype, add_carry, 1);
    generate_overflow_branch(borrow, "Subtraction", tb);
    return result;
  }

  spirv::Value generate_sadd_overflow(const spirv::Value &a,
                                      const spirv::Value &b,
                                      const std::string &tb) {
    // overflow iff (sign(a) == sign(b)) && (sign(a) != sign(result))
    auto result = ir_->make_value(spv::OpIAdd, a.stype, a, b);
    auto zero = ir_->int_immediate_number(a.stype, 0);
    auto a_sign = ir_->make_value(spv::OpSLessThan, ir_->bool_type(), a, zero);
    auto b_sign = ir_->make_value(spv::OpSLessThan, ir_->bool_type(), b, zero);
    auto r_sign =
        ir_->make_value(spv::OpSLessThan, ir_->bool_type(), result, zero);
    auto a_eq_b =
        ir_->make_value(spv::OpLogicalEqual, ir_->bool_type(), a_sign, b_sign);
    auto a_neq_r = ir_->make_value(spv::OpLogicalNotEqual, ir_->bool_type(),
                                   a_sign, r_sign);
    auto overflow =
        ir_->make_value(spv::OpLogicalAnd, ir_->bool_type(), a_eq_b, a_neq_r);
    generate_overflow_branch(overflow, "Addition", tb);
    return result;
  }

  spirv::Value generate_ssub_overflow(const spirv::Value &a,
                                      const spirv::Value &b,
                                      const std::string &tb) {
    // overflow iff (sign(a) != sign(b)) && (sign(a) != sign(result))
    auto result = ir_->make_value(spv::OpISub, a.stype, a, b);
    auto zero = ir_->int_immediate_number(a.stype, 0);
    auto a_sign = ir_->make_value(spv::OpSLessThan, ir_->bool_type(), a, zero);
    auto b_sign = ir_->make_value(spv::OpSLessThan, ir_->bool_type(), b, zero);
    auto r_sign =
        ir_->make_value(spv::OpSLessThan, ir_->bool_type(), result, zero);
    auto a_neq_b = ir_->make_value(spv::OpLogicalNotEqual, ir_->bool_type(),
                                   a_sign, b_sign);
    auto a_neq_r = ir_->make_value(spv::OpLogicalNotEqual, ir_->bool_type(),
                                   a_sign, r_sign);
    auto overflow =
        ir_->make_value(spv::OpLogicalAnd, ir_->bool_type(), a_neq_b, a_neq_r);
    generate_overflow_branch(overflow, "Subtraction", tb);
    return result;
  }

  spirv::Value generate_umul_overflow(const spirv::Value &a,
                                      const spirv::Value &b,
                                      const std::string &tb) {
    // overflow iff high bits != 0
    std::vector<std::tuple<spirv::SType, std::string, size_t>>
        struct_components_;
    struct_components_.emplace_back(a.stype, "low", 0);
    struct_components_.emplace_back(a.stype, "high",
                                    ir_->get_primitive_type_size(a.stype.dt));
    auto struct_type = ir_->create_struct_type(struct_components_);
    auto mul_ext = ir_->make_value(spv::OpUMulExtended, struct_type, a, b);
    auto low = ir_->make_value(spv::OpCompositeExtract, a.stype, mul_ext, 0);
    auto high = ir_->make_value(spv::OpCompositeExtract, a.stype, mul_ext, 1);
    generate_overflow_branch(high, "Multiplication", tb);
    return low;
  }

  spirv::Value generate_smul_overflow(const spirv::Value &a,
                                      const spirv::Value &b,
                                      const std::string &tb) {
    // overflow if high bits are not all sign bit (0 if positive, -1 if
    // negative) or the sign bit of the low bits is not the expected sign bit.
    std::vector<std::tuple<spirv::SType, std::string, size_t>>
        struct_components_;
    struct_components_.emplace_back(a.stype, "low", 0);
    struct_components_.emplace_back(a.stype, "high",
                                    ir_->get_primitive_type_size(a.stype.dt));
    auto struct_type = ir_->create_struct_type(struct_components_);
    auto mul_ext = ir_->make_value(spv::OpSMulExtended, struct_type, a, b);
    auto low = ir_->make_value(spv::OpCompositeExtract, a.stype, mul_ext, 0);
    auto high = ir_->make_value(spv::OpCompositeExtract, a.stype, mul_ext, 1);
    auto zero = ir_->int_immediate_number(a.stype, 0);
    auto minus_one = ir_->int_immediate_number(a.stype, -1);
    auto a_sign = ir_->make_value(spv::OpSLessThan, ir_->bool_type(), a, zero);
    auto b_sign = ir_->make_value(spv::OpSLessThan, ir_->bool_type(), b, zero);
    auto a_not_zero = ir_->ne(a, zero);
    auto b_not_zero = ir_->ne(b, zero);
    auto a_b_not_zero = ir_->make_value(spv::OpLogicalAnd, ir_->bool_type(),
                                        a_not_zero, b_not_zero);
    auto low_sign =
        ir_->make_value(spv::OpSLessThan, ir_->bool_type(), low, zero);
    auto expected_sign = ir_->make_value(spv::OpLogicalNotEqual,
                                         ir_->bool_type(), a_sign, b_sign);
    expected_sign = ir_->make_value(spv::OpLogicalAnd, ir_->bool_type(),
                                    expected_sign, a_b_not_zero);
    auto not_expected_sign = ir_->ne(low_sign, expected_sign);
    auto expected_high = ir_->select(expected_sign, minus_one, zero);
    auto not_expected_high = ir_->ne(high, expected_high);
    auto overflow = ir_->make_value(spv::OpLogicalOr, ir_->bool_type(),
                                    not_expected_high, not_expected_sign);
    generate_overflow_branch(overflow, "Multiplication", tb);
    return low;
  }

  spirv::Value generate_ushl_overflow(const spirv::Value &a,
                                      const spirv::Value &b,
                                      const std::string &tb) {
    // overflow iff a << b >> b != a
    auto result = ir_->make_value(spv::OpShiftLeftLogical, a.stype, a, b);
    auto restore =
        ir_->make_value(spv::OpShiftRightLogical, a.stype, result, b);
    auto overflow = ir_->ne(a, restore);
    generate_overflow_branch(overflow, "Shift left", tb);
    return result;
  }

  spirv::Value generate_sshl_overflow(const spirv::Value &a,
                                      const spirv::Value &b,
                                      const std::string &tb) {
    // overflow iff a << b >> b != a
    auto result = ir_->make_value(spv::OpShiftLeftLogical, a.stype, a, b);
    auto restore =
        ir_->make_value(spv::OpShiftRightArithmetic, a.stype, result, b);
    auto overflow = ir_->ne(a, restore);
    generate_overflow_branch(overflow, "Shift left", tb);
    return result;
  }

  void visit(BinaryOpStmt *bin) override {
    const auto lhs_name = bin->lhs->raw_name();
    const auto rhs_name = bin->rhs->raw_name();
    const auto bin_name = bin->raw_name();
    const auto op_type = bin->op_type;

    spirv::SType dst_type = ir_->get_primitive_type(bin->element_type());
    spirv::Value lhs_value = ir_->query_value(lhs_name);
    spirv::Value rhs_value = ir_->query_value(rhs_name);
    spirv::Value bin_value = spirv::Value();

    TI_WARN_IF(lhs_value.stype.id != rhs_value.stype.id,
               "${} type {} != ${} type {}\n{}", lhs_name,
               lhs_value.stype.dt->to_string(), rhs_name,
               rhs_value.stype.dt->to_string(), bin->get_tb());

    bool debug = caps_->get(DeviceCapability::spirv_has_non_semantic_info);

    if (debug && op_type == BinaryOpType::add && is_integral(dst_type.dt)) {
      if (is_unsigned(dst_type.dt)) {
        bin_value = generate_uadd_overflow(lhs_value, rhs_value, bin->get_tb());
      } else {
        bin_value = generate_sadd_overflow(lhs_value, rhs_value, bin->get_tb());
      }
      bin_value = ir_->cast(dst_type, bin_value);
    } else if (debug && op_type == BinaryOpType::sub &&
               is_integral(dst_type.dt)) {
      if (is_unsigned(dst_type.dt)) {
        bin_value = generate_usub_overflow(lhs_value, rhs_value, bin->get_tb());
      } else {
        bin_value = generate_ssub_overflow(lhs_value, rhs_value, bin->get_tb());
      }
      bin_value = ir_->cast(dst_type, bin_value);
    } else if (debug && op_type == BinaryOpType::mul &&
               is_integral(dst_type.dt)) {
      if (is_unsigned(dst_type.dt)) {
        bin_value = generate_umul_overflow(lhs_value, rhs_value, bin->get_tb());
      } else {
        bin_value = generate_smul_overflow(lhs_value, rhs_value, bin->get_tb());
      }
      bin_value = ir_->cast(dst_type, bin_value);
    }
#define BINARY_OP_TO_SPIRV_ARTHIMATIC(op, func)  \
  else if (op_type == BinaryOpType::op) {        \
    bin_value = ir_->func(lhs_value, rhs_value); \
    bin_value = ir_->cast(dst_type, bin_value);  \
  }

    BINARY_OP_TO_SPIRV_ARTHIMATIC(add, add)
    BINARY_OP_TO_SPIRV_ARTHIMATIC(sub, sub)
    BINARY_OP_TO_SPIRV_ARTHIMATIC(mul, mul)
    BINARY_OP_TO_SPIRV_ARTHIMATIC(div, div)
    BINARY_OP_TO_SPIRV_ARTHIMATIC(mod, mod)
#undef BINARY_OP_TO_SPIRV_ARTHIMATIC

#define BINARY_OP_TO_SPIRV_BITWISE(op, sym)                                \
  else if (op_type == BinaryOpType::op) {                                  \
    bin_value = ir_->make_value(spv::sym, dst_type, lhs_value, rhs_value); \
  }

    else if (debug && op_type == BinaryOpType::bit_shl) {
      if (is_unsigned(dst_type.dt)) {
        bin_value = generate_ushl_overflow(lhs_value, rhs_value, bin->get_tb());
      } else {
        bin_value = generate_sshl_overflow(lhs_value, rhs_value, bin->get_tb());
      }
    }
    BINARY_OP_TO_SPIRV_BITWISE(bit_and, OpBitwiseAnd)
    BINARY_OP_TO_SPIRV_BITWISE(bit_or, OpBitwiseOr)
    BINARY_OP_TO_SPIRV_BITWISE(bit_xor, OpBitwiseXor)
    BINARY_OP_TO_SPIRV_BITWISE(bit_shl, OpShiftLeftLogical)
    // NOTE: `OpShiftRightArithmetic` will treat the first bit as sign bit even
    // it's the unsigned type
    else if (op_type == BinaryOpType::bit_sar) {
      bin_value = ir_->make_value(is_unsigned(dst_type.dt)
                                      ? spv::OpShiftRightLogical
                                      : spv::OpShiftRightArithmetic,
                                  dst_type, lhs_value, rhs_value);
    }
#undef BINARY_OP_TO_SPIRV_BITWISE

#define BINARY_OP_TO_SPIRV_LOGICAL(op, func)     \
  else if (op_type == BinaryOpType::op) {        \
    bin_value = ir_->func(lhs_value, rhs_value); \
    bin_value = ir_->cast(dst_type, bin_value);  \
  }

    BINARY_OP_TO_SPIRV_LOGICAL(cmp_lt, lt)
    BINARY_OP_TO_SPIRV_LOGICAL(cmp_le, le)
    BINARY_OP_TO_SPIRV_LOGICAL(cmp_gt, gt)
    BINARY_OP_TO_SPIRV_LOGICAL(cmp_ge, ge)
    BINARY_OP_TO_SPIRV_LOGICAL(cmp_eq, eq)
    BINARY_OP_TO_SPIRV_LOGICAL(cmp_ne, ne)
    BINARY_OP_TO_SPIRV_LOGICAL(logical_and, logical_and)
    BINARY_OP_TO_SPIRV_LOGICAL(logical_or, logical_or)
#undef BINARY_OP_TO_SPIRV_LOGICAL

#define FLOAT_BINARY_OP_TO_SPIRV_FLOAT_FUNC(op, instruction, instruction_id,   \
                                            max_bits)                          \
  else if (op_type == BinaryOpType::op) {                                      \
    const uint32_t instruction = instruction_id;                               \
    if (is_real(bin->element_type())) {                                        \
      if (data_type_bits(bin->element_type()) > max_bits) {                    \
        TI_ERROR(                                                              \
            "[glsl450] the operand type of instruction {}({}) must <= {}bits", \
            #instruction, instruction_id, max_bits);                           \
      }                                                                        \
      bin_value =                                                              \
          ir_->call_glsl450(dst_type, instruction, lhs_value, rhs_value);      \
    } else {                                                                   \
      TI_NOT_IMPLEMENTED                                                       \
    }                                                                          \
  }

    FLOAT_BINARY_OP_TO_SPIRV_FLOAT_FUNC(atan2, Atan2, 25, 32)
    FLOAT_BINARY_OP_TO_SPIRV_FLOAT_FUNC(pow, Pow, 26, 32)
#undef FLOAT_BINARY_OP_TO_SPIRV_FLOAT_FUNC

#define BINARY_OP_TO_SPIRV_FUNC(op, S_inst, S_inst_id, U_inst, U_inst_id,      \
                                F_inst, F_inst_id)                             \
  else if (op_type == BinaryOpType::op) {                                      \
    const uint32_t S_inst = S_inst_id;                                         \
    const uint32_t U_inst = U_inst_id;                                         \
    const uint32_t F_inst = F_inst_id;                                         \
    const auto dst_dt = bin->element_type();                                   \
    if (is_integral(dst_dt)) {                                                 \
      if (is_signed(dst_dt)) {                                                 \
        bin_value = ir_->call_glsl450(dst_type, S_inst, lhs_value, rhs_value); \
      } else {                                                                 \
        bin_value = ir_->call_glsl450(dst_type, U_inst, lhs_value, rhs_value); \
      }                                                                        \
    } else if (is_real(dst_dt)) {                                              \
      bin_value = ir_->call_glsl450(dst_type, F_inst, lhs_value, rhs_value);   \
    } else {                                                                   \
      TI_NOT_IMPLEMENTED                                                       \
    }                                                                          \
  }

    BINARY_OP_TO_SPIRV_FUNC(min, SMin, 39, UMin, 38, FMin, 37)
    BINARY_OP_TO_SPIRV_FUNC(max, SMax, 42, UMax, 41, FMax, 40)
#undef BINARY_OP_TO_SPIRV_FUNC
    else if (op_type == BinaryOpType::truediv) {
      lhs_value = ir_->cast(dst_type, lhs_value);
      rhs_value = ir_->cast(dst_type, rhs_value);
      bin_value = ir_->div(lhs_value, rhs_value);
    }
    else {TI_NOT_IMPLEMENTED} ir_->register_value(bin_name, bin_value);
  }

  void visit(TernaryOpStmt *tri) override {
    TI_ASSERT(tri->op_type == TernaryOpType::select);
    spirv::Value op1 = ir_->query_value(tri->op1->raw_name());
    spirv::Value op2 = ir_->query_value(tri->op2->raw_name());
    spirv::Value op3 = ir_->query_value(tri->op3->raw_name());
    spirv::Value tri_val =
        ir_->cast(ir_->get_primitive_type(tri->element_type()),
                  ir_->select(ir_->cast(ir_->bool_type(), op1), op2, op3));
    ir_->register_value(tri->raw_name(), tri_val);
  }

  inline bool ends_with(std::string const &value, std::string const &ending) {
    if (ending.size() > value.size())
      return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
  }

  void visit(TexturePtrStmt *stmt) override {
    spirv::Value val;

    auto arg_id = stmt->arg_load_stmt->as<ArgLoadStmt>()->arg_id;
    if (argid_to_tex_value_.find(arg_id) != argid_to_tex_value_.end()) {
      val = argid_to_tex_value_.at(arg_id);
    } else {
      if (stmt->is_storage) {
        BufferFormat format = stmt->format;

        int binding = binding_head_++;
        val =
            ir_->storage_image_argument(/*num_channels=*/4, stmt->dimensions,
                                        /*descriptor_set=*/0, binding, format);
        TextureBind bind;
        bind.arg_id = arg_id;
        bind.binding = binding;
        bind.is_storage = true;
        texture_binds_.push_back(bind);
        argid_to_tex_value_[arg_id] = val;
      } else {
        int binding = binding_head_++;
        val = ir_->texture_argument(/*num_channels=*/4, stmt->dimensions,
                                    /*descriptor_set=*/0, binding);
        TextureBind bind;
        bind.arg_id = arg_id;
        bind.binding = binding;
        texture_binds_.push_back(bind);
        argid_to_tex_value_[arg_id] = val;
      }
    }

    ir_->register_value(stmt->raw_name(), val);
  }

  void visit(TextureOpStmt *stmt) override {
    spirv::Value tex = ir_->query_value(stmt->texture_ptr->raw_name());
    spirv::Value val;
    if (stmt->op == TextureOpType::kSampleLod ||
        stmt->op == TextureOpType::kFetchTexel) {
      // Texture Ops
      std::vector<spirv::Value> args;
      for (int i = 0; i < stmt->args.size() - 1; i++) {
        args.push_back(ir_->query_value(stmt->args[i]->raw_name()));
      }
      spirv::Value lod = ir_->query_value(stmt->args.back()->raw_name());
      if (stmt->op == TextureOpType::kSampleLod) {
        // Sample
        val = ir_->sample_texture(tex, args, lod);
      } else if (stmt->op == TextureOpType::kFetchTexel) {
        // Texel fetch
        val = ir_->fetch_texel(tex, args, lod);
      }
      ir_->register_value(stmt->raw_name(), val);
    } else if (stmt->op == TextureOpType::kLoad ||
               stmt->op == TextureOpType::kStore) {
      // Image Ops
      std::vector<spirv::Value> args;
      for (int i = 0; i < stmt->args.size(); i++) {
        args.push_back(ir_->query_value(stmt->args[i]->raw_name()));
      }
      if (stmt->op == TextureOpType::kLoad) {
        // Image Load
        val = ir_->image_load(tex, args);
        ir_->register_value(stmt->raw_name(), val);
      } else if (stmt->op == TextureOpType::kStore) {
        // Image Store
        ir_->image_store(tex, args);
      }
    } else {
      TI_NOT_IMPLEMENTED;
    }
  }

  void visit(InternalFuncStmt *stmt) override {
    spirv::Value val;

    if (stmt->func_name == "composite_extract_0") {
      val = ir_->make_value(spv::OpCompositeExtract, ir_->f32_type(),
                            ir_->query_value(stmt->args[0]->raw_name()), 0);
    } else if (stmt->func_name == "composite_extract_1") {
      val = ir_->make_value(spv::OpCompositeExtract, ir_->f32_type(),
                            ir_->query_value(stmt->args[0]->raw_name()), 1);
    } else if (stmt->func_name == "composite_extract_2") {
      val = ir_->make_value(spv::OpCompositeExtract, ir_->f32_type(),
                            ir_->query_value(stmt->args[0]->raw_name()), 2);
    } else if (stmt->func_name == "composite_extract_3") {
      val = ir_->make_value(spv::OpCompositeExtract, ir_->f32_type(),
                            ir_->query_value(stmt->args[0]->raw_name()), 3);
    }

    const std::unordered_set<std::string> reduction_ops{
        "subgroupAdd", "subgroupMul", "subgroupMin", "subgroupMax",
        "subgroupAnd", "subgroupOr",  "subgroupXor"};

    const std::unordered_set<std::string> inclusive_scan_ops{
        "subgroupInclusiveAdd", "subgroupInclusiveMul", "subgroupInclusiveMin",
        "subgroupInclusiveMax", "subgroupInclusiveAnd", "subgroupInclusiveOr",
        "subgroupInclusiveXor"};

    const std::unordered_set<std::string> shuffle_ops{
        "subgroupShuffleDown", "subgroupShuffleUp", "subgroupShuffle"};

    if (stmt->func_name == "workgroupBarrier") {
      ir_->make_inst(
          spv::OpControlBarrier,
          ir_->int_immediate_number(ir_->i32_type(), spv::ScopeWorkgroup),
          ir_->int_immediate_number(ir_->i32_type(), spv::ScopeWorkgroup),
          ir_->int_immediate_number(
              ir_->i32_type(), spv::MemorySemanticsWorkgroupMemoryMask |
                                   spv::MemorySemanticsAcquireReleaseMask));
      val = ir_->const_i32_zero_;
    } else if (stmt->func_name == "localInvocationId") {
      val = ir_->cast(ir_->i32_type(), ir_->get_local_invocation_id(0));
    } else if (stmt->func_name == "globalInvocationId") {
      val = ir_->cast(ir_->i32_type(), ir_->get_global_invocation_id(0));
    } else if (stmt->func_name == "workgroupMemoryBarrier") {
      ir_->make_inst(
          spv::OpMemoryBarrier,
          ir_->int_immediate_number(ir_->i32_type(), spv::ScopeWorkgroup),
          ir_->int_immediate_number(
              ir_->i32_type(), spv::MemorySemanticsWorkgroupMemoryMask |
                                   spv::MemorySemanticsAcquireReleaseMask));
      val = ir_->const_i32_zero_;
    } else if (stmt->func_name == "subgroupElect") {
      val = ir_->make_value(
          spv::OpGroupNonUniformElect, ir_->bool_type(),
          ir_->int_immediate_number(ir_->i32_type(), spv::ScopeSubgroup));
      val = ir_->cast(ir_->i32_type(), val);
    } else if (stmt->func_name == "subgroupBarrier") {
      ir_->make_inst(
          spv::OpControlBarrier,
          ir_->int_immediate_number(ir_->i32_type(), spv::ScopeSubgroup),
          ir_->int_immediate_number(ir_->i32_type(), spv::ScopeSubgroup),
          ir_->const_i32_zero_);
      val = ir_->const_i32_zero_;
    } else if (stmt->func_name == "subgroupMemoryBarrier") {
      ir_->make_inst(
          spv::OpMemoryBarrier,
          ir_->int_immediate_number(ir_->i32_type(), spv::ScopeSubgroup),
          ir_->const_i32_zero_);
      val = ir_->const_i32_zero_;
    } else if (stmt->func_name == "subgroupSize") {
      val = ir_->cast(ir_->i32_type(), ir_->get_subgroup_size());
    } else if (stmt->func_name == "subgroupInvocationId") {
      val = ir_->cast(ir_->i32_type(), ir_->get_subgroup_invocation_id());
    } else if (stmt->func_name == "subgroupBroadcast") {
      auto value = ir_->query_value(stmt->args[0]->raw_name());
      auto index = ir_->query_value(stmt->args[1]->raw_name());
      val = ir_->make_value(
          spv::OpGroupNonUniformBroadcast, value.stype,
          ir_->int_immediate_number(ir_->i32_type(), spv::ScopeSubgroup), value,
          index);
    } else if (reduction_ops.find(stmt->func_name) != reduction_ops.end() ||
               inclusive_scan_ops.find(stmt->func_name) !=
                   inclusive_scan_ops.end()) {
      auto arg = ir_->query_value(stmt->args[0]->raw_name());
      auto stype = ir_->get_primitive_type(stmt->args[0]->ret_type);
      spv::Op spv_op;

      if (ends_with(stmt->func_name, "Add")) {
        if (is_integral(stmt->args[0]->ret_type)) {
          spv_op = spv::OpGroupNonUniformIAdd;
        } else {
          spv_op = spv::OpGroupNonUniformFAdd;
        }
      } else if (ends_with(stmt->func_name, "Mul")) {
        if (is_integral(stmt->args[0]->ret_type)) {
          spv_op = spv::OpGroupNonUniformIMul;
        } else {
          spv_op = spv::OpGroupNonUniformFMul;
        }
      } else if (ends_with(stmt->func_name, "Min")) {
        if (is_integral(stmt->args[0]->ret_type)) {
          if (is_signed(stmt->args[0]->ret_type)) {
            spv_op = spv::OpGroupNonUniformSMin;
          } else {
            spv_op = spv::OpGroupNonUniformUMin;
          }
        } else {
          spv_op = spv::OpGroupNonUniformFMin;
        }
      } else if (ends_with(stmt->func_name, "Max")) {
        if (is_integral(stmt->args[0]->ret_type)) {
          if (is_signed(stmt->args[0]->ret_type)) {
            spv_op = spv::OpGroupNonUniformSMax;
          } else {
            spv_op = spv::OpGroupNonUniformUMax;
          }
        } else {
          spv_op = spv::OpGroupNonUniformFMax;
        }
      } else if (ends_with(stmt->func_name, "And")) {
        spv_op = spv::OpGroupNonUniformBitwiseAnd;
      } else if (ends_with(stmt->func_name, "Or")) {
        spv_op = spv::OpGroupNonUniformBitwiseOr;
      } else if (ends_with(stmt->func_name, "Xor")) {
        spv_op = spv::OpGroupNonUniformBitwiseXor;
      } else {
        TI_ERROR("Unsupported operation: {}", stmt->func_name);
      }

      spv::GroupOperation group_op;

      if (reduction_ops.find(stmt->func_name) != reduction_ops.end()) {
        group_op = spv::GroupOperationReduce;
      } else if (inclusive_scan_ops.find(stmt->func_name) !=
                 inclusive_scan_ops.end()) {
        group_op = spv::GroupOperationInclusiveScan;
      }

      val = ir_->make_value(
          spv_op, stype,
          ir_->int_immediate_number(ir_->i32_type(), spv::ScopeSubgroup),
          group_op, arg);
    } else if (shuffle_ops.find(stmt->func_name) != shuffle_ops.end()) {
      auto arg0 = ir_->query_value(stmt->args[0]->raw_name());
      auto arg1 = ir_->query_value(stmt->args[1]->raw_name());
      auto stype = ir_->get_primitive_type(stmt->args[0]->ret_type);
      spv::Op spv_op;

      if (ends_with(stmt->func_name, "Down")) {
        spv_op = spv::OpGroupNonUniformShuffleDown;
      } else if (ends_with(stmt->func_name, "Up")) {
        spv_op = spv::OpGroupNonUniformShuffleUp;
      } else if (ends_with(stmt->func_name, "Shuffle")) {
        spv_op = spv::OpGroupNonUniformShuffle;
      } else {
        TI_ERROR("Unsupported operation: {}", stmt->func_name);
      }

      val = ir_->make_value(
          spv_op, stype,
          ir_->int_immediate_number(ir_->i32_type(), spv::ScopeSubgroup), arg0,
          arg1);
    }
    ir_->register_value(stmt->raw_name(), val);
  }

  void visit(AtomicOpStmt *stmt) override {
    // G9.4: AtomicOpStmt on a bit-pointer (quant_array / bit_struct member)
    // is dispatched to a dedicated CAS-loop helper.  The generic atomic
    // path below assumes a primitive-typed destination and would mis-handle
    // the QuantInt / QuantFixed `dest->element_type()`.
    if (auto *pt = stmt->dest->ret_type->as<PointerType>();
        pt && pt->is_bit_pointer()) {
      emit_bit_pointer_atomic_add(stmt);
      return;
    }
    const auto dt = stmt->dest->element_type().ptr_removed();

    spirv::Value data = ir_->query_value(stmt->val->raw_name());
    spirv::Value val;
    bool use_subgroup_reduction = false;

    if (stmt->is_reduction &&
        caps_->get(DeviceCapability::spirv_has_subgroup_arithmetic)) {
      spv::Op atomic_op = spv::OpNop;
      bool negation = false;
      if (is_integral(dt)) {
        if (stmt->op_type == AtomicOpType::add) {
          atomic_op = spv::OpGroupIAdd;
        } else if (stmt->op_type == AtomicOpType::sub) {
          atomic_op = spv::OpGroupIAdd;
          negation = true;
        } else if (stmt->op_type == AtomicOpType::min) {
          atomic_op = is_signed(dt) ? spv::OpGroupSMin : spv::OpGroupUMin;
        } else if (stmt->op_type == AtomicOpType::max) {
          atomic_op = is_signed(dt) ? spv::OpGroupSMax : spv::OpGroupUMax;
        }
      } else if (is_real(dt)) {
        if (stmt->op_type == AtomicOpType::add) {
          atomic_op = spv::OpGroupFAdd;
        } else if (stmt->op_type == AtomicOpType::sub) {
          atomic_op = spv::OpGroupFAdd;
          negation = true;
        } else if (stmt->op_type == AtomicOpType::min) {
          atomic_op = spv::OpGroupFMin;
        } else if (stmt->op_type == AtomicOpType::max) {
          atomic_op = spv::OpGroupFMax;
        }
      }

      if (atomic_op != spv::OpNop) {
        spirv::Value scope_subgroup =
            ir_->int_immediate_number(ir_->i32_type(), 3);
        spirv::Value operation_reduce = ir_->const_i32_zero_;
        if (negation) {
          if (is_integral(dt)) {
            data = ir_->make_value(spv::OpSNegate, data.stype, data);
          } else {
            data = ir_->make_value(spv::OpFNegate, data.stype, data);
          }
        }
        data = ir_->make_value(atomic_op, ir_->get_primitive_type(dt),
                               scope_subgroup, operation_reduce, data);
        val = data;
        use_subgroup_reduction = true;
      }
    }

    spirv::Label then_label;
    spirv::Label merge_label;

    if (use_subgroup_reduction) {
      spirv::Value subgroup_id = ir_->get_subgroup_invocation_id();
      spirv::Value cond = ir_->make_value(spv::OpIEqual, ir_->bool_type(),
                                          subgroup_id, ir_->const_i32_zero_);

      then_label = ir_->new_label();
      merge_label = ir_->new_label();
      ir_->make_inst(spv::OpSelectionMerge, merge_label,
                     spv::SelectionControlMaskNone);
      ir_->make_inst(spv::OpBranchConditional, cond, then_label, merge_label);
      ir_->start_label(then_label);
    }

    spirv::Value addr_ptr;
    spirv::Value dest_val = ir_->query_value(stmt->dest->raw_name());
    // Shared arrays have already created an accesschain, use it directly.
    const bool dest_is_ptr = dest_val.stype.flag == TypeKind::kPtr;

    if (dt->is_primitive(PrimitiveTypeID::f64)) {
      if (caps_->get(DeviceCapability::spirv_has_atomic_float64_add) &&
          stmt->op_type == AtomicOpType::add) {
        addr_ptr = at_buffer(stmt->dest, dt);
      } else {
        addr_ptr = dest_is_ptr
                       ? dest_val
                       : at_buffer(stmt->dest, ir_->get_taichi_uint_type(dt));
      }
    } else if (dt->is_primitive(PrimitiveTypeID::f32)) {
      if (caps_->get(DeviceCapability::spirv_has_atomic_float_add) &&
          stmt->op_type == AtomicOpType::add) {
        addr_ptr = at_buffer(stmt->dest, dt);
      } else {
        addr_ptr = dest_is_ptr
                       ? dest_val
                       : at_buffer(stmt->dest, ir_->get_taichi_uint_type(dt));
      }
    } else {
      addr_ptr = dest_is_ptr ? dest_val : at_buffer(stmt->dest, dt);
    }

    auto ret_type = ir_->get_primitive_type(dt);

    if (is_real(dt)) {
      spv::Op atomic_fp_op;
      if (stmt->op_type == AtomicOpType::add) {
        atomic_fp_op = spv::OpAtomicFAddEXT;
      }

      bool use_native_atomics = false;

      if (dt->is_primitive(PrimitiveTypeID::f64)) {
        if (caps_->get(DeviceCapability::spirv_has_atomic_float64_add) &&
            stmt->op_type == AtomicOpType::add) {
          use_native_atomics = true;
        }
      } else if (dt->is_primitive(PrimitiveTypeID::f32)) {
        if (caps_->get(DeviceCapability::spirv_has_atomic_float_add) &&
            stmt->op_type == AtomicOpType::add) {
          use_native_atomics = true;
        }
      } else if (dt->is_primitive(PrimitiveTypeID::f16)) {
        if (caps_->get(DeviceCapability::spirv_has_atomic_float16_add) &&
            stmt->op_type == AtomicOpType::add) {
          use_native_atomics = true;
        }
      }

      if (use_native_atomics) {
        val =
            ir_->make_value(atomic_fp_op, ir_->get_primitive_type(dt), addr_ptr,
                            /*scope=*/ir_->const_i32_one_,
                            /*semantics=*/ir_->const_i32_zero_, data);
      } else {
        val = ir_->float_atomic(stmt->op_type, addr_ptr, data, dt);
      }
    } else if (is_integral(dt)) {
      bool use_native_atomics = false;
      spv::Op op;
      if (stmt->op_type == AtomicOpType::add) {
        op = spv::OpAtomicIAdd;
        use_native_atomics = true;
      } else if (stmt->op_type == AtomicOpType::sub) {
        op = spv::OpAtomicISub;
        use_native_atomics = true;
      } else if (stmt->op_type == AtomicOpType::mul) {
        addr_ptr = at_buffer(stmt->dest, ir_->get_taichi_uint_type(dt));
        val = ir_->integer_atomic(stmt->op_type, addr_ptr, data, dt);
        use_native_atomics = false;
      } else if (stmt->op_type == AtomicOpType::min) {
        op = is_signed(dt) ? spv::OpAtomicSMin : spv::OpAtomicUMin;
        use_native_atomics = true;
      } else if (stmt->op_type == AtomicOpType::max) {
        op = is_signed(dt) ? spv::OpAtomicSMax : spv::OpAtomicUMax;
        use_native_atomics = true;
      } else if (stmt->op_type == AtomicOpType::bit_or) {
        op = spv::OpAtomicOr;
        use_native_atomics = true;
      } else if (stmt->op_type == AtomicOpType::bit_and) {
        op = spv::OpAtomicAnd;
        use_native_atomics = true;
      } else if (stmt->op_type == AtomicOpType::bit_xor) {
        op = spv::OpAtomicXor;
        use_native_atomics = true;
      } else {
        TI_NOT_IMPLEMENTED
      }

      if (use_native_atomics) {
        auto uint_type = ir_->get_primitive_uint_type(dt);

        if (data.stype.id != addr_ptr.stype.element_type_id) {
          data = ir_->make_value(spv::OpBitcast, ret_type, data);
        }

        // Semantics = (UniformMemory 0x40) | (AcquireRelease 0x8)
        ir_->make_inst(
            spv::OpMemoryBarrier, ir_->const_i32_one_,
            ir_->uint_immediate_number(
                ir_->u32_type(), spv::MemorySemanticsAcquireReleaseMask |
                                     spv::MemorySemanticsUniformMemoryMask));
        val = ir_->make_value(op, ret_type, addr_ptr,
                              /*scope=*/ir_->const_i32_one_,
                              /*semantics=*/ir_->const_i32_zero_, data);

        if (val.stype.id != ret_type.id) {
          val = ir_->make_value(spv::OpBitcast, ret_type, val);
        }
      }
    } else {
      TI_NOT_IMPLEMENTED
    }

    if (use_subgroup_reduction) {
      ir_->make_inst(spv::OpBranch, merge_label);
      ir_->start_label(merge_label);
    }

    ir_->register_value(stmt->raw_name(), val);
  }

  void visit(IfStmt *if_stmt) override {
    spirv::Value cond_v = ir_->cast(
        ir_->bool_type(), ir_->query_value(if_stmt->cond->raw_name()));
    spirv::Value cond =
        ir_->ne(cond_v, ir_->cast(ir_->bool_type(), ir_->const_i32_zero_));
    spirv::Label then_label = ir_->new_label();
    spirv::Label merge_label = ir_->new_label();
    spirv::Label else_label = ir_->new_label();
    ir_->make_inst(spv::OpSelectionMerge, merge_label,
                   spv::SelectionControlMaskNone);
    ir_->make_inst(spv::OpBranchConditional, cond, then_label, else_label);
    // then block
    ir_->start_label(then_label);
    if (if_stmt->true_statements) {
      if_stmt->true_statements->accept(this);
    }
    // ContinueStmt must be in IfStmt
    if (gen_label_) {  // Skip OpBranch, because ContinueStmt already generated
                       // one
      gen_label_ = false;
    } else {
      ir_->make_inst(spv::OpBranch, merge_label);
    }
    // else block
    ir_->start_label(else_label);
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
    if (gen_label_) {
      gen_label_ = false;
    } else {
      ir_->make_inst(spv::OpBranch, merge_label);
    }
    // merge label
    ir_->start_label(merge_label);
  }

  void visit(RangeForStmt *for_stmt) override {
    auto loop_var_name = for_stmt->raw_name();
    // Must get init label after making value(to make sure they are correct)
    spirv::Label init_label = ir_->current_label();
    spirv::Label head_label = ir_->new_label();
    spirv::Label body_label = ir_->new_label();
    spirv::Label continue_label = ir_->new_label();
    spirv::Label merge_label = ir_->new_label();

    spirv::Value begin_ = ir_->query_value(for_stmt->begin->raw_name());
    spirv::Value end_ = ir_->query_value(for_stmt->end->raw_name());
    spirv::Value init_value;
    spirv::Value extent_value;
    if (!for_stmt->reversed) {
      init_value = begin_;
      extent_value = end_;
    } else {
      // reversed for loop
      init_value = ir_->sub(end_, ir_->const_i32_one_);
      extent_value = begin_;
    }
    ir_->make_inst(spv::OpBranch, head_label);

    // Loop head
    ir_->start_label(head_label);
    spirv::PhiValue loop_var = ir_->make_phi(init_value.stype, 2);
    loop_var.set_incoming(0, init_value, init_label);
    spirv::Value loop_cond;
    if (!for_stmt->reversed) {
      loop_cond = ir_->lt(loop_var, extent_value);
    } else {
      loop_cond = ir_->ge(loop_var, extent_value);
    }
    ir_->make_inst(spv::OpLoopMerge, merge_label, continue_label,
                   spv::LoopControlMaskNone);
    ir_->make_inst(spv::OpBranchConditional, loop_cond, body_label,
                   merge_label);

    // loop body
    ir_->start_label(body_label);
    push_loop_control_labels(continue_label, merge_label);
    ir_->register_value(loop_var_name, spirv::Value(loop_var));
    for_stmt->body->accept(this);
    pop_loop_control_labels();
    ir_->make_inst(spv::OpBranch, continue_label);

    // loop continue
    ir_->start_label(continue_label);
    spirv::Value next_value;
    if (!for_stmt->reversed) {
      next_value = ir_->add(loop_var, ir_->const_i32_one_);
    } else {
      next_value = ir_->sub(loop_var, ir_->const_i32_one_);
    }
    loop_var.set_incoming(1, next_value, ir_->current_label());
    ir_->make_inst(spv::OpBranch, head_label);
    // loop merge
    ir_->start_label(merge_label);
  }

  void visit(WhileStmt *stmt) override {
    spirv::Label head_label = ir_->new_label();
    spirv::Label body_label = ir_->new_label();
    spirv::Label continue_label = ir_->new_label();
    spirv::Label merge_label = ir_->new_label();
    ir_->make_inst(spv::OpBranch, head_label);

    // Loop head
    ir_->start_label(head_label);
    ir_->make_inst(spv::OpLoopMerge, merge_label, continue_label,
                   spv::LoopControlMaskNone);
    ir_->make_inst(spv::OpBranch, body_label);

    // loop body
    ir_->start_label(body_label);
    push_loop_control_labels(continue_label, merge_label);
    stmt->body->accept(this);
    pop_loop_control_labels();
    ir_->make_inst(spv::OpBranch, continue_label);

    // loop continue
    ir_->start_label(continue_label);
    ir_->make_inst(spv::OpBranch, head_label);

    // loop merge
    ir_->start_label(merge_label);
  }

  void visit(WhileControlStmt *stmt) override {
    spirv::Value cond_v =
        ir_->cast(ir_->bool_type(), ir_->query_value(stmt->cond->raw_name()));
    spirv::Value cond =
        ir_->eq(cond_v, ir_->cast(ir_->bool_type(), ir_->const_i32_zero_));
    spirv::Label then_label = ir_->new_label();
    spirv::Label merge_label = ir_->new_label();

    ir_->make_inst(spv::OpSelectionMerge, merge_label,
                   spv::SelectionControlMaskNone);
    ir_->make_inst(spv::OpBranchConditional, cond, then_label, merge_label);
    ir_->start_label(then_label);
    ir_->make_inst(spv::OpBranch, current_merge_label());  // break;
    ir_->start_label(merge_label);
  }

  void visit(ContinueStmt *stmt) override {
    auto stmt_in_off_for = [stmt]() {
      TI_ASSERT(stmt->scope != nullptr);
      if (auto *offl = stmt->scope->cast<OffloadedStmt>(); offl) {
        TI_ASSERT(offl->task_type == OffloadedStmt::TaskType::range_for ||
                  offl->task_type == OffloadedStmt::TaskType::struct_for);
        return true;
      }
      return false;
    };
    if (stmt_in_off_for()) {
      // Return means end THIS main loop and start next loop, not exit kernel
      ir_->make_inst(spv::OpBranch, return_label());
    } else {
      ir_->make_inst(spv::OpBranch, current_continue_label());
    }
    gen_label_ = true;  // Only ContinueStmt will cause duplicate OpBranch,
                        // which should be eliminated
  }

 private:
  void emit_headers() {
    /*
    for (int root = 0; root < compiled_structs_.size(); ++root) {
      get_buffer_value({BufferType::Root, root});
    }
    */
    std::array<int, 3> group_size = {
        task_attribs_.advisory_num_threads_per_group, 1, 1};
    ir_->set_work_group_size(group_size);
    std::vector<spirv::Value> buffers;
    if (caps_->get(DeviceCapability::spirv_version) > 0x10300) {
      buffers = shared_array_binds_;
      // One buffer can be bound to different bind points but has to be unique
      // in OpEntryPoint interface declarations.
      // From Spec: before SPIR-V version 1.4, duplication of these interface id
      // is tolerated. Starting with version 1.4, an interface id must not
      // appear more than once.
      std::unordered_set<spirv::Value, spirv::ValueHasher> entry_point_values;
      for (const auto &bb : task_attribs_.buffer_binds) {
        for (auto &it : buffer_value_map_) {
          if (it.first.first == bb.buffer) {
            entry_point_values.insert(it.second);
          }
        }
      }
      buffers.insert(buffers.end(), entry_point_values.begin(),
                     entry_point_values.end());
    }
    ir_->commit_kernel_function(kernel_function_, "main", buffers,
                                group_size);  // kernel entry
  }

  void generate_gc_noop_kernel(OffloadedStmt *stmt) {
    // G1.b: emit a serial-shaped kernel whose body is empty. The pool
    // freelist is maintained inline by pointer_deactivate, so no GC work
    // is required at this scheduling point.
    task_attribs_.name = task_name_;
    task_attribs_.task_type = OffloadedTaskType::serial;
    task_attribs_.advisory_total_num_threads = 1;
    task_attribs_.advisory_num_threads_per_group = 1;
    ir_->start_function(kernel_function_);
    ir_->make_inst(spv::OpReturn);
    ir_->make_inst(spv::OpFunctionEnd);
    task_attribs_.buffer_binds = get_buffer_binds();
    task_attribs_.texture_binds = get_texture_binds();
  }

  void generate_serial_kernel(OffloadedStmt *stmt) {
    task_attribs_.name = task_name_;
    task_attribs_.task_type = OffloadedTaskType::serial;
    task_attribs_.advisory_total_num_threads = 1;
    task_attribs_.advisory_num_threads_per_group = 1;
    // The computation for a single work is wrapped inside a function, so that
    // we can do grid-strided loop.
    ir_->start_function(kernel_function_);
    spirv::Value cond =
        ir_->eq(ir_->get_global_invocation_id(0),
                ir_->uint_immediate_number(
                    ir_->u32_type(), 0));  // if (gl_GlobalInvocationID.x > 0)
    spirv::Label then_label = ir_->new_label();
    spirv::Label merge_label = ir_->new_label();
    kernel_return_label_ = merge_label;

    ir_->make_inst(spv::OpSelectionMerge, merge_label,
                   spv::SelectionControlMaskNone);
    ir_->make_inst(spv::OpBranchConditional, cond, then_label, merge_label);
    ir_->start_label(then_label);

    // serial kernel
    stmt->body->accept(this);

    ir_->make_inst(spv::OpBranch, merge_label);
    ir_->start_label(merge_label);
    ir_->make_inst(spv::OpReturn);       // return;
    ir_->make_inst(spv::OpFunctionEnd);  // } Close kernel

    task_attribs_.buffer_binds = get_buffer_binds();
    task_attribs_.texture_binds = get_texture_binds();
  }

  void gen_array_range(Stmt *stmt) {
    /* Fix issue 7493
     *
     * Prevent repeated range generation for the same array
     * when loop range has multiple dimensions.
     */
    if (ir_->check_value_existence(stmt->raw_name())) {
      return;
    }
    int num_operands = stmt->num_operands();
    for (int i = 0; i < num_operands; i++) {
      gen_array_range(stmt->operand(i));
    }
    offload_loop_motion_.insert(stmt);
    stmt->accept(this);
  }

  void generate_range_for_kernel(OffloadedStmt *stmt) {
    task_attribs_.name = task_name_;
    task_attribs_.task_type = OffloadedTaskType::range_for;

    task_attribs_.range_for_attribs = TaskAttributes::RangeForAttributes();
    auto &range_for_attribs = task_attribs_.range_for_attribs.value();
    range_for_attribs.const_begin = stmt->const_begin;
    range_for_attribs.const_end = stmt->const_end;
    range_for_attribs.begin =
        (stmt->const_begin ? stmt->begin_value : stmt->begin_offset);
    range_for_attribs.end =
        (stmt->const_end ? stmt->end_value : stmt->end_offset);

    ir_->start_function(kernel_function_);
    const std::string total_elems_name("total_elems");
    spirv::Value total_elems;
    spirv::Value begin_expr_value;
    if (range_for_attribs.const_range()) {
      const int num_elems = range_for_attribs.end - range_for_attribs.begin;
      begin_expr_value = ir_->int_immediate_number(
          ir_->i32_type(), stmt->begin_value, false);  // Named Constant
      total_elems = ir_->int_immediate_number(ir_->i32_type(), num_elems,
                                              false);  // Named Constant
      task_attribs_.advisory_total_num_threads = num_elems;
    } else {
      spirv::Value end_expr_value;
      if (stmt->end_stmt) {
        // Range from args
        TI_ASSERT(stmt->const_begin);
        begin_expr_value = ir_->int_immediate_number(ir_->i32_type(),
                                                     stmt->begin_value, false);
        gen_array_range(stmt->end_stmt);
        end_expr_value = ir_->query_value(stmt->end_stmt->raw_name());
      } else {
        // Range from gtmp / constant
        if (!stmt->const_begin) {
          spirv::Value begin_idx = ir_->make_value(
              spv::OpShiftRightArithmetic, ir_->i32_type(),
              ir_->int_immediate_number(ir_->i32_type(), stmt->begin_offset),
              ir_->int_immediate_number(ir_->i32_type(), 2));
          begin_expr_value = ir_->load_variable(
              ir_->struct_array_access(
                  ir_->i32_type(),
                  get_buffer_value(BufferType::GlobalTmps, PrimitiveType::i32),
                  begin_idx),
              ir_->i32_type());
        } else {
          begin_expr_value = ir_->int_immediate_number(
              ir_->i32_type(), stmt->begin_value, false);  // Named Constant
        }
        if (!stmt->const_end) {
          spirv::Value end_idx = ir_->make_value(
              spv::OpShiftRightArithmetic, ir_->i32_type(),
              ir_->int_immediate_number(ir_->i32_type(), stmt->end_offset),
              ir_->int_immediate_number(ir_->i32_type(), 2));
          end_expr_value = ir_->load_variable(
              ir_->struct_array_access(
                  ir_->i32_type(),
                  get_buffer_value(BufferType::GlobalTmps, PrimitiveType::i32),
                  end_idx),
              ir_->i32_type());
        } else {
          end_expr_value =
              ir_->int_immediate_number(ir_->i32_type(), stmt->end_value, true);
        }
      }
      total_elems = ir_->sub(end_expr_value, begin_expr_value);
      task_attribs_.advisory_total_num_threads = kMaxNumThreadsGridStrideLoop;
    }
    task_attribs_.advisory_num_threads_per_group = stmt->block_dim;
    ir_->debug_name(spv::OpName, begin_expr_value, "begin_expr_value");
    ir_->debug_name(spv::OpName, total_elems, total_elems_name);

    spirv::Value begin_ =
        ir_->add(ir_->cast(ir_->i32_type(), ir_->get_global_invocation_id(0)),
                 begin_expr_value);
    ir_->debug_name(spv::OpName, begin_, "begin_");
    spirv::Value end_ = ir_->add(total_elems, begin_expr_value);
    ir_->debug_name(spv::OpName, end_, "end_");
    const std::string total_invocs_name = "total_invocs";
    // For now, |total_invocs_name| is equal to |total_elems|. Once we support
    // dynamic range, they will be different.
    // https://www.khronos.org/opengl/wiki/Compute_Shader#Inputs

    // HLSL & WGSL cross compilers do not support this builtin
    spirv::Value total_invocs = ir_->cast(
        ir_->i32_type(),
        ir_->mul(ir_->get_num_work_groups(0),
                 ir_->uint_immediate_number(
                     ir_->u32_type(),
                     task_attribs_.advisory_num_threads_per_group, true)));
    /*
    const int group_x = (task_attribs_.advisory_total_num_threads +
                         task_attribs_.advisory_num_threads_per_group - 1) /
                        task_attribs_.advisory_num_threads_per_group;
    spirv::Value total_invocs = ir_->uint_immediate_number(
        ir_->i32_type(), group_x * task_attribs_.advisory_num_threads_per_group,
        false);
        */

    ir_->debug_name(spv::OpName, total_invocs, total_invocs_name);

    // Must get init label after making value(to make sure they are correct)
    spirv::Label init_label = ir_->current_label();
    spirv::Label head_label = ir_->new_label();
    spirv::Label body_label = ir_->new_label();
    spirv::Label continue_label = ir_->new_label();
    spirv::Label merge_label = ir_->new_label();
    ir_->make_inst(spv::OpBranch, head_label);

    // loop head
    ir_->start_label(head_label);
    spirv::PhiValue loop_var = ir_->make_phi(begin_.stype, 2);
    ir_->register_value("ii", loop_var);
    loop_var.set_incoming(0, begin_, init_label);
    spirv::Value loop_cond = ir_->lt(loop_var, end_);
    ir_->make_inst(spv::OpLoopMerge, merge_label, continue_label,
                   spv::LoopControlMaskNone);
    ir_->make_inst(spv::OpBranchConditional, loop_cond, body_label,
                   merge_label);

    // loop body
    ir_->start_label(body_label);
    push_loop_control_labels(continue_label, merge_label);

    // loop kernel
    stmt->body->accept(this);
    pop_loop_control_labels();
    ir_->make_inst(spv::OpBranch, continue_label);

    // loop continue
    ir_->start_label(continue_label);
    spirv::Value next_value = ir_->add(loop_var, total_invocs);
    loop_var.set_incoming(1, next_value, ir_->current_label());
    ir_->make_inst(spv::OpBranch, head_label);

    // loop merge
    ir_->start_label(merge_label);

    ir_->make_inst(spv::OpReturn);
    ir_->make_inst(spv::OpFunctionEnd);

    task_attribs_.buffer_binds = get_buffer_binds();
    task_attribs_.texture_binds = get_texture_binds();
  }

  void generate_struct_for_kernel(OffloadedStmt *stmt) {
    task_attribs_.name = task_name_;
    task_attribs_.task_type = OffloadedTaskType::struct_for;
    task_attribs_.advisory_total_num_threads = 65536;
    task_attribs_.advisory_num_threads_per_group = 128;

    // The computation for a single work is wrapped inside a function, so that
    // we can do grid-strided loop.
    ir_->start_function(kernel_function_);

    auto listgen_buffer =
        get_buffer_value(BufferType::ListGen, PrimitiveType::u32);
    auto listgen_count_ptr = ir_->struct_array_access(
        ir_->u32_type(), listgen_buffer, ir_->const_i32_zero_);
    auto listgen_count = ir_->load_variable(listgen_count_ptr, ir_->u32_type());

    auto invoc_index = ir_->get_global_invocation_id(0);

    spirv::Label loop_head = ir_->new_label();
    spirv::Label loop_body = ir_->new_label();
    spirv::Label loop_merge = ir_->new_label();

    auto loop_index_var = ir_->alloca_variable(ir_->u32_type());
    ir_->store_variable(loop_index_var, invoc_index);

    ir_->make_inst(spv::OpBranch, loop_head);
    ir_->start_label(loop_head);
    // for (; index < list_size; index += gl_NumWorkGroups.x *
    // gl_WorkGroupSize.x)
    auto loop_index = ir_->load_variable(loop_index_var, ir_->u32_type());
    auto loop_cond = ir_->make_value(spv::OpULessThan, ir_->bool_type(),
                                     loop_index, listgen_count);
    ir_->make_inst(spv::OpLoopMerge, loop_merge, loop_body,
                   spv::LoopControlMaskNone);
    ir_->make_inst(spv::OpBranchConditional, loop_cond, loop_body, loop_merge);
    {
      ir_->start_label(loop_body);
      auto listgen_index_ptr = ir_->struct_array_access(
          ir_->u32_type(), listgen_buffer,
          ir_->add(ir_->uint_immediate_number(ir_->u32_type(), 1), loop_index));
      auto listgen_index =
          ir_->load_variable(listgen_index_ptr, ir_->u32_type());

      // kernel
      ir_->register_value("ii", listgen_index);
      stmt->body->accept(this);

      // continue
      spirv::Value total_invocs = ir_->cast(
          ir_->u32_type(),
          ir_->mul(ir_->get_num_work_groups(0),
                   ir_->uint_immediate_number(
                       ir_->u32_type(),
                       task_attribs_.advisory_num_threads_per_group, true)));
      auto next_index = ir_->add(loop_index, total_invocs);
      ir_->store_variable(loop_index_var, next_index);
      ir_->make_inst(spv::OpBranch, loop_head);
    }
    ir_->start_label(loop_merge);

    ir_->make_inst(spv::OpReturn);       // return;
    ir_->make_inst(spv::OpFunctionEnd);  // } Close kernel

    task_attribs_.buffer_binds = get_buffer_binds();
    task_attribs_.texture_binds = get_texture_binds();
  }

  // Phase 1b/1c/1d (taichi-forge 0.3.x): SPIR-V codegen for
  // OffloadedTaskType::listgen. Scope:
  //   * Phase 1b/1c: depth-1 bitmasked (parent == root, child == bitmasked).
  //   * Phase 1d:   nested SNode paths consisting of dense/bitmasked nodes,
  //                 e.g. root.dense.bitmasked, root.bitmasked.bitmasked,
  //                 root.bitmasked.dense, etc.
  //
  // A single 32 MiB listgen buffer holds [count, idx0, idx1, ...] of *flat*
  // global cell indices. The struct_for kernel grid-strides over this list
  // and consumes the indices via "ii".
  //
  // The compiler emits one OffloadedStmt(listgen) per non-root SNode along
  // the path (root -> leaf->parent). Only the LAST listgen task in the chain
  // matters: each listgen task overwrites listgen_buffer, and the runtime
  // zero-fills listgen_buffer before every listgen dispatch
  // (taichi/runtime/gfx/runtime.cpp). Intermediate listgen tasks therefore
  // emit a no-op kernel; the final listgen task (whose snode's children are
  // all places, i.e. the leaf-parent) runs the full ancestor-mask AND scan
  // and writes the active flat indices.
  void generate_listgen_kernel(OffloadedStmt *stmt) {
    auto *child_sn = stmt->snode;
    TI_ASSERT(child_sn != nullptr);
    TI_ASSERT(child_sn->parent != nullptr);

    task_attribs_.name = task_name_;
    task_attribs_.task_type = OffloadedTaskType::listgen;

    ir_->start_function(kernel_function_);

    // Build path from the topmost non-root ancestor down to child_sn.
    std::vector<SNode *> path;
    for (auto *sn = child_sn; sn != nullptr && sn->type != SNodeType::root;
         sn = sn->parent) {
      path.push_back(sn);
    }
    std::reverse(path.begin(), path.end());
    TI_ASSERT(!path.empty());

    // Phase 1d/2c handles dense + bitmasked + pointer nodes. G4 adds
    // dynamic. Reject everything else (hash / quant_array) early.
    for (auto *sn : path) {
      bool ok = sn->type == SNodeType::dense ||
                sn->type == SNodeType::bitmasked ||
                sn->type == SNodeType::pointer;
#if defined(TI_VULKAN_DYNAMIC)
      ok = ok || (sn->type == SNodeType::dynamic);
#endif
      TI_ERROR_IF(!ok,
                  "SPIR-V listgen for SNode type '{}' is not implemented.",
                  snode_type_name(sn->type));
    }

    const int root_id = snode_to_root_.at(child_sn->id);
    const auto &snode_descs = compiled_structs_[root_id].snode_descriptors;

    // Always reference the listgen buffer so its binding is registered. We
    // need this even for the no-op intermediate kernels so the runtime
    // zero-fill loop can find it (and so dispatch validation passes).
    auto listgen_buffer =
        get_buffer_value(BufferType::ListGen, PrimitiveType::u32);
    (void)listgen_buffer;

    // A listgen task lists active elements of `child_sn` at its own depth in
    // the SNode tree. For Phase 1d we always do the full ancestor-mask AND
    // scan and emit `child_sn`-flat-global-axis indices, regardless of
    // whether `child_sn`'s children are places. The struct_for that
    // ultimately consumes the listgen output binds to the last listgen of
    // its chain, and the runtime zero-fills listgen_buffer before each
    // dispatch (taichi/runtime/gfx/runtime.cpp), so prior listgens in the
    // same chain are simply overwritten.

    // ----- Full scan over path -----
    const int n = (int)path.size();
    size_t total_cells = 1;
    for (auto *sn : path) total_cells *= sn->num_cells_per_container;

    task_attribs_.advisory_total_num_threads = (int)total_cells;
    task_attribs_.advisory_num_threads_per_group =
        std::min<int>(128, (int)total_cells);
    if (task_attribs_.advisory_num_threads_per_group == 0) {
      task_attribs_.advisory_num_threads_per_group = 1;
    }

    auto root_buffer = get_buffer_value(BufferInfo(BufferType::Root, root_id),
                                        PrimitiveType::u32);

    auto u32_t = ir_->u32_type();

    // B-3.c-2（2026-05）：listgen 全扫描在不同 level 上访问的容器可能位于
    // 不同 buffer——每跨过一个 pointer SNode（且其 binding_id>=0），后续
    // level 的 slot/mask/length 都从该 pointer 自己的独立池 buffer 读。
    // OFF 默认所有 binding_id==-1，level_buffers 全等于 root_buffer，行为
    // 字节等价旧路径。
    std::vector<spirv::Value> level_buffers((size_t)n);
    {
      spirv::Value cur = root_buffer;
      for (int k = 0; k < n; ++k) {
        level_buffers[(size_t)k] = cur;
        if (path[k]->type == SNodeType::pointer) {
          const auto &c =
              compiled_structs_[root_id].pointer_contracts.at(path[k]->id);
          if (c.pool_buffer_binding_id >= 0) {
            cur = get_buffer_value(
                BufferInfo(BufferType::NodeAllocatorPool,
                           c.pool_buffer_binding_id),
                PrimitiveType::u32);
          }
        }
      }
    }

    auto gid = ir_->cast(u32_t, ir_->get_global_invocation_id(0));
    auto num_cells_const = ir_->uint_immediate_number(u32_t, total_cells);

    // if (gid < total_cells) { ... }
    spirv::Label in_bounds = ir_->new_label();
    spirv::Label after_bounds = ir_->new_label();
    auto in_bounds_cond = ir_->make_value(spv::OpULessThan, ir_->bool_type(),
                                          gid, num_cells_const);
    ir_->make_inst(spv::OpSelectionMerge, after_bounds,
                   spv::SelectionControlMaskNone);
    ir_->make_inst(spv::OpBranchConditional, in_bounds_cond, in_bounds,
                   after_bounds);
    ir_->start_label(in_bounds);

    // Decode per-level cell index: i_path[k] = (gid / acc[k]) % shape[k]
    // with acc[n-1] = 1 and acc[k] = acc[k+1] * shape[k+1] (row-major).
    std::vector<size_t> shape(n), acc(n);
    for (int k = 0; k < n; ++k) {
      shape[k] = path[k]->num_cells_per_container;
    }
    acc[n - 1] = 1;
    for (int k = n - 2; k >= 0; --k) acc[k] = acc[k + 1] * shape[k + 1];

    std::vector<spirv::Value> i_path(n);
    for (int k = 0; k < n; ++k) {
      if (shape[k] == 1) {
        // Single-cell container -- index is always 0. Forcing the constant
        // here matters because the mod-by-1 elision would otherwise leak
        // bits from `gid` into bit-position / word-index math downstream.
        i_path[k] = ir_->uint_immediate_number(u32_t, 0);
        continue;
      }
      spirv::Value v = gid;
      if (acc[k] > 1) {
        v = ir_->make_value(spv::OpUDiv, u32_t, v,
                            ir_->uint_immediate_number(u32_t, acc[k]));
      }
      v = ir_->make_value(spv::OpUMod, u32_t, v,
                          ir_->uint_immediate_number(u32_t, shape[k]));
      i_path[k] = v;
    }

    // Walk path computing the running container address and ANDing every
    // bitmasked ancestor's active bit.
    //
    // addr starts at path[0].mem_offset_in_parent_cell, since root has a
    // single cell at offset 0 and path[0] is a child of root.
    //
    // For each step k:
    //   * if path[k] is bitmasked, mask_area = addr + cell_stride * num_cells
    //     and check bit (i_path[k] >> 5) word, (i_path[k] & 31) bit.
    //   * if k < n-1, advance:
    //       addr = addr + i_path[k] * cell_stride
    //                   + path[k+1].mem_offset_in_parent_cell
    spirv::Value addr = ir_->uint_immediate_number(
        u32_t,
        (uint32_t)snode_descs.at(path[0]->id).mem_offset_in_parent_cell);

    spirv::Value active = ir_->uint_immediate_number(u32_t, 1);

    for (int k = 0; k < n; ++k) {
      const auto &desc_k = snode_descs.at(path[k]->id);
      // Pointer slot lookup is computed once per level when path[k] is a
      // pointer; reused both for the active-mask AND and for the next-
      // level addr advance.
      spirv::Value pointer_effective_slot;  // (slot == 0) ? 0 : slot - 1
      bool path_k_is_pointer = (path[k]->type == SNodeType::pointer);
      if (path_k_is_pointer) {
        // slot_byte_addr = addr + 4 * i_path[k]
        auto slot_byte_off = ir_->make_value(
            spv::OpShiftLeftLogical, u32_t, i_path[k],
            ir_->uint_immediate_number(u32_t, 2));
        auto slot_byte_addr = ir_->add(addr, slot_byte_off);
        auto slot_word_idx = ir_->make_value(
            spv::OpShiftRightLogical, u32_t, slot_byte_addr,
            ir_->uint_immediate_number(u32_t, 2));
        auto slot_ptr =
            ir_->struct_array_access(u32_t, level_buffers[(size_t)k], slot_word_idx);
        auto slot_value = ir_->load_variable(slot_ptr, u32_t);
        // active &= (slot != 0)
        auto is_zero = ir_->make_value(
            spv::OpIEqual, ir_->bool_type(), slot_value,
            ir_->uint_immediate_number(u32_t, 0));
        auto bit = ir_->make_value(
            spv::OpSelect, u32_t, is_zero,
            ir_->uint_immediate_number(u32_t, 0),
            ir_->uint_immediate_number(u32_t, 1));
        active = ir_->make_value(spv::OpBitwiseAnd, u32_t, active, bit);
        // effective_slot = (slot == 0) ? 0 : slot - 1; clamp avoids OOB on
        // inactive cells (the `active` bit will gate the listgen write).
        auto slot_minus_one =
            ir_->sub(slot_value, ir_->uint_immediate_number(u32_t, 1));
        pointer_effective_slot = ir_->make_value(
            spv::OpSelect, u32_t, is_zero,
            ir_->uint_immediate_number(u32_t, 0), slot_minus_one);
      } else if (path[k]->type == SNodeType::bitmasked) {
        auto mask_area_offset_const = ir_->uint_immediate_number(
            u32_t,
            (uint32_t)(desc_k.cell_stride * path[k]->num_cells_per_container));
        auto mask_area = ir_->add(addr, mask_area_offset_const);
        auto word_idx = ir_->make_value(
            spv::OpShiftRightLogical, u32_t, i_path[k],
            ir_->uint_immediate_number(u32_t, 5));
        auto word_byte_offset = ir_->make_value(
            spv::OpShiftLeftLogical, u32_t, word_idx,
            ir_->uint_immediate_number(u32_t, 2));
        auto word_byte_addr = ir_->add(mask_area, word_byte_offset);
        auto word_u32_idx = ir_->make_value(
            spv::OpShiftRightLogical, u32_t, word_byte_addr,
            ir_->uint_immediate_number(u32_t, 2));
        auto word_ptr =
            ir_->struct_array_access(u32_t, level_buffers[(size_t)k], word_u32_idx);
        auto mask_word = ir_->load_variable(word_ptr, u32_t);
        auto bit_idx = ir_->make_value(
            spv::OpBitwiseAnd, u32_t, i_path[k],
            ir_->uint_immediate_number(u32_t, 31));
        auto shifted = ir_->make_value(spv::OpShiftRightLogical, u32_t,
                                       mask_word, bit_idx);
        auto bit = ir_->make_value(spv::OpBitwiseAnd, u32_t, shifted,
                                   ir_->uint_immediate_number(u32_t, 1));
        active = ir_->make_value(spv::OpBitwiseAnd, u32_t, active, bit);
      }
#if defined(TI_VULKAN_DYNAMIC)
      else if (path[k]->type == SNodeType::dynamic) {
        // active &= (i_path[k] < atomicLoad(length))
        auto len_byte_off = ir_->add(
            addr,
            ir_->uint_immediate_number(
                u32_t,
                (uint32_t)desc_k.dynamic_length_offset_in_container));
        auto len_word_idx = ir_->make_value(
            spv::OpShiftRightLogical, u32_t, len_byte_off,
            ir_->uint_immediate_number(u32_t, 2));
        auto len_ptr =
            ir_->struct_array_access(u32_t, level_buffers[(size_t)k], len_word_idx);
        auto len_val = ir_->make_value(
            spv::OpAtomicLoad, u32_t, len_ptr,
            /*scope=*/ir_->const_i32_one_,
            /*semantics=*/ir_->const_i32_zero_);
        auto in_range = ir_->make_value(spv::OpULessThan, ir_->bool_type(),
                                        i_path[k], len_val);
        auto bit = ir_->make_value(
            spv::OpSelect, u32_t, in_range,
            ir_->uint_immediate_number(u32_t, 1),
            ir_->uint_immediate_number(u32_t, 0));
        active = ir_->make_value(spv::OpBitwiseAnd, u32_t, active, bit);
      }
#endif

      if (k < n - 1) {
        auto cell_stride_v =
            ir_->uint_immediate_number(u32_t, (uint32_t)desc_k.cell_stride);
        if (path_k_is_pointer) {
          // Pool jump: addr = pool_offset + effective_slot * cell_stride.
          // The pointer SNode has a single child whose mem_offset is 0 in
          // the pool entry, so we add path[k+1].mem_offset for symmetry
          // with the dense path (in practice this is 0 today).
          // 路线 B B-1：通过 contract 读取 pool 偏移。
          const auto &contract_k =
              compiled_structs_[root_id].pointer_contracts.at(path[k]->id);
          auto pool_offset_v = ir_->uint_immediate_number(
              u32_t, contract_k.pool_data_offset);
          auto step = ir_->mul(pointer_effective_slot, cell_stride_v);
          addr = ir_->add(pool_offset_v, step);
          addr = ir_->add(
              addr,
              ir_->uint_immediate_number(
                  u32_t, (uint32_t)snode_descs.at(path[k + 1]->id)
                             .mem_offset_in_parent_cell));
          continue;
        }
        auto step = ir_->mul(i_path[k], cell_stride_v);
        addr = ir_->add(addr, step);
        addr = ir_->add(
            addr,
            ir_->uint_immediate_number(
                u32_t, (uint32_t)snode_descs.at(path[k + 1]->id)
                           .mem_offset_in_parent_cell));
      }
    }

    auto active_cond = ir_->make_value(spv::OpUGreaterThan, ir_->bool_type(),
                                       active,
                                       ir_->uint_immediate_number(u32_t, 0));
    spirv::Label active_label = ir_->new_label();
    spirv::Label after_active = ir_->new_label();
    ir_->make_inst(spv::OpSelectionMerge, after_active,
                   spv::SelectionControlMaskNone);
    ir_->make_inst(spv::OpBranchConditional, active_cond, active_label,
                   after_active);
    ir_->start_label(active_label);

    // Compute the *globally* flat axis index expected by the loop body's
    // LoopIndexStmt decoder. The decoder uses leaf->num_elements_from_root
    // per axis, so we must emit:
    //   flat_global = sum_a global_axis[a] * (prod_{a'>a} num_from_root[a'])
    // where global_axis[a] = sum_k local_axis_a_k * (prod_{k'>k} shape_a_k')
    // and local_axis_a_k decomposes i_path[k] within level k's container
    // using that level's own extractors.
    //
    // Important: an axis is "active" for the listgen output iff *any* level
    // along the path activates it. The leaf SNode's `extractors[a].active`
    // alone is insufficient for mixed-axis nested layouts (e.g.
    // `bitmasked(ij, ...).bitmasked(i, ...)` where the leaf does not
    // activate `j` itself but `j` still carries a non-trivial shape from
    // the ancestor). Use the path-wide union.
    bool path_active[taichi_max_num_indices] = {};
    for (int kk = 0; kk < n; ++kk) {
      for (int a = 0; a < taichi_max_num_indices; ++a) {
        if (path[kk]->extractors[a].active) path_active[a] = true;
      }
    }

    std::vector<spirv::Value> global_axis(taichi_max_num_indices);
    for (int a = 0; a < taichi_max_num_indices; ++a) {
      global_axis[a] = ir_->uint_immediate_number(u32_t, 0);
    }
    for (int k = 0; k < n; ++k) {
      for (int a = 0; a < taichi_max_num_indices; ++a) {
        if (!path[k]->extractors[a].active) continue;
        int local_acc = path[k]->extractors[a].acc_shape;
        int local_shape = path[k]->extractors[a].shape;
        spirv::Value local = i_path[k];
        if (local_acc > 1) {
          local = ir_->make_value(
              spv::OpUDiv, u32_t, local,
              ir_->uint_immediate_number(u32_t, local_acc));
        }
        if (local_shape > 1) {
          local = ir_->make_value(
              spv::OpUMod, u32_t, local,
              ir_->uint_immediate_number(u32_t, local_shape));
        }
        int desc_prod = 1;
        for (int kk = k + 1; kk < n; ++kk) {
          if (path[kk]->extractors[a].active)
            desc_prod *= path[kk]->extractors[a].shape;
        }
        if (desc_prod > 1) {
          local = ir_->mul(
              local, ir_->uint_immediate_number(u32_t, desc_prod));
        }
        global_axis[a] = ir_->add(global_axis[a], local);
      }
    }
    spirv::Value flat_global = ir_->uint_immediate_number(u32_t, 0);
    for (int a = 0; a < taichi_max_num_indices; ++a) {
      if (!path_active[a]) continue;
      int axis_acc = 1;
      for (int aa = a + 1; aa < taichi_max_num_indices; ++aa) {
        if (path_active[aa])
          axis_acc *=
              (int)child_sn->extractors[aa].num_elements_from_root;
      }
      spirv::Value contrib = global_axis[a];
      if (axis_acc > 1) {
        contrib = ir_->mul(
            contrib, ir_->uint_immediate_number(u32_t, axis_acc));
      }
      flat_global = ir_->add(flat_global, contrib);
    }

    // slot = atomicAdd(listgen_buffer[0], 1); listgen_buffer[1 + slot] = flat_global
    auto count_ptr = ir_->struct_array_access(u32_t, listgen_buffer,
                                              ir_->const_i32_zero_);
    auto slot = ir_->make_value(spv::OpAtomicIAdd, u32_t, count_ptr,
                                /*scope=*/ir_->const_i32_one_,
                                /*semantics=*/ir_->const_i32_zero_,
                                ir_->uint_immediate_number(u32_t, 1));
    auto idx_in_buffer =
        ir_->add(slot, ir_->uint_immediate_number(u32_t, 1));
    auto idx_ptr =
        ir_->struct_array_access(u32_t, listgen_buffer, idx_in_buffer);
    ir_->store_variable(idx_ptr, flat_global);

    ir_->make_inst(spv::OpBranch, after_active);
    ir_->start_label(after_active);
    ir_->make_inst(spv::OpBranch, after_bounds);
    ir_->start_label(after_bounds);

    ir_->make_inst(spv::OpReturn);
    ir_->make_inst(spv::OpFunctionEnd);

    task_attribs_.buffer_binds = get_buffer_binds();
    task_attribs_.texture_binds = get_texture_binds();
  }

  // -----------------------------------------------------------------
  // G9.2 / G9.3a (2026-04-30): bit-pointer load / store helpers.
  //
  // A "bit-pointer" stmt is one whose ret_type is
  // PointerType{is_bit_pointer=true}.  Today (G9.2) this is produced
  // exclusively by SNodeLookupStmt for SNodeType::quant_array and
  // forwarded by GetChStmt for the quant `place` child.  The byte
  // address of the containing physical word lives in the standard
  // SSA register keyed by `stmt->raw_name()`; the bit offset within
  // that word is tracked separately in `bit_ptr_bit_offsets_`.
  //
  // Only QuantIntType / QuantFixedType pointees are supported in this
  // iteration.  QuantFloatType requires a BitStructType parent (shared
  // exponent), which lives behind G9.3b (BitStructStoreStmt SPIR-V
  // visitor) -- not implemented here.
  // -----------------------------------------------------------------

  // Resolve the GetChStmt that produced `bit_ptr_stmt` so we can recover
  // the parent SNode's physical_type.
  const GetChStmt *find_bit_ptr_get_ch(const Stmt *bit_ptr_stmt) {
    auto *get_ch = bit_ptr_stmt->cast<GetChStmt>();
    TI_ASSERT_INFO(get_ch != nullptr,
                   "bit-pointer source must be a GetChStmt (got {})",
                   bit_ptr_stmt->name());
    return get_ch;
  }

  // Look up the bit offset SSA value associated with a bit-pointer stmt.
  // Walks back through GetChStmt forwarding to the SNodeLookupStmt that
  // produced the offset.
  spirv::Value query_bit_offset(const Stmt *bit_ptr_stmt) {
    const Stmt *cur = bit_ptr_stmt;
    while (cur != nullptr) {
      auto it = bit_ptr_bit_offsets_.find(cur);
      if (it != bit_ptr_bit_offsets_.end()) {
        return it->second;
      }
      // Walk through forwarding GetChStmt -> input_ptr (parent
      // SNodeLookupStmt or another GetChStmt).
      if (auto *gc = cur->cast<GetChStmt>()) {
        cur = gc->input_ptr;
        continue;
      }
      break;
    }
    TI_ERROR("query_bit_offset: no bit offset registered for {}",
             bit_ptr_stmt->name());
    return ir_->const_i32_zero_;
  }

  void emit_bit_pointer_load(GlobalLoadStmt *stmt) {
    TI_ERROR_IF(
        !is_extension_supported(Arch::vulkan, Extension::quant),
        "quant load on Vulkan SPIR-V backend requires the experimental "
        "gate (ti.init(vulkan_quant_experimental=True) or env var "
        "TI_VULKAN_QUANT=1).");
    auto *get_ch = find_bit_ptr_get_ch(stmt->src);
    DataType physical_type;
    if (get_ch->input_snode->type == SNodeType::quant_array) {
      physical_type =
          get_ch->input_snode->dt->as<QuantArrayType>()->get_physical_type();
    } else if (get_ch->input_snode->type == SNodeType::bit_struct) {
      physical_type =
          get_ch->input_snode->dt->as<BitStructType>()->get_physical_type();
    } else {
      TI_NOT_IMPLEMENTED;
    }
    SType phys_stype = ir_->get_primitive_type(physical_type);
    SType phys_uint_stype =
        ir_->get_primitive_type(ir_->get_taichi_uint_type(physical_type));

    // Load the physical word as the unsigned variant (so OpBitFieldUExtract
    // / OpBitFieldSExtract have a clean unsigned base; for f32/f64 storage
    // the at_buffer + load_variable pattern is identical to the existing
    // load_buffer path with `ti_buffer_type` = physical uint).
    auto buf_ptr = at_buffer(stmt->src, ir_->get_taichi_uint_type(physical_type));
    spirv::Value phys_uval = ir_->load_variable(buf_ptr, phys_uint_stype);

    spirv::Value bit_off_u32 = query_bit_offset(stmt->src);
    if (bit_off_u32.stype.id != phys_uint_stype.id) {
      bit_off_u32 = ir_->cast(phys_uint_stype, bit_off_u32);
    }

    auto *pt = stmt->src->ret_type->as<PointerType>();
    Type *pointee = pt->get_pointee_type();

    auto extract_quant_int = [&](QuantIntType *qit) -> spirv::Value {
      const int num_bits = qit->get_num_bits();
      const bool is_signed = qit->get_is_signed();
      spirv::Value count = ir_->uint_immediate_number(
          phys_uint_stype, static_cast<uint32_t>(num_bits));
      spv::Op op =
          is_signed ? spv::OpBitFieldSExtract : spv::OpBitFieldUExtract;
      // Result type: physical uint (signed extract still yields the
      // physical-width result in two's-complement).
      spirv::Value extracted =
          ir_->make_value(op, phys_uint_stype, phys_uval, bit_off_u32, count);
      // Cast to compute_type (typically i32).
      DataType compute_type = qit->get_compute_type();
      SType compute_stype = ir_->get_primitive_type(compute_type);
      if (compute_stype.id == phys_uint_stype.id) {
        // Reinterpret unsigned -> signed if needed (no shape change).
        if (is_signed) {
          // Sign-extract returned the value in physical-width signed
          // semantics; bitcast to i32 if requested.
          if (compute_type->is_primitive(PrimitiveTypeID::i32) &&
              physical_type->is_primitive(PrimitiveTypeID::u32)) {
            return ir_->make_value(spv::OpBitcast, compute_stype, extracted);
          }
        }
        return extracted;
      }
      return ir_->cast(compute_stype, extracted);
    };

    spirv::Value result;
    if (auto *qit = pointee->cast<QuantIntType>()) {
      result = extract_quant_int(qit);
    } else if (auto *qfxt = pointee->cast<QuantFixedType>()) {
      auto *digits_qit = qfxt->get_digits_type()->as<QuantIntType>();
      spirv::Value digits = extract_quant_int(digits_qit);
      // reconstruct: float(digits) * scale
      DataType compute_type = qfxt->get_compute_type();
      SType compute_stype = ir_->get_primitive_type(compute_type);
      spirv::Value digits_f = ir_->cast(compute_stype, digits);
      // Build the scale as a same-precision float constant.
      spirv::Value scale_v;
      if (compute_type->is_primitive(PrimitiveTypeID::f64)) {
        scale_v = ir_->float_immediate_number(compute_stype, qfxt->get_scale());
      } else {
        scale_v = ir_->float_immediate_number(
            compute_stype, static_cast<float>(qfxt->get_scale()));
      }
      result = ir_->make_value(spv::OpFMul, compute_stype, digits_f, scale_v);
    } else {
      TI_NOT_IMPLEMENTED;  // QuantFloatType requires bit_struct (G9.3b).
    }
    ir_->register_value(stmt->raw_name(), result);
  }

  void emit_bit_pointer_store(GlobalStoreStmt *stmt) {
    TI_ERROR_IF(
        !is_extension_supported(Arch::vulkan, Extension::quant),
        "quant store on Vulkan SPIR-V backend requires the experimental "
        "gate (ti.init(vulkan_quant_experimental=True) or env var "
        "TI_VULKAN_QUANT=1).");
    auto *get_ch = find_bit_ptr_get_ch(stmt->dest);
    DataType physical_type;
    if (get_ch->input_snode->type == SNodeType::quant_array) {
      physical_type =
          get_ch->input_snode->dt->as<QuantArrayType>()->get_physical_type();
    } else if (get_ch->input_snode->type == SNodeType::bit_struct) {
      physical_type =
          get_ch->input_snode->dt->as<BitStructType>()->get_physical_type();
    } else {
      TI_NOT_IMPLEMENTED;
    }
    SType phys_uint_stype =
        ir_->get_primitive_type(ir_->get_taichi_uint_type(physical_type));

    auto *pt = stmt->dest->ret_type->as<PointerType>();
    Type *pointee = pt->get_pointee_type();
    QuantIntType *qit = nullptr;
    QuantFixedType *qfxt = nullptr;
    int num_bits = 0;
    bool is_signed = false;
    if ((qit = pointee->cast<QuantIntType>())) {
      num_bits = qit->get_num_bits();
      is_signed = qit->get_is_signed();
    } else if ((qfxt = pointee->cast<QuantFixedType>())) {
      auto *digits_qit = qfxt->get_digits_type()->as<QuantIntType>();
      num_bits = digits_qit->get_num_bits();
      is_signed = digits_qit->get_is_signed();
    } else {
      TI_NOT_IMPLEMENTED;
    }

    // Convert the input value to a `num_bits`-wide digits integer in the
    // physical-uint type.
    spirv::Value src_val = ir_->query_value(stmt->val->raw_name());
    spirv::Value digits_uval;
    if (qfxt != nullptr) {
      // digits = round(src / scale)  (LLVM uses fp-to-int with rounding-
      // to-nearest-even via `FPToSI/UI`; SPIR-V does the same).
      DataType compute_type = qfxt->get_compute_type();
      SType compute_stype = ir_->get_primitive_type(compute_type);
      spirv::Value scale_v;
      if (compute_type->is_primitive(PrimitiveTypeID::f64)) {
        scale_v = ir_->float_immediate_number(
            compute_stype, 1.0 / qfxt->get_scale());
      } else {
        scale_v = ir_->float_immediate_number(
            compute_stype, static_cast<float>(1.0 / qfxt->get_scale()));
      }
      // src / scale  ==  src * (1/scale).  Use FMul for precision.
      spirv::Value scaled =
          ir_->make_value(spv::OpFMul, compute_stype, src_val, scale_v);
      // Round to nearest via GLSL std450 RoundEven (== inst id 2).  This
      // matches LLVM's FPToSI behavior under default IEEE rounding (RNE)
      // for all values that fit in num_bits.
      // GLSLstd450RoundEven == 2.
      spirv::Value rounded =
          ir_->call_glsl450(compute_stype, /*GLSLstd450RoundEven=*/2, scaled);
      // FPToSI / FPToUI to the physical-width integer.  Use SIGNED
      // conversion when the quant int is signed (negative floats with
      // OpConvertFToU are undefined in SPIR-V and on most drivers
      // produce 0, which would silently zero out every negative cell).
      // We then bitcast through the physical-width signed integer type
      // before reaching `phys_uint_stype`; ir_->cast handles
      // signed<->unsigned same-width as a clean OpBitcast.
      if (is_signed) {
        DataType phys_signed_dt;
        switch (data_type_bits(physical_type)) {
          case 16:
            phys_signed_dt = PrimitiveType::i16;
            break;
          case 32:
            phys_signed_dt = PrimitiveType::i32;
            break;
          case 64:
            phys_signed_dt = PrimitiveType::i64;
            break;
          default:
            TI_NOT_IMPLEMENTED;
        }
        SType phys_signed_stype = ir_->get_primitive_type(phys_signed_dt);
        spirv::Value digits_signed =
            ir_->make_value(spv::OpConvertFToS, phys_signed_stype, rounded);
        digits_uval = ir_->cast(phys_uint_stype, digits_signed);
      } else {
        digits_uval = ir_->cast(phys_uint_stype, rounded);
      }
    } else {
      // QuantInt: just cast the input integer to physical uint width.
      digits_uval = ir_->cast(phys_uint_stype, src_val);
    }

    // Build the mask: ((1u << num_bits) - 1u) << bit_off
    spirv::Value bit_off_u32 = query_bit_offset(stmt->dest);
    if (bit_off_u32.stype.id != phys_uint_stype.id) {
      bit_off_u32 = ir_->cast(phys_uint_stype, bit_off_u32);
    }
    spirv::Value field_mask;
    if (num_bits == static_cast<int>(ir_->get_primitive_type_size(physical_type) * 8)) {
      // Full word: mask = ~0
      field_mask = ir_->make_value(
          spv::OpNot, phys_uint_stype,
          ir_->uint_immediate_number(phys_uint_stype, 0u));
    } else {
      uint64_t one = 1ull;
      uint64_t mask_lo = (one << static_cast<uint64_t>(num_bits)) - one;
      field_mask =
          ir_->uint_immediate_number(phys_uint_stype, mask_lo);
    }
    // Truncate digits to num_bits and shift into place.
    spirv::Value digits_truncated =
        ir_->make_value(spv::OpBitwiseAnd, phys_uint_stype, digits_uval,
                        field_mask);
    spirv::Value digits_shifted =
        ir_->make_value(spv::OpShiftLeftLogical, phys_uint_stype,
                        digits_truncated, bit_off_u32);
    spirv::Value mask_shifted =
        ir_->make_value(spv::OpShiftLeftLogical, phys_uint_stype, field_mask,
                        bit_off_u32);
    spirv::Value mask_inv =
        ir_->make_value(spv::OpNot, phys_uint_stype, mask_shifted);
    (void)is_signed;  // truncation handles sign for both branches.

    // Read-modify-write on the physical word.  This is non-atomic: races
    // between threads writing different fields of the SAME physical word
    // produce undefined results.  Mirrors the LLVM CPU/CUDA non-atomic
    // store path for quant_array (atomic_add lives in G9.4).
    DataType phys_uint_dt = ir_->get_taichi_uint_type(physical_type);
    auto buf_ptr = at_buffer(stmt->dest, phys_uint_dt);
    spirv::Value old_word = ir_->load_variable(buf_ptr, phys_uint_stype);
    spirv::Value cleared =
        ir_->make_value(spv::OpBitwiseAnd, phys_uint_stype, old_word, mask_inv);
    spirv::Value new_word =
        ir_->make_value(spv::OpBitwiseOr, phys_uint_stype, cleared,
                        digits_shifted);
    ir_->store_variable(buf_ptr, new_word);
  }

  // -----------------------------------------------------------------
  // G9.4 (2026-05-02): atomic_add on a bit-pointer (quant_array /
  // bit_struct member).  Mirrors the LLVM runtime helper
  //   atomic_add_partial_bits_b##N(ptr, off, bits, val):
  //     do {
  //       old = *ptr;
  //       new = (old & ~mask) | ((old + (val << off)) & mask);
  //     } while (!atomicCAS(ptr, old, new));
  // i.e. the addition is performed across the FULL word (so a carry from
  // bit `off + bits - 1` corrupts the next field), exactly as on
  // cpu/cuda — keeping three-backend byte equivalence.
  //
  // Limitations:
  //   * Only AtomicOpType::add is supported (matches the LLVM
  //     `quant_type_atomic` early-return in codegen_llvm.cpp:1402).
  //   * Physical word must be 32 or 64 bits (Vulkan core atomics).
  //   * QuantFloat (shared exponent) is rejected at the field-type
  //     dispatch below; G9.3c will lift that restriction.
  //   * The returned `stmt->raw_name()` value is registered as the input
  //     `stmt->val` (not the old field).  taichi rarely consumes the
  //     return value of atomic_add on a quant field; correctness of the
  //     accumulation is unaffected.
  // -----------------------------------------------------------------
  void emit_bit_pointer_atomic_add(AtomicOpStmt *stmt) {
    TI_ERROR_IF(
        !is_extension_supported(Arch::vulkan, Extension::quant),
        "atomic_add on a quant bit pointer requires the experimental gate "
        "(ti.init(vulkan_quant_experimental=True) or env var "
        "TI_VULKAN_QUANT=1).");
    TI_ERROR_IF(stmt->op_type != AtomicOpType::add,
                "Only atomic_add is supported on quant bit pointers on the "
                "Vulkan SPIR-V backend (got op={}).",
                atomic_op_type_name(stmt->op_type));
    auto *get_ch = find_bit_ptr_get_ch(stmt->dest);
    DataType physical_type;
    if (get_ch->input_snode->type == SNodeType::quant_array) {
      physical_type =
          get_ch->input_snode->dt->as<QuantArrayType>()->get_physical_type();
    } else if (get_ch->input_snode->type == SNodeType::bit_struct) {
      physical_type =
          get_ch->input_snode->dt->as<BitStructType>()->get_physical_type();
    } else {
      TI_NOT_IMPLEMENTED;
    }
    const int phys_bits = data_type_bits(physical_type);
    TI_ERROR_IF(phys_bits != 32 && phys_bits != 64,
                "atomic_add on quant bit pointer requires a 32-bit or 64-bit "
                "physical word (got {} bits).",
                phys_bits);
    SType phys_uint_stype =
        ir_->get_primitive_type(ir_->get_taichi_uint_type(physical_type));

    auto *pt = stmt->dest->ret_type->as<PointerType>();
    Type *pointee = pt->get_pointee_type();
    QuantIntType *qit = nullptr;
    QuantFixedType *qfxt = nullptr;
    int num_bits = 0;
    bool is_signed = false;
    if ((qit = pointee->cast<QuantIntType>())) {
      num_bits = qit->get_num_bits();
      is_signed = qit->get_is_signed();
    } else if ((qfxt = pointee->cast<QuantFixedType>())) {
      auto *digits_qit = qfxt->get_digits_type()->as<QuantIntType>();
      num_bits = digits_qit->get_num_bits();
      is_signed = digits_qit->get_is_signed();
    } else {
      TI_NOT_IMPLEMENTED;  // G9.3c: shared-exponent QuantFloat
    }

    // Convert the user-supplied delta to a `num_bits`-wide digits integer
    // in physical-uint form (same scaling pipeline as emit_bit_pointer_store).
    spirv::Value src_val = ir_->query_value(stmt->val->raw_name());
    spirv::Value digits_uval;
    if (qfxt != nullptr) {
      DataType compute_type = qfxt->get_compute_type();
      SType compute_stype = ir_->get_primitive_type(compute_type);
      spirv::Value scale_v;
      if (compute_type->is_primitive(PrimitiveTypeID::f64)) {
        scale_v = ir_->float_immediate_number(compute_stype,
                                              1.0 / qfxt->get_scale());
      } else {
        scale_v = ir_->float_immediate_number(
            compute_stype, static_cast<float>(1.0 / qfxt->get_scale()));
      }
      spirv::Value scaled =
          ir_->make_value(spv::OpFMul, compute_stype, src_val, scale_v);
      spirv::Value rounded =
          ir_->call_glsl450(compute_stype, /*GLSLstd450RoundEven=*/2, scaled);
      if (is_signed) {
        DataType phys_signed_dt;
        switch (phys_bits) {
          case 32: phys_signed_dt = PrimitiveType::i32; break;
          case 64: phys_signed_dt = PrimitiveType::i64; break;
          default: TI_NOT_IMPLEMENTED;
        }
        SType phys_signed_stype = ir_->get_primitive_type(phys_signed_dt);
        spirv::Value digits_signed =
            ir_->make_value(spv::OpConvertFToS, phys_signed_stype, rounded);
        digits_uval = ir_->cast(phys_uint_stype, digits_signed);
      } else {
        digits_uval = ir_->cast(phys_uint_stype, rounded);
      }
    } else {
      digits_uval = ir_->cast(phys_uint_stype, src_val);
    }

    // Build masks/shifts (truncate digits to num_bits, shift to bit_off).
    spirv::Value bit_off_u32 = query_bit_offset(stmt->dest);
    if (bit_off_u32.stype.id != phys_uint_stype.id) {
      bit_off_u32 = ir_->cast(phys_uint_stype, bit_off_u32);
    }
    spirv::Value field_mask;
    if (num_bits == phys_bits) {
      field_mask = ir_->make_value(
          spv::OpNot, phys_uint_stype,
          ir_->uint_immediate_number(phys_uint_stype, 0u));
    } else {
      uint64_t mask_lo =
          (uint64_t(1) << static_cast<uint64_t>(num_bits)) - 1;
      field_mask = ir_->uint_immediate_number(phys_uint_stype, mask_lo);
    }
    spirv::Value digits_truncated = ir_->make_value(
        spv::OpBitwiseAnd, phys_uint_stype, digits_uval, field_mask);
    spirv::Value digits_shifted =
        ir_->make_value(spv::OpShiftLeftLogical, phys_uint_stype,
                        digits_truncated, bit_off_u32);
    spirv::Value mask_shifted = ir_->make_value(
        spv::OpShiftLeftLogical, phys_uint_stype, field_mask, bit_off_u32);
    spirv::Value mask_inv =
        ir_->make_value(spv::OpNot, phys_uint_stype, mask_shifted);
    (void)is_signed;

    // CAS-loop on the physical word:
    //   do {
    //     old = *buf_ptr;
    //     sum = old + digits_shifted;
    //     new = (old & ~mask_shifted) | (sum & mask_shifted);
    //   } while (atomicCAS(buf_ptr, old, new) != old);
    DataType phys_uint_dt = ir_->get_taichi_uint_type(physical_type);
    auto buf_ptr = at_buffer(stmt->dest, phys_uint_dt);

    spirv::Label head_lbl = ir_->new_label();
    spirv::Label body_lbl = ir_->new_label();
    spirv::Label win_lbl = ir_->new_label();
    spirv::Label lose_lbl = ir_->new_label();
    spirv::Label continue_lbl = ir_->new_label();
    spirv::Label exit_lbl = ir_->new_label();

    ir_->make_inst(spv::OpBranch, head_lbl);
    ir_->start_label(head_lbl);
    ir_->make_inst(spv::OpLoopMerge, exit_lbl, continue_lbl,
                   spv::LoopControlMaskNone);
    ir_->make_inst(spv::OpBranch, body_lbl);
    ir_->make_inst(spv::OpLabel, body_lbl);

    spirv::Value old_word =
        ir_->make_value(spv::OpAtomicLoad, phys_uint_stype, buf_ptr,
                        /*scope=*/ir_->const_i32_one_,
                        /*semantics=*/ir_->const_i32_zero_);
    spirv::Value sum =
        ir_->make_value(spv::OpIAdd, phys_uint_stype, old_word, digits_shifted);
    spirv::Value sum_field = ir_->make_value(
        spv::OpBitwiseAnd, phys_uint_stype, sum, mask_shifted);
    spirv::Value cleared = ir_->make_value(
        spv::OpBitwiseAnd, phys_uint_stype, old_word, mask_inv);
    spirv::Value new_word = ir_->make_value(
        spv::OpBitwiseOr, phys_uint_stype, cleared, sum_field);
    spirv::Value cas = ir_->make_value(
        spv::OpAtomicCompareExchange, phys_uint_stype, buf_ptr,
        /*scope=*/ir_->const_i32_one_,
        /*semantics_eq=*/ir_->const_i32_zero_,
        /*semantics_uneq=*/ir_->const_i32_zero_, new_word, old_word);
    spirv::Value won =
        ir_->make_value(spv::OpIEqual, ir_->bool_type(), cas, old_word);
    ir_->make_inst(spv::OpSelectionMerge, lose_lbl,
                   spv::SelectionControlMaskNone);
    ir_->make_inst(spv::OpBranchConditional, won, win_lbl, lose_lbl);
    ir_->make_inst(spv::OpLabel, win_lbl);
    ir_->make_inst(spv::OpBranch, exit_lbl);
    ir_->make_inst(spv::OpLabel, lose_lbl);
    ir_->make_inst(spv::OpBranch, continue_lbl);
    ir_->make_inst(spv::OpLabel, continue_lbl);
    ir_->make_inst(spv::OpBranch, head_lbl);

    ir_->start_label(exit_lbl);

    // Register the AtomicOpStmt's SSA value.  taichi semantics return
    // the OLD value, but for a quantized field that requires
    // dequantization on every code path; since user code on quant fields
    // overwhelmingly ignores the return, we register the input delta to
    // satisfy any downstream consumers without paying the conversion
    // cost.  Documented in the experimental quant guide.
    ir_->register_value(stmt->raw_name(), src_val);
  }


  // -----------------------------------------------------------------
  // G9.3b (2026-05-01): BitStructStoreStmt visitor.
  //
  // BitStructStoreStmt is produced by transforms/optimize_bit_struct_stores
  // .cpp from grouped GlobalStoreStmt's whose dest is a GetChStmt(bit_struct).
  // It carries (ptr, ch_ids, values, is_atomic) where `ptr` is the pointer
  // to the bit_struct's physical word and `ch_ids[i]` indexes into
  // BitStructType.  This visitor packs all (ch_id, value) pairs into one
  // RMW on the physical word.
  //
  // Limitations (intentional, in this iteration):
  //   * QuantFloat with shared exponent: TI_NOT_IMPLEMENTED (G9.3c).
  //   * is_atomic == true after the IR pass means atomic-demotion did
  //     NOT cover all members (e.g., user only wrote 1 of 3 fields).
  //     Vulkan SPIR-V atomic CAS-loop is G9.4 -- we still execute a
  //     non-atomic RMW here so single-threaded correctness holds, but
  //     concurrent writers to the SAME word produce UB.  This matches
  //     the existing G9.3a quant_array store contract.
  // -----------------------------------------------------------------
  void visit(BitStructStoreStmt *stmt) override {
    TI_ERROR_IF(
        !is_extension_supported(Arch::vulkan, Extension::quant),
        "BitStructStoreStmt on Vulkan SPIR-V backend requires the "
        "experimental gate (ti.init(vulkan_quant_experimental=True) or "
        "env var TI_VULKAN_QUANT=1).");
    auto *bst = stmt->get_bit_struct();
    DataType physical_type = bst->get_physical_type();
    SType phys_uint_stype =
        ir_->get_primitive_type(ir_->get_taichi_uint_type(physical_type));
    const int phys_bits = data_type_bits(physical_type);

    // Reject shared-exponent QuantFloat layouts (any member that owns a
    // shared exponent or has an exponent_users list).  The MPM/baseline
    // BitpackedFields used by g9_quant_baseline.py never trips this.
    for (int i = 0; i < bst->get_num_members(); i++) {
      auto *mtype = bst->get_member_type(i);
      if (mtype->is<QuantFloatType>()) {
        TI_NOT_IMPLEMENTED  // G9.3c: shared-exponent quant_float on Vulkan
      }
    }

    spirv::Value packed =
        ir_->uint_immediate_number(phys_uint_stype, uint64_t(0));
    spirv::Value total_mask =
        ir_->uint_immediate_number(phys_uint_stype, uint64_t(0));

    for (int i = 0; i < (int)stmt->ch_ids.size(); i++) {
      const int ch_id = stmt->ch_ids[i];
      Stmt *val_stmt = stmt->values[i];
      Type *member_type = bst->get_member_type(ch_id);
      const int bit_off = bst->get_member_bit_offset(ch_id);

      QuantIntType *qit = nullptr;
      QuantFixedType *qfxt = nullptr;
      int num_bits = 0;
      bool is_signed = false;
      if ((qit = member_type->cast<QuantIntType>())) {
        num_bits = qit->get_num_bits();
        is_signed = qit->get_is_signed();
      } else if ((qfxt = member_type->cast<QuantFixedType>())) {
        auto *digits_qit = qfxt->get_digits_type()->as<QuantIntType>();
        num_bits = digits_qit->get_num_bits();
        is_signed = digits_qit->get_is_signed();
      } else {
        TI_NOT_IMPLEMENTED;
      }

      spirv::Value src_val = ir_->query_value(val_stmt->raw_name());
      spirv::Value digits_uval;
      if (qfxt != nullptr) {
        DataType compute_type = qfxt->get_compute_type();
        SType compute_stype = ir_->get_primitive_type(compute_type);
        spirv::Value scale_v;
        if (compute_type->is_primitive(PrimitiveTypeID::f64)) {
          scale_v = ir_->float_immediate_number(
              compute_stype, 1.0 / qfxt->get_scale());
        } else {
          scale_v = ir_->float_immediate_number(
              compute_stype, static_cast<float>(1.0 / qfxt->get_scale()));
        }
        spirv::Value scaled =
            ir_->make_value(spv::OpFMul, compute_stype, src_val, scale_v);
        spirv::Value rounded =
            ir_->call_glsl450(compute_stype, /*GLSLstd450RoundEven=*/2, scaled);
        if (is_signed) {
          DataType phys_signed_dt;
          switch (phys_bits) {
            case 16: phys_signed_dt = PrimitiveType::i16; break;
            case 32: phys_signed_dt = PrimitiveType::i32; break;
            case 64: phys_signed_dt = PrimitiveType::i64; break;
            default: TI_NOT_IMPLEMENTED;
          }
          SType phys_signed_stype = ir_->get_primitive_type(phys_signed_dt);
          spirv::Value digits_signed =
              ir_->make_value(spv::OpConvertFToS, phys_signed_stype, rounded);
          digits_uval = ir_->cast(phys_uint_stype, digits_signed);
        } else {
          digits_uval = ir_->cast(phys_uint_stype, rounded);
        }
      } else {
        digits_uval = ir_->cast(phys_uint_stype, src_val);
      }

      spirv::Value field_mask;
      if (num_bits == phys_bits) {
        field_mask = ir_->make_value(
            spv::OpNot, phys_uint_stype,
            ir_->uint_immediate_number(phys_uint_stype, 0u));
      } else {
        uint64_t mask_lo =
            (uint64_t(1) << static_cast<uint64_t>(num_bits)) - 1;
        field_mask = ir_->uint_immediate_number(phys_uint_stype, mask_lo);
      }
      spirv::Value bit_off_v = ir_->uint_immediate_number(
          phys_uint_stype, static_cast<uint64_t>(bit_off));

      spirv::Value digits_truncated = ir_->make_value(
          spv::OpBitwiseAnd, phys_uint_stype, digits_uval, field_mask);
      spirv::Value digits_shifted = ir_->make_value(
          spv::OpShiftLeftLogical, phys_uint_stype, digits_truncated,
          bit_off_v);
      spirv::Value mask_shifted = ir_->make_value(
          spv::OpShiftLeftLogical, phys_uint_stype, field_mask, bit_off_v);

      packed = ir_->make_value(spv::OpBitwiseOr, phys_uint_stype, packed,
                               digits_shifted);
      total_mask = ir_->make_value(spv::OpBitwiseOr, phys_uint_stype,
                                   total_mask, mask_shifted);
    }

    DataType phys_uint_dt = ir_->get_taichi_uint_type(physical_type);
    auto buf_ptr = at_buffer(stmt->ptr, phys_uint_dt);

    // If total_mask covers the full physical word OR every non-exponent
    // member is being written (which is what optimize_bit_struct_stores
    // demotes to non-atomic when quant_opt_atomic_demotion is on), we
    // could emit a plain store.  We always emit RMW here; for the
    // all-fields case this is at most one extra load+and+or per word.
    spirv::Value mask_inv =
        ir_->make_value(spv::OpNot, phys_uint_stype, total_mask);
    if (!stmt->is_atomic) {
      // Non-atomic RMW (single thread per word).
      spirv::Value old_word = ir_->load_variable(buf_ptr, phys_uint_stype);
      spirv::Value cleared = ir_->make_value(
          spv::OpBitwiseAnd, phys_uint_stype, old_word, mask_inv);
      spirv::Value new_word = ir_->make_value(
          spv::OpBitwiseOr, phys_uint_stype, cleared, packed);
      ir_->store_variable(buf_ptr, new_word);
    } else {
      // G9.4: atomic RMW via CAS-loop.  Reached when
      // optimize_bit_struct_stores left is_atomic=true (e.g.,
      // quant_opt_atomic_demotion=false, or stores not uniquely accessed).
      // Mirrors the LLVM atomic_set_mask path in runtime.cpp.
      const int phys_bits_local = data_type_bits(physical_type);
      TI_ERROR_IF(phys_bits_local != 32 && phys_bits_local != 64,
                  "Atomic BitStructStoreStmt requires 32-bit or 64-bit "
                  "physical word on Vulkan (got {} bits).",
                  phys_bits_local);
      spirv::Label head_lbl = ir_->new_label();
      spirv::Label body_lbl = ir_->new_label();
      spirv::Label win_lbl = ir_->new_label();
      spirv::Label lose_lbl = ir_->new_label();
      spirv::Label continue_lbl = ir_->new_label();
      spirv::Label exit_lbl = ir_->new_label();

      ir_->make_inst(spv::OpBranch, head_lbl);
      ir_->start_label(head_lbl);
      ir_->make_inst(spv::OpLoopMerge, exit_lbl, continue_lbl,
                     spv::LoopControlMaskNone);
      ir_->make_inst(spv::OpBranch, body_lbl);
      ir_->make_inst(spv::OpLabel, body_lbl);

      spirv::Value old_word =
          ir_->make_value(spv::OpAtomicLoad, phys_uint_stype, buf_ptr,
                          /*scope=*/ir_->const_i32_one_,
                          /*semantics=*/ir_->const_i32_zero_);
      spirv::Value cleared = ir_->make_value(
          spv::OpBitwiseAnd, phys_uint_stype, old_word, mask_inv);
      spirv::Value new_word = ir_->make_value(
          spv::OpBitwiseOr, phys_uint_stype, cleared, packed);
      spirv::Value cas = ir_->make_value(
          spv::OpAtomicCompareExchange, phys_uint_stype, buf_ptr,
          /*scope=*/ir_->const_i32_one_,
          /*semantics_eq=*/ir_->const_i32_zero_,
          /*semantics_uneq=*/ir_->const_i32_zero_, new_word, old_word);
      spirv::Value won =
          ir_->make_value(spv::OpIEqual, ir_->bool_type(), cas, old_word);
      ir_->make_inst(spv::OpSelectionMerge, lose_lbl,
                     spv::SelectionControlMaskNone);
      ir_->make_inst(spv::OpBranchConditional, won, win_lbl, lose_lbl);
      ir_->make_inst(spv::OpLabel, win_lbl);
      ir_->make_inst(spv::OpBranch, exit_lbl);
      ir_->make_inst(spv::OpLabel, lose_lbl);
      ir_->make_inst(spv::OpBranch, continue_lbl);
      ir_->make_inst(spv::OpLabel, continue_lbl);
      ir_->make_inst(spv::OpBranch, head_lbl);

      ir_->start_label(exit_lbl);
    }
  }

  spirv::Value at_buffer(const Stmt *ptr, DataType dt) {
    spirv::Value ptr_val = ir_->query_value(ptr->raw_name());

    if (ptr_val.stype.dt == PrimitiveType::u64) {
      spirv::Value paddr_ptr = ir_->make_value(
          spv::OpConvertUToPtr,
          ir_->get_pointer_type(ir_->get_primitive_type(dt),
                                spv::StorageClassPhysicalStorageBuffer),
          ptr_val);
      paddr_ptr.flag = ValueKind::kPhysicalPtr;
      return paddr_ptr;
    }

    TI_ERROR_IF(
        !is_integral(ptr_val.stype.dt),
        "at_buffer failed, `ptr_val.stype.dt` is not integeral. Stmt = {} : {}",
        ptr->name(), ptr->type_hint());

    spirv::Value buffer = get_buffer_value(ptr_to_buffers_.at(ptr), dt);
    size_t width = ir_->get_primitive_type_size(dt);
    spirv::Value idx_val = ir_->make_value(
        spv::OpShiftRightLogical, ptr_val.stype, ptr_val,
        ir_->uint_immediate_number(ptr_val.stype, size_t(std::log2(width))));
    spirv::Value ret =
        ir_->struct_array_access(ir_->get_primitive_type(dt), buffer, idx_val);
    return ret;
  }

  spirv::Value load_buffer(const Stmt *ptr, DataType dt) {
    spirv::Value ptr_val = ir_->query_value(ptr->raw_name());

    DataType ti_buffer_type = ir_->get_taichi_uint_type(dt);

    if (ptr_val.stype.dt == PrimitiveType::u64) {
      ti_buffer_type = dt;
    } else if (dt->is_primitive(PrimitiveTypeID::u1)) {
      ti_buffer_type = PrimitiveType::i32;
    }

    auto buf_ptr = at_buffer(ptr, ti_buffer_type);
    auto val_bits =
        ir_->load_variable(buf_ptr, ir_->get_primitive_type(ti_buffer_type));
    if (dt->is_primitive(PrimitiveTypeID::u1))
      return ir_->cast(ir_->bool_type(), val_bits);
    return ti_buffer_type == dt
               ? val_bits
               : ir_->make_value(spv::OpBitcast, ir_->get_primitive_type(dt),
                                 val_bits);
  }

  void store_buffer(const Stmt *ptr, spirv::Value val) {
    spirv::Value ptr_val = ir_->query_value(ptr->raw_name());

    DataType ti_buffer_type = ir_->get_taichi_uint_type(val.stype.dt);

    if (ptr_val.stype.dt == PrimitiveType::u64) {
      ti_buffer_type = val.stype.dt;
    } else if (val.stype.dt->is_primitive(PrimitiveTypeID::u1)) {
      ti_buffer_type = PrimitiveType::i32;
      val = ir_->make_value(spv::OpSelect, ir_->i32_type(), val,
                            ir_->const_i32_one_, ir_->const_i32_zero_);
    }

    auto buf_ptr = at_buffer(ptr, ti_buffer_type);
    auto val_bits =
        val.stype.dt == ti_buffer_type
            ? val
            : ir_->make_value(spv::OpBitcast,
                              ir_->get_primitive_type(ti_buffer_type), val);
    ir_->store_variable(buf_ptr, val_bits);
  }

  spirv::Value get_buffer_value(BufferInfo buffer, DataType dt) {
    auto type = ir_->get_primitive_type(dt);
    auto key = std::make_pair(buffer, type.id);

    const auto it = buffer_value_map_.find(key);
    if (it != buffer_value_map_.end()) {
      return it->second;
    }

    if (buffer.type == BufferType::Args) {
      compile_args_struct();

      buffer_binding_map_[key] = 0;
      buffer_value_map_[key] = args_buffer_value_;
      return args_buffer_value_;
    }

    if (buffer.type == BufferType::Rets) {
      compile_ret_struct();

      buffer_binding_map_[key] = 1;
      buffer_value_map_[key] = ret_buffer_value_;
      return ret_buffer_value_;
    }

    if (buffer.type == BufferType::ArgPack) {
      // Make sure that Args Buffer are loaded first:
      get_buffer_value(BufferType::Args, PrimitiveType::i32);

      int binding = binding_head_++;
      buffer_binding_map_[key] = binding;

      auto buffer_value = compile_argpack_struct(buffer.root_id, binding,
                                                 buffer_instance_name(buffer));
      buffer_value_map_[key] = buffer_value;
      return buffer_value;
    }

    // Binding head starts at 2, so we don't break args and rets
    int binding = binding_head_++;
    buffer_binding_map_[key] = binding;

    spirv::Value buffer_value =
        ir_->buffer_argument(type, 0, binding, buffer_instance_name(buffer));
    buffer_value_map_[key] = buffer_value;
    TI_TRACE("buffer name = {}, value = {}", buffer_instance_name(buffer),
             buffer_value.id);

    return buffer_value;
  }

  spirv::Value make_pointer(size_t offset) {
    if (use_64bit_pointers) {
      // This is hacky, should check out how to encode uint64 values in spirv
      return ir_->uint_immediate_number(ir_->u64_type(), offset);
    } else {
      return ir_->uint_immediate_number(ir_->u32_type(), uint32_t(offset));
    }
  }

  void compile_args_struct() {
    if (!ctx_attribs_->has_args())
      return;

    // Generate struct IR
    tinyir::Block blk;
    std::unordered_map<std::vector<int>, const tinyir::Type *,
                       hashing::Hasher<std::vector<int>>>
        element_types;
    std::unordered_map<std::vector<int>, const taichi::lang::Type *,
                       hashing::Hasher<std::vector<int>>>
        element_taichi_types;
    std::vector<const tinyir::Type *> root_element_types;
    bool has_buffer_ptr =
        caps_->get(DeviceCapability::spirv_has_physical_storage_buffer);
    std::function<void(const std::vector<int> &indices, const Type *type)>
        add_types_to_element_types =
            [&](const std::vector<int> &indices, const Type *type) {
              auto spirv_type = translate_ti_type(blk, type, has_buffer_ptr);
              if (auto struct_type = type->cast<taichi::lang::StructType>()) {
                for (int j = 0; j < struct_type->elements().size(); ++j) {
                  std::vector<int> indices_copy = indices;
                  indices_copy.push_back(j);
                  add_types_to_element_types(indices_copy,
                                             struct_type->elements()[j].type);
                }
              }
              element_taichi_types[indices] = type;
              element_types[indices] = spirv_type;
            };
    for (int i = 0; i < ctx_attribs_->args_type()->elements().size(); i++) {
      auto *type = ctx_attribs_->args_type()->elements()[i].type;
      auto spirv_type = translate_ti_type(blk, type, has_buffer_ptr);
      element_types[{i}] = spirv_type;
      element_taichi_types[{i}] = type;
      root_element_types.push_back(spirv_type);
      if (auto struct_type = type->cast<taichi::lang::StructType>()) {
        for (int j = 0; j < struct_type->elements().size(); ++j) {
          add_types_to_element_types({i, j}, struct_type->elements()[j].type);
        }
      }
    }
    const tinyir::Type *struct_type =
        blk.emplace_back<StructType>(root_element_types);

    // Reduce struct IR
    std::unordered_map<const tinyir::Type *, const tinyir::Type *> old2new;
    auto reduced_blk = ir_reduce_types(&blk, old2new);
    struct_type = old2new[struct_type];

    for (auto &element : root_element_types) {
      element = old2new[element];
    }
    for (auto &element : element_types) {
      element.second = old2new[element.second];
    }

    // Layout & translate to SPIR-V
    STD140LayoutContext layout_ctx;
    auto ir2spirv_map =
        ir_translate_to_spirv(reduced_blk.get(), layout_ctx, ir_.get());
    args_struct_type_.id = ir2spirv_map[struct_type];

    // Must use the same type in ArgLoadStmt as in the args struct,
    // otherwise the validation will fail.
    for (auto &element : element_types) {
      spirv::SType spirv_type;
      spirv_type.id = ir2spirv_map.at(element.second);
      spirv_type.dt = element_taichi_types[element.first];
      args_struct_types_[element.first] = spirv_type;
    }

    args_buffer_value_ =
        ir_->uniform_struct_argument(args_struct_type_, 0, 0, "args");
  }

  spirv::Value compile_argpack_struct(const std::vector<int> &arg_id,
                                      int binding,
                                      const std::string &buffer_name) {
    spirv::SType argpack_struct_type;
    // Generate struct IR
    tinyir::Block blk;
    std::unordered_map<std::vector<int>, const tinyir::Type *,
                       hashing::Hasher<std::vector<int>>>
        element_types;
    std::unordered_map<std::vector<int>, const taichi::lang::Type *,
                       hashing::Hasher<std::vector<int>>>
        element_taichi_types;
    std::vector<const tinyir::Type *> root_element_types;
    bool has_buffer_ptr =
        caps_->get(DeviceCapability::spirv_has_physical_storage_buffer);
    std::function<void(const std::vector<int> &indices, const Type *type)>
        add_types_to_element_types =
            [&](const std::vector<int> &indices, const Type *type) {
              auto spirv_type = translate_ti_type(blk, type, has_buffer_ptr);
              if (auto struct_type = type->cast<taichi::lang::StructType>()) {
                for (int j = 0; j < struct_type->elements().size(); ++j) {
                  std::vector<int> indices_copy = indices;
                  indices_copy.push_back(j);
                  add_types_to_element_types(indices_copy,
                                             struct_type->elements()[j].type);
                }
              }
              element_taichi_types[indices] = type;
              element_types[indices] = spirv_type;
            };
    const lang::StructType *argpack_type =
        ctx_attribs_->argpack_type(arg_id)->as<lang::StructType>();
    for (int i = 0; i < argpack_type->elements().size(); i++) {
      auto *type = argpack_type->elements()[i].type;
      auto spirv_type = translate_ti_type(blk, type, has_buffer_ptr);
      element_types[{i}] = spirv_type;
      element_taichi_types[{i}] = type;
      root_element_types.push_back(spirv_type);
      if (auto struct_type = type->cast<taichi::lang::StructType>()) {
        for (int j = 0; j < struct_type->elements().size(); ++j) {
          add_types_to_element_types({i, j}, struct_type->elements()[j].type);
        }
      }
    }
    const tinyir::Type *struct_type =
        blk.emplace_back<StructType>(root_element_types);

    // Reduce struct IR
    std::unordered_map<const tinyir::Type *, const tinyir::Type *> old2new;
    auto reduced_blk = ir_reduce_types(&blk, old2new);
    struct_type = old2new[struct_type];

    for (auto &element : root_element_types) {
      element = old2new[element];
    }
    for (auto &element : element_types) {
      element.second = old2new[element.second];
    }

    // Layout & translate to SPIR-V
    STD140LayoutContext layout_ctx;
    auto ir2spirv_map =
        ir_translate_to_spirv(reduced_blk.get(), layout_ctx, ir_.get());
    argpack_struct_type.id = ir2spirv_map[struct_type];
    argpack_struct_type.dt = argpack_type;

    // Must use the same type in ArgLoadStmt as in the args struct,
    // otherwise the validation will fail.
    for (auto &element : element_types) {
      spirv::SType spirv_type;
      spirv_type.id = ir2spirv_map.at(element.second);
      spirv_type.dt = element_taichi_types[element.first];
      argpack_struct_types_[arg_id][element.first] = spirv_type;
    }

    argpack_types_[arg_id] = argpack_struct_type;
    argpack_buffer_values_[arg_id] = ir_->uniform_struct_argument(
        argpack_struct_type, 0, binding, buffer_name);
    return argpack_buffer_values_[arg_id];
  }

  void compile_ret_struct() {
    if (!ctx_attribs_->has_rets())
      return;

    // Generate struct IR
    tinyir::Block blk;
    std::vector<const tinyir::Type *> element_types;
    bool has_buffer_ptr =
        caps_->get(DeviceCapability::spirv_has_physical_storage_buffer);
    for (auto &element : ctx_attribs_->rets_type()->elements()) {
      element_types.push_back(
          translate_ti_type(blk, element.type, has_buffer_ptr));
    }
    const tinyir::Type *struct_type =
        blk.emplace_back<StructType>(element_types);

    // Reduce struct IR
    std::unordered_map<const tinyir::Type *, const tinyir::Type *> old2new;
    auto reduced_blk = ir_reduce_types(&blk, old2new);
    struct_type = old2new[struct_type];

    for (auto &element : element_types) {
      element = old2new[element];
    }

    // Layout & translate to SPIR-V
    STD430LayoutContext layout_ctx;
    auto ir2spirv_map =
        ir_translate_to_spirv(reduced_blk.get(), layout_ctx, ir_.get());
    ret_struct_type_.id = ir2spirv_map[struct_type];

    rets_struct_types_.resize(element_types.size());
    for (int i = 0; i < element_types.size(); i++) {
      rets_struct_types_[i].id = ir2spirv_map.at(element_types[i]);
      if (i < ctx_attribs_->rets_type()->elements().size()) {
        rets_struct_types_[i].dt =
            ctx_attribs_->rets_type()->get_element_type({i});
      } else {
        rets_struct_types_[i].dt = PrimitiveType::i32;
      }
    }

    ret_buffer_value_ =
        ir_->buffer_struct_argument(ret_struct_type_, 0, 1, "rets");
  }

  std::vector<BufferBind> get_buffer_binds() {
    std::vector<BufferBind> result;
    for (auto &[key, val] : buffer_binding_map_) {
      result.push_back(BufferBind{key.first, int(val)});
    }
    return result;
  }

  std::vector<TextureBind> get_texture_binds() {
    return texture_binds_;
  }

  void push_loop_control_labels(spirv::Label continue_label,
                                spirv::Label merge_label) {
    continue_label_stack_.push_back(continue_label);
    merge_label_stack_.push_back(merge_label);
  }

  void pop_loop_control_labels() {
    continue_label_stack_.pop_back();
    merge_label_stack_.pop_back();
  }

  const spirv::Label current_continue_label() const {
    return continue_label_stack_.back();
  }

  const spirv::Label current_merge_label() const {
    return merge_label_stack_.back();
  }

  const spirv::Label return_label() const {
    return continue_label_stack_.front();
  }

  Arch arch_;
  DeviceCapabilityConfig *caps_;

  struct BufferInfoTypeTupleHasher {
    std::size_t operator()(const std::pair<BufferInfo, int> &buf) const {
      return BufferInfoHasher()(buf.first) ^ (buf.second << 5);
    }
  };

  spirv::SType args_struct_type_;
  spirv::Value args_buffer_value_;

  std::unordered_map<std::vector<int>,
                     spirv::SType,
                     hashing::Hasher<std::vector<int>>>
      args_struct_types_;
  std::unordered_map<std::vector<int>,
                     std::unordered_map<std::vector<int>,
                                        spirv::SType,
                                        hashing::Hasher<std::vector<int>>>,
                     hashing::Hasher<std::vector<int>>>
      argpack_struct_types_;
  std::unordered_map<std::vector<int>,
                     spirv::SType,
                     hashing::Hasher<std::vector<int>>>
      argpack_types_;
  std::unordered_map<std::vector<int>,
                     spirv::Value,
                     hashing::Hasher<std::vector<int>>>
      argpack_buffer_values_;

  std::vector<spirv::SType> rets_struct_types_;

  spirv::SType ret_struct_type_;
  spirv::Value ret_buffer_value_;

  std::shared_ptr<spirv::IRBuilder> ir_;  // spirv binary code builder
  std::unordered_map<std::pair<BufferInfo, int>,
                     spirv::Value,
                     BufferInfoTypeTupleHasher>
      buffer_value_map_;
  std::unordered_map<std::pair<BufferInfo, int>,
                     uint32_t,
                     BufferInfoTypeTupleHasher>
      buffer_binding_map_;
  std::vector<TextureBind> texture_binds_;
  std::vector<spirv::Value> shared_array_binds_;
  spirv::Value kernel_function_;
  spirv::Label kernel_return_label_;
  bool gen_label_{false};

  int binding_head_{2};  // Args:0, Ret:1

  /*
  std::unordered_map<int, spirv::CompiledSpirvSNode>
      spirv_snodes_;  // maps root id to spirv snode
      */

  OffloadedStmt *const task_ir_;  // not owned
  std::vector<CompiledSNodeStructs> compiled_structs_;
  std::unordered_map<int, int> snode_to_root_;
  const KernelContextAttributes *const ctx_attribs_;  // not owned
  const std::string task_name_;
  std::vector<spirv::Label> continue_label_stack_;
  std::vector<spirv::Label> merge_label_stack_;

  std::unordered_set<const Stmt *> offload_loop_motion_;

  TaskAttributes task_attribs_;
  std::unordered_map<int, GetRootStmt *>
      root_stmts_;  // maps root id to get root stmt
  std::unordered_map<const Stmt *, BufferInfo> ptr_to_buffers_;
  // G9.2 (2026-04-30): bit-pointer side table.  When a stmt's ret_type is
  // PointerType{is_bit_pointer=true} (produced by SNodeLookupStmt of a
  // quant_array, or in the future by GetChStmt of a bit_struct member),
  // its raw_name SSA value holds the *byte* address of the containing
  // physical word and the bit offset is tracked here as a parallel u32
  // SSA value.  GlobalLoadStmt / GlobalStoreStmt then emit the shift /
  // mask / sign-extend sequence using these two values.  Empty when
  // vulkan_quant_experimental is OFF (the entry is only ever populated
  // by the new quant_array branch in SNodeLookupStmt, which itself only
  // runs for fields whose extension was admitted by extension.cpp).
  std::unordered_map<const Stmt *, spirv::Value> bit_ptr_bit_offsets_;
  std::unordered_map<std::vector<int>, Value, hashing::Hasher<std::vector<int>>>
      argid_to_tex_value_;
};
}  // namespace

static void spriv_message_consumer(spv_message_level_t level,
                                   const char *source,
                                   const spv_position_t &position,
                                   const char *message) {
  // TODO: Maybe we can add a macro, e.g. TI_LOG_AT_LEVEL(lv, ...)
  if (level <= SPV_MSG_FATAL) {
    TI_ERROR("{}\n[{}:{}:{}] {}", source, position.index, position.line,
             position.column, message);
  } else if (level <= SPV_MSG_WARNING) {
    TI_WARN("{}\n[{}:{}:{}] {}", source, position.index, position.line,
            position.column, message);
  } else if (level <= SPV_MSG_INFO) {
    TI_INFO("{}\n[{}:{}:{}] {}", source, position.index, position.line,
            position.column, message);
  } else if (level <= SPV_MSG_INFO) {
    TI_TRACE("{}\n[{}:{}:{}] {}", source, position.index, position.line,
             position.column, message);
  }
}

KernelCodegen::KernelCodegen(const Params &params)
    : params_(params), ctx_attribs_(*params.kernel, &params.caps) {
  TI_ASSERT(params.kernel);
  TI_ASSERT(params.ir_root);

  uint32_t spirv_version = params.caps.get(DeviceCapability::spirv_version);

  if (spirv_version >= 0x10600) {
    target_env_ = SPV_ENV_VULKAN_1_3;
  } else if (spirv_version >= 0x10500) {
    target_env_ = SPV_ENV_VULKAN_1_2;
  } else if (spirv_version >= 0x10400) {
    target_env_ = SPV_ENV_VULKAN_1_1_SPIRV_1_4;
  } else if (spirv_version >= 0x10300) {
    target_env_ = SPV_ENV_VULKAN_1_1;
  } else {
    target_env_ = SPV_ENV_VULKAN_1_0;
  }

  spirv_opt_options_.set_run_validator(false);
}

namespace {

// V3 + V2 (2026-04-26): per-thread cached spvtools::Optimizer/SpirvTools.
// Each OS thread that hits this helper builds its own Optimizer once per
// (target_env, spv_opt_level, skip_loop_unroll, disabled_passes_hash)
// combination, then reuses it across kernels. Thread-locality lets the
// parallel SPIR-V codegen path (V2) fan out work to worker threads
// without violating spvtools' single-instance thread-safety contract.
//
// B2 (2026-04-26): pass groups
//   Tier-1 ("\u5fc5\u9700\u6b63\u786e\u6027 / \u901a\u7528\u5165\u95e8", spv_opt_level >= 1):
//     WrapOpKill, DeadBranchElim, AggressiveDCE
//   Tier-2 ("\u901a\u7528\u4f18\u5316", spv_opt_level >= 2):
//     InlineExhaustive, EliminateDeadFunctions, PrivateToLocal,
//     LocalSingleBlockLoadStoreElim, LocalSingleStoreElim,
//     ScalarReplacement, LocalAccessChainConvert, LocalMultiStoreElim, CCP
//   Tier-3 ("\u6fc0\u8fdb\u4f18\u5316", spv_opt_level == 3):
//     MergeReturn, LoopUnroll (skippable via spirv_skip_loop_unroll),
//     RedundancyElimination, CombineAccessChains, Simplification,
//     SSARewrite, VectorDCE, DeadInsertElim, IfConversion,
//     CopyPropagateArrays, ReduceLoadSize, BlockMerge
//
// Users can disable any individual pass by name via
// CompileConfig::spirv_disabled_passes (e.g. ["LoopUnroll",
// "AggressiveDCE"]). Disabled passes are not registered, so they cost
// neither registration nor Run() time.
struct OptCacheKey {
  int target_env_int;
  int spv_opt_level;
  bool skip_loop_unroll;
  // B2: stable hash of the sorted disabled_passes list. Empty list \u2192 0.
  size_t disabled_passes_hash;
  bool operator==(const OptCacheKey &o) const noexcept {
    return target_env_int == o.target_env_int &&
           spv_opt_level == o.spv_opt_level &&
           skip_loop_unroll == o.skip_loop_unroll &&
           disabled_passes_hash == o.disabled_passes_hash;
  }
};
struct OptCacheKeyHash {
  size_t operator()(const OptCacheKey &k) const noexcept {
    return (static_cast<size_t>(k.target_env_int) << 5) ^
           (static_cast<size_t>(k.spv_opt_level) << 1) ^
           static_cast<size_t>(k.skip_loop_unroll ? 1u : 0u) ^
           (k.disabled_passes_hash << 3);
  }
};
struct OptCacheEntry {
  std::unique_ptr<spvtools::Optimizer> opt;
  std::unique_ptr<spvtools::SpirvTools> tools;
};

// B2: stable hash of a *sorted* disabled-passes vector. Caller is
// responsible for sorting (we also re-sort defensively here) so cache
// keys are insensitive to user-supplied ordering.
size_t hash_disabled_passes(const std::vector<std::string> &passes) {
  if (passes.empty()) {
    return 0;
  }
  std::vector<std::string> sorted = passes;
  std::sort(sorted.begin(), sorted.end());
  size_t h = 1469598103934665603ull;  // FNV offset basis
  for (const auto &s : sorted) {
    for (unsigned char c : s) {
      h ^= c;
      h *= 1099511628211ull;
    }
    h ^= 0xff;  // separator between entries
    h *= 1099511628211ull;
  }
  return h;
}

void get_thread_local_opt(spv_target_env target_env,
                          int spv_opt_level,
                          bool skip_loop_unroll,
                          const std::vector<std::string> &disabled_passes,
                          spvtools::Optimizer **out_opt,
                          spvtools::SpirvTools **out_tools) {
  static thread_local std::unordered_map<OptCacheKey, OptCacheEntry,
                                         OptCacheKeyHash>
      opt_cache;
  const size_t dp_hash = hash_disabled_passes(disabled_passes);
  OptCacheKey key{static_cast<int>(target_env), spv_opt_level,
                  skip_loop_unroll, dp_hash};
  auto it = opt_cache.find(key);
  if (it == opt_cache.end()) {
    // Build a set for O(1) membership tests during pass registration.
    std::unordered_set<std::string> disabled(disabled_passes.begin(),
                                             disabled_passes.end());
    auto enabled = [&disabled](const char *name) {
      return disabled.find(name) == disabled.end();
    };
    OptCacheEntry entry;
    entry.opt = std::make_unique<spvtools::Optimizer>(target_env);
    entry.opt->SetMessageConsumer(spriv_message_consumer);
    if (spv_opt_level >= 1) {
      // Tier-1 \u2014 \u5fc5\u9700\u6b63\u786e\u6027 / \u901a\u7528\u5165\u95e8
      if (enabled("WrapOpKill"))
        entry.opt->RegisterPass(spvtools::CreateWrapOpKillPass());
      if (enabled("DeadBranchElim"))
        entry.opt->RegisterPass(spvtools::CreateDeadBranchElimPass());
      if (enabled("AggressiveDCE"))
        entry.opt->RegisterPass(spvtools::CreateAggressiveDCEPass());
    }
    if (spv_opt_level >= 2) {
      // Tier-2 \u2014 \u901a\u7528\u4f18\u5316
      if (enabled("InlineExhaustive"))
        entry.opt->RegisterPass(spvtools::CreateInlineExhaustivePass());
      if (enabled("EliminateDeadFunctions"))
        entry.opt->RegisterPass(spvtools::CreateEliminateDeadFunctionsPass());
      if (enabled("PrivateToLocal"))
        entry.opt->RegisterPass(spvtools::CreatePrivateToLocalPass());
      if (enabled("LocalSingleBlockLoadStoreElim"))
        entry.opt->RegisterPass(
            spvtools::CreateLocalSingleBlockLoadStoreElimPass());
      if (enabled("LocalSingleStoreElim"))
        entry.opt->RegisterPass(spvtools::CreateLocalSingleStoreElimPass());
      if (enabled("ScalarReplacement"))
        entry.opt->RegisterPass(spvtools::CreateScalarReplacementPass());
      if (enabled("LocalAccessChainConvert"))
        entry.opt->RegisterPass(spvtools::CreateLocalAccessChainConvertPass());
      if (enabled("LocalMultiStoreElim"))
        entry.opt->RegisterPass(spvtools::CreateLocalMultiStoreElimPass());
      if (enabled("CCP"))
        entry.opt->RegisterPass(spvtools::CreateCCPPass());
    }
    if (spv_opt_level >= 3) {
      // Tier-3 \u2014 \u6fc0\u8fdb\u4f18\u5316
      if (enabled("MergeReturn"))
        entry.opt->RegisterPass(spvtools::CreateMergeReturnPass());
      // V6 (2026-04-26): CreateLoopUnrollPass is the most expensive pass
      // in the level-3 chain. When skip_loop_unroll is true (or the user
      // explicitly disables "LoopUnroll" via spirv_disabled_passes) we
      // drop it entirely and rely on the GPU driver's own loop unrolling.
      if (!skip_loop_unroll && enabled("LoopUnroll"))
        entry.opt->RegisterPass(spvtools::CreateLoopUnrollPass(true));
      if (enabled("RedundancyElimination"))
        entry.opt->RegisterPass(spvtools::CreateRedundancyEliminationPass());
      if (enabled("CombineAccessChains"))
        entry.opt->RegisterPass(spvtools::CreateCombineAccessChainsPass());
      if (enabled("Simplification"))
        entry.opt->RegisterPass(spvtools::CreateSimplificationPass());
      if (enabled("SSARewrite"))
        entry.opt->RegisterPass(spvtools::CreateSSARewritePass());
      if (enabled("VectorDCE"))
        entry.opt->RegisterPass(spvtools::CreateVectorDCEPass());
      if (enabled("DeadInsertElim"))
        entry.opt->RegisterPass(spvtools::CreateDeadInsertElimPass());
      if (enabled("IfConversion"))
        entry.opt->RegisterPass(spvtools::CreateIfConversionPass());
      if (enabled("CopyPropagateArrays"))
        entry.opt->RegisterPass(spvtools::CreateCopyPropagateArraysPass());
      if (enabled("ReduceLoadSize"))
        entry.opt->RegisterPass(spvtools::CreateReduceLoadSizePass());
      if (enabled("BlockMerge"))
        entry.opt->RegisterPass(spvtools::CreateBlockMergePass());
    }
    entry.tools = std::make_unique<spvtools::SpirvTools>(target_env);
    it = opt_cache.emplace(key, std::move(entry)).first;
  }
  *out_opt = it->second.opt.get();
  *out_tools = it->second.tools.get();
}

}  // namespace

void KernelCodegen::run(TaichiKernelAttributes &kernel_attribs,
                        std::vector<std::vector<uint32_t>> &generated_spirv) {
  auto *root = params_.ir_root->as<Block>();
  auto &tasks = root->statements;
  const int n = static_cast<int>(tasks.size());

  // Per-task results, indexed by task_id so iteration order matches the
  // serial path even when codegen runs on worker threads.
  struct TaskOut {
    std::vector<uint32_t> spv;
    TaskAttributes attribs;
    std::unordered_map<std::vector<int>,
                       irpass::ExternalPtrAccess,
                       hashing::Hasher<std::vector<int>>>
        arr_access;
  };
  std::vector<TaskOut> outs(n);

  // V2 (2026-04-26): per-task SPIR-V emission + spvtools::Optimizer::Run
  // are independent (no shared mutable state across tasks; ctx_attribs_
  // is read-only inside TaskCodegen, mutated only in the post-loop
  // aggregation below). The lambda is thread-safe: each invocation
  // constructs its own TaskCodegen + IRBuilder and fetches its caller
  // thread's thread_local Optimizer via get_thread_local_opt().
  auto run_task = [&, this](int i) {
    TaskCodegen::Params tp;
    tp.task_ir = tasks[i]->as<OffloadedStmt>();
    tp.task_id_in_kernel = i;
    tp.compiled_structs = params_.compiled_structs;
    tp.ctx_attribs = &ctx_attribs_;
    tp.ti_kernel_name = fmt::format("{}_{}", params_.ti_kernel_name, i);
    tp.arch = params_.arch;
    tp.caps = &params_.caps;

    TaskCodegen cgen(tp);
    auto task_res = cgen.run();

    std::vector<uint32_t> optimized_spv(task_res.spirv_code);

    if (params_.spv_opt_level > 0) {
      spvtools::Optimizer *opt = nullptr;
      spvtools::SpirvTools *tools = nullptr;
      get_thread_local_opt(target_env_, params_.spv_opt_level,
                           params_.skip_loop_unroll,
                           params_.disabled_passes, &opt, &tools);
      bool result = false;
      TI_WARN_IF(
          (result = !opt->Run(optimized_spv.data(), optimized_spv.size(),
                              &optimized_spv, spirv_opt_options_)),
          "SPIRV optimization failed");
      (void)result;
    }

    TI_TRACE("SPIRV-Tools-opt: binary size, before={}, after={}",
             task_res.spirv_code.size(), optimized_spv.size());

    outs[i].spv = std::move(optimized_spv);
    outs[i].attribs = std::move(task_res.task_attribs);
    outs[i].arr_access = std::move(task_res.arr_access);
  };

  // Single-task kernels always go through the serial path: thread-spawn
  // overhead dwarfs any potential gain.
  // V8.b (2026-04-26): also bypass the inner pool when the outer
  // compile_kernels worker is already saturating cores. Symmetric with
  // V7's LLVM path; see compile_doc/优化总规划.md §3.5.
  const bool use_parallel = params_.parallel_codegen && n >= 2 &&
                            !(params_.compile_dag_scheduler &&
                              Program::in_compile_kernels_worker());
  if (use_parallel) {
    const int n_workers = std::max(1, std::min(n, params_.num_compile_threads));
    // Fan out tasks in chunks of n_workers; each chunk's std::async
    // futures join before the next chunk starts. This caps live worker
    // threads at n_workers and keeps memory footprint bounded.
    std::vector<std::future<void>> futs;
    futs.reserve(n_workers);
    int i = 0;
    while (i < n) {
      const int end = std::min(n, i + n_workers);
      futs.clear();
      for (int j = i; j < end; ++j) {
        futs.emplace_back(
            std::async(std::launch::async, [&run_task, j]() { run_task(j); }));
      }
      for (auto &f : futs) {
        f.get();
      }
      i = end;
    }
  } else {
    for (int i = 0; i < n; ++i) {
      run_task(i);
    }
  }

  // Aggregate results sequentially in task order.
  for (int i = 0; i < n; ++i) {
    for (auto &[id, access] : outs[i].arr_access) {
      for (auto &arr_access_element : ctx_attribs_.arr_access) {
        if (arr_access_element.first == id) {
          arr_access_element.second = arr_access_element.second | access;
        }
      }
    }
    kernel_attribs.tasks_attribs.push_back(std::move(outs[i].attribs));
    generated_spirv.push_back(std::move(outs[i].spv));
  }
  kernel_attribs.ctx_attribs = std::move(ctx_attribs_);
  kernel_attribs.name = params_.ti_kernel_name;
}

}  // namespace spirv
}  // namespace taichi::lang
