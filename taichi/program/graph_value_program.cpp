#include "taichi/program/graph_value_program.h"

#include <unordered_map>
#include <utility>

#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/program/compile_config.h"
#include "taichi/program/kernel.h"

namespace taichi::lang {
namespace {

constexpr int kMaxValueNodes = 64;

bool integer32(DataType type) {
  return type == PrimitiveType::i32 || type == PrimitiveType::u32;
}

DataType value_type(const Callable::Parameter &parameter) {
  if (!parameter.is_array) {
    return parameter.get_dtype();
  }
  const auto *descriptor = parameter.get_dtype()->cast<StructType>();
  if (parameter.ptype != ParameterType::kNdarray || descriptor == nullptr ||
      parameter.total_dim != 1 || !parameter.element_shape.empty()) {
    return PrimitiveType::unknown;
  }
  // Callable ndarray parameters are shape/data-pointer descriptors, not the
  // scalar dtype itself. Follow the same data member as IR type checking.
  return DataType(descriptor->get_element_type({1})).ptr_removed();
}

// The ndarray frontend can flatten a rank-one loop as i % shape[0]. This is
// exactly i only when that shape is the certified iteration domain. A general
// remainder (including a remainder in the computed value) is not admitted.
bool pointwise_index(Stmt *stmt,
                     RangeForStmt *loop,
                     const GraphKernelIterationDomain &domain) {
  if (auto *index = stmt->cast<LoopIndexStmt>()) {
    return index->loop == loop && index->index == 0;
  }
  auto *binary = stmt->cast<BinaryOpStmt>();
  if (binary == nullptr || binary->op_type != BinaryOpType::mod ||
      domain.kind != "external_tensor") {
    return false;
  }
  auto *shape = binary->rhs->cast<ExternalTensorShapeAlongAxisStmt>();
  return shape != nullptr && shape->arg_id == domain.arg_id &&
         shape->axis == domain.axis &&
         pointwise_index(binary->lhs, loop, domain);
}

bool setup_statement(Stmt *stmt) {
  return stmt->is<ConstStmt>() || stmt->is<ArgLoadStmt>() ||
         stmt->is<ExternalTensorShapeAlongAxisStmt>() ||
         stmt->is<UnaryOpStmt>() || stmt->is<BinaryOpStmt>();
}

class ValueBuilder {
 public:
  ValueBuilder(Kernel *kernel,
               RangeForStmt *loop,
               GraphPointwiseValueProgram *program)
      : kernel_(kernel), loop_(loop), program_(program) {
  }

  int array_argument(Stmt *stmt) const {
    auto *pointer = stmt->cast<ExternalPtrStmt>();
    if (pointer == nullptr || pointer->ndim != 1 ||
        pointer->indices.size() != 1 || !pointer->element_shape.empty() ||
        pointer->is_grad || pointer->boundary != BoundaryMode::kUnsafe ||
        pointer->byte_offset != 0 || pointer->byte_stride != 0 ||
        !pointwise_index(pointer->indices[0], loop_,
                         program_->metadata.iteration_domain)) {
      return -1;
    }
    auto *base = pointer->base_ptr->cast<ArgLoadStmt>();
    const int argument = flat_argument(base, true);
    if (argument < 0 || pointer->ret_type.ptr_removed() !=
                            value_type(kernel_->parameter_list[argument])) {
      return -1;
    }
    return argument;
  }

  int value(Stmt *stmt, int depth = 0) {
    const auto found = ids_.find(stmt);
    if (found != ids_.end()) {
      return found->second;
    }
    if (depth >= kMaxValueNodes || program_->nodes.size() >= kMaxValueNodes) {
      return fail("value_program_too_large");
    }
    if (!integer32(stmt->ret_type)) {
      return fail("non_integer32_value");
    }
    GraphValueNode node;
    node.dtype = stmt->ret_type;
    if (pointwise_index(stmt, loop_, program_->metadata.iteration_domain)) {
      node.kind = GraphValueNode::Kind::index;
    } else if (auto *constant = stmt->cast<ConstStmt>()) {
      node.kind = GraphValueNode::Kind::constant;
      node.constant_bits =
          static_cast<std::uint32_t>(constant->val.val_as_int64());
    } else if (auto *argument = stmt->cast<ArgLoadStmt>()) {
      node.kind = GraphValueNode::Kind::scalar_argument;
      node.argument = flat_argument(argument, false);
      if (node.argument < 0) {
        return fail("unsupported_scalar_argument");
      }
    } else if (auto *load = stmt->cast<GlobalLoadStmt>()) {
      node.kind = GraphValueNode::Kind::array_load;
      node.argument = array_argument(load->src);
      if (node.argument < 0) {
        return fail("unsupported_array_load");
      }
      node.runtime_affine = load->src->as<ExternalPtrStmt>()->runtime_affine;
    } else if (auto *unary = stmt->cast<UnaryOpStmt>()) {
      node.kind = GraphValueNode::Kind::unary;
      node.unary_op = unary->op_type;
      if (node.unary_op != UnaryOpType::cast_value &&
          node.unary_op != UnaryOpType::cast_bits &&
          node.unary_op != UnaryOpType::neg &&
          node.unary_op != UnaryOpType::bit_not) {
        return fail("unsupported_unary_operation");
      }
      node.operands = {value(unary->operand, depth + 1)};
    } else if (auto *binary = stmt->cast<BinaryOpStmt>()) {
      node.kind = GraphValueNode::Kind::binary;
      node.binary_op = binary->op_type;
      switch (node.binary_op) {
        case BinaryOpType::add:
        case BinaryOpType::sub:
        case BinaryOpType::mul:
        case BinaryOpType::bit_and:
        case BinaryOpType::bit_or:
        case BinaryOpType::bit_xor:
          break;
        case BinaryOpType::bit_shl:
        case BinaryOpType::bit_shr:
        case BinaryOpType::bit_sar: {
          auto *shift = binary->rhs->cast<ConstStmt>();
          if (shift == nullptr || !integer32(shift->ret_type) ||
              shift->val.val_as_int64() < 0 ||
              shift->val.val_as_int64() >= 32) {
            return fail("unbounded_shift");
          }
          break;
        }
        default:
          return fail("unsupported_binary_operation");
      }
      if (binary->is_bit_vectorized) {
        return fail("vectorized_value");
      }
      node.operands = {value(binary->lhs, depth + 1),
                       value(binary->rhs, depth + 1)};
    } else {
      return fail("unsupported_value_statement");
    }
    for (int operand : node.operands) {
      if (operand < 0) {
        return -1;
      }
    }
    if (program_->nodes.size() >= kMaxValueNodes) {
      return fail("value_program_too_large");
    }
    const int id = static_cast<int>(program_->nodes.size());
    program_->nodes.push_back(std::move(node));
    ids_.emplace(stmt, id);
    return id;
  }

 private:
  int fail(const char *reason) {
    if (program_->blocker.empty()) {
      program_->blocker = reason;
    }
    return -1;
  }

  int flat_argument(const ArgLoadStmt *stmt, bool array) const {
    if (stmt == nullptr || stmt->arg_id.size() != 1 || stmt->arg_depth != 0 ||
        stmt->is_ptr != array || stmt->create_load == array) {
      return -1;
    }
    const int id = stmt->arg_id[0];
    if (id < 0 || id >= static_cast<int>(kernel_->parameter_list.size())) {
      return -1;
    }
    return kernel_->parameter_list[id].is_array == array ? id : -1;
  }

  Kernel *kernel_;
  RangeForStmt *loop_;
  GraphPointwiseValueProgram *program_;
  std::unordered_map<Stmt *, int> ids_;
};

}  // namespace

GraphPointwiseValueProgram inspect_graph_pointwise_value_program(
    const CompileConfig &config,
    Kernel *kernel) {
  GraphPointwiseValueProgram result;
  auto reject = [&](std::string reason) {
    result.blocker = std::move(reason);
    result.nodes.clear();
    result.result = -1;
    result.output_argument = -1;
    result.output_runtime_affine = false;
    return result;
  };
  if (kernel == nullptr || kernel->definition_retired() || !kernel->ir ||
      kernel->autodiff_mode != AutodiffMode::kNone || !kernel->rets.empty()) {
    return reject("callable_boundary");
  }
  if (config.debug || config.check_out_of_bound) {
    return reject("instrumented_kernel");
  }
  if (kernel->nested_parameters.size() != kernel->parameter_list.size()) {
    return reject("non_flat_arguments");
  }
  for (std::size_t index = 0; index < kernel->parameter_list.size(); ++index) {
    const auto &parameter = kernel->parameter_list[index];
    if (!integer32(value_type(parameter)) || parameter.is_argpack ||
        parameter.needs_grad ||
        (parameter.is_array && parameter.total_dim != 1) ||
        kernel->nested_parameters.find({static_cast<int>(index)}) ==
            kernel->nested_parameters.end()) {
      return reject("unsupported_argument_type");
    }
  }
  auto ir = irpass::analysis::clone(kernel->ir.get());
  irpass::compile_to_offloads(ir.get(), config, kernel, false,
                              AutodiffMode::kNone, true, kernel->ir_is_ast(),
                              &result.metadata, /*stop_before_offload=*/true);
  const auto &metadata = result.metadata;
  if (!metadata.available || metadata.opaque || !metadata.elementwise ||
      metadata.synchronization || !metadata.side_effects.empty()) {
    return reject(metadata.blocker.empty() ? "non_pointwise_effects"
                                           : metadata.blocker);
  }
  for (const auto &effect : metadata.effects) {
    if (effect.resource_kind != "argument" || effect.is_grad ||
        effect.footprint.pattern != "exact_pointwise" ||
        effect.footprint.iteration_rank != 1 ||
        (effect.access != "read" && effect.access != "write" &&
         effect.access != "read_write")) {
      return reject("unsupported_resource_effect");
    }
  }
  auto *root = ir->cast<Block>();
  RangeForStmt *loop = nullptr;
  for (const auto &statement : root->statements) {
    if (auto *candidate = statement->cast<RangeForStmt>()) {
      loop = candidate;
    } else if (!setup_statement(statement.get())) {
      return reject("unsupported_loop_setup");
    }
  }
  if (loop == nullptr) {
    return reject("missing_range_for");
  }
  GlobalStoreStmt *store = nullptr;
  for (const auto &statement : loop->body->statements) {
    if (auto *decoration = statement->cast<DecorationStmt>()) {
      // Rank-one ndarray iteration attaches this non-executing hint. Prove
      // its operand ourselves; do not carry a loop-uniqueness assertion into
      // the reduction's different iteration topology.
      if (decoration->decoration.size() != 2 ||
          decoration->decoration[1] != 0 ||
          decoration->decoration[0] !=
              static_cast<std::uint32_t>(
                  DecorationStmt::Decoration::kLoopUnique) ||
          !pointwise_index(decoration->operand, loop,
                           metadata.iteration_domain)) {
        return reject("unsupported_decoration");
      }
    } else if (auto *candidate = statement->cast<GlobalStoreStmt>()) {
      if (store != nullptr) {
        return reject("multiple_stores");
      }
      store = candidate;
    } else if (!setup_statement(statement.get()) &&
               !statement->is<ExternalPtrStmt>() &&
               !statement->is<LoopIndexStmt>() &&
               !statement->is<GlobalLoadStmt>()) {
      // In particular reject conditional/multi-store, random, local mutation,
      // calls and other statements which a value-only DAG cannot preserve.
      return reject("unsupported_body_statement");
    }
  }
  if (store == nullptr) {
    return reject("missing_store");
  }
  result.blocker.clear();
  ValueBuilder builder(kernel, loop, &result);
  result.output_argument = builder.array_argument(store->dest);
  if (result.output_argument < 0) {
    return reject("unsupported_array_store");
  }
  result.output_runtime_affine =
      store->dest->as<ExternalPtrStmt>()->runtime_affine;
  result.result = builder.value(store->val);
  if (result.result < 0) {
    return reject(result.blocker);
  }
  result.available = true;
  return result;
}

}  // namespace taichi::lang
