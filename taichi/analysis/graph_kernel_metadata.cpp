#include "taichi/analysis/graph_kernel_metadata.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <tuple>

#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/visitors.h"
#include "taichi/program/kernel.h"

namespace taichi::lang {
namespace {

using ResourceKey =
    std::tuple<std::string, std::vector<int>, int, int, bool>;

bool constant_integer(Stmt *stmt, std::int64_t *value) {
  auto *constant = stmt ? stmt->cast<ConstStmt>() : nullptr;
  if (constant == nullptr || !is_integral(constant->val.dt)) {
    return false;
  }
  *value = constant->val.val_as_int64();
  return true;
}

bool constant_one(Stmt *stmt) {
  std::int64_t value = 0;
  return constant_integer(stmt, &value) && value == 1;
}

bool external_shape(Stmt *stmt,
                    std::vector<int> *arg_id,
                    int *axis) {
  if (auto *shape = stmt ? stmt->cast<ExternalTensorShapeAlongAxisStmt>()
                         : nullptr) {
    *arg_id = shape->arg_id;
    *axis = shape->axis;
    return true;
  }
  auto *binary = stmt ? stmt->cast<BinaryOpStmt>() : nullptr;
  if (binary == nullptr || binary->op_type != BinaryOpType::mul) {
    return false;
  }
  if (constant_one(binary->lhs)) {
    return external_shape(binary->rhs, arg_id, axis);
  }
  if (constant_one(binary->rhs)) {
    return external_shape(binary->lhs, arg_id, axis);
  }
  return false;
}

GraphKernelIterationDomain iteration_domain(RangeForStmt *loop) {
  GraphKernelIterationDomain result;
  std::int64_t begin = 0;
  if (!constant_integer(loop->begin, &begin)) {
    return result;
  }
  std::int64_t end = 0;
  if (constant_integer(loop->end, &end)) {
    result.kind = "constant_range";
    result.begin = begin;
    result.end = end;
    return result;
  }
  std::vector<int> arg_id;
  int axis = -1;
  if (begin == 0 && external_shape(loop->end, &arg_id, &axis) && axis == 0) {
    result.kind = "external_tensor";
    result.arg_id = std::move(arg_id);
    result.axis = axis;
    return result;
  }
  if (begin == 0) {
    if (auto *arg = loop->end->cast<ArgLoadStmt>()) {
      if (!arg->is_ptr) {
        result.kind = "scalar_argument";
        result.arg_id = arg->arg_id;
      }
    }
  }
  return result;
}

bool same_external_shape(Stmt *stmt,
                         const GraphKernelIterationDomain &domain) {
  std::vector<int> arg_id;
  int axis = -1;
  return domain.kind == "external_tensor" &&
         external_shape(stmt, &arg_id, &axis) && arg_id == domain.arg_id &&
         axis == domain.axis;
}

bool is_loop_index(Stmt *stmt,
                   RangeForStmt *loop,
                   const GraphKernelIterationDomain &domain) {
  auto *index = stmt ? stmt->cast<LoopIndexStmt>() : nullptr;
  if (index != nullptr) {
    return index->loop == loop && index->index == 0;
  }
  auto *binary = stmt ? stmt->cast<BinaryOpStmt>() : nullptr;
  return binary != nullptr && binary->op_type == BinaryOpType::mod &&
         is_loop_index(binary->lhs, loop, domain) &&
         same_external_shape(binary->rhs, domain);
}

bool checked_add(std::int64_t lhs,
                 std::int64_t rhs,
                 std::int64_t *result) {
  if ((rhs > 0 && lhs > std::numeric_limits<std::int64_t>::max() - rhs) ||
      (rhs < 0 && lhs < std::numeric_limits<std::int64_t>::min() - rhs)) {
    return false;
  }
  *result = lhs + rhs;
  return true;
}

bool checked_subtract(std::int64_t lhs,
                      std::int64_t rhs,
                      std::int64_t *result) {
  if ((rhs > 0 && lhs < std::numeric_limits<std::int64_t>::min() + rhs) ||
      (rhs < 0 && lhs > std::numeric_limits<std::int64_t>::max() + rhs)) {
    return false;
  }
  *result = lhs - rhs;
  return true;
}

bool affine_loop_offset(Stmt *stmt,
                        RangeForStmt *loop,
                        const GraphKernelIterationDomain &domain,
                        std::int64_t *offset) {
  if (is_loop_index(stmt, loop, domain)) {
    *offset = 0;
    return true;
  }
  auto *binary = stmt ? stmt->cast<BinaryOpStmt>() : nullptr;
  if (binary == nullptr) {
    return false;
  }
  std::int64_t base_offset = 0;
  std::int64_t constant = 0;
  if (binary->op_type == BinaryOpType::add) {
    if (affine_loop_offset(binary->lhs, loop, domain, &base_offset) &&
        constant_integer(binary->rhs, &constant)) {
      return checked_add(base_offset, constant, offset);
    }
    if (constant_integer(binary->lhs, &constant) &&
        affine_loop_offset(binary->rhs, loop, domain, &base_offset)) {
      return checked_add(constant, base_offset, offset);
    }
  }
  if (binary->op_type == BinaryOpType::sub &&
      affine_loop_offset(binary->lhs, loop, domain, &base_offset) &&
      constant_integer(binary->rhs, &constant)) {
    return checked_subtract(base_offset, constant, offset);
  }
  return false;
}

Stmt *pointer_origin(Stmt *stmt) {
  while (auto *matrix = stmt ? stmt->cast<MatrixPtrStmt>() : nullptr) {
    stmt = matrix->origin;
  }
  return stmt;
}

bool is_thread_private_pointer(Stmt *stmt) {
  auto *origin = pointer_origin(stmt);
  auto *allocation = origin ? origin->cast<AllocaStmt>() : nullptr;
  return allocation != nullptr && !allocation->is_shared;
}

std::string merge_access(const std::string &lhs, const std::string &rhs) {
  if (lhs == rhs) {
    return lhs;
  }
  if (lhs == "opaque" || rhs == "opaque") {
    return "opaque";
  }
  if (lhs == "atomic" || rhs == "atomic") {
    return "atomic";
  }
  return "read_write";
}

bool direct_loop_index(Stmt *stmt, RangeForStmt *loop) {
  auto *index = stmt ? stmt->cast<LoopIndexStmt>() : nullptr;
  return index != nullptr && index->loop == loop && index->index == 0;
}

bool flattened_coordinate_offset(Stmt *stmt,
                                 RangeForStmt *loop,
                                 BinaryOpType coordinate_op,
                                 std::int64_t *inner_extent,
                                 std::int64_t *offset) {
  auto *binary = stmt ? stmt->cast<BinaryOpStmt>() : nullptr;
  if (binary != nullptr && binary->op_type == coordinate_op &&
      direct_loop_index(binary->lhs, loop) &&
      constant_integer(binary->rhs, inner_extent) && *inner_extent > 0) {
    *offset = 0;
    return true;
  }
  if (binary != nullptr && coordinate_op == BinaryOpType::mod &&
      binary->op_type == BinaryOpType::sub &&
      direct_loop_index(binary->lhs, loop)) {
    auto *product = binary->rhs->cast<BinaryOpStmt>();
    if (product != nullptr && product->op_type == BinaryOpType::mul) {
      auto matches_product = [&](Stmt *quotient_stmt, Stmt *extent_stmt) {
        std::int64_t product_extent = 0;
        auto *quotient = quotient_stmt->cast<BinaryOpStmt>();
        return quotient != nullptr &&
               quotient->op_type == BinaryOpType::floordiv &&
               direct_loop_index(quotient->lhs, loop) &&
               constant_integer(quotient->rhs, inner_extent) &&
               constant_integer(extent_stmt, &product_extent) &&
               *inner_extent > 0 && product_extent == *inner_extent;
      };
      if (matches_product(product->lhs, product->rhs) ||
          matches_product(product->rhs, product->lhs)) {
        *offset = 0;
        return true;
      }
    }
  }
  if (binary == nullptr) {
    return false;
  }
  std::int64_t base_extent = 0;
  std::int64_t base_offset = 0;
  std::int64_t constant = 0;
  if (binary->op_type == BinaryOpType::add) {
    if (flattened_coordinate_offset(binary->lhs, loop, coordinate_op,
                                    &base_extent, &base_offset) &&
        constant_integer(binary->rhs, &constant) &&
        checked_add(base_offset, constant, offset)) {
      *inner_extent = base_extent;
      return true;
    }
    if (constant_integer(binary->lhs, &constant) &&
        flattened_coordinate_offset(binary->rhs, loop, coordinate_op,
                                    &base_extent, &base_offset) &&
        checked_add(constant, base_offset, offset)) {
      *inner_extent = base_extent;
      return true;
    }
  }
  if (binary->op_type == BinaryOpType::sub &&
      flattened_coordinate_offset(binary->lhs, loop, coordinate_op,
                                  &base_extent, &base_offset) &&
      constant_integer(binary->rhs, &constant) &&
      checked_subtract(base_offset, constant, offset)) {
    *inner_extent = base_extent;
    return true;
  }
  return false;
}

struct EffectSummary {
  std::string access;
  int iteration_rank{0};
  std::int64_t flattened_inner_extent{0};
  std::set<std::vector<std::int64_t>> affine_offsets;
  std::set<std::vector<std::int64_t>> read_offsets;
  std::set<std::vector<std::int64_t>> write_offsets;
  bool affine{true};
};

class MetadataVisitor final : public BasicStmtVisitor {
 public:
  MetadataVisitor(RangeForStmt *loop,
                  const GraphKernelIterationDomain &domain)
      : loop_(loop), domain_(domain) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  using BasicStmtVisitor::visit;

  void visit(Stmt *stmt) override {
    if (auto *global = stmt->cast<GlobalPtrStmt>()) {
      // Frontend stores conservatively carry activate=true even for a fully
      // dense SNode path. Dense/root nodes are permanently resident, so that
      // flag has no lifecycle side effect; the enclosing GlobalStoreStmt is
      // still recorded as a write below. Sparse paths remain fail-closed.
      if (global->activate &&
          (global->snode == nullptr || !global->snode->is_path_all_dense)) {
        block("sparse_activation", "opaque_access");
      }
      return;
    }
    if (stmt->has_global_side_effect()) {
      block("unsupported_side_effect", "opaque");
    }
  }

  void preprocess_container_stmt(Stmt *stmt) override {
    if (stmt == loop_ || stmt->is<IfStmt>()) {
      return;
    }
    block("nested_control_flow", "control_flow");
  }

  void visit(GlobalLoadStmt *stmt) override {
    record(stmt->src, "read");
  }

  void visit(AccelerationStructurePtrStmt *stmt) override {
    auto *argument = stmt->arg_load_stmt->cast<ArgLoadStmt>();
    if (argument == nullptr) {
      block("unknown_acceleration_structure", "opaque_access");
      return;
    }
    // Traversal reads the whole AS, not an affine element of an array. Keep
    // this resource effect without inventing a synchronization operation or
    // allowing pointwise fusion to erase the AS dependency.
    acceleration_structure_reads_.insert(argument->arg_id);
  }

  void visit(GlobalStoreStmt *stmt) override {
    record(stmt->dest, "write");
  }

  void visit(AtomicOpStmt *stmt) override {
    // Frontend augmented assignments may survive until this pre-offload stage
    // as atomics on a thread-private local accumulator. They neither publish a
    // Graph resource effect nor synchronize different loop iterations. Keep
    // global, external, shared, and unknown atomics fail-closed below.
    if (is_thread_private_pointer(stmt->dest)) {
      return;
    }
    block("atomic_effect", "atomic");
    record(stmt->dest, "atomic");
  }

  bool blocked() const {
    return !blocker_.empty();
  }

  const std::string &blocker() const {
    return blocker_;
  }

  const std::vector<std::string> &side_effects() const {
    return side_effects_;
  }

  void finalize(GraphKernelIterationDomain *domain) {
    if (blocked()) {
      return;
    }
    int rank = 0;
    std::int64_t inner_extent = 0;
    for (const auto &[_, summary] : effects_) {
      if (!summary.affine || summary.iteration_rank <= 0) {
        block("non_pointwise_access", "opaque_access");
        return;
      }
      if (rank == 0) {
        rank = summary.iteration_rank;
        inner_extent = summary.flattened_inner_extent;
      } else if (summary.iteration_rank != rank ||
                 (rank == 2 &&
                  summary.flattened_inner_extent != inner_extent)) {
        block("inconsistent_iteration_mapping", "opaque_access");
        return;
      }
    }
    if (rank != 2) {
      return;
    }
    if (domain->kind != "constant_range" || domain->begin != 0 ||
        domain->end <= 0 || inner_extent <= 0 ||
        domain->end % inner_extent != 0) {
      block("unknown_flattened_ndrange", "opaque_access");
      return;
    }
    std::vector<std::int64_t> origin;
    for (const auto &[_, summary] : effects_) {
      if (summary.write_offsets.empty()) {
        continue;
      }
      if (summary.write_offsets.size() != 1) {
        block("non_pointwise_output", "opaque_access");
        return;
      }
      const auto &candidate = *summary.write_offsets.begin();
      if (origin.empty()) {
        origin = candidate;
      } else if (origin != candidate) {
        block("inconsistent_iteration_origin", "opaque_access");
        return;
      }
    }
    if (origin.size() != 2) {
      block("missing_iteration_origin", "opaque_access");
      return;
    }
    logical_origin_ = origin;
    domain->logical_origin = origin;
    domain->logical_shape = {domain->end / inner_extent, inner_extent};
  }

  std::vector<GraphKernelResourceEffect> effects() const {
    std::vector<GraphKernelResourceEffect> result;
    result.reserve(effects_.size() + acceleration_structure_reads_.size());
    for (const auto &[key, summary] : effects_) {
      GraphKernelResourceEffect effect;
      effect.resource_kind = std::get<0>(key);
      effect.arg_id = std::get<1>(key);
      effect.snode_tree_id = std::get<2>(key);
      effect.snode_id = std::get<3>(key);
      effect.is_grad = std::get<4>(key);
      effect.access = summary.access;
      if (summary.affine && !summary.affine_offsets.empty() &&
          (summary.iteration_rank != 2 || logical_origin_.size() == 2)) {
        const int rank = summary.iteration_rank;
        effect.footprint.iteration_rank = rank;
        effect.footprint.affine_coefficients.assign(
            rank, std::vector<std::int64_t>(rank, 0));
        for (int axis = 0; axis < rank; ++axis) {
          effect.footprint.affine_coefficients[axis][axis] = 1;
        }
        std::set<std::vector<std::int64_t>> normalized;
        for (auto offset : summary.affine_offsets) {
          if (rank == 2) {
            for (int axis = 0; axis < rank; ++axis) {
              offset[axis] -= logical_origin_[axis];
            }
          }
          normalized.insert(std::move(offset));
        }
        effect.footprint.affine_index_offsets.assign(normalized.begin(),
                                                     normalized.end());
        effect.footprint.halo.assign(rank, {0, 0});
        for (int axis = 0; axis < rank; ++axis) {
          auto minimum = (*normalized.begin())[axis];
          auto maximum = minimum;
          for (const auto &offset : normalized) {
            minimum = std::min(minimum, offset[axis]);
            maximum = std::max(maximum, offset[axis]);
          }
          effect.footprint.halo[axis] = {minimum, maximum};
        }
        const std::vector<std::int64_t> zero(rank, 0);
        const bool exact = normalized.size() == 1 &&
                           *normalized.begin() == zero;
        effect.footprint.pattern =
            exact ? "exact_pointwise"
                  : (normalized.size() == 1 ? "affine" : "stencil");
        if (rank == 1) {
          for (const auto &offset : normalized) {
            effect.footprint.affine_offsets.push_back(offset.front());
          }
        } else {
          effect.footprint.affine_offsets =
              normalized.count(zero) ? zero : *normalized.begin();
          effect.footprint.contiguous_axis = rank - 1;
        }
        effect.footprint.reuse_class = exact ? "none" : "neighbor";
      }
      result.push_back(std::move(effect));
    }
    for (const auto &arg_id : acceleration_structure_reads_) {
      GraphKernelResourceEffect effect;
      effect.resource_kind = "argument";
      effect.arg_id = arg_id;
      effect.access = "read";
      result.push_back(std::move(effect));
    }
    return result;
  }

  bool elementwise() const {
    if (!acceleration_structure_reads_.empty()) {
      return false;
    }
    return std::all_of(
        effects_.begin(), effects_.end(), [&](const auto &item) {
          const auto &summary = item.second;
          if (!summary.affine || summary.affine_offsets.size() != 1) {
            return false;
          }
          auto offset = *summary.affine_offsets.begin();
          if (summary.iteration_rank == 2) {
            if (logical_origin_.size() != 2) {
              return false;
            }
            for (int axis = 0; axis < 2; ++axis) {
              offset[axis] -= logical_origin_[axis];
            }
          }
          return std::all_of(offset.begin(), offset.end(),
                             [](std::int64_t value) { return value == 0; });
        });
  }

 private:
  void block(const std::string &reason, const std::string &side_effect) {
    if (blocker_.empty()) {
      blocker_ = reason;
    }
    if (!side_effect.empty() &&
        std::find(side_effects_.begin(), side_effects_.end(), side_effect) ==
            side_effects_.end()) {
      side_effects_.push_back(side_effect);
    }
  }

  void add_effect(const ResourceKey &key,
                  const std::string &access,
                  bool affine,
                  const std::vector<std::int64_t> &offset = {},
                  std::int64_t flattened_inner_extent = 0) {
    auto [it, inserted] = effects_.emplace(key, EffectSummary{});
    if (inserted) {
      it->second.access = access;
      it->second.affine = affine;
      it->second.iteration_rank = static_cast<int>(offset.size());
      it->second.flattened_inner_extent = flattened_inner_extent;
    }
    if (!inserted) {
      it->second.access = merge_access(it->second.access, access);
      it->second.affine =
          it->second.affine && affine &&
          it->second.iteration_rank == static_cast<int>(offset.size()) &&
          (offset.size() != 2 ||
           it->second.flattened_inner_extent == flattened_inner_extent);
    }
    if (affine) {
      it->second.affine_offsets.insert(offset);
      if (access == "read") {
        it->second.read_offsets.insert(offset);
      } else if (access == "write") {
        it->second.write_offsets.insert(offset);
      }
    }
  }

  bool affine_indices(const std::vector<Stmt *> &indices,
                      int external_dimensions,
                      std::vector<std::int64_t> *offset,
                      std::int64_t *flattened_inner_extent) const {
    // A physics map commonly iterates records in the first dimension while
    // selecting a vector/matrix component dynamically. Runtime affine views
    // require positive-stride unique mappings, so trailing indices remain
    // inside record |i| and cannot introduce a cross-iteration dependency.
    // The leading index must remain an integer affine shift of |i|. Wrapped
    // stencils, gathers, and permutations are rejected even when every
    // trailing component is row-local.
    if (indices.empty() ||
        indices.size() < static_cast<std::size_t>(external_dimensions)) {
      return false;
    }
    if (external_dimensions == 1) {
      std::int64_t scalar_offset = 0;
      if (!affine_loop_offset(indices.front(), loop_, domain_,
                              &scalar_offset)) {
        return false;
      }
      *offset = {scalar_offset};
      *flattened_inner_extent = 0;
      return true;
    }
    if (external_dimensions != 2) {
      return false;
    }
    std::int64_t outer_extent = 0;
    std::int64_t inner_extent = 0;
    std::int64_t outer_offset = 0;
    std::int64_t inner_offset = 0;
    if (!flattened_coordinate_offset(indices[0], loop_,
                                     BinaryOpType::floordiv, &outer_extent,
                                     &outer_offset) ||
        !flattened_coordinate_offset(indices[1], loop_, BinaryOpType::mod,
                                     &inner_extent, &inner_offset) ||
        outer_extent != inner_extent) {
      return false;
    }
    *offset = {outer_offset, inner_offset};
    *flattened_inner_extent = inner_extent;
    return true;
  }

  void record(Stmt *pointer, const std::string &access) {
    pointer = pointer_origin(pointer);
    if (auto *external =
            pointer ? pointer->cast<ExternalPtrStmt>() : nullptr) {
      auto *base = external->base_ptr->cast<ArgLoadStmt>();
      std::vector<std::int64_t> offset;
      std::int64_t flattened_inner_extent = 0;
      if (base == nullptr ||
          !affine_indices(external->indices, external->ndim, &offset,
                          &flattened_inner_extent)) {
        add_effect({"opaque", {}, -1, -1, false}, "opaque", false);
        block("non_pointwise_access", "opaque_access");
        return;
      }
      add_effect({"argument", base->arg_id, -1, -1, external->is_grad},
                 access, true, offset, flattened_inner_extent);
      return;
    }
    if (auto *global = pointer ? pointer->cast<GlobalPtrStmt>() : nullptr) {
      const bool sparse_activation =
          global->activate &&
          (global->snode == nullptr || !global->snode->is_path_all_dense);
      std::int64_t offset = 0;
      if (sparse_activation || global->indices.size() != 1 ||
          !affine_loop_offset(global->indices.front(), loop_, domain_,
                              &offset)) {
        add_effect({"opaque", {}, -1, -1, false}, "opaque", false);
        block(sparse_activation ? "sparse_activation" :
                                  "non_pointwise_access",
              "opaque_access");
        return;
      }
      add_effect({"snode", {}, global->snode->get_snode_tree_id(),
                  global->snode->id, false},
                 access, true, {offset});
      return;
    }
    add_effect({"opaque", {}, -1, -1, false}, "opaque", false);
    block("unsupported_pointer", "opaque_access");
  }

  RangeForStmt *loop_{nullptr};
  const GraphKernelIterationDomain &domain_;
  std::map<ResourceKey, EffectSummary> effects_;
  std::set<std::vector<int>> acceleration_structure_reads_;
  std::vector<std::int64_t> logical_origin_;
  std::vector<std::string> side_effects_;
  std::string blocker_;
};

}  // namespace

GraphKernelMetadata analyze_graph_kernel_metadata(IRNode *root,
                                                  const Kernel *kernel) {
  GraphKernelMetadata result;
  result.available = true;
  result.blocker.clear();
  if (root == nullptr || kernel == nullptr || !root->is<Block>()) {
    result.blocker = "invalid_ir";
    return result;
  }
  if (kernel->autodiff_mode != AutodiffMode::kNone || !kernel->rets.empty()) {
    result.blocker = "callable_boundary";
    return result;
  }

  RangeForStmt *loop = nullptr;
  for (const auto &statement : root->as<Block>()->statements) {
    if (auto *candidate = statement->cast<RangeForStmt>()) {
      if (loop != nullptr) {
        result.blocker = "multiple_iteration_domains";
        return result;
      }
      loop = candidate;
      continue;
    }
    if (statement->is_container_statement() ||
        statement->has_global_side_effect()) {
      result.blocker = "top_level_side_effect";
      return result;
    }
  }
  if (loop == nullptr) {
    result.blocker = "missing_range_for";
    return result;
  }
  if (loop->strictly_serialized || loop->reversed || loop->is_bit_vectorized) {
    result.blocker = "unsupported_iteration_mode";
    return result;
  }

  result.iteration_domain = iteration_domain(loop);
  if (result.iteration_domain.kind == "unknown") {
    result.blocker = "unknown_iteration_domain";
    return result;
  }

  MetadataVisitor visitor(loop, result.iteration_domain);
  root->accept(&visitor);
  visitor.finalize(&result.iteration_domain);
  result.effects = visitor.effects();
  result.side_effects = visitor.side_effects();
  if (visitor.blocked()) {
    result.blocker = visitor.blocker();
    result.synchronization = true;
    return result;
  }
  result.opaque = false;
  result.elementwise = visitor.elementwise();
  result.synchronization = false;
  result.blocker.clear();
  return result;
}

}  // namespace taichi::lang
