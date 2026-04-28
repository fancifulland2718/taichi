#include "taichi/ir/ir.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/system/profiler.h"

#include <typeindex>

namespace taichi::lang {

// Build the reverse def-use map: for each stmt, which stmts reference it as
// an operand.  O(N) to build; allows O(direct-users) MarkUndone instead of
// the previous O(N) full-IR traversal.
class BuildUsesMap : public BasicStmtVisitor {
 public:
  using UsesMap = std::unordered_map<int, std::vector<Stmt *>>;

  explicit BuildUsesMap(UsesMap &uses) : uses_(uses) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void visit(Stmt *stmt) override {
    for (Stmt *op : stmt->get_operands()) {
      if (op)
        uses_[op->instance_id].push_back(stmt);
    }
  }

  void preprocess_container_stmt(Stmt *stmt) override {
    for (Stmt *op : stmt->get_operands()) {
      if (op)
        uses_[op->instance_id].push_back(stmt);
    }
  }

  static void run(IRNode *root, UsesMap &uses) {
    uses.clear();
    BuildUsesMap builder(uses);
    root->accept(&builder);
  }

 private:
  UsesMap &uses_;
};

// Whole Kernel Common Subexpression Elimination
class WholeKernelCSE : public BasicStmtVisitor {
 private:
  std::unordered_set<int> visited_;
  // each scope corresponds to an unordered_set
  std::vector<std::unordered_map<std::size_t, std::unordered_set<Stmt *> > >
      visible_stmts_;
  DelayedIRModifier modifier_;

  // Reverse def-use map: stmt instance_id -> stmts that use it as operand.
  // Rebuilt once per inner iteration; used for O(direct-users) MarkUndone.
  BuildUsesMap::UsesMap uses_;

 public:
  using BasicStmtVisitor::visit;

  WholeKernelCSE() {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  bool is_done(Stmt *stmt) {
    return visited_.find(stmt->instance_id) != visited_.end();
  }

  void set_done(Stmt *stmt) {
    visited_.insert(stmt->instance_id);
  }

  // Mark all direct users of `replaced` as needing reprocessing, then
  // transfer ownership of those users to `replacement` in the uses_ map so
  // that subsequent eliminations of `replacement` can find them.
  void mark_undone_and_transfer(Stmt *replaced, Stmt *replacement) {
    auto it = uses_.find(replaced->instance_id);
    if (it == uses_.end())
      return;
    auto &users = it->second;
    auto &rep_users = uses_[replacement->instance_id];
    for (Stmt *user : users) {
      visited_.erase(user->instance_id);
      rep_users.push_back(user);
    }
    users.clear();
  }

  static std::size_t operand_hash(const Stmt *stmt) {
    std::size_t hash_code{0};
    auto hash_type =
        std::hash<std::type_index>{}(std::type_index(typeid(stmt)));
    if (stmt->is<GlobalPtrStmt>() || stmt->is<LoopUniqueStmt>()) {
      // special cases in common_statement_eliminable()
      return hash_type;
    }
    auto op = stmt->get_operands();
    for (auto &x : op) {
      if (x == nullptr)
        continue;
      // Hash the addresses of the operand pointers.
      hash_code =
          (hash_code * 33) ^
          (std::hash<unsigned long>{}(reinterpret_cast<unsigned long>(x)));
    }
    return hash_type ^ hash_code;
  }

  static bool common_statement_eliminable(Stmt *this_stmt, Stmt *prev_stmt) {
    // Is this_stmt eliminable given that prev_stmt appears before it and has
    // the same type with it?
    if (this_stmt->type() != prev_stmt->type())
      return false;
    if (this_stmt->is<GlobalPtrStmt>()) {
      auto this_ptr = this_stmt->as<GlobalPtrStmt>();
      auto prev_ptr = prev_stmt->as<GlobalPtrStmt>();
      return irpass::analysis::definitely_same_address(this_ptr, prev_ptr) &&
             (this_ptr->activate == prev_ptr->activate || prev_ptr->activate);
    }
    if (this_stmt->is<ExternalPtrStmt>()) {
      auto this_ptr = this_stmt->as<ExternalPtrStmt>();
      auto prev_ptr = prev_stmt->as<ExternalPtrStmt>();
      return irpass::analysis::definitely_same_address(this_ptr, prev_ptr);
    }
    if (this_stmt->is<LoopUniqueStmt>()) {
      auto this_loop_unique = this_stmt->as<LoopUniqueStmt>();
      auto prev_loop_unique = prev_stmt->as<LoopUniqueStmt>();
      if (irpass::analysis::same_value(this_loop_unique->input,
                                       prev_loop_unique->input)) {
        // Merge the "covers" information into prev_loop_unique.
        // Notice that this_loop_unique->covers is corrupted here.
        prev_loop_unique->covers.insert(this_loop_unique->covers.begin(),
                                        this_loop_unique->covers.end());
        return true;
      }
      return false;
    }
    return irpass::analysis::same_statements(this_stmt, prev_stmt);
  }

  void visit(Stmt *stmt) override {
    if (!stmt->common_statement_eliminable())
      return;
    // container_statement does not need to be CSE-ed
    if (stmt->is_container_statement())
      return;
    // Generic visitor for all CSE-able statements.
    std::size_t hash_value = operand_hash(stmt);
    if (is_done(stmt)) {
      visible_stmts_.back()[hash_value].insert(stmt);
      return;
    }
    for (auto &scope : visible_stmts_) {
      for (auto &prev_stmt : scope[hash_value]) {
        if (common_statement_eliminable(stmt, prev_stmt)) {
          // Un-mark direct users of stmt using the O(users) fast path,
          // and transfer them to prev_stmt in the uses_ map.
          mark_undone_and_transfer(stmt, prev_stmt);
          stmt->replace_usages_with(prev_stmt);
          modifier_.erase(stmt);
          return;
        }
      }
    }
    visible_stmts_.back()[hash_value].insert(stmt);
    set_done(stmt);
  }

  void visit(Block *stmt_list) override {
    visible_stmts_.emplace_back();
    for (auto &stmt : stmt_list->statements) {
      stmt->accept(this);
    }
    visible_stmts_.pop_back();
  }

  void visit(IfStmt *if_stmt) override {
    if (if_stmt->true_statements) {
      if (if_stmt->true_statements->statements.empty()) {
        if_stmt->set_true_statements(nullptr);
      }
    }

    if (if_stmt->false_statements) {
      if (if_stmt->false_statements->statements.empty()) {
        if_stmt->set_false_statements(nullptr);
      }
    }

    // Move common statements at the beginning or the end of both branches
    // outside.
    // P9.B (2026-04-27): Loop the head/tail hoist so that K consecutive
    // matching prefix/suffix statements are extracted in a single visit.
    // Previously only 1 stmt at each end was hoisted per outer fixed-point
    // iteration, so a K-stmt shared prefix needed K outer iterations (each
    // rebuilding the def-use map). Physics-simulation pair-wise dispatch
    // commonly shares 3-10 stmt setup (relative position / mat3 transform /
    // contact frame) before diverging — looping cuts CSE outer iterations
    // proportionally on those kernels.
    if (if_stmt->true_statements && if_stmt->false_statements) {
      auto &true_clause = if_stmt->true_statements;
      auto &false_clause = if_stmt->false_statements;
      // Extend prefix hoist: keep extracting matching head stmts as long as
      // both clauses still have entries and the leading pair compares equal.
      while (!true_clause->statements.empty() &&
             !false_clause->statements.empty() &&
             irpass::analysis::same_statements(
                 true_clause->statements[0].get(),
                 false_clause->statements[0].get())) {
        // Directly modify this because it won't invalidate any iterators.
        auto common_stmt = true_clause->extract(0);
        irpass::replace_all_usages_with(false_clause.get(),
                                        false_clause->statements[0].get(),
                                        common_stmt.get());
        modifier_.insert_before(if_stmt, std::move(common_stmt));
        false_clause->erase(0);
      }
      // Extend suffix hoist: same idea on the tail. Note insert_after stacks
      // hoisted stmts in reverse extraction order, but the iteration itself
      // walks the suffix from outermost-tail inward, so the relative order of
      // hoisted stmts in the parent block is preserved.
      while (!true_clause->statements.empty() &&
             !false_clause->statements.empty() &&
             irpass::analysis::same_statements(
                 true_clause->statements.back().get(),
                 false_clause->statements.back().get())) {
        // Directly modify this because it won't invalidate any iterators.
        auto common_stmt = true_clause->extract((int)true_clause->size() - 1);
        irpass::replace_all_usages_with(false_clause.get(),
                                        false_clause->statements.back().get(),
                                        common_stmt.get());
        modifier_.insert_after(if_stmt, std::move(common_stmt));
        false_clause->erase((int)false_clause->size() - 1);
      }
    }

    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements)
      if_stmt->false_statements->accept(this);
  }

  static bool run(IRNode *node) {
    WholeKernelCSE eliminator;
    bool modified = false;
    while (true) {
      // Rebuild the reverse def-use map once per inner iteration (O(N)).
      // This lets mark_undone_and_transfer run in O(direct-users) rather than
      // the previous O(N) full-IR traversal (MarkUndone), giving an overall
      // improvement from O(N_replacements * N) to O(N + N_replacements *
      // avg_users) per iteration.
      BuildUsesMap::run(node, eliminator.uses_);
      node->accept(&eliminator);
      if (eliminator.modifier_.modify_ir())
        modified = true;
      else
        break;
    }
    return modified;
  }
};

namespace irpass {
bool whole_kernel_cse(IRNode *root) {
  TI_AUTO_PROF;
  // P9.B-2 (2026-04-27): per-OffloadedStmt segmentation.
  //
  // When the root is the post-lower_offload kernel block — a Block whose
  // children are all OffloadedStmts — CSE is provably independent across
  // offloads (each offload generates a separate launch; their stmts cannot
  // be operands of each other). Running the pass per-offload turns each
  // fixed-point iteration's BuildUsesMap rebuild from O(whole-IR) into
  // O(per-offload), and stops a tiny modification in offload K from
  // re-triggering an inner iteration that re-walks all other offloads.
  //
  // For pre-offload IR, ti.func bodies, or any other shape we keep the
  // original whole-root behavior — no functional change.
  if (auto *block = dynamic_cast<Block *>(root)) {
    if (!block->statements.empty()) {
      bool all_offloaded = true;
      for (auto &s : block->statements) {
        if (!s->is<OffloadedStmt>()) {
          all_offloaded = false;
          break;
        }
      }
      if (all_offloaded) {
        bool modified = false;
        for (auto &s : block->statements) {
          if (WholeKernelCSE::run(s.get()))
            modified = true;
        }
        return modified;
      }
    }
  }
  return WholeKernelCSE::run(root);
}
}  // namespace irpass

}  // namespace taichi::lang
