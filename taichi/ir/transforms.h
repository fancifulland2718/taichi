#pragma once

#include <atomic>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include "taichi/ir/control_flow_graph.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/pass.h"
#include "taichi/transforms/check_out_of_bound.h"
#include "taichi/transforms/constant_fold.h"
#include "taichi/transforms/inlining.h"
#include "taichi/transforms/lower_access.h"
#include "taichi/transforms/make_block_local.h"
#include "taichi/transforms/make_mesh_block_local.h"
#include "taichi/transforms/demote_mesh_statements.h"
#include "taichi/transforms/simplify.h"
#include "taichi/common/trait.h"
#include "taichi/program/function.h"

namespace taichi::lang {

class ScratchPads;

class Function;

// IR passes
namespace irpass {

// ---------------------------------------------------------------------------
// `bool` return-value contract for IR-mutating passes (P-Compile-1 phase 2-B)
// ---------------------------------------------------------------------------
// Every pass below that returns `bool` follows a single, strict contract:
//
//   true   <=>  the pass wrote at least one IR Stmt (insert / erase / replace
//               / mutate field). The IR is no longer bit-identical to its
//               input.
//   false  <=>  the IR is bit-identical to the input (no Stmt was added,
//               removed, replaced, or mutated).
//
// This contract is depended on by:
//   * driver-level dirty tracking in compile_to_offloads.cpp
//     (`pipeline_dirty`, P-Compile-1 phase 1) — used to skip downstream
//     `full_simplify` calls when the pipeline is provably clean;
//   * the `g_full_simplify_run` / `g_full_simplify_skipped` /
//     `g_fs_entries` / `g_fs_noop` / `g_fs_iters` profiling counters
//     (P-Compile-1 phase 2-A, exposed via pybind as
//     `_ti_core.get_full_simplify_stats` / `_ti_core.get_fs_inner_stats`);
//   * the outer-loop convergence check in `simplify.cpp::full_simplify`
//     (`if (!modified) break;`);
//   * the future debug verifier sandwich (`CompileConfig::fused_pass_verify`,
//     planned for phase 2-B onward).
//
// **Any future short-circuit / early-exit path inside these passes MUST
// preserve the contract**: the returned bool must be bit-identical to what
// the full path would return. A pass that early-exits with `return false`
// while skipping work that would have set the bool to `true` is a
// correctness bug — it lies to driver-level dirty tracking, breaks
// profiling, and lets downstream passes operate on stale IR. Reordering
// sub-passes inside a multi-pass driver (such as full_simplify) is only
// safe if convergence is preserved AND the final returned bool still
// reflects "did anything change".
// ---------------------------------------------------------------------------

void re_id(IRNode *root);
void flag_access(IRNode *root);
void eliminate_immutable_local_vars(IRNode *root);
bool scalarize(IRNode *root, bool half2_optimization_enabled = false);
void lower_matrix_ptr(IRNode *root, bool force_scalarize = false);
bool die(IRNode *root);
bool simplify(IRNode *root, const CompileConfig &config);
bool cfg_optimization(
    IRNode *root,
    bool after_lower_access,
    bool autodiff_enabled,
    bool real_matrix_enabled,
    const std::optional<ControlFlowGraph::LiveVarAnalysisConfig>
        &lva_config_opt = std::nullopt);
bool alg_simp(IRNode *root, const CompileConfig &config);
bool demote_operations(IRNode *root, const CompileConfig &config);
bool binary_op_simplify(IRNode *root, const CompileConfig &config);
bool whole_kernel_cse(IRNode *root);
bool extract_constant(IRNode *root, const CompileConfig &config);
bool unreachable_code_elimination(IRNode *root);
bool loop_invariant_code_motion(IRNode *root, const CompileConfig &config);
bool cache_loop_invariant_global_vars(IRNode *root,
                                      const CompileConfig &config);
// Returns true iff any inner pass actually mutated the IR. Callers may use
// this together with `CompileConfig::use_fused_passes` to short-circuit
// subsequent full_simplify / type_check calls when no IR-mutating pass has
// run since (P-Compile-1 phase 1).
bool full_simplify(IRNode *root,
                   const CompileConfig &config,
                   const FullSimplifyPass::Args &args);
void print(IRNode *root,
           std::string *output = nullptr,
           bool print_ir_dbg_info = false);
std::function<void(const std::string &)> make_pass_printer(
    bool verbose,
    bool print_ir_dbg_info,
    const std::string &kernel_name,
    IRNode *ir);
void frontend_type_check(IRNode *root);
void lower_ast(IRNode *root);
void type_check(IRNode *root, const CompileConfig &config);
bool inlining(IRNode *root,
              const CompileConfig &config,
              const InliningPass::Args &args);
void bit_loop_vectorize(IRNode *root);
void slp_vectorize(IRNode *root);
void replace_all_usages_with(IRNode *root, Stmt *old_stmt, Stmt *new_stmt);
bool check_out_of_bound(IRNode *root,
                        const CompileConfig &config,
                        const CheckOutOfBoundPass::Args &args);
// Returns true iff the IR was structurally modified (any new statement
// inserted). Callers may use this to skip a subsequent full_simplify pass when
// no other transformation has run since the previous simplify.
bool handle_external_ptr_boundary(IRNode *root, const CompileConfig &config);
void make_thread_local(IRNode *root, const CompileConfig &config);
std::unique_ptr<ScratchPads> initialize_scratch_pad(OffloadedStmt *root);
void make_block_local(IRNode *root,
                      const CompileConfig &config,
                      const MakeBlockLocalPass::Args &args);
void make_cpu_multithreaded_range_for(IRNode *root,
                                      const CompileConfig &config);
void make_mesh_thread_local(IRNode *root,
                            const CompileConfig &config,
                            const MakeBlockLocalPass::Args &args);
void make_mesh_block_local(IRNode *root,
                           const CompileConfig &config,
                           const MakeMeshBlockLocal::Args &args);
void demote_mesh_statements(IRNode *root,
                            const CompileConfig &config,
                            const DemoteMeshStatements::Args &args);
bool remove_loop_unique(IRNode *root);
bool remove_range_assumption(IRNode *root);
bool lower_access(IRNode *root,
                  const CompileConfig &config,
                  const LowerAccessPass::Args &args);
void auto_diff(IRNode *root,
               const CompileConfig &config,
               AutodiffMode autodiffMode,
               bool use_stack = false);
/**
 * Check whether the kernel obeys the autodiff limitation e.g., gloabl data
 * access rule
 */
void differentiation_validation_check(IRNode *root,
                                      const CompileConfig &config,
                                      const std::string &kernel_name);
/**
 * Determine all adaptive AD-stacks' size. This pass is idempotent, i.e.,
 * there are no side effects if called more than once or called when not needed.
 * @return Whether the IR is modified, i.e., whether there exists adaptive
 * AD-stacks before this pass.
 */
bool determine_ad_stack_size(IRNode *root, const CompileConfig &config);
bool constant_fold(IRNode *root);
void associate_continue_scope(IRNode *root, const CompileConfig &config);
void offload(IRNode *root, const CompileConfig &config);
bool transform_statements(
    IRNode *root,
    std::function<bool(Stmt *)> filter,
    std::function<void(Stmt *, DelayedIRModifier *)> transformer);
/**
 * @param root The IR root to be traversed.
 * @param filter A function which tells if a statement need to be replaced.
 * @param generator If a statement |s| need to be replaced, generate a new
 * statement |s1| with the argument |s|, insert |s1| to where |s| is defined,
 * remove |s|'s definition, and replace all usages of |s| with |s1|.
 * @return Whether the IR is modified.
 */
bool replace_and_insert_statements(
    IRNode *root,
    std::function<bool(Stmt *)> filter,
    std::function<std::unique_ptr<Stmt>(Stmt *)> generator);
/**
 * @param finder If a statement |s| need to be replaced, find the existing
 * statement |s1| with the argument |s|, remove |s|'s definition, and replace
 * all usages of |s| with |s1|.
 */
bool replace_statements(IRNode *root,
                        std::function<bool(Stmt *)> filter,
                        std::function<Stmt *(Stmt *)> finder);
void demote_dense_struct_fors(IRNode *root);
void demote_no_access_mesh_fors(IRNode *root);
bool demote_atomics(IRNode *root, const CompileConfig &config);
void reverse_segments(IRNode *root);  // for autograd
void detect_read_only(IRNode *root);
void optimize_bit_struct_stores(IRNode *root,
                                const CompileConfig &config,
                                AnalysisManager *amgr);

ENUM_FLAGS(ExternalPtrAccess){NONE = 0, READ = 1, WRITE = 2};

/**
 * Checks the access to external pointers in an offload.
 *
 * @param val1
 *   The offloaded statement to check
 *
 * @return
 *   The analyzed result.
 */
std::unordered_map<std::vector<int>,
                   ExternalPtrAccess,
                   hashing::Hasher<std::vector<int>>>
detect_external_ptr_access_in_task(OffloadedStmt *offload);

// compile_to_offloads does the basic compilation to create all the offloaded
// tasks of a Taichi kernel.
void compile_to_offloads(IRNode *ir,
                         const CompileConfig &config,
                         const Kernel *kernel,
                         bool verbose,
                         AutodiffMode autodiff_mode,
                         bool ad_use_stack,
                         bool start_from_ast);

void offload_to_executable(IRNode *ir,
                           const CompileConfig &config,
                           const Kernel *kernel,
                           bool verbose,
                           bool determine_ad_stack_size,
                           bool lower_global_access,
                           bool make_thread_local,
                           bool make_block_local);
// compile_to_executable fully covers compile_to_offloads, and also does
// additional optimizations so that |ir| can be directly fed into codegen.
void compile_to_executable(IRNode *ir,
                           const CompileConfig &config,
                           const Kernel *kernel,
                           AutodiffMode autodiff_mode,
                           bool ad_use_stack,
                           bool verbose,
                           bool lower_global_access = true,
                           bool make_thread_local = false,
                           bool make_block_local = false,
                           bool start_from_ast = true);
// Compile a function with some basic optimizations
void compile_function(IRNode *ir,
                      const CompileConfig &config,
                      Function *func,
                      AutodiffMode autodiff_mode,
                      bool verbose,
                      Function::IRStage target_stage);

void compile_taichi_functions(IRNode *ir,
                              const CompileConfig &compile_config,
                              Function::IRStage target_stage);

// P-Compile-1 phase 2-A profiling — counts how many `full_simplify` calls in
// `offload_to_executable` actually ran versus were short-circuited by the
// `pipeline_dirty` tracker. Counters are process-global and updated only by
// the driver in compile_to_offloads.cpp. Use `reset_full_simplify_stats()`
// before a benchmark, and `get_full_simplify_stats(run, skipped)` after.
void get_full_simplify_stats(uint64_t *run, uint64_t *skipped);
void reset_full_simplify_stats();

// P-Compile-1 phase 2-A inner profiling — counts `full_simplify()` entries at
// the function level (across ALL call sites in the codebase) and how many
// returned `any_modified == false` (i.e. the IR was already at the simplify
// fixed point on entry). Also tracks total outer-loop iterations consumed.
// Used to drive the Phase 2-B go/no-go decision: if `noop_returns / entries`
// is low, the dirty-tracker driver-level skip already captures the easy
// wins and deeper internal pass fusion is the only remaining lever.
void get_fs_inner_stats(uint64_t *entries,
                        uint64_t *noop_returns,
                        uint64_t *total_iterations);
void reset_fs_inner_stats();
}  // namespace irpass

}  // namespace taichi::lang
