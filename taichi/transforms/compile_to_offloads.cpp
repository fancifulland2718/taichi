#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/pass.h"
#include "taichi/ir/visitors.h"
#include "taichi/program/compile_config.h"
#include "taichi/program/extension.h"
#include "taichi/program/function.h"
#include "taichi/program/kernel.h"
#include "taichi/util/lang_util.h"

#include <atomic>

namespace taichi::lang::irpass {
namespace {
std::atomic<uint64_t> g_full_simplify_run{0};
std::atomic<uint64_t> g_full_simplify_skipped{0};
}  // namespace
void get_full_simplify_stats(uint64_t *run, uint64_t *skipped) {
  if (run) *run = g_full_simplify_run.load(std::memory_order_relaxed);
  if (skipped) *skipped = g_full_simplify_skipped.load(std::memory_order_relaxed);
}
void reset_full_simplify_stats() {
  g_full_simplify_run.store(0, std::memory_order_relaxed);
  g_full_simplify_skipped.store(0, std::memory_order_relaxed);
  reset_fs_inner_stats();
}
}  // namespace taichi::lang::irpass


namespace taichi::lang {

namespace irpass {

void compile_to_offloads(IRNode *ir,
                         const CompileConfig &config,
                         const Kernel *kernel,
                         bool verbose,
                         AutodiffMode autodiff_mode,
                         bool ad_use_stack,
                         bool start_from_ast) {
  TI_AUTO_PROF;

  auto print = make_pass_printer(verbose, config.print_ir_dbg_info,
                                 kernel->get_name(), ir);
  print("Initial IR");

  if (!verbose && config.print_preprocessed_ir && start_from_ast) {
    TI_INFO("[{}] {}:", kernel->get_name(), "Preprocessed IR");
    std::cout << std::flush;
    irpass::re_id(ir);
    irpass::print(ir);
    std::cout << std::flush;
  }

  if (autodiff_mode == AutodiffMode::kReverse) {
    irpass::reverse_segments(ir);
    print("Segment reversed (for autodiff)");
  }

  if (start_from_ast) {
    irpass::frontend_type_check(ir);
    irpass::lower_ast(ir);
    print("Lowered");
  }

  irpass::compile_taichi_functions(ir, config,
                                   Function::IRStage::BeforeLowerAccess);
  irpass::analysis::gather_func_store_dests(ir);
  irpass::compile_taichi_functions(ir, config, Function::IRStage::OptimizedIR);
  irpass::analysis::gather_func_store_dests(ir);

  irpass::eliminate_immutable_local_vars(ir);
  print("Immutable local vars eliminated");

  irpass::type_check(ir, config);
  print("Typechecked");
  irpass::analysis::verify(ir);

  // TODO: strictly enforce bit vectorization for x86 cpu and CUDA now
  //       create a separate CompileConfig flag for the new pass
  if (arch_is_cpu(config.arch) || config.arch == Arch::cuda ||
      config.arch == Arch::amdgpu) {
    irpass::bit_loop_vectorize(ir);
    irpass::type_check(ir, config);
    print("Bit Loop Vectorized");
    irpass::analysis::verify(ir);
  }

  // Removes MatrixOfMatrixPtrStmt & MatrixOfGlobalPtrStmt
  irpass::lower_matrix_ptr(ir, config.force_scalarize_matrix);
  print("Matrix ptr lowered");

  if (config.force_scalarize_matrix) {
    irpass::scalarize(ir, false /*half2_optimization_enabled*/);

    irpass::die(ir);
    print("Scalarized");
  }

  irpass::full_simplify(
      ir, config,
      {false, /*autodiff_enabled*/ autodiff_mode != AutodiffMode::kNone,
       kernel->get_name(), verbose});
  print("Simplified I");
  irpass::analysis::verify(ir);

  // Track whether any IR-mutating pass has run since "Simplified I". If not,
  // "Simplified II" below is a no-op (full_simplify on an already-simplified
  // IR converges in one round) and can be skipped — saving one full pass over
  // every kernel that has no external arrays / no autodiff / no debug checks.
  // (P2.a: full_simplify dirty-flag dedupe.)
  bool dirty_since_simplify_i =
      irpass::handle_external_ptr_boundary(ir, config);
  print("External ptr boundary processed");

  if (is_extension_supported(config.arch, Extension::mesh)) {
    irpass::analysis::gather_meshfor_relation_types(ir);
  }

  if (config.debug && autodiff_mode == AutodiffMode::kCheckAutodiffValid) {
    // Check whether the kernel obeys the autodiff limitation e.g., gloabl data
    // access rule
    // This check should be performed in the forward kernel i.e., autodiff_mode
    // == AutodiffMode::kCheckAutodiffValid
    irpass::demote_atomics(ir, config);
    irpass::differentiation_validation_check(ir, config, kernel->get_name());
    irpass::analysis::verify(ir);
    // demote_atomics + differentiation_validation_check both mutate IR.
    dirty_since_simplify_i = true;
  }

  if (autodiff_mode == AutodiffMode::kReverse ||
      autodiff_mode == AutodiffMode::kForward) {
    // Remove local atomics here so that we don't have to handle their gradients
    irpass::demote_atomics(ir, config);

    irpass::full_simplify(
        ir, config,
        {false, /*autodiff_enabled*/ true, kernel->get_name(), verbose});
    irpass::auto_diff(ir, config, autodiff_mode, ad_use_stack);
    // TODO: Be carefull with the full_simplify when do high-order autodiff
    irpass::full_simplify(
        ir, config,
        {false, /*autodiff_enabled*/ false, kernel->get_name(), verbose});
    print("Gradient");
    irpass::analysis::verify(ir);
    // The two full_simplify calls above already left the IR in a fixed-point
    // state, so the post-flag_access full_simplify below is also redundant
    // for this branch. Keep the dirty flag false.
    dirty_since_simplify_i = false;
  }

  if (config.check_out_of_bound) {
    irpass::check_out_of_bound(ir, config, {kernel->get_name()});
    print("Bound checked");
    irpass::analysis::verify(ir);
    dirty_since_simplify_i = true;
  }

  irpass::flag_access(ir);
  print("Access flagged I");
  irpass::analysis::verify(ir);
  // flag_access only mutates GlobalPtrStmt::activate metadata; full_simplify
  // ignores that field entirely. So flag_access does NOT make the IR dirty
  // for the purposes of simplification.

  if (dirty_since_simplify_i) {
    irpass::full_simplify(
        ir, config,
        {false, /*autodiff_enabled*/ false, kernel->get_name(), verbose});
    print("Simplified II");
    irpass::analysis::verify(ir);
  } else {
    print("Simplified II (skipped: IR unchanged since Simplified I)");
  }

  irpass::offload(ir, config);
  print("Offloaded");
  irpass::analysis::verify(ir);
  // NOTE: There was an additional CFG pass here, removed in
  // https://github.com/taichi-dev/taichi/pull/8691
  irpass::flag_access(ir);
  print("Access flagged II");

  irpass::full_simplify(
      ir, config,
      {false, /*autodiff_enabled*/ false, kernel->get_name(), verbose});
  print("Simplified III");
  irpass::analysis::verify(ir);
}

void offload_to_executable(IRNode *ir,
                           const CompileConfig &config,
                           const Kernel *kernel,
                           bool verbose,
                           bool determine_ad_stack_size,
                           bool lower_global_access,
                           bool make_thread_local,
                           bool make_block_local) {
  TI_AUTO_PROF;

  auto print = make_pass_printer(verbose, config.print_ir_dbg_info,
                                 kernel->get_name(), ir);

  // TODO: This is just a proof that we can demote struct-fors after offloading.
  // Eventually we might want the order to be TLS/BLS -> demote struct-for.
  // For now, putting this after TLS will disable TLS, because it can only
  // handle range-fors at this point.

  auto amgr = std::make_unique<AnalysisManager>();

  print("Start offload_to_executable");
  irpass::analysis::verify(ir);

  // P-Compile-1 phase 1: track whether any IR-mutating pass has run since
  // the previous full_simplify. When `use_fused_passes` is true and this
  // flag is false at a full_simplify call site, the call can be skipped
  // (the IR is already at the simplify fixed-point). Default behavior
  // (flag off) keeps every full_simplify, matching pre-P-Compile-1.
  //
  // We start with `pipeline_dirty = true` because `offload(ir)` ran in the
  // caller and always rewrites the IR. Each pass below ORs in its own
  // "modified" return value; passes that don't return bool but don't
  // mutate IR (flag_access — see comment in compile_to_offloads_internal)
  // are intentionally NOT marked dirty.
  bool pipeline_dirty = true;
  const bool fused = config.use_fused_passes;

  if (config.detect_read_only) {
    irpass::detect_read_only(ir);
    print("Detect read-only accesses");
    // detect_read_only is a pure analysis pass — no IR mutation.
  }

  if (irpass::demote_atomics(ir, config))
    pipeline_dirty = true;
  print("Atomics demoted I");
  irpass::analysis::verify(ir);

  if (config.cache_loop_invariant_global_vars) {
    irpass::cache_loop_invariant_global_vars(ir, config);
    pipeline_dirty = true;
    print("Cache loop-invariant global vars");
  }

  if (config.demote_dense_struct_fors) {
    irpass::demote_dense_struct_fors(ir);
    irpass::type_check(ir, config);
    pipeline_dirty = true;
    print("Dense struct-for demoted");
    irpass::analysis::verify(ir);
  }

  if (config.make_cpu_multithreading_loop && arch_is_cpu(config.arch)) {
    irpass::make_cpu_multithreaded_range_for(ir, config);
    irpass::type_check(ir, config);
    pipeline_dirty = true;
    print("Make CPU multithreaded range-for");
    irpass::analysis::verify(ir);
  }

  if (is_extension_supported(config.arch, Extension::mesh) &&
      config.demote_no_access_mesh_fors) {
    irpass::demote_no_access_mesh_fors(ir);
    irpass::type_check(ir, config);
    pipeline_dirty = true;
    print("No-access mesh-for demoted");
    irpass::analysis::verify(ir);
  }

  if (make_thread_local) {
    irpass::make_thread_local(ir, config);
    pipeline_dirty = true;
    print("Make thread local");
  }

  if (is_extension_supported(config.arch, Extension::mesh)) {
    irpass::make_mesh_thread_local(ir, config, {kernel->get_name()});
    pipeline_dirty = true;
    print("Make mesh thread local");
    if (config.make_mesh_block_local && config.arch == Arch::cuda) {
      irpass::make_mesh_block_local(ir, config, {kernel->get_name()});
      pipeline_dirty = true;
      const bool sx_modified = irpass::full_simplify(
          ir, config,
          {false, /*autodiff_enabled*/ false, kernel->get_name(), verbose});
      pipeline_dirty = sx_modified;
      print("Simplified X");
    }
  }

  if (make_block_local) {
    irpass::make_block_local(ir, config, {kernel->get_name(), verbose});
    pipeline_dirty = true;
    print("Make block local");
  }

  if (is_extension_supported(config.arch, Extension::mesh)) {
    irpass::demote_mesh_statements(ir, config, {kernel->get_name()});
    pipeline_dirty = true;
    print("Demote mesh statements");
  }

  if (irpass::demote_atomics(ir, config))
    pipeline_dirty = true;
  print("Atomics demoted II");
  irpass::analysis::verify(ir);

  if (is_extension_supported(config.arch, Extension::quant) &&
      config.quant_opt_atomic_demotion) {
    irpass::analysis::gather_uniquely_accessed_bit_structs(ir, amgr.get());
  }

  if (irpass::remove_range_assumption(ir))
    pipeline_dirty = true;
  print("Remove range assumption");

  if (irpass::remove_loop_unique(ir))
    pipeline_dirty = true;
  print("Remove loop_unique");
  irpass::analysis::verify(ir);

  if (lower_global_access) {
    if (!fused || pipeline_dirty) {
      const bool sa_modified = irpass::full_simplify(
          ir, config,
          {false, /*autodiff_enabled*/ false, kernel->get_name(), verbose});
      pipeline_dirty = sa_modified;
      g_full_simplify_run.fetch_add(1, std::memory_order_relaxed);
      print("Simplified before lower access");
    } else if (config.fused_pass_verify) {
      // P-Compile-1 phase 2-B fallback: dirty tracking believes the IR is
      // a simplify fixed-point. Re-run full_simplify and verify it agrees
      // (returns false). On disagreement, fall back to using the freshly-
      // simplified IR and emit a warning so the buggy dirty-tracking site
      // can be located.
      const bool sa_modified = irpass::full_simplify(
          ir, config,
          {false, /*autodiff_enabled*/ false, kernel->get_name(), verbose});
      if (sa_modified) {
        TI_WARN(
            "fused_pass_verify: dirty tracking missed an IR-mutating pass "
            "before \"Simplified before lower access\" (kernel='{}'); "
            "falling back to the freshly simplified IR. Please report this "
            "with reproducer.",
            kernel->get_name());
        pipeline_dirty = true;
      }
      g_full_simplify_run.fetch_add(1, std::memory_order_relaxed);
      print("Simplified before lower access (verified)");
    } else {
      g_full_simplify_skipped.fetch_add(1, std::memory_order_relaxed);
      print("Simplified before lower access (skipped: pipeline clean)");
    }
    if (irpass::lower_access(ir, config, {kernel->no_activate, true}))
      pipeline_dirty = true;
    print("Access lowered");
    irpass::analysis::verify(ir);

    if (irpass::die(ir))
      pipeline_dirty = true;
    print("DIE");
    irpass::analysis::verify(ir);

    irpass::flag_access(ir);
    // flag_access mutates GlobalPtrStmt::activate. whole_kernel_cse (a
    // full_simplify sub-pass, see whole_kernel_cse.cpp
    // common_statement_eliminable) consults `activate` when deciding
    // whether two GlobalPtrStmts are CSE-mergeable, so a flag_access
    // run can expose new CSE opportunities. Mark the pipeline dirty so
    // the downstream "Simplified IV" call is not short-circuited away.
    pipeline_dirty = true;
    print("Access flagged III");
    irpass::analysis::verify(ir);
  }

  if (irpass::demote_operations(ir, config))
    pipeline_dirty = true;
  print("Operations demoted");

  if (!fused || pipeline_dirty) {
    const bool s4_modified = irpass::full_simplify(
        ir, config,
        {lower_global_access, /*autodiff_enabled*/ false, kernel->get_name(),
         verbose});
    pipeline_dirty = s4_modified;
    g_full_simplify_run.fetch_add(1, std::memory_order_relaxed);
    print("Simplified IV");
  } else if (config.fused_pass_verify) {
    const bool s4_modified = irpass::full_simplify(
        ir, config,
        {lower_global_access, /*autodiff_enabled*/ false, kernel->get_name(),
         verbose});
    if (s4_modified) {
      TI_WARN(
          "fused_pass_verify: dirty tracking missed an IR-mutating pass "
          "before \"Simplified IV\" (kernel='{}'); falling back to the "
          "freshly simplified IR. Please report this with reproducer.",
          kernel->get_name());
      pipeline_dirty = true;
    }
    g_full_simplify_run.fetch_add(1, std::memory_order_relaxed);
    print("Simplified IV (verified)");
  } else {
    g_full_simplify_skipped.fetch_add(1, std::memory_order_relaxed);
    print("Simplified IV (skipped: pipeline clean)");
  }

  if (determine_ad_stack_size) {
    irpass::determine_ad_stack_size(ir, config);
    print("Autodiff stack size determined");
  }

  if (is_extension_supported(config.arch, Extension::quant)) {
    irpass::optimize_bit_struct_stores(ir, config, amgr.get());
    print("Bit struct stores optimized");
  }

  bool half2_optimization_enabled =
      (config.arch == Arch::cuda && config.half2_vectorization &&
       !get_custom_cuda_library_path().empty());
  if (config.real_matrix_scalarize) {
    if (irpass::scalarize(ir, half2_optimization_enabled)) {
      irpass::die(ir);
      print("DIE");

      // Remove redundant MatrixInitStmt inserted during scalarization
      irpass::full_simplify(
          ir, config,
          {false, /*autodiff_enabled*/ false, kernel->get_name(), verbose});
      print("Scalarized");
    }
  }

  // Final field registration correctness & type checking
  irpass::type_check(ir, config);
  irpass::analysis::verify(ir);
}

void compile_to_executable(IRNode *ir,
                           const CompileConfig &config,
                           const Kernel *kernel,
                           AutodiffMode autodiff_mode,
                           bool ad_use_stack,
                           bool verbose,
                           bool lower_global_access,
                           bool make_thread_local,
                           bool make_block_local,
                           bool start_from_ast) {
  TI_AUTO_PROF;

  compile_to_offloads(ir, config, kernel, verbose, autodiff_mode, ad_use_stack,
                      start_from_ast);

  offload_to_executable(
      ir, config, kernel, verbose,
      /*determine_ad_stack_size=*/autodiff_mode == AutodiffMode::kReverse &&
          ad_use_stack,
      lower_global_access, make_thread_local, make_block_local);
}

void compile_function(IRNode *ir,
                      const CompileConfig &config,
                      Function *func,
                      AutodiffMode autodiff_mode,
                      bool verbose,
                      Function::IRStage target_stage) {
  TI_AUTO_PROF;

  auto current_stage = func->ir_stage();
  auto print = make_pass_printer(verbose, config.print_ir_dbg_info,
                                 func->get_name(), ir);
  print("Initial IR");

  if (target_stage >= Function::IRStage::BeforeLowerAccess &&
      current_stage < Function::IRStage::BeforeLowerAccess) {
    if (autodiff_mode == AutodiffMode::kReverse) {
      irpass::reverse_segments(ir);
      print("Segment reversed (for autodiff)");
    }

    if (current_stage < Function::IRStage::InitialIR) {
      irpass::frontend_type_check(ir);
      irpass::lower_ast(ir);
      print("Lowered");
    }

    // Removes MatrixOfMatrixPtrStmt & MatrixOfGlobalPtrStmt
    irpass::lower_matrix_ptr(ir, config.force_scalarize_matrix);
    print("Matrix ptr lowered");

    irpass::demote_atomics(ir, config);
    print("Atomics demoted");
    irpass::associate_continue_scope(ir, config);
    print("Associated continue scope");
    func->set_ir_stage(Function::IRStage::BeforeLowerAccess);
  }

  if (config.force_scalarize_matrix) {
    irpass::scalarize(ir, false /*half2_optimization_enabled*/);
  }

  if (target_stage >= Function::IRStage::OptimizedIR &&
      current_stage < Function::IRStage::OptimizedIR) {
    irpass::lower_access(ir, config, {{}, true});
    print("Access lowered");
    irpass::analysis::verify(ir);

    irpass::die(ir);
    print("DIE");
    irpass::analysis::verify(ir);

    irpass::flag_access(ir);
    print("Access flagged III");
    irpass::analysis::verify(ir);

    irpass::type_check(ir, config);
    print("Typechecked");

    irpass::demote_operations(ir, config);
    print("Operations demoted");

    if (config.real_matrix_scalarize) {
      if (irpass::scalarize(ir)) {
        // Remove redundant MatrixInitStmt inserted during scalarization
        irpass::die(ir);
        print("Scalarized");
      }
    }

    irpass::full_simplify(ir, config,
                          {true, autodiff_mode != AutodiffMode::kNone,
                           func->get_name(), verbose});
    print("Simplified");
    irpass::analysis::verify(ir);
    func->set_ir_stage(Function::IRStage::OptimizedIR);
  }
}

}  // namespace irpass

}  // namespace taichi::lang
