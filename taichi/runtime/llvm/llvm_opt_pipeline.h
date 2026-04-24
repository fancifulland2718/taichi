// Shared LLVM New-PassManager optimization pipeline used by Taichi JIT
// backends (CPU, CUDA/PTX, AMDGPU/HSACO, DX12/DXIL).
//
// LLVM 17 removed `llvm::PassManagerBuilder` and its legacy IPO pipeline
// populator in favour of the New Pass Manager (`llvm::PassBuilder`).
// This file centralises the replacement so each backend only has to
// describe its tuning parameters (opt level, vectorize flags, target
// machine) instead of duplicating ~30 lines of boilerplate.
//
// The code path here is fully New-PM and therefore remains valid
// through at least LLVM 22 — only the codegen emission step
// (`TargetMachine::addPassesToEmitFile`) still requires the legacy PM,
// which each backend keeps isolated at the end of its pipeline.
#pragma once

#include <string>

#include "llvm/IR/Module.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Target/TargetMachine.h"

namespace taichi::lang {

/// Map an integer llvm_opt_level (0-3) to the matching llvm::OptimizationLevel.
/// Values out of range are clamped: negative → O0, >3 → O3.
inline llvm::OptimizationLevel llvm_opt_level_from_int(int level) {
  switch (level) {
    case 0:  return llvm::OptimizationLevel::O0;
    case 1:  return llvm::OptimizationLevel::O1;
    case 2:  return llvm::OptimizationLevel::O2;
    default: return llvm::OptimizationLevel::O3;
  }
}

/// Effective opt level after considering `compile_tier`.
///
/// `compile_tier == "fast"` forces the opt level down so the full
/// P1.d/P2.c tier semantics line up — Vulkan already caps
/// `spv_opt_level` at 1 in that tier (see `kernel_compiler.cpp`), and
/// this mirrors that policy for CPU / CUDA / AMDGPU / DX12.
///
/// `min_level` is a backend-specific floor: CPU and DX12 tolerate
/// LLVM O0 fine, but NVPTX (CUDA) and AMDGCN codegen both rely on
/// mid-level legalization passes (e.g. StackSave/StackRestore lowering,
/// addrspace canonicalization) that are skipped at O0 — those backends
/// must pass `min_level=1` to stay correct under `tier=fast`.
///
/// Trade-off (measured on 16-kernel CPU cold compile, commit `ca2e062c8`):
///   - CPU `tier=fast` + O0: **~21% faster** cold compile, bit-exact
///     on simple arithmetic kernels; up to 2e-7 relative drift on
///     reassoc-sensitive accumulation kernels (well within Taichi's
///     1e-5 cross-backend numerical bar from P2 protocol).
///   - CUDA `tier=fast` + O1 floor: ~5-8% compile win (estimated),
///     avoids NVPTX codegen fatal errors at O0.
///   - Other tiers: returns `level` unchanged (default 3 = O3).
inline int effective_llvm_opt_level(int level,
                                    const std::string &tier,
                                    int min_level = 0) {
  if (tier == "fast") {
    return min_level > 0 ? min_level : 0;
  }
  return level;
}

struct LLVMOptPipelineOptions {
  // Optimization level. Mirrors the historical `PassManagerBuilder::OptLevel = 3`.
  llvm::OptimizationLevel opt_level = llvm::OptimizationLevel::O3;

  // Vectorization knobs. Forwarded to `PipelineTuningOptions` and hence
  // control whether LoopVectorize / SLPVectorize are run inside the
  // default pipeline built by `buildPerModuleDefaultPipeline`.
  bool loop_vectorize = true;
  bool slp_vectorize = true;

  // Re-run LoopStrengthReduce + IndVarSimplify +
  // SeparateConstOffsetFromGEP + EarlyCSE after the default pipeline.
  // This is a Taichi-specific tweak (see taichi-dev/taichi#5472) that
  // measurably improves GEP lowering for the GPU backends; CPU benefits
  // from it as well.
  bool run_post_gep_passes = true;
};

// Run the module-level optimization pipeline.
//
// `target_machine` may be null (in which case the pipeline runs without
// target-specific tuning), but every in-tree caller passes a valid
// machine so that `PassBuilder` sees an accurate `TargetIRAnalysis`.
void run_module_opt_pipeline(llvm::Module &module,
                             llvm::TargetMachine *target_machine,
                             const LLVMOptPipelineOptions &opts);

}  // namespace taichi::lang
