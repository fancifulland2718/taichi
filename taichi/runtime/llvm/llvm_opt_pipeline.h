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

#include "llvm/IR/Module.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Target/TargetMachine.h"

namespace taichi::lang {

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
