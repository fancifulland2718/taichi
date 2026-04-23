#include "taichi/runtime/llvm/llvm_opt_pipeline.h"

#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/IndVarSimplify.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Scalar/LoopStrengthReduce.h"
#include "llvm/Transforms/Scalar/SeparateConstOffsetFromGEP.h"
#include "llvm/Transforms/Utils/LCSSA.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"

namespace taichi::lang {

void run_module_opt_pipeline(llvm::Module &module,
                             llvm::TargetMachine *target_machine,
                             const LLVMOptPipelineOptions &opts) {
  llvm::PipelineTuningOptions pto;
  pto.LoopVectorization = opts.loop_vectorize;
  pto.SLPVectorization = opts.slp_vectorize;

  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  llvm::PassBuilder pb(target_machine, pto);

  // Let the target register its own PassBuilder callbacks (equivalent
  // to the legacy `TargetMachine::adjustPassManager`). NVPTX, AMDGPU
  // and other targets rely on this to insert backend-specific IR
  // transforms (e.g. NVVM reflect) into the optimization pipeline.
  if (target_machine) {
    target_machine->registerPassBuilderCallbacks(pb);
  }

  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  llvm::ModulePassManager mpm =
      pb.buildPerModuleDefaultPipeline(opts.opt_level);

  if (opts.run_post_gep_passes) {
    // Mirror the original Taichi post-O3 tweak: LoopStrengthReduce and
    // IndVars are loop passes, SeparateConstOffsetFromGEP and EarlyCSE
    // are function passes. LoopSimplify + LCSSA are required canonical
    // forms for the loop pipeline when we are no longer riding on the
    // `PassManagerBuilder` helper that used to add them implicitly.
    llvm::LoopPassManager lpm;
    lpm.addPass(llvm::LoopStrengthReducePass());
    lpm.addPass(llvm::IndVarSimplifyPass());

    llvm::FunctionPassManager fpm;
    fpm.addPass(llvm::LoopSimplifyPass());
    fpm.addPass(llvm::LCSSAPass());
    fpm.addPass(llvm::createFunctionToLoopPassAdaptor(
        std::move(lpm), /*UseMemorySSA=*/false,
        /*UseBlockFrequencyInfo=*/false));
    fpm.addPass(llvm::SeparateConstOffsetFromGEPPass());
    fpm.addPass(llvm::EarlyCSEPass(/*UseMemorySSA=*/true));

    mpm.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(fpm)));
  }

  mpm.run(module, mam);
}

}  // namespace taichi::lang
