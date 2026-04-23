
#include "taichi/common/core.h"
#include "taichi/util/io.h"
#include "taichi/program/program.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/util/file_sequence_writer.h"
#include "taichi/runtime/llvm/llvm_context.h"
#include "taichi/runtime/llvm/llvm_opt_pipeline.h"

#include "dx12_llvm_passes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Function.h"

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Attributes.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Target/TargetMachine.h"
// PassManagerBuilder was removed in LLVM 17; see llvm_opt_pipeline.h.
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/GlobalVariable.h"

using namespace llvm;

namespace taichi::lang {
namespace directx12 {

const char *NumWorkGroupsCBName = "num_work_groups.cbuf";

const llvm::StringRef ShaderAttrKindStr = "hlsl.shader";

void mark_function_as_cs_entry(::llvm::Function *F) {
  F->addFnAttr(ShaderAttrKindStr, "compute");
}
bool is_cs_entry(::llvm::Function *F) {
  return F->hasFnAttribute(ShaderAttrKindStr);
}

void set_num_threads(llvm::Function *F, unsigned x, unsigned y, unsigned z) {
  const llvm::StringRef NumThreadsAttrKindStr = "hlsl.numthreads";
  std::string Str = llvm::formatv("{0},{1},{2}", x, y, z);
  F->addFnAttr(NumThreadsAttrKindStr, Str);
}

GlobalVariable *createGlobalVariableForResource(Module &M,
                                                const char *Name,
                                                llvm::Type *Ty) {
  auto *GV = new GlobalVariable(M, Ty, /*isConstant*/ false,
                                GlobalValue::LinkageTypes::ExternalLinkage,
                                /*Initializer*/ nullptr, Name);
  GV->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::None);
  return GV;
}

std::vector<uint8_t> global_optimize_module(llvm::Module *module,
                                            const CompileConfig &config) {
  TI_AUTO_PROF
  if (llvm::verifyModule(*module, &llvm::errs())) {
    module->print(llvm::errs(), nullptr);
    TI_ERROR("Module broken");
  }

  for (llvm::Function &F : module->functions()) {
    if (directx12::is_cs_entry(&F))
      continue;
    F.addFnAttr(llvm::Attribute::AlwaysInline);
  }
  // FIXME: choose shader model based on feature used.
  llvm::StringRef triple = "dxil-pc-shadermodel6.0-compute";
  module->setTargetTriple(triple);
  module->setSourceFileName("");
  std::string err_str;
  const llvm::Target *target =
      TargetRegistry::lookupTarget(triple.str(), err_str);
  TI_ERROR_UNLESS(target, err_str);

  TargetOptions options;
  if (config.fast_math) {
    options.AllowFPOpFusion = FPOpFusion::Fast;
    options.UnsafeFPMath = 1;
    options.NoInfsFPMath = 1;
    options.NoNaNsFPMath = 1;
  } else {
    options.AllowFPOpFusion = FPOpFusion::Strict;
    options.UnsafeFPMath = 0;
    options.NoInfsFPMath = 0;
    options.NoNaNsFPMath = 0;
  }
  options.HonorSignDependentRoundingFPMathOption = false;
  options.NoZerosInBSS = false;
  options.GuaranteedTailCallOpt = false;

  legacy::FunctionPassManager function_pass_manager(module);
  legacy::PassManager module_pass_manager;

  llvm::StringRef mcpu = "";
  std::unique_ptr<TargetMachine> target_machine(target->createTargetMachine(
      triple.str(), mcpu.str(), "", options, llvm::Reloc::PIC_,
      llvm::CodeModel::Small,
      config.opt_level > 0 ? llvm::CodeGenOptLevel::Aggressive
                           : llvm::CodeGenOptLevel::None));

  TI_ERROR_UNLESS(target_machine.get(), "Could not allocate target machine!");

  module->setDataLayout(target_machine->createDataLayout());

  // Phase 1: lower Taichi intrinsics via the legacy PM (these are
  // legacy ModulePass derivations living in taichi/codegen/dx12).
  {
    TI_PROFILER("llvm_pre_opt_taichi_passes");
    llvm::legacy::PassManager pre_opt_pm;
    pre_opt_pm.add(createTaichiIntrinsicLowerPass(&config));
    pre_opt_pm.run(*module);
  }

  // Phase 2: run the standard O3 optimization pipeline via the New
  // PassManager (PassManagerBuilder was removed in LLVM 17).
  {
    TI_PROFILER("llvm_module_opt_pipeline");
    LLVMOptPipelineOptions opts;
    opts.opt_level = llvm::OptimizationLevel::O3;
    opts.loop_vectorize = true;
    opts.slp_vectorize = true;
    opts.run_post_gep_passes = true;
    run_module_opt_pipeline(*module, target_machine.get(), opts);
  }

  // Phase 3: lower runtime context references and emit DXIL via the
  // legacy PM (codegen still requires legacy infrastructure).
  llvm::SmallString<256> str;
  llvm::raw_svector_ostream OS(str);
  {
    TI_PROFILER("llvm_emit_dxil");
    llvm::legacy::PassManager emit_pm;
    emit_pm.add(createTaichiRuntimeContextLowerPass());
    target_machine->addPassesToEmitFile(emit_pm, OS, nullptr,
                                        llvm::CodeGenFileType::ObjectFile);
    emit_pm.run(*module);
  }
  if (config.print_kernel_llvm_ir_optimized) {
    static FileSequenceWriter writer(
        "taichi_kernel_dx12_llvm_ir_optimized_{:04d}.ll",
        "optimized LLVM IR (DX12)");
    writer.write(module);
  }
  return std::vector<uint8_t>(str.begin(), str.end());
}

}  // namespace directx12
}  // namespace taichi::lang
