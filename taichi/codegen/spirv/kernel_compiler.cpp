#include "taichi/codegen/spirv/kernel_compiler.h"

#include "taichi/ir/analysis.h"
#include "taichi/codegen/spirv/spirv_codegen.h"
#include "taichi/codegen/spirv/compiled_kernel_data.h"

namespace taichi::lang {
namespace spirv {

KernelCompiler::KernelCompiler(Config config) : config_(std::move(config)) {
}

KernelCompiler::IRNodePtr KernelCompiler::compile(
    const CompileConfig &compile_config,
    const Kernel &kernel_def) const {
  auto ir = irpass::analysis::clone(kernel_def.ir.get());
  irpass::compile_to_executable(ir.get(), compile_config, &kernel_def,
                                kernel_def.autodiff_mode,
                                /*ad_use_stack=*/false, compile_config.print_ir,
                                /*lower_global_access=*/true,
                                /*make_thread_local=*/false);
  return ir;
}

KernelCompiler::CKDPtr KernelCompiler::compile(
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &device_caps,
    const Kernel &kernel_def,
    IRNode &chi_ir) const {
  TI_TRACE("VK codegen for Taichi kernel={}", kernel_def.name);
  KernelCodegen::Params params;
  params.ti_kernel_name = kernel_def.name;
  params.kernel = &kernel_def;
  params.ir_root = &chi_ir;
  params.compiled_structs = *config_.compiled_struct_data;
  params.arch = compile_config.arch;
  params.caps = device_caps;
  // P1.d + V1: compile_tier-aware SPIR-V opt level.
  //   "fast"     — skip the spvtools optimizer entirely (level 0, 0 passes).
  //                V1 revision: previously capped at level 1 (3 cheap passes),
  //                but DCE/DeadBranchElim are cheap and their savings do not
  //                materialise over pass registration + Run() overhead on
  //                SPV-bound kernels. Going straight to 0 gives the largest
  //                compile-time win for the "fast" tier and is safe because
  //                TaskCodegen already emits legal SPIR-V (no optimiser
  //                required to pass Vulkan driver validation, and the
  //                spvtools validator is disabled anyway).
  //   "balanced" — no change vs legacy (default external_optimization_level=3,
  //                23 passes). Preserves default runtime perf.
  //   "full"     — no change vs legacy (23 passes).
  // Different tiers produce different artifacts and are segregated by the
  // P2.c offline-cache key (compile_tier is already serialized into the
  // hash; see taichi/analysis/offline_cache_util.cpp).
  int spv_level = compile_config.external_optimization_level;
  if (compile_config.compile_tier == "fast") {
    spv_level = 0;
  }
  params.enable_spv_opt = spv_level > 0;
  params.spv_opt_level = spv_level;
  // V2: opt-in parallel per-task SPIR-V codegen + spvtools::Optimizer::Run.
  // Default false in CompileConfig so behaviour is unchanged for users
  // who do not set ti.init(spirv_parallel_codegen=True).
  params.parallel_codegen = compile_config.spirv_parallel_codegen;
  params.num_compile_threads = std::max(1, compile_config.num_compile_threads);
  // V6: opt-in skip of CreateLoopUnrollPass at level 3.
  params.skip_loop_unroll = compile_config.spirv_skip_loop_unroll;
  // B2: user-facing fine-grained pass disable list. Empty by default.
  params.disabled_passes = compile_config.spirv_disabled_passes;
  // V8.b: anti-double-pool — when the surrounding compile_kernels outer
  // pool is active, suppress this inner per-task fan-out.
  params.compile_dag_scheduler = compile_config.compile_dag_scheduler;
  spirv::KernelCodegen codegen(params);
  spirv::CompiledKernelData::InternalData internal_data;
  codegen.run(internal_data.metadata.kernel_attribs,
              internal_data.src.spirv_src);
  internal_data.metadata.num_snode_trees = config_.compiled_struct_data->size();
  return std::make_unique<spirv::CompiledKernelData>(compile_config.arch,
                                                     internal_data);
}

}  // namespace spirv
}  // namespace taichi::lang
