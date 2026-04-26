#pragma once

#include "taichi/util/lang_util.h"

#include "taichi/codegen/spirv/snode_struct_compiler.h"
#include "taichi/codegen/spirv/kernel_utils.h"

#include <spirv-tools/libspirv.hpp>
#include <spirv-tools/optimizer.hpp>

namespace taichi::lang {

class Kernel;

namespace spirv {

class KernelCodegen {
 public:
  struct Params {
    std::string ti_kernel_name;
    const Kernel *kernel{nullptr};
    const IRNode *ir_root{nullptr};
    std::vector<CompiledSNodeStructs> compiled_structs;
    Arch arch;
    DeviceCapabilityConfig caps;
    bool enable_spv_opt{true};  // kept for back-compat; ignored when spv_opt_level>0
    // 0 = no SPIR-V optimisation
    // 1 = fast (dead-code / dead-branch only)
    // 2 = standard (+ inlining, mem2reg-equivalent)
    // 3 = full  (all passes; legacy default behaviour)
    int spv_opt_level{3};
    // V2 (2026-04-26): when true and there are >= 2 offload tasks, run
    // per-task TaskCodegen + spvtools::Optimizer::Run on a fan-out of
    // worker threads. SPIR-V output is byte-identical to the serial path.
    bool parallel_codegen{false};
    // Cap on workers when parallel_codegen is true. Mirrors
    // CompileConfig::num_compile_threads. Single-task kernels always run
    // serially regardless of this value.
    int num_compile_threads{4};
    // V6 (2026-04-26): when true and spv_opt_level == 3, the level-3 pass
    // chain skips spvtools::CreateLoopUnrollPass(true). Saves a meaningful
    // fraction of compile time for loop-heavy kernels at the cost of
    // letting the GPU driver do the unrolling instead. Output SPIR-V will
    // differ from the unrolled version, so we extend the thread_local
    // OptCacheKey with this flag to keep cached Optimizer instances
    // separate.
    bool skip_loop_unroll{false};
    // B2 (2026-04-26): per-kernel SPIR-V optimizer pass disable list.
    // Sourced from CompileConfig::spirv_disabled_passes; the
    // thread_local Optimizer cache (OptCacheKey) keys on a stable hash
    // of this list so disabled-set toggles produce distinct cached
    // Optimizer instances. Empty list = full pass chain at the chosen
    // spv_opt_level (legacy behaviour, byte-identical SPIR-V).
    std::vector<std::string> disabled_passes;
    // V8.b (2026-04-26): when true, the run() per-task fan-out checks
    // Program::in_compile_kernels_worker() and falls back to serial when
    // already inside an outer compile_kernels worker. Mirrors V7's LLVM
    // anti-double-pool behaviour. Symmetric with parallel_codegen so both
    // flags must be enabled to get the V2 inner pool, and the inner pool
    // is bypassed only when the outer pool is active.
    bool compile_dag_scheduler{false};
  };

  explicit KernelCodegen(const Params &params);

  void run(TaichiKernelAttributes &kernel_attribs,
           std::vector<std::vector<uint32_t>> &generated_spirv);

 private:
  Params params_;
  KernelContextAttributes ctx_attribs_;
  // V2 (2026-04-26): no per-instance Optimizer pointer. Each task body
  // (whether on caller thread or a worker thread) fetches its own
  // thread_local Optimizer/SpirvTools via the file-scope helper in
  // spirv_codegen.cpp, keyed by (target_env, spv_opt_level). This keeps
  // the parallel path lock-free and respects spvtools' single-instance
  // thread-safety contract.
  spv_target_env target_env_{SPV_ENV_VULKAN_1_0};
  spvtools::OptimizerOptions spirv_opt_options_;
};

}  // namespace spirv
}  // namespace taichi::lang
