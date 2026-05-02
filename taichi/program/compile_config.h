#pragma once

#include "taichi/rhi/arch.h"
#include "taichi/util/lang_util.h"

namespace taichi::lang {

struct CompileConfig {
  Arch arch;
  bool debug;
  bool cfg_optimization;
  bool check_out_of_bound;
  bool validate_autodiff;
  int simd_width;
  int opt_level;
  int external_optimization_level;
  // LLVM backend: 0=O0 (fast compile, worse runtime) … 3=O3 (default, best
  // runtime). Exposed so users can trade compile-time speed against kernel
  // performance on first cold launch. Does NOT affect SPIR-V opt (use
  // external_optimization_level for that). Included in the offline-cache key.
  int llvm_opt_level;
  // P2.c: Compile-time optimization tier.
  // "fast"     — aggressively skip the most expensive IR passes
  //              (whole_kernel_cse entirely; keep single-shot LICM /
  //              cfg_optimization). Best cold-compile wall clock; may
  //              leave some redundant computations in the IR.
  // "balanced" — (default) P2.0/P2.a/P2.b behavior: whole_kernel_cse /
  //              cfg_optimization / loop_invariant_code_motion run only
  //              in the first full_simplify iteration.
  // "full"     — re-run the three expensive passes on every
  //              full_simplify iteration, matching the <=1.7.4 release
  //              behavior. Longest compile, occasionally squeezes out
  //              a few more redundant stmts.
  // Included in the offline-cache key.
  std::string compile_tier{"balanced"};
  // P-Compile-1 phase 1: short-circuit consecutive full_simplify /
  // type_check calls in compile_to_offloads.cpp when no IR-mutating pass
  // has run since the previous one. Default false → behavior is bit-
  // identical to the pre-P-Compile-1 pipeline (including the existing
  // P2.a `dirty_since_simplify_i` short-circuit, which is unconditional
  // and orthogonal). When true and TI_DEBUG is set, a verifier sandwich
  // re-runs the skipped pass and asserts it returns no-modification.
  // Included in the offline-cache key.
  bool use_fused_passes{false};
  // P-Compile-1 phase 2-B: when use_fused_passes is true, every skip
  // decision in compile_to_offloads.cpp is double-checked by re-running
  // the would-be-skipped full_simplify and verifying it returns false
  // (no IR change). On mismatch the pipeline emits an English warning
  // and falls back to using the freshly-run (modified) IR, so a stale
  // dirty-tracking decision can never produce a less-optimized kernel.
  // Default false (zero overhead). Not part of the offline-cache key
  // because it is a runtime safety knob, not a codegen toggle.
  bool fused_pass_verify{false};
  // V2 (2026-04-26): when true, the SPIR-V backend's KernelCodegen::run
  // dispatches per-task TaskCodegen + spvtools::Optimizer::Run on parallel
  // worker threads (cap = num_compile_threads). Each worker fetches its own
  // thread_local Optimizer cache (V3), so spvtools' single-instance
  // thread-safety constraints are not violated. Default false to keep
  // legacy serial behaviour for all existing users; opt-in via
  // ti.init(spirv_parallel_codegen=True). Not part of the offline-cache
  // key (per-task SPIR-V output must be byte-identical between OFF and ON).
  bool spirv_parallel_codegen{false};
  // V6 (2026-04-26): when true and spv_opt_level == 3, the SPIR-V backend
  // skips spvtools' CreateLoopUnrollPass(true) — the single most expensive
  // optimizer pass in the level-3 chain. Modern Vulkan drivers (NV / AMD /
  // Intel / Mesa) all unroll loops in their own shader compilers, so the
  // spvtools-side unroll is largely redundant and only inflates compile
  // time. Default false to preserve legacy SPIR-V output bit-exactly;
  // opt-in via ti.init(spirv_skip_loop_unroll=True). Not part of the
  // offline-cache key (treated like spirv_parallel_codegen — the user
  // explicitly accepts a different optimizer chain when toggling).
  bool spirv_skip_loop_unroll{false};
  // Phase 1c-D (taichi-forge 0.3.x): opt-in for the experimental sparse SNode
  // path on the Vulkan backend (bitmasked storage + listgen for depth-1
  // single/multi-axis bitmasked). Default false matches the vanilla taichi
  // 1.7.4 behaviour of refusing sparse on Vulkan. When true, Program ctor
  // calls set_vulkan_sparse_experimental(true) which makes
  // is_extension_supported(Arch::vulkan, Extension::sparse) return true; the
  // legacy env var TI_VULKAN_SPARSE=1 is still honoured for backwards
  // compatibility (see extension.cpp). Not part of the offline-cache key:
  // the resulting SPIR-V differs structurally only when the user actually
  // declares sparse SNodes, which already segregates the cache by SNode tree
  // hash.
  // §13 (2026-05-02): default flipped from false -> true after the full P4
  // matrix (g10_full_ndrange_pointer_3d / g10_inactive_read_zero /
  // vulkan_pointer_smoke / listgen / ported / deactivate_all / recycle /
  // race / c2_4b_demo_workload, 9 scripts) was verified PASS on the current
  // pyd. A one-shot TI_WARN at Program ctor advertises the experimental
  // status whenever the path is actually exercised on Arch::vulkan. Users
  // can opt out via ti.init(vulkan_sparse_experimental=False) which fully
  // restores the vanilla 1.7.4 "reject sparse on Vulkan" behaviour.
  bool vulkan_sparse_experimental{true};
  // G9.1 (taichi-forge 0.3.0): opt-in for the experimental quant_array /
  // bit_struct path on the Vulkan backend. Default false matches vanilla
  // taichi 1.7.4 (quant_array is LLVM-only). When true, Program ctor calls
  // set_vulkan_quant_experimental(true) which makes
  // is_extension_supported(Arch::vulkan, Extension::quant) and
  // is_extension_supported(Arch::vulkan, Extension::quant_basic) return
  // true; the env var TI_VULKAN_QUANT=1 is honoured as a fallback. Codegen
  // is delivered incrementally: unimplemented sites (atomic add on
  // bit_struct, bit-pack/unpack on SPIR-V, etc.) raise TI_NOT_IMPLEMENTED
  // rather than silently miscompiling. Not part of the offline-cache key:
  // the resulting SPIR-V only differs when the user actually declares
  // bit_struct fields, which already segregates the cache by SNode tree
  // hash. Production quant_array workloads should remain on cpu / cuda
  // until the Vulkan codegen completes.
  bool vulkan_quant_experimental{false};
  // B2 (2026-04-26): user-facing fine-grained SPIR-V optimizer pass
  // disable list. Each entry is a pass name (case-sensitive) matching one
  // of the spvtools::Create*Pass identifiers used in
  // spirv_codegen.cpp::get_thread_local_opt — without the "Create" prefix
  // and without the "Pass" suffix. Examples: "LoopUnroll",
  // "AggressiveDCE", "InlineExhaustive", "MergeReturn".
  // Default empty = full pass chain at the chosen spv_opt_level (legacy
  // behaviour, byte-identical SPIR-V output). Disabled passes are skipped
  // at Optimizer registration, so they cost neither register nor Run()
  // time. Different lists produce different SPIR-V bytes and so are
  // segregated in the offline cache (added to the cache key in
  // taichi/analysis/offline_cache_util.cpp).
  std::vector<std::string> spirv_disabled_passes;
  // V7 (2026-04-26): anti double-pool oversubscription in the batched
  // Program::compile_kernels path. The outer ParallelExecutor already
  // saturates `num_compile_threads` worker threads with kernel-level tasks;
  // the inner LLVM compilation_workers pool would then fan out each
  // kernel's offloads on top of that, producing N*M concurrent threads on
  // a machine with far fewer CPU cores. P5_并行编译.md §4 measured this as
  // a 0.89x CPU regression. When true, while a thread is acting as an
  // outer compile_kernels worker, KernelCodeGen::compile_kernel_to_module
  // bypasses the inner pool and processes offloads serially inline (same
  // path as the single-offload bypass). Default true after V8.e validated
  // the LLVMContext type-pollution fix (tests/p4/v8d_validate.py: 20/20
  // stress reps incl. scalar N=16 T=8, sincos N=8 T=8, multi-offload N=16
  // T=8). Opt-out via ti.init(compile_dag_scheduler=False). Not part of
  // the offline-cache key (pure scheduling toggle — emitted bytecode is
  // byte-identical between OFF and ON).
  bool compile_dag_scheduler{true};
  // B-2.b (2026-05): runtime gating for the four pointer-SNode allocator
  // strategy macros that B-1 / B-2.a baked into the contract POD. Default
  // values mirror the legacy CMake-ON state so behaviour is byte-equivalent
  // when the user does not set the kwargs. Opt-out toggles let the user
  // recover vanilla 1.7.4 / earlier-fork behaviour without rebuilding.
  //   - vulkan_pointer_freelist: G1.b 自由链表回收（true=ON，layout 预留 head/links）
  //   - vulkan_pointer_ambient_zone: G10-P2 inactive read 路由零页（true=ON）
  //   - vulkan_pointer_cas_marker: G1.a CAS-marker-first alloc 协议（true=ON）
  //   - vulkan_pointer_pool_fraction: 池容量收缩比例 (0,1]，1.0=不收缩
  // 各字段语义详见 compile_doc/SNode_Vulkan_规划.md §10.4.2。
  bool vulkan_pointer_freelist{true};
  bool vulkan_pointer_ambient_zone{true};
  bool vulkan_pointer_cas_marker{true};
  // C-1.b (2026-05)：默认值由 1.0 改为 -1.0 哨兵（"未设置"）。
  // 仅当 (0, 1) 严格开区间内显式赋值时启用 fraction 缩放；其它
  // (含 -1.0 默认 / 1.0 / 越界值) 一律走 worst-case，与 LLVM 后端
  // "按需分配，不预先估算" 的语义对齐。fraction 仅作为 *显式* 全局
  // 缩放工具存在；per-SNode 精确容量请用 vk_max_active。
  double vulkan_pointer_pool_fraction{-1.0};
  // B-3.b (2026-05): 当 SNodeTree 内恰好 1 个 pointer SNode 时，把该 pointer 的
  // pool 元数据切到独立 NodeAllocatorPool descriptor binding（B-3.b 仅做 plumbing：
  // 申请独立 DeviceAllocation + 注册 input_buffer，codegen 仍读 root_buffer 子区
  // 间，行为字节等价）。多 pointer / 嵌套 pointer 自动回退 root。
  // §13 (2026-05-02): default ON. Earlier flip-then-revert in this same
  // session blamed c2_4b_demo_workload.py's chunked-path failure on
  // independent_pool=true; root cause was actually that
  // gfx_program.cpp built `PointerLayoutPolicy` straight from this flag,
  // so chunked+max_chunks>1 would land at indep_pool=false and emit
  // BufferInfo(Root,...) (kStorageBuffer) for cells that had a recorded
  // ptr_to_chunk_idx_ entry, tripping the kChunkedArrayPtr assert in
  // spirv_codegen.cpp::at_buffer. Fix lives in gfx_program.cpp
  // (coerce_pointer_policy_for_chunked) and re-enables this default.
  bool vulkan_pointer_independent_pool{true};
  // C-2.1 (2026-05): pointer SNode device-side allocator 选择。append-only
  // 骨架：合法值 {"bump", "chunked"}。默认 "bump" = 路线 B 行为，byte-equivalent。
  // "chunked" 在 C-2.2 之前**不可用**，C-2.1 仅做 plumbing（CompileConfig →
  //  Python kwarg → PointerLayoutPolicy → SpirvAllocatorContract → BumpOnly
  //  Params → cache key），factory 端遇到 "chunked" 抛 TI_NOT_IMPLEMENTED，
  //  绝不 silent 降级回 bump（与 C-1.b 删除 OOC 静默 fallback 同一原则）。
  // 详见 compile_doc/SNode_Vulkan_规划.md §12.2.0 / §12.2.1。
  std::string vulkan_pointer_allocator_kind{"bump"};
  // C-2.4.a (2026-05): chunked allocator 允许的最大 chunk 数。默认 1
  // 与 C-2.3 完全等价（单 chunk = 整池）。C-2.4.a 仅将该字段 plumbing
  // 到 contract.max_chunks，后续 step ii–v 才会在 SPIR-V 侧发射
  // descriptor array of buffers 以跨 chunk 寻址。该字段进入 offline cache
  // key，避免同一 SNode 在 max_chunks 改变后复用旧缓存。
  uint32_t vulkan_pointer_max_chunks{1};
  // C-9 (2026-05): pointer SNode alloc 协议改为 deterministic slot mapping
  // （new_slot = idx_u32 + 1）。详见 compile_doc/SNode_Vulkan_规划.md §14。
  // 默认 ON：消除 G10-P1 触发的 spin-loop device-lost；layout 端自动 gating
  //   `capacity >= worst_capacity && allocator_kind == "bump"`，不满足时强制
  //   降级到 cas_marker 路径（与本节落地前字节等价）。设 false 也可整体降级。
  bool vulkan_pointer_deterministic_slot{true};
  int max_vector_width;
  bool print_preprocessed_ir;
  bool print_ir;
  bool print_accessor_ir;
  bool print_ir_dbg_info;
  bool serial_schedule;
  bool simplify_before_lower_access;
  bool lower_access;
  bool simplify_after_lower_access;
  bool move_loop_invariant_outside_if;
  // 2026-04-27 (P-Compile-8 默认值审计): fork-added pass，vanilla 1.7.4 无此项。
  // 实测 cold compile +5-15%（SNode-heavy kernel），runtime 仅 1-3% 帮助且仅作用于
  // 反复读取同一全局指针的窄场景。物理仿真主路径不敏感，故默认翻 false 对齐 vanilla。
  // 用户仍可显式 ti.init(cache_loop_invariant_global_vars=True) 打开。
  bool cache_loop_invariant_global_vars{false};
  // 2026-04-27 (P9.D): full_simplify outer fixed-point 改用 local→global
  // 分级调度：先把 local pass（extract_constant / binary_op_simplify /
  // constant_fold / die / alg_simp / simplify）跑到本地不动点，再跑一次
  // 全部 global pass（LICM / whole_kernel_cse / cfg_optimization），如
  // global 改动了 IR 则回到 local 不动点继续。语义与既有 outer-loop
  // 等价（每次 global 看到的都是 local 不动点 IR，只多不少候选），但
  // 砍掉了 "local 还在小改 → 又跑一次大 global" 的浪费。
  // P2.b/P2.c 的 first_iteration 跳过 global 是错的；本调度始终至少跑
  // 一次 global 来检查是否还有候选，避免当年的语义退化。
  // 灰度阶段默认 false；profile 验证无回归后翻 true。
  bool tiered_full_simplify{true};
  // 2026-04-28 (P9.E): full_simplify outer-loop global-pass cap.
  // Limits how many times each `full_simplify` outer iteration is allowed
  // to run the three expensive global passes (loop_invariant_code_motion,
  // whole_kernel_cse, cfg_optimization). Once the cap is reached the
  // outer loop only continues when local passes are still mutating the
  // IR; further global runs are skipped.
  //   1 (default) — equivalent to the historical `first_iteration` guard
  //                 P2.b/P2.c had: globals run exactly once per
  //                 full_simplify call. The IR after this still goes
  //                 through downstream passes (lower_access, demote_*,
  //                 ...), each of which calls full_simplify again, so the
  //                 globals will eventually see new candidates produced
  //                 by those passes — they are just not re-run within a
  //                 single full_simplify call after their candidate set
  //                 is already drained on iter 1.
  //   0           — unlimited (re-revert to pre-P9.E behaviour: globals
  //                 run every outer iteration until fixed-point).
  //   N > 1       — explicit cap for diagnostic / safety net use.
  // Per [optimization-workflow.instructions.md] §1.1, the runtime-
  // performance / correctness baseline is vanilla 1.7.4 (≥95% step
  // time / numerically equivalent). Cap=1 produces IR that is at least
  // as well-optimized as vanilla 1.7.4's first-outer-iter output (vanilla
  // also runs CSE/LICM/CFG on iter 1; vanilla then keeps iterating, but
  // §2.5 of compile_doc/性能回归根因分析.md established that the marginal
  // optimization beyond iter 1 is ≪0.5% on physics workloads while the
  // compile-time cost is 30-80%).
  int full_simplify_global_iter_cap{1};
  // 2026-04-28 (P9.A-2 / F2): auto-promote @ti.func from Python-side AST
  // inline expansion to is_real_function=True (C++ FuncCallStmt + per-
  // signature IR cache) once expansion wall time exceeds threshold.
  // LLVM-only (FuncCallStmt visitor exists only in codegen_llvm.cpp).
  // Default OFF; preserves vanilla 1.7.4 semantics. Cache key includes
  // both fields so toggling invalidates offline cache cleanly.
  bool auto_real_function{false};
  // Cumulative inline-expansion wall time (microseconds) per @ti.func that
  // triggers promotion. 1 ms ~ "function is being repeatedly traced for
  // multi-millisecond cost" — see compile_doc/优化总规划.md §P9.A.
  int auto_real_function_threshold_us{1000};
  // 2026-04-28 (P9.A-3 / F3): IR-level reverse fallback for auto_real_function.
  // After F2 promotes a func to FuncCallStmt, the inliner can selectively
  // inline back small callees so that "small + frequent" funcs do not pay
  // a perma-call cost. budget = max statement count of a callee for which
  // inlining is allowed. 0 = disabled (no inline-back, FuncCallStmt
  // preserved). Default OFF — F3 is plumbed but quiescent until F1
  // telemetry confirms the real distribution warrants activation.
  int auto_real_function_inline_budget{0};
  bool demote_dense_struct_fors;
  bool advanced_optimization;
  bool constant_folding;
  bool use_llvm;
  bool verbose_kernel_launches;
  bool kernel_profiler;
  bool timeline{false};
  bool verbose;
  bool fast_math;
  bool flatten_if;
  bool make_thread_local;
  bool make_block_local;
  bool detect_read_only;
  bool real_matrix_scalarize;
  bool force_scalarize_matrix;
  bool half2_vectorization;
  bool make_cpu_multithreading_loop;
  DataType default_fp;
  DataType default_ip;
  DataType default_up;
  std::string extra_flags;
  int default_cpu_block_dim;
  bool cpu_block_dim_adaptive;
  int default_gpu_block_dim;
  int gpu_max_reg;
  int ad_stack_size{0};  // 0 = adaptive
  // The default size when the Taichi compiler is unable to automatically
  // determine the autodiff stack size.
  int default_ad_stack_size{32};

  int saturating_grid_dim;
  int max_block_dim;
  int cpu_max_num_threads;
  int random_seed;

  // LLVM backend options:
  bool print_struct_llvm_ir;
  bool print_kernel_llvm_ir;
  bool print_kernel_llvm_ir_optimized;
  bool print_kernel_asm;
  bool print_kernel_amdgcn;

  // CUDA/AMDGPU backend options:
  float64 device_memory_GB;
  float64 device_memory_fraction;

  // Opengl backend options:
  bool allow_nv_shader_extension{true};

  bool quant_opt_store_fusion{true};
  bool quant_opt_atomic_demotion{true};

  // Mesh related.
  // MeshTaichi options
  bool make_mesh_block_local{true};
  bool optimize_mesh_reordered_mapping{true};
  bool mesh_localize_to_end_mapping{true};
  bool mesh_localize_from_end_mapping{false};
  bool mesh_localize_all_attr_mappings{false};
  bool demote_no_access_mesh_fors{true};
  bool experimental_auto_mesh_local{false};
  int auto_mesh_local_default_occupacy{4};

  // Offline cache options
  bool offline_cache{false};
  std::string offline_cache_file_path{get_repo_dir() + "ticache"};
  std::string offline_cache_cleaning_policy{
      "lru"};  // "never"|"version"|"lru"|"fifo"
  int offline_cache_max_size_of_files{100 * 1024 *
                                      1024};   // bytes, default: 100MB
  double offline_cache_cleaning_factor{0.25};  // [0.f, 1.f]

  int num_compile_threads{4};
  std::string vk_api_version;

  // R2.a: Vulkan launch args/ret buffer pool. Default OFF (vanilla behavior).
  // Only effective on gfx backends (vulkan/opengl/metal/dx). Buffers are
  // recycled across `synchronize()` boundaries; capacity bounded.
  bool vulkan_launch_buffer_pool{false};
  int vulkan_launch_buffer_pool_capacity{64};

  size_t cuda_stack_limit{0};

  CompileConfig();

  void fit();
};

extern TI_DLL_EXPORT CompileConfig default_compile_config;

}  // namespace taichi::lang
