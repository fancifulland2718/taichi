# P3 — Frontend IR size-control guardrails

Commit base: `c06bbf830` (V1/V2 Vulkan work).

Scope implemented (Python-only, no C++/wheel rebuild needed):

- **P3.a `unrolling_hard_limit`** — per-`ti.static(for ...)` hard cap. 0 = disabled.
  When an unrolled iteration count exceeds the cap, compile aborts with a
  `TaichiCompilationError` carrying the offending source line and the knob
  name, instead of spending tens of seconds expanding the body.
- **P3.a `unrolling_kernel_hard_limit`** — cumulative cap across all
  `ti.static` loops in one kernel/function compile. Catches pyramidal nested
  unrolls (e.g. 27³ = 19 683) whose individual loops look innocent.
- **P3.b `func_inline_depth_limit`** — hard cap on `@ti.func` inline
  recursion depth. Non-real `@ti.func` calls compound AST expansion; this
  fails fast when the chain exceeds the configured depth.

All three default to `0` (disabled) so the new build is drop-in compatible
with existing user kernels (parity verified below).

Skipped from the original P3 list (require C++ changes / rebuild + runtime
perf gating; deferred to a later pass):
- P3.c matrix scalarize volume-aware
- P3.d batch Python→C++ FFI

## Code changes

- [python/taichi/lang/misc.py](python/taichi/lang/misc.py) — `_SpecialConfig` + `env_spec` + dispatch to runtime for the 3 new knobs.
- [python/taichi/lang/impl.py](python/taichi/lang/impl.py) — `PyTaichi.__init__` defaults + `func_inline_depth` live counter.
- [python/taichi/lang/ast/ast_transformer_utils.py](python/taichi/lang/ast/ast_transformer_utils.py) — `ASTTransformerContext.unrolled_iterations` per-compile accumulator.
- [python/taichi/lang/ast/ast_transformer.py](python/taichi/lang/ast/ast_transformer.py) — new `ASTTransformer._check_unroll_hard_limit` helper, called once per iteration in both `build_static_for` branches.
- [python/taichi/lang/kernel_impl.py](python/taichi/lang/kernel_impl.py) — `Func.__call__` non-real path wraps `transform_tree` with a depth counter (try/finally balanced).

## Tests (`tests/p3/`)

| test | purpose | result |
| --- | --- | --- |
| [smoke_p3a.py](smoke_p3a.py) | static(range(8)) under hard_limit=128 must compile & produce 28·i | **OK** |
| [test_p3a_per_loop.py](test_p3a_per_loop.py) | static(range(100)) with hard_limit=16 must raise fast | **OK**, aborted in 14.9 ms |
| [test_p3a_kernel_total.py](test_p3a_kernel_total.py) | 8×8 nested static, per-loop=100 but kernel-total=32 must raise | **OK** |
| [test_p3b_depth.py](test_p3b_depth.py) | 3 nested @ti.func with depth_limit=2 must raise | **OK** |
| [parity_p3.py](parity_p3.py) | 3-backend numeric parity with/without budgets; cpu/cuda/vulkan | **OK**, Δ=0 default-vs-budgeted on all 3; Δ≤4e-6 cuda/vulkan vs cpu |

## Bench — early-abort latency (`bench_p3_abort.py`)

CPU arch; `unrolling_hard_limit=50` vs disabled. Runaway `ti.static(range(N))`:

|   N | baseline (s) | abort (s) |  speedup |
| ---:| ---:| ---:| ---:|
|  400 |        0.764 |     0.0134 |    57.0x |
|  800 |        2.460 |     0.0139 |   177.1x |
| 1600 |        9.035 |     0.0128 |   707.2x |

Baseline scales roughly O(N); abort time is constant (≈13 ms — the budget
check fires on the 51st iteration before body expansion balloons). At
N=1600 the user gets a clear error in 13 ms instead of waiting 9 s.

### Cold-process verification (`bench_p3_abort_cold.py`)

Each row spawns a fresh python interpreter — zero in-process cache. Inner
`dt` measures just the kernel compile + sync (excludes `import taichi`).

|   N | HL |  inner dt (s) | result |
| ---:| ---:| ---:| ---:|
|  800 |  0 |        2.499 | compiled + ran (baseline) |
|  800 | 50 |        0.023 | raised `TaichiCompilationError` |
| 1600 |  0 |        9.062 | compiled + ran (baseline) |
| 1600 | 50 |        0.023 | raised `TaichiCompilationError` |

**Semantics of the `raised` rows:** the guard is fail-fast, *not* silent
trimming. When the unroll count exceeds `unrolling_hard_limit`,
`ASTTransformer._check_unroll_hard_limit` throws `TaichiCompilationError`
before any IR/codegen runs — the kernel is **never executed with a reduced
unroll**. The user must either raise the limit or rewrite the loop. Default
is `unrolling_hard_limit=0` (disabled), so baseline behaviour is unchanged
unless the user opts in.

Baselines match the in-process bench within <1 % (2.46 s→2.50 s, 9.04 s→9.06 s):
the savings are genuinely cold-compile, not a JIT-warm artefact. The abort
floor is ~23 ms in a fresh interpreter vs ~13 ms when warm; the delta is the
one-time kernel-wrap / launcher initialisation inside `run()` on the first
call. Cold latency at N=1600 drops from 9.062 s to a 23 ms error = **≈394×
faster failure**, not a 394× compile speed-up for a working kernel.

## Parity vs V1/V2

- Default config (`unrolling_hard_limit=0`) → `ASTTransformer._check_unroll_hard_limit` is still called but both guards short-circuit on the 0-check. No semantic change — confirmed bit-exact on cpu/cuda/vulkan vs budgeted config and Δ≤4e-6 across backends on a 16-iter static-sin kernel.
- Existing `unrolling_limit=32` SyntaxWarning path preserved unchanged.

---

## P3.c — irpass::scalarize early-exit (C++, wheel rebuild)

**Correctness argument (why no user-facing opt-in is needed).** The four
sub-passes we skip (`Scalarize` / `ScalarizePointers` / `ExtractLocalPointers`
/ `FuseMatrixPtr`) are pure IR rewriters whose visitors only mutate matrix
statements. When the `HasMatrixStmt` pre-scan reports zero such statements,
those passes are provably no-ops on the input IR, so pre- and post-state
are identical — 3-backend bit-exact parity (`parity_p3.py` Δ=0) confirms
this empirically. This is a semantics-preserving short-circuit, not a
perf/accuracy trade-off, so it ships enabled with no knob.

Scope: pre-scan IR once; if no `TensorType` ret_types and no
`MatrixInitStmt` / `MatrixPtrStmt` / `MatrixOfGlobalPtrStmt` /
`MatrixOfMatrixPtrStmt`, return `false` immediately — skipping 4 later
sub-passes (`Scalarize` / `ScalarizePointers` / `ExtractLocalPointers`
/ `FuseMatrixPtr`). Provably semantics-preserving: when the pre-scan
reports zero matrix stmts, the 4 sub-passes have nothing to mutate, so the
post-state is identical to the pre-state.

### Bench (subprocess-per-row, `bench_p3c_scalar.py`)

Scalar-only saxpy-like kernel on CPU, `N = 1 << 20`. Compile wall-clock:

| unroll | compile dt (s) |
| ---:| ---:|
|   32 |         0.097 |
|  128 |         0.181 |
|  512 |         0.849 |

The early-exit path is reached at all 3 + 1 call sites
(`compile_to_offloads.cpp` L76/L317/L418, `make_block_local.cpp` L47).
Compared to pre-P3.c wheel the scalarize wrapper total drops from the sum
of 5 sub-pass walks to a single `HasMatrixStmt` scan — measured
6–34 μs per invocation via `TI_COMPILE_PROFILE` on realistic kernels.

### Parity

3-backend (cpu/cuda/vulkan) `parity_p3.py` and `smoke_p3a.py` pass
bit-exact on the freshly rebuilt wheel `taichi-1.8.0-cp310-cp310-win_amd64.whl`
(commit `8c1ceec6`): Δ=0 on default-vs-budgeted, Δ≤4e-6 across backends.

### P3.c correctness fix (commit `bfd6871fc`)

The initial P3.c landing (`274268544`) had a latent bug in the
`HasMatrixStmt` visitor. `BasicStmtVisitor` leaves
`invoke_default_visitor=false`, so the generic `visit(Stmt*)` override was
never dispatched to for any typed statement — the predicate always returned
`false` and the early-exit fired on **every** kernel. Scalar tests passed by
accident (scalarize is a no-op on scalar IR), but matrix kernels
miscompiled: `ti.Vector([...])` / `ti.Matrix(...) @ ti.Matrix(...)` /
`M[i]` field loads leaked through unscalarized into LLVM codegen, triggering
asserts like

    Floating-point arithmetic operators only work with floating-point types!
      %20 = fadd reassoc ninf nsz [3 x float] %15, %19

and (on CUDA) `Intrinsic has incorrect return type: ldg.global.i.a16f32.p0`.

Fix: `invoke_default_visitor=true` in the `HasMatrixStmt` constructor, so
every typed stmt falls through to the generic predicate.

Added regression test [parity_p3c_matrix.py](parity_p3c_matrix.py) — 3 cpu
cases, all bit-exact Δ=0 vs hand-computed reference:

| test                   | exercises                 | |Δ|    | result |
| :--------------------- | :------------------------ | ------:| :----- |
| `matrix_init_and_arith` | `MatrixInitStmt` + element arith | 0.000e+00 | OK |
| `matrix_field_matmul`   | `MatrixInitStmt` + `@` matmul    | 0.000e+00 | OK |
| `matrix_of_global_ptr`  | `MatrixOfGlobalPtrStmt` load/store | 0.000e+00 | OK |

Full P3 regression suite on the fixed wheel (commit `bfd6871fc`):

| test                         | result               |
| :--------------------------- | :------------------- |
| `parity_p3.py`               | Δ=0 default-vs-budgeted; Δ≤4e-6 cross-backend (cpu/cuda/vulkan) |
| `smoke_p3a.py`               | OK                   |
| `test_p3a_per_loop.py`       | OK, aborted 21.4 ms  |
| `test_p3a_kernel_total.py`   | OK                   |
| `test_p3b_depth.py`          | OK                   |
| `parity_p3c_matrix.py` (new) | 3/3 OK (see above)   |

### P3.c isolated A/B (P3.a hard-limit disabled, `bench_p3c_ab.py`)

To isolate the C++ early-exit benefit from the Python `_check_unroll_hard_limit`
short-circuit, we rebuilt two wheels from identical sources except for the
44-line `HasMatrixStmt` block in `taichi/transforms/scalarize.cpp`:

- **A = baseline** — `git checkout 15c155343 -- taichi/transforms/scalarize.cpp`, no P3.c.
- **B = P3.c** — HEAD `274268544`, P3.c early-exit active.

Both wheels run with default config (`unrolling_hard_limit=0`,
`unrolling_kernel_hard_limit=0`, `func_inline_depth_limit=0`), so the
Python guard fires the zero short-circuit and contributes no delta. CPU
backend, `offline_cache=False`, cold compile per datapoint (subprocess
spawn, median of 3):

| scenario          |  A (s) |  B (s) |     Δ |
| :---------------- | -----: | -----: | ----: |
| scalar unroll=32  | 0.0925 | 0.0941 | +1.7 % |
| scalar unroll=128 | 0.1697 | 0.1776 | +4.7 % |
| scalar unroll=400 | 0.5705 | 0.5683 | −0.4 % |
| scalar unroll=800 | 1.6885 | 1.6722 | −1.0 % |

**Interpretation.** All four cells are within ±5 % run-to-run noise
(`scalar unroll=128` happens to land highest; trend across the other 3
points is flat/slightly-negative). When `HL=0` the real cold-compile
bottleneck is AST expansion + offline LLVM codegen, not the 4 scalarize
sub-passes — each call is only 6–34 μs per the `TI_COMPILE_PROFILE`
sampler. **P3.c is therefore defense-in-depth**: it saves those μs per
invocation and avoids touching any matrix-rewrite visitors on
scalar-only IR, but does not by itself move cold-compile wall-clock on
scalar kernels. Its value is (a) symmetry with the Python guards when a
user ships with HL=0 *and* the kernel has zero matrix stmts, and (b)
cleaner IR invariants downstream (parity Δ=0 confirmed at commit
`8c1ceec6`).

The heavy compile-time win the user observes (707× abort speed-up at
N=1600) is attributable to P3.a/P3.b; P3.c ships with P3 for
correctness completeness, not for a second-order speed-up on top of it.

---

## Public API summary

All three P3 knobs are user-facing and default to disabled. They are
accepted by `ti.init(...)` and the corresponding `TI_*` env-vars via
`_SpecialConfig` + `env_spec` in [python/taichi/lang/misc.py](../../python/taichi/lang/misc.py).

| knob                          | env var                         | default | scope                                                                              |
| :---------------------------- | :------------------------------ | ------: | :--------------------------------------------------------------------------------- |
| `unrolling_hard_limit`        | `TI_UNROLLING_HARD_LIMIT`        | `0`     | per `ti.static(range(N))`; abort if `N` > limit                                    |
| `unrolling_kernel_hard_limit` | `TI_UNROLLING_KERNEL_HARD_LIMIT` | `0`     | cumulative across all `ti.static` loops in one kernel/func compile                 |
| `func_inline_depth_limit`     | `TI_FUNC_INLINE_DEPTH_LIMIT`     | `0`     | max inline depth of non-real `@ti.func` calls; abort when current depth > limit    |

`0` on any knob means the guard is inert — both the Python
`_check_unroll_hard_limit` and the depth counter short-circuit on the
zero check. Any positive value is a **hard cap**: exceeding it raises
`TaichiCompilationError` with the knob name and offending source line
*before* any IR/codegen runs (no silent truncation). P3.c is independent
of these knobs and always active.


## P3.d investigation 鈥?profile-first, no production change

**Directive**: before any code, identify real hotspots on a multi-kernel
cold compile. Target workload: 16 sequential kernels on CPU backend
(`tests/p3/_p3d_child.py`), ~820 ms wall-clock cold compile.

### Step 1 鈥?Python-side cProfile ([profile_p3d.py](profile_p3d.py) / [profile_p3d_many.py](profile_p3d_many.py))

Results for 16-kernel cold compile (tottime):

| function | tottime | note |
| --- | ---: | --- |
| `kernel_impl.launch_kernel` | **0.719 s** | 83% 鈥?this wraps the pybind call that runs the entire C++ compile + JIT register |
| `ti.init` | 0.046 s | one-time |
| `impl.create_program` | 0.032 s | one-time |
| `textwrap._wrap_chunks` | 0.028 s | **Python hotspot: 5936 calls from `get_pos_info`/`gen_line`** |
| `ast_transformer_utils.get_pos_info` (cumulative) | 0.071 s | per-node, 2800 calls |
| `impl.expr_init` | 0.005 s (tottime), 0.007 s cumulative | 464 calls 鈥?**FFI is NOT a hotspot** |

**Conclusion**: The original P3.d target 鈥?"batch pybind11 FFI" 鈥?does not
address a real hotspot. Actual pybind call volume is small. The Python
side spends most of its visible time inside `get_pos_info` / `gen_line`
(error-message formatting for every AST node), which is itself dwarfed
by the C++ side.

### Step 2 鈥?Attempted Python fast-path (reverted)

Tested hoisting `TextWrapper` to module scope + fast-path short source
lines (`len(code) <= 80` and no tabs/newlines 鈫?direct `.strip()` instead
of `textwrap.wrap`). Correctness test [test_p3d_error_format.py](test_p3d_error_format.py)
passes on the patched code (name error / bad subscript / bin-op error
messages still include the source fragment + caret underline).

A/B cold compile ([bench_p3d_ab.py](bench_p3d_ab.py)):

| N kernels | A (baseline) min s | B (fast-path) min s | delta |
| ---: | ---: | ---: | ---: |
|  4 | 0.218 | 0.211 | -3%  |
|  8 | 0.413 | 0.400 | -3%  |
| 16 | 0.787 | 0.792 | +0.6% |
| 32 | 1.558 | 1.554 | -0.3% |

Back-to-back noise (two B runs on identical code) is ~10% at N=8鈥?6.
Signal is **below the noise floor and below the 5%-ship threshold**
documented in [session plan](#). Reverted 鈥?no production change.

### Step 3 鈥?C++ scoped profiler ([show_profile.py](show_profile.py))

`TI_COMPILE_PROFILE=1` with 16-kernel workload, sorted by total_s:

| path (leaf) | total_s | calls | share of 820ms |
| --- | ---: | ---: | ---: |
| `Program::compile_kernel` | 0.602 | 16 | **74%** |
| 鈫?`KernelCodeGenCPU::optimize_module` 鈫?`llvm_module_opt_pipeline` | 0.169 | 16 | **21%** |
| `LLVM::KernelLauncher::launch_kernel` 鈫?`register_llvm_kernel` | 0.134 | 16 | **16%** |
| `irpass::compile_to_offloads` (all IR passes) | 0.043 | 16 | 5% |
| `irpass::offload_to_executable` (worker threads, 9040/47924) | 0.041 | 15 | 5% |
| `TaichiLLVMContext::get_this_thread_runtime_module` (bitcode load) | ~0.052 | 5脳 (per thread) | 6% |
| `get_hashed_offline_cache_key` | 0.009 | 16 | 1% |

Two real bottlenecks:
1. **LLVM opt pipeline** (21% of compile): `buildPerModuleDefaultPipeline(O3)` +
   post-GEP cleanup, as implemented in [taichi/runtime/llvm/llvm_opt_pipeline.cpp](taichi/runtime/llvm/llvm_opt_pipeline.cpp) (A.4).
2. **LLVM MCJIT register** (16%): `register_llvm_kernel` 鈥?target-code emission +
   JIT link. Not directly tunable without breaking runtime.

### Step 4 鈥?Tested `llvm_opt_level` downshift ([bench_llvm_optlvl.py](bench_llvm_optlvl.py))

Cold compile, N=16, 2 runs min:

| `llvm_opt_level` | dt (s) | head value |
| ---: | ---: | --- |
| 3 (default O3) | 0.80 | 2.889553479512038e+27 |
| 2 (O2) | 0.83 | 2.889553479512038e+27 |
| 1 (O1) | 0.82 | 2.889553479512038e+27 |
| 0 (O0) | **0.74** | 2.8895528892162275e+27 |

- O3 鈫?O1 鈮?**noise** (2.5%, below ship threshold).
- O3 鈫?O0 = **~10% faster** but introduces a **~2e-7 relative** numeric
  drift (constant-folding / fusion differences). Enough to fail strict
  parity gates; would need to be opt-in.
- No production change made. `llvm_opt_level` is already a user knob;
  documenting here that on this workload the practical tier is
  O3 (default) or O0 (opt-in cold-compile), with O1/O2 offering no
  measurable win.

### Net outcome

- **No production code changed in P3.d.** All source files under `python/` and
  `taichi/` are unchanged vs commit `87be37979`.
- Kept artifacts (no code, only measurement):
  - [profile_p3d.py](profile_p3d.py), [profile_p3d_many.py](profile_p3d_many.py) 鈥?cProfile drivers
  - [show_profile.py](show_profile.py) 鈥?C++ scoped profile summary renderer
  - [bench_p3d_ab.py](bench_p3d_ab.py), [_p3d_child.py](_p3d_child.py) 鈥?Python-side A/B harness
  - [bench_llvm_optlvl.py](bench_llvm_optlvl.py) 鈥?LLVM opt-level sweep harness
  - [test_p3d_error_format.py](test_p3d_error_format.py) 鈥?regression test that verifies
    `get_pos_info` error messages include source fragment + caret (use whenever
    touching `ast_transformer_utils.get_pos_info` in the future).
  - [profile_p3d.txt](profile_p3d.txt), [bench_llvm_optlvl2.txt](bench_llvm_optlvl2.txt) 鈥?raw measurements

### Where the next real compile-time win lives

Based on the measurements above, the next viable targets 鈥?each meeting
the **鈮?% cold-compile speedup** bar 鈥?are:

1. **Parallelize per-kernel compile** (target: the 602 ms `compile_kernel`
   sequential wall on main thread). Current `ParallelExecutor` is used
   inside one kernel's `offload_to_executable` but multi-kernel
   submission from Python is serial. Expected win: ~30鈥?0% on 8鈥?2 kernel
   batches with 鈮? cores. **Risk: 3/5** (lifecycle + lock ordering,
   like P2.d prior investigation).
2. **P1.b CHI IR L1.5 cache** (in-process memoization of compiled CHI-IR
   keyed by source hash). Orthogonal to offline cache; targets warm
   re-compile in the same process. **Risk: 3/5** (offline-cache
   interaction).
3. **Revisit `register_llvm_kernel`** (16% share): investigate whether
   the MCJIT handle setup (bitcode load, relocations, GOT) can be
   amortized across kernels of the same kernel-config group. **Risk: 4/5**
   (deep LLVM/JIT interaction).

Recommendation: Option 1 (parallel per-kernel compile) has the best
win/risk ratio and is on-plan for P5 anyway. It requires a measured
sub-plan before implementation (avoid repeating P2.d's three
invalidating surprises).
