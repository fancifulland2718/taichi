# Taichi Forge — Compile, Runtime, Architecture & Modernization Options

> Applies to **Taichi Forge 0.3.0**. Every option listed here is **opt-in**; defaults preserve bit-identical behaviour vs. upstream Taichi 1.7.4.
>
> 中文版：[forge_options.zh.md](forge_options.zh.md)

This document is the single canonical reference for Forge-specific knobs and toolchain changes. Older fork-only settings are kept here once they are part of the public API; experimental or internal-only flags are intentionally excluded.

---

## 1. New Python APIs (fork-only)

| Symbol | Purpose |
|---|---|
| `ti.compile_kernels(kernels)` | Pre-compile a list of kernels on a background thread pool **before** the hot loop. Accepts either decorated kernels or `(kernel, args_tuple)` pairs. Returns the number of kernels submitted. Available on every backend. |
| `ti.compile_profile()` | Context manager. On exit, prints a per-pass timing report and (optionally) writes a CSV / Chrome-trace JSON. Use during development to find compile-time hot spots. |
| `@ti.kernel(opt_level=...)` | Per-kernel LLVM optimisation override. Accepts `"fast"` / `"balanced"` / `"full"` or `0`–`3`. Cache key is isolated per override, so mixed-tier batches do not poison each other. |

### CLI

| Command | Purpose |
|---|---|
| `ti cache warmup script.py [-- script-args]` | Run `script.py` once with the offline cache forced on, populating kernel artifacts for subsequent cold starts. Same arch / driver as the eventual run. |

---

## 2. `ti.init(...)` / `CompileConfig` keyword arguments

All defaults match upstream 1.7.4 unless noted.

### 2.1 Compile-time tier selection

| Kwarg | Default | Purpose |
|---|---|---|
| `compile_tier` | `"balanced"` | `"fast"` lowers LLVM to `-O0` (floor `-O1` on NVPTX / AMDGCN) and SPIR-V to optimization level 1. `"full"` preserves the pre-fork pipeline. |
| `llvm_opt_level` | `-1` (use tier) | Explicit LLVM `-O` override (`0`–`3`). |
| `spv_opt_level` | `-1` (use tier) | Explicit `spirv-opt` optimisation level override. |

### 2.2 Compile-pipeline batch & threading

| Kwarg | Default | Purpose |
|---|---|---|
| `num_compile_threads` | logical-core count | Thread-pool size used by `ti.compile_kernels`. |
| `compile_dag_scheduler` | `True` | DAG-aware anti-saturation scheduler for `ti.compile_kernels` batches; balances inner LLVM thread pool vs. outer kernel pool. Set `False` to fall back to the legacy two-tier model. |
| `spirv_parallel_codegen` | `False` | Opt-in per-kernel task-level parallel SPIR-V codegen. |
| `spirv_disabled_passes` | `[]` | Per-call disable list for individual `spirv-opt` passes (e.g. `["loop-unroll"]`). Cache-key isolated. Disabling `loop-unroll` alone yields ~54 % SPIR-V codegen wall-time reduction on the validated Vulkan suite; the three heaviest passes together yield ~61 %, with byte-identical kernel results. |

### 2.3 Pass / IR controls

| Kwarg | Default | Purpose |
|---|---|---|
| `use_fused_passes` | `False` | Enable `pipeline_dirty` short-circuit around `full_simplify`; numerically bit-identical to off. |
| `tiered_full_simplify` | `True` | Splits `full_simplify` into a local fixed-point phase plus a single global round per outer iteration. Set `False` for the legacy cadence. |
| `unrolling_hard_limit` | `0` (off) | Per-`ti.static(for ...)` unroll iteration cap. Aborts with `TaichiCompilationError` instead of silently consuming compile time. |
| `unrolling_kernel_hard_limit` | `0` (off) | Total unroll iteration cap across a single kernel. |
| `func_inline_depth_limit` | upstream default | Hard cap on `@ti.func` inline recursion depth. |
| `cache_loop_invariant_global_vars` | `False` | Opt in to SNode loop-invariant caching in hot loops. |

### 2.4 Real-function & inlining

| Kwarg | Default | Purpose |
|---|---|---|
| `auto_real_function` | `False` | Auto-promote expensive `@ti.func` instances to `is_real_function=True` (LLVM-only, non-autodiff). |
| `auto_real_function_threshold_us` | `1000` | Promotion threshold in microseconds of estimated compile cost. |

### 2.5 Compatibility-only

| Kwarg | Default | Purpose |
|---|---|---|
| `offline_cache_l_sem` | (off) | Internal / testing flag, not for production use. |
| `vulkan_quant_experimental` | `False` | **New in 0.3.0.** When ON, the Vulkan backend accepts `quant_array` / `bit_struct` fields (i.e. `Extension::quant` / `Extension::quant_basic` are reported supported on Vulkan). Supported: `QuantInt` / `QuantFixed` read, write, and concurrent multi-thread `ti.atomic_add` (via SPIR-V `OpAtomicCompareExchange` spin RMW) on `quant_array` and on multi-field `BitpackedFields` / `bit_struct`, byte-equivalent to cpu / cuda. **Explicitly not supported**: `QuantFloat` shared-exponent and the non-add atomic ops (`atomic_min/max/and/or/xor`, identical restriction to the LLVM backend). Unsupported sites raise `TI_NOT_IMPLEMENTED` / `TI_ERROR` rather than silently miscompile. Equivalent env var: `TI_VULKAN_QUANT=1`. |

---

## 3. Environment variables

| Variable | Range | Default | Purpose |
|---|---|---|---|
| `TI_VULKAN_POOL_FRACTION` | `(0.0, 1.0]` | `1.0` | Shrinks each `pointer` SNode's physical cell pool to `max(num_cells_per_container, round(total × fraction))`. Out-of-capacity activates fall through the existing `cap_v` silent-inactive guard. Invalid / `≤ 0` / `> 1` falls back to `1.0`. Detailed semantics: see [sparse_snode_on_vulkan.en.md](sparse_snode_on_vulkan.en.md). |
| `TI_VULKAN_QUANT` | `0` / `1` | `0` | **New in 0.3.0.** Equivalent to `ti.init(arch=ti.vulkan, vulkan_quant_experimental=True)`. When ON, `quant_array` and `BitpackedFields` / `bit_struct` read, write, and `ti.atomic_add` are all available on Vulkan. `QuantFloat` shared-exponent and non-add atomics are explicitly not supported. OFF preserves vanilla 1.7.4 behaviour. |

> Other environment variables documented in upstream Taichi remain unchanged (`TI_ARCH`, `TI_DEVICE_MEMORY_GB`, etc.). They are not re-listed here.

---

## 4. CMake build options (developer-side)

> These are surfaced **only** when building Forge from source. End users installing the published wheel get every default-ON path; no flags need to be set.

| Option | Default | Purpose |
|---|---|---|
| `TI_VULKAN_POINTER` | ON | Master switch for `pointer` / `bitmasked` SNode on Vulkan. OFF → vanilla `TI_NOT_IMPLEMENTED`. |
| `TI_VULKAN_DYNAMIC` | ON | Master switch for `dynamic` SNode on Vulkan. OFF → vanilla `TI_NOT_IMPLEMENTED`. |
| `TI_VULKAN_POINTER_POOL_FRACTION` | ON | Activates `TI_VULKAN_POOL_FRACTION`. OFF makes the env var a no-op; capacity is reserved for the worst case. |

The wheel published to PyPI builds with all three flags ON.

---

## 5. SNode coverage extensions

| SNode type | vanilla 1.7.4 Vulkan | Taichi Forge 0.3.0 Vulkan |
|---|---|---|
| `dense` | ✅ | ✅ |
| `bitmasked` | ❌ | ✅ |
| `pointer` | ❌ | ✅ |
| `dynamic` | ❌ | ✅ |

Full usage and semantics: [sparse_snode_on_vulkan.en.md](sparse_snode_on_vulkan.en.md).

---

## 6. Toolchain & dependency upgrades

Forge ships against modern toolchains; the table below summarises the versions vs. vanilla 1.7.4.

| Component | vanilla 1.7.4 | Forge 0.3.0 |
|---|---|---|
| LLVM | 15 | **20.1.7** |
| Python | 3.7 – 3.12 | **3.10 – 3.14** |
| Windows MSVC | VS 2019 / 2022 | **VS 2026 (MSVC 14.50+)** |
| `spdlog` | 1.14.1 | **1.15.3** |
| `Vulkan-Headers` / `volk` / `SPIRV-Headers` / `SPIRV-Tools` | older | aligned to **Vulkan SDK 1.4.341** |
| `googletest` | 1.10.0 | **1.17.0** |
| `glm` | 0.9.9.8 + 187 | **1.0.3** |
| `imgui` | v1.84 (WIP) | **v1.91.9b** (non-docking branch) |

The Vulkan ImGui backend was migrated to the new `ImGui_ImplVulkan_InitInfo` layout (`RenderPass` + `ApiVersion` fields, self-managed font texture, `LoadFunctions(api_version, loader)` signature). The GGUI visual-regression suite passes 90 / 90 on the Vulkan + CUDA backends.

---

## 7. Architecture / robustness improvements

These are not user-tunable; they ship enabled by default. Listed for visibility / debugging.

- **Offline cache cross-version safety** — corrupt or version-mismatched `ticache.tcb` triggers an automatic fallback recompile rather than crashing.
- **`rhi_cache.bin` atomic write** — write-then-rename eliminates half-written cache files after abrupt termination.
- **`pipeline_dirty` precise tracking** — explicit OR-combined dirty marks across the five mutating passes that affect `full_simplify`. Removes spurious dirty marks at no-op call sites; verified across CPU / CUDA / Vulkan smoke matrices with no regression.
- **Single-offload bypass on the LLVM CPU path** — removes the prior 0.89× CPU regression introduced by earlier batch-compile work.
- **Defensive type-context guards** on `linking_context_data->llvm_context` to catch accidental cross-context type queries.

---

## 8. Compatibility statement

- All public Python and C-API surfaces from upstream Taichi 1.7.4 remain unchanged.
- Every fork-only knob in this document is additive and defaults to upstream behaviour.
- The published wheel is drop-in for any code that imports `taichi`.

---

## 9. See also

- Sparse SNode on Vulkan user guide: [sparse_snode_on_vulkan.en.md](sparse_snode_on_vulkan.en.md)
