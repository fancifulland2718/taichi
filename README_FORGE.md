# Taichi Forge

> **A community-maintained fork of [`taichi`](https://github.com/taichi-dev/taichi) focused on compile-time performance, modern toolchains (LLVM 20, VS 2026, Python 3.10-3.14), and tighter compile-time safety rails.**

[![license](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

## Install

```bash
pip install taichi-forge
```

The **import name is unchanged** — existing code continues to work as-is:

```python
import taichi as ti
ti.init(arch=ti.cuda)
```

Every public API from upstream Taichi 1.7.4 that we still ship behaves the same way.

---

## Why a fork?

Upstream Taichi 1.7.4 shipped against LLVM 15, Python ≤ 3.12, and the Visual Studio 2019/2022 toolchain. Since then the JIT ecosystem has moved on:

- LLVM 15 no longer compiles cleanly with current CUDA / NVPTX toolchains.
- Python 3.13 dropped `distutils`; 3.14 removes further deprecated stdlib APIs.
- Modern Windows developer setups default to VS 2026 (MSVC 14.50+), which rejects some headers hard-wired in the original build scripts.

Taichi Forge is the rolling result of those maintenance upgrades, along with compile-time performance improvements that reduce cold-start and warm-start latency.

---

## Headline feature: sparse SNode on Vulkan

Vanilla Taichi 1.7.4's Vulkan/SPIRV backend supports **only** `dense` + `root`. Every other SNode type — `pointer`, `bitmasked`, `dynamic`, `hash` — falls back to a `TI_NOT_IMPLEMENTED` on Vulkan, blocking macOS-via-MoltenVK, Linux-AMD-without-ROCm, and mobile / embedded users from running sparse data structures at all.

**Taichi Forge 0.3.0 ships `pointer` / `bitmasked` / `dynamic` on Vulkan** with end-to-end three-backend (cpu / cuda / vulkan) numerical equivalence. This is the fork's headline functional differentiator and is fully validated under the regression matrix in [tests/p4/](tests/p4/) (`vulkan_pointer_*.py`, `vulkan_bitmasked_*.py`, `vulkan_dynamic_basic.py`, `g2_pool_fraction.py`, `g4_probe.py`, `g8_cache_compat.py`).

| SNode type | vanilla 1.7.4 Vulkan | Taichi Forge 0.3.0 Vulkan |
|---|---|---|
| `dense` | ✅ | ✅ |
| `bitmasked` | ❌ | ✅ |
| `pointer` | ❌ | ✅ |
| `dynamic` | ❌ | ✅ |
| `quant_array` / `bit_struct` | ❌ | ⚠️ experimental (read + write + concurrent `ti.atomic_add` via CAS-loop; opt-in via `vulkan_quant_experimental=True`) |
| `hash` | ❌ | ❌ (default-disabled in upstream Python frontend; see feasibility note below) |

Highlights:

- **No new public API** — existing `ti.root.pointer(...).dense(...).place(...)` / `ti.activate` / `ti.deactivate` / `ti.is_active` / `ti.length` / `ti.append` Just Work on Vulkan with the same semantics as the LLVM backends.
- **Static capacity by design** — Vulkan has no device-side dynamic allocator, so each `pointer` / `dynamic` SNode reserves its worst-case cell pool in the root buffer at compile time. Out-of-capacity activates degrade silently rather than crashing (a `cap_v` guard verified by `vulkan_pointer_race.py`).
- **Memory knob `TI_VULKAN_POOL_FRACTION`** — opt-in env var (∈ (0, 1], default 1.0) shrinks the pointer pool to `max(num_cells_per_container, round(total × fraction))`. Combine with the `G1.b` deactivate-freelist (always on) to handle steady-state working sets far below the worst case. Verified at `0.25` against three-backend equivalence.
- **`dynamic` uses a flat-array + length-suffix protocol** on Vulkan instead of LLVM's chunk-list (no shader-side `malloc` exists). `ti.append` / `ti.length` / `ti.deactivate` preserve full LLVM semantics; total capacity is the static `N`.
- **Offline cache cross-version safety** — corrupt or version-mismatched `ticache.tcb` automatically triggers fallback recompile, never crashes (verified end-to-end by `g8_cache_compat.py`).
- **Experimental `quant_array` / `bit_struct` on Vulkan** — opt-in via `ti.init(arch=ti.vulkan, vulkan_quant_experimental=True)` or env var `TI_VULKAN_QUANT=1`. With the gate ON, `QuantInt` / `QuantFixed` member reads, writes, and concurrent multi-thread `ti.atomic_add` (via SPIR-V `OpAtomicCompareExchange` spin RMW) are byte-equivalent to the LLVM backends, including multi-field `BitpackedFields` packing (verified by `tests/p4/g9_quant_baseline.py` MPM-style 11/11/10 packing, `tests/p4/g9_quant_array_baseline.py`, and the same-word race baseline `tests/p4/g9_quant_atomic_race.py`). Default OFF preserves vanilla 1.7.4 behaviour exactly. `QuantFloat` shared-exponent and the non-add atomic ops (`atomic_min/max/and/or/xor`, identical restriction to LLVM) are **explicitly out of scope** and raise a clear `TI_NOT_IMPLEMENTED` rather than miscompiling.

📖 **Full usage guide and limitations** (bilingual): [docs/forge/sparse_snode_on_vulkan.en.md](docs/forge/sparse_snode_on_vulkan.en.md) / [docs/forge/sparse_snode_on_vulkan.zh.md](docs/forge/sparse_snode_on_vulkan.zh.md) — covers static-capacity semantics, the `TI_VULKAN_POOL_FRACTION` knob, dynamic-protocol differences, troubleshooting, and the verification matrix.

> ✅ **Both historical Vulkan-sparse stability gaps are now fixed.** The 0.3.0 inactive-read correctness gap (inner-loop reads of inactive sparse cells returning pool-slot-0 data) was fixed in **0.3.1** via the ambient-zone path (CMake `TI_VULKAN_POINTER_AMBIENT_ZONE`, default ON). The 0.3.1 device-lost on 3D pointer full-activation (every cell of a `pointer.*.dense` 3D field first-written within a single frame, then immediately read by a listgen-dependent kernel) is fixed in **0.3.2** via the *C-9 deterministic-slot codegen* path (CompileConfig `vulkan_pointer_deterministic_slot`, default ON): pointer activation collapses to a single `OpAtomicCompareExchange` per outer cell with no spin-loop, byte-eliminating the warp-lockstep deadlock. Manual opt-out is available via `ti.init(vulkan_pointer_deterministic_slot=False)` to fall back to the 0.3.1 CAS-marker path. See the user guide's “Known issues” section for the complete status, fallback conditions, and verification matrix.

📖 **All fork-only knobs** (compile / runtime / architecture / modernization options): [docs/forge/forge_options.en.md](docs/forge/forge_options.en.md) / [docs/forge/forge_options.zh.md](docs/forge/forge_options.zh.md).

> `hash` SNode remains permanently deferred (no real-time physics or rendering pipeline depends on it; see the user guide §6.1 for the survey). `quant_array` / `bit_struct` ship **experimental scaffolding** on Vulkan in 0.3.0 — frontend gate is opt-in via `vulkan_quant_experimental=True`; codegen is incremental. See user guide §7.

### Quick example (Vulkan-on-anything)

```python
import taichi as ti
ti.init(arch=ti.vulkan)              # works on macOS via MoltenVK, Linux-AMD without ROCm, etc.

x = ti.field(ti.f32)
ti.root.pointer(ti.ij, 32).dense(ti.ij, 8).place(x)

@ti.kernel
def fill():
    for i, j in ti.ndrange(256, 256):
        if (i + j) % 17 == 0:
            x[i, j] = i * 0.1 + j * 0.01

fill()
print(x.to_numpy().sum())            # identical to ti.cpu / ti.cuda
```

---

## Supported toolchain

| Area | Requirement |
|---|---|
| Python | 3.10 – 3.14 (3.9 dropped) |
| Windows MSVC | VS 2026 (`Visual Studio 17 2026`, MSVC 14.50+) |
| LLVM | 20.1.7 (included in the wheel) |
| CMake | 3.20+ |
| CUDA (optional) | NVCC 12.x |

---

## Validated backends

End-to-end tested on Linux x86_64 and Windows x86_64:

- ✅ CPU (LLVM JIT)
- ✅ CUDA
- ✅ Vulkan
- ✅ OpenGL / GLES

**Not yet regression-tested** since the LLVM 20 migration:

- ⚠️ macOS (Apple Silicon / Intel) — Metal backend
- ⚠️ AMDGPU backend
- ⚠️ Android ARM64 (C-API)

Patches and reports welcome.

---

## New APIs and settings (fork-only)

All additions are strictly opt-in; default values preserve bit-identical behaviour vs. upstream 1.7.4.

### New functions

| Symbol | Purpose |
|---|---|
| `ti.compile_kernels(kernels)` | Pre-compile a list of kernels on a background thread pool before the hot loop. Accepts decorated kernels or `(kernel, args_tuple)` pairs. Returns the number of kernels submitted. |
| `ti cache warmup script.py` | CLI command — runs `script.py` once with the offline cache forced on, warming up kernel artifacts for subsequent cold starts. |
| `ti.compile_profile()` | Context manager — on exit, prints a per-pass timing report and optionally writes a CSV / Chrome trace. |
| `@ti.kernel(opt_level=...)` | Per-kernel LLVM optimization level override (`"fast"` / `"balanced"` / `"full"` or 0–3). Cache key is isolated per override. |

### `ti.init(...)` / `CompileConfig` knobs

| Kwarg | Default | Purpose |
|---|---|---|
| `compile_tier` | `"balanced"` | `"fast"` lowers LLVM to `-O0` (floor `-O1` on NVPTX/AMDGCN) and SPIR-V optimizer to level 1. `"full"` preserves pre-fork behaviour. |
| `llvm_opt_level` | `-1` (use tier) | Explicit LLVM `-O` override (0–3). |
| `spv_opt_level` | `-1` (use tier) | Explicit SPIR-V `spirv-opt` optimization level override. |
| `num_compile_threads` | logical-core count | Thread pool size for `ti.compile_kernels`. |
| `unrolling_hard_limit` | `0` (off) | Per-`ti.static(for ...)` unroll iteration cap. Aborts with `TaichiCompilationError` instead of silently burning seconds. |
| `unrolling_kernel_hard_limit` | `0` (off) | Total unroll iteration cap across a single kernel. |
| `func_inline_depth_limit` | upstream default | Hard cap on `@ti.func` inline recursion depth. |
| `cache_loop_invariant_global_vars` | `False` | Set `True` to opt in to SNode loop-invariant caching in hot loops. (Default matches vanilla 1.7.4.) |
| `use_fused_passes` | `False` | Enable `pipeline_dirty` short-circuit for redundant `full_simplify` invocations. Numerically bit-identical to off. |
| `tiered_full_simplify` | `True` | Splits `full_simplify` into a local fixed-point pass followed by a single global round per iteration. Set `False` to match the legacy cadence. |
| `compile_dag_scheduler` | `True` | Anti-saturation scheduler for `ti.compile_kernels` batches; balances inner LLVM thread pool and outer kernel pool. Set `False` for the legacy two-tier model. |
| `spirv_parallel_codegen` | `False` | Opt-in task-level parallel SPIR-V codegen per kernel. |
| `spirv_disabled_passes` | `[]` | Per-call disable list for individual `spirv-opt` passes (e.g. `["loop-unroll"]`). |
| `auto_real_function` | `False` | Auto-promote expensive `@ti.func` instances to `is_real_function=True` (LLVM-only, non-autodiff). |
| `auto_real_function_threshold_us` | `1000` | Promotion threshold in microseconds of estimated compile cost. |

### Compatibility note

- `SNode.snode_tree_id` — backported from upstream `master` (not in 1.7.4 release); available on all backends.
- `offline_cache_l_sem` — internal/testing flag, default off. Not for production use.

---

## Quick start

```python
import taichi as ti

ti.init(arch=ti.cuda, compile_tier="fast")

@ti.kernel
def add(a: ti.types.ndarray(), b: ti.types.ndarray(), c: ti.types.ndarray()):
    for i in a:
        c[i] = a[i] + b[i]

import numpy as np
n = 1 << 20
a = np.random.rand(n).astype(np.float32)
b = np.random.rand(n).astype(np.float32)
c = np.empty_like(a)
add(a, b, c)
```

### Pre-compiling a batch of kernels (fork-only)

```python
import taichi as ti
ti.init(arch=ti.cuda)

@ti.kernel
def k1(x: ti.types.ndarray()): ...
@ti.kernel
def k2(x: ti.types.ndarray(), y: ti.types.ndarray()): ...

# Specialize + compile both on the thread pool before the hot loop.
ti.compile_kernels([k1, k2])
```

### Command-line cache warmup (fork-only)

```bash
ti cache warmup train.py -- --epochs 1
# Subsequent `python train.py` runs start with a populated offline cache.
```

---

## Building from source

```bash
git clone https://github.com/fancifulland2718/taichi-forge/taichi.git
cd taichi
python -m pip install -r requirements_dev.txt
python -m pip install -e . --no-build-isolation -v
```

The build is driven entirely by `pyproject.toml` / `scikit-build-core`. On Windows, build a local LLVM 20 snapshot first:

```powershell
.\scripts\build_llvm20_local.ps1   # produces dist\taichi-llvm-20\
```

---

## Versioning

Taichi Forge uses its own SemVer track starting at **0.1.2**. Fork release numbers do **not** match upstream `taichi` versions.

- `0.1.x` — LLVM 20 + VS 2026 + Python 3.14 + initial compile-performance improvements. Backends: Linux/Windows x86_64, CUDA, Vulkan, OpenGL, GLES, CPU.
- `0.2.x` — deeper compile-time, runtime cache, and toolchain modernization. **Stabilization line, superseded by 0.3.0.**
- `0.3.x` — sparse SNode (`pointer` / `bitmasked` / `dynamic`) on Vulkan + experimental `quant_array` scaffolding on Vulkan. **Current line.**

---

## Release notes

### 0.3.0 (current)

First release with **sparse SNode on Vulkan** as a public feature. Inherits the full 0.2.x compile-time, runtime-cache, IR-pass, and dependency-modernization stack (every knob below remains available and bit-identical to 0.2.4 with defaults off).

**Sparse SNode on Vulkan (headline functional differentiator)**

- `pointer` / `bitmasked` / `dynamic` SNodes now run end-to-end on the Vulkan/SPIRV backend, with three-backend (cpu / cuda / vulkan) numerical equivalence verified across 30+ tests in [tests/p4/](tests/p4/).
- Static-capacity model: pool size = `total_num_cells_from_root` (worst case), no shader-side dynamic allocator. Out-of-capacity activates degrade silently via the `cap_v` guard rather than crashing the device.
- New env var `TI_VULKAN_POOL_FRACTION` (∈ (0, 1], default 1.0) shrinks per-pointer pool capacity for memory-tight steady-state workloads. Combined with the deactivate-freelist (always on), supports re-activation cycles without leaking root-buffer space.
- `dynamic` SNode uses a **flat-array + length-suffix protocol** on Vulkan (length atomic-stored at `cell_stride * N` offset of each container) — full LLVM `ti.append` / `ti.length` / `ti.deactivate` semantics preserved, no chunk-list.
- Offline cache cross-version safety: corrupt or version-mismatched `ticache.tcb` triggers fallback recompile, never crashes (validated by `g8_cache_compat.py` three-phase test).
- Build-time guards: `TI_VULKAN_POINTER` / `TI_VULKAN_DYNAMIC` / `TI_VULKAN_POINTER_POOL_FRACTION` CMake flags (all default ON) allow byte-for-byte revert to vanilla 1.7.4 behaviour for regression bisecting.
- `hash` SNode remains unimplemented on every backend in this fork — vanilla taichi 1.7.4 itself default-disables `ti.root.hash(...)` in the Python frontend, and no real-world demo currently depends on it. See the user guide §6 for substitutes (`pointer` + `TI_VULKAN_POOL_FRACTION`, or user-level hash + `dense` buckets).
- Full user guide (limitations, env vars, troubleshooting, verification matrix): [docs/forge/sparse_snode_on_vulkan.en.md](docs/forge/sparse_snode_on_vulkan.en.md) / [docs/forge/sparse_snode_on_vulkan.zh.md](docs/forge/sparse_snode_on_vulkan.zh.md).

**Compile-time performance**

- Fused-pass driver: `use_fused_passes` adds a `pipeline_dirty` short-circuit around `full_simplify` so that no-op iterations are skipped. Measured ~48.6% of `full_simplify` invocations are observably no-op on representative workloads.
- Tiered `full_simplify` (`tiered_full_simplify`, default on): splits the legacy fixed-point loop into a local fixed-point phase plus a single global round per outer iteration, while preserving final IR.
- DAG-aware scheduler for `ti.compile_kernels` (`compile_dag_scheduler`, default on): balances the inner LLVM thread pool against the outer kernel pool to avoid thread oversubscription on batch warm-up.
- Single-offload bypass on the LLVM CPU path: removes the prior 0.89× CPU regression introduced by earlier batch-compile work.
- Per-kernel `opt_level=` override and `compile_tier="fast"|"balanced"|"full"` presets, with isolated cache keys so mixed-tier batches do not poison each other.
- SPIR-V pipeline gains a per-call `spirv_disabled_passes` allowlist, with cache-key isolation. Disabling `loop-unroll` alone gives ~54% SPIR-V codegen wall-time reduction on the validated Vulkan suite; disabling the three heaviest passes gives ~61%, with byte-identical kernel results.
- Optional task-level parallel SPIR-V codegen per kernel (`spirv_parallel_codegen`).
- Auto real-function promotion (`auto_real_function` + `auto_real_function_threshold_us`) and budget-aware inlining fallback in the LLVM-only path; both default off.

**Offline cache and runtime caches**

- Parallel disk-read for offline cache: metadata-hit but ckd-miss path now reads outside the cache mutex and serializes duplicate requests via an in-progress key set. Validated 12-kernel × Vulkan double-process cold start: 290.1 ms (prime) → 83.1 ms (hit), **3.49× faster** with byte-identical per-kernel artifacts.
- `CompileConfig` key audit + offline-cache schema versioning: unrecognized cache versions now fall back to recompile cleanly instead of crashing.
- `rhi_cache.bin` now uses atomic write-then-rename to eliminate half-written cache files after abrupt termination.

**IR / passes**

- `pipeline_dirty` is now explicit and OR-combined across the five mutating passes that can dirty the pipeline, removing spurious dirty marks at no-op call sites. Validated across CPU / CUDA / Vulkan smoke matrices with no regression.
- Defensive `assert` + "type-query forbidden zone" notes on `linking_context_data->llvm_context` to catch accidental cross-context type queries early.

**Toolchain and third-party libraries**

- `spdlog` 1.14.1 → 1.15.3.
- `Vulkan-Headers` / `volk` / `SPIRV-Headers` / `SPIRV-Tools` aligned to **Vulkan SDK 1.4.341** as a single coordinated bump.
- `googletest` 1.10.0 → 1.17.0 (test-only, no runtime impact).
- `glm` 0.9.9.8+187 → **1.0.3**.
- `imgui` v1.84 (WIP) → **v1.91.9b** (non-docking branch). The Vulkan backend was migrated to the new `ImGui_ImplVulkan_InitInfo` layout (`RenderPass` + `ApiVersion` fields, self-managed font texture, `LoadFunctions(api_version, loader)` signature). GGUI visual-regression suite: **90 / 90 passing** on Vulkan + CUDA backends.

**Compatibility**

- All public Python and C-API surfaces from upstream Taichi 1.7.4 remain unchanged. New configuration knobs are additive; their defaults preserve pre-fork behaviour.
- Build toolchain: LLVM 20.1.7, MSVC 14.50+ (VS 2026), Python 3.10–3.14 — unchanged from 0.1.x.

---

## License

Apache 2.0, same as upstream. See [LICENSE](LICENSE). All upstream copyright notices are preserved.

---

## Acknowledgements

Taichi Forge is built on top of the work of the upstream Taichi developers at [taichi-dev/taichi](https://github.com/taichi-dev/taichi). The core compiler, runtime, and the vast majority of the Python frontend are theirs. This fork carries only the delta described above.



