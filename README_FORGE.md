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

- `0.1.x` — LLVM 20 + VS 2026 + Python 3.14 + compile-performance improvements. Backends: Linux/Windows x86_64, CUDA, Vulkan, OpenGL, GLES, CPU.
- `0.2.x` — deeper compile-time upgrades.
- `0.3.x` — future, potential runtime and architecture changes (e.g. RHI unification, C++20 features).

---

## License

Apache 2.0, same as upstream. See [LICENSE](LICENSE). All upstream copyright notices are preserved.

---

## Acknowledgements

Taichi Forge is built on top of the work of the upstream Taichi developers at [taichi-dev/taichi](https://github.com/taichi-dev/taichi). The core compiler, runtime, and the vast majority of the Python frontend are theirs. This fork carries only the delta described above.



