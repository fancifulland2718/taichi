# Taichi Forge

> **A community-maintained fork of [`taichi`](https://github.com/taichi-dev/taichi) focused on compile-time performance, modern toolchains (LLVM 20, VS 2026, Python 3.14), and tighter compile-time safety rails.**

[![license](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

## Install

```bash
pip install taichi-forge
```

The **import name is changed**:

```python
import taichi-forge as ti
ti.init(arch=ti.cuda)
```

Every public API from upstream Taichi 1.7.4 that we still ship behaves the same way — existing user code runs without modification.

---

## Why a fork?

Upstream Taichi 1.7.4 shipped in mid-2024 against LLVM 15, Python ≤ 3.12, and the legacy LLVM PassManager / typed-pointer API. Since then the JIT ecosystem has moved on:

* LLVM 15 no longer compiles cleanly with current CUDA / NVPTX toolchains, and typed pointers were fully removed in LLVM 17.
* Python 3.13 dropped distutils; 3.14 drops a few more.
* Modern Windows developer setups default to VS 2026, which rejects older MSVC-incompatible headers that Taichi's build scripts had hard-wired.

Taichi Forge is the rolling result of those maintenance upgrades **plus** a compile-time-performance work stream (the P1–P5 phases in the commit log) that reduces cold-start and warm-start compile latency.

---

## Differences vs. upstream Taichi 1.7.4

### Toolchain

| Area | Upstream 1.7.4 | Taichi Forge 0.1.2 |
|---|---|---|
| LLVM | 15.x | **20.1.7** (Phase A.1 → A.4 migration: typed→opaque pointers, legacy-PM→NewPM, `nvvm_ldg_global_{f,i}` → `load + !invariant.load`) |
| CUDA PTX | via LLVM 15 NVPTX | via LLVM 20 NVPTX (NVCC 12.x compatible) |
| Python | 3.9 – 3.12 | **3.10 – 3.14** (3.9 dropped; 3.13/3.14 added) |
| Windows MSVC | VS 2019 / 2022 | **VS 2026** (`Visual Studio 17 2026`, MSVC 14.50+) |
| Build backend | legacy `scikit-build` + `setup.py bdist_wheel` | **`scikit-build-core`** via `python -m build` |
| CMake floor | 3.15 | **3.20** |

### Public API additions

These are **new, fork-only** APIs. Nothing in this list breaks existing 1.7.4 programs; they are all strictly additive.

| Symbol | Introduced | Purpose |
|---|---|---|
| `ti.compile_kernels(kernels_iterable)` | P5.b | **Parallel multi-kernel pre-compile.** Submits a batch of kernels to `num_compile_threads` worker threads for cross-kernel compile parallelism on CPU (LLVM), CUDA and Vulkan. Accepts either decorated kernels or `(kernel, args_tuple)` specialization pairs. Returns the number of kernels submitted. |
| `ti cache warmup script.py [-- args...]` | P1 | **CLI entry point** that pre-runs `script.py` with offline cache forced on, populating disk-backed kernel artifacts for later cold-start re-use. |
| `ti.CompileConfig.compile_tier` | P2.c / P3.d | Enum-valued knob: `"fast"` / `"balanced"` / `"full"`. `fast` caps LLVM at `-O0` (floor `-O1` on NVPTX/AMDGCN) and SPIR-V optimization at level 1; `full` preserves pre-fork behaviour. |
| `ti.CompileConfig.llvm_opt_level` | P1 | Explicit LLVM `-O` override (0–3); takes precedence over `compile_tier`. |
| `ti.CompileConfig.spv_opt_level` | P1.d | Same for SPIR-V (`spirv-opt -O0..-O3`). |
| `ti.CompileConfig.num_compile_threads` | P5.b | Thread pool size for `ti.compile_kernels`. Defaults to the machine's logical-core count. |
| `ti.cfg.unrolling_hard_limit` | P3.a | Per-`ti.static(for ...)` iteration cap. When a single unroll would emit more than this many iterations, compilation aborts with `TaichiCompilationError` instead of quietly taking tens of seconds. `0` (default) preserves 1.7.4 behaviour. |
| `ti.cfg.unrolling_kernel_hard_limit` | P3.a | Cumulative iteration cap across *all* unrolls in a single kernel. Catches pathological nested static fors (e.g. `27³ = 19683`) that individually stay below `unrolling_hard_limit`. |
| `ti.cfg.func_inline_depth_limit` | P3.b | Hard cap on non-real `@ti.func` inline recursion depth. |
| `SNode.snode_tree_id` | (inherited from 1.8.0 backport) | Numeric ID of the owning SNode tree — available on upstream `master` but not released in 1.7.4. |

### Behavioural changes

These changes alter observable behaviour relative to 1.7.4. Most are performance-positive and have no API surface change; a few are documented below because they can shift numerical results at the bit level.

| Area | Change | Impact |
|---|---|---|
| Offline cache key | Dropped `random_seed` (P1.a) | Two runs with the same kernel but different seeds now share cache entries, eliminating spurious recompiles between RNG-using iterations. |
| Offline cache loader | In-process bytes mirror (P1.b) | Hot-start repeat `ti.init(arch=ti.cuda)` in the same process is up to **1.14×** faster. |
| CUDA `__ldg` intrinsic | `nvvm_ldg_global_{f,i}` intrinsics replaced with `load + !invariant.load` metadata | Generated PTX still emits `ld.global.nc`; no perf delta observed, but IR differs. |
| IR passes | `simplify` is deduped via dirty-flag in `compile_to_offloads` (P2.a); `loop_invariant_code_motion` is guarded by first-iteration check (P2.b); `WholeKernelCSE`'s `MarkUndone` walker is O(users) instead of O(N) (pre-P2) | CPU compile 0.89×–1.00×, CUDA 0.99×–1.30×, Vulkan **1.30–1.38×** faster on the heavy-kernel suite. |
| `scalarize` pass | Typed-stmt visit bug in `HasMatrixStmt` fixed (P3.c). Early-exit experiment was *reverted* (miscompile risk). | Behaves correctly in presence of typed matrix stmts. No change if you weren't hitting the miscompile. |
| Kernel compilation thread safety | `KernelCompilationManager` now holds an internal mutex (P5.a) | Enables `ti.compile_kernels`. Single-threaded callers pay one uncontended mutex lock per compile. |

### Removed / deprecated

| Symbol | Status |
|---|---|
| Python 3.9 support | **Removed.** Minimum is 3.10. |
| `wheel` direct build-system dependency | Removed — scikit-build-core integrates `bdist_wheel` natively. |
| `setup.py bdist_wheel` invocation | Still works via a compatibility shim that delegates to `python -m build`. Use PEP 517 entry points (`pip install .`, `python -m build -w`) in new code. |

### Not yet validated in this fork

The main branch is tested end-to-end on Linux x86_64 and Windows x86_64 with the CUDA, Vulkan, OpenGL, GLES and CPU backends. The following paths build but have **not** been regression-tested since the LLVM 20 migration:

* macOS (Apple Silicon / Intel) — Metal backend
* AMDGPU backend
* Android ARM64 C-API

Patches and reports welcome.

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
git clone https://github.com/taichi-dev/taichi.git
cd taichi
python -m pip install -r requirements_dev.txt
python -m pip install -e . --no-build-isolation -v
```

The build is driven entirely by `pyproject.toml` / `scikit-build-core`. See [`docs/design/pypi_release.md`](docs/design/pypi_release.md) and [`compile_doc/LLVM20_升级分析.md`](compile_doc/LLVM20_升级分析.md) for the toolchain details. Windows developers can run `scripts/build_llvm20_local.ps1` to produce a local LLVM 20 snapshot under `dist/taichi-llvm-20/` before building the wheel.

---

## Versioning

Taichi Forge uses its own SemVer track starting at **0.1.2**. Fork release numbers do **not** match upstream `taichi` versions.

* `0.1.x` — LLVM 20 + VS 2026 + Python 3.14 + compile-perf work (P1–P5). Backend coverage: Linux/Windows x86_64, CUDA, Vulkan, OpenGL, GLES, CPU.
* `0.2.x` — planned: macOS/Metal regression suite, scikit-build-core wheel tags for manylinux_2_28.

---

## License

Apache 2.0, same as upstream. See [LICENSE](LICENSE). All upstream copyright notices are preserved.

---

## Acknowledgements

Taichi Forge is built on top of the work of the upstream Taichi developers at [taichi-dev/taichi](https://github.com/taichi-dev/taichi) — the core compiler, runtime, and the vast majority of the Python frontend are theirs. This fork only carries the delta described above.
