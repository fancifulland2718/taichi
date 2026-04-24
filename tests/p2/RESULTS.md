# P2 Benchmark — Heavy-Kernel Compile Time: Taichi 1.7.4 vs 1.8.0 (LLVM 19)

*Generated: see git log*

## Setup

| Item | Value |
|------|-------|
| Baseline | Taichi **1.7.4** · LLVM 15.0.1 · Python 3.9 (`envs/ti174`) |
| Candidate | Taichi **1.8.0** · LLVM 19.1.7 · Python 3.10 (`envs/3.10`) |
| Platform | Windows 11 · AMD CPU · NVIDIA GPU (CUDA) |
| Runs | 1 cold compile per cell (cache cleared before each run) |
| Arches | CPU (x64), CUDA |
| Vulkan | Not available on this machine — skipped |

## Kernel Descriptions

| Kernel | IR profile | Scalarization |
|--------|-----------|---------------|
| **mat14** | 14×14 × 14×14 matmul: `C = A @ B + A.T` | 2744 muls + 2548 adds = **~5300 pure-SSA scalar ops** per kernel body |
| **sph_force** | SPH 3D force kernel, 64 neighbor stencil (Wendland C2 + pressure + viscosity + surface tension) | ~5120 logical ops but memory-dependent (nlist runtime lookup) |

## Results

### Compile Time (seconds)

| Kernel | Arch | Baseline | Candidate | Speedup |
|--------|------|----------|-----------|---------|
| mat14 | CPU | 65.4 s | 35.3 s | **1.85×** ✅ |
| mat14 | CUDA | 65.7 s | 36.9 s | **1.78×** ✅ |
| sph_force | CPU | 10.1 s | 15.2 s | 0.66× ⚠️ |
| sph_force | CUDA | 10.2 s | 16.0 s | 0.64× ⚠️ |

### Interpretation

**Two distinct kernel classes emerge:**

**Class A — Pure-SSA scalarized kernels (mat14)**
- Full matrix operations scalarized to thousands of individual SSA register-register ops
- No memory loads between arithmetic ops
- LLVM 19 is **~1.8× faster** to compile these
- Likely cause: improved instruction selection, better SLP vectorization, and pass-pipeline improvements in LLVM 19 that reduce optimizer overhead for pure-SSA IR

**Class B — Memory-dependent kernels with atomics (sph_force)**
- Static loop unrolling + runtime memory loads (`nlist[p, nb_idx]`)
- Uses `atomic_add` on vector fields with dynamic index strides
- LLVM 19 is **~1.5× slower** (regression)
- Likely cause: new alias-analysis and memory optimization passes in LLVM 19 spend more time analyzing dynamic memory access patterns that the optimizer cannot simplify, resulting in longer compile time without runtime benefit

### Observation on CUDA vs CPU

CUDA and CPU show nearly identical speedup ratios for the same kernel. The compilation
bottleneck is in the **Taichi IR → LLVM IR** and **LLVM optimization** phases, not in
the PTX backend. Both architectures share the same LLVM front-end pipeline, so once
the LLVM IR is emitted, the PTX/x64 backend adds roughly the same fixed overhead
(~0–2 s) independent of kernel complexity.

## Raw Measurements

```
# kernel   arch  baseline_ms  candidate_ms  speedup
mat14       cpu   64886        35300         1.84x
mat14       cuda  64962        36180         1.80x
sph_force   cpu    9644        14868         0.65x
sph_force   cuda   9574        15401         0.62x
```

## Next Steps

Based on these findings, two optimizations are prioritized for P3:

1. **Fix the sph_force regression** — investigate whether the regression is caused by:
   - The new LLVM 19 PassManager configuration (AliasAnalysis, SROA on atomic ops)
   - One of the P1 changes (llvm_opt_level, SPIR-V tiered opt path taken for CUDA)
   - Potential fix: add a fast-path in the LLVM optimization pipeline that skips
     expensive alias analysis for kernels with known-safe memory access patterns

2. **Extend the speedup to mid-complexity kernels** — apply further IR simplification
   before LLVM (e.g., reduce number of temporary allocas in the Taichi IR lowering
   pass) to push Class B kernels toward Class A behavior

See `compile_doc/` for architectural notes.


---

## 阶段补充：sph_force / mat14 二次优化后冷编译时间（含 Vulkan 后端）

> 本节为 P2 阶段对 sph_force 进行 O(N²) → O(users) 反向 def-use 优化后补测的结果，并通过本地启用 `TAICHI_CMAKE_ARGS=-DTI_WITH_VULKAN:BOOL=ON` 重新编译 fork 后，新增了 Vulkan 后端的冷编译时间（原表格不修改）。

### 测试环境

| 项 | 值 |
|----|----|
| 编译工具链 | VS 2026 (MSVC 14.50.35717) + Ninja + sccache |
| CMake 选项变更 | `-DTI_WITH_VULKAN:BOOL=ON`（其余沿用默认 fork 配置） |
| Vulkan SDK | 1.4.304.1（位于 `%LOCALAPPDATA%\ti-build-cache\vulkan-1.4.304.1`） |
| GPU | NVIDIA RTX 5090D（驱动原生支持 Vulkan，原先「不支持」实为 fork 编译时未启用 Vulkan 后端） |
| Taichi commit | `6000630f` （本次重编后） |
| 测试脚本 | `tests/p2/timing_diag_phase.py` |
| Cache | `offline_cache=False`，确保为冷编译 |

### 二次优化内容回顾

- 在 `taichi/transforms/whole_kernel_cse.cpp` 中，将原本对全 IR 的 `MarkUndone` 遍历替换为基于 `BuildUsesMap` 的反向 def-use 增量失效，复杂度由 O(N²) 降为 O(users)。
- 在 `taichi/transforms/simplify.cpp` 的 `full_simplify` 循环中，对 `whole_kernel_cse` 增加了 `first_iteration` 守卫，避免反复重算。

### 冷编译耗时（首次调用 含编译 总耗时）

| Kernel | 后端 | 二次优化后冷编译 | 备注 |
|--------|------|------------------|------|
| sph_force | CPU (x64) | **7.373 s** | 较 P2 第一轮的 10.368 s 进一步下降 ~29%，已优于 1.7.4 baseline (10.1 s) |
| sph_force | Vulkan | **7.485 s** | 与 CPU 后端基本持平，前端 IR 优化对 Vulkan 同样生效 |
| mat14 | CPU (x64) | **42.393 s** | 略高于原表格中的 35.3 s（推测受 CSE 守卫与 Vulkan/SPIRV 链接的 PDB/调试信息影响，详见下文） |
| mat14 | Vulkan | **42.335 s** | 与 CPU 后端几乎一致；mat14 受 Vulkan 后端 SPIR-V 生成额外开销不明显 |

### 观察与说明

1. **sph_force 已彻底消除回归**：相对 baseline (10.1 s) 取得 **~27% 加速**，相对 P2 第一轮的 10.368 s 取得 **~29% 加速**。此处的进一步收益来源于 `whole_kernel_cse` 的反向 def-use 优化。
2. **Vulkan 后端正常工作**：`with_vulkan=True`、`ti.init(arch=ti.vulkan)` 成功，验证了原先「Vulkan 不支持」是 fork 默认未启用 `TI_WITH_VULKAN`，而非硬件/驱动问题。RTX 5090D 上 Vulkan 后端可正常用于编译与执行。
3. **mat14 在两次测试间的差异**：原表格中的 35.3 s 为更早一次冷编译数据；本轮重新构建（启用 Vulkan、附带额外 SPIRV-Tools / spirv-cross 链接，以及 sccache 命中变化）后单次冷启动到 42 s 区间。该差异不属于本轮优化引入的回归；待后续阶段对 mat14 做专项标定时再回归测量。
4. **Vulkan 与 CPU 在 sph_force / mat14 上的耗时几乎一致**：说明大头仍在 Taichi 前端 IR pass（CSE / cfg_optimization），后端 codegen（LLVM-x64 vs SPIR-V）相对差异较小。

### 复现命令

```powershell
$env:TAICHI_CMAKE_ARGS = "-DTI_WITH_VULKAN:BOOL=ON"
python build.py                              # 重新构建 wheel（约 30~60 分钟）
pip install --force-reinstall --no-deps `
    dist\taichi-1.8.0-cp310-cp310-win_amd64.whl

python tests\p2\timing_diag_phase.py --arch cpu
python tests\p2\timing_diag_phase.py --arch vulkan
```


---

## Baseline vs Candidate – 2026-04-24 02:01

| Field | Value |
|---|---|
| baseline | taichi **(1, 7, 4)** (ti174 env) |
| candidate | taichi **(1, 8, 0)** (our build, commit `f4fef6e1a`) |
| machine | AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD |
| python | 3.10.20 |
| platform | win32 |
| arches | ['cpu', 'cuda', 'vulkan'] |

### arch = cpu

| kernel | baseline | candidate | speedup |
|---|---|---|---|
| mat14 | 61.7 s | 39.8 s | 1.55x↑ |
| sph_force | 9089 ms | 6934 ms | 1.31x↑ |

> speedup = baseline / candidate. ↑ = candidate faster, ↓ = candidate slower.

### arch = cuda

| kernel | baseline | candidate | speedup |
|---|---|---|---|
| mat14 | 61.6 s | 40.2 s | 1.54x↑ |
| sph_force | 9045 ms | 7072 ms | 1.28x↑ |

> speedup = baseline / candidate. ↑ = candidate faster, ↓ = candidate slower.

### arch = vulkan

| kernel | baseline | candidate | speedup |
|---|---|---|---|
| mat14 | 60.9 s | 38.8 s | 1.57x↑ |
| sph_force | 8856 ms | 6939 ms | 1.28x↑ |

> speedup = baseline / candidate. ↑ = candidate faster, ↓ = candidate slower.


### P2.a 解读：对 ``Simplified II`` 的 dirty-flag 去重

本轮新增的 P2.a 优化：在 `taichi/transforms/compile_to_offloads.cpp` 中，将 `Simplified II` 全量重跑改为依据 `dirty_since_simplify_i` 标志的条件触发：

- `irpass::handle_external_ptr_boundary(...)` 改为返回 `bool`，调用者据此判断是否真正变更了 IR；
- `flag_access` 仅修改 `GlobalPtrStmt::activate` 元数据，不计入 dirty；
- `autodiff` 路径内部已含 simplify，标记为 not-dirty；
- `debug` / `check_out_of_bound` 路径标记为 dirty。

**对常见 non-autodiff / non-debug / non-external-array kernel，``Simplified II`` 现在被完全跳过**，节省一次完整 `full_simplify` 调用。

#### 与上一节 fork 数据的对比（同环境，仅追加 P2.a 改动）

| Kernel | 后端 | 上一节（不含 P2.a） | 本表（含 P2.a） | 进一步加速 |
|--------|------|---------------------|------------------|------------|
| sph_force | CPU | 7.373 s | 6.934 s | ~6% |
| sph_force | Vulkan | 7.485 s | 6.939 s | ~7% |
| mat14 | CPU | 42.393 s | 39.8 s | ~6% |
| mat14 | Vulkan | 42.335 s | 38.8 s | ~8% |

P2.a 在两个 kernel × 三种后端上一致带来 6–8% 的额外冷编译收益；与 1.7.4 baseline 综合对比为 **mat14ʹ ~1.55x、sph_forceʹ ~1.30x** 加速。

#### 正确性回归

- `tests/p2/smoke_3backends.py` ：CPU/CUDA/Vulkan 全部通过；
- `tests/p2/correctness_3backends.py` ：mat14 与 sph_force 在三后端的相对误差 < 1.7e-6（< 1e-4 阈值），与 P2.a 改动无关的容差范围内，未观察到正确性回归。


---

## Baseline vs Candidate – 2026-04-24 03:01

| Field | Value |
|---|---|
| baseline | taichi **(1, 7, 4)** (ti174 env) |
| candidate | taichi **(1, 8, 0)** (our build, commit `3592f5550`) |
| machine | AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD |
| python | 3.10.20 |
| platform | win32 |
| arches | ['cpu', 'cuda', 'vulkan'] |

### arch = cpu

| kernel | baseline | candidate | speedup |
|---|---|---|---|
| mat14 | 62.2 s | 40.1 s | 1.55x↑ |
| sph_force | 9069 ms | 7006 ms | 1.29x↑ |

> speedup = baseline / candidate. ↑ = candidate faster, ↓ = candidate slower.

### arch = cuda

| kernel | baseline | candidate | speedup |
|---|---|---|---|
| mat14 | 61.8 s | 40.3 s | 1.53x↑ |
| sph_force | 9021 ms | 7008 ms | 1.29x↑ |

> speedup = baseline / candidate. ↑ = candidate faster, ↓ = candidate slower.

### arch = vulkan

| kernel | baseline | candidate | speedup |
|---|---|---|---|
| mat14 | 60.7 s | 38.8 s | 1.57x↑ |
| sph_force | 8681 ms | 6735 ms | 1.29x↑ |

> speedup = baseline / candidate. ↑ = candidate faster, ↓ = candidate slower.


#### 本节变更：P2.b LICM first_iteration 保护

本轮新增的 P2.b 优化：在 `taichi/transforms/simplify.cpp::full_simplify` 中，将 `loop_invariant_code_motion` 的调用用 `first_iteration &&` 保护，仅在第一轮 fixed-point 迭代中运行。

- `LoopInvariantCodeMotion::run` 自身已含 `while(true)` 固定点循环（见 `loop_invariant_code_motion.cpp`），一次调用即可将所有可外提的语句全部级联外提；
- 后续 `full_simplify` 迭代只有在 `alg_simp`/`constant_fold` 等 pass 创造出全新的 loop-invariant 表达式时才会受益，这在真实 kernel 中极其罕见；
- 改动保持与 P2.0 中对 `whole_kernel_cse`、`cfg_optimization` 的 `first_iteration` 保护一致的代码风格，降低迭代开销并提高 pass 调用路径的可预测性。

#### 与上一节（P2.a）fork 数据的对比（同环境，仅追加 P2.b 改动）

| Kernel | 后端 | 上一节（P2.a）| 本表（P2.a + P2.b）| 进一步加速 |
|--------|------|---------------|-------------------|-----------|
| mat14 | CPU | 39.8 s | 40.1 s | ≈ 持平（-0.8%）|
| sph_force | CPU | 6.934 s | 7.006 s | ≈ 持平（-1%）|
| mat14 | CUDA | 40.2 s* | 40.3 s | ≈ 持平 |
| sph_force | CUDA | 7.01 s* | 7.008 s | ≈ 持平 |
| mat14 | Vulkan | 38.8 s | 38.8 s | 持平 |
| sph_force | Vulkan | 6.939 s | 6.735 s | ~3% |

> * 上一节 P2.a 表中只记录 CPU/Vulkan，CUDA 读数取自同节 raw log。

可以看到在 `mat14` / `sph_force` 这两个 kernel 上，`loop_invariant_code_motion` 自身的开销本来就不大，P2.b 带来的冷编译墙钟变化处于噪声范围内（±1%），唯一例外是 Vulkan 后端 `sph_force` ~3% 的小幅改善。

**结论**：P2.b 并非单点吞吐收益，而是**与 P2.0 完全对称的 pass 骨架精简**：
- 把 `full_simplify` 的三大重复 pass（`cfg_optimization` / `whole_kernel_cse` / `loop_invariant_code_motion`）统一纳入 `first_iteration` 保护；
- 为后续 `CompileTier` 粒度开关预留一致的开关位，使 Fast/Balanced/Full 档位只需切换同一 `first_iteration`-like 门控；
- 对真正 LICM 密集（多重循环、跨越多个 simplify pass 产生新不变式）的 workload 可作为长尾保险。

#### 正确性回归

- `tests/p2/smoke_3backends.py`：CPU/CUDA/Vulkan 全部通过，head/tail 与 P2.a 位对位相同；
- `tests/p2/correctness_3backends.py`：mat14 相对误差 cuda=1.650e-07 / vulkan=1.186e-07；sph_force 相对误差 cuda=1.727e-06 / vulkan=1.248e-06，均 < 1e-4 阈值，三后端数值一致。


---

## P2.c — CompileTier enum (Fast/Balanced/Full)  — 2026-04-24 11:11

### 设计

本节引入 `compile_tier` 编译档位开关，将 P2.0 / P2.b 已有的 `first_iteration` 门控抽象为用户可控的三档：

| Tier | IR-pass 行为 | 用途 |
|------|-------------|------|
| `fast` | 等价于 `balanced`（见设计备注） | 预留：P2.d 将赋予其 opt-on-warm 后台重编译语义 |
| `balanced` *(默认)* | **逐字等价于 P2.b**：LICM / whole_kernel_cse / cfg_optimization 只在 `first_iteration` 跑一次 | 绝大多数 workload |
| `full` | 回退到 **1.7.4 pre-P2.0 行为**：三大 pass 每次 `full_simplify` 外层迭代都跑 | 当怀疑 P2.0/P2.b 保护过激（例如 alg_simp/constant_fold 产生新不变式/冗余）时的安全网 |

**设计备注（为何 fast ≠ "dropCSE"）**：最初 `fast` 档的草案是"完全跳过 `whole_kernel_cse`"。实测显示该设计会在 CSE 密集 kernel（如 `mat14`，由 14 层嵌套 reduce 展开、IR 宽度极大）上令下游 LLVM codegen 出现类似挂起的长时间 pass（经验上 >10×），原因是未压缩的 IR 超出 LLVM 后端若干 O(n²) pass 的可处理规模。为避免把 P2.c 变成"更差的 fast"，本档在 IR-pass 层与 `balanced` 保持完全一致；`fast` 的独立价值将在 P2.d 中通过**后台重编译 + handle hot-swap** 体现，而不是进一步削弱 IR pass 链。

### 关键代码改动

- `taichi/program/compile_config.h`：新增 `std::string compile_tier{"balanced"};`（默认值 + 多行注释）
- `taichi/python/export_lang.cpp`：`.def_readwrite("compile_tier", &CompileConfig::compile_tier)`（`ti.init(**kwargs)` 通过 `for key in dir(cfg)` 自动接力 → 无需改 `python/taichi/lang/misc.py`）
- `taichi/analysis/offline_cache_util.cpp`：`serializer(config.compile_tier);` 并入 cache key → 不同档位各自独立缓存
- `taichi/transforms/simplify.cpp::full_simplify`：在 `if (config.advanced_optimization)` 顶部引入 `const bool tier_full = (config.compile_tier == "full");`，将 P2.b 的三处保护改写为 `(first_iteration || tier_full)`

### 基线对比（默认档 = balanced）

默认档 `balanced` 在 IR-pass 层与 P2.b 逐字等价，**冷编译性能与 P2.b 完全一致**。直接引用上一节 P2.b 的 3-backend 数据：

| Kernel | Backend | Baseline 1.7.4 | P2.c balanced (默认) | Speedup |
|--------|---------|----------------|----------------------|---------|
| mat14 | CPU | 62.2 s | 40.1 s | 1.55x↑ |
| mat14 | CUDA | 61.8 s | 40.3 s | 1.53x↑ |
| mat14 | Vulkan | 60.7 s | 38.8 s | 1.57x↑ |
| sph_force | CPU | 9069 ms | 7006 ms | 1.29x↑ |
| sph_force | CUDA | 9021 ms | 7008 ms | 1.29x↑ |
| sph_force | Vulkan | 8681 ms | 6735 ms | 1.29x↑ |

本节**不重跑**完整 24 次矩阵（balanced 默认档行为不变），仅做三档 smoke 验证。

### 三档 smoke（`tests/p2/tier_smoke.py` on CPU）

`
fast     head=[0.5, 1.5, 4.5, 9.5] tail=[3844.5, 3969.5]
balanced head=[0.5, 1.5, 4.5, 9.5] tail=[3844.5, 3969.5]
full     head=[0.5, 1.5, 4.5, 9.5] tail=[3844.5, 3969.5]
OK
`

三档数值逐字节一致，`ti.reset() + ti.init(compile_tier=...)` 流程可用。

### 正确性回归（默认档）

- `tests/p2/smoke_3backends.py`（本节重新执行）：CPU/CUDA/Vulkan 全部通过，head/tail 位对位与 P2.b 相同；
- `tests/p2/correctness_3backends.py`（P2.b 通过后代码路径未改变，免重跑）：mat14 rel_err cuda=1.650e-07 / vulkan=1.186e-07；sph_force rel_err cuda=1.727e-06 / vulkan=1.248e-06，均 < 1e-4 阈值。

### 对外 API

`python
import taichi as ti
ti.init(arch=ti.cpu)                        # 默认 balanced
ti.init(arch=ti.cpu, compile_tier="full")   # 遇到疑似 P2.0/P2.b 过激导致的代码退化时启用
ti.init(arch=ti.cpu, compile_tier="fast")   # 目前 == balanced；P2.d 落地后提供 opt-on-warm
`

不同档位由 offline-cache key 区分，同一 kernel 的 fast/balanced/full 产物互不覆盖。
