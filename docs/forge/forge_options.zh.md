# Taichi Forge — 编译 / 运行时 / 架构 / 现代化选项一览

> 适用于 **Taichi Forge 0.3.0**。本文列出的所有选项均为**可选启用**；默认值与 vanilla taichi 1.7.4 字节等价。
>
> English: [forge_options.en.md](forge_options.en.md)

本文档是本 fork 所有公开新增配置项与工具链变更的**唯一权威清单**。实验性或内部专用 flag 不收录于此。

---

## 1. 新增 Python API（fork 专属）

| 符号 | 用途 |
|---|---|
| `ti.compile_kernels(kernels)` | 在热循环**之前**用线程池预编译一组 kernel。支持装饰器 kernel 或 `(kernel, args_tuple)` 对。返回提交数量。所有后端可用。 |
| `ti.compile_profile()` | Context manager。退出时输出每个 pass 的耗时报告，可选导出 CSV / Chrome trace JSON。开发期定位编译热点。 |
| `@ti.kernel(opt_level=...)` | 每个 kernel 单独覆盖 LLVM 优化级别。接受 `"fast"` / `"balanced"` / `"full"` 或 `0`–`3`。每个覆盖独立 cache key，混合 tier 批次互不污染。 |

### CLI

| 命令 | 用途 |
|---|---|
| `ti cache warmup script.py [-- script-args]` | 强制开 offline cache 跑一次 `script.py`，为后续冷启动写入 kernel 产物。需与最终运行同 arch / driver。 |

---

## 2. `ti.init(...)` / `CompileConfig` 关键字参数

未注明的默认值均与 vanilla 1.7.4 一致。

### 2.1 编译期 tier 选择

| 参数 | 默认 | 用途 |
|---|---|---|
| `compile_tier` | `"balanced"` | `"fast"` 把 LLVM 拉到 `-O0`（NVPTX / AMDGCN 下兜底 `-O1`），SPIR-V 优化等级降到 1。`"full"` 保持 pre-fork 管线。 |
| `llvm_opt_level` | `-1`（沿用 tier） | 显式 LLVM `-O` 覆盖（`0`–`3`）。 |
| `spv_opt_level` | `-1`（沿用 tier） | 显式 `spirv-opt` 优化级别覆盖。 |

### 2.2 编译管线 / 线程

| 参数 | 默认 | 用途 |
|---|---|---|
| `num_compile_threads` | 逻辑核数 | `ti.compile_kernels` 使用的线程池大小。 |
| `compile_dag_scheduler` | `True` | `ti.compile_kernels` 批次的 DAG 反饱和调度器；平衡内 LLVM 线程池与外 kernel 池。`False` 回退两层调度。 |
| `spirv_parallel_codegen` | `False` | 启用每 kernel 的任务级并行 SPIR-V codegen。 |
| `spirv_disabled_passes` | `[]` | 单次调用禁用某些 `spirv-opt` pass（例如 `["loop-unroll"]`），独立 cache key。仅禁 `loop-unroll` 即可在验证 Vulkan 套件上获得 ~54% SPIR-V codegen wall-time 缩减；最重的 3 个一起禁则 ~61%，kernel 字节等价。 |

### 2.3 Pass / IR 控制

| 参数 | 默认 | 用途 |
|---|---|---|
| `use_fused_passes` | `False` | 在 `full_simplify` 周围加 `pipeline_dirty` 短路；与关闭等价的数值字节兼容。 |
| `tiered_full_simplify` | `True` | 把 `full_simplify` 拆为局部 fixed-point + 每外圈一次 global pass。`False` 回退旧节奏。 |
| `unrolling_hard_limit` | `0`（关） | 每个 `ti.static(for ...)` 的 unroll 迭代上限；超出抛 `TaichiCompilationError`，避免静默吃编译时间。 |
| `unrolling_kernel_hard_limit` | `0`（关） | 单 kernel 内 unroll 总迭代上限。 |
| `func_inline_depth_limit` | 上游默认 | `@ti.func` 内联递归深度硬上限。 |
| `cache_loop_invariant_global_vars` | `False` | 在热循环中启用 SNode 循环不变量缓存。 |

### 2.4 Real-function 与内联

| 参数 | 默认 | 用途 |
|---|---|---|
| `auto_real_function` | `False` | 自动将昂贵 `@ti.func` 实例提升为 `is_real_function=True`（LLVM-only，非 autodiff）。 |
| `auto_real_function_threshold_us` | `1000` | 提升阈值（微秒，估算编译耗时）。 |

### 2.5 兼容性占位

| 参数 | 默认 | 用途 |
|---|---|---|
| `offline_cache_l_sem` | （关） | 内部测试 flag，不应在生产中使用。 |
| `vulkan_quant_experimental` | `False` | **0.3.0 新增**。启用后 Vulkan 后端接受 `quant_array` / `bit_struct` 字段（即 `Extension::quant` / `Extension::quant_basic` 在 Vulkan 上可用）。已支持 `QuantInt` / `QuantFixed` 的读、写与多线程并发 `ti.atomic_add`（`OpAtomicCompareExchange` 自旋 RMW，`quant_array` 与 `BitpackedFields` / `bit_struct` 多字段同字均 OK），三后端字节等价。明确不支持：`QuantFloat` 共享指数、非 add 的原子操作（`atomic_min/max/and/or/xor`，与 LLVM 后端一致）。未实现路径会抛 `TI_NOT_IMPLEMENTED` / `TI_ERROR` 而非静默误编译。等价 env var：`TI_VULKAN_QUANT=1`。 |

---

## 3. 环境变量

| 变量 | 取值 | 默认 | 用途 |
|---|---|---|---|
| `TI_VULKAN_POOL_FRACTION` | `(0.0, 1.0]` | `1.0` | 缩减每个 `pointer` SNode 的物理 cell pool 到 `max(num_cells_per_container, round(total × fraction))`。越界 activate 走既有 `cap_v` silent-inactive 守卫。非法 / `≤ 0` / `> 1` 回退 `1.0`。详细语义见 [sparse_snode_on_vulkan.zh.md](sparse_snode_on_vulkan.zh.md)。 |
| `TI_VULKAN_QUANT` | `0` / `1` | `0` | **0.3.0 新增**。等价于 `ti.init(arch=ti.vulkan, vulkan_quant_experimental=True)`。开启后 `quant_array` 与 `BitpackedFields` / `bit_struct` 的读、写、`ti.atomic_add` 均可用。`QuantFloat` 共享指数、非 add 原子明确不支持。OFF 时行为与 vanilla 1.7.4 相同。 |

> 上游 taichi 已有的环境变量（`TI_ARCH` / `TI_DEVICE_MEMORY_GB` 等）保持原行为，不在此重列。

---

## 4. CMake 构建选项（开发者向）

> 仅在从源码构建 Forge 时暴露。安装发布 wheel 的最终用户无需关心；所有默认 ON 路径已编入。

| 选项 | 默认 | 用途 |
|---|---|---|
| `TI_VULKAN_POINTER` | ON | Vulkan 上 `pointer` / `bitmasked` SNode 总开关。OFF 回退 vanilla 的 `TI_NOT_IMPLEMENTED`。 |
| `TI_VULKAN_DYNAMIC` | ON | Vulkan 上 `dynamic` SNode 总开关。OFF 回退 `TI_NOT_IMPLEMENTED`。 |
| `TI_VULKAN_POINTER_POOL_FRACTION` | ON | 启用 `TI_VULKAN_POOL_FRACTION`。OFF 时该 env var 完全失效，按最坏情况预留 capacity。 |

PyPI 发布的 wheel 三项均为 ON。

---

## 5. SNode 覆盖度扩展

| SNode 类型 | vanilla 1.7.4 Vulkan | Taichi Forge 0.3.0 Vulkan |
|---|---|---|
| `dense` | ✅ | ✅ |
| `bitmasked` | ❌ | ✅ |
| `pointer` | ❌ | ✅ |
| `dynamic` | ❌ | ✅ |

完整用法与语义见 [sparse_snode_on_vulkan.zh.md](sparse_snode_on_vulkan.zh.md)。

---

## 6. 工具链与依赖升级

Forge 同步至现代工具链；下表对比 vanilla 1.7.4。

| 组件 | vanilla 1.7.4 | Forge 0.3.0 |
|---|---|---|
| LLVM | 15 | **20.1.7** |
| Python | 3.7 – 3.12 | **3.10 – 3.14** |
| Windows MSVC | VS 2019 / 2022 | **VS 2026（MSVC 14.50+）** |
| `spdlog` | 1.14.1 | **1.15.3** |
| `Vulkan-Headers` / `volk` / `SPIRV-Headers` / `SPIRV-Tools` | 较旧 | 对齐至 **Vulkan SDK 1.4.341** |
| `googletest` | 1.10.0 | **1.17.0** |
| `glm` | 0.9.9.8 + 187 | **1.0.3** |
| `imgui` | v1.84（WIP） | **v1.91.9b**（non-docking 分支） |

Vulkan ImGui 后端已迁移到新的 `ImGui_ImplVulkan_InitInfo` 布局（`RenderPass` + `ApiVersion` 字段、自管理 font texture、`LoadFunctions(api_version, loader)` 签名）。GGUI 视觉回归 90 / 90 通过（Vulkan + CUDA）。

---

## 7. 架构 / 健壮性改进

以下改进默认启用，**不可调**；列出仅供调试参考。

- **Offline cache 跨版本兼容**：损坏或版本不匹配的 `ticache.tcb` 自动 fallback 重编译，不崩溃。
- **`rhi_cache.bin` 原子写入**：write-then-rename 避免崩溃中断时留下半写文件。
- **`pipeline_dirty` 精确跟踪**：在影响 `full_simplify` 的 5 个修改性 pass 上显式 OR；消除空操作时的伪 dirty 标记。CPU / CUDA / Vulkan 烟测无回归。
- **LLVM CPU 路径单 offload 旁路**：移除前期 batch-compile 工作引入的 0.89× CPU 回归。
- **类型上下文防御性 assert**：`linking_context_data->llvm_context` 上的 forbidden-zone 注释 + assert，提前捕捉跨上下文类型查询。

---

## 8. 兼容性声明

- 上游 Taichi 1.7.4 的所有公开 Python 与 C-API 行为保持不变。
- 本文所列 fork 专属选项全部为**新增**，默认值保留上游行为。
- PyPI wheel 对任何 `import taichi` 的代码均为 drop-in。

---

## 9. 另见

- Sparse SNode on Vulkan 使用指南：[sparse_snode_on_vulkan.zh.md](sparse_snode_on_vulkan.zh.md)
