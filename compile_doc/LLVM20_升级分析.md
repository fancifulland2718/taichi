# 升级到 LLVM 20 — 影响分析与改造清单

> 当前基线：LLVM 19.1.x（stock upstream，[`.github/workflows/scripts/ti_build/llvm.py`](.github/workflows/scripts/ti_build/llvm.py#L16)）
>
> 目标：LLVM 20.1.x（最新稳定版）
>
> 本文档只是分析，不包含代码改动。

## TL;DR

LLVM 20 与 19 之间是 minor 版本跨越，**没有像 14→15、16→17 那样的"断崖式"破坏性
变更**（opaque pointers 已经在 LLVM 19 强制启用、`PassManagerBuilder` 已在 LLVM 17
移除）。Taichi 当前对 LLVM 19 的兼容工作（`llvm_opt_pipeline.{h,cpp}` 新 PassBuilder、
opaque pointer 适配）让升级到 20 的工作量从"重构"降级为"小补丁"。

**预估改动量**：
- C++ 代码：~50–150 行（集中在 4–6 个文件）
- CMake / 构建脚本：~20 行（仅 LLVM 下载链接 + 版本号字符串）
- 测试基线：可能需要重新刷一遍 PTX/SPV 黄金文件

**预估风险**：低-中。主要风险点是 NVPTX 后端在 LLVM 20 引入了 NVVM IR 7.x（CUDA 12.5+）
的几条新 intrinsic，旧 intrinsic 在某些情况下会发出 deprecation warning；以及 ORC JIT
在 LLVM 20 调整了 `IRCompileLayer` 的回调签名。

## 1. 已经做完的工作（不需要重做）

LLVM 17→19 的迁移过程中已经处理掉的麻烦事，在 20 上仍然有效：

| 已完成项 | 当前位置 | LLVM 20 影响 |
| ------- | ------- | ----------- |
| Opaque pointers | [llvm_codegen_utils.cpp:32](taichi/codegen/llvm/llvm_codegen_utils.cpp#L32)、[codegen_llvm.cpp:1801](taichi/codegen/llvm/codegen_llvm.cpp#L1801) | 无变化，LLVM 20 也是 opaque-only |
| 新 PassBuilder 优化管线 | [llvm_opt_pipeline.h](taichi/runtime/llvm/llvm_opt_pipeline.h)、[llvm_opt_pipeline.cpp](taichi/runtime/llvm/llvm_opt_pipeline.cpp) | API 不变 |
| `OptimizationLevel::O0..O3` | [llvm_opt_pipeline.h:26-31](taichi/runtime/llvm/llvm_opt_pipeline.h#L26-L31) | 不变 |
| `find_package(LLVM REQUIRED CONFIG)` 不锁版本 | [cmake/TaichiCore.cmake:155](cmake/TaichiCore.cmake#L155) | 自动适配 |

## 2. 需要改动的部分

### 2.1 构建脚本（必改，~10 行）

[`.github/workflows/scripts/ti_build/llvm.py`](.github/workflows/scripts/ti_build/llvm.py)

* `_local_llvm19_prefix()` → `_local_llvm20_prefix()`，路径 `dist/taichi-llvm-19/` → `dist/taichi-llvm-20/`。
* 注释里所有 "LLVM 19" → "LLVM 20"。
* `legacy_url` 的 LLVM 15 fallback 链接保持不变（仍然是历史兜底）。
* Windows 路径下的提示文本里 "build_llvm19_local.ps1" 需要新建一个 `build_llvm20_local.ps1`，本质是把旧脚本里 `19.1.7` → `20.1.x`。

[`scripts/build_llvm19_local.ps1`](scripts/build_llvm19_local.ps1)（如果存在）→ 复制成 `build_llvm20_local.ps1`，更新源 tarball URL。

### 2.2 CMake 兼容性 flag（小心，~5 行）

[`cmake/utils.cmake:11-15`](cmake/utils.cmake#L11-L15)

```cmake
# Silence LLVM 19 [[deprecated]] warnings on legacy PassManager / ...
```

LLVM 20 把更多的 legacy API 标记为 `[[deprecated]]`：
* `legacy::PassManager::run` 在 LLVM 20 仍然存在，但部分头文件（比如 `llvm/Transforms/IPO/PassManagerBuilder.h` 早就删除）已经无法包含，需要确认所有 `.cpp` 都不再 include。
* `llvm::JITTargetMachineBuilder::detectHost()` 在 LLVM 20 没变，但 `JITTargetMachineBuilder` 的某些构造器签名加了 `noexcept`，影响 perfect-forwarding。

应对：把这条 flag 的注释和影响范围更新到 "LLVM 19/20"，源码无需改动；若编译时出现新的 `-Wdeprecated-declarations`，按需追加 `-Wno-deprecated-declarations`（已有）。

### 2.3 NVPTX intrinsic 列表（关键，~10–30 行）

[`taichi/runtime/llvm/llvm_context.cpp:18`](taichi/runtime/llvm/llvm_context.cpp#L18) → `#include "llvm/IR/IntrinsicsNVPTX.h"`

LLVM 20 在 NVPTX 后端引入了 NVVM IR 7.x（伴随 CUDA 12.5+）的若干新 intrinsic，**同时把
旧 intrinsic 标记为 deprecated 但仍可用**。Taichi 用到的 intrinsic 集中在：
* `llvm.nvvm.barrier0`（同步）
* `llvm.nvvm.shfl.sync.*`（warp shuffle）
* `llvm.nvvm.atomic.*`（原子操作）
* `llvm.nvvm.read.ptx.sreg.*`（线程/块 ID）

这些 intrinsic 在 LLVM 20 全部仍然存在。**不需要改 intrinsic 名字**，但需要：
* 验证 `Intrinsic::nvvm_*` 枚举值在 LLVM 20 头文件里仍然有定义（语法上不变，重新编译即可）。
* 检查 `tests/p3/_parity_p3c_matrix_old.py:83` 注释里提到的"CUDA backend has an independent LLVM-19 NVVM intrinsic"路径，确认 LLVM 20 上 NVVM IR 也接受同样的 intrinsic。
* 重新生成 [`external/cuda_libdevice/`](external/cuda_libdevice/) 里的 libdevice bitcode（CUDA 工具链对应升级）。

### 2.4 ORC JIT API 微调（中风险，~20–50 行）

LLVM 20 对 ORC v2 的几个回调点改了签名：

* `IRCompileLayer::IRCompiler` 的 `operator()(Module&)` 在 LLVM 20 加了一个可选的
  `MaterializationResponsibility&` 参数。Taichi 目前在 [`taichi/runtime/cpu/jit_cpu.cpp`](taichi/runtime/cpu/jit_cpu.cpp) 没有自定义 IRCompiler（用的是默认的 `ConcurrentIRCompiler`），所以**不受影响**。
* `LLJITBuilder::setObjectLinkingLayerCreator` 的回调改成了接受 `ExecutionSession&` 而非 `LLJIT&`。当前代码用的是 `LLJIT::create()` 默认 builder，**不受影响**。
* `JITDylib::define` 在 LLVM 20 可以返回 `Expected<void>` 替代 `Error`。当前代码 [`jit_cpu.cpp:161`](taichi/runtime/cpu/jit_cpu.cpp#L161) 用 `cantFail(...)` 包裹，**自动兼容**。

主要的潜在影响：
* `llvm::orc::ThreadSafeContext` 的 ctor 在 LLVM 20 新增了一个 deleted move constructor 来防止误用。Taichi 在 [`llvm_context.h:29`](taichi/runtime/llvm/llvm_context.h#L29) 持有 `unique_ptr<ThreadSafeContext>`，所以使用方式安全，但需要检查 [`llvm_context.cpp:1022`](taichi/runtime/llvm/llvm_context.cpp#L1022) 的 `ThreadLocalData(std::unique_ptr<ThreadSafeContext>)` 构造函数是否还能 `std::move(ctx)` 进去（应该可以，因为 `unique_ptr` 是 move）。

### 2.5 `Triple` 类的迁移（轻微，~3 行）

LLVM 20 把 `llvm::Triple` 从 `llvm/ADT/Triple.h` 正式迁移到 `llvm/TargetParser/Triple.h`
（19 已经支持但 19 的旧路径仍然兼容，20 还在）。当前 [`taichi/runtime/cuda/jit_cuda.cpp:99`](taichi/runtime/cuda/jit_cuda.cpp#L99) 用 `llvm::Triple triple(...)` 但没显式 include 该头文件（间接 include），如果 LLVM 20 拆掉了过渡头，需要显式 `#include "llvm/TargetParser/Triple.h"`。

### 2.6 DataLayout 与 SubtargetInfo（轻微）

LLVM 20 在某些 target 上扩展了 DataLayout 字符串（增加了 `n128` 的合法位宽支持等）。
Taichi 在 [`llvm_context.cpp`](taichi/runtime/llvm/llvm_context.cpp) 里靠 `JITTargetMachineBuilder::detectHost()` 自动获取 DataLayout，**无需手写**。

### 2.7 测试黄金文件（必改，~10 个文件）

LLVM 19 → 20 优化器输出会有微小差异（指令排序、寄存器分配等）。受影响的：

* [`tests/p2/correct_p2a*.txt`](tests/p2)、`timing_out_*.txt` 等带版本字符串 `llvm 19.1.7`
  的快照文件 → 需要重跑后重新提交基线。
* PTX 黄金文件（如果有）→ 重新生成。
* SPIR-V 黄金文件 → 与 LLVM 无关，不受影响。

### 2.8 dx12 后端（可选）

[`taichi/codegen/dx12/dx12_global_optimize_module.cpp:106-148`](taichi/codegen/dx12/dx12_global_optimize_module.cpp#L106-L148) 仍然用 `legacy::PassManager` 走 dx12 codegen。LLVM 20 的 dxbc/DXIL 后端有持续演进，但 Taichi 当前的 dx12 后端代码路径不依赖新接口，**短期内可保持不动**。长期 phase 6 计划仍然是迁移到 New PassManager。

## 3. 验证步骤建议

按风险从低到高：

1. **构建通过**：换上 LLVM 20，`build.py` 全跑，关注 deprecation warning 数量是否仅小幅增加。
2. **`tests/python/test_offline_cache.py`**：106 个测试覆盖了大部分编译路径，跑通可证明 IR
   生成 + 序列化兼容。
3. **`tests/python/test_basic.py` + `test_arith.py`**：核心数值正确性。
4. **CUDA 后端**：在有 CUDA GPU 的机器上跑 [`tests/p5/bench_parallel_compile.py --arch cuda`](tests/p5/bench_parallel_compile.py)，
   既能验证 NVPTX 后端，又能复测 P5 性能（CUDA 1.0× 是已知噪声内）。
5. **Vulkan 后端**：与 LLVM 无关，但跑一遍 SPIR-V 测试确认没引入 link 错误。
6. **AMDGPU**：[`taichi/runtime/amdgpu/jit_amdgpu.cpp`](taichi/runtime/amdgpu/jit_amdgpu.cpp) 用的是 legacy PassManager + ROCm 后端，需要单独测。
7. **AOT C-API**：跑 [`c_api/tests/`](c_api/tests/) 一轮。

## 4. 与 P5 / P1.b 的相互影响

* **P5 并行编译**：LLVM 20 的 `PassBuilder` / `PassManager` 都是线程安全的（每线程自己的实例），P5 的并发模型不受影响。
* **P1.b 字节镜像**：镜像存的是 `CompiledKernelData::dump()` 序列化后的字节，与 LLVM
  版本强绑定。**升级 LLVM 后必须 `ti cache clean`**，否则会反序列化失败（已经有 `check()`
  保护，会安全 fall-back 重新编译，但浪费一次 IO）。考虑后续在 cache 元数据里加一个
  LLVM 版本号字段做主动失效。

## 5. 推荐路径

短期（1–2 周）：
1. 起一个 `feature/llvm-20-upgrade` 分支。
2. 在本地 build LLVM 20.1.x 安装到 `dist/taichi-llvm-20/`。
3. 按本文档 §2.1 - §2.5 顺序改，每改一项跑一次最小验证。
4. 跑 §3 的 1–3 项测试，用 CI 跑 4–7 项。

中期（同 phase 6）：把 [`runtime/cpu/jit_cpu.cpp`](taichi/runtime/cpu/jit_cpu.cpp)、
[`runtime/cuda/jit_cuda.cpp`](taichi/runtime/cuda/jit_cuda.cpp)、
[`runtime/amdgpu/jit_amdgpu.cpp`](taichi/runtime/amdgpu/jit_amdgpu.cpp) 里残留的
`legacy::PassManager` 全部迁移到 New PassManager —— LLVM 21+ 可能彻底删除 legacy
头文件。

长期：在 P1.b 的镜像/磁盘元数据里加 LLVM 版本号 + libdevice 哈希做主动失效。
