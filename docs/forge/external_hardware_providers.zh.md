# 可选外部硬件 Provider 配置指南

[English version](external_hardware_providers.en.md)

Taichi Forge 的官方 runtime wheel 始终保持 driver-only。可选 CUDA/vendor library 由
应用自行安装和绑定版本，只在显式 probe 或实际使用时加载，并且绝不因此新增
`cu12`/`cu13` Forge wheel 变体。本文说明这一用户管理边界，并为 simulation 与 rendering
中最相关的可选 library 给出推荐配置。

本文是安装与部署指南。安装 library 本身不会选择执行路线；下列有界 operation 已有显式
retained-provider API，而 discovery probe 始终不执行算法。

## 支持状态与调用边界

| Library | Forge 状态 | 安装责任方 | Forge 发现方式 | 调用位置 |
| --- | --- | --- | --- | --- |
| cuBLAS | 已注册 D1 provider | 用户 CUDA 环境 | `ti.hardware.probe("cublas")` | direct Python 或 root Graph；不能在 kernel 内调用 |
| cuSPARSE | 已注册 D1 provider | 用户 CUDA 环境 | `ti.hardware.probe("cusparse")` | 领域级 auto/explicit 或 root Graph；不能在 kernel 内调用 |
| cuFFT | 已注册 D1 provider | 用户 CUDA 环境 | `ti.hardware.probe("cufft")` | 显式 plan 或 root Graph；不能在 kernel 内调用 |
| VkFFT 1.3.4 | 可选 ABI1 Vulkan JIT adapter | 当前 runtime 构建配置包含，旧产物可能没有 | `ti.hardware.probe("vkfft")` 或显式路径 | 固定存储 `VulkanFftPlan` 或 root Graph；无 FFT recipe 搜索 |
| cuDSS 0.8.x | 已注册 bundled-adapter ABI | Forge 提供 adapter；用户提供 vendor runtime | `ti.hardware.probe("cudss", library_path=...)` | 领域级 auto/explicit 或 root Graph；不能在 kernel 内调用 |
| OptiX ABI 93/105/118 | 已注册 bundled-adapter ABI | Forge 提供 adapter；用户/driver 提供 vendor runtime | `ti.hardware.probe("optix", library_path=...)` | 显式 scene/launch 或 root Graph；不能在 kernel 内调用 |
| Vulkan driver/ICD | D0 backend 依赖，不是 D1 provider | OS/GPU driver 安装 | `ti.init(arch=ti.vulkan)` 加 capability query | kernel 与已公开 native Vulkan API |
| cuSPARSELt 0.8.x-0.9.x | 已注册 bundled-adapter ABI | Forge 提供 adapter；用户安装可选包 | `ti.hardware.probe(...)` 或 `ti.hardware.tensor.CusparseLtProvider` | 显式 FP16 2:4 matmul plan；无 Graph/kernel/auto 路线 |
| cuTENSOR 2.0.x-2.7.x | 已注册 bundled-adapter ABI | Forge 提供 adapter；用户安装可选包 | `ti.hardware.probe(...)` 或 `ti.hardware.tensor.CutensorProvider` | 显式 FP32 contraction plan；无 Graph/kernel/auto 路线 |
| AmgX stable C API | 已注册 bundled-adapter ABI | Forge 提供 adapter；用户源码构建 | `ti.hardware.probe(...)` 或 `ti.hardware.linalg.AmgxProvider` | 显式 host-CSR solver；无 Graph/kernel/auto 路线 |
| NCCL | 不属于 Forge 当前单 GPU 范围 | 用户安装系统包 | 没有公开 Forge probe 或执行 API | 仅外部 multi-GPU communication |

已注册外部 provider 会出现在 `ti.hardware.providers()` 中。它们的 probe 检查有界版本族和
execution-symbol surface，但不创建 plan，也不资格化 workload。执行只通过已说明的领域 API
或显式 provider/plan API 开始。NCCL 仍未注册。

这些 host library 都不能从 `@ti.kernel` 内调用。当前只有文档明确说明的领域 API 可以自动
选择，例如已资格化的 cuSPARSE SpMV 与 cuDSS solver selection。安装 cuSPARSELt、
cuTENSOR、AmgX 或 NCCL 绝不会触发 compiler rewrite。

### Recording 与完整 recipe 搜索是不同能力

下表描述当前源码 API，不代表所有 vendor release、driver、GPU 或 workload 均已资格化。
库可以执行或录制，不等于其算法已作为 CompileIQ 搜索轴开放。

`ti.hardware.capability(operation_id).to_dict()["recipe_search"]` 提供静态的语义/provider 入口及更窄的
适用范围，与专家执行和 `graph_integration` 分开；它不加载可选库，也不证明当前 workload 可用。
`no_builtin_entry_declared` 只表示该 operation 未声明内建完整 recipe 入口，不禁止应用提供自己的 provider。

| Operation | 语义入口与准备 | Graph 与搜索边界 |
| --- | --- | --- |
| 固定 pattern 稀疏-稠密乘法 | `SparseMatrix.record_spmm(...)`，随后 `operation.prepare(input_array, output_array)` | CUDA f32 CSR / 紧凑 row-major 稠密数组；通过 `GraphBuilder.append_native()` 追加。显式 `ti.hardware.linalg.SparseSpmmRecipeProvider()` 将冻结的 direct/preprocessed 策略加入完整 recipe。 |
| Batched 2D complex FFT | `ti.linalg.record_fft(...)`，随后 `operation.prepare()` | CUDA complex-f32，紧凑 `(H, W, 2)` 或 `(batch, H, W, 2)` 数组，输入输出分离。显式 `ti.hardware.fft.FftRecipeProvider()` 在 whole-transform baseline 外提供逐图像列计划，以及 native 支持时的跨 batch 列计划。 |
| Toolkit reset-monoid segmented scan | 既有 `GraphBuilder.segmented_scan()` 加 `taichi_forge.hardware.source_providers` 中的 `CubSegmentedScanRecipeProvider(manifest_path)` | 可选 source-provider addon；有界 i32/u32 sum 与不可变 segmented layout。prepared capture、workspace 和 head-bitset 生命周期形成物理 recipe；addon 不在 portable runtime wheel 内。 |
| Driver-native segmented scan | `GraphBuilder.segmented_scan()` 与默认 recipe providers | 固定、互不重叠的 i32/u32 数组和不可变 segment；global correction 使用 retained CUDA recording 与 Graph-bound scratch，不依赖外部 Toolkit 库；仍是 fixed-resource action，不是 binding-frame region。 |
| Vulkan VkFFT | 显式 fixed-storage plan 或 root Graph recording | Vulkan JIT/source adapter；不意味着已有内建完整 FFT recipe 搜索或 CUDA binding-frame 接入。 |
| 其他 cuSPARSE / cuFFT / cuDSS expert operation | 既有显式 plan 和已说明的 root Graph recording | recording 本身不提供 recipe generator；cuDSS root 有序调用不能描述成 CUDA Graph capture。 |
| cuBLASLt matmul region | `ti.linalg.record_matmul(...)`，随后 `operation.prepare()` | CUDA 紧凑 scalar-f32、固定形状及可选 strided batch。显式 `ti.hardware.linalg.MatmulRecipeProvider()` 组合冻结算法/workspace、真实输入打包、独立/融合 ReLU；专家 retained-plan API 仍为私有。 |
| cuSPARSELt / cuTENSOR / AmgX | 下文的显式 provider plan | 当前没有公开 complete-recipe provider 或通用 Graph recording 路线。 |

先准备数学 operation，再 freeze Graph。SpMM/FFT/matmul 要求显式的 finite-input / f32 tolerance 合同，
Forge 不在每次 replay 扫描数值。FFT 正向、逆向均不归一化，连续应用两者会将输入乘以 `H * W`。
layout、精度和归一化属于语义要求，不是优化器选择。vendor 不开放的内部信息报告为 unknown，
不能据此虚构内部 kernel 数。

### Matmul 准备与复用

语义为 `D = activation(alpha * op(A) @ op(B) + beta * D)`，activation 支持 `identity` 或 `relu`。
transpose、系数、dtype、形状和调用者确认的 tolerance 都是语义事实，不是搜索轴。输入值可在每次 replay 改变；
输出不能与任一输入重叠，输入之间允许只读别名。tolerance 声明不自动保证任意输入的精度，仍由 evaluator/下游验证。

```python
operation = ti.linalg.record_matmul(
    512, 512, 512, transpose_a=True, activation="relu",
    absolute_tolerance=2e-5, relative_tolerance=2e-5,
)
operation.prepare(workspace_limit_bytes=32 << 20, heuristic_limit=4)
builder = ti.graph.GraphBuilder()
builder.append_native(operation)
definition = builder.freeze()
providers = (*ti.graph.default_recipe_providers(),
             ti.hardware.linalg.MatmulRecipeProvider())
```

把同一 provider set 与常规 workload/evaluation/backend 合同传给 `definition.search_recipes()`；CompileIQ
只调度完整 recipe ID。`definition.compile()` 物化 baseline。此语义描述必须先 **freeze 再执行**，不是可以直接
运行的专家 plan。prepare 只查询元数据、冻结公开算法配置，不序列化 opaque 执行字节、不执行 matmul，也不分配
候选 GPU workspace。物化时只重建所请求的计划和精确 workspace，不重跑 heuristic。
选中输入转置打包时，每次 replay 都刷新 Graph 私有数组，不假设输入恒定、不引入隐式值缓存。仅改变等价描述符
写法不生成布局候选。融合 activation 的中间输出不可见；需要该输出时应建立分开的语义操作。

与 `decision.selection_artifact` 一起保存 `operation.preparation_artifact()`。新进程用相同参数构造
`record_matmul(..., preparation=saved)` 和等价 Graph，再调用 `resolve_recipe(saved_selection, providers=providers)`
与 `materialize(selection)`。导入的准备耗时仍是历史事实；设备/component/配置漂移在冷边界拒绝，不静默替换算法。
operation.close 不破坏存活 Graph 持有的计划。固定绑定检查放在 Graph.bind 发布边界，immutable binding replay
不增加 matmul 数值扫描或 provider 校验。

支持的 native runtime 还可组合不可变参数帧 recipe；仅有 typed matmul capture、尚无 frame 能力的旧 runtime
不会生成这一组合。cuBLASLt 及传递库仍由用户配置（`TI_CUBLASLT_LIBRARY_PATH`），不改变普通 kernel、runtime auto
或 wheel 依赖类别。精确 workspace/私有数组与参数映像单列，vendor/driver 未知存储不伪装为零。融合和打包都不是
通用赢家，host、device 和显存证据分开判断。

### FFT 与 SpMM 准备及复用

两种分离 FFT 都先批量变换所有行，再利用输出数组原地变换列，无额外 dense transpose buffer。
逐图像方案在列阶段调用 batch 数次；跨 batch 方案每列调用一次，单次批量处理独立图像。这改变物理
launch 和访存组织，不改变数学归一化或布局；都不是通用赢家，whole-transform 也可能优于两者。
`prepare()` 记录实际 workspace/准备事实；FFT provider domain 扩展使旧 provider-bound 搜索证据失效，
不与 wheel commit 绑定。旧 native 不生成新增策略；导入的选择若需要它，物化时明确失败，不替换成其他计划。

FFT Graph recording 只持有所用的物理计划，不反向持有搜索 operation 的全部候选计划。`operation.close()`
释放 operation 的准备阶段所有权，并禁止继续调用它的 `prepare()` 或 `compile()`；已构建 Graph 的计划租约
继续有效。未使用计划在最后一个执行拥有者释放后可退休。冻结 recipe 元数据仍可读取，后续物化只重建所请求的
已退休计划，在该冷边界核对 component/workspace 与准备事实是否一致。冻结 definition 持有 FFT 描述，
不再持有 baseline 计划；释放搜索 operation 与 builder 后可仅驻留所选计划，其他仍存活的 Graph 则合法保有
自己的计划。baseline `definition.compile()` 在编译边界重新获取计划，不在 replay 时恢复。
跨进程无计划恢复时，与搜索 selection artifact 一起保存 `operation.preparation_artifact()`。在新进程用
相同参数重建 `record_fft(..., preparation=saved_preparation)`，加入等价 builder 后 freeze，再正常调用
`definition.resolve_recipe(...)` 和 `definition.materialize(...)`。这条路径不要调用 `prepare()`，否则会
显式准备全部导入候选。freeze、catalog discovery、选择解析均不创建 FFT 计划，物化只创建所请求的计划。
此功能需要新的 native capture-description 能力；旧兼容 runtime 仍可运行普通 FFT，但会明确拒绝该恢复路径。

SpMM 通过 `matrix.record_spmm(..., preparation=saved_preparation)` 使用同样的准备 JSON/选择恢复流程。
native 支持计划租约时，`operation.close()` 只释放搜索拥有的计划；同一 matrix/RHS/algorithm 的计划仍与
存活 Graph 命令和原有专家接口缓存共享。冻结 definition 持有矩阵身份和预期事实，不持有 baseline 计划。
与 FFT 不同，SpMM 创建所选计划需要真实 dense bindings：freeze、resolve、materialize 均不创建计划，
准备参数帧的 bind（普通 executor 则是首次 native preparation）只创建并核对所选计划。导入准备事实不代表
已经测量恢复耗时。workspace 是计划分配量，不是 driver 总驻留或 device peak。旧 runtime 保持原矩阵缓存，
明确拒绝 selected-only artifact 导入。

准备 artifact 仅包含预期 JSON 事实，不含 Python executable 或 vendor 二进制。恢复时核对语义、设备和 component，
物化时再次核对实际所选计划的 component/workspace；库探测只在这个显式冷边界发生。报告标明导入的准备观测属于
历史数据，计划重建的实测另行记录。库名/版本相同不证明二进制身份或生产性能，该 artifact 不是 AOT 或二进制缓存。

这些 provider 需要在 `definition.search_recipes(engine="compileiq", providers=..., ...)` 中，
与 `ti.graph.default_recipe_providers()` 一同显式传入；没有匹配的已准备语义 region 时，不能凭
provider 名称制造 region。维护的魔改 CompileIQ fork 只调度 opaque complete-recipe identity；
Forge 负责组合、冻结物理配置和物化。计划重建失败不能静默改选另一 vendor heuristic。
安装库或选中一个实测 recipe 都不改变普通 runtime auto。

选择报告保留 setup/first/steady 成本、声明的数值合同、component identity 和显存范围。
CompileIQ 的 trial memory 最大观测值不是 driver 实测的 device peak。冷物化、after-evaluator
资源快照、请求 workspace 和 pool reservation 是不同观测；缺测写 unavailable，不能填零。
生产采用由下游 workload 的实际复用次数和精度要求决定。search、resume、选择重解析和生命周期
成本报告见 [Graph API 参考](forge_api_reference.zh.md)。

已准备的 FFT、SpMM region 策略也可与默认 CUDA immutable binding-frame 执行器组合。计算 region
仍独占其语义覆盖；执行器包围装配后的计算，不替换计算本身。只有显式声明兼容的 region provider
参与，一个 recipe 最多选择一个执行器，其他 family 不会自动获得兼容性。物化保留所选固定计划，
`graph.bind(...)` 在排队 replay 前准备不可变参数帧；传入普通 mapping 仍包含准备成本。组合可在
保留所选 device 算法的同时减少 host 重提交，但不保证所有组合更快或显存更少。

### 显式启用的诊断设施

NVTX 3 标注使用随构建引入的 header，不要求 `nvToolsExt` shared library；用于显式 profiling 中
关联 stage/trial/recipe 与 GPU 工作，不是物理策略或自动性能门槛。

`ti.hardware.gpu_environment()` 显式采集 NVIDIA 设备的 driver-provided NVML 信息。
在同一线程以 `ti.hardware.capture_trial_environment()` 包围 `session.run(evaluator)`，可把边界
观测附加到报告。NVML 缺失或字段不支持时返回结构化 unavailable。NVML 显存是全设备值，包含
其他进程，不是 recipe/process peak；clock、power、temperature 快照也不是 trial 均值。
采集没有 replay 轮询线程或额外 device 同步，但 host 采集时间计入外层搜索预算。被动
`report()` / `telemetry()` 不会隐式启用它或探测外部库。

## 打包与版本规则

所有可选 provider 都应遵守以下规则：

1. 正常安装 Forge，随后在应用环境中安装可选 vendor 包；不得把 vendor runtime 复制进
   Forge wheel 或 Forge package 目录。Forge 自有的薄 C-ABI adapter 可以位于现有 runtime
   wheel，但不得链接或携带 vendor runtime。
2. 同一个环境中，同一种 library 只选择一个 CUDA-major package family。`-cu12` 或
   `-cu13` 后缀描述 vendor package，不描述 Forge wheel。
3. 核对 GPU architecture、driver、provider release、CUDA family、OS 与 Forge operation
   合同的交集。Forge CUDA kernel 可运行不能证明可选 provider 兼容。
4. Forge provider 支持显式路径时，优先绑定绝对 `library_path`；否则必须在 Python process
   启动前配置 OS loader。
5. 在应用部署 manifest 中记录选中的 package version 与解析后的 shared-library path；不要
   依赖搜索顺序中偶然出现的第一个看似兼容 DLL/shared object。
6. 先做显式 probe，再用与生产一致的 dtype 和 operation family 跑最小正确性检查。probe
   成功只证明 discovery 和 ABI symbol，不证明数值正确性或性能。

对于 Python wheel 安装的 component，以下命令可以列出真实文件，而不假定 package 布局：

```bash
python -m pip show -f PACKAGE_NAME
```

Linux 应在 process 启动前，把所选 `.so` 所在目录写入 adapter RPATH 或
`LD_LIBRARY_PATH`。Windows 应把所选 `.dll` 所在目录加入当前 process 的 `PATH`；也可以在
native load 之前调用 `os.add_dll_directory()`。transitive dependency 目录必须在同一时间段
保持可见。

不要用 `setx PATH` 验证当前 shell：它只影响未来启动的 process。在 PowerShell 中，临时
测试应使用 `$env:PATH`。

### Vulkan driver、ICD 与 SDK 边界

官方 Forge wheel 的普通 Vulkan 使用需要兼容 GPU driver 与 Vulkan ICD，但不要求用户安装
Vulkan SDK。device extension 与 feature combination 会在 runtime 资格化，不支持的切片
fail closed。安装 validation layer 或更新 SDK 不能增加 driver 没有暴露的 device feature。

Vulkan SDK 是提供 header、tool 与 validation 的源码构建/开发依赖，不是可选执行 provider，
也不能成为创建 Vulkan-versioned Forge wheel 的理由。未来任何 external Vulkan library 都
必须定义自己的 provider ABI 与 lifetime 合同，不能仅因环境中存在 SDK 就被隐式加载。

### 可选 CUDA 编译 provider

Forge 的默认 CUDA kernel 路线仍把 PTX 交给 CUDA Driver JIT，不要求 CUDA Toolkit，也不
执行外部程序。需要离线 cubin 或编译器级实验优化的部署，可以在启动前选择外部 `ptxas`：

```powershell
$env:TI_CUDA_PTXAS_MODE = "external"
$env:TI_CUDA_PTXAS_PATH = "C:\CUDA\bin\ptxas.exe"
$env:TI_CUDA_ARTIFACT_CACHE_PATH = "D:\cache\taichi-cuda-artifacts"
```

Linux 可以把 `TI_CUDA_PTXAS_PATH` 设为绝对路径，或让 `ptxas` 可由当前 process 的 `PATH`
解析。Forge 不把 `ptxas`、CUDA Toolkit、CompileIQ 或 Python 优化包放进 wheel；这些工具
及其版本均由应用环境管理。未设置 `TI_CUDA_PTXAS_MODE=external` 时，其它编译 provider
变量不会改变默认 Driver JIT 路线。

| 变量 | 合同 |
| --- | --- |
| `TI_CUDA_PTXAS_MODE` | `driver`（默认）或 `external` |
| `TI_CUDA_PTXAS_PATH` | 可选的 `ptxas` 绝对路径；省略时使用 `PATH` |
| `TI_CUDA_ARTIFACT_CACHE_PATH` | cubin、校验和、lock 与 worker manifest 的持久 cache 根目录 |
| `TI_CUDA_PTXAS_TIMEOUT_SECONDS` | 单次 cache-miss `ptxas` process 的有界 timeout，默认 60 秒 |
| `TI_CUDA_PTXAS_ACF_PATH` | 可选的静态 Advanced Controls File；与 worker 二选一 |
| `TI_CUDA_COMPILEIQ_WORKER` | 可选的用户 worker executable 或 Python script |
| `TI_CUDA_COMPILEIQ_PYTHON` | 执行 worker script 的独立 Python，可与 Forge Python 不同 |
| `TI_CUDA_COMPILEIQ_TIMEOUT_SECONDS` | 单次 cache-miss worker 的有界 timeout，默认 3600 秒 |

所有变量必须在 `ti.init()` 前确定。同一个 CUDA session 首次加载 module 后，Forge 会拒绝
切换 provider identity；需要改变配置时先让旧 work 完成，调用 `ti.reset()`，再用新配置
初始化。cache key 绑定 PTX、GPU target、编译选项、Forge artifact schema、`ptxas` 内容与
版本，以及 ACF/worker identity。cache hit 直接加载已校验 cubin，不重复启动 worker 或
`ptxas`；首次 binary hash、worker 与 `ptxas` 都属于固定编译成本，不属于 kernel 的规模
相关执行成本。

CUDA Advanced Controls File 通过 `ptxas --apply-controls` 应用，因此要求 `ptxas` 13.3 或
更新版本。静态 ACF 适合已离线资格化的固定 kernel family。由于 ACF 是实验性 compiler
control，应用必须保留数值 oracle、compile timeout、目标 GPU 和 `ptxas` 版本，并在任何
compile/校验失败时停用该配置；Forge 不会在失败后静默执行另一个显式 provider。

对于本节的 external PTXAS/ACF process 路线，CompileIQ 不导入 Forge application；用户应在
独立、受支持的 Python 环境中安装选定的上游 release，并提供 workload-specific worker：

```powershell
py -3.11 -m venv C:\venvs\compileiq
C:\venvs\compileiq\Scripts\python.exe -m pip install compileiq
$env:TI_CUDA_PTXAS_MODE = "external"
$env:TI_CUDA_COMPILEIQ_WORKER = "D:\app\forge_compileiq_worker.py"
$env:TI_CUDA_COMPILEIQ_PYTHON = "C:\venvs\compileiq\Scripts\python.exe"
```

所选上游 CompileIQ release 的 Python 支持范围可能窄于 Forge；部署时必须重新核对其 Python
与 CUDA/`ptxas` support table。这个独立解释器约束不会改变 Forge wheel 自身的 Python
支持矩阵。

这个 process worker 与 `ti.graph.compileiq_recipe_search()` 不同。后者是可选离线 Graph recipe
API，要求魔改 fork 提供兼容的 V2 完整 recipe capability 与 main-thread staged-search worker。
接受条件是协议 epoch、必需 schema/API 以及自洽的 core/capability identity，不绑定某个 fork
commit 或 wheel hash。Forge 会记录已安装 Python-source identity 并绑定 checkpoint，源码漂移会
使 resume 证据失效。该资格化 fork 支持 Python 3.10--3.14；普通上游安装或上述 external JSON
worker 都不能替代它。按 task 索引的 kernel/offload 搜索仅保留为私有资格化基础，不属于公共 API。

Forge 使用 versioned JSON v1 process protocol 调用：

```text
PYTHON WORKER --request REQUEST.json --response RESPONSE.json
```

request 包含 PTX 临时路径、artifact key、target、entry manifest、编译选项和精确 `ptxas`
identity。worker 必须原子写出以下一种 response：

```json
{"schema_version": 1, "status": "pass"}
```

或：

```json
{
  "schema_version": 1,
  "status": "ok",
  "acf_path": "C:/absolute/path/controls.acf",
  "acf_sha256": "EXPECTED_SHA256"
}
```

`pass` 表示该 artifact 使用普通 external `ptxas`；`ok` 表示先校验并复制 ACF，再调用
`ptxas`。worker 必须自己定义代表性输入、目标函数、正确性与生命周期 gate。Forge 只在
compile 阶段拥有 PTX 和静态选项，并不了解任意 kernel 的生产输入或物理不变量，因此不会
自动替应用运行全局 autotuning。worker 非零退出、timeout、非法 JSON/status/path/checksum
或不支持的 `ptxas` 都 fail closed。

## 用户环境中的已注册 provider

### cuBLAS、cuSPARSE 与 cuFFT

这些 provider 使用复制的稳定声明与 runtime symbol loading；Forge 不使用 Toolkit header，
也不把 library 链接进官方 wheel。用户可通过兼容 CUDA Toolkit，或与应用 CUDA family
匹配的 NVIDIA component wheel 提供它们。例如，把下面的 `XX` 替换为 `12` 或 `13`，不要
原样执行：

```text
python -m pip install nvidia-cublas-cuXX nvidia-cusparse-cuXX nvidia-cufft-cuXX
```

这三个 provider 的 library path 是 implicit，必须在 Python 启动前配置系统 loader；
`library_path=` 会被有意拒绝。CUDA 12+ cuSPARSE 还可能依赖 `nvJitLink`，因此所选
provider 的 transitive-library 目录也必须可见。

在已初始化 CUDA runtime 上显式验证：

```python
import taichi_forge as ti

ti.init(arch=ti.cuda)
print(ti.hardware.probe("cublas"))
print(ti.hardware.probe("cusparse"))
print(ti.hardware.probe("cufft"))
```

推荐 lifecycle：

- 复用 Program-scoped cuBLAS handle 和 matrix-scoped cuSPARSE
  descriptor/preprocessing，不要在每次调用前后重新创建。
- 复用 fixed-size cuFFT plan。workspace 与 layout 属于 plan 合同；shape、transform kind、
  dtype、device 或 runtime 改变时重建。
- 显式请求必须保持失败可见。显式 provider 不可用或执行失败，不等于允许静默复制到 host
  或选择另一算法。

### cuDSS 0.8.x

平台 `taichi-forge-runtime` wheel 携带一份基于官方 cuDSS 0.8 header 构建的 Forge 薄
C-ABI 1 adapter。adapter 不链接 `cudss`、CUDA runtime、cuBLAS 或 Python，也不会创建新的
wheel 变体；用户无需重编 Forge。厂商 cuDSS runtime 及其传递依赖仍由应用环境提供。

Forge 公开切片绑定 cuDSS 0.8.x。安装与应用 CUDA family 匹配的 package：

```bash
# 一个环境中二选一，不要同时安装。
python -m pip install "nvidia-cudss-cu12>=0.8,<0.9"
python -m pip install "nvidia-cudss-cu13>=0.8,<0.9"
```

cuDSS 还需要兼容 cuBLAS library。当前 Forge resolver 的优先级为：

1. `library_path=` 参数；
2. `TI_CUDSS_LIBRARY_PATH`；
3. 与当前 CUDA driver family 匹配的已知 `nvidia` namespace-package 路径。

显式 path 可以是 shared library，也可以是包含它的目录。它是唯一候选：路径错误时 Forge
不会回退到其它位置。Linux 上 NVIDIA cuDSS wheel 可能只包含 `libcudss.so.0` 而没有无版本
symlink；Forge 会直接解析 versioned library。`library_path` 始终指 vendor runtime；wheel
内部 adapter 不属于公开路径合同，也不能被覆盖。

```powershell
# 可选：为当前 Windows process 显式绑定部署库。
$env:TI_CUDSS_LIBRARY_PATH = "C:\vendor\cudss\bin\cudss64_0.dll"
```

```bash
# 可选：Linux 显式部署绑定。
export TI_CUDSS_LIBRARY_PATH=/opt/vendor/cudss/lib/libcudss.so.0
```

创建 solver 前，先验证精确部署候选：

```python
import os
import taichi_forge as ti

ti.init(arch=ti.cuda)
path = os.environ.get("TI_CUDSS_LIBRARY_PATH")
report = ti.hardware.probe("cudss", library_path=path)
print(report)
```

probe 会瞬时加载 adapter 和 vendor runtime、查询其 0.8.x 版本后立即释放；不会创建或保留
solver handle、factor 或 workspace。只有 `CudssPlan` 拥有执行期 adapter/runtime handle，
并随 plan 确定性关闭。

Forge 当前要求 CUDA Driver API 12.0 或更高版本，以及 square scalar f32 CUDA CSR matrix。
`CudssPlan` 把 `analyze()`、`factorize()`/`refactorize()` 与 `solve()` 分开：

- sparsity pattern 不变时复用 analysis；
- pattern 与 values 都不变时复用 factor；只有 values 变化时使用 `refactorize()`；
- 在 direct call 和已提交 root-Graph action 全部 retire 前保持 plan 存活，之后确定性关闭；
- 为不透明的 analysis、factor 与 workspace memory 预留预算；CSR input bytes 不是 provider
  总峰值；
- 只有存在匹配的 Forge admission evidence 时才使用 `provider="auto"`。没有 evidence 时，
  auto 不会 probe cuDSS，并保留 cuSOLVERSp。应用有意选择时使用 `provider="cudss"`。

推荐的 physics workload 是反复求解的 fixed-pattern sparse system，其中 analysis，通常还有
refactorization，能够被充分摊销。对于一次性、小规模或频繁 remesh 的系统，应测量完整
analysis-factor-solve lifecycle，不能只看 solve 时间。

### OptiX runtime provider

每个平台的 `taichi-forge-runtime` wheel 都携带三份 Forge 自有的薄 adapter，分别用固定的
NVIDIA 官方 header 构建：ABI 93（OptiX 8.1）、ABI 105（OptiX 9.0）和 ABI 118
（OptiX 9.1）。三者共用 Forge provider C ABI 1，并位于同一个 wheel 中。用户不需要安装
OptiX SDK、CUDA Toolkit，也不需要重编 Forge。wheel 仍不包含 `nvoptix.dll` 或
`libnvoptix.so.1`，也不会产生 CUDA/OptiX 版本化 wheel 变体。

vendor runtime 通常由 NVIDIA display driver 提供。ABI 93、105、118 的最低 driver branch
分别为 R555、R570、R590。Forge 从最新到最旧尝试 adapter，保留 installed runtime 接受的
第一项。较新 ABI 不受支持是允许回退的明确条件；context 或 scene 创建之后的执行错误不能
触发静默换实现。

adapter 内嵌由发布流程固定的 CUDA 12.5.x 编译器生成的 `compute_75` PTX 8.5。这个依赖只
存在于构建期，不进入 wheel，也不要求用户安装 CUDA Toolkit。构建过程会审计 PTX 上限，
防止未来发布编译器静默抬高 ABI 93 / R555 的 driver 下限。

runtime 发现顺序如下：

1. 指向 `nvoptix.dll` 或 `libnvoptix.so.1` 的 `library_path=` 参数；
2. `TAICHI_FORGE_OPTIX_LIBRARY`；
3. OptiX loader 实现的标准 NVIDIA driver 搜索路径。

显式路径是唯一候选，适用于 container 或非标准 driver layout。它始终表示 vendor
runtime；Forge adapter 是 runtime wheel 的内部资源，不能通过公开 API 覆盖。`probe()`
会瞬时加载 adapter 与 vendor runtime 来核对精确 ABI，但不会创建或保留 CUDA/OptiX
context。

```python
import taichi_forge as ti

ti.init(arch=ti.cuda)
print(ti.hardware.probe("optix"))

with ti.hardware.ray.load_optix_provider() as provider:
    # 创建 scene，在已提交任务完成前保持存活，然后关闭 scene。
    pass
```

如果 vendor runtime 位于非标准路径，可显式绑定：

```python
vendor_runtime = "/opt/nvidia/lib/libnvoptix.so.1"  # Windows 使用 nvoptix.dll
print(ti.hardware.probe("optix", library_path=vendor_runtime))
with ti.hardware.ray.load_optix_provider(vendor_runtime) as provider:
    pass
```

provider 持有 OptiX context，scene 持有 provider，已提交 Graph work 同时持有两者。必须先
关闭 scene，再关闭 provider；`ti.reset()` 后不得复用任何旧对象。`validation=True` 推荐用于
开发阶段，而不是默认性能配置。

## 显式 optional runtime 执行 provider

标准 runtime wheel 随附以下三个 Forge 自有薄 adapter。adapter 不包含也不链接 vendor
code；`probe()` 只瞬时加载用户 runtime，而创建 provider 会保留选中的 runtime 并公开有界
execution ABI。所有路线都是显式 host-side resource，不是 Graph action、kernel intrinsic
或 automatic rewrite。

### cuSPARSELt 推荐配置

cuSPARSELt 加速某一 operand 满足 provider 50% structured-sparsity 合同的 matrix
multiplication。它不是通用 sparse-matrix solver，也不替代普通 CSR SpMV。

安装一个 CUDA family，并查看真实 shared-library 位置：

```bash
# 二选一。
python -m pip install nvidia-cusparselt-cu12
python -m pip install nvidia-cusparselt-cu13
python -m pip show -f nvidia-cusparselt-cu13
```

可以显式传入文件/目录。省略 `library_path` 时，Forge 先读取
`TI_CUSPARSELT_LIBRARY_PATH`，再检查已安装 NVIDIA package 文件，最后交给 OS loader：

```python
report = ti.hardware.probe(
    "cusparselt", library_path=r"C:\absolute\path\cusparseLt64_0.dll"
)
probe = next(
    item for item in report.operations
    if item.descriptor.operation_id == "runtime.probe.cusparselt"
)
assert probe.discovery == "available"
```

使用显式 retained plan 执行。`A` 必须已经严格满足 2:4 sparsity；Forge 不做 pruning，也不
静默修改数值 operator。`B` 使用 row-major `(n, k)` 转置存储：

```python
with ti.hardware.tensor.CusparseLtProvider(runtime_path) as provider:
    with provider.matmul_plan(m, n, k) as plan:
        plan.compress(a).execute(b_transposed, c, d, alpha=1.0, beta=0.0)
        ti.sync()
```

必须遵守所选 release 的 support table。当前 cuSPARSELt 文档要求 compute capability 8.0
或更高版本；当前 release line 要求 CUDA 12.9 或更新的软件栈和兼容 driver。旧 package
release 的要求不同；package 可安装不等于运行兼容。

应用 adapter 应持有以下 lifecycle：

1. 创建 handle、dense/structured matrix descriptor、matmul descriptor、algorithm
   selection 与 plan；
2. 根据精确 dtype、layout、transpose mode、alignment 和受支持 architecture，检查或有意
   prune structured operand；
3. 查询 compressed storage 和 workspace size，压缩 structured operand，并保留 compressed
   data 与 plan；
4. 执行多次兼容 matmul。structured operand values 变化时重新压缩；shape、stride、dtype、
   operation、device 或 library ABI 变化时重建 descriptor/plan；
5. 使用这些资源的 stream work 完成后才能销毁资源。

推荐应用策略（这不是 Forge API 参数）：

```yaml
provider: cusparselt
activation: explicit
sparsity_policy: already_valid_2_of_4
automatic_pruning: false
plan_cache_key: [device, library_version, dtype, shape, strides, transpose]
algorithm_search: offline_opt_in
min_matmuls_per_compression: 4
fallback_owner: application
```

`min_matmuls_per_compression: 4` 只应视为保守起始门槛；必须资格化精确 workload，并在
compression/setup 占主导时提高门槛。`cusparseLtMatmulSearch()` 只能作为会反复执行的
operation 的离线/初始化 autotuning；不要把 search、pruning、compression 或 plan creation
放进 simulation step loop。

对于 physics workload，automatic pruning 通常不安全：为满足 2:4 sparsity 而修改 mass、
stiffness、Jacobian、contact 或 constraint matrix，会改变数值 operator。只有模型或 learned
operator 原本就定义了 structured pattern，或应用明确接受 approximation 并配置 residual、
conservation 与 stability 检查时，才能使用 cuSPARSELt。dense-like、反复使用的 local/block
operator，以及 batched constitutive/reduced-order transform，通常比不规则 global CSR
system 更合适。

admission 必须计入完整摊销成本：

```text
plan + prune/check + compression + repeated matmul + extra memory
```

先通过数值验收，再要求在预期复用次数下最差 timing 仍有正收益。只有最慢的资格化 sample
仍通过应用门槛时，较高 timing variance 才是可接受的。

### cuTENSOR 推荐配置

安装与应用 CUDA family 匹配的 package：

```bash
# 二选一。
python -m pip install cutensor-cu12
python -m pip install cutensor-cu13
```

显式传入路径或设置 `TI_CUTENSOR_LIBRARY_PATH`。两者都没有时，Forge 先检查已安装的
`cutensor-cu13`/`cutensor-cu12` package 文件，再使用 system loader：

```python
report = ti.hardware.probe("cutensor", library_path="/opt/cutensor/lib/libcutensor.so.2")
```

当前执行面是 compact row-major scalar `f32` contraction，compute 支持 `f32` 或 `tf32`；
mode 显式定义 contraction：

```python
with ti.hardware.tensor.CutensorProvider(runtime_path) as provider:
    with provider.contraction_plan(
        (m, k), "ik", (k, n), "kj", (m, n), "ij", (m, n), "ij"
    ) as plan:
        plan.execute(a, b, c, d, alpha=1.0, beta=0.0)
        ti.sync()
```

重复 contraction 可通过 `plan.record(alpha=..., beta=...)` 进入 root CUDA Graph；
`a`、`b`、`c`、`d` 是可在创建 recording 时指定的符号绑定名：

```python
recording = plan.record(alpha=1.0, beta=0.25)
builder = ti.graph.GraphBuilder()
builder.append_native(recording)
graph = builder.compile()
bindings = graph.bind(dict(a=a, b=b, c=c, d=d))
graph.run(bindings)
```

recording 持有已准备的 vendor plan 和精确 workspace。依赖仍存活时，plan/provider 的
close 会明确拒绝；应先释放 Graph、builder、definition 和 recording，再关闭 plan。
runtime reset 会在 CUDA Program finalize 前释放这些资源。capture 不执行数学运算，
不会提前推进 `beta*C` 反馈。C/D 仅在 layout/modes 相同时可以共享存储，D 不得 alias A/B。
shape、dtype 和存储合法性在 bind/capture 边界确定，不在 steady replay 扫描。
此固定 recording 也可组合 immutable binding frames；不提供独立 `recording.execute()`、
nested sequential 或 AOT recording。这个执行入口本身不开放 contraction 策略搜索，
也不改变普通自动选择。

cuTENSOR vendor runtime 及其 CUDA 依赖仍在 portable wheel 外，Forge 自有 thin adapter
和 capture bridge 可以随 wheel 提供。当前 Forge 执行/recording 只覆盖 contraction，
不代表 vendor 更广的 reduction、permutation 和 elementwise API 都已接入。
Windows capture 已使用 cuTENSOR 2.7/CUDA 13 验证，不代表所有支持的 vendor 版本或平台
均已测试。资源报告区分已知 workspace 字节和未知 vendor state，不宣称观测了 driver
峰值显存。

推荐 adapter 策略：

- 按 operation、dtype/compute type、layout、shape、workspace limit、device、CUDA version
  与 cuTENSOR version 缓存 descriptor/plan；
- 设置显式 workspace budget，并查询 plan 的实际需求；
- 为获得可预测 startup 和 Graph compatibility，默认关闭 JIT；只有经过 profiling、且
  contraction 会反复执行时才启用；
- 持久 plan cache 只对相同 cuTENSOR version、CUDA version，以及匹配的 GPU
  architecture/multiprocessor configuration 有效；cache 不匹配必须拒绝，不能静默复用；
- 对 setup 与通用 layout dispatch 成本无法摊销的小型 fixed-shape elementwise/contraction，
  继续使用手写 kernel；
- 不能根据 library 名称推断使用了 Tensor Core；实际硬件路线由 dtype、compute descriptor、
  selected plan 与 device 共同决定。

### AmgX 推荐配置

AmgX 是完整、可配置的 algebraic-multigrid/Krylov solver，不是 kernel intrinsic。应从
CUDA 与 architecture 支持匹配部署环境的 NVIDIA release，把它构建为应用依赖：

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" \
  -DCMAKE_NO_MPI=ON
cmake --build build --config Release --target amgxsh
```

用部署 GPU 替换 architecture list。近期 AmgX release 使用
`CMAKE_CUDA_ARCHITECTURES`；采用旧 `CUDA_ARCH` 示例前必须核对对应 release notes。
single-GPU adapter 设置 `CMAKE_NO_MPI=ON`；distributed build 还需要兼容 MPI
implementation。把生成的 `amgxsh.dll` 或 `libamgxsh.so` 目录加入 runtime loader path。
upstream 支持 Windows，但其 Windows test coverage 更有限，因此必须放在显式部署资格化之后。

AmgX 不做默认 Python package 搜索。显式传入构建出的 library，或设置
`TI_AMGX_LIBRARY_PATH`；兼容的 CUDA、cuBLAS 与 cuSPARSE 依赖必须同时对 loader 可见：

```python
report = ti.hardware.probe("amgx", library_path="/opt/amgx/lib/libamgxsh.so")
```

执行接口接受连续的 host scalar CSR array 与 host vector。AmgX 持有 device upload、
hierarchy 和 solver resource；topology 复用时应保留 solver，并传入应用自有的精确配置：

```python
with ti.hardware.linalg.AmgxProvider(runtime_path) as provider:
    with provider.solver(offsets, columns, values, "PCG_V.json", config_file=True) as solver:
        solution, info = solver.solve(rhs)
        assert info["converged"]
```

`replace_coefficients()` 在替换数值后总会刷新 solver setup。vendor 导出可选/已弃用的
`AMGX_solver_resetup` C symbol 时，adapter 使用该 fast path；否则执行完整的
`AMGX_solver_setup`。因此，只保留稳定 setup API、未导出 resetup entry point 的 runtime
仍可执行；fallback 数值语义正确，但可能重建更多状态。

对 CSR topology 固定、coefficient 反复变化的 workload，应在 AMG 或 AMG preconditioner
scope 中显式调整 AmgX 自己的 `structure_reuse_levels`。`0` 重建 hierarchy；正数逐级保留
更多已有 level structure。复用 level 时不会重算 prolongation/restriction operator，但会重算
coarse matrix，因此 Forge 绝不自动提高该设置。部分 AmgX release 还接受 `-1` 以保留所有
level；它只能作为 release-qualified 选择，不能作为可移植默认值。任何非零设置都必须针对
预期 coefficient 变化范围，通过 residual、convergence、iteration count、最差 update time
和 peak memory 门禁。

必须先关闭每个 plan/solver，再关闭 provider；`ti.reset()` 后两者都不得复用。存在 live child
resource 时 provider close 会失败。显式选择后的 load、数值和 lifetime 错误会直接暴露，
不会静默 fallback。

推荐的 physics 起点：

- SPD elliptic/Poisson-like system：从随附 `PCG_V.json` 或
  `PCG_AGGREGATION_JACOBI.json` 开始，并验证 symmetry/positive-definite 合同；
- nonsymmetric system：从 `FGMRES_AGGREGATION.json` 或随附 BiCGSTAB 配置开始；不能只按
  matrix size 选择 PCG；
- sparsity topology 不变时保留 setup 与 hierarchy object。只有所选 AmgX release 明确支持
  `structure_reuse_levels`，且应用实际验证该 lifecycle 时，才允许复用 hierarchy；
- 同时为 residual/convergence、iteration、setup time、solve time 和 peak memory 设门禁；
  AMG hierarchy memory 可能超过原始 CSR storage；
- 随部署保存精确 JSON configuration。AmgX tuning surface 很大，只按 library version
  宣称性能没有意义。

## 未注册候选：NCCL

本次 adapter 挂载明确不包含 NCCL；它目前没有 Forge probe 或执行 API。

### NCCL 推荐配置

NCCL 只与 multi-GPU 或 multi-node communication 有关，不能加速 single-GPU kernel。
Forge 推荐的候选范围为 Linux；NVIDIA install guide 在 Linux 上提供 `libnccl2` 与
`libnccl-dev` package：

```bash
sudo apt install libnccl2 libnccl-dev
```

未 pin 的 repository install 可能升级 CUDA。需要保留旧应用栈时，应 pin NCCL/CUDA
package version。adapter 应为每个参与 device/process group 保留 communicator，把 collective
绑定到显式 CUDA stream，传播 asynchronous error，并在部分初始化失败时 abort/close 所有
communicator。

physics candidate 包括 halo exchange、distributed vector reduction、dot product，以及已
partition solver 中的 coarse-grid/global synchronization。admission 必须在实际
PCIe/NVLink/network topology 下测量 communication 与 synchronization；本地 compute
microbenchmark 不能资格化 NCCL。

## 故障排查

| 现象 | 检查项 | 必须采取的动作 |
| --- | --- | --- |
| Probe 报 unavailable | active backend、精确 provider ID、shared-library file、transitive library | 修复 discovery；不能凭假设开启 auto selection |
| Windows DLL 存在但不能加载 | 当前 process `PATH`、`os.add_dll_directory()`、architecture、dependent DLL | load 前完成配置，只绑定一个 CUDA-major family |
| Linux `.so` 存在但不能加载 | `LD_LIBRARY_PATH`/RPATH、SONAME、dependent `.so` | package 没有 symlink 时使用 versioned SONAME |
| Provider 可加载但执行失败 | dtype、shape、layout、device、stream、provider ABI/version | 暴露 provider failure；显式选择后不能静默 fallback |
| 正确性不同 | matrix property、pruning/precision、transpose/layout、stale plan 或 values | 性能评估前先让数值门禁失败 |
| 首次调用很慢 | plan creation、JIT、analysis、compression、allocation | 分离 setup 与 steady state，并使用生产复用次数 |
| 内存增长 | live plan/scene/factor、workspace、cache、in-flight Graph lease | 可用时检查 provider memory report，并在 retire 后关闭 owner |
| 性能不稳定 | synchronization、cold cache、clock/power state、algorithm search、topology | 使用 fresh-process AB/BA，并要求最差情况仍通过应用正收益门槛 |

若 native Windows adapter 在较深的 provider plan creation 中失败，还应检查 host
thread/executable stack reserve。增大 reserve 只能视为特定 provider version 的部署
workaround，不能写成 Forge runtime requirement。

## 部署验收清单

在生产环境启用 external provider 前，应为以下内容保留证据：

- 精确 Forge build/runtime identity 与 active backend；
- GPU UUID/architecture 与 driver version；
- provider package version、shared-library content identity 与 transitive dependency family；
- operation shape、dtype、layout、topology 与 reuse/update policy；
- numerical oracle 或 solver residual/convergence gate；
- 带显式 synchronization 的 setup 与 steady-state timing；
- worst-case 结果和 variability，而不仅是最好值或 median；
- provider-owned、workspace、compressed/factor 与 peak memory budget；
- resource close/reset 行为与 in-flight submission lifetime。

不要把本地 benchmark 直接变成全局自动 heuristic。automatic selection 必须建立
exact-scope、fail-closed admission 合同；否则应保持 provider 显式选择。

## 可选 Vulkan FFT 计划

当前源码提供 `ti.hardware.fft.VulkanFftPlan`，要求包含 FFT 原生桥接的 runtime，以及独立
`taichi_forge_vkfft_provider_abi1_vkfft134` DLL/SO。adapter 使用 VkFFT 1.3.4 和匹配的静态
glslang/SPIRV-Tools 构建；执行需要 Vulkan loader/driver，不需要 CUDA、Vulkan SDK 或共享
glslang runtime。当前标准 runtime 构建启用 `TI_BUILD_VKFFT_PROVIDER`，adapter 安装于
`taichi_forge_runtime/_lib/hardware_providers`，上游声明安装于 `_lib/licenses/vkfft`。旧产物可能没有
该 adapter，应探测实际安装结果，不把当前源码当作已发布 wheel。离线构建可按
`cmake/TaichiVkfftProvider.cmake` 提供 `TI_VKFFT_ROOT` 和匹配的静态库；未给源码路径时仅在构建期
获取固定版本 VkFFT，不在用户创建计划时下载源码或编译 C++。

```python
ti.init(arch=ti.vulkan)
data = ti.ndarray(ti.f32, shape=(2, 16, 8, 2))
# 执行前填充实部/虚部交错的标量数据。
with ti.hardware.fft.VulkanFftPlan(
    data, (16, 8), batch_count=2, direction="inverse",
    normalization="inverse",
) as plan:
    plan.run()  # 原地变换，复用 Forge 有序 compute queue。
    builder = ti.graph.GraphBuilder()
    builder.append_native(plan.record(data="signal"))
    graph = builder.compile()
    bindings = graph.bind({"signal": data})
    graph.run(bindings)
    memory = plan.memory_report()
    build_and_allocation_facts = plan.statistics()
```

显式使用时才从 runtime 包解析 adapter，不在普通 import/replay 中发现或全局启用 provider。
可用 `adapter_path` 或 `TI_VKFFT_LIBRARY_PATH` 覆盖；显式路径无效时直接失败，不静默回落。
旧的 `ti.hardware.fft.is_available()` /
`cache_statistics()` 仍只描述 cuFFT；`ti.hardware.probe("vkfft", library_path=...)` 检查 adapter
ABI，不创建计划、不证明设备或 workload 可执行。被动状态只统计已知的公开未关闭计划。

首版支持原地 compact C2C f32、rank 1--3 与显式 batch；尺寸只能包含 2/3/5/7/11/13 素因子。
更大素因子因上游错误清理路径尚待资格化而暂缓，不是性能淘汰。默认 `normalization="none"`
使正逆变换均不归一化；`"inverse"` 将逆变换除以 transform volume。存储、shape、方向和归一化
在计划中冻结，Graph 必须绑定原 ndarray，不支持运行时替换存储。

创建计划可能 JIT 并同步初始化查找表。重放执行保留的 secondary GPU 命令序列，但每个 FFT action
仍有 root-ordered host call；不能称为整个 Graph 的原生 capture，也不等同于 `ti.linalg.record_fft()`
的 CUDA 输入输出分离合同，不新增 CompileIQ 路由轴。关闭计划拒绝后续调用，已提交 command buffer
仍保留资源。请求分配统计不含用户存储和不透明驱动对象；close 和初始化请求分配峰值均不能证明
显存已退役或真实 device peak。这里不声明生产加速，也不保证所有驱动组合。

## 官方参考

- [cuDSS 文档](https://docs.nvidia.com/cuda/cudss/index.html)
- [cuSPARSELt 入门](https://docs.nvidia.com/cuda/cusparselt/getting_started.html)
- [cuTENSOR 文档](https://docs.nvidia.com/cuda/cutensor/index.html)
- [AmgX 源码与构建指南](https://github.com/NVIDIA/AMGX)
- [NCCL 安装指南](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html)
- [OptiX SDK 下载与 release 要求](https://developer.nvidia.com/designworks/optix/download)
- [CUDA compiler Advanced Controls](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html)
- [NVIDIA CompileIQ](https://developer.nvidia.com/cuda/compileiq)
