# 完整 Graph recipe：下游接入

[English](graph_recipe_integration.en.md)

本指南使用现有公开合同，不要求应用读取内部规划，也不代替应用正确性与生产测试。

## 安装边界

使用兼容的 `taichi-forge` CPython shim 和 `taichi-forge-runtime` Windows wheel；两者按 package
version/private ABI 配对，不按 Git HEAD 配对。当前本地开发交付组合可以是 headless：仿真不依赖 GGUI，
需要 `ti.ui` 窗口的测试则须选择启用 GGUI 的构建。具体产物由提供者说明，不从版本号推断所有可选能力。
构建方式见[wheel 指南](build_wheels.zh.md)，库安装见[外部硬件配置](external_hardware_providers.zh.md)。

完整 Graph 搜索仅使用 [Forge 维护的 CompileIQ fork](https://github.com/fancifulland2718/CompileIQ)。
从该 fork 提供的 Windows wheel 安装，而不是直接 `pip install compileiq` 安装基础版本：

```powershell
python -m pip install C:\artifacts\taichi_forge_runtime-0.6.3-py3-none-win_amd64.whl
python -m pip install C:\artifacts\taichi_forge-0.6.3-cp310-cp310-win_amd64.whl
python -m pip install C:\artifacts\compileiq-<compatible-fork-wheel>.whl
```

路径和 Python tag 是示意，须替换为实际文件。Fork 按 V2 protocol/capability 接受，commit/hash 是来源事实，
不是安装白名单。只运行已选 Graph 不需要启动 CompileIQ 搜索；历史测量适用性检查仍可能需要它。
可选 vendor runtime、DXC、NVCC 等只为用到的 provider 配置，不安装整套库作为前提。

在应用测试环境中明确安装这一组 shim、runtime 和 fork wheel，运行示例前确认实际 import 路径。
安装包测试不要继承开发用的 `PYTHONPATH`、`TAICHI_NATIVE_RUNTIME_DIR` 或 `TAICHI_RUNTIME_DIR`
覆盖。本地 headless 接入组合不是正式发布或渲染窗口资格；窗口测试需要启用 GGUI 的 runtime。

## 可运行示例与结果处理

[complete_recipe_provider.py](../../python/taichi_forge/examples/graph/complete_recipe_provider.py)
使用纯公开接口：应用提供一个精确整数算子的两遍 baseline 和一遍替代 Graph。无需修改中央 family 或 CompileIQ。

```powershell
python -m taichi_forge.examples.graph.complete_recipe_provider --output result --environment-id my-device-driver-runtime
python -m taichi_forge.examples.graph.complete_recipe_provider --output restored --restore result/selection.json --environment-id my-device-driver-runtime
```

`--evaluation-limit 2` 可演示部分搜索；随后用较大预算和 `--resume result/checkpoint.json` 继续。
示例目标是含完成等待的 wall time，不是 device 时间或应用加速证据。环境描述由调用者提供且须真实稳定。

- `selected`：保存 selection artifact，使用 `with definition.materialize(selection) as handle`，
  再通过 `handle.executor.bind(...)` 与 `handle.executor.run(...)` 执行。
- `resumable`：保留 report/checkpoint，以相同 provider、workload、evaluation、environment 和 target 恢复。
- `no_feasible_candidate` / `failed`：检查结构化失败；不要把空 selection 传给 materialize 后称作优化成功。
- baseline 始终可由 `definition.compile()` 明确选择；搜索结果不改变普通 auto。

缺任一 `GraphWorkloadContext`、`GraphEvaluationContract`、`GraphBackendEnvironment` 时，测量只属于当前 session。
完整报告与选择 artifact 不同；vendor operation 还可能需要单独保存 `preparation_artifact()`。
新进程重新建立等价 Graph 和 provider，再调用 `check_recipe_applicability`、`resolve_recipe`。
结构可恢复而历史测量不适用时，可以重新测量，不应伪称旧性能仍有效。无 Python executable/AOT 二进制反序列化。

## 外部 provider 的职责

| 方法/对象 | 必须表达的内容 |
| --- | --- |
| `GraphRecipeProviderDescriptor` | namespace、provider/domain version、semantic fingerprint、装配协议与所需能力 |
| `discover(definition)` | 只针对自己明确理解的语义生成 fragment；空集合不等于性能拒绝 |
| `resolve(definition, key)` | 按稳定 key 重建同一物理策略，不保存 callback 地址或随机身份 |
| `expand(definition, key)` | 已有 survivor 的真实邻居；无邻居返回空集合 |
| `materialize(scope, fragment)` | 冷创建资源/执行组件，及时用 `scope.own(..., release=...)` 登记失败回滚 |
| `assemble(...)` | provider-owned whole Graph 返回执行器及实际物理清单；不是换名字重复同一执行 |
| `describe(...)` | JSON-safe 的解释与限制，不冒充实测数据 |

示例使用 `PROVIDER_OWNED_WHOLE_GRAPH_V1`，必须完整覆盖定义；它不把任意 Python callback 嵌进普通 Graph。
已有 Forge region provider 使用 `RUNTIME_GRAPH_ASSEMBLY_V1` 贡献装配片段，沿用各 owner 的冷物化实现；
不要为了接一个应用 family 修改私有 source/环境变量表。需要独立实现时优先使用示例的完整 Graph 协议。

`CompiledGraphPhysicalManifest.from_graph(definition, recipe, graph)` 复用 Forge 的冷观测，
不要求下游手写 kernel/command manifest；它证明实际执行组织，不证明数学等价。
外部 provider 仍须声明真实 coverage、资源/绑定/数值条件。物理改变时更新 domain/implementation 身份。
示例没有自有 workspace；有 workspace 时应声明 requested ownership/lifetime，并实现相应释放，不能填成零。

## 执行身份与内存观测

provider 在物理观测前应调用 `scope.own_executor(graph)` 登记执行器，因为观测本身也可能失败。
最后一个 handle 关闭时显式退役其 Forge Graph，即使其他 Python 变量仍引用执行器；`Graph.close()` 可重复调用，
不销毁调用者输入。runtime reset 会关闭已有 materialization context。reset 后重新构造 definition，
不要继续使用旧 runtime 执行器；自定义执行器类型仍需提供自己的 release 回调。

物理 manifest schema v2 将执行/分配方案和显存观测分开。`materialized_physical_id` 对编译工作、绑定和
`resource_plan`（请求大小、分组、生命周期）求身份，不包含冷/热缓存分配或 backing page 大小；
`resources`、`memory` 保留观测。旧 v1 physical ID 与 v2 不等价，需重新解析结构选择，并按需更新测量证据。

`handle.resource_instance_id` 表示具体资源所有权实例，不是跨进程选择键。物理 ID 相同不授权共享可变执行器。
只有 provider 明确保证共享状态安全并返回 `GraphMaterializationProduct(..., shareable_executor=True)`，
同一 context 才跨 recipe 共享实例。同一 recipe 在同一 context 的显式重复请求仍使用已有实例。

## evaluator 不应制造错误结论

每次评估先建立等价输入状态、发布 binding、预热，再测量。输入恢复、正确性 readback 和库 probe 不应混入
稳态 submit 时间；若应用确实每步需要更新，应将该更新放进所比较的完整流程，两路保持相同口径。
有反馈输出的 matmul、原地 sort、破坏输入的 C2R 必须显式处理状态；Forge 不自动复制全部输入。

通过已有 `metric_definitions` 声明 device event 区间、kernel-active、host submit、完成等待各自口径。
事件区间可含空隙；并行 kernel 活跃时间求和不等于总耗时。setup/first/steady 可用既有 cost_profiles 分账。
已知 caller/workspace 请求、pool reservation 与未知 vendor/driver 驻留不同；缺测不能写零。
Nsight/NVML 仅显式采样，性能计时与诊断分开，不加每 replay 校验/探测/同步。

目前 cuSOLVERDn、AmgX、Parallel Sort 的具体执行与 Graph 边界以
[外部硬件指南](external_hardware_providers.zh.md) 为准；有执行 API 不等于有完整 recipe generator。
