# Graph 兼容性与迁移指南

> 基础 Graph modernization 与 native replay 模型首次公开于 `0.4.1`；Dense Field
> Graph、replay 生命周期加固、诊断和更严格的并发/AD 合同属于 `0.5.0`。
> `0.6.0` 增加结构化控制、异步 ticket/pacing、runtime dense-storage binding、
> recordable operator action 与 Vulkan device-written indirect dispatch。
> 版本归属见[版本更新说明](release_notes.zh.md)。

本文说明 Forge graph 相比 vanilla Taichi 1.7.4 的公开行为与兼容边界。后端架构、性能、
显存策略与诊断统一维护在 [Graph Runtime 与优化](graph_runtime_optimization.zh.md)。
使用 dense Field 的用户还应阅读 [Dense Field Graph](dense_field_graph.zh.md)中的
静态 binding 合同。

`0.6.3` 新接入请阅读[完整 recipe 与外部 provider 指南](graph_recipe_integration.zh.md)：
`freeze/search/resolve` 是显式离线流程，不改变原 `compile/run` 或 runtime `auto`。
部分搜索结果、跨进程合同和物化 handle 与执行器的区别均在该指南说明。

Forge-only graph 与 native replay API 的精确签名见
[Forge API 参考](forge_api_reference.zh.md)。

## 公开兼容性

Forge 保留用户熟悉的 graph builder 表面：

- `ti.graph.GraphBuilder`
- `GraphBuilder.dispatch(kernel, *args, template_args=None, label=None)`
- `GraphBuilder.create_sequential()`
- `GraphBuilder.append(sequential)`
- `GraphBuilder.compile()`
- `Graph.run(args)`
- `ti.graph.Arg(...)`

普通 kernel graph 继续使用公开 CGraph 模型。只包含普通 kernel dispatch 的 AOT graph
序列化仍兼容既有公开模型。

## Forge 增加的能力

Forge 保留 graph-builder 模型，并增加显式 Forge extension 与 backend-owned execution
planning，用于结构化控制、异步提交、device-driven dispatch 和 DSL-defined native replay。

用户可见的新增能力包括：

- `GraphBuilder.dispatch()` 与 `Sequential.dispatch()` 可通过 keyword-only
  `template_args=` 正式绑定 data-oriented `self`、Field 等构图期参数；
- 可选的逐 dispatch `label=` 与 thread-local `ti.profiler.dispatch_label()` scope 将稳定
  task identity 关联到 profiler/NVTX event；
- 只读 `kernel.task_manifest()` 与单 segment JIT `Graph.task_manifest()` report 可观测
  requested/selected/actual geometry，但不增加 launch control；
- scalar、matrix、ndarray、texture 与 RW texture 的 runtime 参数路径；
- 兼容 dense Field、`DenseNdarrayView` 与受管 external storage 的 runtime-storage binding；
- `GraphBuilder.while_loop()`、`if_then_else()`、`switch()` 与最大 depth=2 的
  structured `Sequential`；
- `GraphBuilder.dispatch_indirect()` / `Sequential.dispatch_indirect()` 的 Vulkan
  device-written 单 task 路径；CPU/CUDA 当前明确失败；
- `Graph.submit()`、`SubmissionTicket`、opt-in region telemetry 与共享
  `SubmissionPacer`；
- 优化 replay 路径不支持时稳定回退到 ordinary dispatch；
- 公开 `Graph.execution_stats()` execution/fallback/resource report；
- `Graph.control_flow_stats()` 与 `Graph.run(trace=True)` 的停止位置/分支报告；
- 可显式 dispatch `kernel.grad`，在自动 AD context 外手工管理 gradient Graph；
- algorithm 层生成的 Forge-defined primitive sequence 可内部 native replay；
- `GraphBuilder.append_native(node, prewarm=False)` 可追加
  `PrimitiveSequence`、`DeviceCheckResult`、`DeviceMetricResult` 等
  Forge DSL-defined node；
- `LinearOperator.graph_action()` 等合格 recordable provider action 可进入 Graph root 或
  structured body，并与相邻 ordinary CGraph 合并为 same-backend region。

## 定义与生命周期合同

- `GraphBuilder.compile()` 冻结当前 dispatch 与 sequential 定义；之后修改 builder
  不会改变已经编译的 graph 或其 lazy AOT 结果；
- `Graph.run(args)` 的参数 key 必须与声明完全一致，missing/unexpected key 会在提交前
  抛出 `TaichiRuntimeError`；
- 同一个 `Graph` 的一次调用是完整 host invocation；不同 graph 仍可独立提交，该保护
  不等待 GPU 完成，也不增加默认 `ti.sync()`；
- `ti.reset()` 会使旧 runtime 的 graph 失效，reset 后应重建 builder 与 graph；
- 闭包或 `template_args` dense Field 是 definition-time binding；内容可变，但替换
  identity/layout 或销毁其带 generation 的 SNodeTree 后必须重建 Graph。需要在 run 之间
  替换兼容 dense Field 时，使用 `ArgKind.NDARRAY` runtime slot；复用相同数值 tree id
  不会恢复旧静态 binding；
- 结构相同的 runtime resource 可使用后端 replay；结构变化可触发 recapture 或 fallback，
  两条路径都保持 binding 与执行语义；
- runtime graph 安全不能替代应用对共享仿真/渲染数据使用 snapshot、slot 或
  producer-consumer 协议。
- 新引擎代码应使用 `template_args=` 绑定 `self`、Field 等 `ti.template()` 参数；Field
  仍不进入每次 run 的字典。旧适配器直接写入 durable AOT plan 时，Forge 继续恢复真实
  runtime 参数名，missing/unexpected key 的严格检查不变。
- `Graph.run()` 是 primal-only；active `ti.ad.Tape()` 与 `ti.ad.FwdMode()` 会在提交前
  被拒绝，而不是静默漏记 gradient/dual。显式 `kernel.grad` Graph 可在这些 context 外
  手工运行。

CUDA resource lease 与 dynamic patch、Vulkan identity 与延迟退役、固定 replay 容量、失败
恢复和 `Graph.execution_stats()` 的实现策略见
[Graph Runtime 与优化](graph_runtime_optimization.zh.md)。

## 严格 runtime key 迁移

definition-time template binding 必须从 runtime 字典移除。旧的 permissive adapter 可能
曾容忍重复传入，但 Forge 现在会拒绝这个 extra key：

```python
builder.dispatch(
    solver.step,
    template_args={"self": solver, "state": solver.state},
)
graph = builder.compile()

graph.run({"state": solver.state})  # 错误：unexpected runtime argument
graph.run({})                       # 正确：该 Graph 没有 runtime argument
```

`Graph.run()` 中只保留已声明的 `ti.graph.Arg` 名称。这会在 backend submission 之前发现
过期 engine adapter 和参数拼写错误。

## 异构多环境 layout

不要为每个同构环境创建一张 Graph。应按稳定的 solver-layout-shape-feature signature
建立 block/Graph，并把同构环境放到 Field 前导维度。不同异构 block 持有独立可写 Field
并独立提交。simulation/render overlap 仍须使用 slot/epoch-owned snapshot Field；
Graph 不增加 data-hazard tracking。

## Native graph 边界

Native graph 支持有意保持窄边界：

- 支持：Forge 自有 DSL/native algorithm 层产生的 native node；
- 不支持：任意用户 native callback 直接放入 graph；
- 不支持：包含 Forge native node 的 graph 做 AOT 序列化；
  `ti.aot.Module.add_graph()` 只接受普通 kernel CGraph；
- same-backend ordinary CGraph 与 recordable provider 可以融合；所有 node 必须匹配 active
  runtime/backend，不支持借此跨设备执行；
- 数值检查 result node 只 replay device-side native 工作，读取结果仍须显式调用
  `to_int()`、`to_bool()`、`ok()` 或 `to_float()`。

这使资源所有权与后端生命周期边界保持明确。

## 适用工作负载

Graph 最适合 dispatch 拓扑稳定且会重复 replay 多次的场景：

- fixed-shape 仿真 substep；
- 重复 native primitive chain；
- 资源结构稳定的 rendering/staging chain；
- 需要命名 graph 入口的 AOT 部署。

若 Python 每帧改变 dispatch 拓扑、shape 或资源结构频繁变化，或一个大 kernel 已占主要
launch 开销，graph 收益通常较低。

## 与 vanilla 的区别

Vanilla Taichi 1.7.4 主要记录普通 kernel dispatch。Forge 保留该模型，并在同一公开入口
下增加后端 execution planning 与 native primitive replay。不支持的优化路径会回退，而
不是改变 graph 语义。
