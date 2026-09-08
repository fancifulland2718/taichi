# Native 算法

本文说明 Forge 公开算法入口中可能走 CPU、CUDA 或 Vulkan native 实现的部分。对于
`method="auto"`，只有当 dtype、shape、layout 和后端能力都满足已知合同时才选择 native
路径；不支持的组合必须回退到正确的通用路径。显式指定 native method 时，不支持应清晰拒绝。

按模块整理的 Forge-only API 符号清单见 [Forge API 参考](forge_api_reference.zh.md)。

核心 native 算法族首次发布于 Forge 0.4.0；Graph native replay 与 device-side 检查分别
在 0.4.1 和 0.4.23 进入公开版本。本文说明已正式发布的 0.6.2 可移植性与安全合同；各项
能力的首次公开版本见[发行说明](release_notes.zh.md)。

## 公开入口

| 入口 | 用途 |
| --- | --- |
| `ti.algorithms.sort(keys, values=None, ...)` | Forge 稳定排序调度器。 |
| `ti.algorithms.sort_by_key(keys, values, ...)` | 排序 keys 并同步移动 payload values。 |
| `ti.algorithms.parallel_sort(keys, values=None)` | vanilla 兼容的 legacy sorter。 |
| `ti.algorithms.PrefixSumExecutor(n).run(values)` | Prefix sum / scan。 |
| `ti.algorithms.device_prefix(values, extent, ...)` | 通过 device-resident 有效数量组合固定容量 primitive 输入。 |
| `ti.algorithms.DevicePrefixSequence(capacity)` | 把 fixed-topology 有效前缀 pipeline 记录成一个逻辑 Graph native node。 |
| `ti.algorithms.DevicePrefixWorkspace(max_items)` | 在有效前缀 pipeline 间复用 staging 与 child primitive workspace。 |
| `ti.algorithms.DeviceWorklist(capacity, dtype, telemetry=..., transition_mode=..., unique_key_capacity=...)` | 持有可复用 front/back storage，并选择 staged、direct 或 dense-key unique 增量 transition。 |
| `ti.algorithms.device_worklist_append(...)` | 从 Taichi scope atomic append，不读取 host count。 |
| `ti.algorithms.device_worklist_append_direct(...)` | atomic append 并直接发布 bounded extent。 |
| `ti.algorithms.device_worklist_append_unique_direct(...)` | 在本次 transition 内按 dense integer key 唯一 append，无需清空 tag table。 |
| `ti.algorithms.DeviceWorklistSequence(args)` | 把一次 worklist transition 记录为 Graph native action。 |
| `ti.algorithms.DeterministicScatterReducePlan(indices, num_groups)` | 复用固定 scatter topology，按稳定 source 顺序做浮点 reduction。 |
| `ti.algorithms.experimental_compact(values, flags, output, count, ...)` | 按 flags 过滤并写入紧凑输出。 |
| `ti.algorithms.experimental_run_length_encode(keys, unique_keys, run_lengths, run_count, ...)` | 完全在 device 上编码连续整数 key run。 |
| `ti.algorithms.experimental_unique(values, output, count, ...)` | 选择每个连续相等 run 的首项。 |
| `ti.algorithms.experimental_unique_by_key(keys, values, unique_keys, unique_values, count, ...)` | 选择每个连续 key run 的第一个 payload。 |
| `ti.algorithms.experimental_segmented_reduce(values, layout, output, ...)` | 无 host round-trip 地 reduce 每个可复用 dense segment。 |
| `ti.algorithms.experimental_segmented_scan(values, layout, output, ...)` | 在每个可复用 dense segment 内做 inclusive/exclusive scan。 |
| `ti.algorithms.experimental_reduce(values, output, op="sum", ...)` | 将 values reduce 到 `output[0]`。 |
| `ti.algorithms.experimental_histogram(values, bins, ...)` | 将整数 values 统计到 bins。 |
| `ti.algorithms.experimental_transform(src, dst, scale=..., bias=..., ...)` | 元素级 affine transform 和 copy。 |
| `ti.algorithms.experimental_gather(src, indices, dst, ...)` | Indexed read。 |
| `ti.algorithms.experimental_scatter(src, indices, dst, ...)` | Indexed write。 |
| `ti.algorithms.experimental_scatter_add(src, indices, dst, ...)` | Indexed add；支持时走后端 atomics 或 staged reduction。 |
| `ti.algorithms.experimental_bucket_builder(keys, values, offsets, output, ...)` | 构建 grouped/bucketed value output。 |
| `ti.algorithms.experimental_grouped_reduce(keys, values, output, ...)` | 按整数 group key reduce values。 |
| `ti.algorithms.count_if(flags, ...)` / `any_if()` / `all_if()` | 在 kernel 外部发起 device-side 数值谓词检查。 |
| `ti.algorithms.nan_count(values, ...)` / `inf_count()` / `all_finite()` | 在 device 上统计 NaN/Inf/非有限值。 |
| `ti.algorithms.index_bounds_check(indices, lower=..., upper=...)` | 在 device 上统计越界 index。 |
| `ti.algorithms.max_abs(values, ...)` / `max_abs_delta(values, reference, ...)` | 在 device 上计算收敛/误差类最大绝对值指标。 |

名称里的 `experimental_` 表示这是 Forge 公开入口，但演进节奏会比长期 vanilla API 更保守。

## 后端选择

大多数 API 支持 `method="auto"`。auto 只在输入合同明确支持时选择 native 后端，否则保持正确
性并走通用或 host fallback。

常见显式 method 家族包括：

- `cpu_native`
- CUDA driver-only native method
- 仅在 reference-enabled 开发构建中存在的、已弃用 CUDA CUB reference method
- 可用时的 Vulkan native method
- sort 类操作的 `host_stable` 或 legacy fallback method

显式 native method 适合测试或受控部署，不应被当成跨所有后端的可移植承诺。

## CompileIQ 边界

`0.6.3` 不公开算法级 CompileIQ 搜索入口。reduce provider 与 segmented-scan 的历史离线搜索器
保留为私有资格化工具，用于审计既有正负证据；它们不是算法公共合同，也不会成为新的 provider
路由层。应用应继续使用上面的普通 primitive、显式 `method=` 或现有 `method="auto"`。

公开的 CompileIQ 入口归 Graph 所有：`ti.graph.compileiq_recipe_search(graph)` 只搜索 Forge 已证明
合法且可精确物化的完整 Graph execution recipe。CompileIQ 不接收 provider、block size、workgroup
shape、PTXAS flag 或 segment offset 等裸轴。搜索和资格化保持离线，不修改 primitive 默认值；
compile/search build 时间只作诊断，不是准入门禁。

## 机器可读 capability 合同

从 Forge 0.6.0 起，每个当前 primitive family 都公开不可变的 schema-v1 描述：

```python
contract = ti.algorithms.primitive_capability("experimental_reduce")
for operand in contract.operands:
    print(operand.name, operand.dtypes, operand.ranks, operand.layouts)

ti.init(arch=ti.vulkan)
active = ti.algorithms.resolve_primitive_capability("reduce")
for method in active.methods:
    print(method.method, method.program_available)
```

`primitive_capability(name)` 与 `primitive_capabilities()` 是静态查询，可在
`ti.init()` 前调用。它们按 operand role 报告 dtype、rank、layout、storage，
并描述 backend method、稳定性、确定性、atomic 顺序依赖、primal/forward/reverse/
explicit-adjoint、Graph、AOT、workspace 与 fallback 合同。也可以用
`experimental_reduce` 等公开入口名作为 family alias。

`resolve_primitive_capability(name)` 要求已有 active Program。它将 method 过滤到
当前 CPU/CUDA/Vulkan 后端，并执行调度层使用的同一组无副作用 provider probe。
`program_available=True` 只表示当前 Program 含有该 provider，不表示任意 dtype/
layout 请求一定有效；method 仍标记 `input_dependent=True`，具体公开操作会在写入前
完成最终请求校验。

catalog 同时是公开 `method=` 校验与 native AD policy 的单一来源，避免文档中的
method/adjoint 能力与真实调度逐渐漂移。

### 自动微分

- 在 `ti.ad.Tape()` 中，只有具体输入同时具备完整注册 backward 时，
  `method="auto"` 才选择 native primal；否则走声明的 kernel fallback。没有
  backward 的显式 native method 会在写入前拒绝。
- 在 `ti.ad.FwdMode()` 中，transform、reduce-sum、gather、scatter 和 scatter-add
  走可微 helper-kernel fallback；其 JVP 已在 CPU、CUDA、Vulkan 上做回归。
  由于还没有 native forward launcher，显式 native method 会拒绝。
- scan、grouped-reduce、segmented scan 与 segmented reduce 当前会在写入前拒绝
  `FwdMode`。segmented-reduce reverse AD 只覆盖 grouped ndarray sum；
  serial/dense-field mode 会拒绝。
- sort、compact、RLE/Unique、histogram、bucket-builder、device check 与 device
  metric 被明确标为不可微，在写入前拒绝 automatic AD context。应在
  Tape/FwdMode 外完成这类预处理或诊断。

这些规则只说明 automatic AD。native Graph node 仍是 primal-only，native-node AOT
serialization 仍不支持。

### CUDA runtime 可移植性

当前源码中的标准 CUDA primitive provider 使用动态加载的 CUDA Driver API 和 Forge 自有
kernel。标准 runtime wheel 不依赖 CUB/CUDART，用户不需要本机 CUDA Toolkit；项目按操作系统
各发布一个 runtime wheel，不按 CUDA 版本分叉。`method="auto"` 永远不会选择 Toolkit
reference provider。

显式 `cuda_cub` / `cuda_cub_*` method 已成为弃用的开发参考路径：只由独立 reference
workflow 编译，调用时发出 warning，标准 runtime wheel 不包含它们。已经发布、仍采用 0.5.0
包内 CUDART 布局的 runtime wheel 会继续被 loader 兼容，但不能把它当成新 driver-only
dependency class 的发行证据。

driver-only 消除了 CUDA Runtime 动态库依赖，但本身不能证明最低 NVIDIA driver 已降低。
PTX 是否可加载以及任何 driver 下限声明，仍必须在目标旧 driver 上真实执行。当前构建边界见
[构建 Wheel](build_wheels.zh.md)，Linux 和旧 driver 的待补证据见
[Linux 复测状态](linux_revalidation.zh.md)。

### CUDA 0.6.0 历史性能快照与当前边界

下表是 0.6.0 资格快照，不代表之后每个 `master` 优化的测量结果。数据来自 Windows
开发机（RTX 5090、driver 610.62、Python 3.10.11）上的 1,048,576 个 i32 item：每项
30 个 sample，每个 sample 批量提交 20 次后同步并折算单次 median；测量前 idle guard
确认没有其它 Python 或 GPU compute process。CUB 只来自不发布的 CUDA 13.2 reference
build，正确性另由 NumPy oracle 验证。

| Primitive | driver-only median | CUB reference median | 相对吞吐 | 资格参考线 | driver workspace |
| --- | ---: | ---: | ---: | ---: | ---: |
| scan | 0.0272 ms | 0.0190 ms | 69.8% | 90% | 4 KiB |
| reduce-sum | 0.0228 ms | 0.0193 ms | 84.6% | 90% | 4 KiB |
| histogram-256 | 0.1243 ms | 0.1215 ms | 97.7% | 90% | 0 |
| stable compact | 0.0279 ms | 0.0228 ms | 81.8% | 80% | 4.00 MiB |
| stable i32 key/value sort | 0.4883 ms | 0.1491 ms | 30.5% | 80% | 28.06 MiB |

按表中的资格参考线，histogram 和 compact 当时达到参考值，scan、reduce 和 sort 没有。
标准 wheel 仍选择正确、异步且 driver-only 的 Forge provider，因为 CUB 不属于发行依赖，
host round-trip 也不适合作为 GPU 热路径默认值。这不是与 CUB 等速的声明，也不是跨设备、
跨驱动性能保证。

成对的 0.6.1 release-candidate wheel 保持相同的 1024-item tiled scan、fused tiled
compact rank 和稳定分层
4-bit LSD radix 合同，但在 radix histogram 顶层已经能由一个 scan tile 完成时立即终止
hierarchy。在表中 1,048,576-item 的规模上，32-bit sort 的 histogram-scan launch 会从
16 降为 8，histogram uniform-add launch 从 8 降为 0，workspace 不增长。上表 timing
不能重新标成 0.6.1 结果。另一次 wheel-to-wheel 测试在同一 RTX 5090/610.62 系统上对比
公开 0.6.0 wheel（`dbc683028`）与成对的 0.6.1 release-candidate wheel：每只 wheel 分别启动
三个新进程，每个进程 10 次 warmup、100 次逐次同步 native sort。process median 再取
median 后分别为 0.51245 ms 和 0.36455 ms，延迟降低 28.9%；报告的 peak workspace 从
29,426,176 B 降为 29,425,664 B。安装 wheel 后的 13 个 CUDA dtype/payload 与大 hierarchy
稳定性用例全部通过。该配对测试的同步方法与历史表不同，因此用于补充而不是改写上表快照。

## 数据合同

- Dense 1D `ti.ndarray` 是主要 native 算法 ABI。
- Dense field/SNode 只有在能证明兼容 dense layout，或能提供安全 staging 路径时才走支持路径。
- `StructNdarray` 可作为 order/copy 类 primitive 的 opaque payload。部分数值 primitive 支持 scalar 或 packed tensor member view；完整语义见 [StructNdarray primitive 语义](struct_ndarray_api.zh.md)。
- 稀疏、非连续、复杂 SNode 拓扑不能默认假设走 native。
- 普通 `experimental_scatter()` 要求所有有效 destination index 唯一。
  CPU native scatter 会在写入前验证，并拒绝 duplicate；需要 duplicate target 时应使用
  `experimental_scatter_add()`。
- duplicate target 的 floating scatter-add 可能受后端 atomic 顺序影响；只有数值合同允许时才应使用。

## 设备端有效前缀 pipeline

`DevicePrefix` 允许固定容量 primitive 共享一个 device 写入的 `DeviceExtent`，而不在 host
观察 count：

```python
workspace = ti.algorithms.DevicePrefixWorkspace(capacity)
prefix = ti.algorithms.device_prefix(values, extent, workspace=workspace)
active = prefix.compact(flags, compacted, compacted_extent)
active.sort()
active.scan(scanned)
```

wrapper 当前可组合 stable compact、scan、reduce、stable sort、consecutive unique/RLE、
grouped sum 与 bucket building。支持一维 scalar `i32/u32/i64/u64/f32/f64` storage，
同时受各 primitive 更窄合同约束。结果 storage 保持固定物理容量，只有 paired extent 以下的
prefix 有定义；准备 provider 时可以用 neutral value 或 sentinel 覆盖 inactive suffix。

它是组合层，不是第二套算法 provider。它复用普通 `method="auto"` 选择与
`DevicePrefixWorkspace`，所以操作之间不需要 count readback、按 count 分配或重建 Graph。
底层固定容量 provider 仍可能处理 capacity-sized scratch。workspace 可串行复用，但不可用于
并发 submission。

Graph 执行可用 `DevicePrefixSequence` 在 symbolic ndarray 参数上记录同一组 prefix 操作，
再通过 `GraphBuilder.append_native()` 追加，从而获得一个用户 ticket 且不观察 host count。
provider、workspace topology 与 operation routing 在 materialization 时固定，replay 不再重复
operation-kind 分支。当前核心 Prefix provider 尚未提供可并入外层 backend Graph 的 command
recipe，因此它仍是 segmented native 诊断路线：`admission="auto"` 会拒绝，显式 admission
则如实报告 loose helper 与 queue topology，而不会把逻辑 node 伪装成 backend-recorded。
这一合同不表示 provider kernel fusion。compact 结果若接入 Vulkan bounded dispatch，可创建
`output_extent.dispatch_state(block_dim)` 并同时传给 compact 与 `dispatch_bounded()`；compact
scatter 会把 indirect packet 与 count 一起发布，删除一次 preparation dispatch。CPU/CUDA
不消费该 packet；CUDA 独立使用 exact logical range，并可选择 12.4+ adaptive physical
control。

在当前 Windows 资格机器上，10% active prefix 的 compact-to-scan chain 相对在两个操作间
显式调用 `DeviceExtent.snapshot()` 的同一 chain，CPU、CUDA、Vulkan 分别快 1.05x、
1.32x、1.90x。这是消除同步的测量结果，不是跨设备吞吐保证。带执行末端同步的成对基准为
`benchmarks/dynamic_workload_bench.py`。

## Device-resident worklist

`DeviceWorklist` 在有效前缀 primitive 之上增加生命周期和可选计数合同。它持有两份固定容量
scalar ndarray、两个 `DeviceExtent`、可复用 primitive workspace，以及完整 counter 或精简的
mandatory state。自定义 producer 可向 back storage append，并且不在 host 观察 count：

```python
worklist = ti.algorithms.DeviceWorklist(capacity, ti.i32)

@ti.kernel
def produce(values: ti.types.ndarray(dtype=ti.i32, ndim=1),
            extent_state: ti.types.ndarray(dtype=ti.i32, ndim=1),
            generated: ti.types.ndarray(dtype=ti.i32, ndim=0),
            overflow: ti.types.ndarray(dtype=ti.i32, ndim=0),
            limit: ti.i32, requested: ti.i32):
    for i in range(requested):
        ti.algorithms.device_worklist_append(
            values, extent_state, generated, overflow, limit, i
        )

worklist.prepare_next()
produce(*worklist.append_arguments(), requested)
worklist.commit_next()
```

`telemetry=False` 会完全省略 accepted/rejected/conflict/winner 数组、绑定与 device write。
staged transition 保留 generated/overflow/generation，共 12 bytes；
`transition_mode="direct"` 进一步只保留 overflow/generation（8 bytes），并用
`device_worklist_append_direct()` 在 slot reservation 时直接发布 extent。direct 模式仅适用于
atomic append，要求 `telemetry=False`，之后不得再调用 `finalize_next()`。

对于有界 dense key domain，可为 direct worklist 设置 `unique_key_capacity`。这会增加每个 key
一个持久 i32 generation tag 与一个 scalar epoch。producer 在遍历当前 active prefix 时调用
`device_worklist_append_unique_direct()`；tag 会为每个输出 key 选出唯一 append，且 transition
之间无需清空整张表，因此 producer 可把 cell 及重叠 halo neighbor 直接写入下一 frontier。
非法 key、capacity overflow 与 epoch 耗尽都会设置 sticky overflow，并让后续 stage fail closed。
该 primitive 本身不会扫描 dense domain；只有应用同时删除 full-domain copy/select，并在 producer
或 consumer 内处理 retired/default state，端到端工作量才真正随 active domain 缩放。

若 workload 已有全局有序的 boundary/record kernel，该 kernel 可用
`worklist.recycle_arguments()` 调用 `device_worklist_recycle_direct()`，随后 host 调用
`commit_recycled_next()`。这样会回收已消费 front、推进 generation，并删除下一层独立 prepare
dispatch。它必须在前一 producer 与旧 front read 完成后恰好执行一次，不能替代 producer 内的
跨 block barrier。
unique worklist 必须改用 `unique_recycle_arguments()` 调用
`device_worklist_recycle_unique_direct()`；该边界同时推进 generation 与 unique epoch。普通 recycle
ABI 会被拒绝，以免旧 tag 错误抑制有效 append。generation 或 epoch 耗尽时会设置 sticky overflow，
不会发生环绕复用。

无 overflow 时，每个 item 只执行一次 atomic slot reservation。atomic append 顺序不保证；
一次 transition 只有一个 producer owner，多个独立 Graph submission 写同一 worklist 前必须
显式排序。overflow 会把发布 count 钳制到 capacity，并同时保留在 `DeviceExtent` 与 worklist
counter 中；伪造或误绑的 Graph capacity 会在写 value 前 fail closed。`select(flags)` 保持
source order。`resolve_conflicts(keys, priorities=..., policy="min_priority",
strategy="auto", key_capacity=..., output_shape="compact_winner_list")` 将 conflict algorithm
与 sort provider 分开。有界紧凑
integer domain 可使用确定性的 `dense_atomic` arbitration；其他情况由 `radix_grouped` 使用
backend native stable-sort provider。两条路线都按 priority、ordinal、source index 处理 tie。
dense 路线把越界 key 记为 rejected + overflow；radix winner reduction 扫描每个 sorted key
run，由一个或少数超长 run 主导的分布并行度更低，应单独做性能资格。可用
`benchmarks/device_worklist_conflict_bench.py` 做同输入配对资格；脚本会验证 parity，并报告
raw sample/CV 与 workspace accounting。
若 consumer 只需要逐 key ownership，可在 `dense_atomic`、`telemetry=False` 下请求
`output_shape="dense_winner_table"`。结果是长度为 `key_capacity` 的 source-index table，空 key
为 `0x7fffffff`；不会生成 compact extent、winner list、scan 或 compact materialization。

fixed-domain producer 可用 dense key 与 active-flag 数组调用
`resolve_conflicts_from_mask()`。该路径保留原始 source index，并删除 stable candidate
compact 与 attribute gather。无需显式 ordinal 时，source index 作为确定性 tie break，ordinal
pass 和 buffer 也会省略。这是一条 telemetry-free 的显式 winner-table 路线；若工作负载专用
的 fused direct-claim kernel 具有更短的物理计划，用户应继续保留后者。

未提供 custom ordinal 的 priority/claim arbitration 在 CPU/CUDA 上会把 signed priority 顺序与
source-index tie break 打包为一次 u64 atomic-min pass；Vulkan 保持 portable 32-bit multi-pass。
返回值的 `arbitration_route` 暴露实际选择。packed route 少一个 dispatch，但每个 key 增加 4 bytes
workspace，因此仍应与 workload-specific fused claim 做配对基准。

Graph replay 使用 `worklist.graph_args(name)` 创建 symbolic 参数。staged producer 两侧可追加
独立的 `DeviceWorklistSequence(args).prepare_next()` / `.finalize_next()` node；direct transition
通常只记录 `prepare_next()` 与 direct producer。host-driven level loop 也可使用上述有序 boundary
kernel recycle 合同；该底层路径不是 `DeviceWorklistSequence` action。sequence 还可记录
`select()`、compact-list conflict
resolve 或 `resolve_conflict_winner_table()`。transition helper 可被 backend record；没有 integrated
action 的 provider pipeline 仍是显式 segmented 路线。Graph staging 在 submission 前分配，steady-state
replay 不分配、也不读取 host count；首次执行仍可能编译 kernel 并准备 native provider
workspace。strategy、provider 与 workspace topology 在 materialization 时固定。
增量 multi-stage Graph 可使用 `args.transition_arguments(step)`，并在每个 bounded producer 前记录
`DeviceWorklistSequence(args).prepare(target="next"|"current")`。连续 step 在两份稳定 buffer
之间交替；共享 epoch table 对每次 transition 独立去重，overflow 会传播到下一 stage。同步完成后，
或等待异步 ticket 后，调用 `commit_direct_transitions(steps)` 只更新 Python-side front ownership。
一个 worklist 仍是一条 completion-ordered workspace lane；并发 submission 需要独立 worklist。
若 worklist 永久由一个 Graph 持有，可改用 `worklist.fixed_graph_args(builder, name)`。它把
front/back、extent、counter 与 unique tag 作为 provider-private binding 固定到 Graph，只把无关的
用户资源留在公开 replay ABI。该绑定受 runtime generation 与当前 front parity 约束：偶数次
transition 可直接复用；奇数次 pipeline 在下一次 replay 前需要使用为另一 parity 编译的 Graph；
错 parity 会在提交前拒绝。provider-fixed worklist 要求 `workspace_lanes=1`；异步复用遵循 runtime
提交顺序，并在后端暴露 pending work 时使用 completion fence。真实并发仍需独立 worklist/Graph。
该所有权形式不会省略提交期资源保活，也不承诺自动加速。
Vulkan 会让相邻的 recordable prepare 与 bounded consumer 共享一个 backend Graph region：
prepare 在同一次 dispatch 中 reset target 并发布 source indirect packet，因此该 pair 只有两个
physical dispatch，也没有 loose packet helper。
完整 telemetry 下，`args.observe()` 把 counter 加到 completion-attached ticket observation；
completion 后由 `args.decode_observation()` 生成 `DeviceWorklistStatistics`。lean arguments 会拒绝
observation，而不会偷偷物化 telemetry。
`execution_report()` 是显式同步边界，可把这些 counter 与 `dispatch_bounded()` snapshot 合并。

Vulkan 上相邻的 recorded `finalize_next()` 与 `dispatch_bounded()` 现在会自动共享
Graph-owned indirect packet：finalizer 在一次 dispatch 中发布 count 与 grid，连续匹配 consumer
复用该 packet，无需公共 launch-state 对象或 preparation dispatch。中间插入其他 action 会
保守禁用该 specialization。把 `worklist.next_extent.dispatch_state(block_dim)` 同时传给两端，
仍是显式 packet publication 的兼容路线。CPU/CUDA 使用相同 source-level 组合但不消费该
packet；CPU 使用 exact scheduler chunk，CUDA 则在所有受支持 driver 上使用 exact logical
device range，并可在 12.4+ 上选择 device update 进一步缩小物理 grid。应查询
`ti.graph.dynamic_work_capabilities()["worklist"]`，不能从通用 API 反推
exact launch 行为。

## 固定拓扑确定性 scatter reduction

`DeterministicScatterReducePlan(indices, num_groups)` 是 immutable connectivity 下浮点 atomic
scatter-add 的显式可复现替代。构造时只读取并验证一次 host-visible integer topology，按 destination
稳定分组有效 source ordinal，再上传 permutation 与 segment layout。binding 使用一个 indexed kernel
按固定顺序读取变化的 contribution，并在 CPU、CUDA、Vulkan 上对每个 destination 从左到右
reduce。支持 scalar/vector ndarray 与 root-dense Field；负数和越界 index 会按统一合同忽略。

该路线不会被自动选择，也不会改变现有 atomic API。它以每个 destination 内的串行工作换取稳定
accumulation order，但融合实现不再物化 ordered-value buffer 或独立 gather stage。因此更适合作为
资格 baseline、可复现 fixed-topology assembly，以及每个 destination valence 不大的场景。
`binding.graph_action()` 只记录一个 dispatch；`report()` 公开 topology、component shape、
`fused_indexed_serial` route，以及为零的 ordered/workspace bytes。稳定顺序只承诺同一 backend/build
合同内可重复，不承诺跨 backend bit 一致，也不表示数值精度更高。atomic assembly 继续作为默认
性能路线，仅在需要可复现时显式选择 stable serial。

## Consecutive RLE 与 Unique

Forge 提供 device-resident consecutive-run primitive：

```python
workspace = ti.algorithms.RunLengthWorkspace(max_items=capacity)
ti.algorithms.experimental_run_length_encode(
    keys,
    unique_keys,
    run_lengths,
    run_count,
    size=active_count,
    workspace=workspace,
)
```

`experimental_unique()` 选择每个连续相等 run 的第一个 value；
`experimental_unique_by_key()` 同时选择每个 key run 的第一个 payload。这些 API
都不会隐式排序或构造 hash table。已排序输入会得到全局 sorted unique；任意输入
保持 consecutive run 顺序。当前合同不实现 global first-occurrence unique。
unique-by-key 接受 StructNdarray raw payload；dense MatrixField payload 当前要求
输入输出同形且元素为 `ti.i32`。

当前 RLE/Unique 合同的 key 支持 `i32/u32/i64/u64`。`size=None` 使用完整固定容量；integer
`size` 只处理 active prefix，`size=0` 是支持的逻辑空输入表示。count 与 length
均为 i32，所以容量上限为 `2^31-1`。input/output storage 不可 alias；只有
device-side count 以下的输出有效。

这是固定容量 tradeoff：`size` 改变语义，但 flags、compact dispatch 与 scratch
仍按物理 capacity 保留，从而避免 Graph replay 重建。若利用率长期偏低，应按 capacity
对 workload 分桶。

实现把一个 boundary kernel 与既有 native compact provider 组合。RLE 还会 compact
run start 并执行一个 length kernel，因此不新增 backend ABI 或 versioned CUDA
library 依赖。Unique 的最低可复用 scratch 为 4 bytes/item，RLE 为 12 bytes/item，
另加 compact provider 的临时空间。`RunLengthWorkspace` 可以复用但不可并发共享；
CPU、CUDA、Vulkan 已用两个 Python submission thread 和独立 workspace 做压力回归。

Windows 开发机（Ryzen 9 9950X、RTX 5090 driver 610.62）上，1,048,576 个 i32 key、
262,144 个 run 的实测如下：

| 后端 | public RLE | PrimitiveSequence Graph | host round-trip | host/public |
| --- | ---: | ---: | ---: | ---: |
| CPU | 4.85 ms | 4.22 ms | 4.98 ms | 1.03x |
| CUDA | 0.418 ms | 0.456 ms | 12.19 ms | 29.2x |
| Vulkan | 0.643 ms | 0.632 ms | 16.03 ms | 24.9x |

compile/warmup 不计时，workspace 已复用，测量前没有其他 Python/GPU compute
process。这是开发证据，不是跨驱动性能保证。CUDA Graph 差异约 38 microseconds，
记录为通用 native-node replay 开销；当前实现不为此增加 RLE 专用路径。

## 可复用 Segmented Reduce 与 Scan

Forge 用可复用 `SegmentedLayout` 表达固定容量 dense topology：

```python
layout = ti.algorithms.SegmentedLayout.from_offsets(
    np.array([0, 256, 512, 512, 768], np.int32),
    capacity=1024,
)
workspace = ti.algorithms.SegmentedWorkspace(
    max_items=layout.capacity,
    max_segments=layout.num_segments,
)
ti.algorithms.experimental_segmented_reduce(
    values, layout, per_segment_sum, workspace=workspace
)
ti.algorithms.experimental_segmented_scan(
    values, layout, scanned, inclusive=True, workspace=workspace
)
```

offset 必须从零开始并 nondecreasing；重复 offset 表示空 segment。也可以用
`from_segment_ids()` 输入 nondecreasing active prefix，并允许缺失 ID。构造器在
host 校验 topology，再上传 offset 与规范化 ID；若输入来自 Taichi，构造会同步一次。
在 direct call 或 `PrimitiveSequence` Graph replay 中复用 layout 时保持
device-resident。

当前 segmented 合同支持 `i32/u32/i64/u64/f32/f64` 的 scalar 1D plain ndarray 与 root-dense
field。values 长度必须等于 layout capacity；reduce output 恰有每 segment 一个值，
scan output 长度等于 capacity，且只有 active prefix 有定义。空 segment reduce 为零；
scan 可以 exact in-place 或 disjoint。MatrixField、StructNdarray 与 sparse SNode
不在当前合同内。

Segmented reduce 当前只实现 sum。ndarray `auto` 组合既有 grouped-reduce provider；
dense field 与显式 `serial` 按 segment 内 left-to-right 累加。整数 sum exact；
grouped 浮点 sum 受 method/顺序影响，serial 浮点 sum 保持公开顺序。只有 grouped
ndarray sum 具有 reverse AD；FwdMode 与 serial AD 都在输出变化前拒绝。

Segmented scan 实现 inclusive/exclusive sum。浮点 scan 始终保持 segment 内
left-to-right 顺序。Integer `auto` 刻意采用粗粒度策略：CPU/Vulkan 与普通短 CUDA
segment 使用 zero-scratch serial；只有 active item 至少 65,536，且最长 segment
至少 4,096 item 的 CUDA layout 才切到 `global_scan`，随后用无竞态的 segment-base
correction 修正。用户可检查 `workspace.last_scan_method`，或显式选择 `serial` /
`global_scan`。这是稳定 policy 边界，不承诺该阈值对每种 device 都是最优点。

topology 内存由 `layout.topology_bytes` 单独报告：每个 capacity item 4 bytes，
每个 offset 4 bytes。Workspace current/peak 只统计可复用执行 scratch。短分段
serial scan 的 scratch 为零；global scan 可持有 provider storage 与每 segment
一个 base value。不可变 layout 可跨 Python submission thread 共享，但每个
producer/Graph 必须使用独立 workspace。

Windows 开发机的代表 workload 为 1,048,576 items、4,096 个长度 256 的 segment；
每项取 5 个 trial 的 median，每 trial 20 次 hot replay，复用 layout/workspace，
compile/warmup 不计时。GPU 仅在确认没有其他 Python/GPU compute process 时测量。

| 后端 | reduce public | reduce Graph | host round-trip | host/public |
| --- | ---: | ---: | ---: | ---: |
| CPU | 0.770 ms | 0.805 ms | 1.003 ms | 1.30x |
| CUDA | 0.0756 ms | 0.0736 ms | 2.881 ms | 38.1x |
| Vulkan | 0.0751 ms | 0.0716 ms | 4.538 ms | 60.4x |

| 后端 | i32 scan public | i32 scan Graph | host round-trip | host/public |
| --- | ---: | ---: | ---: | ---: |
| CPU | 0.500 ms | 0.495 ms | 3.108 ms | 6.22x |
| CUDA | 0.165 ms | 0.161 ms | 6.304 ms | 38.3x |
| Vulkan | 0.176 ms | 0.187 ms | 8.859 ms | 50.3x |

| 后端 | f32 scan public | f32 scan Graph | host round-trip | host/public |
| --- | ---: | ---: | ---: | ---: |
| CPU | 0.604 ms | 0.516 ms | 3.714 ms | 6.15x |
| CUDA | 0.146 ms | 0.161 ms | 8.008 ms | 54.9x |
| Vulkan | 0.167 ms | 0.197 ms | 10.237 ms | 61.2x |

不可变 topology 占 4,210,692 bytes；一次性构建/上传在 CPU、CUDA、Vulkan 上分别为
10.67 ms、17.40 ms、32.56 ms。短分段 scan scratch 为零；CPU grouped reduce
持有 262,144 bytes，实测 CUDA/Vulkan grouped provider 没有 Python-owned scratch。

为避免只对短 workload 过拟合，只增加了一个 64 segment、每 segment 16,384 items
的反例：

| 后端 | 显式 global scan | 显式 serial | 实测优选 |
| --- | ---: | ---: | --- |
| CPU | 5.984 ms | 0.586 ms | serial，10.2x |
| CUDA | 0.871 ms | 1.800 ms | global，2.07x |
| Vulkan | 3.855 ms | 1.597 ms | serial，2.41x |

这些历史结果描述普通 API 的粗粒度 backend 分派，不是阈值扫参、跨 driver 保证，
也不能作为否定新 whole-Graph 实现的依据。

`GraphBuilder.segmented_scan()` 对固定、互不重叠的 1D i32/u32 ndarray 与不可变
segment layout 提供独立的 CUDA 完整 recipe 域。默认 provider 保留 serial、shared-memory
chunk carry、global correction，以及混合段长下的 bucketed 策略，同时提供寄存器 shuffle
和分层 warp carry。新方案执行精确的模 2^32 inclusive/exclusive sum，不分配 global scratch；
分层方案每 block 使用 16 字节 shared memory，每 chunk 两个 block barrier，纯 warp 方案
不需要 shared memory 或 block barrier。这些是完整操作的物理实现，不是公开 block 参数轴。

global-correction 策略使用 retained CUDA Graph 和独立的 Graph-bound scan scratch，普通 Program
arena 增长/清理不影响录制。它只处理 `num_items`，保持未使用 capacity 不变；改变的是 host 提交
与 scratch 生命周期，不改变数学求和或普通 primitive 的 `method="auto"`。录制、显存及首次编译
代价见 [Graph runtime 优化](graph_runtime_optimization.zh.md#global-segmented-scan-的-retained-recording)。

通过 `builder.freeze().search_recipes(...)` 按实际 workload 指标搜索，普通 `method="auto"`
不变。这不等于任意相邻 producer/consumer kernel 融合：值语义、可见中间写入与跨语言下沉
仍需要单独的合同。

### 完整 Graph 分段归约

`GraphBuilder.segmented_reduce(values, layout, output, op="sum")` 为固定、互不重叠的普通
1D i32/u32 ndarray 和 host 发布的不可变 segment offsets 增加 CUDA recipe 域。每段输出一个值，
空段输出零，所有策略均执行精确模 2^32 求和；此入口不支持浮点重排或自动微分。已有
`experimental_segmented_reduce()`、`PrimitiveSequence.segmented_reduce()` 的更广合同与 auto 分派不变；
新 Graph 域的 baseline 是段内串行，并非已有 auto 路径。

默认完整 recipe provider 发现段内串行、warp 协作、block 协作，以及段跨多个内部 chunk 时的
partial/finalize 策略；不公开 block 参数。两阶段方案持有 tile offsets、段到 partial 的 offsets 和
partial 结果，精确 requested bytes 进入物理/资源报告，不冒称 driver 显存峰值。调用者已有 layout
存储共享，不逐候选重建。kernel 与绑定在稳态前准备，replay 提交 retained nested CUDA Graph，
不向 Python 读回结果；释放沿用已有 Graph allocation 生命周期，只支持一个 workspace lane。

通过 `builder.freeze().search_recipes(...)` 和已有选择/resolve 流程使用。短段可能更适合串行，长段
可能受益于 partial/finalize 的并行度，但增加 scratch 与准备成本；不保证统一加速，不改变普通 auto，
也不由该操作直接融合任意相邻 kernel。默认搜索还可发现
[认证 pointwise 值 recipe](graph_runtime_optimization.zh.md#分段归约两侧的认证-pointwise-值融合)，
仅处理有界 i32/u32 producer/consumer，保留所有可见 store。

## Device-side 数值检查

这些 API 是 Forge 新增公开 API，不是 vanilla Taichi 1.7.4/1.8.0 API。它们必须在
Python scope 调用，不能在 `@ti.kernel` 或 `@ti.func` 内部调用。调用本身只提交后端
native primitive，并把结果写入 device-side scalar；只有调用 `to_int()`、`to_bool()`、
`ok()` 或 `to_float()` 时才把标量读回 Python。

`check_count` 类 API 支持 `i32/u32/i64/u64/f32/f64` 的 scalar 1D `ti.ndarray`、
`StructNdarray` scalar member view，以及 root-dense-place dense field。`metric_reduce`
类 API 支持 `f32/f64` 的同类输入；Vulkan 当前只开放 `f32` metric fast path。
`max_abs_delta` 可以直接比较 shape/dtype 相同的 dense field 与 plain ndarray 或
`StructNdarray` scalar member view；这会走后端 native mixed-storage 路径，不经过
host staging。

```python
workspace = ti.algorithms.CheckWorkspace(max_items=n)
bad = ti.algorithms.index_bounds_check(indices, lower=0, upper=n, workspace=workspace)

# Python 分支会同步读取 1 个标量。
if not bad.ok():
    raise RuntimeError("indices out of bounds")
```

对热循环，显式复用 `CheckWorkspace` / `MetricWorkspace` 可以复用 result scalar、scratch
buffer 和后端 replay plan。

## Workspace

多数 native 算法接受 `workspace=` 或返回可复用 workspace。复用 workspace 可以让后端 scratch
buffer 和 native plan 跨帧或跨重复调用存活，是热循环推荐写法。除非具体 workspace
明确提供同步，否则并发调用必须使用独立 workspace。

普通 primitive 的 GPU scratch 归 active Program 的 primitive arena 所有，通过既有 workspace clear/reset API
回收，不再保存在 process-global owner map。CPU primitive scratch 对每个算法族、每个 worker
thread 最多保留 8 MiB；8 MiB 到 64 MiB 使用瞬时分配，超过 64 MiB 保持既有 serial
fallback。这是驻留上界策略，不表示每次操作的峰值临时空间都不超过 8 MiB。

当公开调用传入 `workspace=None` 时，允许缓存的算法会按 active Program 和 Python
submission thread 分离隐式 workspace。每个 thread context 默认最多 64 个 entry，整个
进程默认最多 16 个 context；可分别用
`TAICHI_FORGE_DEFAULT_WORKSPACE_CACHE_LIMIT` 和
`TAICHI_FORGE_DEFAULT_WORKSPACE_CONTEXT_LIMIT` 在启动前调低或关闭。达到 context 上限的
新 thread 使用不缓存 workspace，不会驱逐或清理另一个 thread 可能仍在异步使用的资源。

`clear_default_workspaces()` 只应在 primitive submission 已静止时调用。它会先原子分离
cache metadata，再在锁外清理资源；并发清理正在使用的显式或隐式 workspace 不属于支持
合同。`ti.reset()` 已建立同类静止边界并清理这些 cache。

```python
workspace = None
for _ in range(num_steps):
    workspace = ti.algorithms.experimental_transform(
        src, dst, scale=2.0, bias=1.0, method="auto", workspace=workspace
    )
```

可观测性是 opt-in 且读取时不等待 device：

```python
ti.algorithms.set_primitive_diagnostics_enabled(True, clear=True)
ti.algorithms.experimental_transform(src, dst, method="auto")
snapshot = ti.algorithms.get_primitive_runtime_diagnostics()
print(snapshot["providers"], snapshot["fallbacks"])
print(ti.algorithms.get_primitive_workspace_statistics())
```

两个 snapshot 都使用 `schema_version=1`。runtime diagnostics 报告 provider、
`dependency_class`、fallback 和原始 counter；workspace snapshot 分开报告 Program-owned
provider bytes 与 Python 默认 cache 的逻辑 current/peak bytes。二者可能引用同一底层资源，
不能相加。CUDA CUB reference 名称作为 canonical driver family 的 alias 报告，避免重复
计数。读取这些 metadata 不调用 `ti.sync()`；要证明某一次自动选择，应先清空并只执行该
调用，再读取 snapshot。

## 与 graph 的关系

Forge 可以把由算法层产出的 DSL-defined native primitive sequence 放进 graph replay。
`PrimitiveSequence.run_length_encode()`、`unique()`、`unique_by_key()`、
`segmented_reduce()` 与 `segmented_scan()` 会持有固定 array/layout 与可复用
workspace；replay 不把 device state 读回 Python。
`DeviceCheckResult`、`DeviceMetricResult` 和 `PrimitiveSequence` 都可以通过
`GraphBuilder.append_native(...)` 作为 native graph node 追加。graph replay 只更新
device-side scalar，不会自动把结果读回 Python，也不会把检查结果转换成 graph 内部 host
控制流。

```python
workspace = ti.algorithms.MetricWorkspace(max_items=n)
err = ti.algorithms.max_abs_delta(values, reference, workspace=workspace)

builder = ti.graph.GraphBuilder()
builder.append_native(err)
graph = builder.compile()

graph.run({})
print(err.to_float())  # 显式读取 1 个 device scalar
```

这不等于向用户暴露任意 native callback；普通 Python 算法调用也不要求 graph 参与。
native graph node 是 JIT replay node；`ti.aot.Module.add_graph()` 当前只导出普通
kernel CGraph，会拒绝包含 Forge native node 的 graph。

## 与 vanilla Taichi 的关系

vanilla 兼容的 `parallel_sort()` 仍然保留。更广的 `sort()` dispatcher 和 `experimental_*`
primitive API 是 Forge 增量能力。
