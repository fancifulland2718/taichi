# Taichi Forge 版本更新说明

本文是 Taichi Forge 用户可见更新的唯一版本索引。`0.6.2` 是最新正式发布版本，其最终
用户可见实现边界是 `662affa64`，已发布 native runtime 的 build identity 是
`9b38428667e4`，仅涉及 CI 的最终发行验证边界是 `8fb5856f9`。当前 `master` 面向
`0.6.3` 开发。`0.6.1` 继续保留最终 Python shim/source 边界 `b129ad94c` 和配对 native
runtime build identity `c268ca5671e8`；`0.4.25` 仍是最后一个公开的 `0.4.x` 基线。

由于 PyPI 项目容量有限，部分不再重要的旧发行文件已经移除。因此，当前 PyPI 列表中
找不到某个版本，并不表示它从未存在。下表的源码边界是长期历史锚点；仅涉及打包、CI、
测试或文档的内部提交会归并到它实际随附的用户行为中。

## 快速索引

| 版本 | 历史状态 | 源码边界 | 主要范围 |
| --- | --- | --- | --- |
| [待发布](#unreleased) | 0.6.3 开发版本 | 当前 `master` | hardware capability/provider 合同、CUDA/Vulkan intrinsic 与 retained native execution |
| [0.6.2](#062) | 最新正式发布版本 | `662affa64` | execution-plan 收口、dynamic-work/Worklist 合同、生产 Graph replay、runtime export 控制、device-convergent 线性代数与最小 MUSA 准入 |
| [0.6.1](#061) | 已正式发布 | `b129ad94c` | task launch manifest/policy、动态 LLVM SNode directory、设备端 dynamic worklist、有界 Graph dispatch 与关联 pipeline telemetry |
| [0.6.0](#060) | 已正式发布 | `106ad65d25` | 结构化 Graph 控制/遥测与 Vulkan indirect dispatch、稀疏 runtime/线性代数、driver-only CUDA primitive、受管互操作/显示与 runtime 生命周期有界化 |
| [0.1.0](#010) | 历史源码版本；发行文件可能已移除 | `91ad177685` | scikit-build-core 迁移与 Forge 发行包重命名 |
| [0.1.1](#011) | 历史源码版本；发行文件可能已移除 | `c771969781` | `taichi_forge` import 重命名与安装布局修复 |
| [0.1.2](#012) | 历史源码版本；发行文件可能已移除 | `fe5844390b` | import 修复与 CUDA 构建选项 |
| [0.1.3](#013) | PyPI 当前保留 | `e87d42433` | LLVM 20 工具链、Forge 包身份、编译/缓存控制 |
| [0.2.4](#024) | PyPI 当前保留 | `b42aca5d9` | 编译器/缓存扩展、SPIR-V 并行、内存诊断 |
| [0.3.0](#030) | 历史版本；发行文件可能已移除 | `e9056fa7c` | 首个 Vulkan sparse/quantized 开发版本 |
| [0.3.1](#031) | 历史版本；发行文件可能已移除 | `166da399b` | inactive-read 与 allocator 修复 |
| [0.3.2](#032) | 历史版本；发行文件可能已移除 | `fac7faad9` | deterministic pointer-slot 激活 |
| [0.3.4](#034) | 历史版本；发行文件可能已移除 | `769622584` | bitmasked/deactivation 修复 |
| [0.3.5](#035) | 历史版本；发行文件可能已移除 | `6df723879` | list generation 与 sparse-pool 控制 |
| [0.3.7](#037) | 历史版本；发行文件可能已移除 | `7d095f5d5` | 保守回退 CUDA sparse-pool 策略 |
| [0.3.9](#039) | PyPI 当前保留 | `11b321dce` | Vulkan/CUDA sparse 容量策略 |
| [0.3.11](#0311) | PyPI 当前保留 | `da79573cf` | per-SNode CUDA pool sizing 与诊断 |
| [0.3.12](#0312) | PyPI 当前保留 | `653eaf468` | sparse reuse、自适应 SPIR-V、GGUI/pipeline 生命周期 |
| [0.3.13](#0313) | PyPI 当前保留 | `5e58c34b7` | 实验性 Hash SNode |
| [0.4.0](#040) | PyPI 当前保留 | `1c788298d` | native 算法、StructNdarray 路径、Vulkan offscreen |
| [0.4.1](#041) | PyPI 当前保留 | `0382b4d6b` | Graph 现代化、PrimitiveSequence、DisplayFrame、编译 profiling |
| [0.4.2](#042) | 历史版本；发行文件可能已移除 | `a1bac433b` | ArgPack、小整数、ndarray 生命周期、hidden-window 修复 |
| [0.4.23](#0423) | PyPI 当前保留 | `1f36185c7` | runtime/shim 拆包、device checks、Vulkan 修复 |
| [0.4.24](#0424) | PyPI 当前保留 | `f8dfb3e2a` | GGUI device-image staging 与渲染 cadence |
| [0.4.25](#0425) | PyPI 当前保留；最后一个公开 0.4.x 基线 | `7dad067ca` | GGUI 事件泵与 ImGui 生命周期修复 |
| [0.5.0](#050) | 已发布 runtime 源码边界 | `95626e8036` | 异步 runtime 安全、Graph replay/lifetime、Dense Field Graph |

## 待发布 {#unreleased}

- 完成 Forge-owned whole-Graph recipe 工作流。冻结的 `GraphDefinition` 现在可接入版本化内建或
  外部 provider，把兼容 fragment 组合为完整物理 recipe，并通过魔改 CompileIQ V2 协议进行
  survivor-driven staged search。显式 workload、evaluation、backend-environment、budget 与
  checkpoint 合同支持可恢复搜索；稳定 selection artifact 可在等价 Graph 上检查测量适用性并
  重新解析结构，而不反序列化 Python executable。内建优化 family 各自拥有 descriptor 与
  materializer，不再扩张已退役的中央 executable-space adapter。

  搜索返回 `GraphOptimizationReportV2`：确定性 JSON 事实源以及由它生成的 Markdown，包含终止状态、
  Pareto trade-off、recipe/physical 注释、结构化失败、provenance 与 checkpoint identity。魔改
  CompileIQ fork 提供底层版本化 `OpaqueOptimizationReportV1`，并覆盖 Python 3.10--3.14 wheel
  smoke。运行时按 V2 协议 epoch、必需 schema/API 与自洽 capability/core identity 接受；fork
  commit 和平台 wheel SHA-256 只描述某个已验证快照，后续兼容 build 不会仅因这些值或动态计算的
  Python source identity 变化而被拒绝。source identity 仍绑定每个 session/checkpoint，所以跨 build
  resume 会明确失败。该离线工作流不改变普通 compile/runtime `auto`，不增加 replay-time report
  或检查，不声称 AOT binary 复用，也不替代下游生产验证。

- 本待发布周期的历史实现与资格化记录：最初新增面向经审查魔改 CompileIQ fork、且包含 baseline 的
  Graph-owned 离线 recipe 搜索。初始审查身份为 `forge/opaque-recipes-v1.2` 的 `579b572`，支持
  Forge Python 3.10 与确定性的
  bounded-exhaustive opaque 搜索。完整、有序、按 task 索引的 CUDA kernel plan、覆盖审计和
  qualification 仍作为私有诊断基础保留；它们不再导出
  `ti.compileiq_offload_execution_plan_search()` 公共入口。无调用方的 fixed-axis kernel-wide
  adapter 已删除，raw `TaskLaunchPolicy` 也不作为 CompileIQ 搜索面。编译 identity 与
  launch-only identity 继续分离，launch request 不会从共享 compiled artifact 泄漏。

  `ti.graph.compileiq_recipe_search()` 现在搜索精确、保持 barrier 的连续 map partition，或一个
  精确 CUDA structured-control 域；位置不同但宽度相同的 source group 仍保持独立。2 至 8
  map chain 的非 baseline partition 数分别为 1、3、7、14、28、55、107；跨 phase 乘积不
  超过 4,095 个 candidate 时完整枚举，更大的积明确进入 staged domain。精确 map 物化要求
  一个 ordinary JIT CGraph 与一个 Forge-owned 源 builder，因此 composed 与 multi-builder
  Graph 会 fail closed。R11/R12 的 10-process 平衡 AB/BA 资格测试中，小调度与
  bandwidth Graph candidate 以中位比值 0.97800、0.97232 通过 worst-positive；compute-heavy
  Graph（中位 0.98703、最差 1.01958）和完整 two-task kernel plan（中位 1.00021、最差
  1.01134）作为负面项保留。四个 scope 都通过 correctness、精确 route 与 memory stability；
  两个负面项未触发三个 scope 的扎堆复审阈值。

  除 flat `auto while` 的 conditional/masked 域外，
  R6 新增独立 depth-2 域：一个 root outer `auto while` 按序直接拥有 1 至 8 个 leaf inner
  `auto while`，搜索 device-update 与 masked-bounded 两种物理 plan。nested 域使用独立的
  opaque namespace、semantic identity 与精确显式重建。显式 lowering policy、portable
  control、多个 root region、native/observation node，以及 fusion × control 或 flat × nested
  组合仍被排除。map-fusion 的运行时采纳仍由 Forge 独立的精确作用域 qualification cache
  控制；structured-control R5/R6 只能离线显式
  重建，不生成 runtime cache，也不修改 `auto`；每个 Graph 构造时会冻结物理控制路线。在
  4,096 项、实际 12 次迭代的 10-process
  平衡 AB/BA 资格测试中，max 20 保留 masked（中位 1.129x，worst-positive），max 128 保留
  conditional baseline（masked/conditional 为 2.730x），两个 scope 都保持精确结果和稳定
  memory。R6 depth-2 资格测试在稳态保留 device-update：一条与两条 inner region 的
  masked/device-update 分别为 2.130x 与 2.139x。masked 负面项继续保留，因为 cold 中位数
  反而由 masked 占优（65.61 对 74.79 ms、71.30 对 83.06 ms），persistent allocation 也
  分别从 30,884 降至 532 bytes、从 59,996 降至 796 bytes。这些本机 crossover 不改变
  runtime `auto`；compile/build 时间只作诊断。此前刻意收窄的 reduce-provider 与
  segmented-scan CompileIQ 搜索器仅保留为私有资格化工具，不再从 `ti.algorithms` 导出，也不形成
  provider 路由层。算法实现、显式 `method=` 和既有 `method="auto"` 保持不变。历史
  segmented-scan 域只覆盖 immutable `SegmentedLayout` 上互不 alias 的 CUDA plain 1D i32/u32
  ndarray sum，并比较完整的 `serial` 与 `global_scan` 物理路线。在当前 RTX 5090 /
  driver 610.62 主机上，独立 10-process x 10-block 平衡 AB/BA 资格测试为单个 32,768-item
  inclusive i32 segment 采纳 global scan：candidate/baseline 中位数为 0.31792，最差进程为
  0.31802。负面对照在 4,096 items / 64-item segment 保留 serial（global/serial 中位
  1.819x），并在 262,144 items / 65,536-item segment 保留 global scan（serial/global 中位
  3.161x）。所有路线都保持精确结果、memory stable，并由真实魔改 CompileIQ core 完整覆盖；
  测试期间存在外部 GameViewer compute process，属于性能 caveat。该证据只适用于精确 scope，
  不改变 runtime 默认值。该历史路径通过精确 lock 拒绝源码漂移；上面完成的 V2 路径改为接受协议
  兼容的 fork build，并把实际 source identity 绑定到 session/checkpoint 复用。CompileIQ 仍只是
  可选的离线依赖。

  Graph memory 现在以一个独立的 Forge-owned 完整 recipe 域接入经审查的魔改 CompileIQ fork。
  严格单 dispatch CUDA f32 affine-stencil Graph 会公开 direct baseline 与一个完整
  `shared_staged_1d` candidate；Forge 惰性保留 source lineage，并冻结 offload plan、materialized
  task manifest、halo、shared bytes 与 binding requirements。CompileIQ 只搜索 opaque spec token，
  看不到 `memory_strategy`、block、tile 或 halo 裸轴；该域也不与 map fusion 或 structured control
  组合。worker 只能在 Graph definition 构造期精确重建，之后 Forge 再验证 semantic/physical identity。
  binding 发布一次性证明 owner、extent、layout 和 alias requirement，稳定合格 replay 复用该证书；
  winner 不改变 runtime `auto`，也不生成 runtime qualification cache。

  在此完整域实现之前，正式 S4 机制资格化已在本机 RTX 5090、driver
  610.62、source/native commit `ada3e9eba912a5368c5cd1a21f1b8f5d12c08b7e` 上对精确的 radius-1
  与 radius-4 recipe 得到正项。主计时使用稳定 `GraphBindingSet`；每个 scope 包含 10 个 fresh
  process、均衡 AB/BA 顺序，以及每进程 5 组、每路线至少 250 ms 的 paired block。
  staged/direct 中位比值分别为 0.946262 和 0.953218，最差进程比值分别为 0.947035 和
  0.953970。全部 worker 均通过全数组 exact correctness、route/materialization 与 source/native
  provenance、稳定 replay 的 storage 描述/owner 校验/alias 分析零增量、submit/paced/A-B-A smoke，
  以及两轮各 10,000 replay、无固定内存上限的 memory plateau。raw dict replay 仍只是不可晋级的
  诊断项。被 Git 忽略的本机机器可读 artifact 位于
  `.agent/experiments/graph-memory-s4-ada3e9eb/qualification.json`（SHA-256
  `f788b4d109ef0b1817807fab5ef51100cea9703cb5eda8cd5de6cae471a3fa1e`）。该历史结果只资格化这两个
  精确 recipe scope，不等同于当前完整 CompileIQ 重建的性能准入；后者继续使用 matching-commit、
  fresh-process AB/BA、worst-positive 与进程显存平台 Gate。Forge 仍不把全局
  `CompileConfig.make_block_local` 或私有 per-kernel task 轴暴露为 Graph 搜索域。

  当前完整 CompileIQ 域随后在 RTX 5090、driver 610.62、matching source/shim/native commit
  `835eea2cb18c49ef66470ae4a378493fb97a0db2` 上完成独立正式资格化。每个 scope 使用 10 个
  fresh process、均衡 AB/BA、每进程 5 个每路线至少 250 ms 的 paired block，并且每个 worker
  都由经审查的 CompileIQ fork 完整穷举并精确重建 2/2 个 opaque recipe。16,777,216-item
  radius-1 与 radius-4 的 staged/direct 中位比值分别为 0.947679 与 0.965961，最差进程分别为
  0.948109 与 0.967921，因此两个精确 scope 通过 worst-positive；16,384-item radius-1
  中位为 1.007171、最差为 1.035936，作为负面 scope 保留。全部 30 个 worker 的正确性、
  route/binding identity、source/native provenance、噪声与显存平台门均通过。两轮各 10,000
  replay 之间，Forge device pool 固定为 168,173,696 bytes/2 raw chunks，进程级 GPU 显存的
  波次增量为 0；大规模 scope 为 1061.66 MiB，小规模 scope 为 741.660--741.664 MiB。
  staged route 的每 block 静态 shared memory 为 radius-1 的 520 bytes 与 radius-4 的
  544 bytes；这些数字是完整双路线 worker 的平台观测，不是单 recipe 独占显存声明。
  工件状态为 `partially_qualified` 且 strict gate 通过，位于
  `.agent/experiments/graph-memory-compileiq-835eea2c/qualification.json`（SHA-256
  `7d97c2c2bce00b8d42a4e7e9f47888b088fefc77b0d4a91ef9207e7d4bc2e6d6`）。结果只资格化上述
  大规模精确 scope，负面小规模 scope 不准入，且不改变 runtime `auto`。
- 新增稳定公开的 `Graph.bind(arguments)` 与 `ti.graph.GraphBindingSet`。发布时会按值快照
  scalar/matrix、按 identity 保留 device resource，并在版本可见前完成 owner、layout、
  structured-control 与 alias 资格化；失败的 `update()`/`replace()` 不替换旧版本。同一已发布且
  满足资格的版本在后续 `Graph.run()`/`Graph.submit()` 中复用预扁平化 frame，不再逐 replay
  构造 Python storage descriptor。有限的 ping-pong/ring 资源应各自预发布一个 BindingSet，
  使 A→B→A 同时复用 Python 证书与原生 CGraph MRU plan。原生层仍在每次提交核对带
  generation 的 resource identity 并取得异步 lease；动态 provider、replacement、temporary 与
  逐提交 owner 保持保守 slow path。paced submit 在 admission 等待后选取当前完整版本，
  `ti.reset()` 会使旧 Graph 及其 BindingSet 失效。
- 新增精简且稳定的 `HardwareCapability`、`HardwareProviderStatus`、
  `HardwareExecutionReport` 状态层，以及 schema-v4 diagnostic operation/provider 合同。
  被动 status/report 不加载、benchmark、启用或选择可选 provider。`graph_integration`
  现在区分 inline、root-ordered rerecord、backend-recorded、stream-captured、opaque 与
  unsupported，不再把所有 root-Graph action 混称为 recordable。显式 cuBLAS、cuSPARSE 与
  cuFFT probe 使用瞬时 vendor handle；cuDSS probe 通过 wheel 内薄 C-ABI adapter 瞬时核对
  用户 vendor runtime。它们都不会隐式启用 provider。
- 新增 Forge 自有的 runtime-loaded execution adapter，用于用户管理的 cuSPARSELt
  0.8.x-0.9.x、cuTENSOR 2.0.x-2.7.x 与 AmgX stable C API。显式 provider plan 会保留用户
  runtime，并执行有界的 FP16 2:4 matmul、FP32 contraction 或 host-CSR solve；probe 仍不
  执行算法。薄 adapter 随不变的 runtime wheel 提供，不增加 vendor link dependency、
  Graph、automatic 或 kernel-inline 路线。AmgX coefficient update 在存在可选 resetup
  entry point 时使用它，否则对未导出该 symbol 的兼容 runtime 回退到完整 setup。NCCL
  仍不进入已注册 provider catalog。
- 新增可选 D1 `ti.hardware.linalg.gemm_f32`，通过 direct Python 与 root Graph 对 compact
  row-major f32 matrix 执行 `C = alpha * A @ B + beta * C`。真实执行才 lazy-load 用户的
  兼容 cuBLAS，并在每个 Program 内复用一个 handle；Forge 不新增 Toolkit header、
  link dependency、bundled runtime、package dependency、build switch 或 wheel 变体，
  也不会改写普通 matrix multiplication 或 kernel。
- 在 hardware catalog 中资格化 CUDA sparse 路线：
  `SparseMatrix.spmv(method="auto")` 只有在存在匹配且稳定的 matrix-scoped 成本证据时才
  考虑 cuSPARSE；证据只能从严格的 fresh-process qualification artifact 加载。
  admission-schema-v2 证据绑定精确 topology、device、Python extension、split native runtime、
  runtime bitcode 与 provider ABI；cuDSS 证据还分别绑定精确 adapter 与 vendor runtime
  binary，旧 profile 会
  fail closed。原手填 timing setter 已移除；当前 workload 可提供正整数 `expected_reuse`
  覆盖值，以重新计算 provider 和内嵌 baseline 两侧首次成本。显式
  provider 路线不受成本 gate 限制；
  `SparseSolver(provider="auto", provider_profile=profile)` 对用户管理的 cuDSS 0.8.x 使用
  同一套 exact-scope admission，并对称摊销 cuDSS 与 cuSOLVERSp 的首次成本。缺少证据时不会
  探测可选库并保留 cuSOLVERSp；显式 provider 不受性能 gate 限制。旧版 CUDA、缺失/不兼容
  provider、不合格合同或未通过成本资格的场景同样继续使用 cuSOLVERSp；显式 cuDSS 要求
  CUDA Driver API 12.0 或更高版本。
  新增 staged `CudssPlan`、root-ordered solve recording，以及读取当前 device values、
  不依赖 host numeric-version 推断的 fixed-pattern 事务化 `record_refactor_solve()` action。
  provider 执行失败会在 submission retire 前保持 transaction reserved 并使 factor 失效，
  之后可通过完整 factorization 恢复；
  cuDSS 的 0.8 厂商 ABI 已从核心 runtime 移到 plan-owned bundled adapter；adapter 不链接或
  捆绑厂商库，不新增 wheel 变体，用户也无需重编 Forge；
  另新增独立的手动
  `ti.hardware.linalg.spmv_f32`/`CusparseSpmvRecording` 路线，支持调用方持有 output，
  并在复用 matrix-owned provider
  state 的同时录入 root Graph。所有路线都不是 kernel rewrite，vendor library 继续保持
  可选且不捆绑。
- 资格化 CUDA `cuBLAS -> kernel -> cuSPARSE -> kernel` 与 Vulkan
  `AS refit -> batch Ray Query` 的 source-ordered root-Graph 组合。backend command 仍为
  独立分段 node，不声称 backend-Graph fusion；provider-generation lease 持有 cuSPARSE/AS
  state，cuSPARSE replay 复用缓存 plan。
- 在 catalog 中记录现有 D0 kernel 硬件路线并明确其调用边界：atomic、CUDA warp、Vulkan
  已实现的 subgroup 子集、`SharedArray` 与 `ti.block_local` 是显式的 kernel-inline 语义；
  grouped reduction aggregation 与 opt-in Vulkan list-generation ballot aggregation 是带普通
  atomic fallback 的自动内部选路。该变更不增加语法、依赖、Python native action 或 wheel
  变体。稀疏 pointer-SNode block-local scatter/write-back 因 CUDA 正确性失败可稳定复现，
  继续保持未资格化。
- 官方 wheel 构建与验证会拒绝 D1/D2 vendor runtime 成为必需依赖或新增发行变体；D0
  CUDA/Vulkan runtime 边界保持不变。
- 新增 D0 `ti.graph.VulkanBufferCommand` 与 `VulkanBufferCommandRecording`，支持显式
  buffer fill/copy/barrier 直接执行和 root Graph automatic admission。当前 replay 为
  runtime-ordered `rerecord`，不支持 kernel 内调用、structured Graph 或 AOT。
- 新增 D0 `ti.hardware.image.VulkanImageCopyRecording`，通过 direct 或自动加入 root
  Graph 的命令执行完整 color image copy，精确声明 source READ / destination WRITE，持有
  runtime Texture lease，并在不 readback host 的情况下与 kernel 排序。它要求 format 与
  extent 相同，拒绝 alias 和 depth/stencil。由于 Vulkan 不保证物理 copy engine，catalog
  只声明 native device command 与 implementation-defined hardware acceleration。
- 资格化现有 D0 Vulkan `ti.Texture` 硬件采样路线：显式 texture op 会自动 lowering 到
  SPIR-V image/sampler 指令；普通 field/ndarray 不会自动替换。新增 immutable
  `ti.hardware.sampling.SamplerConfig` min/mag filter 与逐轴
  repeat/mirrored-repeat/clamp-to-edge address，由 device sampler cache 复用；精确
  `fetch()` 继续与 sampler 无关，边界仍是单 mip normalized sampling。另新增匹配的显式
  CUDA JIT-kernel 路线，通过 Driver API array/texture object 支持合格的一、二、四通道格式；
  cached CUDA Graph capture 保留 sampled texture generation 并支持 replay/换绑。
  immutable-binding-frame recipe 同样保留 sampled texture、接受 dispatch label，且只在
  binding 发布时上传参数；CUDA
  read/write texture 与 texture AOT manifest 仍不支持。两个 backend 都不会
  静默替换 field/ndarray 访问，也不增加官方 wheel 变体。
- 新增 D0 `ti.hardware.graphics.VulkanGraphicsPipeline`，在调用方提供的 SPIR-V、精确
  vertex/index layout、runtime-owned color/depth texture 上提供 renderer-neutral 的 direct
  或 root-Graph draw recording。compute/graphics 排序使用 device-side semaphore bridge，
  不隐式等待 host。接口刻意不提供 scene、camera、material、lighting、shader compiler 或
  presentation policy，不新增依赖或官方 wheel 变体；driver pipeline 内存按 opaque 报告。
- 扩展 Vulkan graphics command，加入 fixed/count-buffer indirect draw、有界 immutable
  bindless storage-buffer table snapshot，以及持有 descriptor generation 的 replay。indirect
  输入可由 kernel 生成，无需把 draw count 读回 host。另新增显式
  `ti.hardware.graphics.VulkanMeshPipeline` 命令，在设备完整提供 feature/limit chain 时执行
  调用方提供的 `SPV_EXT_mesh_shader` mesh/task shader。Forge 不提供 meshlet、culling、
  material 或 renderer policy，也不作通用 mesh-shader 加速声明。
- 新增 D0 `ti.hardware.raster.RasterPass`，复用当前 Program 的 Vulkan GGUI/RHI
  graphics pipeline，作为兼容/资格 adapter 提供 mesh/instance/particle/line 的显式
  offscreen 硬件光栅与
  独立 color/depth readback。当前支持 Python direct 与显式分段 root-Graph；kernel 调用、
  automatic Graph admission、structured region 与 AOT 均明确拒绝。分段路线保持顺序与
  lifetime，但不声称 backend-Graph fusion；不增加依赖或官方 wheel 变体。
- 新增首个 D0 `ti.hardware.matrix` 切片：在 NVIDIA compute capability 7.0 及以上
  设备上，通过 CUDA Driver/PTX WMMA 显式执行 compact row-major `m16n16k16`、f16
  输入、f32 累加/输出，支持 direct 与 root Graph。它不依赖 CUDA Toolkit runtime 或
  厂商算法包；普通矩阵乘法不会被改写，kernel 内调用仍不支持。
- 新增底层 D0 `ti.hardware.ray` 独立资源：fixed-topology `TriangleBLAS`、fixed-order
  multi-instance `InstanceTLAS`、instance transform/mask/custom index，以及 direct Python /
  root Graph batch Ray Query。BLAS/TLAS build/refit 都是 root-ordered action；refit 保持
  BLAS topology 和 TLAS 的 BLAS 数量/顺序。`TriangleScene` 保留为单 identity-instance
  兼容 wrapper。procedural geometry 与改变 topology 的 update 仍不支持。另新增显式 JIT
  Vulkan kernel-inline closest-hit 路线，作用于 opaque triangle acceleration structure；
  acceleration-structure Graph argument 与 AOT 仍不支持。不增加 SDK runtime 依赖或官方
  wheel 变体。
- 新增可选 D1 `ti.hardware.fft.CufftPlan1D`/`CufftPlanND`，通过 direct Python 与 root
  Graph 执行 fixed-size、batched、single-precision C2C/R2C/C2R transform，并支持带显式
  embed/stride/distance 的 rank-2/rank-3 layout。显式 probe 仍为无副作用的
  transient load；只有创建 plan 才会 lazy-load 用户提供的兼容 cuFFT shared library。
  Forge 不捆绑或链接 vendor library，也不新增 package 或官方 wheel 变体。
- 新增首个已资格化的 D0 透明 specialization。在 CUDA compute capability 8.0+
  与 PTX 7.0+ 上，至少 8 KiB 的 compiler-generated `ti.block_local` struct-for
  prologue 会对 primitive 4/8/16-byte direct global-to-BLS copy 自动使用 PTX
  `cp.async`，但仅限没有 write-back epilogue 的 read-only cache；较小、非 direct、
  read-write 或较旧 target 的 workload 保留同步 lowering。
  `ti.hardware.report()` 会区分 provider eligibility 与实际 compiled kernel
  specialization 的 selection。该路线不新增公开 CUDA 指令语法、Toolkit runtime、
  package 或 wheel 变体。
- 新增显式 JIT Vulkan cooperative-matrix kernel lowering：
  `ti.hardware.matrix.cooperative_mma_f16_f32`。支持的 subgroup-scoped
  f16/f16/f32/f32 M/N/K tuple 从当前设备枚举，不照搬 CUDA WMMA。普通 matrix
  multiplication 不会被改写；不支持的 tuple、AOT、逃逸 tile 与未资格设备均 fail closed。

## 0.6.2 {#062}

`0.6.2` 在最终用户可见实现边界 `662affa64` 收口 `0.6.1` 之后的更新。已发布 native
runtime 构建于 `9b38428667e4`；该边界晚于实现边界的内容只有 Windows split build 顺序
修复。`8fb5856f9` 的 Python 3.14 load-order validator 收口仅作用于 CI，不改变 wheel 或
runtime 行为。纯文档提交 `40a82bed4` 与 `f757463f7` 分别归入其描述的 Vulkan 和 Graph
行为。以下条目均属于已发布行为；只有明确标记为 experimental、opt-in、diagnostic 或
source-build-only 的 API/后端路线仍保持相应边界。

- Vulkan feature 探测现在严格遵守 core promotion 合同，即使合规驱动不再枚举已提升的旧
  extension 名称也能正确工作。physical-device Features2 在 Vulkan 1.1+ 使用 core 入口，
  只有 Vulkan 1.0 加对应扩展时才使用 KHR 入口；8-bit storage 与 shader int64 atomic
  均从 Vulkan 1.2 起按 core 候选处理，而不是 Vulkan 1.1。请求的 instance version 还会
  限制在 loader 实际支持的版本内。这既避免严格 Vulkan 1.1 驱动收到非法的 8-bit-storage
  查询链，也恢复 Vulkan 1.1/1.2 驱动上不重复枚举旧扩展名时的 promoted feature 探测。
- CPU、CUDA 与 Vulkan 上，直接 root-dense Field template kernel 现在可以跨 serial
  SNodeTree generation 复用经 compiler 资格化的 executable template。frontend 会验证完整
  direct dependency 集合，compiler 会分类 embedded state，每个新 generation 仍创建独立的
  root/runtime/backend binding。旧 Graph 继续 stale，绝不自动 retarget。pointer、bitmasked、
  dynamic、hash SNode、隐藏 capture、grad/dual Field、data-oriented template 以及不匹配的
  layout/policy 继续走 generation-specific fail-closed 路径。resident specialization 预算按
  executable template 与被 pin 的 handle 计费，不再按历史 compile 次数计费；生命周期遥测会
  分别报告 template hit、generation binding、reclaim 与 pinned retired handle。
- device-count bounded dispatch 新增逐节点的
  `physical_grid="auto|extent|capacity"` 策略。`extent` 选择已通过资格、无 host readback 的
  物理 range 路线（CPU scheduler chunk、CUDA 12.4+ adaptive node update 或 Vulkan
  indirect dispatch），`capacity` 则是显式固定 grid 基线。capability 与 handle telemetry 会
  如实报告请求策略和后端实际 route，不会把 logical mask 伪装成 exact physical launch。CUDA
  12.4+ setup qualification 对瞬时 device-node status 最多允许两次 retry，并公开 retry count；持续
  status 或 driver error 仍 fail closed，普通 replay 不承担 retry 或 host synchronization 成本。
- Graph 可通过 `GraphBuilder.private_ndarray()` 与 `Sequential.private_ndarray()` 声明
  instance-owned 私有 ndarray；recordable provider 可在 fixed binding 中使用
  `GraphOwnedNdarray`。私有存储不进入公开 run schema，地址在 instance 生命周期内稳定，默认
  通过 completion fence 独占复用，并可通过独立 workspace lane 支持异步重叠；内存统计会如实
  计入每条 lane。
- cached CGraph replay 现在保留带 generation 的 resource binding plan。纯 ndarray/scalar
  Graph 不再取得 SNode lifecycle guard；CUDA exact replay 无需重建 allocation vector 即可比较
  稳定 signature；Vulkan 在资源 generation 与 scalar/matrix 值不变时复用 immutable launch
  context。重复 signature 使用有界 MRU：每个 CGraph 最多保留四个 runtime binding plan、
  两个 CUDA executable resource signature 与四个 Vulkan immutable launch signature；
  scalar/matrix 值变化会 patch 或复用兼容 slot，不形成无界资源历史。dispatch label 保持为
  task metadata，不再禁用 CUDA/Vulkan native replay。Field/SNode binding 继续使用完整
  guarded path，各后端仍保留 submission-scoped resource。公开
  `Graph.execution_stats()` schema v6 保留关闭的 host attribution 结构，但 snapshot 严格无
  副作用：读取不会启用时钟、counter 或让后续执行增加 host readback；逐次测量使用显式
  submission telemetry。
- `ti.cuda` 下新增实验性的最小 MUSA Driver API provider。Linux 在没有 NVIDIA CUDA 时可
  fallback 到 `libmusa.so`；Windows 必须显式设置 `TI_CUDA_DRIVER_PROVIDER=musa`。该路线要求
  32-lane warp，offline cache identity 与 NVIDIA CUDA 分离，并公开实际 provider；目前只准入
  基础 PTX kernel 执行。CUDA 专用 library、native primitive、Graph 路径、异步 memory pool
  与外部互操作均未资格化，因此这是可运行性准入，不是性能承诺。
- `ti.profiler` 新增默认关闭的 CPU ThreadPool telemetry。显式窗口报告 job、chunk、worker
  admission/underfill、queue occupancy、nested serial、异常与累计 queue/execution/wait 时间。
  关闭时每次 ThreadPool invocation 只增加一次 relaxed flag load，不读取时钟，也不更新逐
  chunk counter。
- 普通 Forge-owned ndarray launch 现在通过稳定 generation slot 绑定；无 SNode dependency
  的 compiled kernel 还会复用 immutable registered execution plan，并不再取得全局 SNode
  生命周期读锁。external/mixed resource 与 Field/SNode-dependent kernel 保留完整通用所有权
  路径，CUDA/Vulkan 异步 submission 继续持有 completion-scoped lease。fast path 不增加 host
  readback 或 replay allocation。在 Windows CPU 上交错执行的 10 对 fresh-process 资格中，
  zero/one/two/four resource 与 65k range fill 的 Forge/vanilla median throughput ratio 分别为
  1.024x/1.154x/1.104x/0.993x/0.988x，成对 CV 为 1.2%-2.5%。该数据只资格化这台机器上的
  fixed launch overhead，不代表通用 kernel throughput。
- 可复用 CUDA/Vulkan 普通 launch plan 现在通过一次有类型的 native patch 刷新全部 scalar
  参数，不再让每个 scalar 单独跨越 Python/pybind。整数符号、NumPy scalar 接受范围、资源
  identity 与 generation guard 均不改变。本地七 scalar enqueue 诊断的 200 样本 median 在
  CUDA 上从 42.9 降至 34.0 微秒，在 Vulkan 上从 60.6 降至 51.4 微秒；这只是隔离 host
  path 方向，不是应用吞吐声明。SNode-dependent kernel 仍排除在 ordinary launch-plan cache
  之外，因为 generation binding 与 launch context 会随 tree 退役；上述单独资格化的
  root-dense executable template 路径只有在新 generation 显式验证并完成 binding 后才复用代码。
- 普通 launch plan 现在使用每 Kernel 四槽的有界 LRU，不再只替换一个资源槽。常见
  ping-pong 与 triple-buffer ndarray signature 会保留各自 launch context；MRU 稳态仍只做一次
  比较，entry 只持有弱资源 guard，失效 generation 会被清理，`ti.reset()` 会清空整个缓存。
  ndarray kernel 捕获 SNode state 等稳定 admission 失败会按 compiled specialization 负缓存，
  通用路径不再每次 launch 都重复 registration。一个本地 696 次 ping-pong 诊断中，相对模拟
  的旧单槽策略，同步 batch median 在 CUDA 上从 135.9 降至 29.2 ms，在 Vulkan 上从
  144.7 降至 23.2 ms；这些隔离 host-path 数据不代表应用吞吐。
- 新增默认关闭的 startup phase 归因：可通过 `TI_STARTUP_PROFILE=1`、
  `ti.runtime.configure_startup_profile()`、`ti.init(startup_profile=True)` 与
  `ti.runtime.startup_profile()` 使用。import、split-runtime load、pybind、Program creation、
  runtime materialization 与 primitive-capability registration 保持为独立 checkpoint。关闭路径
  不读取时钟；snapshot 不要求 active Program 或 CUDA driver。
- Graph、SolvePlan 与 device-convergent batched submission 现在区分
  `telemetry="summary"` 和 `telemetry="timestamps"`；`True` 保留为 timestamps 的兼容别名。
  summary 保留停止 snapshot、queue/submission taxonomy 与 pipeline structure，但不插入 backend
  timestamp marker。Vulkan timestamp marker 会写入 runtime 已拥有的 command list，而不再产生
  marker-only command list；timestamp report 仍明确标为 measurement-path changed。本地短
  structured probe 的 non-exact queue window 都观测到三次 queue call，而 submitted command
  buffer 从 12 降到 9；200 个诊断样本的 timestamp submit/wait/materialization median 从
  1028.8 降到 942.0 微秒。
- 新增 `Graph.prepare_telemetry(mode, slots=1)`，可在不执行 Graph、也不读取用户资源的
  前提下显式物化有界 telemetry arena、编译 packed snapshot kernel，并在 timestamp 模式下
  预热 backend event/query 初始化。`SolvePlan.prepare_telemetry()` 委托到所选 cached
  submission variant；device-convergent `BatchedSolvePlan.prepare_telemetry()` 则预热其单个
  Graph replay。默认 `telemetry=False` submission 仍不产生这些分配。
- 普通 compact ndarray specialization 现在在 LLVM/SPIR-V 中保留 canonical addressing，
  不再让每次访问都读取 affine offset/stride metadata。positive-stride storage view 使用独立的
  runtime-affine specialization；symbolic Graph ndarray 因 replay 可能绑定两种 layout，继续按
  runtime-affine 编译。三对本地诊断中，代表性 CPU stencil 相对同一 vanilla control 从
  0.661x 提升到 1.113x，reduction 从 0.938x 提升到 1.220x；样本数与 CV 不足以把这些倍率作为
  可移植 speedup 发布。
- dynamic-work capability report 升级为 schema v5，并分别报告 device-extent publication
  合同、backend reuse、静态 route admission 与 opt-in physical blocks/threads 观测。每次
  publication 都携带 immutable generation，consumer 只会复用同一 generation；CUDA 无法消费
  producer-owned launch packet 时会拒绝，而不是验证后丢弃。Graph physical plan 分别报告
  logical native action、backend Graph launch、physical queue submission 与 loose helper。
  `admission="auto"` 拒绝 fragmented native provider；显式 admission 保留诊断用 segmented
  execution。Prefix/worklist sequence 在 materialization 时固定 provider 与 workspace topology。
  Worklist conflict resolve 现在把 `dense_atomic`/`radix_grouped` strategy 与 native sort
  provider 分开；有界 dense domain 按 priority、ordinal、source index 确定性处理 tie，越界
  key 记为 overflow；小规模 CPU worklist 默认保守保留 radix。配对的
  `benchmarks/device_worklist_conflict_bench.py` 使用完全相同的输入，校验输出一致性，并报告
  raw sample/CV 与 workspace byte，用于独立资格化 route。
- `DeviceWorklist(telemetry=False)` 现在会省略 optional counter allocation、binding 与 device
  write。atomic-append workload 还可选择 `transition_mode="direct"`，把 mandatory counter state
  从 12 bytes 降到 8 bytes，并删除 finalize action。dense arbitration 可直接返回 source-index
  winner table，不再生成 scan/compact/winner list。在 capacity=65,536、item=16,384 的配对
  transition harness 中，direct 路线的 median latency 在 CPU/Vulkan 上分别降低 7.5%/13.1%；
  CUDA 观测到的 4.8% 方向因 staged CV 超过 5% 而不作为合格结论。
- direct worklist 现在可通过 `unique_key_capacity` 增加有界 dense-key generation table，并在
  producer 内调用 `device_worklist_append_unique_direct()`。静态 Graph stage 使用
  `transition_arguments(step)` 在两份 frontier 间交替，不清空 dense table 即推进 epoch，传播
  overflow，并把每次新发布的 `DeviceExtent` 直接交给 bounded consumer。这样不再需要 Forge
  侧 full-domain select/rebuild；但端到端 active-domain scaling 仍要求调用方融合 retired/default
  state 处理，并删除自身的 full-domain pass。该路线保持显式，不附带自动加速声明。
  Vulkan recordable prepare 会在同一次 dispatch 中 reset target 并发布 source indirect packet；
  mixed indirect recording 会保留该 recipe，使一组 prepare/consume 降为单 backend Graph region、
  两个 physical dispatch，且没有 loose helper。
- 新增 `DeterministicScatterReducePlan`，作为显式 CPU/CUDA/Vulkan 资格与 fixed-topology
  assembly 路线。它只对 immutable integer destination 稳定分组一次，现在用一个融合 indexed
  dispatch 按 source 顺序读取 scalar/vector contribution 并从左到右求和。既有 atomic scatter-add
  仍是默认；stable serial 需显式选择，不分配 ordered-value workspace，并只承诺同一 backend/build
  可重复，不承诺跨后端 bit 一致或数值精度提升。
- fixed-domain conflict producer 现在可调用
  `DeviceWorklist.resolve_conflicts_from_mask()`，直接消费 dense key/active 数组，不再执行
  stable compact 或 attribute gather。winner identity 仍是原始 source index；省略显式
  ordinal 还会删除 ordinal arbitration pass 与 buffer。current-contract 资格入口在通用
  Falling Sand 案例中使用该路线，并保留单独命名的 legacy 入口。本地诊断 A/B 仍由工作负载
  专用四 kernel atomic control 占优，因此 self-contained resolver 保持显式使用，不宣称自动加速。
- 未提供自定义 ordinal 时，`resolve_conflicts_from_mask()` 的 priority 形式会在 CPU/CUDA 上用
  packed 64-bit `(signed priority, source index)` 做一次 atomic arbitration；Vulkan 与不支持的
  layout 保持 portable 32-bit multi-pass。packed route 少一个 dispatch，但每个 key 增加 4 bytes
  scratch。五对本地诊断的 median 方向为 CPU 约 5.4%、CUDA 约 22.6%，但 CUDA control 噪声使其
  只能作为 route 决策依据，不能作为可发布吞吐结论。
- `DenseConflictClaimTable` 现在提供不嵌入具体 workload 逻辑的确定性 producer-fused dense
  arbitration。声明有界 priority/source domain 后，CPU/CUDA/Vulkan 共用一个 packed u32 table；
  更宽 domain 在 CPU/CUDA 上使用 u64，并在 Vulkan 上 fail closed。调用方可以把 slot reset 融入
  已有 reset kernel、把 claim 融入 candidate producer，并在 materialization 中直接读取 winner，
  因而删除三个独立 conflict helper 和 decoded winner table。本地 65,536 source/32,768 key 的
  parity probe 把完整流水从六个 kernel 降为三个，conflict storage 从 384 KiB 降为 128 KiB；
  五组交替 pair 的同步 median 方向为 CPU 1.81x、CUDA 1.72x、Vulkan 1.47x，所有 route 的
  pair-median CV 均不超过 2.8%。该数据只描述这一有界诊断 shape，不代表自动 route 或普遍
  吞吐承诺。
- direct worklist 可在已有全局有序的 record/boundary kernel 内调用
  `device_worklist_recycle_direct()`，随后用 `commit_recycled_next()` 完成 front recycle 与 generation
  前进。这样不削弱 global-ordering 合同，同时每层删除一个独立 prepare helper。64 层配对诊断删除
  全部 64 个 helper，CPU 方向约 11.9%，CUDA/Vulkan 约 21%；这些只是本地诊断方向，不是跨设备保证。
- unique direct worklist 新增 epoch-safe fused recycle boundary，同时推进 generation 与 dense-tag
  epoch，并在耗尽时 fail closed。provider-owned `fixed_graph_args()` 还可从 Graph 的公开 replay ABI
  移除稳定 worklist storage，同时强制 runtime generation、front parity、单 lane 所有权与按 runtime
  顺序执行的异步复用。固定所有权仍保留必要的 native resource lease，不附带自动加速声明。
- CPU ThreadPool completion protocol 现在会先用 atomic sentinel 封闭 stack-owned job，再发布
  completion；迟到 worker 无法重新 join 或重复最终状态转换，非最后 worker 也无需取得 pool
  mutex。installed-runtime manifest validation 还会在 `ti.init()` 前只比较静态 schema，不再
  为此构造 backend Program。这两项属于发行稳定性修复，不附带吞吐提升声明。
- compile profile 现在会报告粗粒度 SNode 生命周期 scope，包括 tree materialization、backend
  synchronization/resource release、executable/kernel-definition retirement 与 lifecycle-lock wait。
  这些 scope 只在 compile profiling 开启时产生，不改变普通 launch 路径。
- Windows split-runtime 构建现在从 pybind object 与显式 runtime anchor 推导 export closure，
  生成确定性的 ABI manifest，并在链接后审计最终 DLL。本地 MSVC split build 将超过
  114,000 个 raw definition 收窄到 1,378 个 requested export；最终 DLL 连同源码显式
  export 共暴露 2,597 个符号，显著低于 32,768 safety cap。显式 Taichi-owned export 仍然保留，因为
  MSVC 需要 class `dllexport` declaration 为独立编译的 shim 生成 special member 与 vtable；链接后审计
  会拒绝 third-party definition owner。该方案通过控制 ABI surface 修复 LNK1189，
  无需删除 compiler/backend module。显式 target dependency 还保证 multi-config MSBuild 在
  export-closure scanner 启动前已生成全部 pybind shim object。本版本仍保持一个 runtime 包与
  一个 DLL。
- 同一 package-private ABI manifest 现在也驱动 Linux ELF 导出隔离。最终 runtime version
  script 只保留 shim 实际 import 的 Taichi 符号与 ABI anchor；
  bundled LLVM、SPIR-V、UI、logging 和 allocator API 保持 local。shim 记录显式
  `DT_NEEDED` 和包相对 `RUNPATH`，loader 对 runtime 与可选 CUDART 使用 `RTLD_LOCAL`。
  发行验证覆盖 runtime-first 与 driver-first 两种碰撞顺序，同时不把 runtime 重复打入每个
  CPython wheel。若不兼容 Taichi ABI 已经被放入进程全局域，import-time 有界 private-ABI probe
  会 fail closed。
- split-runtime macOS 源码构建使用同一 schema、精确 Mach-O exported-symbols list 与包相对
  `@loader_path`。这只是源码构建合同；公开发行矩阵仍为 Windows 与 Linux。
- CUDA-enabled runtime 只使用过 CPU 或 Vulkan Graph state 时，Graph cache reset/析构不再
  构造 CUDA context；真正的 CUDA cache 继续保持 submission lock 顺序。0.6.1 split shim
  中的兼容 override 已迁移到最终的 native runtime 所有权位置。
- nested Graph execution statistics 统一消费扁平 backend mapping；异步 telemetry 以稳定
  region path 记录 recordable/native Sequential 内的 structured region。缺失或重复 region
  仍然 fail closed。
- 完整 SolvePlan Graph action 在 structured sequence 内联后仍保留 action-owned host telemetry。
  每次成功的外层 Graph submission 记一次；ticket poll/wait 会幂等记录 observed completion，
  不读取 terminal；显式 terminal snapshot 则恰好一次记录 packet 的最终逻辑迭代数。单 action
  与 plan aggregate 都不增加 kernel、queue submission、device byte 或隐式同步；nested region
  invocation/replay 数继续由 ticket telemetry 提供。
- recordable f32 compiled-Graph A/M provider 现在可在 CUDA/Vulkan 上执行 device-resident
  convergence control 的 fixed-linear PCG。compiled-Graph PCG 自动选择该路径；
  compiled-Graph CG 保留原默认策略并可显式 opt in。满足资格的连续 Field operand 直接绑定，
  不产生 SolvePlan pack/unpack submission。solver statistics 同时公开 logical、encoded 与
  masked iteration，使 exact 和 bounded backend 上的早停均可观察。
- recordable scale/sum composition 现在会把有序 subtree 规范化为 weighted leaf，并用单个
  in-place `axpby` 完成每次累加。该 lowering 保持 provider 顺序与 generation/lifetime
  检查，只复用一条 Graph-owned scratch vector，也不宣称跨 provider kernel fusion。在本地
  配对的 262,144 项、三 leaf 资格测试中，dispatch 从 8 降到 5，temporary 从 2 MiB 降到
  1 MiB；CUDA warm submit/wait 中位数从 270.2 降到 203.5 us，Vulkan 从 555.3 降到
  465.9 us。该数据只描述这一 workload，不是普遍加速承诺。
- 等长的纯 `compose()` chain 现在会按精确 forward/adjoint leaf 顺序扁平化，并在
  destination 与一条 Graph scratch vector 之间 ping-pong。depth 4/8 因而只使用一条
  N-vector，而不是三/七条。本地 262,144 项源码资格中，depth-8 warm median 在
  CPU/CUDA/Vulkan 上分别改善 12.6%/15.9%/11.5%；depth-2 保持在原有噪声范围。
  rectangular 与 mixed-extent chain 继续使用保守 nested lowering。
- CUDA/Vulkan 上 recordable f32 CG/PCG 的 `SolvePlan.submit()` 现在会把完整
  device-convergent 求解包装进缓存的单 action Graph，并返回一个 `SolvePlanSubmission`。
  terminal packet 在 `done()`/`wait()` 期间保持
  device-resident，只由 `result()` 物化一次；可选 workspace lane 提供独立 Krylov storage，
  statistics 会公开 variant/lane memory 与 terminal readback。该路径与等价手写 Graph 使用
  相同 persistent bytes。在本地配对的 262,144 项测试中，wrapper 相对该 Graph 的开销在
  CUDA 为 2.1%，Vulkan 为噪声范围内的 -1.2%；terminal result 的物化仍是显式的额外同步/
  readback。CPU 使用既有精确 native solve 并返回 completed lane-0 submission，避免 GPU
  风格 masked replay；它不开放 Graph terminal packet/Graph telemetry。
- 非平凡资格用三个 dispatch 表达 SPD stencil A、两个 dispatch 表达 Jacobi M。在 262,144
  项下两个 GPU backend 均于第 13 轮停止：CUDA encoded=13、无 masked tail；Vulkan
  encoded=64、masked=51。在最终 60-sample 源码资格中，warm managed submit/wait 中位数在
  CUDA/Vulkan 上分别为 1.856/4.107 ms；同一手写 Graph 为 1.825/4.053 ms，
  fused-provider 上界为 1.789/3.864 ms，kernel host-check-K4 对照为 4.042/4.727 ms。managed/manual Graph 均使用
  5,244,972 persistent bytes。该本地配对测量只验证 overhead 与停止遥测，不是普遍 solver
  加速承诺。
- 兼容的 values-only `LinearOperator` generation 现在会在 launch 时 rebind 到缓存的 Graph
  action 与 SolvePlan Graph。所有 composition leaf 先完成两阶段验证；topology/schema/state
  tree/runtime 变化仍 fail closed，每张异步 ticket 会 pin 本次实际提交的 immutable numeric
  owner。本地 262,144 项 update/run 资格中，cached rebind 相对重建 Graph 在 CPU 上为
  0.924/1.583 ms、CUDA 为 0.438/1.044 ms、Vulkan 为 0.587/1.094 ms。完成后 13 个 retired
  generation 均已释放，active lease 为 0。
- `BatchedSolvePlan` 在 A 与可选 fixed-linear M 都是 recordable f32 action 时，新增显式
  `device_convergent` CUDA/Vulkan 执行。一个 structured Graph 包含初始化、A/M、reduction、
  recurrence、逐系统 status 和全局 active predicate；`submit()` 返回一张 ticket，terminal
  materialization 会报告精确 logical stop iteration，迭代 loop 内不经 host readback。Vulkan
  继续使用 bounded encoded/masked tail 语义；既有 host-check 默认策略不变。Vulkan
  fixed-budget 使用 direct recurrence dispatch，因为 active submission batch 内的 nested
  replay synchronization 不属于已资格化操作。
- Graph provider argument 与 generation owner 现在会作为一个 immutable launch snapshot
  一次性准备。全部 composition leaf 会先完成验证，再发布任何 binding，因此 values-only
  update 不会把一个 generation 的 replacement argument 与另一 generation 的 lifetime owner
  混合。已经 setup 的 fixed-linear `PreconditionerPlan` 在 approved action 可录制时也可录制；
  每张 Graph ticket 会 pin 精确 approved target/action pair，stale 或未批准 generation 仍然
  fail closed。
- batched solver statistics 升级为 schema v5，每次 solve 只发布一个 packed terminal packet。
  opt-in per-ticket telemetry 报告 logical/executed/provider work、active efficiency、可用
  encoded/masked work、Graph launch、physical queue submission 与不做推测的 timing。惰性
  `workspace_pool()` 提供各自带 completion fence、带内存计账的 Graph/workspace lane，并用
  wait/raise 明确处理饱和；它不宣称独立 GPU stream 或物理重叠。
- CUDA device-convergent batched CG/PCG 可显式压缩活跃系统的 recurrence reduction 与 vector
  update，且不回读 host count。A/M provider apply 仍作用于完整 batch；capability 会如实报告
  capacity-grid masked prefix，而不是 exact indirect dispatch。该选项默认关闭，CPU/Vulkan
  不可用。本地 262,144 scalar、64 system 的交错资格测试中，异构 batch 从 19.855 降至
  18.044 ms（快 9.1%），全 hard 对照从 58.508 降至 56.679 ms（快 3.1%）；runtime、host
  pool 与 device pool memory 保持稳定，异构 case 的阈值边缘浮点终止最多相差一轮。这些数据
  只资格化该 workload，不是普遍加速承诺。
- 新增 `inverse_block_diagonal()`，接受调用方提供的 row-major f32 inverse block，block size
  为 1 到 4。每种尺寸现在使用专门化 kernel 与常数大小 topology word，不再为每个 scalar
  row 保存两个 offset；在 262,144 scalar 下，operator-owned topology snapshot 从约 2 MiB
  降至 4 bytes，配对 warm apply median 在各后端噪声范围内。它在 CPU/CUDA/Vulkan 上可录制，
  并复用普通 compiled provider 的 numeric rebind/pinning 合同。调用方必须显式断言 SPD；
  Forge 不回读、求逆、正则化或推断 block。
  本地 64 系统、262,144 scalar diagonal workload 中，精确 inverse 把约 72-100 次未充分
  预条件/sqrt-scaled iteration 降至 2 次。较难的 sqrt-scaled preconditioner 下，
  device-convergent PCG 相对 host-check-K4 在 CUDA/Vulkan 分别快 3.1%/17.3%；精确 inverse
  只有两轮时 host-check 反而更快，因此新 policy 保持 explicit-only，也说明 crossover 首先
  取决于预条件质量和收敛长度。
- 新增使用公开 `LinearOperator`、2x2 `inverse_block_diagonal`、`PreconditionerPlan`、compact
  Vector Field、numeric-generation update 与单个可复用 `SolvePlan` 的 headless 固定拓扑
  隐式弹簧参考。本地 2,304 节点资格中，logical iteration 从 CG 的 54 次降至 PCG 的 6 次；
  warm median 在 CPU 上从 37.061 降至 7.281 ms，在 CUDA 上从 3.455 降至 2.412 ms。
  Vulkan 为噪声范围内的 7.377/7.460 ms，因为 bounded 路径为 6 次有效迭代编码了 96 个
  slot（90 masked）。CUDA/Vulkan 的 rebind-plus-solve 明显低于 rebuild-plus-solve；三后端
  1,000 帧测试均只保留一张 GPU Graph，释放全部 1,004 个 retired generation，active lease
  为 0，停止位置保持在 4 到 6 次。该数据只资格化此参考并公开 Vulkan tail 成本，不是普遍
  solver 加速承诺。
- 新增 `LinearOperator.shifted(shift)`。recordable f32 GPU lowering 执行 base provider 后只
  增加一个 in-place `axpby`，不会发射第二个 identity provider，也不分配 identity-sized
  temporary。非方阵、非有限 shift 及不受支持的 dtype/backend 组合都会明确失败。
  本地 262,144 项配对 Graph 资格中，dispatch 从 3 降到 2，CUDA warm submit/wait 中位数
  从 0.298 降到 0.222 ms，Vulkan 从 0.568 降到 0.441 ms，数值误差为 0。

- `LinearOperator.apply(alpha=..., beta=..., addend=...)` 的通用形式现在会在 CUDA/Vulkan
  上使用 device-native f32 ndarray transform/scaled-add lowering。非 alias 路径不发生 host
  readback，也不分配 N-vector；addend/output 精确 alias 时复用一个持久 scratch。本地
  262,144 元素成对资格测试中，native apply 与等价双 dispatch Graph 分别为：CPU
  0.822/0.955 ms、CUDA 0.240/0.380 ms、Vulkan 0.401/0.422 ms。该同步边界结果只代表当前
  workload，不是通用加速承诺。
- 新增可原子更新的 `parameterized_affine()`。必须声明的系数闭区间限定保守 trait 推导；
  optimistic version 会拒绝 stale update，每次 submission pin 一个完整 alpha/beta generation。
  f32 系数与区间端点会在 trait 推导前规范化，无法表示的值会失败，舍入为零的正下界也不会
  保留无效 SPD 结论。缓存 Graph 从同一个原生 immutable snapshot 直接 patch 两个 scalar
  binding，不重建、不创建虚拟 provider，也不上传 device parameter array。本地 262,144
  元素 update-plus-run/rebuild-plus-run 分别为：CPU 0.996/3.534 ms、CUDA
  0.372/3.675 ms、Vulkan 0.410/4.531 ms。该同步结果只代表当前 workload；正确性测试还会
  在 update 与重叠 submission 之间保留并验证旧 snapshot。
- fixed-layout `block_diagonal()` standalone apply 现在支持合格的 CUDA/Vulkan f32 leaf；
  Program-bound CPU leaf 也使用相同 runtime-storage subrange 合同。连续 domain/range slice
  一次解析并按 leaf 顺序提交，不执行 gather/scatter、全向量 staging 或 N-sized temporary。
  leaf 可录制时，container 现在可作为一个 Graph/SolvePlan action；private derived subview
  保持双参数公开 ABI，在嵌套 composition 中仍为 zero-staging，并让一次外层 submission 在
  全部 solver iteration 间摊销绑定成本。该合同不做 kernel fusion：双 leaf、262,144 元素
  identity probe 在 CPU/CUDA/Vulkan 分别为
  0.792/0.264/0.458 ms，而单融合 kernel 为 0.388/0.135/0.335 ms。permutation、overlap、
  non-affine 形式仍不支持。同步单 action Graph 资格测试为 1.037/0.504/0.486 ms，standalone
  CPU/CUDA/Vulkan apply 为 0.815/0.367/0.563 ms；均为两个 dispatch，Graph planned temporary
  bytes 为零。该边界数据不是通用加速承诺；预期收益是一次外层 Graph ticket，且迭代间不做
  staging。
- 新增 `SmallBlockInverseBuilder`，面向 size 1-4 的固定 f32 row-major block。direct 与单 dispatch
  Graph 形式都在 device 上以相对 block 尺度的 pivot tolerance 执行带 partial pivot 的 Gauss-Jordan，把 success/non-finite/
  singular 的逐 block status 留在 device，并把失败输出清零且不推断 SPD。对 16,384 个 4x4
  identity block，本地 device build 与 host readback/NumPy inverse/upload 分别为：CPU
  0.824/5.175 ms、CUDA 0.410/6.195 ms、Vulkan 0.722/7.667 ms。三后端 1,000 次复用压力
  测试保持调用方提供的 output/status allocation 不变。implicit spring 参考路径现在会在
  device 上完成 coefficient 组装、inverse 构造和 status 保留，再发布 preconditioner
  generation。本地 2,304 节点、30 步端到端资格中，device refresh 与原 host 组装路径分别为：
  CPU 9.282/8.966 ms、CUDA 6.240/6.527 ms、Vulkan 10.017/13.130 ms；三端逻辑 PCG
  迭代均为 6，失败 block 为 0。CPU 只保留一个较小的实测开销，GPU 则消除了 host
  round trip；该时间只代表当前 workload。

## 0.6.1

- `0.6.1` 的发布来源有意分开记录：`b129ad94c` 是最终 Python shim/source 边界，已经冻结的
  runtime wheel 则报告 native build identity `c268ca5671e8`。两者 distribution version
  保持一致；经过兼容性验证的 shim-only 修复不要求源码 commit 相等。
- 最终 split-wheel shim 在释放 CPU/Vulkan Graph cache 时不再初始化 CUDA driver；只有 cache
  确实持有 CUDA Graph state 时才取得 CUDA submission lock。该修复保持 CUDA 锁顺序，不给
  Graph 执行热路径增加工作，也不改变已经发布的 native runtime wheel，并修复 Windows/Linux
  无 CUDA driver 主机上的 installed-wheel 验证。
- LLVM CPU/CUDA 的 SNode metadata 现在通过按几何级增长的 Program 级 tree directory
  与逐树精确尺寸 runtime-state block 寻址；旧的全局固定 SNode/tree 表不再构成 runtime
  容量上限，但 allocation overflow 与 stale tree generation 仍会 fail closed。逐树诊断会
  单列 runtime-state component 而不重复计入，内部 Program 诊断还会报告 directory
  capacity、active tree、reserved bytes 与 growth events。资格测试覆盖 CPU/CUDA 上
  4,098-node 的 dense/pointer/dynamic/hash 混合树、超过 1,024 的全局 id、513 棵同时
  存活的树、销毁与 generation-safe slot 复用。当前实现的 scaling benchmark 中，
  4,099-node tree 相对 3-node tree 的 lookup 中位数在 CPU/CUDA 上分别为 1.011x/0.919x；
  513-tree 阶段把 directory 从 16 项增长到 1,024 项（8 KiB），销毁后 active-tree count
  恢复。这些是 scaling/ownership 结论，不是相对旧二进制的历史加速比。AMDGPU 采用同一
  LLVM representation，但尚未取得资格；Vulkan 使用独立 sparse runtime。CUDA 无返回值
  kernel 现在会在注册时验证 live tree，随后获得不可变的紧凑 root binding：单 tree 直接携带
  root pointer，不产生 binding allocation；多 tree 每个依赖只占 8 字节，而且即使不同 offload
  访问互不相交的 tree，也统一使用完整 kernel 的稳定映射。生成的 root load 位于 grid-stride
  loop 外，Graph replay 不执行 directory lookup、host readback、分配或同步。CPU 与有返回值的
  LLVM kernel 继续使用通用 directory accessor；directory 边界、generation 与 lifetime 检查
  仍保留在注册和 launch 边界。同一台 Windows RTX 5090 资格机器上的配对 synthetic HVP
  probe 中，candidate 与公开 0.6.0 在 Nsight Compute 下分别执行 5,504/5,632 条 SASS 指令、
  耗时 1.66/2.144 us，Graph replay 中位数为 11.122/11.255 us。这证明热点路径回退已被移除，
  不代表所有应用都能获得额外加速。23 项定向生命周期与高 id 资格覆盖双 tree/双 offload
  映射、销毁、无关 tree 退休、tree-id 复用、4,098-node tree 与 513 棵同时存活的 tree，且未
  放宽 fail-closed 行为。
- Windows CPU JIT session 现在会把每个 LLVM RuntimeDyld COFF object 放入同一个按页对齐的
  `Code -> read-only -> read-write` 映射中，满足 `ADDR32NB` 对 image-relative 顺序、32-bit
  span 以及大于系统页的 section alignment 要求；这替代了此前仅保留全部 object section、
  但不足以约束实际分配地址的策略。该修改避免反复 reset runtime 或交替初始化 CPU/GPU 后端时偶发
  `IMAGE_REL_AMD64_ADDR32NB` 有序布局错误。该修改仅作用于 Windows COFF JIT 初始化，不给
  CUDA/Vulkan Graph replay 增加同步；修复后混合后端 bounded Graph 生命周期整套测试连续五轮
  自然退出，环境敏感的 CUDA driver setup probe 在隔离运行时也通过。
- CPU 的 device-known bounded Graph dispatch 现在默认选择 exact scheduler。scheduler
  只读取并钳制一次 extent，零工作量直接跳过；正数 workload 以连续 JIT loop 的形式按自适应
  chunk 执行，不再逐元素调用 callback。这使 CPU grain 与 GPU
  `block_dim` 解耦，并恢复 LLVM loop vectorization。强制 `masked_capacity` route 继续作为
  fallback 与 A/B 诊断入口。lowering 复用已有 CPU bounded binding 与 runtime scheduler
  symbol，不新增 split-runtime ABI 或 symbol 要求。在 Windows 资格机器上，262,144 项、每项 16 次非平凡操作的
  payload 相对同轮 fixed masked Graph，在 zero/10%/full count 下的 p50 比值分别为
  6.55x/2.78x/0.997x，p95 比值为 6.50x/2.67x/0.999x。正确性覆盖钳制、overflow、TLS
  reduction、`continue` 与两个相互独立的并发 Graph caller；1,000 次交替 replay 中
  runtime、host pool 与 device pool ownership 均保持稳定。
- CUDA device-known bounded dispatch 现在在所有受支持 driver 上都有 exact logical-range
  lowering。默认路线在 device 端读取并钳制 `DeviceExtent`，沿用普通的 saturation-capped
  grid-stride launch，因此既不需要 host readback，也不依赖 CUDA 12.4 node update。在通过
  资格的 12.4+ driver 上，强制 `device_update` 仍可作为物理优化：updated grid 会钳制到同一个
  saturation grid，并可跳过 zero-count payload；正确性仍由 logical range 保证。capability
  schema v4 会分别报告 logical exactness 与 physical launch kind；`masked_capacity` 保留为显式
  A/B 基线。同一 runtime 的成对测试覆盖 zero、block 边界、overflow、ndarray rebind、当时的
  带 label 普通诊断基线、两 block saturation cap、并发 replay 和 1,000-2,000 次 replay memory
  stress。
  在这台 Windows CUDA 机器上，4,194,304 项容量的 full count 与 masked 相差不到 0.4%，10%
  count 快 2.2%；16,777,216 项容量下 zero/1%/full count 分别快 4.9%/4.0%/1.2%。adaptive
  route 存在依 workload 而变的 updater crossover，因此不作为默认路线。最终成对 wheel 的
  forced-route 测试采用 4,194,304 capacity 与每项 16 次 payload 操作；zero、10%、full count
  下 masked/exact median 分别为 34.026/34.428 us、34.328/33.384 us、34.782/38.144 us。
  各路线都通过正确性，明确报告不同的物理 count，并在 1,000 replay 中保持 runtime、host
  pool 与 device pool ownership 稳定。这个接近且依 workload 变化的 crossover，正是保留旧
  driver masked route 作为合格 fallback 的原因。
- 可选的 CUDA 12.4+ adaptive 物理路线现在会把两个或更多连续、且
  extent/capacity/block 合同相同的 bounded payload 交给一个 stateful updater。grid/enabled
  未变化时复用持久状态，单个 payload 保持逐节点路线。在 64 payload、16,777,216 capacity、
  每项一次操作的重复资格测试中，grouped/stateful 在 zero/1%/full count 相比逐节点 control
  约快 1.3x/1.9x/1.04x；一次低离散度 full-count 复测为 5056 us 对 5420 us。persistent
  grouped control 现在还包含 opt-in 的 device-side replay、状态变化、cache hit 与实际 node API
  调用计数，由 `Graph.execution_stats()` 暴露且不增加 replay-time host readback。64 payload 的
  persistent bounded control 为 592 B，逐节点为 2,048 B；1,000 次交替 replay 中 ownership 保持稳定。这些结果只限定本次
  RTX 5090 上的策略 crossover，不会把 adaptive route 提升为通用 CUDA 默认路线。
- CUDA structured control 新增 Forge 自有的 bounded masked Graph 路径，覆盖低于 12.8 的
  driver。满足资格的 Driver API 12.8+ runtime 继续使用原生 conditional Graph node；较旧
  runtime 在普通 CUDA Graph capture 可用时使用 device latch 与 task-entry gate，否则保留
  exact portable control。capability 会明确区分 exact native、bounded masked 与 portable
  执行，不把它们伪装成相同的物理 launch。在当前 driver 上强制 masked route，可以对
  Forge 自有 fallback 的语义和性能完成资格验证，不再要求为此保留旧硬件；但这不等于验证
  某一个旧 driver 的 loader 或厂商实现。成对的 0.6.1 wheel 已通过强制 while、`if`、
  `switch` 合同。在 262,144 item、16 次迭代、1,000 replay 的 workload 上，强制 masked
  Graph median 为 366.9 us，原生 conditional Graph 为 465.3 us，portable control 为
  1,410.9 us，三条路线都准确停在第 16 次。这是特定 workload 的 crossover，不表示 masked
  control 通常快于原生 CUDA control。
- 新增严格限定的 depth-2 `while -> ordered while[1..8]` 单 ticket 执行。outer body
  可以在各 leaf inner loop 之间放置普通 action；inner control 必须互不别名，完整层级仍
  最多编码 4,096 个 action。CPU 使用精确 host control 并返回 completed ticket；Vulkan
  使用 bounded conditional replay；CUDA 使用 bounded
  静态拓扑，且两层之间不做 host readback。通过资格的 Driver API 12.4+ CUDA runtime 会
  先执行缓存的 setup probe，再使用 device-updatable kernel-node group，每个业务 dispatch
  只编译一次；较旧或未通过资格的 runtime 使用 Forge 自有、与版本无关的双 gate
  task-entry masking。可以设置 `TI_GRAPH_CUDA_FORCE_MASKED_CONTROL=1` 在当前 driver 上
  资格化 fallback。两层上限都不超过 64，完整程序最多编码 4096 个 action。capability 会
  分别报告 selected/candidate/fallback route，并明确保持 exact dynamic command
  termination 为 false。outer suffix 可以在 device 上保留每次 inner 的停止位置；
  recordable provider 可以在 ticket 完成后公开 terminal packet。
  本地 Windows 资格 workload 使用 4,096 item、8x16 budget、4 次 active outer、停止位置
  6/7/8/6、5 次 warmup 与 30 次计时 replay。两个新进程中的 CUDA node-update median 为
  652.55/640.95 us，两个强制 fallback median 为 742.45/745.15 us；process median 再取
  median 后 warm 延迟降低 13.0%。同一 nested Graph 在 Vulkan 上为 1,592.85 us，乐观的
  host-known direct-call oracle 为 9,392.0 us，compact root-level Graph 为 1,929.7 us。
  CPU 为 12,260.3 us，优于 direct 的 17,207.95 us，但仍比 11,171.55 us compact oracle
  慢 9.7%。persistent argument storage 分别为 CUDA node update 14,808 B（其中 control
  14,272 B）、强制 CUDA masking 648 B、Vulkan 1,304 B、CPU 0 B。CUDA cold invocation
  约为 104 ms，fallback 约为 101 ms，因此高内存路径是 warm replay 优化，不是 cold-start
  提速声明。这些数字只限定本机和当前 workload，不代表所有 backend 的无条件收益。
  另一组 ordered-two-inner 资格 workload 使用 4 个 active outer step、4,096 item、第一
  inner 停在 6/7/8/9、第二 inner 停在 2。单 ticket 在 CUDA/Vulkan 上分别为
  796.15/2,045.45 us；乐观的 host-known outer loop 仍需等待每个 adaptive inner Graph，
  分别为 4,463.7/6,522.1 us，即快 5.61x/3.19x。
- Graph submission telemetry schema v5 现在分别报告 logical Graph/region invocation、
  backend Graph launch、CUDA stream enqueue 与物理 queue submission。CUDA 会明确把物理
  queue count 标记为不可用；Vulkan 报告 device transaction-window delta，并标记为非精确。
  ticket 持有的 nested telemetry 会保留每个 inner 的停止位置，并以
  `logical_invocations` 区分 child 被重复调用的次数与最终 iteration count。该 opt-in 路径
  不给普通 submission 增加 host readback。
- 含 exclusive Graph-owned solver storage 的 Graph 现在可以配置一至 64 个 workspace
  lane。lane 按需惰性物化，自动 round-robin 优先选择已完成的 lane，也可以逐 submission
  固定；全部占用时按策略等待或立即失败。它消除了多个已排队 solve 之间的 completion-fence
  依赖，同时保持 terminal 与 workspace 独立；不会创建 backend stream，也不承诺 GPU
  重叠执行。memory report 会报告 lane capacity、materialized/busy 数量、wait、saturation
  failure 与聚合 persistent bytes；每个额外物化 lane 都线性增加 workspace 成本。本地
  Windows 的正反顺序诊断中，双 lane 在 4,194,304-item CUDA 上把两次 solve 总完成时间
  降低 11.6-22.5%，在 262,144-item Vulkan 上降低 6.1-7.9%。internal storage 在 CUDA
  上严格从 83,918,892 增至 167,837,784 B，在 Vulkan 上从 5,244,972 增至
  10,489,944 B。这些非 strict-idle 数字只资格化 queueing 机制，不代表通用 solver
  throughput 或 GPU 物理并行。
- 新增只读 offloaded-task manifest 与 JIT dispatch label。manifest 在不发起 profiler probe
  的前提下报告稳定 task identity、`cpu_scheduler`/`grid_stride`/
  `device_bounded_grid_stride`/`one_to_one`/`not_applicable` range mapping、
  requested/selected/actual grid/block geometry 和静态
  shared-memory 上下文。CPU 不填充 GPU geometry；runtime-indirect Vulkan workload 明确
  报告 actual geometry 由 device 决定。label 在 profiler 与可选 NVTX 名称中保持同一个
  task identity；manifest 查询不提交工作，且内存占用保持稳定。
- 新增 `TaskLaunchPolicy`，用于受约束的 direct-JIT block 调优。CUDA/Vulkan 对单一安全
  parallel range task 支持 `hint`/`require`，通过 immutable report 暴露最终 geometry 与
  编译期资源约束，并在 enqueue 前拒绝不支持的 task shape 或 block-sensitive 改写。CPU
  hint 明确保留 worker scheduler，require 明确失败。policy 不覆盖 grid extent，作为独立
  cache specialization，并在只读验证后复用普通 warm launch 路径。无提交查询不能安全取得
  register/local-memory 时会明确保留为空；不会引入 autotuning 或 profiler launch。worker
  thread 使用前应在 Python 主线程通过 `report()` 准备冷 GPU policy，并保留 `auto` 作为性能
  对照。
- opt-in Graph submission telemetry 现在包含由 ticket 持有、immutable 的
  `GraphPipelineReport`，描述优化后的 execution root。每个 stage 会报告逻辑/物理
  dispatch 数、runtime 参数名、native-action 组成、声明的 temporary 字节，以及已有的
  structured-region GPU timestamp；普通 CGraph stage 还会公开物理 `GraphTaskManifest`。
  pipeline schema v2 会把
  带 label 的 bounded dispatch 与 task identity、selected/actual launch geometry、count
  source、capacity、block size、selected route 和 ticket-owned useful/executed/encoded work
  关联起来。device-known count 对每个 distinct extent 只增加一个去重后的双字 tail
  snapshot；host-known count 不增加 device buffer。ordered segment 只报告可靠的 aggregate
  extent；没有 offsets snapshot 时不会伪造逐 segment useful work。普通 CGraph stage 把
  mapping 标记为 `available`；structured while/if/switch stage 则明确标记为
  `structured_runtime_dependent`，不会虚构扁平物理 task 序列。`NativeActionManifest` 会冻结
  provider 的符号 binding、
  effect、temporary requirement 与 recordability/backend 合同，但不暴露 storage 对象或
  device 地址。只有 whole-ticket timestamp 时，普通 stage 不会虚构 per-stage duration；
  默认 `telemetry=False` 路径不会物化该 report 或 telemetry arena。在 Windows RTX 5090/CPU
  资格机上，4,097/65,536 bounded payload 只保留一个 8-byte slot；相对
  `submit().wait()`，完整可选 report 的观测成本约为 CPU 0.529 ms、CUDA 0.350 ms、Vulkan
  0.510 ms。这是采样/诊断成本，不应理解为可以每步持续免费开启。
- CUDA driver-only stable radix sort 现在直接在每个 scatter block 中导出 16 个 digit base，
  删除独立 digit-base kernel 与 workspace。在 RTX 5090 上对 1,048,576 项随机 key 做匹配的
  full-pipeline A/B 时，合格候选的 median 从 508.11 us 降至 454.44 us（11.8%），p95 从
  562.41 us 降至 498.49 us，同时保持重复 key 的稳定顺序与 replay memory 有界。这组本地
  资格数据只刻画该设备和 workload，不是对所有场景的无条件提速承诺。
- CUDA driver-only stable radix sort 在当前 histogram level 已能由一个 1024-item scan
  tile 完成时就终止 hierarchy。对于首层包含 1024 个 block 的 32-bit sort，每次排序会删除
  8 次冗余 scan launch 和 8 次不执行有效 uniform-add 的 launch（device kernel launch 总数
  `53 -> 37`），workspace 也不再保留未使用的 one-element parent。在同一 RTX 5090/610.62
  系统上，公开 0.6.0 wheel 与 release-candidate 0.6.1 成对测试分别为每只 wheel 启动三个
  新进程，每个进程 10 次 warmup、100 次逐次同步调用。1,048,576 个 `i32` item 的 process
  median 再取 median 后从 0.51245 ms 降到 0.36455 ms，延迟降低 28.9%（吞吐 1.41x）；
  报告的 peak workspace 还减少 512 B 至 29,425,664 B。安装 wheel 后的 13 个 CUDA
  key/payload dtype 与大 hierarchy 稳定性用例全部通过。这仍是设备/workload 资格数据，
  不是新的 CUB 等速或通用提速声明。
- 新增 Graph-independent 的固定容量 `DeviceWorklist`：它持有稳定 front/back storage、
  device-owned `DeviceExtent`、atomic append、stable selection 和确定性整数 key 冲突消解。
  无 overflow 时每个接受项只需要一次 slot-reservation atomic；append 顺序不保证，每次
  transition 只允许一个 producer owner，消费前必须由 `commit_next()` 或已记录 finalize node
  发布 extent 与计数。`DeviceWorklistSequence` 可把 reset、finalize、selection 或 keyed claim
  记录成可复用 Graph native action。Graph 参数可把 generated/accepted/rejected/conflict/
  winner/overflow 计数附着到 `SubmissionTicket`，steady-state replay 不读取 host count、也不
  重新分配；首次执行仍可能准备 native provider workspace。
  相邻的 Vulkan finalize 与 bounded consumer 会自动发布到一个 Graph-owned exact indirect
  packet，无需公共 launch-state 对象或 preparation dispatch；连续匹配 consumer 会复用该
  packet。CPU 默认使用 exact adaptive scheduler；CUDA 不消费 Vulkan packet，使用 exact
  logical range，并可在通过资格的 12.4+ driver 上进一步缩小物理 grid。
  确定性 keyed claim 存在明确 workload crossover：262,144 个 active item 时相对完整 host
  round trip 在 CUDA/Vulkan 上分别为 8.63x/9.05x；稀疏 1,638 项在三后端上都更慢。
  资格基准会分开报告这两类输入，并在 1,000 次 CPU、3,000 次 CUDA/Vulkan replay 中观察到
  memory 稳定。
- 新增 `DeviceDispatchState` 与 `DevicePrefixSequence`，用于 fixed-topology、
  device-count-driven pipeline。Vulkan compact 可把 bounded dispatch packet 与输出 count
  一起发布，再交给 `dispatch_bounded(launch_state=...)`，从而删除 consumer preparation
  dispatch；CPU 默认使用 exact adaptive scheduler；CUDA 独立使用 exact logical range，
  并可选择 12.4+ adaptive physical control。统一的
  `dynamic_work_capabilities()` 会分别报告物理 launch、structured iteration termination 与
  completion observation。
- Graph 终态 observation 默认附着到 completion。Vulkan/CPU 使用 host-visible arena slot；
  CUDA 在 ticket completion 前把 device-local snapshot 异步拷贝到持久 pinned host memory。
  `ticket.observations()` 不再触发第二次 readback，同时避免 managed CUDA storage 的页迁移。
  旧 deferred 路径保留为诊断回退。
- 新增 `DevicePrefix` 与 `DevicePrefixWorkspace`，通过共享、device 写入的
  `DeviceExtent` 组合 compact、scan、reduce、sort、consecutive unique/RLE、grouped
  reduce 与 bucket building。固定容量 provider 与可复用 workspace 合同保持可见；wrapper
  消除操作间的 count readback，但不宣称每个 primitive 都按 active count 执行。10% active
  prefix 的 compact-to-scan 资格测试相对显式 host observation，在 CPU/CUDA/Vulkan 上分别为
  1.05x/1.32x/1.90x。
- 新增 `GraphBuilder.dispatch_bounded()`，支持 host-known exact range 与 device-known
  bounded work；新增 `dispatch_ordered_segments()`，用同一个 payload specialization 执行
  具有全局顺序的 offset range。Vulkan 使用 device-written indirect packet 与编译器证明的
  one-to-one range mapping。CPU 的普通 bounded dispatch 使用 exact adaptive scheduler，
  ordered segmented CPU dispatch 则保留全局有序的 masked route；CUDA 会分别如实报告 logical
  exactness 与 saturation-capped static 或 12.4+ adaptive physical launch。extent、capacity、
  block dimension 相同的连续 standalone Vulkan consumer 现在共享一个已准备的 12-byte
  packet；任何中间 action 都会保守失效。在 64 consumer、4,194,304 capacity、每项一次操作的
  资格测试中，packet 复用把 zero/1%/full median 从 3.14/3.14/3.17 ms 降到
  1.68/1.70/1.72 ms，packet storage 从 768 B 降到 12 B；bounded/fixed median ratio 从约
  0.53-0.54x 恢复到 0.97-0.98x，1,000 次 bounded-slot replay 中 ownership 保持稳定。
  固定 8-slot Vulkan replay ring 现在会在 indirect Graph 的在途提交超过可用 slot 时施加
  有界背压。该 Graph 无法通过 ordinary-launch fallback 保留 device-written dispatch packet，
  因而饱和时等待最旧 slot，而不是失败或扩张 replay 内存；未饱和路径与普通 Graph 路径不变。
  capability 与显式
  snapshot 可观察 overflow、useful/executed/skipped/encoded work、非法 offset、workspace 与
  zero-command 行为。通过资格的 recorded producer 现在可直接发布 Graph-owned Vulkan
  packet；中间插入其他 action 会恢复保守 prepare 路线。Vulkan exact 工作量减少不被表述为
  无条件提速：轻量 standalone payload 中，preparation dispatch 的成本可能更高。
- 新增 `DeviceExtent`：以稳定的两槽 device state 保存有界 count 与 sticky overflow。
  device-side publish 无 host readback 地完成钳制；同一 allocation 可由普通 kernel、JIT Graph
  参数和兼容的 count-producing primitive 共享。reset/normalize 保持 device-side，显式
  snapshot/check 才同步，旧 runtime generation 会 fail closed。该状态合同本身不宣称
  exact indirect dispatch，也不改变 kernel grid。

## 0.6.0

`0.6.0` 汇总已发布 `0.5.0` runtime 源码边界之后的更新，不会追溯改写
`0.5.0` 发行产物的行为归属：

### 从 0.5.0 升级概览

| 范围 | 0.6.0 相对 0.5.0 的主要变化 |
| --- | --- |
| Graph 与执行 | fixed-schema `while`/`if`/`switch`、最大 depth=2 的结构化组合、CUDA conditional Graph、Vulkan bounded/compound/nested while、Vulkan device-written indirect dispatch，以及 stop position、region 与 queue 遥测。 |
| 线性代数与稀疏 runtime | 公开 runtime-bound `LinearOperator`、实验性 `SolvePlan`/batch plan、fixed sparse pattern/value update，以及经文档矩阵限定的 CG/PCG、MINRES、BiCGSTAB、GMRES、FGMRES CPU/CUDA/Vulkan provider。 |
| 数据、互操作与显示 | dense storage/view 统一合同、受管 DLPack/external allocation、CUDA-Vulkan shared display、边缘根区域、连续字体缩放和可折叠自动高度面板。 |
| Native primitive 与打包 | CUDA 标准 wheel 改用 Forge 自有 driver-only primitive provider；Program-owned workspace、诊断、稳定 radix/compact/scan 改进，以及 runtime/shim build identity 门禁。 |
| 正确性与生命周期 | `SharedArray` block ownership、Tensor/AD/SVD 与 dense-field 对齐修复、crash-safe offline-cache lock，以及 allocator、specialization、trace、SNode/reset 资源的有界生命周期。 |

升级现有 0.5.0 应用时，重点检查：

- 本地或离线安装应使用 distribution version 相同的 runtime/shim 组合。split-wheel workflow
  显式选择兼容 runtime 时，两者源码 commit 可以不同；是否可用由最终 link、import、
  dependency 与功能验证决定，而不是要求 commit 相等；
- CUDA primitive 代码应使用 `method="auto"`，不要依赖只存在于不发布 reference build 的
  `cuda_cub*` provider；
- `ti.simt.block.SharedArray` 必须在 parallel range-for 的 block 作用域内声明；CUDA 单 block
  总量上限为 48 KiB，超出时明确失败，不会自动启用 dynamic shared memory；
- 结构化 Graph 与 `dispatch_indirect()` 应先查询 capability。Vulkan indirect dispatch 当前是
  单 offloaded-task 能力；CPU/CUDA 不会静默模拟该路径；
- `ti.reset()` 后应重建 Graph、storage view、external owner 与 solver plan，不能复用旧 generation；
- 旧 `from_dlpack()` 与 provider-specific Vulkan-CUDA import 名称保持兼容；新代码可使用统一的
  `from_external()` / `import_external_allocation()` 入口。

- Offline-cache metadata lock 改为由打开文件句柄持有的操作系统 advisory lock。进程
  终止会自动释放所有权，因此持久 `.lock` 文件不再导致反复的 load/dump 警告，也不再
  要求删除已编译 cache 状态。

- 新增 `ti.experimental.ndarray_view(source, slices=...)`，可在 CPU、CUDA、Vulkan
  上把经过资格验证、由 runtime 持有的 dense storage 严格 zero-copy 地绑定到
  `ti.types.ndarray(...)` kernel 参数。支持 contiguous Ndarray、合格的 dense
  scalar/vector/matrix field，以及保持 rank 的正步长 subview。组合 view 只合并经过
  checked arithmetic 验证的 byte offset 与 per-axis stride，不 staging，也不分配临时
  storage。负 stride、broadcast、overlap、permutation、sparse 与 external-owned layout
  会在 enqueue 前失败。stale owner 会被拒绝，GPU submission 会把 runtime resource 保留
  到执行完成。
- 新增 `ti.interop.from_dlpack()` 与 `ExternalDenseView`，提供严格、受管的 zero-copy
  import。CPU/CUDA-host producer 可在 CPU 绑定；CUDA/CUDA-managed producer 可在 CUDA
  绑定。runtime 持有 capsule deleter，每次 submission 前验证 byte range 与 owner
  generation，让 in-flight work 完成后再 retire，并保证 runtime reset 后 `close()` 安全。
  Vulkan、跨设备 import、不支持的 layout 与 copy fallback 会明确失败。
- 受管 external submission 使用 synchronization-domain access epoch。普通 launch 或
  Graph submission 对每个不同 domain 只 acquire 一次，并在 enqueue 完成或异常时逆序
  release。既有 NumPy、PyTorch、Paddle 参数签名保持兼容；同步 CPU NumPy 保留低开销
  direct ABI 与既有 incompatible-layout fallback。
- 新增 provider-neutral `ti.interop.from_external()` 与
  `import_external_allocation()`。既有 `from_dlpack()` 名称保持兼容，并进入同一套
  managed owner/view 协议。首个 raw provider 把 Vulkan 导出的 dedicated memory 与
  paired binary semaphore 导入 CUDA，开放多个 compact typed-offset view，并把它们
  合并为一次 Graph access epoch；device、handle、layout、lifetime 或 synchronization
  不匹配都会 fail closed。provider-specific `import_vulkan_cuda_allocation()` 名称继续
  保留。
- 公开 Vulkan-CUDA provider 与 GGUI shared-display path 现在共用同一个 checked
  raw-handle import core。与旧内部 importer 相比，GPU resource topology 和实测
  per-process GPU-memory 峰值不变，Windows 并发显示 timing 无回退；非法/重复 handle
  及部分构造或 cleanup 失败均可安全处理，且无需全局 CUDA submission lock。
- GGUI `canvas.set_image()` 会自动把合格的 CUDA field/ndarray 图像 pack 到 Vulkan-owned
  exportable storage，再由 CUDA 导入同一 allocation。External semaphore 构成有界的
  CUDA-produce/Vulkan-consume cycle；steady state 复用正常 render submission，不经过
  host，也不执行同帧 cross-device copy。capability/device 不匹配时自动使用既有 staging。
  `Window.get_display_stats()` 可报告实际 zero-copy render submission。资格验证后的
  Windows 2048 x 2048 workload 中，完整 warm frame loop 提升 6.2%，输出逐字节一致。
- 并发 CUDA production 与 Vulkan presentation 在复用 superseded shared-display frame
  前会重新完成 rearm，关闭间歇性的
  `Shared display storage is not available for CUDA` 失败，同时不引入会令相关引擎
  workload 损失 4.5%-8.8% 的全局 CUDA submission lock。
- GGUI 通过 `Gui.set_font_scale()` 与
  `Gui.set_font_scale_from_window_height()` 提供固定字体倍率和连续逻辑高度跟随。
  Vulkan 与 Metal 共用同一线性策略；每个 frame boundary 直接使用既有逻辑显示高度，
  不发生 GPU 回读，也不重建 font atlas。
- GGUI 同时开放带上下限的逻辑像素字体大小、自动高度 subwindow 和可独立折叠的分区。
  响应式控制面板会刚好容纳当前可见文字与 widget，并在分区切换时自动展开或收缩，
  不需要应用层计算高度，也不增加 GPU submission。
- GGUI Window 新增可按用户选择独立启用的 top、bottom、left、right 根区域，围绕
  中央 render viewport 布局。region resize/collapse、Vulkan/Metal viewport 与
  scissor、scene aspect、viewport-local input 和全屏 image 共用同一份 logical/
  framebuffer 布局快照；不增加中间 render target 或 copy。响应式字体策略现在可
  叠加 per-window 用户缩放，并在 edge region 内支持 Ctrl+wheel、Ctrl++/- 与
  Ctrl+0，过程中不重建 font atlas。
- `ti.simt.block.SharedArray` 现在使用 fail-closed 的 block ownership 合同。在并行
  Taichi range-for 内声明（包括从循环内调用的内联 `@ti.func`）仍走既有单 task
  快速路径；kernel root 与 serialized loop 内的声明会在 offload 拆分将其错误提升为
  kernel-global temporary 之前，由 JIT、AOT 与 Graph 编译一致拒绝。CUDA 与 Vulkan
  有 runtime 回归覆盖；本次更新不据此新增其他 GPU 后端的资格声明。CUDA 的静态
  `SharedArray` 单 block 总量限制为 48 KiB；更大的请求会给出明确错误，Forge 不启用
  opt-in dynamic shared memory。
- JIT Graph 的 `ArgKind.NDARRAY` runtime 参数现在通过通用 runtime-storage 协议消费
  Ndarray、dense field 与显式 `DenseNdarrayView`。compact Program-owned Ndarray 与
  SNode payload binding 可使用 CUDA capture、exact replay 和兼容 allocation patch；
  replay 前会重新验证 owner generation 与 byte range。positive affine view 在 CUDA
  使用 ordinary fallback，在 Vulkan 使用 command record/replay，并保持相同结果合同。
  受管 external owner 使用 ordinary/replay access epoch，而不进入 CUDA capture。AOT
  borrowed storage 与 ArgPack 嵌套仍不支持。
- 新增 `GraphBuilder.dispatch_indirect()` 与 `Sequential.dispatch_indirect()`。Vulkan Graph
  replay 可从 device-written 三元素 u32 packet 直接执行 `vkCmdDispatchIndirect`，零 group
  可跳过 payload，packet allocation 变化时会安全重录。目标 kernel 必须只产生一个
  offloaded task；packet 当前必须是 owning Taichi ndarray。Field、external storage、AOT
  packet 以及 CPU/CUDA 执行会明确失败，不会伪装为固定大小或 exact indirect dispatch。
- 新增 fixed-schema 结构化 Graph 控制：`GraphBuilder.while_loop()`、
  `if_then_else()` 与 `switch()`。condition kernel 可组合 tolerance、用户取消、active
  状态与 breakdown，不调用 Python callback。continue predicate 与用户定义 terminal
  status 保持独立；`Graph.control_flow_stats()` 报告 lowering、逻辑/执行迭代、状态轨迹、
  观测流量与 fallback 原因。满足资格的 CUDA `while`、`if` 与 `switch` 使用原生
  conditional node；`native_required` region 可通过 `Graph.submit()` 异步提交，且不做
  host 控制回读。conditional metadata 异步上传，并最多保留两个 deferred replay batch。
  CPU 保留精确 portable 控制。`Sequential` 现在也公开相同的 structured builder，
  允许再嵌套一级，最大 structured depth 为 2。CPU 精确执行两层。在 depth=2 时 parent
  使用 exact portable control；满足资格的 `auto` leaf 可保留 flat native route：
  CUDA `while`/`if`/`switch` 或 Vulkan `while`。这是默认的 portable-parent/native-leaf
  路径，不代表通用的原生 depth=2 合同；满足严格资格的 Vulkan while-to-while
  `auto` 定义还可升级为下述单次 bounded replay。nested `native_required` 定义会明确失败。
  Vulkan 同时支持精确 portable
  控制与满足资格的有界
  `native_required` `while`：`chunk_size` 按 region 生效并封顶为 64，每个 region 最多
  八个 chunk/512 轮；多个有序 region 可作为 compound asynchronous submission 通过一个
  terminal ticket 提交。每个 region 可以选择 compact 或 coarse-gated 首 chunk；
  Vulkan 自动 lowering 在 active chunk 内使用 compact masking，并用 coarse conditional
  rendering gate 跳过后续 chunk。runtime transaction 把其中的 command buffer 合并为一次
  queue batch，同时保持 semaphore 顺序和有界 replay-slot 退役。显式
  `submit(telemetry=True)` 会记录逐 region 的进入与终态 snapshot，并在 ticket 完成后报告
  真实停止轮次、encoded/masked 工作、active/skipped chunk、enqueue 时间和经过说明的
  queue-counter 窗口；默认提交路径不分配 telemetry buffer，也不增加 snapshot kernel。
  满足资格的 Vulkan while-to-while 定义可在 conditional rendering 可用、两层上限
  各自不超过 64、完整程序最多编码 4096 个 action，且两层使用独立单元素 i32 control 时，
  把两层编码为一次 bounded replay；其他 nested 结构使用 exact portable-parent
  control，满足资格的 leaf `while` 仍可保留上述 flat Vulkan route。
  `Graph.run(trace=True)` 使用 portable-parent exact 执行并返回每次 nested invocation；
  `GraphWhileReport` 提供 nested path 与 logical/encoded stop position。nested
  structured Graph 仍不支持异步提交。Vulkan structured replay 只能在 queue submission
  之前 fallback；提交后的 completion 或终态观测失败会直接报错，不会再次执行有副作用的
  body。Vulkan `if`/`switch` 与 exact dynamic command termination 仍不支持，并由
  `structured_control_capabilities()` 独立报告。
- 新增 `LinearOperator.graph_action()`，可把 compiled-kernel f32 provider 直接录入
  Graph root 或结构化 body。provider-owned topology/numeric generation 保持 zero-copy
  fixed binding，input/output dense storage 使用通用 runtime 协议；numeric generation
  失效后必须重建 Graph。通用控制与 provider 合同通过 preconditioned CG 和非对称
  BiCGSTAB 程序完成资格验证，不增加 solver-specific Graph API。连续 CGraph/provider
  region 会融合为一个 backend region；provider 可绑定 invocation-private Graph temporary，
  而不把它公开为 runtime 参数。`LinearOperator.from_graph(..., state=...)` 还可为每个
  distinct dependent pure-dense SNodeTree 接收一个代表性的 live root-dense scalar、
  Vector 或 Matrix Field，并无复制保留该 tree 的原 storage。匹配粒度是 tree；key 和
  Field component 不是访问级 capability。generic compiled-Graph operator 保留有序 multi-dispatch
  forward 与显式 adjoint action；legacy square 形式可录制 forward action，但不会推导
  adjoint。依赖 tree 漏报或多报、dependent tree 含任一 sparse/dynamic descendant、
  indirect dispatch，以及 stale numeric、SNode 或 runtime generation 都会明确失败。
  单独录制一个 action 不保证提速；预期收益来自与周围 Graph
  action 的组合。
- 当所有 leaf 都提供可录制 f32 action 时，`LinearOperator` 的 scale、sum、compose 与显式
  adjoint tree 现在会递归 lowering。sum/compose 使用 Graph-owned bounded temporary
  arena 的 typed f32 storage，不公开 scratch runtime 参数，并为并发 submission 保持独立
  lane。standalone f32 scale/sum/compose 也可在 CUDA/Vulkan 执行；sum/compose 持有私有
  persistent workspace。recordable composed CG/PCG provider 在两个 GPU backend 上自动使用
  已资格化 device-convergent 路径，不会改选 host-check 替代方案。
  `linear_operator_composition_bench.py` 会比较自动 Graph、等价显式 Graph、standalone/无
  Graph 执行，以及 compact Field 的 direct/staged binding，并报告正确性和 temporary 内存。
- 新增 compiled-kernel f32 CG/PCG 的显式 CUDA `device_convergent` 执行。该路径通过通用
  结构化 Graph 和可录制 A/M action 完成，每次 solve 只读取一个 terminal packet，并在
  provider 不可用或 stale 时明确失败。并行 vector update 与持久两级 shared-block
  reduction 让 recurrence work 留在 device 上，迭代内部不做 host observation；plan 会报告
  reduction geometry 与 fixed scratch bytes。它以 `explicit_only` 完成 correctness 资格；
  自动 compiled-kernel plan 保留 K=4 `host_check_every_k`，使 construction/first-execution
  摊销仍由 workload 显式决定。stored f32 CSR/BSR CG/PCG 继续保留自动
  conditional-Graph upgrade。新增 `linear_operator_graph_krylov_bench.py`，按 policy 报告
  build、first、warm、profiler、terminal 与真实 residual 证据。
- CUDA/Vulkan 上满足资格的 recordable f32 CG/PCG plan 现在会把 canonical compact
  full-field RHS、output 与 initial guess 绑定为 solver Graph runtime argument。Graph
  preamble/epilogue 通过一个 plan-owned iterative ndarray 搬运边界值，删除独立
  pack/unpack submission、一次 completion sync 和原有两个 boundary staging vector 中的
  一个。这是 Graph-fused boundary 路径，不是 provider-native zero-copy；indexed/non-compact
  layout 与其它 solver/provider 组合继续 staging。新增 telemetry 会区分能力支持、启用状态、
  最近一次完整边界选择、direct binding 与 fallback transfer。在本地 Windows、10 次有效 CG
  的资格测试中，重复 262,144 scalar composition 的 warm median 在 CUDA/Vulkan 上分别
  降低 2.3%–4.3%/3.9%–11.2%；2,304 scalar 时分别约为 4.1%–4.2%/10.3%–11.1%。
  这些数字同时呈现本机 crossover 与桌面运行波动，不是无条件提速承诺。新增
  `linear_operator_graph_field_solve_bench.py` 提供
  ndarray/forced-staging/direct 配对测试。
- `ti.linalg.LinearOperator.apply()` 与单系统 `SolvePlan.solve()` 接受受支持的
  1D/2D/3D root-dense scalar、Vector 和 Matrix field。overwrite
  `LinearOperator.apply()` 可在 CPU/CUDA/Vulkan 的 compiled-kernel 与
  compiled-Graph provider 上直接绑定 canonical compact full field，fixed native
  CSR/BSR provider 则覆盖 CPU/CUDA。generalized apply、Vulkan native sparse
  provider、indexed/non-compact view，以及不属于 recordable f32 CG/PCG 资格范围的
  SolvePlan 组合使用可复用 device staging。warm solve 不重新分配 staging，转换不会进入
  Krylov iteration。
  稳定 raw-field binding 复用已验证的 implicit view 与 transfer plan；执行 telemetry
  明确区分 direct submission、native pack/unpack 和 compiled Graph pack/unpack 路径。
- 新增 runtime-bound `VectorView` 与 `vector_view(field, indices=...)`，用于声明经过
  验证、冻结的 scalar subset/permutation。新增版本化 capability、layout metadata、
  direct/staging/pack/unpack/indexed-copy telemetry 与 provider-qualified zero-copy
  candidate 报告。sparse SNode、noncanonical layout、非法 index 与不安全 alias 明确
  失败，不执行 host vector fallback。
- native algorithm 与 LinearOperator vector adapter 现在统一从 dense storage descriptor
  获取 dtype、shape、owner generation、byte range、offset 与 record stride，同时保留
  provider 特定 handle 和 warm native-plan replay。

- `ti.linalg.LinearOperator` 现在是公开的 runtime-bound operator API。
  operator trait、composition、vector view 与 operator qualification 均从
  `ti.linalg` 导出。仅接受 field callback 的旧包装类型命名为
  `ti.linalg.FieldLinearOperator`，从而消除两个不同 `LinearOperator` 合同之间的歧义。
  solver execution plan 仍位于 `ti.linalg.experimental`。
- LinearOperator 的 compiled-kernel 与 compiled-Graph provider 现在通过统一
  runtime-storage argument 协议绑定经过资格验证的 Program-owned Ndarray、dense field
  和显式 `DenseNdarrayView`。compact operand 在 CPU、CUDA 与 Vulkan 上直接绑定；一维
  scalar positive-stride view 可直接绑定 compiled kernel，并在 Graph 中保持 zero-copy
  执行。不支持 affine 的 provider 组合会明确失败。

### 数值工具支持边界

`0.6.0` 的 `LinearOperator` 工具支持 fixed-topology、runtime-owned operator，以及
经过资格验证的 CPU/CUDA/Vulkan Krylov 执行。文档 provider 矩阵覆盖 CG/PCG、
MINRES、BiCGSTAB、restarted GMRES，以及使用有限 cyclic variable-linear action table
的 FGMRES。solver plan 提供真实 residual 终止、immutable generation 所有权、持久
workspace、结构化 capability 结果与 provider-neutral qualification report。

当前合同不包括 nonlinear 或 callback 驱动的 preconditioner、自动 restart 选择、
block/recycling/pipelined GMRES、MINRES-QLP 或 singular minimum-norm 语义、GPU
`f64` GMRES-family 执行、variable-action CUDA Graph/Vulkan command replay，以及
single-system 异步提交。Forge 也不构造 IC/ILU/AMG、multigrid、Schur/field split、
domain decomposition、contact、KKT 或 nonlinear outer-solver policy。不受支持的
组合会明确失败，不进行 silent host staging 或 provider replacement。

### 稀疏 runtime 与线性代数现代化

- 通过按需增长 active-list metadata、自适应 traversal-list chunk、独立 ambient
  allocation、有界 traversal/recycle budget 和正确保留 non-contiguous SNode slot，
  降低 sparse runtime 固定开销。CPU listgen 使用并行执行，稳定拓扑可复用生成的
  list；CUDA 合并重复 activation；Vulkan 对常驻 traversal list 设置上界。
- 增加 CPU、CUDA、Vulkan 上经过验证的 scalar sparse assembly。builder insertion
  有明确上界，CUDA/Vulkan transactional 地发布完整 CSR generation；unsupported
  format 明确失败，matrix ownership 在 `ti.reset()` 时保持安全。
- 增加 immutable `SparsePattern.csr()` 与
  `SparsePattern.bsr()`。多个 matrix 共享 canonical indices，各自持有独立 numeric
  buffer；`update_values()` 不重建 topology，只替换 values。BSR 支持 2、3、6、12
  block size 和 rectangular SpMV operator。CPU values 支持 `f32/f64`，
  CUDA/Vulkan fixed storage 使用 `f32`。
- 扩展 `SparseCG`：增加 relative tolerance、显式 scalar Jacobi、fixed CPU
  CSR/BSR 和 fixed CUDA BSR provider。fixed provider 复用 solve workspace，并在
  value update 后只自动刷新 numeric Jacobi/block-Jacobi state。
- 增加 CPU `SparseMINRES`，用于完整显式对称不定 CSR/BSR；增加 CPU
  `SparseBiCGSTAB`，用于非对称 CSR/BSR。迭代 solver 根据真实残差合同
  `||b-Ax|| <= max(atol, rtol*||b||)` 报告收敛。这两个旧 stored-solver
  constructor 仍仅支持 CPU。
- 增加 `ti.linalg.LinearOperator` 与实验性 `SolvePlan` API。fixed
  stored CSR/BSR、精确 compiled-kernel provider 和按 role 分类的 compiled Graph
  使用统一 runtime/lifetime/capability 合同。显式数学 trait 作为 CG/PCG 门禁；
  persistent plan 通过统一 `SolveResult` 返回 terminal state，并在文档支持矩阵内提供
  CPU/CUDA/Vulkan CG、fixed stored Jacobi/block-Jacobi PCG、provider-neutral
  MINRES、provider-neutral BiCGSTAB、restarted GMRES 与 variable-linear FGMRES。
  CPU 还支持最小 operator composition。
- 扩展 compiled-kernel/Graph `LinearOperator`，支持 `(range, domain)` 矩形
  shape、独立显式 adjoint、`A.adjoint().adjoint()` 和共享 immutable numeric
  generation。`apply()` 增加 CPU 通用 `alpha/beta/addend` 合同与 `beta=0`
  no-read 语义；GPU 不支持的系数组合明确失败。新增 provider-neutral
  `qualify_operator()`，生成包含 oracle、adjoint、capability、计时和 native counter 的
  版本化 JSON 证据。
- 扩展 `SolvePlan(method="pcg")`，支持可信 fixed-linear `LinearOperator`
  preconditioner。CPU 接受受支持的 operator provider 组合；CUDA/Vulkan 接受成对的
  compiled-kernel A/M provider。CUDA 可把 CG recurrence scalar 常驻 device，并每 4
  或 8 iteration 检查收敛；Vulkan 支持相同 chunk size 与 relative tolerance，同时保留
  fixed-budget masked execution 作为兼容默认值。logical、executed、wasted iteration
  与 host check 会分别报告。
- 为 fixed stored f32 CSR/BSR 增加 CUDA/Vulkan 原生 solver-chunk replay。CUDA 将
  K=4/8 的 CG/PCG chunk capture 为 CUDA Graph；Vulkan 将相同 sparse recurrence 录制为
  可复用 command sequence，并覆盖 identity、Jacobi 与 block-Jacobi。values-only numeric
  refresh 保留既有序列，外部 output binding 或结构变化会显式失效并重建；compiled-kernel/
  Graph A/M 保持 direct submission，不做 host fallback。
- 扩展 `SolvePlan(method="bicgstab")`，支持 fixed-linear 右预条件与
  CUDA/Vulkan `f32` 执行。device plan 保持 recurrence state 常驻，以原系统真实
  residual 认定 terminal status，并报告结构化 rho/alpha/omega breakdown reason。
  fixed stored identity plan 复用 CUDA Graph 或 Vulkan command-sequence chunk；
  compiled A/M action 保持 direct submission。`statistics()` 精确报告 A/M、dot、
  vector、logical/executed/wasted work 与持久 workspace。
- 增加 restarted `SolvePlan(method="gmres")`，restart 可为 8、16 或 32。
  CPU 支持兼容的 `f32/f64` provider；CUDA/Vulkan 支持 `f32` fixed CSR/BSR
  与 compiled kernel/Graph provider。每个 Arnoldi step 使用两遍 CGS、
  multi-dot reduction 与 fused projection，每个 restart boundary 校验原系统真实
  residual。支持 fixed-linear 右预条件。fixed stored identity cycle 使用 CUDA Graph
  或 Vulkan command replay，其它合格 provider 使用 direct native submission。
  `statistics()` 报告 basis/workspace bytes、A/M、dot/multi-dot/vector work、
  restart cycle、happy breakdown 与 logical/executed/wasted iteration。
- 增加 restarted `SolvePlan(method="fgmres")`，消费有界 variable-linear
  `PreconditionerPlan` action table。1 到 32 个兼容 linear action 按 solve-global
  scheduled inner iteration 循环选择，restart boundary 不重置调度。CPU 支持
  `f32/f64` host action；CUDA/Vulkan 支持兼容 device-native `f32` fixed stored
  与 compiled provider。solve 进入时会 pin 全部 action generation，持久 `Z` basis
  保存预条件 vector；`statistics()` 报告其 bytes、action selection、schedule wrap 与
  update outcome。GPU 使用 direct native submission，并显式报告 replay 合同不可用。
- 增加公开 `PreconditionerPlan` 与 pinned `PreconditionerSession`。外部近似逆可显式执行
  setup、rebuild update 或 lagged reuse，并分别记录 built-from provenance 与 accepted-target
  compatibility；target 更新默认 stale。CPU/CUDA/Vulkan PCG 消费批准后的 immutable
  generation，不在 iteration 热路径执行 Python callback。variable-linear table 在发布
  update 前会 preflight 全部 action，并且只由 FGMRES 消费。10k numeric-generation churn
  验证 retired generation 有界释放；nonlinear behavior 仍明确不受支持。
- 增加单系统 fixed stored f32 CSR/BSR CG/PCG 的精确 CUDA conditional-Graph 执行。
  driver 与 identity、Jacobi 或 block-Jacobi provider 的 capture 资格满足时，默认的
  `bounded_convergent` policy 会自动选用该路径。每次 solve 只保留初始和 terminal state
  两次观察，不再逐轮执行 host scalar synchronization，并在精确的逻辑 iteration 上终止，
  不产生 masked tail work。资格不满足的 runtime 通过文档规定的 Graph chunk fallback
  保持相同数值合同；显式 `host_each_iteration` 仍可选用。
- 增加 CPU、CUDA 与 Vulkan 上的同构独立批量 f32 CG/PCG。每个连续系统拥有独立
  tolerance、status、iteration 与 residual state；fixed stored 和 compiled-kernel A/M
  provider 已完成资格验证。CUDA/Vulkan plan 会复用 plan-owned Taichi Graph 执行稳定的
  iteration recurrence，同时把 A/M 保持为 pinned provider action；更换 output 会 patch
  Graph binding，每个 workspace clone 拥有独立 replay plan。CUDA/Vulkan fixed-budget plan
  还提供 `submit()`、
  `SolveSubmission` 与显式 workspace clone，用于有界并发执行。执行策略 capability 与
  unsupported reason 可查询；batched plan 仍不支持 device-convergent 条件执行，单系统
  路径则按上述资格启用。plan 统计报告每个
  clone 的逻辑 workspace payload 与排除项；host 异步 completion 不代表设备并行保证。
- 增加面向 `SolvePlan`/`BatchedSolvePlan` 的 provider-neutral solver qualification。
  版本化 detached report 覆盖 solution/真实残差、terminal state、A/M 身份、policy、
  logical/executed/provider work、完整 preconditioner action-table provenance、chunk
  counter、transfer、resource、memory-pool 增量和可选
  pacing。factory build、first solve、warm wall time 与合格 fixed-budget host submit 分开记录；
  无法取得的 device timestamp 与 driver identity 明确保持 unavailable，不进行推测。
- 增加 `ti.graph.SubmissionPacer`，为共享 CUDA/Vulkan backend 的
  `Graph.submit()` 与 fixed-budget batch solve 提供显式协作式节奏控制。全局/单 lane
  in-flight 上限、有限等待队列、跨 lane round-robin 与 lane 内 FIFO 可组合配置；调用方可
  选择阻塞 backpressure 或提交前立即拒绝。公开统计覆盖队列峰值、准入等待、逐 lane 完成
  与后端失败，并明确其准入单位为 invocation count，不预算 workspace 或 numeric generation
  字节；runtime reset 与第一次 completion failure 均采用明确的失效语义。
- 强化 direct solver symbolic reuse：只有完整 compressed index pattern 相同，
  `factorize()` 才能复用已分析 pattern；factorization 后更新 values 会使分解
  stale，必须刷新后才能 solve。stable-fluid example 固定 pressure gauge，implicit
  mass-spring example 跨 value-only step 复用 symbolic analysis。
- 完整用户流程、特性、backend/format/dtype 矩阵、失败语义和生命周期规则见
  [稀疏 runtime 与线性代数](sparse_runtime_and_linear_algebra.zh.md)，并可参考
  [LinearOperator 与 SolvePlan](linear_operator.zh.md)、
  [稀疏布局选择指南](sparse_layout_selection.zh.md)与
  [物理稀疏 solver 选择指南](physics_sparse_solver_selection.zh.md)。

- 将 CUDA native primitive 的自动调度切换到 Forge 自有 driver-only provider，覆盖诊断、
  scan/reduce/histogram、组合 primitive 与 stable radix sort。标准 runtime 不再链接或
  打包 CUB/CUDART；显式 `cuda_cub*` method 已弃用，并隔离到不发布产物的
  Toolkit-reference workflow。
- 将 CUDA/Vulkan primitive 资源迁入 Program-owned arena，提供有界 lease 与显式
  clear/statistics。Vulkan 在不做 queue-wide wait 的前提下回收已完成 descriptor/resource
  set；CPU 每个算法族/worker 最多保留 8 MiB primitive scratch，更大请求采用有界瞬时分配
  和 fallback 策略。
- 新增 opt-in `get_primitive_runtime_diagnostics()` 与
  `get_primitive_workspace_statistics()` schema-v1 snapshot；provider dependency、fallback、
  Program provider bytes 和 per-Python-thread 默认 cache 可观测，读取不增加 device sync。
  `workspace=None` cache 默认限制为每 context 64 项、全进程 16 个 context；显式 clear
  要求 submission 已静止。
- 将 CUDA scan 改为 1024-item tiled hierarchy，将 compact 的 flag normalize 与局部 rank
  融合并只扫描 tile count；stable sort 从 1-bit pass 改为分层 4-bit LSD radix。Windows
  百万元素正确性、两 host submitter stress 与 idle-guarded reference 对照已完成。相对测量中
  histogram 与 compact 最接近列出的 CUB reference；scan、reduce 与 stable sort 仍明显落后。
  标准 wheel 选择正确、异步、driver-only 的 Forge provider，但不声明与 CUB 性能相同；具体
  数字与测试条件见 [Native 算法](native_algorithms.zh.md)。
- 0.6.0 标准 runtime wheel 改用 `driver-only` dependency class 门禁，同时继续兼容已经发布的
  0.5.0 包内 CUDART wheel 的 loader、repair 与验证。项目仍按操作系统各发布一个 runtime
  wheel，不按 CUDA 版本分叉。
- CPU native dense-field 路径现在直接使用编译后 SNode layout 中的 root-child offset，
  不再通过前序 payload 大小推算地址。混合 f32/f64 root child 之间的 alignment padding
  因此不会让 `to_numpy()`、`from_numpy()` 或 native field operation 错读相邻 field；
  普通 kernel hot path 不增加分支或复制。
- 最终 runtime/shim wheel 门禁现在记录并校验 native runtime commit identity 的有效性，
  但不要求它与 shim 源码 commit 相等；同时覆盖 shape `()`、`1`、`7` 的 f32/f64 field、
  host/kernel round-trip、serial/atomic f64 reduction、offline cache 与单线程/默认线程配置。
- Windows driver-only/reference build 与 primitive 正确性矩阵已经完成。降低任何公开 driver
  下限之前，仍必须补齐 Linux wheel/import/依赖扫描、compute-sanitizer 和每个声明支持的旧
  NVIDIA driver 真机执行。

### 宿主内存与运行时生命周期有界化

- Host allocator 现在会在 non-exclusive chunk 中的有效请求全部 release 后解除对应 OS
  mapping，并同步扣除 capacity、cursor、alignment waste 与 released-byte 统计。反复创建并
  释放 large/adaptive chunk 不再让进程保留每一份历史 mapping；仍有 live allocation 的
  chunk 按所有权合同继续保留。
- 将此前可能随进程寿命增长的内部记录改为有界合同：Blender 临时源码使用 32 项 LRU 并
  清理淘汰文件；compile/timeline trace、kernel-profiler raw record 与 Python kernel
  specialization 都有固定预算。当前 Program 默认最多编译 1024 个 specialization；达到
  上限后，已有 specialization 仍可使用，新的 cache miss 会明确失败，而不是继续吃掉
  host memory。
- 重复 `ti.init()`/`ti.reset()` 的生命周期不再保留已销毁 SNodeTree 的 launcher、accessor、
  frontend field 映射或 GFX runtime state；Python runtime object registry 使用弱引用，版本
  检查线程每个进程最多启动一次。普通 kernel、Graph 与 UI runtime 不创建持久 helper
  subprocess；应用自己启动的 multiprocessing worker 仍由应用负责 join/terminate。
- 这些修复关闭的是 runtime-owned 的无界历史增长源，不是全进程 RSS 上限。用户仍持有的
  field/ndarray/Graph、尚未完全释放的 allocator chunk、有限 specialization 集、driver/
  context high-water mark 与磁盘 offline cache 都可能按真实 workload 占用资源。诊断与
  配置边界见 [Forge API 参考](forge_api_reference.zh.md#内存增长与所有权边界)和
  [Forge 选项](forge_options.zh.md)。

### 正确性、能力与明确支持边界

- 0.6.0 补全 CPU、CUDA、Vulkan 共享前端、IR、AD、AOT、runtime 与 RHI 中具有明确
  正确性、安全性或生产价值的合同。完整 tile/block/warp/
  subgroup DSL、异构多设备 runtime、稀疏专项和其它后端的新能力仍在当前范围之外；相关入口
  必须明确返回 unsupported/fail-fast，不能用空实现或静默降级伪装成功。
- 补齐 lifecycle/capability/observability 的基础合同：field/AD 枚举只返回 active
  SNodeTree，已销毁 generation 不再重新进入执行；Vulkan 分开声明 f16/f32/f64 atomic-add
  capability，unsupported feature 不会冒充 native 支持；CUDA profiler 只累计上次 query 后的
  新 record，重复查询保持幂等；12 个尚未实现的 subgroup operation 在编译期报告操作名、
  arch 与支持状态，不再由 Python `pass` 返回 `None`。
- Windows 原生构建和 CPU/CUDA/Vulkan 定向矩阵已经完成；GPU 用例只在没有其它 Python/GPU
  compute process 时运行。矩阵也覆盖固定维 3x3 tuple/vector/outer-product/matrix 组合、
  local Vector/Matrix 动态读取的一阶 reverse AD，以及 rotation、inversion、near-singular
  和 repeated-singular-value 的 3D SVD primal 边界。Linux GCC/Clang、headless Vulkan validation、CUDA driver-only
  import/execution 与真实 Torch AD 仍属于发布前复测项，详见
  [Linux 复测清单](linux_revalidation.zh.md)，不会用 Windows 结果替代跨平台结论。
- 强化 debug 执行与索引契约。CPU assertion 失败后会协作取消剩余 debug work，发布一致的
  首个错误，并保持 worker pool 可复用；矩阵/向量访问会逐逻辑轴检查和 clamp，不再接受
  线性化后碰巧落在存储范围内的别名分量；`assume_in_range` 会在支持的整数范围内避免窄
  整数溢出地执行验证。显式 `check_out_of_bound=False` 现在可以覆盖 `debug=True` 的隐式
  bounds 默认值，同时不关闭其它 debug 行为。Vulkan 仍不支持生成 assertion，但已支持
  逐轴 clamp 行为。
- 外部 PyTorch tensor 不再仅因 primal kernel 看到 `requires_grad=True` 就分配完整
  `zeros_like` gradient。Forge 只在 reverse/forward AD、Tape 或 kernel 内显式访问 `.grad`
  时延迟分配对应 tensor 大小的 gradient，并且不会替换用户已有的 gradient。因此纯 primal
  调用可为每个受影响 tensor 避免一份同尺寸分配，同时保留既有 AD 路径。
- GFX kernel 现在分别记录 external array 的 primal 与 gradient 访问。Vulkan 会把 host
  gradient 放入独立 device buffer，并且只在 grad kernel 确实写入时回读；device
  `ti.ndarray` gradient 继续直接使用 device allocation。Torch gradient 的 shape、dtype、
  contiguous layout 或 device 不匹配会在 launch 前被拒绝，不再返回伪正确或不安全梯度。
- 将 `ti.ad.FwdMode` 参数 seed 从 scalar field 扩展到 CPU、CUDA、Vulkan 的 dense vector
  与 matrix field。shaped seed 使用 `field_shape + element_shape`，flat seed 使用 row-major
  顺序；该合同不依赖 AoS/SoA layout，并保留每个 context 只接受一个参数组的既有边界。
- 明确定义 automatic differentiation 的阶数边界：一阶 Tape、手工 reverse 与 FwdMode
  路径已在 CPU/CUDA/Vulkan 验证；嵌套 context、Tape 内手工 reverse 和
  forward-on-reverse 现在会在编译/提交前拒绝。Tape 正文抛出异常后不再对不完整 trace
  运行 adjoint；动态 early-return CFG 继续由前端明确拒绝，不会产生不完整梯度。
- AOT module 创建现在强制执行实际的 same-target 合同。传入与 active `ti.init()` arch
  不同的 `arch` 会在 backend builder 创建前报错，不再 warning 后静默替换 artifact target。
- CUDA LLVM AOT 现在针对显式、进入 cache key 的 target capability 编译（默认 SM 60 / PTX
  50），target-sensitive codegen 不再读取构建机 GPU。artifact 会在 sidecar 中记录
  compute/PTX 要求，loader 在 kernel 注册前拒绝能力不足的 device；更高的精确 target 需要
  显式选择，且不增加 Toolkit/CUDART runtime 依赖。此 sidecar 合同建立前生成的 CUDA LLVM
  AOT artifact 必须重新构建。
- GFX AOT metadata 现在保存全部稠密 root buffer 大小、每 field 的 tree id 和每 kernel 的
  SNodeTree 依赖。C API loader 会分配所有 artifact root，并按记录的 tree 数注册 kernel，不再
  硬编码单 tree；非连续 live tree id 在构建时 fail-fast，稀疏 SNode AOT 仍不在支持范围内。
  显式 tree 已提供有效 layout 时，AOT module 创建前后的首次 kernel materialize 都不会再追加
  metadata 中不存在、也未被使用的尾部空 root；完全无 field 的首次 kernel 仍保留默认空 root。
- AOT kernel template 现在在 CPU、CUDA、Vulkan 上接受边界明确的 ndarray/external-array
  exemplar。specialization 使用与 capacity 无关的 element/layout ABI key，在编译前拒绝不支持
  或非连续输入，去重相同合同，并使用文件系统安全名称及长 signature 的 SHA-256 fallback。
- Vulkan storage image 现在根据声明格式选择 f32、i32 或 u32 sampled value；r/rg/rgba
  16/32-bit 有符号与无符号 image 的 frontend、SPIR-V 和 Vulkan format 合同已经一致，
  原先 r16u 格式族错误使用 UNORM 的映射已改为 UINT。
- kernel launch、Graph 与 AOT 的类型检查现在共享同一个内部结构描述，统一覆盖 scalar、
  vector、matrix、ndarray、texture 与 StructNdarray。Graph ndarray metadata 会端到端
  保留 tensor element type；StructNdarray 继续受普通 kernel 支持，但在序列化 Graph
  schema 能表达结构化 element 之前会由 Graph 明确拒绝。
- Matrix/Vector Graph 参数现在分别使用规范的 rank-2/rank-1 tensor shape；Graph injection
  cache 按相同结构合同复用，不再依赖 Python 对象 identity。0.5.x 的 flat Matrix 和嵌套
  symbolic-list adapter 继续兼容；真实的 rank-2 shape 不匹配会明确拒绝，128-byte 运行时
  上限也会在复制数据前检查。
- 不支持的 rank>2 ndarray element 与超出合同的 quant width 现在由 Python/type validation
  fail-fast。quant float 会按 native exp/significand 与 f32-compute 边界明确报错，不再触发 C++
  assertion；任意 stride 外部 array 继续明确拒绝。Graph texture descriptor 会在编译前校验
  dimension 和 RW format。LLVM 20 signed constant 现在保留固定宽度 bit pattern，signed quant
  host access 不再导致进程 abort。
- 混合类型全局归约的 TLS 布局现在会尝试稳定的按大小重排，但只有候选布局严格减少
  scratch bytes 时才采用；例如同一 offload 中依次出现 f32、f64 reduction 时，每个 TLS 实例
  的 TLS 从 16 bytes 降为 12 bytes。相同 dtype 的归约次序保持不变，tensor 等非二次幂
  大小若不能受益则保留原布局，不增加 runtime 分支或额外设备同步。

## 0.1.0

- 将 Python 构建迁移至 scikit-build-core，并建立最初的 `taichi-forge` 发行包身份。
- 在保留 upstream Taichi DSL 模型的同时，开始 Forge 专用构建/工具链与编译配置线。

## 0.1.1

- 将 Python import tree 从 `taichi` 重命名为 `taichi_forge`。
- 修复新包身份下的 scikit-build-core 安装路径、manifest、package data、示例与内部
  import。

## 0.1.2

- 修复剩余 Python import/rewrite 问题。
- 在发行构建路径中开放 CUDA 编译选项。

## 0.1.3

- 在 LLVM 20/scikit-build-core 工具链上确立 `taichi-forge` 发行包与
  `taichi_forge` import 身份。
- 增加首批 compile profiling、cache warmup、compiler tier 与后端隔离 cache 控制。
- 发布 Python 3.10-3.14 的 Windows/Linux wheel 线。

## 0.2.4

- 扩展 per-kernel optimization level、compile profiling、materialize fast path、
  source/backend cache 隔离与原子 cache 写入。
- 增加缓存/并行 SPIR-V codegen 与 optimizer 复用，并避免嵌套 compiler pool
  oversubscription。
- 增加 memory-pool statistics、Vulkan buffer pool、compiler telemetry，并更新
  MSVC/UTF-8/toolchain 依赖。

## 0.3.0

- 首次加入实验性 Vulkan `pointer`、`bitmasked`、`dynamic` SNode，包括 SPIR-V
  list generation 与 pointer allocation。
- 增加实验性 Vulkan quantized-field 开关；未支持 quantized 操作继续明确拒绝，
  不静默误编译。

## 0.3.1

- 通过 ambient zone 让 inactive Vulkan pointer-cell 读取返回 dtype 零值。
- 加固 pointer allocator、freelist、嵌套 SNode list generation 与 allocator metadata。

## 0.3.2

- 增加 deterministic-slot pointer activation，消除全激活时 CAS/spin 导致的
  device-lost 路径。
- 对不能使用 deterministic slot 的 layout 保留已记录的 fallback。

## 0.3.4

- 为 bitmasked node 增加 clear-on-deactivate。
- 融合两级 sparse deactivation，并修复 index 校验。

## 0.3.5

- 增加 intermediate-list-generation 控制、ballot/grid-dimension 改进和显式 CUDA
  sparse-pool 调优参数。

## 0.3.7

- 回退不安全的隐式 CUDA sparse-pool auto-sizing，在继续测量期间恢复保守行为。

## 0.3.9

- 将 `vk_max_active` 作为 Vulkan pointer SNode 与 CUDA sparse-pool sizing 的显式
  capacity hint。
- 完成首个广泛可用的公开 Vulkan sparse-SNode 发布线。

## 0.3.11

- 增加 per-SNode CUDA sparse-pool auto-sizing、`element_list` budget tracing 和
  LLVM runtime 诊断。

## 0.3.12

- 增加 CUDA deterministic pointer slot、fast reset、sparse-list reuse 和更安全的
  pool 生命周期。
- 改进 Vulkan list-generation reuse、descriptor/resource cache、task-adaptive SPIR-V
  优化、lazy submit 与 runtime statistics。
- 让 GGUI window 在 reset 时退役，并增加 pipeline-cache 持久化。

## 0.3.13

- 在 CPU、CUDA、Vulkan 上增加实验性固定容量 Hash SNode。
- 增加可选 active list、compact child pool、probe/list-generation telemetry、测试和
  benchmark。

## 0.4.0

- 增加 Forge 稳定排序调度器，以及 CPU/CUDA/Vulkan sort、scan、compact、reduce、
  histogram、transform、gather、scatter、scatter-add、bucket-builder 与
  grouped-reduce 路径。
- 增加可复用 native plan/workspace、基于 capability 的 `method="auto"` fallback、
  多 dtype 与 Vulkan shader 实现。
- 增加 StructNdarray opaque payload 和 scalar/tensor member-view 路径。
- 增加 Vulkan offscreen，以及 Linux/GCC wheel 构建修复。

## 0.4.1

- 增加 `ti.compile_kernels()`、`ti.parallel_compile()`，扩展
  `ti.compile_profile()`、compile tier 与 offline-cache sharding/locking。
- 在既有 GraphBuilder/CGraph API 下现代化 Graph 执行，并加入 Forge native replay
  node 与 `PrimitiveSequence`。
- 增加 `ti.ui.DisplayFrame`、`Canvas.submit_frame()`、display statistics、
  packed-u32 Vulkan 直接显示、texture upload 和有界 in-flight frame。
- 优化 native primitive plan、workspace reuse、dense-field route 与 GGUI staging。

## 0.4.2

- 修复 ArgPack allocation 生命周期、Vulkan 小整数 Field、Vector/Matrix ndarray
  释放和 PrefixSum 内部 warning。
- 修复 hidden/offscreen GGUI window teardown，以及早期 Vulkan sparse-SNode
  inactive-read/全激活问题。

## 0.4.23

- 将平台原生 runtime 拆为 `taichi-forge-runtime`，保留小型 per-CPython
  `taichi-forge` shim。
- 修复 Vulkan ArgPack 重复更新，以及创建 sparse SNode 后的 CPU/CUDA dense native
  Field 访问。
- 增加 device-side 数值 checks/metrics 与 native Graph result node。
- 加固 Vulkan ArgPack mapping、小整数 SPIR-V、CUDART 链接、版本传播与发布 workflow。
- 在确认 ROI 很低后，移除废弃的 `use_fused_passes` / `pipeline_dirty` 实验，并
  退役独立 Vulkan buffer-pool/listgen-barrier 实现。后两者的字段仍作为 no-op 兼容
  名称被接受；cache schema 会拒绝 fused-pass 过渡配置产生的 artifact。

## 0.4.24

- 将常见 CUDA/Vulkan Field 与 ndarray 图像在 device 上 pack 为 RGBA8，并为连续
  `uint8` RGBA NumPy 图像使用直接 host 路径。
- 降低仅渲染帧开销，并修正 package/version metadata。

## 0.4.25

- 为 GGUI event API 增加 `poll=False`，阻止每帧重复更新 native cursor，使异步渲染
  循环可以只让 `window.show()` 执行事件泵。
- 使用 `EndFrame()` 平衡空 ImGui frame 生命周期，并跳过不必要的 ImGui draw 提交。

## 0.5.0

这里只列 `0.4.25` 边界之后的工作。native algorithms、最初的 Graph modernization、
`PrimitiveSequence`、DisplayFrame、compile profiling 与 GGUI device-image staging
在 `0.4.25` 前已经公开。

- 按真实 queue handle 对 Vulkan submit/present 做 external synchronization，用
  submission fence wait 替代 queue-wide idle，并保护 per-thread stream、profiler
  query、descriptor、pipeline cache 与 GFX recording state。
- 加固 CPU/CUDA/Vulkan runtime 初始化、完整 kernel submission、allocation
  identity/generation/range 校验、mapping/reset 生命周期、CUDA-Vulkan external-memory
  fallback，以及 CPU scheduler/native replay。
- 增加 Program-owned first-fault domain，统一处理 context-fatal CUDA 错误与 Vulkan
  device loss。kernel、Graph、ticket、同步和 GGUI 路径会引用原始根因快速拒绝；
  fault-aware reset/finalize 跳过不安全后端等待，但不宣称同进程 device 恢复。
- 增加 `ti.runtime.stats()`、`ti.runtime.capabilities()` 和有界
  `ti.runtime.trace()` context。不可变的 Program-generation snapshot 公开
  submission、synchronization、memory、transfer、Graph、display、fault 与 trace
  数据；不可用的可选测量保持 `None`，trace 导出只记录有界 host event，不伪装成
  GPU profiler。
- 将 runtime statistics 扩展到 schema v2，准确区分 host allocator capacity、
  cursor consumption、alignment/已释放不可复用 waste、slab/large/exclusive chunk
  与 lifetime peak。Windows reserve+commit 和 Linux anonymous-mapping residency
  分开表达，不伪造 committed bytes。
- 将固定 1 GiB host slab 改为从 16 MiB 开始、按几何级数增长并以既有 1 GiB
  为上限的策略；超出下一 slab 的单次大请求仍使用按请求大小的 mapping，并用每 chunk
  地址索引和 newest-slab search 避免 mapping 增多后的线性扫描回退；保留仅供发行诊断
  的内部 legacy policy 回退环境变量。
  在受控 Windows fresh-process A/B 中，CPU/Vulkan 初始化 host commit 从 1 GiB
  降到 16 MiB；CPU incremental private bytes 约降低 97.4%，Vulkan 约降低 86.8%。
  CPU/CUDA/Vulkan 的普通 kernel/Graph median 变化均在 1.5% 内；Linux 测量待复测。
- 分离 CUDA device capability 与 LLVM code-generation target，隔离 target cache，
  移除 CUDA-13.2-only iterator 依赖，加固单 runtime wheel，并避免无返回值 CUDA
  kernel 的无用 result allocation。
- 增加安全的 CUDA Graph argument patch/capture recovery，以及 Vulkan Graph identity、
  in-flight retirement 和固定八 slot replay fallback。
- 增加稳定的 `Graph.execution_stats()`、严格 runtime argument 校验、mixed-segment
  隔离、Graph/reset/resource 生命周期合同，以及按需启用的 `Graph.submit()` /
  `SubmissionTicket` 完成跟踪；默认 `Graph.run()` 热路径不变。
- 增加 Dense Field Graph：静态绑定 scalar/vector/matrix Field、构图期
  `template_args`、带 generation 的 SNodeTree dependency、零参数 CUDA capture、
  显式 AD guard 和 block 级异构多环境模型。
- 增加不可变 schema-v1 native primitive capability 描述与 active Program provider
  解析。operand dtype/rank/layout/storage、backend method、determinism/atomic 顺序、
  AD、Graph/AOT、workspace 与 fallback 合同现在和调度共用 method/AD registry。
  FwdMode 对 transform、reduce-sum、gather、scatter、scatter-add 在 CPU/CUDA/Vulkan
  上使用已验证 kernel fallback；不支持的 native、scan/grouped-reduce 和离散
  automatic-AD 路径都会在写入前拒绝。
- 增加面向整数 key、dense ndarray/field storage 的 device-resident consecutive
  run-length encode、unique 与 unique-by-key primitive。CPU/CUDA/Vulkan 已覆盖
  固定容量 `size=0` 逻辑空输入、device-side count、first-payload 语义、可复用
  `RunLengthWorkspace`、PrimitiveSequence Graph replay、alias/AD guard、
  StructNdarray payload 与独立 workspace 的多线程提交。实现复用既有 compact
  provider，不增加 runtime-wheel ABI 依赖。
- 增加可复用 dense `SegmentedLayout` topology，以及 scalar ndarray/root-dense
  field 的 device-resident segmented sum reduce 与 inclusive/exclusive sum scan。
  CPU/CUDA/Vulkan 已覆盖空 segment、固定容量 padding、稳定 serial 浮点顺序、
  grouped-ndarray reverse AD、Graph replay、独立 workspace 并发、scratch/topology
  统计与粗粒度 backend-aware integer scan 分派。实现组合既有 provider，不增加
  runtime-wheel ABI 依赖。
- 增加面向生产形态的 CPU/CUDA/Vulkan 并发、数值、生命周期、内存与 replay
  regression/benchmark。剩余 Linux 发行证据见
  [Linux 复测状态](linux_revalidation.zh.md)。

当前详细合同集中在：

- [Graph Runtime 与优化](graph_runtime_optimization.zh.md)
- [Dense Field Graph](dense_field_graph.zh.md)
- [Native 算法](native_algorithms.zh.md)
- [编译与高级优化权衡](compilation_tradeoffs.zh.md)
- [构建 Forge wheel](build_wheels.zh.md)
