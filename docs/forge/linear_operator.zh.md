# LinearOperator 与实验性 SolvePlan

> 本文说明 Taichi Forge `0.6.2` 正式发行合同；功能归属见
> [版本更新说明](release_notes.zh.md)。

`ti.linalg.LinearOperator` 提供绑定 runtime 的线性映射抽象，可使用 fixed sparse
matrix、已编译 Taichi kernel 或已编译 Graph 作为 provider。这是通用数值 API：应用可以
用它处理离散 PDE、隐式系统、图问题、优化和其它线性代数任务，无需把物理领域对象引入
Taichi DSL。

`LinearOperator`、`OperatorTraits`、storage-view helper、composition helper 与 operator
qualification 是公开的 `ti.linalg` API。`SolvePlan`、`PreconditionerPlan` 与
`BatchedSolvePlan` 等求解执行对象仍位于 `ti.linalg.experimental`；下文分别说明其 backend
和数值支持边界。

## 核心模型

一个 `LinearOperator` 表示 `y = A x`，包含：

- scalar dtype 和 `(rows, columns)` shape；
- 绑定当前 Taichi `Program` 的一个具体 provider；
- self-adjoint、positive definite 等显式数学 trait；
- 可观察的 capability 与 resource-generation metadata；
- 一个可复用的 native execution plan。

公开 vector 参数保持一维 scalar-flat 数学 ABI。`LinearOperator.apply()` 可接收一维
scalar Taichi ndarray、受支持的 dense field、显式 `DenseNdarrayView` 或 `VectorView`。
单系统 `SolvePlan.solve()` 可接收一维 scalar ndarray、受支持的 dense field 或
`VectorView`；显式 `DenseNdarrayView` 不属于其合同。vector payload 不经 host copy，
也不 materialize matrix-free provider；请求不受支持的操作时不会切换 backend。
operator、plan、provider、ndarray、field view 都属于同一 runtime generation。执行
`ti.reset()` 后它们全部失效，也不能重新绑定到后续 `ti.init()` session。

### 精度边界

compiled-kernel 与 compiled-Graph matrix-free provider 当前在 CPU、CUDA、Vulkan
上均使用精确的 `ti.f32` vector ABI。因此，经由这些 provider 执行的 GPU
matrix-free 求解路径目前只对 `ti.f32` 完成资格支持；该 ABI 的 `ti.f64` 支持不属于
当前版本范围。

runtime 不会静默地把 `ti.f64` vector 降为 `ti.f32`，不会 materialize
matrix-free provider，也不会替换为其它 provider 或 backend。不受支持的
dtype/provider/backend 组合会明确失败。stored operator 与 dense field 仍可在下文
支持表所列的 provider、solver 和 backend 组合中使用 `ti.f64`。

## Prepared apply（0.6.3）

固定存储上的重复 apply 可使用：

```python
prepared = operator.prepare_apply(input_array, output_array, adjoint=False)
prepared.run()
builder.append_native(prepared.record())
graph = builder.compile()
prepared.close()
graph.run({})  # 独立编译的 recording 持有自己的快照引用。
graph.close()
```

`ti.linalg.PreparedOperatorPlan` 在准备时固定一次 provider topology/numeric/coefficient generation，并检查
dense 输入输出的范围、布局、不重叠；不执行数学、不做 field staging。输入内容和显式声明的 live field state
可原位更新。相反，后续 `update_numeric()` 或参数发布**不会改变已有 prepared plan**，需重新 prepare 才采用
新 generation。原 `graph_action()` 的兼容 generation 动态 rebind 行为不变。

支持已有资格的 f32 compiled-kernel/compiled-Graph action 及其可记录组合，不新增 stored/vendor sparse 路由、
f64 ABI、广义 alpha/beta apply 或 indexed view packing。依赖尚未解析 composition workspace 的派生子视图
明确拒绝。`record()` 仅在 Graph root 展开，语义保持 opaque，不因 prepare 宣称可融合。
provider 快照仅保留引用、不复制；每个编译 Graph 自己持有 composition workspace。准备可能编译 helper kernel
并分配 workspace，性能计时需预热。prepared replay 不重新发现 provider/schema，仍遵循既有 capture/replay 能力。

提供 `run()`、`report()`、`close()` 与上下文管理。报告给出固定 generation 版本、dispatch/workspace 事实和
绑定状态，不保证性能收益，也不是完整 driver 显存测量。关闭 plan 只退休自己的 Graph；独立编译的 recording
保留其快照与 workspace 直到自己关闭。runtime reset 或 live source tree 退休使执行失效。plan 不是跨进程可执行
对象的序列化格式。

## Dense field 与 VectorView

受支持的 dense field 可以直接作为 `LinearOperator.apply()` 的 `input`、`out` 或
`addend`，也可以作为 `SolvePlan.solve()` 的 `rhs`、`initial_guess` 或 `out`：

```python
rhs = ti.field(ti.f32, shape=(nx, ny))
solution = ti.field(ti.f32, shape=(nx, ny))

operator.apply(rhs, out=solution)
result = plan.solve(rhs, initial_guess=solution, out=solution)
assert result.solution is solution
```

支持的 field 合同为：

- `ti.f32` 或 `ti.f64`；具体 operator/provider/backend 仍须支持相同 dtype；
- 1D、2D 或 3D `root -> dense -> place` scalar field；
- canonical packed `ti.Vector.field` 或 `ti.Matrix.field`；
- fixed shape、当前 `Program` 和仍然存活的 `SNodeTree`。

scalar field 的 record 即使含有固定 sibling padding，仍可作为 staged full view，但不属于
direct compact candidate。Vector/Matrix field 必须保持 canonical component layout。

field 按 scalar-flat 顺序映射到 operator space：先按 index shape 的 canonical 顺序遍历，
再按 Vector lane 或 Matrix row-major component 顺序展开。因此 scalar extent 为
`prod(index_shape) * prod(element_shape)`，并且必须与 operator 的 domain/range extent
精确匹配。`pointer`、`bitmasked`、`dynamic`、`hash` 等 sparse SNode、quantized storage、
arbitrary nested dense tree 和 noncanonical packed-component placement 会在提交前明确失败。

`vector_view()` 可声明 dense field 的显式 scalar 子集或排列：

```python
indices = ti.ndarray(ti.i32, shape=active_size)
indices.from_numpy(active_scalar_indices)

rhs_view = ti.linalg.vector_view(rhs, indices=indices)
solution_view = ti.linalg.vector_view(solution, indices=indices)
result = active_plan.solve(rhs_view, out=solution_view)
```

participant 拥有连续区间时，应使用 immutable scalar-flat range，而不是构造 index map：

```python
rhs_view = ti.linalg.vector_view(rhs, offset=offset, length=count)
solution_view = ti.linalg.vector_view(solution, offset=offset, length=count)
```

`offset`、`length` 与 `stride` 是 host-known integer，在 field 按 canonical scalar
顺序 flatten 后解释。要求 `offset >= 0`、`length > 0`、`stride > 0`，最后一个选中
scalar 必须仍在 source extent 内；range 参数与 `indices` 互斥。compact field 上
`stride=1` 的 range 保留 direct dense-storage descriptor，可以直接绑定通过资格验证的
CUDA/Vulkan Graph Krylov boundary，不产生 gather/scatter。`stride>1` 仍是 device-staged
affine view，且绝不会报告为 direct contiguous storage。

`indices` 可以是一维 `ti.i32` ndarray 或 root-dense scalar field。构造 view 时会复制、
检查并冻结 index topology；index 必须非空、在 source scalar extent 范围内且唯一。
后续修改原始 indices 不会改变既有 view。该构造执行一次显式 host validation；vector
数值的 apply/solve 路径始终留在设备上。indexed scatter 只覆盖选中的 scalar entry，
其它 field entry 保持不变。

dense field 的执行路径由 provider capability 决定。对于
`operator.apply(input, out=output, alpha=1, beta=0)` overwrite 形式，当 provider 报告
`dense_storage_operands=True` 时，canonical compact full field 会直接绑定：

```text
dense field descriptor -> resolved range + submission lease -> provider operands
```

compiled-kernel 与 direct-dispatch compiled-Graph provider 在 CPU、CUDA、Vulkan 上均接受
direct field operand，其中包括有序 multi-dispatch Graph provider。fixed native CSR/BSR
provider 在 CPU 与 CUDA 上接受，Vulkan native sparse provider 仍使用 staging。direct
input/output 必须互不 alias，dtype 与 scalar extent 精确匹配，并且都能证明是 compact
scalar-flat mapping。

其它受支持的情况继续使用显式 device staging：

```text
dense field/view -> device pack or gather -> scalar ndarray provider ABI
scalar ndarray result -> device unpack or scatter -> dense field/view
```

这包括 indexed view、padded 或 non-compact field、generalized apply、`out=None`，以及
不属于 Graph Krylov 资格范围的 `SolvePlan.solve()` field 边界。每个 operator/plan 持有并
复用兼容的 staging ndarray；warm solve 不重新分配，转换只发生在 apply/solve 边界，不进入
Krylov iteration。field `out` 会在同步 API 返回前完成 unpack/scatter。

CUDA/Vulkan 上可录制的 f32 CG/PCG 对 canonical compact full-field 或 `stride=1`
scalar-range RHS、output 与 initial guess 提供更窄的 direct-binding 路径。Field 直接成为 solver Graph 的 runtime
argument；Graph preamble/epilogue 把边界值复制到一个 plan-owned recurrence ndarray，或从中
写回 Field。它会删除独立 pack/unpack submission、一次 completion sync 和两个 boundary
staging vector 中的一个，但不是 provider-native zero-copy：iteration solution 有意保持
ndarray backing，避免每次 Krylov update 重复 Field/SNode 寻址。indexed、strided、padded、
masked 或 scatter view 继续使用可复用 staging。

RHS/input 不能与 output 重叠；`initial_guess` 或 `addend` 可以与 output 是精确相同的 view，
非精确重叠会失败。稳定 raw field binding 只完成一次资格解析，并复用 implicit view、
transfer plan 或 Graph runtime binding；backend 支持时使用 native bulk transfer，其它 staged
layout 使用 compiled conversion Graph replay。

可以查询支持面和实际转换开销：

```python
capabilities = ti.linalg.vector_io_capabilities()
view_metadata = rhs_view.metadata
stats = plan.statistics()["vector_io"]
```

`VectorView.metadata["zero_copy_candidate"]` 只表示 full-field 的物理 layout 可无复制地
flatten；实际执行仍由 provider capability 与本次 operation 决定。`stats` 报告 direct
dense-field submission、staging build/reuse/reserved bytes、implicit view 与 transfer plan
的 build/reuse/eviction、native/Graph transfer submission、pack/unpack、indexed
gather/scatter、logical bytes、direct ndarray binding、direct Graph-solve boundary
submission/binding、completion sync 和合并的 operator sync 次数。
`execution_capabilities()["direct_dense_field_solve"]` 会区分能力是否支持、是否启用，以及
最近一次 solve 是否真正选中了完整 direct Field 边界。
`execution_mode="device_staged"` 表示支持 field API 且数值不经过 host；它不等同于
provider-native zero-copy。

## Stored operator 与 CG

fixed CSR/BSR matrix 可通过 `aslinearoperator()` 或
`LinearOperator.from_sparse_matrix()` 转换。operator 会强引用 matrix，并直接复用现有
pattern 与 numeric storage，不复制矩阵。

```python
import taichi_forge as ti

ti.init(arch=ti.cuda)

operator = ti.linalg.aslinearoperator(
    A,
    traits=ti.linalg.OperatorTraits.spd(),
)

plan = ti.linalg.experimental.SolvePlan(
    operator,
    method="cg",
    max_iterations=200,
    atol=1e-7,
    rtol=1e-5,
)
result = plan.solve(rhs)

if not result.converged:
    raise RuntimeError(result.termination_reason)
x = result.solution
```

mutable Eigen CSR/CSC matrix 继续由已有的 `SparseCG`、`SparseMINRES`、
`SparseBiCGSTAB` 与 `SparseSolver` API 支持。stored provider 只接受 fixed
CSR/BSR，从而让 topology、numeric generation 和 provider ownership 具有统一合同。

## 已编译 kernel provider

`LinearOperator.from_kernel()` 接受精确的 f32 ndarray ABI。topology 与 numeric 分离时，
签名为：

```python
@ti.kernel
def apply_diagonal(
    active_size: ti.i32,
    topology: ti.types.ndarray(dtype=ti.i32, ndim=1),
    numeric: ti.types.ndarray(dtype=ti.f32, ndim=1),
    x: ti.types.ndarray(dtype=ti.f32, ndim=1),
    y: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    for i in range(active_size):
        y[i] = numeric[i] * x[topology[i]]

operator = ti.linalg.LinearOperator.from_kernel(
    apply_diagonal,
    size,
    topology,
    numeric=numeric,
    traits=ti.linalg.OperatorTraits.spd(),
)
```

`size` 也可以写成 `(range_extent, domain_extent)`。矩形 provider 必须通过
`adjoint=` 登记独立的伴随 kernel，才能使用 `operator.adjoint()`：

```python
operator = ti.linalg.LinearOperator.from_kernel(
    forward_kernel,
    (rows, columns),
    topology,
    adjoint=adjoint_kernel,
    numeric=values,
)
```

forward 的 `active_size` 是 range extent，adjoint 的 `active_size` 是 domain
extent；两侧共同需要的尺寸必须存放在显式 topology resource 中。没有登记 adjoint
时会明确失败，不使用 autodiff、host materialization 或 self-adjoint trait 猜测。

未传入 `numeric=` 时，精确签名为 `(active_size, operator_data, x, y)`。所有 data
参数都是 scalar ndarray；topology 与 numeric 可以有各自的 scalar dtype，vector 必须是
f32。kernel 必须覆盖写入每个 output entry，且不能依赖 SNode tree。构造时编译一个
specialization，并把 topology/numeric 输入复制到 operator-owned snapshot。

## 可录制 Graph action

compiled-kernel provider 或满足资格的 compiled-Graph provider 可把 apply 操作公开为
recordable Graph action：

```python
input_arg = ti.graph.Arg(
    ti.graph.ArgKind.NDARRAY, "input", ti.f32, ndim=1
)
output_arg = ti.graph.Arg(
    ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1
)

builder = ti.graph.GraphBuilder()
builder.append_native(operator.graph_action(input_arg, output_arg))
graph = builder.compile()
graph.run({"input": x, "output": y})
```

`operator.graph_action()` 是 f32 zero-copy recording 边界。它既可追加到 Graph root，
也可追加到结构化 `while`、`if` 或 `switch` 使用的 `Sequential` body。provider-owned
topology、numeric 与 workspace snapshot 成为 compiled Graph 的 fixed binding；不会再
复制一份，也不会出现在 `Graph.run()` 参数中。显式声明的 root-dense Field state
继续绑定原 storage。外层 Graph 可保持原顺序录制 provider 的每个 dispatch，并与相邻
kernel 组合，因此迭代 body 不会为每次 operator apply 回到 Python。

symbolic input/output 使用一维 scalar vector ABI。runtime 可传入匹配的 scalar ndarray，
或 Graph runtime 能无复制 scalar-linearize 的 dense storage。input/output 必须能证明互不
alias。`adjoint=True` 要求显式登记 adjoint action。兼容的 values-only update 会在 launch
时 rebind，不需要重建 Graph。submission 会 pin 本次实际使用的 immutable numeric
generation，因此发布新 generation 时在途工作仍安全。topology、schema、state tree 或
runtime generation 变化仍会 fail closed，并要求新建 Graph。

当前 recordable 合同适用于 compiled-kernel provider、只含 direct dispatch 的
compiled-Graph provider、fixed-layout `block_diagonal()` container，以及 leaf 均可录制的
f32 scale/shift/sum/compose/adjoint 树。组合会递归 lowering 并保持 child dispatch 顺序。
block container 会在 submission 时从两个公开 vector binding 派生连续 domain/range subview；
这些 private view 不会成为调用方参数，不执行 staging 或 storage allocation，并且 container
嵌入其它 recordable composition 后仍保持该合同。有序 scale/sum subtree 会规范化为
weighted leaf：两项加权和只执行两个 provider action 与一个 in-place `axpby`，更多项也只
复用一条 scratch vector；`compose` 仍保持 operator 边界与有序 intermediate。需要 scratch
的路径从 Graph-owned bounded arena 取得 typed f32 temporary，因此 scratch 不会成为公共
runtime 参数，并发 submission 会使用独立 arena lane。memory report 会公开
planned/persistent temporary bytes。通用的
矩形/显式 adjoint 形式和 legacy 方阵 forward-only 形式都会保持 compiled Graph 原有的
有序 multi-dispatch sequence。`adjoint=True` 录制显式 adjoint Graph；legacy 方阵形式
不会推断 adjoint。含 indirect dispatch 的 compiled-Graph、stored sparse 与其它不支持的
provider 会明确失败，不会 materialize operator 或插入隐藏的 apply fallback。block-diagonal
录制只覆盖 leaf 可录制的固定连续布局；permutation、overlap 与 non-affine view 仍不支持。
provider recording 协议本身不是公开的自定义 native callback API。

只录制一个 action 并不保证性能提升。它的主要价值是把多个 operator action 与相邻
kernel 组合为更大的 Graph region，以摊销固定提交成本。应用应测量完整组合 workload，
不能假设包装一次 apply 必然更快。

## 已编译 Graph provider

`LinearOperator.from_graph()` 绑定已编译 Graph，其中动态 vector 参数必须命名为
`input` 与 `output`。其它每个 Graph 参数必须且只能分配一个角色：

```python
operator = ti.linalg.LinearOperator.from_graph(
    graph,
    size,
    fixed_i32={"active_size": size},
    topology={"row_offsets": row_offsets, "columns": columns},
    numeric={"values": values},
    workspace={"temporary": temporary},
    # 为每个 dependent SNodeTree 提供一个 live root-dense 代表 Field；
    # key 只是标签，不声明逐 Field 或逐 component 的访问能力。
    state={"coefficients": coefficients},
    traits=ti.linalg.OperatorTraits.spd(),
)
```

该 provider 是 f32 operator；`size` 可为整数方阵简写或 `(range, domain)`，并可通过
`adjoint=adjoint_graph` 登记具有相同 resource role schema 的独立伴随 Graph。它至少需要
一个 topology ndarray。topology、numeric data 与 workspace 会复制到 operator-owned
resource。

`state=` 是 snapshot 策略的显式例外。forward Graph 依赖集合中的每个 distinct
SNodeTree 都必须由该 mapping 中至少一个来自该 tree 的 live root-dense scalar、
Vector 或 Matrix Field 代表。若提供 adjoint Graph，它必须具有相同的 SNodeTree
依赖集合，并使用同一声明完成验证。Forge 保留 tree 的 storage identity 并直接绑定，
不发生 device copy；只要 layout 与 SNodeTree generation 不变，其内容可以在多次 apply
之间原位更新。mapping key 只是非空诊断标签，不单独匹配 Field anchor 或 component；
dependency 比较包含 tree id、generation 与 layout fingerprint，lifetime ownership
和外层 resource effect 仍以 tree 为粒度。同一 tree 的多个代表项不增加语义。

每个 dependent tree 必须整体为 pure-dense。tree 只要包含任一 sparse/dynamic
descendant 就会保守拒绝，即使 provider Graph 只访问其中的 dense sibling。
noncanonical Field storage、依赖 tree 漏报或多报也会在构造时明确失败。

CPU 把 Graph lowering 成 explicit sequence；CUDA/Vulkan 使用 compiled-Graph execution
合同。当已记录的 Graph runtime 规则要求时，backend capture/replay 可以使用 ordinary
Graph fallback，但不会改变数学 provider。Vulkan Graph replay 要求至少两个 dispatch；
单 dispatch Graph 仍正确执行，但 `operator.statistics()` 会将路径报告为
`ordinary_graph_fallback`。

通用和 legacy compiled-Graph provider 都可通过 `graph_action()` 导出保持顺序的 forward
dispatch；通用形式还可导出显式登记的 adjoint Graph。兼容的 numeric generation 会在
launch 时 rebind 到已编译的外层 Graph；所有 leaf 先完成两阶段验证，再统一替换 binding，
submission 则 pin 一个自洽的 generation snapshot。销毁或替换已声明 state 的
SNodeTree、改变 topology/schema、在 `ti.reset()` 后继续使用 action，或改变 owning
runtime 都会 fail closed；复用相同数值 tree id 不会恢复旧 action。

## 数学 trait

`OperatorTraits` 用 `None` 表示未知，用 `bool` 表示调用方的显式声明：

```python
traits = ti.linalg.OperatorTraits(
    self_adjoint=True,
    positive_definite=True,
    positive_semidefinite=True,
    singular=False,
)
```

`OperatorTraits.spd()` 是等价的便捷构造。CG/PCG 要求可信的
`self_adjoint=True` 与 `positive_definite=True`，并拒绝声明为 singular 的
operator。仅凭 shape、一次正 diagonal sample 或经验性乘积检查，不能建立这些性质。

trait 是 operator 后续使用的每个 numeric generation 都必须满足的合同。通过
`update_numeric()` 改变系数后，调用方必须确保已声明性质仍成立。结构上安全的组合会推导
能够证明的 trait；不能证明的性质保持 unknown。

## Apply 与组合

```python
y = operator.apply(x)
operator.apply(x, out=y)
y = operator.apply(x, alpha=2.0, beta=-0.5, addend=z)
y = operator @ x

B = 2.0 * operator
C = operator + B
D = operator.compose(B)       # operator(B(x))
E = operator.adjoint()
F = ti.linalg.block_diagonal((operator, B))
I = ti.linalg.identity(size, dtype=ti.f32)
S = operator.shifted(0.01)    # operator(x) + 0.01 * x
P = operator.parameterized_affine(
    alpha=1.0,
    beta=0.01,
    alpha_range=(1.0, 1.0),
    beta_range=(0.0, 0.1),
)
P.update_parameters(alpha=1.0, beta=0.02, expected_version=1)

# flat row-major 的 1x1、2x2、3x3 或 4x4 inverse block
M = ti.linalg.inverse_block_diagonal(
    inverse_blocks, block_size=3, assume_spd=True
)

inverse = ti.linalg.SmallBlockInverseBuilder(3, block_count).build(blocks)
M = ti.linalg.inverse_block_diagonal(
    inverse.inverse_blocks, block_size=3, assume_spd=True
)
```

通用形式为 `out = alpha * A(x) + beta * addend`。input/output alias 始终被拒绝；
`addend` 可以与 `out` 相同，以表达原地累加。当 `beta == 0` 时，`addend` 不会被验证或
读取。通用系数 lowering 支持 CPU，并在 CUDA/Vulkan 上支持 f32 ndarray operand。GPU 使用
device transform/scaled-add primitive，不发生 host readback；非 alias 路径不分配 N-sized
scratch。当 `addend is out` 时，一个持久 scratch 会在计算 `A(x)` 前保留旧 addend。GPU f64
与不支持的 storage layout 明确失败，不经 host fallback。

只有 provider 提供显式 adjoint apply 时才能调用
`adjoint()`；实现不会把 self-adjoint trait 当作 fallback。

scale、shift、sum 与 compose 在 CPU 上执行；对于 Program-bound f32 operand，也支持 CUDA
与 Vulkan。recordable shifted operator 会 lower 为 base provider action 加一个 in-place
`axpby`，不会再发射 identity provider，也不分配等长 temporary。standalone sum/compose
持有私有 persistent ndarray workspace；recordable 形式改用 Graph-owned typed temporary。
等长的纯 compose chain 会按精确 leaf/adjoint 顺序扁平化，并在 destination 与一条 scratch
vector 之间交替，因此 depth 2 以后 Graph temporary 始终只有一条 vector，不再随语法树深度
增长；rectangular 或 mixed-extent chain 继续使用保守的 nested lowering。组合嵌入
`graph_action()` 时可直接绑定 compact Field；standalone 组合仍使用上文可复用的边界 staging。
`parameterized_affine()` 会把 alpha 与 beta 作为一个 immutable generation 原子发布。构造时
必须声明的系数闭区间同时也是 trait 证明边界：只有 SPD/PSD 在整个区间上都成立时才会保留。
对于 f32 operator，系数和区间端点会先规范化为 f32 kernel 实际执行的值；超出 f32
表示范围的有限 Python 值会明确失败，正下界若舍入为零也不会错误保留 SPD 结论。
更新采用乐观 `expected_version` 检查；缓存 Graph action 会 rebind 新的二标量 snapshot 而不
重建，in-flight submission 继续 pin 它实际提交的旧 snapshot。snapshot 由原生 affine
operator 直接持有；更新不再分配或上传虚拟 device parameter array。

公开 `identity()` 仍只支持 CPU。通用 `block_diagonal()` 还支持 CUDA/Vulkan f32 standalone
apply，但要求每个 leaf 都提供经过资格验证的 direct affine dense storage。leaf 按顺序直接
绑定连续 domain/range subview，不执行 gather、scatter、全向量 staging 或 N-sized temporary。
当所有 leaf 都可录制时，同一 fixed-layout container 也是公开 Graph action，并可直接作为
SolvePlan operator 或 fixed-linear preconditioner。private subview 每次外层 submission 只派生
一次，因此多迭代 SolvePlan 可摊销绑定成本，同时保持单 ticket 且不产生 block-sized scratch。
这是 dispatch composition，不是 kernel fusion。permutation、overlap、non-affine 或不具备
recordable 资格的 layout 会明确失败。GPU f64 composition 仍不支持。

`inverse_block_diagonal()` 是面向常见物理布局的 recordable fixed-linear preconditioner
helper。调用方提供已经求逆、row-major 的 f32 block，并必须声明 `assume_spd=True`；Forge
不会回读、求逆、正则化或推断其 SPD 性质。CPU、CUDA、Vulkan 支持 block size 1 到 4。
每种尺寸使用静态专门化 kernel，并从 scalar index 推导 row/block offset；operator 只快照
一个常数大小的 topology word，不再持有随 vector 长度增长的 offset table。兼容的
values-only update 只复制 inverse values，并使用普通 immutable-generation rebind 合同，
因此同一个单系统或 batched PCG Graph 可以安全消费更新后的 preconditioner。

`SmallBlockInverseBuilder` 可在 device 上构造上述 f32 row-major inverse block，固定 block
size 为 1 到 4。它同时提供 direct `build()` 与单 dispatch `graph_action()`。带 partial pivot 的
Gauss-Jordan 会应用调用方给出的非负对角 regularization；`pivot_tolerance` 会乘以
regularized block 的最大绝对系数，因此 block 整体缩放不会改变状态判断。无法由 f32 kernel
表示的控制值会在构造时失败。builder 为每个 block 写入 device-resident status：0 表示成功、
1 表示输入非有限、2 表示 pivot 奇异或病态；失败 block 的输出为零。
Forge 不自动读回 status，也不推断 SPD；调用方应按自身策略检查 status，并独立作出
`assume_spd` 断言。

### 固定拓扑隐式求解参考

[`implicit_linear_operator.py`](../../python/taichi_forge/examples/simulation/implicit_linear_operator.py)
是一个 headless CPU/CUDA/Vulkan 参考路径。它保持 spring connectivity 不变，逐步发布新的
A/M numeric generation，在 device 上组装变化的 2x2 system block 及其 inverse，通过
`PreconditionerPlan.update()` 显式批准重建后的 inverse-block action，并复用同一个 PCG
`SolvePlan`：

```bash
python -m taichi_forge.examples.simulation.implicit_linear_operator \
  --arch cuda --nodes 2304 --steps 120 --telemetry
```

示例只使用公开 primitive。CUDA/Vulkan 通过一张缓存 Graph ticket 提交完整求解；compact
Vector Field 的边界处理位于已录制 action 内，不产生独立 pack/unpack submission。CPU 使用
文档所述的同步 completed-ticket 路径。`result()` 是唯一 terminal materialization 点，报告的
iteration 是逻辑早停位置；Vulkan 还可能编码有界 masked tail，该部分通过 opt-in telemetry
可见，但不会被记作有效 solver work。逐 block inverse status 在时间步内始终留在 device，
仅由示例末尾的资格报告读取一次。

## SolvePlan 与 SolveResult

`SolvePlan` 会跨多次调用保留 operator、solver state 和 persistent workspace。支持：

- `method="cg"`：面向 SPD 系统、使用 identity preconditioner 的 conjugate gradient；
- `method="pcg"`：fixed CSR 上使用 `"jacobi"`、fixed BSR 上使用
  `"block_jacobi"`，或使用一个可信 SPD `LinearOperator`/`PreconditionerPlan` 应用 fixed-linear
  近似逆的 PCG；
- `method="minres"`：面向允许不定的 square self-adjoint 系统，使用 identity 或 SPD
  preconditioner 的 MINRES；
- `method="bicgstab"`：面向一般 square 系统、使用 identity 或 fixed-linear 右
  preconditioner 的 BiCGSTAB；
- `method="gmres"`：面向一般 square 系统、使用 identity 或 fixed-linear 右
  preconditioner 的 restarted GMRES；
- `method="fgmres"`：使用有限 variable-linear 右预条件 action table 的 restarted
  flexible GMRES。

```python
result = plan.solve(rhs, initial_guess=x0, out=x)
print(result.iterations, result.residual_norm, result.termination_reason)
stats = plan.statistics()
```

在 CUDA/Vulkan 上，若 f32 CG/PCG plan 的 A/M provider 均可录制且选择了取得资格的
`device_convergent` policy，也可以由 plan 持有一个缓存的单 action Graph，并异步提交
完整求解：

```python
plan = ti.linalg.experimental.SolvePlan(
    operator,
    method="pcg",
    preconditioner=preconditioner,
    submission_workspace_lanes=2,
    submission_workspace_saturation="raise",
)
submission = plan.submit(rhs, out=x, telemetry=True)
# 此处尚未读回 terminal packet
submission.wait()                 # 只等待 backend completion
result = submission.result()      # 一次显式 terminal materialization
print(submission.workspace_lane, result.iterations)
```

在 CUDA/Vulkan 上，`SolvePlanSubmission` 会把 plan、runtime operand、output、terminal
packet 与缓存 Graph 保留到 completion。`done()`/`wait()` 不读 terminal；`result()` 只读
一次并缓存 immutable `SolveResult`；`telemetry()` 暴露同一次 Graph submission 的遥测。无 initial guess 与显式
initial guess 会分别缓存，每个已物化 variant 有自己的 lane pool。
`submission_statistics()` 报告 variant、lane、persistent/transient bytes、提交/失败、
telemetry request 与 terminal materialization。

GPU 便捷路径不新增 solver/backend primitive；它严格等价于把 `graph_action()` 放入缓存
Graph 后调用 `Graph.submit()`，因此资格与失败边界仍是相同的 recordable f32 CG/PCG
device-convergent 合同。`submission_workspace_lanes=N` 为每条在途 GPU lane 提供独立
Krylov storage；额外 lane 会线性增加 persistent memory，但不承诺 kernel 的物理并行。
CPU 则同步执行既有精确 native `solve()` 并返回已经完成的 `SolvePlanSubmission`：只使用
lane 0，没有 device terminal packet/Graph telemetry，也不会复刻 GPU 风格的 masked capacity。

### 完整 SolvePlan Graph action

若 f32 CG/PCG plan 的 operator 与可选 fixed-linear preconditioner 均可录制，
可以把完整求解直接内联进外层 Graph。这里得到的是 structured action，不会在
Graph 内部再次调用 `Graph.run()`：

```python
rhs_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "rhs", ti.f32, ndim=1)
x_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "x", ti.f32, ndim=1)

solve = plan.graph_action(rhs_arg, x_arg, name="inner_pcg")
builder = ti.graph.GraphBuilder()
builder.append_native(solve)
graph = builder.compile(workspace_lanes=2, workspace_saturation="raise")

terminal = solve.allocate_terminal()
ticket = graph.submit({"rhs": rhs, "x": x, **terminal.arguments})
ticket.wait()
result = terminal.snapshot()
print(ticket.workspace_lane)
```

`solve.terminal.state` 是 i32[4] symbolic resource，依次保存 status、逻辑迭代数、
breakdown code 和完成标记；`solve.terminal.metrics` 是 f32[4] resource，保存初始
残差平方、最终残差平方、reference norm 平方和 effective tolerance 平方。外层
structured region 可以直接在 device 上消费这些 symbolic resource。Forge 不会隐式
读回 terminal；`terminal.snapshot()` 是显式 host 边界，应在外层
`SubmissionTicket` 完成后调用。

`solve.statistics()` 会在相同边界归属执行信息，不对 Krylov loop 插桩。每次包含该 action 的
外层 Graph 成功执行，`enclosing_graph_submissions` 增加一次；`done()`/`wait()` 观察到 ticket
完成时，`observed_completions` 增加（已经完成的 submission 会立即记录），这一过程不读取
terminal storage。`terminal.snapshot()` 执行显式读回时，每个 packet 的 `terminal_snapshots`
与 `terminal_iteration_sum` 恰好增加一次。action 位于 outer loop 内时，packet 描述本 packet
最后一次 invocation；outer region invocation 与 backend replay 数仍以 ticket region telemetry
为准。`plan.graph_action_statistics()` 汇总该 plan 创建的全部 action，`solve.statistics()` 返回
单个 action 的记录。

该 action 可以追加到 outer `Sequential`，并与最多另外七个有序 inner `while` action
共同组成 depth-2 Graph。取得资格的结构是一个 outer `while`，其 body 顺序包含一至八个
leaf inner `while`，inner 之间允许普通 dispatch/native action；各 inner control resource
必须互不别名，完整层级最多编码 4,096 个 action。CPU 使用精确 nested host control；
通过显式 setup probe 的 CUDA Driver API 12.4+ runtime 使用 device-updatable kernel-node
group，较旧或未通过资格的 runtime 使用 Forge 自带、与 CUDA 版本无关的 bounded 双 gate
Graph；Vulkan 使用 bounded conditional replay。CUDA/Vulkan 会把完整层级保持在一个
ticket 中，两层之间不做 host readback。outer suffix kernel 可以读取每个 solve 的 terminal
state，在推进 outer counter 前把各自迭代数写入 device trace。这些 GPU 路径仍保留
bounded 静态拓扑，不宣称 exact dynamic command termination。

Krylov vector 与 recurrence scalar 都是 compiled Graph workspace lane 私有且地址稳定的
storage。默认单 lane 保留安全的 completion-fence 串行化；`workspace_lanes=N` 会按需惰性
物化最多 `N` 份独立 storage。自动 round-robin 优先选择已经完成的 lane，
`workspace_lane=i` 可以固定 lane。所有 lane 忙时，`workspace_saturation="wait"` 等待，
`"raise"` 立即报错。每个实际物化的额外 lane 都线性增加 persistent memory；
`Graph.execution_stats().memory` 会报告 materialized/busy 数量、acquisition、wait、
saturation error、persistent bytes 与 reuse。

workspace lane 消除的是已排队 solve 之间的 storage-reuse 等待，不会创建 backend stream，
也不承诺 GPU 物理并行。若需要独立 submission stream 或并发 host setup，仍应编译相互独立的
Graph。runtime operand 可使用合格的一维 scalar ndarray、完整 dense Field，以及 compact
scalar-flat `vector_view(..., offset=..., length=..., stride=1)` 区间。RHS 与 output 必须可
证明互不相交；独立 initial guess 必须与 output 不相交，或与它是完全相同的 view。

对于系数不变的兼容路径，fixed-linear preconditioner 可以直接作为 operator 传入，而不是
应用回调：

```python
plan = ti.linalg.experimental.SolvePlan(
    operator,
    method="pcg",
    preconditioner=inverse_operator,
    max_iterations=200,
    atol=1e-7,
    rtol=1e-5,
)
```

`inverse_operator` 必须把 `operator` 的 range 映射回 domain，且 dtype 相同；它必须
携带可信的 self-adjoint、positive-definite 与 nonsingular trait。CPU 接受 operator
execution plan 支持的 provider 组合；CUDA/Vulkan 接受下方支持矩阵列出的可录制
compiled-kernel、compiled-Graph 或合格 composition 组合。每次 solve 会成对 pin 它们的
topology 与 numeric
generation，不调用 host callback，也不执行 backend fallback。

### MINRES

`method="minres"` 使用与 CG/PCG 相同的 `LinearOperator` 与 lifecycle 合同，但要求可信的
`self_adjoint=True` trait，而不要求 operator 为正定：

```python
operator = ti.linalg.LinearOperator.from_sparse_matrix(
    A,
    traits=ti.linalg.OperatorTraits(
        self_adjoint=True,
        singular=False,
    ),
)
plan = ti.linalg.experimental.SolvePlan(
    operator,
    method="minres",
    max_iterations=300,
    atol=1e-8,
    rtol=1e-5,
    execution_policy="host_check_every_k",
    check_interval=4,
)
result = plan.solve(rhs)
```

CPU 为所有兼容的 CPU operator provider 提供 identity-preconditioned `f32/f64` MINRES。
CUDA 与 Vulkan 为 fixed CSR/BSR 和 compiled provider 提供 `f32`
identity-preconditioned MINRES。在 CUDA/Vulkan 上，fixed CSR 可选择 `"jacobi"`，block
size 为 2、3、6 或 12 的 fixed BSR 可选择 `"block_jacobi"`，也可直接提供可信的
device-native fixed-linear `LinearOperator` 或 `PreconditionerPlan`。MINRES preconditioner
必须 self-adjoint、positive-definite 且 nonsingular；应用仍需保证所选 scalar Jacobi
inverse 满足该数学合同。

MINRES 拒绝声明为 `singular=True` 的 operator，不提供 MINRES-QLP，也不提供兼容 singular
系统的 minimum-length 语义。显式存储的对称 matrix 必须完整且一致地保存两个 half。
即使启用 preconditioner，terminal status 也使用原系统的真实 residual 进行资格判断。

一个 CUDA/Vulkan MINRES plan 持有九个长度为 `n` 的持久 `f32` vector 与 144 bytes
持久 scalar state。该数字不包含调用方持有的 operator values、preconditioner resource、
RHS/output array、backend handle 与原生 replay object；具体配置应通过 `statistics()`
检查完整 plan/provider telemetry。

### BiCGSTAB

`method="bicgstab"` 为 square 非对称 operator 提供固定内存 Krylov 路径。它可使用
identity preconditioner，也可在右侧应用 fixed-linear
`LinearOperator`/`PreconditionerPlan`：

```python
plan = ti.linalg.experimental.SolvePlan(
    operator,
    method="bicgstab",
    preconditioner=inverse_operator,
    max_iterations=300,
    atol=1e-8,
    rtol=1e-5,
    execution_policy="host_check_every_k",
    check_interval=4,
)
result = plan.solve(rhs)
```

preconditioner 必须把 operator 的 range 映射回 domain、dtype 相同、为 fixed-linear，
并且不能声明为 singular。它与 PCG/MINRES preconditioner 不同，不要求 self-adjoint
或 positive-definite。右预条件使 terminal qualification 保持在原系统上：只有计算
真实 residual `b - A x` 后，求解才会报告收敛。

CPU 支持 `f32/f64` host-action provider。CUDA/Vulkan 支持 `f32` fixed CSR/BSR
以及 compiled kernel/Graph provider。compiled A/M provider 采用直接原生提交；
identity-preconditioned fixed stored A 可复用 CUDA Graph 或 Vulkan command-sequence
iteration chunk。系统不会为了适配其它 backend 路径把 provider 复制到 host。

device identity plan 持有六个长度为 `n` 的持久 vector；右预条件额外持有两个
preconditioned direction vector。两种配置均持有 112 bytes scalar state。
`statistics()` 精确报告 A/M apply、dot product、vector update、
logical/executed/wasted iteration、host observation、replay、workspace bytes 与
`preconditioning_side`。`SolveResult.breakdown_reason` 将 `nonfinite`、`rho`、
`alpha_denominator`、`omega_denominator`、`omega` 与普通 max-iteration
终止区分开。

BiCGSTAB 在 nonsingular 问题上仍可能停滞或 breakdown。它是低存储一般系统选项，
不能替代经过资格验证的 GMRES-family 方法在稳定性上的作用。

### Restarted GMRES

当 BiCGSTAB 的短 recurrence 不够稳健时，`method="gmres"` 为一般 square operator
提供有界内存的 Krylov 路径。`restart` 是 plan 构造参数，只能取 `8`、`16` 或
`32`，默认值为 `16`。

```python
plan = ti.linalg.experimental.SolvePlan(
    operator,
    method="gmres",
    preconditioner=inverse_operator,
    restart=16,
    max_iterations=160,
    atol=1e-8,
    rtol=1e-5,
    execution_policy="host_check_every_k",
    check_interval=16,
)
result = plan.solve(rhs)
```

每个 Arnoldi step 都执行两遍 classical Gram-Schmidt（CGS2）。正交化先通过
multi-dot reduction 一次生成一组内积，再执行 fused projection，而不是为每个 basis
vector 发起一次由 host 观察的 dot product。每个 cycle 在 device 上更新 Givens rotation，
并使用原系统真实 residual `b - A x` 判断最终状态。`happy_breakdown` 单独计数，
不会与 `arnoldi_breakdown`、`orthogonalization_failure`、
`hessenberg_singular` 或 `nonfinite` 混为一类。

CPU 支持兼容的 `f32/f64` host-action provider；CUDA/Vulkan 支持 `f32` fixed
CSR/BSR 与 compiled kernel/Graph provider。可选 preconditioner 必须是 nonsingular
fixed-linear map，把 operator 的 range 映射回 domain，并应用在右侧；它不需要 PCG
或 MINRES 所要求的 SPD trait。`method="gmres"` 有意只接受 identity 或 fixed-linear
preconditioning；受支持的 variable-linear schedule 应使用 FGMRES。

plan 会预分配连续的 `(restart + 1) * n` basis。device identity plan 持有
`restart + 5` 个长度为 `n` 的持久 vector；右预条件再增加一个 vector。
Hessenberg、Givens、least-squares、multi-dot partial 与 terminal state 也都使用
持久内存，warm solve 不分配 transient solver workspace。`statistics()` 报告
`basis_vector_count`、`basis_reserved_bytes`、
`persistent_vector_reserved_bytes`、`persistent_scalar_reserved_bytes`、
精确的 A/M、dot、multi-dot、vector-pass 计数、restart cycle、
logical/executed/wasted iteration 与 replay 活动。应用在选择更大的 restart 前应检查
这些数据。

对于使用 identity preconditioner 的 fixed stored operator，CUDA Graph 与 Vulkan
command replay 覆盖完整 restart cycle。compiled provider 与右预条件 plan 保持相同
数值实现，但采用 direct native submission。CUDA 支持
`host_check_every_k` 且要求 `check_interval == restart`；Vulkan 支持该策略与
`fixed_budget_masked`。host 观察 terminal state 前会提交完整 cycle，因此最多可能执行
`restart - 1` 个 inactive tail step。更大的 restart 可能改善困难系统的收敛，但也会
同时扩大 basis 内存、cycle 工作量和最坏情况下的 inactive tail。

### 使用 variable-linear action table 的 FGMRES

`method="fgmres"` 接受 `behavior="variable_linear"` 的 `PreconditionerPlan`。plan
包含 1 到 32 个 linear action 的有限表；这一有界结构使 allocation、generation pinning
和 backend execution 均可显式审计，同时不会把 Python callback 引入 Arnoldi step：

```python
schedule = ti.linalg.experimental.PreconditionerPlan(
    operator,
    (inverse0, inverse1, inverse2),
    method="external_multilevel_cycle",
    behavior="variable_linear",
    selection="cyclic",
).setup()

plan = ti.linalg.experimental.SolvePlan(
    operator,
    method="fgmres",
    preconditioner=schedule,
    restart=16,
    max_iterations=160,
    atol=1e-8,
    rtol=1e-5,
    execution_policy="host_check_every_k",
    check_interval=16,
)
result = plan.solve(rhs)
```

solve-global scheduled inner slot `k` 选择 `actions[k % len(actions)]`，restart 不会重置
调度。CPU 的 scheduled slot 与 logical slot 一致；masked GPU execution 在较早发生逻辑
终止后仍可能调度 inactive tail slot，因此 telemetry 会把 action selection 与 executed
iteration 同 logical iteration 分开报告。每次 solve 进入时会同时 pin target 和所有
action generation。每个 action 都必须属于同一 Program、使用 solver dtype、把 operator
range 映射回 domain，并且不能声明为 singular。

CPU 支持兼容的 `f32/f64` host-action provider；CUDA/Vulkan 支持 `f32` 兼容 fixed
stored 与 compiled kernel/Graph provider。FGMRES 复用 GMRES 的 CGS2 Arnoldi、真实
residual 终止、restart 取值和 GPU observation policy，但会把每个预条件 basis vector
保存在持久 `Z` basis 中。额外预留量为 `restart * n * sizeof(dtype)` bytes，并通过
`preconditioned_basis_vector_count` 与 `preconditioned_basis_reserved_bytes` 暴露。
当 `restart=8` 时，经过资格验证的 CPU/device plan 分别持有 20/22 个 solver 持久
vector；该数字不包含 operator、action、RHS/output、backend 与 replay resource。

variable-action FGMRES 当前采用 direct native submission。只有 action table 中每个
action 都具备经过资格验证的 capture/binding 合同后，系统才会声明 CUDA Graph 或 Vulkan
command replay；当前 `solver_replay_unavailable_reason` 返回
`variable_action_capture_contract_unavailable`。API 不支持 nonlinear preconditioner、
Python iteration callback、自动 restart 选择、block GMRES 或领域 outer-solver policy。
把 variable-linear plan 传给 CG、PCG、MINRES、BiCGSTAB 或普通 GMRES 会在 plan 构造
阶段失败。

### 当前未支持边界

`0.6.2` 数值工具合同有意不提供：

- nonlinear、依赖 residual、adaptive 或 Python callback 驱动的 preconditioner；
- 自动 restart 选择、block/multi-RHS Krylov、recycling、deflation、pipelining 或
  communication-avoiding GMRES 变体；
- MINRES-QLP、singular minimum-norm/minimum-length 保证或自动 nullspace 处理；
- GPU `f64` GMRES-family 执行、GPU f64/block-diagonal composition 或通用 GPU
  `alpha/beta/addend` apply；
- variable-action CUDA Graph/Vulkan command replay 或 single-system 异步 solve
  submission；以及下文已资格化 CUDA/Vulkan 范围之外的 device-convergent 执行；
- dynamic-topology solve plan、ragged batch 或透明 host fallback；
- 内建 IC/ILU/AMG、multigrid hierarchy 构建、Schur/field split、domain
  decomposition、离散、contact/KKT policy 或 nonlinear outer iteration。

独立 fixed-size batched CG/PCG 与 block Krylov/multi-RHS solve 是不同合同。不受支持的
backend/provider/policy 组合会在构造或 capability 校验阶段失败；系统不会通过更换
provider、execution policy 或数学问题来近似执行。

## PreconditionerPlan 生命周期

需要更新系数、显式复用或审计来源时，应使用 `PreconditionerPlan`：

```python
preconditioner = ti.linalg.experimental.PreconditionerPlan(
    operator,
    inverse_operator,
    method="external_block_inverse",
).setup()

z = preconditioner.apply(r)
pinned = preconditioner.pin()

# 应用先通过各自 provider 发布新的 numeric generation。
operator.update_numeric(next_a, expected_topology_version=1,
                        expected_numeric_version=3)
inverse_operator.update_numeric(next_m, expected_topology_version=1,
                                expected_numeric_version=7)
preconditioner.update()  # 声明 next_m 由当前 operator generation 重建

pcg = ti.linalg.experimental.SolvePlan(
    operator, method="pcg", preconditioner=preconditioner
)
```

`built_from_operator_stamp` 记录 action 实际由哪个 operator generation 构建；
`accepted_target_stamp` 记录它当前被批准服务于哪个 generation。target 更新后 plan 默认
stale。若算法允许 lagged preconditioning，可在 action 未改变时显式调用
`preconditioner.update(accept_reuse=True)`；此操作只更新 accepted stamp，不改写来源。
action 已改变时必须使用普通 `update()`。

`pin()` 同时保留精确 target 与 action generation，返回的 `PreconditionerSession` 可在后续
generation 发布后继续安全应用旧 generation。fixed-linear session 使用
`apply(r, out=None)`；variable-linear session 可通过 `iteration=k` 选择该 scheduled
slot 对应的 pinned cyclic action，默认 `iteration=0` 选择第一个 action。`metadata`
暴露 provenance/compatibility stamp；`PreconditionerPlan.statistics()` 暴露 setup、
rebuild、reuse、stale rejection、schedule update success/failure 以及 approved
generation 的 publish/retire/release 计数，`SolvePlan.statistics()` 另行报告 action
selection 与 schedule wrap。setup/update 在 host 边界执行；session apply 与 solver
iteration 只调用 native `OperatorAction`，不执行 Python callback。

已经 setup 且 approved action 可录制的 fixed-linear plan 可直接由 recordable PCG 与 batched
PCG 消费。Graph recording 不会绕过 plan 生命周期：每次 launch 会原子验证并 pin 已批准的
target/action generation，异步 ticket 保留这一精确组合直到完成。stale target、未批准的
action generation 或不可录制 action 都会明确失败。variable-linear action table 仍是 FGMRES
的 direct native 能力，不会被提升为 Graph replay。

对于 variable-linear table，`update(accept_reuse=...)` 可以接收一个应用于所有 action 的
boolean，也可以为每个 action 分别提供 boolean。任何 generation 发布前都会验证完整的
next table；一个 stale 或不兼容 action 会拒绝整个 update。已经 pin 旧 table 的 solve
继续持有该 immutable snapshot。`fixed_linear` 由文档列出的 PCG、MINRES、BiCGSTAB 与
GMRES consumer 支持；`variable_linear` 仅由 FGMRES 支持。`nonlinear` 仍只有描述能力，
并返回结构化 unsupported reason。内置 Jacobi/block-Jacobi 使用同一 native
setup/update/pin 生命周期；其 provider 构造仍通过 `preconditioner="jacobi"` 或
`"block_jacobi"` 选择。

`rhs`、`initial_guess` 和 `out` 必须与 operator 的 dtype、scalar extent 一致，并属于当前
runtime。未提供 `out` 时会创建结果 ndarray；未提供 `initial_guess` 时结果初始化为零。
RHS/output alias 会被拒绝。

`SolveResult` 同时包含 solution 与 terminal snapshot：status code、termination reason、
convergence/breakdown/max-iteration flag、iteration count、初始与最终 residual norm、两个
tolerance、relative reference norm 和 effective tolerance。CG、PCG、MINRES、
BiCGSTAB、GMRES 与 FGMRES 使用：

result record 本身是 frozen 的；其中的 `solution` ndarray 仍可由调用方写入。

```text
||b - A x||_2 <= max(atol, rtol * ||b||_2)
```

`statistics()` 返回 backend-neutral plan/provider/workspace counter；它是诊断信息，
不属于数值结果。

### GPU 执行策略

`execution_policy` 控制 GPU solve 在何时由 host 观察收敛状态：

```python
plan = ti.linalg.experimental.SolvePlan(
    operator,
    method="cg",
    max_iterations=200,
    atol=1e-7,
    rtol=1e-5,
    execution_policy="host_check_every_k",
    check_interval=8,
)
```

- CPU 默认使用 `"host_each_iteration"`；stored f32 CG/PCG 也接受显式
  `"bounded_convergent"`，其内部仍保持同一原生循环。
- CUDA fixed stored f32 CSR/BSR CG/PCG 默认使用 `"bounded_convergent"`。
  `bounded_mode="auto"` 会在资格满足时选用 device 侧精确终止的 CUDA conditional
  Graph；原生路径不可用时改用带 host check 的可复用 Graph chunk。显式
  `"host_each_iteration"` 仍作为关闭自动路径的选项。
- 具备原生 replay 资格的 CUDA fixed stored f32 MINRES/BiCGSTAB 默认使用
  `"host_check_every_k"`，且 `check_interval=4`。
- CUDA 上满足资格的方形 matrix-free Kernel/Graph plan 也会自动选择
  `"host_check_every_k"`：CG、Kernel-PCG、MINRES 与 BiCGSTAB 使用 K=4，
  GMRES/FGMRES 使用 `check_interval == restart`。分块执行把 recurrence scalar
  保留在 device 上，每个 chunk 只读取一次 terminal snapshot。非 GMRES 方法仍可
  显式选择 K=8；在支持的组合上也可显式使用 `"host_each_iteration"`。CUDA
  BiCGSTAB 的 compiled Graph provider 默认保留 `"host_each_iteration"`，因为短程收敛
  workload 上 K=4 没有表现出稳定收益；调用方仍可显式 opt in。
- CUDA compiled-kernel f32 CG/PCG 在 conditional Graph 可用时，可显式请求
  `execution_policy="device_convergent"`。通用结构化 Graph 会录制 A action；PCG 还会录制
  fixed-linear compiled-kernel M action，在 device 上保持 recurrence control，并且每次 solve
  只读取一个 terminal packet。vector update 保持并行 range kernel；dot product 使用由
  plan 持有 fixed partial buffer 的两级 shared-block reduction。该组合完成 correctness
  资格，但标记为 `qualification_scope="explicit_only"`；自动默认仍采用 K=4
  `"host_check_every_k"`，因为 Graph 构造/capture 与 first execution 的成本需要由多次
  warm solve 摊薄。
- 当 A 与 fixed-linear M action 都满足录制资格时，recordable compiled-Graph f32 PCG
  会自动选择 `"device_convergent"`。provider 的每个有序 dispatch 都直接消费符号化 Krylov
  input/output，因此 canonical compact Field 或连续 Field range 不会增加 SolvePlan
  pack/unpack submission。recordable compiled-Graph CG 也可显式请求 device-convergent，
  但在尚无普遍 latency 优势时保留既有 host-check 默认策略。
- recordable f32 scale/sum/compose operator 用于 CUDA 或 Vulkan CG/PCG 时，会自动选择
  `"device_convergent"`。这是 composed provider 唯一满足资格的 GPU solve policy：系统
  不会用 host-check 路径替换 standalone apply，也不会在 Krylov 内部搬运 Field/ndarray
  vector。CUDA 在精确逻辑轮次停止；Vulkan 报告精确逻辑 stop，但可能执行 inactive encoded
  tail。两个 backend 每次 solve 都只读取一个 terminal packet；structured-Graph 路径不可用
  时明确失败，不更换 policy。
- CUDA GMRES/FGMRES 默认使用 `"host_check_every_k"`，并要求
  `check_interval == restart`。stored identity-preconditioned GMRES 会录制可复用的
  restart-cycle Graph；FGMRES 与其它不可录制的 provider 组合保持 direct submission。
- Vulkan 上可录制的 compiled-kernel f32 CG/PCG 与 recordable compiled-Graph PCG 默认使用
  `"device_convergent"`。通用结构化 Graph 会录制 A action；PCG 还会录制
  fixed-linear recordable M action。compact device-control plan 把 recurrence
  state 保留在 device 上，不做逐 iteration host observation。超过单一 command plan
  容量的预算会按最多 64 轮的有界 chunk 执行，每个 chunk 只观察一次 terminal state。
  Vulkan kernel-profiler 或 dispatch-cache mode 会关闭 command replay；自动策略此时
  使用 `"host_check_every_k"`，显式 `"device_convergent"` 请求则带 capability reason
  失败。
- 其它具备 replay 资格的 Vulkan fixed stored 或 compiled f32
  CG/PCG/MINRES/BiCGSTAB plan 默认使用 K=4 的 `"host_check_every_k"`；stored
  identity-preconditioned GMRES 使用同一默认策略，并要求
  `check_interval == restart`。不属于可录制 provider CG/PCG 范围的
  matrix-free 方法使用 K=4 或 restart 大小的 host check。对于明确需要消耗完整
  iteration 预算的 workload，仍可显式选择 `"fixed_budget_masked"`。这些策略均支持
  `atol`、`rtol` 及其组合后的 effective tolerance。

对于 fixed stored f32 CSR/BSR，CUDA 的 `host_check_every_k` 以及 Vulkan 的
`host_check_every_k`/`fixed_budget_masked` 会把受支持的 CG/PCG/MINRES 与
identity-preconditioned BiCGSTAB iteration chunk 与 identity-preconditioned GMRES
restart cycle 录制为可复用的原生执行序列。
可录制的 CG/PCG/MINRES 组合包括 identity、stored Jacobi 和 stored block-Jacobi
preconditioner。首次兼容执行建立 CUDA Graph 或 Vulkan command sequence；相同
topology、workspace 与 output binding 的后续执行直接 replay。仅更新 matrix values
不会重录 identity-preconditioned GMRES 序列；受支持的 preconditioner refresh 会保留
现有 CG/PCG/MINRES 序列。更换 output ndarray、改变 topology/schema 或重建 runtime
会使旧序列失效并安全重建。

对于资格满足的 CUDA stored CG/PCG，conditional Graph 持有完整 recurrence loop，
并在每个逻辑 iteration 后于 device 侧判断收敛。每次 solve 仍保留一次初始状态和一次
terminal state 观察，但迭代内部不再执行逐轮 host scalar reduction 或隐式同步。
Graph 在精确的逻辑 iteration 上终止，因此不会产生 masked tail work；只要 topology、
workspace 与 output binding 保持稳定，persistent plan 就会复用同一 executable。

对于资格满足的 CUDA/Vulkan recordable compiled-kernel、compiled-Graph 或 composition
CG/PCG，结构化
Graph 持有完整 recurrence region。Vulkan 使用 compact device masking，而不是动态截断
command stream，因此逻辑收敛轮数精确，但已编码 block 可以包含 inactive tail。provider
action、dense ndarray workspace、predicate、counter、status 与 reduction buffer 均使用固定
runtime binding。canonical compact full Field 使用上文 Graph preamble/epilogue binding；其它
受支持 Field 使用独立 staging。两条路线都不会在 Krylov iteration 内搬运 Field/ndarray 值。

FGMRES action table 在两个 GPU backend 上都使用 direct native submission；系统不会为
variable action schedule 静默复用 identity-GMRES replay 路径。

除可录制 provider CG/PCG 的 device-convergent 范围外，compiled-kernel 与
compiled Graph A/M provider 的外层 solver chunk 均按 direct submission 执行。这个外层
recurrence 边界与 provider 自身的执行方式相互独立：
compiled Graph apply 使用 provider 持有的 compiled Graph plan，compiled-kernel apply
使用普通 compiled kernel launch。满足 replay 资格的多 dispatch Graph 会在 CUDA/Vulkan
上 record/replay；不满足资格的 plan 保留普通执行并报告 backend path。Vulkan 单 dispatch
Graph 会有意保留普通路径，因为为其录制不会带来有效的提交合并。solver 不会再把任一
compiled Graph provider 嵌套捕获进另一层 Graph，也不会通过 host staging 或替换 provider
获取 replay。

`statistics()` 通过 `solver_chunk_builds`、
`solver_chunk_replays`、`solver_chunk_direct_submissions`、`solver_chunk_rebinds`、
`solver_chunk_invalidations`、`solver_graph_enabled` 和
`solver_replay_unavailable_reason` 暴露外层边界；同时通过 `operator_execution_kind`、
`operator_compiled_graph_submissions`、`operator_backend_captures` 与
`operator_backend_replays` 独立报告 provider 执行。构建开销属于 cold execution，性能资格
应分别记录 first solve 与 warm solve。

host 检查状态前，一个 chunk 或 GMRES-family restart cycle 总会完整执行。
`SolveResult.iterations` 表示逻辑上的
convergence 或 breakdown iteration；`statistics()["operations"]` 另行报告
`executed_iterations`、`wasted_iterations`、host synchronization 次数和 direct
chunk submission。因此 chunked solve 最多可能多执行 `K - 1` 轮 masked 或其它 inactive
tail。Vulkan fixed-budget execution 可以执行完整的 `max_iterations`，同时保留更早发生
的逻辑终止结果。

对于 CG/PCG/MINRES/BiCGSTAB，当更早终止比同步频率更重要时可选 `K=4`；
当摊薄 host check 更重要时可选 `K=8`。GMRES 与 FGMRES 使用选定的 restart 作为观察 interval。
较快选择取决于 vector size、operator 成本、iteration count、driver 与 backend。
不受支持的 policy 或 interval 会在 plan 构造时失败，不会静默 fallback。

可使用 `benchmarks/linear_operator_graph_krylov_bench.py` 对完整的 f32
compiled-kernel CG 边界做资格检查。该脚本分别报告 plan construction、first solve、warm
同步 solve completion、terminal observation、真实 residual、最大 solution error、逻辑/执行
迭代、host observation 与 kernel-profiler 可见性；通过 `--policies` 选择一个或多个策略。
CUDA/Vulkan 测量应在目标 GPU 空闲时执行。profiler 无法观察 captured Graph 内部时，
kernel 时间会明确报告为不可用，不做推测。

`plan.execution_capabilities()` 返回执行策略矩阵、条件执行不可用时的结构化原因，
以及当前选定的 `default_execution_policy`。
`automatic_solver_batching` 对象报告 matrix-free Kernel/Graph host-check 是否自动选中、
默认 interval、direct-chunk backend primitive，以及 provider 执行使用 compiled Graph
plan 还是 compiled-kernel launch。batching 不要求、也不宣称启用外层 solver replay。
独立的 `automatic_solver_replay` 对象报告外层 recurrence replay 是否已选择、operator 与
preconditioner
组合是否满足资格，以及具体 backend primitive（`cuda_conditional_graph_or_chunk_replay`、
`cuda_graph_chunk_replay`、`vulkan_structured_graph` 或
`vulkan_command_replay`）。solve 完成后的 statistics 仍是实际 replay 或
direct-submission 路径的权威记录。
直接使用 `"device_convergent"` 覆盖 stored 与 recordable-provider 范围。单系统 stored f32
CSR/BSR CG/PCG 在 driver、conditional-Graph 入口、device setter、provider capture 与
cuBLAS workspace 均满足资格时，可通过 `"bounded_convergent"` 自动选择；否则使用文档
规定的 chunk fallback。在 CUDA 上，单系统 compiled-kernel f32 CG/PCG 在 A/M action
可录制时，可显式请求通用结构化 Graph 路径，但不会自动选中；recordable compiled-Graph
f32 PCG 与 recordable f32 composition 在两个 GPU backend 上都会自动选中，compiled-Graph
CG 则保持 explicit-only。
stored 路径要求 solver-specific conditional setter 与 cuBLAS user-workspace 支持。在
Vulkan 上，所有满足资格的 recordable-provider f32 CG/PCG 范围会报告
`primitive="vulkan_dispatch_indirect"`；不支持的 provider 会直接失败，不会更换 policy
或 backend。
CUDA compiled-kernel 路径要求 general Graph conditional setter，不依赖 cuBLAS
workspace symbol；Vulkan 路径要求已资格化的结构化 Graph runtime 和可录制的固定 f32
binding。`device_convergent.qualification_scope` 与
`automatic_selection_qualified` 会公开该区别。显式请求不可用时失败，不做 fallback。
独立批量 f32 CG/PCG 在全部 A/M action 可录制时还支持显式 device-convergent policy；
CPU 与不可录制 provider 不宣称支持该执行模式。

## 独立批量 CG 与 PCG

`BatchedSolvePlan` 使用一个持久 plan 求解一组同构、相互独立的 SPD 系统。它接受 shape
为 `(B * N, B * N)` 的扁平 direct-sum operator，并把每个 vector 划分为 `B` 个连续、
长度均为 `N` 的系统：

```python
plan = ti.linalg.experimental.BatchedSolvePlan(
    operator,
    batch_size=B,
    independent_systems=True,
    method="cg",
    max_iterations=100,
    atol=1e-7,
    rtol=1e-5,
    execution_policy="host_check_every_k",
    check_interval=4,
)
result = plan.solve(rhs_flat, out=x_flat)

for env, reason in enumerate(result.termination_reasons):
    if not result.converged[env]:
        raise RuntimeError(f"system {env}: {reason}")
```

`independent_systems=True` 是调用方必须给出的显式断言。系统 operator 以及 PCG 的
fixed-linear preconditioner 都必须保持分区：环境 `e` 的 output 只能依赖环境 `e` 的
input。全局 SPD trait 不能证明这一分区性质。违反该合同得到的是无效的独立批量问题，
而不是一个耦合系统求解。

首个公开布局有意保持同构和扁平：

- operator extent 必须能被 `batch_size` 整除；
- 所有系统使用相同 scalar extent 与 f32 dtype；
- `rhs`、`initial_guess` 和 `out` 的 shape 为 `(B * N,)`；
- `atol` 与 `rtol` 可以是 scalar，也可以是长度为 `B` 的 sequence；
- variable offset 与 ragged system 不属于当前合同；可选 active-system compaction
  不改变同构扁平布局。

`method="cg"` 使用 identity preconditioner。`method="pcg"` 要求一个可信 SPD
`LinearOperator`，在相同扁平分区上应用 fixed-linear 近似逆。CPU、CUDA 与 Vulkan
均已验证 fixed stored 和 compiled-kernel A/M provider。其它 provider kind 仍受其普通
backend capability 与资格边界约束；batch plan 不会经 host staging，也不会切换 provider。

每个环境拥有独立的 recurrence scalar、effective tolerance、status、逻辑 iteration count
和 residual norm。一个环境可以收敛、breakdown 或达到 `max_iterations`，而不改变其它
环境的 terminal result。`BatchedSolveResult` 用 immutable tuple 暴露这些值，并单独返回
扁平 solution ndarray。batch size 为 1 时使用同一套单系统 CG/PCG 数值合同。

CPU 使用 `"host_each_iteration"`。CUDA 与 Vulkan 默认使用 K=4 的
`"host_check_every_k"`；也可选择 K=8、显式 `"host_each_iteration"`、
`"fixed_budget_masked"` 或显式 `"device_convergent"`。最后一种要求可录制的 f32 A/M
action，且不会自动选中。host-check chunk 在观察到所有环境已终止前，可能发出 inactive tail
iteration。recurrence 与 vector-update kernel 会屏蔽 inactive 环境，但整体 A/M provider
仍作用于完整扁平 batch；因此 convergence masking 不代表 provider apply 已做 compaction。

host-check 执行会把稳定的 iteration recurrence 编译为 plan-owned Taichi Graph，并在后续
solve 中复用。CG 每轮提交一个 recurrence Graph；PCG 在 A 之后和 M 之后分别提交一个
segment。Vulkan fixed-budget transaction 使用安全的 direct recurrence dispatch，因为在
active submission batch 内同步 nested replay 不属于已资格化操作。

显式 device-convergent 路径则把初始化、全部 A/M provider action、reduction、逐系统
recurrence/status 更新和全局 active predicate 录入一个 structured Graph；`submit()` 只产生
一个 Graph ticket。CUDA 使用已资格化的 structured control，Vulkan 使用 bounded conditional
replay，并可能保留已记录的 encoded/masked tail。两端在 loop active 时都不会经 host 读取
收敛状态。plan-owned device counter 记录真实停止 iteration，只在物化 terminal state 时读取。
该路径在全部系统 terminal 后会停止后续 full-batch A/M payload，但不会对部分活跃 batch 内的
A/M 做 compaction。

CUDA `device_convergent` plan 可显式设置 `active_system_compaction=True`。该路径在设备端
发布稳定的活跃系统列表，只压缩 recurrence reduction 与 vector update；A/M provider 仍作用于
完整扁平 batch，因此 capability 明确报告 `provider_apply_compacted=False`，scope 为
`recurrence_vector_payloads`。compact route 使用 capacity-grid masked prefix，不是 exact
indirect launch，也不回读 host count；CPU/Vulkan 不支持时会明确失败，不做静默模拟。它默认
关闭，因为收益取决于收敛差异、provider 成本和 batch size；并行 reduction 顺序变化还可能使
处在阈值边缘的浮点 stop 相差一轮，但不改变收敛合同。应检查
`statistics()["active_system_compaction"]`，不能只根据 backend 名称推断支持。

所有路径都会 pin 实际提交的 operator/preconditioner generation。兼容的 values-only update
会 rebind 到缓存的 device-convergent Graph，而不重建 Graph；在途 submission 会保留旧
generation 直到完成。上一轮 solve 完成后更换 `out` 只需 patch Graph binding。每个 workspace
clone 拥有独立 replay plan，不会因共享另一 clone 的 Graph lock 而串行化。

`statistics()` 会分别报告 executed system iteration、provider system iteration、masked
provider system iteration、active efficiency、host check、transfer 和 persistent resource
大小，使上述区别可观察。CG plan 拥有三个长度为 `B * N` 的 workspace vector，PCG 拥有
四个；此外还保留逐环境 recurrence、tolerance 与 status state。调用方拥有的 RHS、
solution、initial guess 和 provider resource 不计入这些 plan workspace 数字。batch plan
统计使用 schema version 5，并报告 `recurrence_replay_builds`、
`recurrence_replay_graph_builds`、`recurrence_replay_submissions`、
`recurrence_replay_logical_kernels`、output `recurrence_replay_rebinds` 与 direct recurrence
kernel submission；`recurrence_replay` record 会明确说明 A/M provider apply 不属于 Graph
host-check replay 范围。`device_convergent_replay` 另行报告完整 solve scope、Graph
build/submission 数、logical stop counter 与可用 structured-control telemetry。若 backend
无法在不增加同步的前提下公开 encoded/masked count，则保持 unavailable，不进行推测。

每次完成的 solve 都发布一个 packed terminal packet，其中包含全局 logical counter 以及全部
逐系统 status、iteration、residual 与 tolerance；`result()` 只物化一次该 packet。使用
`submit(..., telemetry=True)` 时，`submission.telemetry()` 还会返回 immutable per-ticket
logical/executed/provider work、active efficiency、可用 encoded/masked count、backend Graph
launch、physical queue submission 与不做推测的 timing 字段。telemetry 为 opt-in；backend
无法提供的 counter 保持 `None`。

使用 `benchmarks/batched_graph_pcg_bench.py` 以交错样本比较 policy 与 preconditioner
质量；脚本还会检查 warmup 后 runtime、host pool 与 device pool 是否稳定。
`linear_operator_graph_rebind_bench.py` 用于 generation churn，
`linear_operator_shifted_bench.py` 用于 shifted 与显式 identity lowering 对照。这些是本地
资格工具，不是固定性能承诺。

### 异步 fixed-budget 或 device-convergent submission

使用 `execution_policy="fixed_budget_masked"` 或满足资格的 `"device_convergent"` 的
CUDA/Vulkan batch plan 可以提交完整 solve，而不在调用返回前读取 terminal state：

```python
plan = ti.linalg.experimental.BatchedSolvePlan(
    operator,
    batch_size=B,
    independent_systems=True,
    max_iterations=8,
    atol=1e-6,
    execution_policy="device_convergent",
)

submission = plan.submit(rhs_flat, out=x_flat)
# 在这里提交应用中无依赖的工作。
submission.wait()
result = submission.result()
```

多个异步 solve producer 共享 GPU 时，可创建惰性、带内存计账的 workspace pool，并用同一个
`SubmissionPacer` 控制整体 backlog 与公平准入：

~~~python
pool = plan.workspace_pool(2, workspace_saturation="wait")
pacer = ti.graph.SubmissionPacer(
    2,
    max_in_flight_per_lane=1,
    max_queued=8,
)
primary_ticket = pool.submit(
    rhs_a, out=x_a, pacer=pacer, lane='primary_physics'
)
secondary_ticket = pool.submit(
    rhs_b, out=x_b, pacer=pacer, lane='secondary_physics'
)
primary_result = primary_ticket.result()
secondary_result = secondary_ticket.result()
~~~

完整 solve 的 host launch 序列以一次准入 turn 提交；提交结束后，两张票据可在
`max_in_flight` 边界内同时保持未完成。这里的异步只承诺宿主完成边界，不承诺两次 solve 的
GPU kernel 并发执行、独立 stream/queue 或设备抢占。`max_in_flight_per_lane` 防止高频 producer
独占全部 slot，lane 间 round-robin 则保证已有等待者获得有限的准入延迟。阻塞调用会以有界
自适应退避轮询全部 in-flight completion，使较晚完成的快 solve 可以先于更早提交的慢 solve
释放容量。该控制只覆盖共享同一 pacer 的调用，不能代替 engine 自身的 frame deadline、
任务依赖或取消策略。

`SolveSubmission.done()` 只观察 backend completion，不释放 workspace slot。`wait()` 在
需要时等待、生成完整逐系统 terminal snapshot，并释放 slot；`result()` 在需要时执行
相同操作并返回 immutable `BatchedSolveResult`。submission 会保留精确的 A/M
generation、input/output ndarray、workspace 和 backend completion，直到这一完成边界。
backend error 会由 `wait()` / `result()` 重新抛出；`ti.reset()` 会先等待已保留的 backend
work，再把尚未完成取值的 ticket 明确标记为 stale。

若在旧 submission 完成前发布新的 operator/preconditioner numeric generation，旧 generation
仍会保留到对应 completion。因此频繁数值更新可能使多个完整 values buffer 同时驻留；应用应
把 update cadence 纳入显存预算，或在更新前完成相关 ticket。Pacer 只限制共享它的 invocation
数量，不按 generation 字节数准入。

一个 `BatchedSolvePlan` 只拥有一个 submission slot。在 pending ticket 完成并生成结果前
再次提交会失败，不会共享 Krylov vector。需要多个独立 in-flight submission 时，使用
`clone = plan.clone_workspace()` 手动管理，或使用 `plan.workspace_pool(lanes)` 做惰性
round-robin 选择。每个已物化 lane 都拥有另一套完整 CG/PCG workspace vector、terminal
state 与独立 Graph instance。可用 `workspace_lane=i` 固定 lane；所有 lane 忙时，
`workspace_saturation="raise"` 立即失败，`"wait"` 等待并物化一个已完成 lane。
`submission.workspace_lane` 报告实际 lane；`pool.statistics()` 报告 capacity/materialized/
pending lane 以及逐 lane、已物化和总容量 payload bytes。workspace lane 消除 mutable state
与 completion fence 的别名，不承诺不同 GPU stream/queue 或物理重叠。每个 root plan 应只
建立一个 pool。chunked host-check policy 和 CPU plan 没有通过 `submit()` 资格；调用会明确
失败，而不是把同步循环移动到 worker thread。

对 batch size `B`、每个系统大小 `N`，f32 host-check CG 每个 plan/clone 的逻辑 workspace
payload 为 `12 * B * N + 92 * B + 24` 字节，PCG 为
`16 * B * N + 92 * B + 24` 字节；device-convergent plan 另加 12 字节，启用 CUDA
active-system compaction 再增加 `4 * B + 8` 字节。这些公式已包含 packed terminal packet。
创建 pool 前应检查 `statistics()["resources"]["clone_workspace_payload_bytes"]`，惰性物化后
以 `pool.statistics()` 为准。该数字不包含 allocator/driver 开销、调用方 vector 或
operator/preconditioner 资源。大型 solve
默认使用 `max_in_flight=1`；只有 profile 证明宿主重叠带来有效收益且资源与尾延迟仍满足预算时，
才增加到 2。小系统应优先扩大 batch，而不是按物理实体创建 plan clone。

该 API 表示 independent batching，不是对隐式耦合 block matrix 使用 global-scalar CG，
也不是 multi-RHS CG、block CG 或其它 block Krylov 方法。耦合系统必须使用在数学上显式
表达该耦合关系的 operator 与 solver。

## 支持矩阵

### Provider 与 apply

| Provider | CPU | CUDA | Vulkan |
| --- | --- | --- | --- |
| Fixed stored CSR/BSR | `f32`、`f64` | `f32` | `f32` |
| Compiled kernel | `f32` | `f32` | `f32` |
| Compiled Graph | `f32` | `f32` | `f32` |
| Identity | `f32`、`f64` | 不支持 | 不支持 |
| fixed-layout block diagonal | `f32`、`f64` | qualified direct-affine leaf，`f32` | qualified direct-affine leaf，`f32` |
| 调用方提供的 inverse block diagonal，size 1-4 | `f32` | `f32` | `f32` |
| Scale/shift/sum/compose | `f32`、`f64` | `f32` | `f32` |

kernel/Graph provider 支持矩形 shape 和显式 adjoint。通用 `alpha/beta` apply 支持 CPU，
并在 CUDA/Vulkan 上支持 f32 ndarray operand；GPU f64 与非 ndarray 的通用 storage 仍不支持。

### Solver

| Method/provider | CPU | CUDA | Vulkan |
| --- | --- | --- | --- |
| CG，fixed stored | CSR/BSR，`f32/f64` | CSR，`f32` | CSR/BSR，`f32` |
| CG，compiled kernel/Graph | `f32` | `f32` | `f32` |
| CG，recordable composition | `f32/f64` | `f32`，仅 device-convergent | `f32`，仅 device-convergent |
| PCG + Jacobi | fixed CSR，`f32/f64` | fixed CSR，`f32` | fixed CSR，`f32` |
| PCG + block-Jacobi | fixed BSR，`f32/f64` | fixed BSR，`f32` | fixed BSR，`f32` |
| PCG + fixed-linear operator/plan | 受支持 provider，`f32/f64` | 可录制 compiled-kernel/Graph/composition A/M，`f32`；Graph/composition 自动使用 device-convergent | 可录制 compiled-kernel/Graph/composition A/M，`f32`；Graph/composition 自动使用 device-convergent |
| MINRES + identity | 受支持 provider，`f32/f64` | fixed CSR/BSR 或 compiled provider，`f32` | fixed CSR/BSR 或 compiled provider，`f32` |
| MINRES + Jacobi/block-Jacobi | 不支持 | 分别为 fixed CSR/BSR，`f32` | 分别为 fixed CSR/BSR，`f32` |
| MINRES + fixed-linear operator/plan | 不支持 | 兼容的 device-native A/M，`f32` | 兼容的 device-native A/M，`f32` |
| 独立批量 CG/PCG | fixed stored 或 compiled-kernel A/M，`f32` | fixed stored 或 recordable A/M，`f32` | fixed stored 或 recordable A/M，`f32` |
| 批量异步 submission | 不支持 | fixed-budget stored/compiled-kernel 或 device-convergent recordable A/M，`f32` | fixed-budget stored/compiled-kernel 或 device-convergent recordable A/M，`f32` |
| device-convergent 条件执行 | 不支持 | 上述单系统范围；独立批量 recordable A/M 仅显式 | 上述单系统范围；独立批量 recordable A/M 仅显式 |
| BiCGSTAB + identity | 受支持 host-action provider，`f32/f64` | fixed CSR/BSR 或 compiled provider，`f32` | fixed CSR/BSR 或 compiled provider，`f32` |
| BiCGSTAB + fixed-linear 右预条件 | 受支持 host-action provider，`f32/f64` | 兼容的 device-native A/M，`f32` | 兼容的 device-native A/M，`f32` |
| GMRES + identity | 受支持 host-action provider，`f32/f64` | fixed CSR/BSR 或 compiled provider，`f32` | fixed CSR/BSR 或 compiled provider，`f32` |
| GMRES + fixed-linear 右预条件 | 受支持 host-action provider，`f32/f64` | 兼容的 device-native A/M，`f32` | 兼容的 device-native A/M，`f32` |
| FGMRES + variable-linear action table | 受支持 host-action provider，`f32/f64` | 兼容的 device-native A/actions，`f32` | 兼容的 device-native A/actions，`f32` |

direct factorization 继续使用 stored-matrix API。

## Numeric update 与所有权

stored operator 调用 `operator.update_numeric(values)`。compiled provider 使用乐观版本
检查：

```python
operator.update_numeric(
    next_values,
    expected_topology_version=1,
    expected_numeric_version=3,
)
```

Graph update 传入完整 numeric role mapping。compiled update 成功后会发布下一个 immutable
numeric generation。in-flight work 会继续保留其 pinned generation；后续 apply/solve
观察新 generation。stored Jacobi/block-Jacobi PCG plan 会在下一次 solve 前刷新 numeric
preconditioner，同时保留 pattern 与 Krylov workspace。scalar CSR Jacobi 保存 diagonal
reciprocal；BSR block-Jacobi 为每个 diagonal block 保存 lower Cholesky factor，支持的
block size 为 2、3、6 和 12。每个 block 都必须有限、对称且正定；不合法的 block 会明确
失败，不会执行对称化、正则化或 fallback，失败的 refresh 也不会发布新的 preconditioner
generation。topology 改变后必须构造新 operator。

CUDA 与 Vulkan 在 device 上完成 value-only Jacobi 和 block-Jacobi refresh。成功刷新后
committed resource address 保持稳定，因此可重放 solve plan 可以只 rebind numeric
generation，而不必重建 recurrence program。warm refresh 不经 host 复制完整 values 或
factor，也不分配 device memory；validation 只回读固定大小的 status。`statistics()` 提供
refresh contract、operation counter、transfer bytes 和 device-allocation count。

公开 API 不提供 borrowed-resource 模式。stored operator 强引用 matrix；compiled
provider 拥有复制后的 topology/numeric/workspace resource；composition 与 solve plan
强引用其 operand。

## 与 legacy matrix-free API 的关系

callback-based `ti.linalg.FieldLinearOperator`、`MatrixFreeCG` 和
`MatrixFreeBICGSTAB` 保留 field-shaped vector ABI。它们使用 `(x, y)` kernel callback；
由于该 ABI 不携带显式 topology、numeric resource、runtime generation 和 capability
信息，不会执行隐式转换。

应用明确需要旧版 field callback 合同时使用 `FieldLinearOperator`；需要 provider
capability、resource generation、runtime storage view、composition 或 solver-plan 集成时
使用 `LinearOperator`。

## 资格验证边界

`qualify_operator()` 可对任意公开 `LinearOperator` 生成版本化、JSON 可序列化的协议证据：

```python
report = ti.linalg.qualify_operator(
    operator,
    reference=dense_reference,
    samples=4,
    warmup=2,
    repetitions=10,
    metadata={"case": "poisson_level_3"},
)
report.to_json()
matrix = ti.linalg.summarize_operator_qualifications([report])
```

报告包含 backend/build、provider、shape、capability、resource stamp、forward/adjoint oracle
误差、线性与 dot-product identity、通用 apply 的 `beta=0` no-read 状态、同步边界计时和
native counter。`summarize_operator_qualifications()` 可从多组 detached
backend/provider report 自动生成确定性的支持矩阵。GPU 不支持的通用系数会记录为
`unsupported`，不会伪装为通过或触发 host fallback。计时只描述本机本次运行；它不是
跨机器性能门。

`qualify_solve_plan()` 为公开 `SolvePlan` 或 `BatchedSolvePlan` 生成对应的执行证据：

```python
report = ti.linalg.experimental.qualify_solve_plan(
    lambda: ti.linalg.experimental.SolvePlan(
        operator,
        method="pcg",
        preconditioner=preconditioner,
        execution_policy="host_check_every_k",
        check_interval=4,
    ),
    rhs,
    reference=expected_solution,
    expected_termination="converged",
    warmup=2,
    repetitions=10,
    metadata={"case": "poisson_level_3"},
)
matrix = ti.linalg.experimental.summarize_solve_qualifications([report])
```

该示例使用 CUDA/Vulkan chunk policy；CPU factory 应改用 `host_each_iteration`。

传入零参数 factory 时会单独记录 plan 构建时间；也可以传入已构造的 plan，此时 build timing
明确记为 unavailable。`reference` 可以是扁平 expected solution，也可以是接收 RHS NumPy
副本的 callable。独立 batch 使用一份 flat reference，并可逐系统指定 expected termination。

报告区分首次与稳态同步 wall time。合格的 GPU fixed-budget batch 还会拆分 host submit 和
completion wait，并可记录调用方提供的 `SubmissionPacer`。记录内容包括 A/M provider、policy/K、
terminal state、独立计算的 `b - A(x)` 真实残差、logical/executed/provider iteration、inactive-work
efficiency、chunk direct/replay counter、transfer、plan resource 和进程全局 device pool 增量。runtime
没有安全查询时，device timestamp span、device identity 和 driver version 会明确保持 unavailable；
不会把 wall time 改名为 device time。Nsight 等 profiler 结果可通过 metadata 作为 sidecar 保存。

一次资格运行会执行一次 first solve、指定数量的 warmup/repetition，并额外执行一次不计时的 operator
apply 来计算独立真实残差；因此会改变 plan counter 和传入的 output。性能证据应使用专用 plan/workspace，
尤其不要让异步 batch 或共享 pacer 的资格运行污染生产调度。函数只返回 detached evidence，不写文件。

API 已覆盖 backend correctness、lifecycle、trait、composition、10k approved-generation
churn 和 solver regression。
应用的生产资格验证仍由 workload 决定：应在具有代表性的物理与非物理系统上验证
operator 语义、conditioning、tolerance、preconditioner 适用性、失败处理、内存预算和
backend driver 行为。

另见[稀疏 runtime 与线性代数](sparse_runtime_and_linear_algebra.zh.md)和
[物理稀疏算子与求解器选择](physics_sparse_solver_selection.zh.md)。
