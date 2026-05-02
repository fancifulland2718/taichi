# Sparse SNode on Vulkan — 使用指南

> 适用于 **Taichi Forge 0.3.0**。vanilla Taichi 1.7.4 的 Vulkan/SPIRV 后端**只支持 `dense` + `root`**；Taichi Forge 在 Vulkan 上额外支持 `pointer` / `bitmasked` / `dynamic` SNode，并保证 cpu / cuda / vulkan 三后端数值等价。
>
> English: [sparse_snode_on_vulkan.en.md](sparse_snode_on_vulkan.en.md)

---

## 1. 速览

| 数据结构 | vanilla 1.7.4 Vulkan | Taichi Forge 0.3.0 Vulkan | LLVM (cpu/cuda) |
|---|---|---|---|
| `dense` | ✅ | ✅ | ✅ |
| `bitmasked` | ❌ | ✅ | ✅ |
| `pointer` | ❌ | ✅ | ✅ |
| `dynamic` | ❌ | ✅ | ✅ |
| `hash` | ❌ | ❌（详见 §6） | ❌（前端默认禁用） |
| `quant_array` / `bit_struct` | ❌ | ⚠️ 实验性（详见 §7） | ✅ |

支持的 op：`activate` / `deactivate` / `is_active` / `length` / `append` / `ti.deactivate` / struct-for (`for I in field:`) / `ti.ndrange` 上的稀疏 listgen。

---

## 2. 启用方式

无需任何额外开关——只要 `ti.init(arch=ti.vulkan)`，上表中的 SNode 即可使用。

```python
import taichi as ti
ti.init(arch=ti.vulkan)

x = ti.field(ti.f32)
ti.root.pointer(ti.ij, 32).dense(ti.ij, 8).place(x)

@ti.kernel
def fill():
    for i, j in ti.ndrange(256, 256):
        if (i + j) % 17 == 0:
            x[i, j] = i * 0.1 + j * 0.01

fill()
```

可选构建期开关（默认全 ON，仅在排查回归时使用）：

| CMake 选项 | 默认 | 用途 |
|---|---|---|
| `TI_VULKAN_POINTER` | ON | 总开关：关闭后 pointer / bitmasked 在 Vulkan 上回退到 vanilla 行为（`TI_NOT_IMPLEMENTED`）。 |
| `TI_VULKAN_DYNAMIC` | ON | dynamic SNode 总开关。OFF 时 `dynamic` 在 Vulkan 上 `TI_NOT_IMPLEMENTED`。 |
| `TI_VULKAN_POINTER_POOL_FRACTION` | ON | 启用 `TI_VULKAN_POOL_FRACTION` 环境变量（见 §3.2）。OFF 时该 env var 完全被忽略，capacity 按最坏情况预留。 |

### 2.1 运行期 env var

| 环境变量 | 取值 | 默认 | 作用 |
|---|---|---|---|
| `TI_VULKAN_POOL_FRACTION` | `(0.0, 1.0]` | `1.0` | 缩减每个 pointer SNode 的物理 cell pool 容量 = `max(num_cells_per_container, round(total_cells × fraction))`。详见 §3.2。非法 / `≤0` / `>1` 自动回退 `1.0`。 |

---

## ⚠️ 已知正确性 / 稳定性问题

> **历史上 Vulkan 稀疏 SNode 曾在两条语义合同上偏离 LLVM cpu/cuda 后端，目前两条均已修复：**
> 1. ✅ 已修复（0.3.1）：inactive 稀疏 cell 读返回 dtype 零值。
> 2. ✅ 已修复（0.3.2，2026-05-02）：3D pointer **全激活**触发 device-lost 已通过 C-9 deterministic-slot codegen 路径解决；详情见下方 Bug 2。

### ✅ 已修复 Bug 1（0.3.1，2026-04-30）：inactive 稀疏 cell 读返回 0

LLVM cpu/cuda 后端：读取 inactive 稀疏 cell **保证返回该 dtype 的零值**（路由到 `ambient_val_addr` 这块独立只读零内存）。这是 SDF 查找、ray-march、邻域采样代码可以「无脑写」的语义基础。

**0.3.0 行为（已被替换）**：inactive 读路由到 pool 第 0 号槽的真实地址。一旦 cell-0 被任何 kernel 合法激活并写入，**所有 inactive 读都会返回 cell-0 的真实数据**——典型表现：SDF ray-march 在空 voxel 读到 > 500 的脏值。

**0.3.1 修复**：[`spirv_codegen.cpp` `pointer_lookup_or_activate(do_activate=false)`](../../taichi/codegen/spirv/spirv_codegen.cpp) 现把 inactive 读重定向到 root buffer 末尾的 ambient zone（per pointer SNode 一段 `cell_stride` 字节零内存，启动时 memset 零，永不被 kernel 写）。三后端 inactive 读字节等价返回 0。验收脚本：[tests/p4/g10_inactive_read_zero.py](../../tests/p4/g10_inactive_read_zero.py)。CMake 选项 `TI_VULKAN_POINTER_AMBIENT_ZONE`（默认 ON）控制此路径；OFF 时还原 0.3.0 slot-0 fallback。

**仍保留的限制**：inactive **写**（容量越界 OOC）保留 silent-loss slot-0 路径，与 §3.1 documented behavior 一致；本次修复**只改正 inactive 读语义**。

### ✅ 已修复 Bug 2（0.3.2，2026-05-02）：3D pointer **全激活** device-lost

**历史触发条件（任一不满足即 PASS，已全部失效）**：

1. SNode 树包含 3D pointer 层（`ti.root.pointer(ti.ijk, ...)` 或 `pointer.pointer.dense(ti.ijk, ...)`）；
2. **同一帧内** pointer 容器中的**全部 cell**都被首次写入（首次写隐含 activate）；
3. 紧接着有 listgen 依赖 kernel：struct-for（`for i,j,k in field:`）/ `to_numpy()` / 读该字段的另一 kernel / 二次 ndrange 全扫；
4. 单 pointer cell 上预期 ≥64 个并发线程同时首次写入。

**历史症状**：写入 kernel 完成（`atomic_add` 计数正确）但下一个 dispatch 触发 `RHI Error: (-4) vkQueueSubmit failed` / Windows 下进程 abort code `0xC0000409`。

**0.3.2 修复（C-9 deterministic-slot codegen）**：观察到默认 `pool_capacity == total_num_cells_from_root`（与 outer cell 数完全相等）后，pointer activation 不再需要原子分配——为每个 outer cell 静态指派一个唯一池槽 `new_slot = idx_u32 + 1 ∈ [1, capacity]`。SPIR-V 端单条 `OpAtomicCompareExchange(slot, 0, new_slot)` 替换原 `CAS-marker + atomicIAdd(watermark) + atomicStore + 结构化 spin-loop` 四指令链；所有线程对同一 outer cell 计算同一 `new_slot`，无 spin、无 watermark 竞争，**warp-lockstep 死锁路径从根上消失**。

详细分析与回归矩阵见 `compile_doc/SNode_Vulkan_规划.md` §14；验收脚本：[tests/p4/g10p1_user_repro.py](../../tests/p4/g10p1_user_repro.py)（5 次连跑 Vulkan rc=0、cpu/cuda/vulkan sum 三后端 ±1e-5 等价）。

**启用门**：CompileConfig `vulkan_pointer_deterministic_slot`（默认 **`True`**，即 `ti.init(arch=ti.vulkan, vulkan_sparse_experimental=True)` 自动生效）。下列三种情况自动 fallback 到旧 CasMarker 路径（与 0.3.1 字节等价）：
- `vk_max_active` hint 把 capacity 收紧到 `< worst_capacity`；
- `vulkan_pointer_max_chunks > 1` 触发 chunked allocator；
- 多实例 SNode（`num_cells_per_container != total_num_cells_from_root`，例如某些深度嵌套 SNode 树）。

**回退手段**：`ti.init(vulkan_pointer_deterministic_slot=False)` 显式关闭，恢复 0.3.1 行为；与 0.3.2 落地前字节等价。

### 当前对 Vulkan 稀疏 SNode 的推荐

- ✅ 全面适合：MPM / SPH / brick 渲染等 atomic_add scatter；小活跃集 + 密集计算；嵌套 pointer 树（如 OpenVDB-like B+ 树）；3D pointer **全激活** "先 fill 后 reduce" 工作流（**0.3.2 起新增支持**）。
- ✅ brick-scatter 模式仍是默认推荐写法（性能更好、内存连续访问），但不再是规避 Bug 2 的必要手段。

---

## 3. 关键限制（**用前必读**）

### 3.1 Pointer / Dynamic 的 capacity 是**编译期静态**的

**与 LLVM 后端最大的语义差**：vanilla LLVM 把每个 cell 当成 `node_allocators` 上的动态 chunk，运行时按需扩容；Vulkan 后端**没有 device-side 动态分配器**，所有 cell 在 root buffer 中静态预留。

后果：

- **超出 capacity 的 activate 会被静默丢弃（silent inactive）**。这是经过 `cap_v` 守卫的安全降级——内核不会崩溃、不会越界写，但写入**不会生效**。
- **没有运行时报错**。要排查需自己加 `ti.length()` / `ti.is_active()` 检查。

例：

```python
# pointer(N=32) 后跟 dense(M=8) → 物理 capacity = 32 cells。
# 超过 32 个独立 i 在同一帧内被 activate 时，多余的会被丢弃。
ptr = ti.root.pointer(ti.i, 32)
blk = ptr.dense(ti.j, 8)
blk.place(x)
```

### 3.2 `TI_VULKAN_POOL_FRACTION` 缩减 capacity

如果你**确知** SNode 的稳态工作集远小于最坏情况（例如稀疏率 < 25%），可以缩减实际 pool 大小以减少 root buffer 与 GPU 内存占用：

```bash
# Linux / macOS
export TI_VULKAN_POOL_FRACTION=0.25
python your_app.py

# Windows PowerShell
$env:TI_VULKAN_POOL_FRACTION = '0.25'
python your_app.py
```

**何时使用**：

- 预设 N 远超实际激活数（例如 `pointer(ti.i, 4096)` 但每帧只激活 ~200 个）；
- 配合 deactivate-freelist（始终启用，`ti.deactivate` 自动归还 cell）；
- 性能/内存敏感的部署场景。

**何时不要用**：

- 你不确定稳态激活数；
- 测试 / 调试阶段；
- 单帧内有"先激活全集再 deactivate" 的工作流（峰值激活数 = 全集）。

**安全降级**：超过缩减后 capacity 的 activate 同样 silent inactive（与 §3.1 同机制）。

### 3.3 Dynamic SNode 协议差异

Vulkan 上的 `dynamic` 使用 **flat-array + length 后缀** 协议：

- 容器布局：`[data: cell_stride × N][length: u32]`；
- `ti.append(field, [i], val)` = `OpAtomicIAdd(length, 1)` + 写入 cell；
- `ti.length(field, [i])` = `OpAtomicLoad(length)`；
- `ti.deactivate(dynamic_node, [i])` = `OpAtomicStore(length, 0)`；
- 不支持 chunk 链；总容量 = 编译期静态 N。
- 超过 N 的 append 会被 silent dropped（同样 `cap_v` 守卫）。

数值结果与 LLVM 完全等价（已通过完整回归集验证）。

### 3.4 SPIR-V warp lockstep 限制

任何"基于 spin 等 winner 写完 slot"的协议（pointer / dynamic 的 race-to-activate）都会受 SPIR-V `OpLoopMerge` + GPU warp lockstep 影响。本 fork 的实现已在 NVIDIA / AMD / Intel iGPU 上验证稳定。如在新硬件上撞到 hang，请打开 issue 附 GPU 型号与 driver 版本。

### 3.5 不支持

- `hash` SNode：详见 §6。
- `quant_array` / `bit_struct`：详见 §7。
- 跨多个 SNode tree 的 cross-tree pointer：与 LLVM 同步限制。
- `ti.deactivate` 在 ambient（root 直挂的 dense）上：与 vanilla 同——dense 不支持 deactivate。

### 3.6 Listgen 顺序非合同

LLVM 后端 struct-for（`for I in field:`）按 SNode 树拓扑序遍历活跃 cell；Vulkan 后端只保证**覆盖所有当前活跃 cell**，不保证遍历顺序。如果你的 reduce kernel 依赖确定遍历顺序（例如 `for i in field: result[0] += i.id`），不同后端的中间累加路径**可能给出不同浮点字节**（最终值在 1e-5 容差内等价，但不字节等价）。

规避：用 `ti.atomic_add` 或在 reduce 前先排序。

### 3.7 显式声明 pointer 容量上限：`vk_max_active`（容量决议 hint）

用于嵌套 pointer 或大 N 但稀疏度低的工作流。给定 `pointer(ti.ij, N)` SNode，默认按最坏情况预留 `N²` cell 到 root buffer；若你确知该层级的稳态最大活跃数远小于此，可显式提示：

```python
# Vulkan 后端：把该 pointer 的物理 pool 容量缩到 1024（而非默认 N²）
blk = ti.root.pointer(ti.ij, 1024, vk_max_active=1024)
blk.dense(ti.ij, 8).place(x)
```

规则：

- kwarg 仅在 Vulkan 后端生效；其他后端忽略并 warn-once。
- 4 级 fallback 优先级：`vk_max_active` (kwarg) > `vulkan_pointer_pool_fraction`（`ti.init` kwarg）> `TI_VULKAN_POOL_FRACTION` (env var) > 最坏情况预留。
- 容量下限锁死 = 该 pointer 容器单 cell 数（`num_cells_per_container`），低于此值会被强制提升以保证 deactivate-freelist 至少能容纳一个 cell。
- 超容时的行为：silent inactive（与 §3.1 一致），不抛异常。
- 嵌套 pointer 上每层独立标注：`outer = ti.root.pointer(ti.ij, OUTER, vk_max_active=N1); inner = outer.pointer(ti.ij, MID, vk_max_active=N2)`。

---

## 4. 验证矩阵

本 fork 已在 cpu / cuda / vulkan **三后端等价回归**：

| 测试 | 覆盖点 |
|---|---|
| `tests/p4/vulkan_pointer_smoke.py` | 基础 activate/lookup |
| `tests/p4/vulkan_pointer_race.py` | 多线程 race-to-activate |
| `tests/p4/vulkan_pointer_recycle.py` | freelist 回收 32 cycle |
| `tests/p4/vulkan_pointer_listgen.py` | struct-for |
| `tests/p4/vulkan_pointer_deactivate_all.py` | 全量 deactivate + 复活 |
| `tests/p4/vulkan_pointer_ported.py` | vanilla pointer test 三后端等价 |
| `tests/p4/vulkan_bitmasked_ported.py` | bitmasked 全套 |
| `tests/p4/vulkan_dynamic_basic.py` | dynamic basic / cycle / is_active |
| `tests/p4/g2_pool_fraction.py` | `TI_VULKAN_POOL_FRACTION=0.25` 三后端等价 |
| `tests/p4/g4_probe.py` | dynamic flat-array 协议 |
| `tests/p4/g8_cache_compat.py` | offline cache fresh / hit / corrupt-recover |
| `tests/p4/c1_jit_capacity.py` | `vk_max_active` 4 级 fallback 容量决议 |
| `tests/p4/c1_nested_pointer_pointer_dense.py` | 嵌套 `pointer.pointer.dense` 三后端等价（含 hint） |
| `tests/p4/g10p1_user_repro.py` | Bug 2 全激活 device-lost 回归 guard（0.3.2 起 PASS） |

跑法（PowerShell）：

```powershell
$env:PYTHONPATH = 'D:\taichi\python'
python tests\p4\vulkan_pointer_smoke.py
```

---

## 5. 排错

| 现象 | 可能原因 | 处置 |
|---|---|---|
| 写入静默丢失，`ti.length()` 显示 < 实际激活次数 | 超出编译期 capacity（§3.1 / §3.3） | 增大 SNode 维度 N，或减少同帧并发 activate。 |
| `TI_VULKAN_POOL_FRACTION=0.5` 后部分 cell 写入丢失 | 缩减后 capacity 不够（§3.2） | 调高 fraction，或恢复默认 1.0。 |
| `RuntimeError: hash not yet supported` | 你在用 `ti.root.hash(...)` | hash 在 vanilla 与本 fork 均默认禁用，见 §6。 |
| Vulkan 上首次启动很慢，第二次秒开 | offline cache 首次编译 | 正常行为；第二次起命中 cache。 |
| `~/.cache/taichi/` 中 `ticache.tcb` 损坏 → 启动 | fallback 重编译路径已内建 | Forge 0.2.4+ 自动处理 `kVersionNotMatched` / `kCorrupted`，不抛异常。 |
| `hang` / `device lost` 在 race 测试 | warp lockstep（§3.4） | 提供 GPU 型号 + driver 版本到 issue。 |

---

## 6. 关于 `hash` SNode

`hash` SNode 在 **vanilla taichi 1.7.4 与本 fork 中均默认禁用**——`ti.root.hash(...)` 抛 `RuntimeError("hash not yet supported")`，已是上游历史状态（自 2021 年起）。

### 6.1 实时物理仿真 / 实时渲染必要性判定

在判定是否为 Vulkan 后端实现 `hash` SNode 之前，我们对主流实时 GPU 负载做了一轮全面的需求扫描。结论：**不存在一个在表组会被堆的实时物理 / 渲染 pipeline 依赖它**，因此不计划进一步投入。

| 负载类型 | 典型稀疏模式 | Forge 中的地道替代方案 |
|---|---|---|
| 刚体 broadphase（PBD / Bullet 类空间哈希） | 世界 AABB 有界 → 固定桤位数哈希表 | `dense` 桤位 + atomic counter，用户在 `floor(x / cell_size)` 上跱 `hash21` |
| MPM / MLS-MPM / SPH / PBF | 仿真域有界，稀疏激活 | `pointer.bitmasked.dense`（上游 MPM-128 / MLS-MPM 法定模式） |
| FEM / 体積网格 | 自适应但有界 | `pointer.dense` 树 |
| 布料 / 毛发 / Eulerian 流体 | 稠密或低稀疏 | `dense` / `bitmasked.dense` |
| OpenVDB 式 SDF / 体素 cone tracing | brick 的 B+ 树 | `pointer.pointer.dense`（Forge 已在 Vulkan 上支持） |
| Instant-NGP / NeRF 哈希网格编码 | 定长哈希表 + 显式冲突处理 | 大小 `T`（常为 `2^14`–`2^19`）的 `dense` + 用户级坐标哈希 |
| 体积全局光照 探重 | 稠密网格 | `dense` |

为什么 `hash` SNode **不是**合适选择：

- 以上负载**坐标空间都是有界的**（仿真域或屏幕视锥），`hash` 的“无界坐标”优势未被使用。
- Vulkan 没有 device-side 动态分配器；忠实实现 `hash` SNode 只能 (a) 静态预留最坏情况桶数组——与“用户级 hash + dense”在功能上等价，或 (b) chunk 重哈希——与本 fork 已明确划出的非目标范围重叠。
- GPU 上哈希插入的竞态 (linear-probe + CAS + warp lockstep) 与 `pointer` race-to-activate 部分重叠，不能纯增得。

### 6.2 当前推荐替代

1. `ti.root.pointer(ti.ij, N).dense(ti.ij, M).place(...)` + `TI_VULKAN_POOL_FRACTION=0.05`（§3.2）：覆盖 95% 真实稀疏场景；
2. 用户级哈希函数（参考 `python/taichi_forge/examples/algorithm/poisson_disk_sampling.py` 中的 `hash21`）+ `dense` 桶，适用于 instant-NGP 类编码器与刚体 broadphase。

---

## 7. 关于 `quant_array` / `bit_struct`

`quant_array`（位打包整数 / 定点字段）与 `bit_struct`（位打包复合结构）在 vanilla taichi 中**仅 LLVM 后端可用**。Forge 0.3.0 在 Vulkan 后端上提供**实验性 codegen**：

- 前端 extension 闸门需主动启用：`ti.init(arch=ti.vulkan, vulkan_quant_experimental=True)` 或 env var `TI_VULKAN_QUANT=1`（详见 [forge_options.zh.md](forge_options.zh.md) §3）。默认 OFF，行为与 vanilla 1.7.4 完全一致（quant 路径在 codegen 入口直接 `TI_ERROR`）。
- 闸门 ON 后已支持的能力：
  - **`quant_array`**：`QuantInt` / `QuantFixed` 子字段的**读 + 写（含多线程并发 `ti.atomic_add` 经 SPIR-V `OpAtomicCompareExchange` 自旋 RMW）**，与 cpu / cuda 后端**字节等价**；
  - **`bit_struct` / `BitpackedFields(max_num_bits=32 或 64)`**：多字段同字 RMW 写（IR pass `optimize_bit_struct_stores` 在 `quant_opt_atomic_demotion=ON` 默认下合并为单条 `BitStructStoreStmt`；`is_atomic == true` 残留路径用 CAS-loop 真原子写），与 cpu / cuda 后端**字节等价**。MPM 风格 11/11/10 quant_fixed 粒子位置打包基线 [tests/p4/g9_quant_baseline.py](https://github.com/taichi-dev/taichi/blob/master/tests/p4/g9_quant_baseline.py) 三后端 max_err 全为 9.77e-4 ≤ bound 1.95e-3；并发原子加 race 基线 [tests/p4/g9_quant_atomic_race.py](https://github.com/taichi-dev/taichi/blob/master/tests/p4/g9_quant_atomic_race.py) N=1024、K=64 路并发同字争用，三后端 max_err 全为 3.94e-3 ≤ bound 1.57e-2。
  - **原子加** `ti.atomic_add(quant_field, delta)`：仅 `AtomicOpType::add`，physical_type 须为 i32 或 i64（与 LLVM `quant_type_atomic` 限制对齐）。返回值为输入 `delta`（非旧值）——quant 字段上的 `atomic_add` 用户代码极少消费返回值，省去 dequant 旁路代价。
- **尚未支持**：
  - **`QuantFloat` 共享指数**（`ti.types.quant.float(...)` + `BitpackedFields(shared_exponent=True)`）：visitor 入口处 `TI_NOT_IMPLEMENTED`。**显式暂缓**：本 fork 本身未需求 shared-exponent（原始诉求是 quant_fixed），且该路径 float 位操作跨驱动微妙差异风险高。生产负载如需使用，请继续走 LLVM cpu / cuda 后端。
  - **非 add 的原子操作**（`atomic_min` / `max` / `bit_and` / `bit_or` / `bit_xor`）到量化字段：与 LLVM 后端一致地不支持（vanilla 设计决定，不是 Vulkan 后端的回退）。
- 未实现点会抛出带详细位置的 `TI_NOT_IMPLEMENTED` / `TI_ERROR`，不会静默误编译。

变通方案（闸门 OFF 或命中未实现 codegen 点时）：

- 用 `ti.f16` 半精度作为简易量化；
- 在 `ti.u32` 字段上手工位运算打包。

回归基线脚本：[tests/p4/g9_quant_baseline.py](https://github.com/taichi-dev/taichi/blob/master/tests/p4/g9_quant_baseline.py)（`bit_struct` MPM 风格 11/11/10 打包）、[tests/p4/g9_quant_array_baseline.py](https://github.com/taichi-dev/taichi/blob/master/tests/p4/g9_quant_array_baseline.py)（`quant_array` 8-bit 单字段）、[tests/p4/g9_quant_atomic_race.py](https://github.com/taichi-dev/taichi/blob/master/tests/p4/g9_quant_atomic_race.py)（atomic_add 多线程同字争用）。

---

## 8. 兼容性与版本

- **API 兼容**：所有公开 Python API（`ti.root.pointer/.dense/.bitmasked/.dynamic/.place`、`ti.activate/.deactivate/.is_active/.length/.append`、`ti.root.deactivate_all` 等）行为与 vanilla 1.7.4 在 LLVM 后端上**严格一致**。Vulkan 上多出的 SNode 类型只新增可用性，不破坏现有用法。
- **Offline cache**：cache key 已纳入 SNode tree 结构 hash，pool fraction / dynamic 协议变更**自动**触发缓存失效。
- **C-API**：sparse SNode 的 root buffer 布局通过既有 `c_api/include/` 接口暴露，AOT 产物格式不变。
- **Wheel 二进制**：默认 build 已 ON 全部稀疏 SNode 后端能力（pointer / bitmasked / dynamic / quant 实验闸门）；用户无需额外编译参数。

---

## 9. 参考

- 本 fork 新增编译/运行时/架构/现代化选项一览：[forge_options.zh.md](forge_options.zh.md)
- 测试代码：本仓库 `tests/p4/vulkan_*.py` 与 `tests/p4/g*.py`
