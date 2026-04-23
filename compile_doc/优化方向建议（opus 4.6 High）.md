
## Taichi 1.7.4 GGUI 可视化与 Kernel 执行性能深度分析

---

### 一、GGUI 渲染管线概览

GGUI 的帧循环流程（window.cpp `show()` 方法）：

```
Python: window.show()
  ├─ draw_frame()
  │   ├─ acquire_next_image()          ← 获取 SwapChain 图像
  │   ├─ image_transition(→ color_attachment)
  │   ├─ record_prepass_commands()     ← compute 预处理
  │   ├─ begin_renderpass()
  │   ├─ for renderable: record_this_frame_commands(cmd)
  │   │   └─ bind pipeline → bind VBO/UBO/SSBO → draw/draw_indexed
  │   ├─ gui->draw()                  ← ImGui 在同一 renderpass 内
  │   ├─ end_renderpass()
  │   ├─ prog->flush()                ← 等待计算完成
  │   └─ stream->submit(wait: [compute_semaphore, acquire_semaphore])
  └─ present_frame()
      ├─ FPS 自适应 sleep（overshoot 补偿）
      └─ surface.present_image()
```

---

### 二、GGUI 关键性能瓶颈

#### 1. **每帧重新分配 VBO numpy 数组**

staging_buffer.py 的 `get_vbo_field_v2` 每次调用都创建新 numpy 数组：

```python
def get_vbo_field_v2(vertices):
    N = vertices.shape[0]
    vertex_stride = 3 + 3 + 2 + 4  # pos + normal + texcoord + color = 12 floats
    vbo = np.ndarray((N, vertex_stride), dtype=np.float32)  # 每帧新分配!
    return vbo
```

而旧版 `get_vbo_field` 有缓存（`vbo_field_cache`），`_v2` 版本刻意去掉了缓存。对于 10 万顶点的场景，每帧分配 ~4.8MB numpy 内存。

**对比**：scene.py 中 mesh/particles/lines 全部调用 `get_vbo_field_v2`，无一例外。

#### 2. **法线每帧从头重算**

在 scene.py 中，`mesh()` 方法的实现：

```python
def mesh(self, vertices, indices=None, normals=None, ...):
    if normals is None:
        normals = gen_normals(vertices, indices)  # 每帧触发!
```

`gen_normals_kernel_indexed` 对每个三角形做叉积+归一化+原子加，对大网格（如 100k 三角形）是一笔不小的开销。尽管 `normals_field_cache` 缓存了 field 本身，但 **kernel 依然每帧执行**。

#### 3. **staging buffer 的 VBO 拷贝路径**

renderable.cpp 中 `copy_helper` 有两条路径：

| 路径 | 条件 | 性能 |
|------|------|------|
| **GPU→GPU** | `src.device == dst.device == ggui_device` | 快：`enqueue_compute_op_lambda` 异步拷贝 + barrier |
| **Host→GPU** | `src.device == nullptr`（host 数据） | **慢：`map` → `memcpy` → `submit_synced`（阻塞!）** |

Host→GPU 路径调用 `stream->submit_synced(cmd_list.get())`，这是一个 **完全同步的操作**，CPU 等待 GPU 拷贝完成才返回。如果 Python 层传入 numpy 数组，每帧都走这条阻塞路径。

#### 4. **Renderable 对象每帧重建**

在 renderer.cpp 中：

```cpp
template <typename T>
T *Renderer::get_renderable_of_type(VertexAttributes vbo_attrs) {
    std::unique_ptr<T> r = std::make_unique<T>(...);  // 每帧 new
    renderables_.push_back(std::move(r));
    return ret;
}
```

且在 `draw_frame()` 结尾（renderer.cpp）：

```cpp
render_queue_.clear();
renderables_.clear();  // 每帧全部销毁
```

这意味着每帧的每个 mesh/particles 对象都经历：`new → init pipeline → create resource_set → bind → draw → delete`。GPU pipeline 虽有 `get_raster_pipeline()` 缓存，但 resource_set 和 raster_state 每帧重建。

#### 5. **Scene UBO 每帧重新分配**

renderer.cpp 中 `init_scene_ubo()` 每帧调用：

```cpp
void Renderer::init_scene_ubo() {
    scene_ubo_.reset();           // 释放旧 buffer
    auto [buf, res] = device.allocate_memory_unique(
        {sizeof(UBOScene), host_write=true, ...});  // 重新分配
    scene_ubo_ = std::move(buf);
}
```

UBOScene 仅 ~128 bytes，但频繁的 `allocate + deallocate` 在 Vulkan 上意味着频繁的 descriptor 更新。

---

### 三、Kernel 运行时执行性能分析

#### 1. CPU 后端执行模型

CPU kernel 的核心在 cpu/kernel_launcher.cpp，启动极轻量：

```cpp
for (auto task : launcher_ctx.task_funcs) {
    task(&ctx.get_context());   // 直接函数调用，无额外开销
}
```

并行 `range_for` 通过 `ThreadPool` 分发（runtime.cpp）：
- 将循环拆分为 `(end - begin) / block_dim` 个 task
- 线程通过 `task_head.fetch_add(1)` 原子抢 task
- 支持 TLS（线程局部存储）：prologue/epilogue 回调，用于 BLS 优化和归约

**性能特征**：
- 启动延迟 ~1-5μs（线程唤醒 + condition variable）
- 原子抢任务 ~10-50ns/task
- `cpu_max_num_threads` 默认为 `hardware_concurrency`
- `default_cpu_block_dim = 32`，`cpu_block_dim_adaptive = true`

**潜在问题**：
- 对 **小循环** kernel（如 N < 1000），线程唤醒开销可能超过计算本身
- `block_dim_adaptive` 会动态调整块大小，但调整逻辑较粗糙

#### 2. CUDA 后端执行模型

cuda/kernel_launcher.cpp 的启动流程更重量级：

```
① malloc_async(device_result_buffer)     ~100ns (pooled)
② 对每个数组参数：
   if on_cuda_device(ptr):               ~1μs (DMA 查询)
     直接使用 device ptr
   else:
     malloc + memcpy_host_to_device       PCIe 受限 (~10GB/s)
③ malloc_async(device_arg_buffer)        ~100ns
④ memcpy_host_to_device_async(args)      异步
⑤ for task: cuLaunchKernel(<<<grid, block>>>)  ~1-5μs/task
⑥ memcpy_device_to_host_async(results)   异步
⑦ mem_free_async(...)                    延迟释放
```

**关键观察**：
- 步骤 ② 中的 `on_cuda_device()` 调用 `cuPointerGetAttribute`，**对每个数组参数执行一次 DMA 类型查询**，这是同步操作（~1μs）
- 如果传入的是 numpy 数组而非 Taichi ndarray，会触发 host→device 拷贝 + `stream_synchronize`（全同步）
- 对于纯 ndarray 参数，走 `DeviceAllocation` 路径，零拷贝

#### 3. Vulkan/SPIR-V (GfxRuntime) 执行模型

runtime.cpp 的 `launch_kernel` 比 LLVM 后端更重：

```
① 分配 args_buffer + ret_buffer (Uniform/Storage)
② 对每个 array 参数：
   if Ndarray: 直接使用 DeviceAllocation
   if 外部数组: allocate_memory_unique + host_to_device
③ 对每个 task:
   ├─ create_resource_set_unique()
   ├─ 遍历 buffer_binds → rw_buffer/buffer 绑定
   ├─ 遍历 texture_binds → image transition + rw_image/image 绑定
   ├─ bind_pipeline(vp)
   ├─ bind_shader_resources(bindings)
   ├─ dispatch(group_x)
   └─ memory_barrier()
④ submit_current_cmdlist_if_timeout()  ← 2ms 超时才批量提交
```

**性能特征**：

| 操作 | 延迟 | 说明 |
|------|------|------|
| 描述符集创建 | ~100-500ns | 每个 task 一个新描述符集 |
| Image transition | ~50-200ns | 纹理状态切换 |
| bind_pipeline | ~50ns | 已缓存的 compute pipeline |
| dispatch | ~200ns | 命令录制，非实际执行 |
| memory_barrier | ~50ns | 命令录制 |
| **提交到 GPU** | **~10-100μs** | 命令缓冲区提交 |

**批量提交策略**（runtime.cpp）：命令列表在 **2ms 未被同步提交时**自动 flush。这意味着如果短时间内连续 launch 多个 kernel，它们会被 **批量打包为一个 command buffer** 提交——这是 Vulkan 后端的一项重要优化。

---

### 四、Kernel 计算性能的关键影响因素

#### 1. Grid/Block 维度选择

- CPU: `default_cpu_block_dim = 32`，自适应调整
- CUDA: `default_gpu_block_dim = 128`，可通过 `ti.init(default_gpu_block_dim=256)` 调整
- Vulkan: `advisory_num_threads_per_group` 由 codegen 决定，`group_x = ceil(total / per_group)`

CUDA 的 occupancy 受限于：
- `gpu_max_reg` 默认 0（使用驱动默认值）
- `saturating_grid_dim` 默认 0（无限制）

#### 2. 内存访存模式

**Dense SNode**（连续数组）：
- 顺序访存，cache 友好
- `struct_for` 直接展开为 `range_for`（由 `demote_dense_struct_fors` pass 完成）
- 最佳吞吐

**Sparse SNode**（pointer/bitmasked/hash）：
- 增加一层间接寻址：`element_lists[snode_id]` → 实际数据
- `struct_for` 需遍历 element list，可能导致线程间负载不均衡
- Bitmasked 通过位扫描(`find_next_bit`)避免完整遍历，但仍有分支开销

#### 3. 自动优化 Pass 对运行时的影响

| IR 优化 | 运行时效果 |
|---------|-----------|
| `make_thread_local` | 热数据放入线程栈/寄存器 → 减少全局内存访问 |
| `make_block_local` | GPU block 共享内存 → 减少 global load（BLS 优化） |
| `cache_loop_invariant_global_vars` | 循环不变量提升 → 减少重复访存 |
| `demote_atomics` | 原子操作退化为普通 load/store → 减少同步开销 |
| `half2_vectorization` (CUDA) | FP16 双打包 → 吞吐翻倍 |
| `detect_read_only` | 标记只读访问 → 允许更激进缓存 |

#### 4. 计算与渲染之间的同步代价

GGUI 的 `draw_frame()` 中（renderer.cpp）：

```cpp
if (app_context_.prog()) {
    auto sema = app_context_.prog()->flush();  // flush 计算流
    if (sema) {
        wait_semaphores.push_back(sema);        // 渲染等待计算
    }
}
```

这是一个 **计算→渲染 pipeline barrier**：渲染命令提交时通过 semaphore 等待计算流完成。如果仿真 kernel 和渲染在同一 Vulkan device 上，这个等待不涉及 CPU，是纯 GPU 侧同步——代价很小。但如果计算在 CUDA、渲染在 Vulkan（不同 device），就需要显式 host-side 同步。

---

### 五、性能问题汇总与优化建议

#### GGUI 可视化层

| 问题 | 影响 | 严重程度 |
|------|------|----------|
| **`get_vbo_field_v2` 每帧分配 numpy**（staging_buffer.py） | Python GC 压力 + 内存带宽浪费 | **高** |
| **法线每帧重算**（scene.py） | 大网格浪费数 ms 的 kernel 执行时间 | **高** |
| **Renderable 对象每帧 new/delete**（renderer.cpp） | 频繁堆分配 + resource_set 重建 | **中** |
| **Host→GPU staging 是阻塞式提交**（renderable.cpp） | CPU 等 GPU 完成拷贝 | **中** |
| **Scene UBO 每帧重新分配**（renderer.cpp） | 128B buffer 频繁 alloc/free | **低** |

#### Kernel 执行层

| 问题 | 影响 | 严重程度 |
|------|------|----------|
| **CUDA `on_cuda_device()` 逐参数查询**（kernel_launcher.cpp） | 每个 array 参数 ~1μs 同步查询 | **中** |
| **Vulkan 每 task 创建新 descriptor set**（runtime.cpp） | 描述符分配开销累积 | **中** |
| **小 kernel 的线程池唤醒开销** (CPU) | N < 1000 时线程开销可能 > 计算 | **中** |
| **外部数组(numpy)触发全同步拷贝** (CUDA/Vulkan) | `stream_synchronize` 破坏异步流水线 | **高** |
| **Vulkan 2ms 超时提交策略** | 低工作量时引入不必要的延迟 | **低** |

#### 仿真性能的核心建议

| 建议 | 原理 |
|------|------|
| **始终使用 `ti.ndarray` 而非 numpy** | 避免 host→device 拷贝和同步 |
| **对不变网格传入预计算好的 normals** | 避免每帧 `gen_normals` kernel 开销 |
| **仿真与 GGUI 使用同一后端 (均 Vulkan)** | 实现 GPU→GPU 零拷贝路径 |
| **用 dense field 代替 sparse** | 去除间接寻址 + 负载均衡 |
| **CPU 后端: 对小 kernel 使用 `ti.loop_config(serialize=True)`** | 跳过线程池调度 |
| **CUDA: 确保数据已在 GPU 上** | 避免 `on_cuda_device` 查询和传输 |
| **利用离线缓存 (`offline_cache=True`)** | 消除重复编译延迟 |