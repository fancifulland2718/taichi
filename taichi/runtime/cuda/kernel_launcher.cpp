#include "taichi/runtime/cuda/kernel_launcher.h"
#include "taichi/runtime/cuda/jit_cuda.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/system/profiler_annotation.h"
#include "taichi/util/environ_config.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <cstdint>
#include <unordered_map>

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"

namespace taichi::lang {
namespace cuda {

namespace {

struct RetainedLaunchBufferTelemetry {
  std::atomic<std::uint64_t> current_bytes{0};
  std::atomic<std::uint64_t> peak_bytes{0};
  std::atomic<std::uint64_t> allocation_calls{0};
  std::atomic<std::uint64_t> release_calls{0};
};

struct GridResidencyTelemetry {
  std::atomic<std::uint64_t> resolution_calls{0};
  std::atomic<std::uint64_t> resolution_failures{0};
  std::atomic<std::uint64_t> last_requested_waves{0};
  std::atomic<std::uint64_t> last_baseline_grid{0};
  std::atomic<std::uint64_t> last_resolved_grid{0};
  std::atomic<std::uint64_t> last_active_blocks_per_multiprocessor{0};
  std::atomic<std::uint64_t> last_multiprocessor_count{0};
  std::atomic<std::uint64_t> coarsening_resolution_calls{0};
  std::atomic<std::uint64_t> coarsening_resolution_failures{0};
  std::atomic<std::uint64_t> last_work_per_thread_target{0};
  std::atomic<std::uint64_t> last_coarsening_baseline_grid{0};
  std::atomic<std::uint64_t> last_coarsened_grid{0};
};

struct ArtifactQualificationTelemetry {
  std::atomic<std::uint64_t> qualification_calls{0};
  std::atomic<std::uint64_t> registration_materializations{0};
  std::atomic<std::uint64_t> function_attribute_queries{0};
  std::atomic<std::uint64_t> occupancy_queries{0};
};

RetainedLaunchBufferTelemetry &retained_launch_buffer_telemetry() {
  // Process-lifetime telemetry avoids static-destruction ordering with the
  // Program singleton, whose CUDA launcher may release buffers during exit.
  static auto *telemetry = new RetainedLaunchBufferTelemetry();
  return *telemetry;
}

GridResidencyTelemetry &grid_residency_telemetry() {
  static auto *telemetry = new GridResidencyTelemetry();
  return *telemetry;
}

ArtifactQualificationTelemetry &artifact_qualification_telemetry() {
  static auto *telemetry = new ArtifactQualificationTelemetry();
  return *telemetry;
}

void update_retained_launch_peak(std::uint64_t candidate) {
  auto &peak = retained_launch_buffer_telemetry().peak_bytes;
  auto observed = peak.load(std::memory_order_relaxed);
  while (observed < candidate &&
         !peak.compare_exchange_weak(observed, candidate,
                                     std::memory_order_relaxed,
                                     std::memory_order_relaxed)) {
  }
}

void record_retained_launch_allocation(std::size_t old_bytes,
                                       std::size_t new_bytes) {
  auto &telemetry = retained_launch_buffer_telemetry();
  telemetry.allocation_calls.fetch_add(1, std::memory_order_relaxed);
  const auto current = telemetry.current_bytes.fetch_add(
                           new_bytes - old_bytes, std::memory_order_relaxed) +
                       new_bytes - old_bytes;
  update_retained_launch_peak(current);
}

void record_retained_launch_release(std::size_t bytes) {
  auto &telemetry = retained_launch_buffer_telemetry();
  telemetry.release_calls.fetch_add(1, std::memory_order_relaxed);
  const auto previous =
      telemetry.current_bytes.fetch_sub(bytes, std::memory_order_relaxed);
  TI_ASSERT(previous >= bytes);
}

void bind_cuda_native_functions(JITModule *jit_module,
                                std::vector<OffloadedTask> &tasks) {
  auto *cuda_module = dynamic_cast<JITModuleCUDA *>(jit_module);
  TI_ASSERT(cuda_module != nullptr);
  for (auto &task : tasks) {
    task.native_function = cuda_module->lookup_function(task.name);
    TI_ASSERT(task.native_function != nullptr);
  }
}

}  // namespace

KernelLauncher::KernelLauncher(LLVM::KernelLauncher::Config config)
    : Base(std::move(config)),
      retain_ordinary_launch_buffers_(
          get_environ_config("TI_CUDA_RETAIN_ORDINARY_LAUNCH_BUFFERS", 1) !=
          0),
      ordinary_launch_arg_ring_size_(static_cast<std::size_t>(std::clamp(
          get_environ_config("TI_CUDA_RETAINED_LAUNCH_ARG_RING_SIZE", 1), 1,
          static_cast<int>(kMaxOrdinaryLaunchArgRingSize)))) {
}

RetainedLaunchBufferTelemetrySnapshot
get_retained_launch_buffer_telemetry_snapshot() {
  const auto &telemetry = retained_launch_buffer_telemetry();
  return {
      telemetry.current_bytes.load(std::memory_order_relaxed),
      telemetry.peak_bytes.load(std::memory_order_relaxed),
      telemetry.allocation_calls.load(std::memory_order_relaxed),
      telemetry.release_calls.load(std::memory_order_relaxed),
  };
}

GridResidencyTelemetrySnapshot get_grid_residency_telemetry_snapshot() {
  const auto &telemetry = grid_residency_telemetry();
  return {
      telemetry.resolution_calls.load(std::memory_order_relaxed),
      telemetry.resolution_failures.load(std::memory_order_relaxed),
      telemetry.last_requested_waves.load(std::memory_order_relaxed),
      telemetry.last_baseline_grid.load(std::memory_order_relaxed),
      telemetry.last_resolved_grid.load(std::memory_order_relaxed),
      telemetry.last_active_blocks_per_multiprocessor.load(
          std::memory_order_relaxed),
      telemetry.last_multiprocessor_count.load(std::memory_order_relaxed),
      telemetry.coarsening_resolution_calls.load(std::memory_order_relaxed),
      telemetry.coarsening_resolution_failures.load(
          std::memory_order_relaxed),
      telemetry.last_work_per_thread_target.load(std::memory_order_relaxed),
      telemetry.last_coarsening_baseline_grid.load(std::memory_order_relaxed),
      telemetry.last_coarsened_grid.load(std::memory_order_relaxed),
  };
}

ArtifactQualificationTelemetrySnapshot
get_artifact_qualification_telemetry_snapshot() {
  const auto &telemetry = artifact_qualification_telemetry();
  return {
      telemetry.qualification_calls.load(std::memory_order_relaxed),
      telemetry.registration_materializations.load(std::memory_order_relaxed),
      telemetry.function_attribute_queries.load(std::memory_order_relaxed),
      telemetry.occupancy_queries.load(std::memory_order_relaxed),
  };
}

KernelLauncher::RetainedDeviceBuffer::~RetainedDeviceBuffer() {
  try {
    release();
  } catch (const std::exception &error) {
    TI_WARN("Failed to release a retained CUDA launch buffer: {}",
            error.what());
  } catch (...) {
    TI_WARN("Failed to release a retained CUDA launch buffer");
  }
}

void *KernelLauncher::RetainedDeviceBuffer::reserve(
    std::size_t required_bytes) const {
  if (required_bytes == 0) {
    return nullptr;
  }
  if (ptr != nullptr && capacity >= required_bytes) {
    return ptr;
  }

  void *replacement = nullptr;
  CUDADriver::get_instance().malloc_async(&replacement, required_bytes,
                                          nullptr);
  const auto old_capacity = capacity;
  if (ptr != nullptr) {
    try {
      CUDADriver::get_instance().mem_free_async(ptr, nullptr);
    } catch (...) {
      try {
        CUDADriver::get_instance().mem_free_async(replacement, nullptr);
      } catch (...) {
      }
      throw;
    }
  }
  ptr = replacement;
  capacity = required_bytes;
  record_retained_launch_allocation(old_capacity, capacity);
  return ptr;
}

void KernelLauncher::RetainedDeviceBuffer::release() const {
  if (ptr == nullptr) {
    return;
  }
  CUDAContext::get_instance().make_current();
  CUDADriver::get_instance().mem_free_async(ptr, nullptr);
  record_retained_launch_release(capacity);
  ptr = nullptr;
  capacity = 0;
}

namespace {

std::shared_ptr<void> own_cuda_allocation(void *ptr) {
  return std::shared_ptr<void>(ptr, [](void *allocation) {
    if (allocation != nullptr) {
      CUDAContext::get_instance().make_current();
      CUDADriver::get_instance().mem_free(allocation);
    }
  });
}

void add_cuda_graph_execution_gate(LLVMCompiledKernel &compiled) {
  TI_ASSERT(compiled.module != nullptr);
  auto &module = *compiled.module;
  auto &llvm_context = module.getContext();
  auto *pointer_type = llvm::PointerType::get(llvm_context, 0);
  auto *i8_type = llvm::Type::getInt8Ty(llvm_context);
  auto *i32_type = llvm::Type::getInt32Ty(llvm_context);
  auto *i64_type = llvm::Type::getInt64Ty(llvm_context);

  for (const auto &task : compiled.tasks) {
    auto *function = module.getFunction(task.name);
    TI_ASSERT(function != nullptr && !function->empty());
    TI_ASSERT(function->getReturnType()->isVoidTy());
    TI_ASSERT(function->arg_size() == 1);
    auto *original_entry = &function->getEntryBlock();
    auto *gate_entry = llvm::BasicBlock::Create(
        llvm_context, "graph_execution_gate", function, original_entry);
    auto *masked_exit = llvm::BasicBlock::Create(
        llvm_context, "graph_execution_masked_exit", function, original_entry);
    llvm::IRBuilder<> builder(gate_entry);
    // RuntimeContext::arg_buffer is its first member. The private gated launch
    // allocation stores GraphExecutionGateBinding immediately before that
    // pointer, so the variant can inline the test without adding a runtime
    // bitcode symbol or changing RuntimeContext's split-wheel ABI.
    auto *arg_buffer = builder.CreateAlignedLoad(
        pointer_type, function->getArg(0), llvm::Align(8), "graph_args");
    auto *binding = builder.CreateInBoundsGEP(
        i8_type, arg_buffer,
        llvm::ConstantInt::getSigned(
            i64_type,
            -static_cast<std::int64_t>(sizeof(GraphExecutionGateBinding))),
        "graph_gate_binding");
    auto *gate_bits = builder.CreateAlignedLoad(
        i64_type, binding, llvm::Align(8), "graph_gate_address");
    auto *expected_address = builder.CreateInBoundsGEP(
        i8_type, binding,
        llvm::ConstantInt::get(i64_type,
                               offsetof(GraphExecutionGateBinding, expected)));
    auto *expected = builder.CreateAlignedLoad(
        i32_type, expected_address, llvm::Align(4), "graph_gate_expected");
    auto *gate = builder.CreateIntToPtr(gate_bits, pointer_type);
    auto *value = builder.CreateAlignedLoad(i32_type, gate, llvm::Align(4),
                                            "graph_gate_value");
    auto *execute = builder.CreateICmpEQ(value, expected);
    builder.CreateCondBr(execute, original_entry, masked_exit);
    builder.SetInsertPoint(masked_exit);
    builder.CreateRetVoid();
  }
  TI_ASSERT(!llvm::verifyModule(module, &llvm::errs()));
}

}  // namespace

void KernelLauncher::configure_root_binding(
    const LLVM::CompiledKernelData &compiled,
    Context &context) {
  // The result-buffer slot is otherwise unused for a no-return kernel. New
  // kernels use it for the compact binding; old cached kernels safely ignore
  // it and continue resolving roots through the runtime directory.
  context.uses_root_binding = compiled.get_internal_data().rets.empty() &&
                              !context.snode_tree_ids.empty();
  if (!context.uses_root_binding) {
    return;
  }

  TI_ASSERT(compiled.get_internal_data().rets.empty());
  TI_ASSERT(!context.snode_tree_ids.empty());
}

void KernelLauncher::ensure_root_binding(const Context &context) {
  if (!context.uses_root_binding) {
    return;
  }

  std::call_once(context.root_binding_once, [&] {
    TI_TRACE("Initializing CUDA root binding for {} SNodeTree(s)",
             context.snode_tree_ids.size());
    auto *executor = get_runtime_executor();
    std::vector<void *> roots;
    roots.reserve(context.snode_tree_ids.size());
    for (int tree_id : context.snode_tree_ids) {
      roots.push_back(executor->get_snode_tree_root_ptr(tree_id));
    }

    if (roots.size() == 1) {
      context.root_binding = roots.front();
      TI_TRACE("Using direct CUDA root binding {}", context.root_binding);
      return;
    }

    CUDAContext::get_instance().make_current();
    const std::size_t binding_bytes = roots.size() * sizeof(void *);
    void *device_binding = nullptr;
    CUDADriver::get_instance().malloc(&device_binding, binding_bytes);
    auto owner = own_cuda_allocation(device_binding);
    CUDADriver::get_instance().memcpy_host_to_device(
        device_binding, roots.data(), binding_bytes);
    context.root_binding = device_binding;
    context.root_binding_owner = std::move(owner);
    TI_TRACE("Using compact CUDA root table {} ({} bytes)",
             context.root_binding, binding_bytes);
  });
}

const std::vector<OffloadedTask> &
KernelLauncher::resolve_grid_policy_tasks(
    const Context &context,
    std::int32_t waves,
    std::int32_t range_work_per_thread_target) {
  TI_ERROR_IF(waves != 0 && waves != 1 && waves != 2 && waves != 4,
              "CUDA grid residency waves must be 0, 1, 2, or 4");
  TI_ERROR_IF(range_work_per_thread_target != 1 &&
                  range_work_per_thread_target != 2 &&
                  range_work_per_thread_target != 4 &&
                  range_work_per_thread_target != 8,
              "CUDA range work-per-thread target must be 1, 2, 4, or 8");
  TI_ERROR_IF(waves == 0 && range_work_per_thread_target == 1,
              "baseline CUDA grid policy does not require resolution");
  const std::size_t wave_slot =
      waves == 0 ? 0 : (waves == 1 ? 1 : (waves == 2 ? 2 : 3));
  const std::size_t work_slot =
      range_work_per_thread_target == 1
          ? 0
          : (range_work_per_thread_target == 2
                 ? 1
                 : (range_work_per_thread_target == 4 ? 2 : 3));
  const std::size_t slot = wave_slot * 4 + work_slot - 1;
  std::call_once(context.grid_policy_once[slot], [&]() {
    auto &telemetry = grid_residency_telemetry();
    if (waves != 0) {
      telemetry.resolution_calls.fetch_add(1, std::memory_order_relaxed);
    }
    if (range_work_per_thread_target != 1) {
      telemetry.coarsening_resolution_calls.fetch_add(
          1, std::memory_order_relaxed);
    }
    try {
      auto resolved = context.offloaded_tasks;
      std::size_t range_tasks = 0;
      for (auto &task : resolved) {
        if (task.task_type != OffloadedTaskType::range_for) {
          continue;
        }
        ++range_tasks;
        TI_ERROR_IF(
            task.one_to_one ||
                task.sparse_list_op != OffloadedTask::kSparseListOpNone ||
                task.may_mutate_sparse_topology || task.grid_dim <= 0 ||
                task.block_dim <= 0 || task.dynamic_shared_array_bytes < 0,
            "CUDA grid policy requires a dense grid-stride range task");
        if (range_work_per_thread_target != 1) {
          TI_ERROR_IF(
              task.constant_range_size < 0,
              "CUDA range work-per-thread control requires a constant range");
          const int baseline_grid = task.grid_dim;
          const std::int64_t workgroup_span =
              static_cast<std::int64_t>(task.block_dim) *
              range_work_per_thread_target;
          const std::int64_t target_grid = std::max<std::int64_t>(
              1, (task.constant_range_size + workgroup_span - 1) /
                     workgroup_span);
          task.grid_dim = static_cast<int>(std::min<std::int64_t>(
              task.grid_dim, target_grid));
          telemetry.last_work_per_thread_target.store(
              range_work_per_thread_target, std::memory_order_relaxed);
          telemetry.last_coarsening_baseline_grid.store(
              baseline_grid, std::memory_order_relaxed);
          telemetry.last_coarsened_grid.store(task.grid_dim,
                                               std::memory_order_relaxed);
        }
        if (waves != 0) {
          void *function = task.native_function;
          TI_ASSERT(function != nullptr);
          int active_blocks_per_multiprocessor = 0;
          CUDADriver::get_instance().kernel_get_occupancy(
              &active_blocks_per_multiprocessor, function, task.block_dim,
              static_cast<std::size_t>(task.dynamic_shared_array_bytes));
          int multiprocessor_count = 0;
          CUDADriver::get_instance().device_get_attribute(
              &multiprocessor_count,
              CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
              CUDAContext::get_instance().get_device());
          TI_ERROR_IF(active_blocks_per_multiprocessor <= 0 ||
                          multiprocessor_count <= 0,
                      "CUDA occupancy did not produce a positive residency");
          const std::int64_t cap =
              static_cast<std::int64_t>(active_blocks_per_multiprocessor) *
              multiprocessor_count * waves;
          const int baseline_grid = task.grid_dim;
          task.grid_dim = static_cast<int>(
              std::max<std::int64_t>(1, std::min<std::int64_t>(task.grid_dim,
                                                              cap)));
          telemetry.last_requested_waves.store(waves,
                                               std::memory_order_relaxed);
          telemetry.last_baseline_grid.store(baseline_grid,
                                             std::memory_order_relaxed);
          telemetry.last_resolved_grid.store(task.grid_dim,
                                             std::memory_order_relaxed);
          telemetry.last_active_blocks_per_multiprocessor.store(
              active_blocks_per_multiprocessor, std::memory_order_relaxed);
          telemetry.last_multiprocessor_count.store(multiprocessor_count,
                                                    std::memory_order_relaxed);
        }
      }
      TI_ERROR_IF(range_tasks != 1,
                  "CUDA grid policy requires exactly one range task; got {}",
                  range_tasks);
      context.grid_policy_tasks[slot] = std::move(resolved);
    } catch (...) {
      if (waves != 0) {
        telemetry.resolution_failures.fetch_add(1,
                                                std::memory_order_relaxed);
      }
      if (range_work_per_thread_target != 1) {
        telemetry.coarsening_resolution_failures.fetch_add(
            1, std::memory_order_relaxed);
      }
      throw;
    }
  });
  return context.grid_policy_tasks[slot];
}

const KernelLauncher::Context::ResolvedTaskExecutionPlan *
KernelLauncher::resolve_task_execution_plan(
    const Context &context,
    const LaunchContextBuilder &launch_context) {
  const auto &identity =
      launch_context.cuda_task_execution_plan_identity();
  const auto &content_digest =
      launch_context.cuda_task_execution_plan_content_digest();
  if (const auto *cached = context.task_execution_plan_mru.load(
          std::memory_order_acquire);
      cached != nullptr && cached->content_digest == content_digest &&
      cached->identity == identity) {
    return cached;
  }
  const auto &task_kinds =
      launch_context.cuda_task_execution_plan_kinds();
  const auto &waves = launch_context.cuda_task_grid_residency_waves();
  const auto &work =
      launch_context.cuda_task_range_work_per_thread_targets();
  const auto &sparse_policies =
      launch_context.cuda_task_sparse_list_policies();
  TI_ERROR_IF(identity.empty(),
              "CUDA task execution plan requires a non-empty identity");
  TI_ERROR_IF(task_kinds.size() != context.offloaded_tasks.size() ||
                  waves.size() != context.offloaded_tasks.size() ||
                  work.size() != context.offloaded_tasks.size() ||
                  sparse_policies.size() != context.offloaded_tasks.size(),
              "CUDA task execution plan topology mismatch: plan has {} "
              "task(s), artifact has {}",
              task_kinds.size(), context.offloaded_tasks.size());

  std::lock_guard<std::mutex> lock(context.task_execution_plan_mutex);
  if (auto found = context.task_execution_plans.find(identity);
      found != context.task_execution_plans.end()) {
    const auto &cached = *found->second;
    TI_ERROR_IF(cached.task_kinds != task_kinds ||
                    cached.grid_residency_waves != waves ||
                    cached.range_work_per_thread_targets != work ||
                    cached.sparse_list_policies != sparse_policies,
                "CUDA task execution plan identity was reused with different "
                "contents");
    TI_ASSERT(cached.content_digest == content_digest);
    context.task_execution_plan_mru.store(found->second.get(),
                                          std::memory_order_release);
    return found->second.get();
  }
  TI_ERROR_IF(context.task_execution_plans.size() >= 4096,
              "CUDA task execution plan cache reached its bounded 4096-plan "
              "capacity");

  auto resolved =
      std::make_shared<Context::ResolvedTaskExecutionPlan>();
  resolved->identity = identity;
  resolved->content_digest = content_digest;
  resolved->task_kinds = task_kinds;
  resolved->grid_residency_waves = waves;
  resolved->range_work_per_thread_targets = work;
  resolved->sparse_list_policies = sparse_policies;
  resolved->tasks = context.offloaded_tasks;
  auto &telemetry = grid_residency_telemetry();
  const bool uses_waves =
      std::any_of(waves.begin(), waves.end(), [](int value) {
        return value != 0;
      });
  const bool uses_work = std::any_of(work.begin(), work.end(), [](int value) {
    return value != 1;
  });
  if (uses_waves) {
    telemetry.resolution_calls.fetch_add(1, std::memory_order_relaxed);
    CUDAContext::get_instance().make_current();
  }
  if (uses_work) {
    telemetry.coarsening_resolution_calls.fetch_add(
        1, std::memory_order_relaxed);
  }
  try {
    for (std::size_t index = 0; index < resolved->tasks.size(); ++index) {
      auto &task = resolved->tasks[index];
      const auto actual_kind = offloaded_task_type_name(task.task_type);
      TI_ERROR_IF(task_kinds[index] != actual_kind,
                  "CUDA task execution plan topology mismatch at task {}: "
                  "expected {}, got {}",
                  index, task_kinds[index], actual_kind);
      TI_ERROR_IF(sparse_policies[index] != 0 && sparse_policies[index] != 1,
                  "CUDA sparse list policy is invalid at task {}", index);
      if (sparse_policies[index] == 1) {
        TI_ERROR_IF(
            task.task_type != OffloadedTaskType::listgen ||
                task.sparse_list_op != OffloadedTask::kSparseListOpListgen ||
                task.sparse_list_parent_grid_bound <= 0 || task.grid_dim <= 0,
            "CUDA parent-bound sparse traversal requires an exact listgen "
            "task at ordinal {}",
            index);
        task.grid_dim =
            std::min(task.grid_dim, task.sparse_list_parent_grid_bound);
      }
      if (waves[index] == 0 && work[index] == 1) {
        continue;
      }
      TI_ERROR_IF(
          task.task_type != OffloadedTaskType::range_for || task.one_to_one ||
              task.sparse_list_op != OffloadedTask::kSparseListOpNone ||
              task.may_mutate_sparse_topology || task.grid_dim <= 0 ||
              task.block_dim <= 0 || task.dynamic_shared_array_bytes < 0,
          "CUDA task execution plan requires an eligible dense grid-stride "
          "range task at ordinal {}",
          index);
      if (work[index] != 1) {
        TI_ERROR_IF(task.constant_range_size < 0,
                    "CUDA task execution plan work-per-thread control "
                    "requires a constant range at task {}",
                    index);
        const int baseline_grid = task.grid_dim;
        const std::int64_t workgroup_span =
            static_cast<std::int64_t>(task.block_dim) * work[index];
        const std::int64_t target_grid = std::max<std::int64_t>(
            1, (task.constant_range_size + workgroup_span - 1) /
                   workgroup_span);
        task.grid_dim = static_cast<int>(std::min<std::int64_t>(
            task.grid_dim, target_grid));
        telemetry.last_work_per_thread_target.store(
            work[index], std::memory_order_relaxed);
        telemetry.last_coarsening_baseline_grid.store(
            baseline_grid, std::memory_order_relaxed);
        telemetry.last_coarsened_grid.store(task.grid_dim,
                                             std::memory_order_relaxed);
      }
      if (waves[index] != 0) {
        void *function = task.native_function;
        TI_ASSERT(function != nullptr);
        int active_blocks_per_multiprocessor = 0;
        CUDADriver::get_instance().kernel_get_occupancy(
            &active_blocks_per_multiprocessor, function, task.block_dim,
            static_cast<std::size_t>(task.dynamic_shared_array_bytes));
        int multiprocessor_count = 0;
        CUDADriver::get_instance().device_get_attribute(
            &multiprocessor_count,
            CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            CUDAContext::get_instance().get_device());
        TI_ERROR_IF(active_blocks_per_multiprocessor <= 0 ||
                        multiprocessor_count <= 0,
                    "CUDA occupancy did not produce a positive residency for "
                    "task {}",
                    index);
        const std::int64_t cap =
            static_cast<std::int64_t>(active_blocks_per_multiprocessor) *
            multiprocessor_count * waves[index];
        const int baseline_grid = task.grid_dim;
        task.grid_dim = static_cast<int>(std::max<std::int64_t>(
            1, std::min<std::int64_t>(task.grid_dim, cap)));
        telemetry.last_requested_waves.store(waves[index],
                                             std::memory_order_relaxed);
        telemetry.last_baseline_grid.store(baseline_grid,
                                           std::memory_order_relaxed);
        telemetry.last_resolved_grid.store(task.grid_dim,
                                           std::memory_order_relaxed);
        telemetry.last_active_blocks_per_multiprocessor.store(
            active_blocks_per_multiprocessor, std::memory_order_relaxed);
        telemetry.last_multiprocessor_count.store(
            multiprocessor_count, std::memory_order_relaxed);
      }
    }
  } catch (...) {
    if (uses_waves) {
      telemetry.resolution_failures.fetch_add(1,
                                              std::memory_order_relaxed);
    }
    if (uses_work) {
      telemetry.coarsening_resolution_failures.fetch_add(
          1, std::memory_order_relaxed);
    }
    throw;
  }
  const auto [inserted, did_insert] =
      context.task_execution_plans.emplace(identity, resolved);
  TI_ASSERT(did_insert);
  context.task_execution_plan_mru.store(inserted->second.get(),
                                        std::memory_order_release);
  return inserted->second.get();
}

bool KernelLauncher::on_cuda_device(void *ptr) {
  unsigned int attr_val = 0;
  uint32_t ret_code = CUDADriver::get_instance().mem_get_attribute.call(
      &attr_val, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (void *)ptr);

  return ret_code == CUDA_SUCCESS && attr_val == CU_MEMORYTYPE_DEVICE;
}

int64 KernelLauncher::get_sparse_list_version(int snode_id) const {
  auto it = sparse_list_states_.find(snode_id);
  if (it == sparse_list_states_.end()) {
    return 0;
  }
  return it->second.version;
}

bool KernelLauncher::sparse_list_task_is_current(const OffloadedTask &task) {
  SparseListgenNodeStatistics *telemetry = nullptr;
  if (sparse_listgen_telemetry_enabled_ &&
      task.sparse_list_op == OffloadedTask::kSparseListOpListgen &&
      task.sparse_list_snode_id >= 0) {
    telemetry = &sparse_listgen_telemetry_[task.sparse_list_snode_id];
    telemetry->snode_id = task.sparse_list_snode_id;
    telemetry->parent_snode_id = task.sparse_list_parent_snode_id;
    ++telemetry->requests;
  }
  if (task.sparse_list_op == OffloadedTask::kSparseListOpNone ||
      task.sparse_list_snode_id < 0 ||
      task.sparse_list_parent_snode_id < 0) {
    return false;
  }

  auto it = sparse_list_states_.find(task.sparse_list_snode_id);
  if (it == sparse_list_states_.end()) {
    if (telemetry != nullptr) {
      telemetry->last_rebuild_reason = "cold";
    }
    if (listgen_reuse_adaptive_) {
      auto &state = sparse_list_states_[task.sparse_list_snode_id];
      record_sparse_list_reuse_sample(state, /*would_skip=*/false);
    }
    return false;
  }
  auto &state = it->second;
  const bool would_skip = state.clean_epoch == state.dirty_epoch &&
                 state.clean_parent_version ==
                   get_sparse_list_version(
                     task.sparse_list_parent_snode_id);
  record_sparse_list_reuse_sample(state, would_skip);
  if (telemetry != nullptr) {
    if (would_skip && !state.adaptive_disabled) {
      ++telemetry->reuse_hits;
    } else if (state.clean_epoch != state.dirty_epoch) {
      telemetry->last_rebuild_reason = "topology_dirty";
    } else if (state.clean_parent_version !=
               get_sparse_list_version(task.sparse_list_parent_snode_id)) {
      telemetry->last_rebuild_reason = "parent_version_changed";
    } else if (state.adaptive_disabled) {
      telemetry->last_rebuild_reason = "adaptive_reuse_disabled";
    } else {
      telemetry->last_rebuild_reason = "not_current";
    }
  }
  return would_skip && !state.adaptive_disabled;
}

void KernelLauncher::record_sparse_list_reuse_sample(SparseListState &state,
                                                     bool would_skip) const {
  if (!listgen_reuse_adaptive_) {
    return;
  }
  constexpr int kWindow = 64;
  constexpr int kDisablePercent = 10;
  constexpr int kEnablePercent = 15;
  const int hit = would_skip ? 1 : 0;
  if (state.adaptive_window_size < kWindow) {
    state.adaptive_window_size++;
  } else {
    state.adaptive_hit_count -=
        int((state.adaptive_window_bits >> (kWindow - 1)) & 1u);
  }
  state.adaptive_window_bits = (state.adaptive_window_bits << 1) |
                               static_cast<std::uint64_t>(hit);
  state.adaptive_hit_count += hit;

  if (state.adaptive_window_size < kWindow) {
    return;
  }
  if (!state.adaptive_disabled &&
      state.adaptive_hit_count * 100 < kDisablePercent * kWindow) {
    state.adaptive_disabled = true;
  } else if (state.adaptive_disabled &&
             state.adaptive_hit_count * 100 >= kEnablePercent * kWindow) {
    state.adaptive_disabled = false;
  }
}

void KernelLauncher::mark_sparse_list_task_launched(
    const OffloadedTask &task) {
  if (task.sparse_list_op != OffloadedTask::kSparseListOpListgen ||
      task.sparse_list_snode_id < 0 ||
      task.sparse_list_parent_snode_id < 0) {
    return;
  }

  if (sparse_listgen_telemetry_enabled_) {
    auto &telemetry = sparse_listgen_telemetry_[task.sparse_list_snode_id];
    telemetry.snode_id = task.sparse_list_snode_id;
    telemetry.parent_snode_id = task.sparse_list_parent_snode_id;
    ++telemetry.rebuilds;
    if (!telemetry.candidate_slots_dispatched.available) {
      telemetry.candidate_slots_dispatched = {0, true};
    }
    telemetry.candidate_slots_dispatched.value +=
        static_cast<std::uint64_t>(task.grid_dim) * task.block_dim;
  }

  auto &state = sparse_list_states_[task.sparse_list_snode_id];
  state.clean_epoch = state.dirty_epoch;
  state.clean_parent_version =
      get_sparse_list_version(task.sparse_list_parent_snode_id);
  state.version++;
}

void KernelLauncher::invalidate_sparse_list_cache(
    int sparse_mutation_snode_id) {
  if (sparse_mutation_snode_id >= 0) {
    sparse_list_states_[sparse_mutation_snode_id].dirty_epoch++;
    if (sparse_listgen_telemetry_enabled_) {
      auto &telemetry =
          sparse_listgen_telemetry_[sparse_mutation_snode_id];
      telemetry.snode_id = sparse_mutation_snode_id;
      ++telemetry.invalidations;
    }
    return;
  }
  for (auto &kv : sparse_list_states_) {
    kv.second.dirty_epoch++;
    if (sparse_listgen_telemetry_enabled_) {
      auto &telemetry = sparse_listgen_telemetry_[kv.first];
      telemetry.snode_id = kv.first;
      ++telemetry.invalidations;
    }
  }
}

bool KernelLauncher::prepare_cuda_graph_context(Handle handle,
                                                LaunchContextBuilder &ctx,
                                                RuntimeContext &context,
                                                std::vector<OffloadedTask>
                                                    *resolved_tasks,
                                                std::string
                                                    *task_plan_identity,
                                                std::size_t
                                                    *resolved_task_count) {
  std::shared_lock<std::shared_mutex> launch_lock(registration_mutex());
  std::shared_ptr<const Context> launcher_ctx;
  auto iter = contexts_.find(handle.get_launch_id());
  TI_ASSERT(iter != contexts_.end());
  launcher_ctx = iter->second;
  ensure_root_binding(*launcher_ctx);
  const auto &parameters = launcher_ctx->parameters;
  const Context::ResolvedTaskExecutionPlan *plan = nullptr;
  const std::vector<OffloadedTask> *offloaded_tasks_ptr =
      &launcher_ctx->offloaded_tasks;
  if (ctx.has_cuda_task_execution_plan()) {
    TI_ERROR_IF(ctx.cuda_grid_residency_waves() != 0 ||
                    ctx.cuda_range_work_per_thread_target() != 1,
                "task-indexed and call-wide CUDA launch controls cannot be "
                "combined during Graph capture");
    plan = resolve_task_execution_plan(*launcher_ctx, ctx);
    offloaded_tasks_ptr = &plan->tasks;
  } else if (ctx.cuda_grid_residency_waves() != 0 ||
             ctx.cuda_range_work_per_thread_target() != 1) {
    // The legacy call-wide policy is intentionally not a Graph contract.
    return false;
  }
  const auto &offloaded_tasks = *offloaded_tasks_ptr;
  for (const auto &task : offloaded_tasks) {
    if (task.sparse_list_op != OffloadedTask::kSparseListOpNone ||
        task.may_mutate_sparse_topology) {
      return false;
    }
  }
  if (ctx.result_buffer_size > 0) {
    return false;
  }

  auto *executor = get_runtime_executor();
  ctx.get_context().runtime = executor->get_llvm_runtime();
  for (int i = 0; i < (int)parameters.size(); i++) {
    const auto &kv = parameters[i];
    const auto &key = kv.first;
    const auto &parameter = kv.second;
    if (parameter.is_array) {
      if (ctx.device_allocation_type[key] ==
          LaunchContextBuilder::DevAllocType::kTexture) {
        // Program resolved the generation-qualified texture before capture;
        // the owning Graph retains its registry lease, not a buffer lease.
        const auto found = ctx.array_ptrs.find(key);
        if (found == ctx.array_ptrs.end() || found->second == nullptr) {
          return false;
        }
        const auto object =
            *static_cast<const std::uint64_t *>(found->second);
        if (object == 0) {
          return false;
        }
        auto object_key = key;
        object_key.push_back(TypeFactory::DATA_PTR_POS_IN_NDARRAY);
        ctx.set_struct_arg<std::uint64_t>(object_key, object);
        continue;
      }
      const auto arr_sz = ctx.array_runtime_sizes[key];
      if (arr_sz == 0) {
        continue;
      }
      std::vector<int> data_ptr_idx = key;
      data_ptr_idx.push_back(TypeFactory::DATA_PTR_POS_IN_NDARRAY);
      auto data_ptr = ctx.array_ptrs[data_ptr_idx];
      std::vector<int> grad_ptr_idx = key;
      grad_ptr_idx.push_back(TypeFactory::GRAD_PTR_POS_IN_NDARRAY);
      auto grad_ptr = ctx.array_ptrs[grad_ptr_idx];
      if (ctx.device_allocation_type[key] ==
          LaunchContextBuilder::DevAllocType::kNone) {
        if (!on_cuda_device(data_ptr)) {
          return false;
        }
        ctx.set_ndarray_ptrs(key, reinterpret_cast<uint64>(data_ptr),
                             reinterpret_cast<uint64>(grad_ptr));
      } else if (ctx.device_allocation_type[key] ==
                 LaunchContextBuilder::DevAllocType::kNdarray) {
        DeviceAllocation *ptr = static_cast<DeviceAllocation *>(data_ptr);
        auto *data_device_ptr = executor->get_device_alloc_info_ptr(*ptr);
        void *grad_device_ptr = nullptr;
        if (grad_ptr != nullptr) {
          ptr = static_cast<DeviceAllocation *>(grad_ptr);
          grad_device_ptr = executor->get_device_alloc_info_ptr(*ptr);
        }
        ctx.set_ndarray_ptrs(key, reinterpret_cast<uint64>(data_device_ptr),
                             reinterpret_cast<uint64>(grad_device_ptr));
      } else if (ctx.device_allocation_type[key] ==
                 LaunchContextBuilder::DevAllocType::kDenseStorage) {
        const auto &binding = ctx.get_resolved_dense_storage(key);
        auto *base = reinterpret_cast<char *>(
            executor->get_device_alloc_info_ptr(binding.allocation));
        ctx.set_ndarray_ptrs(
            key, reinterpret_cast<uint64>(base + binding.byte_offset), 0);
      } else {
        return false;
      }
    } else if (parameter.is_argpack) {
      std::vector<int> data_ptr_idx = key;
      data_ptr_idx.push_back(TypeFactory::DATA_PTR_POS_IN_ARGPACK);
      auto *argpack = ctx.argpack_ptrs[key];
      auto argpack_ptr = argpack->get_device_allocation();
      auto *device_ptr = executor->get_device_alloc_info_ptr(argpack_ptr);
      if (key.size() == 1) {
        ctx.set_argpack_ptr(key, reinterpret_cast<uint64>(device_ptr));
      } else {
        auto key_parent = key;
        key_parent.pop_back();
        auto *argpack_parent = ctx.argpack_ptrs[key_parent];
        argpack_parent->set_arg_nested_argpack_ptr(
            key.back(), reinterpret_cast<uint64>(device_ptr));
      }
    }
  }

  context = ctx.get_context();
  // A no-return CUDA field task repurposes this otherwise unused slot for its
  // registration-owned compact root binding. Other tasks canonicalize it so
  // capture packets do not retain a transient host address.
  context.result_buffer = launcher_ctx->uses_root_binding
                              ? static_cast<uint64_t *>(
                                    launcher_ctx->root_binding)
                              : nullptr;
  if (resolved_tasks != nullptr) {
    *resolved_tasks = offloaded_tasks;
  }
  if (resolved_task_count != nullptr) {
    *resolved_task_count = offloaded_tasks.size();
  }
  if (task_plan_identity != nullptr) {
    *task_plan_identity = ctx.cuda_task_execution_plan_identity();
  }
  return true;
}

bool KernelLauncher::prepare_cuda_graph_launch(Handle handle,
                                               LaunchContextBuilder &ctx,
                                               GraphLaunchPacket &packet,
                                               void *stream) {
  if (!prepare_cuda_graph_launch_packet(handle, ctx, packet)) {
    return false;
  }
  if (packet.arg_buffer_size == 0) {
    return true;
  }
  CUDADriver::get_instance().malloc_async(&packet.device_arg_buffer,
                                          packet.arg_buffer_size, stream);
  CUDADriver::get_instance().memcpy_host_to_device_async(
      packet.device_arg_buffer, packet.context.arg_buffer,
      packet.arg_buffer_size, stream);
  packet.context.arg_buffer = static_cast<char *>(packet.device_arg_buffer);
  return true;
}

bool KernelLauncher::prepare_cuda_graph_launch_packet(
    Handle handle,
    LaunchContextBuilder &ctx,
    GraphLaunchPacket &packet) {
  RuntimeContext context;
  if (!prepare_cuda_graph_context(handle, ctx, context, &packet.offloaded_tasks,
                                  &packet.task_execution_plan_identity)) {
    return false;
  }
  packet.handle = handle;
  packet.dispatch_label = ctx.dispatch_label();
  packet.arg_buffer_size = ctx.arg_buffer_size;
  packet.arg_buffer_prefix_size = 0;
  packet.device_arg_buffer_size = packet.arg_buffer_size;
  packet.bounded_extent = 0;
  packet.bounded_capacity = 0;
  packet.bounded_range = false;
  packet.context = context;
  if (packet.arg_buffer_size == 0) {
    // A Field-only graph has no runtime argument storage. The RuntimeContext
    // still carries the runtime/root state used by statically bound fields,
    // but capture needs neither a device allocation nor an H2D upload.
    packet.device_arg_buffer = nullptr;
    packet.context.arg_buffer = nullptr;
    return true;
  }
  return true;
}

bool KernelLauncher::prepare_cuda_graph_bounded_range(
    Handle handle,
    LaunchContextBuilder &ctx,
    GraphLaunchPacket &packet,
    void *extent,
    std::uint32_t capacity,
    void *stream) {
  if (extent == nullptr || capacity == 0 || capacity > 0x7fffffffu) {
    return false;
  }
  RuntimeContext context;
  if (!prepare_cuda_graph_context(handle, ctx, context,
                                  &packet.offloaded_tasks,
                                  &packet.task_execution_plan_identity)) {
    return false;
  }

  CudaBoundedRangeBinding binding;
  binding.extent = reinterpret_cast<std::uintptr_t>(extent);
  binding.capacity = static_cast<std::int32_t>(capacity);
  packet.handle = handle;
  packet.dispatch_label = ctx.dispatch_label();
  packet.arg_buffer_size = ctx.arg_buffer_size;
  packet.arg_buffer_prefix_size = sizeof(binding);
  packet.device_arg_buffer_size =
      packet.arg_buffer_prefix_size + packet.arg_buffer_size;
  packet.bounded_extent = binding.extent;
  packet.bounded_capacity = capacity;
  packet.bounded_range = true;
  packet.context = context;
  CUDADriver::get_instance().malloc_async(
      &packet.device_arg_buffer, packet.device_arg_buffer_size, stream);
  CUDADriver::get_instance().memcpy_host_to_device_async(
      packet.device_arg_buffer, &binding, sizeof(binding), stream);
  auto *device_args = static_cast<char *>(packet.device_arg_buffer) +
                      packet.arg_buffer_prefix_size;
  if (packet.arg_buffer_size > 0) {
    CUDADriver::get_instance().memcpy_host_to_device_async(
        device_args, packet.context.arg_buffer, packet.arg_buffer_size, stream);
  }
  packet.context.arg_buffer = device_args;
  return true;
}

bool KernelLauncher::prepare_cuda_graph_gated_launch(Handle handle,
                                                     LaunchContextBuilder &ctx,
                                                     GraphLaunchPacket &packet,
                                                     void *gate,
                                                     std::uint32_t expected,
                                                     void *stream) {
  if (gate == nullptr || expected == 0) {
    return false;
  }
  RuntimeContext context;
  if (!prepare_cuda_graph_context(handle, ctx, context,
                                  &packet.offloaded_tasks,
                                  &packet.task_execution_plan_identity)) {
    return false;
  }

  GraphExecutionGateBinding binding;
  binding.gate = reinterpret_cast<std::uintptr_t>(gate);
  binding.expected = expected;
  packet.handle = handle;
  packet.dispatch_label = ctx.dispatch_label();
  packet.arg_buffer_size = ctx.arg_buffer_size;
  packet.arg_buffer_prefix_size = sizeof(binding);
  packet.device_arg_buffer_size =
      packet.arg_buffer_prefix_size + packet.arg_buffer_size;
  packet.context = context;
  CUDADriver::get_instance().malloc_async(
      &packet.device_arg_buffer, packet.device_arg_buffer_size, stream);
  CUDADriver::get_instance().memcpy_host_to_device_async(
      packet.device_arg_buffer, &binding, sizeof(binding), stream);
  auto *device_args = static_cast<char *>(packet.device_arg_buffer) +
                      packet.arg_buffer_prefix_size;
  if (packet.arg_buffer_size > 0) {
    CUDADriver::get_instance().memcpy_host_to_device_async(
        device_args, packet.context.arg_buffer, packet.arg_buffer_size, stream);
  }
  packet.context.arg_buffer = device_args;
  return true;
}

bool KernelLauncher::update_cuda_graph_launch(
    const GraphLaunchPacket &packet,
    LaunchContextBuilder &ctx,
    std::vector<uint8_t> &host_arg_buffer,
    void *stream) {
  if (packet.arg_buffer_size != ctx.arg_buffer_size) {
    return false;
  }
  RuntimeContext context;
  std::size_t resolved_task_count = 0;
  std::string task_plan_identity;
  if (!prepare_cuda_graph_context(packet.handle, ctx, context,
                                  nullptr, &task_plan_identity,
                                  &resolved_task_count)) {
    return false;
  }
  if (task_plan_identity != packet.task_execution_plan_identity ||
      resolved_task_count != packet.offloaded_tasks.size()) {
    return false;
  }
  if (context.runtime != packet.context.runtime ||
      context.result_buffer != packet.context.result_buffer) {
    return false;
  }

  if (packet.arg_buffer_size == 0) {
    if (packet.arg_buffer_prefix_size == 0) {
      if (packet.device_arg_buffer != nullptr ||
          packet.context.arg_buffer != nullptr) {
        return false;
      }
    } else if (packet.device_arg_buffer == nullptr ||
               packet.context.arg_buffer == nullptr ||
               packet.device_arg_buffer_size != packet.arg_buffer_prefix_size) {
      return false;
    }
    host_arg_buffer.clear();
    return true;
  }
  if (packet.device_arg_buffer == nullptr || context.arg_buffer == nullptr) {
    return false;
  }

  host_arg_buffer.resize(packet.arg_buffer_size);
  std::memcpy(host_arg_buffer.data(), context.arg_buffer,
              packet.arg_buffer_size);
  auto *device_args = static_cast<char *>(packet.device_arg_buffer) +
                      packet.arg_buffer_prefix_size;
  CUDADriver::get_instance().memcpy_host_to_device_async(
      device_args, host_arg_buffer.data(), packet.arg_buffer_size, stream);
  return true;
}

bool KernelLauncher::update_cuda_graph_bounded_range(
    GraphLaunchPacket &packet,
    void *extent,
    std::uint32_t capacity,
    std::vector<uint8_t> &host_binding,
    void *stream) {
  if (!packet.bounded_range || extent == nullptr ||
      capacity != packet.bounded_capacity ||
      packet.arg_buffer_prefix_size != sizeof(CudaBoundedRangeBinding) ||
      packet.device_arg_buffer == nullptr) {
    return false;
  }
  const auto extent_address = reinterpret_cast<std::uintptr_t>(extent);
  if (packet.bounded_extent == extent_address) {
    host_binding.clear();
    return true;
  }
  CudaBoundedRangeBinding binding;
  binding.extent = extent_address;
  binding.capacity = static_cast<std::int32_t>(capacity);
  host_binding.resize(sizeof(binding));
  std::memcpy(host_binding.data(), &binding, sizeof(binding));
  CUDADriver::get_instance().memcpy_host_to_device_async(
      packet.device_arg_buffer, host_binding.data(), host_binding.size(),
      stream);
  packet.bounded_extent = extent_address;
  return true;
}

void KernelLauncher::capture_cuda_graph_launch(
    const GraphLaunchPacket &packet,
    void *stream) {
  std::shared_lock<std::shared_mutex> launch_lock(registration_mutex());
  std::shared_ptr<const Context> launcher_ctx;
  auto iter = contexts_.find(packet.handle.get_launch_id());
  TI_ASSERT(iter != contexts_.end());
  launcher_ctx = iter->second;
  auto *cuda_module = launcher_ctx->jit_module;
  auto *cuda_jit_module = dynamic_cast<JITModuleCUDA *>(cuda_module);
  TI_ASSERT(cuda_jit_module != nullptr);
  const auto &offloaded_tasks = packet.offloaded_tasks;
  // A valid compiled kernel can contain no offloaded work (for example an
  // empty kernel or a statically eliminated body). It contributes no CUDA
  // commands; the surrounding stream operations already preserve ordering.
  // Keep the logical dispatch in the Graph, without fabricating a device task
  // or forcing the complete Graph to fall back to ordinary execution.
  for (const auto &task : offloaded_tasks) {
    std::string trace_name;
    std::unique_ptr<ScopedExternalProfilerAnnotation> annotation;
    if (!packet.dispatch_label.empty()) {
      trace_name = make_labeled_task_name(
          task.name, task.task_id, packet.dispatch_label);
      annotation = std::make_unique<ScopedExternalProfilerAnnotation>(
          trace_name);
    }
    TI_TRACE("Capturing kernel {}<<<{}, {}>>>", task.name, task.grid_dim,
             task.block_dim);
    cuda_jit_module->launch_function_with_stream(
        task.native_function, task.name, task.grid_dim, task.block_dim,
        task.dynamic_shared_array_bytes,
        {const_cast<RuntimeContext *>(&packet.context)}, {}, stream,
        trace_name.empty() ? nullptr : &trace_name);
  }
}

bool KernelLauncher::capture_cuda_graph_bounded_launch(
    const GraphLaunchPacket &packet,
    void *stream,
    void **device_node,
    std::uint32_t *driver_error) {
  if (device_node == nullptr || driver_error == nullptr) {
    return false;
  }
  *device_node = nullptr;
  *driver_error = CUDA_SUCCESS;
  std::shared_lock<std::shared_mutex> launch_lock(registration_mutex());
  auto iter = contexts_.find(packet.handle.get_launch_id());
  if (iter == contexts_.end()) {
    return false;
  }
  const auto &launcher_ctx = iter->second;
  auto *cuda_jit_module =
      dynamic_cast<JITModuleCUDA *>(launcher_ctx->jit_module);
  if (cuda_jit_module == nullptr || packet.offloaded_tasks.size() != 1) {
    return false;
  }
  const auto &task = packet.offloaded_tasks.front();
  if (!task.one_to_one || task.grid_dim <= 0 || task.block_dim <= 0 ||
      task.dynamic_shared_array_bytes < 0) {
    return false;
  }
  void *function = task.native_function;
  if (function == nullptr) {
    return false;
  }
  CUDAContext::get_instance().make_current();

  TaichiCudaLaunchAttribute attribute{};
  attribute.id = TAICHI_CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE;
  attribute.value.device_updatable_kernel_node.device_updatable = 1;
  TaichiCudaLaunchConfig config{};
  config.grid_dim_x = static_cast<std::uint32_t>(task.grid_dim);
  config.grid_dim_y = 1;
  config.grid_dim_z = 1;
  config.block_dim_x = static_cast<std::uint32_t>(task.block_dim);
  config.block_dim_y = 1;
  config.block_dim_z = 1;
  config.shared_mem_bytes =
      static_cast<std::uint32_t>(task.dynamic_shared_array_bytes);
  config.stream = stream;
  config.attributes = &attribute;
  config.num_attributes = 1;
  void *kernel_args[] = {
      const_cast<RuntimeContext *>(&packet.context),
  };

  std::string trace_name;
  std::unique_ptr<ScopedExternalProfilerAnnotation> annotation;
  if (!packet.dispatch_label.empty()) {
    trace_name = make_labeled_task_name(task.name, task.task_id,
                                        packet.dispatch_label);
    annotation = std::make_unique<ScopedExternalProfilerAnnotation>(
        trace_name);
  }
  *driver_error = CUDADriver::get_instance().launch_kernel_ex.call(
      &config, function, kernel_args, nullptr);
  if (*driver_error != CUDA_SUCCESS) {
    return false;
  }
  *device_node =
      attribute.value.device_updatable_kernel_node.device_node;
  return *device_node != nullptr;
}

bool KernelLauncher::capture_cuda_graph_updatable_launch(
    const GraphLaunchPacket &packet,
    void *stream,
    std::vector<void *> *device_nodes,
    std::uint32_t *driver_error) {
  if (device_nodes == nullptr || driver_error == nullptr) {
    return false;
  }
  device_nodes->clear();
  *driver_error = CUDA_SUCCESS;
  std::shared_lock<std::shared_mutex> launch_lock(registration_mutex());
  auto iter = contexts_.find(packet.handle.get_launch_id());
  if (iter == contexts_.end()) {
    return false;
  }
  const auto &launcher_ctx = iter->second;
  auto *cuda_jit_module =
      dynamic_cast<JITModuleCUDA *>(launcher_ctx->jit_module);
  if (cuda_jit_module == nullptr || packet.offloaded_tasks.empty()) {
    return false;
  }
  CUDAContext::get_instance().make_current();
  for (const auto &task : packet.offloaded_tasks) {
    if (task.grid_dim <= 0 || task.block_dim <= 0 ||
        task.dynamic_shared_array_bytes < 0 ||
        task.native_function == nullptr) {
      return false;
    }
  }

  device_nodes->reserve(packet.offloaded_tasks.size());
  for (const auto &task : packet.offloaded_tasks) {
    void *function = task.native_function;
    TaichiCudaLaunchAttribute attribute{};
    attribute.id = TAICHI_CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE;
    attribute.value.device_updatable_kernel_node.device_updatable = 1;
    TaichiCudaLaunchConfig config{};
    config.grid_dim_x = static_cast<std::uint32_t>(task.grid_dim);
    config.grid_dim_y = 1;
    config.grid_dim_z = 1;
    config.block_dim_x = static_cast<std::uint32_t>(task.block_dim);
    config.block_dim_y = 1;
    config.block_dim_z = 1;
    config.shared_mem_bytes =
        static_cast<std::uint32_t>(task.dynamic_shared_array_bytes);
    config.stream = stream;
    config.attributes = &attribute;
    config.num_attributes = 1;
    void *kernel_args[] = {
        const_cast<RuntimeContext *>(&packet.context),
    };

    std::string trace_name;
    std::unique_ptr<ScopedExternalProfilerAnnotation> annotation;
    if (!packet.dispatch_label.empty()) {
      trace_name = make_labeled_task_name(task.name, task.task_id,
                                          packet.dispatch_label);
      annotation =
          std::make_unique<ScopedExternalProfilerAnnotation>(trace_name);
    }
    *driver_error = CUDADriver::get_instance().launch_kernel_ex.call(
        &config, function, kernel_args, nullptr);
    if (*driver_error != CUDA_SUCCESS) {
      return false;
    }
    void *device_node =
        attribute.value.device_updatable_kernel_node.device_node;
    if (device_node == nullptr) {
      *driver_error = CUDA_ERROR_NOT_SUPPORTED;
      return false;
    }
    device_nodes->push_back(device_node);
  }
  return true;
}

void KernelLauncher::launch_llvm_kernel(Handle handle,
                                        LaunchContextBuilder &ctx) {
  // Keep argument-buffer preparation, every offloaded task, and stream-ordered
  // cleanup contiguous with respect to CUDA graph replay and other host
  // launchers. The default ndarray/field path only enqueues work while holding
  // this lock; it does not add a device synchronization.
  auto submission_lock =
      CUDAContext::get_instance().get_submission_lock_guard();
  std::shared_lock<std::shared_mutex> launch_lock(registration_mutex());
  std::shared_ptr<const Context> launcher_ctx;
  auto iter = contexts_.find(handle.get_launch_id());
  TI_ASSERT(iter != contexts_.end());
  launcher_ctx = iter->second;
  ensure_root_binding(*launcher_ctx);
  auto *executor = get_runtime_executor();
  const bool listgen_reuse_adaptive =
      executor->get_config().cuda_listgen_reuse_adaptive;
  auto *cuda_module = launcher_ctx->jit_module;
  auto *cuda_jit_module = dynamic_cast<JITModuleCUDA *>(cuda_module);
  TI_ASSERT(cuda_jit_module != nullptr);
  const auto &parameters = launcher_ctx->parameters;

  CUDAContext::get_instance().make_current();
  const Context::ResolvedTaskExecutionPlan *task_execution_plan = nullptr;
  const std::vector<OffloadedTask> *offloaded_tasks_ptr =
      &launcher_ctx->offloaded_tasks;
  if (ctx.has_cuda_task_execution_plan()) {
    TI_ERROR_IF(ctx.cuda_grid_residency_waves() != 0 ||
                    ctx.cuda_range_work_per_thread_target() != 1,
                "task-indexed and call-wide CUDA launch controls cannot be "
                "combined");
    task_execution_plan =
        resolve_task_execution_plan(*launcher_ctx, ctx);
    offloaded_tasks_ptr = &task_execution_plan->tasks;
  } else if (ctx.cuda_grid_residency_waves() != 0 ||
             ctx.cuda_range_work_per_thread_target() != 1) {
    offloaded_tasks_ptr = &resolve_grid_policy_tasks(
        *launcher_ctx, ctx.cuda_grid_residency_waves(),
        ctx.cuda_range_work_per_thread_target());
  }
  const auto &offloaded_tasks = *offloaded_tasks_ptr;

  // |transfers| is only used for external arrays whose data is originally on
  // host. They are first transferred onto device and that device pointer is
  // stored in |device_ptrs| below. |transfers| saves its original pointer so
  // that we can copy the data back once kernel finishes. as well as the
  // temporary device allocations, which can be freed after kernel finishes. Key
  // is [arg_id, ptr_pos], where ptr_pos is TypeFactory::DATA_PTR_POS_IN_NDARRAY
  // for data_ptr and TypeFactory::GRAD_PTR_POS_IN_NDARRAY for grad_ptr. Value
  // is [host_ptr, temporary_device_alloc]. Invariant: temp_devallocs.size() !=
  // 0 <==> transfer happened.
  std::unordered_map<std::vector<int>, std::pair<void *, DeviceAllocation>,
                     hashing::Hasher<std::vector<int>>>
      transfers;

  // |device_ptrs| stores pointers on device for all arrays args, including
  // external arrays and ndarrays, no matter whether the data is originally on
  // device or host.
  // This is the source of truth for us to look for device pointers used in CUDA
  // kernels.
  std::unordered_map<std::vector<int>, void *,
                     hashing::Hasher<std::vector<int>>>
      device_ptrs;

  char *device_result_buffer{nullptr};
  // Most void kernels with only Taichi fields do not need a device result
  // buffer. Allocate lazily for an actual return value or a host-array
  // transfer, both of which use it as the runtime allocation result channel.
  // This preserves the default-stream ordering of the existing allocation and
  // free while avoiding one async allocation/free pair per ordinary launch.
  auto ensure_device_result_buffer = [&] {
    if (device_result_buffer == nullptr) {
      const auto required =
          std::max(ctx.result_buffer_size, sizeof(uint64));
      if (retain_ordinary_launch_buffers_) {
        device_result_buffer = static_cast<char *>(
            launcher_ctx->ordinary_result_buffer.reserve(required));
      } else {
        CUDADriver::get_instance().malloc_async(
            reinterpret_cast<void **>(&device_result_buffer), required,
            nullptr);
      }
    }
    return device_result_buffer;
  };
  ctx.get_context().runtime = executor->get_llvm_runtime();
  for (int i = 0; i < (int)parameters.size(); i++) {
    const auto &kv = parameters[i];
    const auto &key = kv.first;
    const auto &parameter = kv.second;
    if (parameter.is_array) {
      if (ctx.device_allocation_type[key] ==
          LaunchContextBuilder::DevAllocType::kTexture) {
        const auto found = ctx.array_ptrs.find(key);
        TI_ERROR_IF(found == ctx.array_ptrs.end() || found->second == nullptr,
                    "CUDA texture argument resolved to a null resource");
        const auto texture_object =
            *static_cast<const std::uint64_t *>(found->second);
        TI_ERROR_IF(texture_object == 0,
                    "CUDA texture argument resolved to a null texture object");
        auto texture_object_key = key;
        texture_object_key.push_back(TypeFactory::DATA_PTR_POS_IN_NDARRAY);
        ctx.set_struct_arg<std::uint64_t>(texture_object_key, texture_object);
        continue;
      }
      const auto arr_sz = ctx.array_runtime_sizes[key];
      // Note: both numpy and PyTorch support arrays/tensors with zeros
      // in shapes, e.g., shape=(0) or shape=(100, 0, 200). This makes
      // `arr_sz` zero.
      if (arr_sz == 0) {
        continue;
      }

      std::vector<int> data_ptr_idx = key;
      data_ptr_idx.push_back(TypeFactory::DATA_PTR_POS_IN_NDARRAY);
      auto data_ptr = ctx.array_ptrs[data_ptr_idx];
      std::vector<int> grad_ptr_idx = key;
      grad_ptr_idx.push_back(TypeFactory::GRAD_PTR_POS_IN_NDARRAY);

      auto grad_ptr = ctx.array_ptrs[grad_ptr_idx];
      if (ctx.device_allocation_type[key] ==
          LaunchContextBuilder::DevAllocType::kNone) {
        // External array
        // Note: assuming both data & grad are on the same device
        if (on_cuda_device(data_ptr)) {
          // data_ptr is a raw ptr on CUDA device
          device_ptrs[data_ptr_idx] = data_ptr;
          device_ptrs[grad_ptr_idx] = grad_ptr;
        } else {
          DeviceAllocation devalloc =
              executor->allocate_memory_on_device(
                  arr_sz, (uint64 *)ensure_device_result_buffer());
          device_ptrs[data_ptr_idx] =
              executor->get_device_alloc_info_ptr(devalloc);
          transfers[data_ptr_idx] = {data_ptr, devalloc};

          CUDADriver::get_instance().memcpy_host_to_device(
              (void *)device_ptrs[data_ptr_idx], data_ptr, arr_sz);
          if (grad_ptr != nullptr) {
            DeviceAllocation grad_devalloc =
                executor->allocate_memory_on_device(
                    arr_sz, (uint64 *)ensure_device_result_buffer());
            device_ptrs[grad_ptr_idx] =
                executor->get_device_alloc_info_ptr(grad_devalloc);
            transfers[grad_ptr_idx] = {grad_ptr, grad_devalloc};

            CUDADriver::get_instance().memcpy_host_to_device(
                (void *)device_ptrs[grad_ptr_idx], grad_ptr, arr_sz);
          } else {
            device_ptrs[grad_ptr_idx] = nullptr;
          }
        }

        ctx.set_ndarray_ptrs(key, (uint64)device_ptrs[data_ptr_idx],
                             (uint64)device_ptrs[grad_ptr_idx]);
      } else if (ctx.device_allocation_type[key] ==
                 LaunchContextBuilder::DevAllocType::kDenseStorage) {
        const auto &binding = ctx.get_resolved_dense_storage(key);
        auto *base = reinterpret_cast<char *>(
            executor->get_device_alloc_info_ptr(binding.allocation));
        device_ptrs[data_ptr_idx] = base + binding.byte_offset;
        device_ptrs[grad_ptr_idx] = nullptr;
        ctx.set_ndarray_ptrs(key, (uint64)device_ptrs[data_ptr_idx], 0);
      } else if (arr_sz > 0) {
        // Ndarray
        DeviceAllocation *ptr = static_cast<DeviceAllocation *>(data_ptr);
        // Unwrapped raw ptr on device
        device_ptrs[data_ptr_idx] = executor->get_device_alloc_info_ptr(*ptr);

        if (grad_ptr != nullptr) {
          ptr = static_cast<DeviceAllocation *>(grad_ptr);
          device_ptrs[grad_ptr_idx] = executor->get_device_alloc_info_ptr(*ptr);
        } else {
          device_ptrs[grad_ptr_idx] = nullptr;
        }

        ctx.set_ndarray_ptrs(key, (uint64)device_ptrs[data_ptr_idx],
                             (uint64)device_ptrs[grad_ptr_idx]);
      }
    } else if (parameter.is_argpack) {
      std::vector<int> data_ptr_idx = key;
      data_ptr_idx.push_back(TypeFactory::DATA_PTR_POS_IN_ARGPACK);
      auto *argpack = ctx.argpack_ptrs[key];
      auto argpack_ptr = argpack->get_device_allocation();
      device_ptrs[data_ptr_idx] =
          executor->get_device_alloc_info_ptr(argpack_ptr);
      if (key.size() == 1) {
        ctx.set_argpack_ptr(key, (uint64)device_ptrs[data_ptr_idx]);
      } else {
        auto key_parent = key;
        key_parent.pop_back();
        auto *argpack_parent = ctx.argpack_ptrs[key_parent];
        argpack_parent->set_arg_nested_argpack_ptr(
            key.back(), (uint64)device_ptrs[data_ptr_idx]);
      }
    }
  }
  if (transfers.size() > 0) {
    CUDADriver::get_instance().stream_synchronize(nullptr);
  }
  char *host_arg_buffer = ctx.get_context().arg_buffer;
  char *host_result_buffer = (char *)ctx.get_context().result_buffer;
  if (ctx.result_buffer_size > 0) {
    ctx.get_context().result_buffer =
        (uint64 *)ensure_device_result_buffer();
  }
  if (launcher_ctx->uses_root_binding) {
    TI_ASSERT(ctx.result_buffer_size == 0);
    TI_ASSERT(launcher_ctx->root_binding != nullptr);
    ctx.get_context().result_buffer =
        static_cast<uint64_t *>(launcher_ctx->root_binding);
  }
  char *device_arg_buffer = nullptr;
  const std::size_t arg_buffer_prefix_size =
      ctx.has_cuda_bounded_range() ? sizeof(CudaBoundedRangeBinding) : 0;
  const std::size_t device_arg_buffer_size =
      arg_buffer_prefix_size + ctx.arg_buffer_size;
  if (device_arg_buffer_size > 0) {
    if (retain_ordinary_launch_buffers_) {
      std::size_t slot = 0;
      if (ordinary_launch_arg_ring_size_ > 1) {
        slot = launcher_ctx->ordinary_arg_buffer_cursor++ %
               ordinary_launch_arg_ring_size_;
      }
      device_arg_buffer = static_cast<char *>(
          launcher_ctx->ordinary_arg_buffers[slot].reserve(
              device_arg_buffer_size));
    } else {
      CUDADriver::get_instance().malloc_async(
          reinterpret_cast<void **>(&device_arg_buffer),
          device_arg_buffer_size, nullptr);
    }
    if (ctx.has_cuda_bounded_range()) {
      auto extent_ptr_idx = ctx.cuda_bounded_extent_arg_id();
      extent_ptr_idx.push_back(TypeFactory::DATA_PTR_POS_IN_NDARRAY);
      const auto extent_iter = device_ptrs.find(extent_ptr_idx);
      TI_ERROR_IF(extent_iter == device_ptrs.end() ||
                      extent_iter->second == nullptr,
                  "CUDA exact bounded DeviceExtent resolved to a null "
                  "address");
      CudaBoundedRangeBinding binding{
          reinterpret_cast<std::uintptr_t>(extent_iter->second),
          ctx.cuda_bounded_capacity(), 0};
      CUDADriver::get_instance().memcpy_host_to_device_async(
          device_arg_buffer, &binding, sizeof(binding), nullptr);
    }
    auto *device_args = device_arg_buffer + arg_buffer_prefix_size;
    if (ctx.arg_buffer_size > 0) {
      CUDADriver::get_instance().memcpy_host_to_device_async(
          device_args, host_arg_buffer, ctx.arg_buffer_size, nullptr);
    }
    ctx.get_context().arg_buffer = device_args;
  }

  for (const auto &task : offloaded_tasks) {
    const bool uses_sparse_state =
        task.sparse_list_op != OffloadedTask::kSparseListOpNone ||
        task.may_mutate_sparse_topology;
    std::unique_lock<std::mutex> sparse_lock(sparse_list_mutex_,
                                             std::defer_lock);
    if (uses_sparse_state) {
      sparse_lock.lock();
      listgen_reuse_adaptive_ = listgen_reuse_adaptive;
    }
    if (sparse_list_task_is_current(task)) {
      TI_TRACE("Skipping current sparse list kernel {}", task.name);
      continue;
    }
    TI_TRACE("Launching kernel {}<<<{}, {}>>>", task.name, task.grid_dim,
             task.block_dim);
    if (ctx.dispatch_label().empty()) {
      cuda_jit_module->launch_function_with_stream(
          task.native_function, task.name, task.grid_dim, task.block_dim,
          task.dynamic_shared_array_bytes, {&ctx.get_context()}, {}, nullptr);
    } else {
      const auto trace_name = make_labeled_task_name(
          task.name, task.task_id, ctx.dispatch_label());
      ScopedExternalProfilerAnnotation annotation(trace_name);
      cuda_jit_module->launch_function_with_stream(
          task.native_function, task.name, task.grid_dim, task.block_dim,
          task.dynamic_shared_array_bytes, {&ctx.get_context()}, {}, nullptr,
          &trace_name);
    }
    mark_sparse_list_task_launched(task);
    if (task.may_mutate_sparse_topology) {
      invalidate_sparse_list_cache(task.sparse_mutation_snode_id);
    }
  }
  if (!retain_ordinary_launch_buffers_ && device_arg_buffer != nullptr) {
    CUDADriver::get_instance().mem_free_async(device_arg_buffer, nullptr);
  }
  if (ctx.result_buffer_size > 0) {
    CUDADriver::get_instance().memcpy_device_to_host_async(
        host_result_buffer, device_result_buffer, ctx.result_buffer_size,
        nullptr);
  }
  if (!retain_ordinary_launch_buffers_ && device_result_buffer != nullptr) {
    CUDADriver::get_instance().mem_free_async(device_result_buffer, nullptr);
  }
  ctx.get_context().arg_buffer = host_arg_buffer;
  ctx.get_context().result_buffer = (uint64 *)host_result_buffer;
  // copy data back to host
  if (transfers.size() > 0) {
    CUDADriver::get_instance().stream_synchronize(nullptr);
    for (auto itr = transfers.begin(); itr != transfers.end(); itr++) {
      auto &idx = itr->first;
      auto arg_id = idx;
      arg_id.pop_back();
      CUDADriver::get_instance().memcpy_device_to_host(
          itr->second.first, (void *)device_ptrs[idx],
          ctx.array_runtime_sizes[arg_id]);
      executor->deallocate_memory_on_device(itr->second.second);
    }
  }
}

KernelLauncher::Handle KernelLauncher::register_llvm_kernel(
    const LLVM::CompiledKernelData &compiled) {
  TI_ASSERT(compiled.arch() == Arch::cuda);
  std::unique_lock<std::shared_mutex> lock(registration_mutex());

  if (!compiled.get_handle()) {
    auto handle = make_handle();
    auto index = handle.get_launch_id();

    auto ctx = std::make_shared<Context>();
    auto *executor = get_runtime_executor();

    auto data = compiled.get_internal_data().compiled_data.clone();
    auto parameters = compiled.get_internal_data().args;
    auto *jit_module = executor->create_jit_module(std::move(data.module),
                                                   data.cuda_max_registers);
    try {
      bind_cuda_native_functions(jit_module, data.tasks);
    } catch (...) {
      executor->remove_jit_module(jit_module);
      throw;
    }

    // Populate ctx
    ctx->jit_module = jit_module;
    ctx->snode_tree_ids = compiled.snode_tree_ids();
    ctx->parameters = std::move(parameters);
    ctx->offloaded_tasks = std::move(data.tasks);
    configure_root_binding(compiled, *ctx);
    const bool was_inserted = contexts_.emplace(index, std::move(ctx)).second;
    TI_ASSERT(was_inserted);

    compiled.set_handle(handle);
  }
  return *compiled.get_handle();
}

KernelLauncher::Handle KernelLauncher::register_llvm_kernel_graph_gated(
    const LLVM::CompiledKernelData &compiled) {
  TI_ASSERT(compiled.arch() == Arch::cuda);
  std::unique_lock<std::shared_mutex> lock(registration_mutex());
  if (compiled.get_graph_masked_handle()) {
    return *compiled.get_graph_masked_handle();
  }

  auto handle = make_handle();
  auto ctx = std::make_shared<Context>();
  auto *executor = get_runtime_executor();
  auto data = compiled.get_internal_data().compiled_data.clone();
  add_cuda_graph_execution_gate(data);
  auto parameters = compiled.get_internal_data().args;
  auto *jit_module = executor->create_jit_module(std::move(data.module),
                                                 data.cuda_max_registers);
  try {
    bind_cuda_native_functions(jit_module, data.tasks);
  } catch (...) {
    executor->remove_jit_module(jit_module);
    throw;
  }

  ctx->jit_module = jit_module;
  ctx->snode_tree_ids = compiled.snode_tree_ids();
  ctx->parameters = std::move(parameters);
  ctx->offloaded_tasks = std::move(data.tasks);
  configure_root_binding(compiled, *ctx);
  const bool inserted =
      contexts_.emplace(handle.get_launch_id(), std::move(ctx)).second;
  TI_ASSERT(inserted);
  compiled.set_graph_masked_handle(handle);
  return handle;
}

void KernelLauncher::retire_snode_tree(int tree_id) {
  auto submission_lock =
      CUDAContext::get_instance().get_submission_lock_guard();
  std::unique_lock<std::shared_mutex> registration_lock(registration_mutex());
  auto *executor = get_runtime_executor();
  bool retired_any = false;
  std::vector<int> retired_sparse_snode_ids;
  for (auto iter = contexts_.begin(); iter != contexts_.end();) {
    const auto &context = iter->second;
    if (!std::binary_search(context->snode_tree_ids.begin(),
                            context->snode_tree_ids.end(), tree_id)) {
      ++iter;
      continue;
    }
    for (const auto &task : context->offloaded_tasks) {
      if (task.sparse_list_snode_id >= 0) {
        retired_sparse_snode_ids.push_back(task.sparse_list_snode_id);
      }
    }
    for (const auto &buffer : context->ordinary_arg_buffers) {
      buffer.release();
    }
    context->ordinary_result_buffer.release();
    auto module = context->jit_module;
    iter = contexts_.erase(iter);
    executor->remove_jit_module(module);
    retired_any = true;
  }
  if (retired_any) {
    std::lock_guard<std::mutex> sparse_lock(sparse_list_mutex_);
    sparse_list_states_.clear();
    for (int snode_id : retired_sparse_snode_ids) {
      sparse_listgen_telemetry_.erase(snode_id);
    }
  }
}

std::size_t KernelLauncher::debug_registered_kernel_count() {
  std::shared_lock<std::shared_mutex> lock(registration_mutex());
  return contexts_.size();
}

std::vector<KernelLauncher::ArtifactQualification>
KernelLauncher::qualify_llvm_kernel_artifacts(
    const LLVM::CompiledKernelData &compiled) {
  TI_ASSERT(compiled.arch() == Arch::cuda);
  auto &telemetry = artifact_qualification_telemetry();
  telemetry.qualification_calls.fetch_add(1, std::memory_order_relaxed);

  const bool was_registered = compiled.get_handle().has_value();
  const Handle handle = register_llvm_kernel(compiled);
  if (!was_registered) {
    telemetry.registration_materializations.fetch_add(
        1, std::memory_order_relaxed);
  }

  std::shared_lock<std::shared_mutex> lock(registration_mutex());
  const auto iter = contexts_.find(handle.get_launch_id());
  TI_ERROR_IF(iter == contexts_.end(),
              "CUDA artifact qualification lost its registered context");
  const auto &context = *iter->second;
  auto *cuda_module = dynamic_cast<JITModuleCUDA *>(context.jit_module);
  TI_ERROR_IF(cuda_module == nullptr,
              "CUDA artifact qualification requires a CUDA JIT module");

  CUDAContext::get_instance().make_current();
  int multiprocessor_count = 0;
  CUDADriver::get_instance().device_get_attribute(
      &multiprocessor_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
      CUDAContext::get_instance().get_device());

  std::vector<ArtifactQualification> result;
  result.reserve(context.offloaded_tasks.size());
  for (const auto &task : context.offloaded_tasks) {
    ArtifactQualification item;
    item.entry_point = task.name;
    void *function = task.native_function;
    TI_ASSERT(function != nullptr);
    item.function_identity = reinterpret_cast<std::uintptr_t>(function);
    const auto query_attribute = [&](int attribute, int *value) {
      CUDADriver::get_instance().kernel_get_attribute(value, attribute,
                                                       function);
      telemetry.function_attribute_queries.fetch_add(
          1, std::memory_order_relaxed);
    };
    query_attribute(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                    &item.max_threads_per_block);
    query_attribute(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                    &item.static_shared_memory_bytes);
    query_attribute(CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
                    &item.constant_memory_bytes);
    query_attribute(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                    &item.local_memory_bytes_per_thread);
    query_attribute(CU_FUNC_ATTRIBUTE_NUM_REGS, &item.registers_per_thread);
    query_attribute(CU_FUNC_ATTRIBUTE_PTX_VERSION, &item.ptx_version);
    query_attribute(CU_FUNC_ATTRIBUTE_BINARY_VERSION, &item.binary_version);
    query_attribute(CU_FUNC_ATTRIBUTE_CACHE_MODE_CA, &item.cache_mode_ca);
    query_attribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    &item.max_dynamic_shared_bytes);
    query_attribute(CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                    &item.preferred_shared_carveout);
    item.block_dim = task.block_dim;
    item.dynamic_shared_bytes = task.dynamic_shared_array_bytes;
    item.multiprocessor_count = multiprocessor_count;
    if (item.block_dim > 0 && item.dynamic_shared_bytes >= 0) {
      CUDADriver::get_instance().kernel_get_occupancy(
          &item.active_blocks_per_multiprocessor, function, item.block_dim,
          static_cast<std::size_t>(item.dynamic_shared_bytes));
      telemetry.occupancy_queries.fetch_add(1, std::memory_order_relaxed);
    }
    result.push_back(std::move(item));
  }
  return result;
}

void KernelLauncher::debug_reset_sparse_listgen_statistics() {
  std::lock_guard<std::mutex> lock(sparse_list_mutex_);
  sparse_listgen_telemetry_.clear();
  sparse_listgen_telemetry_enabled_ = true;
}

SparseSNodeTreeListgenStatistics
KernelLauncher::debug_sparse_listgen_statistics(
    const std::vector<int> &snode_ids) {
  std::lock_guard<std::mutex> lock(sparse_list_mutex_);
  SparseSNodeTreeListgenStatistics result;
  result.available = sparse_listgen_telemetry_enabled_;
  if (!result.available) {
    return result;
  }
  for (const auto &[snode_id, telemetry] : sparse_listgen_telemetry_) {
    if (std::binary_search(snode_ids.begin(), snode_ids.end(), snode_id)) {
      result.nodes.push_back(telemetry);
    }
  }
  std::sort(result.nodes.begin(), result.nodes.end(),
            [](const auto &lhs, const auto &rhs) {
              return lhs.snode_id < rhs.snode_id;
            });
  return result;
}

}  // namespace cuda
}  // namespace taichi::lang
