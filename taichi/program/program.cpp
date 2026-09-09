// Program, context for Taichi program execution

#include "program.h"

#include "taichi/ir/statements.h"
#include "taichi/program/extension.h"
#include "taichi/codegen/cpu/codegen_cpu.h"
#include "taichi/struct/struct.h"
#include "taichi/runtime/program_impls/opengl/opengl_program.h"
#include "taichi/runtime/program_impls/metal/metal_program.h"
#include "taichi/platform/cuda/detect_cuda.h"
#include "taichi/system/timeline.h"
#include "taichi/system/threading.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/frontend_ir.h"
#include "taichi/program/snode_expr_utils.h"
#include "taichi/math/arithmetic.h"
#include "taichi/rhi/common/host_memory_pool.h"
#include "taichi/rhi/interop/external_sync.h"
#include "taichi/rhi/interop/external_access_epoch.h"
#include "taichi/util/environ_config.h"
#ifdef TI_WITH_LLVM
#include "taichi/rhi/cpu/cpu_device.h"
#endif
#include "taichi/program/parallel_executor.h"

#include <chrono>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>

#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/rhi/cuda/cuda_primitives.h"
#include "taichi/rhi/cuda/primitives/hierarchical_ptx.h"
#endif

#ifdef TI_WITH_LLVM
#include "taichi/rhi/llvm/device_memory_pool.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#include "taichi/codegen/llvm/struct_llvm.h"
#endif

#ifdef TI_WITH_VULKAN
#include "taichi/program/vulkan_command_replay.h"
#include "taichi/runtime/program_impls/vulkan/vulkan_program.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#endif
#ifdef TI_WITH_OPENGL
#include "taichi/runtime/program_impls/opengl/opengl_program.h"
#include "taichi/rhi/opengl/opengl_api.h"
#endif
#ifdef TI_WITH_DX11
#include "taichi/runtime/program_impls/dx/dx_program.h"
#include "taichi/rhi/dx/dx_api.h"
#endif
#ifdef TI_WITH_DX12
#include "taichi/runtime/program_impls/dx12/dx12_program.h"
#include "taichi/rhi/dx12/dx12_api.h"
#endif
#ifdef TI_WITH_METAL
#include "taichi/runtime/program_impls/metal/metal_program.h"
#include "taichi/rhi/metal/metal_api.h"
#endif  // TI_WITH_METAL

#if defined(_M_X64) || defined(__x86_64)
// For _MM_SET_FLUSH_ZERO_MODE
#include <xmmintrin.h>
#endif  // defined(_M_X64) || defined(__x86_64)

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <vector>

namespace taichi::lang {

class ProgramLifetimeToken {
 public:
  explicit ProgramLifetimeToken(Program *program) : program_(program) {
  }

 private:
  friend class Program;
  std::mutex mutex_;
  Program *program_{nullptr};
};

Program *&runtime_completion_detail::active_runtime_submission_program()
    noexcept {
  static thread_local Program *active_program = nullptr;
  return active_program;
}

namespace {

thread_local Program *active_snode_tree_lifecycle_program = nullptr;
thread_local Program *active_runtime_resource_graph_program = nullptr;
thread_local Program::RuntimeResourceGraphScope *active_runtime_resource_graph_scope = nullptr;

std::uint64_t ordinary_launch_now_ns() noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

class RuntimeProgramSyncStatisticsScope {
 public:
  explicit RuntimeProgramSyncStatisticsScope(Program *program)
      : statistics_(program->runtime_statistics()),
        trace_(&program->runtime_trace(),
               RuntimeTraceEventKind::kProgramSynchronize),
        started_(std::chrono::steady_clock::now()),
        uncaught_exceptions_(std::uncaught_exceptions()) {
  }

  ~RuntimeProgramSyncStatisticsScope() {
    if (std::uncaught_exceptions() != uncaught_exceptions_) {
      trace_.mark_failed();
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started_);
    statistics_.record_program_sync(
        static_cast<std::uint64_t>(elapsed.count()));
  }

 private:
  RuntimeStatistics &statistics_;
  RuntimeTraceRecorder::Scope trace_;
  std::chrono::steady_clock::time_point started_;
  int uncaught_exceptions_;
};

std::atomic<std::uint64_t> next_runtime_resource_domain{1};

std::uint64_t allocate_runtime_resource_domain() {
  std::uint64_t domain =
      next_runtime_resource_domain.load(std::memory_order_relaxed);
  for (;;) {
    TI_ASSERT(domain != 0 &&
              domain != (std::numeric_limits<std::uint64_t>::max)());
    if (next_runtime_resource_domain.compare_exchange_weak(
            domain, domain + 1, std::memory_order_relaxed,
            std::memory_order_relaxed)) {
      return domain;
    }
  }
}

std::uint64_t saturating_counter_delta(std::uint64_t current,
                                       std::uint64_t baseline) noexcept {
  return current >= baseline ? current - baseline : 0;
}

std::uint64_t saturating_counter_add(std::uint64_t lhs,
                                     std::uint64_t rhs) noexcept {
  const auto maximum = (std::numeric_limits<std::uint64_t>::max)();
  return lhs > maximum - rhs ? maximum : lhs + rhs;
}

void advance_snode_tree_epoch(std::atomic<std::uint64_t> &epoch) {
  const std::uint64_t current = epoch.load(std::memory_order_relaxed);
  TI_ASSERT(current != std::numeric_limits<std::uint64_t>::max());
  epoch.store(current + 1, std::memory_order_release);
}

}  // namespace

Program::SNodeTreeLifecycleReadGuard::SNodeTreeLifecycleReadGuard(
    Program *program)
    : program_(program),
      previous_program_(active_snode_tree_lifecycle_program),
      lock_(program->snode_tree_lifecycle_mutex_),
      epoch_(program->snode_tree_mutation_epoch()) {
  active_snode_tree_lifecycle_program = program_;
}

Program::SNodeTreeLifecycleReadGuard::SNodeTreeLifecycleReadGuard(
    SNodeTreeLifecycleReadGuard &&other) noexcept
    : program_(std::exchange(other.program_, nullptr)),
      previous_program_(std::exchange(other.previous_program_, nullptr)),
      lock_(std::move(other.lock_)),
      epoch_(other.epoch_) {
}

Program::SNodeTreeLifecycleReadGuard::~SNodeTreeLifecycleReadGuard() {
  if (program_ == nullptr) {
    return;
  }
  TI_ASSERT(active_snode_tree_lifecycle_program == program_);
  active_snode_tree_lifecycle_program = previous_program_;
}

Program::SNodeTreeLifecycleReadGuard
Program::acquire_snode_tree_lifecycle_read_guard() {
  return SNodeTreeLifecycleReadGuard(this);
}

Program::RuntimeResourceGraphScope::RuntimeResourceGraphScope(Program *program)
    : program_(program),
      previous_program_(active_runtime_resource_graph_program),
      lock_(program->runtime_resource_submission_mutex_, std::defer_lock),
      previous_scope_(active_runtime_resource_graph_scope) {
  program_->ensure_runtime_submission_allowed("Graph resource submission");
  lock_.lock();
  TI_ASSERT(previous_program_ == nullptr || previous_program_ == program_);
  active_runtime_resource_graph_program = program_;
  active_runtime_resource_graph_scope = this;
}

Program::RuntimeResourceGraphScope::RuntimeResourceGraphScope(
    RuntimeResourceGraphScope &&other) noexcept
    : program_(std::exchange(other.program_, nullptr)),
      previous_program_(std::exchange(other.previous_program_, nullptr)),
      lock_(std::move(other.lock_)),
      previous_scope_(std::exchange(other.previous_scope_, nullptr)),
      external_access_epoch_(std::move(other.external_access_epoch_)) {
  if (active_runtime_resource_graph_scope == &other) {
    active_runtime_resource_graph_scope = this;
  }
}

Program::RuntimeResourceGraphScope::~RuntimeResourceGraphScope() {
  if (program_ == nullptr) {
    return;
  }
  try {
    finish_external_access_epoch();
  } catch (...) {
    TI_WARN("External Graph access epoch release failed during unwinding");
  }
  TI_ASSERT(active_runtime_resource_graph_program == program_);
  TI_ASSERT(active_runtime_resource_graph_scope == this);
  active_runtime_resource_graph_scope = previous_scope_;
  active_runtime_resource_graph_program = previous_program_;
}

void Program::RuntimeResourceGraphScope::finish_external_access_epoch() {
  if (external_access_epoch_) {
    external_access_epoch_->release();
    external_access_epoch_.reset();
  }
}

Program::RuntimeSubmissionWriteScope::RuntimeSubmissionWriteScope(
    Program *program)
    : program_(program) {
  program_->acquire_runtime_submission_writer();
}

Program::RuntimeSubmissionWriteScope::~RuntimeSubmissionWriteScope() {
  program_->release_runtime_submission_writer();
}

Program::RuntimeSubmissionTransaction::RuntimeSubmissionTransaction(
    Program *program,
    bool gpu_timing,
    bool cuda_concurrent_batch)
    : program_(program),
      gpu_timing_requested_(gpu_timing),
      cuda_concurrent_batch_(cuda_concurrent_batch) {
  TI_ASSERT(program_ != nullptr);
  // The outer scope must observe tracking enabled so nested kernel/CGraph
  // launches reuse this reader instead of opening segment-local boundaries.
  program_->runtime_completion_tracking_enabled_.store(
      true, std::memory_order_release);
  submission_scope_.emplace(program_->acquire_runtime_submission_scope());
  program_->program_impl_->begin_runtime_submission_batch();
  submission_batch_open_ = true;
  try {
    if (gpu_timing) {
      if (program_->compile_config().arch == Arch::cuda) {
        gpu_timing_ = RuntimeCompletion::begin_cuda_gpu_timing(
            nullptr, program_->runtime_fault_domain_);
      } else {
        gpu_timing_ = program_->program_impl_->begin_runtime_gpu_timing();
      }
    }
    if (cuda_concurrent_batch_) {
#if defined(TI_WITH_CUDA)
      TI_ERROR_IF(program_->compile_config().arch != Arch::cuda,
                  "Concurrent Graph submission batches require CUDA");
      CUDAContext::get_instance().make_current();
      auto &driver = CUDADriver::get_instance();
      cuda_concurrent_fork_event_ =
          program_->runtime_completion_cuda_event_pool_->acquire();
      const auto error =
          driver.event_record.call(cuda_concurrent_fork_event_, nullptr);
      TI_ERROR_IF(error != CUDA_SUCCESS, "CUDA batch fork event record failed: {}",
                  driver.event_record.get_error_message(error));
#else
      TI_ERROR("Concurrent Graph submission batches require a CUDA build");
#endif
    }
    if (gpu_timing || cuda_concurrent_batch_) {
      previous_telemetry_transaction_ =
          Program::active_runtime_submission_telemetry_transaction();
      Program::active_runtime_submission_telemetry_transaction() = this;
    }
  } catch (...) {
#if defined(TI_WITH_CUDA)
    if (cuda_concurrent_fork_event_ != nullptr) {
      program_->runtime_completion_cuda_event_pool_->release(
          cuda_concurrent_fork_event_, /*reusable=*/false);
      cuda_concurrent_fork_event_ = nullptr;
    }
#endif
    try {
      program_->program_impl_->end_runtime_submission_batch();
    } catch (...) {
    }
    submission_batch_open_ = false;
    throw;
  }
}

Program::RuntimeSubmissionTransaction::~RuntimeSubmissionTransaction() {
  if ((gpu_timing_requested_ || cuda_concurrent_batch_) &&
      Program::active_runtime_submission_telemetry_transaction() == this) {
    Program::active_runtime_submission_telemetry_transaction() =
        previous_telemetry_transaction_;
  }
  if (program_ != nullptr && submission_batch_open_) {
    // Preserve already-recorded work on exception paths. A backend failure is
    // reported by the next explicit completion/synchronize boundary.
    try {
      join_cuda_concurrent_streams();
    } catch (...) {
    }
    try {
      program_->program_impl_->end_runtime_submission_batch();
    } catch (...) {
    }
    submission_batch_open_ = false;
  }
  // Exception paths may have submitted an earlier segment. Closing the reader
  // is sufficient: legacy synchronize/next completion retains responsibility
  // for any resources already pinned by that segment.
  submission_scope_.reset();
}

void *Program::RuntimeSubmissionTransaction::register_cuda_concurrent_stream(
    void *stream) {
  TI_ERROR_IF(finished_, "Runtime submission transaction already finished");
  TI_ERROR_IF(!cuda_concurrent_batch_,
              "Runtime submission transaction is not a concurrent CUDA batch");
  TI_ERROR_IF(stream == nullptr,
              "Concurrent CUDA Graph replay requires a non-default stream");
#if defined(TI_WITH_CUDA)
  if (std::find(cuda_concurrent_streams_.begin(),
                cuda_concurrent_streams_.end(), stream) !=
      cuda_concurrent_streams_.end()) {
    return stream;
  }
  auto &driver = CUDADriver::get_instance();
  const auto error = driver.stream_wait_event.call(
      stream, cuda_concurrent_fork_event_, 0);
  TI_ERROR_IF(error != CUDA_SUCCESS, "CUDA batch fork wait failed: {}",
              driver.stream_wait_event.get_error_message(error));
  cuda_concurrent_streams_.push_back(stream);
  return stream;
#else
  TI_ERROR("Concurrent Graph submission batches require a CUDA build");
#endif
}

void Program::RuntimeSubmissionTransaction::join_cuda_concurrent_streams() {
  if (!cuda_concurrent_batch_ || cuda_concurrent_fork_event_ == nullptr) {
    return;
  }
#if defined(TI_WITH_CUDA)
  CUDAContext::get_instance().make_current();
  auto &driver = CUDADriver::get_instance();
  std::uint32_t first_error = CUDA_SUCCESS;
  std::vector<void *> tail_events;
  tail_events.reserve(cuda_concurrent_streams_.size());
  try {
    for (void *stream : cuda_concurrent_streams_) {
      void *event =
          program_->runtime_completion_cuda_event_pool_->acquire();
      tail_events.push_back(event);
      auto error = driver.event_record.call(event, stream);
      if (error == CUDA_SUCCESS) {
        error = driver.stream_wait_event.call(nullptr, event, 0);
      }
      if (error != CUDA_SUCCESS && first_error == CUDA_SUCCESS) {
        first_error = error;
      }
    }
  } catch (...) {
    for (void *stream : cuda_concurrent_streams_) {
      driver.stream_synchronize.call(stream);
    }
    driver.stream_synchronize.call(nullptr);
    for (void *event : tail_events) {
      program_->runtime_completion_cuda_event_pool_->release(
          event, /*reusable=*/true);
    }
    if (cuda_concurrent_fork_event_ != nullptr) {
      program_->runtime_completion_cuda_event_pool_->release(
          cuda_concurrent_fork_event_, /*reusable=*/true);
      cuda_concurrent_fork_event_ = nullptr;
    }
    cuda_concurrent_streams_.clear();
    throw;
  }
  if (first_error != CUDA_SUCCESS) {
    // This is an exception-only recovery path. Preserve already-enqueued work
    // before releasing events; production batches use only event ordering.
    for (void *stream : cuda_concurrent_streams_) {
      driver.stream_synchronize.call(stream);
    }
    driver.stream_synchronize.call(nullptr);
    for (void *event : tail_events) {
      program_->runtime_completion_cuda_event_pool_->release(
          event, /*reusable=*/true);
    }
    if (cuda_concurrent_fork_event_ != nullptr) {
      program_->runtime_completion_cuda_event_pool_->release(
          cuda_concurrent_fork_event_, /*reusable=*/true);
      cuda_concurrent_fork_event_ = nullptr;
    }
  } else {
    tail_events.push_back(cuda_concurrent_fork_event_);
    cuda_concurrent_fork_event_ = nullptr;
    program_->pin_runtime_cuda_batch_events(std::move(tail_events));
  }
  cuda_concurrent_streams_.clear();
  TI_ERROR_IF(first_error != CUDA_SUCCESS, "CUDA batch join failed: {}",
              driver.stream_wait_event.get_error_message(first_error));
#endif
}

void Program::RuntimeSubmissionTransaction::mark_submission() noexcept {
  if (!finished_) {
    // Native calls now record their own operation count at the actual Program
    // method boundary. This transaction only publishes that the native
    // portion may have left backend work for the completion ticket.
    program_->mark_runtime_submission_pending();
  }
}

void Program::RuntimeSubmissionTransaction::begin_gpu_region_timing(
    const std::string &path_id) {
  TI_ERROR_IF(finished_, "Runtime submission transaction already finished");
  TI_ERROR_IF(!gpu_timing_requested_,
              "GPU region timing requires an instrumented transaction");
  TI_ERROR_IF(path_id.empty(), "GPU region timing path must not be empty");
  StreamGpuTiming timing;
  if (program_->compile_config().arch == Arch::cuda) {
    timing = RuntimeCompletion::begin_cuda_gpu_timing(
        nullptr, program_->runtime_fault_domain_);
  } else {
    timing = program_->program_impl_->begin_runtime_gpu_timing();
  }
  gpu_region_timings_.push_back({path_id, std::move(timing)});
  active_gpu_region_timings_.push_back(gpu_region_timings_.size() - 1);
}

void Program::RuntimeSubmissionTransaction::end_gpu_region_timing(
    const std::string &path_id) {
  TI_ERROR_IF(finished_, "Runtime submission transaction already finished");
  TI_ERROR_IF(active_gpu_region_timings_.empty(),
              "GPU region timing ended without a matching begin");
  const std::size_t index = active_gpu_region_timings_.back();
  auto &region = gpu_region_timings_[index];
  TI_ERROR_IF(region.path_id != path_id,
              "GPU region timing must end in nested stack order");
  if (region.timing) {
    if (program_->compile_config().arch == Arch::cuda) {
      RuntimeCompletion::end_cuda_gpu_timing(
          region.timing, nullptr, program_->runtime_fault_domain_);
    } else {
      program_->program_impl_->end_runtime_gpu_timing(region.timing);
    }
  }
  active_gpu_region_timings_.pop_back();
}

RuntimeCompletion Program::RuntimeSubmissionTransaction::finish() {
  TI_ERROR_IF(finished_, "Runtime submission transaction already finished");
  TI_ERROR_IF(!active_gpu_region_timings_.empty(),
              "Runtime submission transaction has unfinished GPU region timing");
  join_cuda_concurrent_streams();
  if (gpu_timing_) {
    if (program_->compile_config().arch == Arch::cuda) {
      RuntimeCompletion::end_cuda_gpu_timing(
          gpu_timing_, nullptr, program_->runtime_fault_domain_);
    } else {
      program_->program_impl_->end_runtime_gpu_timing(gpu_timing_);
    }
  }
  if (submission_batch_open_) {
    submission_batch_open_ = false;
    program_->program_impl_->end_runtime_submission_batch();
  }
  // record_runtime_completion() takes the writer boundary. Never attempt it
  // while this transaction still owns the corresponding reader.
  submission_scope_.reset();
  finished_ = true;
  if ((gpu_timing_requested_ || cuda_concurrent_batch_) &&
      Program::active_runtime_submission_telemetry_transaction() == this) {
    Program::active_runtime_submission_telemetry_transaction() =
        previous_telemetry_transaction_;
  }
  Program *program = std::exchange(program_, nullptr);
  return program->record_runtime_completion(
      std::move(gpu_timing_), std::move(gpu_region_timings_));
}

std::vector<SNodeTreeDependency> Program::snapshot_snode_tree_dependencies(
    const std::vector<int> &tree_ids) const {
  std::optional<std::shared_lock<std::shared_mutex>> lock;
  if (active_snode_tree_lifecycle_program != this) {
    lock.emplace(snode_tree_lifecycle_mutex_);
  }
  return snapshot_snode_tree_dependencies_unlocked(tree_ids);
}

std::vector<SNodeTreeDependency>
Program::snapshot_snode_tree_dependencies_unlocked(
    const std::vector<int> &tree_ids) const {
  std::vector<SNodeTreeDependency> dependencies;
  dependencies.reserve(tree_ids.size());
  for (const int tree_id : tree_ids) {
    TI_ERROR_IF(tree_id < 0 ||
                    static_cast<std::size_t>(tree_id) >=
                        snode_tree_active_.size() ||
                    !snode_tree_active_[tree_id],
                "Cannot compile a graph that references destroyed SNodeTree "
                "id={}.",
                tree_id);
    TI_ASSERT(static_cast<std::size_t>(tree_id) < snode_trees_.size());
    TI_ASSERT(snode_trees_[tree_id] != nullptr);
    dependencies.push_back(
        {tree_id, snode_tree_generations_[tree_id],
         snode_tree_layout_fingerprint(*snode_trees_[tree_id])});
  }
  return dependencies;
}

std::uint64_t Program::snode_tree_layout_fingerprint_for(int tree_id) const {
  return snapshot_snode_tree_dependencies({tree_id})[0].layout_fingerprint;
}

bool Program::prepare_relocatable_kernel_template(
    const Kernel &kernel_def,
    const std::vector<int> &direct_template_tree_ids) {
  if (!kernel_def.definition_retired() ||
      kernel_def.get_cached_kernel_key().empty()) {
    return false;
  }
  std::vector<int> normalized = direct_template_tree_ids;
  std::sort(normalized.begin(), normalized.end());
  normalized.erase(std::unique(normalized.begin(), normalized.end()),
                   normalized.end());
  if (normalized != kernel_def.snode_tree_dependencies()) {
    return false;
  }
  auto lifecycle_guard = acquire_snode_tree_lifecycle_read_guard();
  auto dependencies =
      snapshot_snode_tree_dependencies_unlocked(normalized);
  return program_impl_->get_kernel_compilation_manager()
             .instantiate_relocatable_execution_handle(
                 kernel_def.get_cached_kernel_key(), dependencies) != nullptr;
}

bool Program::register_relocatable_kernel_template_candidate(
    const Kernel &kernel_def,
    const std::vector<int> &direct_template_tree_ids) {
  if (kernel_def.definition_retired()) {
    return false;
  }
  std::vector<int> normalized = direct_template_tree_ids;
  std::sort(normalized.begin(), normalized.end());
  normalized.erase(std::unique(normalized.begin(), normalized.end()),
                   normalized.end());
  std::lock_guard<std::mutex> lock(relocatable_kernel_candidate_mutex_);
  relocatable_kernel_candidates_[&kernel_def] = std::move(normalized);
  return true;
}

std::uint64_t Program::reclaim_relocatable_kernel_templates(
    std::size_t maximum_resident) {
  return program_impl_->get_kernel_compilation_manager()
      .reclaim_relocatable_templates(maximum_resident);
}

bool Program::snode_executable_reuse_enabled() const noexcept {
  return program_impl_->get_kernel_compilation_manager()
      .relocatable_reuse_enabled();
}

void Program::validate_snode_tree_dependencies(
    const std::vector<SNodeTreeDependency> &dependencies,
    const char *consumer) const {
  TI_ASSERT(active_snode_tree_lifecycle_program == this);
  TI_ASSERT(consumer != nullptr);
  const bool graph_consumer = std::strcmp(consumer, "Graph") == 0;
  for (const auto &dependency : dependencies) {
    const int tree_id = dependency.tree_id;
    const bool destroyed =
        tree_id < 0 ||
        static_cast<std::size_t>(tree_id) >= snode_tree_active_.size() ||
        !snode_tree_active_[tree_id];
    if (graph_consumer) {
      TI_ERROR_IF(destroyed,
                  "Graph references destroyed SNodeTree id={} generation={}; "
                  "rebuild the Graph.",
                  tree_id, dependency.generation);
    } else {
      TI_ERROR_IF(destroyed,
                  "{} references destroyed SNodeTree id={} generation={}; "
                  "rebuild it.",
                  consumer, tree_id, dependency.generation);
    }
    const std::uint64_t current_generation =
        snode_tree_generations_[tree_id];
    if (graph_consumer) {
      TI_ERROR_IF(
          current_generation != dependency.generation,
          "Graph references stale SNodeTree id={} generation={}, but the "
          "current generation is {}; rebuild the Graph.",
          tree_id, dependency.generation, current_generation);
    } else {
      TI_ERROR_IF(
          current_generation != dependency.generation,
          "{} references stale SNodeTree id={} generation={}, but the current "
          "generation is {}; rebuild it.",
          consumer, tree_id, dependency.generation, current_generation);
    }
  }
}
std::atomic<int> Program::num_instances_;

namespace {
constexpr std::size_t kCpuPrimitiveMaxLocalWorkspaceBytes = 64ull << 20;
constexpr std::size_t kCpuPrimitiveRetainedScratchBytes = 8ull << 20;
constexpr std::uint64_t kCpuPrimitiveExecutionDomainSlots = 16;

thread_local Program *active_cpu_primitive_program = nullptr;

class ScopedCpuPrimitiveProgram {
 public:
  explicit ScopedCpuPrimitiveProgram(Program *program)
      : submission_guard_(
            program->acquire_runtime_resource_submission_guard()),
        previous_program_(active_cpu_primitive_program) {
    active_cpu_primitive_program = program;
  }

  ~ScopedCpuPrimitiveProgram() {
    active_cpu_primitive_program = previous_program_;
  }

 private:
  Program::RuntimeResourceSubmissionGuard submission_guard_;
  Program *previous_program_;
};

class ScopedRuntimeTransferStatistics {
 public:
  ScopedRuntimeTransferStatistics(Program *program,
                                  RuntimeTransferKind kind,
                                  std::size_t bytes) noexcept
      : program_(program),
        kind_(kind),
        bytes_(static_cast<std::uint64_t>(bytes)),
        trace_(&program->runtime_trace(), runtime_trace_kind(kind), bytes_),
        uncaught_exceptions_(std::uncaught_exceptions()) {
  }

  ~ScopedRuntimeTransferStatistics() {
    // Count logical payload only after the Program-level operation has
    // accepted/enqueued the copy. Backend-internal staging and retries are not
    // separate user-visible transfers.
    if (std::uncaught_exceptions() == uncaught_exceptions_) {
      program_->runtime_statistics().record_transfer(kind_, bytes_);
    } else {
      trace_.mark_failed();
    }
  }

  ScopedRuntimeTransferStatistics(const ScopedRuntimeTransferStatistics &) =
      delete;
  ScopedRuntimeTransferStatistics &operator=(
      const ScopedRuntimeTransferStatistics &) = delete;

 private:
  Program *program_;
  RuntimeTransferKind kind_;
  std::uint64_t bytes_;
  RuntimeTraceRecorder::Scope trace_;
  int uncaught_exceptions_;
};

std::uint64_t cpu_primitive_execution_domain() {
  // Keep retained scratch proportional to supported concurrent callers, not
  // to every short-lived Python thread that has ever touched this Program.
  // PrimitiveWorkspaceArena already serializes callers that hash to one slot.
  thread_local const std::uint64_t domain =
      1 + std::hash<std::thread::id>{}(std::this_thread::get_id()) %
              kCpuPrimitiveExecutionDomainSlots;
  return domain;
}

struct CpuPrimitiveScratch {
  std::unique_ptr<std::max_align_t[]> storage;
  std::size_t capacity_words{0};

  std::size_t allocated_bytes() const noexcept {
    return capacity_words * sizeof(std::max_align_t);
  }

  template <typename T>
  T *zeroed(std::size_t items) {
    TI_ERROR_IF(items > std::numeric_limits<std::size_t>::max() / sizeof(T),
                "CPU primitive scratch size overflowed.");
    const std::size_t bytes = items * sizeof(T);
    const std::size_t words =
        (bytes + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t);
    if (words > capacity_words) {
      storage = std::make_unique<std::max_align_t[]>(words);
      capacity_words = words;
    }
    auto *result = reinterpret_cast<T *>(storage.get());
    std::fill_n(result, items, T{});
    return result;
  }
};

template <typename T>
class CpuPrimitiveScratchBuffer {
 public:
  CpuPrimitiveScratchBuffer(PrimitiveWorkspaceFamily family,
                            std::size_t items) {
    TI_ERROR_IF(items > std::numeric_limits<std::size_t>::max() / sizeof(T),
                "CPU primitive scratch size overflowed.");
    const std::size_t bytes = items * sizeof(T);
    if (bytes <= kCpuPrimitiveRetainedScratchBytes) {
      TI_ASSERT(active_cpu_primitive_program != nullptr);
      auto lease =
          active_cpu_primitive_program->primitive_workspace_arena()
              .acquire<CpuPrimitiveScratch>(
                  {PrimitiveWorkspaceBackend::cpu, family,
                   cpu_primitive_execution_domain(), 0},
                  [] { return std::make_shared<CpuPrimitiveScratch>(); });
      retained_ = std::make_unique<ScratchLease>(std::move(lease));
      data_ = retained_->operator->()->template zeroed<T>(items);
      return;
    }
    transient_.assign(items, T{});
    data_ = transient_.data();
  }

  T *data() const noexcept {
    return data_;
  }

 private:
  using ScratchLease =
      PrimitiveWorkspaceArena::Lease<CpuPrimitiveScratch>;
  std::unique_ptr<ScratchLease> retained_;
  std::vector<T> transient_;
  T *data_{nullptr};
};

std::size_t cpu_primitive_workspace_bytes(
    const Program *program,
    PrimitiveWorkspaceFamily family) {
  return static_cast<std::size_t>(
      program->primitive_workspace_arena()
          .snapshot(PrimitiveWorkspaceBackend::cpu, family)
          .reserved_bytes);
}

uint32_t cpu_sortable_f32_key(float value, int nan_policy) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  constexpr uint32_t kSign = 0x80000000u;
  constexpr uint32_t kAbsMask = 0x7fffffffu;
  constexpr uint32_t kInfBits = 0x7f800000u;
  if (nan_policy == 0 && (bits & kAbsMask) > kInfBits) {
    return 0xffffffffu;
  }
  if (nan_policy == 0 && (bits & kAbsMask) == 0) {
    return kSign;
  }
  return (bits & kSign) ? ~bits : (bits ^ kSign);
}

uint64_t cpu_sortable_f64_key(double value, int nan_policy) {
  uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  constexpr uint64_t kSign = 0x8000000000000000ull;
  constexpr uint64_t kAbsMask = 0x7fffffffffffffffull;
  constexpr uint64_t kInfBits = 0x7ff0000000000000ull;
  if (nan_policy == 0 && (bits & kAbsMask) > kInfBits) {
    return 0xffffffffffffffffull;
  }
  if (nan_policy == 0 && (bits & kAbsMask) == 0) {
    return kSign;
  }
  return (bits & kSign) ? ~bits : (bits ^ kSign);
}

template <typename KeyT>
bool cpu_sort_key_before(KeyT lhs,
                         KeyT rhs,
                         bool descending,
                         int /*nan_policy*/) {
  if (lhs == rhs) {
    return false;
  }
  return descending ? lhs > rhs : lhs < rhs;
}

template <>
bool cpu_sort_key_before<float>(float lhs,
                                float rhs,
                                bool descending,
                                int nan_policy) {
  uint32_t lhs_bits = 0;
  uint32_t rhs_bits = 0;
  std::memcpy(&lhs_bits, &lhs, sizeof(lhs_bits));
  std::memcpy(&rhs_bits, &rhs, sizeof(rhs_bits));
  constexpr uint32_t kAbsMask = 0x7fffffffu;
  constexpr uint32_t kInfBits = 0x7f800000u;
  const bool lhs_nan = (lhs_bits & kAbsMask) > kInfBits;
  const bool rhs_nan = (rhs_bits & kAbsMask) > kInfBits;
  if (nan_policy == 0 && (lhs_nan || rhs_nan)) {
    return !lhs_nan && rhs_nan;
  }
  if (nan_policy == 0 && (lhs_bits & kAbsMask) == 0 &&
      (rhs_bits & kAbsMask) == 0) {
    return false;
  }
  const uint32_t lhs_key = cpu_sortable_f32_key(lhs, nan_policy);
  const uint32_t rhs_key = cpu_sortable_f32_key(rhs, nan_policy);
  if (lhs_key == rhs_key) {
    return false;
  }
  return descending ? lhs_key > rhs_key : lhs_key < rhs_key;
}

template <>
bool cpu_sort_key_before<double>(double lhs,
                                 double rhs,
                                 bool descending,
                                 int nan_policy) {
  uint64_t lhs_bits = 0;
  uint64_t rhs_bits = 0;
  std::memcpy(&lhs_bits, &lhs, sizeof(lhs_bits));
  std::memcpy(&rhs_bits, &rhs, sizeof(rhs_bits));
  constexpr uint64_t kAbsMask = 0x7fffffffffffffffull;
  constexpr uint64_t kInfBits = 0x7ff0000000000000ull;
  const bool lhs_nan = (lhs_bits & kAbsMask) > kInfBits;
  const bool rhs_nan = (rhs_bits & kAbsMask) > kInfBits;
  if (nan_policy == 0 && (lhs_nan || rhs_nan)) {
    return !lhs_nan && rhs_nan;
  }
  if (nan_policy == 0 && (lhs_bits & kAbsMask) == 0 &&
      (rhs_bits & kAbsMask) == 0) {
    return false;
  }
  const uint64_t lhs_key = cpu_sortable_f64_key(lhs, nan_policy);
  const uint64_t rhs_key = cpu_sortable_f64_key(rhs, nan_policy);
  if (lhs_key == rhs_key) {
    return false;
  }
  return descending ? lhs_key > rhs_key : lhs_key < rhs_key;
}

template <typename KeyT, typename ValueT>
std::size_t cpu_stable_sort_impl(KeyT *keys,
                                 ValueT *values,
                                 std::size_t n,
                                 bool descending,
                                 int nan_policy) {
  if (n <= 1) {
    return 0;
  }
  if (values) {
    struct Item {
      KeyT key;
      ValueT value;
    };
    std::vector<Item> items(n);
    for (std::size_t i = 0; i < n; ++i) {
      items[i] = {keys[i], values[i]};
    }
    std::stable_sort(items.begin(), items.end(), [&](const Item &lhs,
                                                     const Item &rhs) {
      return cpu_sort_key_before<KeyT>(
          lhs.key, rhs.key, descending, nan_policy);
    });
    for (std::size_t i = 0; i < n; ++i) {
      keys[i] = items[i].key;
      values[i] = items[i].value;
    }
    return items.size() * sizeof(Item);
  }

  std::vector<KeyT> sorted_keys(keys, keys + n);
  std::stable_sort(sorted_keys.begin(), sorted_keys.end(), [&](KeyT lhs,
                                                               KeyT rhs) {
    return cpu_sort_key_before<KeyT>(lhs, rhs, descending, nan_policy);
  });
  std::memcpy(keys, sorted_keys.data(), n * sizeof(KeyT));
  return sorted_keys.size() * sizeof(KeyT);
}

template <typename KeyT>
std::size_t cpu_stable_sort_value_dispatch(KeyT *keys,
                                           void *values,
                                           std::size_t n,
                                           int value_type,
                                           bool descending,
                                           int nan_policy) {
  if (!values) {
    return cpu_stable_sort_impl<KeyT, int32_t>(
        keys, nullptr, n, descending, nan_policy);
  }
  switch (value_type) {
    case 0:
      return cpu_stable_sort_impl<KeyT, int32_t>(
          keys, reinterpret_cast<int32_t *>(values), n, descending,
          nan_policy);
    case 1:
      return cpu_stable_sort_impl<KeyT, float>(
          keys, reinterpret_cast<float *>(values), n, descending, nan_policy);
    case 2:
      return cpu_stable_sort_impl<KeyT, uint32_t>(
          keys, reinterpret_cast<uint32_t *>(values), n, descending,
          nan_policy);
    case 3:
      return cpu_stable_sort_impl<KeyT, uint64_t>(
          keys, reinterpret_cast<uint64_t *>(values), n, descending,
          nan_policy);
    case 4:
      return cpu_stable_sort_impl<KeyT, int64_t>(
          keys, reinterpret_cast<int64_t *>(values), n, descending,
          nan_policy);
    case 5:
      return cpu_stable_sort_impl<KeyT, double>(
          keys, reinterpret_cast<double *>(values), n, descending, nan_policy);
    default:
      TI_ERROR("CPU native sort received an unsupported value type.");
  }
}

template <typename KeyT>
std::size_t cpu_stable_sort_raw_values(KeyT *keys,
                                       void *values,
                                       std::size_t n,
                                       std::size_t item_bytes,
                                       bool descending,
                                       int nan_policy) {
  if (n <= 1) {
    return 0;
  }
  auto *value_bytes = static_cast<uint8_t *>(values);
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](std::size_t lhs,
                                                   std::size_t rhs) {
    return cpu_sort_key_before<KeyT>(
        keys[lhs], keys[rhs], descending, nan_policy);
  });

  std::vector<KeyT> sorted_keys(n);
  std::vector<uint8_t> sorted_values(n * item_bytes);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t src = order[i];
    sorted_keys[i] = keys[src];
    std::memcpy(sorted_values.data() + i * item_bytes,
                value_bytes + src * item_bytes, item_bytes);
  }
  std::memcpy(keys, sorted_keys.data(), n * sizeof(KeyT));
  std::memcpy(value_bytes, sorted_values.data(), sorted_values.size());
  return order.size() * sizeof(std::size_t) + sorted_keys.size() * sizeof(KeyT) +
         sorted_values.size();
}

std::size_t sort_key_type_size(int key_type) {
  switch (key_type) {
    case 0:
    case 1:
    case 2:
      return sizeof(uint32_t);
    case 3:
    case 4:
    case 5:
      return sizeof(uint64_t);
    default:
      return 0;
  }
}

template <typename ValueT, typename CounterT>
struct CpuHistogramTaskContext {
  const ValueT *values{nullptr};
  CounterT *partial{nullptr};
  std::size_t n{0};
  std::size_t num_bins{0};
  int num_threads{1};
};

template <typename T>
struct CpuReduceTaskContext {
  const T *values{nullptr};
  T *partial{nullptr};
  std::size_t n{0};
  int num_threads{1};
  int op{0};
};

template <typename T>
struct CpuStridedReduceTaskContext {
  const uint8_t *values{nullptr};
  T *partial{nullptr};
  std::size_t n{0};
  std::size_t offset{0};
  std::size_t stride{0};
  int num_threads{1};
  int op{0};
};

struct CpuFillU32TaskContext {
  uint32_t *data{nullptr};
  std::size_t words{0};
  uint32_t value{0};
  int num_threads{1};
};

struct CpuCopyTaskContext {
  uint8_t *dst{nullptr};
  const uint8_t *src{nullptr};
  std::size_t bytes{0};
  int num_threads{1};
};

struct CpuDenseFieldCopyTaskContext {
  uint8_t *dst{nullptr};
  const uint8_t *src{nullptr};
  std::size_t item_bytes{0};
  std::size_t dst_stride{0};
  std::size_t src_stride{0};
  std::size_t n{0};
  int num_threads{1};
};

template <typename T>
struct CpuTransformTaskContext {
  const T *src{nullptr};
  T *dst{nullptr};
  std::size_t n{0};
  T scale{};
  T bias{};
  int num_threads{1};
};

template <typename T>
struct CpuStridedTransformTaskContext {
  const uint8_t *src{nullptr};
  T *dst{nullptr};
  std::size_t n{0};
  std::size_t offset{0};
  std::size_t stride{0};
  T scale{};
  T bias{};
  int num_threads{1};
};

template <typename T>
struct CpuStridedToStridedTransformTaskContext {
  const uint8_t *src{nullptr};
  uint8_t *dst{nullptr};
  std::size_t n{0};
  std::size_t src_offset{0};
  std::size_t src_stride{0};
  std::size_t dst_offset{0};
  std::size_t dst_stride{0};
  T scale{};
  T bias{};
  int num_threads{1};
};

template <typename T>
struct CpuPackedStridedToStridedTransformTaskContext {
  const uint8_t *src{nullptr};
  uint8_t *dst{nullptr};
  std::size_t n{0};
  int lane_count{1};
  std::size_t src_offset{0};
  std::size_t src_stride{0};
  std::size_t dst_offset{0};
  std::size_t dst_stride{0};
  T scale{};
  T bias{};
  int num_threads{1};
};

struct CpuIndexedCopyTaskContext {
  const uint8_t *src{nullptr};
  const int32_t *indices{nullptr};
  uint8_t *dst{nullptr};
  std::size_t n{0};
  std::size_t index_bound{0};
  std::size_t item_bytes{0};
  bool scatter{false};
  int num_threads{1};
};

struct CpuStridedIndexedCopyTaskContext {
  const uint8_t *src{nullptr};
  const int32_t *indices{nullptr};
  uint8_t *dst{nullptr};
  std::size_t n{0};
  std::size_t index_bound{0};
  std::size_t item_bytes{0};
  std::size_t src_offset{0};
  std::size_t src_stride{0};
  std::size_t dst_offset{0};
  std::size_t dst_stride{0};
  bool scatter{false};
  int num_threads{1};
};

template <typename T>
struct CpuStridedGatherAddTaskContext {
  const uint8_t *src{nullptr};
  const int32_t *indices{nullptr};
  uint8_t *dst{nullptr};
  std::size_t n{0};
  std::size_t index_bound{0};
  std::size_t src_offset{0};
  std::size_t src_stride{sizeof(T)};
  std::size_t dst_offset{0};
  std::size_t dst_stride{sizeof(T)};
  int num_threads{1};
};

template <typename T>
struct CpuScatterAddTaskContext {
  const T *src{nullptr};
  const int32_t *indices{nullptr};
  T *partial{nullptr};
  T *dst{nullptr};
  std::size_t n{0};
  std::size_t dst_items{0};
  int num_threads{1};
};

template <typename T>
struct CpuStridedScatterAddTaskContext {
  const uint8_t *src{nullptr};
  const int32_t *indices{nullptr};
  T *partial{nullptr};
  T *dst{nullptr};
  std::size_t n{0};
  std::size_t dst_items{0};
  std::size_t offset{0};
  std::size_t stride{0};
  int num_threads{1};
};

template <typename T>
struct CpuStridedScatterAddIoTaskContext {
  const uint8_t *src{nullptr};
  const int32_t *indices{nullptr};
  T *partial{nullptr};
  uint8_t *dst{nullptr};
  std::size_t n{0};
  std::size_t dst_items{0};
  std::size_t src_offset{0};
  std::size_t src_stride{0};
  std::size_t dst_offset{0};
  std::size_t dst_stride{0};
  int num_threads{1};
};

template <typename T>
struct CpuPackedScatterAddIoTaskContext {
  const uint8_t *src{nullptr};
  const int32_t *indices{nullptr};
  T *partial{nullptr};
  uint8_t *dst{nullptr};
  std::size_t n{0};
  std::size_t dst_items{0};
  int lane_count{1};
  std::size_t src_offset{0};
  std::size_t src_stride{0};
  std::size_t dst_offset{0};
  std::size_t dst_stride{0};
  int num_threads{1};
};

struct CpuBucketCountTaskContext {
  const int32_t *keys{nullptr};
  int32_t *partial{nullptr};
  std::size_t n{0};
  std::size_t num_bins{0};
  int num_threads{1};
};

template <typename T>
struct CpuBucketScatterTaskContext {
  const int32_t *keys{nullptr};
  const T *values{nullptr};
  int32_t *thread_offsets{nullptr};
  T *output{nullptr};
  std::size_t n{0};
  std::size_t num_bins{0};
  int num_threads{1};
};

struct CpuBucketScatterRawTaskContext {
  const int32_t *keys{nullptr};
  const uint8_t *values{nullptr};
  int32_t *thread_offsets{nullptr};
  uint8_t *output{nullptr};
  std::size_t n{0};
  std::size_t num_bins{0};
  std::size_t item_bytes{0};
  int num_threads{1};
};

struct CpuCompactCountTaskContext {
  const uint8_t *flags{nullptr};
  std::size_t flags_stride{sizeof(int32_t)};
  std::size_t *counts{nullptr};
  std::size_t n{0};
  int num_threads{1};
};

template <typename T>
struct CpuCompactScatterTaskContext {
  const uint8_t *values{nullptr};
  std::size_t values_stride{sizeof(T)};
  const uint8_t *flags{nullptr};
  std::size_t flags_stride{sizeof(int32_t)};
  uint8_t *output{nullptr};
  std::size_t output_stride{sizeof(T)};
  const std::size_t *offsets{nullptr};
  std::size_t n{0};
  int num_threads{1};
};

template <typename T>
struct CpuDenseFieldFillTaskContext {
  uint8_t *data{nullptr};
  std::size_t stride{sizeof(T)};
  T value{};
  std::size_t n{0};
  int num_threads{1};
};

template <typename T>
struct CpuGroupedReduceTaskContext {
  const int32_t *keys{nullptr};
  const T *values{nullptr};
  T *partial{nullptr};
  T *output{nullptr};
  std::size_t n{0};
  std::size_t num_groups{0};
  int num_threads{1};
};

template <typename T>
struct CpuStridedGroupedReduceTaskContext {
  const int32_t *keys{nullptr};
  const uint8_t *values{nullptr};
  T *partial{nullptr};
  T *output{nullptr};
  std::size_t n{0};
  std::size_t num_groups{0};
  std::size_t offset{0};
  std::size_t stride{0};
  int num_threads{1};
};

template <typename T>
struct CpuStridedGroupedReduceIoTaskContext {
  const uint8_t *keys{nullptr};
  const uint8_t *values{nullptr};
  T *partial{nullptr};
  uint8_t *output{nullptr};
  std::size_t n{0};
  std::size_t num_groups{0};
  std::size_t keys_offset{0};
  std::size_t keys_stride{sizeof(int32_t)};
  std::size_t values_offset{0};
  std::size_t values_stride{0};
  std::size_t output_offset{0};
  std::size_t output_stride{0};
  int num_threads{1};
};

ThreadPool *get_cpu_primitive_thread_pool(int max_threads) {
  TI_ERROR_IF(active_cpu_primitive_program == nullptr,
              "CPU native primitive scheduler was used outside a Program "
              "submission scope.");
  auto *const pool = active_cpu_primitive_program->get_cpu_thread_pool();
  TI_ASSERT(pool != nullptr);
  (void)max_threads;
  return pool;
}

bool cpu_use_parallel_simple_loop(std::size_t n, int target_threads) {
  return n >= 262144 && target_threads > 1;
}

bool cpu_use_parallel_aggregation(std::size_t n, int target_threads) {
  return n >= 262144 && target_threads >= 4;
}

int cpu_indexed_copy_target_threads(std::size_t n,
                                    int max_threads,
                                    bool scatter) {
  constexpr int kChunkItems = 32768;
  int target_threads = static_cast<int>(
      std::min<std::size_t>((n + kChunkItems - 1) / kChunkItems,
                            static_cast<std::size_t>(max_threads)));
  if (scatter && n >= 262144) {
    target_threads = std::min(target_threads, 4);
  }
  return std::max(1, target_threads);
}

void validate_cpu_plain_scatter_indices(const int32_t *indices,
                                        std::size_t n,
                                        std::size_t dst_items) {
  // Plain scatter has a unique-target contract. In particular, allowing two
  // worker tasks to memcpy to one destination is a host data race, not a
  // permissible last-writer-wins implementation. Check before either the
  // parallel or serial write path so an invalid request leaves dst untouched.
  // Invalid indices remain ignored, matching the established indexed-copy
  // contract; only in-range targets participate in uniqueness validation.
  if (n == 0 || dst_items == 0) {
    return;
  }

  try {
    // A byte marker is much cheaper than a hash table for the usual
    // permutation-like case. Do not size it wildly beyond input cardinality:
    // sparse destination index spaces instead use an O(n) hash set.
    if (n >= std::numeric_limits<std::size_t>::max() / 8 ||
        dst_items <= n * 8) {
      std::vector<uint8_t> seen(dst_items, 0);
      for (std::size_t i = 0; i < n; ++i) {
        const int32_t raw_index = indices[i];
        if (raw_index < 0) {
          continue;
        }
        const std::size_t index = static_cast<std::size_t>(raw_index);
        if (index >= dst_items) {
          continue;
        }
        TI_ERROR_IF(seen[index] != 0,
                    "CPU native plain scatter requires unique destination "
                    "indices; use experimental_scatter_add() for duplicate "
                    "targets.");
        seen[index] = 1;
      }
      return;
    }

    std::unordered_set<int32_t> seen;
    seen.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
      const int32_t raw_index = indices[i];
      if (raw_index < 0 ||
          static_cast<std::size_t>(raw_index) >= dst_items) {
        continue;
      }
      TI_ERROR_IF(!seen.insert(raw_index).second,
                  "CPU native plain scatter requires unique destination "
                  "indices; use experimental_scatter_add() for duplicate "
                  "targets.");
    }
  } catch (const std::bad_alloc &) {
    TI_ERROR("CPU native plain scatter could not allocate uniqueness "
             "validation storage; use experimental_scatter_add() for "
             "duplicate targets or reduce the request size.");
  }
}

int cpu_aggregation_target_threads(std::size_t n,
                                   std::size_t groups,
                                   int max_threads) {
  int target_threads = std::max(1, static_cast<int>(n / 65536));
  if (groups <= 1024 && n >= 262144) {
    target_threads = std::min(16, std::max(1, static_cast<int>(n / 32768)));
  }
  return std::min(max_threads, target_threads);
}
template <typename T>
T cpu_reduce_identity(int op) {
  if (op == 1) {
    if constexpr (std::is_floating_point_v<T>) {
      return std::numeric_limits<T>::infinity();
    } else {
      return std::numeric_limits<T>::max();
    }
  }
  if (op == 2) {
    if constexpr (std::is_floating_point_v<T>) {
      return -std::numeric_limits<T>::infinity();
    } else {
      return std::numeric_limits<T>::lowest();
    }
  }
  return T{0};
}

template <typename T>
T cpu_reduce_combine(T a, T b, int op) {
  if (op == 1) {
    return std::min(a, b);
  }
  if (op == 2) {
    return std::max(a, b);
  }
  if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
    using U = std::make_unsigned_t<T>;
    U ua = 0;
    U ub = 0;
    std::memcpy(&ua, &a, sizeof(T));
    std::memcpy(&ub, &b, sizeof(T));
    U sum = ua + ub;
    T result{};
    std::memcpy(&result, &sum, sizeof(T));
    return result;
  } else {
    return a + b;
  }
}

template <typename T>
T cpu_reduce_sum_contiguous_range(const T *values,
                                  std::size_t begin,
                                  std::size_t end) {
  if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
    using U = std::make_unsigned_t<T>;
    const auto *unsigned_values = reinterpret_cast<const U *>(values);
    U acc = 0;
    for (std::size_t i = begin; i < end; ++i) {
      acc += unsigned_values[i];
    }
    T result{};
    std::memcpy(&result, &acc, sizeof(T));
    return result;
  } else {
    T acc{};
    for (std::size_t i = begin; i < end; ++i) {
      acc += values[i];
    }
    return acc;
  }
}

template <typename T>
T cpu_reduce_sum_strided_range(const uint8_t *values,
                               std::size_t offset,
                               std::size_t stride,
                               std::size_t begin,
                               std::size_t end) {
  if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
    using U = std::make_unsigned_t<T>;
    U acc = 0;
    for (std::size_t i = begin; i < end; ++i) {
      const auto *value =
          reinterpret_cast<const T *>(values + offset + i * stride);
      U bits = 0;
      std::memcpy(&bits, value, sizeof(T));
      acc += bits;
    }
    T result{};
    std::memcpy(&result, &acc, sizeof(T));
    return result;
  } else {
    T acc{};
    for (std::size_t i = begin; i < end; ++i) {
      const auto *value =
          reinterpret_cast<const T *>(values + offset + i * stride);
      acc += *value;
    }
    return acc;
  }
}

template <typename T>
void cpu_reduce_task(void *raw_ctx, int /*thread_id*/, int task_id) {
  auto *ctx = static_cast<CpuReduceTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  if (ctx->op == 0) {
    ctx->partial[tid] = cpu_reduce_sum_contiguous_range(ctx->values, begin, end);
    return;
  }
  T acc = cpu_reduce_identity<T>(ctx->op);
  for (std::size_t i = begin; i < end; ++i) {
    acc = cpu_reduce_combine(acc, ctx->values[i], ctx->op);
  }
  ctx->partial[tid] = acc;
}

template <typename T>
void cpu_strided_reduce_task(void *raw_ctx,
                             int /*thread_id*/,
                             int task_id) {
  auto *ctx = static_cast<CpuStridedReduceTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  if (ctx->op == 0) {
    ctx->partial[tid] = cpu_reduce_sum_strided_range<T>(
        ctx->values, ctx->offset, ctx->stride, begin, end);
    return;
  }
  T acc = cpu_reduce_identity<T>(ctx->op);
  for (std::size_t i = begin; i < end; ++i) {
    const auto *value = reinterpret_cast<const T *>(
        ctx->values + ctx->offset + i * ctx->stride);
    acc = cpu_reduce_combine(acc, *value, ctx->op);
  }
  ctx->partial[tid] = acc;
}

void cpu_fill_u32_task(void *raw_ctx, int /*thread_id*/, int task_id) {
  auto *ctx = static_cast<CpuFillU32TaskContext *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->words * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->words * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  std::fill(ctx->data + begin, ctx->data + end, ctx->value);
}

template <typename T>
void cpu_dense_field_fill_task(void *raw_ctx,
                               int /*thread_id*/,
                               int task_id) {
  auto *ctx = static_cast<CpuDenseFieldFillTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  if (ctx->stride == sizeof(T)) {
    auto *data = reinterpret_cast<T *>(ctx->data);
    std::fill(data + begin, data + end, ctx->value);
    return;
  }
  for (std::size_t i = begin; i < end; ++i) {
    *reinterpret_cast<T *>(ctx->data + i * ctx->stride) = ctx->value;
  }
}

void cpu_copy_task(void *raw_ctx, int /*thread_id*/, int task_id) {
  auto *ctx = static_cast<CpuCopyTaskContext *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->bytes * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->bytes * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  std::memcpy(ctx->dst + begin, ctx->src + begin, end - begin);
}

void cpu_dense_field_copy_task(void *raw_ctx,
                               int /*thread_id*/,
                               int task_id) {
  auto *ctx = static_cast<CpuDenseFieldCopyTaskContext *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    std::memcpy(ctx->dst + i * ctx->dst_stride,
                ctx->src + i * ctx->src_stride, ctx->item_bytes);
  }
}

void cpu_compact_count_task(void *raw_ctx, int /*thread_id*/, int task_id) {
  auto *ctx = static_cast<CpuCompactCountTaskContext *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  std::size_t count = 0;
  for (std::size_t i = begin; i < end; ++i) {
    const int32_t flag =
        *reinterpret_cast<const int32_t *>(ctx->flags + i * ctx->flags_stride);
    count += flag != 0;
  }
  ctx->counts[tid] = count;
}

template <typename T>
void cpu_compact_scatter_task(void *raw_ctx, int /*thread_id*/, int task_id) {
  auto *ctx = static_cast<CpuCompactScatterTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  std::size_t written = ctx->offsets[tid];
  if (ctx->values_stride == sizeof(T) &&
      ctx->flags_stride == sizeof(int32_t) &&
      ctx->output_stride == sizeof(T)) {
    const auto *values = reinterpret_cast<const T *>(ctx->values);
    const auto *flags = reinterpret_cast<const int32_t *>(ctx->flags);
    auto *output = reinterpret_cast<T *>(ctx->output);
    for (std::size_t i = begin; i < end; ++i) {
      if (flags[i] != 0) {
        output[written++] = values[i];
      }
    }
    return;
  }
  for (std::size_t i = begin; i < end; ++i) {
    const int32_t flag =
        *reinterpret_cast<const int32_t *>(ctx->flags + i * ctx->flags_stride);
    if (flag != 0) {
      *reinterpret_cast<T *>(ctx->output + written * ctx->output_stride) =
          *reinterpret_cast<const T *>(ctx->values + i * ctx->values_stride);
      written++;
    }
  }
}

template <typename T>
void cpu_transform_task(void *raw_ctx, int /*thread_id*/, int task_id) {
  auto *ctx = static_cast<CpuTransformTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    ctx->dst[i] = ctx->src[i] * ctx->scale + ctx->bias;
  }
}

template <typename T>
void cpu_strided_transform_task(void *raw_ctx,
                                int /*thread_id*/,
                                int task_id) {
  auto *ctx = static_cast<CpuStridedTransformTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    const auto *value = reinterpret_cast<const T *>(
        ctx->src + ctx->offset + i * ctx->stride);
    ctx->dst[i] = (*value) * ctx->scale + ctx->bias;
  }
}

template <typename T>
void cpu_strided_to_strided_transform_task(void *raw_ctx,
                                           int /*thread_id*/,
                                           int task_id) {
  auto *ctx =
      static_cast<CpuStridedToStridedTransformTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    const auto *value = reinterpret_cast<const T *>(
        ctx->src + ctx->src_offset + i * ctx->src_stride);
    auto *out = reinterpret_cast<T *>(
        ctx->dst + ctx->dst_offset + i * ctx->dst_stride);
    *out = (*value) * ctx->scale + ctx->bias;
  }
}

template <typename T>
void cpu_packed_strided_to_strided_transform_task(void *raw_ctx,
                                                  int /*thread_id*/,
                                                  int task_id) {
  auto *ctx =
      static_cast<CpuPackedStridedToStridedTransformTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t total =
      ctx->n * static_cast<std::size_t>(ctx->lane_count);
  const std::size_t begin =
      total * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      total * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t scalar_i = begin; scalar_i < end; ++scalar_i) {
    const std::size_t item =
        scalar_i / static_cast<std::size_t>(ctx->lane_count);
    const std::size_t lane =
        scalar_i - item * static_cast<std::size_t>(ctx->lane_count);
    const std::size_t lane_offset = lane * sizeof(T);
    const auto *value = reinterpret_cast<const T *>(
        ctx->src + ctx->src_offset + item * ctx->src_stride + lane_offset);
    auto *out = reinterpret_cast<T *>(
        ctx->dst + ctx->dst_offset + item * ctx->dst_stride + lane_offset);
    *out = (*value) * ctx->scale + ctx->bias;
  }
}

template <typename T>
void cpu_transform_run_typed(const T *src_ptr,
                             T *dst_ptr,
                             std::size_t n,
                             T scale,
                             T bias,
                             bool use_parallel,
                             int target_threads,
                             int max_threads) {
  if (use_parallel) {
    CpuTransformTaskContext<T> ctx;
    ctx.src = src_ptr;
    ctx.dst = dst_ptr;
    ctx.n = n;
    ctx.scale = scale;
    ctx.bias = bias;
    ctx.num_threads = target_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(target_threads, target_threads, &ctx, cpu_transform_task<T>);
    return;
  }
  for (std::size_t i = 0; i < n; ++i) {
    dst_ptr[i] = src_ptr[i] * scale + bias;
  }
}

template <typename T>
void cpu_transform_run_strided_typed(const uint8_t *src_ptr,
                                     T *dst_ptr,
                                     std::size_t n,
                                     std::size_t offset,
                                     std::size_t stride,
                                     T scale,
                                     T bias,
                                     bool use_parallel,
                                     int target_threads,
                                     int max_threads) {
  if (use_parallel) {
    CpuStridedTransformTaskContext<T> ctx;
    ctx.src = src_ptr;
    ctx.dst = dst_ptr;
    ctx.n = n;
    ctx.offset = offset;
    ctx.stride = stride;
    ctx.scale = scale;
    ctx.bias = bias;
    ctx.num_threads = target_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(target_threads, target_threads, &ctx,
             cpu_strided_transform_task<T>);
    return;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const auto *value =
        reinterpret_cast<const T *>(src_ptr + offset + i * stride);
    dst_ptr[i] = (*value) * scale + bias;
  }
}

template <typename T>
void cpu_transform_run_strided_to_strided_typed(const uint8_t *src_ptr,
                                                uint8_t *dst_ptr,
                                                std::size_t n,
                                                std::size_t src_offset,
                                                std::size_t src_stride,
                                                std::size_t dst_offset,
                                                std::size_t dst_stride,
                                                T scale,
                                                T bias,
                                                bool use_parallel,
                                                int target_threads,
                                                int max_threads) {
  if (use_parallel) {
    CpuStridedToStridedTransformTaskContext<T> ctx;
    ctx.src = src_ptr;
    ctx.dst = dst_ptr;
    ctx.n = n;
    ctx.src_offset = src_offset;
    ctx.src_stride = src_stride;
    ctx.dst_offset = dst_offset;
    ctx.dst_stride = dst_stride;
    ctx.scale = scale;
    ctx.bias = bias;
    ctx.num_threads = target_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(target_threads, target_threads, &ctx,
             cpu_strided_to_strided_transform_task<T>);
    return;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const auto *value =
        reinterpret_cast<const T *>(src_ptr + src_offset + i * src_stride);
    auto *out =
        reinterpret_cast<T *>(dst_ptr + dst_offset + i * dst_stride);
    *out = (*value) * scale + bias;
  }
}

template <typename T>
void cpu_transform_run_packed_strided_to_strided_typed(
    const uint8_t *src_ptr,
    uint8_t *dst_ptr,
    std::size_t n,
    int lane_count,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride,
    T scale,
    T bias,
    bool use_parallel,
    int target_threads,
    int max_threads) {
  if (use_parallel) {
    CpuPackedStridedToStridedTransformTaskContext<T> ctx;
    ctx.src = src_ptr;
    ctx.dst = dst_ptr;
    ctx.n = n;
    ctx.lane_count = lane_count;
    ctx.src_offset = src_offset;
    ctx.src_stride = src_stride;
    ctx.dst_offset = dst_offset;
    ctx.dst_stride = dst_stride;
    ctx.scale = scale;
    ctx.bias = bias;
    ctx.num_threads = target_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(target_threads, target_threads, &ctx,
             cpu_packed_strided_to_strided_transform_task<T>);
    return;
  }
  const std::size_t total = n * static_cast<std::size_t>(lane_count);
  for (std::size_t scalar_i = 0; scalar_i < total; ++scalar_i) {
    const std::size_t item =
        scalar_i / static_cast<std::size_t>(lane_count);
    const std::size_t lane =
        scalar_i - item * static_cast<std::size_t>(lane_count);
    const std::size_t lane_offset = lane * sizeof(T);
    const auto *value = reinterpret_cast<const T *>(
        src_ptr + src_offset + item * src_stride + lane_offset);
    auto *out = reinterpret_cast<T *>(
        dst_ptr + dst_offset + item * dst_stride + lane_offset);
    *out = (*value) * scale + bias;
  }
}

template <typename T>
void cpu_add_merge_task(void *raw_ctx, int /*thread_id*/, int task_id) {
  auto *ctx = static_cast<CpuTransformTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    ctx->dst[i] += ctx->src[i];
  }
}

template <typename T>
void cpu_strided_to_strided_add_merge_task(void *raw_ctx,
                                           int /*thread_id*/,
                                           int task_id) {
  auto *ctx = static_cast<CpuStridedToStridedTransformTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    const auto *value =
        reinterpret_cast<const T *>(ctx->src + ctx->src_offset +
                                    i * ctx->src_stride);
    auto *out =
        reinterpret_cast<T *>(ctx->dst + ctx->dst_offset +
                              i * ctx->dst_stride);
    *out += *value;
  }
}

template <typename T>
void cpu_add_scaled_task(void *raw_ctx, int /*thread_id*/, int task_id) {
  auto *ctx = static_cast<CpuTransformTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    ctx->dst[i] += ctx->src[i] * ctx->scale;
  }
}

template <typename T>
void cpu_strided_to_strided_add_scaled_task(void *raw_ctx,
                                            int /*thread_id*/,
                                            int task_id) {
  auto *ctx = static_cast<CpuStridedToStridedTransformTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    const auto *value =
        reinterpret_cast<const T *>(ctx->src + ctx->src_offset +
                                    i * ctx->src_stride);
    auto *out =
        reinterpret_cast<T *>(ctx->dst + ctx->dst_offset +
                              i * ctx->dst_stride);
    *out += (*value) * ctx->scale;
  }
}

template <typename T>
void cpu_add_merge_run_typed(const T *src_ptr,
                             T *dst_ptr,
                             std::size_t n,
                             bool use_parallel,
                             int target_threads,
                             int max_threads) {
  if (use_parallel) {
    CpuTransformTaskContext<T> ctx;
    ctx.src = src_ptr;
    ctx.dst = dst_ptr;
    ctx.n = n;
    ctx.scale = T{};
    ctx.bias = T{};
    ctx.num_threads = target_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(target_threads, target_threads, &ctx, cpu_add_merge_task<T>);
    return;
  }
  for (std::size_t i = 0; i < n; ++i) {
    dst_ptr[i] += src_ptr[i];
  }
}

template <typename T>
void cpu_add_scaled_run_typed(const T *src_ptr,
                              T *dst_ptr,
                              std::size_t n,
                              T scale,
                              bool use_parallel,
                              int target_threads,
                              int max_threads) {
  if (use_parallel) {
    CpuTransformTaskContext<T> ctx;
    ctx.src = src_ptr;
    ctx.dst = dst_ptr;
    ctx.n = n;
    ctx.scale = scale;
    ctx.bias = T{};
    ctx.num_threads = target_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(target_threads, target_threads, &ctx, cpu_add_scaled_task<T>);
    return;
  }
  for (std::size_t i = 0; i < n; ++i) {
    dst_ptr[i] += src_ptr[i] * scale;
  }
}

template <typename T>
void cpu_add_scaled_run_strided_to_strided_typed(
    const uint8_t *src_ptr,
    uint8_t *dst_ptr,
    std::size_t n,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride,
    T scale,
    bool use_parallel,
    int target_threads,
    int max_threads) {
  if (use_parallel) {
    CpuStridedToStridedTransformTaskContext<T> ctx;
    ctx.src = src_ptr;
    ctx.dst = dst_ptr;
    ctx.n = n;
    ctx.src_offset = src_offset;
    ctx.src_stride = src_stride;
    ctx.dst_offset = dst_offset;
    ctx.dst_stride = dst_stride;
    ctx.scale = scale;
    ctx.bias = T{};
    ctx.num_threads = target_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(target_threads, target_threads, &ctx,
             cpu_strided_to_strided_add_scaled_task<T>);
    return;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const auto *value =
        reinterpret_cast<const T *>(src_ptr + src_offset + i * src_stride);
    auto *out =
        reinterpret_cast<T *>(dst_ptr + dst_offset + i * dst_stride);
    *out += (*value) * scale;
  }
}

template <typename T>
void cpu_add_merge_run_strided_to_strided_typed(const uint8_t *src_ptr,
                                                uint8_t *dst_ptr,
                                                std::size_t n,
                                                std::size_t src_offset,
                                                std::size_t src_stride,
                                                std::size_t dst_offset,
                                                std::size_t dst_stride,
                                                bool use_parallel,
                                                int target_threads,
                                                int max_threads) {
  if (use_parallel) {
    CpuStridedToStridedTransformTaskContext<T> ctx;
    ctx.src = src_ptr;
    ctx.dst = dst_ptr;
    ctx.n = n;
    ctx.src_offset = src_offset;
    ctx.src_stride = src_stride;
    ctx.dst_offset = dst_offset;
    ctx.dst_stride = dst_stride;
    ctx.scale = T{};
    ctx.bias = T{};
    ctx.num_threads = target_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(target_threads, target_threads, &ctx,
             cpu_strided_to_strided_add_merge_task<T>);
    return;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const auto *value =
        reinterpret_cast<const T *>(src_ptr + src_offset + i * src_stride);
    auto *out =
        reinterpret_cast<T *>(dst_ptr + dst_offset + i * dst_stride);
    *out += *value;
  }
}

template <typename T>
void cpu_strided_gather_add_task(void *raw_ctx,
                                 int /*thread_id*/,
                                 int task_id) {
  auto *ctx = static_cast<CpuStridedGatherAddTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    const auto index = static_cast<std::size_t>(ctx->indices[i]);
    if (index < ctx->index_bound) {
      const auto *value = reinterpret_cast<const T *>(
          ctx->src + ctx->src_offset + index * ctx->src_stride);
      auto *out = reinterpret_cast<T *>(
          ctx->dst + ctx->dst_offset + i * ctx->dst_stride);
      *out += *value;
    }
  }
}

template <typename T>
void cpu_gather_add_run_strided_to_strided_typed(
    const uint8_t *src_ptr,
    const int32_t *indices_ptr,
    uint8_t *dst_ptr,
    std::size_t n,
    std::size_t index_bound,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride,
    bool use_parallel,
    int target_threads,
    int max_threads) {
  if (use_parallel) {
    CpuStridedGatherAddTaskContext<T> ctx;
    ctx.src = src_ptr;
    ctx.indices = indices_ptr;
    ctx.dst = dst_ptr;
    ctx.n = n;
    ctx.index_bound = index_bound;
    ctx.src_offset = src_offset;
    ctx.src_stride = src_stride;
    ctx.dst_offset = dst_offset;
    ctx.dst_stride = dst_stride;
    ctx.num_threads = target_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(target_threads, target_threads, &ctx,
             cpu_strided_gather_add_task<T>);
    return;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const auto index = static_cast<std::size_t>(indices_ptr[i]);
    if (index < index_bound) {
      const auto *value =
          reinterpret_cast<const T *>(src_ptr + src_offset +
                                      index * src_stride);
      auto *out =
          reinterpret_cast<T *>(dst_ptr + dst_offset + i * dst_stride);
      *out += *value;
    }
  }
}

std::size_t transform_value_size(int value_type) {
  TI_ERROR_IF(value_type < 0 || value_type > 5,
              "transform received an unsupported value type.");
  return (value_type == 3 || value_type == 4 || value_type == 5)
             ? sizeof(uint64_t)
             : sizeof(uint32_t);
}

void check_transform_member_request(const char *backend,
                                    Ndarray *src,
                                    Ndarray *dst,
                                    int value_type,
                                    std::size_t offset,
                                    std::size_t stride) {
  TI_ERROR_IF(!src || !dst, "{} strided transform received a null ndarray.",
              backend);
  TI_ERROR_IF(src->get_nelement() != dst->get_nelement(),
              "{} strided transform source and destination sizes differ.",
              backend);
  const std::size_t value_size = transform_value_size(value_type);
  TI_ERROR_IF(dst->get_element_size() != value_size,
              "{} strided transform destination dtype does not match value "
              "type.",
              backend);
  TI_ERROR_IF(stride < value_size,
              "{} strided transform source stride is smaller than value size.",
              backend);
  TI_ERROR_IF(offset % value_size != 0 || stride % value_size != 0,
              "{} strided transform source offset/stride must align to value "
              "size.",
              backend);
  const std::size_t n = src->get_nelement();
  if (n == 0) {
    return;
  }
  const std::size_t src_bytes = n * src->get_element_size();
  TI_ERROR_IF(src_bytes < value_size,
              "{} strided transform source buffer is smaller than value size.",
              backend);
  TI_ERROR_IF(offset > src_bytes - value_size,
              "{} strided transform source offset is out of bounds.", backend);
  const std::size_t last = offset + (n - 1) * stride + value_size;
  TI_ERROR_IF(last > src_bytes,
              "{} strided transform source range is out of bounds.", backend);
}

void check_transform_strided_range(const char *backend,
                                   const char *role,
                                   Ndarray *arr,
                                   std::size_t logical_items,
                                   std::size_t value_size,
                                   std::size_t offset,
                                   std::size_t stride) {
  TI_ERROR_IF(stride < value_size,
              "{} strided transform {} stride is smaller than value size.",
              backend, role);
  TI_ERROR_IF(offset % value_size != 0 || stride % value_size != 0,
              "{} strided transform {} offset/stride must align to value "
              "size.",
              backend, role);
  if (logical_items == 0) {
    return;
  }
  const std::size_t bytes = arr->get_nelement() * arr->get_element_size();
  TI_ERROR_IF(bytes < value_size,
              "{} strided transform {} buffer is smaller than value size.",
              backend, role);
  TI_ERROR_IF(offset > bytes - value_size,
              "{} strided transform {} offset is out of bounds.", backend,
              role);
  const std::size_t last = offset + (logical_items - 1) * stride + value_size;
  TI_ERROR_IF(last > bytes,
              "{} strided transform {} range is out of bounds.", backend,
              role);
}

void check_transform_strided_request(const char *backend,
                                     Ndarray *src,
                                     Ndarray *dst,
                                     int value_type,
                                     std::size_t src_offset,
                                     std::size_t src_stride,
                                     std::size_t dst_offset,
                                     std::size_t dst_stride) {
  TI_ERROR_IF(!src || !dst, "{} strided transform received a null ndarray.",
              backend);
  TI_ERROR_IF(src->get_nelement() != dst->get_nelement(),
              "{} strided transform source and destination sizes differ.",
              backend);
  const std::size_t value_size = transform_value_size(value_type);
  const std::size_t n = src->get_nelement();
  check_transform_strided_range(backend, "source", src, n, value_size,
                                src_offset, src_stride);
  check_transform_strided_range(backend, "destination", dst, n, value_size,
                                dst_offset, dst_stride);
}

void check_add_merge_strided_request(const char *backend,
                                     Ndarray *src,
                                     Ndarray *dst,
                                     int value_type,
                                     std::size_t src_offset,
                                     std::size_t src_stride,
                                     std::size_t dst_offset,
                                     std::size_t dst_stride) {
  TI_ERROR_IF(!src || !dst, "{} strided add-merge received a null ndarray.",
              backend);
  TI_ERROR_IF(src->shape.size() != 1 || dst->shape.size() != 1,
              "{} strided add-merge expects 1D ndarrays.", backend);
  TI_ERROR_IF(src->get_nelement() != dst->get_nelement(),
              "{} strided add-merge source and destination sizes differ.",
              backend);
  const std::size_t value_size = transform_value_size(value_type);
  const std::size_t n = src->get_nelement();
  check_transform_strided_range(backend, "add-merge source", src, n,
                                value_size, src_offset, src_stride);
  check_transform_strided_range(backend, "add-merge destination", dst, n,
                                value_size, dst_offset, dst_stride);
}

void check_transform_packed_strided_range(const char *backend,
                                          const char *role,
                                          Ndarray *arr,
                                          std::size_t logical_items,
                                          std::size_t value_size,
                                          int lane_count,
                                          std::size_t offset,
                                          std::size_t stride) {
  TI_ERROR_IF(lane_count <= 0,
              "{} packed strided transform lane count must be positive.",
              backend);
  const std::size_t payload_bytes =
      static_cast<std::size_t>(lane_count) * value_size;
  TI_ERROR_IF(stride < payload_bytes,
              "{} packed strided transform {} stride is smaller than payload.",
              backend, role);
  TI_ERROR_IF(offset % value_size != 0 || stride % value_size != 0,
              "{} packed strided transform {} offset/stride must align to "
              "value size.",
              backend, role);
  if (logical_items == 0) {
    return;
  }
  const std::size_t bytes = arr->get_nelement() * arr->get_element_size();
  TI_ERROR_IF(bytes < payload_bytes,
              "{} packed strided transform {} buffer is smaller than payload.",
              backend, role);
  TI_ERROR_IF(offset > bytes - payload_bytes,
              "{} packed strided transform {} offset is out of bounds.",
              backend, role);
  const std::size_t last =
      offset + (logical_items - 1) * stride + payload_bytes;
  TI_ERROR_IF(last > bytes,
              "{} packed strided transform {} range is out of bounds.",
              backend, role);
}

void check_transform_packed_strided_request(const char *backend,
                                            Ndarray *src,
                                            Ndarray *dst,
                                            int value_type,
                                            int lane_count,
                                            std::size_t src_offset,
                                            std::size_t src_stride,
                                            std::size_t dst_offset,
                                            std::size_t dst_stride) {
  TI_ERROR_IF(!src || !dst,
              "{} packed strided transform received a null ndarray.", backend);
  TI_ERROR_IF(src->get_nelement() != dst->get_nelement(),
              "{} packed strided transform source and destination sizes "
              "differ.",
              backend);
  const std::size_t value_size = transform_value_size(value_type);
  const std::size_t n = src->get_nelement();
  check_transform_packed_strided_range(
      backend, "source", src, n, value_size, lane_count, src_offset,
      src_stride);
  check_transform_packed_strided_range(
      backend, "destination", dst, n, value_size, lane_count, dst_offset,
      dst_stride);
}

void check_indexed_copy_strided_request(const char *backend,
                                        Ndarray *src,
                                        Ndarray *indices,
                                        Ndarray *dst,
                                        std::size_t item_bytes,
                                        std::size_t src_offset,
                                        std::size_t src_stride,
                                        std::size_t dst_offset,
                                        std::size_t dst_stride,
                                        bool scatter) {
  TI_ERROR_IF(!src || !indices || !dst,
              "{} strided indexed-copy received a null ndarray.", backend);
  TI_ERROR_IF(src->shape.size() != 1 || indices->shape.size() != 1 ||
                  dst->shape.size() != 1,
              "{} strided indexed-copy expects 1D ndarrays.", backend);
  TI_ERROR_IF(indices->get_element_size() != sizeof(int32_t),
              "{} strided indexed-copy expects i32 indices.", backend);
  TI_ERROR_IF(item_bytes == 0 || item_bytes % sizeof(uint32_t) != 0,
              "{} strided indexed-copy item size must be a positive "
              "uint32-word multiple.",
              backend);
  if (scatter) {
    TI_ERROR_IF(src->get_nelement() != indices->get_nelement(),
                "{} strided scatter expects source and indices sizes to "
                "match.",
                backend);
  } else {
    TI_ERROR_IF(indices->get_nelement() != dst->get_nelement(),
                "{} strided gather expects indices and destination sizes to "
                "match.",
                backend);
  }
  auto check_range = [&](const char *role, Ndarray *arr,
                         std::size_t logical_items, std::size_t offset,
                         std::size_t stride) {
    TI_ERROR_IF(stride < item_bytes,
                "{} strided indexed-copy {} stride is smaller than item "
                "size.",
                backend, role);
    TI_ERROR_IF(offset % sizeof(uint32_t) != 0 ||
                    stride % sizeof(uint32_t) != 0,
                "{} strided indexed-copy {} offset/stride must be "
                "uint32-word aligned.",
                backend, role);
    if (logical_items == 0) {
      return;
    }
    const std::size_t bytes = arr->get_nelement() * arr->get_element_size();
    TI_ERROR_IF(bytes < item_bytes,
                "{} strided indexed-copy {} buffer is smaller than item "
                "size.",
                backend, role);
    TI_ERROR_IF(offset > bytes - item_bytes,
                "{} strided indexed-copy {} offset is out of bounds.",
                backend, role);
    const std::size_t last = offset + (logical_items - 1) * stride + item_bytes;
    TI_ERROR_IF(last > bytes,
                "{} strided indexed-copy {} range is out of bounds.",
                backend, role);
  };
  check_range("source", src, src->get_nelement(), src_offset, src_stride);
  check_range("destination", dst, dst->get_nelement(), dst_offset,
              dst_stride);
}

inline void cpu_copy_indexed_payload(uint8_t *dst,
                                     const uint8_t *src,
                                     std::size_t item_bytes) {
  auto *dst_words = reinterpret_cast<uint32_t *>(dst);
  const auto *src_words = reinterpret_cast<const uint32_t *>(src);
  if (item_bytes == sizeof(uint32_t)) {
    dst_words[0] = src_words[0];
    return;
  }
  if (item_bytes == sizeof(uint64_t)) {
    dst_words[0] = src_words[0];
    dst_words[1] = src_words[1];
    return;
  }
  if (item_bytes == 4 * sizeof(uint32_t)) {
    dst_words[0] = src_words[0];
    dst_words[1] = src_words[1];
    dst_words[2] = src_words[2];
    dst_words[3] = src_words[3];
    return;
  }
  const std::size_t words = item_bytes / sizeof(uint32_t);
  for (std::size_t word = 0; word < words; ++word) {
    dst_words[word] = src_words[word];
  }
}

inline void cpu_zero_indexed_payload(uint8_t *dst, std::size_t item_bytes) {
  auto *dst_words = reinterpret_cast<uint32_t *>(dst);
  if (item_bytes == sizeof(uint32_t)) {
    dst_words[0] = 0;
    return;
  }
  if (item_bytes == sizeof(uint64_t)) {
    dst_words[0] = 0;
    dst_words[1] = 0;
    return;
  }
  if (item_bytes == 4 * sizeof(uint32_t)) {
    dst_words[0] = 0;
    dst_words[1] = 0;
    dst_words[2] = 0;
    dst_words[3] = 0;
    return;
  }
  const std::size_t words = item_bytes / sizeof(uint32_t);
  for (std::size_t word = 0; word < words; ++word) {
    dst_words[word] = 0;
  }
}

void cpu_indexed_copy_task(void *raw_ctx, int /*thread_id*/, int task_id) {
  auto *ctx = static_cast<CpuIndexedCopyTaskContext *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  if (ctx->scatter) {
    for (std::size_t i = begin; i < end; ++i) {
      const auto index = static_cast<std::size_t>(ctx->indices[i]);
      if (index < ctx->index_bound) {
        cpu_copy_indexed_payload(ctx->dst + index * ctx->item_bytes,
                                 ctx->src + i * ctx->item_bytes,
                                 ctx->item_bytes);
      }
    }
  } else {
    for (std::size_t i = begin; i < end; ++i) {
      const auto index = static_cast<std::size_t>(ctx->indices[i]);
      if (index < ctx->index_bound) {
        cpu_copy_indexed_payload(ctx->dst + i * ctx->item_bytes,
                                 ctx->src + index * ctx->item_bytes,
                                 ctx->item_bytes);
      } else {
        cpu_zero_indexed_payload(ctx->dst + i * ctx->item_bytes,
                                 ctx->item_bytes);
      }
    }
  }
}

void cpu_strided_indexed_copy_task(void *raw_ctx,
                                   int /*thread_id*/,
                                   int task_id) {
  auto *ctx = static_cast<CpuStridedIndexedCopyTaskContext *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  if (ctx->scatter) {
    for (std::size_t i = begin; i < end; ++i) {
      const auto index = static_cast<std::size_t>(ctx->indices[i]);
      if (index < ctx->index_bound) {
        cpu_copy_indexed_payload(
            ctx->dst + ctx->dst_offset + index * ctx->dst_stride,
            ctx->src + ctx->src_offset + i * ctx->src_stride,
            ctx->item_bytes);
      }
    }
  } else {
    for (std::size_t i = begin; i < end; ++i) {
      const auto index = static_cast<std::size_t>(ctx->indices[i]);
      if (index < ctx->index_bound) {
        cpu_copy_indexed_payload(
            ctx->dst + ctx->dst_offset + i * ctx->dst_stride,
            ctx->src + ctx->src_offset + index * ctx->src_stride,
            ctx->item_bytes);
      } else {
        cpu_zero_indexed_payload(
            ctx->dst + ctx->dst_offset + i * ctx->dst_stride,
            ctx->item_bytes);
      }
    }
  }
}

template <typename T>
void cpu_scatter_add_count_task(void *raw_ctx,
                                int /*thread_id*/,
                                int task_id) {
  auto *ctx = static_cast<CpuScatterAddTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  T *local = ctx->partial + ctx->dst_items * static_cast<std::size_t>(tid);
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    const auto index = static_cast<std::size_t>(ctx->indices[i]);
    if (index < ctx->dst_items) {
      local[index] = cpu_reduce_combine(local[index], ctx->src[i], 0);
    }
  }
}

template <typename T>
void cpu_scatter_add_merge_task(void *raw_ctx,
                                int /*thread_id*/,
                                int task_id) {
  auto *ctx = static_cast<CpuScatterAddTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->dst_items * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->dst_items * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    T value{};
    for (int t = 0; t < ctx->num_threads; ++t) {
      value = cpu_reduce_combine(
          value,
          ctx->partial[ctx->dst_items * static_cast<std::size_t>(t) + i], 0);
    }
    ctx->dst[i] = cpu_reduce_combine(ctx->dst[i], value, 0);
  }
}

template <typename T>
void cpu_strided_scatter_add_count_task(void *raw_ctx,
                                        int /*thread_id*/,
                                        int task_id) {
  auto *ctx = static_cast<CpuStridedScatterAddTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  T *local = ctx->partial + ctx->dst_items * static_cast<std::size_t>(tid);
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    const auto index = static_cast<std::size_t>(ctx->indices[i]);
    if (index < ctx->dst_items) {
      const auto *value =
          reinterpret_cast<const T *>(ctx->src + ctx->offset + i * ctx->stride);
      local[index] = cpu_reduce_combine(local[index], *value, 0);
    }
  }
}

template <typename T>
void cpu_strided_scatter_add_merge_task(void *raw_ctx,
                                        int /*thread_id*/,
                                        int task_id) {
  auto *ctx = static_cast<CpuStridedScatterAddTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->dst_items * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->dst_items * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    T value{};
    for (int t = 0; t < ctx->num_threads; ++t) {
      value = cpu_reduce_combine(
          value,
          ctx->partial[ctx->dst_items * static_cast<std::size_t>(t) + i], 0);
    }
    ctx->dst[i] = cpu_reduce_combine(ctx->dst[i], value, 0);
  }
}

template <typename T>
void cpu_strided_scatter_add_io_count_task(void *raw_ctx,
                                           int /*thread_id*/,
                                           int task_id) {
  auto *ctx = static_cast<CpuStridedScatterAddIoTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  T *local = ctx->partial + ctx->dst_items * static_cast<std::size_t>(tid);
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    const auto index = static_cast<std::size_t>(ctx->indices[i]);
    if (index < ctx->dst_items) {
      const auto *value = reinterpret_cast<const T *>(
          ctx->src + ctx->src_offset + i * ctx->src_stride);
      local[index] = cpu_reduce_combine(local[index], *value, 0);
    }
  }
}

template <typename T>
void cpu_strided_scatter_add_io_merge_task(void *raw_ctx,
                                           int /*thread_id*/,
                                           int task_id) {
  auto *ctx = static_cast<CpuStridedScatterAddIoTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->dst_items * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->dst_items * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    T value{};
    for (int t = 0; t < ctx->num_threads; ++t) {
      value = cpu_reduce_combine(
          value,
          ctx->partial[ctx->dst_items * static_cast<std::size_t>(t) + i], 0);
    }
    auto *dst_value =
        reinterpret_cast<T *>(ctx->dst + ctx->dst_offset + i * ctx->dst_stride);
    *dst_value = cpu_reduce_combine(*dst_value, value, 0);
  }
}

template <typename T>
void cpu_packed_scatter_add_io_count_task(void *raw_ctx,
                                          int /*thread_id*/,
                                          int task_id) {
  auto *ctx = static_cast<CpuPackedScatterAddIoTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const auto lanes = static_cast<std::size_t>(ctx->lane_count);
  T *local = ctx->partial +
             ctx->dst_items * lanes * static_cast<std::size_t>(tid);
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    const auto index = static_cast<std::size_t>(ctx->indices[i]);
    if (index < ctx->dst_items) {
      const uint8_t *src_item =
          ctx->src + ctx->src_offset + i * ctx->src_stride;
      T *local_item = local + index * lanes;
      for (std::size_t lane = 0; lane < lanes; ++lane) {
        const auto *value =
            reinterpret_cast<const T *>(src_item + lane * sizeof(T));
        local_item[lane] = cpu_reduce_combine(local_item[lane], *value, 0);
      }
    }
  }
}

template <typename T>
void cpu_packed_scatter_add_io_merge_task(void *raw_ctx,
                                          int /*thread_id*/,
                                          int task_id) {
  auto *ctx = static_cast<CpuPackedScatterAddIoTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const auto lanes = static_cast<std::size_t>(ctx->lane_count);
  const std::size_t total = ctx->dst_items * lanes;
  const std::size_t begin =
      total * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      total * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t scalar_i = begin; scalar_i < end; ++scalar_i) {
    const std::size_t item = scalar_i / lanes;
    const std::size_t lane = scalar_i - item * lanes;
    T value{};
    for (int t = 0; t < ctx->num_threads; ++t) {
      value = cpu_reduce_combine(
          value,
          ctx->partial[(ctx->dst_items * lanes) * static_cast<std::size_t>(t) +
                       scalar_i],
          0);
    }
    auto *dst_value = reinterpret_cast<T *>(
        ctx->dst + ctx->dst_offset + item * ctx->dst_stride +
        lane * sizeof(T));
    *dst_value = cpu_reduce_combine(*dst_value, value, 0);
  }
}

template <typename T>
std::size_t cpu_scatter_add_typed(const T *src_ptr,
                                  const int32_t *indices_ptr,
                                  T *dst_ptr,
                                  std::size_t n,
                                  std::size_t dst_items,
                                  int max_threads,
                                  int target_threads) {
  TI_ERROR_IF(!src_ptr || !dst_ptr,
              "CPU native scatter-add received a null data pointer.");
  const std::size_t workspace_bytes =
      static_cast<std::size_t>(target_threads) * dst_items * sizeof(T);
  if (target_threads > 1 &&
      workspace_bytes <= kCpuPrimitiveMaxLocalWorkspaceBytes) {
    CpuPrimitiveScratchBuffer<T> partial(
        PrimitiveWorkspaceFamily::scatter_add,
        static_cast<std::size_t>(target_threads) * dst_items);
    CpuScatterAddTaskContext<T> ctx;
    ctx.src = src_ptr;
    ctx.indices = indices_ptr;
    ctx.partial = partial.data();
    ctx.dst = dst_ptr;
    ctx.n = n;
    ctx.dst_items = dst_items;
    ctx.num_threads = target_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(target_threads, target_threads, &ctx,
             cpu_scatter_add_count_task<T>);
    pool->run(target_threads, target_threads, &ctx,
             cpu_scatter_add_merge_task<T>);
    return workspace_bytes;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const auto index = static_cast<std::size_t>(indices_ptr[i]);
    if (index < dst_items) {
      dst_ptr[index] = cpu_reduce_combine(dst_ptr[index], src_ptr[i], 0);
    }
  }
  return 0;
}

template <typename T>
std::size_t cpu_scatter_add_strided_typed(const uint8_t *src_ptr,
                                          std::size_t offset,
                                          std::size_t stride,
                                          const int32_t *indices_ptr,
                                          T *dst_ptr,
                                          std::size_t n,
                                          std::size_t dst_items,
                                          int max_threads,
                                          int target_threads) {
  TI_ERROR_IF(!src_ptr || !dst_ptr,
              "CPU native strided scatter-add received a null data pointer.");
  const std::size_t workspace_bytes =
      static_cast<std::size_t>(target_threads) * dst_items * sizeof(T);
  if (target_threads > 1 &&
      workspace_bytes <= kCpuPrimitiveMaxLocalWorkspaceBytes) {
    CpuPrimitiveScratchBuffer<T> partial(
        PrimitiveWorkspaceFamily::scatter_add,
        static_cast<std::size_t>(target_threads) * dst_items);
    CpuStridedScatterAddTaskContext<T> ctx;
    ctx.src = src_ptr;
    ctx.indices = indices_ptr;
    ctx.partial = partial.data();
    ctx.dst = dst_ptr;
    ctx.n = n;
    ctx.dst_items = dst_items;
    ctx.offset = offset;
    ctx.stride = stride;
    ctx.num_threads = target_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(target_threads, target_threads, &ctx,
             cpu_strided_scatter_add_count_task<T>);
    pool->run(target_threads, target_threads, &ctx,
             cpu_strided_scatter_add_merge_task<T>);
    return workspace_bytes;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const auto index = static_cast<std::size_t>(indices_ptr[i]);
    if (index < dst_items) {
      const auto *value =
          reinterpret_cast<const T *>(src_ptr + offset + i * stride);
      dst_ptr[index] = cpu_reduce_combine(dst_ptr[index], *value, 0);
    }
  }
  return 0;
}

template <typename T>
std::size_t cpu_scatter_add_strided_io_typed(const uint8_t *src_ptr,
                                             std::size_t src_offset,
                                             std::size_t src_stride,
                                             const int32_t *indices_ptr,
                                             uint8_t *dst_ptr,
                                             std::size_t dst_offset,
                                             std::size_t dst_stride,
                                             std::size_t n,
                                             std::size_t dst_items,
                                             int max_threads,
                                             int target_threads) {
  TI_ERROR_IF(!src_ptr || !dst_ptr,
              "CPU native strided scatter-add received a null data pointer.");
  const std::size_t workspace_bytes =
      static_cast<std::size_t>(target_threads) * dst_items * sizeof(T);
  if (target_threads > 1 &&
      workspace_bytes <= kCpuPrimitiveMaxLocalWorkspaceBytes) {
    CpuPrimitiveScratchBuffer<T> partial(
        PrimitiveWorkspaceFamily::scatter_add,
        static_cast<std::size_t>(target_threads) * dst_items);
    CpuStridedScatterAddIoTaskContext<T> ctx;
    ctx.src = src_ptr;
    ctx.indices = indices_ptr;
    ctx.partial = partial.data();
    ctx.dst = dst_ptr;
    ctx.n = n;
    ctx.dst_items = dst_items;
    ctx.src_offset = src_offset;
    ctx.src_stride = src_stride;
    ctx.dst_offset = dst_offset;
    ctx.dst_stride = dst_stride;
    ctx.num_threads = target_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(target_threads, target_threads, &ctx,
             cpu_strided_scatter_add_io_count_task<T>);
    pool->run(target_threads, target_threads, &ctx,
             cpu_strided_scatter_add_io_merge_task<T>);
    return workspace_bytes;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const auto index = static_cast<std::size_t>(indices_ptr[i]);
    if (index < dst_items) {
      const auto *value =
          reinterpret_cast<const T *>(src_ptr + src_offset + i * src_stride);
      auto *dst_value = reinterpret_cast<T *>(dst_ptr + dst_offset +
                                              index * dst_stride);
      *dst_value = cpu_reduce_combine(*dst_value, *value, 0);
    }
  }
  return 0;
}

template <typename T>
std::size_t cpu_scatter_add_packed_strided_io_typed(
    const uint8_t *src_ptr,
    std::size_t src_offset,
    std::size_t src_stride,
    const int32_t *indices_ptr,
    uint8_t *dst_ptr,
    std::size_t dst_offset,
    std::size_t dst_stride,
    std::size_t n,
    std::size_t dst_items,
    int lane_count,
    int max_threads,
    int target_threads) {
  TI_ERROR_IF(!src_ptr || !indices_ptr || !dst_ptr,
              "CPU native packed scatter-add received a null data pointer.");
  TI_ERROR_IF(lane_count <= 0,
              "CPU native packed scatter-add received an invalid lane count.");
  const auto lanes = static_cast<std::size_t>(lane_count);
  TI_ERROR_IF(dst_items >
                  std::numeric_limits<std::size_t>::max() / lanes,
              "CPU native packed scatter-add destination scalar count "
              "overflowed.");
  const std::size_t dst_scalars = dst_items * lanes;
  TI_ERROR_IF(static_cast<std::size_t>(target_threads) >
                  std::numeric_limits<std::size_t>::max() / dst_scalars,
              "CPU native packed scatter-add workspace size overflowed.");
  const std::size_t workspace_items =
      static_cast<std::size_t>(target_threads) * dst_scalars;
  TI_ERROR_IF(workspace_items >
                  std::numeric_limits<std::size_t>::max() / sizeof(T),
              "CPU native packed scatter-add workspace byte size overflowed.");
  const std::size_t workspace_bytes = workspace_items * sizeof(T);
  if (target_threads > 1 &&
      workspace_bytes <= kCpuPrimitiveMaxLocalWorkspaceBytes) {
    CpuPrimitiveScratchBuffer<T> partial(
        PrimitiveWorkspaceFamily::scatter_add, workspace_items);
    CpuPackedScatterAddIoTaskContext<T> ctx;
    ctx.src = src_ptr;
    ctx.indices = indices_ptr;
    ctx.partial = partial.data();
    ctx.dst = dst_ptr;
    ctx.n = n;
    ctx.dst_items = dst_items;
    ctx.lane_count = lane_count;
    ctx.src_offset = src_offset;
    ctx.src_stride = src_stride;
    ctx.dst_offset = dst_offset;
    ctx.dst_stride = dst_stride;
    ctx.num_threads = target_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(target_threads, target_threads, &ctx,
             cpu_packed_scatter_add_io_count_task<T>);
    pool->run(target_threads, target_threads, &ctx,
             cpu_packed_scatter_add_io_merge_task<T>);
    return workspace_bytes;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const auto index = static_cast<std::size_t>(indices_ptr[i]);
    if (index < dst_items) {
      const uint8_t *src_item = src_ptr + src_offset + i * src_stride;
      uint8_t *dst_item = dst_ptr + dst_offset + index * dst_stride;
      for (std::size_t lane = 0; lane < lanes; ++lane) {
        const auto *value =
            reinterpret_cast<const T *>(src_item + lane * sizeof(T));
        auto *dst_value =
            reinterpret_cast<T *>(dst_item + lane * sizeof(T));
        *dst_value = cpu_reduce_combine(*dst_value, *value, 0);
      }
    }
  }
  return 0;
}

void cpu_bucket_count_task(void *raw_ctx, int /*thread_id*/, int task_id) {
  auto *ctx = static_cast<CpuBucketCountTaskContext *>(raw_ctx);
  const int tid = task_id;
  int32_t *local = ctx->partial + ctx->num_bins * static_cast<std::size_t>(tid);
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    int32_t key = ctx->keys[i];
    if (key >= 0 && static_cast<std::size_t>(key) < ctx->num_bins) {
      local[key] += 1;
    }
  }
}

template <typename T>
void cpu_bucket_scatter_task(void *raw_ctx, int /*thread_id*/, int task_id) {
  auto *ctx = static_cast<CpuBucketScatterTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  int32_t *local =
      ctx->thread_offsets + ctx->num_bins * static_cast<std::size_t>(tid);
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    int32_t key = ctx->keys[i];
    if (key >= 0 && static_cast<std::size_t>(key) < ctx->num_bins) {
      int32_t pos = local[key]++;
      if (pos >= 0 && static_cast<std::size_t>(pos) < ctx->n) {
        ctx->output[pos] = ctx->values[i];
      }
    }
  }
}

void cpu_bucket_scatter_raw_task(void *raw_ctx,
                                 int /*thread_id*/,
                                 int task_id) {
  auto *ctx = static_cast<CpuBucketScatterRawTaskContext *>(raw_ctx);
  const int tid = task_id;
  int32_t *local =
      ctx->thread_offsets + ctx->num_bins * static_cast<std::size_t>(tid);
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    int32_t key = ctx->keys[i];
    if (key >= 0 && static_cast<std::size_t>(key) < ctx->num_bins) {
      int32_t pos = local[key]++;
      if (pos >= 0 && static_cast<std::size_t>(pos) < ctx->n) {
        std::memcpy(ctx->output + static_cast<std::size_t>(pos) *
                                      ctx->item_bytes,
                    ctx->values + i * ctx->item_bytes, ctx->item_bytes);
      }
    }
  }
}

template <typename T>
std::size_t cpu_bucket_builder_typed(const int32_t *keys_ptr,
                                     const T *values_ptr,
                                     int32_t *offsets_ptr,
                                     T *output_ptr,
                                     std::size_t n,
                                     std::size_t num_bins,
                                     int max_threads) {
  TI_ERROR_IF(!keys_ptr || !values_ptr || !offsets_ptr || !output_ptr,
              "CPU native bucket builder received a null data pointer.");
  std::fill(offsets_ptr, offsets_ptr + num_bins + 1, 0);
  if (n == 0) {
    return 0;
  }

  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const std::size_t parallel_workspace =
      static_cast<std::size_t>(target_threads) * num_bins * sizeof(int32_t) *
      2;
  const bool use_parallel = cpu_use_parallel_aggregation(n, target_threads) &&
                            parallel_workspace <=
                                kCpuPrimitiveMaxLocalWorkspaceBytes;

  if (use_parallel) {
    std::vector<int32_t> partial(
        static_cast<std::size_t>(target_threads) * num_bins, 0);
    CpuBucketCountTaskContext count_ctx;
    count_ctx.keys = keys_ptr;
    count_ctx.partial = partial.data();
    count_ctx.n = n;
    count_ctx.num_bins = num_bins;
    count_ctx.num_threads = target_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(target_threads, target_threads, &count_ctx, cpu_bucket_count_task);

    std::vector<int32_t> thread_offsets(
        static_cast<std::size_t>(target_threads) * num_bins, 0);
    int64_t running = 0;
    offsets_ptr[0] = 0;
    for (std::size_t bin = 0; bin < num_bins; ++bin) {
      int64_t pos = running;
      for (int tid = 0; tid < target_threads; ++tid) {
        const std::size_t idx =
            static_cast<std::size_t>(tid) * num_bins + bin;
        thread_offsets[idx] = static_cast<int32_t>(pos);
        pos += partial[idx];
      }
      running = pos;
      TI_ERROR_IF(running > std::numeric_limits<int32_t>::max(),
                  "CPU native bucket builder valid item count exceeds i32 range.");
      offsets_ptr[bin + 1] = static_cast<int32_t>(running);
    }

    CpuBucketScatterTaskContext<T> scatter_ctx;
    scatter_ctx.keys = keys_ptr;
    scatter_ctx.values = values_ptr;
    scatter_ctx.thread_offsets = thread_offsets.data();
    scatter_ctx.output = output_ptr;
    scatter_ctx.n = n;
    scatter_ctx.num_bins = num_bins;
    scatter_ctx.num_threads = target_threads;
    pool->run(target_threads, target_threads, &scatter_ctx,
             cpu_bucket_scatter_task<T>);
    return parallel_workspace;
  }

  for (std::size_t i = 0; i < n; ++i) {
    int32_t key = keys_ptr[i];
    if (key >= 0 && static_cast<std::size_t>(key) < num_bins) {
      offsets_ptr[static_cast<std::size_t>(key) + 1] += 1;
    }
  }
  int64_t running = 0;
  for (std::size_t bin = 0; bin <= num_bins; ++bin) {
    running += offsets_ptr[bin];
    TI_ERROR_IF(running > std::numeric_limits<int32_t>::max(),
                "CPU native bucket builder valid item count exceeds i32 range.");
    offsets_ptr[bin] = static_cast<int32_t>(running);
  }
  std::vector<int32_t> cursor(offsets_ptr, offsets_ptr + num_bins);
  for (std::size_t i = 0; i < n; ++i) {
    int32_t key = keys_ptr[i];
    if (key >= 0 && static_cast<std::size_t>(key) < num_bins) {
      int32_t pos = cursor[key]++;
      output_ptr[pos] = values_ptr[i];
    }
  }
  return cursor.size() * sizeof(int32_t);
}

std::size_t cpu_bucket_builder_raw(const int32_t *keys_ptr,
                                   const uint8_t *values_ptr,
                                   int32_t *offsets_ptr,
                                   uint8_t *output_ptr,
                                   std::size_t n,
                                   std::size_t num_bins,
                                   std::size_t item_bytes,
                                   int max_threads) {
  TI_ERROR_IF(!keys_ptr || !values_ptr || !offsets_ptr || !output_ptr,
              "CPU native bucket builder received a null data pointer.");
  TI_ERROR_IF(item_bytes == 0,
              "CPU native bucket builder received empty payload items.");
  std::fill(offsets_ptr, offsets_ptr + num_bins + 1, 0);
  if (n == 0) {
    return 0;
  }

  const int chunk_items = 65536;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const std::size_t parallel_workspace =
      static_cast<std::size_t>(target_threads) * num_bins * sizeof(int32_t) *
      2;
  const bool use_parallel = cpu_use_parallel_aggregation(n, target_threads) &&
                            parallel_workspace <=
                                kCpuPrimitiveMaxLocalWorkspaceBytes;

  if (use_parallel) {
    std::vector<int32_t> partial(
        static_cast<std::size_t>(target_threads) * num_bins, 0);
    CpuBucketCountTaskContext count_ctx;
    count_ctx.keys = keys_ptr;
    count_ctx.partial = partial.data();
    count_ctx.n = n;
    count_ctx.num_bins = num_bins;
    count_ctx.num_threads = target_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(target_threads, target_threads, &count_ctx, cpu_bucket_count_task);

    std::vector<int32_t> thread_offsets(
        static_cast<std::size_t>(target_threads) * num_bins, 0);
    int64_t running = 0;
    offsets_ptr[0] = 0;
    for (std::size_t bin = 0; bin < num_bins; ++bin) {
      int64_t pos = running;
      for (int tid = 0; tid < target_threads; ++tid) {
        const std::size_t idx =
            static_cast<std::size_t>(tid) * num_bins + bin;
        thread_offsets[idx] = static_cast<int32_t>(pos);
        pos += partial[idx];
      }
      running = pos;
      TI_ERROR_IF(running > std::numeric_limits<int32_t>::max(),
                  "CPU native bucket builder valid item count exceeds i32 range.");
      offsets_ptr[bin + 1] = static_cast<int32_t>(running);
    }

    CpuBucketScatterRawTaskContext scatter_ctx;
    scatter_ctx.keys = keys_ptr;
    scatter_ctx.values = values_ptr;
    scatter_ctx.thread_offsets = thread_offsets.data();
    scatter_ctx.output = output_ptr;
    scatter_ctx.n = n;
    scatter_ctx.num_bins = num_bins;
    scatter_ctx.item_bytes = item_bytes;
    scatter_ctx.num_threads = target_threads;
    pool->run(target_threads, target_threads, &scatter_ctx,
             cpu_bucket_scatter_raw_task);
    return parallel_workspace;
  }

  for (std::size_t i = 0; i < n; ++i) {
    int32_t key = keys_ptr[i];
    if (key >= 0 && static_cast<std::size_t>(key) < num_bins) {
      offsets_ptr[static_cast<std::size_t>(key) + 1] += 1;
    }
  }
  int64_t running = 0;
  for (std::size_t bin = 0; bin <= num_bins; ++bin) {
    running += offsets_ptr[bin];
    TI_ERROR_IF(running > std::numeric_limits<int32_t>::max(),
                "CPU native bucket builder valid item count exceeds i32 range.");
    offsets_ptr[bin] = static_cast<int32_t>(running);
  }
  std::vector<int32_t> cursor(offsets_ptr, offsets_ptr + num_bins);
  for (std::size_t i = 0; i < n; ++i) {
    int32_t key = keys_ptr[i];
    if (key >= 0 && static_cast<std::size_t>(key) < num_bins) {
      int32_t pos = cursor[key]++;
      std::memcpy(output_ptr + static_cast<std::size_t>(pos) * item_bytes,
                  values_ptr + i * item_bytes, item_bytes);
    }
  }
  return cursor.size() * sizeof(int32_t);
}

template <typename T>
void cpu_grouped_reduce_count_task(void *raw_ctx,
                                   int /*thread_id*/,
                                   int task_id) {
  auto *ctx = static_cast<CpuGroupedReduceTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  T *local =
      ctx->partial + ctx->num_groups * static_cast<std::size_t>(tid);
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    int32_t key = ctx->keys[i];
    if (key >= 0 && static_cast<std::size_t>(key) < ctx->num_groups) {
      local[key] = cpu_reduce_combine(local[key], ctx->values[i], 0);
    }
  }
}

template <typename T>
void cpu_grouped_reduce_merge_task(void *raw_ctx,
                                   int /*thread_id*/,
                                   int task_id) {
  auto *ctx = static_cast<CpuGroupedReduceTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->num_groups * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->num_groups * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t group = begin; group < end; ++group) {
    T value{};
    for (int t = 0; t < ctx->num_threads; ++t) {
      value = cpu_reduce_combine(
          value,
          ctx->partial[ctx->num_groups * static_cast<std::size_t>(t) + group],
          0);
    }
    ctx->output[group] = value;
  }
}

template <typename T>
void cpu_strided_grouped_reduce_count_task(void *raw_ctx,
                                           int /*thread_id*/,
                                           int task_id) {
  auto *ctx = static_cast<CpuStridedGroupedReduceTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  T *local =
      ctx->partial + ctx->num_groups * static_cast<std::size_t>(tid);
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    int32_t key = ctx->keys[i];
    if (key >= 0 && static_cast<std::size_t>(key) < ctx->num_groups) {
      const auto *value =
          reinterpret_cast<const T *>(ctx->values + ctx->offset +
                                      i * ctx->stride);
      local[key] = cpu_reduce_combine(local[key], *value, 0);
    }
  }
}

template <typename T>
void cpu_strided_grouped_reduce_merge_task(void *raw_ctx,
                                           int /*thread_id*/,
                                           int task_id) {
  auto *ctx = static_cast<CpuStridedGroupedReduceTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->num_groups * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->num_groups * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t group = begin; group < end; ++group) {
    T value{};
    for (int t = 0; t < ctx->num_threads; ++t) {
      value = cpu_reduce_combine(
          value,
          ctx->partial[ctx->num_groups * static_cast<std::size_t>(t) + group],
          0);
    }
    ctx->output[group] = value;
  }
}

template <typename T>
void cpu_strided_grouped_reduce_io_count_task(void *raw_ctx,
                                              int /*thread_id*/,
                                              int task_id) {
  auto *ctx = static_cast<CpuStridedGroupedReduceIoTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  T *local =
      ctx->partial + ctx->num_groups * static_cast<std::size_t>(tid);
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    const int32_t key = *reinterpret_cast<const int32_t *>(
        ctx->keys + ctx->keys_offset + i * ctx->keys_stride);
    if (key >= 0 && static_cast<std::size_t>(key) < ctx->num_groups) {
      const auto *value = reinterpret_cast<const T *>(
          ctx->values + ctx->values_offset + i * ctx->values_stride);
      local[key] = cpu_reduce_combine(local[key], *value, 0);
    }
  }
}

template <typename T>
void cpu_strided_grouped_reduce_io_merge_task(void *raw_ctx,
                                              int /*thread_id*/,
                                              int task_id) {
  auto *ctx = static_cast<CpuStridedGroupedReduceIoTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->num_groups * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->num_groups * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t group = begin; group < end; ++group) {
    T value{};
    for (int t = 0; t < ctx->num_threads; ++t) {
      value = cpu_reduce_combine(
          value,
          ctx->partial[ctx->num_groups * static_cast<std::size_t>(t) + group],
          0);
    }
    auto *out_value = reinterpret_cast<T *>(
        ctx->output + ctx->output_offset + group * ctx->output_stride);
    *out_value = value;
  }
}

template <typename T>
std::size_t cpu_grouped_reduce_typed(const int32_t *keys_ptr,
                                     const T *values_ptr,
                                     T *output_ptr,
                                     std::size_t n,
                                     std::size_t num_groups,
                                     int max_threads,
                                     int target_threads) {
  TI_ERROR_IF(!keys_ptr || !values_ptr || !output_ptr,
              "CPU native grouped reduce received a null data pointer.");
  std::fill(output_ptr, output_ptr + num_groups, T{});
  if (n == 0) {
    return 0;
  }
  const std::size_t workspace_bytes =
      static_cast<std::size_t>(target_threads) * num_groups * sizeof(T);
  if (target_threads > 1 &&
      workspace_bytes <= kCpuPrimitiveMaxLocalWorkspaceBytes) {
    CpuPrimitiveScratchBuffer<T> partial(
        PrimitiveWorkspaceFamily::grouped,
        static_cast<std::size_t>(target_threads) * num_groups);
    CpuGroupedReduceTaskContext<T> ctx;
    ctx.keys = keys_ptr;
    ctx.values = values_ptr;
    ctx.partial = partial.data();
    ctx.output = output_ptr;
    ctx.n = n;
    ctx.num_groups = num_groups;
    ctx.num_threads = target_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(target_threads, target_threads, &ctx,
             cpu_grouped_reduce_count_task<T>);
    pool->run(target_threads, target_threads, &ctx,
             cpu_grouped_reduce_merge_task<T>);
    return workspace_bytes;
  }

  for (std::size_t i = 0; i < n; ++i) {
    int32_t key = keys_ptr[i];
    if (key >= 0 && static_cast<std::size_t>(key) < num_groups) {
      output_ptr[key] = cpu_reduce_combine(output_ptr[key], values_ptr[i], 0);
    }
  }
  return 0;
}

template <typename T>
std::size_t cpu_grouped_reduce_strided_typed(const int32_t *keys_ptr,
                                             const uint8_t *values_ptr,
                                             std::size_t offset,
                                             std::size_t stride,
                                             T *output_ptr,
                                             std::size_t n,
                                             std::size_t num_groups,
                                             int max_threads,
                                             int target_threads) {
  TI_ERROR_IF(!keys_ptr || !values_ptr || !output_ptr,
              "CPU native strided grouped reduce received a null data pointer.");
  std::fill(output_ptr, output_ptr + num_groups, T{});
  if (n == 0) {
    return 0;
  }
  const std::size_t workspace_bytes =
      static_cast<std::size_t>(target_threads) * num_groups * sizeof(T);
  if (target_threads > 1 &&
      workspace_bytes <= kCpuPrimitiveMaxLocalWorkspaceBytes) {
    CpuPrimitiveScratchBuffer<T> partial(
        PrimitiveWorkspaceFamily::grouped,
        static_cast<std::size_t>(target_threads) * num_groups);
    CpuStridedGroupedReduceTaskContext<T> ctx;
    ctx.keys = keys_ptr;
    ctx.values = values_ptr;
    ctx.partial = partial.data();
    ctx.output = output_ptr;
    ctx.n = n;
    ctx.num_groups = num_groups;
    ctx.offset = offset;
    ctx.stride = stride;
    ctx.num_threads = target_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(target_threads, target_threads, &ctx,
             cpu_strided_grouped_reduce_count_task<T>);
    pool->run(target_threads, target_threads, &ctx,
             cpu_strided_grouped_reduce_merge_task<T>);
    return workspace_bytes;
  }

  for (std::size_t i = 0; i < n; ++i) {
    int32_t key = keys_ptr[i];
    if (key >= 0 && static_cast<std::size_t>(key) < num_groups) {
      const auto *value =
          reinterpret_cast<const T *>(values_ptr + offset + i * stride);
      output_ptr[key] = cpu_reduce_combine(output_ptr[key], *value, 0);
    }
  }
  return 0;
}

template <typename T>
std::size_t cpu_grouped_reduce_strided_io_typed(const uint8_t *keys_ptr,
                                                std::size_t keys_offset,
                                                std::size_t keys_stride,
                                                const uint8_t *values_ptr,
                                                std::size_t values_offset,
                                                std::size_t values_stride,
                                                uint8_t *output_ptr,
                                                std::size_t output_offset,
                                                std::size_t output_stride,
                                                std::size_t n,
                                                std::size_t num_groups,
                                                int max_threads,
                                                int target_threads) {
  TI_ERROR_IF(!keys_ptr || !values_ptr || !output_ptr,
              "CPU native strided grouped reduce received a null data pointer.");
  for (std::size_t group = 0; group < num_groups; ++group) {
    auto *out_value = reinterpret_cast<T *>(
        output_ptr + output_offset + group * output_stride);
    *out_value = T{};
  }
  if (n == 0) {
    return 0;
  }
  const std::size_t workspace_bytes =
      static_cast<std::size_t>(target_threads) * num_groups * sizeof(T);
  if (target_threads > 1 &&
      workspace_bytes <= kCpuPrimitiveMaxLocalWorkspaceBytes) {
    CpuPrimitiveScratchBuffer<T> partial(
        PrimitiveWorkspaceFamily::grouped,
        static_cast<std::size_t>(target_threads) * num_groups);
    CpuStridedGroupedReduceIoTaskContext<T> ctx;
    ctx.keys = keys_ptr;
    ctx.values = values_ptr;
    ctx.partial = partial.data();
    ctx.output = output_ptr;
    ctx.n = n;
    ctx.num_groups = num_groups;
    ctx.keys_offset = keys_offset;
    ctx.keys_stride = keys_stride;
    ctx.values_offset = values_offset;
    ctx.values_stride = values_stride;
    ctx.output_offset = output_offset;
    ctx.output_stride = output_stride;
    ctx.num_threads = target_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(target_threads, target_threads, &ctx,
             cpu_strided_grouped_reduce_io_count_task<T>);
    pool->run(target_threads, target_threads, &ctx,
             cpu_strided_grouped_reduce_io_merge_task<T>);
    return workspace_bytes;
  }

  for (std::size_t i = 0; i < n; ++i) {
    const int32_t key = *reinterpret_cast<const int32_t *>(
        keys_ptr + keys_offset + i * keys_stride);
    if (key >= 0 && static_cast<std::size_t>(key) < num_groups) {
      const auto *value = reinterpret_cast<const T *>(
          values_ptr + values_offset + i * values_stride);
      auto *out_value = reinterpret_cast<T *>(
          output_ptr + output_offset + static_cast<std::size_t>(key) *
                                         output_stride);
      *out_value = cpu_reduce_combine(*out_value, *value, 0);
    }
  }
  return 0;
}

std::size_t primitive_value_type_size(int value_type) {
  if (value_type >= 0 && value_type <= 2) {
    return sizeof(uint32_t);
  }
  if (value_type >= 3 && value_type <= 5) {
    return sizeof(uint64_t);
  }
  return 0;
}

#ifdef TI_WITH_CUDA
std::size_t cuda_scatter_add_contiguous(void *src,
                                        void *indices,
                                        void *dst,
                                        std::size_t n,
                                        std::size_t dst_n,
                                        int value_type) {
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA scatter-add currently supports at most INT_MAX source "
              "items.");
  TI_ERROR_IF(dst_n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA scatter-add currently supports destination sizes up to "
              "INT_MAX items.");
  void *stream = nullptr;
  const std::size_t value_size = primitive_value_type_size(value_type);
  return cuda::driver_scatter_add_strided(
      src, indices, dst, static_cast<int>(n), static_cast<int>(dst_n),
      static_cast<cuda::CudaTransformValueType>(value_type), 0, value_size, 0,
      sizeof(std::int32_t), 0, value_size, stream);
}
#endif

template <typename T>
std::size_t cpu_compact_dense_field_typed(const uint8_t *values_ptr,
                                          std::size_t values_stride,
                                          const uint8_t *flags_ptr,
                                          std::size_t flags_stride,
                                          uint8_t *output_ptr,
                                          std::size_t output_stride,
                                          std::size_t n,
                                          int max_threads,
                                          int target_threads,
                                          bool use_parallel,
                                          std::size_t *workspace_bytes) {
  if (workspace_bytes) {
    *workspace_bytes = 0;
  }
  if (use_parallel) {
    std::vector<std::size_t> offsets(
        static_cast<std::size_t>(target_threads) + 1, 0);
    CpuCompactCountTaskContext count_ctx;
    count_ctx.flags = flags_ptr;
    count_ctx.flags_stride = flags_stride;
    count_ctx.counts = offsets.data();
    count_ctx.n = n;
    count_ctx.num_threads = target_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(target_threads, target_threads, &count_ctx, cpu_compact_count_task);
    std::size_t running = 0;
    for (int tid = 0; tid < target_threads; ++tid) {
      const std::size_t count = offsets[tid];
      offsets[tid] = running;
      running += count;
    }
    offsets[target_threads] = running;
    CpuCompactScatterTaskContext<T> scatter_ctx;
    scatter_ctx.values = values_ptr;
    scatter_ctx.values_stride = values_stride;
    scatter_ctx.flags = flags_ptr;
    scatter_ctx.flags_stride = flags_stride;
    scatter_ctx.output = output_ptr;
    scatter_ctx.output_stride = output_stride;
    scatter_ctx.offsets = offsets.data();
    scatter_ctx.n = n;
    scatter_ctx.num_threads = target_threads;
    pool->run(target_threads, target_threads, &scatter_ctx,
             cpu_compact_scatter_task<T>);
    if (workspace_bytes) {
      *workspace_bytes = offsets.size() * sizeof(std::size_t);
    }
    return running;
  }

  std::size_t written = 0;
  if (values_stride == sizeof(T) && flags_stride == sizeof(int32_t) &&
      output_stride == sizeof(T)) {
    const auto *values = reinterpret_cast<const T *>(values_ptr);
    const auto *flags = reinterpret_cast<const int32_t *>(flags_ptr);
    auto *output = reinterpret_cast<T *>(output_ptr);
    for (std::size_t i = 0; i < n; ++i) {
      if (flags[i] != 0) {
        output[written++] = values[i];
      }
    }
    return written;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const int32_t flag =
        *reinterpret_cast<const int32_t *>(flags_ptr + i * flags_stride);
    if (flag != 0) {
      *reinterpret_cast<T *>(output_ptr + written * output_stride) =
          *reinterpret_cast<const T *>(values_ptr + i * values_stride);
      written++;
    }
  }
  return written;
}

void check_indexed_copy_dense_field_request(Program *program,
                                            const char *backend,
                                            SNode *src,
                                            Ndarray *indices,
                                            SNode *dst,
                                            int value_type,
                                            std::size_t src_n,
                                            std::size_t dst_n,
                                            bool scatter) {
  TI_ERROR_IF(!program || !src || !indices || !dst,
              "{} dense field indexed-copy received a null argument.",
              backend);
  TI_ERROR_IF(indices->shape.size() != 1,
              "{} dense field indexed-copy expects 1D indices.", backend);
  TI_ERROR_IF(indices->get_element_size() != sizeof(int32_t),
              "{} dense field indexed-copy expects i32 indices.", backend);
  const std::size_t item_bytes = primitive_value_type_size(value_type);
  TI_ERROR_IF(item_bytes == 0 || item_bytes % sizeof(uint32_t) != 0,
              "{} dense field indexed-copy item size must be a positive "
              "uint32-word multiple.",
              backend);
  if (scatter) {
    TI_ERROR_IF(src_n != indices->get_nelement(),
                "{} dense field scatter expects source and indices sizes to "
                "match.",
                backend);
  } else {
    TI_ERROR_IF(indices->get_nelement() != dst_n,
                "{} dense field gather expects indices and destination sizes "
                "to match.",
                backend);
  }
  const std::size_t src_stride = program->get_dense_field_stride(src, item_bytes);
  const std::size_t dst_stride = program->get_dense_field_stride(dst, item_bytes);
  TI_ERROR_IF(src_stride < item_bytes || dst_stride < item_bytes,
              "{} dense field indexed-copy received an invalid field stride.",
              backend);
  TI_ERROR_IF(src_stride % sizeof(uint32_t) != 0 ||
                  dst_stride % sizeof(uint32_t) != 0,
              "{} dense field indexed-copy stride must be uint32-word aligned.",
              backend);
}

void check_indexed_copy_dense_field_indices_field_request(Program *program,
                                                          const char *backend,
                                                          SNode *src,
                                                          SNode *indices,
                                                          SNode *dst,
                                                          int value_type,
                                                          std::size_t src_n,
                                                          std::size_t indices_n,
                                                          std::size_t dst_n,
                                                          bool scatter) {
  TI_ERROR_IF(!program || !src || !indices || !dst,
              "{} dense field indexed-copy received a null argument.",
              backend);
  const std::size_t item_bytes = primitive_value_type_size(value_type);
  TI_ERROR_IF(item_bytes == 0 || item_bytes % sizeof(uint32_t) != 0,
              "{} dense field indexed-copy item size must be a positive "
              "uint32-word multiple.",
              backend);
  if (scatter) {
    TI_ERROR_IF(src_n != indices_n,
                "{} dense field scatter expects source and indices sizes to "
                "match.",
                backend);
  } else {
    TI_ERROR_IF(indices_n != dst_n,
                "{} dense field gather expects indices and destination sizes "
                "to match.",
                backend);
  }
  const std::size_t src_stride = program->get_dense_field_stride(src, item_bytes);
  const std::size_t index_stride =
      program->get_dense_field_stride(indices, sizeof(int32_t));
  const std::size_t dst_stride = program->get_dense_field_stride(dst, item_bytes);
  TI_ERROR_IF(src_stride < item_bytes || dst_stride < item_bytes ||
                  index_stride < sizeof(int32_t),
              "{} dense field indexed-copy received an invalid field stride.",
              backend);
  TI_ERROR_IF(index_stride != sizeof(int32_t),
              "{} dense field indexed-copy currently requires contiguous i32 "
              "indices when indices are stored in a field.",
              backend);
  TI_ERROR_IF(src_stride % sizeof(uint32_t) != 0 ||
                  dst_stride % sizeof(uint32_t) != 0,
              "{} dense field indexed-copy stride must be uint32-word aligned.",
              backend);
}

void check_reduce_member_request(const char *backend,
                                 Ndarray *values,
                                 Ndarray *output,
                                 int value_type,
                                 std::size_t offset,
                                 std::size_t stride,
                                 int op) {
  TI_ERROR_IF(!values || !output,
              "{} strided reduce received a null ndarray.", backend);
  TI_ERROR_IF(values->shape.size() != 1 || output->shape.size() != 1,
              "{} strided reduce expects 1D ndarrays.", backend);
  TI_ERROR_IF(values->get_nelement() == 0,
              "{} strided reduce expects at least one input item.", backend);
  TI_ERROR_IF(output->get_nelement() < 1,
              "{} strided reduce output must contain at least one item.",
              backend);
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "{} strided reduce received an unsupported value type.",
              backend);
  TI_ERROR_IF(output->get_element_size() != value_size,
              "{} strided reduce output dtype does not match value type.",
              backend);
  TI_ERROR_IF(op < 0 || op > 2,
              "{} strided reduce supports only sum/min/max operations.",
              backend);
  TI_ERROR_IF(stride < value_size,
              "{} strided reduce source stride is smaller than value size.",
              backend);
  TI_ERROR_IF(offset % value_size != 0 || stride % value_size != 0,
              "{} strided reduce source offset/stride must align to value "
              "size.",
              backend);
  const std::size_t n = values->get_nelement();
  const std::size_t src_bytes = n * values->get_element_size();
  TI_ERROR_IF(src_bytes < value_size,
              "{} strided reduce source buffer is smaller than value size.",
              backend);
  TI_ERROR_IF(offset > src_bytes - value_size,
              "{} strided reduce source offset is out of bounds.", backend);
  const std::size_t last = offset + (n - 1) * stride + value_size;
  TI_ERROR_IF(last > src_bytes,
              "{} strided reduce source range is out of bounds.", backend);
}

void check_reduce_strided_request(const char *backend,
                                  Ndarray *values,
                                  Ndarray *output,
                                  int value_type,
                                  std::size_t values_offset,
                                  std::size_t values_stride,
                                  std::size_t output_offset,
                                  std::size_t output_stride,
                                  int op) {
  TI_ERROR_IF(!values || !output,
              "{} strided reduce received a null ndarray.", backend);
  TI_ERROR_IF(values->shape.size() != 1 || output->shape.size() != 1,
              "{} strided reduce expects 1D ndarrays.", backend);
  TI_ERROR_IF(values->get_nelement() == 0,
              "{} strided reduce expects at least one input item.", backend);
  TI_ERROR_IF(output->get_nelement() < 1,
              "{} strided reduce output must contain at least one item.",
              backend);
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "{} strided reduce received an unsupported value type.",
              backend);
  TI_ERROR_IF(op < 0 || op > 2,
              "{} strided reduce supports only sum/min/max operations.",
              backend);
  auto check_range = [&](const char *role, Ndarray *arr,
                         std::size_t logical_items, std::size_t offset,
                         std::size_t stride) {
    TI_ERROR_IF(stride < value_size,
                "{} strided reduce {} stride is smaller than value size.",
                backend, role);
    TI_ERROR_IF(offset % value_size != 0 || stride % value_size != 0,
                "{} strided reduce {} offset/stride must align to value "
                "size.",
                backend, role);
    const std::size_t bytes = arr->get_nelement() * arr->get_element_size();
    TI_ERROR_IF(bytes < value_size,
                "{} strided reduce {} buffer is smaller than value size.",
                backend, role);
    TI_ERROR_IF(offset > bytes - value_size,
                "{} strided reduce {} offset is out of bounds.", backend,
                role);
    const std::size_t last =
        offset + (logical_items - 1) * stride + value_size;
    TI_ERROR_IF(last > bytes,
                "{} strided reduce {} range is out of bounds.", backend,
                role);
  };
  check_range("source", values, values->get_nelement(), values_offset,
              values_stride);
  check_range("destination", output, 1, output_offset, output_stride);
}

void check_scan_member_request(const char *backend,
                               Ndarray *data,
                               int value_type,
                               std::size_t offset,
                               std::size_t stride) {
  TI_ERROR_IF(!data, "{} strided scan received a null ndarray.", backend);
  TI_ERROR_IF(data->shape.size() != 1, "{} strided scan expects a 1D ndarray.",
              backend);
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "{} strided scan received an unsupported value type.", backend);
  TI_ERROR_IF(stride < value_size,
              "{} strided scan source stride is smaller than value size.",
              backend);
  TI_ERROR_IF(offset % value_size != 0 || stride % value_size != 0,
              "{} strided scan source offset/stride must align to value size.",
              backend);
  const std::size_t n = data->get_nelement();
  if (n == 0) {
    return;
  }
  const std::size_t src_bytes = n * data->get_element_size();
  TI_ERROR_IF(src_bytes < value_size,
              "{} strided scan source buffer is smaller than value size.",
              backend);
  TI_ERROR_IF(offset > src_bytes - value_size,
              "{} strided scan source offset is out of bounds.", backend);
  const std::size_t last = offset + (n - 1) * stride + value_size;
  TI_ERROR_IF(last > src_bytes,
              "{} strided scan source range is out of bounds.", backend);
}

void check_scatter_add_member_request(const char *backend,
                                      Ndarray *src,
                                      Ndarray *indices,
                                      Ndarray *dst,
                                      int value_type,
                                      std::size_t offset,
                                      std::size_t stride) {
  TI_ERROR_IF(!src || !indices || !dst,
              "{} strided scatter-add received a null ndarray.", backend);
  TI_ERROR_IF(src->shape.size() != 1 || indices->shape.size() != 1 ||
                  dst->shape.size() != 1,
              "{} strided scatter-add expects 1D ndarrays.", backend);
  TI_ERROR_IF(src->get_nelement() != indices->get_nelement(),
              "{} strided scatter-add source and indices sizes differ.",
              backend);
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "{} strided scatter-add received an unsupported value type.",
              backend);
  TI_ERROR_IF(dst->get_element_size() != value_size ||
                  indices->get_element_size() != sizeof(int32_t),
              "{} strided scatter-add destination dtype or i32 index size "
              "mismatch.",
              backend);
  TI_ERROR_IF(stride < value_size,
              "{} strided scatter-add source stride is smaller than value "
              "size.",
              backend);
  TI_ERROR_IF(offset % value_size != 0 || stride % value_size != 0,
              "{} strided scatter-add source offset/stride must align to "
              "value size.",
              backend);
  const std::size_t n = src->get_nelement();
  if (n == 0) {
    return;
  }
  const std::size_t src_bytes = n * src->get_element_size();
  TI_ERROR_IF(src_bytes < value_size,
              "{} strided scatter-add source buffer is smaller than value "
              "size.",
              backend);
  TI_ERROR_IF(offset > src_bytes - value_size,
              "{} strided scatter-add source offset is out of bounds.",
              backend);
  const std::size_t last = offset + (n - 1) * stride + value_size;
  TI_ERROR_IF(last > src_bytes,
              "{} strided scatter-add source range is out of bounds.",
              backend);
}

void check_grouped_reduce_member_request(const char *backend,
                                         Ndarray *keys,
                                         Ndarray *values,
                                         Ndarray *output,
                                         int value_type,
                                         std::size_t offset,
                                         std::size_t stride,
                                         int op) {
  TI_ERROR_IF(!keys || !values || !output,
              "{} strided grouped reduce received a null ndarray.", backend);
  TI_ERROR_IF(keys->shape.size() != 1 || values->shape.size() != 1 ||
                  output->shape.size() != 1,
              "{} strided grouped reduce expects 1D ndarrays.", backend);
  TI_ERROR_IF(keys->get_nelement() != values->get_nelement(),
              "{} strided grouped reduce keys and values sizes differ.",
              backend);
  TI_ERROR_IF(output->get_nelement() == 0,
              "{} strided grouped reduce output must contain at least one "
              "group.",
              backend);
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "{} strided grouped reduce received an unsupported value type.",
              backend);
  TI_ERROR_IF(keys->get_element_size() != sizeof(int32_t) ||
                  output->get_element_size() != value_size,
              "{} strided grouped reduce output dtype or i32 key size "
              "mismatch.",
              backend);
  TI_ERROR_IF(op != 0,
              "{} strided grouped reduce currently supports only sum.",
              backend);
  TI_ERROR_IF(stride < value_size,
              "{} strided grouped reduce source stride is smaller than value "
              "size.",
              backend);
  TI_ERROR_IF(offset % value_size != 0 || stride % value_size != 0,
              "{} strided grouped reduce source offset/stride must align to "
              "value size.",
              backend);
  const std::size_t n = values->get_nelement();
  if (n == 0) {
    return;
  }
  const std::size_t values_bytes = n * values->get_element_size();
  TI_ERROR_IF(values_bytes < value_size,
              "{} strided grouped reduce source buffer is smaller than value "
              "size.",
              backend);
  TI_ERROR_IF(offset > values_bytes - value_size,
              "{} strided grouped reduce source offset is out of bounds.",
              backend);
  const std::size_t last = offset + (n - 1) * stride + value_size;
  TI_ERROR_IF(last > values_bytes,
              "{} strided grouped reduce source range is out of bounds.",
              backend);
}

void check_strided_range(const char *backend,
                         const char *role,
                         Ndarray *arr,
                         std::size_t logical_items,
                         std::size_t value_size,
                         std::size_t offset,
                         std::size_t stride) {
  TI_ERROR_IF(stride < value_size,
              "{} strided {} stride is smaller than value size.", backend,
              role);
  TI_ERROR_IF(offset % value_size != 0 || stride % value_size != 0,
              "{} strided {} offset/stride must align to value size.", backend,
              role);
  if (logical_items == 0) {
    return;
  }
  const std::size_t bytes = arr->get_nelement() * arr->get_element_size();
  TI_ERROR_IF(bytes < value_size,
              "{} strided {} buffer is smaller than value size.", backend,
              role);
  TI_ERROR_IF(offset > bytes - value_size,
              "{} strided {} offset is out of bounds.", backend, role);
  const std::size_t last = offset + (logical_items - 1) * stride + value_size;
  TI_ERROR_IF(last > bytes, "{} strided {} range is out of bounds.", backend,
              role);
}

void check_scatter_add_strided_request(const char *backend,
                                       Ndarray *src,
                                       Ndarray *indices,
                                       Ndarray *dst,
                                       int value_type,
                                       std::size_t src_offset,
                                       std::size_t src_stride,
                                       std::size_t dst_offset,
                                       std::size_t dst_stride) {
  TI_ERROR_IF(!src || !indices || !dst,
              "{} strided scatter-add received a null ndarray.", backend);
  TI_ERROR_IF(src->shape.size() != 1 || indices->shape.size() != 1 ||
                  dst->shape.size() != 1,
              "{} strided scatter-add expects 1D ndarrays.", backend);
  TI_ERROR_IF(src->get_nelement() != indices->get_nelement(),
              "{} strided scatter-add source and indices sizes differ.",
              backend);
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "{} strided scatter-add received an unsupported value type.",
              backend);
  TI_ERROR_IF(indices->get_element_size() != sizeof(int32_t),
              "{} strided scatter-add expects i32 indices.", backend);
  const std::size_t n = src->get_nelement();
  const std::size_t dst_items = dst->get_nelement();
  check_strided_range(backend, "scatter-add source", src, n, value_size,
                      src_offset, src_stride);
  check_strided_range(backend, "scatter-add destination", dst, dst_items,
                      value_size, dst_offset, dst_stride);
}

void check_grouped_reduce_strided_keys_request(const char *backend,
                                               Ndarray *keys,
                                               Ndarray *values,
                                               Ndarray *output,
                                               int value_type,
                                               std::size_t keys_offset,
                                               std::size_t keys_stride,
                                               std::size_t values_offset,
                                               std::size_t values_stride,
                                               std::size_t output_offset,
                                               std::size_t output_stride,
                                               int op) {
  TI_ERROR_IF(!keys || !values || !output,
              "{} strided grouped reduce received a null ndarray.", backend);
  TI_ERROR_IF(keys->shape.size() != 1 || values->shape.size() != 1 ||
                  output->shape.size() != 1,
              "{} strided grouped reduce expects 1D ndarrays.", backend);
  TI_ERROR_IF(keys->get_nelement() != values->get_nelement(),
              "{} strided grouped reduce keys and values sizes differ.",
              backend);
  TI_ERROR_IF(output->get_nelement() == 0,
              "{} strided grouped reduce output must contain at least one "
              "group.",
              backend);
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "{} strided grouped reduce received an unsupported value type.",
              backend);
  TI_ERROR_IF(op != 0,
              "{} strided grouped reduce currently supports only sum.",
              backend);
  check_strided_range(backend, "grouped reduce keys", keys,
                      keys->get_nelement(), sizeof(int32_t), keys_offset,
                      keys_stride);
  check_strided_range(backend, "grouped reduce source", values,
                      values->get_nelement(), value_size, values_offset,
                      values_stride);
  check_strided_range(backend, "grouped reduce output", output,
                      output->get_nelement(), value_size, output_offset,
                      output_stride);
}


std::size_t histogram_bin_type_size(int bin_type) {
  if (bin_type == 0) {
    return sizeof(int32_t);
  }
  if (bin_type == 4) {
    return sizeof(int64_t);
  }
  return 0;
}

template <typename ValueT>
bool cpu_histogram_valid_bin(ValueT bin, std::size_t num_bins) {
  if constexpr (std::is_unsigned_v<ValueT>) {
    return static_cast<std::size_t>(bin) < num_bins;
  } else {
    return bin >= 0 && static_cast<std::size_t>(bin) < num_bins;
  }
}

template <typename ValueT, typename CounterT>
void cpu_histogram_task(void *raw_ctx, int /*thread_id*/, int task_id) {
  auto *ctx = static_cast<CpuHistogramTaskContext<ValueT, CounterT> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  CounterT *local =
      ctx->partial + static_cast<std::size_t>(tid) * ctx->num_bins;
  for (std::size_t i = begin; i < end; ++i) {
    ValueT bin = ctx->values[i];
    if (cpu_histogram_valid_bin(bin, ctx->num_bins)) {
      local[static_cast<std::size_t>(bin)] += 1;
    }
  }
}

template <typename ValueT, typename CounterT>
std::size_t cpu_histogram_typed(const ValueT *values_ptr,
                                CounterT *bins_ptr,
                                std::size_t n,
                                std::size_t num_bins,
                                int max_threads,
                                int target_threads,
                                bool use_parallel) {
  TI_ERROR_IF(!values_ptr || !bins_ptr,
              "CPU native histogram received a null data pointer.");
  std::fill(bins_ptr, bins_ptr + num_bins, CounterT{});

  if (use_parallel) {
    const int num_threads = target_threads;
    std::vector<CounterT> partial(
        static_cast<std::size_t>(num_threads) * num_bins, CounterT{});
    CpuHistogramTaskContext<ValueT, CounterT> ctx;
    ctx.values = values_ptr;
    ctx.partial = partial.data();
    ctx.n = n;
    ctx.num_bins = num_bins;
    ctx.num_threads = num_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(num_threads, num_threads, &ctx,
             cpu_histogram_task<ValueT, CounterT>);
    for (std::size_t bin = 0; bin < num_bins; ++bin) {
      CounterT total{};
      for (int tid = 0; tid < num_threads; ++tid) {
        total += partial[static_cast<std::size_t>(tid) * num_bins + bin];
      }
      bins_ptr[bin] = total;
    }
    return partial.size() * sizeof(CounterT);
  }

  for (std::size_t i = 0; i < n; ++i) {
    ValueT bin = values_ptr[i];
    if (cpu_histogram_valid_bin(bin, num_bins)) {
      bins_ptr[static_cast<std::size_t>(bin)] += 1;
    }
  }
  return 0;
}

template <typename ValueT, typename CounterT>
std::size_t cpu_histogram_strided_typed(const uint8_t *values_ptr,
                                        std::size_t value_stride,
                                        uint8_t *bins_ptr,
                                        std::size_t bin_stride,
                                        std::size_t n,
                                        std::size_t num_bins) {
  TI_ERROR_IF(!values_ptr || !bins_ptr,
              "CPU native strided histogram received a null data pointer.");
  for (std::size_t bin = 0; bin < num_bins; ++bin) {
    *reinterpret_cast<CounterT *>(bins_ptr + bin * bin_stride) = CounterT{};
  }
  for (std::size_t i = 0; i < n; ++i) {
    ValueT bin =
        *reinterpret_cast<const ValueT *>(values_ptr + i * value_stride);
    if (cpu_histogram_valid_bin(bin, num_bins)) {
      auto *counter =
          reinterpret_cast<CounterT *>(bins_ptr +
                                       static_cast<std::size_t>(bin) *
                                           bin_stride);
      *counter += 1;
    }
  }
  return 0;
}

template <typename T>
std::size_t cpu_reduce_typed(const T *values_ptr,
                             T *output_ptr,
                             int op,
                             std::size_t n,
                             int max_threads,
                             int target_threads,
                             bool use_parallel) {
  TI_ERROR_IF(!values_ptr || !output_ptr,
              "CPU native reduce received a null data pointer.");

  T result = cpu_reduce_identity<T>(op);
  if (use_parallel) {
    const int num_threads = target_threads;
    std::vector<T> partial(num_threads);
    CpuReduceTaskContext<T> ctx;
    ctx.values = values_ptr;
    ctx.partial = partial.data();
    ctx.n = n;
    ctx.num_threads = num_threads;
    ctx.op = op;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(num_threads, num_threads, &ctx, cpu_reduce_task<T>);
    for (int tid = 0; tid < num_threads; ++tid) {
      result = cpu_reduce_combine(result, partial[tid], op);
    }
    output_ptr[0] = result;
    return partial.size() * sizeof(T);
  }

  if (op == 0) {
    output_ptr[0] = cpu_reduce_sum_contiguous_range(values_ptr, 0, n);
    return 0;
  }
  for (std::size_t i = 0; i < n; ++i) {
    result = cpu_reduce_combine(result, values_ptr[i], op);
  }
  output_ptr[0] = result;
  return 0;
}

template <typename T>
std::size_t cpu_reduce_strided_typed(const uint8_t *values_ptr,
                                     T *output_ptr,
                                     int op,
                                     std::size_t n,
                                     std::size_t offset,
                                     std::size_t stride,
                                     int max_threads,
                                     int target_threads,
                                     bool use_parallel) {
  TI_ERROR_IF(!values_ptr || !output_ptr,
              "CPU native strided reduce received a null data pointer.");

  T result = cpu_reduce_identity<T>(op);
  if (use_parallel) {
    const int num_threads = target_threads;
    std::vector<T> partial(num_threads);
    CpuStridedReduceTaskContext<T> ctx;
    ctx.values = values_ptr;
    ctx.partial = partial.data();
    ctx.n = n;
    ctx.offset = offset;
    ctx.stride = stride;
    ctx.num_threads = num_threads;
    ctx.op = op;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(num_threads, num_threads, &ctx, cpu_strided_reduce_task<T>);
    for (int tid = 0; tid < num_threads; ++tid) {
      result = cpu_reduce_combine(result, partial[tid], op);
    }
    output_ptr[0] = result;
    return partial.size() * sizeof(T);
  }

  if (op == 0) {
    output_ptr[0] =
        cpu_reduce_sum_strided_range<T>(values_ptr, offset, stride, 0, n);
    return 0;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const auto *value =
        reinterpret_cast<const T *>(values_ptr + offset + i * stride);
    result = cpu_reduce_combine(result, *value, op);
  }
  output_ptr[0] = result;
  return 0;
}

template <typename T>
bool cpu_check_predicate(T value, int check_op, int lower, int upper) {
  switch (check_op) {
    case 0:
      return value != T{};
    case 1:
      return value == T{};
    case 2:
      if constexpr (std::is_floating_point_v<T>) {
        return std::isnan(value);
      }
      return false;
    case 3:
      if constexpr (std::is_floating_point_v<T>) {
        return std::isinf(value);
      }
      return false;
    case 4:
      if constexpr (std::is_floating_point_v<T>) {
        return !std::isfinite(value);
      }
      return false;
    case 5:
      if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
        return value < static_cast<T>(lower) || value >= static_cast<T>(upper);
      } else if constexpr (std::is_integral_v<T>) {
        if (lower < 0) {
          return value >= static_cast<T>(upper);
        }
        return value < static_cast<T>(lower) || value >= static_cast<T>(upper);
      }
      return false;
  }
  return false;
}

template <typename T>
struct CpuCheckCountTaskContext {
  const T *values{nullptr};
  int32_t *partial{nullptr};
  std::size_t n{0};
  int check_op{0};
  int lower{0};
  int upper{0};
  int num_threads{1};
};

template <typename T>
void cpu_check_count_task(void *raw_ctx, int /*thread_id*/, int task_id) {
  auto *ctx = static_cast<CpuCheckCountTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  int32_t local = 0;
  for (std::size_t i = begin; i < end; ++i) {
    local += cpu_check_predicate(ctx->values[i], ctx->check_op, ctx->lower,
                                 ctx->upper)
                 ? 1
                 : 0;
  }
  ctx->partial[tid] = local;
}

template <typename T>
std::size_t cpu_check_count_typed(const T *values_ptr,
                                  int32_t *output_ptr,
                                  int check_op,
                                  int lower,
                                  int upper,
                                  std::size_t n,
                                  int max_threads,
                                  int target_threads,
                                  bool use_parallel) {
  TI_ERROR_IF(!values_ptr || !output_ptr,
              "CPU native check_count received a null data pointer.");
  if (use_parallel) {
    const int num_threads = target_threads;
    std::vector<int32_t> partial(num_threads, 0);
    CpuCheckCountTaskContext<T> ctx;
    ctx.values = values_ptr;
    ctx.partial = partial.data();
    ctx.n = n;
    ctx.check_op = check_op;
    ctx.lower = lower;
    ctx.upper = upper;
    ctx.num_threads = num_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(num_threads, num_threads, &ctx, cpu_check_count_task<T>);
    int32_t total = 0;
    for (int32_t value : partial) {
      total += value;
    }
    output_ptr[0] = total;
    return partial.size() * sizeof(int32_t);
  }
  int32_t total = 0;
  for (std::size_t i = 0; i < n; ++i) {
    total += cpu_check_predicate(values_ptr[i], check_op, lower, upper) ? 1 : 0;
  }
  output_ptr[0] = total;
  return 0;
}

template <typename T>
struct CpuStridedCheckCountTaskContext {
  const uint8_t *values{nullptr};
  std::size_t offset{0};
  std::size_t stride{sizeof(T)};
  int32_t *partial{nullptr};
  std::size_t n{0};
  int check_op{0};
  int lower{0};
  int upper{0};
  int num_threads{1};
};

template <typename T>
void cpu_strided_check_count_task(void *raw_ctx,
                                  int /*thread_id*/,
                                  int task_id) {
  auto *ctx = static_cast<CpuStridedCheckCountTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  int32_t local = 0;
  for (std::size_t i = begin; i < end; ++i) {
    const auto value = *reinterpret_cast<const T *>(
        ctx->values + ctx->offset + i * ctx->stride);
    local += cpu_check_predicate(value, ctx->check_op, ctx->lower, ctx->upper)
                 ? 1
                 : 0;
  }
  ctx->partial[tid] = local;
}

template <typename T>
std::size_t cpu_check_count_strided_typed(const uint8_t *values_ptr,
                                          std::size_t offset,
                                          std::size_t stride,
                                          int32_t *output_ptr,
                                          int check_op,
                                          int lower,
                                          int upper,
                                          std::size_t n,
                                          int max_threads,
                                          int target_threads,
                                          bool use_parallel) {
  TI_ERROR_IF(!values_ptr || !output_ptr,
              "CPU native strided check_count received a null data pointer.");
  if (stride == sizeof(T) && offset == 0) {
    return cpu_check_count_typed(reinterpret_cast<const T *>(values_ptr),
                                 output_ptr, check_op, lower, upper, n,
                                 max_threads, target_threads, use_parallel);
  }
  if (use_parallel) {
    const int num_threads = target_threads;
    std::vector<int32_t> partial(num_threads, 0);
    CpuStridedCheckCountTaskContext<T> ctx;
    ctx.values = values_ptr;
    ctx.offset = offset;
    ctx.stride = stride;
    ctx.partial = partial.data();
    ctx.n = n;
    ctx.check_op = check_op;
    ctx.lower = lower;
    ctx.upper = upper;
    ctx.num_threads = num_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(num_threads, num_threads, &ctx, cpu_strided_check_count_task<T>);
    int32_t total = 0;
    for (int32_t value : partial) {
      total += value;
    }
    output_ptr[0] = total;
    return partial.size() * sizeof(int32_t);
  }
  int32_t total = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const auto value =
        *reinterpret_cast<const T *>(values_ptr + offset + i * stride);
    total += cpu_check_predicate(value, check_op, lower, upper) ? 1 : 0;
  }
  output_ptr[0] = total;
  return 0;
}

template <typename T>
T cpu_metric_abs(T value) {
  if constexpr (std::is_floating_point_v<T>) {
    if (std::isnan(value)) {
      return std::numeric_limits<T>::infinity();
    }
    return value < T{} ? -value : value;
  } else if constexpr (std::is_signed_v<T>) {
    return value < T{} ? -value : value;
  } else {
    return value;
  }
}

template <typename T>
T cpu_metric_value(const T *values, const T *other, std::size_t i, int op) {
  switch (op) {
    case 0:
      return cpu_metric_abs(values[i]);
    case 1:
      return cpu_metric_abs(values[i] - other[i]);
  }
  return T{};
}

template <typename T>
struct CpuMetricReduceTaskContext {
  const T *values{nullptr};
  const T *other{nullptr};
  T *partial{nullptr};
  std::size_t n{0};
  int metric_op{0};
  int num_threads{1};
};

template <typename T>
void cpu_metric_reduce_task(void *raw_ctx, int /*thread_id*/, int task_id) {
  auto *ctx = static_cast<CpuMetricReduceTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  T local{};
  for (std::size_t i = begin; i < end; ++i) {
    local = std::max(local,
                     cpu_metric_value(ctx->values, ctx->other, i,
                                      ctx->metric_op));
  }
  ctx->partial[tid] = local;
}

template <typename T>
std::size_t cpu_metric_reduce_typed(const T *values_ptr,
                                    const T *other_ptr,
                                    T *output_ptr,
                                    int metric_op,
                                    std::size_t n,
                                    int max_threads,
                                    int target_threads,
                                    bool use_parallel) {
  TI_ERROR_IF(!values_ptr || !output_ptr,
              "CPU native metric_reduce received a null data pointer.");
  TI_ERROR_IF(metric_op == 1 && !other_ptr,
              "CPU native max_abs_delta received a null rhs pointer.");
  if (use_parallel) {
    const int num_threads = target_threads;
    std::vector<T> partial(num_threads, T{});
    CpuMetricReduceTaskContext<T> ctx;
    ctx.values = values_ptr;
    ctx.other = other_ptr;
    ctx.partial = partial.data();
    ctx.n = n;
    ctx.metric_op = metric_op;
    ctx.num_threads = num_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(num_threads, num_threads, &ctx, cpu_metric_reduce_task<T>);
    T result{};
    for (T value : partial) {
      result = std::max(result, value);
    }
    output_ptr[0] = result;
    return partial.size() * sizeof(T);
  }
  T result{};
  for (std::size_t i = 0; i < n; ++i) {
    result = std::max(result,
                      cpu_metric_value(values_ptr, other_ptr, i, metric_op));
  }
  output_ptr[0] = result;
  return 0;
}

template <typename T>
T cpu_metric_strided_value(const uint8_t *values,
                           const uint8_t *other,
                           std::size_t values_offset,
                           std::size_t values_stride,
                           std::size_t other_offset,
                           std::size_t other_stride,
                           std::size_t i,
                           int op) {
  const T value =
      *reinterpret_cast<const T *>(values + values_offset + i * values_stride);
  switch (op) {
    case 0:
      return cpu_metric_abs(value);
    case 1:
      return cpu_metric_abs(
          value - *reinterpret_cast<const T *>(other + other_offset +
                                               i * other_stride));
  }
  return T{};
}

template <typename T>
struct CpuStridedMetricReduceTaskContext {
  const uint8_t *values{nullptr};
  const uint8_t *other{nullptr};
  T *partial{nullptr};
  std::size_t n{0};
  std::size_t values_offset{0};
  std::size_t values_stride{sizeof(T)};
  std::size_t other_offset{0};
  std::size_t other_stride{sizeof(T)};
  int metric_op{0};
  int num_threads{1};
};

template <typename T>
void cpu_strided_metric_reduce_task(void *raw_ctx,
                                    int /*thread_id*/,
                                    int task_id) {
  auto *ctx = static_cast<CpuStridedMetricReduceTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  T local{};
  for (std::size_t i = begin; i < end; ++i) {
    local = std::max(local,
                     cpu_metric_strided_value<T>(
                         ctx->values, ctx->other, ctx->values_offset,
                         ctx->values_stride, ctx->other_offset,
                         ctx->other_stride, i, ctx->metric_op));
  }
  ctx->partial[tid] = local;
}

template <typename T>
std::size_t cpu_metric_reduce_strided_typed(const uint8_t *values_ptr,
                                            const uint8_t *other_ptr,
                                            T *output_ptr,
                                            int metric_op,
                                            std::size_t n,
                                            std::size_t values_offset,
                                            std::size_t values_stride,
                                            std::size_t other_offset,
                                            std::size_t other_stride,
                                            int max_threads,
                                            int target_threads,
                                            bool use_parallel) {
  TI_ERROR_IF(!values_ptr || !output_ptr,
              "CPU native strided metric_reduce received a null data pointer.");
  TI_ERROR_IF(metric_op == 1 && !other_ptr,
              "CPU native strided max_abs_delta received a null rhs pointer.");
  if (!other_ptr) {
    other_ptr = values_ptr;
    other_offset = values_offset;
    other_stride = values_stride;
  }
  if (values_offset == 0 && values_stride == sizeof(T) && other_offset == 0 &&
      other_stride == sizeof(T)) {
    return cpu_metric_reduce_typed(reinterpret_cast<const T *>(values_ptr),
                                   reinterpret_cast<const T *>(other_ptr),
                                   output_ptr, metric_op, n, max_threads,
                                   target_threads, use_parallel);
  }
  if (use_parallel) {
    const int num_threads = target_threads;
    std::vector<T> partial(num_threads, T{});
    CpuStridedMetricReduceTaskContext<T> ctx;
    ctx.values = values_ptr;
    ctx.other = other_ptr;
    ctx.partial = partial.data();
    ctx.n = n;
    ctx.values_offset = values_offset;
    ctx.values_stride = values_stride;
    ctx.other_offset = other_offset;
    ctx.other_stride = other_stride;
    ctx.metric_op = metric_op;
    ctx.num_threads = num_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(num_threads, num_threads, &ctx,
             cpu_strided_metric_reduce_task<T>);
    T result{};
    for (T value : partial) {
      result = std::max(result, value);
    }
    output_ptr[0] = result;
    return partial.size() * sizeof(T);
  }
  T result{};
  for (std::size_t i = 0; i < n; ++i) {
    result = std::max(result,
                      cpu_metric_strided_value<T>(
                          values_ptr, other_ptr, values_offset, values_stride,
                          other_offset, other_stride, i, metric_op));
  }
  output_ptr[0] = result;
  return 0;
}

template <typename T>
std::size_t cpu_scan_typed(T *data_ptr, std::size_t n) {
  TI_ERROR_IF(!data_ptr, "CPU native scan received a null data pointer.");
  T prefix{};
  for (std::size_t i = 0; i < n; ++i) {
    prefix += data_ptr[i];
    data_ptr[i] = prefix;
  }
  return 0;
}

template <typename T>
std::size_t cpu_reverse_scan_typed(T *data_ptr, std::size_t n) {
  TI_ERROR_IF(!data_ptr, "CPU native reverse scan received a null data pointer.");
  T suffix{};
  for (std::size_t i = n; i-- > 0;) {
    suffix += data_ptr[i];
    data_ptr[i] = suffix;
  }
  return 0;
}

template <typename T>
std::size_t cpu_scan_strided_typed(uint8_t *data_ptr,
                                   std::size_t n,
                                   std::size_t offset,
                                   std::size_t stride) {
  TI_ERROR_IF(!data_ptr, "CPU native strided scan received a null data pointer.");
  T prefix{};
  for (std::size_t i = 0; i < n; ++i) {
    auto *value = reinterpret_cast<T *>(data_ptr + offset + i * stride);
    prefix += *value;
    *value = prefix;
  }
  return 0;
}

template <typename T>
std::size_t cpu_reverse_scan_strided_typed(uint8_t *data_ptr,
                                           std::size_t n,
                                           std::size_t offset,
                                           std::size_t stride) {
  TI_ERROR_IF(!data_ptr,
              "CPU native reverse strided scan received a null data pointer.");
  T suffix{};
  for (std::size_t i = n; i-- > 0;) {
    auto *value = reinterpret_cast<T *>(data_ptr + offset + i * stride);
    suffix += *value;
    *value = suffix;
  }
  return 0;
}

constexpr char kNativeDenseSimpleLayoutMessage[] =
    "Native dense field path currently supports only root.place "
    "and root.dense.place layouts.";

bool native_dense_linear_root_child_supported(SNode *root_child) {
  return root_child &&
         (root_child->type == SNodeType::place ||
          root_child->type == SNodeType::dense);
}

std::size_t root_child_offset(SNode *root_child) {
  if (!root_child || !root_child->parent ||
      root_child->parent->type != SNodeType::root) {
    throw std::runtime_error(
        "Native dense field path expects a root child SNode.");
  }
  if (!native_dense_linear_root_child_supported(root_child)) {
    throw std::runtime_error(kNativeDenseSimpleLayoutMessage);
  }
  SNode *root = root_child->parent;
  const int child_id = root->child_id(root_child);
  for (int i = 0; i < child_id; ++i) {
    SNode *child = root->ch[i].get();
    if (!native_dense_linear_root_child_supported(child)) {
      throw std::runtime_error(kNativeDenseSimpleLayoutMessage);
    }
  }
  // The struct compiler may insert alignment padding between root children,
  // especially when 32-bit and 64-bit fields share a tree.  Summing the
  // preceding payload sizes therefore points before the real child storage.
  // The compiled SNode layout is the single source of truth for both kernel
  // addressing and native bulk operations.
  return root_child->offset_bytes_in_parent_cell;
}

uint8_t *map_cpu_dense_field(Program *program,
                             SNode *snode,
                             int value_type,
                             std::size_t n,
                             const char *op_name,
                             std::size_t *stride) {
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "{} received an unsupported dense field value type.", op_name);
  DevicePtr ptr = program->get_dense_field_device_ptr(snode);
  const std::size_t field_stride =
      program->get_dense_field_stride(snode, value_size);
  TI_ERROR_IF(field_stride < value_size,
              "{} received an invalid dense field stride.", op_name);
  if (stride) {
    *stride = field_stride;
  }
  const std::size_t span = n == 0 ? value_size : (n - 1) * field_stride + value_size;
  void *mapped = nullptr;
#ifdef TI_WITH_LLVM
  auto *cpu_device = dynamic_cast<cpu::CpuDevice *>(ptr.device);
  TI_ERROR_IF(!cpu_device, "{} expected CPU field storage.", op_name);
  RhiResult res = cpu_device->map_range_for_cpu_native(ptr, span, &mapped);
#else
  RhiResult res = RhiResult::invalid_usage;
#endif
  TI_ERROR_IF(res != RhiResult::success || !mapped,
              "{} failed to map CPU dense field storage.", op_name);
  return reinterpret_cast<uint8_t *>(mapped);
}

std::size_t dense_field_packed_scalar_items(std::size_t n,
                                            int lane_count,
                                            const char *op_name) {
  TI_ERROR_IF(lane_count <= 0,
              "{} received an invalid packed dense field lane count.",
              op_name);
  const auto lanes = static_cast<std::size_t>(lane_count);
  TI_ERROR_IF(n > std::numeric_limits<std::size_t>::max() / lanes,
              "{} received an oversized packed dense field request.",
              op_name);
  return n * lanes;
}

std::size_t dense_field_packed_bytes(int value_type,
                                     std::size_t n,
                                     int lane_count,
                                     const char *op_name) {
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "{} received an unsupported packed dense field value type.",
              op_name);
  const std::size_t scalar_items =
      dense_field_packed_scalar_items(n, lane_count, op_name);
  TI_ERROR_IF(scalar_items >
                  std::numeric_limits<std::size_t>::max() / value_size,
              "{} received an oversized packed dense field byte range.",
              op_name);
  return scalar_items * value_size;
}

std::size_t check_dense_field_packed_stride(Program *program,
                                            SNode *snode,
                                            int value_type,
                                            int lane_count,
                                            const char *op_name) {
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "{} received an unsupported packed dense field value type.",
              op_name);
  const std::size_t expected_stride =
      dense_field_packed_scalar_items(1, lane_count, op_name) * value_size;
  const std::size_t stride = program->get_dense_field_stride(snode, value_size);
  TI_ERROR_IF(stride != expected_stride,
              "{} expects a packed contiguous dense MatrixField layout.",
              op_name);
  return stride;
}

uint8_t *map_cpu_dense_field_packed(Program *program,
                                    SNode *snode,
                                    int value_type,
                                    std::size_t n,
                                    int lane_count,
                                    const char *op_name) {
  const std::size_t bytes =
      dense_field_packed_bytes(value_type, n, lane_count, op_name);
  check_dense_field_packed_stride(program, snode, value_type, lane_count,
                                  op_name);
  DevicePtr ptr = program->get_dense_field_device_ptr(snode);
  void *mapped = nullptr;
#ifdef TI_WITH_LLVM
  auto *cpu_device = dynamic_cast<cpu::CpuDevice *>(ptr.device);
  TI_ERROR_IF(!cpu_device, "{} expected CPU field storage.", op_name);
  RhiResult res = cpu_device->map_range_for_cpu_native(ptr, bytes, &mapped);
#else
  RhiResult res = RhiResult::invalid_usage;
#endif
  TI_ERROR_IF(res != RhiResult::success || !mapped,
              "{} failed to map CPU packed dense field storage.", op_name);
  return reinterpret_cast<uint8_t *>(mapped);
}

}  // namespace

DevicePtr Program::get_dense_field_device_ptr(SNode *snode) {
  TI_ERROR_IF(!snode, "Native dense field path received a null field.");
  TI_ERROR_IF(snode->type != SNodeType::place,
              "Native dense field path expects a place SNode.");
  SNode *parent = snode->parent;
  TI_ERROR_IF(!parent, "Native dense field path expects a placed field.");
  SNode *root_child = nullptr;
  std::size_t leaf_offset = 0;
  if (parent->type == SNodeType::root) {
    root_child = snode;
  } else {
    if (parent->type != SNodeType::dense || !parent->parent ||
        parent->parent->type != SNodeType::root) {
      throw std::runtime_error(kNativeDenseSimpleLayoutMessage);
    }
    root_child = parent;
    leaf_offset = snode->offset_bytes_in_parent_cell;
  }
  const int tree_id = root_child->parent->get_snode_tree_id();
  DevicePtr root_ptr = get_snode_tree_device_ptr(tree_id);
  const std::size_t root_offset =
      compile_config().arch == Arch::vulkan
          ? get_field_in_tree_offset(tree_id, root_child)
          : root_child_offset(root_child);
  return root_ptr.get_ptr(root_offset + leaf_offset);
}

std::size_t Program::get_dense_field_stride(SNode *snode,
                                            std::size_t value_size) {
  TI_ERROR_IF(!snode || !snode->parent,
              "Native dense field path received a null field.");
  if (snode->parent->type == SNodeType::dense) {
    return snode->parent->cell_size_bytes;
  }
  return value_size;
}

Program::Program(Arch desired_arch)
    : snode_rw_accessors_bank_(this),
      lifetime_token_(std::make_shared<ProgramLifetimeToken>(this)),
      runtime_completion_domain_(allocate_runtime_resource_domain()),
      runtime_fault_domain_(std::make_shared<RuntimeFaultDomain>(
          desired_arch, runtime_completion_domain_)),
      runtime_trace_(runtime_fault_domain_->statistics(),
                     runtime_completion_domain_),
      dense_field_staging_resources_(allocate_runtime_resource_domain()),
      argpack_resources_(allocate_runtime_resource_domain()),
      ndarray_resources_(allocate_runtime_resource_domain()),
      texture_resources_(allocate_runtime_resource_domain()),
      external_dense_storage_resources_(
          allocate_runtime_resource_domain()) {
  TI_TRACE("Program initializing...");

  ordinary_launch_attribution_.enabled =
      get_environ_config("TI_DEBUG_ORDINARY_LAUNCH_ATTRIBUTION", 0) != 0;
  ordinary_owned_ndarray_fast_path_enabled_ =
      get_environ_config("TI_DEBUG_ORDINARY_NDARRAY_LEGACY_PATH", 0) == 0;
  ordinary_snode_guard_elision_enabled_ =
      get_environ_config("TI_DEBUG_ORDINARY_FORCE_SNODE_GUARD", 0) == 0;

#ifdef TI_WITH_CUDA
  if (desired_arch == Arch::cuda) {
    runtime_completion_cuda_event_pool_ =
        std::make_shared<RuntimeCompletionCudaEventPool>(
            runtime_fault_domain_,
            kRuntimeCompletionCudaEventCacheCapacity);
  }
#endif

  auto [staging_result, staging_handle] =
      dense_field_staging_resources_.emplace(
          kDenseFieldStagingResourceKind);
  TI_ERROR_IF(staging_result != DenseFieldStagingRegistry::Result::kSuccess,
              "Unable to register dense-field staging cache");
  auto [staging_lease_result, staging_lease] =
      dense_field_staging_resources_.acquire(staging_handle);
  TI_ERROR_IF(
      staging_lease_result != DenseFieldStagingRegistry::Result::kSuccess,
      "Unable to acquire dense-field staging cache");
  dense_field_staging_handle_ = staging_handle;
  dense_field_staging_lease_ = std::move(staging_lease);

  // For performance considerations and correctness of QuantFloatType
  // operations, we force floating-point operations to flush to zero on all
  // backends (including CPUs).
#if defined(_M_X64) || defined(__x86_64)
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
#endif  // defined(_M_X64) || defined(__x86_64)
#if defined(__arm64__) || defined(__aarch64__)
  // Enforce flush to zero on arm64 CPUs
  // https://developer.arm.com/documentation/100403/0201/register-descriptions/advanced-simd-and-floating-point-registers/aarch64-register-descriptions/fpcr--floating-point-control-register?lang=en
  std::uint64_t fpcr;
  __asm__ __volatile__("");
  __asm__ __volatile__("MRS %0, FPCR" : "=r"(fpcr));
  __asm__ __volatile__("");
  __asm__ __volatile__("MSR FPCR, %0"
                       :
                       : "ri"(fpcr | (1 << 24)));  // Bit 24 is FZ
  __asm__ __volatile__("");
#endif  // defined(__arm64__) || defined(__aarch64__)
  auto &config = compile_config_;
  config = default_compile_config;
  config.arch = desired_arch;
  config.fit();

  profiler = make_profiler(config.arch, config.kernel_profiler);
  if (arch_uses_llvm(config.arch)) {
#ifdef TI_WITH_LLVM
    if (config.arch != Arch::dx12) {
      program_impl_ = std::make_unique<LlvmProgramImpl>(config, profiler.get());
    } else {
      // NOTE: use Dx12ProgramImpl to avoid using LlvmRuntimeExecutor for dx12.
#ifdef TI_WITH_DX12
      TI_ASSERT(directx12::is_dx12_api_available());
      program_impl_ = std::make_unique<Dx12ProgramImpl>(config);
#else
      TI_ERROR("This taichi is not compiled with DX12");
#endif
    }
#else
    TI_ERROR("This taichi is not compiled with LLVM");
#endif
  } else if (config.arch == Arch::metal) {
#ifdef TI_WITH_METAL
    TI_ASSERT(metal::is_metal_api_available());
    program_impl_ = std::make_unique<MetalProgramImpl>(config);
#else
    TI_ERROR("This taichi is not compiled with Metal")
#endif
  } else if (config.arch == Arch::vulkan) {
#ifdef TI_WITH_VULKAN
    TI_ASSERT(vulkan::is_vulkan_api_available());
    program_impl_ = std::make_unique<VulkanProgramImpl>(config);
#else
    TI_ERROR("This taichi is not compiled with Vulkan")
#endif
  } else if (config.arch == Arch::dx11) {
#ifdef TI_WITH_DX11
    TI_ASSERT(directx11::is_dx_api_available());
    program_impl_ = std::make_unique<Dx11ProgramImpl>(config);
#else
    TI_ERROR("This taichi is not compiled with DX11");
#endif
  } else if (config.arch == Arch::opengl) {
#ifdef TI_WITH_OPENGL
    TI_ASSERT(opengl::initialize_opengl(false));
    program_impl_ = std::make_unique<OpenglProgramImpl>(config);
#else
    TI_ERROR("This taichi is not compiled with OpenGL");
#endif
  } else if (config.arch == Arch::gles) {
#ifdef TI_WITH_OPENGL
    TI_ASSERT(opengl::initialize_opengl(true));
    program_impl_ = std::make_unique<OpenglProgramImpl>(config);
#else
    TI_ERROR("This taichi is not compiled with OpenGL");
#endif
  } else {
    TI_NOT_IMPLEMENTED
  }

  // program_impl_ should be set in the if-else branch above
  TI_ASSERT(program_impl_);

  // Phase 1c-D: propagate the user's vulkan_sparse_experimental opt-in to
  // the process-global extension table BEFORE any is_extension_supported()
  // query (the very next block already calls it for Extension::assertion).
  // Replaced for each Program; OR'd with legacy TI_VULKAN_SPARSE=1.
  set_vulkan_sparse_experimental(config.vulkan_sparse_experimental);
  // §13 (2026-05-02): default for vulkan_sparse_experimental is now true.
  // Emit a one-shot informational warning whenever the experimental sparse
  // path is exercised on Arch::vulkan, so users can correlate any unexpected
  // behaviour with this opt-in path. Skipped on cpu/cuda (which never read
  // the flag) and skipped when the user explicitly disables it.
  if (config.arch == Arch::vulkan && config.vulkan_sparse_experimental) {
    static bool sparse_warn_emitted = false;
    if (!sparse_warn_emitted) {
      sparse_warn_emitted = true;
      TI_WARN(
          "Vulkan sparse SNode support is experimental and enabled by default "
          "as of taichi-forge 0.3.x; pass ti.init(vulkan_sparse_experimental="
          "False) to disable if you observe regressions vs. cuda/cpu.");
    }
  }
  // G9.1: same propagation pattern for quant_array / bit_struct on Vulkan.
  set_vulkan_quant_experimental(config.vulkan_quant_experimental);

  Device *compute_device = nullptr;
  compute_device = program_impl_->get_compute_device();
  // Must have handled all the arch fallback logic by this point.
  TI_ASSERT_INFO(num_instances_ == 0, "Only one instance at a time");
  total_compilation_time_ = 0;
  SNode::counter = 0;

  result_buffer = nullptr;
  finalized_ = false;

  if (!is_extension_supported(config.arch, Extension::assertion)) {
    if (config.check_out_of_bound) {
      TI_WARN("Out-of-bound access checking is not supported on arch={}",
              arch_name(config.arch));
      config.check_out_of_bound = false;
    }
  }

  Timelines::get_instance().set_enabled(config.timeline);

  initialize_runtime_backend_telemetry_baseline();

  // Install process-wide backend reporting only after every fallible Program
  // construction step and the single-instance check have succeeded. A failed
  // second Program must never overwrite the live Program's reporter.
  attach_runtime_fault_reporter();
  num_instances_ += 1;

  TI_TRACE("Program ({}) arch={} initialized.", fmt::ptr(this),
           arch_name(config.arch));
}

TypeFactory &Program::get_type_factory() {
  TI_WARN(
      "Program::get_type_factory() will be deprecated, Please use "
      "TypeFactory::get_instance()");
  return TypeFactory::get_instance();
}

Function *Program::create_function(const FunctionKey &func_key) {
  TI_TRACE("Creating function {}...", func_key.get_full_name());
  functions_.emplace_back(std::make_unique<Function>(this, func_key));
  TI_ASSERT(function_map_.count(func_key) == 0);
  function_map_[func_key] = functions_.back().get();
  return functions_.back().get();
}

namespace {

constexpr int kDefaultFullSimplifyGlobalIterCap = 1;

CompileConfig make_effective_kernel_compile_config(
    const CompileConfig &base_config,
    const Kernel &kernel_def) {
  CompileConfig effective_config = base_config;

  // D2: per-kernel opt_level is represented as a compile_tier override on the
  // C++ Kernel. Normalize it before cache lookup so single-kernel and
  // compile_kernels batch paths share the same IR/codegen/cache behavior.
  const auto &override = kernel_def.get_compile_tier_override();
  if (override.has_value()) {
    effective_config.compile_tier = *override;
  }
  if (effective_config.compile_tier != "fast" &&
      effective_config.compile_tier != "balanced" &&
      effective_config.compile_tier != "full") {
    TI_ERROR("compile_tier must be one of fast, balanced, full; got {}",
             effective_config.compile_tier);
  }

  // D2: "full" should mean the global IR passes may run to fixed point.
  // Preserve explicit advanced tuning: only rewrite the cap when it still has
  // the default balanced value.
  if (effective_config.compile_tier == "full" &&
      effective_config.full_simplify_global_iter_cap ==
          kDefaultFullSimplifyGlobalIterCap) {
    effective_config.full_simplify_global_iter_cap = 0;
  }

  return effective_config;
}

}  // namespace

const CompiledKernelData &Program::compile_kernel(
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &caps,
    const Kernel &kernel_def) {
  auto start_t = Time::get_time();
  TI_AUTO_PROF;
  auto &mgr = program_impl_->get_kernel_compilation_manager();
  if (kernel_def.ir == nullptr) {
    auto handle = mgr.instantiate_relocatable_execution_handle(
        kernel_def.get_cached_kernel_key(),
        snapshot_snode_tree_dependencies(
            kernel_def.snode_tree_dependencies()));
    TI_ERROR_IF(
        handle == nullptr,
        "Cannot compile a kernel whose SNodeTree dependency has been "
        "destroyed; rebuild the kernel/Graph.");
    total_compilation_time_ += Time::get_time() - start_t;
    return handle->compiled();
  }
  const auto effective_config =
      make_effective_kernel_compile_config(compile_config, kernel_def);
  const auto &ckd = mgr.load_or_compile(effective_config, caps, kernel_def);
  kernel_def.set_snode_tree_dependencies(ckd.snode_tree_ids());
  total_compilation_time_ += Time::get_time() - start_t;
  return ckd;
}

std::shared_ptr<KernelExecutionHandle>
Program::compile_kernel_execution_handle(
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &caps,
    const Kernel &kernel_def) {
  auto start_t = Time::get_time();
  TI_AUTO_PROF;
  auto &mgr = program_impl_->get_kernel_compilation_manager();
  if (kernel_def.ir == nullptr) {
    auto handle = mgr.instantiate_relocatable_execution_handle(
        kernel_def.get_cached_kernel_key(),
        snapshot_snode_tree_dependencies(
            kernel_def.snode_tree_dependencies()));
    TI_ERROR_IF(
        handle == nullptr,
        "Cannot compile a kernel whose SNodeTree dependency has been "
        "destroyed; rebuild the kernel/Graph.");
    total_compilation_time_ += Time::get_time() - start_t;
    return handle;
  }
  const auto effective_config =
      make_effective_kernel_compile_config(compile_config, kernel_def);
  auto handle = mgr.load_or_compile_execution_handle(
      effective_config, caps, kernel_def);
  kernel_def.set_snode_tree_dependencies(handle->compiled().snode_tree_ids());
  total_compilation_time_ += Time::get_time() - start_t;
  return handle;
}

const CompiledKernelData *Program::find_cached_kernel(
    const CompileConfig &compile_config,
    const std::string &kernel_key,
    const Kernel &kernel_def) {
  if (kernel_def.ir == nullptr) {
    auto handle = program_impl_->get_kernel_compilation_manager()
                      .instantiate_relocatable_execution_handle(
                          kernel_key,
                          snapshot_snode_tree_dependencies(
                              kernel_def.snode_tree_dependencies()));
    return handle == nullptr ? nullptr : &handle->compiled();
  }
  auto &mgr = program_impl_->get_kernel_compilation_manager();
  const auto *compiled = mgr.find_cached_kernel(
      kernel_key, kernel_def, compile_config.arch,
      compile_config.offline_cache);
  if (compiled != nullptr) {
    kernel_def.set_snode_tree_dependencies(compiled->snode_tree_ids());
  }
  return compiled;
}

std::shared_ptr<KernelExecutionHandle>
Program::find_cached_kernel_execution_handle(
    const CompileConfig &compile_config,
    const std::string &kernel_key,
    const Kernel &kernel_def) {
  if (kernel_def.ir == nullptr) {
    return program_impl_->get_kernel_compilation_manager()
        .instantiate_relocatable_execution_handle(
            kernel_key,
            snapshot_snode_tree_dependencies(
                kernel_def.snode_tree_dependencies()));
  }
  auto &mgr = program_impl_->get_kernel_compilation_manager();
  auto handle = mgr.find_cached_execution_handle(
      kernel_key, kernel_def, compile_config.arch,
      compile_config.offline_cache);
  if (handle != nullptr) {
    kernel_def.set_snode_tree_dependencies(
        handle->compiled().snode_tree_ids());
  }
  return handle;
}

// P5.b: batch / parallel kernel compilation.
//
// Design:
// 1. Compilation is dispatched to a ParallelExecutor with
//    `compile_config.num_compile_threads` workers.
// 2. All heavy lifting (IR passes, LLVM opt, SPIR-V codegen, GPU module load)
//    runs on worker threads. The main thread only submits + flushes.
// 3. Ordering: kernel compilation is order-independent at the C++ level —
//    @ti.func inlining and template specialization are resolved in Python
//    before a `Kernel` object ever reaches C++. Each kernel is compiled as a
//    self-contained unit.
// 4. Thread-safety:
//    - `KernelCompilationManager::load_or_compile` is guarded by its own
//      cache_mutex_ (P5.a) so concurrent cache hits/inserts are safe.
//    - LLVM: TaichiLLVMContext maintains per-thread state in a synchronized
//      registry; first-touch on a worker lazily clones the runtime module +
//      struct_modules from the main thread (which is already quiescent after
//      materialize_runtime). Exited workers retire their registry entries.
//    - CUDA: `cuModuleLoadDataEx` is serialized by CUDAContext::get_lock_guard
//      inside JITSessionCUDA; all optimization runs in parallel.
//    - Vulkan: SPIR-V codegen touches no shared state.
// 5. Error propagation: the first exception from any worker is captured and
//    rethrown on the calling thread after flush(); remaining workers still
//    finish their in-flight tasks so we never leave the executor in a bad
//    state.
//
// V7 (2026-04-26) — thread-local depth counter that tracks whether the
// current thread is acting as a compile_kernels outer worker. The LLVM
// codegen path consults this via Program::in_compile_kernels_worker() to
// avoid double-pool oversubscription. Only incremented when
// compile_config.compile_dag_scheduler is true.
namespace {
thread_local int g_compile_kernels_worker_depth = 0;

bool collect_unique_compile_kernels(
    const std::vector<const Kernel *> &kernels,
    std::vector<const Kernel *> &deduplicated) {
  if (kernels.size() <= 1) {
    return false;
  }

  bool has_duplicate = false;
  if (kernels.size() <= 32) {
    for (std::size_t i = 1; i < kernels.size() && !has_duplicate; ++i) {
      for (std::size_t j = 0; j < i; ++j) {
        if (kernels[i] == kernels[j]) {
          has_duplicate = true;
          break;
        }
      }
    }
    if (!has_duplicate) {
      return false;
    }
  }

  deduplicated.clear();
  deduplicated.reserve(kernels.size());
  std::unordered_set<const Kernel *> seen;
  seen.reserve(kernels.size());
  for (const Kernel *kernel : kernels) {
    if (seen.insert(kernel).second) {
      deduplicated.push_back(kernel);
    }
  }
  return deduplicated.size() != kernels.size();
}

class CompileKernelsWorkerDepthScope {
 public:
  explicit CompileKernelsWorkerDepthScope(bool enabled) : enabled_(enabled) {
    if (enabled_) {
      ++g_compile_kernels_worker_depth;
    }
  }

  ~CompileKernelsWorkerDepthScope() {
    if (enabled_) {
      --g_compile_kernels_worker_depth;
    }
  }

 private:
  bool enabled_{false};
};
}  // namespace

bool Program::in_compile_kernels_worker() {
  return g_compile_kernels_worker_depth > 0;
}

// Caller contract: do NOT destroy SNode trees concurrently with this call.
void Program::compile_kernels(
    const CompileConfig &compile_config,
    const std::vector<const Kernel *> &kernels) {
  if (kernels.empty()) {
    return;
  }
  auto start_t = Time::get_time();
  const auto caps = get_device_caps();
  // D4: the Python API returns the submitted task count, but the backend only
  // needs to compile each materialized specialization once. Keep the common
  // unique-task path allocation-free.
  std::vector<const Kernel *> deduplicated_jobs;
  const auto *compile_jobs = &kernels;
  if (collect_unique_compile_kernels(kernels, deduplicated_jobs)) {
    compile_jobs = &deduplicated_jobs;
  }

  const int n_compile_threads =
      std::max(1, compile_config.num_compile_threads);
  int n_workers = std::min<int>(n_compile_threads, (int)compile_jobs->size());

  auto &mgr = program_impl_->get_kernel_compilation_manager();
  const bool dag_mode = compile_config.compile_dag_scheduler;
  // V8.a (2026-04-26): when dag_mode is on and there are fewer kernels than
  // compile threads, skip the outer ParallelExecutor entirely. The serial
  // outer loop lets each kernel's inner offload pool (LLVM
  // compilation_workers / SPIR-V V2 std::async) consume the full T-wide
  // worker budget on its own. With V7 enabled the previous behaviour would
  // create only N outer workers and force inner-serial, leaving (T-N) cores
  // idle. See compile_doc/优化总规划.md §3.5.
  const bool prefer_inner_parallelism =
      dag_mode && (int)compile_jobs->size() < n_compile_threads;
  if (n_workers <= 1 || prefer_inner_parallelism) {
    // Fast path: honour the same serial path as compile_kernel.
    for (auto *k : *compile_jobs) {
      const auto effective_config =
          make_effective_kernel_compile_config(compile_config, *k);
      const auto &compiled = mgr.load_or_compile(effective_config, caps, *k);
      k->set_snode_tree_dependencies(compiled.snode_tree_ids());
    }
    total_compilation_time_ += Time::get_time() - start_t;
    return;
  }

  std::mutex err_mu;
  std::exception_ptr first_error;

  {
    ParallelExecutor exec("compile_kernels", n_workers);
    for (auto *k : *compile_jobs) {
      exec.enqueue([&, k]() {
        // V7: mark this worker so the LLVM inner pool stays serial.
        CompileKernelsWorkerDepthScope worker_scope(dag_mode);
        try {
          const auto effective_config =
              make_effective_kernel_compile_config(compile_config, *k);
          const auto &compiled =
              mgr.load_or_compile(effective_config, caps, *k);
          k->set_snode_tree_dependencies(compiled.snode_tree_ids());
        } catch (...) {
          std::lock_guard<std::mutex> g(err_mu);
          if (!first_error) {
            first_error = std::current_exception();
          }
        }
      });
    }
    // ~ParallelExecutor runs flush() implicitly via its destructor.
    exec.flush();
  }

  total_compilation_time_ += Time::get_time() - start_t;
  if (first_error) {
    std::rethrow_exception(first_error);
  }
}

void Program::launch_kernel(const CompiledKernelData &compiled_kernel_data,
                            LaunchContextBuilder &ctx) {
  ensure_runtime_submission_allowed("kernel launch");
  if (ctx.argpack_ptrs.empty() && ctx.ndarray_ptrs.empty() &&
      ctx.texture_ptrs.empty() && ctx.dense_storage_ptrs.empty()) {
    const bool attribute = ordinary_launch_attribution_.enabled;
    const std::uint64_t total_started =
        attribute ? ordinary_launch_now_ns() : 0;
    if (attribute) {
      ordinary_launch_attribution_.launches.fetch_add(
          1, std::memory_order_relaxed);
      ordinary_launch_attribution_.no_resource_fast_path.fetch_add(
          1, std::memory_order_relaxed);
    }
    auto launch_backend = [&] {
      const std::uint64_t started = attribute ? ordinary_launch_now_ns() : 0;
      program_impl_->get_kernel_launcher().launch_kernel(compiled_kernel_data,
                                                          ctx);
      if (attribute) {
        ordinary_launch_attribution_.backend_submit_ns.fetch_add(
            ordinary_launch_now_ns() - started, std::memory_order_relaxed);
      }
    };
    auto account_completion = [&] {
      const std::uint64_t started = attribute ? ordinary_launch_now_ns() : 0;
      mark_runtime_submission();
      check_runtime_error_after_kernel_launch(compiled_kernel_data);
      if (attribute) {
        ordinary_launch_attribution_.completion_accounting_ns.fetch_add(
            ordinary_launch_now_ns() - started, std::memory_order_relaxed);
      }
    };
    auto finish_attribution = [&] {
      if (attribute) {
        ordinary_launch_attribution_.total_host_ns.fetch_add(
            ordinary_launch_now_ns() - total_started,
            std::memory_order_relaxed);
      }
    };
    // Keep the pre-registry ordinary-launch machine path intact. Routing this
    // overwhelmingly common case through launch_kernel_impl() added an
    // out-of-line call, optional guard construction, and a registered-handle
    // branch even though none of them can contribute ownership here.
    if (active_snode_tree_lifecycle_program != this &&
        (!ordinary_snode_guard_elision_enabled_ ||
         compiled_kernel_data.has_snode_tree_dependencies())) {
      const std::uint64_t started = attribute ? ordinary_launch_now_ns() : 0;
      auto lifecycle_guard = acquire_snode_tree_lifecycle_read_guard();
      if (attribute) {
        ordinary_launch_attribution_.snode_guard_acquisitions.fetch_add(
            1, std::memory_order_relaxed);
        ordinary_launch_attribution_.snode_guard_wait_ns.fetch_add(
            ordinary_launch_now_ns() - started,
            std::memory_order_relaxed);
      }
      if (!runtime_completion_tracking_enabled_.load(
              std::memory_order_acquire)) {
        launch_backend();
        account_completion();
        finish_attribution();
        return;
      }
      auto completion_scope = acquire_runtime_submission_scope();
      launch_backend();
      account_completion();
      finish_attribution();
      return;
    }
    if (active_snode_tree_lifecycle_program != this && attribute) {
      ordinary_launch_attribution_.snode_guard_elisions.fetch_add(
          1, std::memory_order_relaxed);
    }
    if (!runtime_completion_tracking_enabled_.load(
            std::memory_order_acquire)) {
      launch_backend();
      account_completion();
      finish_attribution();
      return;
    }
    auto completion_scope = acquire_runtime_submission_scope();
    launch_backend();
    account_completion();
    finish_attribution();
    return;
  }

  if (active_runtime_resource_graph_program == this) {
    const bool attribute = ordinary_launch_attribution_.enabled;
    const std::uint64_t total_started =
        attribute ? ordinary_launch_now_ns() : 0;
    if (attribute) {
      ordinary_launch_attribution_.launches.fetch_add(
          1, std::memory_order_relaxed);
      ordinary_launch_attribution_.graph_transaction_dispatches.fetch_add(
          1, std::memory_order_relaxed);
    }
    // The enclosing Graph transaction has validated and pinned every runtime
    // argument while owning runtime_resource_submission_mutex_. Per-dispatch
    // owner/handle checks remain, but no registry lookup or recursive lock is
    // needed here.
    TI_ASSERT(ctx.argpack_ptrs.empty());
    resolve_ndarray_launch_context_under_guard(ctx);
    resolve_runtime_storage_launch_context_under_guard(ctx);
    resolve_texture_launch_context_under_guard(ctx);
    auto completion_scope = acquire_runtime_submission_scope();
    const std::uint64_t backend_started =
        attribute ? ordinary_launch_now_ns() : 0;
    program_impl_->get_kernel_launcher().launch_kernel(compiled_kernel_data,
                                                        ctx);
    if (attribute) {
      ordinary_launch_attribution_.backend_submit_ns.fetch_add(
          ordinary_launch_now_ns() - backend_started,
          std::memory_order_relaxed);
    }
    const std::uint64_t completion_started =
        attribute ? ordinary_launch_now_ns() : 0;
    mark_runtime_submission();
    check_runtime_error_after_kernel_launch(compiled_kernel_data);
    if (attribute) {
      ordinary_launch_attribution_.completion_accounting_ns.fetch_add(
          ordinary_launch_now_ns() - completion_started,
          std::memory_order_relaxed);
      ordinary_launch_attribution_.total_host_ns.fetch_add(
          ordinary_launch_now_ns() - total_started,
          std::memory_order_relaxed);
    }
    return;
  }
  launch_kernel_impl(compiled_kernel_data, ctx, nullptr);
}

void Program::launch_registered_kernel(
    const CompiledKernelData &compiled_kernel_data,
    KernelLaunchHandle handle,
    LaunchContextBuilder &ctx) {
  ensure_runtime_submission_allowed("registered kernel launch");
  launch_kernel_impl(compiled_kernel_data, ctx, &handle);
}

void Program::launch_kernel_impl(
    const CompiledKernelData &compiled_kernel_data,
    LaunchContextBuilder &ctx,
    const KernelLaunchHandle *registered_handle) {
  const bool attribute = ordinary_launch_attribution_.enabled;
  const std::uint64_t total_started =
      attribute ? ordinary_launch_now_ns() : 0;
  if (attribute) {
    ordinary_launch_attribution_.launches.fetch_add(
        1, std::memory_order_relaxed);
    const bool has_resources =
        !ctx.argpack_ptrs.empty() || !ctx.ndarray_ptrs.empty() ||
        !ctx.texture_ptrs.empty() || !ctx.dense_storage_ptrs.empty();
    if (has_resources) {
      ordinary_launch_attribution_.general_resource_launches.fetch_add(
          1, std::memory_order_relaxed);
    } else {
      ordinary_launch_attribution_.no_resource_fast_path.fetch_add(
          1, std::memory_order_relaxed);
    }
    if (ctx.argpack_ptrs.empty() && !ctx.ndarray_ptrs.empty() &&
        ctx.texture_ptrs.empty() && ctx.dense_storage_ptrs.empty()) {
      ordinary_launch_attribution_.owned_ndarray_only_launches.fetch_add(
          1, std::memory_order_relaxed);
    }
  }
  auto finish_attribution = [&] {
    if (attribute) {
      ordinary_launch_attribution_.total_host_ns.fetch_add(
          ordinary_launch_now_ns() - total_started,
          std::memory_order_relaxed);
    }
  };
  struct ResolvedDenseBindingReset {
    LaunchContextBuilder *ctx;
    ~ResolvedDenseBindingReset() {
      ctx->clear_resolved_dense_storage();
    }
  } resolved_dense_binding_reset{&ctx};

  std::optional<SNodeTreeLifecycleReadGuard> lifecycle_guard;
  if (active_snode_tree_lifecycle_program != this &&
      (!ordinary_snode_guard_elision_enabled_ ||
       compiled_kernel_data.has_snode_tree_dependencies())) {
    // Global lock order is SNodeTree lifecycle -> runtime-resource submission.
    // Graph already holds the former; ordinary launches acquire it here.
    const std::uint64_t started = attribute ? ordinary_launch_now_ns() : 0;
    lifecycle_guard.emplace(acquire_snode_tree_lifecycle_read_guard());
    if (attribute) {
      ordinary_launch_attribution_.snode_guard_acquisitions.fetch_add(
          1, std::memory_order_relaxed);
      ordinary_launch_attribution_.snode_guard_wait_ns.fetch_add(
          ordinary_launch_now_ns() - started, std::memory_order_relaxed);
    }
  } else if (active_snode_tree_lifecycle_program != this && attribute) {
    ordinary_launch_attribution_.snode_guard_elisions.fetch_add(
        1, std::memory_order_relaxed);
  }
  // launch_kernel() handles the dominant no-resource path before entering this
  // ownership-oriented slow path. Keep a defensive fast path for internal
  // registered-handle callers that do not make the same promise.
  if (ctx.argpack_ptrs.empty() && ctx.ndarray_ptrs.empty() &&
      ctx.texture_ptrs.empty() && ctx.dense_storage_ptrs.empty()) {
    auto completion_scope = acquire_runtime_submission_scope();
    const std::uint64_t backend_started =
        attribute ? ordinary_launch_now_ns() : 0;
    if (registered_handle) {
      program_impl_->get_kernel_launcher().launch_registered_kernel(
          compiled_kernel_data, *registered_handle, ctx);
    } else {
      program_impl_->get_kernel_launcher().launch_kernel(compiled_kernel_data,
                                                          ctx);
    }
    if (attribute) {
      ordinary_launch_attribution_.backend_submit_ns.fetch_add(
          ordinary_launch_now_ns() - backend_started,
          std::memory_order_relaxed);
    }
    const std::uint64_t completion_started =
        attribute ? ordinary_launch_now_ns() : 0;
    mark_runtime_submission();
    check_runtime_error_after_kernel_launch(compiled_kernel_data);
    if (attribute) {
      ordinary_launch_attribution_.completion_accounting_ns.fetch_add(
          ordinary_launch_now_ns() - completion_started,
          std::memory_order_relaxed);
    }
    finish_attribution();
    return;
  }

  if (active_runtime_resource_graph_program == this) {
    TI_ASSERT(ctx.argpack_ptrs.empty());
    resolve_ndarray_launch_context_under_guard(ctx);
    resolve_runtime_storage_launch_context_under_guard(ctx);
    resolve_texture_launch_context_under_guard(ctx);
    auto completion_scope = acquire_runtime_submission_scope();
    const std::uint64_t backend_started =
        attribute ? ordinary_launch_now_ns() : 0;
    if (registered_handle) {
      program_impl_->get_kernel_launcher().launch_registered_kernel(
          compiled_kernel_data, *registered_handle, ctx);
    } else {
      program_impl_->get_kernel_launcher().launch_kernel(compiled_kernel_data,
                                                          ctx);
    }
    if (attribute) {
      ordinary_launch_attribution_.backend_submit_ns.fetch_add(
          ordinary_launch_now_ns() - backend_started,
          std::memory_order_relaxed);
    }
    const std::uint64_t completion_started =
        attribute ? ordinary_launch_now_ns() : 0;
    mark_runtime_submission();
    check_runtime_error_after_kernel_launch(compiled_kernel_data);
    if (attribute) {
      ordinary_launch_attribution_.completion_accounting_ns.fetch_add(
          ordinary_launch_now_ns() - completion_started,
          std::memory_order_relaxed);
    }
    finish_attribution();
    return;
  }

  if (ctx.argpack_ptrs.empty() && ctx.ndarray_ptrs.empty() &&
      ctx.dense_storage_ptrs.empty()) {
    // Texture-only kernels are common in staging and visualization. Do not
    // value-initialize the unrelated ArgPack/Ndarray inline lease arrays on
    // every submission. This keeps the same submission transaction and
    // failure recovery as the general mixed-resource path below.
    const std::uint64_t lock_started =
        attribute ? ordinary_launch_now_ns() : 0;
    std::unique_lock<std::recursive_mutex> resource_submission_lock(
        runtime_resource_submission_mutex_);
    if (attribute) {
      ordinary_launch_attribution_.resource_lock_acquisitions.fetch_add(
          1, std::memory_order_relaxed);
      ordinary_launch_attribution_.resource_lock_wait_ns.fetch_add(
          ordinary_launch_now_ns() - lock_started,
          std::memory_order_relaxed);
    }
    auto completion_scope = acquire_runtime_submission_scope();
    const bool retain_resources_until_sync =
        arch_is_gpu(compile_config().arch);
    const std::uint64_t resolution_started =
        attribute ? ordinary_launch_now_ns() : 0;
    TextureLaunchLeases texture_leases;
    if (retain_resources_until_sync) {
      texture_leases = acquire_texture_launch_leases(ctx);
    } else {
      resolve_texture_launch_context(ctx);
    }
    if (attribute) {
      ordinary_launch_attribution_.resource_resolution_ns.fetch_add(
          ordinary_launch_now_ns() - resolution_started,
          std::memory_order_relaxed);
    }

    bool pin_attempted = false;
    auto pin_after_submission = [&] {
      if (!retain_resources_until_sync || pin_attempted) {
        return;
      }
      pin_attempted = true;
      if (texture_leases.empty()) {
        return;
      }
      try {
        pin_texture_launch_leases(texture_leases);
      } catch (...) {
        program_impl_->synchronize();
        release_completed_texture_leases();
        throw;
      }
    };
    auto launch = [&] {
      const std::uint64_t backend_started =
          attribute ? ordinary_launch_now_ns() : 0;
      if (registered_handle) {
        program_impl_->get_kernel_launcher().launch_registered_kernel(
            compiled_kernel_data, *registered_handle, ctx);
      } else {
        program_impl_->get_kernel_launcher().launch_kernel(compiled_kernel_data,
                                                            ctx);
      }
      if (attribute) {
        ordinary_launch_attribution_.backend_submit_ns.fetch_add(
            ordinary_launch_now_ns() - backend_started,
            std::memory_order_relaxed);
      }
      const std::uint64_t completion_started =
          attribute ? ordinary_launch_now_ns() : 0;
      mark_runtime_submission();
      pin_after_submission();
      check_runtime_error_after_kernel_launch(compiled_kernel_data);
      if (attribute) {
        ordinary_launch_attribution_.completion_accounting_ns.fetch_add(
            ordinary_launch_now_ns() - completion_started,
            std::memory_order_relaxed);
      }
    };
    try {
      launch();
    } catch (...) {
      const std::exception_ptr launch_error = std::current_exception();
      if (retain_resources_until_sync && !pin_attempted) {
        try {
          pin_after_submission();
        } catch (...) {
        }
      }
      std::rethrow_exception(launch_error);
    }
    finish_attribution();
    return;
  }

  const std::uint64_t lock_started =
      attribute ? ordinary_launch_now_ns() : 0;
  std::unique_lock<std::recursive_mutex> resource_submission_lock(
      runtime_resource_submission_mutex_);
  if (attribute) {
    ordinary_launch_attribution_.resource_lock_acquisitions.fetch_add(
        1, std::memory_order_relaxed);
    ordinary_launch_attribution_.resource_lock_wait_ns.fetch_add(
        ordinary_launch_now_ns() - lock_started,
        std::memory_order_relaxed);
  }
  auto completion_scope = acquire_runtime_submission_scope();
  const bool retain_resources_until_sync = arch_is_gpu(compile_config().arch);
  const std::uint64_t resolution_started =
      attribute ? ordinary_launch_now_ns() : 0;
  ArgPackLaunchLeases argpack_leases;
  if (!ctx.argpack_ptrs.empty()) {
    argpack_leases = acquire_argpack_launch_leases(ctx);
  }
  NdarrayLaunchLeases ndarray_leases;
  if (!ctx.ndarray_ptrs.empty()) {
    if (retain_resources_until_sync) {
      ndarray_leases = acquire_ndarray_launch_leases(ctx);
    } else {
      resolve_ndarray_launch_context(ctx);
    }
  }
  ExternalDenseStorageLaunchLeases external_dense_storage_leases;
  if (!ctx.dense_storage_ptrs.empty()) {
    resolve_dense_storage_launch_context(ctx, ndarray_leases,
                                         external_dense_storage_leases);
  }
  TextureLaunchLeases texture_leases;
  if (!ctx.texture_ptrs.empty()) {
    if (retain_resources_until_sync) {
      texture_leases = acquire_texture_launch_leases(ctx);
    } else {
      resolve_texture_launch_context(ctx);
    }
  }
  if (attribute) {
    ordinary_launch_attribution_.resource_resolution_ns.fetch_add(
        ordinary_launch_now_ns() - resolution_started,
        std::memory_order_relaxed);
  }

  ExternalAccessEpoch external_access_epoch;
  begin_external_access_epoch(external_access_epoch, external_dense_storage_leases);
  bool pin_attempted = false;
  auto pin_after_submission = [&] {
    if (!retain_resources_until_sync || pin_attempted) {
      return;
    }
    pin_attempted = true;
    try {
      if (!argpack_leases.empty()) {
        pin_argpack_launch_leases(argpack_leases);
      }
      if (!ndarray_leases.empty()) {
        pin_ndarray_launch_leases(ndarray_leases);
      }
      if (!texture_leases.empty()) {
        pin_texture_launch_leases(texture_leases);
      }
      if (!external_dense_storage_leases.empty()) {
        pin_external_dense_storage_launch_leases(
            external_dense_storage_leases);
      }
    } catch (...) {
      // Submission already happened. If metadata allocation fails, complete
      // the backend work before allowing the stack leases to unwind.
      program_impl_->synchronize();
      release_completed_argpack_leases();
      release_completed_ndarray_leases();
      release_completed_texture_leases();
      release_completed_external_dense_storage_leases();
      release_completed_vulkan_native_resources();
      throw;
    }
  };
  auto launch = [&] {
    const std::uint64_t backend_started =
        attribute ? ordinary_launch_now_ns() : 0;
    if (registered_handle) {
      program_impl_->get_kernel_launcher().launch_registered_kernel(
          compiled_kernel_data, *registered_handle, ctx);
    } else {
      program_impl_->get_kernel_launcher().launch_kernel(compiled_kernel_data,
                                                          ctx);
    }
    if (attribute) {
      ordinary_launch_attribution_.backend_submit_ns.fetch_add(
          ordinary_launch_now_ns() - backend_started,
          std::memory_order_relaxed);
    }
    const std::uint64_t completion_started =
        attribute ? ordinary_launch_now_ns() : 0;
    mark_runtime_submission();
    pin_after_submission();
    check_runtime_error_after_kernel_launch(compiled_kernel_data);
    if (attribute) {
      ordinary_launch_attribution_.completion_accounting_ns.fetch_add(
          ordinary_launch_now_ns() - completion_started,
          std::memory_order_relaxed);
    }
  };

  try {
    launch();
    external_access_epoch.release();
  } catch (...) {
    const std::exception_ptr launch_error = std::current_exception();
    if (retain_resources_until_sync && !pin_attempted) {
      try {
        pin_after_submission();
      } catch (...) {
        // pin_after_submission synchronized before reporting its own failure;
        // preserve the backend exception that initiated this recovery path.
      }
    }
    try {
      external_access_epoch.release();
    } catch (...) {
    }
    std::rethrow_exception(launch_error);
  }
  finish_attribution();
}

void Program::RegisteredKernelExecutionPlan::launch(
    Program &program,
    LaunchContextBuilder &ctx) const {
  if (!dependencies_.empty()) {
    auto scope = begin_launch(program);
    scope->launch(ctx);
    return;
  }
  TI_ERROR_IF(&program != owner_,
              "Registered kernel execution plan belongs to another Program");
  TI_ASSERT(compiled_ != nullptr);
  if (program.ordinary_launch_attribution_.enabled) {
    program.ordinary_launch_attribution_.registered_execution_plan_launches
        .fetch_add(1, std::memory_order_relaxed);
  }
  program.launch_registered_kernel(*compiled_, handle_, ctx);
}

std::unique_ptr<Program::RegisteredKernelExecutionPlanLaunchScope>
Program::RegisteredKernelExecutionPlan::begin_launch(Program &program) const {
  TI_ERROR_IF(&program != owner_,
              "Registered kernel execution plan belongs to another Program");
  TI_ASSERT(compiled_ != nullptr);
  std::optional<SNodeTreeLifecycleReadGuard> lifecycle_guard;
  if (!dependencies_.empty()) {
    const bool attribute = program.ordinary_launch_attribution_.enabled;
    const std::uint64_t started = attribute ? ordinary_launch_now_ns() : 0;
    lifecycle_guard.emplace(
        program.acquire_snode_tree_lifecycle_read_guard());
    if (attribute) {
      program.ordinary_launch_attribution_.snode_guard_acquisitions.fetch_add(
          1, std::memory_order_relaxed);
      program.ordinary_launch_attribution_.snode_guard_wait_ns.fetch_add(
          ordinary_launch_now_ns() - started, std::memory_order_relaxed);
    }
    // Validate before Python constructs a context from the retained Kernel
    // ABI. The returned scope keeps this transaction live through submission.
    const std::uint64_t current_epoch = lifecycle_guard->epoch();
    if (validated_snode_tree_epoch_.load(std::memory_order_acquire) !=
        current_epoch) {
      if (attribute) {
        program.ordinary_launch_attribution_
            .registered_plan_snode_validation_scans.fetch_add(
                1, std::memory_order_relaxed);
      }
      program.validate_snode_tree_dependencies(
          dependencies_, "Registered kernel execution plan");
      validated_snode_tree_epoch_.store(current_epoch,
                                        std::memory_order_release);
    } else if (attribute) {
      program.ordinary_launch_attribution_
          .registered_plan_snode_validation_epoch_fast_hits.fetch_add(
              1, std::memory_order_relaxed);
    }
  }
  return std::unique_ptr<RegisteredKernelExecutionPlanLaunchScope>(
      new RegisteredKernelExecutionPlanLaunchScope(
          &program, compiled_, handle_, std::move(lifecycle_guard)));
}

void Program::RegisteredKernelExecutionPlanLaunchScope::launch(
    LaunchContextBuilder &ctx) {
  TI_ERROR_IF(launched_,
              "Registered kernel execution plan launch scope was already used");
  TI_ASSERT(program_ != nullptr);
  TI_ASSERT(compiled_ != nullptr);
  launched_ = true;
  if (program_->ordinary_launch_attribution_.enabled) {
    program_->ordinary_launch_attribution_.registered_execution_plan_launches
        .fetch_add(1, std::memory_order_relaxed);
  }
  program_->launch_registered_kernel(*compiled_, handle_, ctx);
}

std::unique_ptr<Program::RegisteredKernelExecutionPlan>
Program::register_kernel_execution_plan(
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &caps,
    const Kernel &kernel_def,
    bool allow_snode_tree_dependencies) {
  auto lifecycle_guard = acquire_snode_tree_lifecycle_read_guard();
  const auto &compiled = compile_kernel(compile_config, caps, kernel_def);
  if (!compiled.get_handle() ||
      (compiled.has_snode_tree_dependencies() &&
       !allow_snode_tree_dependencies)) {
    return nullptr;
  }
  std::vector<SNodeTreeDependency> dependencies;
  if (compiled.has_snode_tree_dependencies()) {
    dependencies =
        snapshot_snode_tree_dependencies_unlocked(compiled.snode_tree_ids());
  }
  const std::uint64_t validated_snode_tree_epoch =
      dependencies.empty() ? 0 : lifecycle_guard.epoch();
  return std::unique_ptr<RegisteredKernelExecutionPlan>(
      new RegisteredKernelExecutionPlan(this, &compiled,
                                        *compiled.get_handle(),
                                        std::move(dependencies),
                                        validated_snode_tree_epoch));
}

void Program::compile_and_launch_kernel(
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &caps,
    const Kernel &kernel_def,
    LaunchContextBuilder &ctx) {
  const bool attribute = ordinary_launch_attribution_.enabled;
  const std::uint64_t total_started =
      attribute ? ordinary_launch_now_ns() : 0;
  std::optional<SNodeTreeLifecycleReadGuard> lifecycle_guard;
  const auto dependency_state = kernel_def.snode_tree_dependency_state();
  const bool needs_lifecycle_guard =
      !ordinary_snode_guard_elision_enabled_ ||
      dependency_state != Kernel::SNodeTreeDependencyState::none;
  if (needs_lifecycle_guard) {
    const std::uint64_t started = attribute ? ordinary_launch_now_ns() : 0;
    lifecycle_guard.emplace(acquire_snode_tree_lifecycle_read_guard());
    if (attribute) {
      ordinary_launch_attribution_.snode_guard_acquisitions.fetch_add(
          1, std::memory_order_relaxed);
      ordinary_launch_attribution_.snode_guard_wait_ns.fetch_add(
          ordinary_launch_now_ns() - started, std::memory_order_relaxed);
    }
  } else if (attribute) {
    ordinary_launch_attribution_.snode_guard_elisions.fetch_add(
        1, std::memory_order_relaxed);
  }
  const std::uint64_t compile_started =
      attribute ? ordinary_launch_now_ns() : 0;
  const auto &compiled = compile_kernel(compile_config, caps, kernel_def);
  if (attribute) {
    ordinary_launch_attribution_.compile_lookup_ns.fetch_add(
        ordinary_launch_now_ns() - compile_started,
        std::memory_order_relaxed);
  }
  if (ordinary_snode_guard_elision_enabled_ &&
      !compiled.has_snode_tree_dependencies()) {
    lifecycle_guard.reset();
  }
  launch_kernel(compiled, ctx);
  if (attribute) {
    ordinary_launch_attribution_.compile_and_launch_total_ns.fetch_add(
        ordinary_launch_now_ns() - total_started,
        std::memory_order_relaxed);
  }
}

void Program::check_runtime_error_after_kernel_launch(
    const CompiledKernelData &compiled_kernel_data) {
  if (!arch_uses_llvm(compiled_kernel_data.arch())) {
    return;
  }
  const bool check_runtime_error =
      compile_config().debug ||
      compiled_kernel_data.may_trigger_hash_overflow();
  if (ordinary_launch_attribution_.enabled) {
    auto &counter = check_runtime_error
                        ? ordinary_launch_attribution_.runtime_error_checks
                        : ordinary_launch_attribution_
                              .runtime_error_check_elisions;
    counter.fetch_add(1, std::memory_order_relaxed);
  }
  if (check_runtime_error) {
    program_impl_->check_runtime_error(result_buffer);
  }
}

void Program::materialize_runtime() {
  program_impl_->materialize_runtime(profiler.get(), &result_buffer);
  // Some backends create their Device lazily while materializing the runtime.
  attach_runtime_fault_reporter();
}

ThreadPool *Program::get_cpu_thread_pool() {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native primitive scheduler requested on a non-CPU "
              "backend.");
#ifdef TI_WITH_LLVM
  auto *const pool = get_llvm_program(this)->get_cpu_thread_pool();
  TI_ASSERT(pool != nullptr);
  return pool;
#else
  TI_ERROR("CPU native primitive scheduler requires LLVM support.");
  return nullptr;
#endif
}

void Program::set_cpu_scheduler_telemetry_enabled(bool enabled) {
  get_cpu_thread_pool()->set_telemetry_enabled(enabled);
}

std::unordered_map<std::string, std::uint64_t>
Program::debug_cpu_scheduler_telemetry(bool reset) {
  const auto stats = get_cpu_thread_pool()->telemetry_statistics(reset);
  return {
      {"schema_version", 1},
      {"enabled", stats.enabled ? 1u : 0u},
      {"jobs_submitted", stats.jobs_submitted},
      {"jobs_completed", stats.jobs_completed},
      {"queued_jobs", stats.queued_jobs},
      {"nested_serial_jobs", stats.nested_serial_jobs},
      {"tasks_requested", stats.tasks_requested},
      {"tasks_completed", stats.tasks_completed},
      {"nested_serial_tasks", stats.nested_serial_tasks},
      {"requested_worker_slots", stats.requested_worker_slots},
      {"joined_workers", stats.joined_workers},
      {"underfilled_jobs", stats.underfilled_jobs},
      {"cancelled_jobs", stats.cancelled_jobs},
      {"exception_jobs", stats.exception_jobs},
      {"queue_wait_ns", stats.queue_wait_ns},
      {"execution_ns", stats.execution_ns},
      {"submitter_wait_ns", stats.submitter_wait_ns},
      {"max_queue_depth", stats.max_queue_depth},
      {"max_requested_threads", stats.max_requested_threads},
      {"max_joined_workers", stats.max_joined_workers},
  };
}

void Program::set_kernel_executable_lifecycle_telemetry_enabled(bool enabled) {
  program_impl_->get_kernel_compilation_manager()
      .set_executable_lifecycle_telemetry_enabled(enabled);
}

KernelExecutableLifecycleStatistics
Program::debug_kernel_executable_lifecycle_statistics(bool reset) {
  return program_impl_->get_kernel_compilation_manager()
      .executable_lifecycle_statistics(reset);
}

static void remove_snode_frontend_caches(
    SNode *parent_snode,
    SNodeRwAccessorsBank *snode_rw_accessors_bank,
    SNodeFieldMap *snode_to_fields,
    std::unordered_set<Kernel *> *retired_accessor_kernels) {
  for (int i = 0; i < (int)parent_snode->ch.size(); i++) {
    auto child_snode = parent_snode->ch[i].get();
    if (child_snode->type == SNodeType::place) {
      auto [reader, writer] =
          snode_rw_accessors_bank->remove_cached_kernels(child_snode);
      if (reader != nullptr) {
        retired_accessor_kernels->insert(reader);
      }
      if (writer != nullptr) {
        retired_accessor_kernels->insert(writer);
      }
      snode_to_fields->erase(child_snode);
    }
    remove_snode_frontend_caches(child_snode, snode_rw_accessors_bank,
                                 snode_to_fields, retired_accessor_kernels);
  }
}

void Program::destroy_snode_tree(SNodeTree *snode_tree) {
  TI_COMPILE_PROFILER("cpp.snode.destroy.total");
  TI_ASSERT(arch_uses_llvm(compile_config().arch) ||
            compile_config().arch == Arch::vulkan ||
            compile_config().arch == Arch::dx11 ||
            compile_config().arch == Arch::dx12);

  std::unique_lock<std::shared_mutex> lifecycle_lock;
  {
    TI_COMPILE_PROFILER("cpp.snode.destroy.lifecycle_lock");
    lifecycle_lock =
        std::unique_lock<std::shared_mutex>(snode_tree_lifecycle_mutex_);
  }
  TI_ERROR_IF(snode_tree == nullptr, "Cannot destroy a null SNodeTree.");
  const int tree_id = snode_tree->id();
  TI_ERROR_IF(tree_id < 0 ||
                  static_cast<std::size_t>(tree_id) >= snode_trees_.size() ||
                  static_cast<std::size_t>(tree_id) >=
                      snode_tree_active_.size() ||
                  !snode_tree_active_[tree_id] ||
                  snode_trees_[tree_id].get() != snode_tree ||
                  snode_tree_generations_[tree_id] !=
                      snode_tree->generation(),
              "SNodeTree id={} generation={} is no longer active.", tree_id,
              snode_tree->generation());

  // The exclusive lifecycle transaction prevents new graph/kernel enqueue
  // sections and waits for current host submissions to finish. Explicit tree
  // destruction is a cold path, so complete outstanding device work before
  // releasing the root allocation.
  {
    TI_COMPILE_PROFILER("cpp.snode.destroy.backend_sync");
    program_impl_->synchronize();
  }

  // When accessing a ti.field at Python scope, SNodeRwAccessorsBank creates
  // a Taichi Kernel to read/write the field in a JIT manner, which caches the
  // compiled JIT Kernel so as to avoid recompilation when accessing the same
  // field.

  // This cache uses the place-SNode's address (SNode*) as the key,
  // which becomes unsafe once the SNodeTree gets destroyed and that
  // place-SNode's address gets reused by another SNode. We have to remove all
  // cached kernels upon SNodeTree destruction.
  SNode *root = snode_tree->root();
  std::vector<int> active_tree_ids;
  active_tree_ids.reserve(snode_tree_active_.size());
  for (std::size_t i = 0; i < snode_tree_active_.size(); ++i) {
    if (snode_tree_active_[i]) {
      active_tree_ids.push_back(static_cast<int>(i));
    }
  }
  // Snapshot every active layout while the exclusive lifecycle transaction
  // still owns all roots. Relocatable templates retain only these immutable
  // fingerprints; no root pointer or generation allocation escapes.
  auto active_dependencies =
      snapshot_snode_tree_dependencies_unlocked(active_tree_ids);

  std::unordered_set<Kernel *> retired_accessor_kernels;
  // Remove frontend state keyed by SNode pointers before destroying the root.
  {
    TI_COMPILE_PROFILER("cpp.snode.destroy.frontend_caches");
    remove_snode_frontend_caches(root, &snode_rw_accessors_bank_,
                                 &snode_to_fields_,
                                 &retired_accessor_kernels);
  }

  // Destroy the root before retiring executable state so an exceptional
  // backend teardown can still leave the old Graph/kernel artifacts intact
  // for the Python transaction's cancellation path.
  {
    TI_COMPILE_PROFILER("cpp.snode.destroy.backend_resources");
    program_impl_->destroy_snode_tree(snode_tree);
  }

  // Compiled Graph/kernel artifacts contain static root bindings. Python Graph
  // retirement already drained host invocation sections; the Program lifecycle
  // lock and device synchronize above make backend module/pipeline unload safe.
  {
    TI_COMPILE_PROFILER("cpp.snode.destroy.executables");
    // Candidate publication happens at frontend materialization, before a
    // backend key necessarily exists. Resolve stable Kernel pointers to their
    // final compiled keys at the destruction boundary, after all launches or
    // Graph builds have completed compilation and before cache invalidation.
    {
      std::lock_guard<std::mutex> candidate_lock(
          relocatable_kernel_candidate_mutex_);
      auto &manager =
          program_impl_->get_kernel_compilation_manager();
      for (const auto &kernel : kernels) {
        const auto candidate =
            relocatable_kernel_candidates_.find(kernel.get());
        if (candidate == relocatable_kernel_candidates_.end() ||
            candidate->second != kernel->snode_tree_dependencies() ||
            kernel->get_cached_kernel_key().empty()) {
          continue;
        }
        const auto &dependencies = kernel->snode_tree_dependencies();
        if (std::binary_search(dependencies.begin(), dependencies.end(),
                               tree_id)) {
          manager.register_relocatable_template_candidate(
              kernel->get_cached_kernel_key());
        }
      }
    }
    program_impl_->get_kernel_compilation_manager().invalidate_snode_tree(
        tree_id, active_dependencies);
    program_impl_->get_kernel_launcher().retire_snode_tree(tree_id);
  }

  // CompiledGraph keeps ordinary Kernel pointers stable for its whole Python
  // lifetime, so retain those definitions as small retired shells. Accessor
  // kernels are private to SNodeRwAccessorsBank, whose entries were removed
  // above; after backend cache retirement they have no remaining observer and
  // can be deleted instead of accumulating one shell per historical field.
  {
    TI_COMPILE_PROFILER("cpp.snode.destroy.kernel_definitions");
    for (auto iter = kernels.begin(); iter != kernels.end();) {
      auto &kernel = *iter;
      if (retired_accessor_kernels.erase(kernel.get()) != 0) {
        {
          std::lock_guard<std::mutex> candidate_lock(
              relocatable_kernel_candidate_mutex_);
          relocatable_kernel_candidates_.erase(kernel.get());
        }
        iter = kernels.erase(iter);
        continue;
      }
      if (kernel->definition_retired()) {
        ++iter;
        continue;
      }
      const auto &dependencies = kernel->snode_tree_dependencies();
      if (std::binary_search(dependencies.begin(), dependencies.end(),
                             tree_id)) {
        const bool preserve_relocatable_abi =
            program_impl_->get_kernel_compilation_manager()
                .has_relocatable_template(kernel->get_cached_kernel_key());
        kernel->retire_definition(preserve_relocatable_abi);
        if (!preserve_relocatable_abi) {
          std::lock_guard<std::mutex> candidate_lock(
              relocatable_kernel_candidate_mutex_);
          relocatable_kernel_candidates_.erase(kernel.get());
        }
      }
      ++iter;
    }
  }
  TI_ASSERT(retired_accessor_kernels.empty());
  snode_tree_active_[tree_id] = 0;
  free_snode_tree_ids_.push(tree_id);
  advance_snode_tree_epoch(snode_tree_mutation_epoch_);
}

SNodeTree *Program::add_snode_tree(std::unique_ptr<SNode> root,
                                   bool compile_only) {
  TI_COMPILE_PROFILER("cpp.snode.add.total");
  std::unique_lock<std::shared_mutex> lifecycle_lock;
  {
    TI_COMPILE_PROFILER("cpp.snode.add.lifecycle_lock");
    lifecycle_lock =
        std::unique_lock<std::shared_mutex>(snode_tree_lifecycle_mutex_);
  }
  const int id = allocate_snode_tree_id();
  if (static_cast<std::size_t>(id) == snode_tree_generations_.size()) {
    snode_tree_generations_.push_back(1);
    snode_tree_active_.push_back(0);
  } else {
    TI_ASSERT(id >= 0 &&
              static_cast<std::size_t>(id) < snode_tree_generations_.size());
    TI_ASSERT(!snode_tree_active_[id]);
    TI_ASSERT(snode_tree_generations_[id] !=
              std::numeric_limits<std::uint64_t>::max());
    ++snode_tree_generations_[id];
  }
  const std::uint64_t generation = snode_tree_generations_[id];
  auto tree =
      std::make_unique<SNodeTree>(id, generation, std::move(root));
  tree->root()->set_snode_tree_id(id);
  try {
    {
      TI_COMPILE_PROFILER("cpp.snode.add.backend_materialize");
      if (compile_only) {
        program_impl_->compile_snode_tree_types(tree.get());
      } else {
        program_impl_->materialize_snode_tree(tree.get(), result_buffer);
      }
    }
    // Layout compilation has now finalized cell sizes, offsets and backend-
    // neutral structural metadata. Cache the diagnostic fingerprint once;
    // Graph dispatch collection remains O(number of referenced trees).
    {
      TI_COMPILE_PROFILER("cpp.snode.add.layout_fingerprint");
      tree->refresh_layout_fingerprint();
    }
  } catch (...) {
    free_snode_tree_ids_.push(id);
    throw;
  }
  if (id < snode_trees_.size()) {
    snode_trees_[id] = std::move(tree);
  } else {
    TI_ASSERT(id == snode_trees_.size());
    snode_trees_.push_back(std::move(tree));
  }
  snode_tree_active_[id] = 1;
  advance_snode_tree_epoch(snode_tree_mutation_epoch_);
  return snode_trees_[id].get();
}

std::vector<int> Program::get_active_snode_tree_ids() const {
  std::shared_lock<std::shared_mutex> lifecycle_lock(
      snode_tree_lifecycle_mutex_);
  TI_ASSERT(snode_tree_active_.size() == snode_trees_.size());
  std::vector<int> active_ids;
  active_ids.reserve(snode_trees_.size() - free_snode_tree_ids_.size());
  for (std::size_t tree_id = 0; tree_id < snode_trees_.size(); ++tree_id) {
    if (snode_tree_active_[tree_id]) {
      active_ids.push_back(static_cast<int>(tree_id));
    }
  }
  return active_ids;
}

SNodeMetadataStatistics Program::debug_snode_metadata_statistics() const {
  std::shared_lock<std::shared_mutex> lifecycle_lock(
      snode_tree_lifecycle_mutex_);
  SNodeMetadataStatistics result;
  result.tree_slots = snode_trees_.size();
  result.free_tree_ids = free_snode_tree_ids_.size();
  result.generation_table_bytes =
      snode_tree_generations_.capacity() * sizeof(std::uint64_t);
  result.active_table_bytes =
      snode_tree_active_.capacity() * sizeof(std::uint8_t);
  result.global_snode_ids_issued =
      static_cast<std::size_t>(std::max(global_id_counter_, 0));
  std::function<std::size_t(const SNode *)> count_snodes =
      [&](const SNode *snode) -> std::size_t {
    if (snode == nullptr) {
      return 0;
    }
    std::size_t count = 1;
    for (const auto &child : snode->ch) {
      count += count_snodes(child.get());
    }
    return count;
  };
  for (std::size_t tree_id = 0; tree_id < snode_trees_.size(); ++tree_id) {
    const auto &tree = snode_trees_[tree_id];
    if (tree == nullptr) {
      continue;
    }
    const auto count = count_snodes(tree->root());
    if (tree_id < snode_tree_active_.size() &&
        snode_tree_active_[tree_id]) {
      ++result.active_tree_count;
      result.active_snode_count += count;
    } else {
      ++result.retired_tree_shells;
      result.retired_snode_count += count;
    }
  }
  result.tree_inline_bytes_lower_bound =
      (result.active_tree_count + result.retired_tree_shells) *
      sizeof(SNodeTree);
  result.snode_inline_bytes_lower_bound =
      (result.active_snode_count + result.retired_snode_count) * sizeof(SNode);
  return result;
}

SparseSNodeTreeStatistics Program::debug_sparse_snode_tree_statistics(
    int tree_id) {
  std::shared_lock<std::shared_mutex> lifecycle_lock(
      snode_tree_lifecycle_mutex_);
  TI_ERROR_IF(tree_id < 0 ||
                  static_cast<std::size_t>(tree_id) >= snode_trees_.size() ||
                  static_cast<std::size_t>(tree_id) >=
                      snode_tree_active_.size() ||
                  !snode_tree_active_[tree_id] ||
                  snode_trees_[tree_id] == nullptr,
              "SNodeTree id={} is no longer active.", tree_id);

  SNodeTree *tree = snode_trees_[tree_id].get();
  SparseSNodeTreeStatistics result;
  result.tree_id = tree_id;
  result.generation = tree->generation();
  result.layout_fingerprint = tree->layout_fingerprint();
  result.backend = compile_config().arch;
  result.memory = program_impl_->get_snode_tree_memory_statistics(
      tree, result_buffer);
  std::vector<int> snode_ids;
  std::function<void(SNode *)> collect_snode_ids = [&](SNode *snode) {
    snode_ids.push_back(snode->id);
    for (const auto &child : snode->ch) {
      collect_snode_ids(child.get());
    }
  };
  collect_snode_ids(tree->root());
  std::sort(snode_ids.begin(), snode_ids.end());
  result.listgen = program_impl_->get_kernel_launcher()
                       .debug_sparse_listgen_statistics(snode_ids);
  return result;
}

void Program::debug_reset_sparse_listgen_statistics() {
  program_impl_->get_kernel_launcher().debug_reset_sparse_listgen_statistics();
}

void Program::ExternalDenseStorageResource::finalize() {
  if (finalized_) {
    return;
  }
  finalized_ = true;
  auto release = std::move(release_);
  if (release) {
    release();
  }
}

SNode *Program::get_snode_root(int tree_id) {
  std::shared_lock<std::shared_mutex> lifecycle_lock(
      snode_tree_lifecycle_mutex_);
  TI_ERROR_IF(tree_id < 0 ||
                  static_cast<std::size_t>(tree_id) >= snode_trees_.size() ||
                  !snode_tree_active_[tree_id],
              "SNodeTree id={} is no longer active.", tree_id);
  TI_ASSERT(snode_trees_[tree_id] != nullptr);
  return snode_trees_[tree_id]->root();
}

bool Program::ArgPackLaunchLeases::contains(
    ArgPackResourceHandle handle) const noexcept {
  for (std::size_t i = 0; i < inline_count_; ++i) {
    if (inline_leases_[i]->handle() == handle) {
      return true;
    }
  }
  for (const auto &lease : overflow_leases_) {
    if (lease.handle() == handle) {
      return true;
    }
  }
  return false;
}

bool Program::ArgPackLaunchLeases::empty() const noexcept {
  return inline_count_ == 0 && overflow_leases_.empty();
}

void Program::ArgPackLaunchLeases::add(ArgPackResourceLease lease) {
  if (inline_count_ < inline_leases_.size()) {
    inline_leases_[inline_count_++].emplace(std::move(lease));
    return;
  }
  overflow_leases_.push_back(std::move(lease));
}

bool Program::NdarrayLaunchLeases::contains(
    NdarrayResourceHandle handle) const noexcept {
  for (std::size_t i = 0; i < inline_count_; ++i) {
    if (inline_leases_[i]->handle() == handle) {
      return true;
    }
  }
  for (const auto &lease : overflow_leases_) {
    if (lease.handle() == handle) {
      return true;
    }
  }
  return false;
}

Ndarray *Program::NdarrayLaunchLeases::find(
    NdarrayResourceHandle handle) const noexcept {
  for (std::size_t i = 0; i < inline_count_; ++i) {
    if (inline_leases_[i]->handle() == handle) {
      return inline_leases_[i]->get();
    }
  }
  for (const auto &lease : overflow_leases_) {
    if (lease.handle() == handle) {
      return lease.get();
    }
  }
  return nullptr;
}

bool Program::NdarrayLaunchLeases::empty() const noexcept {
  return inline_count_ == 0 && overflow_leases_.empty();
}

void Program::NdarrayLaunchLeases::add(NdarrayResourceLease lease) {
  if (inline_count_ < inline_leases_.size()) {
    inline_leases_[inline_count_++].emplace(std::move(lease));
    return;
  }
  overflow_leases_.push_back(std::move(lease));
}

bool Program::TextureLaunchLeases::contains(
    TextureResourceHandle handle) const noexcept {
  for (std::size_t i = 0; i < inline_count_; ++i) {
    if (inline_leases_[i]->handle() == handle) {
      return true;
    }
  }
  for (const auto &lease : overflow_leases_) {
    if (lease.handle() == handle) {
      return true;
    }
  }
  return false;
}

bool Program::TextureLaunchLeases::empty() const noexcept {
  return inline_count_ == 0 && overflow_leases_.empty();
}

void Program::TextureLaunchLeases::add(TextureResourceLease lease) {
  if (inline_count_ < inline_leases_.size()) {
    inline_leases_[inline_count_++].emplace(std::move(lease));
    return;
  }
  overflow_leases_.push_back(std::move(lease));
}

Program::ExternalDenseStorageResource *
Program::ExternalDenseStorageLaunchLeases::find(
    ExternalDenseStorageHandle handle) const noexcept {
  for (std::size_t i = 0; i < inline_count_; ++i) {
    if (inline_leases_[i]->handle() == handle) {
      return inline_leases_[i]->get();
    }
  }
  for (const auto &lease : overflow_leases_) {
    if (lease.handle() == handle) {
      return lease.get();
    }
  }
  return nullptr;
}

bool Program::ExternalDenseStorageLaunchLeases::empty() const noexcept {
  return inline_count_ == 0 && overflow_leases_.empty();
}

void Program::ExternalDenseStorageLaunchLeases::add(
    ExternalDenseStorageLease lease) {
  if (inline_count_ < inline_leases_.size()) {
    inline_leases_[inline_count_++].emplace(std::move(lease));
    return;
  }
  overflow_leases_.push_back(std::move(lease));
}
const std::vector<std::shared_ptr<ExternalSynchronizationDomain>> &
Program::ExternalDenseStorageLaunchLeases::synchronization_domains()
    const noexcept {
  return synchronization_domains_;
}

void Program::ExternalDenseStorageLaunchLeases::track_synchronization_domain(
    const std::shared_ptr<ExternalSynchronizationDomain> &domain) {
  if (!domain) {
    return;
  }
  for (const auto &existing : synchronization_domains_) {
    if (existing->identity() != domain->identity()) {
      continue;
    }
    TI_ERROR_IF(existing.get() != domain.get(),
                "External synchronization domain identity collision");
    return;
  }
  synchronization_domains_.push_back(domain);
}


std::uint64_t Program::argpack_lease_key(ArgPackResourceHandle handle) {
  return (static_cast<std::uint64_t>(handle.generation) << 32u) |
         static_cast<std::uint64_t>(handle.index);
}

std::uint64_t Program::ndarray_lease_key(NdarrayResourceHandle handle) {
  return (static_cast<std::uint64_t>(handle.generation) << 32u) |
         static_cast<std::uint64_t>(handle.index);
}

std::uint64_t Program::texture_lease_key(TextureResourceHandle handle) {
  return (static_cast<std::uint64_t>(handle.generation) << 32u) |
         static_cast<std::uint64_t>(handle.index);
}

std::uint64_t Program::external_dense_storage_lease_key(
    ExternalDenseStorageHandle handle) {
  return (static_cast<std::uint64_t>(handle.generation) << 32u) |
         static_cast<std::uint64_t>(handle.index);
}

Program::ExternalDenseStorageHandle Program::external_dense_storage_handle(
    const storage::StorageOwnerRef &owner) const noexcept {
  if (owner.kind != storage::StorageOwnerKind::kExternalManaged ||
      owner.external_owner_domain !=
          external_dense_storage_resources_.domain()) {
    return {};
  }
  return ExternalDenseStorageHandle{
      owner.external_owner_domain, kExternalDenseStorageResourceKind,
      owner.external_slot, owner.external_generation};
}

RuntimeResourceHandle Program::capture_argpack_resource_handle(
    const ArgPack *view) const {
  std::lock_guard<std::mutex> lock(argpack_lifecycle_mutex_);
  TI_ERROR_IF(!argpack_resources_open_,
              "Cannot bind an ArgPack after Program finalize");
  const auto found = argpack_views_.find(view);
  TI_ERROR_IF(found == argpack_views_.end(),
              "Cannot bind a stale or retired ArgPack");
  return found->second.handle;
}

Program::ArgPackLaunchLeases Program::acquire_argpack_launch_leases(
    const LaunchContextBuilder &ctx) {
  ArgPackLaunchLeases leases;
  std::lock_guard<std::mutex> lock(argpack_lifecycle_mutex_);
  TI_ERROR_IF(!argpack_resources_open_,
              "Cannot launch a kernel with ArgPack after Program finalize");
  for (const auto &entry : ctx.argpack_ptrs) {
    const auto expected = ctx.argpack_resource_handles.find(entry.first);
    TI_ERROR_IF(expected == ctx.argpack_resource_handles.end(),
                "Kernel launch is missing its captured ArgPack generation");
    const ArgPack *view = entry.second;
    const auto found = argpack_views_.find(view);
    TI_ERROR_IF(found == argpack_views_.end() ||
                    found->second.handle != expected->second,
                "Kernel launch references a stale or retired ArgPack");
    const ArgPackResourceView &resource_view = found->second;
    if (leases.contains(resource_view.handle)) {
      continue;
    }
    TI_ASSERT(resource_view.lease &&
              resource_view.lease.handle() == resource_view.handle);
    if (argpack_inflight_leases_.find(
            argpack_lease_key(resource_view.handle)) !=
        argpack_inflight_leases_.end()) {
      continue;
    }
    auto lease = resource_view.lease.clone();
    TI_ERROR_IF(!lease, "Kernel launch could not clone its ArgPack lease");
    leases.add(std::move(lease));
  }
  return leases;
}

Program::NdarrayLaunchLeases Program::acquire_ndarray_leases(
    std::initializer_list<const Ndarray *> views) {
  NdarrayLaunchLeases leases;
  // Every caller owns runtime_resource_submission_mutex_. Create, retire,
  // close, pin and completion release take the same outer gate before they
  // mutate these maps, so a second mutex on this read/clone path is redundant.
  TI_ERROR_IF(!ndarray_resources_open_,
              "Cannot use an Ndarray after Program finalize");
  for (const Ndarray *view : views) {
    if (view == nullptr) {
      continue;
    }
    const auto found = ndarray_views_.find(view);
    TI_ERROR_IF(found == ndarray_views_.end(),
                "Runtime operation references a stale or retired Ndarray");
    const NdarrayResourceView &resource_view = found->second;
    if (leases.contains(resource_view.handle)) {
      continue;
    }
    TI_ASSERT(resource_view.lease &&
              resource_view.lease.handle() == resource_view.handle);
    if (ndarray_inflight_leases_.find(
            ndarray_lease_key(resource_view.handle)) !=
        ndarray_inflight_leases_.end()) {
      continue;
    }
    auto lease = resource_view.lease.clone();
    TI_ERROR_IF(!lease, "Runtime operation could not clone its Ndarray lease");
    leases.add(std::move(lease));
  }
  return leases;
}

Program::NdarrayLaunchLeases Program::acquire_ndarray_leases(
    const std::vector<const Ndarray *> &views) {
  return acquire_ndarray_leases(views.data(), views.size());
}

Program::NdarrayLaunchLeases Program::acquire_ndarray_leases(
    const Ndarray *const *views,
    std::size_t count) {
  NdarrayLaunchLeases leases;
  // See the initializer_list overload: the submission gate is the outer
  // lifecycle transaction and keeps all map entries stable here.
  TI_ERROR_IF(!ndarray_resources_open_,
              "Cannot use an Ndarray after Program finalize");
  for (std::size_t i = 0; i < count; ++i) {
    const Ndarray *view = views[i];
    if (view == nullptr) {
      continue;
    }
    const auto found = ndarray_views_.find(view);
    TI_ERROR_IF(found == ndarray_views_.end(),
                "Runtime operation references a stale or retired Ndarray");
    const NdarrayResourceView &resource_view = found->second;
    if (leases.contains(resource_view.handle)) {
      continue;
    }
    if (ndarray_inflight_leases_.find(
            ndarray_lease_key(resource_view.handle)) !=
        ndarray_inflight_leases_.end()) {
      continue;
    }
    auto lease = resource_view.lease.clone();
    TI_ERROR_IF(!lease, "Runtime operation could not clone its Ndarray lease");
    leases.add(std::move(lease));
  }
  return leases;
}

Program::TextureLaunchLeases Program::acquire_texture_leases(
    std::initializer_list<const Texture *> views) {
  TextureLaunchLeases leases;
  TI_ERROR_IF(!texture_resources_open_,
              "Cannot use a Texture after Program finalize");
  for (const Texture *view : views) {
    if (view == nullptr) {
      continue;
    }
    const auto found = texture_views_.find(view);
    TI_ERROR_IF(found == texture_views_.end(),
                "Runtime operation references a stale or retired Texture");
    const TextureResourceView &resource_view = found->second;
    if (leases.contains(resource_view.handle) ||
        texture_inflight_leases_.find(
            texture_lease_key(resource_view.handle)) !=
            texture_inflight_leases_.end()) {
      continue;
    }
    auto lease = resource_view.lease.clone();
    TI_ERROR_IF(!lease, "Runtime operation could not clone its Texture lease");
    leases.add(std::move(lease));
  }
  return leases;
}

Program::TextureLaunchLeases Program::acquire_texture_leases(
    const std::vector<const Texture *> &views) {
  return acquire_texture_leases(views.data(), views.size());
}

Program::TextureLaunchLeases Program::acquire_texture_leases(
    const Texture *const *views,
    std::size_t count) {
  TextureLaunchLeases leases;
  TI_ERROR_IF(!texture_resources_open_,
              "Cannot use a Texture after Program finalize");
  for (std::size_t i = 0; i < count; ++i) {
    const Texture *view = views[i];
    if (view == nullptr) {
      continue;
    }
    const auto found = texture_views_.find(view);
    TI_ERROR_IF(found == texture_views_.end(),
                "Runtime operation references a stale or retired Texture");
    const TextureResourceView &resource_view = found->second;
    if (leases.contains(resource_view.handle) ||
        texture_inflight_leases_.find(
            texture_lease_key(resource_view.handle)) !=
            texture_inflight_leases_.end()) {
      continue;
    }
    auto lease = resource_view.lease.clone();
    TI_ERROR_IF(!lease, "Runtime operation could not clone its Texture lease");
    leases.add(std::move(lease));
  }
  return leases;
}

Program::NdarrayResourceLease Program::acquire_ndarray_external_lease(
    RuntimeResourceHandle handle) {
  std::lock_guard<std::recursive_mutex> submission_lock(
      runtime_resource_submission_mutex_);
  TI_ERROR_IF(!ndarray_resources_open_,
              "Cannot retain an Ndarray after Program finalize");
  auto [result, lease] = ndarray_resources_.acquire(handle);
  TI_ERROR_IF(result != NdarrayResourceRegistry::Result::kSuccess,
              "Cannot retain a stale or retired Ndarray");
  return std::move(lease);
}

Program::TextureResourceLease Program::acquire_texture_external_lease(
    const Texture *view) {
  std::lock_guard<std::recursive_mutex> submission_lock(
      runtime_resource_submission_mutex_);
  TI_ERROR_IF(!texture_resources_open_,
              "Cannot retain a Texture after Program finalize");
  const auto found = texture_views_.find(view);
  TI_ERROR_IF(found == texture_views_.end() ||
                  found->second.handle != view->runtime_resource_handle(),
              "Cannot retain a stale or retired Texture");
  auto lease = found->second.lease.clone();
  TI_ERROR_IF(!lease, "Cannot clone the Texture external lease");
  return lease;
}

void Program::retain_ndarrays_for_external_submission(
    const std::vector<const Ndarray *> &views) {
  retain_ndarrays_for_external_submission(views.data(), views.size());
}

void Program::retain_ndarrays_for_external_submission(
    const Ndarray *const *views,
    std::size_t count) {
  // The public contract requires the caller to keep
  // runtime_resource_submission_mutex_ for the whole external transaction.
  // Re-locking the recursive mutex here is safe but adds measurable fixed
  // overhead to every Graph replay without extending the protected interval.
  if (!arch_is_gpu(compile_config().arch)) {
    validate_ndarrays_for_external_submission(views, count);
    return;
  }
  auto leases = acquire_ndarray_leases(views, count);
  if (!leases.empty()) {
    pin_ndarray_launch_leases(leases);
  }
}

void Program::retain_runtime_storage_for_graph_submission(
    const storage::RuntimeStorageArgument *const *arguments,
    std::size_t count) {
  TI_ERROR_IF(active_runtime_resource_graph_program != this,
              "Runtime storage retention requires an active Graph scope");
  TI_ASSERT(active_runtime_resource_graph_scope != nullptr);
  NdarrayLaunchLeases ndarray_leases;
  ExternalDenseStorageLaunchLeases external_leases;
  for (std::size_t i = 0; i < count; ++i) {
    const auto *argument = arguments[i];
    TI_ERROR_IF(argument == nullptr,
                "Graph received a null runtime storage argument");
    if (argument->descriptor().owner().kind ==
        storage::StorageOwnerKind::kSNodePayload) {
      // The Graph already owns the SNodeTree lifecycle read guard. Dense
      // Field payloads never create allocator leases, so resolving them here
      // would duplicate every dispatch's address validation without adding
      // lifetime protection.
      continue;
    }
    resolve_dense_storage_descriptor(argument->descriptor(), ndarray_leases,
                                     external_leases, argument);
  }
  if (!external_leases.synchronization_domains().empty()) {
    TI_ERROR_IF(
        active_runtime_resource_graph_scope->external_access_epoch_,
        "Graph external access epoch was already acquired");
    auto epoch = std::make_unique<ExternalAccessEpoch>();
    begin_external_access_epoch(*epoch, external_leases);
    active_runtime_resource_graph_scope->external_access_epoch_ =
        std::move(epoch);
  }
  if (arch_is_gpu(compile_config().arch)) {
    if (!ndarray_leases.empty()) {
      pin_ndarray_launch_leases(ndarray_leases);
    }
    if (!external_leases.empty()) {
      pin_external_dense_storage_launch_leases(external_leases);
    }
  }
}

storage::ResolvedDenseBinding
Program::resolve_runtime_storage_argument_under_graph_guard(
    const storage::RuntimeStorageArgument &argument) {
  const bool requires_snode_guard =
      argument.descriptor().owner().kind ==
      storage::StorageOwnerKind::kSNodePayload;
  TI_ERROR_IF(active_runtime_resource_graph_program != this ||
                  (requires_snode_guard &&
                   active_snode_tree_lifecycle_program != this),
              "Runtime storage Graph signature resolution requires active "
              "resource ownership and, for Field payloads, an SNodeTree "
              "lifecycle guard");
  NdarrayLaunchLeases ndarray_leases;
  ExternalDenseStorageLaunchLeases external_leases;
  auto binding = resolve_dense_storage_descriptor(
      argument.descriptor(), ndarray_leases, external_leases, &argument);
  TI_ERROR_IF(!ndarray_leases.empty() || !external_leases.empty(),
              "Graph runtime storage was not retained before signature "
              "resolution");
  return binding;
}

void Program::validate_ndarrays_for_external_submission(
    const std::vector<const Ndarray *> &views) {
  validate_ndarrays_for_external_submission(views.data(), views.size());
}

void Program::validate_ndarrays_for_external_submission(
    const Ndarray *const *views,
    std::size_t count) {
  // Callers keep the submission guard alive through a synchronous external
  // operation. The temporary leases prove that every view still belongs to
  // this Program; the live-view lease then remains protected from retire by
  // the same gate until the operation completes.
  // Graph, Texture and GGUI callers hold runtime_resource_submission_mutex_
  // across validation and the synchronous operation.
  // The caller's submission guard keeps the live-view map stable through the
  // external operation. Debug readers may run concurrently, but all writers
  // also require the submission gate before taking ndarray_lifecycle_mutex_.
  TI_ERROR_IF(!ndarray_resources_open_,
              "Cannot use an Ndarray after Program finalize");
  for (std::size_t i = 0; i < count; ++i) {
    const Ndarray *view = views[i];
    if (view == nullptr) {
      continue;
    }
    const auto found = ndarray_views_.find(view);
    TI_ERROR_IF(found == ndarray_views_.end() ||
                    found->second.handle != view->runtime_resource_handle(),
                "Runtime operation references a stale or retired Ndarray");
    TI_ASSERT(found->second.lease && found->second.lease.get() == view &&
              found->second.lease.handle() == found->second.handle);
  }
}

void Program::retain_textures_for_external_submission(
    const std::vector<const Texture *> &views) {
  retain_textures_for_external_submission(views.data(), views.size());
}

void Program::retain_textures_for_external_submission(
    const Texture *const *views,
    std::size_t count) {
  if (!arch_is_gpu(compile_config().arch)) {
    validate_textures_for_external_submission(views, count);
    return;
  }
  auto leases = acquire_texture_leases(views, count);
  if (!leases.empty()) {
    pin_texture_launch_leases(leases);
  }
}

void Program::validate_textures_for_external_submission(
    const std::vector<const Texture *> &views) {
  validate_textures_for_external_submission(views.data(), views.size());
}

void Program::validate_textures_for_external_submission(
    const Texture *const *views,
    std::size_t count) {
  TI_ERROR_IF(!texture_resources_open_,
              "Cannot use a Texture after Program finalize");
  for (std::size_t i = 0; i < count; ++i) {
    const Texture *view = views[i];
    if (view == nullptr) {
      continue;
    }
    const auto found = texture_views_.find(view);
    TI_ERROR_IF(found == texture_views_.end() ||
                    found->second.handle != view->runtime_resource_handle(),
                "Runtime operation references a stale or retired Texture");
    TI_ASSERT(found->second.lease && found->second.lease.get() == view &&
              found->second.lease.handle() == found->second.handle);
  }
}

void Program::resolve_ndarray_launch_context(LaunchContextBuilder &ctx) {
  if (ctx.ndarray_ptrs.empty()) {
    return;
  }
  // Both callers hold runtime_resource_submission_mutex_: ordinary launch owns
  // it in launch_kernel_impl(), while AOT Graph owns an explicit transaction.
  TI_ERROR_IF(!ndarray_resources_open_,
              "Cannot resolve an Ndarray after Program finalize");

  auto resolve_view = [&](Program *owner, const Ndarray *view,
                          RuntimeResourceHandle expected_handle)
      -> const Ndarray * {
    if (view == nullptr) {
      return nullptr;
    }
    TI_ERROR_IF(owner != this,
                "Kernel launch references an Ndarray from another Program");
    if (ordinary_owned_ndarray_fast_path_enabled_) {
      TI_ERROR_IF(expected_handle.index >= ndarray_view_slots_.size(),
                  "Kernel launch references a stale or retired Ndarray");
      const auto &slot = ndarray_view_slots_[expected_handle.index];
      TI_ERROR_IF(slot.view != view || slot.handle != expected_handle ||
                      slot.resource == nullptr,
                  "Kernel launch references a stale or retired Ndarray");
      TI_ASSERT(slot.resource->lease &&
                slot.resource->lease.get() == view &&
                slot.resource->handle == expected_handle);
      if (ordinary_launch_attribution_.enabled) {
        ordinary_launch_attribution_.ndarray_slot_validations.fetch_add(
            1, std::memory_order_relaxed);
      }
      return slot.resource->lease.get();
    }
    const auto found = ndarray_views_.find(view);
    if (ordinary_launch_attribution_.enabled) {
      ordinary_launch_attribution_.ndarray_map_lookups.fetch_add(
          1, std::memory_order_relaxed);
    }
    TI_ERROR_IF(found == ndarray_views_.end() ||
                    found->second.handle != expected_handle,
                "Kernel launch references a stale or retired Ndarray");
    TI_ASSERT(found->second.lease && found->second.lease.get() == view);
    return found->second.lease.get();
  };

  for (const auto &resource_ref : ctx.ndarray_ptrs) {
    const Ndarray *data = resolve_view(resource_ref.owner, resource_ref.data,
                                       resource_ref.data_handle);
    TI_ASSERT(data != nullptr);
    const Ndarray *grad = resolve_view(resource_ref.owner, resource_ref.grad,
                                       resource_ref.grad_handle);
    (void)data;
    (void)grad;
  }
}

void Program::resolve_ndarray_launch_context_under_guard(
    LaunchContextBuilder &ctx) {
  // Caller owns runtime_resource_submission_mutex_ continuously from before
  // handle capture until backend submission. That prevents view retirement,
  // so repeating the lifecycle-map lookup per dispatch would add no safety.
  auto resolve_view = [&](Program *owner, const Ndarray *view,
                          RuntimeResourceHandle expected_handle)
      -> const Ndarray * {
    if (view == nullptr) {
      return nullptr;
    }
    TI_ERROR_IF(owner != this || !expected_handle,
                "Graph launch references an invalid Ndarray owner");
    return view;
  };

  for (const auto &ref : ctx.ndarray_ptrs) {
    const Ndarray *data = resolve_view(ref.owner, ref.data, ref.data_handle);
    TI_ASSERT(data != nullptr);
    const Ndarray *grad = resolve_view(ref.owner, ref.grad, ref.grad_handle);
    (void)data;
    (void)grad;
  }
}

void Program::resolve_runtime_storage_launch_context_under_guard(
    LaunchContextBuilder &ctx) {
  if (ctx.dense_storage_ptrs.empty()) {
    return;
  }
  NdarrayLaunchLeases ndarray_leases;
  ExternalDenseStorageLaunchLeases external_leases;
  resolve_dense_storage_launch_context(ctx, ndarray_leases, external_leases);
  TI_ERROR_IF(!ndarray_leases.empty() || !external_leases.empty(),
              "Graph runtime storage arguments were not retained before "
              "dispatch");
}

void Program::resolve_texture_launch_context(LaunchContextBuilder &ctx) {
  if (ctx.texture_ptrs.empty()) {
    return;
  }
  TI_ERROR_IF(!texture_resources_open_,
              "Cannot resolve a Texture after Program finalize");
  for (const auto &ref : ctx.texture_ptrs) {
    TI_ERROR_IF(ref.owner != this,
                "Kernel launch references a Texture from another Program");
    TI_ERROR_IF(ref.handle.index >= texture_view_slots_.size(),
                "Kernel launch references a stale or retired Texture");
    const auto &slot = texture_view_slots_[ref.handle.index];
    TI_ERROR_IF(slot.view != ref.texture || slot.handle != ref.handle,
                "Kernel launch references a stale or retired Texture");
  }
}

void Program::resolve_texture_launch_context_under_guard(
    LaunchContextBuilder &ctx) {
  for (const auto &ref : ctx.texture_ptrs) {
    TI_ERROR_IF(ref.owner != this || !ref.handle,
                "Graph launch references an invalid Texture owner");
  }
}

Program::NdarrayLaunchLeases Program::acquire_ndarray_launch_leases(
    LaunchContextBuilder &ctx) {
  NdarrayLaunchLeases leases;
  // launch_kernel_impl() owns runtime_resource_submission_mutex_. It excludes
  // create/retire/close and in-flight-map mutation for this entire validation
  // and backend submission interval; avoid a second mutex on every kernel.
  TI_ERROR_IF(!ndarray_resources_open_,
              "Cannot launch a kernel with Ndarray after Program finalize");

  auto acquire_view = [&](Program *owner, const Ndarray *view,
                           RuntimeResourceHandle expected_handle) {
    if (view == nullptr) {
      return;
    }
    TI_ERROR_IF(owner != this,
                "Kernel launch references an Ndarray from another Program");
    const NdarrayResourceView *resource_view = nullptr;
    if (ordinary_owned_ndarray_fast_path_enabled_) {
      TI_ERROR_IF(expected_handle.index >= ndarray_view_slots_.size(),
                  "Kernel launch references a stale or retired Ndarray");
      const auto &slot = ndarray_view_slots_[expected_handle.index];
      TI_ERROR_IF(slot.view != view || slot.handle != expected_handle ||
                      slot.resource == nullptr,
                  "Kernel launch references a stale or retired Ndarray");
      resource_view = slot.resource;
      if (ordinary_launch_attribution_.enabled) {
        ordinary_launch_attribution_.ndarray_slot_validations.fetch_add(
            1, std::memory_order_relaxed);
      }
    } else {
      const auto found = ndarray_views_.find(view);
      if (ordinary_launch_attribution_.enabled) {
        ordinary_launch_attribution_.ndarray_map_lookups.fetch_add(
            1, std::memory_order_relaxed);
      }
      TI_ERROR_IF(found == ndarray_views_.end() ||
                      found->second.handle != expected_handle,
                  "Kernel launch references a stale or retired Ndarray");
      resource_view = &found->second;
    }
    TI_ASSERT(resource_view->lease && resource_view->lease.get() == view &&
              resource_view->lease.handle() == expected_handle);
    if (leases.contains(expected_handle)) {
      return;
    }
    if (ndarray_inflight_leases_.find(ndarray_lease_key(expected_handle)) !=
        ndarray_inflight_leases_.end()) {
      if (ordinary_launch_attribution_.enabled) {
        ordinary_launch_attribution_.ndarray_inflight_reuses.fetch_add(
            1, std::memory_order_relaxed);
      }
      return;
    }
    auto lease = resource_view->lease.clone();
    TI_ERROR_IF(!lease, "Kernel launch could not clone its Ndarray lease");
    if (ordinary_launch_attribution_.enabled) {
      ordinary_launch_attribution_.ndarray_lease_clones.fetch_add(
          1, std::memory_order_relaxed);
    }
    leases.add(std::move(lease));
  };

  for (const auto &resource_ref : ctx.ndarray_ptrs) {
    acquire_view(resource_ref.owner, resource_ref.data,
                 resource_ref.data_handle);
    acquire_view(resource_ref.owner, resource_ref.grad,
                 resource_ref.grad_handle);
  }
  return leases;
}

storage::ResolvedDenseBinding Program::resolve_dense_storage_descriptor(
    const storage::DenseStorageDescriptor &descriptor,
    NdarrayLaunchLeases &ndarray_leases,
    ExternalDenseStorageLaunchLeases &external_leases,
    const storage::RuntimeStorageArgument *runtime_argument) {
  TI_ERROR_IF(!ndarray_resources_open_,
              "Cannot resolve dense storage after Program finalize");

  if (runtime_argument != nullptr) {
    const auto &qualification = runtime_argument->qualification();
    TI_ERROR_IF(&runtime_argument->descriptor() != &descriptor,
                "Runtime storage argument descriptor identity was lost");
    TI_ERROR_IF(
        runtime_argument->requirement().backend != compile_config().arch,
        "Runtime storage argument was qualified for backend {} but "
        "submitted to {}",
        arch_name(runtime_argument->requirement().backend),
        arch_name(compile_config().arch));
    TI_ERROR_IF(!qualification.capabilities.bindable ||
                    !qualification.capabilities.zero_copy_qualified,
                "Runtime storage argument is not zero-copy bindable: {}",
                storage::to_string(qualification.reason));
  }

  auto find_snode = [](SNode *root, int target_id) -> SNode * {
    auto visit = [&](auto &&self, SNode *node) -> SNode * {
      if (node == nullptr) {
        return nullptr;
      }
      if (node->id == target_id) {
        return node;
      }
      for (const auto &child : node->ch) {
        if (SNode *found = self(self, child.get())) {
          return found;
        }
      }
      return nullptr;
    };
    return visit(visit, root);
  };

  const auto &owner = descriptor.owner();
  const auto &properties = descriptor.properties();
  if (owner.kind == storage::StorageOwnerKind::kExternalManaged) {
    TI_ERROR_IF(owner.external_owner_domain !=
                    external_dense_storage_resources_.domain(),
                "External dense storage belongs to another Program");
  } else {
    TI_ERROR_IF(owner.program_domain != runtime_program_generation(),
                "Dense storage binding belongs to another Program generation");
  }
  const bool safe_positive_affine =
      !properties.has_negative_stride && properties.element_contiguous &&
      properties.uniqueness ==
          storage::StorageMappingUniqueness::kProvenUnique;
  TI_ERROR_IF((!properties.ndarray_abi_compatible && !safe_positive_affine) ||
                  properties.reachable_begin < 0 ||
                  properties.reachable_end < properties.reachable_begin ||
                  properties.reachable_begin != descriptor.byte_offset(),
              "Dense storage binding is not a supported unique dense range");
  TI_ERROR_IF(!properties.ndarray_abi_compatible &&
                  owner.kind == storage::StorageOwnerKind::kExternalManaged,
              "Affine external dense storage requires an external access "
              "epoch and is not supported for runtime affine bindings");

  storage::ResolvedDenseBinding binding;
  if (runtime_argument != nullptr) {
    binding.runtime_signature = runtime_argument->stable_signature();
    binding.synchronization_domain_identity =
        runtime_argument->synchronization_domain_identity();
    binding.capabilities = runtime_argument->qualification().capabilities;
  }
  binding.byte_offset = static_cast<std::uint64_t>(properties.reachable_begin);
  binding.byte_size = static_cast<std::uint64_t>(properties.reachable_end -
                                                 properties.reachable_begin);

  if (owner.kind == storage::StorageOwnerKind::kProgramNdarray) {
    const auto handle = owner.ndarray_handle;
    TI_ERROR_IF(handle.index >= ndarray_view_slots_.size(),
                "Dense storage binding references a stale or retired Ndarray");
    const auto &slot = ndarray_view_slots_[handle.index];
    TI_ERROR_IF(slot.view == nullptr || slot.handle != handle ||
                    slot.resource == nullptr,
                "Dense storage binding references a stale or retired Ndarray");
    const Ndarray *array = slot.view;
    if (arch_is_gpu(compile_config().arch) &&
        ndarray_leases.find(handle) == nullptr &&
        ndarray_inflight_leases_.find(ndarray_lease_key(handle)) ==
            ndarray_inflight_leases_.end()) {
      TI_ASSERT(slot.resource->handle == handle && slot.resource->lease);
      auto lease = slot.resource->lease.clone();
      TI_ERROR_IF(!lease,
                  "Dense storage binding could not clone its Ndarray lease");
      ndarray_leases.add(std::move(lease));
    }
    TI_ASSERT(array != nullptr);
    const std::size_t element_size = array->get_element_size();
    const std::size_t element_count = array->get_nelement();
    TI_ERROR_IF(element_size != 0 &&
                    element_count > (std::numeric_limits<std::size_t>::max)() /
                                        element_size,
                "Ndarray byte span overflow while resolving storage");
    const std::size_t allocation_bytes = element_size * element_count;
    TI_ERROR_IF(binding.byte_offset > allocation_bytes ||
                    binding.byte_size > allocation_bytes - binding.byte_offset,
                "Dense storage range exceeds its Ndarray allocation");
    binding.allocation = array->get_device_allocation();
    dense_storage_ndarray_bindings_.fetch_add(1, std::memory_order_relaxed);
  } else if (owner.kind == storage::StorageOwnerKind::kSNodePayload) {
    const int tree_id = owner.tree.tree_id;
    TI_ERROR_IF(
        tree_id < 0 ||
            static_cast<std::size_t>(tree_id) >= snode_trees_.size() ||
            static_cast<std::size_t>(tree_id) >= snode_tree_active_.size() ||
            !snode_tree_active_[tree_id] || snode_trees_[tree_id] == nullptr,
        "Dense storage binding references a retired SNodeTree");
    SNodeTree *tree = snode_trees_[tree_id].get();
    TI_ERROR_IF(tree->generation() != owner.tree.generation,
                "Dense storage binding references a retired SNodeTree "
                "generation");
    TI_ERROR_IF(tree->layout_fingerprint() != owner.tree.layout_fingerprint,
                "Dense storage binding SNodeTree layout has changed");
    SNode *anchor = find_snode(tree->root(), owner.anchor_snode_id);
    TI_ERROR_IF(anchor == nullptr || anchor->type != SNodeType::place,
                "Dense storage binding anchor is no longer available");
    const DevicePtr field_ptr = get_dense_field_device_ptr(anchor);
    const std::uint64_t payload_begin = owner.snode_payload_byte_begin;
    const std::uint64_t payload_end = owner.snode_payload_byte_end;
    TI_ERROR_IF(field_ptr.offset != payload_begin,
                "Dense storage binding base no longer matches its SNode "
                "layout");
    TI_ERROR_IF(payload_end < payload_begin ||
                    binding.byte_offset < payload_begin ||
                    binding.byte_offset > payload_end ||
                    binding.byte_size > payload_end - binding.byte_offset,
                "Dense storage range exceeds its SNode payload");
    binding.allocation = DeviceAllocation{field_ptr.device, field_ptr.alloc_id};
    dense_storage_field_bindings_.fetch_add(1, std::memory_order_relaxed);
  } else if (owner.kind == storage::StorageOwnerKind::kExternalManaged) {
    const auto handle = external_dense_storage_handle(owner);
    TI_ERROR_IF(!handle, "External dense storage belongs to another Program");
    ExternalDenseStorageResource *resource = external_leases.find(handle);
    if (resource == nullptr) {
      // An inflight lease keeps retired storage physically alive, but must not
      // make a retired generation eligible for a new submission. Acquire from
      // the registry first to validate the current live state, then reuse the
      // existing ownership only as a launch-time optimization.
      auto [result, lease] = external_dense_storage_resources_.acquire(handle);
      TI_ERROR_IF(
          result != ExternalDenseStorageRegistry::Result::kSuccess || !lease,
          "External dense storage is stale or retired");
      const auto inflight = external_dense_storage_inflight_leases_.find(
          external_dense_storage_lease_key(handle));
      if (inflight != external_dense_storage_inflight_leases_.end()) {
        resource = inflight->second.get();
      } else {
        external_leases.add(std::move(lease));
        resource = external_leases.find(handle);
      }
    }
    TI_ERROR_IF(resource == nullptr,
                "External dense storage lease was not acquired");
    if (runtime_argument != nullptr &&
        runtime_argument->synchronization_domain_identity() != 0) {
      TI_ERROR_IF(!resource->synchronization_domain ||
                      resource->synchronization_domain->identity() !=
                          runtime_argument->synchronization_domain_identity(),
                  "External dense storage synchronization domain is stale or "
                  "mismatched");
    }
    external_leases.track_synchronization_domain(
        resource->synchronization_domain);
    TI_ERROR_IF(binding.byte_offset > resource->allocation_bytes ||
                    binding.byte_size >
                        resource->allocation_bytes - binding.byte_offset,
                "Dense storage range exceeds its external allocation");
    binding.allocation = resource->allocation;
    dense_storage_external_bindings_.fetch_add(1, std::memory_order_relaxed);
  } else {
    TI_ERROR("Dense storage binding does not accept this owner kind");
  }

  binding.valid = true;
  dense_storage_resolved_bindings_.fetch_add(1, std::memory_order_relaxed);
  dense_storage_resolved_bytes_.fetch_add(binding.byte_size,
                                          std::memory_order_relaxed);
  return binding;
}

void Program::resolve_dense_storage_launch_context(
    LaunchContextBuilder &ctx,
    NdarrayLaunchLeases &ndarray_leases,
    ExternalDenseStorageLaunchLeases &external_leases) {
  bool resolved_here = false;
  for (std::size_t i = 0; i < ctx.dense_storage_ptrs.size(); ++i) {
    auto &resource = ctx.dense_storage_ptrs[i];
    TI_ERROR_IF(resource.descriptor == nullptr,
                "Dense storage launch context lost its descriptor");
    if (resource.resolved.valid) {
      continue;
    }
    ctx.set_resolved_dense_storage(
        i, resolve_dense_storage_descriptor(*resource.descriptor,
                                            ndarray_leases, external_leases,
                                            resource.runtime_argument));
    resolved_here = true;
  }
  if (resolved_here) {
    dense_storage_direct_submissions_.fetch_add(1, std::memory_order_relaxed);
  }
}

void Program::with_resolved_dense_storage_bindings(
    const std::vector<const storage::DenseStorageDescriptor *> &descriptors,
    const DenseStorageBindingCallback &callback) {
  ensure_runtime_submission_allowed("dense storage submission");
  TI_ERROR_IF(descriptors.empty() || !callback,
              "Dense storage submission requires descriptors and a callback");
  std::optional<SNodeTreeLifecycleReadGuard> lifecycle_guard;
  if (active_snode_tree_lifecycle_program != this) {
    lifecycle_guard.emplace(acquire_snode_tree_lifecycle_read_guard());
  }
  std::lock_guard<std::recursive_mutex> resource_submission_lock(
      runtime_resource_submission_mutex_);
  NdarrayLaunchLeases ndarray_leases;
  ExternalDenseStorageLaunchLeases external_leases;
  std::vector<storage::ResolvedDenseBinding> bindings;
  bindings.reserve(descriptors.size());
  for (const auto *descriptor : descriptors) {
    TI_ERROR_IF(descriptor == nullptr,
                "Dense storage submission received a null descriptor");
    bindings.push_back(resolve_dense_storage_descriptor(
        *descriptor, ndarray_leases, external_leases));
  }
  dense_storage_direct_submissions_.fetch_add(1, std::memory_order_relaxed);
  // Resolved allocations are submission-scoped capabilities. The callback
  // may enqueue backend work, but must not retain a binding after returning.
  // GPU owner leases are pinned before this transaction unlocks and are then
  // released by RuntimeCompletion or synchronize/finalize.
  ExternalAccessEpoch external_access_epoch;
  begin_external_access_epoch(external_access_epoch, external_leases);
  try {
    callback(bindings.data(), bindings.size());
    external_access_epoch.release();
    if (arch_is_gpu(compile_config().arch)) {
      if (!ndarray_leases.empty()) {
        pin_ndarray_launch_leases(ndarray_leases);
      }
      if (!external_leases.empty()) {
        pin_external_dense_storage_launch_leases(external_leases);
      }
    }
  } catch (...) {
    const std::exception_ptr submission_error = std::current_exception();
    try {
      external_access_epoch.release();
    } catch (...) {
    }
    if (arch_is_gpu(compile_config().arch)) {
      try {
        program_impl_->synchronize();
        release_completed_ndarray_leases();
        release_completed_external_dense_storage_leases();
        release_completed_vulkan_native_resources();
      } catch (...) {
      }
    }
    std::rethrow_exception(submission_error);
  }
}

void Program::with_resolved_runtime_storage_arguments(
    const std::vector<const storage::RuntimeStorageArgument *> &arguments,
    const DenseStorageBindingCallback &callback) {
  ensure_runtime_submission_allowed("runtime storage submission");
  TI_ERROR_IF(arguments.empty() || !callback,
              "Runtime storage submission requires arguments and a callback");
  std::optional<SNodeTreeLifecycleReadGuard> lifecycle_guard;
  if (active_snode_tree_lifecycle_program != this) {
    lifecycle_guard.emplace(acquire_snode_tree_lifecycle_read_guard());
  }
  std::lock_guard<std::recursive_mutex> resource_submission_lock(
      runtime_resource_submission_mutex_);
  NdarrayLaunchLeases ndarray_leases;
  ExternalDenseStorageLaunchLeases external_leases;
  std::vector<storage::ResolvedDenseBinding> bindings;
  bindings.reserve(arguments.size());
  for (const auto *argument : arguments) {
    TI_ERROR_IF(argument == nullptr,
                "Runtime storage submission received a null argument");
    bindings.push_back(resolve_dense_storage_descriptor(
        argument->descriptor(), ndarray_leases, external_leases, argument));
  }
  dense_storage_direct_submissions_.fetch_add(1, std::memory_order_relaxed);
  ExternalAccessEpoch external_access_epoch;
  begin_external_access_epoch(external_access_epoch, external_leases);
  try {
    callback(bindings.data(), bindings.size());
    external_access_epoch.release();
    if (arch_is_gpu(compile_config().arch)) {
      if (!ndarray_leases.empty()) {
        pin_ndarray_launch_leases(ndarray_leases);
      }
      if (!external_leases.empty()) {
        pin_external_dense_storage_launch_leases(external_leases);
      }
    }
  } catch (...) {
    const std::exception_ptr submission_error = std::current_exception();
    try {
      external_access_epoch.release();
    } catch (...) {
    }
    if (arch_is_gpu(compile_config().arch)) {
      try {
        program_impl_->synchronize();
        release_completed_ndarray_leases();
        release_completed_external_dense_storage_leases();
        release_completed_vulkan_native_resources();
      } catch (...) {
      }
    }
    std::rethrow_exception(submission_error);
  }
}

intptr_t Program::get_dense_storage_data_ptr_as_int(
    const storage::ResolvedDenseBinding &binding) {
  TI_ERROR_IF(!binding.valid,
              "Cannot access an unresolved dense storage binding");
  if (!arch_is_cpu(compile_config().arch) &&
      compile_config().arch != Arch::cuda &&
      compile_config().arch != Arch::amdgpu) {
    return 0;
  }
  auto *base = program_impl_->get_device_alloc_info_ptr(binding.allocation);
  TI_ERROR_IF(base == nullptr && binding.byte_size != 0,
              "Dense storage binding resolved to a null device address");
  if (base == nullptr) {
    return 0;
  }
  return reinterpret_cast<intptr_t>(
      reinterpret_cast<std::uint8_t *>(base) + binding.byte_offset);
}

Program::TextureLaunchLeases Program::acquire_texture_launch_leases(
    LaunchContextBuilder &ctx) {
  TextureLaunchLeases leases;
  TI_ERROR_IF(!texture_resources_open_,
              "Cannot launch a kernel with Texture after Program finalize");
  for (const auto &ref : ctx.texture_ptrs) {
    TI_ERROR_IF(ref.owner != this,
                "Kernel launch references a Texture from another Program");
    TI_ERROR_IF(ref.handle.index >= texture_view_slots_.size(),
                "Kernel launch references a stale or retired Texture");
    const auto &slot = texture_view_slots_[ref.handle.index];
    TI_ERROR_IF(slot.view != ref.texture || slot.handle != ref.handle,
                "Kernel launch references a stale or retired Texture");
    if (leases.contains(ref.handle) ||
        texture_inflight_leases_.find(texture_lease_key(ref.handle)) !=
            texture_inflight_leases_.end()) {
      continue;
    }
    const auto found = texture_views_.find(ref.texture);
    TI_ASSERT(found != texture_views_.end() &&
              found->second.handle == ref.handle);
    const TextureResourceView &resource_view = found->second;
    TI_ASSERT(resource_view.lease &&
              resource_view.lease.get() == ref.texture &&
              resource_view.lease.handle() == ref.handle);
    auto lease = resource_view.lease.clone();
    TI_ERROR_IF(!lease, "Kernel launch could not clone its Texture lease");
    leases.add(std::move(lease));
  }
  return leases;
}

void Program::pin_argpack_launch_leases(ArgPackLaunchLeases &leases) {
  std::lock_guard<std::mutex> lock(argpack_lifecycle_mutex_);
  auto pin_lease = [&](ArgPackResourceLease &lease) {
    if (!lease) {
      return;
    }
    const std::uint64_t key = argpack_lease_key(lease.handle());
    if (argpack_inflight_leases_.find(key) ==
        argpack_inflight_leases_.end()) {
      argpack_inflight_leases_.emplace(key, std::move(lease));
    }
  };
  for (std::size_t i = 0; i < leases.inline_count_; ++i) {
    pin_lease(*leases.inline_leases_[i]);
  }
  for (auto &lease : leases.overflow_leases_) {
    pin_lease(lease);
  }
}

void Program::pin_ndarray_launch_leases(NdarrayLaunchLeases &leases) {
  std::lock_guard<std::mutex> lock(ndarray_lifecycle_mutex_);
  auto pin_lease = [&](NdarrayResourceLease &lease) {
    if (!lease) {
      return;
    }
    const std::uint64_t key = ndarray_lease_key(lease.handle());
    if (ndarray_inflight_leases_.find(key) ==
        ndarray_inflight_leases_.end()) {
      ndarray_inflight_leases_.emplace(key, std::move(lease));
      if (ordinary_launch_attribution_.enabled) {
        ordinary_launch_attribution_.ndarray_pins.fetch_add(
            1, std::memory_order_relaxed);
      }
    }
  };
  for (std::size_t i = 0; i < leases.inline_count_; ++i) {
    pin_lease(*leases.inline_leases_[i]);
  }
  for (auto &lease : leases.overflow_leases_) {
    pin_lease(lease);
  }
}

void Program::pin_texture_launch_leases(TextureLaunchLeases &leases) {
  std::lock_guard<std::mutex> lock(texture_lifecycle_mutex_);
  auto pin_lease = [&](TextureResourceLease &lease) {
    if (!lease) {
      return;
    }
    const std::uint64_t key = texture_lease_key(lease.handle());
    if (texture_inflight_leases_.find(key) ==
        texture_inflight_leases_.end()) {
      texture_inflight_leases_.emplace(key, std::move(lease));
    }
  };
  for (std::size_t i = 0; i < leases.inline_count_; ++i) {
    pin_lease(*leases.inline_leases_[i]);
  }
  for (auto &lease : leases.overflow_leases_) {
    pin_lease(lease);
  }
}

void Program::pin_external_dense_storage_launch_leases(
    ExternalDenseStorageLaunchLeases &leases) {
  std::lock_guard<std::mutex> lock(external_dense_storage_lifecycle_mutex_);
  auto pin_lease = [&](ExternalDenseStorageLease &lease) {
    if (!lease) {
      return;
    }
    const auto &domain = lease.get()->synchronization_domain;
    if (domain && domain->retirement_waits_for_consumer()) {
      return;
    }
    const auto key = external_dense_storage_lease_key(lease.handle());
    if (external_dense_storage_inflight_leases_.find(key) ==
        external_dense_storage_inflight_leases_.end()) {
      external_dense_storage_inflight_leases_.emplace(key, std::move(lease));
    }
  };
  for (std::size_t i = 0; i < leases.inline_count_; ++i) {
    pin_lease(*leases.inline_leases_[i]);
  }
  for (auto &lease : leases.overflow_leases_) {
    pin_lease(lease);
  }
}

void Program::begin_external_access_epoch(
    ExternalAccessEpoch &epoch,
    const ExternalDenseStorageLaunchLeases &leases) {
  const auto &domains = leases.synchronization_domains();
  if (domains.empty()) {
    return;
  }
  ExternalStreamDomain stream;
  if (arch_is_cpu(compile_config().arch)) {
    stream = ExternalStreamDomain::host(runtime_program_generation());
  } else if (compile_config().arch == Arch::cuda) {
    stream = ExternalStreamDomain::cuda(runtime_program_generation(), 1);
  } else {
    TI_ERROR("External synchronization is unsupported on backend {}",
             arch_name(compile_config().arch));
  }
  epoch = ExternalAccessEpoch(domains, stream);
}

void Program::release_completed_argpack_leases() {
  ArgPackInflightLeaseMap completed;
  {
    std::lock_guard<std::mutex> lock(argpack_lifecycle_mutex_);
    completed.swap(argpack_inflight_leases_);
  }
  completed.clear();
}

void Program::release_completed_ndarray_leases() {
  NdarrayInflightLeaseMap completed;
  {
    std::lock_guard<std::mutex> lock(ndarray_lifecycle_mutex_);
    completed.swap(ndarray_inflight_leases_);
  }
  completed.clear();
}

void Program::release_completed_texture_leases() {
  TextureInflightLeaseMap completed;
  {
    std::lock_guard<std::mutex> lock(texture_lifecycle_mutex_);
    completed.swap(texture_inflight_leases_);
  }
  completed.clear();
}

void Program::release_completed_external_dense_storage_leases() {
  ExternalDenseStorageInflightLeaseMap completed;
  {
    std::lock_guard<std::mutex> lock(external_dense_storage_lifecycle_mutex_);
    completed.swap(external_dense_storage_inflight_leases_);
  }
  completed.clear();
}

void Program::pin_cuda_provider_plan(
    std::shared_ptr<CudaProviderCompletionResource> plan) {
  TI_ASSERT(plan != nullptr);
  const auto *identity = plan.get();
  const std::uint64_t token = plan->submission_retirement_token();
  const auto found = cuda_provider_inflight_plans_.find(identity);
  if (found == cuda_provider_inflight_plans_.end()) {
    cuda_provider_inflight_plans_.emplace(
        identity, CudaProviderInflightPlan{std::move(plan), token});
  } else if (token != 0) {
    found->second.retirement_token = token;
  }
}

void Program::release_completed_cuda_provider_plans() {
  CudaProviderInflightPlanMap completed;
  completed.swap(cuda_provider_inflight_plans_);
  for (auto &item : completed) {
    item.second.resource->on_submission_retired(item.second.retirement_token);
  }
  completed.clear();
}

void Program::release_completed_vulkan_native_resources() {
  // CUDA provider plans and Vulkan native objects share the same completion
  // retirement boundary even though their backend-specific ownership differs.
  release_completed_cuda_provider_plans();
  std::vector<void *> cuda_batch_events;
  {
    std::lock_guard<std::mutex> lock(cuda_batch_event_mutex_);
    cuda_batch_events.swap(cuda_batch_event_retirements_);
  }
  for (void *event : cuda_batch_events) {
    runtime_completion_cuda_event_pool_->release(event, /*reusable=*/true);
  }
  std::vector<std::shared_ptr<VulkanGraphicsPipelineResource>> graphics;
  std::vector<std::shared_ptr<VulkanTriangleRayScene>> ray_scenes;
  std::vector<std::shared_ptr<VulkanRayResource>> ray_resources;
  {
    std::lock_guard<std::mutex> lock(vulkan_graphics_pipeline_mutex_);
    graphics.swap(vulkan_graphics_pipeline_retirements_);
  }
  {
    std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
    ray_scenes.swap(vulkan_ray_scene_retirements_);
    ray_resources.swap(vulkan_ray_resource_retirements_);
  }
  graphics.clear();
  ray_scenes.clear();
  ray_resources.clear();
}

std::size_t Program::RuntimeCompletionResourceBatch::retained_resource_count(
    std::uint32_t kind) const noexcept {
  if (kind == kArgPackResourceKind) {
    return argpacks.size();
  }
  if (kind == kNdarrayResourceKind) {
    return ndarrays.size();
  }
  if (kind == kTextureResourceKind) {
    return textures.size();
  }
  if (kind == kExternalDenseStorageResourceKind) {
    return external_dense_storage.size();
  }
  if (kind == kCudaProviderPlanResourceKind) {
    return cuda_provider_plans.size();
  }
  if (kind == kVulkanGraphicsPipelineResourceKind) {
    return vulkan_graphics_pipelines.size();
  }
  if (kind == kVulkanRaySceneResourceKind) {
    return vulkan_ray_scenes.size();
  }
  if (kind == kVulkanRayResourceKind) {
    return vulkan_ray_resources.size();
  }
  return 0;
}

Program::RuntimeCompletionResourceBatch::~RuntimeCompletionResourceBatch() {
  for (auto &item : cuda_provider_plans) {
    item.second.resource->on_submission_retired(item.second.retirement_token);
  }
  if (cuda_event_pool) {
    for (void *event : cuda_batch_events) {
      cuda_event_pool->release(event, /*reusable=*/true);
    }
  }
}

void Program::pin_runtime_cuda_batch_events(std::vector<void *> events) {
  if (events.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(cuda_batch_event_mutex_);
  cuda_batch_event_retirements_.insert(cuda_batch_event_retirements_.end(),
                                       events.begin(), events.end());
}

std::shared_ptr<Program::RuntimeCompletionResourceBatch>
Program::detach_runtime_completion_resources() {
  bool has_resources = false;
  {
    std::lock_guard<std::mutex> lock(argpack_lifecycle_mutex_);
    has_resources = !argpack_inflight_leases_.empty();
  }
  if (!has_resources) {
    std::lock_guard<std::mutex> lock(ndarray_lifecycle_mutex_);
    has_resources = !ndarray_inflight_leases_.empty();
  }
  if (!has_resources) {
    std::lock_guard<std::mutex> lock(texture_lifecycle_mutex_);
    has_resources = !texture_inflight_leases_.empty();
  }
  if (!has_resources) {
    std::lock_guard<std::mutex> lock(external_dense_storage_lifecycle_mutex_);
    has_resources = !external_dense_storage_inflight_leases_.empty();
  }
  if (!has_resources) {
    std::lock_guard<std::mutex> lock(vulkan_graphics_pipeline_mutex_);
    has_resources = !vulkan_graphics_pipeline_retirements_.empty();
  }
  if (!has_resources) {
    has_resources = !cuda_provider_inflight_plans_.empty();
  }
  if (!has_resources) {
    std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
    has_resources = !vulkan_ray_scene_retirements_.empty() ||
                    !vulkan_ray_resource_retirements_.empty();
  }
  if (!has_resources) {
    std::lock_guard<std::mutex> lock(cuda_batch_event_mutex_);
    has_resources = !cuda_batch_event_retirements_.empty();
  }
  if (!has_resources) {
    return nullptr;
  }

  // Allocate the control block before moving any lease. A metadata allocation
  // failure therefore leaves the legacy synchronize-owned maps intact.
  auto batch = std::make_shared<RuntimeCompletionResourceBatch>();
  {
    std::lock_guard<std::mutex> lock(argpack_lifecycle_mutex_);
    batch->argpacks.swap(argpack_inflight_leases_);
  }
  {
    std::lock_guard<std::mutex> lock(ndarray_lifecycle_mutex_);
    batch->ndarrays.swap(ndarray_inflight_leases_);
  }
  {
    std::lock_guard<std::mutex> lock(texture_lifecycle_mutex_);
    batch->textures.swap(texture_inflight_leases_);
  }
  {
    std::lock_guard<std::mutex> lock(external_dense_storage_lifecycle_mutex_);
    batch->external_dense_storage.swap(external_dense_storage_inflight_leases_);
  }
  batch->cuda_provider_plans.swap(cuda_provider_inflight_plans_);
  {
    std::lock_guard<std::mutex> lock(vulkan_graphics_pipeline_mutex_);
    batch->vulkan_graphics_pipelines.swap(
        vulkan_graphics_pipeline_retirements_);
  }
  {
    std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
    batch->vulkan_ray_scenes.swap(vulkan_ray_scene_retirements_);
    batch->vulkan_ray_resources.swap(vulkan_ray_resource_retirements_);
  }
  {
    std::lock_guard<std::mutex> lock(cuda_batch_event_mutex_);
    batch->cuda_batch_events.swap(cuda_batch_event_retirements_);
  }
  if (!batch->cuda_batch_events.empty()) {
    batch->cuda_event_pool = runtime_completion_cuda_event_pool_;
  }
  TI_ASSERT(!batch->empty());
  return batch;
}

void Program::acquire_runtime_submission_reader() noexcept {
  for (;;) {
    const std::uint64_t previous =
        runtime_submission_gate_.fetch_add(1, std::memory_order_acquire);
    TI_ASSERT((previous & kRuntimeSubmissionReaderMask) !=
              kRuntimeSubmissionReaderMask);
    if ((previous & kRuntimeSubmissionWriterBit) == 0) {
      return;
    }
    runtime_submission_gate_.fetch_sub(1, std::memory_order_release);
    std::this_thread::yield();
  }
}

void Program::release_runtime_submission_reader() noexcept {
  const std::uint64_t previous =
      runtime_submission_gate_.fetch_sub(1, std::memory_order_release);
  TI_ASSERT((previous & kRuntimeSubmissionReaderMask) != 0);
}

void Program::acquire_runtime_submission_writer() noexcept {
  for (;;) {
    std::uint64_t observed =
        runtime_submission_gate_.load(std::memory_order_acquire);
    if ((observed & kRuntimeSubmissionWriterBit) != 0) {
      std::this_thread::yield();
      continue;
    }
    if (runtime_submission_gate_.compare_exchange_weak(
            observed, observed | kRuntimeSubmissionWriterBit,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
      break;
    }
  }
  while ((runtime_submission_gate_.load(std::memory_order_acquire) &
          kRuntimeSubmissionReaderMask) != 0) {
    std::this_thread::yield();
  }
}

void Program::release_runtime_submission_writer() noexcept {
  const std::uint64_t observed =
      runtime_submission_gate_.load(std::memory_order_relaxed);
  TI_ASSERT(observed == kRuntimeSubmissionWriterBit);
  runtime_submission_gate_.store(0, std::memory_order_release);
}

void Program::track_runtime_completion(
    const RuntimeCompletion &completion) {
  TI_ASSERT(completion.valid() && completion.has_backend_work());
  std::lock_guard<std::mutex> lock(runtime_completion_mutex_);
  runtime_completions_.push_back(completion);
}

void Program::collect_ready_runtime_completions(std::size_t max_polls) {
  std::lock_guard<std::mutex> lock(runtime_completion_mutex_);
  const std::size_t poll_count =
      std::min(max_polls, runtime_completions_.size());
  for (std::size_t i = 0; i < poll_count; ++i) {
    RuntimeCompletion completion = runtime_completions_.front();
    // Query before mutating the registry. A backend exception therefore keeps
    // the completion tracked for reset/synchronize, and appending before
    // popping gives the same guarantee if deque growth reports OOM.
    if (completion.done()) {
      runtime_completions_.pop_front();
    } else {
      runtime_completions_.push_back(std::move(completion));
      runtime_completions_.pop_front();
    }
  }
}

void Program::complete_all_runtime_completions() noexcept {
  std::deque<RuntimeCompletion> completed;
  {
    std::lock_guard<std::mutex> lock(runtime_completion_mutex_);
    completed.swap(runtime_completions_);
  }
  for (const auto &completion : completed) {
    completion.mark_completed();
  }
}

void Program::fail_all_runtime_completions(
    const std::string &reason) noexcept {
  std::deque<RuntimeCompletion> failed;
  {
    std::lock_guard<std::mutex> lock(runtime_completion_mutex_);
    failed.swap(runtime_completions_);
  }
  for (const auto &completion : failed) {
    completion.invalidate_and_release(reason);
  }
}

void Program::attach_runtime_fault_reporter() {
  TI_ASSERT(program_impl_ != nullptr);
  Device *compute_device = program_impl_->get_compute_device();
  Device *graphics_device = program_impl_->get_graphics_device();
  if (compute_device != nullptr) {
    compute_device->set_backend_fault_reporter(runtime_fault_domain_);
  }
  if (graphics_device != nullptr && graphics_device != compute_device) {
    graphics_device->set_backend_fault_reporter(runtime_fault_domain_);
  }
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    CUDADriver::get_instance_without_context().set_fault_reporter(
        runtime_fault_domain_);
  }
#endif
}

void Program::detach_runtime_fault_reporter() noexcept {
  Device::clear_backend_fault_reporter(runtime_fault_domain_);
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    CUDADriver::get_instance_without_context().clear_fault_reporter(
        runtime_fault_domain_);
  }
#endif
}

void Program::debug_inject_runtime_fault(
    std::int64_t backend_code,
    const std::string &operation,
    const std::string &message) {
  const std::uint64_t sequence =
      next_runtime_completion_sequence_.load(std::memory_order_acquire);
  runtime_fault_domain_->report_fatal(
      {compile_config().arch, backend_code, sequence, operation, message});
}

std::size_t Program::runtime_completion_resource_count(
    std::uint32_t kind) const noexcept {
  std::lock_guard<std::mutex> lock(runtime_completion_mutex_);
  std::size_t count = 0;
  for (const auto &completion : runtime_completions_) {
    count += completion.retained_resource_count(kind);
  }
  return count;
}

RuntimeCompletion Program::record_runtime_completion(
    StreamGpuTiming gpu_timing,
    std::vector<RuntimeGpuRegionTiming> gpu_region_timings) {
  ensure_runtime_submission_allowed("runtime completion recording");
  // Bound host polling work, not the number of valid in-flight submissions.
  // Synchronize/reset/finalize fully drain the registry; until then the
  // backend and available device memory define the natural backlog limit.
  collect_ready_runtime_completions(kRuntimeCompletionPollBudget);

  RuntimeCompletion result;
  {
    // Lock order: resource submission -> completion submission. Kernel/Graph
    // paths acquire the corresponding shared scope only after their existing
    // SNode/resource guards, so completion recording cannot invert the order.
    std::lock_guard<std::recursive_mutex> resource_lock(
        runtime_resource_submission_mutex_);
    // New submissions acquire the reader gate from this point onward. A
    // submission that observed the old disabled state may still overlap this
    // writer; its successful dirty-bit publication linearizes it after this
    // completion, which is valid because the two host calls overlap.
    runtime_completion_tracking_enabled_.store(true,
                                               std::memory_order_release);
    RuntimeSubmissionWriteScope submission_scope(this);
    TI_ERROR_IF(finalized_,
                "Cannot record a completion after Program finalize");

    if (!runtime_submission_pending_.exchange(false,
                                              std::memory_order_acq_rel)) {
      const std::uint64_t next =
          next_runtime_completion_sequence_.load(std::memory_order_relaxed);
      return RuntimeCompletion::completed(
          compile_config().arch, runtime_completion_domain_,
          next > 1 ? next - 1 : 0, runtime_fault_domain_);
    }

    const std::uint64_t previous_submission_epoch =
        runtime_submission_epoch_.fetch_add(1, std::memory_order_relaxed);
    TI_ASSERT(previous_submission_epoch !=
              (std::numeric_limits<std::uint64_t>::max)());
    const std::uint64_t submission_epoch = previous_submission_epoch + 1;

    const std::uint64_t sequence =
        next_runtime_completion_sequence_.fetch_add(
            1, std::memory_order_relaxed);
    TI_ASSERT(sequence != 0 &&
              sequence != (std::numeric_limits<std::uint64_t>::max)());

    try {
      if (arch_is_cpu(compile_config().arch)) {
        // CPU launches are synchronous. Preserve that contract and reuse the
        // completed singleton instead of pretending to run a background task.
        program_impl_->synchronize();
        result = RuntimeCompletion::completed(
            compile_config().arch, runtime_completion_domain_, sequence,
            runtime_fault_domain_);
      } else if (compile_config().arch == Arch::cuda) {
        // Existing Driver API symbols are loaded dynamically; no cudart or
        // Toolkit-versioned runtime dependency is introduced.
        result = RuntimeCompletion::from_cuda_stream(
            runtime_completion_domain_, sequence, nullptr,
            runtime_fault_domain_, runtime_completion_cuda_event_pool_,
            std::move(gpu_timing),
            std::move(gpu_region_timings));
      } else if (compile_config().arch == Arch::vulkan) {
        // A work epoch exists, so flush() intentionally records a fence even
        // when the work came from a replay/native path outside current_cmdlist.
        result = RuntimeCompletion::from_stream_semaphore(
            Arch::vulkan, runtime_completion_domain_, sequence,
            program_impl_->flush(), runtime_fault_domain_,
            std::move(gpu_timing), std::move(gpu_region_timings));
      } else {
        // F2's supported contract is CPU/CUDA/Vulkan. Other compiled backends
        // retain their legacy synchronous fallback without a fake token.
        program_impl_->synchronize();
        result = RuntimeCompletion::completed(
            compile_config().arch, runtime_completion_domain_, sequence,
            runtime_fault_domain_);
      }

      if (result.has_backend_work()) {
        result.attach_resources(detach_runtime_completion_resources());
        track_runtime_completion(result);
      } else {
        release_completed_argpack_leases();
        release_completed_ndarray_leases();
        release_completed_texture_leases();
        release_completed_external_dense_storage_leases();
        release_completed_vulkan_native_resources();
      }
      last_runtime_completion_submission_epoch_.store(
          submission_epoch, std::memory_order_release);
    } catch (...) {
      const std::exception_ptr submission_error = std::current_exception();
      if (runtime_fault_domain_->has_fatal_fault()) {
        const auto fault = runtime_fault_domain_->snapshot();
        const std::string reason =
            fault.first_fault ? fault.first_fault->message
                              : "Runtime backend entered a fatal state";
        result.invalidate_and_release(reason);
        fail_all_runtime_completions(reason);
        release_completed_argpack_leases();
        release_completed_ndarray_leases();
        release_completed_texture_leases();
        release_completed_external_dense_storage_leases();
        release_completed_vulkan_native_resources();
        last_runtime_completion_submission_epoch_.store(
            submission_epoch, std::memory_order_release);
        std::rethrow_exception(submission_error);
      }
      // No completion can safely own a partially detached submission. Finish
      // the backend first, then restore the pre-F2 synchronize release rule.
      try {
        program_impl_->synchronize();
      } catch (const BackendRuntimeError &sync_error) {
        runtime_fault_domain_->report_backend_error(sync_error, sequence);
        if (runtime_fault_domain_->has_fatal_fault()) {
          const auto fault = runtime_fault_domain_->snapshot();
          const std::string reason =
              fault.first_fault ? fault.first_fault->message
                                : "Runtime backend entered a fatal state";
          result.invalidate_and_release(reason);
          fail_all_runtime_completions(reason);
          release_completed_argpack_leases();
          release_completed_ndarray_leases();
          release_completed_texture_leases();
          release_completed_external_dense_storage_leases();
          release_completed_vulkan_native_resources();
          last_runtime_completion_submission_epoch_.store(
              submission_epoch, std::memory_order_release);
        }
        throw;
      }
      result.mark_completed();
      complete_all_runtime_completions();
      release_completed_argpack_leases();
      release_completed_ndarray_leases();
      release_completed_texture_leases();
      release_completed_external_dense_storage_leases();
      release_completed_vulkan_native_resources();
      last_runtime_completion_submission_epoch_.store(
          submission_epoch, std::memory_order_release);
      throw;
    }
  }

  return result;
}

std::unique_ptr<Program::RuntimeSubmissionTransaction>
Program::begin_runtime_submission_transaction(bool gpu_timing,
                                               bool cuda_concurrent_batch) {
  ensure_runtime_submission_allowed("submission transaction");
  TI_ERROR_IF(finalized_,
              "Cannot begin a submission transaction after Program finalize");
  return std::unique_ptr<RuntimeSubmissionTransaction>(
      new RuntimeSubmissionTransaction(this, gpu_timing,
                                       cuda_concurrent_batch));
}

void *Program::register_runtime_cuda_concurrent_stream(void *stream) {
  RuntimeSubmissionTransaction *transaction =
      active_runtime_submission_telemetry_transaction();
  TI_ERROR_IF(transaction == nullptr || transaction->program_ != this,
              "Concurrent CUDA Graph replay requires an active batch transaction");
  return transaction->register_cuda_concurrent_stream(stream);
}

Program::RuntimeSubmissionTransaction *&
Program::active_runtime_submission_telemetry_transaction() noexcept {
  static thread_local RuntimeSubmissionTransaction *active_transaction =
      nullptr;
  return active_transaction;
}

std::unordered_map<std::string, std::uint64_t>
Program::debug_runtime_completion_stats() const {
  std::uint64_t active = 0;
  std::uint64_t pending = 0;
  std::uint64_t failed = 0;
  {
    std::lock_guard<std::mutex> lock(runtime_completion_mutex_);
    active = runtime_completions_.size();
    for (const auto &completion : runtime_completions_) {
      if (!completion.first_error_message().empty()) {
        ++failed;
      } else if (completion.has_backend_work()) {
        ++pending;
      }
    }
  }
  const auto cuda_events = runtime_completion_cuda_event_pool_
                               ? runtime_completion_cuda_event_pool_->snapshot()
                               : RuntimeCompletionCudaEventPoolSnapshot{};
  return {
      {"domain", runtime_completion_domain_},
      {"submission_epoch",
       runtime_submission_epoch_.load(std::memory_order_acquire)},
      {"completed_submission_epoch",
       last_runtime_completion_submission_epoch_.load(
           std::memory_order_acquire)},
      {"next_sequence",
       next_runtime_completion_sequence_.load(std::memory_order_acquire)},
      {"active", active},
      {"pending", pending},
      {"failed", failed},
      {"retained_argpacks",
       runtime_completion_resource_count(kArgPackResourceKind)},
      {"retained_ndarrays",
       runtime_completion_resource_count(kNdarrayResourceKind)},
      {"retained_textures",
       runtime_completion_resource_count(kTextureResourceKind)},
      {"retained_vulkan_graphics_pipelines",
       runtime_completion_resource_count(
           kVulkanGraphicsPipelineResourceKind)},
      {"retained_vulkan_ray_scenes",
       runtime_completion_resource_count(kVulkanRaySceneResourceKind)},
      {"retained_vulkan_ray_resources",
       runtime_completion_resource_count(kVulkanRayResourceKind)},
      {"retained_cuda_provider_plans",
       runtime_completion_resource_count(kCudaProviderPlanResourceKind)},
      {"cuda_completion_events_created", cuda_events.created},
      {"cuda_completion_events_reused", cuda_events.reused},
      {"cuda_completion_events_returned", cuda_events.returned},
      {"cuda_completion_events_destroyed", cuda_events.destroyed},
      {"cuda_completion_events_abandoned", cuda_events.abandoned},
      {"cuda_completion_events_cached", cuda_events.cached},
  };
}

RuntimeStatisticsSnapshot Program::runtime_statistics_snapshot() {
  RuntimeStatisticsSnapshot snapshot =
      runtime_fault_domain_->statistics().snapshot();
  const auto argpacks = debug_argpack_resource_stats();
  const auto ndarrays = debug_ndarray_resource_stats();
  const auto textures = debug_texture_resource_stats();
  const auto staging = debug_dense_field_staging_stats();
  auto sum = [&](const char *key) {
    return argpacks.at(key) + ndarrays.at(key) + textures.at(key) +
           staging.at(key);
  };
  snapshot.memory.live_resources = sum("live");
  snapshot.memory.retiring_resources = sum("retiring");
  snapshot.memory.inflight_resources =
      argpacks.at("inflight") + ndarrays.at("inflight") +
      textures.at("inflight");
  const auto graphics = debug_vulkan_graphics_resource_stats();
  const auto ray = debug_vulkan_ray_resource_stats();
  snapshot.memory.live_resources += graphics.at("live") + ray.at("live");
  snapshot.memory.retiring_resources +=
      graphics.at("retiring") + ray.at("retiring");
  snapshot.memory.inflight_resources +=
      graphics.at("completion_retained") + ray.at("completion_retained");
  std::size_t pending_cuda_provider_plans = 0;
  {
    std::lock_guard<std::recursive_mutex> lock(
        runtime_resource_submission_mutex_);
    pending_cuda_provider_plans = cuda_provider_inflight_plans_.size();
  }
  snapshot.memory.inflight_resources +=
      runtime_completion_resource_count(kCudaProviderPlanResourceKind) +
      pending_cuda_provider_plans;

  const HostMemoryPoolStats host =
      HostMemoryPool::get_instance().get_stats();
  snapshot.memory.host_requested_live_bytes = {
      host.requested_live_bytes, true};
  snapshot.memory.host_raw_bytes = {host.reserved_bytes, true};
  snapshot.memory.host_capacity_bytes = {host.capacity_bytes, true};
  auto &host_allocator = snapshot.memory.host_allocator;
  host_allocator.requested_live_bytes = {
      host.requested_live_bytes, true};
  host_allocator.peak_requested_live_bytes = {
      host.peak_requested_live_bytes, true};
  host_allocator.reserved_bytes = {host.reserved_bytes, true};
  host_allocator.committed_bytes = {
      host.committed_bytes, host.committed_bytes_available};
  host_allocator.capacity_bytes = {host.capacity_bytes, true};
  host_allocator.used_bytes = {host.used_bytes, true};
  host_allocator.available_bytes = {host.available_bytes, true};
  host_allocator.alignment_waste_bytes = {
      host.alignment_waste_bytes, true};
  host_allocator.unreclaimed_released_bytes = {
      host.unreclaimed_released_bytes, true};
  host_allocator.wasted_bytes = {host.wasted_bytes, true};
  host_allocator.chunk_count = {host.unified_chunks, true};
  host_allocator.slab_chunk_count = {host.slab_chunks, true};
  host_allocator.large_chunk_count = {host.large_chunks, true};
  host_allocator.exclusive_chunk_count = {
      host.exclusive_chunks, true};
  host_allocator.peak_reserved_bytes = {
      host.peak_reserved_bytes, true};
  host_allocator.peak_used_bytes = {host.peak_used_bytes, true};
  host_allocator.peak_wasted_bytes = {
      host.peak_wasted_bytes, true};
  host_allocator.peak_chunk_count = {host.peak_chunks, true};

#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    const auto driver = CUDADriver::get_instance().get_telemetry_snapshot();
    const auto context =
        CUDAContext::get_instance().get_lock_telemetry_snapshot();
    const auto submission =
        CUDAContext::get_instance().get_submission_lock_telemetry_snapshot();
    const auto lock_samples = saturating_counter_add(
        driver.lock.sampled_acquisitions,
        saturating_counter_add(context.sampled_acquisitions,
                               submission.sampled_acquisitions));
    const auto lock_contentions = saturating_counter_add(
        driver.lock.contended_acquisitions,
        saturating_counter_add(context.contended_acquisitions,
                               submission.contended_acquisitions));
    const auto lock_wait_ns = saturating_counter_add(
        driver.lock.sampled_wait_ns,
        saturating_counter_add(context.sampled_wait_ns,
                               submission.sampled_wait_ns));
    snapshot.synchronization.backend_waits = {
        saturating_counter_delta(
            driver.wait.waits,
            runtime_backend_telemetry_baseline_.backend_waits),
        true};
    snapshot.synchronization.backend_wait_ns = {
        saturating_counter_delta(
            driver.wait.wait_ns,
            runtime_backend_telemetry_baseline_.backend_wait_ns),
        true};
    snapshot.synchronization.backend_lock_samples = {
        saturating_counter_delta(
            lock_samples,
            runtime_backend_telemetry_baseline_.backend_lock_samples),
        true};
    snapshot.synchronization.backend_lock_contentions = {
        saturating_counter_delta(
            lock_contentions,
            runtime_backend_telemetry_baseline_.backend_lock_contentions),
        true};
    snapshot.synchronization.backend_lock_sampled_wait_ns = {
        saturating_counter_delta(
            lock_wait_ns,
            runtime_backend_telemetry_baseline_
                .backend_lock_sampled_wait_ns),
        true};
  }
#endif

#ifdef TI_WITH_VULKAN
  if (compile_config().arch == Arch::vulkan) {
    auto *device = dynamic_cast<vulkan::VulkanDevice *>(
        program_impl_->get_compute_device());
    if (device != nullptr) {
      const auto telemetry = device->runtime_telemetry_snapshot();
      snapshot.synchronization.backend_waits = {
          telemetry.wait.waits, true};
      snapshot.synchronization.backend_wait_ns = {
          telemetry.wait.wait_ns, true};
      snapshot.synchronization.backend_lock_samples = {
          telemetry.queue_lock.sampled_acquisitions, true};
      snapshot.synchronization.backend_lock_contentions = {
          telemetry.queue_lock.contended_acquisitions, true};
      snapshot.synchronization.backend_lock_sampled_wait_ns = {
          telemetry.queue_lock.sampled_wait_ns, true};
    }
  }
#endif

#ifdef TI_WITH_LLVM
  if (arch_uses_llvm(compile_config().arch)) {
    const DeviceMemoryPoolStats device =
        DeviceMemoryPool::get_instance().get_stats();
    const std::uint64_t device_live =
        device.bytes_allocated_total >= device.bytes_released_total
            ? device.bytes_allocated_total - device.bytes_released_total
            : 0;
    snapshot.memory.device_requested_live_bytes = {device_live, true};
    snapshot.memory.device_raw_bytes = {device.raw_bytes, true};
    snapshot.memory.device_cached_bytes = {device.cached_bytes, true};
  }
#endif
  return snapshot;
}

void Program::initialize_runtime_backend_telemetry_baseline() {
#ifdef TI_WITH_CUDA
  if (compile_config().arch != Arch::cuda) {
    return;
  }
  const auto driver = CUDADriver::get_instance().get_telemetry_snapshot();
  const auto context =
      CUDAContext::get_instance().get_lock_telemetry_snapshot();
  const auto submission =
      CUDAContext::get_instance().get_submission_lock_telemetry_snapshot();
  runtime_backend_telemetry_baseline_.backend_waits = driver.wait.waits;
  runtime_backend_telemetry_baseline_.backend_wait_ns = driver.wait.wait_ns;
  runtime_backend_telemetry_baseline_.backend_lock_samples =
      saturating_counter_add(
          driver.lock.sampled_acquisitions,
          saturating_counter_add(context.sampled_acquisitions,
                                 submission.sampled_acquisitions));
  runtime_backend_telemetry_baseline_.backend_lock_contentions =
      saturating_counter_add(
          driver.lock.contended_acquisitions,
          saturating_counter_add(context.contended_acquisitions,
                                 submission.contended_acquisitions));
  runtime_backend_telemetry_baseline_.backend_lock_sampled_wait_ns =
      saturating_counter_add(
          driver.lock.sampled_wait_ns,
          saturating_counter_add(context.sampled_wait_ns,
                                 submission.sampled_wait_ns));
#endif
}

void Program::close_argpack_resources() {
  std::lock_guard<std::recursive_mutex> submission_lock(
      runtime_resource_submission_mutex_);
  std::lock_guard<std::mutex> lifecycle_lock(argpack_lifecycle_mutex_);
  argpack_resources_open_ = false;
  argpack_views_.clear();
}

void Program::close_ndarray_resources() {
  std::lock_guard<std::recursive_mutex> submission_lock(
      runtime_resource_submission_mutex_);
  std::lock_guard<std::mutex> lifecycle_lock(ndarray_lifecycle_mutex_);
  ndarray_resources_open_ = false;
  ndarray_views_.clear();
  ndarray_view_slots_.clear();
}

void Program::close_texture_resources() {
  std::lock_guard<std::recursive_mutex> submission_lock(
      runtime_resource_submission_mutex_);
  std::lock_guard<std::mutex> lifecycle_lock(texture_lifecycle_mutex_);
  texture_resources_open_ = false;
  texture_views_.clear();
  texture_view_slots_.clear();
}

void Program::close_external_dense_storage_resources() {
  std::lock_guard<std::recursive_mutex> submission_lock(
      runtime_resource_submission_mutex_);
  std::lock_guard<std::mutex> lifecycle_lock(
      external_dense_storage_lifecycle_mutex_);
  external_dense_storage_resources_open_ = false;
}

storage::StorageOwnerRef Program::register_external_dense_storage(
    DeviceAllocation allocation,
    std::uint64_t allocation_bytes,
    ExternalDenseStorageRelease release,
    std::shared_ptr<ExternalSynchronizationDomain> synchronization_domain) {
  ensure_runtime_submission_allowed("external dense storage registration");
  std::lock_guard<std::recursive_mutex> submission_lock(
      runtime_resource_submission_mutex_);
  std::lock_guard<std::mutex> lifecycle_lock(
      external_dense_storage_lifecycle_mutex_);
  TI_ERROR_IF(!external_dense_storage_resources_open_ || finalized_,
              "Cannot register external dense storage after finalize");
  TI_ERROR_IF(allocation_bytes != 0 && allocation == kDeviceNullAllocation,
              "Non-empty external dense storage requires an allocation");
  TI_ERROR_IF(allocation != kDeviceNullAllocation &&
                  allocation.device != program_impl_->get_compute_device(),
              "External dense storage belongs to another compute device");
  auto [result, handle] = external_dense_storage_resources_.emplace(
      kExternalDenseStorageResourceKind, allocation, allocation_bytes,
      std::move(release), std::move(synchronization_domain));
  TI_ERROR_IF(result != ExternalDenseStorageRegistry::Result::kSuccess,
              "Unable to register external dense storage resource");
  return storage::StorageOwnerRef::external_managed(handle.domain, handle.index,
                                                    handle.generation);
}

void Program::retire_external_dense_storage(
    const storage::StorageOwnerRef &owner) {
  std::lock_guard<std::recursive_mutex> submission_lock(
      runtime_resource_submission_mutex_);
  const auto handle = external_dense_storage_handle(owner);
  TI_ERROR_IF(!handle, "External dense storage belongs to another Program");
  TI_ERROR_IF(external_dense_storage_resources_.retire(handle) !=
                  ExternalDenseStorageRegistry::Result::kSuccess,
              "External dense storage is stale or already retired");
}

bool Program::validate_external_dense_storage_owner(
    const storage::StorageOwnerRef &owner) noexcept {
  try {
    std::lock_guard<std::recursive_mutex> submission_lock(
        runtime_resource_submission_mutex_);
    const auto handle = external_dense_storage_handle(owner);
    if (!handle) {
      return false;
    }
    auto [result, lease] = external_dense_storage_resources_.acquire(handle);
    return result == ExternalDenseStorageRegistry::Result::kSuccess &&
           static_cast<bool>(lease);
  } catch (...) {
    return false;
  }
}

std::unordered_map<std::string, std::uint64_t>
Program::debug_external_dense_storage_stats() const {
  const auto completion_inflight =
      runtime_completion_resource_count(kExternalDenseStorageResourceKind);
  std::lock_guard<std::mutex> lifecycle_lock(
      external_dense_storage_lifecycle_mutex_);
  const auto stats = external_dense_storage_resources_.stats();
  return {{"slots", stats.slots},
          {"live", stats.live},
          {"retiring", stats.retiring},
          {"released", stats.released},
          {"leases", stats.leases},
          {"created_total", stats.created_total},
          {"retired_total", stats.retired_total},
          {"released_total", stats.released_total},
          {"release_errors", stats.release_errors},
          {"inflight", external_dense_storage_inflight_leases_.size() +
                           completion_inflight},
          {"closed", stats.closed ? 1u : 0u}};
}

std::unordered_map<std::string, std::uint64_t>
Program::debug_argpack_resource_stats() const {
  const auto completion_inflight =
      runtime_completion_resource_count(kArgPackResourceKind);
  std::lock_guard<std::mutex> lock(argpack_lifecycle_mutex_);
  const auto stats = argpack_resources_.stats();
  return {{"slots", stats.slots},
          {"live", stats.live},
          {"retiring", stats.retiring},
          {"released", stats.released},
          {"leases", stats.leases},
          {"created_total", stats.created_total},
          {"retired_total", stats.retired_total},
          {"released_total", stats.released_total},
          {"release_errors", stats.release_errors},
          {"views", argpack_views_.size()},
          {"inflight",
           argpack_inflight_leases_.size() + completion_inflight},
          {"closed", stats.closed ? 1u : 0u}};
}

std::unordered_map<std::string, std::uint64_t>
Program::debug_argpack_resource_identity(const ArgPack *view) const {
  std::lock_guard<std::mutex> lock(argpack_lifecycle_mutex_);
  const auto found = argpack_views_.find(view);
  TI_ERROR_IF(found == argpack_views_.end(),
              "Cannot inspect a stale or retired ArgPack");
  const auto handle = found->second.handle;
  return {{"domain", handle.domain},
          {"kind", handle.kind},
          {"index", handle.index},
          {"generation", handle.generation}};
}

std::unordered_map<std::string, std::uint64_t>
Program::debug_ndarray_resource_stats() const {
  const auto completion_inflight =
      runtime_completion_resource_count(kNdarrayResourceKind);
  std::lock_guard<std::mutex> lock(ndarray_lifecycle_mutex_);
  const auto stats = ndarray_resources_.stats();
  return {{"slots", stats.slots},
          {"live", stats.live},
          {"retiring", stats.retiring},
          {"released", stats.released},
          {"leases", stats.leases},
          {"created_total", stats.created_total},
          {"retired_total", stats.retired_total},
          {"released_total", stats.released_total},
          {"release_errors", stats.release_errors},
          {"views", ndarray_views_.size()},
          {"inflight",
           ndarray_inflight_leases_.size() + completion_inflight},
          {"closed", stats.closed ? 1u : 0u}};
}

std::unordered_map<std::string, std::uint64_t>
Program::debug_ndarray_resource_identity(const Ndarray *view) const {
  std::lock_guard<std::mutex> lock(ndarray_lifecycle_mutex_);
  const auto found = ndarray_views_.find(view);
  TI_ERROR_IF(found == ndarray_views_.end(),
              "Cannot inspect a stale or retired Ndarray");
  const auto handle = found->second.handle;
  return {{"domain", handle.domain},
          {"kind", handle.kind},
          {"index", handle.index},
          {"generation", handle.generation}};
}

std::unordered_map<std::string, std::uint64_t>
Program::debug_texture_resource_stats() const {
  const auto completion_inflight =
      runtime_completion_resource_count(kTextureResourceKind);
  std::lock_guard<std::mutex> lock(texture_lifecycle_mutex_);
  const auto stats = texture_resources_.stats();
  return {{"slots", stats.slots},
          {"live", stats.live},
          {"retiring", stats.retiring},
          {"released", stats.released},
          {"leases", stats.leases},
          {"created_total", stats.created_total},
          {"retired_total", stats.retired_total},
          {"released_total", stats.released_total},
          {"release_errors", stats.release_errors},
          {"views", texture_views_.size()},
          {"inflight",
           texture_inflight_leases_.size() + completion_inflight},
          {"closed", stats.closed ? 1u : 0u}};
}

std::unordered_map<std::string, std::uint64_t>
Program::debug_texture_resource_identity(const Texture *view) const {
  std::lock_guard<std::mutex> lock(texture_lifecycle_mutex_);
  const auto found = texture_views_.find(view);
  TI_ERROR_IF(found == texture_views_.end(),
              "Cannot inspect a stale or retired Texture");
  const auto handle = found->second.handle;
  return {{"domain", handle.domain},
          {"kind", handle.kind},
          {"index", handle.index},
          {"generation", handle.generation}};
}

void Program::synchronize() {
  ensure_runtime_submission_allowed("Program synchronize");
  RuntimeProgramSyncStatisticsScope statistics_scope(this);
  std::lock_guard<std::recursive_mutex> submission_lock(
      runtime_resource_submission_mutex_);
  const bool tracking_was_enabled =
      runtime_completion_tracking_enabled_.exchange(
          true, std::memory_order_acq_rel);
  {
    RuntimeSubmissionWriteScope completion_submission_scope(this);
    try {
      program_impl_->synchronize();
    } catch (const BackendRuntimeError &error) {
      runtime_fault_domain_->report_backend_error(
          error, next_runtime_completion_sequence_.load(
                     std::memory_order_acquire));
      if (runtime_fault_domain_->has_fatal_fault()) {
        runtime_submission_pending_.store(false, std::memory_order_release);
        const auto fault = runtime_fault_domain_->snapshot();
        const std::string reason =
            fault.first_fault ? fault.first_fault->message
                              : "Runtime backend entered a fatal state";
        fail_all_runtime_completions(reason);
        release_completed_argpack_leases();
        release_completed_ndarray_leases();
        release_completed_texture_leases();
        release_completed_external_dense_storage_leases();
        release_completed_vulkan_native_resources();
      }
      throw;
    }
    const bool had_submission = runtime_submission_pending_.exchange(
        false, std::memory_order_acq_rel);
    std::uint64_t submission_epoch =
        runtime_submission_epoch_.load(std::memory_order_relaxed);
    if (had_submission) {
      const std::uint64_t previous = runtime_submission_epoch_.fetch_add(
          1, std::memory_order_relaxed);
      TI_ASSERT(previous != (std::numeric_limits<std::uint64_t>::max)());
      submission_epoch = previous + 1;
    }
    complete_all_runtime_completions();
    release_completed_argpack_leases();
    release_completed_ndarray_leases();
    release_completed_texture_leases();
    release_completed_external_dense_storage_leases();
    release_completed_vulkan_native_resources();
    last_runtime_completion_submission_epoch_.store(
        submission_epoch, std::memory_order_release);
  }
  if (!tracking_was_enabled) {
    runtime_completion_tracking_enabled_.store(false,
                                               std::memory_order_release);
  }
}

StreamSemaphore Program::flush() {
  ensure_runtime_submission_allowed("Program flush");
  return program_impl_->flush();
}

StreamSemaphore Program::flush_if_pending() {
  ensure_runtime_submission_allowed("Program flush");
  if (auto *gfx_program = dynamic_cast<GfxProgramImpl *>(program_impl_.get())) {
    return gfx_program->flush_if_pending();
  }
  return program_impl_->flush();
}

bool Program::has_pending_gfx_command_list() const {
  if (auto *gfx_program = dynamic_cast<GfxProgramImpl *>(program_impl_.get())) {
    return gfx_program->has_pending_command_list();
  }
  return false;
}

int Program::get_snode_tree_size() {
  return snode_trees_.size();
}

Kernel &Program::get_snode_reader(SNode *snode) {
  TI_ASSERT(snode->type == SNodeType::place);
  auto kernel_name = fmt::format("snode_reader_{}", snode->id);
  auto &ker = kernel([snode, this](Kernel *kernel) {
    ExprGroup indices;
    for (int i = 0; i < snode->num_active_indices; i++) {
      auto argload_expr = Expr::make<ArgLoadExpression>(std::vector<int>{i},
                                                        PrimitiveType::i32);
      argload_expr->type_check(&this->compile_config());
      indices.push_back(std::move(argload_expr));
    }
    ASTBuilder &builder = kernel->context->builder();
    auto ret = Stmt::make<FrontendReturnStmt>(ExprGroup(
        builder.expr_subscript(Expr(snode_to_fields_.at(snode)), indices)));
    builder.insert(std::move(ret));
  });
  ker.name = kernel_name;
  ker.is_accessor = true;
  for (int i = 0; i < snode->num_active_indices; i++)
    ker.insert_scalar_param(PrimitiveType::i32);
  ker.insert_ret(snode->dt);
  ker.finalize_params();
  ker.finalize_rets();
  return ker;
}

Kernel &Program::get_snode_writer(SNode *snode) {
  TI_ASSERT(snode->type == SNodeType::place);
  auto kernel_name = fmt::format("snode_writer_{}", snode->id);
  auto &ker = kernel([snode, this](Kernel *kernel) {
    ExprGroup indices;
    for (int i = 0; i < snode->num_active_indices; i++) {
      auto argload_expr = Expr::make<ArgLoadExpression>(std::vector<int>{i},
                                                        PrimitiveType::i32);
      argload_expr->type_check(&this->compile_config());
      indices.push_back(std::move(argload_expr));
    }
    ASTBuilder &builder = kernel->context->builder();
    auto expr =
        builder.expr_subscript(Expr(snode_to_fields_.at(snode)), indices);
    expr.type_check(&this->compile_config());
    auto argload_expr = Expr::make<ArgLoadExpression>(
        std::vector<int>{snode->num_active_indices},
        snode->dt->get_compute_type());
    argload_expr->type_check(&this->compile_config());
    builder.insert_assignment(expr, argload_expr, expr->dbg_info);
  });
  ker.name = kernel_name;
  ker.is_accessor = true;
  for (int i = 0; i < snode->num_active_indices; i++)
    ker.insert_scalar_param(PrimitiveType::i32);
  ker.insert_scalar_param(snode->dt);
  ker.finalize_params();
  ker.finalize_rets();
  return ker;
}

uint64 Program::fetch_result_uint64(int i) {
  return program_impl_->fetch_result_uint64(i, result_buffer);
}

void Program::finalize() {
  if (finalized_) {
    return;
  }

  runtime_fault_domain_->begin_finalizing();

  bool teardown_warning_reported = false;
  auto best_effort = [&](const char *step, auto &&operation) {
    try {
      operation();
      return true;
    } catch (const BackendRuntimeError &error) {
      runtime_fault_domain_->report_backend_error(error, 0);
      if (!runtime_fault_domain_->has_fatal_fault() &&
          !teardown_warning_reported) {
        teardown_warning_reported = true;
        TI_WARN("Program finalize step '{}' failed: {}", step, error.what());
      }
    } catch (const std::exception &error) {
      if (!runtime_fault_domain_->has_fatal_fault() &&
          !teardown_warning_reported) {
        teardown_warning_reported = true;
        TI_WARN("Program finalize step '{}' failed: {}", step, error.what());
      }
    } catch (...) {
      if (!runtime_fault_domain_->has_fatal_fault() &&
          !teardown_warning_reported) {
        teardown_warning_reported = true;
        TI_WARN("Program finalize step '{}' failed with an unknown error",
                step);
      }
    }
    return false;
  };

  best_effort("close runtime resources", [&] {
    std::lock_guard<std::recursive_mutex> submission_lock(
        runtime_resource_submission_mutex_);
    close_argpack_resources();
    close_ndarray_resources();
    close_texture_resources();
    close_external_dense_storage_resources();
    dense_field_staging_open_ = false;
  });
  TI_TRACE("Program finalizing...");

  bool synchronized = false;
  if (!runtime_fault_domain_->has_fatal_fault()) {
    synchronized = best_effort("backend synchronize", [&] {
      std::lock_guard<std::recursive_mutex> submission_lock(
          runtime_resource_submission_mutex_);
      RuntimeSubmissionWriteScope completion_submission_scope(this);
      program_impl_->synchronize();
      const bool had_submission = runtime_submission_pending_.exchange(
          false, std::memory_order_acq_rel);
      std::uint64_t submission_epoch =
          runtime_submission_epoch_.load(std::memory_order_relaxed);
      if (had_submission) {
        const std::uint64_t previous = runtime_submission_epoch_.fetch_add(
            1, std::memory_order_relaxed);
        TI_ASSERT(previous != (std::numeric_limits<std::uint64_t>::max)());
        submission_epoch = previous + 1;
      }
      complete_all_runtime_completions();
      release_completed_argpack_leases();
      release_completed_ndarray_leases();
      release_completed_texture_leases();
      release_completed_external_dense_storage_leases();
      release_completed_vulkan_native_resources();
      last_runtime_completion_submission_epoch_.store(
          submission_epoch, std::memory_order_release);
    });
  }
  if (!synchronized) {
    runtime_submission_pending_.store(false, std::memory_order_release);
    const auto fault = runtime_fault_domain_->snapshot();
    const std::string reason =
        fault.first_fault
            ? fault.first_fault->message
            : "Program finalized after backend synchronization failed";
    fail_all_runtime_completions(reason);
    release_completed_argpack_leases();
    release_completed_ndarray_leases();
    release_completed_texture_leases();
    release_completed_external_dense_storage_leases();
    release_completed_vulkan_native_resources();
  }
  best_effort("clear primitive workspace arena",
              [&] { primitive_workspace_arena_.clear(); });
  if (compile_config().arch == Arch::vulkan) {
    best_effort("clear Vulkan graphics pipelines",
                [&] { vulkan_clear_graphics_pipelines(); });
    best_effort("clear Vulkan ray scenes",
                [&] { vulkan_clear_ray_scenes(); });
    best_effort("clear Vulkan FFT plans", [&] { vulkan_clear_fft_plans(); });
    best_effort("clear Vulkan Parallel Sort plans",
                [&] { vulkan_clear_parallel_sort_plans(); });
    best_effort("clear Vulkan primitive caches",
                [&] { vulkan_clear_primitive_caches(); });
  }
  if (compile_config().arch == Arch::cuda) {
    best_effort("clear CUDA cuDSS plans", [&] { cuda_clear_cudss_plans(); });
    best_effort("clear CUDA cuFFT plans", [&] { cuda_clear_cufft_plans(); });
    best_effort("clear CUDA cuBLAS GEMM provider",
                [&] { cuda_clear_cublas_gemm(); });
  }
  best_effort("finalize ArgPack resources",
              [&] { argpack_resources_.finalize({kArgPackResourceKind}); });
  best_effort("finalize Ndarray resources",
              [&] { ndarray_resources_.finalize({kNdarrayResourceKind}); });
  best_effort("finalize Texture resources",
              [&] { texture_resources_.finalize({kTextureResourceKind}); });
  best_effort("finalize external dense storage", [&] {
    external_dense_storage_resources_.finalize(
        {kExternalDenseStorageResourceKind});
  });
  best_effort("close dense-field staging", [&] {
    std::lock_guard<std::recursive_mutex> submission_lock(
        runtime_resource_submission_mutex_);
    close_dense_field_staging_resource();
  });
  best_effort("close Graph observation staging", [&] {
    std::lock_guard<std::recursive_mutex> submission_lock(
        runtime_resource_submission_mutex_);
    graph_observation_staging_.readback.reset();
    graph_observation_staging_.capacity = 0;
#ifdef TI_WITH_CUDA
    if (compile_config().arch == Arch::cuda &&
        runtime_fault_domain_->backend_calls_safe()) {
      auto context_guard = CUDAContext::get_instance().get_guard();
      auto &driver = CUDADriver::get_instance();
      for (const auto &entry : graph_observation_staging_.cuda_readbacks) {
        if (entry.second.host_ptr) {
          driver.mem_free_host(entry.second.host_ptr);
        }
      }
    }
#endif
    graph_observation_staging_.cuda_readbacks.clear();
    graph_observation_staging_.cuda_pinned_bytes = 0;
  });
  best_effort("clear CUDA completion event cache", [&] {
    if (runtime_completion_cuda_event_pool_) {
      runtime_completion_cuda_event_pool_->clear();
    }
  });
  if (arch_uses_llvm(compile_config().arch) ||
      compile_config().arch == Arch::vulkan) {
    best_effort("finalize backend runtime",
                [&] { program_impl_->finalize(); });
  }
  detach_runtime_fault_reporter();

  Stmt::reset_counter();

  finalized_ = true;
  num_instances_ -= 1;
  runtime_fault_domain_->mark_finalized();
  best_effort("write offline cache",
              [&] { program_impl_->dump_cache_data_to_disk(); });
  compile_config_ = default_compile_config;
  TI_TRACE("Program ({}) finalized_.", fmt::ptr(this));

  // Reset memory pool
  best_effort("reset host memory pool",
              [&] { HostMemoryPool::get_instance().reset(); });
}

void Program::clear_primitive_workspaces() {
  clear_primitive_workspaces_for(PrimitiveWorkspaceBackend::any,
                                 PrimitiveWorkspaceFamily::any);
}

void Program::clear_primitive_workspaces_for(
    PrimitiveWorkspaceBackend backend,
    PrimitiveWorkspaceFamily family) {
  // Block native/Graph host submissions before establishing completion. A
  // synchronize-before-retire sequence alone is unsafe: a caller that had
  // already acquired a workspace lease could enqueue after the wait returned.
  // The existing recursive resource-submission domain closes that gap without
  // adding any lock to ordinary kernel submission.
  auto submission_guard = acquire_runtime_resource_submission_guard();
  // Resource destructors may call backend deallocation APIs. Establish an
  // explicit completion boundary here instead of letting an arena budget or
  // cache lookup introduce an invisible wait after enqueue.
  synchronize();
  primitive_workspace_arena_.clear(backend, family);
}

int Program::default_block_dim(const CompileConfig &config) {
  if (arch_is_cpu(config.arch)) {
    return config.default_cpu_block_dim;
  } else {
    return config.default_gpu_block_dim;
  }
}

void Program::print_memory_profiler_info() {
  program_impl_->print_memory_profiler_info(snode_trees_, result_buffer);
}

std::size_t Program::get_snode_num_dynamically_allocated(SNode *snode) {
  return program_impl_->get_snode_num_dynamically_allocated(snode,
                                                            result_buffer);
}

void Program::reset_hash_snode_probe_stats() {
  program_impl_->reset_hash_snode_probe_stats(result_buffer);
}

std::vector<int64> Program::get_hash_snode_probe_stats() {
  return program_impl_->get_hash_snode_probe_stats(result_buffer);
}

DeviceAllocation Program::allocate_host_read_memory_on_device(
    std::size_t alloc_size,
    AllocUsage usage) {
  if (arch_is_cpu(compile_config().arch)) {
    return program_impl_->allocate_memory_on_device(alloc_size, result_buffer,
                                                    usage);
  }
  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device,
              "Host-readable Graph observation storage requires a compute "
              "device.");
  DeviceAllocation allocation;
  const RhiResult result = device->allocate_memory(
      {alloc_size, /*host_write=*/false, /*host_read=*/true,
       /*export_sharing=*/false, usage},
      &allocation);
  TI_ERROR_IF(result != RhiResult::success,
              "Unable to allocate host-readable Graph observation storage: "
              "{}",
              result);
  return allocation;
}

void Program::debug_reset_ordinary_launch_attribution() noexcept {
  std::lock_guard<std::recursive_mutex> submission_lock(
      runtime_resource_submission_mutex_);
  auto &stats = ordinary_launch_attribution_;
#define TI_RESET_ORDINARY_LAUNCH_COUNTER(name) \
  stats.name.store(0, std::memory_order_relaxed)
  TI_RESET_ORDINARY_LAUNCH_COUNTER(launches);
  TI_RESET_ORDINARY_LAUNCH_COUNTER(no_resource_fast_path);
  TI_RESET_ORDINARY_LAUNCH_COUNTER(registered_execution_plan_launches);
  TI_RESET_ORDINARY_LAUNCH_COUNTER(graph_transaction_dispatches);
  TI_RESET_ORDINARY_LAUNCH_COUNTER(general_resource_launches);
  TI_RESET_ORDINARY_LAUNCH_COUNTER(owned_ndarray_only_launches);
  TI_RESET_ORDINARY_LAUNCH_COUNTER(snode_guard_acquisitions);
  TI_RESET_ORDINARY_LAUNCH_COUNTER(snode_guard_elisions);
  TI_RESET_ORDINARY_LAUNCH_COUNTER(registered_plan_snode_validation_scans);
  TI_RESET_ORDINARY_LAUNCH_COUNTER(
      registered_plan_snode_validation_epoch_fast_hits);
  TI_RESET_ORDINARY_LAUNCH_COUNTER(resource_lock_acquisitions);
  TI_RESET_ORDINARY_LAUNCH_COUNTER(ndarray_slot_validations);
  TI_RESET_ORDINARY_LAUNCH_COUNTER(ndarray_map_lookups);
  TI_RESET_ORDINARY_LAUNCH_COUNTER(ndarray_lease_clones);
  TI_RESET_ORDINARY_LAUNCH_COUNTER(ndarray_inflight_reuses);
  TI_RESET_ORDINARY_LAUNCH_COUNTER(ndarray_pins);
  TI_RESET_ORDINARY_LAUNCH_COUNTER(runtime_error_checks);
  TI_RESET_ORDINARY_LAUNCH_COUNTER(runtime_error_check_elisions);
  TI_RESET_ORDINARY_LAUNCH_COUNTER(total_host_ns);
  TI_RESET_ORDINARY_LAUNCH_COUNTER(compile_lookup_ns);
  TI_RESET_ORDINARY_LAUNCH_COUNTER(compile_and_launch_total_ns);
  TI_RESET_ORDINARY_LAUNCH_COUNTER(snode_guard_wait_ns);
  TI_RESET_ORDINARY_LAUNCH_COUNTER(resource_lock_wait_ns);
  TI_RESET_ORDINARY_LAUNCH_COUNTER(resource_resolution_ns);
  TI_RESET_ORDINARY_LAUNCH_COUNTER(backend_submit_ns);
  TI_RESET_ORDINARY_LAUNCH_COUNTER(completion_accounting_ns);
#undef TI_RESET_ORDINARY_LAUNCH_COUNTER
  if (program_impl_) {
    program_impl_->get_kernel_launcher().debug_reset_launch_attribution();
  }
}

std::unordered_map<std::string, std::uint64_t>
Program::debug_ordinary_launch_attribution() const {
  const auto &stats = ordinary_launch_attribution_;
  auto load = [](const std::atomic<std::uint64_t> &value) {
    return value.load(std::memory_order_relaxed);
  };
  std::unordered_map<std::string, std::uint64_t> result = {
      {"enabled", stats.enabled ? 1u : 0u},
      {"owned_ndarray_fast_path",
       ordinary_owned_ndarray_fast_path_enabled_ ? 1u : 0u},
      {"snode_guard_elision",
       ordinary_snode_guard_elision_enabled_ ? 1u : 0u},
      {"launches", load(stats.launches)},
      {"no_resource_fast_path", load(stats.no_resource_fast_path)},
      {"registered_execution_plan_launches",
       load(stats.registered_execution_plan_launches)},
      {"graph_transaction_dispatches",
       load(stats.graph_transaction_dispatches)},
      {"general_resource_launches", load(stats.general_resource_launches)},
      {"owned_ndarray_only_launches",
       load(stats.owned_ndarray_only_launches)},
      {"snode_guard_acquisitions", load(stats.snode_guard_acquisitions)},
      {"snode_guard_elisions", load(stats.snode_guard_elisions)},
      {"registered_plan_snode_validation_scans",
       load(stats.registered_plan_snode_validation_scans)},
      {"registered_plan_snode_validation_epoch_fast_hits",
       load(stats.registered_plan_snode_validation_epoch_fast_hits)},
      {"resource_lock_acquisitions", load(stats.resource_lock_acquisitions)},
      {"ndarray_slot_validations", load(stats.ndarray_slot_validations)},
      {"ndarray_map_lookups", load(stats.ndarray_map_lookups)},
      {"ndarray_lease_clones", load(stats.ndarray_lease_clones)},
      {"ndarray_inflight_reuses", load(stats.ndarray_inflight_reuses)},
      {"ndarray_pins", load(stats.ndarray_pins)},
      {"runtime_error_checks", load(stats.runtime_error_checks)},
      {"runtime_error_check_elisions",
       load(stats.runtime_error_check_elisions)},
      {"total_host_ns", load(stats.total_host_ns)},
      {"compile_lookup_ns", load(stats.compile_lookup_ns)},
      {"compile_and_launch_total_ns",
       load(stats.compile_and_launch_total_ns)},
      {"snode_guard_wait_ns", load(stats.snode_guard_wait_ns)},
      {"resource_lock_wait_ns", load(stats.resource_lock_wait_ns)},
      {"resource_resolution_ns", load(stats.resource_resolution_ns)},
      {"backend_submit_ns", load(stats.backend_submit_ns)},
      {"completion_accounting_ns", load(stats.completion_accounting_ns)},
  };
  if (program_impl_) {
    for (const auto &[name, value] :
         program_impl_->get_kernel_launcher().debug_launch_attribution()) {
      result.emplace("backend_" + name, value);
    }
  }
  return result;
}

Ndarray *Program::create_ndarray(const DataType type,
                                 const std::vector<int> &shape,
                                 ExternalArrayLayout layout,
                                 bool zero_fill,
                                 const DebugInfo &dbg_info,
                                 bool host_read) {
  return create_ndarray_impl(type, shape, layout, zero_fill, dbg_info,
                             host_read, {});
}

Ndarray *Program::create_ndarray_with_allocator(
    DataType type,
    const std::vector<int> &shape,
    const std::function<DeviceAllocation(std::size_t)> &allocator) {
  return create_ndarray_impl(type, shape, ExternalArrayLayout::kNull, true,
                             DebugInfo(), false, allocator);
}

Ndarray *Program::create_ndarray_impl(
    DataType type,
    const std::vector<int> &shape,
    ExternalArrayLayout layout,
    bool zero_fill,
    const DebugInfo &dbg_info,
    bool host_read,
    const std::function<DeviceAllocation(std::size_t)> &allocator) {
  std::lock_guard<std::recursive_mutex> submission_lock(
      runtime_resource_submission_mutex_);
  {
    std::lock_guard<std::mutex> lifecycle_lock(ndarray_lifecycle_mutex_);
    TI_ERROR_IF(!ndarray_resources_open_,
                "Cannot create Ndarray after Program finalize");
  }

  auto arr = std::make_unique<Ndarray>(this, type, shape, layout, dbg_info,
                                       host_read, allocator);
  Ndarray *view = arr.get();
  std::unique_lock<std::mutex> lock(ndarray_lifecycle_mutex_);
  auto [result, handle] =
      ndarray_resources_.insert(kNdarrayResourceKind, std::move(arr));
  TI_ERROR_IF(result != NdarrayResourceRegistry::Result::kSuccess,
              "Unable to register the new Ndarray runtime resource");
  auto [lease_result, lease] = ndarray_resources_.acquire(handle);
  if (lease_result != NdarrayResourceRegistry::Result::kSuccess) {
    lock.unlock();
    ndarray_resources_.retire(handle);
    TI_ERROR("Unable to acquire the new Ndarray runtime resource");
  }
  view->bind_runtime_resource_handle(handle);
  bool inserted = false;
  try {
    if (handle.index >= ndarray_view_slots_.size()) {
      ndarray_view_slots_.resize(static_cast<std::size_t>(handle.index) + 1);
    }
    auto [view_iter, view_inserted] = ndarray_views_.emplace(
        view, NdarrayResourceView{handle, std::move(lease)});
    inserted = view_inserted;
    if (inserted) {
      TI_ASSERT(ndarray_view_slots_[handle.index].view == nullptr);
      ndarray_view_slots_[handle.index] = {view, handle, &view_iter->second};
    }
  } catch (...) {
    if (handle.index < ndarray_view_slots_.size() &&
        ndarray_view_slots_[handle.index].view == view) {
      ndarray_view_slots_[handle.index] = {};
    }
    lock.unlock();
    ndarray_resources_.retire(handle);
    throw;
  }
  if (!inserted) {
    if (handle.index < ndarray_view_slots_.size() &&
        ndarray_view_slots_[handle.index].view == view) {
      ndarray_view_slots_[handle.index] = {};
    }
    lock.unlock();
    ndarray_resources_.retire(handle);
    TI_ERROR("Ndarray view identity collision inside one Program");
  }
  lock.unlock();

  try {
    if (zero_fill) {
      Arch arch = compile_config().arch;
      if (arch_is_cpu(arch) || arch == Arch::cuda || arch == Arch::amdgpu) {
        fill_ndarray_fast_u32(view, /*data=*/0);
      } else if (arch != Arch::dx12) {
        // Device api support for dx12 backend are not complete yet
        Stream *stream =
            program_impl_->get_compute_device()->get_compute_stream();
        auto [cmdlist, res] = stream->new_command_list_unique();
        TI_ASSERT(res == RhiResult::success);
        cmdlist->buffer_fill(view->ndarray_alloc_.get_ptr(0),
                             view->get_element_size() * view->get_nelement(),
                             /*data=*/0);
        stream->submit_synced(cmdlist.get());
      }
    }
  } catch (...) {
    delete_ndarray(view);
    if (arch_is_gpu(compile_config().arch)) {
      synchronize();
    }
    throw;
  }
  return view;
}

ArgPack *Program::create_argpack(const DataType dt) {
  // Serialize the open check, backend allocation and publication with
  // close_argpack_resources(). This prevents a creator that started around
  // finalize from allocating against a backend whose teardown has begun.
  // The narrower lifecycle mutex is still never held across the backend call.
  std::lock_guard<std::recursive_mutex> submission_lock(
      runtime_resource_submission_mutex_);
  {
    std::lock_guard<std::mutex> lifecycle_lock(argpack_lifecycle_mutex_);
    TI_ERROR_IF(!argpack_resources_open_,
                "Cannot create ArgPack after Program finalize");
  }

  // Device allocation can enter a backend allocator. Keep it outside the
  // lifecycle critical section; only publication into the registry/view map
  // needs serialization with launch, retire and finalize.
  auto pack = std::make_unique<ArgPack>(this, dt);
  ArgPack *view = pack.get();
  std::unique_lock<std::mutex> lock(argpack_lifecycle_mutex_);
  auto [result, handle] =
      argpack_resources_.insert(kArgPackResourceKind, std::move(pack));
  TI_ERROR_IF(result != ArgPackResourceRegistry::Result::kSuccess,
              "Unable to register the new ArgPack runtime resource");

  auto [lease_result, lease] = argpack_resources_.acquire(handle);
  if (lease_result != ArgPackResourceRegistry::Result::kSuccess) {
    lock.unlock();
    argpack_resources_.retire(handle);
    TI_ERROR("Unable to acquire the new ArgPack runtime resource");
  }

  bool inserted = false;
  try {
    inserted =
        argpack_views_
            .emplace(view, ArgPackResourceView{handle, std::move(lease)})
            .second;
  } catch (...) {
    lock.unlock();
    argpack_resources_.retire(handle);
    throw;
  }
  if (!inserted) {
    lock.unlock();
    argpack_resources_.retire(handle);
    TI_ERROR("ArgPack view identity collision inside one Program");
  }
  return view;
}

void Program::delete_ndarray(Ndarray *ndarray) {
  // [Note] Ndarray memory deallocation
  // Ndarray's memory allocation is managed by Taichi and Python can control
  // this via Taichi indirectly. For example, when an ndarray is GC-ed in
  // Python, it signals Taichi to free its memory allocation. But Taichi will
  // make sure **no pending kernels to be executed needs the ndarray** before it
  // actually frees the memory. When `ti.reset()` is called, all ndarrays
  // allocated in this program should be gone and no longer valid in Python.
  // The registry retires an ndarray once Python no longer owns its view and
  // frees it after all submission/completion leases have been released:
  // - Python GC signals taichi that it's no longer useful
  // - All kernels using it are executed.
  std::lock_guard<std::recursive_mutex> submission_lock(
      runtime_resource_submission_mutex_);
  NdarrayResourceHandle handle;
  {
    std::lock_guard<std::mutex> lock(ndarray_lifecycle_mutex_);
    const auto found = ndarray_views_.find(ndarray);
    if (found == ndarray_views_.end()) {
      return;
    }
    handle = found->second.handle;
    // Every asynchronous use pins before releasing the Program submission
    // transaction. The owner is either in ndarray_inflight_leases_ or in a
    // RuntimeCompletion resource batch; if neither exists there is no pending
    // backend dereference to protect. Creating a new lease here would retain a
    // resource after its completion has already proved safety and force an
    // unrelated later ti.sync(). Texture/ArgPack already follow this rule.
    TI_ASSERT(handle.index < ndarray_view_slots_.size() &&
              ndarray_view_slots_[handle.index].view == ndarray &&
              ndarray_view_slots_[handle.index].handle == handle);
    ndarray_view_slots_[handle.index] = {};
    ndarray_views_.erase(found);
  }
  const auto result = ndarray_resources_.retire(handle);
  TI_ASSERT(result == NdarrayResourceRegistry::Result::kSuccess ||
            result == NdarrayResourceRegistry::Result::kInvalidHandle);
}

void Program::delete_ndarray_if_alive(
    Program *program,
    const std::weak_ptr<ProgramLifetimeToken> &lifetime,
    Ndarray *ndarray) noexcept {
  if (!program || !ndarray) {
    return;
  }
  auto token = lifetime.lock();
  if (!token) {
    return;
  }
  try {
    std::lock_guard<std::mutex> lock(token->mutex_);
    if (token->program_ == program) {
      program->delete_ndarray(ndarray);
    }
  } catch (...) {
  }
}

storage::StorageOwnerRef Program::register_external_dense_storage_if_alive(
    Program *program,
    const std::weak_ptr<ProgramLifetimeToken> &lifetime,
    DeviceAllocation allocation,
    std::uint64_t allocation_bytes,
    ExternalDenseStorageRelease release,
    std::shared_ptr<ExternalSynchronizationDomain> synchronization_domain) {
  if (!program) {
    return {};
  }
  auto token = lifetime.lock();
  if (!token) {
    return {};
  }
  std::lock_guard<std::mutex> lock(token->mutex_);
  if (token->program_ != program) {
    return {};
  }
  return program->register_external_dense_storage(
      allocation, allocation_bytes, std::move(release),
      std::move(synchronization_domain));
}

bool Program::retire_external_dense_storage_if_alive(
    Program *program,
    const std::weak_ptr<ProgramLifetimeToken> &lifetime,
    const storage::StorageOwnerRef &owner) noexcept {
  if (!program || !owner.valid()) {
    return false;
  }
  auto token = lifetime.lock();
  if (!token) {
    return false;
  }
  try {
    std::lock_guard<std::mutex> lock(token->mutex_);
    if (token->program_ != program ||
        !program->validate_external_dense_storage_owner(owner)) {
      return false;
    }
    program->retire_external_dense_storage(owner);
    return true;
  } catch (...) {
    return false;
  }
}

void Program::delete_argpack(ArgPack *argpack) {
  std::lock_guard<std::recursive_mutex> submission_lock(
      runtime_resource_submission_mutex_);
  ArgPackResourceHandle handle;
  {
    std::lock_guard<std::mutex> lock(argpack_lifecycle_mutex_);
    const auto found = argpack_views_.find(argpack);
    if (found == argpack_views_.end()) {
      return;
    }
    handle = found->second.handle;
    argpack_views_.erase(found);
  }
  const auto result = argpack_resources_.retire(handle);
  TI_ASSERT(result == ArgPackResourceRegistry::Result::kSuccess ||
            result == ArgPackResourceRegistry::Result::kInvalidHandle);
}

Texture *Program::create_texture(BufferFormat buffer_format,
                                 const std::vector<int> &shape,
                                 ImageSamplerConfig sampler_config) {
  std::lock_guard<std::recursive_mutex> submission_lock(
      runtime_resource_submission_mutex_);
  {
    std::lock_guard<std::mutex> lifecycle_lock(texture_lifecycle_mutex_);
    TI_ERROR_IF(!texture_resources_open_,
                "Cannot create Texture after Program finalize");
  }

  std::unique_ptr<Texture> texture;
  if (shape.size() == 1) {
    texture = std::make_unique<Texture>(this, buffer_format, shape[0], 1, 1,
                                        sampler_config, ImageDimension::d1D);
  } else if (shape.size() == 2) {
    texture =
        std::make_unique<Texture>(this, buffer_format, shape[0], shape[1], 1,
                                  sampler_config, ImageDimension::d2D);
  } else if (shape.size() == 3) {
    texture = std::make_unique<Texture>(this, buffer_format, shape[0],
                                        shape[1], shape[2], sampler_config,
                                        ImageDimension::d3D);
  } else {
    TI_ERROR("Texture shape invalid");
  }

  Texture *view = texture.get();
  std::unique_lock<std::mutex> lock(texture_lifecycle_mutex_);
  auto [result, handle] =
      texture_resources_.insert(kTextureResourceKind, std::move(texture));
  TI_ERROR_IF(result != TextureResourceRegistry::Result::kSuccess,
              "Unable to register the new Texture runtime resource");
  auto [lease_result, lease] = texture_resources_.acquire(handle);
  if (lease_result != TextureResourceRegistry::Result::kSuccess) {
    lock.unlock();
    texture_resources_.retire(handle);
    TI_ERROR("Unable to acquire the new Texture runtime resource");
  }
  view->bind_runtime_resource_handle(handle);
  bool inserted = false;
  try {
    if (texture_view_slots_.size() <= handle.index) {
      texture_view_slots_.resize(static_cast<std::size_t>(handle.index) + 1);
    }
    inserted = texture_views_
                   .emplace(view,
                            TextureResourceView{handle, std::move(lease)})
                   .second;
  } catch (...) {
    lock.unlock();
    texture_resources_.retire(handle);
    throw;
  }
  if (!inserted) {
    lock.unlock();
    texture_resources_.retire(handle);
    TI_ERROR("Texture view identity collision inside one Program");
  }
  TI_ASSERT(texture_view_slots_[handle.index].view == nullptr);
  texture_view_slots_[handle.index] = {view, handle};
  return view;
}

void Program::delete_texture(Texture *texture) {
  std::lock_guard<std::recursive_mutex> submission_lock(
      runtime_resource_submission_mutex_);
  TextureResourceHandle handle;
  {
    std::lock_guard<std::mutex> lock(texture_lifecycle_mutex_);
    const auto found = texture_views_.find(texture);
    if (found == texture_views_.end()) {
      return;
    }
    handle = found->second.handle;
    TI_ASSERT(handle.index < texture_view_slots_.size() &&
              texture_view_slots_[handle.index].view == texture &&
              texture_view_slots_[handle.index].handle == handle);
    texture_view_slots_[handle.index] = {};
    texture_views_.erase(found);
  }
  const auto result = texture_resources_.retire(handle);
  TI_ASSERT(result == TextureResourceRegistry::Result::kSuccess ||
            result == TextureResourceRegistry::Result::kInvalidHandle);
}

intptr_t Program::get_ndarray_data_ptr_as_int(const Ndarray *ndarray) {
  std::lock_guard<std::recursive_mutex> submission_lock(
      runtime_resource_submission_mutex_);
  auto leases = acquire_ndarray_leases({ndarray});
  uint64_t *data_ptr{nullptr};
  if (arch_is_cpu(compile_config().arch) ||
      compile_config().arch == Arch::cuda ||
      compile_config().arch == Arch::amdgpu) {
    // For the LLVM backends, device allocation is a physical pointer.
    data_ptr =
        program_impl_->get_device_alloc_info_ptr(ndarray->ndarray_alloc_);
  }

  // Native CUDA primitives consume the raw pointer after this helper returns.
  // Pin before releasing the submission gate; delete/reset then either sees
  // the existing in-flight owner or waits for backend synchronization.
  if (arch_is_gpu(compile_config().arch)) {
    pin_ndarray_launch_leases(leases);
  }

  return reinterpret_cast<intptr_t>(data_ptr);
}

Program::ExternalCudaSubmissionScope::ExternalCudaSubmissionScope(
    Program *program)
    : program_(program),
      guard_(program->acquire_runtime_resource_submission_guard()) {
  TI_ERROR_IF(program_->compile_config().arch != Arch::cuda,
              "External CUDA provider submission requires the CUDA backend");
  TI_ERROR_IF(program_->finalized_,
              "Cannot register an external CUDA provider submission after "
              "Program finalize");
}

void Program::ExternalCudaSubmissionScope::commit(
    const std::vector<const Ndarray *> &ndarrays,
    bool failed) {
  TI_ERROR_IF(committed_,
              "External CUDA provider submission was already committed");
  auto leases = program_->acquire_ndarray_leases(ndarrays);
  program_->pin_ndarray_launch_leases(leases);
  program_->mark_runtime_submission(RuntimeSubmissionKind::kNative);
  if (failed) {
    program_->record_runtime_submission_failure();
  }
  committed_ = true;
}

std::unique_ptr<Program::ExternalCudaSubmissionScope>
Program::begin_external_cuda_submission() {
  return std::unique_ptr<ExternalCudaSubmissionScope>(
      new ExternalCudaSubmissionScope(this));
}

void Program::fill_ndarray_fast_u32(Ndarray *ndarray, uint32_t val) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!ndarray, "fill_ndarray_fast_u32 received a null ndarray.");
  auto leases = acquire_ndarray_leases({ndarray});
  const std::size_t bytes =
      ndarray->get_nelement() * ndarray->get_element_size();
  if (bytes == 0) {
    return;
  }
  if (compile_config().arch == Arch::vulkan) {
    const DeviceAllocation alloc = ndarray->ndarray_alloc_;
    enqueue_compute_op_lambda(
        [alloc, bytes, val](Device * /*device*/, CommandList *cmdlist) {
          cmdlist->buffer_fill(alloc.get_ptr(0), bytes, val);
          cmdlist->buffer_barrier(alloc);
        },
        {});
    return;
  }
  if (compile_config().arch == Arch::cuda && val == 0 &&
      bytes % sizeof(uint32_t) != 0) {
#ifdef TI_WITH_CUDA
    auto *raw_ptr =
        program_impl_->get_device_alloc_info_ptr(ndarray->ndarray_alloc_);
    TI_ERROR_IF(!raw_ptr, "CUDA ndarray fill received a null data pointer.");
    CUDADriver::get_instance().memset(reinterpret_cast<void *>(raw_ptr), 0,
                                      bytes);
    return;
#else
    TI_NOT_IMPLEMENTED;
#endif
  }
  if (arch_is_cpu(compile_config().arch)) {
    auto *raw_ptr =
        program_impl_->get_device_alloc_info_ptr(ndarray->ndarray_alloc_);
    TI_ERROR_IF(!raw_ptr, "CPU ndarray fill received a null data pointer.");
    if (val == 0) {
      std::memset(raw_ptr, 0, bytes);
      return;
    }
    const std::size_t words = bytes / sizeof(uint32_t);
    auto *ptr = reinterpret_cast<uint32_t *>(raw_ptr);
    TI_ERROR_IF(!ptr, "CPU ndarray fill received a null data pointer.");
    const int max_threads =
        std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
    const int chunk_items = 32768;
    const int target_threads = static_cast<int>(
        std::min<std::size_t>((words + chunk_items - 1) / chunk_items,
                              static_cast<std::size_t>(max_threads)));
    if (words >= 65536 && target_threads > 1) {
      CpuFillU32TaskContext ctx;
      ctx.data = ptr;
      ctx.words = words;
      ctx.value = val;
      ctx.num_threads = target_threads;
      auto pool = get_cpu_primitive_thread_pool(max_threads);
      pool->run(target_threads, target_threads, &ctx, cpu_fill_u32_task);
      return;
    }
    std::fill(ptr, ptr + words, val);
    return;
  }
  // This is a temporary solution to bypass device api on LLVM backends.
  program_impl_->fill_ndarray(
      ndarray->ndarray_alloc_, bytes / sizeof(uint32_t), val);
}

void Program::copy_ndarray_fast(Ndarray *dst, Ndarray *src) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!dst || !src, "copy_ndarray_fast received a null ndarray.");
  auto leases = acquire_ndarray_leases({dst, src});
  const std::size_t dst_bytes = dst->get_nelement() * dst->get_element_size();
  const std::size_t src_bytes = src->get_nelement() * src->get_element_size();
  TI_ERROR_IF(dst_bytes != src_bytes,
              "copy_ndarray_fast requires source and destination to have the "
              "same byte size.");
  if (dst_bytes == 0 || dst == src) {
    return;
  }
  ScopedRuntimeTransferStatistics transfer_statistics(
      this, RuntimeTransferKind::kDeviceToDevice, dst_bytes);

  if (compile_config().arch == Arch::vulkan) {
    const DeviceAllocation dst_alloc = dst->ndarray_alloc_;
    const DeviceAllocation src_alloc = src->ndarray_alloc_;
    enqueue_compute_op_lambda(
        [dst_alloc, src_alloc, dst_bytes](Device * /*device*/,
                                          CommandList *cmdlist) {
          cmdlist->buffer_copy(dst_alloc.get_ptr(0), src_alloc.get_ptr(0),
                               dst_bytes);
          cmdlist->buffer_barrier(dst_alloc);
        },
        {});
    return;
  }

  if (arch_is_cpu(compile_config().arch)) {
    auto *dst_ptr = reinterpret_cast<uint8_t *>(
        program_impl_->get_device_alloc_info_ptr(dst->ndarray_alloc_));
    auto *src_ptr = reinterpret_cast<const uint8_t *>(
        program_impl_->get_device_alloc_info_ptr(src->ndarray_alloc_));
    TI_ERROR_IF(!dst_ptr || !src_ptr,
                "CPU ndarray copy received a null data pointer.");
    const int max_threads =
        std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
    const std::size_t chunk_bytes =
        dst_bytes <= (4 << 20) ? (1 << 20) : (256 << 10);
    const int target_threads = static_cast<int>(
        std::min<std::size_t>((dst_bytes + chunk_bytes - 1) / chunk_bytes,
                              static_cast<std::size_t>(max_threads)));
    if (dst_bytes >= (1 << 20) && target_threads > 1) {
      CpuCopyTaskContext ctx;
      ctx.dst = dst_ptr;
      ctx.src = src_ptr;
      ctx.bytes = dst_bytes;
      ctx.num_threads = target_threads;
      auto pool = get_cpu_primitive_thread_pool(max_threads);
      pool->run(target_threads, target_threads, &ctx, cpu_copy_task);
      return;
    }
    std::memcpy(dst_ptr, src_ptr, dst_bytes);
    return;
  }

  if (compile_config().arch == Arch::cuda ||
      compile_config().arch == Arch::amdgpu) {
    Device::memcpy_direct(dst->ndarray_alloc_.get_ptr(0),
                          src->ndarray_alloc_.get_ptr(0), dst_bytes);
    return;
  }

  Stream *stream = program_impl_->get_compute_device()->get_compute_stream();
  auto [cmdlist, res] = stream->new_command_list_unique();
  TI_ASSERT(res == RhiResult::success);
  cmdlist->buffer_copy(dst->ndarray_alloc_.get_ptr(0),
                       src->ndarray_alloc_.get_ptr(0), dst_bytes);
  stream->submit_synced(cmdlist.get());
}

void Program::record_vulkan_buffer_commands(
    const std::vector<VulkanBufferCommand> &commands) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan buffer command recording requires the Vulkan backend.");
  TI_ERROR_IF(commands.empty(),
              "Vulkan buffer command recording requires at least one command.");
  TI_ERROR_IF(commands.size() > 4096,
              "Vulkan buffer command recording exceeds the 4096-command "
              "safety limit.");

  auto resource_submission_guard = acquire_runtime_resource_submission_guard();
  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan buffer command recording has no compute device.");

  struct ResolvedCommand {
    VulkanBufferCommandKind kind;
    DeviceAllocation destination;
    DeviceAllocation source;
    std::size_t destination_offset;
    std::size_t source_offset;
    std::size_t bytes;
    std::uint32_t value;
  };
  std::vector<ResolvedCommand> resolved;
  std::vector<const Ndarray *> views;
  resolved.reserve(commands.size());
  views.reserve(commands.size() * 2);

  auto storage_bytes = [](const Ndarray *array) {
    TI_ERROR_IF(!array,
                "Vulkan buffer command received a null ndarray binding.");
    const std::size_t elements = array->get_nelement();
    const std::size_t element_bytes = array->get_element_size();
    TI_ERROR_IF(
        element_bytes != 0 &&
            elements > std::numeric_limits<std::size_t>::max() / element_bytes,
        "Vulkan buffer command ndarray size overflows size_t.");
    return elements * element_bytes;
  };
  auto validate_range = [&](const Ndarray *array, std::size_t offset,
                            std::size_t bytes, const char *role) {
    const std::size_t available = storage_bytes(array);
    TI_ERROR_IF(offset > available || bytes > available - offset,
                "Vulkan buffer command {} range offset {} plus {} bytes "
                "exceeds {} bytes.",
                role, offset, bytes, available);
    TI_ERROR_IF((offset & 3u) != 0 || (bytes & 3u) != 0,
                "Vulkan buffer command {} offset and size must be "
                "four-byte aligned.",
                role);
    const DeviceAllocation allocation = array->get_device_allocation();
    TI_ERROR_IF(array->owning_program() != this || allocation.device != device,
                "Vulkan buffer command {} belongs to another runtime or "
                "device.",
                role);
  };

  for (const auto &command : commands) {
    ResolvedCommand item{command.kind,          kDeviceNullAllocation,
                         kDeviceNullAllocation, command.destination_offset,
                         command.source_offset, command.bytes,
                         command.value};
    switch (command.kind) {
      case VulkanBufferCommandKind::kFillU32:
        TI_ERROR_IF(command.bytes == 0,
                    "Vulkan buffer fill requires a nonzero byte count.");
        validate_range(command.destination, command.destination_offset,
                       command.bytes, "destination");
        item.destination = command.destination->get_device_allocation();
        views.push_back(command.destination);
        break;
      case VulkanBufferCommandKind::kCopy: {
        TI_ERROR_IF(command.bytes == 0,
                    "Vulkan buffer copy requires a nonzero byte count.");
        validate_range(command.destination, command.destination_offset,
                       command.bytes, "destination");
        validate_range(command.source, command.source_offset, command.bytes,
                       "source");
        item.destination = command.destination->get_device_allocation();
        item.source = command.source->get_device_allocation();
        if (item.destination == item.source) {
          const std::size_t destination_end =
              command.destination_offset + command.bytes;
          const std::size_t source_end = command.source_offset + command.bytes;
          TI_ERROR_IF(command.destination_offset < source_end &&
                          command.source_offset < destination_end,
                      "Vulkan buffer copy regions overlap within one "
                      "allocation.");
        }
        views.push_back(command.destination);
        views.push_back(command.source);
        break;
      }
      case VulkanBufferCommandKind::kBufferBarrier:
        TI_ERROR_IF(!command.destination,
                    "Vulkan buffer barrier requires a destination binding.");
        validate_range(command.destination, 0, 0, "barrier");
        item.destination = command.destination->get_device_allocation();
        views.push_back(command.destination);
        break;
      case VulkanBufferCommandKind::kMemoryBarrier:
        break;
      default:
        TI_ERROR("Vulkan buffer command kind is invalid.");
    }
    resolved.push_back(item);
  }

  auto leases = acquire_ndarray_leases(views);
  enqueue_compute_op_lambda(
      [resolved = std::move(resolved)](Device * /*device*/,
                                       CommandList *cmdlist) {
        for (const auto &command : resolved) {
          switch (command.kind) {
            case VulkanBufferCommandKind::kFillU32:
              cmdlist->buffer_fill(
                  command.destination.get_ptr(command.destination_offset),
                  command.bytes, command.value);
              break;
            case VulkanBufferCommandKind::kCopy:
              cmdlist->buffer_copy(
                  command.destination.get_ptr(command.destination_offset),
                  command.source.get_ptr(command.source_offset), command.bytes);
              break;
            case VulkanBufferCommandKind::kBufferBarrier:
              cmdlist->buffer_barrier(command.destination);
              break;
            case VulkanBufferCommandKind::kMemoryBarrier:
              cmdlist->memory_barrier();
              break;
          }
        }
      },
      {});
  mark_runtime_submission_pending();
  pin_ndarray_launch_leases(leases);
}

void Program::copy_ndarray_from_host(Ndarray *dst,
                                     const void *src,
                                     std::size_t bytes) {
  std::lock_guard<std::recursive_mutex> submission_lock(
      runtime_resource_submission_mutex_);
  TI_ERROR_IF(!dst || !src,
              "copy_ndarray_from_host received a null pointer.");
  auto leases = acquire_ndarray_leases({dst});
  const std::size_t expected_bytes =
      dst->get_nelement() * dst->get_element_size();
  TI_ERROR_IF(bytes != expected_bytes,
              "copy_ndarray_from_host expected {} bytes, but received {}.",
              expected_bytes, bytes);
  if (bytes == 0) {
    return;
  }
  ScopedRuntimeTransferStatistics transfer_statistics(
      this, RuntimeTransferKind::kHostToDevice, bytes);

  if (arch_is_cpu(compile_config().arch)) {
    auto *dst_ptr = reinterpret_cast<uint8_t *>(
        program_impl_->get_device_alloc_info_ptr(dst->ndarray_alloc_));
    TI_ERROR_IF(!dst_ptr,
                "CPU ndarray host upload received a null data pointer.");
    std::memcpy(dst_ptr, src, bytes);
    return;
  }

  auto *device = program_impl_->get_compute_device();
  DevicePtr dst_ptr = dst->ndarray_alloc_.get_ptr(0);
  const void *src_ptr = src;
  std::size_t size = bytes;
  const RhiResult res = device->upload_data(&dst_ptr, &src_ptr, &size, 1);
  TI_ERROR_IF(res != RhiResult::success,
              "copy_ndarray_from_host failed: {}", res);
}

void Program::copy_ndarray_to_host(Ndarray *src,
                                   void *dst,
                                   std::size_t bytes) {
  std::lock_guard<std::recursive_mutex> submission_lock(
      runtime_resource_submission_mutex_);
  TI_ERROR_IF(!src || !dst, "copy_ndarray_to_host received a null pointer.");
  auto leases = acquire_ndarray_leases({src});
  const std::size_t expected_bytes =
      src->get_nelement() * src->get_element_size();
  TI_ERROR_IF(bytes != expected_bytes,
              "copy_ndarray_to_host expected {} bytes, but received {}.",
              expected_bytes, bytes);
  if (bytes == 0) {
    return;
  }
  ScopedRuntimeTransferStatistics transfer_statistics(
      this, RuntimeTransferKind::kDeviceToHost, bytes);

  if (arch_is_cpu(compile_config().arch)) {
    auto *src_ptr = reinterpret_cast<const uint8_t *>(
        program_impl_->get_device_alloc_info_ptr(src->ndarray_alloc_));
    TI_ERROR_IF(!src_ptr,
                "CPU ndarray host readback received a null data pointer.");
    std::memcpy(dst, src_ptr, bytes);
    return;
  }

  auto *device = program_impl_->get_compute_device();
  DevicePtr src_ptr = src->ndarray_alloc_.get_ptr(0);
  void *dst_ptr = dst;
  std::size_t size = bytes;
  const RhiResult res = device->readback_data(&src_ptr, &dst_ptr, &size, 1);
  TI_ERROR_IF(res != RhiResult::success,
              "copy_ndarray_to_host failed: {}", res);
}

void Program::copy_ndarrays_to_host(const Ndarray *const *srcs,
                                    void *const *dsts,
                                    const std::size_t *bytes,
                                    std::size_t count) {
  std::lock_guard<std::recursive_mutex> submission_lock(
      runtime_resource_submission_mutex_);
  TI_ERROR_IF(count > 0 && (!srcs || !dsts || !bytes),
              "copy_ndarrays_to_host received a null pointer table.");
  if (count == 0) {
    return;
  }
  TI_ERROR_IF(count > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "copy_ndarrays_to_host count {} exceeds the RHI limit.",
              count);
  auto leases = acquire_ndarray_leases(srcs, count);
  std::size_t total_bytes = 0;
  for (std::size_t i = 0; i < count; ++i) {
    TI_ERROR_IF(!srcs[i] || !dsts[i],
                "copy_ndarrays_to_host received a null pointer at index {}.",
                i);
    const std::size_t expected_bytes =
        srcs[i]->get_nelement() * srcs[i]->get_element_size();
    TI_ERROR_IF(bytes[i] != expected_bytes,
                "copy_ndarrays_to_host expected {} bytes at index {}, but "
                "received {}.",
                expected_bytes, i, bytes[i]);
    TI_ERROR_IF(bytes[i] > std::numeric_limits<std::size_t>::max() -
                               total_bytes,
                "copy_ndarrays_to_host byte count overflow.");
    total_bytes += bytes[i];
  }
  if (total_bytes == 0) {
    return;
  }
  ScopedRuntimeTransferStatistics transfer_statistics(
      this, RuntimeTransferKind::kDeviceToHost, total_bytes);

  if (arch_is_cpu(compile_config().arch)) {
    for (std::size_t i = 0; i < count; ++i) {
      if (bytes[i] == 0) {
        continue;
      }
      auto *src_ptr = reinterpret_cast<const uint8_t *>(
          program_impl_->get_device_alloc_info_ptr(srcs[i]->ndarray_alloc_));
      TI_ERROR_IF(!src_ptr,
                  "CPU ndarray host readback received a null data pointer.");
      std::memcpy(dsts[i], src_ptr, bytes[i]);
    }
    return;
  }

  if (compile_config().arch == Arch::vulkan) {
    (void)flush_if_pending();
  }
  auto *device = program_impl_->get_compute_device();
  std::vector<DevicePtr> device_ptrs;
  std::vector<void *> host_ptrs;
  std::vector<std::size_t> copy_sizes;
  device_ptrs.reserve(count);
  host_ptrs.reserve(count);
  copy_sizes.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    if (bytes[i] == 0) {
      continue;
    }
    device_ptrs.push_back(srcs[i]->ndarray_alloc_.get_ptr(0));
    host_ptrs.push_back(dsts[i]);
    copy_sizes.push_back(bytes[i]);
  }
  const RhiResult res = device->readback_data(
      device_ptrs.data(), host_ptrs.data(), copy_sizes.data(),
      static_cast<int>(device_ptrs.size()));
  TI_ERROR_IF(res != RhiResult::success,
              "copy_ndarrays_to_host failed: {}", res);
}

void Program::copy_graph_observations_to_host(
    const Ndarray *const *srcs,
    void *const *dsts,
    const std::size_t *bytes,
    std::size_t count) {
  std::lock_guard<std::recursive_mutex> submission_lock(
      runtime_resource_submission_mutex_);
  TI_ERROR_IF(count > 0 && (!srcs || !dsts || !bytes),
              "copy_graph_observations_to_host received a null pointer table.");
  if (count == 0) {
    return;
  }
  TI_ERROR_IF(count > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "copy_graph_observations_to_host count {} exceeds the RHI limit.",
              count);
  auto leases = acquire_ndarray_leases(srcs, count);
  std::size_t total_bytes = 0;
  bool portable_packed_layout = true;
  for (std::size_t i = 0; i < count; ++i) {
    TI_ERROR_IF(!srcs[i] || !dsts[i],
                "copy_graph_observations_to_host received a null pointer at "
                "index {}.",
                i);
    const std::size_t expected_bytes =
        srcs[i]->get_nelement() * srcs[i]->get_element_size();
    TI_ERROR_IF(bytes[i] != expected_bytes,
                "copy_graph_observations_to_host expected {} bytes at index "
                "{}, but received {}.",
                expected_bytes, i, bytes[i]);
    TI_ERROR_IF(bytes[i] > std::numeric_limits<std::size_t>::max() -
                               total_bytes,
                "copy_graph_observations_to_host byte count overflow.");
    total_bytes += bytes[i];
    portable_packed_layout &= bytes[i] % 4 == 0;
  }
  if (total_bytes == 0) {
    return;
  }
  ScopedRuntimeTransferStatistics transfer_statistics(
      this, RuntimeTransferKind::kDeviceToHost, total_bytes);

  if (arch_is_cpu(compile_config().arch)) {
    graph_observation_staging_.direct_batches += 1;
    for (std::size_t i = 0; i < count; ++i) {
      if (bytes[i] == 0) {
        continue;
      }
      auto *src_ptr = reinterpret_cast<const uint8_t *>(
          program_impl_->get_device_alloc_info_ptr(srcs[i]->ndarray_alloc_));
      TI_ERROR_IF(!src_ptr,
                  "CPU Graph observation received a null data pointer.");
      std::memcpy(dsts[i], src_ptr, bytes[i]);
    }
    return;
  }

  if (compile_config().arch == Arch::vulkan) {
    (void)flush_if_pending();
  }
  auto *device = program_impl_->get_compute_device();
  std::vector<DevicePtr> device_ptrs;
  std::vector<void *> host_ptrs;
  std::vector<std::size_t> copy_sizes;
  device_ptrs.reserve(count);
  host_ptrs.reserve(count);
  copy_sizes.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    if (bytes[i] == 0) {
      continue;
    }
    device_ptrs.push_back(srcs[i]->ndarray_alloc_.get_ptr(0));
    host_ptrs.push_back(dsts[i]);
    copy_sizes.push_back(bytes[i]);
  }

  constexpr std::size_t kMaxPersistentObservationBytes = 4096;
  const bool persistent_staging_enabled =
      compile_config().arch == Arch::vulkan && portable_packed_layout &&
      total_bytes <= kMaxPersistentObservationBytes &&
      get_environ_config("TI_GRAPH_PERSISTENT_OBSERVATION_STAGING", 1) != 0;
  if (persistent_staging_enabled) {
    bool reused = graph_observation_staging_.readback != nullptr &&
                  graph_observation_staging_.capacity >= total_bytes;
    if (!reused) {
      std::size_t capacity = 64;
      while (capacity < total_bytes) {
        capacity *= 2;
      }
      auto [allocation, result] = device->allocate_memory_unique(
          {capacity, /*host_write=*/false, /*host_read=*/true,
           /*export_sharing=*/false, AllocUsage::None});
      if (result == RhiResult::success) {
        graph_observation_staging_.readback = std::move(allocation);
        graph_observation_staging_.capacity = capacity;
        graph_observation_staging_.allocations += 1;
      } else {
        graph_observation_staging_.fallback_batches += 1;
      }
    }
    if (graph_observation_staging_.readback != nullptr &&
        graph_observation_staging_.capacity >= total_bytes) {
      if (reused) {
        graph_observation_staging_.reuses += 1;
      }
      const RhiResult result = device->readback_data_packed(
          device_ptrs.data(), host_ptrs.data(), copy_sizes.data(),
          static_cast<int>(device_ptrs.size()),
          graph_observation_staging_.readback->get_ptr(0),
          graph_observation_staging_.capacity);
      if (result == RhiResult::success) {
        graph_observation_staging_.packed_batches += 1;
        graph_observation_staging_.packed_payload_bytes += total_bytes;
        return;
      }
      graph_observation_staging_.fallback_batches += 1;
    }
  } else {
    graph_observation_staging_.direct_batches += 1;
  }

  const RhiResult result = device->readback_data(
      device_ptrs.data(), host_ptrs.data(), copy_sizes.data(),
      static_cast<int>(device_ptrs.size()));
  TI_ERROR_IF(result != RhiResult::success,
              "copy_graph_observations_to_host failed: {}", result);
}

void Program::copy_host_readable_graph_observations_to_host(
    const Ndarray *const *srcs,
    void *const *dsts,
    const std::size_t *bytes,
    std::size_t count) {
  std::lock_guard<std::recursive_mutex> submission_lock(
      runtime_resource_submission_mutex_);
  TI_ERROR_IF(count > 0 && (!srcs || !dsts || !bytes),
              "copy_host_readable_graph_observations_to_host received a null "
              "pointer table.");
  if (count == 0) {
    return;
  }
  auto leases = acquire_ndarray_leases(srcs, count);
  std::size_t total_bytes = 0;
  for (std::size_t i = 0; i < count; ++i) {
    TI_ERROR_IF(!srcs[i] || !dsts[i],
                "Host-readable Graph observation received a null pointer at "
                "index {}.",
                i);
    TI_ERROR_IF(!srcs[i]->is_host_readable(),
                "Graph observation source {} is not completion-attached "
                "host-readable storage.",
                i);
    const std::size_t expected_bytes =
        srcs[i]->get_nelement() * srcs[i]->get_element_size();
    TI_ERROR_IF(bytes[i] != expected_bytes,
                "Host-readable Graph observation expected {} bytes at index "
                "{}, but received {}.",
                expected_bytes, i, bytes[i]);
    TI_ERROR_IF(bytes[i] > std::numeric_limits<std::size_t>::max() -
                               total_bytes,
                "Host-readable Graph observation byte count overflow.");
    total_bytes += bytes[i];
  }

  const Arch arch = compile_config().arch;
  if (arch_is_cpu(arch) || arch == Arch::cuda) {
    for (std::size_t i = 0; i < count; ++i) {
      if (bytes[i] == 0) {
        continue;
      }
      const auto *src_ptr = reinterpret_cast<const uint8_t *>(
          program_impl_->get_device_alloc_info_ptr(srcs[i]->ndarray_alloc_));
      TI_ERROR_IF(!src_ptr,
                  "Host-readable Graph observation received a null data "
                  "pointer at index {}.",
                  i);
      std::memcpy(dsts[i], src_ptr, bytes[i]);
    }
  } else if (arch == Arch::vulkan) {
    Device *device = program_impl_->get_compute_device();
    TI_ERROR_IF(!device,
                "Host-readable Vulkan Graph observation requires a compute "
                "device.");
    for (std::size_t i = 0; i < count; ++i) {
      if (bytes[i] == 0) {
        continue;
      }
      void *mapped = nullptr;
      const RhiResult result = device->map(srcs[i]->ndarray_alloc_, &mapped);
      TI_ERROR_IF(result != RhiResult::success || !mapped,
                  "Unable to map host-readable Graph observation {}: {}", i,
                  result);
      std::memcpy(dsts[i], mapped, bytes[i]);
      device->unmap(srcs[i]->ndarray_alloc_);
    }
  } else {
    TI_ERROR("Completion-attached Graph observation storage is unavailable on "
             "architecture {}.",
             arch_name(arch));
  }
  graph_observation_staging_.completion_attached_batches += 1;
  graph_observation_staging_.completion_attached_bytes += total_bytes;
}

std::uint64_t Program::create_cuda_graph_observation_readback(
    std::size_t bytes) {
  std::lock_guard<std::recursive_mutex> submission_lock(
      runtime_resource_submission_mutex_);
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA Graph observation readback requires the CUDA backend.");
  TI_ERROR_IF(finalized_,
              "Cannot create CUDA Graph observation readback after Program "
              "finalize.");
  TI_ERROR_IF(bytes == 0,
              "CUDA Graph observation readback capacity must be positive.");
#ifdef TI_WITH_CUDA
  const std::uint64_t handle =
      graph_observation_staging_.next_cuda_readback_handle++;
  TI_ERROR_IF(handle == 0,
              "CUDA Graph observation readback handle overflow.");
  void *host_ptr = nullptr;
  {
    auto context_guard = CUDAContext::get_instance().get_guard();
    CUDADriver::get_instance().mem_host_alloc(&host_ptr, bytes, 0);
  }
  TI_ERROR_IF(!host_ptr,
              "CUDA Graph observation pinned-host allocation returned null.");
  try {
    const bool inserted =
        graph_observation_staging_.cuda_readbacks
            .emplace(handle,
                     GraphObservationStagingState::CudaPinnedReadback{
                         host_ptr, bytes})
            .second;
    TI_ERROR_IF(!inserted,
                "CUDA Graph observation readback handle collision.");
  } catch (...) {
    auto context_guard = CUDAContext::get_instance().get_guard();
    CUDADriver::get_instance().mem_free_host(host_ptr);
    throw;
  }
  graph_observation_staging_.cuda_pinned_bytes += bytes;
  graph_observation_staging_.allocations += 1;
  return handle;
#else
  TI_NOT_IMPLEMENTED;
#endif
}

void Program::destroy_cuda_graph_observation_readback(std::uint64_t handle) {
  std::lock_guard<std::recursive_mutex> submission_lock(
      runtime_resource_submission_mutex_);
  const auto found = graph_observation_staging_.cuda_readbacks.find(handle);
  if (found == graph_observation_staging_.cuda_readbacks.end()) {
    return;
  }
#ifdef TI_WITH_CUDA
  if (found->second.host_ptr && runtime_fault_domain_->backend_calls_safe()) {
    auto context_guard = CUDAContext::get_instance().get_guard();
    CUDADriver::get_instance().mem_free_host(found->second.host_ptr);
  }
#endif
  TI_ASSERT(graph_observation_staging_.cuda_pinned_bytes >=
            found->second.capacity);
  graph_observation_staging_.cuda_pinned_bytes -= found->second.capacity;
  graph_observation_staging_.cuda_readbacks.erase(found);
}

void Program::enqueue_cuda_graph_observation_readback(
    std::uint64_t handle,
    const Ndarray *const *srcs,
    const std::size_t *bytes,
    std::size_t count) {
  std::lock_guard<std::recursive_mutex> submission_lock(
      runtime_resource_submission_mutex_);
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA Graph observation readback requires the CUDA backend.");
  TI_ERROR_IF(count > 0 && (!srcs || !bytes),
              "CUDA Graph observation readback received a null pointer table.");
  const auto found = graph_observation_staging_.cuda_readbacks.find(handle);
  TI_ERROR_IF(found == graph_observation_staging_.cuda_readbacks.end(),
              "CUDA Graph observation readback handle is stale.");
  auto leases = acquire_ndarray_leases(srcs, count);
  std::size_t total_bytes = 0;
  for (std::size_t i = 0; i < count; ++i) {
    TI_ERROR_IF(!srcs[i],
                "CUDA Graph observation source {} is null.", i);
    const std::size_t expected_bytes =
        srcs[i]->get_nelement() * srcs[i]->get_element_size();
    TI_ERROR_IF(bytes[i] != expected_bytes,
                "CUDA Graph observation expected {} bytes at index {}, but "
                "received {}.",
                expected_bytes, i, bytes[i]);
    TI_ERROR_IF(bytes[i] > std::numeric_limits<std::size_t>::max() -
                               total_bytes,
                "CUDA Graph observation byte count overflow.");
    total_bytes += bytes[i];
  }
  TI_ERROR_IF(total_bytes > found->second.capacity,
              "CUDA Graph observation payload {} exceeds pinned capacity {}.",
              total_bytes, found->second.capacity);
  ScopedRuntimeTransferStatistics transfer_statistics(
      this, RuntimeTransferKind::kDeviceToHost, total_bytes);
#ifdef TI_WITH_CUDA
  auto submission_guard =
      CUDAContext::get_instance().get_submission_lock_guard();
  auto context_guard = CUDAContext::get_instance().get_guard();
  auto &driver = CUDADriver::get_instance();
  auto *host_ptr = static_cast<std::uint8_t *>(found->second.host_ptr);
  std::size_t offset = 0;
  for (std::size_t i = 0; i < count; ++i) {
    if (bytes[i] == 0) {
      continue;
    }
    auto *src_ptr =
        program_impl_->get_device_alloc_info_ptr(srcs[i]->ndarray_alloc_);
    TI_ERROR_IF(!src_ptr,
                "CUDA Graph observation received a null device pointer at "
                "index {}.",
                i);
    driver.memcpy_device_to_host_async(host_ptr + offset, src_ptr, bytes[i],
                                       nullptr);
    offset += bytes[i];
  }
  pin_ndarray_launch_leases(leases);
  mark_runtime_submission_pending();
  graph_observation_staging_.completion_attached_batches += 1;
  graph_observation_staging_.completion_attached_bytes += total_bytes;
#else
  TI_NOT_IMPLEMENTED;
#endif
}

void Program::copy_cuda_graph_observation_readback_to_host(
    std::uint64_t handle,
    void *const *dsts,
    const std::size_t *bytes,
    std::size_t count) {
  std::lock_guard<std::recursive_mutex> submission_lock(
      runtime_resource_submission_mutex_);
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA Graph observation readback requires the CUDA backend.");
  TI_ERROR_IF(count > 0 && (!dsts || !bytes),
              "CUDA Graph observation host copy received a null pointer "
              "table.");
  const auto found = graph_observation_staging_.cuda_readbacks.find(handle);
  TI_ERROR_IF(found == graph_observation_staging_.cuda_readbacks.end(),
              "CUDA Graph observation readback handle is stale.");
  std::size_t total_bytes = 0;
  for (std::size_t i = 0; i < count; ++i) {
    TI_ERROR_IF(!dsts[i],
                "CUDA Graph observation destination {} is null.", i);
    TI_ERROR_IF(bytes[i] > std::numeric_limits<std::size_t>::max() -
                               total_bytes,
                "CUDA Graph observation host-copy byte count overflow.");
    total_bytes += bytes[i];
  }
  TI_ERROR_IF(total_bytes > found->second.capacity,
              "CUDA Graph observation host-copy payload {} exceeds pinned "
              "capacity {}.",
              total_bytes, found->second.capacity);
  const auto *host_ptr =
      static_cast<const std::uint8_t *>(found->second.host_ptr);
  std::size_t offset = 0;
  for (std::size_t i = 0; i < count; ++i) {
    if (bytes[i] == 0) {
      continue;
    }
    std::memcpy(dsts[i], host_ptr + offset, bytes[i]);
    offset += bytes[i];
  }
}

GraphObservationStagingStatistics
Program::graph_observation_staging_statistics() {
  std::lock_guard<std::recursive_mutex> submission_lock(
      runtime_resource_submission_mutex_);
  GraphObservationStagingStatistics result;
  result.persistent_bytes = graph_observation_staging_.capacity +
                            graph_observation_staging_.cuda_pinned_bytes;
  result.allocations = graph_observation_staging_.allocations;
  result.reuses = graph_observation_staging_.reuses;
  result.packed_batches = graph_observation_staging_.packed_batches;
  result.direct_batches = graph_observation_staging_.direct_batches;
  result.fallback_batches = graph_observation_staging_.fallback_batches;
  result.packed_payload_bytes =
      graph_observation_staging_.packed_payload_bytes;
  result.completion_attached_batches =
      graph_observation_staging_.completion_attached_batches;
  result.completion_attached_bytes =
      graph_observation_staging_.completion_attached_bytes;
  return result;
}

SNodeRuntimeDirectoryStatistics
Program::debug_snode_runtime_directory_statistics() const {
  std::shared_lock<std::shared_mutex> lifecycle_lock(
      snode_tree_lifecycle_mutex_);
  return program_impl_->get_snode_runtime_directory_statistics();
}

bool Program::cuda_matrix_mma_f16_f32_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda &&
         cuda::driver_matrix_mma_f16_f32_available();
#else
  return false;
#endif
}

bool Program::cuda_async_tile_available() const {
#ifdef TI_WITH_CUDA
  if (compile_config().arch != Arch::cuda) {
    return false;
  }
  auto &context = CUDAContext::get_instance();
  return cuda::detail::cuda_async_tile_copy_admitted(
      context.get_compute_capability(), context.get_ptx_version(),
      cuda::detail::kCudaAsyncTileMinBlsBytes, /*copy_bytes=*/4,
      /*direct_global_to_bls_copy=*/true, /*read_only_bls=*/true);
#else
  return false;
#endif
}

void Program::record_cuda_async_tile_lowering(
    std::size_t copy_sites) noexcept {
  if (copy_sites == 0) {
    return;
  }
  cuda_async_tile_lowered_specializations_.fetch_add(
      1, std::memory_order_relaxed);
  cuda_async_tile_copy_sites_.fetch_add(copy_sites,
                                        std::memory_order_relaxed);
}

void Program::record_cuda_async_tile_candidate(
    cuda::detail::CudaAsyncTileAdmissionReason reason) noexcept {
  using Reason = cuda::detail::CudaAsyncTileAdmissionReason;
  cuda_async_tile_candidates_.fetch_add(1, std::memory_order_relaxed);
  if (reason == Reason::kAdmitted) {
    cuda_async_tile_admitted_.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  cuda_async_tile_rejected_.fetch_add(1, std::memory_order_relaxed);
  std::size_t index = 0;
  switch (reason) {
    case Reason::kBelowSize:
      index = 0;
      break;
    case Reason::kReadWriteBls:
      index = 1;
      break;
    case Reason::kUnsupportedWidth:
      index = 2;
      break;
    case Reason::kNonDirectAddress:
      index = 3;
      break;
    case Reason::kAliasUnknown:
      index = 4;
      break;
    case Reason::kSharedMemoryPressure:
      index = 5;
      break;
    case Reason::kTargetCapability:
      index = 6;
      break;
    case Reason::kCostGate:
      index = 7;
      break;
    case Reason::kAdmitted:
      TI_UNREACHABLE;
  }
  cuda_async_tile_rejection_reasons_[index].fetch_add(
      1, std::memory_order_relaxed);
}

std::unordered_map<std::string, std::uint64_t>
Program::cuda_async_tile_statistics() const {
  return {
      {"candidates",
       cuda_async_tile_candidates_.load(std::memory_order_relaxed)},
      {"admitted",
       cuda_async_tile_admitted_.load(std::memory_order_relaxed)},
      {"lowered", cuda_async_tile_copy_sites_.load(std::memory_order_relaxed)},
      {"fallback",
       cuda_async_tile_rejected_.load(std::memory_order_relaxed)},
      {"rejected",
       cuda_async_tile_rejected_.load(std::memory_order_relaxed)},
      {"below_size",
       cuda_async_tile_rejection_reasons_[0].load(
           std::memory_order_relaxed)},
      {"read_write_bls",
       cuda_async_tile_rejection_reasons_[1].load(
           std::memory_order_relaxed)},
      {"unsupported_width",
       cuda_async_tile_rejection_reasons_[2].load(
           std::memory_order_relaxed)},
      {"non_direct_address",
       cuda_async_tile_rejection_reasons_[3].load(
           std::memory_order_relaxed)},
      {"alias_unknown",
       cuda_async_tile_rejection_reasons_[4].load(
           std::memory_order_relaxed)},
      {"shared_memory_pressure",
       cuda_async_tile_rejection_reasons_[5].load(
           std::memory_order_relaxed)},
      {"target_capability",
       cuda_async_tile_rejection_reasons_[6].load(
           std::memory_order_relaxed)},
      {"cost_gate",
       cuda_async_tile_rejection_reasons_[7].load(
           std::memory_order_relaxed)},
      {"lowered_specializations",
       cuda_async_tile_lowered_specializations_.load(
           std::memory_order_relaxed)},
      {"copy_sites",
       cuda_async_tile_copy_sites_.load(std::memory_order_relaxed)},
  };
}

std::size_t Program::cuda_matrix_mma_f16_f32(Ndarray *a,
                                              Ndarray *b,
                                              Ndarray *output,
                                              std::size_t batch_count) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA matrix MMA is only available on CUDA.");
  TI_ERROR_IF(!a || !b || !output,
              "CUDA matrix MMA received a null ndarray.");
  TI_ERROR_IF(batch_count == 0 ||
                  batch_count > static_cast<std::size_t>(
                                    std::numeric_limits<int>::max()),
              "CUDA matrix MMA batch count must be in [1, INT_MAX].");
  TI_ERROR_IF(batch_count >
                  std::numeric_limits<std::size_t>::max() / 256,
              "CUDA matrix MMA batch size overflow.");
  const std::size_t scalar_count = batch_count * 256;
  auto check_array = [scalar_count](const char *name, Ndarray *array,
                                    DataType dtype, std::size_t item_bytes) {
    TI_ERROR_IF(!array->get_element_shape().empty() ||
                    array->get_element_data_type() != dtype ||
                    array->get_nelement() != scalar_count ||
                    array->get_element_size() != item_bytes,
                "CUDA matrix MMA {} must be a compact scalar ndarray with "
                "exactly batch_count * 256 entries.",
                name);
  };
  check_array("A", a, PrimitiveType::f16, sizeof(std::uint16_t));
  check_array("B", b, PrimitiveType::f16, sizeof(std::uint16_t));
  check_array("output", output, PrimitiveType::f32, sizeof(float32));
  TI_ERROR_IF(output->get_device_allocation() == a->get_device_allocation() ||
                  output->get_device_allocation() ==
                      b->get_device_allocation(),
              "CUDA matrix MMA output must not alias an input allocation.");
#ifdef TI_WITH_CUDA
  TI_ERROR_IF(!cuda::driver_matrix_mma_f16_f32_available(),
              "CUDA matrix MMA requires NVIDIA compute capability 7.0 or "
              "newer.");
  auto *a_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(a));
  auto *b_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(b));
  auto *output_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output));
  return cuda::driver_matrix_mma_f16_f32(
      a_ptr, b_ptr, output_ptr, static_cast<int>(batch_count));
#else
  TI_ERROR("CUDA matrix MMA requires TI_WITH_CUDA=ON.");
#endif
}

bool Program::cuda_device_transform_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda &&
         cuda::driver_transform_available();
#else
  return false;
#endif
}

bool Program::cuda_toolkit_transform_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda &&
         cuda::cub_transform_available();
#else
  return false;
#endif
}

std::size_t Program::cuda_device_transform_affine_ndarray(Ndarray *src,
                                                          Ndarray *dst,
                                                          int value_type,
                                                          double scale,
                                                          double bias) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA device transform is only available on CUDA.");
  TI_ERROR_IF(!src || !dst, "CUDA device transform received a null ndarray.");
  TI_ERROR_IF(src->get_nelement() != dst->get_nelement(),
              "CUDA device transform source and destination sizes differ.");
  TI_ERROR_IF(src->get_element_size() != dst->get_element_size(),
              "CUDA device transform source and destination dtypes differ.");
  TI_ERROR_IF(value_type < 0 || value_type > 5,
              "CUDA device transform received an unsupported value type.");
  const std::size_t expected_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(src->get_element_size() != expected_size,
              "CUDA device transform dtype does not match value type.");
  TI_ERROR_IF(src->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA device transform currently supports at most INT_MAX items.");
#ifdef TI_WITH_CUDA
  const auto cuda_value_type =
      static_cast<cuda::CudaTransformValueType>(value_type);
  auto *src_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(src));
  auto *dst_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(dst));
  TI_ERROR_IF(!cuda::driver_transform_available(),
              "CUDA device transform requires CUDA driver API support.");
  return cuda::driver_transform_affine(
      src_ptr, dst_ptr, static_cast<int>(src->get_nelement()), cuda_value_type,
      scale, bias);
#else
  TI_ERROR("CUDA device transform requires TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_device_transform_affine_member_ndarray(
    Ndarray *src,
    Ndarray *dst,
    int value_type,
    std::size_t offset,
    std::size_t stride,
    double scale,
    double bias) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA strided transform is only available on CUDA.");
  check_transform_member_request("CUDA", src, dst, value_type, offset, stride);
  TI_ERROR_IF(src->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA strided transform currently supports at most INT_MAX "
              "items.");
#ifdef TI_WITH_CUDA
  TI_ERROR_IF(!cuda::driver_transform_available(),
              "CUDA strided transform requires CUDA driver API support.");
  auto *src_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(src));
  auto *dst_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(dst));
  return cuda::driver_transform_affine_strided(
      src_ptr, dst_ptr, static_cast<int>(src->get_nelement()),
      static_cast<cuda::CudaTransformValueType>(value_type), offset, stride,
      offset, stride, scale, bias);
#else
  TI_ERROR("CUDA strided transform requires TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_device_transform_affine_strided_ndarray(
    Ndarray *src,
    Ndarray *dst,
    int value_type,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride,
    double scale,
    double bias) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA strided transform is only available on CUDA.");
  check_transform_strided_request("CUDA", src, dst, value_type, src_offset,
                                  src_stride, dst_offset, dst_stride);
  TI_ERROR_IF(src->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA strided transform currently supports at most INT_MAX "
              "items.");
#ifdef TI_WITH_CUDA
  TI_ERROR_IF(!cuda::driver_transform_available(),
              "CUDA strided transform requires CUDA driver API support.");
  auto *src_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(src));
  auto *dst_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(dst));
  return cuda::driver_transform_affine_strided(
      src_ptr, dst_ptr, static_cast<int>(src->get_nelement()),
      static_cast<cuda::CudaTransformValueType>(value_type), src_offset,
      src_stride, dst_offset, dst_stride, scale, bias);
#else
  TI_ERROR("CUDA strided transform requires TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_device_transform_affine_packed_strided_ndarray(
    Ndarray *src,
    Ndarray *dst,
    int value_type,
    int lane_count,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride,
    double scale,
    double bias) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA packed strided transform is only available on CUDA.");
  check_transform_packed_strided_request("CUDA", src, dst, value_type,
                                         lane_count, src_offset, src_stride,
                                         dst_offset, dst_stride);
  TI_ERROR_IF(src->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA packed strided transform currently supports at most "
              "INT_MAX items.");
#ifdef TI_WITH_CUDA
  TI_ERROR_IF(!cuda::driver_transform_available(),
              "CUDA packed strided transform requires CUDA driver API "
              "support.");
  auto *src_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(src));
  auto *dst_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(dst));
  return cuda::driver_transform_affine_packed_strided(
      src_ptr, dst_ptr, static_cast<int>(src->get_nelement()), lane_count,
      static_cast<cuda::CudaTransformValueType>(value_type), src_offset,
      src_stride, dst_offset, dst_stride, scale, bias);
#else
  TI_ERROR("CUDA packed strided transform requires TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_device_transform_affine_dense_field(SNode *src,
                                                              SNode *dst,
                                                              int value_type,
                                                              std::size_t n,
                                                              double scale,
                                                              double bias) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA dense field transform is only available on CUDA.");
  TI_ERROR_IF(!src || !dst,
              "CUDA dense field transform received a null field.");
  TI_ERROR_IF(value_type < 0 || value_type > 5,
              "CUDA dense field transform received an unsupported value type.");
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA dense field transform currently supports at most INT_MAX "
              "items.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CUDA dense field transform received an unsupported value type.");
  const std::size_t src_stride = get_dense_field_stride(src, value_size);
  const std::size_t dst_stride = get_dense_field_stride(dst, value_size);
  TI_ERROR_IF(src_stride < value_size || dst_stride < value_size,
              "CUDA dense field transform received an invalid field stride.");
  if (n == 0) {
    return 0;
  }
#ifdef TI_WITH_CUDA
  auto raw_ptr = [this](DevicePtr ptr, const char *op_name) {
    TI_ERROR_IF(!ptr.device, "{} received a null dense field device.",
                op_name);
    DeviceAllocation alloc{ptr.device, ptr.alloc_id};
    auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
    TI_ERROR_IF(!base, "{} received a null dense field data pointer.",
                op_name);
    return static_cast<void *>(reinterpret_cast<uint8_t *>(base) + ptr.offset);
  };
  void *src_ptr =
      raw_ptr(get_dense_field_device_ptr(src), "CUDA dense field transform");
  void *dst_ptr =
      raw_ptr(get_dense_field_device_ptr(dst), "CUDA dense field transform");
  const auto cuda_value_type =
      static_cast<cuda::CudaTransformValueType>(value_type);
  if (src_stride == value_size && dst_stride == value_size) {
    if (cuda::driver_transform_available()) {
      return cuda::driver_transform_affine(
          src_ptr, dst_ptr, static_cast<int>(n), cuda_value_type, scale, bias);
    }
  }
  TI_ERROR_IF(!cuda::driver_transform_available(),
              "CUDA dense field strided transform requires CUDA driver API "
              "support.");
  return cuda::driver_transform_affine_strided(
      src_ptr, dst_ptr, static_cast<int>(n), cuda_value_type, 0, src_stride, 0,
      dst_stride, scale, bias);
#else
  TI_ERROR("CUDA dense field transform requires TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_device_zero_dense_field(SNode *dst,
                                                  int value_type,
                                                  std::size_t n) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA dense field zero-fill is only available on CUDA.");
  TI_ERROR_IF(!dst, "CUDA dense field zero-fill received a null field.");
  TI_ERROR_IF(value_type < 0 || value_type > 5,
              "CUDA dense field zero-fill received an unsupported value type.");
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA dense field zero-fill currently supports at most INT_MAX "
              "items.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CUDA dense field zero-fill received an unsupported value type.");
  if (n == 0) {
    return 0;
  }
  const std::size_t dst_stride = get_dense_field_stride(dst, value_size);
  TI_ERROR_IF(dst_stride < value_size,
              "CUDA dense field zero-fill received an invalid field stride.");
#ifdef TI_WITH_CUDA
  DevicePtr dst_device_ptr = get_dense_field_device_ptr(dst);
  TI_ERROR_IF(!dst_device_ptr.device,
              "CUDA dense field zero-fill received a null dense field device.");
  DeviceAllocation alloc{dst_device_ptr.device, dst_device_ptr.alloc_id};
  auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
  TI_ERROR_IF(!base,
              "CUDA dense field zero-fill received a null dense field data "
              "pointer.");
  void *dst_raw =
      static_cast<void *>(reinterpret_cast<uint8_t *>(base) + dst_device_ptr.offset);
  if (dst_stride == value_size) {
    CUDADriver::get_instance().memset(dst_raw, 0, n * value_size);
    return 0;
  }
  TI_ERROR_IF(!cuda::driver_transform_available(),
              "CUDA strided dense field zero-fill requires CUDA driver API "
              "support.");
  return cuda::driver_transform_affine_strided(
      dst_raw, dst_raw, static_cast<int>(n),
      static_cast<cuda::CudaTransformValueType>(value_type), 0, dst_stride, 0,
      dst_stride, 0.0, 0.0);
#else
  TI_ERROR("CUDA dense field zero-fill requires TI_WITH_CUDA=ON.");
#endif
}

bool Program::cuda_device_add_merge_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda &&
         cuda::driver_hierarchical_available();
#else
  return false;
#endif
}

std::size_t Program::cuda_device_add_merge_ndarray(Ndarray *src,
                                                   Ndarray *dst,
                                                   int value_type) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA add-merge is only available on CUDA.");
  TI_ERROR_IF(!src || !dst, "CUDA add-merge received a null ndarray.");
  TI_ERROR_IF(src->get_nelement() != dst->get_nelement(),
              "CUDA add-merge source and destination sizes differ.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CUDA add-merge received an unsupported value type.");
  const std::size_t src_element_size = src->get_element_size();
  const std::size_t dst_element_size = dst->get_element_size();
  TI_ERROR_IF(src_element_size != dst_element_size ||
                  src_element_size < value_size ||
                  src_element_size % value_size != 0,
              "CUDA add-merge payload does not match value type.");
  const std::size_t lanes = src_element_size / value_size;
  if (src->get_nelement() == 0 || lanes == 0) {
    return 0;
  }
  TI_ERROR_IF(
      src->get_nelement() >
          static_cast<std::size_t>(std::numeric_limits<int>::max()) / lanes,
      "CUDA add-merge currently supports at most INT_MAX items.");
  const std::size_t scalar_items = src->get_nelement() * lanes;
#ifdef TI_WITH_CUDA
  void *stream = nullptr;
  return cuda::driver_add_strided(
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(src)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(dst)),
      static_cast<int>(scalar_items),
      static_cast<cuda::CudaTransformValueType>(value_type), 0, value_size, 0,
      value_size, stream);
#else
  TI_ERROR("CUDA add-merge requires TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_device_add_scaled_ndarray(Ndarray *src,
                                                    Ndarray *dst,
                                                    int value_type,
                                                    double scale) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA scaled-add is only available on CUDA.");
  TI_ERROR_IF(!src || !dst, "CUDA scaled-add received a null ndarray.");
  TI_ERROR_IF(src->shape.size() != 1 || dst->shape.size() != 1 ||
                  src->get_nelement() != dst->get_nelement(),
              "CUDA scaled-add expects matching 1D ndarrays.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CUDA scaled-add received an unsupported value type.");
  const std::size_t src_element_size = src->get_element_size();
  const std::size_t dst_element_size = dst->get_element_size();
  TI_ERROR_IF(src_element_size != dst_element_size ||
                  src_element_size < value_size ||
                  src_element_size % value_size != 0,
              "CUDA scaled-add payload does not match value type.");
  TI_ERROR_IF(value_type != 1 && value_type != 5,
              "CUDA scaled-add is supported only for f32/f64 gradients.");
  const std::size_t lanes = src_element_size / value_size;
  if (src->get_nelement() == 0 || lanes == 0) {
    return 0;
  }
  TI_ERROR_IF(
      src->get_nelement() >
          static_cast<std::size_t>(std::numeric_limits<int>::max()) / lanes,
      "CUDA scaled-add currently supports at most INT_MAX items.");
  const std::size_t scalar_items = src->get_nelement() * lanes;
#ifdef TI_WITH_CUDA
  void *stream = nullptr;
  return cuda::driver_add_scaled_strided(
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(src)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(dst)),
      static_cast<int>(scalar_items),
      static_cast<cuda::CudaTransformValueType>(value_type), 0, value_size, 0,
      value_size, scale, stream);
#else
  TI_ERROR("CUDA scaled-add requires TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_device_add_scalar_ndarray_to_ndarray(Ndarray *src,
                                                               Ndarray *dst,
                                                               int value_type,
                                                               double scale) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA scalar-to-ndarray add is only available on CUDA.");
  TI_ERROR_IF(!src || !dst,
              "CUDA scalar-to-ndarray add received a null ndarray.");
  TI_ERROR_IF(src->shape.size() != 1 || dst->shape.size() != 1 ||
                  src->get_nelement() < 1,
              "CUDA scalar-to-ndarray add expects 1D source and destination "
              "ndarrays with at least one source element.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CUDA scalar-to-ndarray add received an unsupported value type.");
  TI_ERROR_IF(src->get_element_size() != value_size ||
                  dst->get_element_size() != value_size,
              "CUDA scalar-to-ndarray add dtype does not match value type.");
  TI_ERROR_IF(value_type != 1 && value_type != 5,
              "CUDA scalar-to-ndarray add is supported only for f32/f64 "
              "gradients.");
  if (dst->get_nelement() == 0) {
    return 0;
  }
  TI_ERROR_IF(dst->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA scalar-to-ndarray add currently supports at most INT_MAX "
              "items.");
#ifdef TI_WITH_CUDA
  void *stream = nullptr;
  return cuda::driver_add_scaled_strided(
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(src)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(dst)),
      static_cast<int>(dst->get_nelement()),
      static_cast<cuda::CudaTransformValueType>(value_type), 0, 0, 0,
      value_size, scale, stream);
#else
  TI_ERROR("CUDA scalar-to-ndarray add requires TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_device_add_merge_strided_ndarray(
    Ndarray *src,
    Ndarray *dst,
    int value_type,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA strided add-merge is only available on CUDA.");
  check_add_merge_strided_request("CUDA", src, dst, value_type, src_offset,
                                  src_stride, dst_offset, dst_stride);
  const std::size_t n = src->get_nelement();
  if (n == 0) {
    return 0;
  }
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA strided add-merge currently supports at most INT_MAX "
              "items.");
#ifdef TI_WITH_CUDA
  void *stream = nullptr;
  return cuda::driver_add_strided(
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(src)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(dst)),
      static_cast<int>(n),
      static_cast<cuda::CudaTransformValueType>(value_type), src_offset,
      src_stride, dst_offset, dst_stride, stream);
#else
  TI_ERROR("CUDA strided add-merge requires TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_device_add_merge_dense_field(Ndarray *src,
                                                       SNode *dst,
                                                       int value_type,
                                                       std::size_t n) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA dense field add-merge is only available on CUDA.");
  TI_ERROR_IF(!src || !dst,
              "CUDA dense field add-merge received a null input.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CUDA dense field add-merge received an unsupported value type.");
  TI_ERROR_IF(src->shape.size() != 1 || src->get_nelement() != n ||
                  src->get_element_size() != value_size,
              "CUDA dense field add-merge source shape or dtype mismatch.");
  if (n == 0) {
    return 0;
  }
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA dense field add-merge currently supports at most INT_MAX "
              "items.");
  const std::size_t dst_stride = get_dense_field_stride(dst, value_size);
  TI_ERROR_IF(dst_stride < value_size,
              "CUDA dense field add-merge received an invalid field stride.");
#ifdef TI_WITH_CUDA
  auto raw_ptr = [this](DevicePtr ptr, const char *op_name) {
    TI_ERROR_IF(!ptr.device, "{} received a null dense field device.", op_name);
    DeviceAllocation alloc{ptr.device, ptr.alloc_id};
    auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
    TI_ERROR_IF(!base, "{} received a null dense field data pointer.", op_name);
    return static_cast<void *>(reinterpret_cast<uint8_t *>(base) + ptr.offset);
  };
  void *src_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(src));
  DevicePtr dst_ptr = get_dense_field_device_ptr(dst);
  void *dst_raw = raw_ptr(dst_ptr, "CUDA dense field add-merge");
  void *stream = nullptr;
  return cuda::driver_add_strided(
      src_ptr, dst_raw, static_cast<int>(n),
      static_cast<cuda::CudaTransformValueType>(value_type), 0, value_size, 0,
      dst_stride, stream);
#else
  TI_ERROR("CUDA dense field add-merge requires TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_device_add_scaled_dense_field(SNode *src,
                                                        SNode *dst,
                                                        int value_type,
                                                        std::size_t n,
                                                        double scale) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA dense field scaled-add is only available on CUDA.");
  TI_ERROR_IF(!src || !dst,
              "CUDA dense field scaled-add received a null field.");
  TI_ERROR_IF(value_type != 1 && value_type != 5,
              "CUDA dense field scaled-add is supported only for f32/f64 "
              "gradients.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CUDA dense field scaled-add received an unsupported value "
              "type.");
  if (n == 0) {
    return 0;
  }
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA dense field scaled-add currently supports at most INT_MAX "
              "items.");
  const std::size_t src_stride = get_dense_field_stride(src, value_size);
  const std::size_t dst_stride = get_dense_field_stride(dst, value_size);
  TI_ERROR_IF(src_stride < value_size || dst_stride < value_size,
              "CUDA dense field scaled-add received an invalid field stride.");
#ifdef TI_WITH_CUDA
  auto raw_ptr = [this](DevicePtr ptr, const char *op_name) {
    TI_ERROR_IF(!ptr.device, "{} received a null dense field device.", op_name);
    DeviceAllocation alloc{ptr.device, ptr.alloc_id};
    auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
    TI_ERROR_IF(!base, "{} received a null dense field data pointer.", op_name);
    return static_cast<void *>(reinterpret_cast<uint8_t *>(base) + ptr.offset);
  };
  void *src_raw =
      raw_ptr(get_dense_field_device_ptr(src), "CUDA dense field scaled-add");
  void *dst_raw =
      raw_ptr(get_dense_field_device_ptr(dst), "CUDA dense field scaled-add");
  void *stream = nullptr;
  return cuda::driver_add_scaled_strided(
      src_raw, dst_raw, static_cast<int>(n),
      static_cast<cuda::CudaTransformValueType>(value_type), 0, src_stride, 0,
      dst_stride, scale, stream);
#else
  TI_ERROR("CUDA dense field scaled-add requires TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_device_add_scalar_field_to_dense_field(
    SNode *src,
    SNode *dst,
    int value_type,
    std::size_t n) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA scalar-to-dense add is only available on CUDA.");
  TI_ERROR_IF(!src || !dst, "CUDA scalar-to-dense add received a null field.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CUDA scalar-to-dense add received an unsupported value type.");
  if (n == 0) {
    return 0;
  }
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA scalar-to-dense add currently supports at most INT_MAX "
              "items.");
  const std::size_t dst_stride = get_dense_field_stride(dst, value_size);
  TI_ERROR_IF(dst_stride < value_size,
              "CUDA scalar-to-dense add received an invalid field stride.");
#ifdef TI_WITH_CUDA
  auto raw_ptr = [this](DevicePtr ptr, const char *op_name) {
    TI_ERROR_IF(!ptr.device, "{} received a null dense field device.", op_name);
    DeviceAllocation alloc{ptr.device, ptr.alloc_id};
    auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
    TI_ERROR_IF(!base, "{} received a null dense field data pointer.", op_name);
    return static_cast<void *>(reinterpret_cast<uint8_t *>(base) + ptr.offset);
  };
  void *src_raw =
      raw_ptr(get_dense_field_device_ptr(src), "CUDA scalar-to-dense add");
  void *dst_raw =
      raw_ptr(get_dense_field_device_ptr(dst), "CUDA scalar-to-dense add");
  void *stream = nullptr;
  return cuda::driver_add_strided(
      src_raw, dst_raw, static_cast<int>(n),
      static_cast<cuda::CudaTransformValueType>(value_type), 0, 0, 0,
      dst_stride, stream);
#else
  TI_ERROR("CUDA scalar-to-dense add requires TI_WITH_CUDA=ON.");
#endif
}

bool Program::cuda_device_indexed_copy_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda &&
         cuda::driver_indexed_copy_available();
#else
  return false;
#endif
}

bool Program::cuda_device_indexed_copy_payload_available(
    std::size_t item_bytes) const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda && item_bytes != 0 &&
         item_bytes % sizeof(uint32_t) == 0 &&
         cuda::driver_indexed_copy_available();
#else
  return false;
#endif
}

std::size_t Program::cuda_device_gather_ndarray(Ndarray *src,
                                                Ndarray *indices,
                                                Ndarray *dst) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA device gather is only available on CUDA.");
  TI_ERROR_IF(!src || !indices || !dst,
              "CUDA device gather received a null ndarray.");
  TI_ERROR_IF(src->shape.size() != 1 || indices->shape.size() != 1 ||
                  dst->shape.size() != 1,
              "CUDA device gather currently expects 1D ndarrays.");
  TI_ERROR_IF(indices->get_nelement() != dst->get_nelement(),
              "CUDA device gather expects indices and destination sizes to "
              "match.");
  TI_ERROR_IF(src->get_element_size() != dst->get_element_size(),
              "CUDA device gather source and destination dtypes differ.");
  const std::size_t item_bytes = src->get_element_size();
  TI_ERROR_IF(item_bytes == 0 || item_bytes % sizeof(uint32_t) != 0 ||
                  indices->get_element_size() != sizeof(int32_t),
              "CUDA device gather currently expects 4-byte aligned values and "
              "i32 indices.");
  TI_ERROR_IF(indices->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA device gather currently supports at most INT_MAX items.");
  TI_ERROR_IF(src->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA device gather currently supports source sizes up to "
              "INT_MAX items.");
#ifdef TI_WITH_CUDA
  auto *src_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(src));
  auto *indices_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(indices));
  auto *dst_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(dst));
  const int item_words = static_cast<int>(item_bytes / sizeof(uint32_t));
  TI_ERROR_IF(indices->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max() /
                                           item_words),
              "CUDA device gather word count exceeds INT_MAX.");
  return cuda::driver_indexed_copy(
      src_ptr, indices_ptr, dst_ptr,
      static_cast<int>(indices->get_nelement()),
      static_cast<int>(src->get_nelement()), item_words,
      cuda::CudaIndexedCopyOp::gather, nullptr);
#else
  TI_ERROR("CUDA device gather requires TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_device_gather_strided_ndarray(
    Ndarray *src,
    Ndarray *indices,
    Ndarray *dst,
    std::size_t item_bytes,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA device strided gather is only available on CUDA.");
  check_indexed_copy_strided_request("CUDA", src, indices, dst, item_bytes,
                                     src_offset, src_stride, dst_offset,
                                     dst_stride, false);
  TI_ERROR_IF(indices->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA device strided gather currently supports at most INT_MAX "
              "items.");
  TI_ERROR_IF(src->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA device strided gather currently supports source sizes up "
              "to INT_MAX items.");
#ifdef TI_WITH_CUDA
  TI_ERROR_IF(!cuda::driver_indexed_copy_available(),
              "CUDA Driver indexed-copy provider is unavailable.");
  auto *src_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(src));
  auto *indices_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(indices));
  auto *dst_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(dst));
  const int item_words = static_cast<int>(item_bytes / sizeof(uint32_t));
  TI_ERROR_IF(indices->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max() /
                                           item_words),
              "CUDA device strided gather word count exceeds INT_MAX.");
  return cuda::driver_indexed_copy_strided(
      src_ptr, indices_ptr, dst_ptr, static_cast<int>(indices->get_nelement()),
      static_cast<int>(src->get_nelement()), item_words,
      src_offset / sizeof(uint32_t), src_stride / sizeof(uint32_t),
      dst_offset / sizeof(uint32_t), dst_stride / sizeof(uint32_t),
      cuda::CudaIndexedCopyOp::gather, nullptr);
#else
  TI_ERROR("CUDA device strided gather requires TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_device_gather_dense_field(SNode *src,
                                                    Ndarray *indices,
                                                    SNode *dst,
                                                    int value_type,
                                                    std::size_t src_n,
                                                    std::size_t dst_n) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA device dense field gather is only available on CUDA.");
  check_indexed_copy_dense_field_request(this, "CUDA", src, indices, dst,
                                         value_type, src_n, dst_n, false);
  const std::size_t n = indices->get_nelement();
  if (n == 0) {
    return 0;
  }
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA device dense field gather currently supports at most "
              "INT_MAX items.");
  TI_ERROR_IF(src_n >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA device dense field gather supports source sizes up to "
              "INT_MAX items.");
  const std::size_t item_bytes = primitive_value_type_size(value_type);
  const std::size_t src_stride = get_dense_field_stride(src, item_bytes);
  const std::size_t dst_stride = get_dense_field_stride(dst, item_bytes);
#ifdef TI_WITH_CUDA
  auto raw_ptr = [this](DevicePtr ptr, const char *op_name) {
    TI_ERROR_IF(!ptr.device, "{} received a null dense field device.",
                op_name);
    DeviceAllocation alloc{ptr.device, ptr.alloc_id};
    auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
    TI_ERROR_IF(!base, "{} received a null dense field data pointer.",
                op_name);
    return static_cast<void *>(reinterpret_cast<uint8_t *>(base) + ptr.offset);
  };
  void *src_ptr =
      raw_ptr(get_dense_field_device_ptr(src), "CUDA dense field gather");
  auto *indices_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(indices));
  void *dst_ptr =
      raw_ptr(get_dense_field_device_ptr(dst), "CUDA dense field gather");
  const int item_words = static_cast<int>(item_bytes / sizeof(uint32_t));
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max() /
                                           item_words),
              "CUDA device dense field gather word count exceeds INT_MAX.");
  if (src_stride == item_bytes && dst_stride == item_bytes) {
    return cuda::driver_indexed_copy(
        src_ptr, indices_ptr, dst_ptr, static_cast<int>(n),
        static_cast<int>(src_n), item_words,
        cuda::CudaIndexedCopyOp::gather, nullptr);
  }
  TI_ERROR_IF(!cuda::driver_indexed_copy_available(),
              "CUDA Driver indexed-copy provider is unavailable.");
  return cuda::driver_indexed_copy_strided(
      src_ptr, indices_ptr, dst_ptr, static_cast<int>(n),
      static_cast<int>(src_n), item_words, 0, src_stride / sizeof(uint32_t), 0,
      dst_stride / sizeof(uint32_t), cuda::CudaIndexedCopyOp::gather, nullptr);
#else
  TI_ERROR("CUDA device dense field gather requires TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_device_gather_dense_field_packed(SNode *src,
                                                           Ndarray *indices,
                                                           SNode *dst,
                                                           int value_type,
                                                           std::size_t src_n,
                                                           std::size_t dst_n,
                                                           int lane_count) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA device packed dense field gather is only available on "
              "CUDA.");
  TI_ERROR_IF(!indices || indices->shape.size() != 1 ||
                  indices->get_nelement() != dst_n ||
                  indices->get_element_size() != sizeof(int32_t),
              "CUDA device packed dense field gather expects 1D i32 indices "
              "matching destination size.");
  const std::size_t n = indices->get_nelement();
  if (n == 0) {
    return 0;
  }
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
                  src_n >
                      static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA device packed dense field gather currently supports up to "
              "INT_MAX items.");
  const std::size_t item_bytes =
      dense_field_packed_bytes(value_type, 1, lane_count,
                               "CUDA packed dense field gather");
  check_dense_field_packed_stride(this, src, value_type, lane_count,
                                  "CUDA packed dense field gather");
  check_dense_field_packed_stride(this, dst, value_type, lane_count,
                                  "CUDA packed dense field gather");
#ifdef TI_WITH_CUDA
  auto raw_ptr = [this](DevicePtr ptr, const char *op_name) {
    TI_ERROR_IF(!ptr.device, "{} received a null dense field device.",
                op_name);
    DeviceAllocation alloc{ptr.device, ptr.alloc_id};
    auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
    TI_ERROR_IF(!base, "{} received a null dense field data pointer.",
                op_name);
    return static_cast<void *>(reinterpret_cast<uint8_t *>(base) + ptr.offset);
  };
  void *src_ptr = raw_ptr(get_dense_field_device_ptr(src),
                          "CUDA packed dense field gather");
  auto *indices_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(indices));
  void *dst_ptr = raw_ptr(get_dense_field_device_ptr(dst),
                          "CUDA packed dense field gather");
  const int item_words = static_cast<int>(item_bytes / sizeof(uint32_t));
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max() /
                                           item_words),
              "CUDA device packed dense field gather word count exceeds "
              "INT_MAX.");
  TI_ERROR_IF(!cuda::driver_indexed_copy_available(),
              "CUDA Driver indexed-copy provider is unavailable.");
  return cuda::driver_indexed_copy(
      src_ptr, indices_ptr, dst_ptr, static_cast<int>(n),
      static_cast<int>(src_n), item_words, cuda::CudaIndexedCopyOp::gather,
      nullptr);
#else
  TI_ERROR("CUDA device packed dense field gather requires TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_device_gather_dense_field_packed_indices_field(
    SNode *src,
    SNode *indices,
    SNode *dst,
    int value_type,
    std::size_t src_n,
    std::size_t indices_n,
    std::size_t dst_n,
    int lane_count) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA device packed dense field gather is only available on "
              "CUDA.");
  TI_ERROR_IF(indices_n != dst_n,
              "CUDA device packed dense field gather expects field indices "
              "matching destination size.");
  const std::size_t n = indices_n;
  if (n == 0) {
    return 0;
  }
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
                  src_n >
                      static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA device packed dense field gather currently supports up to "
              "INT_MAX items.");
  const std::size_t item_bytes =
      dense_field_packed_bytes(value_type, 1, lane_count,
                               "CUDA packed dense field gather");
  check_dense_field_packed_stride(this, src, value_type, lane_count,
                                  "CUDA packed dense field gather");
  check_dense_field_packed_stride(this, dst, value_type, lane_count,
                                  "CUDA packed dense field gather");
  TI_ERROR_IF(get_dense_field_stride(indices, sizeof(int32_t)) !=
                  sizeof(int32_t),
              "CUDA packed dense field gather requires contiguous i32 field "
              "indices.");
#ifdef TI_WITH_CUDA
  auto raw_ptr = [this](DevicePtr ptr, const char *op_name) {
    TI_ERROR_IF(!ptr.device, "{} received a null dense field device.",
                op_name);
    DeviceAllocation alloc{ptr.device, ptr.alloc_id};
    auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
    TI_ERROR_IF(!base, "{} received a null dense field data pointer.",
                op_name);
    return static_cast<void *>(reinterpret_cast<uint8_t *>(base) + ptr.offset);
  };
  void *src_ptr = raw_ptr(get_dense_field_device_ptr(src),
                          "CUDA packed dense field gather");
  void *indices_ptr = raw_ptr(get_dense_field_device_ptr(indices),
                              "CUDA packed dense field gather");
  void *dst_ptr = raw_ptr(get_dense_field_device_ptr(dst),
                          "CUDA packed dense field gather");
  const int item_words = static_cast<int>(item_bytes / sizeof(uint32_t));
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max() /
                                           item_words),
              "CUDA device packed dense field gather word count exceeds "
              "INT_MAX.");
  TI_ERROR_IF(!cuda::driver_indexed_copy_available(),
              "CUDA Driver indexed-copy provider is unavailable.");
  return cuda::driver_indexed_copy(
      src_ptr, indices_ptr, dst_ptr, static_cast<int>(n),
      static_cast<int>(src_n), item_words, cuda::CudaIndexedCopyOp::gather,
      nullptr);
#else
  TI_ERROR("CUDA device packed dense field gather requires TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_device_gather_dense_field_indices_field(
    SNode *src,
    SNode *indices,
    SNode *dst,
    int value_type,
    std::size_t src_n,
    std::size_t indices_n,
    std::size_t dst_n) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA device dense field gather is only available on CUDA.");
  check_indexed_copy_dense_field_indices_field_request(
      this, "CUDA", src, indices, dst, value_type, src_n, indices_n, dst_n,
      false);
  const std::size_t n = indices_n;
  if (n == 0) {
    return 0;
  }
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA device dense field gather currently supports at most "
              "INT_MAX items.");
  TI_ERROR_IF(src_n >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA device dense field gather supports source sizes up to "
              "INT_MAX items.");
  const std::size_t item_bytes = primitive_value_type_size(value_type);
  const std::size_t src_stride = get_dense_field_stride(src, item_bytes);
  const std::size_t dst_stride = get_dense_field_stride(dst, item_bytes);
#ifdef TI_WITH_CUDA
  auto raw_ptr = [this](DevicePtr ptr, const char *op_name) {
    TI_ERROR_IF(!ptr.device, "{} received a null dense field device.",
                op_name);
    DeviceAllocation alloc{ptr.device, ptr.alloc_id};
    auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
    TI_ERROR_IF(!base, "{} received a null dense field data pointer.",
                op_name);
    return static_cast<void *>(reinterpret_cast<uint8_t *>(base) + ptr.offset);
  };
  void *src_ptr =
      raw_ptr(get_dense_field_device_ptr(src), "CUDA dense field gather");
  void *indices_ptr =
      raw_ptr(get_dense_field_device_ptr(indices), "CUDA dense field gather");
  void *dst_ptr =
      raw_ptr(get_dense_field_device_ptr(dst), "CUDA dense field gather");
  const int item_words = static_cast<int>(item_bytes / sizeof(uint32_t));
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max() /
                                           item_words),
              "CUDA device dense field gather word count exceeds INT_MAX.");
  if (src_stride == item_bytes && dst_stride == item_bytes) {
    return cuda::driver_indexed_copy(
        src_ptr, indices_ptr, dst_ptr, static_cast<int>(n),
        static_cast<int>(src_n), item_words,
        cuda::CudaIndexedCopyOp::gather, nullptr);
  }
  TI_ERROR_IF(!cuda::driver_indexed_copy_available(),
              "CUDA Driver indexed-copy provider is unavailable.");
  return cuda::driver_indexed_copy_strided(
      src_ptr, indices_ptr, dst_ptr, static_cast<int>(n),
      static_cast<int>(src_n), item_words, 0, src_stride / sizeof(uint32_t), 0,
      dst_stride / sizeof(uint32_t), cuda::CudaIndexedCopyOp::gather, nullptr);
#else
  TI_ERROR("CUDA device dense field gather requires TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_device_gather_add_ndarray(Ndarray *src,
                                                    Ndarray *indices,
                                                    Ndarray *dst,
                                                    int value_type) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA gather-add is only available on CUDA.");
  TI_ERROR_IF(!src || !indices || !dst,
              "CUDA gather-add received a null ndarray.");
  TI_ERROR_IF(src->shape.size() != 1 || indices->shape.size() != 1 ||
                  dst->shape.size() != 1 ||
                  indices->get_nelement() != dst->get_nelement(),
              "CUDA gather-add expects 1D src/indices/dst and destination "
              "size matching indices size.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CUDA gather-add received an unsupported value type.");
  TI_ERROR_IF(src->get_element_size() != value_size ||
                  dst->get_element_size() != value_size ||
                  indices->get_element_size() != sizeof(int32_t),
              "CUDA gather-add dtype does not match value type or i32 index "
              "size.");
  TI_ERROR_IF(value_type != 1 && value_type != 5,
              "CUDA gather-add is supported only for f32/f64 gradients.");
  if (indices->get_nelement() == 0 || src->get_nelement() == 0) {
    return 0;
  }
  TI_ERROR_IF(indices->get_nelement() > static_cast<std::size_t>(
                                            std::numeric_limits<int>::max()) ||
                  src->get_nelement() >
                      static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA gather-add currently supports at most INT_MAX items.");
#ifdef TI_WITH_CUDA
  void *stream = nullptr;
  return cuda::driver_gather_add_strided(
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(src)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(indices)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(dst)),
      static_cast<int>(indices->get_nelement()),
      static_cast<int>(src->get_nelement()),
      static_cast<cuda::CudaTransformValueType>(value_type), 0, value_size, 0,
      sizeof(std::int32_t), 0, value_size, stream);
#else
  TI_ERROR("CUDA gather-add requires TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_device_gather_add_dense_field(SNode *src,
                                                        Ndarray *indices,
                                                        SNode *dst,
                                                        int value_type,
                                                        std::size_t src_n,
                                                        std::size_t dst_n) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA dense field gather-add is only available on CUDA.");
  check_indexed_copy_dense_field_request(this, "CUDA", src, indices, dst,
                                         value_type, src_n, dst_n, false);
  TI_ERROR_IF(indices->get_nelement() != dst_n,
              "CUDA dense field gather-add expects destination size to match "
              "indices size.");
  TI_ERROR_IF(value_type != 1 && value_type != 5,
              "CUDA dense field gather-add is supported only for f32/f64 "
              "gradients.");
  const std::size_t n = indices->get_nelement();
  if (n == 0 || src_n == 0) {
    return 0;
  }
  TI_ERROR_IF(
      n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
          src_n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
      "CUDA dense field gather-add currently supports at most "
      "INT_MAX items.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  const std::size_t src_stride = get_dense_field_stride(src, value_size);
  const std::size_t dst_stride = get_dense_field_stride(dst, value_size);
#ifdef TI_WITH_CUDA
  auto raw_ptr = [this](DevicePtr ptr, const char *op_name) {
    TI_ERROR_IF(!ptr.device, "{} received a null dense field device.", op_name);
    DeviceAllocation alloc{ptr.device, ptr.alloc_id};
    auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
    TI_ERROR_IF(!base, "{} received a null dense field data pointer.", op_name);
    return static_cast<void *>(reinterpret_cast<uint8_t *>(base) + ptr.offset);
  };
  void *src_ptr =
      raw_ptr(get_dense_field_device_ptr(src), "CUDA dense field gather-add");
  void *indices_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(indices));
  void *dst_ptr =
      raw_ptr(get_dense_field_device_ptr(dst), "CUDA dense field gather-add");
  void *stream = nullptr;
  return cuda::driver_gather_add_strided(
      src_ptr, indices_ptr, dst_ptr, static_cast<int>(n),
      static_cast<int>(src_n),
      static_cast<cuda::CudaTransformValueType>(value_type), 0, src_stride, 0,
      sizeof(std::int32_t), 0, dst_stride, stream);
#else
  TI_ERROR("CUDA dense field gather-add requires TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_device_gather_add_dense_field_indices_field(
    SNode *src,
    SNode *indices,
    SNode *dst,
    int value_type,
    std::size_t src_n,
    std::size_t indices_n,
    std::size_t dst_n) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA dense field gather-add is only available on CUDA.");
  check_indexed_copy_dense_field_indices_field_request(
      this, "CUDA", src, indices, dst, value_type, src_n, indices_n, dst_n,
      false);
  TI_ERROR_IF(indices_n != dst_n,
              "CUDA dense field gather-add expects destination size to match "
              "indices size.");
  TI_ERROR_IF(value_type != 1 && value_type != 5,
              "CUDA dense field gather-add is supported only for f32/f64 "
              "gradients.");
  if (indices_n == 0 || src_n == 0) {
    return 0;
  }
  TI_ERROR_IF(
      indices_n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
          src_n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
      "CUDA dense field gather-add currently supports at most "
      "INT_MAX items.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  const std::size_t src_stride = get_dense_field_stride(src, value_size);
  const std::size_t indices_stride =
      get_dense_field_stride(indices, sizeof(std::int32_t));
  const std::size_t dst_stride = get_dense_field_stride(dst, value_size);
#ifdef TI_WITH_CUDA
  auto raw_ptr = [this](DevicePtr ptr, const char *op_name) {
    TI_ERROR_IF(!ptr.device, "{} received a null dense field device.", op_name);
    DeviceAllocation alloc{ptr.device, ptr.alloc_id};
    auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
    TI_ERROR_IF(!base, "{} received a null dense field data pointer.", op_name);
    return static_cast<void *>(reinterpret_cast<uint8_t *>(base) + ptr.offset);
  };
  void *src_ptr =
      raw_ptr(get_dense_field_device_ptr(src), "CUDA dense field gather-add");
  void *indices_ptr = raw_ptr(get_dense_field_device_ptr(indices),
                              "CUDA dense field gather-add");
  void *dst_ptr =
      raw_ptr(get_dense_field_device_ptr(dst), "CUDA dense field gather-add");
  void *stream = nullptr;
  return cuda::driver_gather_add_strided(
      src_ptr, indices_ptr, dst_ptr, static_cast<int>(indices_n),
      static_cast<int>(src_n),
      static_cast<cuda::CudaTransformValueType>(value_type), 0, src_stride, 0,
      indices_stride, 0, dst_stride, stream);
#else
  TI_ERROR("CUDA dense field gather-add requires TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_device_scatter_ndarray(Ndarray *src,
                                                 Ndarray *indices,
                                                 Ndarray *dst) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA device scatter is only available on CUDA.");
  TI_ERROR_IF(!src || !indices || !dst,
              "CUDA device scatter received a null ndarray.");
  TI_ERROR_IF(src->shape.size() != 1 || indices->shape.size() != 1 ||
                  dst->shape.size() != 1,
              "CUDA device scatter currently expects 1D ndarrays.");
  TI_ERROR_IF(src->get_nelement() != indices->get_nelement(),
              "CUDA device scatter expects source and indices sizes to match.");
  TI_ERROR_IF(src->get_element_size() != dst->get_element_size(),
              "CUDA device scatter source and destination dtypes differ.");
  const std::size_t item_bytes = src->get_element_size();
  TI_ERROR_IF(item_bytes == 0 || item_bytes % sizeof(uint32_t) != 0 ||
                  indices->get_element_size() != sizeof(int32_t),
              "CUDA device scatter currently expects 4-byte aligned values and "
              "i32 indices.");
  TI_ERROR_IF(indices->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA device scatter currently supports at most INT_MAX items.");
  TI_ERROR_IF(dst->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA device scatter currently supports destination sizes up to "
              "INT_MAX items.");
#ifdef TI_WITH_CUDA
  auto *src_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(src));
  auto *indices_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(indices));
  auto *dst_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(dst));
  const int item_words = static_cast<int>(item_bytes / sizeof(uint32_t));
  TI_ERROR_IF(indices->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max() /
                                           item_words),
              "CUDA device scatter word count exceeds INT_MAX.");
  return cuda::driver_indexed_copy(
      src_ptr, indices_ptr, dst_ptr,
      static_cast<int>(indices->get_nelement()),
      static_cast<int>(dst->get_nelement()), item_words,
      cuda::CudaIndexedCopyOp::scatter, nullptr);
#else
  TI_ERROR("CUDA device scatter requires TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_device_scatter_strided_ndarray(
    Ndarray *src,
    Ndarray *indices,
    Ndarray *dst,
    std::size_t item_bytes,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA device strided scatter is only available on CUDA.");
  check_indexed_copy_strided_request("CUDA", src, indices, dst, item_bytes,
                                     src_offset, src_stride, dst_offset,
                                     dst_stride, true);
  TI_ERROR_IF(indices->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA device strided scatter currently supports at most INT_MAX "
              "items.");
  TI_ERROR_IF(dst->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA device strided scatter currently supports destination "
              "sizes up to INT_MAX items.");
#ifdef TI_WITH_CUDA
  TI_ERROR_IF(!cuda::driver_indexed_copy_available(),
              "CUDA Driver indexed-copy provider is unavailable.");
  auto *src_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(src));
  auto *indices_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(indices));
  auto *dst_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(dst));
  const int item_words = static_cast<int>(item_bytes / sizeof(uint32_t));
  TI_ERROR_IF(indices->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max() /
                                           item_words),
              "CUDA device strided scatter word count exceeds INT_MAX.");
  return cuda::driver_indexed_copy_strided(
      src_ptr, indices_ptr, dst_ptr, static_cast<int>(indices->get_nelement()),
      static_cast<int>(dst->get_nelement()), item_words,
      src_offset / sizeof(uint32_t), src_stride / sizeof(uint32_t),
      dst_offset / sizeof(uint32_t), dst_stride / sizeof(uint32_t),
      cuda::CudaIndexedCopyOp::scatter, nullptr);
#else
  TI_ERROR("CUDA device strided scatter requires TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_device_scatter_dense_field(SNode *src,
                                                     Ndarray *indices,
                                                     SNode *dst,
                                                     int value_type,
                                                     std::size_t src_n,
                                                     std::size_t dst_n) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA device dense field scatter is only available on CUDA.");
  check_indexed_copy_dense_field_request(this, "CUDA", src, indices, dst,
                                         value_type, src_n, dst_n, true);
  const std::size_t n = indices->get_nelement();
  if (n == 0) {
    return 0;
  }
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA device dense field scatter currently supports at most "
              "INT_MAX items.");
  TI_ERROR_IF(dst_n >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA device dense field scatter supports destination sizes up "
              "to INT_MAX items.");
  const std::size_t item_bytes = primitive_value_type_size(value_type);
  const std::size_t src_stride = get_dense_field_stride(src, item_bytes);
  const std::size_t dst_stride = get_dense_field_stride(dst, item_bytes);
#ifdef TI_WITH_CUDA
  auto raw_ptr = [this](DevicePtr ptr, const char *op_name) {
    TI_ERROR_IF(!ptr.device, "{} received a null dense field device.",
                op_name);
    DeviceAllocation alloc{ptr.device, ptr.alloc_id};
    auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
    TI_ERROR_IF(!base, "{} received a null dense field data pointer.",
                op_name);
    return static_cast<void *>(reinterpret_cast<uint8_t *>(base) + ptr.offset);
  };
  void *src_ptr =
      raw_ptr(get_dense_field_device_ptr(src), "CUDA dense field scatter");
  auto *indices_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(indices));
  void *dst_ptr =
      raw_ptr(get_dense_field_device_ptr(dst), "CUDA dense field scatter");
  const int item_words = static_cast<int>(item_bytes / sizeof(uint32_t));
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max() /
                                           item_words),
              "CUDA device dense field scatter word count exceeds INT_MAX.");
  if (src_stride == item_bytes && dst_stride == item_bytes) {
    return cuda::driver_indexed_copy(
        src_ptr, indices_ptr, dst_ptr, static_cast<int>(n),
        static_cast<int>(dst_n), item_words,
        cuda::CudaIndexedCopyOp::scatter, nullptr);
  }
  TI_ERROR_IF(!cuda::driver_indexed_copy_available(),
              "CUDA Driver indexed-copy provider is unavailable.");
  return cuda::driver_indexed_copy_strided(
      src_ptr, indices_ptr, dst_ptr, static_cast<int>(n),
      static_cast<int>(dst_n), item_words, 0, src_stride / sizeof(uint32_t), 0,
      dst_stride / sizeof(uint32_t), cuda::CudaIndexedCopyOp::scatter, nullptr);
#else
  TI_ERROR("CUDA device dense field scatter requires TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_device_scatter_dense_field_packed(SNode *src,
                                                            Ndarray *indices,
                                                            SNode *dst,
                                                            int value_type,
                                                            std::size_t src_n,
                                                            std::size_t dst_n,
                                                            int lane_count) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA device packed dense field scatter is only available on "
              "CUDA.");
  TI_ERROR_IF(!indices || indices->shape.size() != 1 ||
                  indices->get_nelement() != src_n ||
                  indices->get_element_size() != sizeof(int32_t),
              "CUDA device packed dense field scatter expects 1D i32 indices "
              "matching source size.");
  const std::size_t n = indices->get_nelement();
  if (n == 0) {
    return 0;
  }
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
                  dst_n >
                      static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA device packed dense field scatter currently supports up "
              "to INT_MAX items.");
  const std::size_t item_bytes =
      dense_field_packed_bytes(value_type, 1, lane_count,
                               "CUDA packed dense field scatter");
  check_dense_field_packed_stride(this, src, value_type, lane_count,
                                  "CUDA packed dense field scatter");
  check_dense_field_packed_stride(this, dst, value_type, lane_count,
                                  "CUDA packed dense field scatter");
#ifdef TI_WITH_CUDA
  auto raw_ptr = [this](DevicePtr ptr, const char *op_name) {
    TI_ERROR_IF(!ptr.device, "{} received a null dense field device.",
                op_name);
    DeviceAllocation alloc{ptr.device, ptr.alloc_id};
    auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
    TI_ERROR_IF(!base, "{} received a null dense field data pointer.",
                op_name);
    return static_cast<void *>(reinterpret_cast<uint8_t *>(base) + ptr.offset);
  };
  void *src_ptr = raw_ptr(get_dense_field_device_ptr(src),
                          "CUDA packed dense field scatter");
  auto *indices_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(indices));
  void *dst_ptr = raw_ptr(get_dense_field_device_ptr(dst),
                          "CUDA packed dense field scatter");
  const int item_words = static_cast<int>(item_bytes / sizeof(uint32_t));
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max() /
                                           item_words),
              "CUDA device packed dense field scatter word count exceeds "
              "INT_MAX.");
  TI_ERROR_IF(!cuda::driver_indexed_copy_available(),
              "CUDA Driver indexed-copy provider is unavailable.");
  return cuda::driver_indexed_copy(
      src_ptr, indices_ptr, dst_ptr, static_cast<int>(n),
      static_cast<int>(dst_n), item_words, cuda::CudaIndexedCopyOp::scatter,
      nullptr);
#else
  TI_ERROR("CUDA device packed dense field scatter requires TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_device_scatter_dense_field_packed_indices_field(
    SNode *src,
    SNode *indices,
    SNode *dst,
    int value_type,
    std::size_t src_n,
    std::size_t indices_n,
    std::size_t dst_n,
    int lane_count) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA device packed dense field scatter is only available on "
              "CUDA.");
  TI_ERROR_IF(src_n != indices_n,
              "CUDA device packed dense field scatter expects field indices "
              "matching source size.");
  const std::size_t n = indices_n;
  if (n == 0) {
    return 0;
  }
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
                  dst_n >
                      static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA device packed dense field scatter currently supports up "
              "to INT_MAX items.");
  const std::size_t item_bytes =
      dense_field_packed_bytes(value_type, 1, lane_count,
                               "CUDA packed dense field scatter");
  check_dense_field_packed_stride(this, src, value_type, lane_count,
                                  "CUDA packed dense field scatter");
  check_dense_field_packed_stride(this, dst, value_type, lane_count,
                                  "CUDA packed dense field scatter");
  TI_ERROR_IF(get_dense_field_stride(indices, sizeof(int32_t)) !=
                  sizeof(int32_t),
              "CUDA packed dense field scatter requires contiguous i32 field "
              "indices.");
#ifdef TI_WITH_CUDA
  auto raw_ptr = [this](DevicePtr ptr, const char *op_name) {
    TI_ERROR_IF(!ptr.device, "{} received a null dense field device.",
                op_name);
    DeviceAllocation alloc{ptr.device, ptr.alloc_id};
    auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
    TI_ERROR_IF(!base, "{} received a null dense field data pointer.",
                op_name);
    return static_cast<void *>(reinterpret_cast<uint8_t *>(base) + ptr.offset);
  };
  void *src_ptr = raw_ptr(get_dense_field_device_ptr(src),
                          "CUDA packed dense field scatter");
  void *indices_ptr = raw_ptr(get_dense_field_device_ptr(indices),
                              "CUDA packed dense field scatter");
  void *dst_ptr = raw_ptr(get_dense_field_device_ptr(dst),
                          "CUDA packed dense field scatter");
  const int item_words = static_cast<int>(item_bytes / sizeof(uint32_t));
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max() /
                                           item_words),
              "CUDA device packed dense field scatter word count exceeds "
              "INT_MAX.");
  TI_ERROR_IF(!cuda::driver_indexed_copy_available(),
              "CUDA Driver indexed-copy provider is unavailable.");
  return cuda::driver_indexed_copy(
      src_ptr, indices_ptr, dst_ptr, static_cast<int>(n),
      static_cast<int>(dst_n), item_words, cuda::CudaIndexedCopyOp::scatter,
      nullptr);
#else
  TI_ERROR("CUDA device packed dense field scatter requires TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_device_scatter_dense_field_indices_field(
    SNode *src,
    SNode *indices,
    SNode *dst,
    int value_type,
    std::size_t src_n,
    std::size_t indices_n,
    std::size_t dst_n) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA device dense field scatter is only available on CUDA.");
  check_indexed_copy_dense_field_indices_field_request(
      this, "CUDA", src, indices, dst, value_type, src_n, indices_n, dst_n,
      true);
  const std::size_t n = indices_n;
  if (n == 0) {
    return 0;
  }
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA device dense field scatter currently supports at most "
              "INT_MAX items.");
  TI_ERROR_IF(dst_n >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA device dense field scatter supports destination sizes up "
              "to INT_MAX items.");
  const std::size_t item_bytes = primitive_value_type_size(value_type);
  const std::size_t src_stride = get_dense_field_stride(src, item_bytes);
  const std::size_t dst_stride = get_dense_field_stride(dst, item_bytes);
#ifdef TI_WITH_CUDA
  auto raw_ptr = [this](DevicePtr ptr, const char *op_name) {
    TI_ERROR_IF(!ptr.device, "{} received a null dense field device.",
                op_name);
    DeviceAllocation alloc{ptr.device, ptr.alloc_id};
    auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
    TI_ERROR_IF(!base, "{} received a null dense field data pointer.",
                op_name);
    return static_cast<void *>(reinterpret_cast<uint8_t *>(base) + ptr.offset);
  };
  void *src_ptr =
      raw_ptr(get_dense_field_device_ptr(src), "CUDA dense field scatter");
  void *indices_ptr =
      raw_ptr(get_dense_field_device_ptr(indices), "CUDA dense field scatter");
  void *dst_ptr =
      raw_ptr(get_dense_field_device_ptr(dst), "CUDA dense field scatter");
  const int item_words = static_cast<int>(item_bytes / sizeof(uint32_t));
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max() /
                                           item_words),
              "CUDA device dense field scatter word count exceeds INT_MAX.");
  if (src_stride == item_bytes && dst_stride == item_bytes) {
    return cuda::driver_indexed_copy(
        src_ptr, indices_ptr, dst_ptr, static_cast<int>(n),
        static_cast<int>(dst_n), item_words,
        cuda::CudaIndexedCopyOp::scatter, nullptr);
  }
  TI_ERROR_IF(!cuda::driver_indexed_copy_available(),
              "CUDA Driver indexed-copy provider is unavailable.");
  return cuda::driver_indexed_copy_strided(
      src_ptr, indices_ptr, dst_ptr, static_cast<int>(n),
      static_cast<int>(dst_n), item_words, 0, src_stride / sizeof(uint32_t), 0,
      dst_stride / sizeof(uint32_t), cuda::CudaIndexedCopyOp::scatter, nullptr);
#else
  TI_ERROR("CUDA device dense field scatter requires TI_WITH_CUDA=ON.");
#endif
}

bool Program::cuda_device_scatter_add_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda &&
         cuda::driver_hierarchical_available();
#else
  return false;
#endif
}

std::size_t Program::cuda_device_scatter_add_ndarray(Ndarray *src,
                                                     Ndarray *indices,
                                                     Ndarray *dst,
                                                     int value_type) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA scatter-add is only available on CUDA.");
  TI_ERROR_IF(!src || !indices || !dst,
              "CUDA toolkit scatter-add received a null ndarray.");
  TI_ERROR_IF(src->shape.size() != 1 || indices->shape.size() != 1 ||
                  dst->shape.size() != 1,
              "CUDA toolkit scatter-add currently expects 1D ndarrays.");
  TI_ERROR_IF(src->get_nelement() != indices->get_nelement(),
              "CUDA toolkit scatter-add expects source and indices sizes to "
              "match.");
  TI_ERROR_IF(src->get_element_size() != dst->get_element_size(),
              "CUDA toolkit scatter-add source and destination dtypes differ.");
  const std::size_t expected_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(expected_size == 0,
              "CUDA toolkit scatter-add received an unsupported value type.");
  TI_ERROR_IF(src->get_element_size() != expected_size ||
                  indices->get_element_size() != sizeof(int32_t),
              "CUDA toolkit scatter-add dtype does not match value type or "
              "indices are not i32.");
  TI_ERROR_IF(indices->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA toolkit scatter-add currently supports at most INT_MAX "
              "source items.");
  TI_ERROR_IF(dst->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA toolkit scatter-add currently supports destination sizes "
              "up to INT_MAX items.");
#ifdef TI_WITH_CUDA
  auto *src_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(src));
  auto *indices_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(indices));
  auto *dst_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(dst));
  return cuda_scatter_add_contiguous(src_ptr, indices_ptr, dst_ptr,
                                     indices->get_nelement(),
                                     dst->get_nelement(), value_type);
#else
  TI_ERROR(
      "CUDA scatter-add requires building Taichi with TI_WITH_CUDA=ON and "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t Program::cuda_device_scatter_add_member_ndarray(
    Ndarray *src,
    Ndarray *indices,
    Ndarray *dst,
    int value_type,
    std::size_t offset,
    std::size_t stride) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA toolkit strided scatter-add is only available on CUDA.");
  check_scatter_add_member_request("CUDA toolkit", src, indices, dst,
                                   value_type, offset, stride);
  TI_ERROR_IF(indices->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA toolkit strided scatter-add currently supports at most "
              "INT_MAX source items.");
  TI_ERROR_IF(dst->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA toolkit strided scatter-add currently supports "
              "destination sizes up to INT_MAX items.");
#ifdef TI_WITH_CUDA
  void *stream = nullptr;
  return cuda::driver_scatter_add_strided(
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(src)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(indices)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(dst)),
      static_cast<int>(indices->get_nelement()),
      static_cast<int>(dst->get_nelement()),
      static_cast<cuda::CudaTransformValueType>(value_type), offset, stride, 0,
      sizeof(std::int32_t), offset, stride, stream);
#else
  TI_ERROR(
      "CUDA strided scatter-add requires building Taichi with TI_WITH_CUDA=ON "
      "and TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t Program::cuda_device_scatter_add_strided_ndarray(
    Ndarray *src,
    Ndarray *indices,
    Ndarray *dst,
    int value_type,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA toolkit strided scatter-add is only available on CUDA.");
  check_scatter_add_strided_request("CUDA toolkit", src, indices, dst,
                                    value_type, src_offset, src_stride,
                                    dst_offset, dst_stride);
  TI_ERROR_IF(indices->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA toolkit strided scatter-add currently supports at most "
              "INT_MAX source items.");
  TI_ERROR_IF(dst->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA toolkit strided scatter-add currently supports "
              "destination sizes up to INT_MAX items.");
#ifdef TI_WITH_CUDA
  void *stream = nullptr;
  return cuda::driver_scatter_add_strided(
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(src)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(indices)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(dst)),
      static_cast<int>(indices->get_nelement()),
      static_cast<int>(dst->get_nelement()),
      static_cast<cuda::CudaTransformValueType>(value_type), src_offset,
      src_stride, 0, sizeof(std::int32_t), dst_offset, dst_stride, stream);
#else
  TI_ERROR(
      "CUDA strided scatter-add requires building Taichi with TI_WITH_CUDA=ON "
      "and TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t Program::cuda_device_scatter_add_dense_field(SNode *src,
                                                         Ndarray *indices,
                                                         SNode *dst,
                                                         int value_type,
                                                         std::size_t src_n,
                                                         std::size_t dst_n) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA toolkit dense field scatter-add is only available on "
              "CUDA.");
  check_indexed_copy_dense_field_request(this, "CUDA toolkit", src, indices,
                                         dst, value_type, src_n, dst_n, true);
  const std::size_t n = indices->get_nelement();
  if (n == 0 || dst_n == 0) {
    return 0;
  }
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA toolkit dense field scatter-add currently supports at "
              "most INT_MAX source items.");
  TI_ERROR_IF(dst_n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA toolkit dense field scatter-add currently supports "
              "destination sizes up to INT_MAX items.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  const std::size_t src_stride = get_dense_field_stride(src, value_size);
  const std::size_t dst_stride = get_dense_field_stride(dst, value_size);
#ifdef TI_WITH_CUDA
  auto raw_ptr = [this](DevicePtr ptr, const char *op_name) {
    TI_ERROR_IF(!ptr.device, "{} received a null dense field device.", op_name);
    DeviceAllocation alloc{ptr.device, ptr.alloc_id};
    auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
    TI_ERROR_IF(!base, "{} received a null dense field data pointer.", op_name);
    return static_cast<void *>(reinterpret_cast<uint8_t *>(base) + ptr.offset);
  };
  void *src_ptr = raw_ptr(get_dense_field_device_ptr(src),
                          "CUDA toolkit dense field scatter-add");
  auto *indices_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(indices));
  void *dst_ptr = raw_ptr(get_dense_field_device_ptr(dst),
                          "CUDA toolkit dense field scatter-add");
  void *stream = nullptr;
  return cuda::driver_scatter_add_strided(
      src_ptr, indices_ptr, dst_ptr, static_cast<int>(n),
      static_cast<int>(dst_n),
      static_cast<cuda::CudaTransformValueType>(value_type), 0, src_stride, 0,
      sizeof(std::int32_t), 0, dst_stride, stream);
#else
  TI_ERROR(
      "CUDA dense field scatter-add requires building Taichi with "
      "TI_WITH_CUDA=ON and TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t Program::cuda_device_scatter_add_dense_field_indices_field(
    SNode *src,
    SNode *indices,
    SNode *dst,
    int value_type,
    std::size_t src_n,
    std::size_t indices_n,
    std::size_t dst_n) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA toolkit dense field scatter-add is only available on "
              "CUDA.");
  check_indexed_copy_dense_field_indices_field_request(
      this, "CUDA toolkit", src, indices, dst, value_type, src_n, indices_n,
      dst_n, true);
  const std::size_t n = indices_n;
  if (n == 0 || dst_n == 0) {
    return 0;
  }
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA toolkit dense field scatter-add currently supports at "
              "most INT_MAX source items.");
  TI_ERROR_IF(dst_n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA toolkit dense field scatter-add currently supports "
              "destination sizes up to INT_MAX items.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  const std::size_t src_stride = get_dense_field_stride(src, value_size);
  const std::size_t indices_stride =
      get_dense_field_stride(indices, sizeof(std::int32_t));
  const std::size_t dst_stride = get_dense_field_stride(dst, value_size);
#ifdef TI_WITH_CUDA
  auto raw_ptr = [this](DevicePtr ptr, const char *op_name) {
    TI_ERROR_IF(!ptr.device, "{} received a null dense field device.", op_name);
    DeviceAllocation alloc{ptr.device, ptr.alloc_id};
    auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
    TI_ERROR_IF(!base, "{} received a null dense field data pointer.", op_name);
    return static_cast<void *>(reinterpret_cast<uint8_t *>(base) + ptr.offset);
  };
  void *src_ptr = raw_ptr(get_dense_field_device_ptr(src),
                          "CUDA toolkit dense field scatter-add");
  void *indices_ptr = raw_ptr(get_dense_field_device_ptr(indices),
                              "CUDA toolkit dense field scatter-add");
  void *dst_ptr = raw_ptr(get_dense_field_device_ptr(dst),
                          "CUDA toolkit dense field scatter-add");
  void *stream = nullptr;
  return cuda::driver_scatter_add_strided(
      src_ptr, indices_ptr, dst_ptr, static_cast<int>(n),
      static_cast<int>(dst_n),
      static_cast<cuda::CudaTransformValueType>(value_type), 0, src_stride, 0,
      indices_stride, 0, dst_stride, stream);
#else
  TI_ERROR(
      "CUDA dense field scatter-add requires building Taichi with "
      "TI_WITH_CUDA=ON and TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

bool Program::cuda_device_bucket_builder_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda &&
         cuda::driver_hierarchical_available();
#else
  return false;
#endif
}

std::size_t Program::cuda_device_bucket_builder_i32_ndarray(Ndarray *keys,
                                                            Ndarray *values,
                                                            Ndarray *offsets,
                                                            Ndarray *output,
                                                            Ndarray *cursor) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  return cuda_device_bucket_builder_ndarray(keys, values, offsets, output,
                                            cursor, 0);
}

std::size_t Program::cuda_device_bucket_builder_ndarray(Ndarray *keys,
                                                        Ndarray *values,
                                                        Ndarray *offsets,
                                                        Ndarray *output,
                                                        Ndarray *cursor,
                                                        int value_type) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA toolkit bucket builder is only available on CUDA.");
  TI_ERROR_IF(!keys || !values || !offsets || !output || !cursor,
              "CUDA toolkit bucket builder received a null ndarray.");
  TI_ERROR_IF(keys->shape.size() != 1 || values->shape.size() != 1 ||
                  offsets->shape.size() != 1 || output->shape.size() != 1 ||
                  cursor->shape.size() != 1,
              "CUDA toolkit bucket builder expects 1D ndarrays.");
  TI_ERROR_IF(keys->get_nelement() != values->get_nelement(),
              "CUDA toolkit bucket builder keys and values sizes differ.");
  TI_ERROR_IF(
      offsets->get_nelement() < 2,
      "CUDA toolkit bucket builder offsets must contain num_bins + 1 items.");
  const std::size_t num_bins = offsets->get_nelement() - 1;
  TI_ERROR_IF(cursor->get_nelement() < num_bins,
              "CUDA toolkit bucket builder cursor is smaller than num_bins.");
  TI_ERROR_IF(
      output->get_nelement() < values->get_nelement(),
      "CUDA toolkit bucket builder output is smaller than input values.");
  const std::size_t expected_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(
      expected_size == 0,
      "CUDA toolkit bucket builder received an unsupported value type.");
  const std::size_t item_bytes = values->get_element_size();
  TI_ERROR_IF(
      keys->get_element_size() != sizeof(int32_t) ||
          offsets->get_element_size() != sizeof(int32_t) || item_bytes == 0 ||
          item_bytes % sizeof(uint32_t) != 0 ||
          output->get_element_size() != item_bytes ||
          cursor->get_element_size() != sizeof(int32_t),
      "CUDA toolkit bucket builder dtype does not match value type or "
      "keys/offsets/cursor are not i32, or payload is not 4-byte aligned.");
  TI_ERROR_IF(
      keys->get_nelement() >
              static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()) ||
          num_bins >
              static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()),
      "CUDA bucket builder input is too large for u32 launch parameters.");
  const int item_words = static_cast<int>(item_bytes / sizeof(uint32_t));
  TI_ERROR_IF(
      keys->get_nelement() > static_cast<std::size_t>(
                                 std::numeric_limits<int>::max() / item_words),
      "CUDA bucket builder word count exceeds INT_MAX.");
#ifdef TI_WITH_CUDA
  void *stream = nullptr;
  return cuda::driver_bucket_builder_strided(
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(keys)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(offsets)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(cursor)),
      static_cast<int>(keys->get_nelement()), static_cast<int>(num_bins),
      item_words, 0, sizeof(std::int32_t), 0, item_bytes, 0,
      sizeof(std::int32_t), 0, item_bytes, stream, &primitive_workspace_arena_);
#else
  TI_ERROR(
      "CUDA bucket builder requires building Taichi with TI_WITH_CUDA=ON and "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t Program::cuda_device_bucket_builder_dense_field(
    SNode *keys,
    SNode *values,
    SNode *offsets,
    SNode *output,
    Ndarray *cursor,
    int value_type,
    std::size_t n,
    std::size_t num_bins) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA toolkit dense field bucket builder is only available on "
              "CUDA.");
  TI_ERROR_IF(!keys || !values || !offsets || !output || !cursor,
              "CUDA toolkit dense field bucket builder received a null input.");
  TI_ERROR_IF(num_bins == 0,
              "CUDA toolkit dense field bucket builder expects at least one "
              "bucket.");
  TI_ERROR_IF(cursor->shape.size() != 1 || cursor->get_nelement() < num_bins ||
                  cursor->get_element_size() != sizeof(int32_t),
              "CUDA toolkit dense field bucket builder cursor must be a 1D "
              "i32 ndarray with at least num_bins items.");
  const std::size_t item_bytes = primitive_value_type_size(value_type);
  TI_ERROR_IF(item_bytes == 0 || item_bytes % sizeof(uint32_t) != 0,
              "CUDA toolkit dense field bucket builder received an unsupported "
              "value type.");
  const std::size_t keys_stride = get_dense_field_stride(keys, sizeof(int32_t));
  const std::size_t values_stride = get_dense_field_stride(values, item_bytes);
  const std::size_t offsets_stride =
      get_dense_field_stride(offsets, sizeof(int32_t));
  const std::size_t output_stride = get_dense_field_stride(output, item_bytes);
  TI_ERROR_IF(keys_stride != sizeof(int32_t) || values_stride != item_bytes ||
                  offsets_stride != sizeof(int32_t) ||
                  output_stride != item_bytes,
              "CUDA toolkit dense field bucket builder requires contiguous "
              "keys, values, offsets, and output fields.");
  TI_ERROR_IF(
      n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
          num_bins > static_cast<std::size_t>(std::numeric_limits<int>::max()),
      "CUDA dense field bucket builder input is too large for int "
      "launch parameters.");
  const int item_words = static_cast<int>(item_bytes / sizeof(uint32_t));
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max() /
                                           item_words),
              "CUDA dense field bucket builder word count exceeds INT_MAX.");
#ifdef TI_WITH_CUDA
  auto raw_ptr = [this](DevicePtr ptr, const char *op_name) {
    TI_ERROR_IF(!ptr.device, "{} received a null dense field device.", op_name);
    DeviceAllocation alloc{ptr.device, ptr.alloc_id};
    auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
    TI_ERROR_IF(!base, "{} received a null dense field data pointer.", op_name);
    return static_cast<void *>(reinterpret_cast<uint8_t *>(base) + ptr.offset);
  };
  void *keys_ptr = raw_ptr(get_dense_field_device_ptr(keys),
                           "CUDA toolkit dense field bucket builder");
  void *values_ptr = raw_ptr(get_dense_field_device_ptr(values),
                             "CUDA toolkit dense field bucket builder");
  void *offsets_ptr = raw_ptr(get_dense_field_device_ptr(offsets),
                              "CUDA toolkit dense field bucket builder");
  void *output_ptr = raw_ptr(get_dense_field_device_ptr(output),
                             "CUDA toolkit dense field bucket builder");
  void *cursor_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(cursor));
  void *stream = nullptr;
  return cuda::driver_bucket_builder_strided(
      keys_ptr, values_ptr, offsets_ptr, output_ptr, cursor_ptr,
      static_cast<int>(n), static_cast<int>(num_bins), item_words, 0,
      keys_stride, 0, values_stride, 0, offsets_stride, 0, output_stride,
      stream, &primitive_workspace_arena_);
#else
  TI_ERROR(
      "CUDA dense field bucket builder requires building Taichi with "
      "TI_WITH_CUDA=ON and TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

bool Program::cuda_device_grouped_reduce_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda &&
         cuda::driver_hierarchical_available();
#else
  return false;
#endif
}

std::size_t Program::cuda_device_grouped_reduce_i32_atomic_ndarray(
    Ndarray *keys,
    Ndarray *values,
    Ndarray *output,
    int op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  return cuda_device_grouped_reduce_atomic_ndarray(keys, values, output, 0, op);
}

std::size_t Program::cuda_device_grouped_reduce_atomic_ndarray(Ndarray *keys,
                                                               Ndarray *values,
                                                               Ndarray *output,
                                                               int value_type,
                                                               int op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA grouped reduce is only available on CUDA.");
  TI_ERROR_IF(!keys || !values || !output,
              "CUDA grouped reduce received a null ndarray.");
  TI_ERROR_IF(keys->shape.size() != 1 || values->shape.size() != 1 ||
                  output->shape.size() != 1,
              "CUDA grouped reduce expects 1D ndarrays.");
  TI_ERROR_IF(keys->get_nelement() != values->get_nelement(),
              "CUDA grouped reduce keys and values sizes differ.");
  TI_ERROR_IF(output->get_nelement() == 0,
              "CUDA grouped reduce output must contain at least one group.");
  const std::size_t num_groups = output->get_nelement();
  const std::size_t expected_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(expected_size == 0,
              "CUDA grouped reduce received an unsupported value type.");
  TI_ERROR_IF(keys->get_element_size() != sizeof(int32_t) ||
                  values->get_element_size() != expected_size ||
                  output->get_element_size() != expected_size,
              "CUDA grouped reduce value type or i32 key size mismatch.");
  TI_ERROR_IF(op != 0, "CUDA grouped reduce currently supports only sum.");
  TI_ERROR_IF(
      keys->get_nelement() >
              static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
          num_groups >
              static_cast<std::size_t>(std::numeric_limits<int>::max()),
      "CUDA grouped reduce input is too large for int launch parameters.");
#ifdef TI_WITH_CUDA
  void *stream = nullptr;
  return cuda::driver_grouped_reduce_strided(
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(keys)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output)),
      static_cast<int>(keys->get_nelement()), static_cast<int>(num_groups),
      static_cast<cuda::CudaTransformValueType>(value_type), 0,
      sizeof(std::int32_t), 0, expected_size, 0, expected_size, stream);
#else
  TI_ERROR(
      "CUDA grouped reduce requires building Taichi with TI_WITH_CUDA=ON and "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t Program::cuda_device_grouped_reduce_atomic_dense_field(
    SNode *keys,
    SNode *values,
    SNode *output,
    int value_type,
    std::size_t n,
    std::size_t num_groups,
    int op) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA dense field grouped reduce is only available on CUDA.");
  TI_ERROR_IF(!keys || !values || !output,
              "CUDA dense field grouped reduce received a null field.");
  TI_ERROR_IF(num_groups == 0,
              "CUDA dense field grouped reduce output must contain at least "
              "one group.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CUDA dense field grouped reduce received an unsupported value "
              "type.");
  const std::size_t keys_stride = get_dense_field_stride(keys, sizeof(int32_t));
  const std::size_t values_stride = get_dense_field_stride(values, value_size);
  const std::size_t output_stride = get_dense_field_stride(output, value_size);
  TI_ERROR_IF(keys_stride < sizeof(int32_t) || values_stride < value_size ||
                  output_stride < value_size,
              "CUDA dense field grouped reduce received an invalid stride.");
  TI_ERROR_IF(op != 0, "CUDA dense field grouped reduce supports only sum.");
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
                  num_groups >
                      static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA dense field grouped reduce input is too large for int "
              "launch parameters.");
#ifdef TI_WITH_CUDA
  auto raw_ptr = [this](DevicePtr ptr, const char *op_name) {
    TI_ERROR_IF(!ptr.device, "{} received a null dense field device.", op_name);
    DeviceAllocation alloc{ptr.device, ptr.alloc_id};
    auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
    TI_ERROR_IF(!base, "{} received a null dense field data pointer.", op_name);
    return static_cast<void *>(reinterpret_cast<uint8_t *>(base) + ptr.offset);
  };
  void *keys_ptr = raw_ptr(get_dense_field_device_ptr(keys),
                           "CUDA dense field grouped reduce");
  void *values_ptr = raw_ptr(get_dense_field_device_ptr(values),
                             "CUDA dense field grouped reduce");
  void *output_ptr = raw_ptr(get_dense_field_device_ptr(output),
                             "CUDA dense field grouped reduce");
  void *stream = nullptr;
  return cuda::driver_grouped_reduce_strided(
      keys_ptr, values_ptr, output_ptr, static_cast<int>(n),
      static_cast<int>(num_groups),
      static_cast<cuda::CudaTransformValueType>(value_type), 0, keys_stride, 0,
      values_stride, 0, output_stride, stream);
#else
  TI_ERROR(
      "CUDA dense field grouped reduce requires building Taichi with "
      "TI_WITH_CUDA=ON and TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t Program::cuda_device_grouped_reduce_atomic_member_ndarray(
    Ndarray *keys,
    Ndarray *values,
    Ndarray *output,
    int value_type,
    std::size_t offset,
    std::size_t stride,
    int op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA strided grouped reduce is only available on CUDA.");
  check_grouped_reduce_member_request("CUDA", keys, values, output, value_type,
                                      offset, stride, op);
  const std::size_t num_groups = output->get_nelement();
  TI_ERROR_IF(keys->get_nelement() > static_cast<std::size_t>(
                                         std::numeric_limits<int>::max()) ||
                  num_groups >
                      static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA strided grouped reduce input is too large for int launch "
              "parameters.");
#ifdef TI_WITH_CUDA
  void *stream = nullptr;
  return cuda::driver_grouped_reduce_strided(
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(keys)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output)),
      static_cast<int>(keys->get_nelement()), static_cast<int>(num_groups),
      static_cast<cuda::CudaTransformValueType>(value_type), 0,
      sizeof(std::int32_t), offset, stride, offset, stride, stream);
#else
  TI_ERROR(
      "CUDA strided grouped reduce requires building Taichi with "
      "TI_WITH_CUDA=ON and TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t Program::cuda_device_grouped_reduce_atomic_strided_ndarray(
    Ndarray *keys,
    Ndarray *values,
    Ndarray *output,
    int value_type,
    std::size_t values_offset,
    std::size_t values_stride,
    std::size_t output_offset,
    std::size_t output_stride,
    int op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  return cuda_device_grouped_reduce_atomic_strided_keys_ndarray(
      keys, values, output, value_type, 0, sizeof(int32_t), values_offset,
      values_stride, output_offset, output_stride, op);
}

std::size_t Program::cuda_device_grouped_reduce_atomic_strided_keys_ndarray(
    Ndarray *keys,
    Ndarray *values,
    Ndarray *output,
    int value_type,
    std::size_t keys_offset,
    std::size_t keys_stride,
    std::size_t values_offset,
    std::size_t values_stride,
    std::size_t output_offset,
    std::size_t output_stride,
    int op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA strided grouped reduce is only available on CUDA.");
  check_grouped_reduce_strided_keys_request(
      "CUDA", keys, values, output, value_type, keys_offset, keys_stride,
      values_offset, values_stride, output_offset, output_stride, op);
  const std::size_t num_groups = output->get_nelement();
  TI_ERROR_IF(keys->get_nelement() > static_cast<std::size_t>(
                                         std::numeric_limits<int>::max()) ||
                  num_groups >
                      static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA strided grouped reduce input is too large for int launch "
              "parameters.");
#ifdef TI_WITH_CUDA
  void *stream = nullptr;
  return cuda::driver_grouped_reduce_strided(
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(keys)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output)),
      static_cast<int>(keys->get_nelement()), static_cast<int>(num_groups),
      static_cast<cuda::CudaTransformValueType>(value_type), keys_offset,
      keys_stride, values_offset, values_stride, output_offset, output_stride,
      stream);
#else
  TI_ERROR(
      "CUDA strided grouped reduce requires building Taichi with "
      "TI_WITH_CUDA=ON and TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t Program::cuda_device_grouped_reduce_i32_ndarray(Ndarray *keys,
                                                            Ndarray *values,
                                                            Ndarray *output,
                                                            Ndarray *offsets,
                                                            Ndarray *scratch,
                                                            Ndarray *cursor,
                                                            int op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  return cuda_device_grouped_reduce_ndarray(keys, values, output, offsets,
                                            scratch, cursor, 0, op);
}

std::size_t Program::cuda_device_grouped_reduce_ndarray(Ndarray *keys,
                                                        Ndarray *values,
                                                        Ndarray *output,
                                                        Ndarray *offsets,
                                                        Ndarray *scratch,
                                                        Ndarray *cursor,
                                                        int value_type,
                                                        int op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA grouped reduce is only available on CUDA.");
  TI_ERROR_IF(!keys || !values || !output || !offsets || !scratch || !cursor,
              "CUDA grouped reduce received a null ndarray.");
  TI_ERROR_IF(keys->shape.size() != 1 || values->shape.size() != 1 ||
                  output->shape.size() != 1 || offsets->shape.size() != 1 ||
                  scratch->shape.size() != 1 || cursor->shape.size() != 1,
              "CUDA grouped reduce expects 1D ndarrays.");
  TI_ERROR_IF(keys->get_nelement() != values->get_nelement(),
              "CUDA grouped reduce keys and values sizes differ.");
  TI_ERROR_IF(output->get_nelement() == 0,
              "CUDA grouped reduce output must contain at least one group.");
  const std::size_t num_groups = output->get_nelement();
  TI_ERROR_IF(offsets->get_nelement() < num_groups + 1,
              "CUDA grouped reduce offsets must contain num_groups + 1 items.");
  TI_ERROR_IF(scratch->get_nelement() < values->get_nelement(),
              "CUDA grouped reduce scratch is smaller than input values.");
  TI_ERROR_IF(cursor->get_nelement() < num_groups,
              "CUDA grouped reduce cursor is smaller than num_groups.");
  const std::size_t expected_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(expected_size == 0,
              "CUDA grouped reduce received an unsupported value type.");
  TI_ERROR_IF(keys->get_element_size() != sizeof(int32_t) ||
                  values->get_element_size() != expected_size ||
                  output->get_element_size() != expected_size ||
                  offsets->get_element_size() != sizeof(int32_t) ||
                  scratch->get_element_size() != expected_size ||
                  cursor->get_element_size() != sizeof(int32_t),
              "CUDA grouped reduce value type or i32 metadata size mismatch.");
  TI_ERROR_IF(op != 0, "CUDA grouped reduce currently supports only sum.");
  TI_ERROR_IF(
      keys->get_nelement() >
              static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
          num_groups >
              static_cast<std::size_t>(std::numeric_limits<int>::max()),
      "CUDA grouped reduce input is too large for int launch parameters.");
#ifdef TI_WITH_CUDA
  void *stream = nullptr;
  return cuda::driver_grouped_reduce_strided(
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(keys)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output)),
      static_cast<int>(keys->get_nelement()), static_cast<int>(num_groups),
      static_cast<cuda::CudaTransformValueType>(value_type), 0,
      sizeof(std::int32_t), 0, expected_size, 0, expected_size, stream);
#else
  TI_ERROR(
      "CUDA grouped reduce requires building Taichi with TI_WITH_CUDA=ON and "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t Program::cuda_device_grouped_reduce_segmented_strided_keys_ndarray(
    Ndarray *keys,
    Ndarray *values,
    Ndarray *output,
    Ndarray *offsets,
    Ndarray *scratch,
    Ndarray *cursor,
    int value_type,
    std::size_t keys_offset,
    std::size_t keys_stride,
    std::size_t values_offset,
    std::size_t values_stride,
    std::size_t output_offset,
    std::size_t output_stride,
    int op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA strided segmented grouped reduce is only available on "
              "CUDA.");
  TI_ERROR_IF(!offsets || !scratch || !cursor,
              "CUDA strided segmented grouped reduce received a null "
              "workspace ndarray.");
  check_grouped_reduce_strided_keys_request(
      "CUDA", keys, values, output, value_type, keys_offset, keys_stride,
      values_offset, values_stride, output_offset, output_stride, op);
  const std::size_t n = keys->get_nelement();
  const std::size_t num_groups = output->get_nelement();
  TI_ERROR_IF(offsets->shape.size() != 1 || scratch->shape.size() != 1 ||
                  cursor->shape.size() != 1,
              "CUDA strided segmented grouped reduce workspace expects 1D "
              "ndarrays.");
  TI_ERROR_IF(offsets->get_nelement() < num_groups + 1,
              "CUDA strided segmented grouped reduce offsets must contain "
              "num_groups + 1 items.");
  TI_ERROR_IF(scratch->get_nelement() < n,
              "CUDA strided segmented grouped reduce scratch is smaller than "
              "input values.");
  TI_ERROR_IF(cursor->get_nelement() < num_groups,
              "CUDA strided segmented grouped reduce cursor is smaller than "
              "num_groups.");
  const std::size_t expected_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(
      offsets->get_element_size() != sizeof(int32_t) ||
          scratch->get_element_size() != expected_size ||
          cursor->get_element_size() != sizeof(int32_t),
      "CUDA strided segmented grouped reduce workspace dtype mismatch.");
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
                  num_groups >
                      static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA strided segmented grouped reduce input is too large for "
              "int launch parameters.");
#ifdef TI_WITH_CUDA
  void *stream = nullptr;
  return cuda::driver_grouped_reduce_strided(
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(keys)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output)),
      static_cast<int>(n), static_cast<int>(num_groups),
      static_cast<cuda::CudaTransformValueType>(value_type), keys_offset,
      keys_stride, values_offset, values_stride, output_offset, output_stride,
      stream);
#else
  TI_ERROR(
      "CUDA strided segmented grouped reduce requires building Taichi with "
      "TI_WITH_CUDA=ON and TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

bool Program::cuda_sparse_assembly_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda &&
         cuda::driver_hierarchical_available();
#else
  return false;
#endif
}

CudaSparseAssemblyDispatchInfo Program::cuda_sparse_assemble_csr(
    Ndarray *packed_triplets,
    Ndarray *triplet_rows,
    Ndarray *triplet_columns,
    Ndarray *triplet_values,
    Ndarray *sorted_keys,
    Ndarray *sorted_values,
    Ndarray *segment_ids,
    Ndarray *unique_keys,
    Ndarray *segment_offsets,
    Ndarray *unique_values,
    Ndarray *row_offsets,
    Ndarray *column_indices,
    Ndarray *active_count,
    Ndarray *control,
    std::size_t capacity,
    std::size_t rows,
    std::size_t cols) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!cuda_sparse_assembly_available(),
              "CUDA sparse assembly requires the Driver hierarchical "
              "primitive provider.");
  TI_ERROR_IF(capacity == 0 ||
                  capacity >=
                      static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA sparse assembly capacity must be in [1, INT_MAX).");
  TI_ERROR_IF(rows == 0 || cols == 0 ||
                  rows >=
                      static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
                  cols >
                      static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA sparse assembly rows must be in [1, INT_MAX), and "
              "columns in [1, INT_MAX].");
  const bool packed_input = packed_triplets != nullptr;
  TI_ERROR_IF(
      packed_input
          ? (triplet_rows || triplet_columns || triplet_values)
          : (!triplet_rows || !triplet_columns || !triplet_values),
      "CUDA sparse assembly requires exactly one input layout: packed "
      "builder storage or three separate triplet arrays.");
  std::vector<Ndarray *> arrays{
      sorted_keys,     sorted_values,  segment_ids, unique_keys,
      segment_offsets, unique_values,  row_offsets, column_indices,
      active_count,    control};
  if (packed_input) {
    arrays.push_back(packed_triplets);
  } else {
    arrays.push_back(triplet_rows);
    arrays.push_back(triplet_columns);
    arrays.push_back(triplet_values);
  }
  for (Ndarray *array : arrays) {
    TI_ERROR_IF(!array, "CUDA sparse assembly received a null ndarray.");
    TI_ERROR_IF(array->shape.size() != 1 ||
                    !array->get_element_shape().empty(),
                "CUDA sparse assembly expects scalar 1D ndarrays.");
  }
  auto check_array = [](const char *name, Ndarray *array, DataType dtype,
                        std::size_t count, std::size_t item_bytes) {
    TI_ERROR_IF(array->get_element_data_type() != dtype ||
                    array->get_nelement() != count ||
                    array->get_element_size() != item_bytes,
                "CUDA sparse assembly {} has an unexpected dtype, shape, or "
                "byte width; expected exactly {} scalar entries.",
                name, count);
  };
  if (packed_input) {
    check_array("packed triplets", packed_triplets, PrimitiveType::f32,
                capacity * 3 + 2, sizeof(float32));
  } else {
    check_array("triplet rows", triplet_rows, PrimitiveType::i32, capacity,
                sizeof(int32_t));
    check_array("triplet columns", triplet_columns, PrimitiveType::i32,
                capacity, sizeof(int32_t));
    check_array("triplet values", triplet_values, PrimitiveType::f32,
                capacity, sizeof(float32));
  }
  check_array("sorted keys", sorted_keys, PrimitiveType::u64, capacity,
              sizeof(uint64_t));
  check_array("sorted values", sorted_values, PrimitiveType::f32, capacity,
              sizeof(float32));
  check_array("segment ids", segment_ids, PrimitiveType::i32, capacity,
              sizeof(int32_t));
  check_array("unique keys", unique_keys, PrimitiveType::u64, capacity,
              sizeof(uint64_t));
  check_array("segment offsets", segment_offsets, PrimitiveType::i32,
              capacity + 1, sizeof(int32_t));
  check_array("unique values", unique_values, PrimitiveType::f32, capacity,
              sizeof(float32));
  check_array("row offsets", row_offsets, PrimitiveType::i32, rows + 1,
              sizeof(int32_t));
  check_array("column indices", column_indices, PrimitiveType::i32, capacity,
              sizeof(int32_t));
  check_array("active count", active_count, PrimitiveType::i32, 1,
              sizeof(int32_t));
  check_array("control", control, PrimitiveType::i32, 2, sizeof(int32_t));

  std::vector<DeviceAllocation> allocs(arrays.size());
  for (std::size_t i = 0; i < arrays.size(); ++i) {
    allocs[i] = arrays[i]->get_device_allocation();
    for (std::size_t j = 0; j < i; ++j) {
      TI_ERROR_IF(allocs[i] == allocs[j],
                  "CUDA sparse assembly buffers must not alias.");
    }
  }

#ifdef TI_WITH_CUDA
  auto ptr = [this](Ndarray *array) {
    return reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(array));
  };
  constexpr auto i32_type = cuda::CudaTransformValueType::i32;
  constexpr auto u64_key_type = cuda::CudaDriverSortKeyType::u64;
  void *stream = nullptr;
  const int capacity_i = static_cast<int>(capacity);
  const int rows_i = static_cast<int>(rows);
  const int cols_i = static_cast<int>(cols);

  cuda::driver_zero_strided(ptr(control), 2, i32_type, 0, sizeof(int32_t),
                            stream);
  if (packed_input) {
    cuda::driver_sparse_assembly_pack_packed_validate(
        ptr(packed_triplets), ptr(sorted_keys), ptr(sorted_values),
        ptr(active_count), ptr(control), capacity_i, rows_i, cols_i, stream);
  } else {
    cuda::driver_sparse_assembly_pack_validate(
        ptr(triplet_rows), ptr(triplet_columns), ptr(triplet_values),
        ptr(sorted_keys), ptr(sorted_values), ptr(active_count), ptr(control),
        capacity_i, rows_i, cols_i, stream);
  }

  CudaSparseAssemblyDispatchInfo result;
  result.radix_sort_workspace_bytes =
      cuda::driver_stable_radix_sort_strided(
          ptr(sorted_keys), ptr(sorted_values), capacity_i, u64_key_type, 1, 0,
          sizeof(uint64_t), 0, sizeof(float32), true, 0, stream,
          &primitive_workspace_arena_);

  cuda::driver_sparse_assembly_mark_segments(
      ptr(sorted_keys), ptr(segment_ids), ptr(active_count), ptr(control),
      capacity_i, stream);
  result.scan_workspace_bytes = cuda::driver_inclusive_scan_strided(
      ptr(segment_ids), capacity_i, i32_type, 0, sizeof(int32_t), false,
      stream, &primitive_workspace_arena_);
  cuda::driver_sparse_assembly_scatter_segments(
      ptr(sorted_keys), ptr(segment_ids), ptr(unique_keys),
      ptr(segment_offsets), ptr(active_count), ptr(control), capacity_i,
      stream);
  cuda::driver_sparse_assembly_reduce_segments(
      ptr(sorted_values), ptr(segment_offsets), ptr(unique_values),
      ptr(active_count), ptr(control), capacity_i, stream);

  cuda::driver_zero_strided(ptr(row_offsets), rows_i + 1, i32_type, 0,
                            sizeof(int32_t), stream);
  cuda::driver_sparse_assembly_emit_csr(
      ptr(unique_keys), ptr(row_offsets), ptr(column_indices),
      ptr(active_count), ptr(control), capacity_i, rows_i, cols_i, stream);
  result.scan_workspace_bytes = cuda::driver_inclusive_scan_strided(
      ptr(row_offsets), rows_i + 1, i32_type, 0, sizeof(int32_t), false,
      stream, &primitive_workspace_arena_);
  cuda::driver_sparse_assembly_finalize_control(
      ptr(active_count), ptr(control), capacity_i, stream);
  // Driver workspaces retain every replaced generation until Program
  // teardown. Growth therefore does not require an assembly-local sync.
  result.workspace_growth_synchronized = false;
  return result;
#else
  TI_NOT_IMPLEMENTED;
#endif
}

bool Program::cuda_device_radix_sort_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda &&
         cuda::driver_hierarchical_available();
#else
  return false;
#endif
}

std::size_t Program::cuda_device_radix_sort_ndarray(Ndarray *keys,
                                                    Ndarray *values,
                                                    int key_type,
                                                    int value_type,
                                                    int nan_policy) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA Driver stable sort is only available on CUDA.");
  TI_ERROR_IF(!keys || keys->shape.size() != 1,
              "CUDA Driver stable sort expects a 1D keys ndarray.");
  const bool has_values = values != nullptr;
  TI_ERROR_IF(has_values && (values->shape.size() != 1 ||
                             values->get_nelement() != keys->get_nelement()),
              "CUDA Driver stable sort expects a matching 1D values ndarray.");
  const std::size_t expected_key_size = sort_key_type_size(key_type);
  TI_ERROR_IF(
      expected_key_size == 0 || keys->get_element_size() != expected_key_size,
      "CUDA Driver stable sort key dtype does not match key_type.");
  TI_ERROR_IF(nan_policy < 0 || nan_policy > 1,
              "CUDA Driver stable sort received an unsupported NaN policy.");
  const std::size_t expected_value_size = primitive_value_type_size(value_type);
  const std::size_t actual_value_size =
      has_values ? values->get_element_size() : 0;
  TI_ERROR_IF(
      has_values && (expected_value_size == 0 || actual_value_size == 0 ||
                     actual_value_size % sizeof(std::uint32_t) != 0),
      "CUDA Driver stable sort value payload must be 4-byte aligned.");
  TI_ERROR_IF(keys->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA Driver stable sort supports at most INT_MAX items.");
  if (keys->get_nelement() <= 1) {
    return 0;
  }
#ifdef TI_WITH_CUDA
  return cuda::driver_stable_radix_sort_strided(
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(keys)),
      has_values ? reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values))
                 : nullptr,
      static_cast<int>(keys->get_nelement()),
      static_cast<cuda::CudaDriverSortKeyType>(key_type),
      has_values ? static_cast<int>(actual_value_size / sizeof(std::uint32_t))
                 : 0,
      0, expected_key_size, 0, actual_value_size, has_values, nan_policy,
      nullptr, &primitive_workspace_arena_);
#else
  TI_NOT_IMPLEMENTED;
#endif
}

std::size_t Program::cuda_device_radix_sort_dense_field(SNode *keys,
                                                        SNode *values,
                                                        int key_type,
                                                        int value_type,
                                                        std::size_t n,
                                                        int nan_policy) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA Driver dense-field stable sort is only available on CUDA.");
  TI_ERROR_IF(!keys, "CUDA Driver dense-field stable sort received null keys.");
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA Driver dense-field stable sort supports at most INT_MAX "
              "items.");
  const std::size_t expected_key_size = sort_key_type_size(key_type);
  TI_ERROR_IF(expected_key_size == 0,
              "CUDA Driver dense-field stable sort received an unsupported "
              "key type.");
  TI_ERROR_IF(nan_policy < 0 || nan_policy > 1,
              "CUDA Driver dense-field stable sort received an unsupported "
              "NaN policy.");
  const bool has_values = values != nullptr;
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(has_values && value_size == 0,
              "CUDA Driver dense-field stable sort received an unsupported "
              "value type.");
  const std::size_t keys_stride =
      get_dense_field_stride(keys, expected_key_size);
  const std::size_t values_stride =
      has_values ? get_dense_field_stride(values, value_size) : 0;
  TI_ERROR_IF(keys_stride < expected_key_size ||
                  (has_values && values_stride < value_size),
              "CUDA Driver dense-field stable sort received an invalid "
              "field stride.");
  if (n <= 1) {
    return 0;
  }
#ifdef TI_WITH_CUDA
  auto raw_ptr = [this](DevicePtr ptr, const char *op_name) {
    TI_ERROR_IF(!ptr.device, "{} received a null dense-field device.", op_name);
    DeviceAllocation alloc{ptr.device, ptr.alloc_id};
    auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
    TI_ERROR_IF(!base, "{} received a null dense-field pointer.", op_name);
    return static_cast<void *>(reinterpret_cast<std::uint8_t *>(base) +
                               ptr.offset);
  };
  return cuda::driver_stable_radix_sort_strided(
      raw_ptr(get_dense_field_device_ptr(keys),
              "CUDA Driver dense-field stable sort"),
      has_values ? raw_ptr(get_dense_field_device_ptr(values),
                           "CUDA Driver dense-field stable sort")
                 : nullptr,
      static_cast<int>(n), static_cast<cuda::CudaDriverSortKeyType>(key_type),
      has_values ? static_cast<int>(value_size / sizeof(std::uint32_t)) : 0, 0,
      keys_stride, 0, values_stride, has_values, nan_policy, nullptr,
      &primitive_workspace_arena_);
#else
  TI_NOT_IMPLEMENTED;
#endif
}

void Program::cuda_device_radix_sort_clear_workspace() {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    auto submission_guard = acquire_runtime_resource_submission_guard();
    synchronize();
    primitive_workspace_arena_.clear(PrimitiveWorkspaceBackend::cuda,
                                     PrimitiveWorkspaceFamily::ordering);
    primitive_workspace_arena_.clear(PrimitiveWorkspaceBackend::cuda,
                                     PrimitiveWorkspaceFamily::ordering_aux);
  }
#endif
}

std::size_t Program::cuda_device_radix_sort_workspace_bytes() const {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    const auto ordering = static_cast<std::size_t>(
        primitive_workspace_arena_
            .snapshot(PrimitiveWorkspaceBackend::cuda,
                      PrimitiveWorkspaceFamily::ordering)
            .reserved_bytes);
    const auto auxiliary = static_cast<std::size_t>(
        primitive_workspace_arena_
            .snapshot(PrimitiveWorkspaceBackend::cuda,
                      PrimitiveWorkspaceFamily::ordering_aux)
            .reserved_bytes);
    return ordering + auxiliary;
  }
#endif
  return 0;
}

bool Program::cuda_cub_radix_sort_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda &&
         cuda::cub_radix_sort_available();
#else
  return false;
#endif
}

std::size_t Program::cuda_cub_radix_sort_ndarray(Ndarray *keys,
                                                 Ndarray *values,
                                                 int key_type,
                                                 int value_type,
                                                 int mode,
                                                 int nan_policy) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB sort is only available on the CUDA backend.");
  TI_ERROR_IF(!keys, "CUDA CUB sort received null keys ndarray.");
  TI_ERROR_IF(keys->shape.size() != 1,
              "CUDA CUB sort currently expects a 1D ndarray.");
  const bool has_values = values != nullptr;
  if (has_values) {
    TI_ERROR_IF(values->shape.size() != 1,
                "CUDA CUB sort values must be a 1D ndarray.");
    TI_ERROR_IF(values->get_nelement() != keys->get_nelement(),
                "CUDA CUB sort keys and values must have the same length.");
  }
#ifdef TI_WITH_CUDA
  std::size_t expected_key_size = 0;
  const std::size_t expected_value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(mode < 0 || mode > 1,
              "CUDA CUB sort received an unsupported sort mode.");
  TI_ERROR_IF(nan_policy < 0 || nan_policy > 1,
              "CUDA CUB sort received an unsupported NaN policy.");
  TI_ERROR_IF(has_values && expected_value_size == 0,
              "CUDA CUB sort received an unsupported value type.");
  const std::size_t actual_value_size =
      has_values ? values->get_element_size() : expected_value_size;
  const auto cub_key_type = static_cast<cuda::CubSortKeyType>(key_type);
  const auto cub_value_type = static_cast<cuda::CubSortValueType>(value_type);
  const auto cub_mode = static_cast<cuda::CubSortMode>(mode);
  const auto cub_nan_policy =
      static_cast<cuda::CubSortNanPolicy>(nan_policy);
  switch (cub_key_type) {
    case cuda::CubSortKeyType::u32:
    case cuda::CubSortKeyType::i32:
    case cuda::CubSortKeyType::f32:
      expected_key_size = 4;
      break;
    case cuda::CubSortKeyType::u64:
    case cuda::CubSortKeyType::i64:
    case cuda::CubSortKeyType::f64:
      expected_key_size = 8;
      break;
  }
  TI_ERROR_IF(expected_key_size == 0,
              "CUDA CUB sort received an unsupported key type.");
  TI_ERROR_IF(keys->get_element_size() != expected_key_size,
              "CUDA CUB sort key dtype does not match the requested key type.");
  TI_ERROR_IF(has_values &&
                  (actual_value_size == 0 ||
                   actual_value_size % sizeof(uint32_t) != 0),
              "CUDA CUB sort value payload must be 4-byte aligned.");
  if (cub_mode == cuda::CubSortMode::split32) {
    TI_ERROR_IF(cub_key_type != cuda::CubSortKeyType::u64 &&
                    cub_key_type != cuda::CubSortKeyType::i64 &&
                    cub_key_type != cuda::CubSortKeyType::f64,
                "CUDA CUB split32 sort supports only u64/i64/f64 keys.");
  }
#endif
  auto key_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(keys));
  auto value_ptr = has_values
                       ? reinterpret_cast<void *>(
                             get_ndarray_data_ptr_as_int(values))
                       : nullptr;
#ifdef TI_WITH_CUDA
  void *stream = nullptr;
  return cuda::cub_radix_sort(
      key_ptr, value_ptr, static_cast<int>(keys->get_nelement()),
      cub_key_type, cub_value_type, cub_mode, cub_nan_policy, has_values,
      has_values ? static_cast<int>(actual_value_size / sizeof(uint32_t)) : 0,
      stream, &primitive_workspace_arena_);
#else
  TI_ERROR(
      "CUDA CUB sort requires building Taichi with TI_WITH_CUDA=ON and "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t Program::cuda_cub_radix_sort_dense_field(SNode *keys,
                                                     SNode *values,
                                                     int key_type,
                                                     int value_type,
                                                     std::size_t n,
                                                     int mode,
                                                     int nan_policy) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB dense field sort is only available on CUDA.");
  TI_ERROR_IF(!keys, "CUDA CUB dense field sort received null keys.");
#ifdef TI_WITH_CUDA
  const std::size_t expected_key_size = sort_key_type_size(key_type);
  const std::size_t expected_value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(expected_key_size == 0,
              "CUDA CUB dense field sort received an unsupported key type.");
  TI_ERROR_IF(mode < 0 || mode > 1,
              "CUDA CUB dense field sort received an unsupported sort mode.");
  TI_ERROR_IF(nan_policy < 0 || nan_policy > 1,
              "CUDA CUB dense field sort received an unsupported NaN policy.");
  const bool has_values = values != nullptr;
  TI_ERROR_IF(has_values && expected_value_size == 0,
              "CUDA CUB dense field sort received an unsupported value type.");
  const std::size_t key_stride = get_dense_field_stride(keys, expected_key_size);
  TI_ERROR_IF(key_stride != expected_key_size,
              "CUDA CUB dense field sort requires contiguous dense field keys.");
  std::size_t value_stride = 0;
  if (has_values) {
    value_stride = get_dense_field_stride(values, expected_value_size);
    TI_ERROR_IF(value_stride != expected_value_size,
                "CUDA CUB dense field sort requires contiguous dense field values.");
  }
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA CUB dense field sort currently supports at most INT_MAX items.");
  if (n <= 1) {
    return 0;
  }
  auto raw_ptr = [this](DevicePtr ptr, const char *op_name) {
    TI_ERROR_IF(!ptr.device, "{} received a null dense field device.",
                op_name);
    DeviceAllocation alloc{ptr.device, ptr.alloc_id};
    auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
    TI_ERROR_IF(!base, "{} received a null dense field data pointer.",
                op_name);
    return static_cast<void *>(reinterpret_cast<uint8_t *>(base) + ptr.offset);
  };
  void *key_ptr =
      raw_ptr(get_dense_field_device_ptr(keys), "CUDA CUB dense field sort");
  void *value_ptr =
      has_values
          ? raw_ptr(get_dense_field_device_ptr(values),
                    "CUDA CUB dense field sort")
          : nullptr;
  void *stream = nullptr;
  return cuda::cub_radix_sort(
      key_ptr, value_ptr, static_cast<int>(n),
      static_cast<cuda::CubSortKeyType>(key_type),
      static_cast<cuda::CubSortValueType>(value_type),
      static_cast<cuda::CubSortMode>(mode),
      static_cast<cuda::CubSortNanPolicy>(nan_policy), has_values,
      has_values ? static_cast<int>(value_stride / sizeof(uint32_t)) : 0,
      stream, &primitive_workspace_arena_);
#else
  TI_ERROR(
      "CUDA CUB dense field sort requires building Taichi with TI_WITH_CUDA=ON "
      "and TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

void Program::cuda_cub_radix_sort_clear_workspace() {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    clear_primitive_workspaces_for(PrimitiveWorkspaceBackend::cuda,
                                   PrimitiveWorkspaceFamily::ordering);
  }
#endif
}

std::size_t Program::cuda_cub_radix_sort_workspace_bytes() const {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    return cuda::cub_radix_sort_cached_bytes(
        const_cast<PrimitiveWorkspaceArena *>(&primitive_workspace_arena_));
  }
#endif
  return 0;
}

bool Program::cpu_stable_sort_available() const {
  return arch_is_cpu(compile_config().arch);
}

std::size_t Program::cpu_stable_sort_ndarray(Ndarray *keys,
                                             Ndarray *values,
                                             int key_type,
                                             int value_type,
                                             bool descending,
                                             int nan_policy) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native sort is only available on CPU backends.");
  TI_ERROR_IF(!keys, "CPU native sort received null keys ndarray.");
  TI_ERROR_IF(keys->shape.size() != 1,
              "CPU native sort currently expects a 1D ndarray.");
  TI_ERROR_IF(nan_policy < 0 || nan_policy > 1,
              "CPU native sort received an unsupported NaN policy.");
  const bool has_values = values != nullptr;
  if (has_values) {
    TI_ERROR_IF(values->shape.size() != 1,
                "CPU native sort values must be a 1D ndarray.");
    TI_ERROR_IF(values->get_nelement() != keys->get_nelement(),
                "CPU native sort keys and values must have the same length.");
    const std::size_t expected_value_size = primitive_value_type_size(value_type);
    TI_ERROR_IF(expected_value_size == 0,
                "CPU native sort received an unsupported value type.");
    TI_ERROR_IF(values->get_element_size() == 0 ||
                    values->get_element_size() % sizeof(uint32_t) != 0,
                "CPU native sort value payload must be 4-byte aligned.");
  }

  const std::size_t n = keys->get_nelement();
  void *value_ptr = has_values
                        ? reinterpret_cast<void *>(
                              get_ndarray_data_ptr_as_int(values))
                        : nullptr;
  auto key_ptr = get_ndarray_data_ptr_as_int(keys);
  TI_ERROR_IF(!key_ptr, "CPU native sort received a null key pointer.");
  TI_ERROR_IF(has_values && !value_ptr,
              "CPU native sort received a null value pointer.");
  const std::size_t expected_value_size =
      has_values ? primitive_value_type_size(value_type) : 0;
  const std::size_t value_item_bytes =
      has_values ? values->get_element_size() : 0;
  const bool raw_value_payload =
      has_values && value_item_bytes != expected_value_size;

  switch (key_type) {
    case 0:
      TI_ERROR_IF(keys->get_element_size() != sizeof(uint32_t),
                  "CPU native sort key dtype does not match ti.u32.");
      if (raw_value_payload) {
        return cpu_stable_sort_raw_values(reinterpret_cast<uint32_t *>(key_ptr),
                                          value_ptr, n, value_item_bytes,
                                          descending, nan_policy);
      }
      return cpu_stable_sort_value_dispatch(
          reinterpret_cast<uint32_t *>(key_ptr), value_ptr, n, value_type,
          descending, nan_policy);
    case 1:
      TI_ERROR_IF(keys->get_element_size() != sizeof(int32_t),
                  "CPU native sort key dtype does not match ti.i32.");
      if (raw_value_payload) {
        return cpu_stable_sort_raw_values(reinterpret_cast<int32_t *>(key_ptr),
                                          value_ptr, n, value_item_bytes,
                                          descending, nan_policy);
      }
      return cpu_stable_sort_value_dispatch(
          reinterpret_cast<int32_t *>(key_ptr), value_ptr, n, value_type,
          descending, nan_policy);
    case 2:
      TI_ERROR_IF(keys->get_element_size() != sizeof(float),
                  "CPU native sort key dtype does not match ti.f32.");
      if (raw_value_payload) {
        return cpu_stable_sort_raw_values(reinterpret_cast<float *>(key_ptr),
                                          value_ptr, n, value_item_bytes,
                                          descending, nan_policy);
      }
      return cpu_stable_sort_value_dispatch(
          reinterpret_cast<float *>(key_ptr), value_ptr, n, value_type,
          descending, nan_policy);
    case 3:
      TI_ERROR_IF(keys->get_element_size() != sizeof(uint64_t),
                  "CPU native sort key dtype does not match ti.u64.");
      if (raw_value_payload) {
        return cpu_stable_sort_raw_values(reinterpret_cast<uint64_t *>(key_ptr),
                                          value_ptr, n, value_item_bytes,
                                          descending, nan_policy);
      }
      return cpu_stable_sort_value_dispatch(
          reinterpret_cast<uint64_t *>(key_ptr), value_ptr, n, value_type,
          descending, nan_policy);
    case 4:
      TI_ERROR_IF(keys->get_element_size() != sizeof(int64_t),
                  "CPU native sort key dtype does not match ti.i64.");
      if (raw_value_payload) {
        return cpu_stable_sort_raw_values(reinterpret_cast<int64_t *>(key_ptr),
                                          value_ptr, n, value_item_bytes,
                                          descending, nan_policy);
      }
      return cpu_stable_sort_value_dispatch(
          reinterpret_cast<int64_t *>(key_ptr), value_ptr, n, value_type,
          descending, nan_policy);
    case 5:
      TI_ERROR_IF(keys->get_element_size() != sizeof(double),
                  "CPU native sort key dtype does not match ti.f64.");
      if (raw_value_payload) {
        return cpu_stable_sort_raw_values(reinterpret_cast<double *>(key_ptr),
                                          value_ptr, n, value_item_bytes,
                                          descending, nan_policy);
      }
      return cpu_stable_sort_value_dispatch(
          reinterpret_cast<double *>(key_ptr), value_ptr, n, value_type,
          descending, nan_policy);
    default:
      TI_ERROR("CPU native sort received an unsupported key type.");
  }
}

std::size_t Program::cpu_stable_sort_dense_field(SNode *keys,
                                                 SNode *values,
                                                 int key_type,
                                                 int value_type,
                                                 std::size_t n,
                                                 bool descending,
                                                 int nan_policy) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU dense field sort is only available on CPU backends.");
  TI_ERROR_IF(!keys, "CPU dense field sort received null keys.");
  TI_ERROR_IF(nan_policy < 0 || nan_policy > 1,
              "CPU dense field sort received an unsupported NaN policy.");
  const std::size_t key_size = sort_key_type_size(key_type);
  TI_ERROR_IF(key_size == 0,
              "CPU dense field sort received an unsupported key type.");
  const bool has_values = values != nullptr;
  const std::size_t expected_value_size =
      has_values ? primitive_value_type_size(value_type) : 0;
  TI_ERROR_IF(has_values && expected_value_size == 0,
              "CPU dense field sort received an unsupported value type.");
  if (n <= 1) {
    return 0;
  }
  std::size_t key_stride = 0;
  auto *key_ptr_bytes = map_cpu_dense_field(
      this, keys, key_size == sizeof(uint64_t) ? 3 : 0, n,
      "CPU dense field sort", &key_stride);
  TI_ERROR_IF(key_stride != key_size,
              "CPU dense field sort requires contiguous dense field keys.");
  void *value_ptr = nullptr;
  std::size_t value_stride = 0;
  if (has_values) {
    auto *value_ptr_bytes = map_cpu_dense_field(
        this, values, value_type, n, "CPU dense field sort", &value_stride);
    TI_ERROR_IF(value_stride != expected_value_size,
                "CPU dense field sort requires contiguous dense field values.");
    value_ptr = value_ptr_bytes;
  }
  switch (key_type) {
    case 0:
      return cpu_stable_sort_value_dispatch(
          reinterpret_cast<uint32_t *>(key_ptr_bytes), value_ptr, n,
          value_type, descending, nan_policy);
    case 1:
      return cpu_stable_sort_value_dispatch(
          reinterpret_cast<int32_t *>(key_ptr_bytes), value_ptr, n, value_type,
          descending, nan_policy);
    case 2:
      return cpu_stable_sort_value_dispatch(
          reinterpret_cast<float *>(key_ptr_bytes), value_ptr, n, value_type,
          descending, nan_policy);
    case 3:
      return cpu_stable_sort_value_dispatch(
          reinterpret_cast<uint64_t *>(key_ptr_bytes), value_ptr, n,
          value_type, descending, nan_policy);
    case 4:
      return cpu_stable_sort_value_dispatch(
          reinterpret_cast<int64_t *>(key_ptr_bytes), value_ptr, n, value_type,
          descending, nan_policy);
    case 5:
      return cpu_stable_sort_value_dispatch(
          reinterpret_cast<double *>(key_ptr_bytes), value_ptr, n, value_type,
          descending, nan_policy);
    default:
      TI_ERROR("CPU dense field sort received an unsupported key type.");
  }
}

bool Program::cuda_device_scan_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda &&
         cuda::driver_hierarchical_available();
#else
  return false;
#endif
}

std::size_t Program::cuda_device_inclusive_scan_ndarray(Ndarray *data,
                                                        int value_type) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA Driver scan is only available on CUDA.");
  TI_ERROR_IF(!data, "CUDA Driver scan received a null ndarray.");
  TI_ERROR_IF(data->shape.size() != 1,
              "CUDA Driver scan currently expects a 1D ndarray.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0 || data->get_element_size() != value_size,
              "CUDA Driver scan dtype does not match its value type.");
  TI_ERROR_IF(data->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA Driver scan currently supports at most INT_MAX items.");
  if (data->get_nelement() <= 1) {
    return 0;
  }
#ifdef TI_WITH_CUDA
  auto *data_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(data));
  return cuda::driver_inclusive_scan_strided(
      data_ptr, static_cast<int>(data->get_nelement()),
      static_cast<cuda::CudaTransformValueType>(value_type), 0, value_size,
      false, nullptr, &primitive_workspace_arena_);
#else
  TI_NOT_IMPLEMENTED;
#endif
}

std::size_t Program::cuda_device_inclusive_reverse_scan_ndarray(
    Ndarray *data,
    int value_type) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA Driver reverse scan is only available on CUDA.");
  TI_ERROR_IF(!data, "CUDA Driver reverse scan received a null ndarray.");
  TI_ERROR_IF(data->shape.size() != 1,
              "CUDA Driver reverse scan currently expects a 1D ndarray.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0 || data->get_element_size() != value_size,
              "CUDA Driver reverse scan dtype does not match its value type.");
  TI_ERROR_IF(data->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA Driver reverse scan supports at most INT_MAX items.");
  if (data->get_nelement() <= 1) {
    return 0;
  }
#ifdef TI_WITH_CUDA
  auto *data_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(data));
  return cuda::driver_inclusive_scan_strided(
      data_ptr, static_cast<int>(data->get_nelement()),
      static_cast<cuda::CudaTransformValueType>(value_type), 0, value_size,
      true, nullptr, &primitive_workspace_arena_);
#else
  TI_NOT_IMPLEMENTED;
#endif
}

std::size_t Program::cuda_device_inclusive_scan_member_ndarray(
    Ndarray *data,
    int value_type,
    std::size_t offset,
    std::size_t stride) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA Driver strided scan is only available on CUDA.");
  check_scan_member_request("CUDA Driver", data, value_type, offset, stride);
  TI_ERROR_IF(data->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA Driver strided scan supports at most INT_MAX items.");
  if (data->get_nelement() <= 1) {
    return 0;
  }
#ifdef TI_WITH_CUDA
  auto *data_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(data));
  return cuda::driver_inclusive_scan_strided(
      data_ptr, static_cast<int>(data->get_nelement()),
      static_cast<cuda::CudaTransformValueType>(value_type), offset, stride,
      false, nullptr, &primitive_workspace_arena_);
#else
  TI_NOT_IMPLEMENTED;
#endif
}

std::size_t Program::cuda_device_inclusive_reverse_scan_member_ndarray(
    Ndarray *data,
    int value_type,
    std::size_t offset,
    std::size_t stride) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA Driver reverse strided scan is only available on CUDA.");
  check_scan_member_request("CUDA Driver reverse", data, value_type, offset,
                            stride);
  TI_ERROR_IF(data->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA Driver reverse strided scan supports at most INT_MAX "
              "items.");
  if (data->get_nelement() <= 1) {
    return 0;
  }
#ifdef TI_WITH_CUDA
  auto *data_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(data));
  return cuda::driver_inclusive_scan_strided(
      data_ptr, static_cast<int>(data->get_nelement()),
      static_cast<cuda::CudaTransformValueType>(value_type), offset, stride,
      true, nullptr, &primitive_workspace_arena_);
#else
  TI_NOT_IMPLEMENTED;
#endif
}

std::size_t Program::cuda_device_inclusive_scan_dense_field(
    SNode *data,
    int value_type,
    std::size_t n) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA Driver dense field scan is only available on CUDA.");
  TI_ERROR_IF(!data, "CUDA Driver scan received a null dense field.");
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA Driver dense field scan supports at most INT_MAX items.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CUDA Driver dense field scan received an unsupported type.");
  const std::size_t stride = get_dense_field_stride(data, value_size);
  TI_ERROR_IF(stride < value_size,
              "CUDA Driver dense field scan received an invalid stride.");
  if (n <= 1) {
    return 0;
  }
#ifdef TI_WITH_CUDA
  DevicePtr device_ptr = get_dense_field_device_ptr(data);
  DeviceAllocation allocation{device_ptr.device, device_ptr.alloc_id};
  auto *base = program_impl_->get_device_alloc_info_ptr(allocation);
  TI_ERROR_IF(!base,
              "CUDA Driver dense field scan received a null data pointer.");
  auto *data_ptr = reinterpret_cast<std::uint8_t *>(base) + device_ptr.offset;
  return cuda::driver_inclusive_scan_strided(
      data_ptr, static_cast<int>(n),
      static_cast<cuda::CudaTransformValueType>(value_type), 0, stride, false,
      nullptr, &primitive_workspace_arena_);
#else
  TI_NOT_IMPLEMENTED;
#endif
}

std::size_t Program::cuda_device_inclusive_reverse_scan_dense_field(
    SNode *data,
    int value_type,
    std::size_t n) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA Driver reverse dense field scan is only available on "
              "CUDA.");
  TI_ERROR_IF(!data,
              "CUDA Driver reverse scan received a null dense field.");
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA Driver reverse dense field scan supports at most INT_MAX "
              "items.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CUDA Driver reverse dense field scan received an unsupported "
              "type.");
  const std::size_t stride = get_dense_field_stride(data, value_size);
  TI_ERROR_IF(stride < value_size,
              "CUDA Driver reverse dense field scan received an invalid "
              "stride.");
  if (n <= 1) {
    return 0;
  }
#ifdef TI_WITH_CUDA
  DevicePtr device_ptr = get_dense_field_device_ptr(data);
  DeviceAllocation allocation{device_ptr.device, device_ptr.alloc_id};
  auto *base = program_impl_->get_device_alloc_info_ptr(allocation);
  TI_ERROR_IF(!base,
              "CUDA Driver reverse dense field scan received a null data "
              "pointer.");
  auto *data_ptr = reinterpret_cast<std::uint8_t *>(base) + device_ptr.offset;
  return cuda::driver_inclusive_scan_strided(
      data_ptr, static_cast<int>(n),
      static_cast<cuda::CudaTransformValueType>(value_type), 0, stride, true,
      nullptr, &primitive_workspace_arena_);
#else
  TI_NOT_IMPLEMENTED;
#endif
}

std::size_t Program::cuda_device_inclusive_scan_dense_field_packed(
    SNode *data,
    int value_type,
    std::size_t n,
    int lane_count) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA Driver packed dense field scan is only available on "
              "CUDA.");
  TI_ERROR_IF(!data,
              "CUDA Driver packed scan received a null dense field.");
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA Driver packed scan supports at most INT_MAX items.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  const std::size_t item_bytes = dense_field_packed_bytes(
      value_type, 1, lane_count, "CUDA Driver packed dense field scan");
  check_dense_field_packed_stride(this, data, value_type, lane_count,
                                  "CUDA Driver packed dense field scan");
  if (n <= 1) {
    return 0;
  }
#ifdef TI_WITH_CUDA
  DevicePtr device_ptr = get_dense_field_device_ptr(data);
  DeviceAllocation allocation{device_ptr.device, device_ptr.alloc_id};
  auto *base = program_impl_->get_device_alloc_info_ptr(allocation);
  TI_ERROR_IF(!base,
              "CUDA Driver packed scan received a null data pointer.");
  auto *data_ptr = reinterpret_cast<std::uint8_t *>(base) + device_ptr.offset;
  std::size_t workspace_bytes = 0;
  for (int lane = 0; lane < lane_count; ++lane) {
    workspace_bytes = std::max(
        workspace_bytes,
        cuda::driver_inclusive_scan_strided(
            data_ptr, static_cast<int>(n),
            static_cast<cuda::CudaTransformValueType>(value_type),
            static_cast<std::size_t>(lane) * value_size, item_bytes, false,
            nullptr, &primitive_workspace_arena_));
  }
  return workspace_bytes;
#else
  TI_NOT_IMPLEMENTED;
#endif
}

std::size_t Program::cuda_device_inclusive_reverse_scan_dense_field_packed(
    SNode *data,
    int value_type,
    std::size_t n,
    int lane_count) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA Driver reverse packed dense field scan is only available "
              "on CUDA.");
  TI_ERROR_IF(!data,
              "CUDA Driver reverse packed scan received a null field.");
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA Driver reverse packed scan supports at most INT_MAX "
              "items.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  const std::size_t item_bytes = dense_field_packed_bytes(
      value_type, 1, lane_count,
      "CUDA Driver reverse packed dense field scan");
  check_dense_field_packed_stride(
      this, data, value_type, lane_count,
      "CUDA Driver reverse packed dense field scan");
  if (n <= 1) {
    return 0;
  }
#ifdef TI_WITH_CUDA
  DevicePtr device_ptr = get_dense_field_device_ptr(data);
  DeviceAllocation allocation{device_ptr.device, device_ptr.alloc_id};
  auto *base = program_impl_->get_device_alloc_info_ptr(allocation);
  TI_ERROR_IF(!base,
              "CUDA Driver reverse packed scan received a null data pointer.");
  auto *data_ptr = reinterpret_cast<std::uint8_t *>(base) + device_ptr.offset;
  std::size_t workspace_bytes = 0;
  for (int lane = 0; lane < lane_count; ++lane) {
    workspace_bytes = std::max(
        workspace_bytes,
        cuda::driver_inclusive_scan_strided(
            data_ptr, static_cast<int>(n),
            static_cast<cuda::CudaTransformValueType>(value_type),
            static_cast<std::size_t>(lane) * value_size, item_bytes, true,
            nullptr, &primitive_workspace_arena_));
  }
  return workspace_bytes;
#else
  TI_NOT_IMPLEMENTED;
#endif
}

void Program::cuda_device_scan_clear_workspace() {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    clear_primitive_workspaces_for(PrimitiveWorkspaceBackend::cuda,
                                   PrimitiveWorkspaceFamily::scan);
  }
#endif
}

std::size_t Program::cuda_device_scan_workspace_bytes() const {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    return static_cast<std::size_t>(
        primitive_workspace_arena_
            .snapshot(PrimitiveWorkspaceBackend::cuda,
                      PrimitiveWorkspaceFamily::scan)
            .reserved_bytes);
  }
#endif
  return 0;
}

bool Program::cuda_cub_scan_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda &&
         cuda::cub_inclusive_scan_available();
#else
  return false;
#endif
}

std::size_t Program::cuda_cub_inclusive_scan_ndarray(Ndarray *data,
                                                     int value_type) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB scan is only available on the CUDA backend.");
  TI_ERROR_IF(!data, "CUDA CUB scan received a null ndarray.");
  TI_ERROR_IF(data->shape.size() != 1,
              "CUDA CUB scan currently expects a 1D ndarray.");
#ifdef TI_WITH_CUDA
  const auto cub_value_type = static_cast<cuda::CubScanValueType>(value_type);
  std::size_t expected_value_size = 0;
  switch (cub_value_type) {
    case cuda::CubScanValueType::i32:
      expected_value_size = sizeof(int32_t);
      break;
    case cuda::CubScanValueType::f32:
      expected_value_size = sizeof(float);
      break;
    case cuda::CubScanValueType::u32:
      expected_value_size = sizeof(uint32_t);
      break;
    case cuda::CubScanValueType::u64:
      expected_value_size = sizeof(uint64_t);
      break;
    case cuda::CubScanValueType::i64:
      expected_value_size = sizeof(int64_t);
      break;
    case cuda::CubScanValueType::f64:
      expected_value_size = sizeof(double);
      break;
  }
  TI_ERROR_IF(expected_value_size == 0,
              "CUDA CUB scan received an unsupported value type.");
  TI_ERROR_IF(data->get_element_size() != expected_value_size,
              "CUDA CUB scan dtype does not match the requested value type.");
  auto data_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(data));
  void *stream = nullptr;
  return cuda::cub_inclusive_scan(
      data_ptr, static_cast<int>(data->get_nelement()), cub_value_type, stream,
      &primitive_workspace_arena_);
#else
  TI_ERROR(
      "CUDA CUB scan requires building Taichi with TI_WITH_CUDA=ON and "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t Program::cuda_cub_inclusive_reverse_scan_ndarray(Ndarray *data,
                                                             int value_type) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB reverse scan is only available on the CUDA backend.");
  TI_ERROR_IF(!data, "CUDA CUB reverse scan received a null ndarray.");
  TI_ERROR_IF(data->shape.size() != 1,
              "CUDA CUB reverse scan currently expects a 1D ndarray.");
#ifdef TI_WITH_CUDA
  const auto cub_value_type = static_cast<cuda::CubScanValueType>(value_type);
  std::size_t expected_value_size = 0;
  switch (cub_value_type) {
    case cuda::CubScanValueType::i32:
      expected_value_size = sizeof(int32_t);
      break;
    case cuda::CubScanValueType::f32:
      expected_value_size = sizeof(float);
      break;
    case cuda::CubScanValueType::u32:
      expected_value_size = sizeof(uint32_t);
      break;
    case cuda::CubScanValueType::u64:
      expected_value_size = sizeof(uint64_t);
      break;
    case cuda::CubScanValueType::i64:
      expected_value_size = sizeof(int64_t);
      break;
    case cuda::CubScanValueType::f64:
      expected_value_size = sizeof(double);
      break;
  }
  TI_ERROR_IF(expected_value_size == 0,
              "CUDA CUB reverse scan received an unsupported value type.");
  TI_ERROR_IF(data->get_element_size() != expected_value_size,
              "CUDA CUB reverse scan dtype does not match the requested "
              "value type.");
  auto data_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(data));
  void *stream = nullptr;
  return cuda::cub_inclusive_reverse_scan(
      data_ptr, static_cast<int>(data->get_nelement()), cub_value_type, stream,
      &primitive_workspace_arena_);
#else
  TI_ERROR(
      "CUDA CUB reverse scan requires building Taichi with TI_WITH_CUDA=ON "
      "and TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t Program::cuda_cub_inclusive_scan_member_ndarray(
    Ndarray *data,
    int value_type,
    std::size_t offset,
    std::size_t stride) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB strided scan is only available on the CUDA backend.");
  check_scan_member_request("CUDA CUB", data, value_type, offset, stride);
  TI_ERROR_IF(data->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA CUB strided scan currently supports at most INT_MAX "
              "items.");
  if (data->get_nelement() <= 1) {
    return 0;
  }
#ifdef TI_WITH_CUDA
  auto data_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(data));
  void *stream = nullptr;
  return cuda::cub_inclusive_scan_strided(
      data_ptr, static_cast<int>(data->get_nelement()),
      static_cast<cuda::CubScanValueType>(value_type), offset, stride, stream,
      &primitive_workspace_arena_);
#else
  TI_ERROR(
      "CUDA CUB strided scan requires building Taichi with TI_WITH_CUDA=ON "
      "and TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t Program::cuda_cub_inclusive_reverse_scan_member_ndarray(
    Ndarray *data,
    int value_type,
    std::size_t offset,
    std::size_t stride) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB reverse strided scan is only available on the CUDA "
              "backend.");
  check_scan_member_request("CUDA CUB reverse", data, value_type, offset,
                            stride);
  TI_ERROR_IF(data->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA CUB reverse strided scan currently supports at most "
              "INT_MAX items.");
  if (data->get_nelement() <= 1) {
    return 0;
  }
#ifdef TI_WITH_CUDA
  auto data_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(data));
  void *stream = nullptr;
  return cuda::cub_inclusive_reverse_scan_strided(
      data_ptr, static_cast<int>(data->get_nelement()),
      static_cast<cuda::CubScanValueType>(value_type), offset, stride, stream,
      &primitive_workspace_arena_);
#else
  TI_ERROR(
      "CUDA CUB reverse strided scan requires building Taichi with "
      "TI_WITH_CUDA=ON and TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t Program::cuda_cub_inclusive_scan_dense_field(SNode *data,
                                                         int value_type,
                                                         std::size_t n) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB dense field scan is only available on CUDA.");
  TI_ERROR_IF(!data, "CUDA CUB dense field scan received a null field.");
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA CUB dense field scan currently supports at most INT_MAX "
              "items.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CUDA CUB dense field scan received an unsupported value type.");
  const std::size_t stride = get_dense_field_stride(data, value_size);
  TI_ERROR_IF(stride < value_size,
              "CUDA CUB dense field scan received an invalid field stride.");
  if (n <= 1) {
    return 0;
  }
#ifdef TI_WITH_CUDA
  DevicePtr device_ptr = get_dense_field_device_ptr(data);
  DeviceAllocation alloc{device_ptr.device, device_ptr.alloc_id};
  auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
  TI_ERROR_IF(!base,
              "CUDA CUB dense field scan received a null data pointer.");
  void *data_ptr =
      static_cast<void *>(reinterpret_cast<uint8_t *>(base) + device_ptr.offset);
  void *stream = nullptr;
  return cuda::cub_inclusive_scan_strided(
      data_ptr, static_cast<int>(n),
      static_cast<cuda::CubScanValueType>(value_type), 0, stride, stream,
      &primitive_workspace_arena_);
#else
  TI_ERROR(
      "CUDA CUB dense field scan requires building Taichi with "
      "TI_WITH_CUDA=ON and TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t Program::cuda_cub_inclusive_reverse_scan_dense_field(
    SNode *data,
    int value_type,
    std::size_t n) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB dense field reverse scan is only available on CUDA.");
  TI_ERROR_IF(!data,
              "CUDA CUB dense field reverse scan received a null field.");
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA CUB dense field reverse scan currently supports at most "
              "INT_MAX items.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CUDA CUB dense field reverse scan received an unsupported "
              "value type.");
  const std::size_t stride = get_dense_field_stride(data, value_size);
  TI_ERROR_IF(stride < value_size,
              "CUDA CUB dense field reverse scan received an invalid field "
              "stride.");
  if (n <= 1) {
    return 0;
  }
#ifdef TI_WITH_CUDA
  DevicePtr device_ptr = get_dense_field_device_ptr(data);
  DeviceAllocation alloc{device_ptr.device, device_ptr.alloc_id};
  auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
  TI_ERROR_IF(!base,
              "CUDA CUB dense field reverse scan received a null data "
              "pointer.");
  void *data_ptr =
      static_cast<void *>(reinterpret_cast<uint8_t *>(base) + device_ptr.offset);
  void *stream = nullptr;
  return cuda::cub_inclusive_reverse_scan_strided(
      data_ptr, static_cast<int>(n),
      static_cast<cuda::CubScanValueType>(value_type), 0, stride, stream,
      &primitive_workspace_arena_);
#else
  TI_ERROR(
      "CUDA CUB dense field reverse scan requires building Taichi with "
      "TI_WITH_CUDA=ON and TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t Program::cuda_cub_inclusive_scan_dense_field_packed(
    SNode *data,
    int value_type,
    std::size_t n,
    int lane_count) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB packed dense field scan is only available on CUDA.");
  TI_ERROR_IF(!data, "CUDA CUB packed dense field scan received a null field.");
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA CUB packed dense field scan currently supports at most "
              "INT_MAX items.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  const std::size_t item_bytes = dense_field_packed_bytes(
      value_type, 1, lane_count, "CUDA CUB packed dense field scan");
  check_dense_field_packed_stride(this, data, value_type, lane_count,
                                  "CUDA CUB packed dense field scan");
  if (n <= 1) {
    return 0;
  }
#ifdef TI_WITH_CUDA
  DevicePtr device_ptr = get_dense_field_device_ptr(data);
  DeviceAllocation alloc{device_ptr.device, device_ptr.alloc_id};
  auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
  TI_ERROR_IF(!base,
              "CUDA CUB packed dense field scan received a null data pointer.");
  auto *data_ptr = reinterpret_cast<uint8_t *>(base) + device_ptr.offset;
  void *stream = nullptr;
  std::size_t temp_bytes = 0;
  for (int lane = 0; lane < lane_count; ++lane) {
    temp_bytes = std::max(
        temp_bytes,
        cuda::cub_inclusive_scan_strided(
            data_ptr, static_cast<int>(n),
            static_cast<cuda::CubScanValueType>(value_type),
            static_cast<std::size_t>(lane) * value_size, item_bytes, stream,
            &primitive_workspace_arena_));
  }
  return temp_bytes;
#else
  TI_ERROR(
      "CUDA CUB packed dense field scan requires building Taichi with "
      "TI_WITH_CUDA=ON and TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t Program::cuda_cub_inclusive_reverse_scan_dense_field_packed(
    SNode *data,
    int value_type,
    std::size_t n,
    int lane_count) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB packed dense field reverse scan is only available on "
              "CUDA.");
  TI_ERROR_IF(!data,
              "CUDA CUB packed dense field reverse scan received a null "
              "field.");
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA CUB packed dense field reverse scan currently supports at "
              "most INT_MAX items.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  const std::size_t item_bytes = dense_field_packed_bytes(
      value_type, 1, lane_count, "CUDA CUB packed dense field reverse scan");
  check_dense_field_packed_stride(this, data, value_type, lane_count,
                                  "CUDA CUB packed dense field reverse scan");
  if (n <= 1) {
    return 0;
  }
#ifdef TI_WITH_CUDA
  DevicePtr device_ptr = get_dense_field_device_ptr(data);
  DeviceAllocation alloc{device_ptr.device, device_ptr.alloc_id};
  auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
  TI_ERROR_IF(!base,
              "CUDA CUB packed dense field reverse scan received a null data "
              "pointer.");
  auto *data_ptr = reinterpret_cast<uint8_t *>(base) + device_ptr.offset;
  void *stream = nullptr;
  std::size_t temp_bytes = 0;
  for (int lane = 0; lane < lane_count; ++lane) {
    temp_bytes = std::max(
        temp_bytes,
        cuda::cub_inclusive_reverse_scan_strided(
            data_ptr, static_cast<int>(n),
            static_cast<cuda::CubScanValueType>(value_type),
            static_cast<std::size_t>(lane) * value_size, item_bytes, stream,
            &primitive_workspace_arena_));
  }
  return temp_bytes;
#else
  TI_ERROR(
      "CUDA CUB packed dense field reverse scan requires building Taichi with "
      "TI_WITH_CUDA=ON and TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

void Program::cuda_cub_scan_clear_workspace() {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    clear_primitive_workspaces_for(PrimitiveWorkspaceBackend::cuda,
                                   PrimitiveWorkspaceFamily::scan);
  }
#endif
}

std::size_t Program::cuda_cub_scan_workspace_bytes() const {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    return cuda::cub_inclusive_scan_cached_bytes(
        const_cast<PrimitiveWorkspaceArena *>(&primitive_workspace_arena_));
  }
#endif
  return 0;
}

bool Program::cuda_device_compact_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda &&
         cuda::driver_hierarchical_available();
#else
  return false;
#endif
}
std::size_t Program::cuda_device_compact_ndarray(Ndarray *values,
                                                 Ndarray *flags,
                                                 Ndarray *output,
                                                 Ndarray *count,
                                                 int value_type) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA Driver compact is only available on CUDA.");
  TI_ERROR_IF(!values || !flags || !output || !count,
              "CUDA Driver compact received a null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || flags->shape.size() != 1 ||
                  output->shape.size() != 1 || count->shape.size() != 1,
              "CUDA Driver compact expects 1D ndarrays.");
  TI_ERROR_IF(values->get_nelement() != flags->get_nelement() ||
                  values->get_nelement() > output->get_nelement(),
              "CUDA Driver compact expects matching values/flags and "
              "sufficient output capacity.");
  TI_ERROR_IF(count->get_nelement() < 1,
              "CUDA Driver compact count must contain at least one item.");
  const std::size_t expected_value_bytes =
      primitive_value_type_size(value_type);
  TI_ERROR_IF(expected_value_bytes == 0,
              "CUDA Driver compact received an unsupported value type.");
  const std::size_t item_bytes = values->get_element_size();
  TI_ERROR_IF(item_bytes == 0 || item_bytes % sizeof(std::uint32_t) != 0 ||
                  output->get_element_size() != item_bytes ||
                  flags->get_element_size() != sizeof(std::int32_t) ||
                  count->get_element_size() != sizeof(std::int32_t),
              "CUDA Driver compact received mismatched dtypes or a payload "
              "that is not 4-byte aligned.");
  TI_ERROR_IF(values->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA Driver compact supports at most INT_MAX items.");
  const int item_words = static_cast<int>(item_bytes / sizeof(std::uint32_t));
#ifdef TI_WITH_CUDA
  return cuda::driver_compact_strided(
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(flags)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(count)),
      static_cast<int>(values->get_nelement()), item_words, 0, item_bytes, 0,
      sizeof(std::int32_t), 0, item_bytes, 0, nullptr,
      &primitive_workspace_arena_);
#else
  TI_NOT_IMPLEMENTED;
#endif
}
std::size_t Program::cuda_device_compact_dense_field(SNode *values,
                                                     SNode *flags,
                                                     SNode *output,
                                                     SNode *count,
                                                     int value_type,
                                                     std::size_t n) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA Driver dense-field compact is only available on CUDA.");
  TI_ERROR_IF(!values || !flags || !output || !count,
              "CUDA Driver dense-field compact received a null field.");
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA Driver dense-field compact supports at most INT_MAX "
              "items.");
  const std::size_t item_bytes = primitive_value_type_size(value_type);
  TI_ERROR_IF(item_bytes == 0 || item_bytes % sizeof(std::uint32_t) != 0,
              "CUDA Driver dense-field compact received an unsupported value "
              "type.");
  const std::size_t values_stride = get_dense_field_stride(values, item_bytes);
  const std::size_t flags_stride =
      get_dense_field_stride(flags, sizeof(std::int32_t));
  const std::size_t output_stride = get_dense_field_stride(output, item_bytes);
  const std::size_t count_stride =
      get_dense_field_stride(count, sizeof(std::int32_t));
  TI_ERROR_IF(
      values_stride < item_bytes || flags_stride < sizeof(std::int32_t) ||
          output_stride < item_bytes || count_stride < sizeof(std::int32_t),
      "CUDA Driver dense-field compact received an invalid stride.");
#ifdef TI_WITH_CUDA
  auto raw_ptr = [this](DevicePtr ptr, const char *op_name) {
    TI_ERROR_IF(!ptr.device, "{} received a null dense-field device.", op_name);
    DeviceAllocation alloc{ptr.device, ptr.alloc_id};
    auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
    TI_ERROR_IF(!base, "{} received a null dense-field pointer.", op_name);
    return static_cast<void *>(reinterpret_cast<std::uint8_t *>(base) +
                               ptr.offset);
  };
  return cuda::driver_compact_strided(
      raw_ptr(get_dense_field_device_ptr(values),
              "CUDA Driver dense-field compact"),
      raw_ptr(get_dense_field_device_ptr(flags),
              "CUDA Driver dense-field compact"),
      raw_ptr(get_dense_field_device_ptr(output),
              "CUDA Driver dense-field compact"),
      raw_ptr(get_dense_field_device_ptr(count),
              "CUDA Driver dense-field compact"),
      static_cast<int>(n), static_cast<int>(item_bytes / sizeof(std::uint32_t)),
      0, values_stride, 0, flags_stride, 0, output_stride, 0, nullptr,
      &primitive_workspace_arena_);
#else
  TI_NOT_IMPLEMENTED;
#endif
}

void Program::cuda_device_compact_clear_workspace() {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    clear_primitive_workspaces_for(PrimitiveWorkspaceBackend::cuda,
                                   PrimitiveWorkspaceFamily::compact);
  }
#endif
}

std::size_t Program::cuda_device_compact_workspace_bytes() const {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    return static_cast<std::size_t>(
        primitive_workspace_arena_
            .snapshot(PrimitiveWorkspaceBackend::cuda,
                      PrimitiveWorkspaceFamily::compact)
            .reserved_bytes);
  }
#endif
  return 0;
}

bool Program::cuda_cub_select_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda && cuda::cub_select_available();
#else
  return false;
#endif
}

std::size_t Program::cuda_cub_select_ndarray(Ndarray *values,
                                             Ndarray *flags,
                                             Ndarray *output,
                                             Ndarray *count,
                                             int value_type) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB select is only available on the CUDA backend.");
  TI_ERROR_IF(!values || !flags || !output || !count,
              "CUDA CUB select received a null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || flags->shape.size() != 1 ||
                  output->shape.size() != 1 || count->shape.size() != 1,
              "CUDA CUB select currently expects 1D ndarrays.");
  TI_ERROR_IF(values->get_nelement() != flags->get_nelement() ||
                  values->get_nelement() > output->get_nelement(),
              "CUDA CUB select expects values/flags to have the same length "
              "and output to have at least that many elements.");
  TI_ERROR_IF(count->get_nelement() < 1,
              "CUDA CUB select count ndarray must have at least one element.");
  std::size_t expected_value_bytes = 0;
  switch (value_type) {
    case 0:
    case 1:
    case 2:
      expected_value_bytes = sizeof(uint32_t);
      break;
    case 3:
    case 4:
    case 5:
      expected_value_bytes = sizeof(uint64_t);
      break;
    default:
      TI_ERROR("CUDA CUB select received an unsupported value type.");
  }
  TI_ERROR_IF(expected_value_bytes == 0,
              "CUDA CUB select received an unsupported value type.");
  const std::size_t item_bytes = values->get_element_size();
  TI_ERROR_IF(item_bytes == 0 || item_bytes % sizeof(uint32_t) != 0 ||
                  output->get_element_size() != item_bytes ||
                  flags->get_element_size() != sizeof(int32_t) ||
                  count->get_element_size() != sizeof(int32_t),
              "CUDA CUB select received mismatched value/flag/count dtypes or "
              "a non-4-byte-aligned payload.");
  TI_ERROR_IF(values->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA CUB select currently supports at most INT_MAX items.");
  const int item_words = static_cast<int>(item_bytes / sizeof(uint32_t));
  TI_ERROR_IF(values->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max() /
                                           item_words),
              "CUDA CUB select word count exceeds INT_MAX.");
#ifdef TI_WITH_CUDA
  auto values_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values));
  auto flags_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(flags));
  auto output_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output));
  auto count_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(count));
  void *stream = nullptr;
  return cuda::cub_select_flagged(
      values_ptr, flags_ptr, output_ptr, count_ptr,
      static_cast<int>(values->get_nelement()),
      static_cast<cuda::CubSelectValueType>(value_type), item_words, stream,
      &primitive_workspace_arena_);
#else
  TI_ERROR(
      "CUDA CUB select requires building Taichi with TI_WITH_CUDA=ON and "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t Program::cuda_cub_select_dense_field(SNode *values,
                                                 SNode *flags,
                                                 SNode *output,
                                                 SNode *count,
                                                 int value_type,
                                                 std::size_t n) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB dense field select is only available on CUDA.");
  TI_ERROR_IF(!values || !flags || !output || !count,
              "CUDA CUB dense field select received a null field.");
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA CUB dense field select currently supports at most INT_MAX "
              "items.");
  const std::size_t item_bytes = primitive_value_type_size(value_type);
  TI_ERROR_IF(item_bytes == 0,
              "CUDA CUB dense field select received an unsupported value type.");
  TI_ERROR_IF(item_bytes % sizeof(uint32_t) != 0,
              "CUDA CUB dense field select requires 4-byte-aligned payloads.");
  const std::size_t values_stride = get_dense_field_stride(values, item_bytes);
  const std::size_t flags_stride =
      get_dense_field_stride(flags, sizeof(int32_t));
  const std::size_t output_stride = get_dense_field_stride(output, item_bytes);
  const std::size_t count_stride =
      get_dense_field_stride(count, sizeof(int32_t));
  TI_ERROR_IF(values_stride != item_bytes || output_stride != item_bytes ||
                  flags_stride != sizeof(int32_t) ||
                  count_stride < sizeof(int32_t),
              "CUDA CUB dense field select requires contiguous values, flags, "
              "and output fields.");
  const std::size_t item_words = item_bytes / sizeof(uint32_t);
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max() /
                                           item_words),
              "CUDA CUB dense field select word count exceeds INT_MAX.");
#ifdef TI_WITH_CUDA
  auto raw_ptr = [this](DevicePtr ptr, const char *op_name) {
    TI_ERROR_IF(!ptr.device, "{} received a null dense field device.",
                op_name);
    DeviceAllocation alloc{ptr.device, ptr.alloc_id};
    auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
    TI_ERROR_IF(!base, "{} received a null dense field data pointer.",
                op_name);
    return static_cast<void *>(reinterpret_cast<uint8_t *>(base) + ptr.offset);
  };
  void *values_ptr = raw_ptr(get_dense_field_device_ptr(values),
                             "CUDA CUB dense field select");
  void *flags_ptr =
      raw_ptr(get_dense_field_device_ptr(flags), "CUDA CUB dense field select");
  void *output_ptr = raw_ptr(get_dense_field_device_ptr(output),
                             "CUDA CUB dense field select");
  void *count_ptr =
      raw_ptr(get_dense_field_device_ptr(count), "CUDA CUB dense field select");
  void *stream = nullptr;
  return cuda::cub_select_flagged(
      values_ptr, flags_ptr, output_ptr, count_ptr, static_cast<int>(n),
      static_cast<cuda::CubSelectValueType>(value_type),
      static_cast<int>(item_words), stream, &primitive_workspace_arena_);
#else
  TI_ERROR(
      "CUDA CUB dense field select requires building Taichi with "
      "TI_WITH_CUDA=ON and TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t Program::cuda_cub_select_i32_ndarray(Ndarray *values,
                                                 Ndarray *flags,
                                                 Ndarray *output,
                                                 Ndarray *count) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  return cuda_cub_select_ndarray(values, flags, output, count, 0);
}

void Program::cuda_cub_select_clear_workspace() {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    clear_primitive_workspaces_for(PrimitiveWorkspaceBackend::cuda,
                                   PrimitiveWorkspaceFamily::compact);
  }
#endif
}

std::size_t Program::cuda_cub_select_workspace_bytes() const {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    return cuda::cub_select_cached_bytes(
        const_cast<PrimitiveWorkspaceArena *>(&primitive_workspace_arena_));
  }
#endif
  return 0;
}

bool Program::cuda_device_histogram_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda &&
         cuda::driver_hierarchical_available();
#else
  return false;
#endif
}

std::size_t Program::cuda_device_histogram_ndarray(Ndarray *values,
                                                   Ndarray *bins,
                                                   int value_type,
                                                   int bin_type) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA Driver histogram is only available on CUDA.");
  TI_ERROR_IF(!values || !bins,
              "CUDA Driver histogram received a null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || bins->shape.size() != 1,
              "CUDA Driver histogram currently expects 1D ndarrays.");
  TI_ERROR_IF(value_type != 0 && value_type != 2,
              "CUDA Driver histogram supports only i32/u32 bin ids.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  const std::size_t bin_size = histogram_bin_type_size(bin_type);
  TI_ERROR_IF(bin_size == 0,
              "CUDA Driver histogram supports only i32/i64 counters.");
  TI_ERROR_IF(values->get_element_size() != value_size ||
                  bins->get_element_size() != bin_size,
              "CUDA Driver histogram received mismatched dtypes.");
  TI_ERROR_IF(bins->get_nelement() == 0,
              "CUDA Driver histogram expects at least one bin.");
  TI_ERROR_IF(values->get_nelement() >
                      static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
                  bins->get_nelement() >
                      static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA Driver histogram supports at most INT_MAX items/bins.");
#ifdef TI_WITH_CUDA
  auto *values_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values));
  auto *bins_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(bins));
  return cuda::driver_histogram_strided(
      values_ptr, bins_ptr, static_cast<int>(values->get_nelement()),
      static_cast<int>(bins->get_nelement()),
      static_cast<cuda::CudaTransformValueType>(value_type),
      static_cast<cuda::CudaTransformValueType>(bin_type), 0, value_size, 0,
      bin_size, nullptr);
#else
  TI_NOT_IMPLEMENTED;
#endif
}

std::size_t Program::cuda_device_histogram_dense_field(
    SNode *values,
    SNode *bins,
    int value_type,
    int bin_type,
    std::size_t n,
    std::size_t num_bins) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA Driver dense field histogram is only available on CUDA.");
  TI_ERROR_IF(!values || !bins,
              "CUDA Driver dense field histogram received a null field.");
  TI_ERROR_IF(value_type != 0 && value_type != 2,
              "CUDA Driver histogram supports only i32/u32 bin ids.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  const std::size_t bin_size = histogram_bin_type_size(bin_type);
  TI_ERROR_IF(bin_size == 0,
              "CUDA Driver histogram supports only i32/i64 counters.");
  TI_ERROR_IF(num_bins == 0,
              "CUDA Driver histogram expects at least one bin.");
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
                  num_bins >
                      static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA Driver histogram supports at most INT_MAX items/bins.");
  const std::size_t value_stride =
      get_dense_field_stride(values, value_size);
  const std::size_t bin_stride = get_dense_field_stride(bins, bin_size);
  TI_ERROR_IF(value_stride < value_size || bin_stride < bin_size,
              "CUDA Driver histogram received an invalid field stride.");
#ifdef TI_WITH_CUDA
  auto raw_ptr = [this](DevicePtr ptr, const char *op_name) {
    TI_ERROR_IF(!ptr.device, "{} received a null dense field device.", op_name);
    DeviceAllocation allocation{ptr.device, ptr.alloc_id};
    auto *base = program_impl_->get_device_alloc_info_ptr(allocation);
    TI_ERROR_IF(!base, "{} received a null dense field data pointer.",
                op_name);
    return static_cast<void *>(reinterpret_cast<std::uint8_t *>(base) +
                               ptr.offset);
  };
  void *values_ptr = raw_ptr(get_dense_field_device_ptr(values),
                             "CUDA Driver dense field histogram");
  void *bins_ptr = raw_ptr(get_dense_field_device_ptr(bins),
                           "CUDA Driver dense field histogram");
  return cuda::driver_histogram_strided(
      values_ptr, bins_ptr, static_cast<int>(n), static_cast<int>(num_bins),
      static_cast<cuda::CudaTransformValueType>(value_type),
      static_cast<cuda::CudaTransformValueType>(bin_type), 0, value_stride, 0,
      bin_stride, nullptr);
#else
  TI_NOT_IMPLEMENTED;
#endif
}

void Program::cuda_device_histogram_clear_workspace() {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    clear_primitive_workspaces_for(PrimitiveWorkspaceBackend::cuda,
                                   PrimitiveWorkspaceFamily::histogram);
  }
#endif
}

std::size_t Program::cuda_device_histogram_workspace_bytes() const {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    return static_cast<std::size_t>(
        primitive_workspace_arena_
            .snapshot(PrimitiveWorkspaceBackend::cuda,
                      PrimitiveWorkspaceFamily::histogram)
            .reserved_bytes);
  }
#endif
  return 0;
}

bool Program::cuda_cub_histogram_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda && cuda::cub_histogram_available();
#else
  return false;
#endif
}

std::size_t Program::cuda_cub_histogram_i32_ndarray(Ndarray *values,
                                                    Ndarray *bins) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  return cuda_cub_histogram_ndarray(values, bins, 0, 0);
}

std::size_t Program::cuda_cub_histogram_ndarray(Ndarray *values,
                                                Ndarray *bins,
                                                int value_type,
                                                int bin_type) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB histogram is only available on CUDA.");
  TI_ERROR_IF(!values || !bins,
              "CUDA CUB histogram received a null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || bins->shape.size() != 1,
              "CUDA CUB histogram currently expects 1D ndarrays.");
  TI_ERROR_IF(value_type != 0 && value_type != 2,
              "CUDA CUB histogram currently supports only i32/u32 bin ids.");
  const std::size_t value_size = value_type == 2 ? sizeof(uint32_t)
                                                 : sizeof(int32_t);
  const std::size_t bin_size = histogram_bin_type_size(bin_type);
  TI_ERROR_IF(bin_size == 0,
              "CUDA CUB histogram currently supports only i32/i64 bins.");
  TI_ERROR_IF(values->get_element_size() != value_size ||
                  bins->get_element_size() != bin_size,
              "CUDA CUB histogram received mismatched value/bin dtypes.");
  TI_ERROR_IF(bins->get_nelement() == 0,
              "CUDA CUB histogram expects at least one bin.");
#ifdef TI_WITH_CUDA
  auto values_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values));
  auto bins_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(bins));
  void *stream = nullptr;
  return cuda::cub_histogram_even(
      values_ptr, bins_ptr, static_cast<int>(values->get_nelement()),
      static_cast<int>(bins->get_nelement()),
      static_cast<cuda::CubHistogramValueType>(value_type),
      static_cast<cuda::CubHistogramBinType>(bin_type),
      stream, &primitive_workspace_arena_);
#else
  TI_ERROR(
      "CUDA CUB histogram requires building Taichi with TI_WITH_CUDA=ON and "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t Program::cuda_cub_histogram_dense_field(SNode *values,
                                                    SNode *bins,
                                                    int value_type,
                                                    int bin_type,
                                                    std::size_t n,
                                                    std::size_t num_bins) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB dense field histogram is only available on CUDA.");
  TI_ERROR_IF(!values || !bins,
              "CUDA CUB dense field histogram received a null field.");
  TI_ERROR_IF(value_type != 0 && value_type != 2,
              "CUDA CUB dense field histogram currently supports only i32/u32 "
              "bin ids.");
  const std::size_t value_size = value_type == 2 ? sizeof(uint32_t)
                                                 : sizeof(int32_t);
  const std::size_t bin_size = histogram_bin_type_size(bin_type);
  TI_ERROR_IF(bin_size == 0,
              "CUDA CUB dense field histogram currently supports only i32/i64 "
              "bins.");
  TI_ERROR_IF(num_bins == 0,
              "CUDA CUB dense field histogram expects at least one bin.");
  const std::size_t value_stride = get_dense_field_stride(values, value_size);
  const std::size_t bin_stride = get_dense_field_stride(bins, bin_size);
  TI_ERROR_IF(value_stride != value_size || bin_stride != bin_size,
              "CUDA CUB dense field histogram requires contiguous dense field "
              "values and bins.");
#ifdef TI_WITH_CUDA
  auto raw_ptr = [this](DevicePtr ptr, const char *op_name) {
    TI_ERROR_IF(!ptr.device, "{} received a null dense field device.",
                op_name);
    DeviceAllocation alloc{ptr.device, ptr.alloc_id};
    auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
    TI_ERROR_IF(!base, "{} received a null dense field data pointer.",
                op_name);
    return static_cast<void *>(reinterpret_cast<uint8_t *>(base) + ptr.offset);
  };
  void *values_ptr = raw_ptr(get_dense_field_device_ptr(values),
                             "CUDA CUB dense field histogram");
  void *bins_ptr =
      raw_ptr(get_dense_field_device_ptr(bins), "CUDA CUB dense field histogram");
  void *stream = nullptr;
  return cuda::cub_histogram_even(
      values_ptr, bins_ptr, static_cast<int>(n), static_cast<int>(num_bins),
      static_cast<cuda::CubHistogramValueType>(value_type),
      static_cast<cuda::CubHistogramBinType>(bin_type), stream,
      &primitive_workspace_arena_);
#else
  TI_ERROR(
      "CUDA CUB dense field histogram requires building Taichi with "
      "TI_WITH_CUDA=ON and TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

void Program::cuda_cub_histogram_clear_workspace() {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    clear_primitive_workspaces_for(PrimitiveWorkspaceBackend::cuda,
                                   PrimitiveWorkspaceFamily::histogram);
  }
#endif
}

std::size_t Program::cuda_cub_histogram_workspace_bytes() const {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    return cuda::cub_histogram_cached_bytes(
        const_cast<PrimitiveWorkspaceArena *>(&primitive_workspace_arena_));
  }
#endif
  return 0;
}

bool Program::cuda_device_reduce_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda &&
         cuda::driver_hierarchical_available();
#else
  return false;
#endif
}

std::size_t Program::cuda_device_reduce_ndarray(Ndarray *values,
                                                Ndarray *output,
                                                int value_type,
                                                int op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA Driver reduce is only available on CUDA.");
  TI_ERROR_IF(!values || !output,
              "CUDA Driver reduce received a null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || output->shape.size() != 1,
              "CUDA Driver reduce currently expects 1D ndarrays.");
  TI_ERROR_IF(values->get_nelement() == 0 || output->get_nelement() == 0,
              "CUDA Driver reduce requires non-empty input/output.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0 || values->get_element_size() != value_size ||
                  output->get_element_size() != value_size,
              "CUDA Driver reduce received mismatched dtypes.");
  TI_ERROR_IF(op < 0 || op > 2,
              "CUDA Driver reduce received an unsupported operation.");
  TI_ERROR_IF(values->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA Driver reduce supports at most INT_MAX items.");
#ifdef TI_WITH_CUDA
  auto *values_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values));
  auto *output_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output));
  return cuda::driver_reduce_strided(
      values_ptr, output_ptr, static_cast<int>(values->get_nelement()),
      static_cast<cuda::CudaTransformValueType>(value_type), 0, value_size, 0,
      value_size, static_cast<cuda::CudaHierarchicalReduceOp>(op), nullptr,
      &primitive_workspace_arena_);
#else
  TI_NOT_IMPLEMENTED;
#endif
}

std::size_t Program::cuda_device_reduce_member_ndarray(
    Ndarray *values,
    Ndarray *output,
    int value_type,
    std::size_t offset,
    std::size_t stride,
    int op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA Driver strided reduce is only available on CUDA.");
  check_reduce_member_request("CUDA Driver", values, output, value_type,
                              offset, stride, op);
  TI_ERROR_IF(values->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA Driver strided reduce supports at most INT_MAX items.");
#ifdef TI_WITH_CUDA
  auto *values_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values));
  auto *output_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output));
  return cuda::driver_reduce_strided(
      values_ptr, output_ptr, static_cast<int>(values->get_nelement()),
      static_cast<cuda::CudaTransformValueType>(value_type), offset, stride, 0,
      output->get_element_size(),
      static_cast<cuda::CudaHierarchicalReduceOp>(op), nullptr,
      &primitive_workspace_arena_);
#else
  TI_NOT_IMPLEMENTED;
#endif
}

std::size_t Program::cuda_device_reduce_strided_ndarray(
    Ndarray *values,
    Ndarray *output,
    int value_type,
    std::size_t values_offset,
    std::size_t values_stride,
    std::size_t output_offset,
    std::size_t output_stride,
    int op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA Driver strided reduce is only available on CUDA.");
  check_reduce_strided_request("CUDA Driver", values, output, value_type,
                               values_offset, values_stride, output_offset,
                               output_stride, op);
  TI_ERROR_IF(values->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA Driver strided reduce supports at most INT_MAX items.");
#ifdef TI_WITH_CUDA
  auto *values_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values));
  auto *output_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output));
  return cuda::driver_reduce_strided(
      values_ptr, output_ptr, static_cast<int>(values->get_nelement()),
      static_cast<cuda::CudaTransformValueType>(value_type), values_offset,
      values_stride, output_offset, output_stride,
      static_cast<cuda::CudaHierarchicalReduceOp>(op), nullptr,
      &primitive_workspace_arena_);
#else
  TI_NOT_IMPLEMENTED;
#endif
}

std::size_t Program::cuda_device_reduce_dense_field(SNode *values,
                                                    SNode *output,
                                                    int value_type,
                                                    std::size_t n,
                                                    int op) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA Driver dense field reduce is only available on CUDA.");
  TI_ERROR_IF(!values || !output,
              "CUDA Driver dense field reduce received a null field.");
  TI_ERROR_IF(n == 0 ||
                  n > static_cast<std::size_t>(
                          std::numeric_limits<int>::max()),
              "CUDA Driver dense field reduce requires 1..INT_MAX items.");
  TI_ERROR_IF(op < 0 || op > 2,
              "CUDA Driver dense field reduce received an unsupported op.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CUDA Driver dense field reduce received an unsupported type.");
  const std::size_t values_stride =
      get_dense_field_stride(values, value_size);
  const std::size_t output_stride =
      get_dense_field_stride(output, value_size);
  TI_ERROR_IF(values_stride < value_size || output_stride < value_size,
              "CUDA Driver dense field reduce received an invalid stride.");
#ifdef TI_WITH_CUDA
  auto raw_ptr = [this](DevicePtr ptr, const char *op_name) {
    TI_ERROR_IF(!ptr.device, "{} received a null dense field device.", op_name);
    DeviceAllocation allocation{ptr.device, ptr.alloc_id};
    auto *base = program_impl_->get_device_alloc_info_ptr(allocation);
    TI_ERROR_IF(!base, "{} received a null dense field data pointer.",
                op_name);
    return static_cast<void *>(reinterpret_cast<std::uint8_t *>(base) +
                               ptr.offset);
  };
  void *values_ptr = raw_ptr(get_dense_field_device_ptr(values),
                             "CUDA Driver dense field reduce");
  void *output_ptr = raw_ptr(get_dense_field_device_ptr(output),
                             "CUDA Driver dense field reduce");
  return cuda::driver_reduce_strided(
      values_ptr, output_ptr, static_cast<int>(n),
      static_cast<cuda::CudaTransformValueType>(value_type), 0, values_stride,
      0, output_stride, static_cast<cuda::CudaHierarchicalReduceOp>(op),
      nullptr, &primitive_workspace_arena_);
#else
  TI_NOT_IMPLEMENTED;
#endif
}

std::size_t Program::cuda_device_reduce_dense_field_packed(
    SNode *values,
    SNode *output,
    int value_type,
    std::size_t n,
    int lane_count,
    int op) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA Driver packed dense field reduce is only available on "
              "CUDA.");
  TI_ERROR_IF(!values || !output,
              "CUDA Driver packed reduce received a null field.");
  TI_ERROR_IF(n == 0 ||
                  n > static_cast<std::size_t>(
                          std::numeric_limits<int>::max()),
              "CUDA Driver packed reduce requires 1..INT_MAX items.");
  TI_ERROR_IF(op < 0 || op > 2,
              "CUDA Driver packed reduce received an unsupported op.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  const std::size_t item_bytes = dense_field_packed_bytes(
      value_type, 1, lane_count, "CUDA Driver packed dense field reduce");
  check_dense_field_packed_stride(this, values, value_type, lane_count,
                                  "CUDA Driver packed dense field reduce");
  check_dense_field_packed_stride(this, output, value_type, lane_count,
                                  "CUDA Driver packed dense field reduce");
#ifdef TI_WITH_CUDA
  auto raw_ptr = [this](DevicePtr ptr, const char *op_name) {
    TI_ERROR_IF(!ptr.device, "{} received a null dense field device.", op_name);
    DeviceAllocation allocation{ptr.device, ptr.alloc_id};
    auto *base = program_impl_->get_device_alloc_info_ptr(allocation);
    TI_ERROR_IF(!base, "{} received a null dense field data pointer.",
                op_name);
    return reinterpret_cast<std::uint8_t *>(base) + ptr.offset;
  };
  auto *values_ptr = raw_ptr(get_dense_field_device_ptr(values),
                             "CUDA Driver packed dense field reduce");
  auto *output_ptr = raw_ptr(get_dense_field_device_ptr(output),
                             "CUDA Driver packed dense field reduce");
  std::size_t workspace_bytes = 0;
  for (int lane = 0; lane < lane_count; ++lane) {
    const std::size_t lane_offset =
        static_cast<std::size_t>(lane) * value_size;
    workspace_bytes = std::max(
        workspace_bytes,
        cuda::driver_reduce_strided(
            values_ptr, output_ptr, static_cast<int>(n),
            static_cast<cuda::CudaTransformValueType>(value_type), lane_offset,
            item_bytes, lane_offset, item_bytes,
            static_cast<cuda::CudaHierarchicalReduceOp>(op), nullptr,
            &primitive_workspace_arena_));
  }
  return workspace_bytes;
#else
  TI_NOT_IMPLEMENTED;
#endif
}

void Program::cuda_device_reduce_clear_workspace() {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    clear_primitive_workspaces_for(PrimitiveWorkspaceBackend::cuda,
                                   PrimitiveWorkspaceFamily::reduce);
  }
#endif
}

std::size_t Program::cuda_device_reduce_workspace_bytes() const {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    return static_cast<std::size_t>(
        primitive_workspace_arena_
            .snapshot(PrimitiveWorkspaceBackend::cuda,
                      PrimitiveWorkspaceFamily::reduce)
            .reserved_bytes);
  }
#endif
  return 0;
}

bool Program::cuda_cub_reduce_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda && cuda::cub_reduce_available();
#else
  return false;
#endif
}

std::size_t Program::cuda_cub_reduce_ndarray(Ndarray *values,
                                             Ndarray *output,
                                             int value_type,
                                             int op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB reduce is only available on CUDA.");
  TI_ERROR_IF(!values || !output,
              "CUDA CUB reduce received a null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || output->shape.size() != 1,
              "CUDA CUB reduce currently expects 1D ndarrays.");
  TI_ERROR_IF(values->get_nelement() == 0,
              "CUDA CUB reduce expects at least one input item.");
  TI_ERROR_IF(output->get_nelement() < 1,
              "CUDA CUB reduce output ndarray must have at least one item.");
  TI_ERROR_IF(values->get_element_size() != output->get_element_size(),
              "CUDA CUB reduce expects matching input/output element sizes.");
  const std::size_t expected_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(expected_size == 0,
              "CUDA CUB reduce received an unsupported value type.");
  TI_ERROR_IF(values->get_element_size() != expected_size,
              "CUDA CUB reduce dtype does not match value type.");
  TI_ERROR_IF(op < 0 || op > 2,
              "CUDA CUB reduce received an unsupported op.");
#ifdef TI_WITH_CUDA
  auto values_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values));
  auto output_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output));
  void *stream = nullptr;
  return cuda::cub_reduce(values_ptr, output_ptr,
                          static_cast<int>(values->get_nelement()),
                          static_cast<cuda::CubReduceValueType>(value_type),
                          static_cast<cuda::CubReduceOp>(op), stream,
                          &primitive_workspace_arena_);
#else
  TI_ERROR(
      "CUDA CUB reduce requires building Taichi with TI_WITH_CUDA=ON and "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t Program::cuda_cub_reduce_member_ndarray(Ndarray *values,
                                                    Ndarray *output,
                                                    int value_type,
                                                    std::size_t offset,
                                                    std::size_t stride,
                                                    int op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB strided reduce is only available on CUDA.");
  check_reduce_member_request("CUDA CUB", values, output, value_type, offset,
                              stride, op);
  TI_ERROR_IF(values->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA CUB strided reduce currently supports at most INT_MAX "
              "items.");
#ifdef TI_WITH_CUDA
  auto values_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values));
  auto output_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output));
  void *stream = nullptr;
  return cuda::cub_reduce_strided(
      values_ptr, output_ptr, static_cast<int>(values->get_nelement()),
      static_cast<cuda::CubReduceValueType>(value_type), offset, stride,
      static_cast<cuda::CubReduceOp>(op), stream,
      &primitive_workspace_arena_);
#else
  TI_ERROR(
      "CUDA CUB strided reduce requires building Taichi with TI_WITH_CUDA=ON "
      "and TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t Program::cuda_cub_reduce_strided_ndarray(
    Ndarray *values,
    Ndarray *output,
    int value_type,
    std::size_t values_offset,
    std::size_t values_stride,
    std::size_t output_offset,
    std::size_t output_stride,
    int op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB strided reduce is only available on CUDA.");
  check_reduce_strided_request("CUDA CUB", values, output, value_type,
                               values_offset, values_stride, output_offset,
                               output_stride, op);
  TI_ERROR_IF(values->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA CUB strided reduce currently supports at most INT_MAX "
              "items.");
#ifdef TI_WITH_CUDA
  auto values_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values));
  auto output_ptr = reinterpret_cast<void *>(
      get_ndarray_data_ptr_as_int(output) + output_offset);
  void *stream = nullptr;
  return cuda::cub_reduce_strided(
      values_ptr, output_ptr, static_cast<int>(values->get_nelement()),
      static_cast<cuda::CubReduceValueType>(value_type), values_offset,
      values_stride, static_cast<cuda::CubReduceOp>(op), stream,
      &primitive_workspace_arena_);
#else
  TI_ERROR(
      "CUDA CUB strided reduce requires building Taichi with TI_WITH_CUDA=ON "
      "and TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t Program::cuda_cub_reduce_dense_field(SNode *values,
                                                 SNode *output,
                                                 int value_type,
                                                 std::size_t n,
                                                 int op) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB dense field reduce is only available on CUDA.");
  TI_ERROR_IF(!values || !output,
              "CUDA CUB dense field reduce received a null field.");
  TI_ERROR_IF(n == 0,
              "CUDA CUB dense field reduce expects at least one input item.");
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA CUB dense field reduce currently supports at most INT_MAX "
              "items.");
  TI_ERROR_IF(op < 0 || op > 2,
              "CUDA CUB dense field reduce received an unsupported op.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CUDA CUB dense field reduce received an unsupported value type.");
  const std::size_t stride = get_dense_field_stride(values, value_size);
  TI_ERROR_IF(stride < value_size,
              "CUDA CUB dense field reduce received an invalid field stride.");
#ifdef TI_WITH_CUDA
  auto raw_ptr = [this](DevicePtr ptr, const char *op_name) {
    TI_ERROR_IF(!ptr.device, "{} received a null dense field device.",
                op_name);
    DeviceAllocation alloc{ptr.device, ptr.alloc_id};
    auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
    TI_ERROR_IF(!base, "{} received a null dense field data pointer.",
                op_name);
    return static_cast<void *>(reinterpret_cast<uint8_t *>(base) + ptr.offset);
  };
  void *values_ptr =
      raw_ptr(get_dense_field_device_ptr(values), "CUDA CUB dense field reduce");
  void *output_ptr =
      raw_ptr(get_dense_field_device_ptr(output), "CUDA CUB dense field reduce");
  void *stream = nullptr;
  return cuda::cub_reduce_strided(
      values_ptr, output_ptr, static_cast<int>(n),
      static_cast<cuda::CubReduceValueType>(value_type), 0, stride,
      static_cast<cuda::CubReduceOp>(op), stream,
      &primitive_workspace_arena_);
#else
  TI_ERROR(
      "CUDA CUB dense field reduce requires building Taichi with "
      "TI_WITH_CUDA=ON and TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t Program::cuda_cub_reduce_dense_field_packed(SNode *values,
                                                        SNode *output,
                                                        int value_type,
                                                        std::size_t n,
                                                        int lane_count,
                                                        int op) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB packed dense field reduce is only available on CUDA.");
  TI_ERROR_IF(!values || !output,
              "CUDA CUB packed dense field reduce received a null field.");
  TI_ERROR_IF(n == 0,
              "CUDA CUB packed dense field reduce expects at least one input "
              "item.");
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA CUB packed dense field reduce currently supports at most "
              "INT_MAX items.");
  TI_ERROR_IF(op < 0 || op > 2,
              "CUDA CUB packed dense field reduce received an unsupported op.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CUDA CUB packed dense field reduce received an unsupported "
              "value type.");
  const std::size_t item_bytes =
      dense_field_packed_bytes(value_type, 1, lane_count,
                               "CUDA CUB packed dense field reduce");
  check_dense_field_packed_stride(this, values, value_type, lane_count,
                                  "CUDA CUB packed dense field reduce");
  check_dense_field_packed_stride(this, output, value_type, lane_count,
                                  "CUDA CUB packed dense field reduce");
#ifdef TI_WITH_CUDA
  auto raw_ptr = [this](DevicePtr ptr, const char *op_name) {
    TI_ERROR_IF(!ptr.device, "{} received a null dense field device.",
                op_name);
    DeviceAllocation alloc{ptr.device, ptr.alloc_id};
    auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
    TI_ERROR_IF(!base, "{} received a null dense field data pointer.",
                op_name);
    return reinterpret_cast<uint8_t *>(base) + ptr.offset;
  };
  auto *values_ptr = raw_ptr(get_dense_field_device_ptr(values),
                             "CUDA CUB packed dense field reduce");
  auto *output_ptr = raw_ptr(get_dense_field_device_ptr(output),
                             "CUDA CUB packed dense field reduce");
  void *stream = nullptr;
  std::size_t temp_bytes = 0;
  for (int lane = 0; lane < lane_count; ++lane) {
    const std::size_t lane_offset =
        static_cast<std::size_t>(lane) * value_size;
    temp_bytes = std::max(
        temp_bytes,
        cuda::cub_reduce_strided(
            values_ptr, output_ptr + lane_offset, static_cast<int>(n),
            static_cast<cuda::CubReduceValueType>(value_type), lane_offset,
            item_bytes, static_cast<cuda::CubReduceOp>(op), stream,
            &primitive_workspace_arena_));
  }
  return temp_bytes;
#else
  TI_ERROR(
      "CUDA CUB packed dense field reduce requires building Taichi with "
      "TI_WITH_CUDA=ON and TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

void Program::cuda_cub_reduce_clear_workspace() {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    clear_primitive_workspaces_for(PrimitiveWorkspaceBackend::cuda,
                                   PrimitiveWorkspaceFamily::reduce);
  }
#endif
}

std::size_t Program::cuda_cub_reduce_workspace_bytes() const {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    return cuda::cub_reduce_cached_bytes(
        const_cast<PrimitiveWorkspaceArena *>(&primitive_workspace_arena_));
  }
#endif
  return 0;
}

bool Program::cuda_device_check_count_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda &&
         cuda::driver_check_count_available();
#else
  return false;
#endif
}

std::size_t Program::cuda_device_check_count_ndarray(Ndarray *values,
                                                  Ndarray *output,
                                                  int value_type,
                                                  int check_op,
                                                  int lower,
                                                  int upper) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA driver check_count is only available on CUDA.");
  TI_ERROR_IF(!values || !output,
              "CUDA driver check_count received a null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || output->shape.size() != 1,
              "CUDA driver check_count expects 1D ndarrays.");
  TI_ERROR_IF(values->get_nelement() == 0,
              "CUDA driver check_count expects at least one input item.");
  TI_ERROR_IF(output->get_nelement() < 1,
              "CUDA driver check_count output must contain at least one item.");
  TI_ERROR_IF(output->get_element_size() != sizeof(int32_t),
              "CUDA driver check_count output must be i32.");
  const std::size_t expected_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(expected_size == 0,
              "CUDA driver check_count received an unsupported value type.");
  TI_ERROR_IF(values->get_element_size() != expected_size,
              "CUDA driver check_count dtype does not match value type.");
  TI_ERROR_IF(check_op < 0 || check_op > 5,
              "CUDA driver check_count received an unsupported check op.");
  TI_ERROR_IF(values->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA driver check_count currently supports at most INT_MAX items.");
#ifdef TI_WITH_CUDA
  void *values_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values));
  void *output_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output));
  void *stream = nullptr;
  return cuda::driver_check_count(
      values_ptr, output_ptr, static_cast<int>(values->get_nelement()),
      static_cast<cuda::CudaTransformValueType>(value_type), 0, expected_size,
      static_cast<cuda::CudaCheckOp>(check_op), lower, upper, stream);
#else
  TI_ERROR(
      "CUDA driver check_count requires building Taichi with TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_device_check_count_strided_ndarray(
    Ndarray *values,
    Ndarray *output,
    int value_type,
    std::size_t offset,
    std::size_t stride,
    int check_op,
    int lower,
    int upper) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA driver strided check_count is only available on CUDA.");
  TI_ERROR_IF(!values || !output,
              "CUDA driver strided check_count received a null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || output->shape.size() != 1,
              "CUDA driver strided check_count expects 1D ndarrays.");
  TI_ERROR_IF(values->get_nelement() == 0,
              "CUDA driver strided check_count expects at least one input item.");
  TI_ERROR_IF(output->get_nelement() < 1,
              "CUDA driver strided check_count output must contain at least one "
              "item.");
  TI_ERROR_IF(output->get_element_size() != sizeof(int32_t),
              "CUDA driver strided check_count output must be i32.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CUDA driver strided check_count received an unsupported value "
              "type.");
  TI_ERROR_IF(check_op < 0 || check_op > 5,
              "CUDA driver strided check_count received an unsupported check op.");
  TI_ERROR_IF(stride < value_size || offset % value_size != 0 ||
                  stride % value_size != 0,
              "CUDA driver strided check_count received invalid offset/stride.");
  const std::size_t n = values->get_nelement();
  const std::size_t src_bytes = n * values->get_element_size();
  TI_ERROR_IF(src_bytes < value_size || offset > src_bytes - value_size ||
                  offset + (n - 1) * stride + value_size > src_bytes,
              "CUDA driver strided check_count source range is out of bounds.");
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA driver strided check_count currently supports at most "
              "INT_MAX items.");
#ifdef TI_WITH_CUDA
  void *values_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values));
  void *output_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output));
  void *stream = nullptr;
  return cuda::driver_check_count(
      values_ptr, output_ptr, static_cast<int>(n),
      static_cast<cuda::CudaTransformValueType>(value_type), offset, stride,
      static_cast<cuda::CudaCheckOp>(check_op), lower, upper, stream);
#else
  TI_ERROR(
      "CUDA driver strided check_count requires building Taichi with "
      "TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_device_check_count_dense_field(SNode *values,
                                                      Ndarray *output,
                                                      int value_type,
                                                      std::size_t n,
                                                      int check_op,
                                                      int lower,
                                                      int upper) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA driver dense field check_count is only available on CUDA.");
  TI_ERROR_IF(!values || !output,
              "CUDA driver dense field check_count received a null argument.");
  TI_ERROR_IF(n == 0,
              "CUDA driver dense field check_count expects at least one item.");
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA driver dense field check_count currently supports at most "
              "INT_MAX items.");
  TI_ERROR_IF(output->shape.size() != 1 || output->get_nelement() < 1 ||
                  output->get_element_size() != sizeof(int32_t),
              "CUDA driver dense field check_count output must be a non-empty "
              "i32 ndarray.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CUDA driver dense field check_count received an unsupported value "
              "type.");
  TI_ERROR_IF(check_op < 0 || check_op > 5,
              "CUDA driver dense field check_count received an unsupported check "
              "op.");
  const std::size_t stride = get_dense_field_stride(values, value_size);
  TI_ERROR_IF(stride < value_size,
              "CUDA driver dense field check_count received an invalid field "
              "stride.");
#ifdef TI_WITH_CUDA
  auto raw_ptr = [this](DevicePtr ptr, const char *op_name) {
    TI_ERROR_IF(!ptr.device, "{} received a null dense field device.",
                op_name);
    DeviceAllocation alloc{ptr.device, ptr.alloc_id};
    auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
    TI_ERROR_IF(!base, "{} received a null dense field data pointer.",
                op_name);
    return static_cast<void *>(reinterpret_cast<uint8_t *>(base) + ptr.offset);
  };
  void *values_ptr = raw_ptr(get_dense_field_device_ptr(values),
                             "CUDA driver dense field check_count");
  void *output_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output));
  void *stream = nullptr;
  return cuda::driver_check_count(
      values_ptr, output_ptr, static_cast<int>(n),
      static_cast<cuda::CudaTransformValueType>(value_type), 0, stride,
      static_cast<cuda::CudaCheckOp>(check_op), lower, upper, stream);
#else
  TI_ERROR(
      "CUDA driver dense field check_count requires building Taichi with "
      "TI_WITH_CUDA=ON.");
#endif
}

void Program::cuda_device_check_count_clear_workspace() {
  // Driver diagnostics own no temporary allocation.
}

std::size_t Program::cuda_device_check_count_workspace_bytes() const {
  return 0;
}

bool Program::cuda_device_metric_reduce_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda &&
         cuda::driver_metric_reduce_available();
#else
  return false;
#endif
}

bool Program::cuda_device_metric_reduce_value_type_available(
    int value_type) const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda &&
         (value_type == static_cast<int>(cuda::CudaTransformValueType::f32) ||
          value_type == static_cast<int>(cuda::CudaTransformValueType::f64));
#else
  return false;
#endif
}

std::size_t Program::cuda_device_metric_reduce_ndarray(Ndarray *values,
                                                    Ndarray *other,
                                                    Ndarray *output,
                                                    int value_type,
                                                    int metric_op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA driver metric_reduce is only available on CUDA.");
  TI_ERROR_IF(!values || !output,
              "CUDA driver metric_reduce received a null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || output->shape.size() != 1,
              "CUDA driver metric_reduce expects 1D ndarrays.");
  TI_ERROR_IF(values->get_nelement() == 0,
              "CUDA driver metric_reduce expects at least one input item.");
  TI_ERROR_IF(output->get_nelement() < 1,
              "CUDA driver metric_reduce output must contain at least one item.");
  TI_ERROR_IF(metric_op < 0 || metric_op > 1,
              "CUDA driver metric_reduce received an unsupported metric op.");
  TI_ERROR_IF(metric_op == 1 && !other,
              "CUDA driver max_abs_delta received a null rhs ndarray.");
  if (other) {
    TI_ERROR_IF(other->shape.size() != 1,
                "CUDA driver metric_reduce rhs must be a 1D ndarray.");
    TI_ERROR_IF(other->get_nelement() != values->get_nelement(),
                "CUDA driver metric_reduce inputs must have the same length.");
  }
  const std::size_t expected_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(expected_size == 0,
              "CUDA driver metric_reduce received an unsupported value type.");
  TI_ERROR_IF(!cuda_device_metric_reduce_value_type_available(value_type),
              "CUDA driver metric_reduce currently supports only f32/f64.");
  TI_ERROR_IF(values->get_element_size() != expected_size ||
                  output->get_element_size() != expected_size ||
                  (other && other->get_element_size() != expected_size),
              "CUDA driver metric_reduce dtype does not match value type.");
  TI_ERROR_IF(values->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA driver metric_reduce currently supports at most INT_MAX "
              "items.");
#ifdef TI_WITH_CUDA
  void *values_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values));
  void *other_ptr =
      other ? reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(other))
            : nullptr;
  void *output_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output));
  void *stream = nullptr;
  return cuda::driver_metric_reduce(
      values_ptr, other_ptr, output_ptr,
      static_cast<int>(values->get_nelement()),
      static_cast<cuda::CudaTransformValueType>(value_type), 0, expected_size, 0,
      expected_size, static_cast<cuda::CudaMetricOp>(metric_op), stream);
#else
  TI_ERROR(
      "CUDA driver metric_reduce requires building Taichi with TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_device_metric_reduce_strided_ndarray(
    Ndarray *values,
    Ndarray *other,
    Ndarray *output,
    int value_type,
    std::size_t values_offset,
    std::size_t values_stride,
    std::size_t other_offset,
    std::size_t other_stride,
    int metric_op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA driver strided metric_reduce is only available on CUDA.");
  TI_ERROR_IF(!values || !output,
              "CUDA driver strided metric_reduce received a null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || output->shape.size() != 1,
              "CUDA driver strided metric_reduce expects 1D ndarrays.");
  TI_ERROR_IF(values->get_nelement() == 0,
              "CUDA driver strided metric_reduce expects at least one item.");
  TI_ERROR_IF(output->get_nelement() < 1,
              "CUDA driver strided metric_reduce output must contain at least "
              "one item.");
  TI_ERROR_IF(metric_op < 0 || metric_op > 1,
              "CUDA driver strided metric_reduce received an unsupported metric "
              "op.");
  TI_ERROR_IF(metric_op == 1 && !other,
              "CUDA driver strided max_abs_delta received a null rhs ndarray.");
  if (!other) {
    other = values;
    other_offset = values_offset;
    other_stride = values_stride;
  }
  TI_ERROR_IF(other->shape.size() != 1,
              "CUDA driver strided metric_reduce rhs must be a 1D ndarray.");
  TI_ERROR_IF(other->get_nelement() != values->get_nelement(),
              "CUDA driver strided metric_reduce inputs must have the same "
              "length.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CUDA driver strided metric_reduce received an unsupported value "
              "type.");
  TI_ERROR_IF(!cuda_device_metric_reduce_value_type_available(value_type),
              "CUDA driver strided metric_reduce currently supports only "
              "f32/f64.");
  TI_ERROR_IF(output->get_element_size() != value_size,
              "CUDA driver strided metric_reduce output dtype does not match "
              "value type.");
  auto check_range = [&](const char *role, Ndarray *arr, std::size_t offset,
                         std::size_t stride) {
    TI_ERROR_IF(stride < value_size || offset % value_size != 0 ||
                    stride % value_size != 0,
                "CUDA driver strided metric_reduce {} received invalid "
                "offset/stride.",
                role);
    const std::size_t bytes = arr->get_nelement() * arr->get_element_size();
    TI_ERROR_IF(bytes < value_size || offset > bytes - value_size ||
                    offset + (arr->get_nelement() - 1) * stride + value_size >
                        bytes,
                "CUDA driver strided metric_reduce {} range is out of bounds.",
                role);
  };
  check_range("source", values, values_offset, values_stride);
  check_range("rhs", other, other_offset, other_stride);
  const std::size_t n = values->get_nelement();
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA driver strided metric_reduce currently supports at most "
              "INT_MAX items.");
#ifdef TI_WITH_CUDA
  void *values_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values));
  void *other_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(other));
  void *output_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output));
  void *stream = nullptr;
  return cuda::driver_metric_reduce(
      values_ptr, other_ptr, output_ptr, static_cast<int>(n),
      static_cast<cuda::CudaTransformValueType>(value_type), values_offset,
      values_stride, other_offset, other_stride,
      static_cast<cuda::CudaMetricOp>(metric_op), stream);
#else
  TI_ERROR(
      "CUDA driver strided metric_reduce requires building Taichi with "
      "TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_device_metric_reduce_dense_field(SNode *values,
                                                        SNode *other,
                                                        Ndarray *output,
                                                        int value_type,
                                                        std::size_t n,
                                                        int metric_op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA driver dense field metric_reduce is only available on CUDA.");
  TI_ERROR_IF(!values || !output,
              "CUDA driver dense field metric_reduce received a null argument.");
  if (!other) {
    other = values;
  }
  TI_ERROR_IF(n == 0,
              "CUDA driver dense field metric_reduce expects at least one item.");
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA driver dense field metric_reduce currently supports at most "
              "INT_MAX items.");
  TI_ERROR_IF(output->shape.size() != 1 || output->get_nelement() < 1,
              "CUDA driver dense field metric_reduce output must be a non-empty "
              "ndarray.");
  TI_ERROR_IF(metric_op < 0 || metric_op > 1,
              "CUDA driver dense field metric_reduce received an unsupported "
              "metric op.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CUDA driver dense field metric_reduce received an unsupported "
              "value type.");
  TI_ERROR_IF(!cuda_device_metric_reduce_value_type_available(value_type),
              "CUDA driver dense field metric_reduce currently supports only "
              "f32/f64.");
  TI_ERROR_IF(output->get_element_size() != value_size,
              "CUDA driver dense field metric_reduce output dtype does not match "
              "value type.");
  const std::size_t values_stride = get_dense_field_stride(values, value_size);
  const std::size_t other_stride = get_dense_field_stride(other, value_size);
  TI_ERROR_IF(values_stride < value_size || other_stride < value_size,
              "CUDA driver dense field metric_reduce received an invalid field "
              "stride.");
#ifdef TI_WITH_CUDA
  auto raw_ptr = [this](DevicePtr ptr, const char *op_name) {
    TI_ERROR_IF(!ptr.device, "{} received a null dense field device.",
                op_name);
    DeviceAllocation alloc{ptr.device, ptr.alloc_id};
    auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
    TI_ERROR_IF(!base, "{} received a null dense field data pointer.",
                op_name);
    return static_cast<void *>(reinterpret_cast<uint8_t *>(base) + ptr.offset);
  };
  void *values_ptr = raw_ptr(get_dense_field_device_ptr(values),
                             "CUDA driver dense field metric_reduce");
  void *other_ptr = raw_ptr(get_dense_field_device_ptr(other),
                            "CUDA driver dense field metric_reduce");
  void *output_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output));
  void *stream = nullptr;
  return cuda::driver_metric_reduce(
      values_ptr, other_ptr, output_ptr, static_cast<int>(n),
      static_cast<cuda::CudaTransformValueType>(value_type), 0, values_stride, 0,
      other_stride, static_cast<cuda::CudaMetricOp>(metric_op), stream);
#else
  TI_ERROR(
      "CUDA driver dense field metric_reduce requires building Taichi with "
      "TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_device_metric_reduce_dense_field_strided_ndarray(
    SNode *field,
    Ndarray *array,
    Ndarray *output,
    int value_type,
    std::size_t n,
    std::size_t array_offset,
    std::size_t array_stride,
    bool field_is_values,
    int metric_op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA driver mixed metric_reduce is only available on CUDA.");
  TI_ERROR_IF(!field || !array || !output,
              "CUDA driver mixed metric_reduce received a null argument.");
  TI_ERROR_IF(n == 0,
              "CUDA driver mixed metric_reduce expects at least one item.");
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA driver mixed metric_reduce currently supports at most "
              "INT_MAX items.");
  TI_ERROR_IF(array->shape.size() != 1 || output->shape.size() != 1,
              "CUDA driver mixed metric_reduce expects 1D ndarrays.");
  TI_ERROR_IF(array->get_nelement() != n,
              "CUDA driver mixed metric_reduce inputs must have the same length.");
  TI_ERROR_IF(output->get_nelement() < 1,
              "CUDA driver mixed metric_reduce output must be a non-empty "
              "ndarray.");
  TI_ERROR_IF(metric_op < 0 || metric_op > 1,
              "CUDA driver mixed metric_reduce received an unsupported metric "
              "op.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CUDA driver mixed metric_reduce received an unsupported value "
              "type.");
  TI_ERROR_IF(!cuda_device_metric_reduce_value_type_available(value_type),
              "CUDA driver mixed metric_reduce currently supports only f32/f64.");
  TI_ERROR_IF(output->get_element_size() != value_size,
              "CUDA driver mixed metric_reduce output dtype does not match value "
              "type.");
  const std::size_t array_bytes =
      array->get_nelement() * array->get_element_size();
  TI_ERROR_IF(array_stride < value_size || array_offset % value_size != 0 ||
                  array_stride % value_size != 0 || array_bytes < value_size ||
                  array_offset > array_bytes - value_size ||
                  array_offset + (n - 1) * array_stride + value_size >
                      array_bytes,
              "CUDA driver mixed metric_reduce ndarray range is out of bounds.");
  const std::size_t field_stride = get_dense_field_stride(field, value_size);
  TI_ERROR_IF(field_stride < value_size,
              "CUDA driver mixed metric_reduce received an invalid field stride.");
#ifdef TI_WITH_CUDA
  auto raw_field_ptr = [this](DevicePtr ptr, const char *op_name) {
    TI_ERROR_IF(!ptr.device, "{} received a null dense field device.",
                op_name);
    DeviceAllocation alloc{ptr.device, ptr.alloc_id};
    auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
    TI_ERROR_IF(!base, "{} received a null dense field data pointer.",
                op_name);
    return static_cast<void *>(reinterpret_cast<uint8_t *>(base) + ptr.offset);
  };
  void *field_ptr = raw_field_ptr(get_dense_field_device_ptr(field),
                                  "CUDA driver mixed metric_reduce");
  void *array_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(array));
  void *output_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output));
  void *stream = nullptr;
  if (field_is_values) {
    return cuda::driver_metric_reduce(
        field_ptr, array_ptr, output_ptr, static_cast<int>(n),
        static_cast<cuda::CudaTransformValueType>(value_type), 0, field_stride,
        array_offset, array_stride, static_cast<cuda::CudaMetricOp>(metric_op),
        stream);
  }
  return cuda::driver_metric_reduce(
      array_ptr, field_ptr, output_ptr, static_cast<int>(n),
      static_cast<cuda::CudaTransformValueType>(value_type), array_offset,
      array_stride, 0, field_stride,
      static_cast<cuda::CudaMetricOp>(metric_op), stream);
#else
  TI_ERROR(
      "CUDA driver mixed metric_reduce requires building Taichi with "
      "TI_WITH_CUDA=ON.");
#endif
}

void Program::cuda_device_metric_reduce_clear_workspace() {
  // Driver diagnostics own no temporary allocation.
}

std::size_t Program::cuda_device_metric_reduce_workspace_bytes() const {
  return 0;
}


bool Program::cuda_cub_check_count_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda &&
         cuda::cub_check_count_available();
#else
  return false;
#endif
}

std::size_t Program::cuda_cub_check_count_ndarray(Ndarray *values,
                                                  Ndarray *output,
                                                  int value_type,
                                                  int check_op,
                                                  int lower,
                                                  int upper) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB check_count is only available on CUDA.");
  TI_ERROR_IF(!values || !output,
              "CUDA CUB check_count received a null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || output->shape.size() != 1,
              "CUDA CUB check_count expects 1D ndarrays.");
  TI_ERROR_IF(values->get_nelement() == 0,
              "CUDA CUB check_count expects at least one input item.");
  TI_ERROR_IF(output->get_nelement() < 1,
              "CUDA CUB check_count output must contain at least one item.");
  TI_ERROR_IF(output->get_element_size() != sizeof(int32_t),
              "CUDA CUB check_count output must be i32.");
  const std::size_t expected_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(expected_size == 0,
              "CUDA CUB check_count received an unsupported value type.");
  TI_ERROR_IF(values->get_element_size() != expected_size,
              "CUDA CUB check_count dtype does not match value type.");
  TI_ERROR_IF(check_op < 0 || check_op > 5,
              "CUDA CUB check_count received an unsupported check op.");
  TI_ERROR_IF(values->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA CUB check_count currently supports at most INT_MAX items.");
#ifdef TI_WITH_CUDA
  void *values_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values));
  void *output_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output));
  void *stream = nullptr;
  return cuda::cub_check_count(
      values_ptr, output_ptr, static_cast<int>(values->get_nelement()),
      static_cast<cuda::CubReduceValueType>(value_type),
      static_cast<cuda::CudaCheckOp>(check_op), lower, upper, stream,
      &primitive_workspace_arena_);
#else
  TI_ERROR(
      "CUDA CUB check_count requires building Taichi with TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_cub_check_count_strided_ndarray(
    Ndarray *values,
    Ndarray *output,
    int value_type,
    std::size_t offset,
    std::size_t stride,
    int check_op,
    int lower,
    int upper) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB strided check_count is only available on CUDA.");
  TI_ERROR_IF(!values || !output,
              "CUDA CUB strided check_count received a null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || output->shape.size() != 1,
              "CUDA CUB strided check_count expects 1D ndarrays.");
  TI_ERROR_IF(values->get_nelement() == 0,
              "CUDA CUB strided check_count expects at least one input item.");
  TI_ERROR_IF(output->get_nelement() < 1,
              "CUDA CUB strided check_count output must contain at least one "
              "item.");
  TI_ERROR_IF(output->get_element_size() != sizeof(int32_t),
              "CUDA CUB strided check_count output must be i32.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CUDA CUB strided check_count received an unsupported value "
              "type.");
  TI_ERROR_IF(check_op < 0 || check_op > 5,
              "CUDA CUB strided check_count received an unsupported check op.");
  TI_ERROR_IF(stride < value_size || offset % value_size != 0 ||
                  stride % value_size != 0,
              "CUDA CUB strided check_count received invalid offset/stride.");
  const std::size_t n = values->get_nelement();
  const std::size_t src_bytes = n * values->get_element_size();
  TI_ERROR_IF(src_bytes < value_size || offset > src_bytes - value_size ||
                  offset + (n - 1) * stride + value_size > src_bytes,
              "CUDA CUB strided check_count source range is out of bounds.");
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA CUB strided check_count currently supports at most "
              "INT_MAX items.");
#ifdef TI_WITH_CUDA
  void *values_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values));
  void *output_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output));
  void *stream = nullptr;
  return cuda::cub_check_count_strided(
      values_ptr, output_ptr, static_cast<int>(n),
      static_cast<cuda::CubReduceValueType>(value_type), offset, stride,
      static_cast<cuda::CudaCheckOp>(check_op), lower, upper, stream,
      &primitive_workspace_arena_);
#else
  TI_ERROR(
      "CUDA CUB strided check_count requires building Taichi with "
      "TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_cub_check_count_dense_field(SNode *values,
                                                      Ndarray *output,
                                                      int value_type,
                                                      std::size_t n,
                                                      int check_op,
                                                      int lower,
                                                      int upper) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB dense field check_count is only available on CUDA.");
  TI_ERROR_IF(!values || !output,
              "CUDA CUB dense field check_count received a null argument.");
  TI_ERROR_IF(n == 0,
              "CUDA CUB dense field check_count expects at least one item.");
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA CUB dense field check_count currently supports at most "
              "INT_MAX items.");
  TI_ERROR_IF(output->shape.size() != 1 || output->get_nelement() < 1 ||
                  output->get_element_size() != sizeof(int32_t),
              "CUDA CUB dense field check_count output must be a non-empty "
              "i32 ndarray.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CUDA CUB dense field check_count received an unsupported value "
              "type.");
  TI_ERROR_IF(check_op < 0 || check_op > 5,
              "CUDA CUB dense field check_count received an unsupported check "
              "op.");
  const std::size_t stride = get_dense_field_stride(values, value_size);
  TI_ERROR_IF(stride < value_size,
              "CUDA CUB dense field check_count received an invalid field "
              "stride.");
#ifdef TI_WITH_CUDA
  auto raw_ptr = [this](DevicePtr ptr, const char *op_name) {
    TI_ERROR_IF(!ptr.device, "{} received a null dense field device.",
                op_name);
    DeviceAllocation alloc{ptr.device, ptr.alloc_id};
    auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
    TI_ERROR_IF(!base, "{} received a null dense field data pointer.",
                op_name);
    return static_cast<void *>(reinterpret_cast<uint8_t *>(base) + ptr.offset);
  };
  void *values_ptr = raw_ptr(get_dense_field_device_ptr(values),
                             "CUDA CUB dense field check_count");
  void *output_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output));
  void *stream = nullptr;
  return cuda::cub_check_count_strided(
      values_ptr, output_ptr, static_cast<int>(n),
      static_cast<cuda::CubReduceValueType>(value_type), 0, stride,
      static_cast<cuda::CudaCheckOp>(check_op), lower, upper, stream,
      &primitive_workspace_arena_);
#else
  TI_ERROR(
      "CUDA CUB dense field check_count requires building Taichi with "
      "TI_WITH_CUDA=ON.");
#endif
}

void Program::cuda_cub_check_count_clear_workspace() {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    clear_primitive_workspaces_for(PrimitiveWorkspaceBackend::cuda,
                                   PrimitiveWorkspaceFamily::check);
  }
#endif
}

std::size_t Program::cuda_cub_check_count_workspace_bytes() const {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    return cuda::cub_check_count_cached_bytes(
        const_cast<PrimitiveWorkspaceArena *>(&primitive_workspace_arena_));
  }
#endif
  return 0;
}

bool Program::cuda_cub_metric_reduce_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda &&
         cuda::cub_metric_reduce_available();
#else
  return false;
#endif
}

bool Program::cuda_cub_metric_reduce_value_type_available(
    int value_type) const {
#ifdef TI_WITH_CUDA
  return cuda::cub_metric_reduce_value_type_available(
      static_cast<cuda::CubReduceValueType>(value_type));
#else
  return false;
#endif
}

std::size_t Program::cuda_cub_metric_reduce_ndarray(Ndarray *values,
                                                    Ndarray *other,
                                                    Ndarray *output,
                                                    int value_type,
                                                    int metric_op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB metric_reduce is only available on CUDA.");
  TI_ERROR_IF(!values || !output,
              "CUDA CUB metric_reduce received a null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || output->shape.size() != 1,
              "CUDA CUB metric_reduce expects 1D ndarrays.");
  TI_ERROR_IF(values->get_nelement() == 0,
              "CUDA CUB metric_reduce expects at least one input item.");
  TI_ERROR_IF(output->get_nelement() < 1,
              "CUDA CUB metric_reduce output must contain at least one item.");
  TI_ERROR_IF(metric_op < 0 || metric_op > 1,
              "CUDA CUB metric_reduce received an unsupported metric op.");
  TI_ERROR_IF(metric_op == 1 && !other,
              "CUDA CUB max_abs_delta received a null rhs ndarray.");
  if (other) {
    TI_ERROR_IF(other->shape.size() != 1,
                "CUDA CUB metric_reduce rhs must be a 1D ndarray.");
    TI_ERROR_IF(other->get_nelement() != values->get_nelement(),
                "CUDA CUB metric_reduce inputs must have the same length.");
  }
  const std::size_t expected_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(expected_size == 0,
              "CUDA CUB metric_reduce received an unsupported value type.");
  TI_ERROR_IF(!cuda_cub_metric_reduce_value_type_available(value_type),
              "CUDA CUB metric_reduce currently supports only f32/f64.");
  TI_ERROR_IF(values->get_element_size() != expected_size ||
                  output->get_element_size() != expected_size ||
                  (other && other->get_element_size() != expected_size),
              "CUDA CUB metric_reduce dtype does not match value type.");
  TI_ERROR_IF(values->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA CUB metric_reduce currently supports at most INT_MAX "
              "items.");
#ifdef TI_WITH_CUDA
  void *values_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values));
  void *other_ptr =
      other ? reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(other))
            : nullptr;
  void *output_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output));
  void *stream = nullptr;
  return cuda::cub_metric_reduce(
      values_ptr, other_ptr, output_ptr,
      static_cast<int>(values->get_nelement()),
      static_cast<cuda::CubReduceValueType>(value_type),
      static_cast<cuda::CudaMetricOp>(metric_op), stream,
      &primitive_workspace_arena_);
#else
  TI_ERROR(
      "CUDA CUB metric_reduce requires building Taichi with TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_cub_metric_reduce_strided_ndarray(
    Ndarray *values,
    Ndarray *other,
    Ndarray *output,
    int value_type,
    std::size_t values_offset,
    std::size_t values_stride,
    std::size_t other_offset,
    std::size_t other_stride,
    int metric_op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB strided metric_reduce is only available on CUDA.");
  TI_ERROR_IF(!values || !output,
              "CUDA CUB strided metric_reduce received a null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || output->shape.size() != 1,
              "CUDA CUB strided metric_reduce expects 1D ndarrays.");
  TI_ERROR_IF(values->get_nelement() == 0,
              "CUDA CUB strided metric_reduce expects at least one item.");
  TI_ERROR_IF(output->get_nelement() < 1,
              "CUDA CUB strided metric_reduce output must contain at least "
              "one item.");
  TI_ERROR_IF(metric_op < 0 || metric_op > 1,
              "CUDA CUB strided metric_reduce received an unsupported metric "
              "op.");
  TI_ERROR_IF(metric_op == 1 && !other,
              "CUDA CUB strided max_abs_delta received a null rhs ndarray.");
  if (!other) {
    other = values;
    other_offset = values_offset;
    other_stride = values_stride;
  }
  TI_ERROR_IF(other->shape.size() != 1,
              "CUDA CUB strided metric_reduce rhs must be a 1D ndarray.");
  TI_ERROR_IF(other->get_nelement() != values->get_nelement(),
              "CUDA CUB strided metric_reduce inputs must have the same "
              "length.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CUDA CUB strided metric_reduce received an unsupported value "
              "type.");
  TI_ERROR_IF(!cuda_cub_metric_reduce_value_type_available(value_type),
              "CUDA CUB strided metric_reduce currently supports only "
              "f32/f64.");
  TI_ERROR_IF(output->get_element_size() != value_size,
              "CUDA CUB strided metric_reduce output dtype does not match "
              "value type.");
  auto check_range = [&](const char *role, Ndarray *arr, std::size_t offset,
                         std::size_t stride) {
    TI_ERROR_IF(stride < value_size || offset % value_size != 0 ||
                    stride % value_size != 0,
                "CUDA CUB strided metric_reduce {} received invalid "
                "offset/stride.",
                role);
    const std::size_t bytes = arr->get_nelement() * arr->get_element_size();
    TI_ERROR_IF(bytes < value_size || offset > bytes - value_size ||
                    offset + (arr->get_nelement() - 1) * stride + value_size >
                        bytes,
                "CUDA CUB strided metric_reduce {} range is out of bounds.",
                role);
  };
  check_range("source", values, values_offset, values_stride);
  check_range("rhs", other, other_offset, other_stride);
  const std::size_t n = values->get_nelement();
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA CUB strided metric_reduce currently supports at most "
              "INT_MAX items.");
#ifdef TI_WITH_CUDA
  void *values_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values));
  void *other_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(other));
  void *output_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output));
  void *stream = nullptr;
  return cuda::cub_metric_reduce_strided(
      values_ptr, other_ptr, output_ptr, static_cast<int>(n),
      static_cast<cuda::CubReduceValueType>(value_type), values_offset,
      values_stride, other_offset, other_stride,
      static_cast<cuda::CudaMetricOp>(metric_op), stream,
      &primitive_workspace_arena_);
#else
  TI_ERROR(
      "CUDA CUB strided metric_reduce requires building Taichi with "
      "TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_cub_metric_reduce_dense_field(SNode *values,
                                                        SNode *other,
                                                        Ndarray *output,
                                                        int value_type,
                                                        std::size_t n,
                                                        int metric_op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB dense field metric_reduce is only available on CUDA.");
  TI_ERROR_IF(!values || !output,
              "CUDA CUB dense field metric_reduce received a null argument.");
  if (!other) {
    other = values;
  }
  TI_ERROR_IF(n == 0,
              "CUDA CUB dense field metric_reduce expects at least one item.");
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA CUB dense field metric_reduce currently supports at most "
              "INT_MAX items.");
  TI_ERROR_IF(output->shape.size() != 1 || output->get_nelement() < 1,
              "CUDA CUB dense field metric_reduce output must be a non-empty "
              "ndarray.");
  TI_ERROR_IF(metric_op < 0 || metric_op > 1,
              "CUDA CUB dense field metric_reduce received an unsupported "
              "metric op.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CUDA CUB dense field metric_reduce received an unsupported "
              "value type.");
  TI_ERROR_IF(!cuda_cub_metric_reduce_value_type_available(value_type),
              "CUDA CUB dense field metric_reduce currently supports only "
              "f32/f64.");
  TI_ERROR_IF(output->get_element_size() != value_size,
              "CUDA CUB dense field metric_reduce output dtype does not match "
              "value type.");
  const std::size_t values_stride = get_dense_field_stride(values, value_size);
  const std::size_t other_stride = get_dense_field_stride(other, value_size);
  TI_ERROR_IF(values_stride < value_size || other_stride < value_size,
              "CUDA CUB dense field metric_reduce received an invalid field "
              "stride.");
#ifdef TI_WITH_CUDA
  auto raw_ptr = [this](DevicePtr ptr, const char *op_name) {
    TI_ERROR_IF(!ptr.device, "{} received a null dense field device.",
                op_name);
    DeviceAllocation alloc{ptr.device, ptr.alloc_id};
    auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
    TI_ERROR_IF(!base, "{} received a null dense field data pointer.",
                op_name);
    return static_cast<void *>(reinterpret_cast<uint8_t *>(base) + ptr.offset);
  };
  void *values_ptr = raw_ptr(get_dense_field_device_ptr(values),
                             "CUDA CUB dense field metric_reduce");
  void *other_ptr = raw_ptr(get_dense_field_device_ptr(other),
                            "CUDA CUB dense field metric_reduce");
  void *output_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output));
  void *stream = nullptr;
  return cuda::cub_metric_reduce_strided(
      values_ptr, other_ptr, output_ptr, static_cast<int>(n),
      static_cast<cuda::CubReduceValueType>(value_type), 0, values_stride, 0,
      other_stride, static_cast<cuda::CudaMetricOp>(metric_op), stream,
      &primitive_workspace_arena_);
#else
  TI_ERROR(
      "CUDA CUB dense field metric_reduce requires building Taichi with "
      "TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_cub_metric_reduce_dense_field_strided_ndarray(
    SNode *field,
    Ndarray *array,
    Ndarray *output,
    int value_type,
    std::size_t n,
    std::size_t array_offset,
    std::size_t array_stride,
    bool field_is_values,
    int metric_op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB mixed metric_reduce is only available on CUDA.");
  TI_ERROR_IF(!field || !array || !output,
              "CUDA CUB mixed metric_reduce received a null argument.");
  TI_ERROR_IF(n == 0,
              "CUDA CUB mixed metric_reduce expects at least one item.");
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA CUB mixed metric_reduce currently supports at most "
              "INT_MAX items.");
  TI_ERROR_IF(array->shape.size() != 1 || output->shape.size() != 1,
              "CUDA CUB mixed metric_reduce expects 1D ndarrays.");
  TI_ERROR_IF(array->get_nelement() != n,
              "CUDA CUB mixed metric_reduce inputs must have the same length.");
  TI_ERROR_IF(output->get_nelement() < 1,
              "CUDA CUB mixed metric_reduce output must be a non-empty "
              "ndarray.");
  TI_ERROR_IF(metric_op < 0 || metric_op > 1,
              "CUDA CUB mixed metric_reduce received an unsupported metric "
              "op.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CUDA CUB mixed metric_reduce received an unsupported value "
              "type.");
  TI_ERROR_IF(!cuda_cub_metric_reduce_value_type_available(value_type),
              "CUDA CUB mixed metric_reduce currently supports only f32/f64.");
  TI_ERROR_IF(output->get_element_size() != value_size,
              "CUDA CUB mixed metric_reduce output dtype does not match value "
              "type.");
  const std::size_t array_bytes =
      array->get_nelement() * array->get_element_size();
  TI_ERROR_IF(array_stride < value_size || array_offset % value_size != 0 ||
                  array_stride % value_size != 0 || array_bytes < value_size ||
                  array_offset > array_bytes - value_size ||
                  array_offset + (n - 1) * array_stride + value_size >
                      array_bytes,
              "CUDA CUB mixed metric_reduce ndarray range is out of bounds.");
  const std::size_t field_stride = get_dense_field_stride(field, value_size);
  TI_ERROR_IF(field_stride < value_size,
              "CUDA CUB mixed metric_reduce received an invalid field stride.");
#ifdef TI_WITH_CUDA
  auto raw_field_ptr = [this](DevicePtr ptr, const char *op_name) {
    TI_ERROR_IF(!ptr.device, "{} received a null dense field device.",
                op_name);
    DeviceAllocation alloc{ptr.device, ptr.alloc_id};
    auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
    TI_ERROR_IF(!base, "{} received a null dense field data pointer.",
                op_name);
    return static_cast<void *>(reinterpret_cast<uint8_t *>(base) + ptr.offset);
  };
  void *field_ptr = raw_field_ptr(get_dense_field_device_ptr(field),
                                  "CUDA CUB mixed metric_reduce");
  void *array_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(array));
  void *output_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output));
  void *stream = nullptr;
  if (field_is_values) {
    return cuda::cub_metric_reduce_strided(
        field_ptr, array_ptr, output_ptr, static_cast<int>(n),
        static_cast<cuda::CubReduceValueType>(value_type), 0, field_stride,
        array_offset, array_stride, static_cast<cuda::CudaMetricOp>(metric_op),
        stream, &primitive_workspace_arena_);
  }
  return cuda::cub_metric_reduce_strided(
      array_ptr, field_ptr, output_ptr, static_cast<int>(n),
      static_cast<cuda::CubReduceValueType>(value_type), array_offset,
      array_stride, 0, field_stride,
      static_cast<cuda::CudaMetricOp>(metric_op), stream,
      &primitive_workspace_arena_);
#else
  TI_ERROR(
      "CUDA CUB mixed metric_reduce requires building Taichi with "
      "TI_WITH_CUDA=ON.");
#endif
}

void Program::cuda_cub_metric_reduce_clear_workspace() {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    clear_primitive_workspaces_for(PrimitiveWorkspaceBackend::cuda,
                                   PrimitiveWorkspaceFamily::metric);
  }
#endif
}

std::size_t Program::cuda_cub_metric_reduce_workspace_bytes() const {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    return cuda::cub_metric_reduce_cached_bytes(
        const_cast<PrimitiveWorkspaceArena *>(&primitive_workspace_arena_));
  }
#endif
  return 0;
}

bool Program::cpu_scan_available() const {
  return arch_is_cpu(compile_config().arch);
}

std::size_t Program::cpu_inclusive_scan_ndarray(Ndarray *data,
                                                int value_type) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native scan is only available on CPU backends.");
  TI_ERROR_IF(!data, "CPU native scan received a null ndarray.");
  TI_ERROR_IF(data->shape.size() != 1,
              "CPU native scan currently expects a 1D ndarray.");
  const std::size_t expected_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(expected_size == 0,
              "CPU native scan received an unsupported value type.");
  TI_ERROR_IF(data->get_element_size() != expected_size,
              "CPU native scan dtype does not match the requested value type.");
  auto ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(data));
  TI_ERROR_IF(!ptr, "CPU native scan received a null data pointer.");
  const std::size_t n = data->get_nelement();

  auto scan_typed = [n](auto *typed_ptr) {
    using T = std::remove_pointer_t<decltype(typed_ptr)>;
    T prefix{};
    for (std::size_t i = 0; i < n; ++i) {
      prefix += typed_ptr[i];
      typed_ptr[i] = prefix;
    }
  };

  switch (value_type) {
    case 0:
      scan_typed(reinterpret_cast<int32_t *>(ptr));
      break;
    case 1:
      scan_typed(reinterpret_cast<float *>(ptr));
      break;
    case 2:
      scan_typed(reinterpret_cast<uint32_t *>(ptr));
      break;
    case 3:
      scan_typed(reinterpret_cast<uint64_t *>(ptr));
      break;
    case 4:
      scan_typed(reinterpret_cast<int64_t *>(ptr));
      break;
    case 5:
      scan_typed(reinterpret_cast<double *>(ptr));
      break;
    default:
      TI_ERROR("CPU native scan received an unsupported value type.");
  }
  return 0;
}

std::size_t Program::cpu_inclusive_reverse_scan_ndarray(Ndarray *data,
                                                        int value_type) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native reverse scan is only available on CPU backends.");
  TI_ERROR_IF(!data, "CPU native reverse scan received a null ndarray.");
  TI_ERROR_IF(data->shape.size() != 1,
              "CPU native reverse scan currently expects a 1D ndarray.");
  const std::size_t expected_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(expected_size == 0,
              "CPU native reverse scan received an unsupported value type.");
  TI_ERROR_IF(data->get_element_size() != expected_size,
              "CPU native reverse scan dtype does not match the requested "
              "value type.");
  auto ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(data));
  TI_ERROR_IF(!ptr, "CPU native reverse scan received a null data pointer.");
  const std::size_t n = data->get_nelement();

  switch (value_type) {
    case 0:
      return cpu_reverse_scan_typed(reinterpret_cast<int32_t *>(ptr), n);
    case 1:
      return cpu_reverse_scan_typed(reinterpret_cast<float *>(ptr), n);
    case 2:
      return cpu_reverse_scan_typed(reinterpret_cast<uint32_t *>(ptr), n);
    case 3:
      return cpu_reverse_scan_typed(reinterpret_cast<uint64_t *>(ptr), n);
    case 4:
      return cpu_reverse_scan_typed(reinterpret_cast<int64_t *>(ptr), n);
    case 5:
      return cpu_reverse_scan_typed(reinterpret_cast<double *>(ptr), n);
    default:
      TI_ERROR("CPU native reverse scan received an unsupported value type.");
  }
}

std::size_t Program::cpu_inclusive_scan_member_ndarray(Ndarray *data,
                                                       int value_type,
                                                       std::size_t offset,
                                                       std::size_t stride) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native strided scan is only available on CPU backends.");
  check_scan_member_request("CPU native", data, value_type, offset, stride);
  auto ptr = reinterpret_cast<uint8_t *>(get_ndarray_data_ptr_as_int(data));
  TI_ERROR_IF(!ptr, "CPU native strided scan received a null data pointer.");
  const std::size_t n = data->get_nelement();

  switch (value_type) {
    case 0:
      return cpu_scan_strided_typed<int32_t>(ptr, n, offset, stride);
    case 1:
      return cpu_scan_strided_typed<float>(ptr, n, offset, stride);
    case 2:
      return cpu_scan_strided_typed<uint32_t>(ptr, n, offset, stride);
    case 3:
      return cpu_scan_strided_typed<uint64_t>(ptr, n, offset, stride);
    case 4:
      return cpu_scan_strided_typed<int64_t>(ptr, n, offset, stride);
    case 5:
      return cpu_scan_strided_typed<double>(ptr, n, offset, stride);
    default:
      TI_ERROR("CPU native strided scan received an unsupported value type.");
  }
}

std::size_t Program::cpu_inclusive_reverse_scan_member_ndarray(
    Ndarray *data,
    int value_type,
    std::size_t offset,
    std::size_t stride) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native reverse strided scan is only available on CPU "
              "backends.");
  check_scan_member_request("CPU native reverse", data, value_type, offset,
                            stride);
  auto ptr = reinterpret_cast<uint8_t *>(get_ndarray_data_ptr_as_int(data));
  TI_ERROR_IF(!ptr,
              "CPU native reverse strided scan received a null data pointer.");
  const std::size_t n = data->get_nelement();

  switch (value_type) {
    case 0:
      return cpu_reverse_scan_strided_typed<int32_t>(ptr, n, offset, stride);
    case 1:
      return cpu_reverse_scan_strided_typed<float>(ptr, n, offset, stride);
    case 2:
      return cpu_reverse_scan_strided_typed<uint32_t>(ptr, n, offset, stride);
    case 3:
      return cpu_reverse_scan_strided_typed<uint64_t>(ptr, n, offset, stride);
    case 4:
      return cpu_reverse_scan_strided_typed<int64_t>(ptr, n, offset, stride);
    case 5:
      return cpu_reverse_scan_strided_typed<double>(ptr, n, offset, stride);
    default:
      TI_ERROR(
          "CPU native reverse strided scan received an unsupported value type.");
  }
}

std::size_t Program::cpu_inclusive_scan_dense_field(SNode *data,
                                                    int value_type,
                                                    std::size_t n) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native dense field scan is only available on CPU backends.");
  TI_ERROR_IF(n == 0, "CPU native dense field scan expects at least one item.");
  std::size_t stride = 0;
  auto *ptr = map_cpu_dense_field(this, data, value_type, n,
                                  "CPU native dense field scan", &stride);
  switch (value_type) {
    case 0:
      if (stride == sizeof(int32_t)) {
        return cpu_scan_typed(reinterpret_cast<int32_t *>(ptr), n);
      }
      return cpu_scan_strided_typed<int32_t>(ptr, n, 0, stride);
    case 1:
      if (stride == sizeof(float)) {
        return cpu_scan_typed(reinterpret_cast<float *>(ptr), n);
      }
      return cpu_scan_strided_typed<float>(ptr, n, 0, stride);
    case 2:
      if (stride == sizeof(uint32_t)) {
        return cpu_scan_typed(reinterpret_cast<uint32_t *>(ptr), n);
      }
      return cpu_scan_strided_typed<uint32_t>(ptr, n, 0, stride);
    case 3:
      if (stride == sizeof(uint64_t)) {
        return cpu_scan_typed(reinterpret_cast<uint64_t *>(ptr), n);
      }
      return cpu_scan_strided_typed<uint64_t>(ptr, n, 0, stride);
    case 4:
      if (stride == sizeof(int64_t)) {
        return cpu_scan_typed(reinterpret_cast<int64_t *>(ptr), n);
      }
      return cpu_scan_strided_typed<int64_t>(ptr, n, 0, stride);
    case 5:
      if (stride == sizeof(double)) {
        return cpu_scan_typed(reinterpret_cast<double *>(ptr), n);
      }
      return cpu_scan_strided_typed<double>(ptr, n, 0, stride);
    default:
      TI_ERROR("CPU native dense field scan received an unsupported value type.");
  }
}

std::size_t Program::cpu_inclusive_reverse_scan_dense_field(SNode *data,
                                                            int value_type,
                                                            std::size_t n) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native dense field reverse scan is only available on CPU "
              "backends.");
  TI_ERROR_IF(n == 0,
              "CPU native dense field reverse scan expects at least one item.");
  std::size_t stride = 0;
  auto *ptr = map_cpu_dense_field(this, data, value_type, n,
                                  "CPU native dense field reverse scan",
                                  &stride);
  switch (value_type) {
    case 0:
      if (stride == sizeof(int32_t)) {
        return cpu_reverse_scan_typed(reinterpret_cast<int32_t *>(ptr), n);
      }
      return cpu_reverse_scan_strided_typed<int32_t>(ptr, n, 0, stride);
    case 1:
      if (stride == sizeof(float)) {
        return cpu_reverse_scan_typed(reinterpret_cast<float *>(ptr), n);
      }
      return cpu_reverse_scan_strided_typed<float>(ptr, n, 0, stride);
    case 2:
      if (stride == sizeof(uint32_t)) {
        return cpu_reverse_scan_typed(reinterpret_cast<uint32_t *>(ptr), n);
      }
      return cpu_reverse_scan_strided_typed<uint32_t>(ptr, n, 0, stride);
    case 3:
      if (stride == sizeof(uint64_t)) {
        return cpu_reverse_scan_typed(reinterpret_cast<uint64_t *>(ptr), n);
      }
      return cpu_reverse_scan_strided_typed<uint64_t>(ptr, n, 0, stride);
    case 4:
      if (stride == sizeof(int64_t)) {
        return cpu_reverse_scan_typed(reinterpret_cast<int64_t *>(ptr), n);
      }
      return cpu_reverse_scan_strided_typed<int64_t>(ptr, n, 0, stride);
    case 5:
      if (stride == sizeof(double)) {
        return cpu_reverse_scan_typed(reinterpret_cast<double *>(ptr), n);
      }
      return cpu_reverse_scan_strided_typed<double>(ptr, n, 0, stride);
    default:
      TI_ERROR(
          "CPU native dense field reverse scan received an unsupported value "
          "type.");
  }
}

std::size_t Program::cpu_inclusive_scan_dense_field_packed(SNode *data,
                                                           int value_type,
                                                           std::size_t n,
                                                           int lane_count) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native packed dense field scan is only available on CPU "
              "backends.");
  if (n <= 1) {
    return 0;
  }
  const std::size_t value_size = primitive_value_type_size(value_type);
  const std::size_t item_bytes = dense_field_packed_bytes(
      value_type, 1, lane_count, "CPU native packed dense field scan");
  auto *ptr = map_cpu_dense_field_packed(
      this, data, value_type, n, lane_count,
      "CPU native packed dense field scan");
  switch (value_type) {
    case 0:
      for (int lane = 0; lane < lane_count; ++lane) {
        cpu_scan_strided_typed<int32_t>(
            ptr, n, static_cast<std::size_t>(lane) * value_size, item_bytes);
      }
      return 0;
    case 1:
      for (int lane = 0; lane < lane_count; ++lane) {
        cpu_scan_strided_typed<float>(
            ptr, n, static_cast<std::size_t>(lane) * value_size, item_bytes);
      }
      return 0;
    case 2:
      for (int lane = 0; lane < lane_count; ++lane) {
        cpu_scan_strided_typed<uint32_t>(
            ptr, n, static_cast<std::size_t>(lane) * value_size, item_bytes);
      }
      return 0;
    case 3:
      for (int lane = 0; lane < lane_count; ++lane) {
        cpu_scan_strided_typed<uint64_t>(
            ptr, n, static_cast<std::size_t>(lane) * value_size, item_bytes);
      }
      return 0;
    case 4:
      for (int lane = 0; lane < lane_count; ++lane) {
        cpu_scan_strided_typed<int64_t>(
            ptr, n, static_cast<std::size_t>(lane) * value_size, item_bytes);
      }
      return 0;
    case 5:
      for (int lane = 0; lane < lane_count; ++lane) {
        cpu_scan_strided_typed<double>(
            ptr, n, static_cast<std::size_t>(lane) * value_size, item_bytes);
      }
      return 0;
    default:
      TI_ERROR(
          "CPU native packed dense field scan received an unsupported value "
          "type.");
  }
}

std::size_t Program::cpu_inclusive_reverse_scan_dense_field_packed(
    SNode *data,
    int value_type,
    std::size_t n,
    int lane_count) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native packed dense field reverse scan is only available "
              "on CPU backends.");
  if (n <= 1) {
    return 0;
  }
  const std::size_t value_size = primitive_value_type_size(value_type);
  const std::size_t item_bytes = dense_field_packed_bytes(
      value_type, 1, lane_count, "CPU native packed dense field reverse scan");
  auto *ptr = map_cpu_dense_field_packed(
      this, data, value_type, n, lane_count,
      "CPU native packed dense field reverse scan");
  switch (value_type) {
    case 0:
      for (int lane = 0; lane < lane_count; ++lane) {
        cpu_reverse_scan_strided_typed<int32_t>(
            ptr, n, static_cast<std::size_t>(lane) * value_size, item_bytes);
      }
      return 0;
    case 1:
      for (int lane = 0; lane < lane_count; ++lane) {
        cpu_reverse_scan_strided_typed<float>(
            ptr, n, static_cast<std::size_t>(lane) * value_size, item_bytes);
      }
      return 0;
    case 2:
      for (int lane = 0; lane < lane_count; ++lane) {
        cpu_reverse_scan_strided_typed<uint32_t>(
            ptr, n, static_cast<std::size_t>(lane) * value_size, item_bytes);
      }
      return 0;
    case 3:
      for (int lane = 0; lane < lane_count; ++lane) {
        cpu_reverse_scan_strided_typed<uint64_t>(
            ptr, n, static_cast<std::size_t>(lane) * value_size, item_bytes);
      }
      return 0;
    case 4:
      for (int lane = 0; lane < lane_count; ++lane) {
        cpu_reverse_scan_strided_typed<int64_t>(
            ptr, n, static_cast<std::size_t>(lane) * value_size, item_bytes);
      }
      return 0;
    case 5:
      for (int lane = 0; lane < lane_count; ++lane) {
        cpu_reverse_scan_strided_typed<double>(
            ptr, n, static_cast<std::size_t>(lane) * value_size, item_bytes);
      }
      return 0;
    default:
      TI_ERROR(
          "CPU native packed dense field reverse scan received an unsupported "
          "value type.");
  }
}

std::size_t Program::cpu_scan_workspace_bytes() const {
  return 0;
}

bool Program::cpu_compact_available() const {
  return arch_is_cpu(compile_config().arch);
}

template <typename T>
void cpu_fill_dense_field_typed(uint8_t *dst_ptr,
                                std::size_t dst_stride,
                                std::size_t n,
                                uint64_t value_bits,
                                int max_threads) {
  T value{};
  std::memcpy(&value, &value_bits, sizeof(T));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  if (cpu_use_parallel_simple_loop(n, target_threads)) {
    CpuDenseFieldFillTaskContext<T> ctx;
    ctx.data = dst_ptr;
    ctx.stride = dst_stride;
    ctx.value = value;
    ctx.n = n;
    ctx.num_threads = target_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(target_threads, target_threads, &ctx,
             cpu_dense_field_fill_task<T>);
    return;
  }
  if (dst_stride == sizeof(T)) {
    auto *data = reinterpret_cast<T *>(dst_ptr);
    std::fill(data, data + n, value);
    return;
  }
  for (std::size_t i = 0; i < n; ++i) {
    *reinterpret_cast<T *>(dst_ptr + i * dst_stride) = value;
  }
}

double dense_field_fill_value_as_double(int value_type, uint64_t value_bits) {
  switch (value_type) {
    case 0: {
      int32_t value{};
      std::memcpy(&value, &value_bits, sizeof(value));
      return static_cast<double>(value);
    }
    case 1: {
      float value{};
      std::memcpy(&value, &value_bits, sizeof(value));
      return static_cast<double>(value);
    }
    case 2: {
      uint32_t value{};
      std::memcpy(&value, &value_bits, sizeof(value));
      return static_cast<double>(value);
    }
    default:
      TI_ERROR(
          "Native dense field fill cannot represent this value type as a "
          "32-bit device fill value.");
  }
  return 0.0;
}

bool native_dense_field_bulk_arch(Arch arch) {
  return arch_is_cpu(arch) || arch == Arch::cuda || arch == Arch::vulkan;
}

DeviceAllocationUnique ensure_vulkan_dense_host_staging(
    Device *device,
    DeviceAllocationUnique &staging,
    std::size_t &capacity,
    std::size_t bytes,
    bool host_write,
    bool host_read,
    AllocUsage usage) {
  TI_ERROR_IF(!device,
              "Native dense field host copy requires a valid Vulkan device.");
  if (!staging || capacity < bytes) {
    auto [new_staging, res] = device->allocate_memory_unique(
        {bytes, host_write, host_read, false, usage});
    TI_ERROR_IF(res != RhiResult::success,
                "Native dense field host copy failed to allocate staging "
                "buffer: {}",
                res);
    capacity = bytes;
    return std::move(new_staging);
  }
  return nullptr;
}

Program::DenseFieldHostCopyStagingResource &
Program::dense_field_staging_resource() {
  TI_ERROR_IF(!dense_field_staging_open_ || !dense_field_staging_lease_,
              "Dense-field staging cache is unavailable after Program "
              "finalize");
  TI_ASSERT(dense_field_staging_lease_.handle() ==
            dense_field_staging_handle_);
  return *dense_field_staging_lease_;
}

void Program::close_dense_field_staging_resource() {
  if (dense_field_staging_lease_) {
    auto &staging = *dense_field_staging_lease_;
    std::lock_guard<std::mutex> lock(staging.mutex);
    staging.upload.reset();
    staging.upload_capacity = 0;
    staging.readback.reset();
    staging.readback_capacity = 0;
  }
  dense_field_staging_lease_.reset();
  if (dense_field_staging_handle_) {
    const auto result =
        dense_field_staging_resources_.retire(dense_field_staging_handle_);
    TI_ASSERT(result == DenseFieldStagingRegistry::Result::kSuccess ||
              result == DenseFieldStagingRegistry::Result::kInvalidHandle);
    dense_field_staging_handle_ = {};
  }
  dense_field_staging_resources_.finalize(
      {kDenseFieldStagingResourceKind});
}

std::unordered_map<std::string, std::uint64_t>
Program::debug_dense_field_staging_stats() {
  auto resource_submission_guard =
      acquire_runtime_resource_submission_guard();
  const auto stats = dense_field_staging_resources_.stats();
  std::unordered_map<std::string, std::uint64_t> result{
      {"slots", stats.slots},
      {"live", stats.live},
      {"retiring", stats.retiring},
      {"released", stats.released},
      {"leases", stats.leases},
      {"created_total", stats.created_total},
      {"retired_total", stats.retired_total},
      {"released_total", stats.released_total},
      {"release_errors", stats.release_errors},
      {"closed", stats.closed ? 1u : 0u},
      {"open", dense_field_staging_open_ ? 1u : 0u},
      {"domain", dense_field_staging_handle_.domain},
      {"kind", dense_field_staging_handle_.kind},
      {"index", dense_field_staging_handle_.index},
      {"generation", dense_field_staging_handle_.generation},
      {"upload_capacity", 0},
      {"readback_capacity", 0},
      {"has_upload", 0},
      {"has_readback", 0},
  };
  if (dense_field_staging_lease_) {
    auto &staging = *dense_field_staging_lease_;
    std::lock_guard<std::mutex> lock(staging.mutex);
    result["upload_capacity"] = staging.upload_capacity;
    result["readback_capacity"] = staging.readback_capacity;
    result["has_upload"] = staging.upload ? 1u : 0u;
    result["has_readback"] = staging.readback ? 1u : 0u;
  }
  return result;
}

std::unordered_map<std::string, std::uint64_t>
Program::debug_dense_storage_binding_stats() const {
  return {
      {"direct_submissions",
       dense_storage_direct_submissions_.load(std::memory_order_relaxed)},
      {"resolved_bindings",
       dense_storage_resolved_bindings_.load(std::memory_order_relaxed)},
      {"resolved_bytes",
       dense_storage_resolved_bytes_.load(std::memory_order_relaxed)},
      {"ndarray_bindings",
       dense_storage_ndarray_bindings_.load(std::memory_order_relaxed)},
      {"field_bindings",
       dense_storage_field_bindings_.load(std::memory_order_relaxed)},
      {"external_bindings",
       dense_storage_external_bindings_.load(std::memory_order_relaxed)},
      {"temporary_allocations", 0},
      {"temporary_bytes", 0},
  };
}

void Program::fill_dense_field(SNode *dst,
                               int value_type,
                               uint64_t value_bits,
                               std::size_t n) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  const Arch arch = compile_config().arch;
  TI_ERROR_IF(!native_dense_field_bulk_arch(arch),
              "Native dense field fill is only available on CPU, CUDA, and "
              "Vulkan backends.");
  TI_ERROR_IF(!dst, "Native dense field fill received a null field.");
  const std::size_t item_bytes = primitive_value_type_size(value_type);
  TI_ERROR_IF(item_bytes == 0,
              "Native dense field fill received an unsupported value type.");
  if (n == 0) {
    return;
  }
  if (arch == Arch::cuda) {
#ifdef TI_WITH_CUDA
    if (value_bits == 0) {
      cuda_device_zero_dense_field(dst, value_type, n);
      return;
    }
    TI_ERROR_IF(value_type > 2,
                "Native dense field fill currently supports non-zero CUDA "
                "device fills only for 32-bit primitive fields.");
    DevicePtr dst_device_ptr = get_dense_field_device_ptr(dst);
    const std::size_t dst_stride = get_dense_field_stride(dst, item_bytes);
    TI_ERROR_IF(dst_stride < item_bytes,
                "Native dense field fill received an invalid field stride.");
    if (dst_stride == item_bytes) {
      DeviceAllocation alloc{dst_device_ptr.device, dst_device_ptr.alloc_id};
      auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
      TI_ERROR_IF(!base,
                  "Native dense field fill received a null CUDA field data "
                  "pointer.");
      auto *dst_raw =
          reinterpret_cast<uint8_t *>(base) + dst_device_ptr.offset;
      CUDADriver::get_instance().memsetd32(
          dst_raw, static_cast<uint32_t>(value_bits), n);
      return;
    }
    const double value = dense_field_fill_value_as_double(value_type, value_bits);
    cuda_device_transform_affine_dense_field(dst, dst, value_type, n, 0.0,
                                             value);
    return;
#else
    TI_ERROR("Native CUDA dense field fill requires TI_WITH_CUDA=ON.");
#endif
  }
  if (arch == Arch::vulkan) {
    if (value_bits == 0) {
      vulkan_zero_dense_field(dst, value_type, n);
      return;
    }
    TI_ERROR_IF(value_type > 2,
                "Native dense field fill currently supports non-zero Vulkan "
                "device fills only for 32-bit primitive fields.");
    DevicePtr dst_device_ptr = get_dense_field_device_ptr(dst);
    const std::size_t dst_stride = get_dense_field_stride(dst, item_bytes);
    TI_ERROR_IF(dst_stride < item_bytes,
                "Native dense field fill received an invalid field stride.");
    DeviceAllocation dst_alloc{dst_device_ptr.device, dst_device_ptr.alloc_id};
    TI_ERROR_IF(!dst_alloc.device,
                "Native dense field fill received null Vulkan storage.");
    if (dst_stride == item_bytes) {
      const std::size_t dst_bytes = n * item_bytes;
      enqueue_compute_op_lambda(
          [dst_alloc, dst_offset = dst_device_ptr.offset, dst_bytes,
           value = static_cast<uint32_t>(value_bits)](
              Device * /*device*/, CommandList *cmdlist) {
            cmdlist->buffer_fill(dst_alloc.get_ptr(dst_offset), dst_bytes,
                                 value);
            cmdlist->buffer_barrier(dst_alloc.get_ptr(dst_offset), dst_bytes);
          },
          {});
      return;
    }
    const double value = dense_field_fill_value_as_double(value_type, value_bits);
    vulkan_transform_affine_dense_field(dst, dst, value_type, n, 0.0, value);
    return;
  }
  std::size_t dst_stride = 0;
  auto *dst_ptr = map_cpu_dense_field(this, dst, value_type, n,
                                      "Native dense field fill", &dst_stride);
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  switch (value_type) {
    case 0:
      cpu_fill_dense_field_typed<int32_t>(dst_ptr, dst_stride, n, value_bits,
                                          max_threads);
      return;
    case 1:
      cpu_fill_dense_field_typed<float>(dst_ptr, dst_stride, n, value_bits,
                                        max_threads);
      return;
    case 2:
      cpu_fill_dense_field_typed<uint32_t>(dst_ptr, dst_stride, n, value_bits,
                                           max_threads);
      return;
    case 3:
      cpu_fill_dense_field_typed<uint64_t>(dst_ptr, dst_stride, n, value_bits,
                                           max_threads);
      return;
    case 4:
      cpu_fill_dense_field_typed<int64_t>(dst_ptr, dst_stride, n, value_bits,
                                          max_threads);
      return;
    case 5:
      cpu_fill_dense_field_typed<double>(dst_ptr, dst_stride, n, value_bits,
                                         max_threads);
      return;
    default:
      TI_ERROR("Native dense field fill received an unsupported value type.");
  }
}

void Program::fill_dense_field_packed(SNode *dst,
                                      int value_type,
                                      uint64_t value_bits,
                                      std::size_t n,
                                      int lane_count) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  const Arch arch = compile_config().arch;
  TI_ERROR_IF(!native_dense_field_bulk_arch(arch),
              "Native packed dense field fill is only available on CPU, CUDA, "
              "and Vulkan backends.");
  TI_ERROR_IF(!dst, "Native packed dense field fill received a null field.");
  const std::size_t item_bytes = primitive_value_type_size(value_type);
  TI_ERROR_IF(item_bytes == 0,
              "Native packed dense field fill received an unsupported value "
              "type.");
  const std::size_t scalar_items =
      dense_field_packed_scalar_items(n, lane_count,
                                      "Native packed dense field fill");
  const std::size_t bytes =
      dense_field_packed_bytes(value_type, n, lane_count,
                               "Native packed dense field fill");
  if (n == 0) {
    return;
  }
  check_dense_field_packed_stride(this, dst, value_type, lane_count,
                                  "Native packed dense field fill");
  if (arch == Arch::cuda) {
#ifdef TI_WITH_CUDA
    DevicePtr dst_device_ptr = get_dense_field_device_ptr(dst);
    DeviceAllocation alloc{dst_device_ptr.device, dst_device_ptr.alloc_id};
    auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
    TI_ERROR_IF(!base,
                "Native packed dense field fill received a null CUDA field "
                "data pointer.");
    auto *dst_raw =
        reinterpret_cast<uint8_t *>(base) + dst_device_ptr.offset;
    if (value_bits == 0) {
      CUDADriver::get_instance().memset(dst_raw, 0, bytes);
      return;
    }
    TI_ERROR_IF(value_type > 2,
                "Native packed dense field fill currently supports non-zero "
                "CUDA device fills only for 32-bit primitive fields.");
    CUDADriver::get_instance().memsetd32(
        dst_raw, static_cast<uint32_t>(value_bits), scalar_items);
    return;
#else
    TI_ERROR("Native CUDA packed dense field fill requires TI_WITH_CUDA=ON.");
#endif
  }
  if (arch == Arch::vulkan) {
    TI_ERROR_IF(value_bits != 0 && value_type > 2,
                "Native packed dense field fill currently supports non-zero "
                "Vulkan device fills only for 32-bit primitive fields.");
    DevicePtr dst_device_ptr = get_dense_field_device_ptr(dst);
    DeviceAllocation dst_alloc{dst_device_ptr.device, dst_device_ptr.alloc_id};
    TI_ERROR_IF(!dst_alloc.device,
                "Native packed dense field fill received null Vulkan "
                "storage.");
    enqueue_compute_op_lambda(
        [dst_alloc, dst_offset = dst_device_ptr.offset, bytes,
         value = static_cast<uint32_t>(value_bits)](
            Device * /*device*/, CommandList *cmdlist) {
          cmdlist->buffer_fill(dst_alloc.get_ptr(dst_offset), bytes, value);
          cmdlist->buffer_barrier(dst_alloc.get_ptr(dst_offset), bytes);
        },
        {});
    return;
  }
  auto *dst_ptr = map_cpu_dense_field_packed(
      this, dst, value_type, n, lane_count, "Native packed dense field fill");
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  switch (value_type) {
    case 0:
      cpu_fill_dense_field_typed<int32_t>(dst_ptr, item_bytes, scalar_items,
                                          value_bits, max_threads);
      return;
    case 1:
      cpu_fill_dense_field_typed<float>(dst_ptr, item_bytes, scalar_items,
                                        value_bits, max_threads);
      return;
    case 2:
      cpu_fill_dense_field_typed<uint32_t>(dst_ptr, item_bytes, scalar_items,
                                           value_bits, max_threads);
      return;
    case 3:
      cpu_fill_dense_field_typed<uint64_t>(dst_ptr, item_bytes, scalar_items,
                                           value_bits, max_threads);
      return;
    case 4:
      cpu_fill_dense_field_typed<int64_t>(dst_ptr, item_bytes, scalar_items,
                                          value_bits, max_threads);
      return;
    case 5:
      cpu_fill_dense_field_typed<double>(dst_ptr, item_bytes, scalar_items,
                                         value_bits, max_threads);
      return;
    default:
      TI_ERROR(
          "Native packed dense field fill received an unsupported value type.");
  }
}

std::size_t Program::transform_affine_dense_field_packed(SNode *src,
                                                         SNode *dst,
                                                         int value_type,
                                                         std::size_t n,
                                                         int lane_count,
                                                         double scale,
                                                         double bias) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  const Arch arch = compile_config().arch;
  TI_ERROR_IF(!native_dense_field_bulk_arch(arch),
              "Native packed dense field transform is only available on CPU, "
              "CUDA, and Vulkan backends.");
  TI_ERROR_IF(!src || !dst,
              "Native packed dense field transform received a null field.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "Native packed dense field transform received an unsupported "
              "value type.");
  const std::size_t scalar_items =
      dense_field_packed_scalar_items(n, lane_count,
                                      "Native packed dense field transform");
  if (n == 0) {
    return 0;
  }
  check_dense_field_packed_stride(this, src, value_type, lane_count,
                                  "Native packed dense field transform");
  check_dense_field_packed_stride(this, dst, value_type, lane_count,
                                  "Native packed dense field transform");

  if (arch == Arch::vulkan) {
    return vulkan_transform_affine_dense_field_packed(
        src, dst, value_type, n, lane_count, scale, bias);
  }

  if (arch == Arch::cuda) {
#ifdef TI_WITH_CUDA
    TI_ERROR_IF(
        scalar_items >
            static_cast<std::size_t>(std::numeric_limits<int>::max()),
        "CUDA packed dense field transform currently supports at most INT_MAX "
        "scalar items.");
    auto raw_ptr = [this](DevicePtr ptr, const char *op_name) {
      TI_ERROR_IF(!ptr.device, "{} received a null dense field device.",
                  op_name);
      DeviceAllocation alloc{ptr.device, ptr.alloc_id};
      auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
      TI_ERROR_IF(!base, "{} received a null dense field data pointer.",
                  op_name);
      return static_cast<void *>(reinterpret_cast<uint8_t *>(base) +
                                 ptr.offset);
    };
    void *src_raw = raw_ptr(get_dense_field_device_ptr(src),
                            "CUDA packed dense field transform");
    void *dst_raw = raw_ptr(get_dense_field_device_ptr(dst),
                            "CUDA packed dense field transform");
    const auto cuda_value_type =
        static_cast<cuda::CudaTransformValueType>(value_type);
    TI_ERROR_IF(!cuda::driver_transform_available(),
                "CUDA packed dense field transform requires CUDA driver API "
                "support.");
    return cuda::driver_transform_affine(
        src_raw, dst_raw, static_cast<int>(scalar_items), cuda_value_type,
        scale, bias);
#else
    TI_ERROR("CUDA packed dense field transform requires TI_WITH_CUDA=ON.");
#endif
  }

  const auto *src_ptr = map_cpu_dense_field_packed(
      this, src, value_type, n, lane_count,
      "CPU native packed dense field transform");
  auto *dst_ptr = map_cpu_dense_field_packed(
      this, dst, value_type, n, lane_count,
      "CPU native packed dense field transform");
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((scalar_items + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel =
      cpu_use_parallel_simple_loop(scalar_items, target_threads);
  switch (value_type) {
    case 0:
      cpu_transform_run_typed<uint32_t>(
          reinterpret_cast<const uint32_t *>(src_ptr),
          reinterpret_cast<uint32_t *>(dst_ptr), scalar_items,
          static_cast<uint32_t>(static_cast<int32_t>(scale)),
          static_cast<uint32_t>(static_cast<int32_t>(bias)), use_parallel,
          target_threads, max_threads);
      return 0;
    case 1:
      cpu_transform_run_typed<float>(
          reinterpret_cast<const float *>(src_ptr),
          reinterpret_cast<float *>(dst_ptr), scalar_items,
          static_cast<float>(scale), static_cast<float>(bias), use_parallel,
          target_threads, max_threads);
      return 0;
    case 2:
      cpu_transform_run_typed<uint32_t>(
          reinterpret_cast<const uint32_t *>(src_ptr),
          reinterpret_cast<uint32_t *>(dst_ptr), scalar_items,
          static_cast<uint32_t>(scale), static_cast<uint32_t>(bias),
          use_parallel, target_threads, max_threads);
      return 0;
    case 3:
      cpu_transform_run_typed<uint64_t>(
          reinterpret_cast<const uint64_t *>(src_ptr),
          reinterpret_cast<uint64_t *>(dst_ptr), scalar_items,
          static_cast<uint64_t>(scale), static_cast<uint64_t>(bias),
          use_parallel, target_threads, max_threads);
      return 0;
    case 4:
      cpu_transform_run_typed<uint64_t>(
          reinterpret_cast<const uint64_t *>(src_ptr),
          reinterpret_cast<uint64_t *>(dst_ptr), scalar_items,
          static_cast<uint64_t>(static_cast<int64_t>(scale)),
          static_cast<uint64_t>(static_cast<int64_t>(bias)), use_parallel,
          target_threads, max_threads);
      return 0;
    case 5:
      cpu_transform_run_typed<double>(
          reinterpret_cast<const double *>(src_ptr),
          reinterpret_cast<double *>(dst_ptr), scalar_items,
          static_cast<double>(scale), static_cast<double>(bias), use_parallel,
          target_threads, max_threads);
      return 0;
    default:
      TI_ERROR(
          "Native packed dense field transform received an unsupported value "
          "type.");
  }
  return 0;
}

void Program::copy_dense_field(SNode *dst,
                               SNode *src,
                               int value_type,
                               std::size_t n) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  const Arch arch = compile_config().arch;
  TI_ERROR_IF(!native_dense_field_bulk_arch(arch),
              "Native dense field copy is only available on CPU, CUDA, and "
              "Vulkan backends.");
  TI_ERROR_IF(!dst || !src,
              "Native dense field copy received a null field.");
  const std::size_t item_bytes = primitive_value_type_size(value_type);
  TI_ERROR_IF(item_bytes == 0,
              "Native dense field copy received an unsupported value type.");
  if (n == 0 || dst == src) {
    return;
  }
  if (n > std::numeric_limits<std::size_t>::max() / item_bytes) {
    TI_ERROR("Native dense field copy received an oversized request.");
  }
  ScopedRuntimeTransferStatistics transfer_statistics(
      this, RuntimeTransferKind::kDeviceToDevice, n * item_bytes);

  if (arch == Arch::cuda || arch == Arch::vulkan) {
    const std::size_t dst_stride = get_dense_field_stride(dst, item_bytes);
    const std::size_t src_stride = get_dense_field_stride(src, item_bytes);
    DevicePtr dst_ptr = get_dense_field_device_ptr(dst);
    DevicePtr src_ptr = get_dense_field_device_ptr(src);
    Device *device = program_impl_->get_compute_device();
    TI_ERROR_IF(!device || dst_ptr.device != device || src_ptr.device != device,
                "Native dense field copy received invalid device storage.");
    const std::size_t bytes = n * item_bytes;
    if (arch == Arch::vulkan) {
      if (dst_stride != item_bytes || src_stride != item_bytes) {
        vulkan_transform_affine_dense_field(src, dst, value_type, n, 1.0, 0.0);
        return;
      }
      DeviceAllocation dst_alloc{dst_ptr.device, dst_ptr.alloc_id};
      DeviceAllocation src_alloc{src_ptr.device, src_ptr.alloc_id};
      enqueue_compute_op_lambda(
          [dst_alloc, src_alloc, dst_offset = dst_ptr.offset,
           src_offset = src_ptr.offset,
           bytes](Device * /*device*/, CommandList *cmdlist) {
            cmdlist->buffer_copy(dst_alloc.get_ptr(dst_offset),
                                 src_alloc.get_ptr(src_offset), bytes);
            cmdlist->buffer_barrier(dst_alloc.get_ptr(dst_offset), bytes);
          },
          {});
      return;
    }
    if (dst_stride != item_bytes || src_stride != item_bytes) {
      cuda_device_transform_affine_dense_field(src, dst, value_type, n, 1.0,
                                               0.0);
      return;
    }
    Device::memcpy_direct(dst_ptr, src_ptr, bytes);
    return;
  }

  std::size_t dst_stride = 0;
  std::size_t src_stride = 0;
  auto *dst_ptr = map_cpu_dense_field(this, dst, value_type, n,
                                      "Native dense field copy", &dst_stride);
  const auto *src_ptr = map_cpu_dense_field(this, src, value_type, n,
                                           "Native dense field copy",
                                           &src_stride);
  const std::size_t bytes = n * item_bytes;
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  if (dst_stride == item_bytes && src_stride == item_bytes) {
    const std::size_t chunk_bytes =
        bytes <= (4 << 20) ? (1 << 20) : (256 << 10);
    const int target_threads = static_cast<int>(
        std::min<std::size_t>((bytes + chunk_bytes - 1) / chunk_bytes,
                              static_cast<std::size_t>(max_threads)));
    if (bytes >= (1 << 20) && target_threads > 1) {
      CpuCopyTaskContext ctx;
      ctx.dst = dst_ptr;
      ctx.src = src_ptr;
      ctx.bytes = bytes;
      ctx.num_threads = target_threads;
      auto pool = get_cpu_primitive_thread_pool(max_threads);
      pool->run(target_threads, target_threads, &ctx, cpu_copy_task);
      return;
    }
    std::memcpy(dst_ptr, src_ptr, bytes);
    return;
  }

  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  if (cpu_use_parallel_simple_loop(n, target_threads)) {
    CpuDenseFieldCopyTaskContext ctx;
    ctx.dst = dst_ptr;
    ctx.src = src_ptr;
    ctx.item_bytes = item_bytes;
    ctx.dst_stride = dst_stride;
    ctx.src_stride = src_stride;
    ctx.n = n;
    ctx.num_threads = target_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(target_threads, target_threads, &ctx, cpu_dense_field_copy_task);
    return;
  }
  for (std::size_t i = 0; i < n; ++i) {
    std::memcpy(dst_ptr + i * dst_stride, src_ptr + i * src_stride,
                item_bytes);
  }
}

void Program::copy_dense_field_packed(SNode *dst,
                                      SNode *src,
                                      int value_type,
                                      std::size_t n,
                                      int lane_count) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  const Arch arch = compile_config().arch;
  TI_ERROR_IF(!native_dense_field_bulk_arch(arch),
              "Native packed dense field copy is only available on CPU, CUDA, "
              "and Vulkan backends.");
  TI_ERROR_IF(!dst || !src,
              "Native packed dense field copy received a null field.");
  const std::size_t bytes =
      dense_field_packed_bytes(value_type, n, lane_count,
                               "Native packed dense field copy");
  if (n == 0 || dst == src) {
    return;
  }
  ScopedRuntimeTransferStatistics transfer_statistics(
      this, RuntimeTransferKind::kDeviceToDevice, bytes);
  check_dense_field_packed_stride(this, dst, value_type, lane_count,
                                  "Native packed dense field copy");
  check_dense_field_packed_stride(this, src, value_type, lane_count,
                                  "Native packed dense field copy");

  if (arch == Arch::cuda || arch == Arch::vulkan) {
    DevicePtr dst_ptr = get_dense_field_device_ptr(dst);
    DevicePtr src_ptr = get_dense_field_device_ptr(src);
    Device *device = program_impl_->get_compute_device();
    TI_ERROR_IF(!device || dst_ptr.device != device || src_ptr.device != device,
                "Native packed dense field copy received invalid device "
                "storage.");
    if (arch == Arch::vulkan) {
      DeviceAllocation dst_alloc{dst_ptr.device, dst_ptr.alloc_id};
      DeviceAllocation src_alloc{src_ptr.device, src_ptr.alloc_id};
      enqueue_compute_op_lambda(
          [dst_alloc, src_alloc, dst_offset = dst_ptr.offset,
           src_offset = src_ptr.offset,
           bytes](Device * /*device*/, CommandList *cmdlist) {
            cmdlist->buffer_copy(dst_alloc.get_ptr(dst_offset),
                                 src_alloc.get_ptr(src_offset), bytes);
            cmdlist->buffer_barrier(dst_alloc.get_ptr(dst_offset), bytes);
          },
          {});
      return;
    }
    Device::memcpy_direct(dst_ptr, src_ptr, bytes);
    return;
  }

  auto *dst_ptr = map_cpu_dense_field_packed(
      this, dst, value_type, n, lane_count, "Native packed dense field copy");
  const auto *src_ptr = map_cpu_dense_field_packed(
      this, src, value_type, n, lane_count, "Native packed dense field copy");
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const std::size_t chunk_bytes =
      bytes <= (4 << 20) ? (1 << 20) : (256 << 10);
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((bytes + chunk_bytes - 1) / chunk_bytes,
                            static_cast<std::size_t>(max_threads)));
  if (bytes >= (1 << 20) && target_threads > 1) {
    CpuCopyTaskContext ctx;
    ctx.dst = dst_ptr;
    ctx.src = src_ptr;
    ctx.bytes = bytes;
    ctx.num_threads = target_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(target_threads, target_threads, &ctx, cpu_copy_task);
    return;
  }
  std::memcpy(dst_ptr, src_ptr, bytes);
}

void Program::copy_dense_field_to_ndarray(Ndarray *dst,
                                          SNode *src,
                                          int value_type,
                                          std::size_t n,
                                          int lane_count) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  auto resource_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!dst || !src,
              "Native dense-field-to-ndarray copy received a null operand.");
  const Arch arch = compile_config().arch;
  TI_ERROR_IF(!native_dense_field_bulk_arch(arch),
              "Native dense-field-to-ndarray copy is only available on CPU, "
              "CUDA, and Vulkan backends.");
  const std::size_t bytes = dense_field_packed_bytes(
      value_type, n, lane_count, "Native dense-field-to-ndarray copy");
  TI_ERROR_IF(dst->get_nelement() * dst->get_element_size() != bytes,
              "Native dense-field-to-ndarray copy requires an exactly sized "
              "destination ndarray.");
  check_dense_field_packed_stride(
      this, src, value_type, lane_count,
      "Native dense-field-to-ndarray copy");
  if (bytes == 0) {
    return;
  }
  auto leases = acquire_ndarray_leases({dst});
  ScopedRuntimeTransferStatistics transfer_statistics(
      this, RuntimeTransferKind::kDeviceToDevice, bytes);

  if (arch == Arch::cuda || arch == Arch::vulkan) {
    DevicePtr src_ptr = get_dense_field_device_ptr(src);
    const DeviceAllocation dst_alloc = dst->ndarray_alloc_;
    Device *device = program_impl_->get_compute_device();
    TI_ERROR_IF(!device || src_ptr.device != device ||
                    dst_alloc.device != device,
                "Native dense-field-to-ndarray copy received invalid device "
                "storage.");
    if (arch == Arch::vulkan) {
      DeviceAllocation src_alloc{src_ptr.device, src_ptr.alloc_id};
      enqueue_compute_op_lambda(
          [dst_alloc, src_alloc, src_offset = src_ptr.offset,
           bytes](Device * /*device*/, CommandList *cmdlist) {
            cmdlist->buffer_copy(dst_alloc.get_ptr(0),
                                 src_alloc.get_ptr(src_offset), bytes);
            cmdlist->buffer_barrier(dst_alloc);
          },
          {});
      mark_runtime_submission_pending();
      pin_ndarray_launch_leases(leases);
      return;
    }
    Device::memcpy_direct(dst_alloc.get_ptr(0), src_ptr, bytes);
    return;
  }

  auto *dst_ptr = reinterpret_cast<uint8_t *>(
      program_impl_->get_device_alloc_info_ptr(dst->ndarray_alloc_));
  const auto *src_ptr = map_cpu_dense_field_packed(
      this, src, value_type, n, lane_count,
      "Native dense-field-to-ndarray copy");
  TI_ERROR_IF(!dst_ptr,
              "Native dense-field-to-ndarray copy received null CPU ndarray "
              "storage.");
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const std::size_t chunk_bytes =
      bytes <= (4 << 20) ? (1 << 20) : (256 << 10);
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((bytes + chunk_bytes - 1) / chunk_bytes,
                            static_cast<std::size_t>(max_threads)));
  if (bytes >= (1 << 20) && target_threads > 1) {
    CpuCopyTaskContext ctx;
    ctx.dst = dst_ptr;
    ctx.src = src_ptr;
    ctx.bytes = bytes;
    ctx.num_threads = target_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(target_threads, target_threads, &ctx, cpu_copy_task);
    return;
  }
  std::memcpy(dst_ptr, src_ptr, bytes);
}

void Program::copy_ndarray_to_dense_field(SNode *dst,
                                          Ndarray *src,
                                          int value_type,
                                          std::size_t n,
                                          int lane_count) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  auto resource_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!dst || !src,
              "Native ndarray-to-dense-field copy received a null operand.");
  const Arch arch = compile_config().arch;
  TI_ERROR_IF(!native_dense_field_bulk_arch(arch),
              "Native ndarray-to-dense-field copy is only available on CPU, "
              "CUDA, and Vulkan backends.");
  const std::size_t bytes = dense_field_packed_bytes(
      value_type, n, lane_count, "Native ndarray-to-dense-field copy");
  TI_ERROR_IF(src->get_nelement() * src->get_element_size() != bytes,
              "Native ndarray-to-dense-field copy requires an exactly sized "
              "source ndarray.");
  check_dense_field_packed_stride(
      this, dst, value_type, lane_count,
      "Native ndarray-to-dense-field copy");
  if (bytes == 0) {
    return;
  }
  auto leases = acquire_ndarray_leases({src});
  ScopedRuntimeTransferStatistics transfer_statistics(
      this, RuntimeTransferKind::kDeviceToDevice, bytes);

  if (arch == Arch::cuda || arch == Arch::vulkan) {
    DevicePtr dst_ptr = get_dense_field_device_ptr(dst);
    const DeviceAllocation src_alloc = src->ndarray_alloc_;
    Device *device = program_impl_->get_compute_device();
    TI_ERROR_IF(!device || dst_ptr.device != device ||
                    src_alloc.device != device,
                "Native ndarray-to-dense-field copy received invalid device "
                "storage.");
    if (arch == Arch::vulkan) {
      DeviceAllocation dst_alloc{dst_ptr.device, dst_ptr.alloc_id};
      enqueue_compute_op_lambda(
          [dst_alloc, src_alloc, dst_offset = dst_ptr.offset,
           bytes](Device * /*device*/, CommandList *cmdlist) {
            cmdlist->buffer_copy(dst_alloc.get_ptr(dst_offset),
                                 src_alloc.get_ptr(0), bytes);
            cmdlist->buffer_barrier(dst_alloc.get_ptr(dst_offset), bytes);
          },
          {});
      mark_runtime_submission_pending();
      pin_ndarray_launch_leases(leases);
      return;
    }
    Device::memcpy_direct(dst_ptr, src_alloc.get_ptr(0), bytes);
    return;
  }

  auto *dst_ptr = map_cpu_dense_field_packed(
      this, dst, value_type, n, lane_count,
      "Native ndarray-to-dense-field copy");
  const auto *src_ptr = reinterpret_cast<const uint8_t *>(
      program_impl_->get_device_alloc_info_ptr(src->ndarray_alloc_));
  TI_ERROR_IF(!src_ptr,
              "Native ndarray-to-dense-field copy received null CPU ndarray "
              "storage.");
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const std::size_t chunk_bytes =
      bytes <= (4 << 20) ? (1 << 20) : (256 << 10);
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((bytes + chunk_bytes - 1) / chunk_bytes,
                            static_cast<std::size_t>(max_threads)));
  if (bytes >= (1 << 20) && target_threads > 1) {
    CpuCopyTaskContext ctx;
    ctx.dst = dst_ptr;
    ctx.src = src_ptr;
    ctx.bytes = bytes;
    ctx.num_threads = target_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(target_threads, target_threads, &ctx, cpu_copy_task);
    return;
  }
  std::memcpy(dst_ptr, src_ptr, bytes);
}
void Program::copy_dense_field_from_host(SNode *dst,
                                         std::uintptr_t src,
                                         std::size_t src_bytes,
                                         int value_type,
                                         std::size_t n) {
  const Arch arch = compile_config().arch;
  std::optional<RuntimeResourceSubmissionGuard> resource_submission_guard;
  if (arch == Arch::vulkan) {
    resource_submission_guard.emplace(
        acquire_runtime_resource_submission_guard());
  }
  TI_ERROR_IF(!native_dense_field_bulk_arch(arch),
              "Native dense field host copy is only available on CPU, CUDA, "
              "and Vulkan backends.");
  TI_ERROR_IF(!dst || !src, "Native dense field host copy received a null "
                            "field or source pointer.");
  const std::size_t item_bytes = primitive_value_type_size(value_type);
  TI_ERROR_IF(item_bytes == 0,
              "Native dense field host copy received an unsupported value "
              "type.");
  TI_ERROR_IF(n > std::numeric_limits<std::size_t>::max() / item_bytes,
              "Native dense field host copy received an oversized request.");
  const std::size_t expected_bytes = n * item_bytes;
  TI_ERROR_IF(src_bytes != expected_bytes,
              "Native dense field host copy received mismatched source size.");
  if (n == 0) {
    return;
  }
  ScopedRuntimeTransferStatistics transfer_statistics(
      this, RuntimeTransferKind::kHostToDevice, src_bytes);
  if (arch == Arch::cuda || arch == Arch::vulkan) {
    const std::size_t dst_stride = get_dense_field_stride(dst, item_bytes);
    TI_ERROR_IF(dst_stride != item_bytes,
                "Native dense field path currently supports device host copy "
                "only for contiguous dense fields.");
    DevicePtr dst_ptr = get_dense_field_device_ptr(dst);
    Device *device = program_impl_->get_compute_device();
    TI_ERROR_IF(!device || dst_ptr.device != device,
                "Native dense field host copy received invalid device "
                "storage.");
    if (arch == Arch::vulkan) {
      auto &staging = dense_field_staging_resource();
      std::lock_guard<std::mutex> lock(staging.mutex);
      if (auto new_staging = ensure_vulkan_dense_host_staging(
              device, staging.upload, staging.upload_capacity, src_bytes,
              /*host_write=*/true, /*host_read=*/false, AllocUsage::Upload)) {
        staging.upload = std::move(new_staging);
      }
      void *mapped{nullptr};
      RhiResult res = device->map(*staging.upload, &mapped);
      TI_ERROR_IF(res != RhiResult::success || !mapped,
                  "Native dense field host copy failed to map staging buffer: "
                  "{}",
                  res);
      std::memcpy(mapped, reinterpret_cast<const void *>(src), src_bytes);
      device->unmap(*staging.upload);
      Stream *stream = device->get_compute_stream();
      auto [cmdlist, cmd_res] = stream->new_command_list_unique();
      TI_ERROR_IF(cmd_res != RhiResult::success,
                  "Native dense field host copy failed to create command "
                  "list: {}",
                  cmd_res);
      cmdlist->buffer_copy(dst_ptr, staging.upload->get_ptr(0), src_bytes);
      stream->submit_synced(cmdlist.get());
      return;
    }
    const void *src_ptr = reinterpret_cast<const void *>(src);
    std::size_t size = src_bytes;
    const RhiResult res = device->upload_data(&dst_ptr, &src_ptr, &size, 1);
    TI_ERROR_IF(res != RhiResult::success,
                "Native dense field host copy failed: {}", res);
    return;
  }
  std::size_t dst_stride = 0;
  auto *dst_ptr = map_cpu_dense_field(this, dst, value_type, n,
                                      "Native dense field host copy",
                                      &dst_stride);
  const auto *src_ptr = reinterpret_cast<const uint8_t *>(src);
  if (dst_stride == item_bytes) {
    std::memcpy(dst_ptr, src_ptr, src_bytes);
    return;
  }
  for (std::size_t i = 0; i < n; ++i) {
    std::memcpy(dst_ptr + i * dst_stride, src_ptr + i * item_bytes, item_bytes);
  }
}

void Program::copy_dense_field_packed_from_host(SNode *dst,
                                                std::uintptr_t src,
                                                std::size_t src_bytes,
                                                int value_type,
                                                std::size_t n,
                                                int lane_count) {
  const Arch arch = compile_config().arch;
  std::optional<RuntimeResourceSubmissionGuard> resource_submission_guard;
  if (arch == Arch::vulkan) {
    resource_submission_guard.emplace(
        acquire_runtime_resource_submission_guard());
  }
  TI_ERROR_IF(!native_dense_field_bulk_arch(arch),
              "Native packed dense field host copy is only available on CPU, "
              "CUDA, and Vulkan backends.");
  TI_ERROR_IF(!dst || !src,
              "Native packed dense field host copy received a null field or "
              "source pointer.");
  const std::size_t expected_bytes =
      dense_field_packed_bytes(value_type, n, lane_count,
                               "Native packed dense field host copy");
  TI_ERROR_IF(src_bytes != expected_bytes,
              "Native packed dense field host copy received mismatched source "
              "size.");
  if (n == 0) {
    return;
  }
  ScopedRuntimeTransferStatistics transfer_statistics(
      this, RuntimeTransferKind::kHostToDevice, src_bytes);
  check_dense_field_packed_stride(this, dst, value_type, lane_count,
                                  "Native packed dense field host copy");
  if (arch == Arch::cuda || arch == Arch::vulkan) {
    DevicePtr dst_ptr = get_dense_field_device_ptr(dst);
    Device *device = program_impl_->get_compute_device();
    TI_ERROR_IF(!device || dst_ptr.device != device,
                "Native packed dense field host copy received invalid device "
                "storage.");
    if (arch == Arch::vulkan) {
      auto &staging = dense_field_staging_resource();
      std::lock_guard<std::mutex> lock(staging.mutex);
      if (auto new_staging = ensure_vulkan_dense_host_staging(
              device, staging.upload, staging.upload_capacity, src_bytes,
              /*host_write=*/true, /*host_read=*/false, AllocUsage::Upload)) {
        staging.upload = std::move(new_staging);
      }
      void *mapped{nullptr};
      RhiResult res = device->map(*staging.upload, &mapped);
      TI_ERROR_IF(res != RhiResult::success || !mapped,
                  "Native packed dense field host copy failed to map staging "
                  "buffer: {}",
                  res);
      std::memcpy(mapped, reinterpret_cast<const void *>(src), src_bytes);
      device->unmap(*staging.upload);
      Stream *stream = device->get_compute_stream();
      auto [cmdlist, cmd_res] = stream->new_command_list_unique();
      TI_ERROR_IF(cmd_res != RhiResult::success,
                  "Native packed dense field host copy failed to create "
                  "command list: {}",
                  cmd_res);
      cmdlist->buffer_copy(dst_ptr, staging.upload->get_ptr(0), src_bytes);
      stream->submit_synced(cmdlist.get());
      return;
    }
    const void *src_ptr = reinterpret_cast<const void *>(src);
    std::size_t size = src_bytes;
    const RhiResult res = device->upload_data(&dst_ptr, &src_ptr, &size, 1);
    TI_ERROR_IF(res != RhiResult::success,
                "Native packed dense field host copy failed: {}", res);
    return;
  }
  auto *dst_ptr = map_cpu_dense_field_packed(
      this, dst, value_type, n, lane_count,
      "Native packed dense field host copy");
  std::memcpy(dst_ptr, reinterpret_cast<const void *>(src), src_bytes);
}

void Program::copy_dense_field_to_host(SNode *src,
                                       std::uintptr_t dst,
                                       std::size_t dst_bytes,
                                       int value_type,
                                       std::size_t n) {
  const Arch arch = compile_config().arch;
  std::optional<RuntimeResourceSubmissionGuard> resource_submission_guard;
  if (arch == Arch::vulkan) {
    resource_submission_guard.emplace(
        acquire_runtime_resource_submission_guard());
  }
  TI_ERROR_IF(!native_dense_field_bulk_arch(arch),
              "Native dense field host readback is only available on CPU, "
              "CUDA, and Vulkan backends.");
  TI_ERROR_IF(!src || !dst,
              "Native dense field host readback received a null field or "
              "destination pointer.");
  const std::size_t item_bytes = primitive_value_type_size(value_type);
  TI_ERROR_IF(item_bytes == 0,
              "Native dense field host readback received an unsupported "
              "value type.");
  TI_ERROR_IF(n > std::numeric_limits<std::size_t>::max() / item_bytes,
              "Native dense field host readback received an oversized "
              "request.");
  const std::size_t expected_bytes = n * item_bytes;
  TI_ERROR_IF(dst_bytes != expected_bytes,
              "Native dense field host readback received mismatched "
              "destination size.");
  if (n == 0) {
    return;
  }
  ScopedRuntimeTransferStatistics transfer_statistics(
      this, RuntimeTransferKind::kDeviceToHost, dst_bytes);
  if (arch == Arch::cuda || arch == Arch::vulkan) {
    const std::size_t src_stride = get_dense_field_stride(src, item_bytes);
    TI_ERROR_IF(src_stride != item_bytes,
                "Native dense field path currently supports device host "
                "readback only for contiguous dense fields.");
    DevicePtr src_ptr = get_dense_field_device_ptr(src);
    Device *device = program_impl_->get_compute_device();
    TI_ERROR_IF(!device || src_ptr.device != device,
                "Native dense field host readback received invalid device "
                "storage.");
    if (arch == Arch::vulkan) {
      auto &staging = dense_field_staging_resource();
      std::lock_guard<std::mutex> lock(staging.mutex);
      if (auto new_staging = ensure_vulkan_dense_host_staging(
              device, staging.readback, staging.readback_capacity, dst_bytes,
              /*host_write=*/false, /*host_read=*/true, AllocUsage::None)) {
        staging.readback = std::move(new_staging);
      }
      Stream *stream = device->get_compute_stream();
      auto [cmdlist, cmd_res] = stream->new_command_list_unique();
      TI_ERROR_IF(cmd_res != RhiResult::success,
                  "Native dense field host readback failed to create command "
                  "list: {}",
                  cmd_res);
      cmdlist->buffer_copy(
          staging.readback->get_ptr(0), src_ptr, dst_bytes);
      stream->submit_synced(cmdlist.get());
      void *mapped{nullptr};
      RhiResult res = device->map(*staging.readback, &mapped);
      TI_ERROR_IF(res != RhiResult::success || !mapped,
                  "Native dense field host readback failed to map staging "
                  "buffer: {}",
                  res);
      std::memcpy(reinterpret_cast<void *>(dst), mapped, dst_bytes);
      device->unmap(*staging.readback);
      return;
    }
    void *dst_ptr = reinterpret_cast<void *>(dst);
    std::size_t size = dst_bytes;
    const RhiResult res = device->readback_data(&src_ptr, &dst_ptr, &size, 1);
    TI_ERROR_IF(res != RhiResult::success,
                "Native dense field host readback failed: {}", res);
    return;
  }
  std::size_t src_stride = 0;
  const auto *src_ptr = map_cpu_dense_field(this, src, value_type, n,
                                           "Native dense field host readback",
                                           &src_stride);
  auto *dst_ptr = reinterpret_cast<uint8_t *>(dst);
  if (src_stride == item_bytes) {
    std::memcpy(dst_ptr, src_ptr, dst_bytes);
    return;
  }
  for (std::size_t i = 0; i < n; ++i) {
    std::memcpy(dst_ptr + i * item_bytes, src_ptr + i * src_stride,
                item_bytes);
  }
}

void Program::copy_dense_field_packed_to_host(SNode *src,
                                              std::uintptr_t dst,
                                              std::size_t dst_bytes,
                                              int value_type,
                                              std::size_t n,
                                              int lane_count) {
  const Arch arch = compile_config().arch;
  std::optional<RuntimeResourceSubmissionGuard> resource_submission_guard;
  if (arch == Arch::vulkan) {
    resource_submission_guard.emplace(
        acquire_runtime_resource_submission_guard());
  }
  TI_ERROR_IF(!native_dense_field_bulk_arch(arch),
              "Native packed dense field host readback is only available on "
              "CPU, CUDA, and Vulkan backends.");
  TI_ERROR_IF(!src || !dst,
              "Native packed dense field host readback received a null field "
              "or destination pointer.");
  const std::size_t expected_bytes =
      dense_field_packed_bytes(value_type, n, lane_count,
                               "Native packed dense field host readback");
  TI_ERROR_IF(dst_bytes != expected_bytes,
              "Native packed dense field host readback received mismatched "
              "destination size.");
  if (n == 0) {
    return;
  }
  ScopedRuntimeTransferStatistics transfer_statistics(
      this, RuntimeTransferKind::kDeviceToHost, dst_bytes);
  check_dense_field_packed_stride(this, src, value_type, lane_count,
                                  "Native packed dense field host readback");
  if (arch == Arch::cuda || arch == Arch::vulkan) {
    DevicePtr src_ptr = get_dense_field_device_ptr(src);
    Device *device = program_impl_->get_compute_device();
    TI_ERROR_IF(!device || src_ptr.device != device,
                "Native packed dense field host readback received invalid "
                "device storage.");
    if (arch == Arch::vulkan) {
      auto &staging = dense_field_staging_resource();
      std::lock_guard<std::mutex> lock(staging.mutex);
      if (auto new_staging = ensure_vulkan_dense_host_staging(
              device, staging.readback, staging.readback_capacity, dst_bytes,
              /*host_write=*/false, /*host_read=*/true, AllocUsage::None)) {
        staging.readback = std::move(new_staging);
      }
      Stream *stream = device->get_compute_stream();
      auto [cmdlist, cmd_res] = stream->new_command_list_unique();
      TI_ERROR_IF(cmd_res != RhiResult::success,
                  "Native packed dense field host readback failed to create "
                  "command list: {}",
                  cmd_res);
      cmdlist->buffer_copy(
          staging.readback->get_ptr(0), src_ptr, dst_bytes);
      stream->submit_synced(cmdlist.get());
      void *mapped{nullptr};
      RhiResult res = device->map(*staging.readback, &mapped);
      TI_ERROR_IF(res != RhiResult::success || !mapped,
                  "Native packed dense field host readback failed to map "
                  "staging buffer: {}",
                  res);
      std::memcpy(reinterpret_cast<void *>(dst), mapped, dst_bytes);
      device->unmap(*staging.readback);
      return;
    }
    void *dst_ptr = reinterpret_cast<void *>(dst);
    std::size_t size = dst_bytes;
    const RhiResult res = device->readback_data(&src_ptr, &dst_ptr, &size, 1);
    TI_ERROR_IF(res != RhiResult::success,
                "Native packed dense field host readback failed: {}", res);
    return;
  }
  const auto *src_ptr = map_cpu_dense_field_packed(
      this, src, value_type, n, lane_count,
      "Native packed dense field host readback");
  std::memcpy(reinterpret_cast<void *>(dst), src_ptr, dst_bytes);
}

std::size_t Program::cpu_compact_ndarray(Ndarray *values,
                                         Ndarray *flags,
                                         Ndarray *output,
                                         Ndarray *count,
                                         int value_type) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native compact is only available on CPU backends.");
  TI_ERROR_IF(!values || !flags || !output || !count,
              "CPU native compact received a null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || flags->shape.size() != 1 ||
                  output->shape.size() != 1 || count->shape.size() != 1,
              "CPU native compact expects 1D ndarrays.");
  TI_ERROR_IF(values->get_nelement() != flags->get_nelement(),
              "CPU native compact values and flags must have the same length.");
  TI_ERROR_IF(output->get_nelement() < values->get_nelement(),
              "CPU native compact output must have at least input length.");
  TI_ERROR_IF(count->get_nelement() < 1,
              "CPU native compact count must contain at least one item.");
  const std::size_t value_bytes =
      (value_type == 0 || value_type == 1 || value_type == 2)
          ? sizeof(uint32_t)
          : (value_type == 3 || value_type == 4 || value_type == 5)
                ? sizeof(uint64_t)
                : 0;
  TI_ERROR_IF(value_bytes == 0,
              "CPU native compact received an unsupported value type.");
  const std::size_t item_bytes = values->get_element_size();
  TI_ERROR_IF(item_bytes == 0 || item_bytes % sizeof(uint32_t) != 0 ||
                  output->get_element_size() != item_bytes ||
                  flags->get_element_size() != sizeof(int32_t) ||
                  count->get_element_size() != sizeof(int32_t),
              "CPU native compact received mismatched value/flag/count dtypes "
              "or a non-4-byte-aligned payload.");

  auto *values_ptr =
      reinterpret_cast<const uint8_t *>(get_ndarray_data_ptr_as_int(values));
  auto *flags_ptr =
      reinterpret_cast<int32_t *>(get_ndarray_data_ptr_as_int(flags));
  auto *output_ptr =
      reinterpret_cast<uint8_t *>(get_ndarray_data_ptr_as_int(output));
  auto *count_ptr =
      reinterpret_cast<int32_t *>(get_ndarray_data_ptr_as_int(count));
  TI_ERROR_IF(!values_ptr || !flags_ptr || !output_ptr || !count_ptr,
              "CPU native compact received a null data pointer.");

  std::size_t written = 0;
  const std::size_t n = values->get_nelement();
  for (std::size_t i = 0; i < n; ++i) {
    if (flags_ptr[i] != 0) {
      std::memcpy(output_ptr + written * item_bytes,
                  values_ptr + i * item_bytes, item_bytes);
      written++;
    }
  }
  TI_ERROR_IF(written > static_cast<std::size_t>(
                            std::numeric_limits<int32_t>::max()),
              "CPU native compact output count exceeds i32 range.");
  count_ptr[0] = static_cast<int32_t>(written);
  return 0;
}

std::size_t Program::cpu_compact_dense_field(SNode *values,
                                             SNode *flags,
                                             SNode *output,
                                             SNode *count,
                                             int value_type,
                                             std::size_t n) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native dense field compact is only available on CPU "
              "backends.");
  TI_ERROR_IF(!values || !flags || !output || !count,
              "CPU native dense field compact received a null field.");
  const std::size_t item_bytes = primitive_value_type_size(value_type);
  TI_ERROR_IF(item_bytes == 0 || item_bytes % sizeof(uint32_t) != 0,
              "CPU native dense field compact received an unsupported value "
              "type.");
  std::size_t values_stride = 0;
  std::size_t flags_stride = 0;
  std::size_t output_stride = 0;
  std::size_t count_stride = 0;
  const auto *values_ptr = map_cpu_dense_field(
      this, values, value_type, n, "CPU native dense field compact values",
      &values_stride);
  const auto *flags_ptr = map_cpu_dense_field(
      this, flags, 0, n, "CPU native dense field compact flags",
      &flags_stride);
  auto *output_ptr = map_cpu_dense_field(
      this, output, value_type, n, "CPU native dense field compact output",
      &output_stride);
  auto *count_ptr = map_cpu_dense_field(
      this, count, 0, 1, "CPU native dense field compact count", &count_stride);

  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 131072;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = n >= 262144 && target_threads > 1;
  std::size_t workspace_bytes = 0;
  std::size_t written = 0;
  switch (value_type) {
    case 0:
      written = cpu_compact_dense_field_typed<int32_t>(
          values_ptr, values_stride, flags_ptr, flags_stride, output_ptr,
          output_stride, n, max_threads, target_threads, use_parallel,
          &workspace_bytes);
      break;
    case 1:
      written = cpu_compact_dense_field_typed<float>(
          values_ptr, values_stride, flags_ptr, flags_stride, output_ptr,
          output_stride, n, max_threads, target_threads, use_parallel,
          &workspace_bytes);
      break;
    case 2:
      written = cpu_compact_dense_field_typed<uint32_t>(
          values_ptr, values_stride, flags_ptr, flags_stride, output_ptr,
          output_stride, n, max_threads, target_threads, use_parallel,
          &workspace_bytes);
      break;
    case 3:
      written = cpu_compact_dense_field_typed<uint64_t>(
          values_ptr, values_stride, flags_ptr, flags_stride, output_ptr,
          output_stride, n, max_threads, target_threads, use_parallel,
          &workspace_bytes);
      break;
    case 4:
      written = cpu_compact_dense_field_typed<int64_t>(
          values_ptr, values_stride, flags_ptr, flags_stride, output_ptr,
          output_stride, n, max_threads, target_threads, use_parallel,
          &workspace_bytes);
      break;
    case 5:
      written = cpu_compact_dense_field_typed<double>(
          values_ptr, values_stride, flags_ptr, flags_stride, output_ptr,
          output_stride, n, max_threads, target_threads, use_parallel,
          &workspace_bytes);
      break;
    default:
      TI_ERROR("CPU native dense field compact received an unsupported value "
               "type.");
  }
  TI_ERROR_IF(written > static_cast<std::size_t>(
                            std::numeric_limits<int32_t>::max()),
              "CPU native dense field compact output count exceeds i32 range.");
  *reinterpret_cast<int32_t *>(count_ptr) = static_cast<int32_t>(written);
  return workspace_bytes;
}

std::size_t Program::cpu_compact_i32_ndarray(Ndarray *values,
                                             Ndarray *flags,
                                             Ndarray *output,
                                             Ndarray *count) {
  return cpu_compact_ndarray(values, flags, output, count, 0);
}

std::size_t Program::cpu_compact_workspace_bytes() const {
  return 0;
}

bool Program::cpu_histogram_available() const {
  return arch_is_cpu(compile_config().arch);
}

std::size_t Program::cpu_histogram_i32_ndarray(Ndarray *values,
                                               Ndarray *bins) {
  return cpu_histogram_ndarray(values, bins, 0, 0);
}

std::size_t Program::cpu_histogram_ndarray(Ndarray *values,
                                           Ndarray *bins,
                                           int value_type,
                                           int bin_type) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native histogram is only available on CPU backends.");
  TI_ERROR_IF(!values || !bins,
              "CPU native histogram received a null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || bins->shape.size() != 1,
              "CPU native histogram expects 1D ndarrays.");
  TI_ERROR_IF(value_type != 0 && value_type != 2,
              "CPU native histogram currently supports only i32/u32 bin ids.");
  const std::size_t value_size = value_type == 2 ? sizeof(uint32_t)
                                                 : sizeof(int32_t);
  const std::size_t bin_size = histogram_bin_type_size(bin_type);
  TI_ERROR_IF(bin_size == 0,
              "CPU native histogram currently supports only i32/i64 bins.");
  TI_ERROR_IF(values->get_element_size() != value_size ||
                  bins->get_element_size() != bin_size,
              "CPU native histogram received mismatched value/bin dtypes.");
  TI_ERROR_IF(bins->get_nelement() == 0,
              "CPU native histogram expects at least one bin.");
  if (bin_type == 0) {
    TI_ERROR_IF(values->get_nelement() >
                    static_cast<std::size_t>(
                        std::numeric_limits<int32_t>::max()),
                "CPU native histogram input is too large for i32 bin counts.");
  }

  auto *values_ptr =
      reinterpret_cast<const void *>(get_ndarray_data_ptr_as_int(values));
  auto *bins_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(bins));
  TI_ERROR_IF(!values_ptr || !bins_ptr,
              "CPU native histogram received a null data pointer.");

  const std::size_t n = values->get_nelement();
  const std::size_t num_bins = bins->get_nelement();
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = num_bins <= 4096 &&
                            cpu_use_parallel_aggregation(n, target_threads);
  if (value_type == 2 && bin_type == 4) {
    return cpu_histogram_typed(static_cast<const uint32_t *>(values_ptr),
                               static_cast<int64_t *>(bins_ptr), n, num_bins,
                               max_threads, target_threads, use_parallel);
  }
  if (value_type == 2) {
    return cpu_histogram_typed(static_cast<const uint32_t *>(values_ptr),
                               static_cast<int32_t *>(bins_ptr), n, num_bins,
                               max_threads, target_threads, use_parallel);
  }
  if (bin_type == 4) {
    return cpu_histogram_typed(static_cast<const int32_t *>(values_ptr),
                               static_cast<int64_t *>(bins_ptr), n, num_bins,
                               max_threads, target_threads, use_parallel);
  }
  return cpu_histogram_typed(static_cast<const int32_t *>(values_ptr),
                             static_cast<int32_t *>(bins_ptr), n, num_bins,
                             max_threads, target_threads, use_parallel);
}

std::size_t Program::cpu_histogram_dense_field(SNode *values,
                                               SNode *bins,
                                               int value_type,
                                               int bin_type,
                                               std::size_t n,
                                               std::size_t num_bins) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native dense field histogram is only available on CPU "
              "backends.");
  TI_ERROR_IF(!values || !bins,
              "CPU native dense field histogram received a null field.");
  TI_ERROR_IF(value_type != 0 && value_type != 2,
              "CPU native dense field histogram currently supports only "
              "i32/u32 bin ids.");
  const std::size_t value_size = value_type == 2 ? sizeof(uint32_t)
                                                 : sizeof(int32_t);
  const std::size_t bin_size = histogram_bin_type_size(bin_type);
  TI_ERROR_IF(bin_size == 0,
              "CPU native dense field histogram currently supports only "
              "i32/i64 bins.");
  TI_ERROR_IF(num_bins == 0,
              "CPU native dense field histogram expects at least one bin.");
  if (bin_type == 0) {
    TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()),
                "CPU native dense field histogram input is too large for i32 "
                "bin counts.");
  }
  std::size_t value_stride = 0;
  std::size_t bin_stride = 0;
  const auto *values_ptr = map_cpu_dense_field(
      this, values, value_type, n, "CPU native dense field histogram",
      &value_stride);
  auto *bins_ptr = map_cpu_dense_field(
      this, bins, bin_type, num_bins, "CPU native dense field histogram",
      &bin_stride);
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool contiguous =
      value_stride == value_size && bin_stride == bin_size;
  const bool use_parallel =
      contiguous && num_bins <= 4096 &&
      cpu_use_parallel_aggregation(n, target_threads);
  if (value_type == 2 && bin_type == 4) {
    if (contiguous) {
      return cpu_histogram_typed(reinterpret_cast<const uint32_t *>(values_ptr),
                                 reinterpret_cast<int64_t *>(bins_ptr), n,
                                 num_bins, max_threads, target_threads,
                                 use_parallel);
    }
    return cpu_histogram_strided_typed<uint32_t, int64_t>(
        values_ptr, value_stride, bins_ptr, bin_stride, n, num_bins);
  }
  if (value_type == 2) {
    if (contiguous) {
      return cpu_histogram_typed(reinterpret_cast<const uint32_t *>(values_ptr),
                                 reinterpret_cast<int32_t *>(bins_ptr), n,
                                 num_bins, max_threads, target_threads,
                                 use_parallel);
    }
    return cpu_histogram_strided_typed<uint32_t, int32_t>(
        values_ptr, value_stride, bins_ptr, bin_stride, n, num_bins);
  }
  if (bin_type == 4) {
    if (contiguous) {
      return cpu_histogram_typed(reinterpret_cast<const int32_t *>(values_ptr),
                                 reinterpret_cast<int64_t *>(bins_ptr), n,
                                 num_bins, max_threads, target_threads,
                                 use_parallel);
    }
    return cpu_histogram_strided_typed<int32_t, int64_t>(
        values_ptr, value_stride, bins_ptr, bin_stride, n, num_bins);
  }
  if (contiguous) {
    return cpu_histogram_typed(reinterpret_cast<const int32_t *>(values_ptr),
                               reinterpret_cast<int32_t *>(bins_ptr), n,
                               num_bins, max_threads, target_threads,
                               use_parallel);
  }
  return cpu_histogram_strided_typed<int32_t, int32_t>(
      values_ptr, value_stride, bins_ptr, bin_stride, n, num_bins);
}

std::size_t Program::cpu_histogram_workspace_bytes() const {
  return 0;
}

bool Program::cpu_reduce_available() const {
  return arch_is_cpu(compile_config().arch);
}

std::size_t Program::cpu_reduce_ndarray(Ndarray *values,
                                        Ndarray *output,
                                        int value_type,
                                        int op) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native reduce is only available on CPU backends.");
  TI_ERROR_IF(!values || !output,
              "CPU native reduce received a null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || output->shape.size() != 1,
              "CPU native reduce expects 1D ndarrays.");
  TI_ERROR_IF(values->get_nelement() == 0,
              "CPU native reduce expects at least one input item.");
  TI_ERROR_IF(output->get_nelement() < 1,
              "CPU native reduce output must contain at least one item.");
  TI_ERROR_IF(values->get_element_size() != output->get_element_size(),
              "CPU native reduce expects matching input/output element sizes.");
  const std::size_t expected_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(expected_size == 0,
              "CPU native reduce received an unsupported value type.");
  TI_ERROR_IF(values->get_element_size() != expected_size,
              "CPU native reduce dtype does not match value type.");
  TI_ERROR_IF(op < 0 || op > 2,
              "CPU native reduce received an unsupported op.");

  const std::size_t n = values->get_nelement();
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = cpu_use_parallel_aggregation(n, target_threads);

  switch (value_type) {
    case 0:
      return cpu_reduce_typed(
          reinterpret_cast<int32_t *>(get_ndarray_data_ptr_as_int(values)),
          reinterpret_cast<int32_t *>(get_ndarray_data_ptr_as_int(output)), op,
          n, max_threads, target_threads, use_parallel);
    case 1:
      return cpu_reduce_typed(
          reinterpret_cast<float *>(get_ndarray_data_ptr_as_int(values)),
          reinterpret_cast<float *>(get_ndarray_data_ptr_as_int(output)), op, n,
          max_threads, target_threads, use_parallel);
    case 2:
      return cpu_reduce_typed(
          reinterpret_cast<uint32_t *>(get_ndarray_data_ptr_as_int(values)),
          reinterpret_cast<uint32_t *>(get_ndarray_data_ptr_as_int(output)), op,
          n, max_threads, target_threads, use_parallel);
    case 3:
      return cpu_reduce_typed(
          reinterpret_cast<uint64_t *>(get_ndarray_data_ptr_as_int(values)),
          reinterpret_cast<uint64_t *>(get_ndarray_data_ptr_as_int(output)), op,
          n, max_threads, target_threads, use_parallel);
    case 4:
      return cpu_reduce_typed(
          reinterpret_cast<int64_t *>(get_ndarray_data_ptr_as_int(values)),
          reinterpret_cast<int64_t *>(get_ndarray_data_ptr_as_int(output)), op,
          n, max_threads, target_threads, use_parallel);
    case 5:
      return cpu_reduce_typed(
          reinterpret_cast<double *>(get_ndarray_data_ptr_as_int(values)),
          reinterpret_cast<double *>(get_ndarray_data_ptr_as_int(output)), op,
          n, max_threads, target_threads, use_parallel);
  }
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::cpu_reduce_member_ndarray(Ndarray *values,
                                               Ndarray *output,
                                               int value_type,
                                               std::size_t offset,
                                               std::size_t stride,
                                               int op) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native strided reduce is only available on CPU backends.");
  check_reduce_member_request("CPU native", values, output, value_type, offset,
                              stride, op);

  const std::size_t n = values->get_nelement();
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = cpu_use_parallel_aggregation(n, target_threads);

  const auto values_addr = get_ndarray_data_ptr_as_int(values);
  const auto output_addr = get_ndarray_data_ptr_as_int(output);
  TI_ERROR_IF(!values_addr || !output_addr,
              "CPU native strided reduce received a null data pointer.");
  const auto *values_ptr = reinterpret_cast<const uint8_t *>(values_addr);
  switch (value_type) {
    case 0:
      return cpu_reduce_strided_typed<int32_t>(
          values_ptr, reinterpret_cast<int32_t *>(output_addr), op, n, offset,
          stride, max_threads, target_threads, use_parallel);
    case 1:
      return cpu_reduce_strided_typed<float>(
          values_ptr, reinterpret_cast<float *>(output_addr), op, n, offset,
          stride, max_threads, target_threads, use_parallel);
    case 2:
      return cpu_reduce_strided_typed<uint32_t>(
          values_ptr, reinterpret_cast<uint32_t *>(output_addr), op, n, offset,
          stride, max_threads, target_threads, use_parallel);
    case 3:
      return cpu_reduce_strided_typed<uint64_t>(
          values_ptr, reinterpret_cast<uint64_t *>(output_addr), op, n, offset,
          stride, max_threads, target_threads, use_parallel);
    case 4:
      return cpu_reduce_strided_typed<int64_t>(
          values_ptr, reinterpret_cast<int64_t *>(output_addr), op, n, offset,
          stride, max_threads, target_threads, use_parallel);
    case 5:
      return cpu_reduce_strided_typed<double>(
          values_ptr, reinterpret_cast<double *>(output_addr), op, n, offset,
          stride, max_threads, target_threads, use_parallel);
  }
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::cpu_reduce_strided_ndarray(Ndarray *values,
                                                Ndarray *output,
                                                int value_type,
                                                std::size_t values_offset,
                                                std::size_t values_stride,
                                                std::size_t output_offset,
                                                std::size_t output_stride,
                                                int op) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native strided reduce is only available on CPU backends.");
  check_reduce_strided_request("CPU native", values, output, value_type,
                               values_offset, values_stride, output_offset,
                               output_stride, op);

  const std::size_t n = values->get_nelement();
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = cpu_use_parallel_aggregation(n, target_threads);

  const auto values_addr = get_ndarray_data_ptr_as_int(values);
  const auto output_addr = get_ndarray_data_ptr_as_int(output);
  TI_ERROR_IF(!values_addr || !output_addr,
              "CPU native strided reduce received a null data pointer.");
  const auto *values_ptr = reinterpret_cast<const uint8_t *>(values_addr);
  auto *output_ptr = reinterpret_cast<uint8_t *>(output_addr + output_offset);
  switch (value_type) {
    case 0:
      return cpu_reduce_strided_typed<int32_t>(
          values_ptr, reinterpret_cast<int32_t *>(output_ptr), op, n,
          values_offset, values_stride, max_threads, target_threads,
          use_parallel);
    case 1:
      return cpu_reduce_strided_typed<float>(
          values_ptr, reinterpret_cast<float *>(output_ptr), op, n,
          values_offset, values_stride, max_threads, target_threads,
          use_parallel);
    case 2:
      return cpu_reduce_strided_typed<uint32_t>(
          values_ptr, reinterpret_cast<uint32_t *>(output_ptr), op, n,
          values_offset, values_stride, max_threads, target_threads,
          use_parallel);
    case 3:
      return cpu_reduce_strided_typed<uint64_t>(
          values_ptr, reinterpret_cast<uint64_t *>(output_ptr), op, n,
          values_offset, values_stride, max_threads, target_threads,
          use_parallel);
    case 4:
      return cpu_reduce_strided_typed<int64_t>(
          values_ptr, reinterpret_cast<int64_t *>(output_ptr), op, n,
          values_offset, values_stride, max_threads, target_threads,
          use_parallel);
    case 5:
      return cpu_reduce_strided_typed<double>(
          values_ptr, reinterpret_cast<double *>(output_ptr), op, n,
          values_offset, values_stride, max_threads, target_threads,
          use_parallel);
  }
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::cpu_reduce_dense_field(SNode *values,
                                            SNode *output,
                                            int value_type,
                                            std::size_t n,
                                            int op) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native dense field reduce is only available on CPU "
              "backends.");
  TI_ERROR_IF(n == 0,
              "CPU native dense field reduce expects at least one input item.");
  TI_ERROR_IF(op < 0 || op > 2,
              "CPU native dense field reduce received an unsupported op.");
  std::size_t values_stride = 0;
  const auto *values_ptr = map_cpu_dense_field(
      this, values, value_type, n, "CPU native dense field reduce",
      &values_stride);
  auto *output_ptr = map_cpu_dense_field(
      this, output, value_type, 1, "CPU native dense field reduce", nullptr);
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = cpu_use_parallel_aggregation(n, target_threads);
  switch (value_type) {
    case 0:
      if (values_stride == sizeof(int32_t)) {
        return cpu_reduce_typed(
            reinterpret_cast<const int32_t *>(values_ptr),
            reinterpret_cast<int32_t *>(output_ptr), op, n, max_threads,
            target_threads, use_parallel);
      }
      return cpu_reduce_strided_typed<int32_t>(
          values_ptr, reinterpret_cast<int32_t *>(output_ptr), op, n, 0,
          values_stride, max_threads, target_threads, use_parallel);
    case 1:
      if (values_stride == sizeof(float)) {
        return cpu_reduce_typed(reinterpret_cast<const float *>(values_ptr),
                                reinterpret_cast<float *>(output_ptr), op, n,
                                max_threads, target_threads, use_parallel);
      }
      return cpu_reduce_strided_typed<float>(
          values_ptr, reinterpret_cast<float *>(output_ptr), op, n, 0,
          values_stride, max_threads, target_threads, use_parallel);
    case 2:
      if (values_stride == sizeof(uint32_t)) {
        return cpu_reduce_typed(
            reinterpret_cast<const uint32_t *>(values_ptr),
            reinterpret_cast<uint32_t *>(output_ptr), op, n, max_threads,
            target_threads, use_parallel);
      }
      return cpu_reduce_strided_typed<uint32_t>(
          values_ptr, reinterpret_cast<uint32_t *>(output_ptr), op, n, 0,
          values_stride, max_threads, target_threads, use_parallel);
    case 3:
      if (values_stride == sizeof(uint64_t)) {
        return cpu_reduce_typed(
            reinterpret_cast<const uint64_t *>(values_ptr),
            reinterpret_cast<uint64_t *>(output_ptr), op, n, max_threads,
            target_threads, use_parallel);
      }
      return cpu_reduce_strided_typed<uint64_t>(
          values_ptr, reinterpret_cast<uint64_t *>(output_ptr), op, n, 0,
          values_stride, max_threads, target_threads, use_parallel);
    case 4:
      if (values_stride == sizeof(int64_t)) {
        return cpu_reduce_typed(
            reinterpret_cast<const int64_t *>(values_ptr),
            reinterpret_cast<int64_t *>(output_ptr), op, n, max_threads,
            target_threads, use_parallel);
      }
      return cpu_reduce_strided_typed<int64_t>(
          values_ptr, reinterpret_cast<int64_t *>(output_ptr), op, n, 0,
          values_stride, max_threads, target_threads, use_parallel);
    case 5:
      if (values_stride == sizeof(double)) {
        return cpu_reduce_typed(reinterpret_cast<const double *>(values_ptr),
                                reinterpret_cast<double *>(output_ptr), op, n,
                                max_threads, target_threads, use_parallel);
      }
      return cpu_reduce_strided_typed<double>(
          values_ptr, reinterpret_cast<double *>(output_ptr), op, n, 0,
          values_stride, max_threads, target_threads, use_parallel);
    default:
      TI_ERROR(
          "CPU native dense field reduce received an unsupported value type.");
  }
}

std::size_t Program::cpu_reduce_dense_field_packed(SNode *values,
                                                   SNode *output,
                                                   int value_type,
                                                   std::size_t n,
                                                   int lane_count,
                                                   int op) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native packed dense field reduce is only available on CPU "
              "backends.");
  TI_ERROR_IF(n == 0,
              "CPU native packed dense field reduce expects at least one "
              "input item.");
  TI_ERROR_IF(op < 0 || op > 2,
              "CPU native packed dense field reduce received an unsupported "
              "op.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CPU native packed dense field reduce received an unsupported "
              "value type.");
  const std::size_t item_bytes =
      dense_field_packed_bytes(value_type, 1, lane_count,
                               "CPU native packed dense field reduce");
  const auto *values_ptr = map_cpu_dense_field_packed(
      this, values, value_type, n, lane_count,
      "CPU native packed dense field reduce");
  auto *output_ptr = map_cpu_dense_field_packed(
      this, output, value_type, 1, lane_count,
      "CPU native packed dense field reduce");
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = cpu_use_parallel_aggregation(n, target_threads);
  std::size_t temp_bytes = 0;
  for (int lane = 0; lane < lane_count; ++lane) {
    const std::size_t lane_offset =
        static_cast<std::size_t>(lane) * value_size;
    auto *lane_output = output_ptr + lane_offset;
    switch (value_type) {
      case 0:
        temp_bytes = std::max(
            temp_bytes,
            cpu_reduce_strided_typed<int32_t>(
                values_ptr, reinterpret_cast<int32_t *>(lane_output), op, n,
                lane_offset, item_bytes, max_threads, target_threads,
                use_parallel));
        break;
      case 1:
        temp_bytes = std::max(
            temp_bytes,
            cpu_reduce_strided_typed<float>(
                values_ptr, reinterpret_cast<float *>(lane_output), op, n,
                lane_offset, item_bytes, max_threads, target_threads,
                use_parallel));
        break;
      case 2:
        temp_bytes = std::max(
            temp_bytes,
            cpu_reduce_strided_typed<uint32_t>(
                values_ptr, reinterpret_cast<uint32_t *>(lane_output), op, n,
                lane_offset, item_bytes, max_threads, target_threads,
                use_parallel));
        break;
      case 3:
        temp_bytes = std::max(
            temp_bytes,
            cpu_reduce_strided_typed<uint64_t>(
                values_ptr, reinterpret_cast<uint64_t *>(lane_output), op, n,
                lane_offset, item_bytes, max_threads, target_threads,
                use_parallel));
        break;
      case 4:
        temp_bytes = std::max(
            temp_bytes,
            cpu_reduce_strided_typed<int64_t>(
                values_ptr, reinterpret_cast<int64_t *>(lane_output), op, n,
                lane_offset, item_bytes, max_threads, target_threads,
                use_parallel));
        break;
      case 5:
        temp_bytes = std::max(
            temp_bytes,
            cpu_reduce_strided_typed<double>(
                values_ptr, reinterpret_cast<double *>(lane_output), op, n,
                lane_offset, item_bytes, max_threads, target_threads,
                use_parallel));
        break;
      default:
        TI_ERROR(
            "CPU native packed dense field reduce received an unsupported "
            "value type.");
    }
  }
  return temp_bytes;
}

std::size_t Program::cpu_reduce_workspace_bytes() const {
  return 0;
}

bool Program::cpu_check_count_available() const {
  return arch_is_cpu(compile_config().arch);
}

std::size_t Program::cpu_check_count_ndarray(Ndarray *values,
                                             Ndarray *output,
                                             int value_type,
                                             int check_op,
                                             int lower,
                                             int upper) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native check_count is only available on CPU backends.");
  TI_ERROR_IF(!values || !output,
              "CPU native check_count received a null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || output->shape.size() != 1,
              "CPU native check_count expects 1D ndarrays.");
  TI_ERROR_IF(values->get_nelement() == 0,
              "CPU native check_count expects at least one input item.");
  TI_ERROR_IF(output->get_nelement() < 1,
              "CPU native check_count output must contain at least one item.");
  TI_ERROR_IF(output->get_element_size() != sizeof(int32_t),
              "CPU native check_count output must be i32.");
  const std::size_t expected_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(expected_size == 0,
              "CPU native check_count received an unsupported value type.");
  TI_ERROR_IF(values->get_element_size() != expected_size,
              "CPU native check_count dtype does not match value type.");
  TI_ERROR_IF(check_op < 0 || check_op > 5,
              "CPU native check_count received an unsupported check op.");

  const std::size_t n = values->get_nelement();
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = cpu_use_parallel_aggregation(n, target_threads);
  const auto values_addr = get_ndarray_data_ptr_as_int(values);
  const auto output_addr = get_ndarray_data_ptr_as_int(output);
  auto *output_ptr = reinterpret_cast<int32_t *>(output_addr);
  switch (value_type) {
    case 0:
      return cpu_check_count_typed(
          reinterpret_cast<const int32_t *>(values_addr), output_ptr, check_op,
          lower, upper, n, max_threads, target_threads, use_parallel);
    case 1:
      return cpu_check_count_typed(
          reinterpret_cast<const float *>(values_addr), output_ptr, check_op,
          lower, upper, n, max_threads, target_threads, use_parallel);
    case 2:
      return cpu_check_count_typed(
          reinterpret_cast<const uint32_t *>(values_addr), output_ptr, check_op,
          lower, upper, n, max_threads, target_threads, use_parallel);
    case 3:
      return cpu_check_count_typed(
          reinterpret_cast<const uint64_t *>(values_addr), output_ptr, check_op,
          lower, upper, n, max_threads, target_threads, use_parallel);
    case 4:
      return cpu_check_count_typed(
          reinterpret_cast<const int64_t *>(values_addr), output_ptr, check_op,
          lower, upper, n, max_threads, target_threads, use_parallel);
    case 5:
      return cpu_check_count_typed(
          reinterpret_cast<const double *>(values_addr), output_ptr, check_op,
          lower, upper, n, max_threads, target_threads, use_parallel);
  }
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::cpu_check_count_strided_ndarray(Ndarray *values,
                                                     Ndarray *output,
                                                     int value_type,
                                                     std::size_t offset,
                                                     std::size_t stride,
                                                     int check_op,
                                                     int lower,
                                                     int upper) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native strided check_count is only available on CPU "
              "backends.");
  TI_ERROR_IF(!values || !output,
              "CPU native strided check_count received a null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || output->shape.size() != 1,
              "CPU native strided check_count expects 1D ndarrays.");
  TI_ERROR_IF(values->get_nelement() == 0,
              "CPU native strided check_count expects at least one item.");
  TI_ERROR_IF(output->get_nelement() < 1 ||
                  output->get_element_size() != sizeof(int32_t),
              "CPU native strided check_count output must be a non-empty i32 "
              "ndarray.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CPU native strided check_count received an unsupported value "
              "type.");
  TI_ERROR_IF(check_op < 0 || check_op > 5,
              "CPU native strided check_count received an unsupported check "
              "op.");
  const std::size_t n = values->get_nelement();
  const std::size_t src_bytes = n * values->get_element_size();
  TI_ERROR_IF(stride < value_size || offset % value_size != 0 ||
                  stride % value_size != 0 || src_bytes < value_size ||
                  offset > src_bytes - value_size ||
                  offset + (n - 1) * stride + value_size > src_bytes,
              "CPU native strided check_count source range is out of bounds.");
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = cpu_use_parallel_aggregation(n, target_threads);
  const auto values_addr = get_ndarray_data_ptr_as_int(values);
  const auto output_addr = get_ndarray_data_ptr_as_int(output);
  auto *values_ptr = reinterpret_cast<const uint8_t *>(values_addr);
  auto *output_ptr = reinterpret_cast<int32_t *>(output_addr);
  switch (value_type) {
    case 0:
      return cpu_check_count_strided_typed<int32_t>(
          values_ptr, offset, stride, output_ptr, check_op, lower, upper, n,
          max_threads, target_threads, use_parallel);
    case 1:
      return cpu_check_count_strided_typed<float>(
          values_ptr, offset, stride, output_ptr, check_op, lower, upper, n,
          max_threads, target_threads, use_parallel);
    case 2:
      return cpu_check_count_strided_typed<uint32_t>(
          values_ptr, offset, stride, output_ptr, check_op, lower, upper, n,
          max_threads, target_threads, use_parallel);
    case 3:
      return cpu_check_count_strided_typed<uint64_t>(
          values_ptr, offset, stride, output_ptr, check_op, lower, upper, n,
          max_threads, target_threads, use_parallel);
    case 4:
      return cpu_check_count_strided_typed<int64_t>(
          values_ptr, offset, stride, output_ptr, check_op, lower, upper, n,
          max_threads, target_threads, use_parallel);
    case 5:
      return cpu_check_count_strided_typed<double>(
          values_ptr, offset, stride, output_ptr, check_op, lower, upper, n,
          max_threads, target_threads, use_parallel);
  }
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::cpu_check_count_dense_field(SNode *values,
                                                 Ndarray *output,
                                                 int value_type,
                                                 std::size_t n,
                                                 int check_op,
                                                 int lower,
                                                 int upper) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native dense field check_count is only available on CPU "
              "backends.");
  TI_ERROR_IF(!values || !output,
              "CPU native dense field check_count received a null argument.");
  TI_ERROR_IF(n == 0,
              "CPU native dense field check_count expects at least one item.");
  TI_ERROR_IF(output->shape.size() != 1 || output->get_nelement() < 1 ||
                  output->get_element_size() != sizeof(int32_t),
              "CPU native dense field check_count output must be a non-empty "
              "i32 ndarray.");
  TI_ERROR_IF(check_op < 0 || check_op > 5,
              "CPU native dense field check_count received an unsupported "
              "check op.");
  std::size_t stride = 0;
  const auto *values_ptr = map_cpu_dense_field(
      this, values, value_type, n, "CPU native dense field check_count",
      &stride);
  auto *output_ptr =
      reinterpret_cast<int32_t *>(get_ndarray_data_ptr_as_int(output));
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = cpu_use_parallel_aggregation(n, target_threads);
  switch (value_type) {
    case 0:
      return cpu_check_count_strided_typed<int32_t>(
          values_ptr, 0, stride, output_ptr, check_op, lower, upper, n,
          max_threads, target_threads, use_parallel);
    case 1:
      return cpu_check_count_strided_typed<float>(
          values_ptr, 0, stride, output_ptr, check_op, lower, upper, n,
          max_threads, target_threads, use_parallel);
    case 2:
      return cpu_check_count_strided_typed<uint32_t>(
          values_ptr, 0, stride, output_ptr, check_op, lower, upper, n,
          max_threads, target_threads, use_parallel);
    case 3:
      return cpu_check_count_strided_typed<uint64_t>(
          values_ptr, 0, stride, output_ptr, check_op, lower, upper, n,
          max_threads, target_threads, use_parallel);
    case 4:
      return cpu_check_count_strided_typed<int64_t>(
          values_ptr, 0, stride, output_ptr, check_op, lower, upper, n,
          max_threads, target_threads, use_parallel);
    case 5:
      return cpu_check_count_strided_typed<double>(
          values_ptr, 0, stride, output_ptr, check_op, lower, upper, n,
          max_threads, target_threads, use_parallel);
    default:
      TI_ERROR(
          "CPU native dense field check_count received an unsupported value "
          "type.");
  }
}

std::size_t Program::cpu_check_count_workspace_bytes() const {
  return 0;
}

bool Program::cpu_metric_reduce_available() const {
  return arch_is_cpu(compile_config().arch);
}

bool Program::cpu_metric_reduce_value_type_available(int value_type) const {
  return value_type == 1 || value_type == 5;
}

std::size_t Program::cpu_metric_reduce_ndarray(Ndarray *values,
                                               Ndarray *other,
                                               Ndarray *output,
                                               int value_type,
                                               int metric_op) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native metric_reduce is only available on CPU backends.");
  TI_ERROR_IF(!values || !output,
              "CPU native metric_reduce received a null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || output->shape.size() != 1,
              "CPU native metric_reduce expects 1D ndarrays.");
  TI_ERROR_IF(values->get_nelement() == 0,
              "CPU native metric_reduce expects at least one input item.");
  TI_ERROR_IF(output->get_nelement() < 1,
              "CPU native metric_reduce output must contain at least one item.");
  TI_ERROR_IF(metric_op < 0 || metric_op > 1,
              "CPU native metric_reduce received an unsupported metric op.");
  TI_ERROR_IF(metric_op == 1 && !other,
              "CPU native max_abs_delta received a null rhs ndarray.");
  if (other) {
    TI_ERROR_IF(other->shape.size() != 1,
                "CPU native metric_reduce rhs must be a 1D ndarray.");
    TI_ERROR_IF(other->get_nelement() != values->get_nelement(),
                "CPU native metric_reduce inputs must have the same length.");
  }
  const std::size_t expected_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(expected_size == 0,
              "CPU native metric_reduce received an unsupported value type.");
  TI_ERROR_IF(!cpu_metric_reduce_value_type_available(value_type),
              "CPU native metric_reduce currently supports only f32/f64.");
  TI_ERROR_IF(values->get_element_size() != expected_size ||
                  output->get_element_size() != expected_size ||
                  (other && other->get_element_size() != expected_size),
              "CPU native metric_reduce dtype does not match value type.");

  const std::size_t n = values->get_nelement();
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = cpu_use_parallel_aggregation(n, target_threads);
  const auto values_addr = get_ndarray_data_ptr_as_int(values);
  const auto other_addr = other ? get_ndarray_data_ptr_as_int(other) : 0;
  const auto output_addr = get_ndarray_data_ptr_as_int(output);
  switch (value_type) {
    case 1:
      return cpu_metric_reduce_typed(
          reinterpret_cast<const float *>(values_addr),
          reinterpret_cast<const float *>(other_addr),
          reinterpret_cast<float *>(output_addr), metric_op, n, max_threads,
          target_threads, use_parallel);
    case 5:
      return cpu_metric_reduce_typed(
          reinterpret_cast<const double *>(values_addr),
          reinterpret_cast<const double *>(other_addr),
          reinterpret_cast<double *>(output_addr), metric_op, n, max_threads,
          target_threads, use_parallel);
  }
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::cpu_metric_reduce_strided_ndarray(
    Ndarray *values,
    Ndarray *other,
    Ndarray *output,
    int value_type,
    std::size_t values_offset,
    std::size_t values_stride,
    std::size_t other_offset,
    std::size_t other_stride,
    int metric_op) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native strided metric_reduce is only available on CPU "
              "backends.");
  TI_ERROR_IF(!values || !output,
              "CPU native strided metric_reduce received a null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || output->shape.size() != 1,
              "CPU native strided metric_reduce expects 1D ndarrays.");
  TI_ERROR_IF(values->get_nelement() == 0,
              "CPU native strided metric_reduce expects at least one item.");
  TI_ERROR_IF(output->get_nelement() < 1,
              "CPU native strided metric_reduce output must contain at least "
              "one item.");
  TI_ERROR_IF(metric_op < 0 || metric_op > 1,
              "CPU native strided metric_reduce received an unsupported "
              "metric op.");
  TI_ERROR_IF(metric_op == 1 && !other,
              "CPU native strided max_abs_delta received a null rhs ndarray.");
  if (!other) {
    other = values;
    other_offset = values_offset;
    other_stride = values_stride;
  }
  TI_ERROR_IF(other->shape.size() != 1,
              "CPU native strided metric_reduce rhs must be a 1D ndarray.");
  TI_ERROR_IF(other->get_nelement() != values->get_nelement(),
              "CPU native strided metric_reduce inputs must have the same "
              "length.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CPU native strided metric_reduce received an unsupported value "
              "type.");
  TI_ERROR_IF(!cpu_metric_reduce_value_type_available(value_type),
              "CPU native strided metric_reduce currently supports only "
              "f32/f64.");
  TI_ERROR_IF(output->get_element_size() != value_size,
              "CPU native strided metric_reduce output dtype does not match "
              "value type.");
  auto check_range = [&](const char *role, Ndarray *arr, std::size_t offset,
                         std::size_t stride) {
    const std::size_t bytes = arr->get_nelement() * arr->get_element_size();
    TI_ERROR_IF(stride < value_size || offset % value_size != 0 ||
                    stride % value_size != 0 || bytes < value_size ||
                    offset > bytes - value_size ||
                    offset + (arr->get_nelement() - 1) * stride + value_size >
                        bytes,
                "CPU native strided metric_reduce {} range is out of bounds.",
                role);
  };
  check_range("source", values, values_offset, values_stride);
  check_range("rhs", other, other_offset, other_stride);
  const std::size_t n = values->get_nelement();
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = cpu_use_parallel_aggregation(n, target_threads);
  const auto values_addr = get_ndarray_data_ptr_as_int(values);
  const auto other_addr = get_ndarray_data_ptr_as_int(other);
  const auto output_addr = get_ndarray_data_ptr_as_int(output);
  const auto *values_ptr = reinterpret_cast<const uint8_t *>(values_addr);
  const auto *other_ptr = reinterpret_cast<const uint8_t *>(other_addr);
  switch (value_type) {
    case 1:
      return cpu_metric_reduce_strided_typed<float>(
          values_ptr, other_ptr, reinterpret_cast<float *>(output_addr),
          metric_op, n, values_offset, values_stride, other_offset,
          other_stride, max_threads, target_threads, use_parallel);
    case 5:
      return cpu_metric_reduce_strided_typed<double>(
          values_ptr, other_ptr, reinterpret_cast<double *>(output_addr),
          metric_op, n, values_offset, values_stride, other_offset,
          other_stride, max_threads, target_threads, use_parallel);
  }
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::cpu_metric_reduce_dense_field(SNode *values,
                                                   SNode *other,
                                                   Ndarray *output,
                                                   int value_type,
                                                   std::size_t n,
                                                   int metric_op) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native dense field metric_reduce is only available on CPU "
              "backends.");
  TI_ERROR_IF(!values || !output,
              "CPU native dense field metric_reduce received a null argument.");
  if (!other) {
    other = values;
  }
  TI_ERROR_IF(n == 0,
              "CPU native dense field metric_reduce expects at least one "
              "item.");
  TI_ERROR_IF(output->shape.size() != 1 || output->get_nelement() < 1,
              "CPU native dense field metric_reduce output must be a "
              "non-empty ndarray.");
  TI_ERROR_IF(metric_op < 0 || metric_op > 1,
              "CPU native dense field metric_reduce received an unsupported "
              "metric op.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CPU native dense field metric_reduce received an unsupported "
              "value type.");
  TI_ERROR_IF(!cpu_metric_reduce_value_type_available(value_type),
              "CPU native dense field metric_reduce currently supports only "
              "f32/f64.");
  TI_ERROR_IF(output->get_element_size() != value_size,
              "CPU native dense field metric_reduce output dtype does not "
              "match value type.");
  std::size_t values_stride = 0;
  std::size_t other_stride = 0;
  const auto *values_ptr = map_cpu_dense_field(
      this, values, value_type, n, "CPU native dense field metric_reduce",
      &values_stride);
  const auto *other_ptr = map_cpu_dense_field(
      this, other, value_type, n, "CPU native dense field metric_reduce",
      &other_stride);
  const auto output_addr = get_ndarray_data_ptr_as_int(output);
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = cpu_use_parallel_aggregation(n, target_threads);
  switch (value_type) {
    case 1:
      return cpu_metric_reduce_strided_typed<float>(
          values_ptr, other_ptr, reinterpret_cast<float *>(output_addr),
          metric_op, n, 0, values_stride, 0, other_stride, max_threads,
          target_threads, use_parallel);
    case 5:
      return cpu_metric_reduce_strided_typed<double>(
          values_ptr, other_ptr, reinterpret_cast<double *>(output_addr),
          metric_op, n, 0, values_stride, 0, other_stride, max_threads,
          target_threads, use_parallel);
  }
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::cpu_metric_reduce_dense_field_strided_ndarray(
    SNode *field,
    Ndarray *array,
    Ndarray *output,
    int value_type,
    std::size_t n,
    std::size_t array_offset,
    std::size_t array_stride,
    bool field_is_values,
    int metric_op) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native mixed metric_reduce is only available on CPU "
              "backends.");
  TI_ERROR_IF(!field || !array || !output,
              "CPU native mixed metric_reduce received a null argument.");
  TI_ERROR_IF(n == 0,
              "CPU native mixed metric_reduce expects at least one item.");
  TI_ERROR_IF(array->shape.size() != 1 || output->shape.size() != 1,
              "CPU native mixed metric_reduce expects 1D ndarrays.");
  TI_ERROR_IF(array->get_nelement() != n,
              "CPU native mixed metric_reduce inputs must have the same "
              "length.");
  TI_ERROR_IF(output->get_nelement() < 1,
              "CPU native mixed metric_reduce output must contain at least "
              "one item.");
  TI_ERROR_IF(metric_op < 0 || metric_op > 1,
              "CPU native mixed metric_reduce received an unsupported metric "
              "op.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CPU native mixed metric_reduce received an unsupported value "
              "type.");
  TI_ERROR_IF(!cpu_metric_reduce_value_type_available(value_type),
              "CPU native mixed metric_reduce currently supports only "
              "f32/f64.");
  TI_ERROR_IF(output->get_element_size() != value_size,
              "CPU native mixed metric_reduce output dtype does not match "
              "value type.");
  const std::size_t array_bytes =
      array->get_nelement() * array->get_element_size();
  TI_ERROR_IF(array_stride < value_size || array_offset % value_size != 0 ||
                  array_stride % value_size != 0 || array_bytes < value_size ||
                  array_offset > array_bytes - value_size ||
                  array_offset + (n - 1) * array_stride + value_size >
                      array_bytes,
              "CPU native mixed metric_reduce ndarray range is out of bounds.");
  std::size_t field_stride = 0;
  const auto *field_ptr = map_cpu_dense_field(
      this, field, value_type, n, "CPU native mixed metric_reduce",
      &field_stride);
  const auto array_addr = get_ndarray_data_ptr_as_int(array);
  const auto output_addr = get_ndarray_data_ptr_as_int(output);
  const auto *array_ptr = reinterpret_cast<const uint8_t *>(array_addr);
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = cpu_use_parallel_aggregation(n, target_threads);
  const auto *values_ptr = field_is_values ? field_ptr : array_ptr;
  const auto *other_ptr = field_is_values ? array_ptr : field_ptr;
  const std::size_t values_offset = field_is_values ? 0 : array_offset;
  const std::size_t values_stride = field_is_values ? field_stride : array_stride;
  const std::size_t other_offset = field_is_values ? array_offset : 0;
  const std::size_t other_stride = field_is_values ? array_stride : field_stride;
  switch (value_type) {
    case 1:
      return cpu_metric_reduce_strided_typed<float>(
          values_ptr, other_ptr, reinterpret_cast<float *>(output_addr),
          metric_op, n, values_offset, values_stride, other_offset,
          other_stride, max_threads, target_threads, use_parallel);
    case 5:
      return cpu_metric_reduce_strided_typed<double>(
          values_ptr, other_ptr, reinterpret_cast<double *>(output_addr),
          metric_op, n, values_offset, values_stride, other_offset,
          other_stride, max_threads, target_threads, use_parallel);
  }
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::cpu_metric_reduce_workspace_bytes() const {
  return 0;
}

bool Program::cpu_transform_available() const {
  return arch_is_cpu(compile_config().arch);
}

std::size_t Program::cpu_transform_affine_ndarray(Ndarray *src,
                                                  Ndarray *dst,
                                                  int value_type,
                                                  double scale,
                                                  double bias) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native transform is only available on CPU backends.");
  TI_ERROR_IF(!src || !dst, "CPU native transform received a null ndarray.");
  TI_ERROR_IF(src->get_nelement() != dst->get_nelement(),
              "CPU native transform source and destination sizes differ.");
  TI_ERROR_IF(src->get_element_size() != dst->get_element_size(),
              "CPU native transform source and destination dtypes differ.");
  TI_ERROR_IF(value_type < 0 || value_type > 5,
              "CPU native transform received an unsupported value type.");
  const bool is_64bit = value_type == 3 || value_type == 4 || value_type == 5;
  TI_ERROR_IF(src->get_element_size() !=
                  (is_64bit ? sizeof(uint64_t) : sizeof(uint32_t)),
              "CPU native transform dtype does not match value type.");

  const std::size_t n = src->get_nelement();
  if (n == 0) {
    return 0;
  }
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = cpu_use_parallel_simple_loop(n, target_threads);

  const auto src_addr = get_ndarray_data_ptr_as_int(src);
  const auto dst_addr = get_ndarray_data_ptr_as_int(dst);
  TI_ERROR_IF(!src_addr || !dst_addr,
              "CPU native transform received a null data pointer.");
  switch (value_type) {
    case 0:
      cpu_transform_run_typed<uint32_t>(
          reinterpret_cast<const uint32_t *>(src_addr),
          reinterpret_cast<uint32_t *>(dst_addr), n,
          static_cast<uint32_t>(static_cast<int32_t>(scale)),
          static_cast<uint32_t>(static_cast<int32_t>(bias)), use_parallel,
          target_threads, max_threads);
      return 0;
    case 2:
      cpu_transform_run_typed<uint32_t>(
          reinterpret_cast<const uint32_t *>(src_addr),
          reinterpret_cast<uint32_t *>(dst_addr), n,
          static_cast<uint32_t>(scale), static_cast<uint32_t>(bias),
          use_parallel, target_threads, max_threads);
      return 0;
    case 1:
      cpu_transform_run_typed<float>(
          reinterpret_cast<const float *>(src_addr),
          reinterpret_cast<float *>(dst_addr), n, static_cast<float>(scale),
          static_cast<float>(bias), use_parallel, target_threads, max_threads);
      return 0;
    case 3:
      cpu_transform_run_typed<uint64_t>(
          reinterpret_cast<const uint64_t *>(src_addr),
          reinterpret_cast<uint64_t *>(dst_addr), n,
          static_cast<uint64_t>(scale), static_cast<uint64_t>(bias),
          use_parallel, target_threads, max_threads);
      return 0;
    case 4:
      cpu_transform_run_typed<uint64_t>(
          reinterpret_cast<const uint64_t *>(src_addr),
          reinterpret_cast<uint64_t *>(dst_addr), n,
          static_cast<uint64_t>(static_cast<int64_t>(scale)),
          static_cast<uint64_t>(static_cast<int64_t>(bias)), use_parallel,
          target_threads, max_threads);
      return 0;
    case 5:
      cpu_transform_run_typed<double>(
          reinterpret_cast<const double *>(src_addr),
          reinterpret_cast<double *>(dst_addr), n, scale, bias, use_parallel,
          target_threads, max_threads);
      return 0;
    default:
      TI_ERROR("CPU native transform received an unsupported value type.");
  }
  return 0;
}

std::size_t Program::cpu_transform_affine_member_ndarray(Ndarray *src,
                                                         Ndarray *dst,
                                                         int value_type,
                                                         std::size_t offset,
                                                         std::size_t stride,
                                                         double scale,
                                                         double bias) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native strided transform is only available on CPU "
              "backends.");
  check_transform_member_request("CPU native", src, dst, value_type, offset,
                                 stride);
  const std::size_t n = src->get_nelement();
  if (n == 0) {
    return 0;
  }
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = cpu_use_parallel_simple_loop(n, target_threads);

  const auto src_addr = get_ndarray_data_ptr_as_int(src);
  const auto dst_addr = get_ndarray_data_ptr_as_int(dst);
  TI_ERROR_IF(!src_addr || !dst_addr,
              "CPU native strided transform received a null data pointer.");
  const auto *src_bytes = reinterpret_cast<const uint8_t *>(src_addr);
  switch (value_type) {
    case 0:
      cpu_transform_run_strided_typed<uint32_t>(
          src_bytes, reinterpret_cast<uint32_t *>(dst_addr), n, offset, stride,
          static_cast<uint32_t>(static_cast<int32_t>(scale)),
          static_cast<uint32_t>(static_cast<int32_t>(bias)), use_parallel,
          target_threads, max_threads);
      return 0;
    case 2:
      cpu_transform_run_strided_typed<uint32_t>(
          src_bytes, reinterpret_cast<uint32_t *>(dst_addr), n, offset, stride,
          static_cast<uint32_t>(scale), static_cast<uint32_t>(bias),
          use_parallel, target_threads, max_threads);
      return 0;
    case 1:
      cpu_transform_run_strided_typed<float>(
          src_bytes, reinterpret_cast<float *>(dst_addr), n, offset, stride,
          static_cast<float>(scale), static_cast<float>(bias), use_parallel,
          target_threads, max_threads);
      return 0;
    case 3:
      cpu_transform_run_strided_typed<uint64_t>(
          src_bytes, reinterpret_cast<uint64_t *>(dst_addr), n, offset, stride,
          static_cast<uint64_t>(scale), static_cast<uint64_t>(bias),
          use_parallel, target_threads, max_threads);
      return 0;
    case 4:
      cpu_transform_run_strided_typed<uint64_t>(
          src_bytes, reinterpret_cast<uint64_t *>(dst_addr), n, offset, stride,
          static_cast<uint64_t>(static_cast<int64_t>(scale)),
          static_cast<uint64_t>(static_cast<int64_t>(bias)), use_parallel,
          target_threads, max_threads);
      return 0;
    case 5:
      cpu_transform_run_strided_typed<double>(
          src_bytes, reinterpret_cast<double *>(dst_addr), n, offset, stride,
          scale, bias, use_parallel, target_threads, max_threads);
      return 0;
    default:
      TI_ERROR("CPU native strided transform received an unsupported value "
               "type.");
  }
  return 0;
}

std::size_t Program::cpu_transform_affine_strided_ndarray(
    Ndarray *src,
    Ndarray *dst,
    int value_type,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride,
    double scale,
    double bias) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native strided transform is only available on CPU "
              "backends.");
  check_transform_strided_request("CPU native", src, dst, value_type,
                                  src_offset, src_stride, dst_offset,
                                  dst_stride);
  const std::size_t n = src->get_nelement();
  if (n == 0) {
    return 0;
  }
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = cpu_use_parallel_simple_loop(n, target_threads);

  const auto src_addr = get_ndarray_data_ptr_as_int(src);
  const auto dst_addr = get_ndarray_data_ptr_as_int(dst);
  TI_ERROR_IF(!src_addr || !dst_addr,
              "CPU native strided transform received a null data pointer.");
  const auto *src_bytes = reinterpret_cast<const uint8_t *>(src_addr);
  auto *dst_bytes = reinterpret_cast<uint8_t *>(dst_addr);
  switch (value_type) {
    case 0:
      cpu_transform_run_strided_to_strided_typed<uint32_t>(
          src_bytes, dst_bytes, n, src_offset, src_stride, dst_offset,
          dst_stride, static_cast<uint32_t>(static_cast<int32_t>(scale)),
          static_cast<uint32_t>(static_cast<int32_t>(bias)), use_parallel,
          target_threads, max_threads);
      return 0;
    case 2:
      cpu_transform_run_strided_to_strided_typed<uint32_t>(
          src_bytes, dst_bytes, n, src_offset, src_stride, dst_offset,
          dst_stride, static_cast<uint32_t>(scale), static_cast<uint32_t>(bias),
          use_parallel, target_threads, max_threads);
      return 0;
    case 1:
      cpu_transform_run_strided_to_strided_typed<float>(
          src_bytes, dst_bytes, n, src_offset, src_stride, dst_offset,
          dst_stride, static_cast<float>(scale), static_cast<float>(bias),
          use_parallel, target_threads, max_threads);
      return 0;
    case 3:
      cpu_transform_run_strided_to_strided_typed<uint64_t>(
          src_bytes, dst_bytes, n, src_offset, src_stride, dst_offset,
          dst_stride, static_cast<uint64_t>(scale), static_cast<uint64_t>(bias),
          use_parallel, target_threads, max_threads);
      return 0;
    case 4:
      cpu_transform_run_strided_to_strided_typed<uint64_t>(
          src_bytes, dst_bytes, n, src_offset, src_stride, dst_offset,
          dst_stride, static_cast<uint64_t>(static_cast<int64_t>(scale)),
          static_cast<uint64_t>(static_cast<int64_t>(bias)), use_parallel,
          target_threads, max_threads);
      return 0;
    case 5:
      cpu_transform_run_strided_to_strided_typed<double>(
          src_bytes, dst_bytes, n, src_offset, src_stride, dst_offset,
          dst_stride, scale, bias, use_parallel, target_threads, max_threads);
      return 0;
    default:
      TI_ERROR("CPU native strided transform received an unsupported value "
               "type.");
  }
  return 0;
}

std::size_t Program::cpu_transform_affine_packed_strided_ndarray(
    Ndarray *src,
    Ndarray *dst,
    int value_type,
    int lane_count,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride,
    double scale,
    double bias) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native packed strided transform is only available on CPU "
              "backends.");
  check_transform_packed_strided_request("CPU native", src, dst, value_type,
                                         lane_count, src_offset, src_stride,
                                         dst_offset, dst_stride);
  const std::size_t n = src->get_nelement();
  if (n == 0) {
    return 0;
  }
  const std::size_t total =
      n * static_cast<std::size_t>(std::max(1, lane_count));
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((total + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = total >= 65536 && target_threads > 1;

  const auto src_addr = get_ndarray_data_ptr_as_int(src);
  const auto dst_addr = get_ndarray_data_ptr_as_int(dst);
  TI_ERROR_IF(!src_addr || !dst_addr,
              "CPU native packed strided transform received a null data "
              "pointer.");
  const auto *src_bytes = reinterpret_cast<const uint8_t *>(src_addr);
  auto *dst_bytes = reinterpret_cast<uint8_t *>(dst_addr);
  switch (value_type) {
    case 0:
      cpu_transform_run_packed_strided_to_strided_typed<uint32_t>(
          src_bytes, dst_bytes, n, lane_count, src_offset, src_stride,
          dst_offset, dst_stride,
          static_cast<uint32_t>(static_cast<int32_t>(scale)),
          static_cast<uint32_t>(static_cast<int32_t>(bias)), use_parallel,
          target_threads, max_threads);
      return 0;
    case 2:
      cpu_transform_run_packed_strided_to_strided_typed<uint32_t>(
          src_bytes, dst_bytes, n, lane_count, src_offset, src_stride,
          dst_offset, dst_stride, static_cast<uint32_t>(scale),
          static_cast<uint32_t>(bias), use_parallel, target_threads,
          max_threads);
      return 0;
    case 1:
      cpu_transform_run_packed_strided_to_strided_typed<float>(
          src_bytes, dst_bytes, n, lane_count, src_offset, src_stride,
          dst_offset, dst_stride, static_cast<float>(scale),
          static_cast<float>(bias), use_parallel, target_threads, max_threads);
      return 0;
    case 3:
      cpu_transform_run_packed_strided_to_strided_typed<uint64_t>(
          src_bytes, dst_bytes, n, lane_count, src_offset, src_stride,
          dst_offset, dst_stride, static_cast<uint64_t>(scale),
          static_cast<uint64_t>(bias), use_parallel, target_threads,
          max_threads);
      return 0;
    case 4:
      cpu_transform_run_packed_strided_to_strided_typed<uint64_t>(
          src_bytes, dst_bytes, n, lane_count, src_offset, src_stride,
          dst_offset, dst_stride,
          static_cast<uint64_t>(static_cast<int64_t>(scale)),
          static_cast<uint64_t>(static_cast<int64_t>(bias)), use_parallel,
          target_threads, max_threads);
      return 0;
    case 5:
      cpu_transform_run_packed_strided_to_strided_typed<double>(
          src_bytes, dst_bytes, n, lane_count, src_offset, src_stride,
          dst_offset, dst_stride, scale, bias, use_parallel, target_threads,
          max_threads);
      return 0;
    default:
      TI_ERROR("CPU native packed strided transform received an unsupported "
               "value type.");
  }
  return 0;
}

std::size_t Program::cpu_transform_affine_dense_field(SNode *src,
                                                      SNode *dst,
                                                      int value_type,
                                                      std::size_t n,
                                                      double scale,
                                                      double bias) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native dense field transform is only available on CPU "
              "backends.");
  if (n == 0) {
    return 0;
  }
  std::size_t src_stride = 0;
  std::size_t dst_stride = 0;
  const auto *src_ptr = map_cpu_dense_field(
      this, src, value_type, n, "CPU native dense field transform",
      &src_stride);
  auto *dst_ptr = map_cpu_dense_field(
      this, dst, value_type, n, "CPU native dense field transform",
      &dst_stride);
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = cpu_use_parallel_simple_loop(n, target_threads);
  switch (value_type) {
    case 0:
      if (src_stride == sizeof(uint32_t) && dst_stride == sizeof(uint32_t)) {
        cpu_transform_run_typed<uint32_t>(
            reinterpret_cast<const uint32_t *>(src_ptr),
            reinterpret_cast<uint32_t *>(dst_ptr), n,
            static_cast<uint32_t>(static_cast<int32_t>(scale)),
            static_cast<uint32_t>(static_cast<int32_t>(bias)), use_parallel,
            target_threads, max_threads);
        return 0;
      }
      cpu_transform_run_strided_to_strided_typed<uint32_t>(
          src_ptr, dst_ptr, n, 0, src_stride, 0, dst_stride,
          static_cast<uint32_t>(static_cast<int32_t>(scale)),
          static_cast<uint32_t>(static_cast<int32_t>(bias)), use_parallel,
          target_threads, max_threads);
      return 0;
    case 2:
      if (src_stride == sizeof(uint32_t) && dst_stride == sizeof(uint32_t)) {
        cpu_transform_run_typed<uint32_t>(
            reinterpret_cast<const uint32_t *>(src_ptr),
            reinterpret_cast<uint32_t *>(dst_ptr), n,
            static_cast<uint32_t>(scale), static_cast<uint32_t>(bias),
            use_parallel, target_threads, max_threads);
        return 0;
      }
      cpu_transform_run_strided_to_strided_typed<uint32_t>(
          src_ptr, dst_ptr, n, 0, src_stride, 0, dst_stride,
          static_cast<uint32_t>(scale), static_cast<uint32_t>(bias),
          use_parallel, target_threads, max_threads);
      return 0;
    case 1:
      if (src_stride == sizeof(float) && dst_stride == sizeof(float)) {
        cpu_transform_run_typed<float>(
            reinterpret_cast<const float *>(src_ptr),
            reinterpret_cast<float *>(dst_ptr), n, static_cast<float>(scale),
            static_cast<float>(bias), use_parallel, target_threads,
            max_threads);
        return 0;
      }
      cpu_transform_run_strided_to_strided_typed<float>(
          src_ptr, dst_ptr, n, 0, src_stride, 0, dst_stride,
          static_cast<float>(scale), static_cast<float>(bias), use_parallel,
          target_threads, max_threads);
      return 0;
    case 3:
      if (src_stride == sizeof(uint64_t) && dst_stride == sizeof(uint64_t)) {
        cpu_transform_run_typed<uint64_t>(
            reinterpret_cast<const uint64_t *>(src_ptr),
            reinterpret_cast<uint64_t *>(dst_ptr), n,
            static_cast<uint64_t>(scale), static_cast<uint64_t>(bias),
            use_parallel, target_threads, max_threads);
        return 0;
      }
      cpu_transform_run_strided_to_strided_typed<uint64_t>(
          src_ptr, dst_ptr, n, 0, src_stride, 0, dst_stride,
          static_cast<uint64_t>(scale), static_cast<uint64_t>(bias),
          use_parallel, target_threads, max_threads);
      return 0;
    case 4:
      if (src_stride == sizeof(uint64_t) && dst_stride == sizeof(uint64_t)) {
        cpu_transform_run_typed<uint64_t>(
            reinterpret_cast<const uint64_t *>(src_ptr),
            reinterpret_cast<uint64_t *>(dst_ptr), n,
            static_cast<uint64_t>(static_cast<int64_t>(scale)),
            static_cast<uint64_t>(static_cast<int64_t>(bias)), use_parallel,
            target_threads, max_threads);
        return 0;
      }
      cpu_transform_run_strided_to_strided_typed<uint64_t>(
          src_ptr, dst_ptr, n, 0, src_stride, 0, dst_stride,
          static_cast<uint64_t>(static_cast<int64_t>(scale)),
          static_cast<uint64_t>(static_cast<int64_t>(bias)), use_parallel,
          target_threads, max_threads);
      return 0;
    case 5:
      if (src_stride == sizeof(double) && dst_stride == sizeof(double)) {
        cpu_transform_run_typed<double>(
            reinterpret_cast<const double *>(src_ptr),
            reinterpret_cast<double *>(dst_ptr), n, scale, bias, use_parallel,
            target_threads, max_threads);
        return 0;
      }
      cpu_transform_run_strided_to_strided_typed<double>(
          src_ptr, dst_ptr, n, 0, src_stride, 0, dst_stride, scale, bias,
          use_parallel, target_threads, max_threads);
      return 0;
    default:
      TI_ERROR("CPU native dense field transform received an unsupported "
               "value type.");
  }
  return 0;
}

std::size_t Program::cpu_transform_workspace_bytes() const {
  return 0;
}

bool Program::cpu_add_merge_available() const {
  return arch_is_cpu(compile_config().arch);
}

std::size_t Program::cpu_add_merge_ndarray(Ndarray *src,
                                           Ndarray *dst,
                                           int value_type) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native add-merge is only available on CPU backends.");
  TI_ERROR_IF(!src || !dst, "CPU add-merge received a null ndarray.");
  TI_ERROR_IF(src->get_nelement() != dst->get_nelement(),
              "CPU add-merge source and destination sizes differ.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CPU add-merge received an unsupported value type.");
  const std::size_t src_element_size = src->get_element_size();
  const std::size_t dst_element_size = dst->get_element_size();
  TI_ERROR_IF(src_element_size != dst_element_size ||
                  src_element_size < value_size ||
                  src_element_size % value_size != 0,
              "CPU add-merge payload does not match value type.");
  const std::size_t n = src->get_nelement() * (src_element_size / value_size);
  if (n == 0) {
    return 0;
  }
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = cpu_use_parallel_simple_loop(n, target_threads);
  const auto src_addr = get_ndarray_data_ptr_as_int(src);
  const auto dst_addr = get_ndarray_data_ptr_as_int(dst);
  TI_ERROR_IF(!src_addr || !dst_addr,
              "CPU add-merge received a null data pointer.");
  switch (value_type) {
    case 0:
      cpu_add_merge_run_typed(
          reinterpret_cast<const int32_t *>(src_addr),
          reinterpret_cast<int32_t *>(dst_addr), n, use_parallel,
          target_threads, max_threads);
      return 0;
    case 1:
      cpu_add_merge_run_typed(reinterpret_cast<const float *>(src_addr),
                              reinterpret_cast<float *>(dst_addr), n,
                              use_parallel, target_threads, max_threads);
      return 0;
    case 2:
      cpu_add_merge_run_typed(
          reinterpret_cast<const uint32_t *>(src_addr),
          reinterpret_cast<uint32_t *>(dst_addr), n, use_parallel,
          target_threads, max_threads);
      return 0;
    case 3:
      cpu_add_merge_run_typed(
          reinterpret_cast<const uint64_t *>(src_addr),
          reinterpret_cast<uint64_t *>(dst_addr), n, use_parallel,
          target_threads, max_threads);
      return 0;
    case 4:
      cpu_add_merge_run_typed(
          reinterpret_cast<const int64_t *>(src_addr),
          reinterpret_cast<int64_t *>(dst_addr), n, use_parallel,
          target_threads, max_threads);
      return 0;
    case 5:
      cpu_add_merge_run_typed(reinterpret_cast<const double *>(src_addr),
                              reinterpret_cast<double *>(dst_addr), n,
                              use_parallel, target_threads, max_threads);
      return 0;
    default:
      TI_ERROR("CPU add-merge received an unsupported value type.");
  }
  return 0;
}

std::size_t Program::cpu_add_scaled_ndarray(Ndarray *src,
                                            Ndarray *dst,
                                            int value_type,
                                            double scale) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native scaled-add is only available on CPU backends.");
  TI_ERROR_IF(!src || !dst, "CPU scaled-add received a null ndarray.");
  TI_ERROR_IF(src->shape.size() != 1 || dst->shape.size() != 1 ||
                  src->get_nelement() != dst->get_nelement(),
              "CPU scaled-add expects matching 1D ndarrays.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CPU scaled-add received an unsupported value type.");
  const std::size_t src_element_size = src->get_element_size();
  const std::size_t dst_element_size = dst->get_element_size();
  TI_ERROR_IF(src_element_size != dst_element_size ||
                  src_element_size < value_size ||
                  src_element_size % value_size != 0,
              "CPU scaled-add payload does not match value type.");
  const std::size_t n = src->get_nelement() * (src_element_size / value_size);
  if (n == 0) {
    return 0;
  }
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = cpu_use_parallel_simple_loop(n, target_threads);
  const auto src_addr = get_ndarray_data_ptr_as_int(src);
  const auto dst_addr = get_ndarray_data_ptr_as_int(dst);
  TI_ERROR_IF(!src_addr || !dst_addr,
              "CPU scaled-add received a null data pointer.");
  switch (value_type) {
    case 1:
      cpu_add_scaled_run_typed(reinterpret_cast<const float *>(src_addr),
                               reinterpret_cast<float *>(dst_addr), n,
                               static_cast<float>(scale), use_parallel,
                               target_threads, max_threads);
      return 0;
    case 5:
      cpu_add_scaled_run_typed(reinterpret_cast<const double *>(src_addr),
                               reinterpret_cast<double *>(dst_addr), n, scale,
                               use_parallel, target_threads, max_threads);
      return 0;
    default:
      TI_ERROR("CPU scaled-add is supported only for f32/f64 gradients.");
  }
  return 0;
}

std::size_t Program::cpu_add_scalar_ndarray_to_ndarray(Ndarray *src,
                                                       Ndarray *dst,
                                                       int value_type,
                                                       double scale) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native scalar-to-ndarray add is only available on CPU "
              "backends.");
  TI_ERROR_IF(!src || !dst,
              "CPU scalar-to-ndarray add received a null ndarray.");
  TI_ERROR_IF(src->shape.size() != 1 || dst->shape.size() != 1 ||
                  src->get_nelement() < 1,
              "CPU scalar-to-ndarray add expects 1D source and destination "
              "ndarrays with at least one source element.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CPU scalar-to-ndarray add received an unsupported value type.");
  TI_ERROR_IF(src->get_element_size() != value_size ||
                  dst->get_element_size() != value_size,
              "CPU scalar-to-ndarray add dtype does not match value type.");
  const std::size_t n = dst->get_nelement();
  if (n == 0) {
    return 0;
  }
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = cpu_use_parallel_simple_loop(n, target_threads);
  const auto src_addr = get_ndarray_data_ptr_as_int(src);
  const auto dst_addr = get_ndarray_data_ptr_as_int(dst);
  TI_ERROR_IF(!src_addr || !dst_addr,
              "CPU scalar-to-ndarray add received a null data pointer.");
  const auto *src_ptr = reinterpret_cast<const uint8_t *>(src_addr);
  auto *dst_ptr = reinterpret_cast<uint8_t *>(dst_addr);
  switch (value_type) {
    case 1:
      cpu_add_scaled_run_strided_to_strided_typed<float>(
          src_ptr, dst_ptr, n, 0, 0, 0, value_size,
          static_cast<float>(scale), use_parallel, target_threads,
          max_threads);
      return 0;
    case 5:
      cpu_add_scaled_run_strided_to_strided_typed<double>(
          src_ptr, dst_ptr, n, 0, 0, 0, value_size, scale, use_parallel,
          target_threads, max_threads);
      return 0;
    default:
      TI_ERROR(
          "CPU scalar-to-ndarray add is supported only for f32/f64 gradients.");
  }
  return 0;
}

std::size_t Program::cpu_add_merge_strided_ndarray(Ndarray *src,
                                                   Ndarray *dst,
                                                   int value_type,
                                                   std::size_t src_offset,
                                                   std::size_t src_stride,
                                                   std::size_t dst_offset,
                                                   std::size_t dst_stride) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native strided add-merge is only available on CPU "
              "backends.");
  check_add_merge_strided_request("CPU native", src, dst, value_type,
                                  src_offset, src_stride, dst_offset,
                                  dst_stride);
  const std::size_t n = src->get_nelement();
  if (n == 0) {
    return 0;
  }
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = cpu_use_parallel_simple_loop(n, target_threads);
  const auto src_addr = get_ndarray_data_ptr_as_int(src);
  const auto dst_addr = get_ndarray_data_ptr_as_int(dst);
  TI_ERROR_IF(!src_addr || !dst_addr,
              "CPU strided add-merge received a null data pointer.");
  const auto *src_ptr = reinterpret_cast<const uint8_t *>(src_addr);
  auto *dst_ptr = reinterpret_cast<uint8_t *>(dst_addr);
  switch (value_type) {
    case 0:
      cpu_add_merge_run_strided_to_strided_typed<int32_t>(
          src_ptr, dst_ptr, n, src_offset, src_stride, dst_offset, dst_stride,
          use_parallel, target_threads, max_threads);
      return 0;
    case 1:
      cpu_add_merge_run_strided_to_strided_typed<float>(
          src_ptr, dst_ptr, n, src_offset, src_stride, dst_offset, dst_stride,
          use_parallel, target_threads, max_threads);
      return 0;
    case 2:
      cpu_add_merge_run_strided_to_strided_typed<uint32_t>(
          src_ptr, dst_ptr, n, src_offset, src_stride, dst_offset, dst_stride,
          use_parallel, target_threads, max_threads);
      return 0;
    case 3:
      cpu_add_merge_run_strided_to_strided_typed<uint64_t>(
          src_ptr, dst_ptr, n, src_offset, src_stride, dst_offset, dst_stride,
          use_parallel, target_threads, max_threads);
      return 0;
    case 4:
      cpu_add_merge_run_strided_to_strided_typed<int64_t>(
          src_ptr, dst_ptr, n, src_offset, src_stride, dst_offset, dst_stride,
          use_parallel, target_threads, max_threads);
      return 0;
    case 5:
      cpu_add_merge_run_strided_to_strided_typed<double>(
          src_ptr, dst_ptr, n, src_offset, src_stride, dst_offset, dst_stride,
          use_parallel, target_threads, max_threads);
      return 0;
    default:
      TI_ERROR("CPU strided add-merge received an unsupported value type.");
  }
  return 0;
}

std::size_t Program::cpu_add_merge_dense_field(Ndarray *src,
                                               SNode *dst,
                                               int value_type,
                                               std::size_t n) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native dense field add-merge is only available on CPU "
              "backends.");
  TI_ERROR_IF(!src || !dst,
              "CPU dense field add-merge received a null input.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CPU dense field add-merge received an unsupported value type.");
  TI_ERROR_IF(src->shape.size() != 1 || src->get_nelement() != n ||
                  src->get_element_size() != value_size,
              "CPU dense field add-merge source shape or dtype mismatch.");
  if (n == 0) {
    return 0;
  }
  std::size_t dst_stride = 0;
  auto *dst_ptr = map_cpu_dense_field(this, dst, value_type, n,
                                      "CPU native dense field add-merge",
                                      &dst_stride);
  const auto src_addr = get_ndarray_data_ptr_as_int(src);
  TI_ERROR_IF(!src_addr,
              "CPU dense field add-merge received a null source pointer.");
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = cpu_use_parallel_simple_loop(n, target_threads);
  const auto *src_ptr = reinterpret_cast<const uint8_t *>(src_addr);
  switch (value_type) {
    case 0:
      cpu_add_merge_run_strided_to_strided_typed<int32_t>(
          src_ptr, dst_ptr, n, 0, value_size, 0, dst_stride, use_parallel,
          target_threads, max_threads);
      return 0;
    case 1:
      cpu_add_merge_run_strided_to_strided_typed<float>(
          src_ptr, dst_ptr, n, 0, value_size, 0, dst_stride, use_parallel,
          target_threads, max_threads);
      return 0;
    case 2:
      cpu_add_merge_run_strided_to_strided_typed<uint32_t>(
          src_ptr, dst_ptr, n, 0, value_size, 0, dst_stride, use_parallel,
          target_threads, max_threads);
      return 0;
    case 3:
      cpu_add_merge_run_strided_to_strided_typed<uint64_t>(
          src_ptr, dst_ptr, n, 0, value_size, 0, dst_stride, use_parallel,
          target_threads, max_threads);
      return 0;
    case 4:
      cpu_add_merge_run_strided_to_strided_typed<int64_t>(
          src_ptr, dst_ptr, n, 0, value_size, 0, dst_stride, use_parallel,
          target_threads, max_threads);
      return 0;
    case 5:
      cpu_add_merge_run_strided_to_strided_typed<double>(
          src_ptr, dst_ptr, n, 0, value_size, 0, dst_stride, use_parallel,
          target_threads, max_threads);
      return 0;
    default:
      TI_ERROR("CPU dense field add-merge received an unsupported value type.");
  }
  return 0;
}

std::size_t Program::add_merge_dense_field_packed(SNode *src,
                                                  SNode *dst,
                                                  int value_type,
                                                  std::size_t n,
                                                  int lane_count) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  const Arch arch = compile_config().arch;
  TI_ERROR_IF(!native_dense_field_bulk_arch(arch),
              "Native packed dense field add-merge is only available on CPU, "
              "CUDA, and Vulkan backends.");
  TI_ERROR_IF(!src || !dst,
              "Native packed dense field add-merge received a null field.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "Native packed dense field add-merge received an unsupported "
              "value type.");
  const std::size_t scalar_items = dense_field_packed_scalar_items(
      n, lane_count, "Native packed dense field add-merge");
  if (n == 0) {
    return 0;
  }
  check_dense_field_packed_stride(this, src, value_type, lane_count,
                                  "Native packed dense field add-merge");
  check_dense_field_packed_stride(this, dst, value_type, lane_count,
                                  "Native packed dense field add-merge");

  if (arch == Arch::vulkan) {
    return vulkan_add_merge_dense_field_packed(src, dst, value_type, n,
                                               lane_count);
  }

  if (arch == Arch::cuda) {
#ifdef TI_WITH_CUDA
    TI_ERROR_IF(
        scalar_items >
            static_cast<std::size_t>(std::numeric_limits<int>::max()),
        "CUDA packed dense field add-merge currently supports at most INT_MAX "
        "scalar items.");
    auto raw_ptr = [this](DevicePtr ptr, const char *op_name) {
      TI_ERROR_IF(!ptr.device, "{} received a null dense field device.",
                  op_name);
      DeviceAllocation alloc{ptr.device, ptr.alloc_id};
      auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
      TI_ERROR_IF(!base, "{} received a null dense field data pointer.",
                  op_name);
      return static_cast<void *>(reinterpret_cast<uint8_t *>(base) +
                                 ptr.offset);
    };
    void *src_raw = raw_ptr(get_dense_field_device_ptr(src),
                            "CUDA packed dense field add-merge");
    void *dst_raw = raw_ptr(get_dense_field_device_ptr(dst),
                            "CUDA packed dense field add-merge");
    void *stream = nullptr;
    return cuda::driver_add_strided(
        src_raw, dst_raw, static_cast<int>(scalar_items),
        static_cast<cuda::CudaTransformValueType>(value_type), 0, value_size, 0,
        value_size, stream);
#else
    TI_ERROR("CUDA packed dense field add-merge requires TI_WITH_CUDA=ON.");
#endif
  }

  const auto *src_ptr =
      map_cpu_dense_field_packed(this, src, value_type, n, lane_count,
                                 "CPU native packed dense field add-merge");
  auto *dst_ptr =
      map_cpu_dense_field_packed(this, dst, value_type, n, lane_count,
                                 "CPU native packed dense field add-merge");
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((scalar_items + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel =
      cpu_use_parallel_simple_loop(scalar_items, target_threads);
  switch (value_type) {
    case 0:
      cpu_add_merge_run_strided_to_strided_typed<int32_t>(
          src_ptr, dst_ptr, scalar_items, 0, value_size, 0, value_size,
          use_parallel, target_threads, max_threads);
      return 0;
    case 1:
      cpu_add_merge_run_strided_to_strided_typed<float>(
          src_ptr, dst_ptr, scalar_items, 0, value_size, 0, value_size,
          use_parallel, target_threads, max_threads);
      return 0;
    case 2:
      cpu_add_merge_run_strided_to_strided_typed<uint32_t>(
          src_ptr, dst_ptr, scalar_items, 0, value_size, 0, value_size,
          use_parallel, target_threads, max_threads);
      return 0;
    case 3:
      cpu_add_merge_run_strided_to_strided_typed<uint64_t>(
          src_ptr, dst_ptr, scalar_items, 0, value_size, 0, value_size,
          use_parallel, target_threads, max_threads);
      return 0;
    case 4:
      cpu_add_merge_run_strided_to_strided_typed<int64_t>(
          src_ptr, dst_ptr, scalar_items, 0, value_size, 0, value_size,
          use_parallel, target_threads, max_threads);
      return 0;
    case 5:
      cpu_add_merge_run_strided_to_strided_typed<double>(
          src_ptr, dst_ptr, scalar_items, 0, value_size, 0, value_size,
          use_parallel, target_threads, max_threads);
      return 0;
    default:
      TI_ERROR(
          "CPU packed dense field add-merge received an unsupported value "
          "type.");
  }
  return 0;
}

std::size_t Program::scatter_add_dense_field_packed(SNode *src,
                                                    Ndarray *indices,
                                                    SNode *dst,
                                                    int value_type,
                                                    std::size_t src_n,
                                                    std::size_t dst_n,
                                                    int lane_count) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  const Arch arch = compile_config().arch;
  TI_ERROR_IF(!native_dense_field_bulk_arch(arch),
              "Native packed dense field scatter-add is only available on "
              "CPU, CUDA, and Vulkan backends.");
  TI_ERROR_IF(!src || !indices || !dst,
              "Native packed dense field scatter-add received a null "
              "argument.");
  TI_ERROR_IF(!indices || indices->shape.size() != 1 ||
                  indices->get_nelement() != src_n ||
                  indices->get_element_size() != sizeof(int32_t),
              "Native packed dense field scatter-add expects 1D i32 indices "
              "matching source size.");
  const std::size_t n = indices->get_nelement();
  if (n == 0 || dst_n == 0) {
    return 0;
  }
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "Native packed dense field scatter-add received an unsupported "
              "value type.");
  const std::size_t item_bytes = dense_field_packed_bytes(
      value_type, 1, lane_count, "Native packed dense field scatter-add");
  check_dense_field_packed_stride(this, src, value_type, lane_count,
                                  "Native packed dense field scatter-add");
  check_dense_field_packed_stride(this, dst, value_type, lane_count,
                                  "Native packed dense field scatter-add");

  if (arch == Arch::vulkan) {
    return vulkan_scatter_add_dense_field_packed(src, indices, dst, value_type,
                                                 src_n, dst_n, lane_count);
  }

  if (arch == Arch::cuda) {
#ifdef TI_WITH_CUDA
    TI_ERROR_IF(
        n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            dst_n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
        "CUDA packed dense field scatter-add currently supports up "
        "to INT_MAX items.");
    TI_ERROR_IF(static_cast<std::size_t>(lane_count) >
                    static_cast<std::size_t>(std::numeric_limits<int>::max()),
                "CUDA packed dense field scatter-add received an invalid "
                "lane count.");
    auto raw_ptr = [this](DevicePtr ptr, const char *op_name) {
      TI_ERROR_IF(!ptr.device, "{} received a null dense field device.",
                  op_name);
      DeviceAllocation alloc{ptr.device, ptr.alloc_id};
      auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
      TI_ERROR_IF(!base, "{} received a null dense field data pointer.",
                  op_name);
      return static_cast<void *>(reinterpret_cast<uint8_t *>(base) +
                                 ptr.offset);
    };
    void *src_raw = raw_ptr(get_dense_field_device_ptr(src),
                            "CUDA packed dense field scatter-add");
    auto *indices_ptr =
        reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(indices));
    void *dst_raw = raw_ptr(get_dense_field_device_ptr(dst),
                            "CUDA packed dense field scatter-add");
    void *stream = nullptr;
    for (int lane = 0; lane < lane_count; ++lane) {
      const std::size_t lane_offset =
          static_cast<std::size_t>(lane) * value_size;
      cuda::driver_scatter_add_strided(
          src_raw, indices_ptr, dst_raw, static_cast<int>(n),
          static_cast<int>(dst_n),
          static_cast<cuda::CudaTransformValueType>(value_type), lane_offset,
          item_bytes, 0, sizeof(std::int32_t), lane_offset, item_bytes, stream);
    }
    return 0;
#else
    TI_ERROR("CUDA packed dense field scatter-add requires TI_WITH_CUDA=ON.");
#endif
  }

  const auto *src_ptr =
      map_cpu_dense_field_packed(this, src, value_type, src_n, lane_count,
                                 "CPU native packed dense field scatter-add");
  auto *dst_ptr =
      map_cpu_dense_field_packed(this, dst, value_type, dst_n, lane_count,
                                 "CPU native packed dense field scatter-add");
  auto *indices_ptr =
      reinterpret_cast<const int32_t *>(get_ndarray_data_ptr_as_int(indices));
  const int max_threads = std::max(1, compile_config().cpu_max_num_threads);
  const int target_threads =
      cpu_aggregation_target_threads(n, dst_n, max_threads);
  switch (value_type) {
    case 0:
      return cpu_scatter_add_packed_strided_io_typed<int32_t>(
          src_ptr, 0, item_bytes, indices_ptr, dst_ptr, 0, item_bytes, n, dst_n,
          lane_count, max_threads, target_threads);
    case 1:
      return cpu_scatter_add_packed_strided_io_typed<float>(
          src_ptr, 0, item_bytes, indices_ptr, dst_ptr, 0, item_bytes, n, dst_n,
          lane_count, max_threads, target_threads);
    case 2:
      return cpu_scatter_add_packed_strided_io_typed<uint32_t>(
          src_ptr, 0, item_bytes, indices_ptr, dst_ptr, 0, item_bytes, n, dst_n,
          lane_count, max_threads, target_threads);
    case 3:
      return cpu_scatter_add_packed_strided_io_typed<uint64_t>(
          src_ptr, 0, item_bytes, indices_ptr, dst_ptr, 0, item_bytes, n, dst_n,
          lane_count, max_threads, target_threads);
    case 4:
      return cpu_scatter_add_packed_strided_io_typed<int64_t>(
          src_ptr, 0, item_bytes, indices_ptr, dst_ptr, 0, item_bytes, n, dst_n,
          lane_count, max_threads, target_threads);
    case 5:
      return cpu_scatter_add_packed_strided_io_typed<double>(
          src_ptr, 0, item_bytes, indices_ptr, dst_ptr, 0, item_bytes, n, dst_n,
          lane_count, max_threads, target_threads);
    default:
      TI_ERROR(
          "Native packed dense field scatter-add received an unsupported "
          "value type.");
  }
  return 0;
}

std::size_t Program::scatter_add_dense_field_packed_indices_field(
    SNode *src,
    SNode *indices,
    SNode *dst,
    int value_type,
    std::size_t src_n,
    std::size_t indices_n,
    std::size_t dst_n,
    int lane_count) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  const Arch arch = compile_config().arch;
  TI_ERROR_IF(!native_dense_field_bulk_arch(arch),
              "Native packed dense field scatter-add is only available on "
              "CPU, CUDA, and Vulkan backends.");
  TI_ERROR_IF(!src || !indices || !dst,
              "Native packed dense field scatter-add received a null "
              "argument.");
  TI_ERROR_IF(src_n != indices_n,
              "Native packed dense field scatter-add expects source and "
              "field-index sizes to match.");
  const std::size_t n = indices_n;
  if (n == 0 || dst_n == 0) {
    return 0;
  }
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "Native packed dense field scatter-add received an unsupported "
              "value type.");
  const std::size_t item_bytes = dense_field_packed_bytes(
      value_type, 1, lane_count, "Native packed dense field scatter-add");
  check_dense_field_packed_stride(this, src, value_type, lane_count,
                                  "Native packed dense field scatter-add");
  check_dense_field_packed_stride(this, dst, value_type, lane_count,
                                  "Native packed dense field scatter-add");
  TI_ERROR_IF(
      get_dense_field_stride(indices, sizeof(int32_t)) != sizeof(int32_t),
      "Native packed dense field scatter-add requires contiguous i32 "
      "field indices.");

  if (arch == Arch::vulkan) {
    return vulkan_scatter_add_dense_field_packed_indices_field(
        src, indices, dst, value_type, src_n, indices_n, dst_n, lane_count);
  }

  if (arch == Arch::cuda) {
#ifdef TI_WITH_CUDA
    TI_ERROR_IF(
        n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            dst_n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
        "CUDA packed dense field scatter-add currently supports up "
        "to INT_MAX items.");
    TI_ERROR_IF(static_cast<std::size_t>(lane_count) >
                    static_cast<std::size_t>(std::numeric_limits<int>::max()),
                "CUDA packed dense field scatter-add received an invalid "
                "lane count.");
    auto raw_ptr = [this](DevicePtr ptr, const char *op_name) {
      TI_ERROR_IF(!ptr.device, "{} received a null dense field device.",
                  op_name);
      DeviceAllocation alloc{ptr.device, ptr.alloc_id};
      auto *base = program_impl_->get_device_alloc_info_ptr(alloc);
      TI_ERROR_IF(!base, "{} received a null dense field data pointer.",
                  op_name);
      return static_cast<void *>(reinterpret_cast<uint8_t *>(base) +
                                 ptr.offset);
    };
    void *src_raw = raw_ptr(get_dense_field_device_ptr(src),
                            "CUDA packed dense field scatter-add");
    void *indices_raw = raw_ptr(get_dense_field_device_ptr(indices),
                                "CUDA packed dense field scatter-add");
    void *dst_raw = raw_ptr(get_dense_field_device_ptr(dst),
                            "CUDA packed dense field scatter-add");
    void *stream = nullptr;
    for (int lane = 0; lane < lane_count; ++lane) {
      const std::size_t lane_offset =
          static_cast<std::size_t>(lane) * value_size;
      cuda::driver_scatter_add_strided(
          src_raw, indices_raw, dst_raw, static_cast<int>(n),
          static_cast<int>(dst_n),
          static_cast<cuda::CudaTransformValueType>(value_type), lane_offset,
          item_bytes, 0, sizeof(std::int32_t), lane_offset, item_bytes, stream);
    }
    return 0;
#else
    TI_ERROR("CUDA packed dense field scatter-add requires TI_WITH_CUDA=ON.");
#endif
  }

  const auto *src_ptr =
      map_cpu_dense_field_packed(this, src, value_type, src_n, lane_count,
                                 "CPU native packed dense field scatter-add");
  const auto *indices_ptr_bytes =
      map_cpu_dense_field(this, indices, 0, indices_n,
                          "CPU native packed dense field scatter-add", nullptr);
  auto *dst_ptr =
      map_cpu_dense_field_packed(this, dst, value_type, dst_n, lane_count,
                                 "CPU native packed dense field scatter-add");
  const auto *indices_ptr =
      reinterpret_cast<const int32_t *>(indices_ptr_bytes);
  const int max_threads = std::max(1, compile_config().cpu_max_num_threads);
  const int target_threads =
      cpu_aggregation_target_threads(n, dst_n, max_threads);
  switch (value_type) {
    case 0:
      return cpu_scatter_add_packed_strided_io_typed<int32_t>(
          src_ptr, 0, item_bytes, indices_ptr, dst_ptr, 0, item_bytes, n, dst_n,
          lane_count, max_threads, target_threads);
    case 1:
      return cpu_scatter_add_packed_strided_io_typed<float>(
          src_ptr, 0, item_bytes, indices_ptr, dst_ptr, 0, item_bytes, n, dst_n,
          lane_count, max_threads, target_threads);
    case 2:
      return cpu_scatter_add_packed_strided_io_typed<uint32_t>(
          src_ptr, 0, item_bytes, indices_ptr, dst_ptr, 0, item_bytes, n, dst_n,
          lane_count, max_threads, target_threads);
    case 3:
      return cpu_scatter_add_packed_strided_io_typed<uint64_t>(
          src_ptr, 0, item_bytes, indices_ptr, dst_ptr, 0, item_bytes, n, dst_n,
          lane_count, max_threads, target_threads);
    case 4:
      return cpu_scatter_add_packed_strided_io_typed<int64_t>(
          src_ptr, 0, item_bytes, indices_ptr, dst_ptr, 0, item_bytes, n, dst_n,
          lane_count, max_threads, target_threads);
    case 5:
      return cpu_scatter_add_packed_strided_io_typed<double>(
          src_ptr, 0, item_bytes, indices_ptr, dst_ptr, 0, item_bytes, n, dst_n,
          lane_count, max_threads, target_threads);
    default:
      TI_ERROR(
          "Native packed dense field scatter-add received an unsupported "
          "value type.");
  }
  return 0;
}

std::size_t Program::cpu_add_scaled_dense_field(SNode *src,
                                                SNode *dst,
                                                int value_type,
                                                std::size_t n,
                                                double scale) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native dense field scaled-add is only available on CPU "
              "backends.");
  TI_ERROR_IF(!src || !dst,
              "CPU dense field scaled-add received a null field.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CPU dense field scaled-add received an unsupported value type.");
  if (n == 0) {
    return 0;
  }
  std::size_t src_stride = 0;
  std::size_t dst_stride = 0;
  const auto *src_ptr = map_cpu_dense_field(
      this, src, value_type, n, "CPU native dense field scaled-add",
      &src_stride);
  auto *dst_ptr = map_cpu_dense_field(
      this, dst, value_type, n, "CPU native dense field scaled-add",
      &dst_stride);
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = cpu_use_parallel_simple_loop(n, target_threads);
  switch (value_type) {
    case 1:
      cpu_add_scaled_run_strided_to_strided_typed<float>(
          src_ptr, dst_ptr, n, 0, src_stride, 0, dst_stride,
          static_cast<float>(scale), use_parallel, target_threads,
          max_threads);
      return 0;
    case 5:
      cpu_add_scaled_run_strided_to_strided_typed<double>(
          src_ptr, dst_ptr, n, 0, src_stride, 0, dst_stride, scale,
          use_parallel, target_threads, max_threads);
      return 0;
    default:
      TI_ERROR(
          "CPU dense field scaled-add is supported only for f32/f64 gradients.");
  }
  return 0;
}

std::size_t Program::cpu_add_scalar_field_to_dense_field(SNode *src,
                                                         SNode *dst,
                                                         int value_type,
                                                         std::size_t n) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native scalar-to-dense add is only available on CPU "
              "backends.");
  TI_ERROR_IF(!src || !dst,
              "CPU scalar-to-dense add received a null field.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CPU scalar-to-dense add received an unsupported value type.");
  if (n == 0) {
    return 0;
  }
  auto *src_ptr = map_cpu_dense_field(this, src, value_type, 1,
                                      "CPU native scalar-to-dense add", nullptr);
  std::size_t dst_stride = 0;
  auto *dst_ptr = map_cpu_dense_field(this, dst, value_type, n,
                                      "CPU native scalar-to-dense add",
                                      &dst_stride);
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = cpu_use_parallel_simple_loop(n, target_threads);
  switch (value_type) {
    case 0:
      cpu_add_merge_run_strided_to_strided_typed<int32_t>(
          src_ptr, dst_ptr, n, 0, 0, 0, dst_stride, use_parallel,
          target_threads, max_threads);
      return 0;
    case 1:
      cpu_add_merge_run_strided_to_strided_typed<float>(
          src_ptr, dst_ptr, n, 0, 0, 0, dst_stride, use_parallel,
          target_threads, max_threads);
      return 0;
    case 2:
      cpu_add_merge_run_strided_to_strided_typed<uint32_t>(
          src_ptr, dst_ptr, n, 0, 0, 0, dst_stride, use_parallel,
          target_threads, max_threads);
      return 0;
    case 3:
      cpu_add_merge_run_strided_to_strided_typed<uint64_t>(
          src_ptr, dst_ptr, n, 0, 0, 0, dst_stride, use_parallel,
          target_threads, max_threads);
      return 0;
    case 4:
      cpu_add_merge_run_strided_to_strided_typed<int64_t>(
          src_ptr, dst_ptr, n, 0, 0, 0, dst_stride, use_parallel,
          target_threads, max_threads);
      return 0;
    case 5:
      cpu_add_merge_run_strided_to_strided_typed<double>(
          src_ptr, dst_ptr, n, 0, 0, 0, dst_stride, use_parallel,
          target_threads, max_threads);
      return 0;
    default:
      TI_ERROR("CPU scalar-to-dense add received an unsupported value type.");
  }
  return 0;
}

bool Program::cpu_indexed_copy_available() const {
  return arch_is_cpu(compile_config().arch);
}

std::size_t Program::cpu_gather_ndarray(Ndarray *src,
                                        Ndarray *indices,
                                        Ndarray *dst) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native gather is only available on CPU backends.");
  TI_ERROR_IF(!src || !indices || !dst,
              "CPU native gather received a null ndarray.");
  TI_ERROR_IF(src->shape.size() != 1 || indices->shape.size() != 1 ||
                  dst->shape.size() != 1,
              "CPU native gather currently expects 1D ndarrays.");
  TI_ERROR_IF(indices->get_nelement() != dst->get_nelement(),
              "CPU native gather expects indices and destination sizes to "
              "match.");
  TI_ERROR_IF(src->get_element_size() != dst->get_element_size(),
              "CPU native gather source and destination dtypes differ.");
  const std::size_t item_bytes = src->get_element_size();
  TI_ERROR_IF(item_bytes == 0 || item_bytes % sizeof(uint32_t) != 0 ||
                  indices->get_element_size() != sizeof(int32_t),
              "CPU native gather currently expects 4-byte aligned values and "
              "i32 indices.");
  const std::size_t n = indices->get_nelement();
  const std::size_t src_items = src->get_nelement();
  if (n == 0) {
    return 0;
  }
  auto *src_ptr =
      reinterpret_cast<const uint8_t *>(get_ndarray_data_ptr_as_int(src));
  auto *indices_ptr =
      reinterpret_cast<const int32_t *>(get_ndarray_data_ptr_as_int(indices));
  auto *dst_ptr = reinterpret_cast<uint8_t *>(get_ndarray_data_ptr_as_int(dst));
  TI_ERROR_IF(!src_ptr || !indices_ptr || !dst_ptr,
              "CPU native gather received a null data pointer.");
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  if (n >= 65536 && target_threads > 1) {
    CpuIndexedCopyTaskContext ctx;
    ctx.src = src_ptr;
    ctx.indices = indices_ptr;
    ctx.dst = dst_ptr;
    ctx.n = n;
    ctx.index_bound = src_items;
    ctx.item_bytes = item_bytes;
    ctx.scatter = false;
    ctx.num_threads = target_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(target_threads, target_threads, &ctx, cpu_indexed_copy_task);
    return 0;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const auto index = static_cast<std::size_t>(indices_ptr[i]);
    if (index < src_items) {
      std::memcpy(dst_ptr + i * item_bytes, src_ptr + index * item_bytes,
                  item_bytes);
    } else {
      std::memset(dst_ptr + i * item_bytes, 0, item_bytes);
    }
  }
  return 0;
}

std::size_t Program::cpu_gather_strided_ndarray(Ndarray *src,
                                                Ndarray *indices,
                                                Ndarray *dst,
                                                std::size_t item_bytes,
                                                std::size_t src_offset,
                                                std::size_t src_stride,
                                                std::size_t dst_offset,
                                                std::size_t dst_stride) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native strided gather is only available on CPU backends.");
  check_indexed_copy_strided_request("CPU native", src, indices, dst,
                                     item_bytes, src_offset, src_stride,
                                     dst_offset, dst_stride, false);
  const std::size_t n = indices->get_nelement();
  const std::size_t src_items = src->get_nelement();
  if (n == 0) {
    return 0;
  }
  auto *src_ptr =
      reinterpret_cast<const uint8_t *>(get_ndarray_data_ptr_as_int(src));
  auto *indices_ptr =
      reinterpret_cast<const int32_t *>(get_ndarray_data_ptr_as_int(indices));
  auto *dst_ptr = reinterpret_cast<uint8_t *>(get_ndarray_data_ptr_as_int(dst));
  TI_ERROR_IF(!src_ptr || !indices_ptr || !dst_ptr,
              "CPU native strided gather received a null data pointer.");
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  if (n >= 65536 && target_threads > 1) {
    CpuStridedIndexedCopyTaskContext ctx;
    ctx.src = src_ptr;
    ctx.indices = indices_ptr;
    ctx.dst = dst_ptr;
    ctx.n = n;
    ctx.index_bound = src_items;
    ctx.item_bytes = item_bytes;
    ctx.src_offset = src_offset;
    ctx.src_stride = src_stride;
    ctx.dst_offset = dst_offset;
    ctx.dst_stride = dst_stride;
    ctx.scatter = false;
    ctx.num_threads = target_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(target_threads, target_threads, &ctx,
             cpu_strided_indexed_copy_task);
    return 0;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const auto index = static_cast<std::size_t>(indices_ptr[i]);
    if (index < src_items) {
      std::memcpy(dst_ptr + dst_offset + i * dst_stride,
                  src_ptr + src_offset + index * src_stride, item_bytes);
    } else {
      std::memset(dst_ptr + dst_offset + i * dst_stride, 0, item_bytes);
    }
  }
  return 0;
}

std::size_t Program::cpu_gather_dense_field(SNode *src,
                                            Ndarray *indices,
                                            SNode *dst,
                                            int value_type,
                                            std::size_t src_n,
                                            std::size_t dst_n) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native dense field gather is only available on CPU "
              "backends.");
  check_indexed_copy_dense_field_request(this, "CPU native", src, indices, dst,
                                         value_type, src_n, dst_n, false);
  const std::size_t n = indices->get_nelement();
  if (n == 0) {
    return 0;
  }
  const std::size_t item_bytes = primitive_value_type_size(value_type);
  std::size_t src_stride = 0;
  std::size_t dst_stride = 0;
  const auto *src_ptr = map_cpu_dense_field(
      this, src, value_type, src_n, "CPU native dense field gather",
      &src_stride);
  auto *dst_ptr = map_cpu_dense_field(
      this, dst, value_type, dst_n, "CPU native dense field gather",
      &dst_stride);
  auto *indices_ptr =
      reinterpret_cast<const int32_t *>(get_ndarray_data_ptr_as_int(indices));
  TI_ERROR_IF(!src_ptr || !indices_ptr || !dst_ptr,
              "CPU native dense field gather received a null data pointer.");
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  if (n >= 65536 && target_threads > 1) {
    CpuStridedIndexedCopyTaskContext ctx;
    ctx.src = src_ptr;
    ctx.indices = indices_ptr;
    ctx.dst = dst_ptr;
    ctx.n = n;
    ctx.index_bound = src_n;
    ctx.item_bytes = item_bytes;
    ctx.src_offset = 0;
    ctx.src_stride = src_stride;
    ctx.dst_offset = 0;
    ctx.dst_stride = dst_stride;
    ctx.scatter = false;
    ctx.num_threads = target_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(target_threads, target_threads, &ctx,
             cpu_strided_indexed_copy_task);
    return 0;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const auto index = static_cast<std::size_t>(indices_ptr[i]);
    if (index < src_n) {
      std::memcpy(dst_ptr + i * dst_stride, src_ptr + index * src_stride,
                  item_bytes);
    } else {
      std::memset(dst_ptr + i * dst_stride, 0, item_bytes);
    }
  }
  return 0;
}

std::size_t Program::cpu_gather_dense_field_packed(SNode *src,
                                                   Ndarray *indices,
                                                   SNode *dst,
                                                   int value_type,
                                                   std::size_t src_n,
                                                   std::size_t dst_n,
                                                   int lane_count) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native packed dense field gather is only available on CPU "
              "backends.");
  TI_ERROR_IF(!indices || indices->shape.size() != 1 ||
                  indices->get_nelement() != dst_n ||
                  indices->get_element_size() != sizeof(int32_t),
              "CPU native packed dense field gather expects 1D i32 indices "
              "matching destination size.");
  const std::size_t n = indices->get_nelement();
  if (n == 0) {
    return 0;
  }
  const std::size_t item_bytes =
      dense_field_packed_bytes(value_type, 1, lane_count,
                               "CPU packed dense field gather");
  const auto *src_ptr = map_cpu_dense_field_packed(
      this, src, value_type, src_n, lane_count,
      "CPU packed dense field gather");
  auto *dst_ptr = map_cpu_dense_field_packed(
      this, dst, value_type, dst_n, lane_count,
      "CPU packed dense field gather");
  auto *indices_ptr =
      reinterpret_cast<const int32_t *>(get_ndarray_data_ptr_as_int(indices));
  TI_ERROR_IF(!src_ptr || !indices_ptr || !dst_ptr,
              "CPU native packed dense field gather received a null data "
              "pointer.");
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  if (n >= 65536 && target_threads > 1) {
    CpuStridedIndexedCopyTaskContext ctx;
    ctx.src = src_ptr;
    ctx.indices = indices_ptr;
    ctx.dst = dst_ptr;
    ctx.n = n;
    ctx.index_bound = src_n;
    ctx.item_bytes = item_bytes;
    ctx.src_offset = 0;
    ctx.src_stride = item_bytes;
    ctx.dst_offset = 0;
    ctx.dst_stride = item_bytes;
    ctx.scatter = false;
    ctx.num_threads = target_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(target_threads, target_threads, &ctx,
             cpu_strided_indexed_copy_task);
    return 0;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const auto index = static_cast<std::size_t>(indices_ptr[i]);
    if (index < src_n) {
      std::memcpy(dst_ptr + i * item_bytes, src_ptr + index * item_bytes,
                  item_bytes);
    } else {
      std::memset(dst_ptr + i * item_bytes, 0, item_bytes);
    }
  }
  return 0;
}

std::size_t Program::cpu_gather_dense_field_packed_indices_field(
    SNode *src,
    SNode *indices,
    SNode *dst,
    int value_type,
    std::size_t src_n,
    std::size_t indices_n,
    std::size_t dst_n,
    int lane_count) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native packed dense field gather is only available on CPU "
              "backends.");
  TI_ERROR_IF(indices_n != dst_n,
              "CPU native packed dense field gather expects field indices "
              "matching destination size.");
  const std::size_t n = indices_n;
  if (n == 0) {
    return 0;
  }
  const std::size_t item_bytes =
      dense_field_packed_bytes(value_type, 1, lane_count,
                               "CPU packed dense field gather");
  const auto *src_ptr = map_cpu_dense_field_packed(
      this, src, value_type, src_n, lane_count,
      "CPU packed dense field gather");
  std::size_t indices_stride = 0;
  const auto *indices_ptr_bytes = map_cpu_dense_field(
      this, indices, 0, indices_n, "CPU packed dense field gather",
      &indices_stride);
  auto *dst_ptr = map_cpu_dense_field_packed(
      this, dst, value_type, dst_n, lane_count,
      "CPU packed dense field gather");
  TI_ERROR_IF(indices_stride != sizeof(int32_t),
              "CPU packed dense field gather requires contiguous i32 field "
              "indices.");
  const auto *indices_ptr =
      reinterpret_cast<const int32_t *>(indices_ptr_bytes);
  TI_ERROR_IF(!src_ptr || !indices_ptr || !dst_ptr,
              "CPU native packed dense field gather received a null data "
              "pointer.");
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  if (n >= 65536 && target_threads > 1) {
    CpuStridedIndexedCopyTaskContext ctx;
    ctx.src = src_ptr;
    ctx.indices = indices_ptr;
    ctx.dst = dst_ptr;
    ctx.n = n;
    ctx.index_bound = src_n;
    ctx.item_bytes = item_bytes;
    ctx.src_offset = 0;
    ctx.src_stride = item_bytes;
    ctx.dst_offset = 0;
    ctx.dst_stride = item_bytes;
    ctx.scatter = false;
    ctx.num_threads = target_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(target_threads, target_threads, &ctx,
             cpu_strided_indexed_copy_task);
    return 0;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const auto index = static_cast<std::size_t>(indices_ptr[i]);
    if (index < src_n) {
      std::memcpy(dst_ptr + i * item_bytes, src_ptr + index * item_bytes,
                  item_bytes);
    } else {
      std::memset(dst_ptr + i * item_bytes, 0, item_bytes);
    }
  }
  return 0;
}

std::size_t Program::cpu_gather_dense_field_indices_field(
    SNode *src,
    SNode *indices,
    SNode *dst,
    int value_type,
    std::size_t src_n,
    std::size_t indices_n,
    std::size_t dst_n) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native dense field gather is only available on CPU "
              "backends.");
  check_indexed_copy_dense_field_indices_field_request(
      this, "CPU native", src, indices, dst, value_type, src_n, indices_n,
      dst_n, false);
  const std::size_t n = indices_n;
  if (n == 0) {
    return 0;
  }
  const std::size_t item_bytes = primitive_value_type_size(value_type);
  std::size_t src_stride = 0;
  std::size_t indices_stride = 0;
  std::size_t dst_stride = 0;
  const auto *src_ptr = map_cpu_dense_field(
      this, src, value_type, src_n, "CPU native dense field gather",
      &src_stride);
  const auto *indices_ptr_bytes = map_cpu_dense_field(
      this, indices, 0, indices_n, "CPU native dense field gather",
      &indices_stride);
  auto *dst_ptr = map_cpu_dense_field(
      this, dst, value_type, dst_n, "CPU native dense field gather",
      &dst_stride);
  TI_ERROR_IF(indices_stride != sizeof(int32_t),
              "CPU native dense field gather requires contiguous i32 field "
              "indices.");
  const auto *indices_ptr =
      reinterpret_cast<const int32_t *>(indices_ptr_bytes);
  TI_ERROR_IF(!src_ptr || !indices_ptr || !dst_ptr,
              "CPU native dense field gather received a null data pointer.");
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  if (n >= 65536 && target_threads > 1) {
    CpuStridedIndexedCopyTaskContext ctx;
    ctx.src = src_ptr;
    ctx.indices = indices_ptr;
    ctx.dst = dst_ptr;
    ctx.n = n;
    ctx.index_bound = src_n;
    ctx.item_bytes = item_bytes;
    ctx.src_offset = 0;
    ctx.src_stride = src_stride;
    ctx.dst_offset = 0;
    ctx.dst_stride = dst_stride;
    ctx.scatter = false;
    ctx.num_threads = target_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(target_threads, target_threads, &ctx,
             cpu_strided_indexed_copy_task);
    return 0;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const auto index = static_cast<std::size_t>(indices_ptr[i]);
    if (index < src_n) {
      std::memcpy(dst_ptr + i * dst_stride, src_ptr + index * src_stride,
                  item_bytes);
    } else {
      std::memset(dst_ptr + i * dst_stride, 0, item_bytes);
    }
  }
  return 0;
}

std::size_t Program::cpu_gather_add_ndarray(Ndarray *src,
                                            Ndarray *indices,
                                            Ndarray *dst,
                                            int value_type) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native gather-add is only available on CPU backends.");
  TI_ERROR_IF(!src || !indices || !dst,
              "CPU native gather-add received a null ndarray.");
  TI_ERROR_IF(src->shape.size() != 1 || indices->shape.size() != 1 ||
                  dst->shape.size() != 1,
              "CPU native gather-add expects 1D ndarrays.");
  TI_ERROR_IF(indices->get_nelement() != dst->get_nelement(),
              "CPU native gather-add expects indices and destination sizes to "
              "match.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CPU native gather-add received an unsupported value type.");
  TI_ERROR_IF(src->get_element_size() != value_size ||
                  dst->get_element_size() != value_size ||
                  indices->get_element_size() != sizeof(int32_t),
              "CPU native gather-add dtype does not match value type or i32 "
              "index size.");
  const std::size_t n = indices->get_nelement();
  if (n == 0) {
    return 0;
  }
  const auto *src_ptr =
      reinterpret_cast<const uint8_t *>(get_ndarray_data_ptr_as_int(src));
  const auto *indices_ptr =
      reinterpret_cast<const int32_t *>(get_ndarray_data_ptr_as_int(indices));
  auto *dst_ptr = reinterpret_cast<uint8_t *>(get_ndarray_data_ptr_as_int(dst));
  TI_ERROR_IF(!src_ptr || !indices_ptr || !dst_ptr,
              "CPU native gather-add received a null data pointer.");
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = cpu_use_parallel_simple_loop(n, target_threads);
  switch (value_type) {
    case 1:
      cpu_gather_add_run_strided_to_strided_typed<float>(
          src_ptr, indices_ptr, dst_ptr, n, src->get_nelement(), 0,
          value_size, 0, value_size, use_parallel, target_threads,
          max_threads);
      return 0;
    case 5:
      cpu_gather_add_run_strided_to_strided_typed<double>(
          src_ptr, indices_ptr, dst_ptr, n, src->get_nelement(), 0,
          value_size, 0, value_size, use_parallel, target_threads,
          max_threads);
      return 0;
    default:
      TI_ERROR("CPU native gather-add is supported only for f32/f64 "
               "gradients.");
  }
  return 0;
}

std::size_t Program::cpu_gather_add_dense_field(SNode *src,
                                                Ndarray *indices,
                                                SNode *dst,
                                                int value_type,
                                                std::size_t src_n,
                                                std::size_t dst_n) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native dense field gather-add is only available on CPU "
              "backends.");
  check_indexed_copy_dense_field_request(this, "CPU native", src, indices, dst,
                                         value_type, src_n, dst_n, false);
  const std::size_t n = indices->get_nelement();
  TI_ERROR_IF(n != dst_n,
              "CPU native dense field gather-add expects destination size to "
              "match indices size.");
  if (n == 0) {
    return 0;
  }
  std::size_t src_stride = 0;
  std::size_t dst_stride = 0;
  const auto *src_ptr = map_cpu_dense_field(
      this, src, value_type, src_n, "CPU native dense field gather-add",
      &src_stride);
  auto *dst_ptr = map_cpu_dense_field(
      this, dst, value_type, dst_n, "CPU native dense field gather-add",
      &dst_stride);
  const auto *indices_ptr =
      reinterpret_cast<const int32_t *>(get_ndarray_data_ptr_as_int(indices));
  TI_ERROR_IF(!src_ptr || !indices_ptr || !dst_ptr,
              "CPU native dense field gather-add received a null data "
              "pointer.");
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = cpu_use_parallel_simple_loop(n, target_threads);
  switch (value_type) {
    case 1:
      cpu_gather_add_run_strided_to_strided_typed<float>(
          src_ptr, indices_ptr, dst_ptr, n, src_n, 0, src_stride, 0,
          dst_stride, use_parallel, target_threads, max_threads);
      return 0;
    case 5:
      cpu_gather_add_run_strided_to_strided_typed<double>(
          src_ptr, indices_ptr, dst_ptr, n, src_n, 0, src_stride, 0,
          dst_stride, use_parallel, target_threads, max_threads);
      return 0;
    default:
      TI_ERROR("CPU native dense field gather-add is supported only for "
               "f32/f64 gradients.");
  }
  return 0;
}

std::size_t Program::cpu_gather_add_dense_field_indices_field(
    SNode *src,
    SNode *indices,
    SNode *dst,
    int value_type,
    std::size_t src_n,
    std::size_t indices_n,
    std::size_t dst_n) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native dense field gather-add is only available on CPU "
              "backends.");
  check_indexed_copy_dense_field_indices_field_request(
      this, "CPU native", src, indices, dst, value_type, src_n, indices_n,
      dst_n, false);
  TI_ERROR_IF(indices_n != dst_n,
              "CPU native dense field gather-add expects destination size to "
              "match indices size.");
  if (indices_n == 0) {
    return 0;
  }
  std::size_t src_stride = 0;
  std::size_t indices_stride = 0;
  std::size_t dst_stride = 0;
  const auto *src_ptr = map_cpu_dense_field(
      this, src, value_type, src_n, "CPU native dense field gather-add",
      &src_stride);
  const auto *indices_ptr_bytes = map_cpu_dense_field(
      this, indices, 0, indices_n, "CPU native dense field gather-add",
      &indices_stride);
  auto *dst_ptr = map_cpu_dense_field(
      this, dst, value_type, dst_n, "CPU native dense field gather-add",
      &dst_stride);
  TI_ERROR_IF(indices_stride != sizeof(int32_t),
              "CPU native dense field gather-add requires contiguous i32 "
              "field indices.");
  const auto *indices_ptr =
      reinterpret_cast<const int32_t *>(indices_ptr_bytes);
  TI_ERROR_IF(!src_ptr || !indices_ptr || !dst_ptr,
              "CPU native dense field gather-add received a null data "
              "pointer.");
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((indices_n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel =
      cpu_use_parallel_simple_loop(indices_n, target_threads);
  switch (value_type) {
    case 1:
      cpu_gather_add_run_strided_to_strided_typed<float>(
          src_ptr, indices_ptr, dst_ptr, indices_n, src_n, 0, src_stride, 0,
          dst_stride, use_parallel, target_threads, max_threads);
      return 0;
    case 5:
      cpu_gather_add_run_strided_to_strided_typed<double>(
          src_ptr, indices_ptr, dst_ptr, indices_n, src_n, 0, src_stride, 0,
          dst_stride, use_parallel, target_threads, max_threads);
      return 0;
    default:
      TI_ERROR("CPU native dense field gather-add is supported only for "
               "f32/f64 gradients.");
  }
  return 0;
}

std::size_t Program::cpu_scatter_ndarray(Ndarray *src,
                                         Ndarray *indices,
                                         Ndarray *dst) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native scatter is only available on CPU backends.");
  TI_ERROR_IF(!src || !indices || !dst,
              "CPU native scatter received a null ndarray.");
  TI_ERROR_IF(src->shape.size() != 1 || indices->shape.size() != 1 ||
                  dst->shape.size() != 1,
              "CPU native scatter currently expects 1D ndarrays.");
  TI_ERROR_IF(src->get_nelement() != indices->get_nelement(),
              "CPU native scatter expects source and indices sizes to match.");
  TI_ERROR_IF(src->get_element_size() != dst->get_element_size(),
              "CPU native scatter source and destination dtypes differ.");
  const std::size_t item_bytes = src->get_element_size();
  TI_ERROR_IF(item_bytes == 0 || item_bytes % sizeof(uint32_t) != 0 ||
                  indices->get_element_size() != sizeof(int32_t),
              "CPU native scatter currently expects 4-byte aligned values and "
              "i32 indices.");
  const std::size_t n = indices->get_nelement();
  const std::size_t dst_items = dst->get_nelement();
  if (n == 0) {
    return 0;
  }
  auto *src_ptr =
      reinterpret_cast<const uint8_t *>(get_ndarray_data_ptr_as_int(src));
  auto *indices_ptr =
      reinterpret_cast<const int32_t *>(get_ndarray_data_ptr_as_int(indices));
  auto *dst_ptr = reinterpret_cast<uint8_t *>(get_ndarray_data_ptr_as_int(dst));
  TI_ERROR_IF(!src_ptr || !indices_ptr || !dst_ptr,
              "CPU native scatter received a null data pointer.");
  validate_cpu_plain_scatter_indices(indices_ptr, n, dst_items);
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int target_threads =
      cpu_indexed_copy_target_threads(n, max_threads, true);
  if (n >= 65536 && target_threads > 1) {
    CpuIndexedCopyTaskContext ctx;
    ctx.src = src_ptr;
    ctx.indices = indices_ptr;
    ctx.dst = dst_ptr;
    ctx.n = n;
    ctx.index_bound = dst_items;
    ctx.item_bytes = item_bytes;
    ctx.scatter = true;
    ctx.num_threads = target_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(target_threads, target_threads, &ctx, cpu_indexed_copy_task);
    return 0;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const auto index = static_cast<std::size_t>(indices_ptr[i]);
    if (index < dst_items) {
      std::memcpy(dst_ptr + index * item_bytes, src_ptr + i * item_bytes,
                  item_bytes);
    }
  }
  return 0;
}

std::size_t Program::cpu_scatter_strided_ndarray(Ndarray *src,
                                                 Ndarray *indices,
                                                 Ndarray *dst,
                                                 std::size_t item_bytes,
                                                 std::size_t src_offset,
                                                 std::size_t src_stride,
                                                 std::size_t dst_offset,
                                                 std::size_t dst_stride) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native strided scatter is only available on CPU backends.");
  check_indexed_copy_strided_request("CPU native", src, indices, dst,
                                     item_bytes, src_offset, src_stride,
                                     dst_offset, dst_stride, true);
  const std::size_t n = indices->get_nelement();
  const std::size_t dst_items = dst->get_nelement();
  if (n == 0) {
    return 0;
  }
  auto *src_ptr =
      reinterpret_cast<const uint8_t *>(get_ndarray_data_ptr_as_int(src));
  auto *indices_ptr =
      reinterpret_cast<const int32_t *>(get_ndarray_data_ptr_as_int(indices));
  auto *dst_ptr = reinterpret_cast<uint8_t *>(get_ndarray_data_ptr_as_int(dst));
  TI_ERROR_IF(!src_ptr || !indices_ptr || !dst_ptr,
              "CPU native strided scatter received a null data pointer.");
  validate_cpu_plain_scatter_indices(indices_ptr, n, dst_items);
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int target_threads =
      cpu_indexed_copy_target_threads(n, max_threads, true);
  if (n >= 65536 && target_threads > 1) {
    CpuStridedIndexedCopyTaskContext ctx;
    ctx.src = src_ptr;
    ctx.indices = indices_ptr;
    ctx.dst = dst_ptr;
    ctx.n = n;
    ctx.index_bound = dst_items;
    ctx.item_bytes = item_bytes;
    ctx.src_offset = src_offset;
    ctx.src_stride = src_stride;
    ctx.dst_offset = dst_offset;
    ctx.dst_stride = dst_stride;
    ctx.scatter = true;
    ctx.num_threads = target_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(target_threads, target_threads, &ctx,
             cpu_strided_indexed_copy_task);
    return 0;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const auto index = static_cast<std::size_t>(indices_ptr[i]);
    if (index < dst_items) {
      std::memcpy(dst_ptr + dst_offset + index * dst_stride,
                  src_ptr + src_offset + i * src_stride, item_bytes);
    }
  }
  return 0;
}

std::size_t Program::cpu_scatter_dense_field(SNode *src,
                                             Ndarray *indices,
                                             SNode *dst,
                                             int value_type,
                                             std::size_t src_n,
                                             std::size_t dst_n) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native dense field scatter is only available on CPU "
              "backends.");
  check_indexed_copy_dense_field_request(this, "CPU native", src, indices, dst,
                                         value_type, src_n, dst_n, true);
  const std::size_t n = indices->get_nelement();
  if (n == 0) {
    return 0;
  }
  const std::size_t item_bytes = primitive_value_type_size(value_type);
  std::size_t src_stride = 0;
  std::size_t dst_stride = 0;
  const auto *src_ptr = map_cpu_dense_field(
      this, src, value_type, src_n, "CPU native dense field scatter",
      &src_stride);
  auto *dst_ptr = map_cpu_dense_field(
      this, dst, value_type, dst_n, "CPU native dense field scatter",
      &dst_stride);
  auto *indices_ptr =
      reinterpret_cast<const int32_t *>(get_ndarray_data_ptr_as_int(indices));
  TI_ERROR_IF(!src_ptr || !indices_ptr || !dst_ptr,
              "CPU native dense field scatter received a null data pointer.");
  validate_cpu_plain_scatter_indices(indices_ptr, n, dst_n);
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  if (n >= 65536 && target_threads > 1) {
    CpuStridedIndexedCopyTaskContext ctx;
    ctx.src = src_ptr;
    ctx.indices = indices_ptr;
    ctx.dst = dst_ptr;
    ctx.n = n;
    ctx.index_bound = dst_n;
    ctx.item_bytes = item_bytes;
    ctx.src_offset = 0;
    ctx.src_stride = src_stride;
    ctx.dst_offset = 0;
    ctx.dst_stride = dst_stride;
    ctx.scatter = true;
    ctx.num_threads = target_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(target_threads, target_threads, &ctx,
             cpu_strided_indexed_copy_task);
    return 0;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const auto index = static_cast<std::size_t>(indices_ptr[i]);
    if (index < dst_n) {
      std::memcpy(dst_ptr + index * dst_stride, src_ptr + i * src_stride,
                  item_bytes);
    }
  }
  return 0;
}

std::size_t Program::cpu_scatter_dense_field_packed(SNode *src,
                                                    Ndarray *indices,
                                                    SNode *dst,
                                                    int value_type,
                                                    std::size_t src_n,
                                                    std::size_t dst_n,
                                                    int lane_count) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native packed dense field scatter is only available on CPU "
              "backends.");
  TI_ERROR_IF(!indices || indices->shape.size() != 1 ||
                  indices->get_nelement() != src_n ||
                  indices->get_element_size() != sizeof(int32_t),
              "CPU native packed dense field scatter expects 1D i32 indices "
              "matching source size.");
  const std::size_t n = indices->get_nelement();
  if (n == 0) {
    return 0;
  }
  const std::size_t item_bytes =
      dense_field_packed_bytes(value_type, 1, lane_count,
                               "CPU packed dense field scatter");
  const auto *src_ptr = map_cpu_dense_field_packed(
      this, src, value_type, src_n, lane_count,
      "CPU packed dense field scatter");
  auto *dst_ptr = map_cpu_dense_field_packed(
      this, dst, value_type, dst_n, lane_count,
      "CPU packed dense field scatter");
  auto *indices_ptr =
      reinterpret_cast<const int32_t *>(get_ndarray_data_ptr_as_int(indices));
  TI_ERROR_IF(!src_ptr || !indices_ptr || !dst_ptr,
              "CPU native packed dense field scatter received a null data "
              "pointer.");
  validate_cpu_plain_scatter_indices(indices_ptr, n, dst_n);
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  if (n >= 65536 && target_threads > 1) {
    CpuStridedIndexedCopyTaskContext ctx;
    ctx.src = src_ptr;
    ctx.indices = indices_ptr;
    ctx.dst = dst_ptr;
    ctx.n = n;
    ctx.index_bound = dst_n;
    ctx.item_bytes = item_bytes;
    ctx.src_offset = 0;
    ctx.src_stride = item_bytes;
    ctx.dst_offset = 0;
    ctx.dst_stride = item_bytes;
    ctx.scatter = true;
    ctx.num_threads = target_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(target_threads, target_threads, &ctx,
             cpu_strided_indexed_copy_task);
    return 0;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const auto index = static_cast<std::size_t>(indices_ptr[i]);
    if (index < dst_n) {
      std::memcpy(dst_ptr + index * item_bytes, src_ptr + i * item_bytes,
                  item_bytes);
    }
  }
  return 0;
}

std::size_t Program::cpu_scatter_dense_field_packed_indices_field(
    SNode *src,
    SNode *indices,
    SNode *dst,
    int value_type,
    std::size_t src_n,
    std::size_t indices_n,
    std::size_t dst_n,
    int lane_count) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native packed dense field scatter is only available on CPU "
              "backends.");
  TI_ERROR_IF(src_n != indices_n,
              "CPU native packed dense field scatter expects field indices "
              "matching source size.");
  const std::size_t n = indices_n;
  if (n == 0) {
    return 0;
  }
  const std::size_t item_bytes =
      dense_field_packed_bytes(value_type, 1, lane_count,
                               "CPU packed dense field scatter");
  const auto *src_ptr = map_cpu_dense_field_packed(
      this, src, value_type, src_n, lane_count,
      "CPU packed dense field scatter");
  std::size_t indices_stride = 0;
  const auto *indices_ptr_bytes = map_cpu_dense_field(
      this, indices, 0, indices_n, "CPU packed dense field scatter",
      &indices_stride);
  auto *dst_ptr = map_cpu_dense_field_packed(
      this, dst, value_type, dst_n, lane_count,
      "CPU packed dense field scatter");
  TI_ERROR_IF(indices_stride != sizeof(int32_t),
              "CPU packed dense field scatter requires contiguous i32 field "
              "indices.");
  const auto *indices_ptr =
      reinterpret_cast<const int32_t *>(indices_ptr_bytes);
  TI_ERROR_IF(!src_ptr || !indices_ptr || !dst_ptr,
              "CPU native packed dense field scatter received a null data "
              "pointer.");
  validate_cpu_plain_scatter_indices(indices_ptr, n, dst_n);
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  if (n >= 65536 && target_threads > 1) {
    CpuStridedIndexedCopyTaskContext ctx;
    ctx.src = src_ptr;
    ctx.indices = indices_ptr;
    ctx.dst = dst_ptr;
    ctx.n = n;
    ctx.index_bound = dst_n;
    ctx.item_bytes = item_bytes;
    ctx.src_offset = 0;
    ctx.src_stride = item_bytes;
    ctx.dst_offset = 0;
    ctx.dst_stride = item_bytes;
    ctx.scatter = true;
    ctx.num_threads = target_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(target_threads, target_threads, &ctx,
             cpu_strided_indexed_copy_task);
    return 0;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const auto index = static_cast<std::size_t>(indices_ptr[i]);
    if (index < dst_n) {
      std::memcpy(dst_ptr + index * item_bytes, src_ptr + i * item_bytes,
                  item_bytes);
    }
  }
  return 0;
}

std::size_t Program::cpu_scatter_dense_field_indices_field(
    SNode *src,
    SNode *indices,
    SNode *dst,
    int value_type,
    std::size_t src_n,
    std::size_t indices_n,
    std::size_t dst_n) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native dense field scatter is only available on CPU "
              "backends.");
  check_indexed_copy_dense_field_indices_field_request(
      this, "CPU native", src, indices, dst, value_type, src_n, indices_n,
      dst_n, true);
  const std::size_t n = indices_n;
  if (n == 0) {
    return 0;
  }
  const std::size_t item_bytes = primitive_value_type_size(value_type);
  std::size_t src_stride = 0;
  std::size_t indices_stride = 0;
  std::size_t dst_stride = 0;
  const auto *src_ptr = map_cpu_dense_field(
      this, src, value_type, src_n, "CPU native dense field scatter",
      &src_stride);
  const auto *indices_ptr_bytes = map_cpu_dense_field(
      this, indices, 0, indices_n, "CPU native dense field scatter",
      &indices_stride);
  auto *dst_ptr = map_cpu_dense_field(
      this, dst, value_type, dst_n, "CPU native dense field scatter",
      &dst_stride);
  TI_ERROR_IF(indices_stride != sizeof(int32_t),
              "CPU native dense field scatter requires contiguous i32 field "
              "indices.");
  const auto *indices_ptr =
      reinterpret_cast<const int32_t *>(indices_ptr_bytes);
  TI_ERROR_IF(!src_ptr || !indices_ptr || !dst_ptr,
              "CPU native dense field scatter received a null data pointer.");
  validate_cpu_plain_scatter_indices(indices_ptr, n, dst_n);
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  if (n >= 65536 && target_threads > 1) {
    CpuStridedIndexedCopyTaskContext ctx;
    ctx.src = src_ptr;
    ctx.indices = indices_ptr;
    ctx.dst = dst_ptr;
    ctx.n = n;
    ctx.index_bound = dst_n;
    ctx.item_bytes = item_bytes;
    ctx.src_offset = 0;
    ctx.src_stride = src_stride;
    ctx.dst_offset = 0;
    ctx.dst_stride = dst_stride;
    ctx.scatter = true;
    ctx.num_threads = target_threads;
    auto pool = get_cpu_primitive_thread_pool(max_threads);
    pool->run(target_threads, target_threads, &ctx,
             cpu_strided_indexed_copy_task);
    return 0;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const auto index = static_cast<std::size_t>(indices_ptr[i]);
    if (index < dst_n) {
      std::memcpy(dst_ptr + index * dst_stride, src_ptr + i * src_stride,
                  item_bytes);
    }
  }
  return 0;
}

std::size_t Program::cpu_indexed_copy_workspace_bytes() const {
  return 0;
}

bool Program::cpu_scatter_add_available() const {
  return arch_is_cpu(compile_config().arch);
}

std::size_t Program::cpu_scatter_add_ndarray(Ndarray *src,
                                             Ndarray *indices,
                                             Ndarray *dst,
                                             int value_type) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native scatter-add is only available on CPU backends.");
  TI_ERROR_IF(!src || !indices || !dst,
              "CPU native scatter-add received a null ndarray.");
  TI_ERROR_IF(src->shape.size() != 1 || indices->shape.size() != 1 ||
                  dst->shape.size() != 1,
              "CPU native scatter-add currently expects 1D ndarrays.");
  TI_ERROR_IF(src->get_nelement() != indices->get_nelement(),
              "CPU native scatter-add expects source and indices sizes to "
              "match.");
  TI_ERROR_IF(src->get_element_size() != dst->get_element_size(),
              "CPU native scatter-add source and destination dtypes differ.");
  const std::size_t expected_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(expected_size == 0,
              "CPU native scatter-add received an unsupported value type.");
  TI_ERROR_IF(src->get_element_size() != expected_size ||
                  indices->get_element_size() != sizeof(int32_t),
              "CPU native scatter-add value type or i32 index size mismatch.");
  const std::size_t n = indices->get_nelement();
  const std::size_t dst_items = dst->get_nelement();
  if (n == 0 || dst_items == 0) {
    return 0;
  }
  auto *indices_ptr =
      reinterpret_cast<const int32_t *>(get_ndarray_data_ptr_as_int(indices));
  TI_ERROR_IF(!indices_ptr,
              "CPU native scatter-add received a null index pointer.");
  const int max_threads = std::max(1, compile_config().cpu_max_num_threads);
  const int target_threads = cpu_aggregation_target_threads(n, dst_items, max_threads);
  switch (value_type) {
    case 0:
      return cpu_scatter_add_typed(
          reinterpret_cast<const int32_t *>(get_ndarray_data_ptr_as_int(src)),
          indices_ptr,
          reinterpret_cast<int32_t *>(get_ndarray_data_ptr_as_int(dst)), n,
          dst_items, max_threads, target_threads);
    case 1:
      return cpu_scatter_add_typed(
          reinterpret_cast<const float *>(get_ndarray_data_ptr_as_int(src)),
          indices_ptr, reinterpret_cast<float *>(get_ndarray_data_ptr_as_int(dst)),
          n, dst_items, max_threads, target_threads);
    case 2:
      return cpu_scatter_add_typed(
          reinterpret_cast<const uint32_t *>(get_ndarray_data_ptr_as_int(src)),
          indices_ptr,
          reinterpret_cast<uint32_t *>(get_ndarray_data_ptr_as_int(dst)), n,
          dst_items, max_threads, target_threads);
    case 3:
      return cpu_scatter_add_typed(
          reinterpret_cast<const uint64_t *>(get_ndarray_data_ptr_as_int(src)),
          indices_ptr,
          reinterpret_cast<uint64_t *>(get_ndarray_data_ptr_as_int(dst)), n,
          dst_items, max_threads, target_threads);
    case 4:
      return cpu_scatter_add_typed(
          reinterpret_cast<const int64_t *>(get_ndarray_data_ptr_as_int(src)),
          indices_ptr,
          reinterpret_cast<int64_t *>(get_ndarray_data_ptr_as_int(dst)), n,
          dst_items, max_threads, target_threads);
    case 5:
      return cpu_scatter_add_typed(
          reinterpret_cast<const double *>(get_ndarray_data_ptr_as_int(src)),
          indices_ptr,
          reinterpret_cast<double *>(get_ndarray_data_ptr_as_int(dst)), n,
          dst_items, max_threads, target_threads);
    default:
      TI_ERROR("CPU native scatter-add received an unsupported value type.");
  }
  return 0;
}

std::size_t Program::cpu_scatter_add_member_ndarray(Ndarray *src,
                                                    Ndarray *indices,
                                                    Ndarray *dst,
                                                    int value_type,
                                                    std::size_t offset,
                                                    std::size_t stride) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native strided scatter-add is only available on CPU "
              "backends.");
  check_scatter_add_member_request("CPU native", src, indices, dst, value_type,
                                   offset, stride);
  const std::size_t n = indices->get_nelement();
  const std::size_t dst_items = dst->get_nelement();
  if (n == 0 || dst_items == 0) {
    return 0;
  }
  auto *src_ptr =
      reinterpret_cast<const uint8_t *>(get_ndarray_data_ptr_as_int(src));
  auto *indices_ptr =
      reinterpret_cast<const int32_t *>(get_ndarray_data_ptr_as_int(indices));
  TI_ERROR_IF(!src_ptr || !indices_ptr,
              "CPU native strided scatter-add received a null data pointer.");
  const int max_threads = std::max(1, compile_config().cpu_max_num_threads);
  const int target_threads = cpu_aggregation_target_threads(n, dst_items, max_threads);
  switch (value_type) {
    case 0:
      return cpu_scatter_add_strided_typed<int32_t>(
          src_ptr, offset, stride, indices_ptr,
          reinterpret_cast<int32_t *>(get_ndarray_data_ptr_as_int(dst)), n,
          dst_items, max_threads, target_threads);
    case 1:
      return cpu_scatter_add_strided_typed<float>(
          src_ptr, offset, stride, indices_ptr,
          reinterpret_cast<float *>(get_ndarray_data_ptr_as_int(dst)), n,
          dst_items, max_threads, target_threads);
    case 2:
      return cpu_scatter_add_strided_typed<uint32_t>(
          src_ptr, offset, stride, indices_ptr,
          reinterpret_cast<uint32_t *>(get_ndarray_data_ptr_as_int(dst)), n,
          dst_items, max_threads, target_threads);
    case 3:
      return cpu_scatter_add_strided_typed<uint64_t>(
          src_ptr, offset, stride, indices_ptr,
          reinterpret_cast<uint64_t *>(get_ndarray_data_ptr_as_int(dst)), n,
          dst_items, max_threads, target_threads);
    case 4:
      return cpu_scatter_add_strided_typed<int64_t>(
          src_ptr, offset, stride, indices_ptr,
          reinterpret_cast<int64_t *>(get_ndarray_data_ptr_as_int(dst)), n,
          dst_items, max_threads, target_threads);
    case 5:
      return cpu_scatter_add_strided_typed<double>(
          src_ptr, offset, stride, indices_ptr,
          reinterpret_cast<double *>(get_ndarray_data_ptr_as_int(dst)), n,
          dst_items, max_threads, target_threads);
    default:
      TI_ERROR(
          "CPU native strided scatter-add received an unsupported value type.");
  }
  return 0;
}

std::size_t Program::cpu_scatter_add_strided_ndarray(
    Ndarray *src,
    Ndarray *indices,
    Ndarray *dst,
    int value_type,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native strided scatter-add is only available on CPU "
              "backends.");
  check_scatter_add_strided_request("CPU native", src, indices, dst,
                                    value_type, src_offset, src_stride,
                                    dst_offset, dst_stride);
  const std::size_t n = indices->get_nelement();
  const std::size_t dst_items = dst->get_nelement();
  if (n == 0 || dst_items == 0) {
    return 0;
  }
  auto *src_ptr =
      reinterpret_cast<const uint8_t *>(get_ndarray_data_ptr_as_int(src));
  auto *indices_ptr =
      reinterpret_cast<const int32_t *>(get_ndarray_data_ptr_as_int(indices));
  auto *dst_ptr = reinterpret_cast<uint8_t *>(get_ndarray_data_ptr_as_int(dst));
  TI_ERROR_IF(!src_ptr || !indices_ptr || !dst_ptr,
              "CPU native strided scatter-add received a null data pointer.");
  const int max_threads = std::max(1, compile_config().cpu_max_num_threads);
  const int target_threads = cpu_aggregation_target_threads(n, dst_items, max_threads);
  switch (value_type) {
    case 0:
      return cpu_scatter_add_strided_io_typed<int32_t>(
          src_ptr, src_offset, src_stride, indices_ptr, dst_ptr, dst_offset,
          dst_stride, n, dst_items, max_threads, target_threads);
    case 1:
      return cpu_scatter_add_strided_io_typed<float>(
          src_ptr, src_offset, src_stride, indices_ptr, dst_ptr, dst_offset,
          dst_stride, n, dst_items, max_threads, target_threads);
    case 2:
      return cpu_scatter_add_strided_io_typed<uint32_t>(
          src_ptr, src_offset, src_stride, indices_ptr, dst_ptr, dst_offset,
          dst_stride, n, dst_items, max_threads, target_threads);
    case 3:
      return cpu_scatter_add_strided_io_typed<uint64_t>(
          src_ptr, src_offset, src_stride, indices_ptr, dst_ptr, dst_offset,
          dst_stride, n, dst_items, max_threads, target_threads);
    case 4:
      return cpu_scatter_add_strided_io_typed<int64_t>(
          src_ptr, src_offset, src_stride, indices_ptr, dst_ptr, dst_offset,
          dst_stride, n, dst_items, max_threads, target_threads);
    case 5:
      return cpu_scatter_add_strided_io_typed<double>(
          src_ptr, src_offset, src_stride, indices_ptr, dst_ptr, dst_offset,
          dst_stride, n, dst_items, max_threads, target_threads);
    default:
      TI_ERROR(
          "CPU native strided scatter-add received an unsupported value type.");
  }
  return 0;
}

std::size_t Program::cpu_scatter_add_dense_field(SNode *src,
                                                 Ndarray *indices,
                                                 SNode *dst,
                                                 int value_type,
                                                 std::size_t src_n,
                                                 std::size_t dst_n) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native dense field scatter-add is only available on CPU "
              "backends.");
  check_indexed_copy_dense_field_request(this, "CPU native", src, indices, dst,
                                         value_type, src_n, dst_n, true);
  const std::size_t n = indices->get_nelement();
  if (n == 0 || dst_n == 0) {
    return 0;
  }
  std::size_t src_stride = 0;
  std::size_t dst_stride = 0;
  const auto *src_ptr = map_cpu_dense_field(
      this, src, value_type, src_n, "CPU native dense field scatter-add",
      &src_stride);
  auto *dst_ptr = map_cpu_dense_field(
      this, dst, value_type, dst_n, "CPU native dense field scatter-add",
      &dst_stride);
  auto *indices_ptr =
      reinterpret_cast<const int32_t *>(get_ndarray_data_ptr_as_int(indices));
  TI_ERROR_IF(!src_ptr || !indices_ptr || !dst_ptr,
              "CPU native dense field scatter-add received a null data "
              "pointer.");
  const int max_threads = std::max(1, compile_config().cpu_max_num_threads);
  const int target_threads = cpu_aggregation_target_threads(n, dst_n, max_threads);
  switch (value_type) {
    case 0:
      return cpu_scatter_add_strided_io_typed<int32_t>(
          src_ptr, 0, src_stride, indices_ptr, dst_ptr, 0, dst_stride, n,
          dst_n, max_threads, target_threads);
    case 1:
      return cpu_scatter_add_strided_io_typed<float>(
          src_ptr, 0, src_stride, indices_ptr, dst_ptr, 0, dst_stride, n,
          dst_n, max_threads, target_threads);
    case 2:
      return cpu_scatter_add_strided_io_typed<uint32_t>(
          src_ptr, 0, src_stride, indices_ptr, dst_ptr, 0, dst_stride, n,
          dst_n, max_threads, target_threads);
    case 3:
      return cpu_scatter_add_strided_io_typed<uint64_t>(
          src_ptr, 0, src_stride, indices_ptr, dst_ptr, 0, dst_stride, n,
          dst_n, max_threads, target_threads);
    case 4:
      return cpu_scatter_add_strided_io_typed<int64_t>(
          src_ptr, 0, src_stride, indices_ptr, dst_ptr, 0, dst_stride, n,
          dst_n, max_threads, target_threads);
    case 5:
      return cpu_scatter_add_strided_io_typed<double>(
          src_ptr, 0, src_stride, indices_ptr, dst_ptr, 0, dst_stride, n,
          dst_n, max_threads, target_threads);
    default:
      TI_ERROR(
          "CPU native dense field scatter-add received an unsupported value "
          "type.");
  }
  return 0;
}

std::size_t Program::cpu_scatter_add_dense_field_indices_field(
    SNode *src,
    SNode *indices,
    SNode *dst,
    int value_type,
    std::size_t src_n,
    std::size_t indices_n,
    std::size_t dst_n) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native dense field scatter-add is only available on CPU "
              "backends.");
  check_indexed_copy_dense_field_indices_field_request(
      this, "CPU native", src, indices, dst, value_type, src_n, indices_n,
      dst_n, true);
  const std::size_t n = indices_n;
  if (n == 0 || dst_n == 0) {
    return 0;
  }
  std::size_t src_stride = 0;
  std::size_t indices_stride = 0;
  std::size_t dst_stride = 0;
  const auto *src_ptr = map_cpu_dense_field(
      this, src, value_type, src_n, "CPU native dense field scatter-add",
      &src_stride);
  const auto *indices_ptr_bytes = map_cpu_dense_field(
      this, indices, 0, indices_n, "CPU native dense field scatter-add",
      &indices_stride);
  auto *dst_ptr = map_cpu_dense_field(
      this, dst, value_type, dst_n, "CPU native dense field scatter-add",
      &dst_stride);
  TI_ERROR_IF(indices_stride != sizeof(int32_t),
              "CPU native dense field scatter-add requires contiguous i32 "
              "field indices.");
  const auto *indices_ptr =
      reinterpret_cast<const int32_t *>(indices_ptr_bytes);
  TI_ERROR_IF(!src_ptr || !indices_ptr || !dst_ptr,
              "CPU native dense field scatter-add received a null data "
              "pointer.");
  const int max_threads = std::max(1, compile_config().cpu_max_num_threads);
  const int target_threads = cpu_aggregation_target_threads(n, dst_n, max_threads);
  switch (value_type) {
    case 0:
      return cpu_scatter_add_strided_io_typed<int32_t>(
          src_ptr, 0, src_stride, indices_ptr, dst_ptr, 0, dst_stride, n,
          dst_n, max_threads, target_threads);
    case 1:
      return cpu_scatter_add_strided_io_typed<float>(
          src_ptr, 0, src_stride, indices_ptr, dst_ptr, 0, dst_stride, n,
          dst_n, max_threads, target_threads);
    case 2:
      return cpu_scatter_add_strided_io_typed<uint32_t>(
          src_ptr, 0, src_stride, indices_ptr, dst_ptr, 0, dst_stride, n,
          dst_n, max_threads, target_threads);
    case 3:
      return cpu_scatter_add_strided_io_typed<uint64_t>(
          src_ptr, 0, src_stride, indices_ptr, dst_ptr, 0, dst_stride, n,
          dst_n, max_threads, target_threads);
    case 4:
      return cpu_scatter_add_strided_io_typed<int64_t>(
          src_ptr, 0, src_stride, indices_ptr, dst_ptr, 0, dst_stride, n,
          dst_n, max_threads, target_threads);
    case 5:
      return cpu_scatter_add_strided_io_typed<double>(
          src_ptr, 0, src_stride, indices_ptr, dst_ptr, 0, dst_stride, n,
          dst_n, max_threads, target_threads);
    default:
      TI_ERROR(
          "CPU native dense field scatter-add received an unsupported value "
          "type.");
  }
  return 0;
}

std::size_t Program::cpu_scatter_add_workspace_bytes() const {
  return cpu_primitive_workspace_bytes(
      this, PrimitiveWorkspaceFamily::scatter_add);
}

void Program::cpu_scatter_add_clear_workspace() {
  primitive_workspace_arena_.clear(PrimitiveWorkspaceBackend::cpu,
                                   PrimitiveWorkspaceFamily::scatter_add);
}

bool Program::cpu_bucket_builder_available() const {
  return arch_is_cpu(compile_config().arch);
}

std::size_t Program::cpu_bucket_builder_i32_ndarray(Ndarray *keys,
                                                    Ndarray *values,
                                                    Ndarray *offsets,
                                                    Ndarray *output) {
  return cpu_bucket_builder_ndarray(keys, values, offsets, output, 0);
}

std::size_t Program::cpu_bucket_builder_ndarray(Ndarray *keys,
                                                Ndarray *values,
                                                Ndarray *offsets,
                                                Ndarray *output,
                                                int value_type) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native bucket builder is only available on CPU backends.");
  TI_ERROR_IF(!keys || !values || !offsets || !output,
              "CPU native bucket builder received a null ndarray.");
  TI_ERROR_IF(keys->shape.size() != 1 || values->shape.size() != 1 ||
                  offsets->shape.size() != 1 || output->shape.size() != 1,
              "CPU native bucket builder expects 1D ndarrays.");
  TI_ERROR_IF(keys->get_nelement() != values->get_nelement(),
              "CPU native bucket builder keys and values sizes differ.");
  TI_ERROR_IF(offsets->get_nelement() < 2,
              "CPU native bucket builder offsets must contain num_bins + 1 items.");
  const std::size_t n = keys->get_nelement();
  const std::size_t num_bins = offsets->get_nelement() - 1;
  TI_ERROR_IF(output->get_nelement() < n,
              "CPU native bucket builder output is smaller than input values.");
  const std::size_t expected_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(expected_size == 0,
              "CPU native bucket builder received an unsupported value type.");
  const std::size_t item_bytes = values->get_element_size();
  TI_ERROR_IF(keys->get_element_size() != sizeof(int32_t) ||
                  offsets->get_element_size() != sizeof(int32_t) ||
                  item_bytes == 0 ||
                  item_bytes % sizeof(uint32_t) != 0 ||
                  output->get_element_size() != item_bytes,
              "CPU native bucket builder dtype does not match value type, "
              "keys/offsets are not i32, or payload is not 4-byte aligned.");
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()),
              "CPU native bucket builder input count exceeds i32 range.");

  auto *keys_ptr =
      reinterpret_cast<int32_t *>(get_ndarray_data_ptr_as_int(keys));
  auto *offsets_ptr =
      reinterpret_cast<int32_t *>(get_ndarray_data_ptr_as_int(offsets));
  TI_ERROR_IF(!keys_ptr || !offsets_ptr,
              "CPU native bucket builder received a null data pointer.");

  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  if (item_bytes != expected_size) {
    return cpu_bucket_builder_raw(
        keys_ptr,
        reinterpret_cast<const uint8_t *>(get_ndarray_data_ptr_as_int(values)),
        offsets_ptr,
        reinterpret_cast<uint8_t *>(get_ndarray_data_ptr_as_int(output)), n,
        num_bins, item_bytes, max_threads);
  }
  switch (value_type) {
    case 0:
      return cpu_bucket_builder_typed(
          keys_ptr,
          reinterpret_cast<const int32_t *>(get_ndarray_data_ptr_as_int(values)),
          offsets_ptr,
          reinterpret_cast<int32_t *>(get_ndarray_data_ptr_as_int(output)), n,
          num_bins, max_threads);
    case 1:
      return cpu_bucket_builder_typed(
          keys_ptr,
          reinterpret_cast<const float *>(get_ndarray_data_ptr_as_int(values)),
          offsets_ptr,
          reinterpret_cast<float *>(get_ndarray_data_ptr_as_int(output)), n,
          num_bins, max_threads);
    case 2:
      return cpu_bucket_builder_typed(
          keys_ptr,
          reinterpret_cast<const uint32_t *>(get_ndarray_data_ptr_as_int(values)),
          offsets_ptr,
          reinterpret_cast<uint32_t *>(get_ndarray_data_ptr_as_int(output)), n,
          num_bins, max_threads);
    case 3:
      return cpu_bucket_builder_typed(
          keys_ptr,
          reinterpret_cast<const uint64_t *>(get_ndarray_data_ptr_as_int(values)),
          offsets_ptr,
          reinterpret_cast<uint64_t *>(get_ndarray_data_ptr_as_int(output)), n,
          num_bins, max_threads);
    case 4:
      return cpu_bucket_builder_typed(
          keys_ptr,
          reinterpret_cast<const int64_t *>(get_ndarray_data_ptr_as_int(values)),
          offsets_ptr,
          reinterpret_cast<int64_t *>(get_ndarray_data_ptr_as_int(output)), n,
          num_bins, max_threads);
    case 5:
      return cpu_bucket_builder_typed(
          keys_ptr,
          reinterpret_cast<const double *>(get_ndarray_data_ptr_as_int(values)),
          offsets_ptr,
          reinterpret_cast<double *>(get_ndarray_data_ptr_as_int(output)), n,
          num_bins, max_threads);
    default:
      TI_ERROR("CPU native bucket builder received an unsupported value type.");
  }
  return 0;
}

std::size_t Program::cpu_bucket_builder_dense_field(SNode *keys,
                                                    SNode *values,
                                                    SNode *offsets,
                                                    SNode *output,
                                                    int value_type,
                                                    std::size_t n,
                                                    std::size_t num_bins) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native dense field bucket builder is only available on CPU "
              "backends.");
  TI_ERROR_IF(!keys || !values || !offsets || !output,
              "CPU native dense field bucket builder received a null field.");
  TI_ERROR_IF(num_bins == 0,
              "CPU native dense field bucket builder expects at least one "
              "bucket.");
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()),
              "CPU native dense field bucket builder input count exceeds i32 "
              "range.");
  const std::size_t item_bytes = primitive_value_type_size(value_type);
  TI_ERROR_IF(item_bytes == 0,
              "CPU native dense field bucket builder received an unsupported "
              "value type.");
  std::size_t keys_stride = 0;
  std::size_t values_stride = 0;
  std::size_t offsets_stride = 0;
  std::size_t output_stride = 0;
  const auto *keys_ptr = map_cpu_dense_field(
      this, keys, 0, n, "CPU native dense field bucket builder keys",
      &keys_stride);
  const auto *values_ptr = map_cpu_dense_field(
      this, values, value_type, n,
      "CPU native dense field bucket builder values", &values_stride);
  auto *offsets_ptr = map_cpu_dense_field(
      this, offsets, 0, num_bins + 1,
      "CPU native dense field bucket builder offsets", &offsets_stride);
  auto *output_ptr = map_cpu_dense_field(
      this, output, value_type, n,
      "CPU native dense field bucket builder output", &output_stride);
  TI_ERROR_IF(keys_stride != sizeof(int32_t) || values_stride != item_bytes ||
                  offsets_stride != sizeof(int32_t) ||
                  output_stride != item_bytes,
              "CPU native dense field bucket builder requires contiguous "
              "keys, values, offsets, and output fields.");
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  switch (value_type) {
    case 0:
      return cpu_bucket_builder_typed(
          reinterpret_cast<const int32_t *>(keys_ptr),
          reinterpret_cast<const int32_t *>(values_ptr),
          reinterpret_cast<int32_t *>(offsets_ptr),
          reinterpret_cast<int32_t *>(output_ptr), n, num_bins, max_threads);
    case 1:
      return cpu_bucket_builder_typed(
          reinterpret_cast<const int32_t *>(keys_ptr),
          reinterpret_cast<const float *>(values_ptr),
          reinterpret_cast<int32_t *>(offsets_ptr),
          reinterpret_cast<float *>(output_ptr), n, num_bins, max_threads);
    case 2:
      return cpu_bucket_builder_typed(
          reinterpret_cast<const int32_t *>(keys_ptr),
          reinterpret_cast<const uint32_t *>(values_ptr),
          reinterpret_cast<int32_t *>(offsets_ptr),
          reinterpret_cast<uint32_t *>(output_ptr), n, num_bins, max_threads);
    case 3:
      return cpu_bucket_builder_typed(
          reinterpret_cast<const int32_t *>(keys_ptr),
          reinterpret_cast<const uint64_t *>(values_ptr),
          reinterpret_cast<int32_t *>(offsets_ptr),
          reinterpret_cast<uint64_t *>(output_ptr), n, num_bins, max_threads);
    case 4:
      return cpu_bucket_builder_typed(
          reinterpret_cast<const int32_t *>(keys_ptr),
          reinterpret_cast<const int64_t *>(values_ptr),
          reinterpret_cast<int32_t *>(offsets_ptr),
          reinterpret_cast<int64_t *>(output_ptr), n, num_bins, max_threads);
    case 5:
      return cpu_bucket_builder_typed(
          reinterpret_cast<const int32_t *>(keys_ptr),
          reinterpret_cast<const double *>(values_ptr),
          reinterpret_cast<int32_t *>(offsets_ptr),
          reinterpret_cast<double *>(output_ptr), n, num_bins, max_threads);
    default:
      TI_ERROR("CPU native dense field bucket builder received an unsupported "
               "value type.");
  }
  return 0;
}

std::size_t Program::cpu_bucket_builder_workspace_bytes() const {
  return 0;
}

bool Program::cpu_grouped_reduce_available() const {
  return arch_is_cpu(compile_config().arch);
}

std::size_t Program::cpu_grouped_reduce_i32_ndarray(Ndarray *keys,
                                                    Ndarray *values,
                                                    Ndarray *output,
                                                    int op) {
  return cpu_grouped_reduce_ndarray(keys, values, output, 0, op);
}

std::size_t Program::cpu_grouped_reduce_ndarray(Ndarray *keys,
                                                Ndarray *values,
                                                Ndarray *output,
                                                int value_type,
                                                int op) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native grouped reduce is only available on CPU backends.");
  TI_ERROR_IF(!keys || !values || !output,
              "CPU native grouped reduce received a null ndarray.");
  TI_ERROR_IF(keys->shape.size() != 1 || values->shape.size() != 1 ||
                  output->shape.size() != 1,
              "CPU native grouped reduce expects 1D ndarrays.");
  TI_ERROR_IF(keys->get_nelement() != values->get_nelement(),
              "CPU native grouped reduce keys and values sizes differ.");
  TI_ERROR_IF(output->get_nelement() == 0,
              "CPU native grouped reduce output must contain at least one group.");
  const std::size_t expected_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(expected_size == 0,
              "CPU native grouped reduce received an unsupported value type.");
  TI_ERROR_IF(keys->get_element_size() != sizeof(int32_t) ||
                  values->get_element_size() != expected_size ||
                  output->get_element_size() != expected_size,
              "CPU native grouped reduce value type or i32 key size mismatch.");
  TI_ERROR_IF(op != 0, "CPU native grouped reduce currently supports only sum.");
  const std::size_t n = keys->get_nelement();
  const std::size_t num_groups = output->get_nelement();
  auto *keys_ptr =
      reinterpret_cast<const int32_t *>(get_ndarray_data_ptr_as_int(keys));
  const int max_threads = std::max(1, compile_config().cpu_max_num_threads);
  const int target_threads = cpu_aggregation_target_threads(n, num_groups, max_threads);
  switch (value_type) {
    case 0:
      return cpu_grouped_reduce_typed(
          keys_ptr,
          reinterpret_cast<const int32_t *>(get_ndarray_data_ptr_as_int(values)),
          reinterpret_cast<int32_t *>(get_ndarray_data_ptr_as_int(output)), n,
          num_groups, max_threads, target_threads);
    case 1:
      return cpu_grouped_reduce_typed(
          keys_ptr,
          reinterpret_cast<const float *>(get_ndarray_data_ptr_as_int(values)),
          reinterpret_cast<float *>(get_ndarray_data_ptr_as_int(output)), n,
          num_groups, max_threads, target_threads);
    case 2:
      return cpu_grouped_reduce_typed(
          keys_ptr,
          reinterpret_cast<const uint32_t *>(get_ndarray_data_ptr_as_int(values)),
          reinterpret_cast<uint32_t *>(get_ndarray_data_ptr_as_int(output)), n,
          num_groups, max_threads, target_threads);
    case 3:
      return cpu_grouped_reduce_typed(
          keys_ptr,
          reinterpret_cast<const uint64_t *>(get_ndarray_data_ptr_as_int(values)),
          reinterpret_cast<uint64_t *>(get_ndarray_data_ptr_as_int(output)), n,
          num_groups, max_threads, target_threads);
    case 4:
      return cpu_grouped_reduce_typed(
          keys_ptr,
          reinterpret_cast<const int64_t *>(get_ndarray_data_ptr_as_int(values)),
          reinterpret_cast<int64_t *>(get_ndarray_data_ptr_as_int(output)), n,
          num_groups, max_threads, target_threads);
    case 5:
      return cpu_grouped_reduce_typed(
          keys_ptr,
          reinterpret_cast<const double *>(get_ndarray_data_ptr_as_int(values)),
          reinterpret_cast<double *>(get_ndarray_data_ptr_as_int(output)), n,
          num_groups, max_threads, target_threads);
    default:
      TI_ERROR("CPU native grouped reduce received an unsupported value type.");
  }
  return 0;
}

std::size_t Program::cpu_grouped_reduce_dense_field(SNode *keys,
                                                    SNode *values,
                                                    SNode *output,
                                                    int value_type,
                                                    std::size_t n,
                                                    std::size_t num_groups,
                                                    int op) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native dense field grouped reduce is only available on CPU "
              "backends.");
  TI_ERROR_IF(!keys || !values || !output,
              "CPU native dense field grouped reduce received a null field.");
  TI_ERROR_IF(num_groups == 0,
              "CPU native dense field grouped reduce output must contain at "
              "least one group.");
  TI_ERROR_IF(op != 0,
              "CPU native dense field grouped reduce currently supports only "
              "sum.");
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CPU native dense field grouped reduce received an unsupported "
              "value type.");
  std::size_t keys_stride = 0;
  std::size_t values_stride = 0;
  std::size_t output_stride = 0;
  const auto *keys_ptr = map_cpu_dense_field(
      this, keys, 0, n, "CPU native dense field grouped reduce keys",
      &keys_stride);
  const auto *values_ptr = map_cpu_dense_field(
      this, values, value_type, n,
      "CPU native dense field grouped reduce values", &values_stride);
  auto *output_ptr = map_cpu_dense_field(
      this, output, value_type, num_groups,
      "CPU native dense field grouped reduce output", &output_stride);
  TI_ERROR_IF(keys_stride != sizeof(int32_t) || values_stride != value_size ||
                  output_stride != value_size,
              "CPU native dense field grouped reduce requires contiguous keys, "
              "values, and output fields.");
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int target_threads = cpu_aggregation_target_threads(n, num_groups, max_threads);
  switch (value_type) {
    case 0:
      return cpu_grouped_reduce_typed(
          reinterpret_cast<const int32_t *>(keys_ptr),
          reinterpret_cast<const int32_t *>(values_ptr),
          reinterpret_cast<int32_t *>(output_ptr), n, num_groups, max_threads,
          target_threads);
    case 1:
      return cpu_grouped_reduce_typed(
          reinterpret_cast<const int32_t *>(keys_ptr),
          reinterpret_cast<const float *>(values_ptr),
          reinterpret_cast<float *>(output_ptr), n, num_groups, max_threads,
          target_threads);
    case 2:
      return cpu_grouped_reduce_typed(
          reinterpret_cast<const int32_t *>(keys_ptr),
          reinterpret_cast<const uint32_t *>(values_ptr),
          reinterpret_cast<uint32_t *>(output_ptr), n, num_groups, max_threads,
          target_threads);
    case 3:
      return cpu_grouped_reduce_typed(
          reinterpret_cast<const int32_t *>(keys_ptr),
          reinterpret_cast<const uint64_t *>(values_ptr),
          reinterpret_cast<uint64_t *>(output_ptr), n, num_groups, max_threads,
          target_threads);
    case 4:
      return cpu_grouped_reduce_typed(
          reinterpret_cast<const int32_t *>(keys_ptr),
          reinterpret_cast<const int64_t *>(values_ptr),
          reinterpret_cast<int64_t *>(output_ptr), n, num_groups, max_threads,
          target_threads);
    case 5:
      return cpu_grouped_reduce_typed(
          reinterpret_cast<const int32_t *>(keys_ptr),
          reinterpret_cast<const double *>(values_ptr),
          reinterpret_cast<double *>(output_ptr), n, num_groups, max_threads,
          target_threads);
    default:
      TI_ERROR("CPU native dense field grouped reduce received an unsupported "
               "value type.");
  }
  return 0;
}

std::size_t Program::cpu_grouped_reduce_member_ndarray(Ndarray *keys,
                                                       Ndarray *values,
                                                       Ndarray *output,
                                                       int value_type,
                                                       std::size_t offset,
                                                       std::size_t stride,
                                                       int op) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native strided grouped reduce is only available on CPU "
              "backends.");
  check_grouped_reduce_member_request("CPU native", keys, values, output,
                                      value_type, offset, stride, op);
  const std::size_t n = keys->get_nelement();
  const std::size_t num_groups = output->get_nelement();
  auto *keys_ptr =
      reinterpret_cast<const int32_t *>(get_ndarray_data_ptr_as_int(keys));
  auto *values_ptr =
      reinterpret_cast<const uint8_t *>(get_ndarray_data_ptr_as_int(values));
  TI_ERROR_IF(!keys_ptr || !values_ptr,
              "CPU native strided grouped reduce received a null data pointer.");
  const int max_threads = std::max(1, compile_config().cpu_max_num_threads);
  const int target_threads = cpu_aggregation_target_threads(n, num_groups, max_threads);
  switch (value_type) {
    case 0:
      return cpu_grouped_reduce_strided_typed<int32_t>(
          keys_ptr, values_ptr, offset, stride,
          reinterpret_cast<int32_t *>(get_ndarray_data_ptr_as_int(output)), n,
          num_groups, max_threads, target_threads);
    case 1:
      return cpu_grouped_reduce_strided_typed<float>(
          keys_ptr, values_ptr, offset, stride,
          reinterpret_cast<float *>(get_ndarray_data_ptr_as_int(output)), n,
          num_groups, max_threads, target_threads);
    case 2:
      return cpu_grouped_reduce_strided_typed<uint32_t>(
          keys_ptr, values_ptr, offset, stride,
          reinterpret_cast<uint32_t *>(get_ndarray_data_ptr_as_int(output)), n,
          num_groups, max_threads, target_threads);
    case 3:
      return cpu_grouped_reduce_strided_typed<uint64_t>(
          keys_ptr, values_ptr, offset, stride,
          reinterpret_cast<uint64_t *>(get_ndarray_data_ptr_as_int(output)), n,
          num_groups, max_threads, target_threads);
    case 4:
      return cpu_grouped_reduce_strided_typed<int64_t>(
          keys_ptr, values_ptr, offset, stride,
          reinterpret_cast<int64_t *>(get_ndarray_data_ptr_as_int(output)), n,
          num_groups, max_threads, target_threads);
    case 5:
      return cpu_grouped_reduce_strided_typed<double>(
          keys_ptr, values_ptr, offset, stride,
          reinterpret_cast<double *>(get_ndarray_data_ptr_as_int(output)), n,
          num_groups, max_threads, target_threads);
    default:
      TI_ERROR("CPU native strided grouped reduce received an unsupported "
               "value type.");
  }
  return 0;
}

std::size_t Program::cpu_grouped_reduce_strided_ndarray(
    Ndarray *keys,
    Ndarray *values,
    Ndarray *output,
    int value_type,
    std::size_t values_offset,
    std::size_t values_stride,
    std::size_t output_offset,
    std::size_t output_stride,
    int op) {
  return cpu_grouped_reduce_strided_keys_ndarray(
      keys, values, output, value_type, 0, sizeof(int32_t), values_offset,
      values_stride, output_offset, output_stride, op);
}

std::size_t Program::cpu_grouped_reduce_strided_keys_ndarray(
    Ndarray *keys,
    Ndarray *values,
    Ndarray *output,
    int value_type,
    std::size_t keys_offset,
    std::size_t keys_stride,
    std::size_t values_offset,
    std::size_t values_stride,
    std::size_t output_offset,
    std::size_t output_stride,
    int op) {
  ScopedCpuPrimitiveProgram cpu_primitive_program_scope(this);
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native strided grouped reduce is only available on CPU "
              "backends.");
  check_grouped_reduce_strided_keys_request(
      "CPU native", keys, values, output, value_type, keys_offset, keys_stride,
      values_offset, values_stride, output_offset, output_stride, op);
  const std::size_t n = keys->get_nelement();
  const std::size_t num_groups = output->get_nelement();
  auto *keys_ptr =
      reinterpret_cast<const uint8_t *>(get_ndarray_data_ptr_as_int(keys));
  auto *values_ptr =
      reinterpret_cast<const uint8_t *>(get_ndarray_data_ptr_as_int(values));
  auto *output_ptr =
      reinterpret_cast<uint8_t *>(get_ndarray_data_ptr_as_int(output));
  TI_ERROR_IF(!keys_ptr || !values_ptr || !output_ptr,
              "CPU native strided grouped reduce received a null data pointer.");
  const int max_threads = std::max(1, compile_config().cpu_max_num_threads);
  const int target_threads = cpu_aggregation_target_threads(n, num_groups, max_threads);
  switch (value_type) {
    case 0:
      return cpu_grouped_reduce_strided_io_typed<int32_t>(
          keys_ptr, keys_offset, keys_stride, values_ptr, values_offset,
          values_stride, output_ptr, output_offset, output_stride, n,
          num_groups, max_threads, target_threads);
    case 1:
      return cpu_grouped_reduce_strided_io_typed<float>(
          keys_ptr, keys_offset, keys_stride, values_ptr, values_offset,
          values_stride, output_ptr, output_offset, output_stride, n,
          num_groups, max_threads, target_threads);
    case 2:
      return cpu_grouped_reduce_strided_io_typed<uint32_t>(
          keys_ptr, keys_offset, keys_stride, values_ptr, values_offset,
          values_stride, output_ptr, output_offset, output_stride, n,
          num_groups, max_threads, target_threads);
    case 3:
      return cpu_grouped_reduce_strided_io_typed<uint64_t>(
          keys_ptr, keys_offset, keys_stride, values_ptr, values_offset,
          values_stride, output_ptr, output_offset, output_stride, n,
          num_groups, max_threads, target_threads);
    case 4:
      return cpu_grouped_reduce_strided_io_typed<int64_t>(
          keys_ptr, keys_offset, keys_stride, values_ptr, values_offset,
          values_stride, output_ptr, output_offset, output_stride, n,
          num_groups, max_threads, target_threads);
    case 5:
      return cpu_grouped_reduce_strided_io_typed<double>(
          keys_ptr, keys_offset, keys_stride, values_ptr, values_offset,
          values_stride, output_ptr, output_offset, output_stride, n,
          num_groups, max_threads, target_threads);
    default:
      TI_ERROR(
          "CPU native strided grouped reduce received an unsupported value "
          "type.");
  }
  return 0;
}

std::size_t Program::cpu_grouped_reduce_workspace_bytes() const {
  return cpu_primitive_workspace_bytes(this,
                                       PrimitiveWorkspaceFamily::grouped);
}

void Program::cpu_grouped_reduce_clear_workspace() {
  primitive_workspace_arena_.clear(PrimitiveWorkspaceBackend::cpu,
                                   PrimitiveWorkspaceFamily::grouped);
}

std::pair<const ArgPackType *, size_t>
Program::get_argpack_type_with_data_layout(const ArgPackType *old_ty,
                                           const std::string &layout) {
  // Convert to StructType
  auto *struct_type_old =
      TypeFactory::get_instance()
          .get_struct_type(old_ty->elements(), old_ty->get_layout())
          ->as<StructType>();
  // Call get_struct_type_with_data_layout
  auto [struct_type, size] = program_impl_->get_struct_type_with_data_layout(
      const_cast<StructType *>(struct_type_old), layout);
  // Convert back to ArgPackType
  auto *new_ty =
      TypeFactory::get_instance()
          .get_argpack_type(struct_type->elements(), struct_type->get_layout())
          ->as<ArgPackType>();
  return {new_ty, size};
}

std::pair<const StructType *, size_t> Program::get_struct_type_with_data_layout(
    const StructType *old_ty,
    const std::string &layout) {
  return program_impl_->get_struct_type_with_data_layout(old_ty, layout);
}

Program::~Program() {
  if (lifetime_token_) {
    std::lock_guard<std::mutex> lock(lifetime_token_->mutex_);
    lifetime_token_->program_ = nullptr;
  }
  lifetime_token_.reset();
  finalize();
}

DeviceCapabilityConfig translate_devcaps(const std::vector<std::string> &caps) {
  // Each device capability assignment is named like this:
  // - `spirv_version=1.3`
  // - `spirv_has_int8`
  DeviceCapabilityConfig cfg{};
  for (const std::string &cap : caps) {
    std::string_view key;
    uint32_t value;
    size_t ieq = cap.find('=');
    if (ieq == std::string::npos) {
      key = cap;
      value = 1;
    } else {
      key = std::string_view(cap.c_str(), ieq);
      value = (uint32_t)std::atol(cap.c_str() + ieq + 1);
    }
    DeviceCapability devcap = str2devcap(key);
    cfg.set(devcap, value);
  }

  // Assign default caps (that always present).
  if (!cfg.contains(DeviceCapability::spirv_version)) {
    cfg.set(DeviceCapability::spirv_version, 0x10300);
  }
  return cfg;
}

std::unique_ptr<AotModuleBuilder> Program::make_aot_module_builder(
    Arch arch,
    const std::vector<std::string> &caps) {
  DeviceCapabilityConfig cfg = translate_devcaps(caps);
  // FIXME: This couples the runtime backend with the target AOT backend. E.g.
  // If we want to build a Metal AOT module, we have to be on the macOS
  // platform. Consider decoupling this part
  if (arch_uses_llvm(compile_config().arch) ||
      compile_config().arch == Arch::metal ||
      compile_config().arch == Arch::vulkan ||
      compile_config().arch == Arch::opengl ||
      compile_config().arch == Arch::gles ||
      compile_config().arch == Arch::dx12) {
    return program_impl_->make_aot_module_builder(cfg);
  }
  return nullptr;
}

int Program::allocate_snode_tree_id() {
  if (free_snode_tree_ids_.empty()) {
    return snode_trees_.size();
  } else {
    int id = free_snode_tree_ids_.top();
    free_snode_tree_ids_.pop();
    return id;
  }
}

void Program::enqueue_compute_op_lambda(
    std::function<void(Device *device, CommandList *cmdlist)> op,
    const std::vector<ComputeOpImageRef> &image_refs) {
#ifdef TI_WITH_VULKAN
  if (try_record_vulkan_native_command(this, op)) {
    return;
  }
#endif
  program_impl_->enqueue_compute_op_lambda(op, image_refs);
}

void Program::enqueue_graphics_op_lambda(
    std::function<void(GraphicsDevice *device, CommandList *cmdlist)> op,
    const std::vector<ComputeOpImageRef> &image_refs,
    const std::vector<std::uint64_t> &replay_key) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Runtime-ordered graphics commands currently require Vulkan.");
#ifdef TI_WITH_VULKAN
  TI_ERROR_IF(vulkan_native_command_recording_context != nullptr,
              "Graphics commands cannot be nested into a compute native "
              "command recording.");
#endif
  program_impl_->enqueue_graphics_op_lambda(std::move(op), image_refs,
                                            replay_key);
}

}  // namespace taichi::lang
