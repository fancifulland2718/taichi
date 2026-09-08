#include "taichi/runtime/cuda/graph_binding_frame.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <set>

#include "taichi/program/kernel.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"
#include "taichi/program/runtime_completion.h"
#include "taichi/program/storage_view.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/runtime/cuda/kernel_launcher.h"

namespace taichi::lang::cuda {

struct GraphBindingFrame::State {
  const void *owner{nullptr};
  CudaDevice *device{nullptr};
  bool valid{false};
  CUgraph graph{nullptr};
  std::vector<KernelLauncher::GraphLaunchPacket> packets;
  // Source bytes outlive asynchronous preparation uploads.
  std::vector<std::unique_ptr<LaunchContextBuilder>> contexts;
  void *argument_image{nullptr};
  std::vector<std::uint8_t> host_argument_image;
  std::vector<std::unique_ptr<CudaDevice::AllocationLease>> allocations;
  std::vector<std::shared_ptr<void>> provider_capture_resources;
  std::size_t bytes{0};
  std::size_t nodes{0};

  void release(bool backend_safe) noexcept {
    valid = false;
    auto &driver = CUDADriver::get_instance();
    if (backend_safe) {
      try {
        auto context = CUDAContext::get_instance().get_guard();
        if (graph) {
          driver.graph_destroy(graph);
        }
        if (argument_image) {
          // Packet pointers are slices, not independently owned allocations.
          // Retirement follows completion on the preparation/replay stream.
          driver.mem_free_async(argument_image, nullptr);
        }
      } catch (...) {
        // The driver wrapper already reported any backend fault. Never
        // terminate while unwinding a failed context.
      }
    }
    graph = nullptr;
    argument_image = nullptr;
    packets.clear();
    host_argument_image.clear();
    contexts.clear();
    allocations.clear();
    provider_capture_resources.clear();
    device = nullptr;
    bytes = 0;
  }
};

struct GraphBindingExecutor::State : CudaDevice::RetainedGraphResource {
  struct Pending {
    RuntimeCompletion completion;
    std::shared_ptr<GraphBindingFrame> frame;
  };

  aot::CompiledGraph graph;
  CompileConfig config;
  Program *program{nullptr};
  CudaDevice *device{nullptr};
  KernelLauncher *launcher{nullptr};
  std::vector<std::shared_ptr<KernelExecutionHandle>> kernels;
  std::vector<KernelLauncher::Handle> handles;
  std::vector<std::shared_ptr<void>> provider_plans;
  std::mutex mutex;
  std::atomic<bool> closed{false};
  bool failed{false};
  CUgraphExec executable{nullptr};
  void *capture_stream{nullptr};
  std::shared_ptr<GraphBindingFrame> active;
  std::vector<std::weak_ptr<GraphBindingFrame>> frames;
  std::deque<Pending> pending;
  std::shared_ptr<RuntimeCompletionCudaEventPool> event_pool;
  std::uint64_t program_domain{0};
  std::uint64_t upload_calls{0};
  std::uint64_t upload_bytes{0};

  State(const aot::CompiledGraph &graph_, const CompileConfig &config_)
      : graph(graph_), config(config_) {
    event_pool = std::make_shared<RuntimeCompletionCudaEventPool>(
        std::weak_ptr<RuntimeFaultDomain>{}, 0,
        RuntimeCompletionCudaEventPool::Retention::until_clear);
  }

  ~State() override {
    retire_graph_resource();
  }

  void collect(bool sweep_directory = false) {
    // Default-stream completions are ordered. No poll on exact replay.
    while (!pending.empty() && pending.front().completion.done()) {
      pending.pop_front();
    }
    if (sweep_directory) {
      frames.erase(
          std::remove_if(frames.begin(), frames.end(),
                         [](const auto &frame) { return frame.expired(); }),
          frames.end());
    }
  }

  void retain_pending(const std::shared_ptr<GraphBindingFrame> &frame) {
    pending.push_back({RuntimeCompletion::from_cuda_stream(
                           program_domain, 0, nullptr, nullptr, event_pool),
                       frame});
  }

  void update(CUgraph source) {
    void *error_node = nullptr;
    std::uint32_t result = 0;
    auto &driver = CUDADriver::get_instance();
    const auto error = driver.graph_exec_update_v1.call(executable, source,
                                                        &error_node, &result);
    if (error != CUDA_SUCCESS || result != 0) {
      // Never silently substitute recapture/reinstantiation for this recipe.
      failed = true;
      BackendRuntimeError failure(
          Arch::cuda, error, "graph_exec_update_v1",
          fmt::format(
              "CUDA binding-frame update rejected: driver={}, result={}", error,
              result));
      program->report_backend_runtime_error(failure);
      throw failure;
    }
  }

  void retire_graph_resource() noexcept override {
    auto submission = CUDAContext::get_instance().get_submission_lock_guard();
    std::lock_guard<std::mutex> lock(mutex);
    if (closed) {
      return;
    }
    closed = true;
    bool safe = device && device->backend_calls_safe();
    auto &driver = CUDADriver::get_instance();
    if (safe) {
      try {
        auto context = CUDAContext::get_instance().get_guard();
        // Close/reset is a lifetime boundary, never replay. Also covers
        // preparation uploads for frames that were never launched.
        if (executable || !pending.empty()) {
          driver.stream_synchronize(nullptr);
        }
        if (executable) {
          driver.graph_exec_destroy(executable);
        }
        if (capture_stream) {
          driver.stream_destroy(capture_stream);
        }
      } catch (...) {
        safe = false;
      }
    }
    executable = nullptr;
    capture_stream = nullptr;
    for (const auto &entry : frames) {
      if (auto frame = entry.lock()) {
        frame->state_->release(safe);
      }
    }
    pending.clear();
    active.reset();
    frames.clear();
    event_pool->clear();
    kernels.clear();
    handles.clear();
    provider_plans.clear();
    device = nullptr;
    // The Python binding pins the Program shell. Keep its address immutable
    // so a preparation racing explicit close can acquire the existing resource
    // guard, then observe closed before touching retired backend objects.
    launcher = nullptr;
  }
};

GraphBindingFrame::GraphBindingFrame() : state_(std::make_unique<State>()) {
}

GraphBindingFrame::~GraphBindingFrame() {
  // Device retirement invalidates all still-published frames first.
  if (state_->valid) {
    auto submission = CUDAContext::get_instance().get_submission_lock_guard();
    state_->release(state_->device->backend_calls_safe());
  }
}

GraphBindingExecutor::GraphBindingExecutor(const aot::CompiledGraph &graph,
                                           const CompileConfig &config,
                                           Program &program_owner) {
  TI_ERROR_IF(
      config.arch != Arch::cuda || config.debug || config.kernel_profiler,
      "CUDA binding frames require non-debug CUDA execution without "
      "kernel-profiler instrumentation");
  TI_ERROR_IF(
      graph.dispatches.empty() || !graph.snode_tree_dependencies.empty() ||
          graph.has_indirect_dispatches() ||
          graph.has_cuda_parallel_dispatch_groups() ||
          graph.has_dispatch_labels(),
      "CUDA binding frames require an ordinary unlabeled ndarray Graph");
  auto state = std::make_shared<State>(graph, config);
  state->program = &program_owner;
  auto &driver = CUDADriver::get_instance();
  TI_ERROR_IF(!available(),
              "CUDA binding-frame executable update is unavailable");
  for (const auto &dispatch : graph.dispatches) {
    if (dispatch.cuda_capture_command) {
      TI_ERROR_IF(
          !dispatch.cuda_capture_command->supports_binding_frames() ||
              dispatch.cuda_capture_command->program() != state->program,
          "CUDA provider command does not support immutable binding frames");
      continue;
    }
    TI_ERROR_IF(!dispatch.ti_kernel || dispatch.cuda_bounded_dispatch ||
                    dispatch.cpu_bounded_dispatch ||
                    !dispatch.snode_tree_dependencies.empty(),
                "CUDA binding frames require ordinary JIT kernel dispatches");
    auto *program = dispatch.ti_kernel->program;
    TI_ERROR_IF(program != state->program,
                "CUDA binding frames cannot mix Program owners");
  }
  auto resource_guard =
      state->program->acquire_runtime_resource_submission_guard();
  auto submission = CUDAContext::get_instance().get_submission_lock_guard();
  state->device =
      dynamic_cast<CudaDevice *>(state->program->get_compute_device());
  state->launcher = dynamic_cast<KernelLauncher *>(
      &state->program->get_program_impl()->get_kernel_launcher());
  TI_ERROR_IF(!state->device || !state->launcher ||
                  !CUDAContext::get_instance().supports_mem_pool(),
              "CUDA binding frames require the CUDA launcher and async memory");
  state->program_domain = state->program->runtime_program_generation();
  for (const auto &dispatch : graph.dispatches) {
    if (dispatch.cuda_capture_command) {
      state->handles.emplace_back();
      state->kernels.push_back(nullptr);
      state->provider_plans.push_back(
          dispatch.cuda_capture_command->retain_binding_frame_plan(program_owner));
      continue;
    }
    auto kernel = state->program->compile_kernel_execution_handle(
        config, state->program->get_device_caps(), *dispatch.ti_kernel);
    const auto &compiled =
        dynamic_cast<const LLVM::CompiledKernelData &>(kernel->compiled());
    state->handles.push_back(state->launcher->register_llvm_kernel(compiled));
    state->kernels.push_back(std::move(kernel));
  }
  auto context = CUDAContext::get_instance().get_guard();
  driver.stream_create(&state->capture_stream, 1 /* nonblocking */);
  state->device->register_graph_resource(state);
  state_ = std::move(state);
}

GraphBindingExecutor::~GraphBindingExecutor() {
  // Keep the weak device registration lockable until retirement has finished.
  state_->retire_graph_resource();
}

bool GraphBindingExecutor::available() {
  auto &driver = CUDADriver::get_instance();
  return driver.graph_exec_update_v1.available() &&
         driver.graph_node_get_type.available() &&
         driver.graph_instantiate_with_flags.available() &&
         driver.stream_begin_capture.available() &&
         driver.stream_end_capture.available() &&
         CUDAContext::get_instance().supports_mem_pool();
}

const aot::CompiledGraph &GraphBindingExecutor::graph() const {
  return state_->graph;
}

std::shared_ptr<GraphBindingFrame> GraphBindingExecutor::prepare(
    const std::unordered_map<std::string, aot::IValue> &args) {
  auto &state = *state_;
  TI_ERROR_IF(state.closed, "CUDA binding-frame executor is closed or failed");
  auto resource_guard = state.program->acquire_runtime_resource_graph_scope();
  auto submission = CUDAContext::get_instance().get_submission_lock_guard();
  auto context = CUDAContext::get_instance().get_guard();
  std::lock_guard<std::mutex> lock(state.mutex);
  TI_ERROR_IF(state.closed || state.failed,
              "CUDA binding-frame executor is closed or failed");
  state.collect(true);
  auto frame = std::shared_ptr<GraphBindingFrame>(new GraphBindingFrame());
  auto &data = *frame->state_;
  data.owner = &state;
  data.device = state.device;
  data.valid = true;
  state.frames.push_back(frame);
  std::set<std::uint64_t> allocations;
  for (const auto &[name, value] : args) {
    if (value.tag == aot::ArgKind::kNdarray) {
      DeviceAllocation allocation;
      if (value.runtime_storage) {
        // Accept immutable ndarray descriptors published by Graph.bind(),
        // including qualified views. Field/external owners need a different
        // synchronization/lifecycle contract, not merely an allocation lease.
        const auto &argument = *value.runtime_storage;
        TI_ERROR_IF(
            argument.descriptor().owner().kind !=
                storage::StorageOwnerKind::kProgramNdarray,
            "CUDA binding frames require Program-owned ndarray storage: {}",
            name);
        const auto *argument_ptr = &argument;
        state.program->retain_runtime_storage_for_graph_submission(
            &argument_ptr, 1);
        const auto resolved =
            state.program->resolve_runtime_storage_argument_under_graph_guard(
                argument);
        TI_ERROR_IF(
            !resolved.valid || resolved.synchronization_domain_identity != 0,
            "CUDA binding-frame storage requires external synchronization: {}",
            name);
        allocation = resolved.allocation;
      } else {
        TI_ERROR_IF(!value.val, "CUDA binding-frame ndarray is null: {}", name);
        const auto *array = reinterpret_cast<const Ndarray *>(value.val);
        state.program->validate_ndarrays_for_external_submission(&array, 1);
        allocation = array->get_device_allocation();
      }
      TI_ERROR_IF(allocation.device != state.device,
                  "CUDA binding frame ndarray belongs to another device");
      if (allocations.insert(allocation.alloc_id).second) {
        auto lease = state.device->acquire_allocation_lease(allocation);
        TI_ERROR_IF(!lease, "CUDA binding-frame allocation is retired: {}",
                    name);
        data.allocations.push_back(std::move(lease));
      }
    } else {
      TI_ERROR_IF(value.tag != aot::ArgKind::kScalar &&
                      value.tag != aot::ArgKind::kMatrix,
                  "CUDA binding frame argument kind is unsupported: {}", name);
    }
  }
  auto &driver = CUDADriver::get_instance();
  bool capturing = false;
  bool uploaded = false;
  try {
    data.packets.reserve(state.graph.dispatches.size());
    std::vector<std::size_t> argument_offsets;
    argument_offsets.reserve(state.graph.dispatches.size());
    for (std::size_t i = 0; i < state.graph.dispatches.size(); ++i) {
      const auto &dispatch = state.graph.dispatches[i];
      if (dispatch.cuda_capture_command) {
        auto &command = *dispatch.cuda_capture_command;
        TI_ERROR_IF(!command.supports(args, *state.program),
                    "CUDA provider binding-frame arguments do not match the fixed plan");
        // Provider descriptors/preprocessing only, never mathematical work.
        command.prepare(args, *state.program);
        data.packets.emplace_back();
        argument_offsets.push_back(0);
        continue;
      }
      data.contexts.push_back(
          std::make_unique<LaunchContextBuilder>(dispatch.ti_kernel));
      auto &launch = *data.contexts.back();
      state.graph.init_runtime_context(dispatch.symbolic_args, args, launch);
      state.program->resolve_ndarray_launch_context_under_guard(launch);
      state.program->resolve_runtime_storage_launch_context_under_guard(launch);
      data.packets.emplace_back();
      auto &packet = data.packets.back();
      TI_ERROR_IF(!state.launcher->prepare_cuda_graph_launch_packet(
                      state.handles[i], launch, packet),
                  "CUDA binding-frame launch packet is unsupported");
      // Preserve the 256-byte alignment each separate CUDA allocation had.
      // Padding is part of actual argument storage and its reported bytes.
      constexpr std::size_t alignment = 256;
      const auto previous_size = data.host_argument_image.size();
      TI_ERROR_IF(previous_size >
                      std::numeric_limits<std::size_t>::max() - (alignment - 1),
                  "CUDA binding-frame argument image is too large");
      const auto offset = (previous_size + alignment - 1) & ~(alignment - 1);
      argument_offsets.push_back(offset);
      if (packet.arg_buffer_size) {
        TI_ERROR_IF(packet.arg_buffer_size >
                        std::numeric_limits<std::size_t>::max() - offset,
                    "CUDA binding-frame argument image is too large");
        data.host_argument_image.resize(offset + packet.arg_buffer_size);
        std::memcpy(data.host_argument_image.data() + offset,
                    packet.context.arg_buffer, packet.arg_buffer_size);
      }
    }
    data.bytes = data.host_argument_image.size();
    if (data.bytes) {
      driver.malloc_async(&data.argument_image, data.bytes, nullptr);
      // Keep the host image until the frame retires, including failed capture.
      // Only preparation uploads; exact replay retains its previous path.
      uploaded = true;
      driver.memcpy_host_to_device_async(data.argument_image,
                                         data.host_argument_image.data(),
                                         data.bytes, nullptr);
      ++state.upload_calls;
      state.upload_bytes += data.bytes;
    }
    for (std::size_t i = 0; i < data.packets.size(); ++i) {
      auto &packet = data.packets[i];
      if (packet.arg_buffer_size) {
        packet.device_arg_buffer =
            static_cast<char *>(data.argument_image) + argument_offsets[i];
        packet.context.arg_buffer =
            static_cast<char *>(packet.device_arg_buffer);
      }
    }
    driver.stream_begin_capture(state.capture_stream,
                                CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
    capturing = true;
    for (std::size_t i = 0; i < state.graph.dispatches.size(); ++i) {
      const auto &command = state.graph.dispatches[i].cuda_capture_command;
      if (command) {
        command->record(args, *state.program, state.capture_stream);
        if (auto resource = command->take_capture_resources()) {
          data.provider_capture_resources.push_back(std::move(resource));
        }
      } else {
        state.launcher->capture_cuda_graph_launch(data.packets[i],
                                                  state.capture_stream);
      }
    }
    const auto error =
        driver.stream_end_capture.call(state.capture_stream, &data.graph);
    capturing = false;
    TI_ERROR_IF(error != CUDA_SUCCESS, "CUDA binding-frame capture failed: {}",
                error);
    driver.graph_get_nodes(data.graph, nullptr, &data.nodes);
    TI_ERROR_IF(data.nodes == 0, "CUDA binding frame contains no kernel nodes");
    std::vector<void *> nodes(data.nodes);
    driver.graph_get_nodes(data.graph, nodes.data(), &data.nodes);
    for (auto *node : nodes) {
      std::uint32_t type = 0;
      driver.graph_node_get_type(node, &type);
      // Fixed-plan providers also record workspace clears and copies. Their
      // resource lease owns any host source bytes per immutable frame. This
      // does not admit host callbacks, allocations, or opaque child graphs.
      const bool provider_memory = !state.provider_plans.empty() &&
                                   (type == 1 || type == 2);  // memcpy/memset
      TI_ERROR_IF(type != 0 /* kernel */ && !provider_memory,
                  "CUDA binding frame contains an unsupported node");
    }
    if (!state.executable) {
      driver.graph_instantiate_with_flags(&state.executable, data.graph, 0);
      state.active = frame;
    } else {
      // Preparation may leave the executable configured for this frame: every
      // run explicitly names its frame, and no mathematical work executes here.
      // Retain the previous configuration for already queued launches before
      // updating. Restoring it only to update again on run would add two
      // redundant driver calls to the common prepare-then-run path.
      state.retain_pending(state.active);
      state.update(data.graph);
      state.active = frame;
    }
    state.program->mark_runtime_submission_pending();
    state.retain_pending(frame);
    resource_guard.finish_external_access_epoch();
    return frame;
  } catch (...) {
    if (capturing) {
      CUgraph abandoned = nullptr;
      driver.stream_end_capture.call(state.capture_stream, &abandoned);
      if (abandoned) {
        driver.graph_destroy.call(abandoned);
      }
    }
    if (uploaded) {
      // Unpublished failed preparation may still have host staging in flight.
      // This cold rollback is deliberately not a replay fallback.
      driver.stream_synchronize(nullptr);
    }
    if (state.active != frame) {
      data.release(state.device->backend_calls_safe());
    }
    throw;
  }
}

void GraphBindingExecutor::run(
    const std::shared_ptr<GraphBindingFrame> &frame) {
  auto submission = CUDAContext::get_instance().get_submission_lock_guard();
  auto context = CUDAContext::get_instance().get_guard();
  auto &state = *state_;
  std::lock_guard<std::mutex> lock(state.mutex);
  TI_ERROR_IF(state.closed || state.failed || !frame || !frame->state_->valid ||
                  frame->state_->owner != &state,
              "CUDA binding frame is closed or belongs to another executor");
  auto completion_scope = state.program->acquire_runtime_submission_scope();
  if (state.active != frame) {
    state.collect();
    // Only an actual binding switch needs a last-use completion event.
    state.retain_pending(state.active);
    state.update(frame->state_->graph);
    state.active = frame;
  }
  CUDADriver::get_instance().graph_launch(state.executable, nullptr);
  state.program->mark_runtime_submission(
      RuntimeSubmissionKind::kGraphBackendSubmission);
}

void GraphBindingExecutor::close() {
  state_->retire_graph_resource();
}

std::unordered_map<std::string, std::uint64_t>
GraphBindingExecutor::snapshot() {
  auto submission = CUDAContext::get_instance().get_submission_lock_guard();
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto &state = *state_;
  if (!state.closed) {
    state.collect(true);
  }
  std::uint64_t bytes = 0;
  std::uint64_t frames = 0;
  std::uint64_t nodes = 0;
  for (const auto &entry : state.frames) {
    if (auto frame = entry.lock()) {
      bytes += frame->state_->bytes;
      nodes += frame->state_->nodes;
      ++frames;
    }
  }
  const auto events = state.event_pool->snapshot();
  return {{"closed", state.closed.load(std::memory_order_relaxed)},
          {"failed", state.failed},
          {"frames", frames},
          {"argument_bytes", bytes},
          {"kernel_nodes", nodes},
          {"executables", state.executable ? 1 : 0},
          {"pending_frame_leases", state.pending.size()},
          // Driver event bytes are opaque. Counts describe retained handles,
          // never requested ndarray bytes or a measured VRAM peak.
          {"completion_events_created", events.created},
          {"completion_events_reused", events.reused},
          {"completion_events_cached", events.cached},
          {"completion_events_destroyed", events.destroyed},
          {"completion_events_abandoned", events.abandoned},
          {"completion_event_cache_until_close", 1},
          {"preparation_upload_calls", state.upload_calls},
          {"preparation_upload_bytes", state.upload_bytes}};
}

}  // namespace taichi::lang::cuda
