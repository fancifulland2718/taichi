#include "taichi/program/program.h"
#include "taichi/program/ndarray.h"
#include "taichi/runtime/gfx/graph_recording.h"

#ifdef TI_WITH_VULKAN
#include "taichi/runtime/gfx/kernel_launcher.h"
#include "taichi/runtime/gfx/runtime.h"
#include "taichi/program/storage_view.h"

namespace taichi::lang::gfx {
aot::CompiledGraph graph_recording_argument_schema(
    const std::vector<GraphRecordingSource> &sources) {
  aot::CompiledGraph result;
  auto merge = [&](const aot::Arg &arg) {
    auto [entry, inserted] = result.args.emplace(arg.name, arg);
    const auto &prior = entry->second;
    TI_ERROR_IF(
        !inserted && (arg.tag != prior.tag || arg.dtype_id != prior.dtype_id ||
                      arg.field_dim != prior.field_dim ||
                      arg.element_shape != prior.element_shape),
        "Prepared Vulkan Graph has conflicting argument declarations: {}",
        arg.name);
  };
  for (const auto &source : sources) {
    if (const auto *graph = std::get_if<aot::CompiledGraph *>(&source.value)) {
      TI_ERROR_IF(!*graph, "Prepared Vulkan Graph segment is null");
      for (const auto &[name, arg] : (*graph)->args) {
        merge(arg);
      }
    } else {
      const auto &command =
          std::get<std::shared_ptr<ExternalGraphCommand>>(source.value);
      TI_ERROR_IF(!command, "Prepared Vulkan Graph command is null");
      for (const auto &arg : command->arguments()) {
        merge(arg);
      }
    }
  }
  return result;
}

FixedGraphRecording::FixedGraphRecording(
    Program &program,
    std::unique_ptr<GraphReplayRegistration> registration)
    : program_(&program), registration_(std::move(registration)) {
}
FixedGraphRecording::~FixedGraphRecording() = default;
void FixedGraphRecording::run() {
  auto guard = program_->acquire_runtime_resource_submission_guard();
  std::lock_guard<std::mutex> lock(mutex_);
  TI_ERROR_IF(!registration_, "Prepared Vulkan Graph is closed");
  registration_->launch_prepared();
  program_->mark_runtime_submission_pending();
}
void FixedGraphRecording::close() {
  std::lock_guard<std::mutex> lock(mutex_);
  registration_.reset();
}
std::uint64_t FixedGraphRecording::argument_bytes() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return registration_
             ? registration_->snapshot_stats().known_persistent_argument_bytes
             : 0;
}
}  // namespace taichi::lang::gfx

namespace taichi::lang {
std::shared_ptr<gfx::FixedGraphRecording>
Program::create_vulkan_graph_recording(
    const std::vector<gfx::GraphRecordingSource> &sources,
    const std::unordered_map<std::string, aot::IValue> &args) {
  auto scope = acquire_runtime_resource_graph_scope();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan || compile_config().debug ||
                  compile_config().kernel_profiler,
              "Prepared Vulkan Graph requires non-debug Vulkan execution");
  auto *launcher = dynamic_cast<gfx::KernelLauncher *>(&get_kernel_launcher());
  TI_ERROR_IF(!launcher, "Prepared Vulkan Graph launcher is unavailable");
  std::vector<const Ndarray *> arrays;
  for (const auto &[name, value] : args) {
    if (value.tag == aot::ArgKind::kNdarray) {
      TI_ERROR_IF(!value.val,
                  "Prepared Vulkan Graph requires Program ndarray owners: {}",
                  name);
      const auto *array = reinterpret_cast<const Ndarray *>(value.val);
      TI_ERROR_IF(array->owning_program() != this,
                  "Prepared Vulkan Graph ndarray belongs to another runtime");
      arrays.push_back(array);
      if (value.runtime_storage) {
        TI_ERROR_IF(
            value.runtime_storage->descriptor().owner().kind !=
                    storage::StorageOwnerKind::kProgramNdarray ||
                value.runtime_storage->descriptor().owner().ndarray_handle !=
                    array->runtime_resource_handle(),
            "Prepared Vulkan Graph descriptor must match its retained ndarray "
            "owner");
        const auto *argument = value.runtime_storage;
        retain_runtime_storage_for_graph_submission(&argument, 1);
      }
    } else {
      TI_ERROR_IF(
          value.tag != aot::ArgKind::kScalar &&
              value.tag != aot::ArgKind::kMatrix,
          "Prepared Vulkan Graph supports buffer and value arguments only");
    }
  }
  std::vector<std::shared_ptr<void>> owners;
  owners.push_back(
      std::make_shared<NdarrayLaunchLeases>(acquire_ndarray_leases(arrays)));
  std::vector<std::unique_ptr<LaunchContextBuilder>> contexts;
  std::vector<gfx::GfxRuntime::GraphRecordingOperation> operations;
  for (const auto &source : sources) {
    if (const auto *graph_pointer =
            std::get_if<aot::CompiledGraph *>(&source.value)) {
      const auto &graph = **graph_pointer;
      TI_ERROR_IF(!graph.snode_tree_dependencies.empty() ||
                      graph.has_indirect_dispatches() ||
                      graph.has_cuda_parallel_dispatch_groups() ||
                      graph.has_dispatch_labels(),
                  "Prepared Vulkan Graph requires ordinary ndarray segments");
      // Retain synthetic kernels and their compiled payloads independently of
      // Python builders and subsequent compilation-cache eviction.
      for (auto &kernel : graph.owned_jit_kernels) {
        owners.push_back(kernel);
      }
      for (const auto &dispatch : graph.dispatches) {
        TI_ERROR_IF(!dispatch.ti_kernel ||
                        dispatch.ti_kernel->program != this ||
                        dispatch.cuda_capture_command ||
                        dispatch.cuda_bounded_dispatch ||
                        dispatch.cpu_bounded_dispatch ||
                        !dispatch.snode_tree_dependencies.empty(),
                    "Prepared Vulkan Graph cannot lower this dispatch");
        auto kernel = compile_kernel_execution_handle(
            compile_config(), get_device_caps(), *dispatch.ti_kernel);
        auto handle = launcher->get_or_register_kernel(kernel->compiled());
        owners.push_back(std::move(kernel));
        auto context =
            std::make_unique<LaunchContextBuilder>(dispatch.ti_kernel);
        graph.init_runtime_context(dispatch.symbolic_args, args, *context);
        resolve_ndarray_launch_context_under_guard(*context);
        resolve_runtime_storage_launch_context_under_guard(*context);
        operations.push_back({{handle, context.get()}, {}});
        contexts.push_back(std::move(context));
      }
    } else {
      const auto &command =
          std::get<std::shared_ptr<gfx::ExternalGraphCommand>>(source.value);
      command->validate(*this, args);
      owners.push_back(command);
      operations.push_back(
          {{}, [command](Device *device, CommandList *commands) {
             command->record(device, commands);
           }});
    }
  }
  auto registration =
      launcher->runtime()->prepare_fixed_graph(operations, std::move(owners));
  return std::make_shared<gfx::FixedGraphRecording>(*this,
                                                    std::move(registration));
}
}  // namespace taichi::lang
#else
namespace taichi::lang {
std::shared_ptr<gfx::FixedGraphRecording>
Program::create_vulkan_graph_recording(
    const std::vector<gfx::GraphRecordingSource> &,
    const std::unordered_map<std::string, aot::IValue> &) {
  TI_ERROR("Prepared Vulkan Graph is unavailable in this build");
}
}  // namespace taichi::lang
#endif
