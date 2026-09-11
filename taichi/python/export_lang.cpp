// Bindings for the python frontend

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>
#include "taichi/ir/snode.h"
#include "taichi/common/commit_hash.h"
#include "taichi/common/runtime_contract.h"
#include "taichi/rhi/cuda/cuda_capability.h"

#if TI_WITH_LLVM
#include "llvm/Config/llvm-config.h"
#endif

#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"
#include "pybind11/numpy.h"
#include "fp16.h"

#include "taichi/ir/expression_ops.h"
#include "taichi/ir/frontend_ir.h"
#include "taichi/ir/statements.h"
#include "taichi/analysis/offline_cache_util.h"
#include "taichi/program/graph_builder.h"
#include "taichi/program/graph_value_program.h"
#include "taichi/program/graph_value_fusion.h"
#include "taichi/program/cuda_scan_capture.h"
#include "taichi/program/cuda_cublaslt_capture.h"
#include "taichi/program/cuda_cutensor_capture.h"
#include "taichi/program/cuda_cusparselt_capture.h"
#include "taichi/program/extension.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/matrix.h"
#include "taichi/program/runtime_statistics.h"
#include "taichi/program/storage_view.h"
#include "taichi/python/export_storage_view.h"
#include "taichi/python/export.h"
#include "taichi/math/svd.h"
#include "taichi/system/timeline.h"
#include "taichi/system/profiler_annotation.h"
#include "taichi/util/environ_config.h"
#include "taichi/codegen/spirv/spv_stats.h"
#include "taichi/python/snode_registry.h"
#include "taichi/program/sparse_matrix.h"
#include "taichi/program/sparse_preconditioner.h"
#include "taichi/program/sparse_solver.h"
#include "taichi/program/conjugate_gradient.h"
#include "taichi/program/sparse_bicgstab.h"
#include "taichi/program/sparse_fixed_bicgstab.h"
#include "taichi/program/sparse_fixed_gmres.h"
#include "taichi/program/sparse_minres.h"
#include "taichi/program/sparse_operator_minres.h"
#include "taichi/program/sparse_device_bicgstab.h"
#include "taichi/program/sparse_device_gmres.h"
#include "taichi/program/sparse_device_minres.h"
#include "taichi/aot/graph_data.h"
#include "taichi/runtime/gfx/runtime.h"
#include "taichi/runtime/gfx/graph_recording.h"
#include "taichi/ir/mesh.h"

#include "taichi/program/kernel_profiler.h"

#if defined(TI_WITH_CUDA)
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/runtime/cuda/cuda_artifact_provider.h"
#include "taichi/runtime/cuda/cuda_compileiq_protocol.h"
#include "taichi/runtime/cuda/graph_binding_frame.h"
#include "taichi/runtime/cuda/graph_memory_pool.h"
#include "taichi/runtime/cuda/kernel_launcher.h"
#endif
#if defined(TI_WITH_VULKAN)
#include "taichi/codegen/spirv/compiled_kernel_data.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#endif

namespace taichi {
bool test_threading();

}  // namespace taichi

namespace taichi::lang {

std::string libdevice_path();

namespace irpass {
void get_fs_inner_stats(uint64_t *entries,
                        uint64_t *noop_returns,
                        uint64_t *total_iterations);
void reset_fs_inner_stats();
}  // namespace irpass

}  // namespace taichi::lang

namespace taichi {
namespace {

class PythonScalarArgumentPatchPlan {
 public:
  PythonScalarArgumentPatchPlan(std::vector<int> indices, std::string kinds) {
    TI_ERROR_IF(indices.size() != kinds.size(),
                "Scalar argument patch indices and kinds must have equal size");
    py::module_ numpy = py::module_::import("numpy");
    numpy_floating_ = numpy.attr("floating");
    numpy_integer_ = numpy.attr("integer");
    entries_.reserve(indices.size());
    for (std::size_t position = 0; position < indices.size(); ++position) {
      TI_ERROR_IF(indices[position] < 0,
                  "Scalar argument patch indices must be non-negative");
      const char kind = kinds[position];
      TI_ERROR_IF(kind != 'f' && kind != 'i' && kind != 'u',
                  "Scalar argument patch kind must be f, i, or u");
      entries_.push_back({indices[position], kind, {indices[position]}});
    }
  }

  int apply(lang::LaunchContextBuilder &context,
            const py::sequence &args) const {
    for (const Entry &entry : entries_) {
      TI_ERROR_IF(static_cast<std::size_t>(entry.index) >= args.size(),
                  "Scalar argument patch index is outside the argument tuple");
      const py::handle value = args[entry.index];
      const bool integer =
          PyLong_Check(value.ptr()) || is_instance(value, numpy_integer_);
      if (entry.kind == 'f') {
        const bool real =
            integer || PyFloat_Check(value.ptr()) ||
            is_instance(value, numpy_floating_);
        if (!real) {
          return entry.index;
        }
        context.set_arg_float(entry.arg_id, py::cast<double>(value));
      } else {
        if (!integer) {
          return entry.index;
        }
        if (entry.kind == 'i') {
          context.set_arg_int(entry.arg_id, py::cast<std::int64_t>(value));
        } else {
          context.set_arg_uint(entry.arg_id, py::cast<std::uint64_t>(value));
        }
      }
    }
    return -1;
  }

 private:
  struct Entry {
    int index;
    char kind;
    std::vector<int> arg_id;
  };

  static bool is_instance(const py::handle &value,
                          const py::object &type) {
    const int result = PyObject_IsInstance(value.ptr(), type.ptr());
    if (result < 0) {
      throw py::error_already_set();
    }
    return result != 0;
  }

  std::vector<Entry> entries_;
  py::object numpy_floating_;
  py::object numpy_integer_;
};

py::dict offloaded_task_manifest_to_python(
    const lang::OffloadedTaskManifest &task) {
  py::dict item;
  item["task_id"] = task.task_id;
  item["logical_task_id"] = task.logical_task_id;
  item["optimization_spec_id"] = task.optimization_spec_id;
  item["task_name"] = task.task_name;
  item["backend"] = arch_name(task.arch);
  item["task_index"] = task.task_index;
  item["task_type"] = lang::offloaded_task_type_name(task.task_type);
  auto set_optional = [&](const char *name, const auto &value) {
    if (value.has_value()) {
      item[name] = *value;
    } else {
      item[name] = py::none();
    }
  };
  set_optional("requested_grid_size", task.requested_grid_size);
  set_optional("requested_block_size", task.requested_block_size);
  item["source_block_size_explicit"] = task.source_block_size_explicit;
  item["requested_thread_local_mode"] = task.requested_thread_local_mode;
  item["requested_cuda_min_blocks_per_sm"] =
      task.requested_cuda_min_blocks_per_sm;
  set_optional("requested_cuda_max_registers",
               task.requested_cuda_max_registers);
  set_optional("requested_grid_residency_waves",
               task.requested_grid_residency_waves);
  item["requested_range_work_per_thread_target"] =
      task.requested_range_work_per_thread_target;
  item["requested_memory_strategy"] = task.requested_memory_strategy;
  item["requested_sparse_list_policy"] =
      task.requested_sparse_list_policy;
  set_optional("selected_grid_size", task.selected_grid_size);
  set_optional("selected_block_size", task.selected_block_size);
  set_optional("actual_grid_size", task.actual_grid_size);
  set_optional("actual_block_size", task.actual_block_size);
  item["actual_geometry_kind"] = task.actual_geometry_kind;
  item["actual_geometry_reason"] = task.actual_geometry_reason;
  item["range_mapping"] = task.range_mapping;
  set_optional("staged_external_arg_index", task.staged_external_arg_index);
  set_optional("staged_halo_low", task.staged_halo_low);
  set_optional("staged_halo_high", task.staged_halo_high);
  item["staged_external_arg_indices"] = task.staged_external_arg_indices;
  item["staged_halo_lows"] = task.staged_halo_lows;
  item["staged_halo_highs"] = task.staged_halo_highs;
  item["staged_byte_offsets"] = task.staged_byte_offsets;
  item["staged_element_bytes"] = task.staged_element_bytes;
  item["staged_scalar_bytes"] = task.staged_scalar_bytes;
  item["staged_element_shapes"] = task.staged_element_shapes;
  item["staged_iteration_shape"] = task.staged_iteration_shape;
  item["staged_iteration_origin"] = task.staged_iteration_origin;
  item["staged_tile_shape"] = task.staged_tile_shape;
  item["staged_halo_lows_nd"] = task.staged_halo_lows_nd;
  item["staged_halo_highs_nd"] = task.staged_halo_highs_nd;
  item["staged_access_offsets"] = task.staged_access_offsets;
  item["sparse_list_op"] = task.sparse_list_op;
  item["sparse_list_snode_id"] = task.sparse_list_snode_id;
  item["sparse_list_parent_snode_id"] = task.sparse_list_parent_snode_id;
  set_optional("sparse_list_parent_grid_bound",
               task.sparse_list_parent_grid_bound);
  item["may_mutate_sparse_topology"] = task.may_mutate_sparse_topology;
  item["sparse_mutation_snode_id"] = task.sparse_mutation_snode_id;
  if (task.constant_range_size.has_value()) {
    item["constant_range_size"] = *task.constant_range_size;
  } else {
    item["constant_range_size"] = py::none();
  }
  item["static_shared_bytes"] = task.static_shared_bytes;
  item["dynamic_shared_bytes"] = task.dynamic_shared_bytes;
  item["thread_local_bytes"] = task.thread_local_bytes;
  return item;
}

lang::OffloadedTaskManifest overlay_execution_plan_launch_requests(
    const lang::OffloadedTaskManifest &compiled_task,
    const lang::Kernel *kernel) {
  auto task = compiled_task;
  if (kernel == nullptr ||
      !kernel->get_offload_execution_plan().has_value()) {
    return task;
  }
  const auto &spec = kernel->offload_task_optimization_spec(
      task.task_index, task.task_type);
  if (spec.grid_residency_waves > 0) {
    task.requested_grid_residency_waves = spec.grid_residency_waves;
  } else {
    task.requested_grid_residency_waves.reset();
  }
  task.requested_range_work_per_thread_target =
      spec.range_work_per_thread_target;
  task.requested_memory_strategy = spec.memory_strategy;
  task.requested_sparse_list_policy = spec.sparse_list_policy;
  if (spec.sparse_list_policy == "parent_capacity_bound" &&
      task.sparse_list_parent_grid_bound.has_value() &&
      task.selected_grid_size.has_value()) {
    const auto resolved = std::min(*task.sparse_list_parent_grid_bound,
                                   *task.selected_grid_size);
    task.actual_grid_size = resolved;
    task.actual_geometry_kind = "sparse_parent_capacity_bound";
    task.actual_geometry_reason =
        "the complete sparse traversal plan capped listgen CTAs by the "
        "compiler-proven parent-list capacity";
  }
  return task;
}

py::dict graph_kernel_metadata_to_python(
    const lang::GraphKernelMetadata &metadata) {
  py::dict item;
  item["version"] = metadata.version;
  item["available"] = metadata.available;
  item["opaque"] = metadata.opaque;
  item["elementwise"] = metadata.elementwise;
  item["synchronization"] = metadata.synchronization;
  item["blocker"] = metadata.blocker;
  item["side_effects"] = metadata.side_effects;
  py::dict domain;
  domain["kind"] = metadata.iteration_domain.kind;
  domain["arg_id"] = metadata.iteration_domain.arg_id;
  domain["axis"] = metadata.iteration_domain.axis;
  domain["begin"] = metadata.iteration_domain.begin;
  domain["end"] = metadata.iteration_domain.end;
  domain["logical_shape"] = metadata.iteration_domain.logical_shape;
  domain["logical_origin"] = metadata.iteration_domain.logical_origin;
  item["iteration_domain"] = std::move(domain);
  py::list effects;
  for (const auto &effect : metadata.effects) {
    py::dict encoded;
    encoded["resource_kind"] = effect.resource_kind;
    encoded["arg_id"] = effect.arg_id;
    encoded["snode_tree_id"] = effect.snode_tree_id;
    encoded["snode_id"] = effect.snode_id;
    encoded["is_grad"] = effect.is_grad;
    encoded["access"] = effect.access;
    py::dict footprint;
    footprint["pattern"] = effect.footprint.pattern;
    footprint["iteration_rank"] = effect.footprint.iteration_rank;
    footprint["affine_coefficients"] =
        effect.footprint.affine_coefficients;
    footprint["affine_index_offsets"] =
        effect.footprint.affine_index_offsets;
    footprint["affine_offsets"] = effect.footprint.affine_offsets;
    footprint["halo"] = effect.footprint.halo;
    footprint["contiguous_axis"] = effect.footprint.contiguous_axis;
    footprint["reuse_class"] = effect.footprint.reuse_class;
    encoded["footprint"] = std::move(footprint);
    effects.append(std::move(encoded));
  }
  item["effects"] = std::move(effects);
  return item;
}

const char *autodiff_mode_name(AutodiffMode mode) {
  switch (mode) {
    case AutodiffMode::kNone:
      return "primal";
    case AutodiffMode::kForward:
      return "forward";
    case AutodiffMode::kReverse:
      return "adjoint";
    case AutodiffMode::kCheckAutodiffValid:
      return "validation";
  }
  return "unknown";
}

#if defined(TI_WITH_VULKAN)
const char *spirv_buffer_type_name(
    lang::spirv::TaskAttributes::BufferType type) {
  using BufferType = lang::spirv::TaskAttributes::BufferType;
  switch (type) {
    case BufferType::Root:
      return "root";
    case BufferType::GlobalTmps:
      return "global_tmps";
    case BufferType::Args:
      return "args";
    case BufferType::Rets:
      return "rets";
    case BufferType::ListGen:
      return "listgen";
    case BufferType::ExtArr:
      return "external_array";
    case BufferType::ArgPack:
      return "argpack";
    case BufferType::NodeAllocatorPool:
      return "node_allocator_pool";
    case BufferType::HashOverflow:
      return "hash_overflow";
  }
  return "unknown";
}

const char *spirv_buffer_access_name(std::uint32_t access) {
  using BufferBind = lang::spirv::TaskAttributes::BufferBind;
  if (access == BufferBind::kAccessRead) {
    return "read";
  }
  if (access == BufferBind::kAccessWrite) {
    return "write";
  }
  if (access == BufferBind::kAccessReadWrite) {
    return "read_write";
  }
  return "opaque";
}

py::dict spirv_task_metadata_to_python(
    const lang::spirv::TaskAttributes &task) {
  py::dict metadata;
  metadata["entry_point"] = task.name;
  metadata["local_size"] =
      py::make_tuple(task.advisory_num_threads_per_group, 1, 1);
  py::list bindings;
  for (const auto &binding : task.buffer_binds) {
    py::dict encoded;
    encoded["kind"] = "storage_buffer";
    encoded["buffer_type"] = spirv_buffer_type_name(binding.buffer.type);
    encoded["logical_path"] = binding.buffer.root_id;
    encoded["binding"] = binding.binding;
    encoded["chunk_count"] = binding.chunk_count;
    encoded["access"] = spirv_buffer_access_name(binding.access);
    bindings.append(std::move(encoded));
  }
  for (const auto &binding : task.texture_binds) {
    py::dict encoded;
    encoded["kind"] =
        binding.is_storage ? "storage_image" : "sampled_image";
    encoded["buffer_type"] =
        binding.is_storage ? "storage_image" : "sampled_image";
    encoded["logical_path"] = binding.arg_id;
    encoded["binding"] = binding.binding;
    encoded["chunk_count"] = 0;
    encoded["access"] = binding.is_storage ? "read_write" : "read";
    bindings.append(std::move(encoded));
  }
  for (const auto &binding : task.acceleration_structure_binds) {
    py::dict encoded;
    encoded["kind"] = "acceleration_structure";
    encoded["buffer_type"] = "acceleration_structure";
    encoded["logical_path"] = binding.arg_id;
    encoded["binding"] = binding.binding;
    encoded["chunk_count"] = 0;
    encoded["access"] = "read";
    bindings.append(std::move(encoded));
  }
  metadata["bindings"] = std::move(bindings);
  return metadata;
}
#endif

py::dict compiled_kernel_gpu_semantics_to_python(
    const lang::CompiledKernelData &compiled,
    AutodiffMode autodiff_mode,
    const lang::Kernel *kernel,
    const lang::aot::CompiledDispatch *graph_dispatch = nullptr,
    std::size_t graph_dispatch_index = 0) {
  const auto backend = compiled.arch();
  py::dict result;
  result["backend"] = arch_name(backend);
  result["kernel_identity"] = compiled.kernel_identity();
  result["logical_kernel_identity"] = compiled.logical_kernel_identity();
  result["optimization_spec_identity"] =
      compiled.optimization_spec_identity();
  result["autodiff_role"] = autodiff_mode_name(autodiff_mode);
  result["graph_metadata"] =
      graph_kernel_metadata_to_python(compiled.graph_metadata());
  result["regular_handle_registered"] = compiled.get_handle().has_value();
  result["graph_masked_handle_registered"] =
      compiled.get_graph_masked_handle().has_value();
  if (graph_dispatch != nullptr) {
    result["graph_dispatch_index"] = graph_dispatch_index;
    result["graph_kernel_name"] = graph_dispatch->kernel_name;
    result["graph_dispatch_label"] = graph_dispatch->dispatch_label;
    result["graph_indirect"] =
        graph_dispatch->indirect_dispatch_arg.has_value();
    result["graph_source_dispatch_count"] =
        std::max<std::size_t>(1, graph_dispatch->source_dispatches.size());
  }
#if defined(TI_WITH_VULKAN)
  const lang::spirv::CompiledKernelData *spirv_compiled = nullptr;
  if (backend == Arch::vulkan) {
    spirv_compiled =
        dynamic_cast<const lang::spirv::CompiledKernelData *>(&compiled);
    TI_ERROR_IF(spirv_compiled == nullptr,
                "Vulkan GPU semantics require SPIR-V compiled data");
  }
#endif
  py::list tasks;
  const auto manifest = compiled.task_manifest();
  for (std::size_t index = 0; index < manifest.size(); ++index) {
    const auto task =
        overlay_execution_plan_launch_requests(manifest[index], kernel);
    auto encoded = offloaded_task_manifest_to_python(task);
    if (graph_dispatch != nullptr) {
      encoded["graph_dispatch_index"] = graph_dispatch_index;
      encoded["graph_kernel_name"] = graph_dispatch->kernel_name;
      encoded["graph_dispatch_label"] = graph_dispatch->dispatch_label;
      encoded["graph_indirect"] =
          graph_dispatch->indirect_dispatch_arg.has_value();
      encoded["graph_source_dispatch_count"] =
          std::max<std::size_t>(1, graph_dispatch->source_dispatches.size());
      if (graph_dispatch->indirect_dispatch_arg.has_value()) {
        encoded["actual_grid_size"] = py::none();
        encoded["actual_block_size"] = py::none();
        encoded["actual_geometry_kind"] = "runtime_indirect";
        encoded["actual_geometry_reason"] =
            "the device-owned dispatch packet determines geometry for each "
            "invocation without host readback";
      }
    }
#if defined(TI_WITH_VULKAN)
    if (spirv_compiled != nullptr) {
      const auto &attributes = spirv_compiled->get_internal_data()
                                   .metadata.kernel_attribs.tasks_attribs;
      TI_ERROR_IF(attributes.size() != manifest.size(),
                  "Vulkan task metadata count mismatch");
      encoded["backend_metadata"] =
          spirv_task_metadata_to_python(attributes[index]);
    }
#endif
    tasks.append(std::move(encoded));
  }
  result["tasks"] = std::move(tasks);
  return result;
}

// Record native primitive telemetry inside the existing pybind call. Keeping
// this at the binding boundary covers cold calls, cached plan descriptors and
// direct advanced usage without adding a second Python-to-C++ round trip.
template <typename Result, typename... Args>
auto tracked_native_program_method(Result (lang::Program::*method)(Args...)) {
  return [method](lang::Program *program, Args... args) -> Result {
    try {
      if constexpr (std::is_void_v<Result>) {
        (program->*method)(std::forward<Args>(args)...);
        program->record_runtime_submission_stat(
            lang::RuntimeSubmissionKind::kNative);
      } else {
        Result result = (program->*method)(std::forward<Args>(args)...);
        program->record_runtime_submission_stat(
            lang::RuntimeSubmissionKind::kNative);
        return result;
      }
    } catch (...) {
      program->record_runtime_submission_failure();
      throw;
    }
  };
}

std::size_t checked_buffer_nbytes(const py::buffer_info &info,
                                  const char *invalid_message,
                                  const char *oversized_message) {
  TI_ERROR_IF(info.ndim < 0 || info.itemsize < 0 || info.size < 0, "{}",
              invalid_message);
  const auto size = static_cast<std::size_t>(info.size);
  const auto itemsize = static_cast<std::size_t>(info.itemsize);
  TI_ERROR_IF(itemsize != 0 &&
                  size > std::numeric_limits<std::size_t>::max() / itemsize,
              "{}", oversized_message);
  return size * itemsize;
}

py::dict runtime_trace_snapshot_to_dict(
    const lang::RuntimeTraceSnapshot &snapshot) {
  py::dict result;
  result["program_domain"] = snapshot.program_domain;
  result["session"] = snapshot.session;
  result["enabled"] = snapshot.enabled;
  result["max_threads"] = snapshot.max_threads;
  result["events_per_thread"] = snapshot.events_per_thread;
  result["event_capacity"] = snapshot.event_capacity;
  result["allocated_bytes"] = snapshot.allocated_bytes;
  result["recorded_events"] = snapshot.recorded_events;
  result["dropped_events"] = snapshot.dropped_events;
  return result;
}

py::dict primitive_workspace_snapshot_to_dict(
    const lang::PrimitiveWorkspaceSnapshot &snapshot) {
  py::dict result;
  result["budget_bytes"] = snapshot.budget_bytes;
  result["reserved_bytes"] = snapshot.reserved_bytes;
  result["in_use_bytes"] = snapshot.in_use_bytes;
  result["persistent_bytes"] = snapshot.persistent_bytes;
  result["reclaimable_bytes"] = snapshot.reclaimable_bytes;
  result["over_budget_bytes"] = snapshot.over_budget_bytes;
  result["peak_reserved_bytes"] = snapshot.peak_reserved_bytes;
  result["peak_in_use_bytes"] = snapshot.peak_in_use_bytes;
  result["entries"] = snapshot.entries;
  result["active_leases"] = snapshot.active_leases;
  result["acquisitions"] = snapshot.acquisitions;
  result["cache_hits"] = snapshot.cache_hits;
  result["cache_misses"] = snapshot.cache_misses;
  result["growth_events"] = snapshot.growth_events;
  result["clear_calls"] = snapshot.clear_calls;
  result["cleared_entries"] = snapshot.cleared_entries;
  result["trim_calls"] = snapshot.trim_calls;
  result["evictions"] = snapshot.evictions;
  result["lock_samples"] = snapshot.lock_samples;
  result["lock_contentions"] = snapshot.lock_contentions;
  result["lock_wait_ns"] = snapshot.lock_wait_ns;
  return result;
}

}  // namespace

void export_lang(py::module &m) {
  m.def("_compile_graph_segmented_reduce_values",
        [](lang::Program &program, lang::Kernel *reduction,
           const std::vector<lang::aot::Arg> &reduction_args,
           lang::Kernel *producer, const std::vector<lang::aot::Arg> &producer_args,
           lang::Kernel *consumer, const std::vector<lang::aot::Arg> &consumer_args,
           int consumer_input) {
          auto tree_guard = program.acquire_snode_tree_lifecycle_read_guard();
          TI_ERROR_IF(reduction == nullptr || reduction->program != &program,
                      "Graph value fusion requires the active source program");
          return lang::compile_graph_segmented_reduce_values(
              program.compile_config(), {reduction, reduction_args},
              {producer, producer_args}, {consumer, consumer_args}, consumer_input);
        }, py::keep_alive<0, 1>(), py::arg("program"), py::arg("reduction"),
        py::arg("reduction_args"), py::arg("producer") = nullptr,
        py::arg("producer_args") = std::vector<lang::aot::Arg>{},
        py::arg("consumer") = nullptr,
        py::arg("consumer_args") = std::vector<lang::aot::Arg>{},
        py::arg("consumer_input") = -1);
  m.def("_graph_pointwise_value_program", [](lang::Kernel *kernel) {
    TI_ERROR_IF(kernel == nullptr, "Value-program query received a null kernel");
    auto tree_guard = kernel->program->acquire_snode_tree_lifecycle_read_guard();
    const auto program = lang::inspect_graph_pointwise_value_program(
        kernel->program->compile_config(), kernel);
    py::dict result;
    result["contract"] = "pointwise-integer32-preoffload";
    result["available"] = program.available;
    result["blocker"] = program.blocker;
    result["metadata"] = graph_kernel_metadata_to_python(program.metadata);
    result["output_argument"] = program.output_argument;
    result["output_runtime_affine"] = program.output_runtime_affine;
    result["result"] = program.result;
    py::list nodes;
    for (const auto &node : program.nodes) {
      py::dict item;
      using Kind = lang::GraphValueNode::Kind;
      const char *kind = "constant";
      switch (node.kind) {
        case Kind::constant: break;
        case Kind::index: kind = "index"; break;
        case Kind::scalar_argument: kind = "scalar_argument"; break;
        case Kind::array_load: kind = "array_load"; break;
        case Kind::unary: kind = "unary"; break;
        case Kind::binary: kind = "binary"; break;
      }
      item["kind"] = kind;
      item["dtype"] = node.dtype->to_string();
      item["operands"] = node.operands;
      item["argument"] = node.argument;
      item["constant_bits"] = node.constant_bits;
      item["runtime_affine"] = node.runtime_affine;
      item["operation"] = node.kind == Kind::unary ?
          unary_op_type_name(node.unary_op) : node.kind == Kind::binary ?
          binary_op_type_name(node.binary_op) : "";
      nodes.append(std::move(item));
    }
    result["nodes"] = std::move(nodes);
    return result;
  });
  m.def("_cuda_scan_capture_workspace_bytes", &lang::cuda_scan_capture_workspace_bytes);
  py::class_<lang::CudaCublasLtCapturePlan>(m, "_CudaCublasLtCapturePlan")
      .def(py::init<>())
      .def_static("supports_binding_frames", []() { return true; })
      .def_readwrite("matmul_address", &lang::CudaCublasLtCapturePlan::matmul_address)
      .def_readwrite("handle", &lang::CudaCublasLtCapturePlan::handle)
      .def_readwrite("descriptor", &lang::CudaCublasLtCapturePlan::descriptor)
      .def_readwrite("layouts", &lang::CudaCublasLtCapturePlan::layouts)
      .def_property("algorithm",
                    [](const lang::CudaCublasLtCapturePlan &plan) {
                      return py::bytes(plan.algorithm);
                    },
                    [](lang::CudaCublasLtCapturePlan &plan, py::bytes data) {
                      plan.algorithm = data;
                    })
      .def_readwrite("shapes", &lang::CudaCublasLtCapturePlan::shapes)
      .def_readwrite("workspace_bytes", &lang::CudaCublasLtCapturePlan::workspace_bytes)
      .def_readwrite("alpha", &lang::CudaCublasLtCapturePlan::alpha)
      .def_readwrite("beta", &lang::CudaCublasLtCapturePlan::beta);
  using namespace taichi::lang;
  using namespace std::placeholders;

  py::class_<CudaCutensorCapturePlan>(m, "_CudaCutensorCapturePlan")
      .def(py::init<>())
      .def_static("supports_binding_frames", []() { return true; })
      .def_readwrite("execute_address", &CudaCutensorCapturePlan::execute_address)
      .def_readwrite("handle", &CudaCutensorCapturePlan::handle)
      .def_readwrite("shapes", &CudaCutensorCapturePlan::shapes)
      .def_readwrite("workspace_bytes", &CudaCutensorCapturePlan::workspace_bytes)
      .def_readwrite("alignment_bytes", &CudaCutensorCapturePlan::alignment_bytes)
      .def_readwrite("output_alias_compatible",
                     &CudaCutensorCapturePlan::output_alias_compatible)
      .def_readwrite("alpha", &CudaCutensorCapturePlan::alpha)
      .def_readwrite("beta", &CudaCutensorCapturePlan::beta);

  py::class_<CudaCusparseLtCapturePlan>(m, "_CudaCusparseLtCapturePlan")
      .def(py::init<>())
      .def_static("supports_binding_frames", []() { return true; })
      .def_readwrite("compress_address", &CudaCusparseLtCapturePlan::compress_address)
      .def_readwrite("execute_address", &CudaCusparseLtCapturePlan::execute_address)
      .def_readwrite("handle", &CudaCusparseLtCapturePlan::handle)
      .def_readwrite("m", &CudaCusparseLtCapturePlan::m)
      .def_readwrite("n", &CudaCusparseLtCapturePlan::n)
      .def_readwrite("k", &CudaCusparseLtCapturePlan::k)
      .def_readwrite("matmul_count", &CudaCusparseLtCapturePlan::matmul_count)
      .def_readwrite("compressed_bytes", &CudaCusparseLtCapturePlan::compressed_bytes)
      .def_readwrite("compression_buffer_bytes",
                     &CudaCusparseLtCapturePlan::compression_buffer_bytes)
      .def_readwrite("workspace_bytes", &CudaCusparseLtCapturePlan::workspace_bytes)
      .def_readwrite("alignment_bytes", &CudaCusparseLtCapturePlan::alignment_bytes)
      .def_readwrite("recompress", &CudaCusparseLtCapturePlan::recompress)
      .def_readwrite("alpha", &CudaCusparseLtCapturePlan::alpha)
      .def_readwrite("beta", &CudaCusparseLtCapturePlan::beta);

  py::register_exception<TaichiTypeError>(m, "TaichiTypeError",
                                          PyExc_TypeError);
  py::register_exception<TaichiSyntaxError>(m, "TaichiSyntaxError",
                                            PyExc_SyntaxError);
  py::register_exception<TaichiIndexError>(m, "TaichiIndexError",
                                           PyExc_IndexError);
  py::register_exception<TaichiRuntimeError>(m, "TaichiRuntimeError",
                                             PyExc_RuntimeError);
  py::register_exception<TaichiAssertionError>(m, "TaichiAssertionError",
                                               PyExc_AssertionError);
  py::enum_<Arch>(m, "Arch", py::arithmetic())
#define PER_ARCH(x) .value(#x, Arch::x)
#include "taichi/inc/archs.inc.h"
#undef PER_ARCH
      .export_values();

  m.def("arch_name", arch_name);
  m.def("arch_from_name", arch_from_name);

  py::enum_<SNodeType>(m, "SNodeType", py::arithmetic())
#define PER_SNODE(x) .value(#x, SNodeType::x)
#include "taichi/inc/snodes.inc.h"
#undef PER_SNODE
      .export_values();

  py::enum_<Extension>(m, "Extension", py::arithmetic())
#define PER_EXTENSION(x) .value(#x, Extension::x)
#include "taichi/inc/extensions.inc.h"
#undef PER_EXTENSION
      .export_values();

  py::enum_<ExternalArrayLayout>(m, "Layout", py::arithmetic())
      .value("AOS", ExternalArrayLayout::kAOS)
      .value("SOA", ExternalArrayLayout::kSOA)
      .value("NULL", ExternalArrayLayout::kNull)
      .export_values();

  py::enum_<AutodiffMode>(m, "AutodiffMode", py::arithmetic())
      .value("NONE", AutodiffMode::kNone)
      .value("VALIDATION", AutodiffMode::kCheckAutodiffValid)
      .value("FORWARD", AutodiffMode::kForward)
      .value("REVERSE", AutodiffMode::kReverse)
      .export_values();

  py::enum_<SNodeGradType>(m, "SNodeGradType", py::arithmetic())
      .value("PRIMAL", SNodeGradType::kPrimal)
      .value("ADJOINT", SNodeGradType::kAdjoint)
      .value("DUAL", SNodeGradType::kDual)
      .value("ADJOINT_CHECKBIT", SNodeGradType::kAdjointCheckbit)
      .export_values();

  py::enum_<BoundaryMode>(m, "BoundaryMode", py::arithmetic())
      .value("UNSAFE", BoundaryMode::kUnsafe)
      .value("CLAMP", BoundaryMode::kClamp)
      .export_values();

  // TODO(type): This should be removed
  py::class_<DataType>(m, "DataType")
      .def(py::init<Type *>())
      .def(py::self == py::self)
      .def("__hash__", &DataType::hash)
      .def("to_string", &DataType::to_string)
      .def("__str__", &DataType::to_string)
      .def("shape", &DataType::get_shape)
      .def("element_type", &DataType::get_element_type)
      .def("ptr_removed", &DataType::ptr_removed)
      .def(
          "get_ptr", [](DataType *dtype) -> Type * { return *dtype; },
          py::return_value_policy::reference)
      .def("__call__",
           [](DataType *dtype, py::args args, const py::kwargs &kwargs) {
             // Defining __call__ here to make DataType callable in Python,
             // which enables us to write `typing.Tuple[ti.i32, ti.i32]`.
             throw TaichiSyntaxError(
                 "Taichi data types cannot be called outside Taichi kernels.");
           })
      .def(py::pickle(
          [](const DataType &dt) {
            // Note: this only works for primitive types, which is fine for now.
            auto primitive =
                dynamic_cast<const PrimitiveType *>((const Type *)dt);
            TI_ASSERT(primitive);
            return py::make_tuple((std::size_t)primitive->type);
          },
          [](py::tuple t) {
            if (t.size() != 1)
              throw std::runtime_error("Invalid state!");

            DataType dt =
                PrimitiveType::get((PrimitiveTypeID)(t[0].cast<std::size_t>()));

            return dt;
          }));

  py::class_<DebugInfo>(m, "DebugInfo")
      .def(py::init<>())
      .def(py::init<std::string>())
      .def(py::init<>())
      .def_readwrite("tb", &DebugInfo::tb)
      .def_readwrite("src_loc", &DebugInfo::src_loc);

  py::class_<CompileConfig>(m, "CompileConfig")
      .def(py::init<>())
      .def_readwrite("arch", &CompileConfig::arch)
      .def_readwrite("opt_level", &CompileConfig::opt_level)
      .def_readwrite("llvm_opt_level", &CompileConfig::llvm_opt_level)
      .def_readwrite("compile_tier", &CompileConfig::compile_tier)
      .def_readwrite("spirv_parallel_codegen",
                     &CompileConfig::spirv_parallel_codegen)
      .def_readwrite("spirv_skip_loop_unroll",
                     &CompileConfig::spirv_skip_loop_unroll)
      .def_readwrite("vulkan_sparse_experimental",
                     &CompileConfig::vulkan_sparse_experimental)
      .def_readwrite("vulkan_quant_experimental",
                     &CompileConfig::vulkan_quant_experimental)
      .def_readwrite("hash_snode_experimental",
                     &CompileConfig::hash_snode_experimental)
      .def_readwrite("hash_snode_default_load_factor",
                     &CompileConfig::hash_snode_default_load_factor)
      .def_readwrite("hash_snode_active_list",
                     &CompileConfig::hash_snode_active_list)
      .def_readwrite("hash_snode_diagnostics",
                     &CompileConfig::hash_snode_diagnostics)
      .def_readwrite("hash_snode_compact_child_pool",
                     &CompileConfig::hash_snode_compact_child_pool)
      .def_readwrite("spirv_disabled_passes",
                     &CompileConfig::spirv_disabled_passes)
      .def_readwrite("spirv_adaptive_opt",
            &CompileConfig::spirv_adaptive_opt)
      .def_readwrite("spirv_adaptive_opt_threshold",
            &CompileConfig::spirv_adaptive_opt_threshold)
      .def_readwrite("compile_dag_scheduler",
                     &CompileConfig::compile_dag_scheduler)
      .def_readwrite("vulkan_pointer_freelist",
                     &CompileConfig::vulkan_pointer_freelist)
      .def_readwrite("vulkan_pointer_ambient_zone",
                     &CompileConfig::vulkan_pointer_ambient_zone)
      .def_readwrite("vulkan_pointer_cas_marker",
                     &CompileConfig::vulkan_pointer_cas_marker)
      .def_readwrite("vulkan_pointer_pool_fraction",
                     &CompileConfig::vulkan_pointer_pool_fraction)
      .def_readwrite("vulkan_pointer_independent_pool",
                     &CompileConfig::vulkan_pointer_independent_pool)
      .def_readwrite("vulkan_pointer_allocator_kind",
                     &CompileConfig::vulkan_pointer_allocator_kind)
      .def_readwrite("vulkan_pointer_max_chunks",
                     &CompileConfig::vulkan_pointer_max_chunks)
      .def_readwrite("vulkan_pointer_deterministic_slot",
                     &CompileConfig::vulkan_pointer_deterministic_slot)
      .def_readwrite("cuda_pointer_deterministic_slot",
                     &CompileConfig::cuda_pointer_deterministic_slot)
      .def_readwrite("cuda_pointer_fast_reset",
                     &CompileConfig::cuda_pointer_fast_reset)
      .def_readwrite("cuda_listgen_reuse",
            &CompileConfig::cuda_listgen_reuse)
      .def_readwrite("cuda_listgen_reuse_adaptive",
             &CompileConfig::cuda_listgen_reuse_adaptive)
      .def_readwrite("bitmasked_clear_data_on_deactivate",
                     &CompileConfig::bitmasked_clear_data_on_deactivate)
      .def_readwrite("print_ir", &CompileConfig::print_ir)
      .def_readwrite("print_preprocessed_ir",
                     &CompileConfig::print_preprocessed_ir)
      .def_readwrite("print_ir_dbg_info", &CompileConfig::print_ir_dbg_info)
      .def_readwrite("debug", &CompileConfig::debug)
      .def_readwrite("cfg_optimization", &CompileConfig::cfg_optimization)
      .def_readwrite("check_out_of_bound", &CompileConfig::check_out_of_bound)
      .def_readwrite("print_accessor_ir", &CompileConfig::print_accessor_ir)
      .def_readwrite("use_llvm", &CompileConfig::use_llvm)
      .def_readwrite("print_struct_llvm_ir",
                     &CompileConfig::print_struct_llvm_ir)
      .def_readwrite("print_kernel_llvm_ir",
                     &CompileConfig::print_kernel_llvm_ir)
      .def_readwrite("print_kernel_llvm_ir_optimized",
                     &CompileConfig::print_kernel_llvm_ir_optimized)
      .def_readwrite("print_kernel_asm", &CompileConfig::print_kernel_asm)
      .def_readwrite("print_kernel_amdgcn", &CompileConfig::print_kernel_amdgcn)
      .def_readwrite("simplify_before_lower_access",
                     &CompileConfig::simplify_before_lower_access)
      .def_readwrite("simplify_after_lower_access",
                     &CompileConfig::simplify_after_lower_access)
      .def_readwrite("lower_access", &CompileConfig::lower_access)
      .def_readwrite("move_loop_invariant_outside_if",
                     &CompileConfig::move_loop_invariant_outside_if)
      .def_readwrite("cache_loop_invariant_global_vars",
                     &CompileConfig::cache_loop_invariant_global_vars)
      .def_readwrite("tiered_full_simplify",
                     &CompileConfig::tiered_full_simplify)
      .def_readwrite("full_simplify_global_iter_cap",
                     &CompileConfig::full_simplify_global_iter_cap)
      .def_readwrite("auto_real_function",
                     &CompileConfig::auto_real_function)
      .def_readwrite("auto_real_function_threshold_us",
                     &CompileConfig::auto_real_function_threshold_us)
      .def_readwrite("auto_real_function_inline_budget",
                     &CompileConfig::auto_real_function_inline_budget)
      .def_readwrite("default_cpu_block_dim",
                     &CompileConfig::default_cpu_block_dim)
      .def_readwrite("cpu_block_dim_adaptive",
                     &CompileConfig::cpu_block_dim_adaptive)
      .def_readwrite("default_gpu_block_dim",
                     &CompileConfig::default_gpu_block_dim)
      .def_readwrite("gpu_max_reg", &CompileConfig::gpu_max_reg)
      .def_readwrite("saturating_grid_dim", &CompileConfig::saturating_grid_dim)
      .def_readwrite("max_block_dim", &CompileConfig::max_block_dim)
      .def_readwrite("cpu_max_num_threads", &CompileConfig::cpu_max_num_threads)
      .def_readwrite("random_seed", &CompileConfig::random_seed)
      .def_readwrite("verbose_kernel_launches",
                     &CompileConfig::verbose_kernel_launches)
      .def_readwrite("verbose", &CompileConfig::verbose)
      .def_readwrite("demote_dense_struct_fors",
                     &CompileConfig::demote_dense_struct_fors)
      .def_readwrite("spirv_skip_intermediate_listgen",
                     &CompileConfig::spirv_skip_intermediate_listgen)
      .def_readwrite("spirv_listgen_subgroup_ballot",
                     &CompileConfig::spirv_listgen_subgroup_ballot)
      .def_readwrite("listgen_static_grid_dim",
                     &CompileConfig::listgen_static_grid_dim)
      .def_readwrite("vulkan_listgen_dynamic_size",
                     &CompileConfig::vulkan_listgen_dynamic_size)
      .def_readwrite("vulkan_listgen_buffer_MB",
                     &CompileConfig::vulkan_listgen_buffer_MB)
      .def_readwrite("vulkan_dispatch_cache",
                     &CompileConfig::vulkan_dispatch_cache)
      .def_readwrite("vulkan_dispatch_cache_size",
             &CompileConfig::vulkan_dispatch_cache_size)
      .def_readwrite("vulkan_descriptor_cache_lru",
             &CompileConfig::vulkan_descriptor_cache_lru)
      .def_readwrite("vulkan_listgen_lite_barrier",
             &CompileConfig::vulkan_listgen_lite_barrier)
      .def_readwrite("vulkan_listgen_reuse",
            &CompileConfig::vulkan_listgen_reuse)
      .def_readwrite("vulkan_listgen_reuse_adaptive",
             &CompileConfig::vulkan_listgen_reuse_adaptive)
      .def_readwrite("vulkan_spv_stats", &CompileConfig::vulkan_spv_stats)
      .def_readwrite("vulkan_spv_stats_filter",
            &CompileConfig::vulkan_spv_stats_filter)
      .def_readwrite("vulkan_spv_stats_capacity",
            &CompileConfig::vulkan_spv_stats_capacity)
      .def_readwrite("vulkan_spv_stats_to_stderr",
            &CompileConfig::vulkan_spv_stats_to_stderr)
      .def_readwrite("kernel_profiler", &CompileConfig::kernel_profiler)
      .def_readwrite("timeline", &CompileConfig::timeline)
      .def_readwrite("default_fp", &CompileConfig::default_fp)
      .def_readwrite("default_ip", &CompileConfig::default_ip)
      .def_readwrite("default_up", &CompileConfig::default_up)
      .def_readwrite("device_memory_GB", &CompileConfig::device_memory_GB)
      .def_readwrite("cuda_sparse_pool_size_GB",
                     &CompileConfig::cuda_sparse_pool_size_GB)
      .def_readwrite("cuda_sparse_pool_size_floor_MiB",
                     &CompileConfig::cuda_sparse_pool_size_floor_MiB)
      .def_readwrite("cuda_sparse_pool_auto_size",
                     &CompileConfig::cuda_sparse_pool_auto_size)
      .def_readwrite("cuda_sparse_per_snode_pool",
                     &CompileConfig::cuda_sparse_per_snode_pool)
      .def_readwrite("device_memory_fraction",
                     &CompileConfig::device_memory_fraction)
      .def_readwrite("fast_math", &CompileConfig::fast_math)
      .def_readwrite("advanced_optimization",
                     &CompileConfig::advanced_optimization)
      .def_readwrite("ad_stack_size", &CompileConfig::ad_stack_size)
      .def_readwrite("flatten_if", &CompileConfig::flatten_if)
      .def_readwrite("make_thread_local", &CompileConfig::make_thread_local)
      .def_readwrite("make_block_local", &CompileConfig::make_block_local)
      .def_readwrite("detect_read_only", &CompileConfig::detect_read_only)
      .def_readwrite("real_matrix_scalarize",
                     &CompileConfig::real_matrix_scalarize)
      .def_readwrite("force_scalarize_matrix",
                     &CompileConfig::force_scalarize_matrix)
      .def_readwrite("half2_vectorization", &CompileConfig::half2_vectorization)
      .def_readwrite("make_cpu_multithreading_loop",
                     &CompileConfig::make_cpu_multithreading_loop)
      .def_readwrite("quant_opt_store_fusion",
                     &CompileConfig::quant_opt_store_fusion)
      .def_readwrite("quant_opt_atomic_demotion",
                     &CompileConfig::quant_opt_atomic_demotion)
      .def_readwrite("allow_nv_shader_extension",
                     &CompileConfig::allow_nv_shader_extension)
      .def_readwrite("make_mesh_block_local",
                     &CompileConfig::make_mesh_block_local)
      .def_readwrite("mesh_localize_to_end_mapping",
                     &CompileConfig::mesh_localize_to_end_mapping)
      .def_readwrite("mesh_localize_from_end_mapping",
                     &CompileConfig::mesh_localize_from_end_mapping)
      .def_readwrite("optimize_mesh_reordered_mapping",
                     &CompileConfig::optimize_mesh_reordered_mapping)
      .def_readwrite("mesh_localize_all_attr_mappings",
                     &CompileConfig::mesh_localize_all_attr_mappings)
      .def_readwrite("demote_no_access_mesh_fors",
                     &CompileConfig::demote_no_access_mesh_fors)
      .def_readwrite("experimental_auto_mesh_local",
                     &CompileConfig::experimental_auto_mesh_local)
      .def_readwrite("auto_mesh_local_default_occupacy",
                     &CompileConfig::auto_mesh_local_default_occupacy)
      .def_readwrite("offline_cache", &CompileConfig::offline_cache)
      .def_readwrite("offline_cache_file_path",
                     &CompileConfig::offline_cache_file_path)
      .def_readwrite("offline_cache_cleaning_policy",
                     &CompileConfig::offline_cache_cleaning_policy)
      .def_readwrite("offline_cache_max_size_of_files",
                     &CompileConfig::offline_cache_max_size_of_files)
      .def_readwrite("offline_cache_cleaning_factor",
                     &CompileConfig::offline_cache_cleaning_factor)
      .def_readwrite("num_compile_threads", &CompileConfig::num_compile_threads)
      .def_readwrite("vk_api_version", &CompileConfig::vk_api_version)
      .def_readwrite("vulkan_launch_buffer_pool",
                     &CompileConfig::vulkan_launch_buffer_pool)
      .def_readwrite("vulkan_launch_buffer_pool_capacity",
                     &CompileConfig::vulkan_launch_buffer_pool_capacity)
      .def_readwrite("gfx_ctx_buffer_ring",
            &CompileConfig::gfx_ctx_buffer_ring)
      .def_readwrite("gfx_ctx_buffer_ring_size",
            &CompileConfig::gfx_ctx_buffer_ring_size)
      .def_readwrite("gfx_cmdlist_lazy_submit",
                     &CompileConfig::gfx_cmdlist_lazy_submit)
      .def_readwrite("gfx_cmdlist_max_dispatches",
                     &CompileConfig::gfx_cmdlist_max_dispatches)
      .def_readwrite("cuda_stack_limit", &CompileConfig::cuda_stack_limit)
      .def_readwrite("external_optimization_level",
                     &CompileConfig::external_optimization_level);

  m.def("reset_default_compile_config",
        [&]() { default_compile_config = CompileConfig(); });
  m.def("_set_check_out_of_bound_explicit", [](bool explicit_value) {
    default_compile_config.check_out_of_bound_explicit = explicit_value;
  });

  m.def(
      "default_compile_config",
      [&]() -> CompileConfig & { return default_compile_config; },
      py::return_value_policy::reference);

  m.def("get_last_vulkan_spv_stats", []() {
    py::list ret;
    for (const auto &s : spirv::get_last_spv_stats()) {
      py::dict item;
      item["kernel"] = s.kernel_name;
      item["task_id"] = s.task_id;
      item["task_name"] = s.task_name;
      item["type"] = s.task_type;
      item["snode_id"] = s.snode_id;
      item["word_before"] = py::int_(s.before_words);
      item["word_after"] = py::int_(s.after_words);
      item["ray_query_initialize_before"] =
          py::int_(s.ray_query_initialize_before);
      item["ray_query_initialize_after"] =
          py::int_(s.ray_query_initialize_after);
      item["ray_query_getter_before"] =
          py::int_(s.ray_query_getter_before);
      item["ray_query_getter_after"] = py::int_(s.ray_query_getter_after);
      item["cooperative_matrix_load_before"] =
          py::int_(s.cooperative_matrix_load_before);
      item["cooperative_matrix_load_after"] =
          py::int_(s.cooperative_matrix_load_after);
      item["cooperative_matrix_mul_add_before"] =
          py::int_(s.cooperative_matrix_mul_add_before);
      item["cooperative_matrix_mul_add_after"] =
          py::int_(s.cooperative_matrix_mul_add_after);
      item["cooperative_matrix_store_before"] =
          py::int_(s.cooperative_matrix_store_before);
      item["cooperative_matrix_store_after"] =
          py::int_(s.cooperative_matrix_store_after);
      item["function_variable_before"] =
          py::int_(s.function_variable_before);
      item["function_variable_after"] =
          py::int_(s.function_variable_after);
      item["phi_before"] = py::int_(s.phi_before);
      item["phi_after"] = py::int_(s.phi_after);
      item["opt_run"] = s.opt_run;
      item["opt_ok"] = s.opt_ok;
      item["duration_us"] = s.opt_us;
      item["is_listgen"] = s.listgen_related;
      item["is_pointer"] = s.pointer_related;
      item["skipped_passes"] = s.skipped_passes;
      ret.append(item);
    }
    return ret;
  });

  m.def("get_fs_inner_stats", []() {
    uint64_t e = 0, n = 0, it = 0;
    taichi::lang::irpass::get_fs_inner_stats(&e, &n, &it);
    return py::make_tuple(e, n, it);
  });
  m.def("reset_fs_inner_stats",
        &taichi::lang::irpass::reset_fs_inner_stats);

  py::class_<Program::KernelProfilerQueryResult>(m, "KernelProfilerQueryResult")
      .def_readwrite("counter", &Program::KernelProfilerQueryResult::counter)
      .def_readwrite("min", &Program::KernelProfilerQueryResult::min)
      .def_readwrite("max", &Program::KernelProfilerQueryResult::max)
      .def_readwrite("avg", &Program::KernelProfilerQueryResult::avg);

  py::class_<KernelProfileTracedRecord>(m, "KernelProfileTracedRecord")
      .def_readwrite("register_per_thread",
                     &KernelProfileTracedRecord::register_per_thread)
      .def_readwrite("shared_mem_per_block",
                     &KernelProfileTracedRecord::shared_mem_per_block)
      .def_readwrite("grid_size", &KernelProfileTracedRecord::grid_size)
      .def_readwrite("block_size", &KernelProfileTracedRecord::block_size)
      .def_readwrite(
          "active_blocks_per_multiprocessor",
          &KernelProfileTracedRecord::active_blocks_per_multiprocessor)
      .def_readwrite("kernel_time",
                     &KernelProfileTracedRecord::kernel_elapsed_time_in_ms)
      .def_readwrite("base_time", &KernelProfileTracedRecord::time_since_base)
      .def_readwrite("name", &KernelProfileTracedRecord::name)
      .def_readwrite("metric_values",
                     &KernelProfileTracedRecord::metric_values);

  py::enum_<SNodeAccessFlag>(m, "SNodeAccessFlag", py::arithmetic())
      .value("block_local", SNodeAccessFlag::block_local)
      .value("read_only", SNodeAccessFlag::read_only)
      .value("mesh_local", SNodeAccessFlag::mesh_local)
      .export_values();

  // Export ASTBuilder
  py::class_<ASTBuilder>(m, "ASTBuilder")
      .def("make_id_expr", &ASTBuilder::make_id_expr)
      .def("create_kernel_exprgroup_return",
           &ASTBuilder::create_kernel_exprgroup_return)
      .def("create_print", &ASTBuilder::create_print)
      .def("begin_func", &ASTBuilder::begin_func)
      .def("end_func", &ASTBuilder::end_func)
      .def("stop_grad", &ASTBuilder::stop_gradient)
      .def("begin_frontend_if", &ASTBuilder::begin_frontend_if)
      .def("begin_frontend_if_true", &ASTBuilder::begin_frontend_if_true)
      .def("pop_scope", &ASTBuilder::pop_scope)
      .def("begin_frontend_if_false", &ASTBuilder::begin_frontend_if_false)
      .def("insert_deactivate", &ASTBuilder::insert_snode_deactivate)
      .def("insert_activate", &ASTBuilder::insert_snode_activate)
      .def("expr_snode_get_addr", &ASTBuilder::snode_get_addr)
      .def("expr_snode_append", &ASTBuilder::snode_append)
      .def("expr_snode_is_active", &ASTBuilder::snode_is_active)
      .def("expr_snode_length", &ASTBuilder::snode_length)
      .def("insert_external_func_call", &ASTBuilder::insert_external_func_call)
      .def("make_matrix_expr", &ASTBuilder::make_matrix_expr)
      .def("expr_alloca", &ASTBuilder::expr_alloca)
      .def("expr_alloca_shared_array", &ASTBuilder::expr_alloca_shared_array)
      .def("create_assert_stmt", &ASTBuilder::create_assert_stmt)
      .def("expr_assign", &ASTBuilder::expr_assign)
      .def("begin_frontend_range_for", &ASTBuilder::begin_frontend_range_for)
      .def("end_frontend_range_for", &ASTBuilder::pop_scope)
      .def("begin_frontend_struct_for_on_snode",
           &ASTBuilder::begin_frontend_struct_for_on_snode)
      .def("begin_frontend_struct_for_on_external_tensor",
           &ASTBuilder::begin_frontend_struct_for_on_external_tensor)
      .def("end_frontend_struct_for", &ASTBuilder::pop_scope)
      .def("begin_frontend_mesh_for", &ASTBuilder::begin_frontend_mesh_for)
      .def("end_frontend_mesh_for", &ASTBuilder::pop_scope)
      .def("begin_frontend_while", &ASTBuilder::begin_frontend_while)
      .def("insert_break_stmt", &ASTBuilder::insert_break_stmt)
      .def("insert_continue_stmt", &ASTBuilder::insert_continue_stmt)
      .def("insert_expr_stmt", &ASTBuilder::insert_expr_stmt)
      .def("insert_thread_idx_expr", &ASTBuilder::insert_thread_idx_expr)
      .def("insert_patch_idx_expr", &ASTBuilder::insert_patch_idx_expr)
      .def("make_texture_op_expr", &ASTBuilder::make_texture_op_expr)
      .def("expand_exprs", &ASTBuilder::expand_exprs)
      .def("mesh_index_conversion", &ASTBuilder::mesh_index_conversion)
      .def("expr_subscript", &ASTBuilder::expr_subscript)
      .def("insert_func_call", &ASTBuilder::insert_func_call)
      .def("sifakis_svd_f32", sifakis_svd_export<float32, int32>)
      .def("sifakis_svd_f64", sifakis_svd_export<float64, int64>)
      .def("expr_var", &ASTBuilder::make_var)
      .def("bit_vectorize", &ASTBuilder::bit_vectorize)
      .def("parallelize", &ASTBuilder::parallelize)
      .def("strictly_serialize", &ASTBuilder::strictly_serialize)
      .def("block_dim", &ASTBuilder::block_dim)
      .def("one_to_one", &ASTBuilder::one_to_one)
      .def("insert_snode_access_flag", &ASTBuilder::insert_snode_access_flag)
      .def("reset_snode_access_flag", &ASTBuilder::reset_snode_access_flag);

  py::class_<DeviceCapabilityConfig>(
      m, "DeviceCapabilityConfig");  // NOLINT(bugprone-unused-raii)

  py::class_<CompiledKernelData>(
      m, "CompiledKernelData");  // NOLINT(bugprone-unused-raii)

  m.def("_push_dispatch_label", &push_dispatch_label);
  m.def("_restore_dispatch_label", &restore_dispatch_label);
  m.def("_push_external_profiler_range", &push_external_profiler_range);
  m.def("_pop_external_profiler_range", &pop_external_profiler_range);

  // Internal F2 validation surface. The public API remains Graph.run() -> None
  // until CPU/CUDA/Vulkan Ticket semantics and reset/fault lifetime complete
  // the F2.3 acceptance gate.
  py::class_<RuntimeCompletion>(m, "_RuntimeCompletion")
      .def("done", &RuntimeCompletion::done,
           py::call_guard<py::gil_scoped_release>())
      .def("wait", &RuntimeCompletion::wait,
           py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("backend", [](const RuntimeCompletion &value) {
        return arch_name(value.backend());
      })
      .def_property_readonly("program_domain",
                             &RuntimeCompletion::program_domain)
      .def_property_readonly("sequence", &RuntimeCompletion::sequence)
      .def_property_readonly("has_backend_work",
                             &RuntimeCompletion::has_backend_work)
      .def_property_readonly("first_error",
                             &RuntimeCompletion::first_error_message)
      .def("_gpu_timing", [](const RuntimeCompletion &value) {
        const auto snapshot = value.gpu_timing_snapshot();
        py::dict result;
        result["available"] = snapshot.available;
        result["duration_ns"] = snapshot.duration_ns;
        result["exact"] = snapshot.exact;
        result["measurement_path_changed"] =
            snapshot.measurement_path_changed;
        result["stream_id"] = snapshot.stream_id;
        result["driver_owned_bytes"] = snapshot.driver_owned_bytes;
        result["driver_owned_bytes_known"] =
            snapshot.driver_owned_bytes_known;
        result["status"] = snapshot.status;
        return result;
      })
      .def("_gpu_region_timings", [](const RuntimeCompletion &value) {
        py::list results;
        for (const auto &region : value.gpu_region_timing_snapshots()) {
          py::dict result;
          result["path_id"] = region.path_id;
          result["available"] = region.timing.available;
          result["duration_ns"] = region.timing.duration_ns;
          result["exact"] = region.timing.exact;
          result["measurement_path_changed"] =
              region.timing.measurement_path_changed;
          result["stream_id"] = region.timing.stream_id;
          result["driver_owned_bytes"] = region.timing.driver_owned_bytes;
          result["driver_owned_bytes_known"] =
              region.timing.driver_owned_bytes_known;
          result["status"] = region.timing.status;
          results.append(std::move(result));
        }
        return results;
      });

  py::class_<Program::RuntimeSubmissionTransaction>(
      m, "_RuntimeSubmissionTransaction")
      .def("_mark_submission",
           &Program::RuntimeSubmissionTransaction::mark_submission)
      .def("_begin_gpu_region_timing",
           &Program::RuntimeSubmissionTransaction::begin_gpu_region_timing)
      .def("_end_gpu_region_timing",
           &Program::RuntimeSubmissionTransaction::end_gpu_region_timing)
      .def("_submission_statistics",
           [](const Program::RuntimeSubmissionTransaction &transaction) {
        const RuntimeSubmissionStatistics snapshot =
            transaction.submission_statistics();
        py::dict result;
        result["kernel_submissions"] = snapshot.kernel_submissions;
        result["graph_submissions"] = snapshot.graph_submissions;
        result["graph_backend_submissions"] =
            snapshot.graph_backend_submissions;
        result["backend_graph_launches"] =
            snapshot.backend_graph_launches;
        result["native_submissions"] = snapshot.native_submissions;
        return result;
      })
      .def("_finish", &Program::RuntimeSubmissionTransaction::finish,
           py::call_guard<py::gil_scoped_release>());

  py::class_<Program::ExternalCudaSubmissionScope>(
      m, "_ExternalCudaSubmissionScope")
      .def("_commit",
           [](Program::ExternalCudaSubmissionScope &scope,
              const std::vector<Ndarray *> &ndarrays, bool failed) {
             std::vector<const Ndarray *> views(ndarrays.begin(),
                                                ndarrays.end());
             scope.commit(views, failed);
           },
           py::arg("ndarrays"), py::arg("failed") = false);

  py::class_<PythonScalarArgumentPatchPlan>(m, "_ScalarArgumentPatchPlan")
      .def(py::init<std::vector<int>, std::string>(), py::arg("indices"),
           py::arg("kinds"))
      .def("apply", &PythonScalarArgumentPatchPlan::apply, py::arg("context"),
           py::arg("args"));

  py::class_<Program::RegisteredKernelExecutionPlanLaunchScope>(
      m, "_RegisteredKernelExecutionPlanLaunchScope")
      .def("launch",
           &Program::RegisteredKernelExecutionPlanLaunchScope::launch,
           py::arg("context"), py::call_guard<py::gil_scoped_release>());

  py::class_<Program::RegisteredKernelExecutionPlan>(
      m, "_RegisteredKernelExecutionPlan")
      .def(
          "launch", &Program::RegisteredKernelExecutionPlan::launch,
          py::arg("program"), py::arg("context"),
          py::call_guard<py::gil_scoped_release>())
      .def("_begin_launch",
           &Program::RegisteredKernelExecutionPlan::begin_launch,
           py::arg("program"), py::call_guard<py::gil_scoped_release>())
      .def_property_readonly(
          "has_snode_tree_dependencies",
          &Program::RegisteredKernelExecutionPlan::
              has_snode_tree_dependencies);

  py::class_<VulkanRayInstanceInfo>(m, "VulkanRayInstanceInfo")
      .def(py::init<>())
      .def_readwrite("transform", &VulkanRayInstanceInfo::transform)
      .def_readwrite("mask", &VulkanRayInstanceInfo::mask)
      .def_readwrite("custom_index", &VulkanRayInstanceInfo::custom_index);

  py::class_<VulkanRayQueryCommand>(m, "_VulkanRayQueryCommand");
  py::class_<PreparedVulkanBufferCommands>(m, "_PreparedVulkanBufferCommands");

  py::class_<Program>(m, "Program")
      .def(py::init<>())
      .def("config", &Program::compile_config,
           py::return_value_policy::reference)
      .def("sync_kernel_profiler",
           [](Program *program) { program->profiler->sync(); })
      .def("update_kernel_profiler",
           [](Program *program) { program->profiler->update(); })
      .def("clear_kernel_profiler",
           [](Program *program) { program->profiler->clear(); })
      .def("query_kernel_profile_info",
           [](Program *program, const std::string &name) {
             return program->query_kernel_profile_info(name);
           })
      .def("get_kernel_profiler_records",
           [](Program *program) {
             return program->profiler->get_traced_records();
           })
      .def(
          "get_kernel_profiler_device_name",
          [](Program *program) { return program->profiler->get_device_name(); })
      .def("reinit_kernel_profiler_with_metrics",
           [](Program *program, const std::vector<std::string> metrics) {
             return program->profiler->reinit_with_metrics(metrics);
           })
      .def("kernel_profiler_total_time",
           [](Program *program) { return program->profiler->get_total_time(); })
      .def("_kernel_cache_key_no_compile",
           [](Program &program, Kernel *kernel) {
             TI_ERROR_IF(kernel == nullptr,
                         "Kernel cache-key query received a null kernel");
             auto tree_guard =
                 program.acquire_snode_tree_lifecycle_read_guard();
             auto key = kernel->get_cached_kernel_key();
             if (key.empty()) {
               key = get_hashed_offline_cache_key(
                   program.compile_config(), program.get_device_caps(),
                   kernel);
               kernel->set_kernel_key_for_cache(key);
             }
             return key;
           })
      .def("_kernel_task_manifest",
           [](Program &program, Kernel *kernel) {
             TI_ERROR_IF(kernel == nullptr,
                         "Task manifest query received a null kernel");
             auto tree_guard =
                 program.acquire_snode_tree_lifecycle_read_guard();
             const auto &compiled = program.compile_kernel(
                 program.compile_config(), program.get_device_caps(),
                 *kernel);
             py::list result;
             for (const auto &task : compiled.task_manifest()) {
               result.append(offloaded_task_manifest_to_python(
                   overlay_execution_plan_launch_requests(task, kernel)));
             }
             return result;
           })
      .def("_kernel_gpu_semantics_snapshot",
           [](Program &program, Kernel *kernel) {
             TI_ERROR_IF(kernel == nullptr,
                         "GPU semantics query received a null kernel");
             const auto backend = program.compile_config().arch;
             TI_ERROR_IF(backend != Arch::cuda && backend != Arch::vulkan,
                         "GPU semantics are supported only on CUDA and Vulkan, "
                         "not {}",
                         arch_name(backend));
             auto tree_guard =
                 program.acquire_snode_tree_lifecycle_read_guard();
             const auto &compiled = program.compile_kernel(
                 program.compile_config(), program.get_device_caps(),
                 *kernel);
             return compiled_kernel_gpu_semantics_to_python(
                 compiled, kernel->autodiff_mode, kernel);
           })
      .def("_kernel_gpu_artifact_qualification",
           [](Program &program, Kernel *kernel) {
             TI_ERROR_IF(kernel == nullptr,
                         "GPU artifact qualification received a null kernel");
             const auto backend = program.compile_config().arch;
             TI_ERROR_IF(backend != Arch::cuda && backend != Arch::vulkan,
                         "GPU artifact qualification is supported only on "
                         "CUDA and Vulkan, not {}",
                         arch_name(backend));
             auto tree_guard =
                 program.acquire_snode_tree_lifecycle_read_guard();
             const auto &compiled = program.compile_kernel(
                 program.compile_config(), program.get_device_caps(),
                 *kernel);
             py::dict result;
             result["backend"] = arch_name(backend);
             result["registered_before"] = compiled.get_handle().has_value();
             py::list tasks;
#if defined(TI_WITH_CUDA)
             if (backend == Arch::cuda) {
               const auto *llvm_compiled =
                   dynamic_cast<const LLVM::CompiledKernelData *>(&compiled);
               TI_ERROR_IF(
                   llvm_compiled == nullptr,
                   "CUDA artifact qualification requires LLVM compiled data");
               auto *launcher = dynamic_cast<cuda::KernelLauncher *>(
                   &program.get_kernel_launcher());
               TI_ERROR_IF(launcher == nullptr,
                           "CUDA artifact qualification requires the CUDA "
                           "kernel launcher");
               for (const auto &item :
                    launcher->qualify_llvm_kernel_artifacts(*llvm_compiled)) {
                 py::dict encoded;
                 encoded["entry_point"] = item.entry_point;
                 encoded["function_identity"] =
                     py::int_(item.function_identity);
                 encoded["max_threads_per_block"] =
                     item.max_threads_per_block;
                 encoded["static_shared_memory_bytes"] =
                     item.static_shared_memory_bytes;
                 encoded["constant_memory_bytes"] =
                     item.constant_memory_bytes;
                 encoded["local_memory_bytes_per_thread"] =
                     item.local_memory_bytes_per_thread;
                 encoded["registers_per_thread"] =
                     item.registers_per_thread;
                 encoded["ptx_version"] = item.ptx_version;
                 encoded["binary_version"] = item.binary_version;
                 encoded["cache_mode_ca"] = item.cache_mode_ca;
                 encoded["max_dynamic_shared_bytes"] =
                     item.max_dynamic_shared_bytes;
                 encoded["preferred_shared_carveout"] =
                     item.preferred_shared_carveout;
                 encoded["block_dim"] = item.block_dim;
                 encoded["dynamic_shared_bytes"] =
                     item.dynamic_shared_bytes;
                 encoded["active_blocks_per_multiprocessor"] =
                     item.active_blocks_per_multiprocessor;
                 encoded["multiprocessor_count"] =
                     item.multiprocessor_count;
                 tasks.append(std::move(encoded));
               }
               result["provider"] = "cuda_driver";
               result["materialized"] = true;
             }
#endif
#if defined(TI_WITH_VULKAN)
             if (backend == Arch::vulkan) {
               const auto *spirv_compiled =
                   dynamic_cast<const lang::spirv::CompiledKernelData *>(
                       &compiled);
               TI_ERROR_IF(spirv_compiled == nullptr,
                           "Vulkan artifact qualification requires SPIR-V "
                           "compiled data");
               for (const auto &item : spirv_compiled->get_internal_data()
                                                .metadata.kernel_attribs
                                                .tasks_attribs) {
                 py::dict encoded = spirv_task_metadata_to_python(item);
                 encoded["pipeline_executable_statistics_available"] = false;
                 encoded["pipeline_executable_statistics_reason"] =
                     "no optional Vulkan pipeline executable provider is "
                     "installed";
                 tasks.append(std::move(encoded));
               }
               result["provider"] = "spirv_static_reflection";
               result["materialized"] = false;
             }
#endif
             result["registered_after"] = compiled.get_handle().has_value();
             result["tasks"] = std::move(tasks);
             return result;
           })
      .def("_debug_gpu_artifact_qualification_stats", [](Program &program) {
        py::dict result;
        result["schema_version"] = 1;
        result["backend"] = arch_name(program.compile_config().arch);
#if defined(TI_WITH_CUDA)
        const auto stats =
            cuda::get_artifact_qualification_telemetry_snapshot();
        result["qualification_calls"] = stats.qualification_calls;
        result["registration_materializations"] =
            stats.registration_materializations;
        result["function_attribute_queries"] =
            stats.function_attribute_queries;
        result["occupancy_queries"] = stats.occupancy_queries;
#else
        result["qualification_calls"] = 0;
        result["registration_materializations"] = 0;
        result["function_attribute_queries"] = 0;
        result["occupancy_queries"] = 0;
#endif
        return result;
      })
      .def("_debug_snode_relocation_manifest",
           [](Program &program, Kernel *kernel) {
             TI_ERROR_IF(kernel == nullptr,
                         "SNode relocation manifest query received a null "
                         "kernel");
             std::vector<int> tree_ids;
             std::vector<OffloadedTaskManifest> tasks;
             std::string kernel_identity;
             Arch backend = program.compile_config().arch;
             SNodeRelocationDescriptor relocation;
             bool regular_handle_registered = false;
             bool graph_masked_handle_registered = false;
             {
               auto tree_guard =
                   program.acquire_snode_tree_lifecycle_read_guard();
               const auto &compiled = program.compile_kernel(
                   program.compile_config(), program.get_device_caps(),
                   *kernel);
               tree_ids = compiled.snode_tree_ids();
               tasks = compiled.task_manifest();
               kernel_identity = compiled.kernel_identity();
               backend = compiled.arch();
               relocation = compiled.snode_relocation_descriptor();
               regular_handle_registered = compiled.get_handle().has_value();
               graph_masked_handle_registered =
                   compiled.get_graph_masked_handle().has_value();
             }

             // The compiler/runtime descriptor is the product contract. This
             // binding only serializes it and adds the live generation
             // snapshot; it must not infer relocation admission itself.
             const auto dependencies =
                 program.snapshot_snode_tree_dependencies(tree_ids);
             py::dict result;
             result["schema_version"] = relocation.schema_version;
             result["purpose"] =
                 "relocatable_snode_executable_readiness";
             result["backend"] = arch_name(backend);
             result["kernel_identity"] = kernel_identity;
             result["has_snode_tree_dependencies"] =
                 relocation.has_snode_tree_dependencies;
             result["coverage"] = relocation.compiler_emitted
                                      ? "compiler_typed_relocation_contract"
                                      : "legacy_cache_fail_closed_contract";
             result["compiler_emitted"] = relocation.compiler_emitted;
             result["relocation_class"] = std::string(
                 snode_relocation_class_name(relocation.relocation_class));
             result["compiler_embedded_state_fully_classified"] =
                 relocation.compiler_embedded_state_fully_classified;
             result["reuse_admitted"] = relocation.reuse_admitted;
             result["regular_handle_registered"] =
                 regular_handle_registered;
             result["graph_masked_handle_registered"] =
                 graph_masked_handle_registered;

             py::list dependency_items;
             for (const auto &dependency : dependencies) {
               py::dict item;
               item["tree_id"] = dependency.tree_id;
               item["generation"] = dependency.generation;
               item["layout_fingerprint"] =
                   dependency.layout_fingerprint;
               item["binding_ownership"] = "generation";
               dependency_items.append(std::move(item));
             }
             result["tree_dependencies"] = std::move(dependency_items);

             py::list task_items;
             TI_ASSERT(tasks.size() == relocation.tasks.size());
             for (std::size_t index = 0; index < tasks.size(); ++index) {
               const auto &task = tasks[index];
               const auto &task_relocation = relocation.tasks[index];
               TI_ASSERT(task.task_index == task_relocation.task_index);
               TI_ASSERT(task.task_type == task_relocation.task_type);
               py::dict item = offloaded_task_manifest_to_python(task);
               py::list generation_state;
               for (const auto state :
                    task_relocation.generation_bound_state) {
                 generation_state.append(
                     std::string(snode_relocation_state_name(state)));
               }
               item["layout_code_candidate"] = true;
               item["relocation_class"] = std::string(
                   snode_relocation_class_name(
                       task_relocation.relocation_class));
               item["generation_bound_state"] =
                   std::move(generation_state);
               item["embedded_state_audited"] =
                   relocation.compiler_embedded_state_fully_classified;
               task_items.append(std::move(item));
             }
             result["tasks"] = std::move(task_items);

             py::list blockers;
             for (const auto blocker : relocation.blockers) {
               blockers.append(
                   std::string(snode_relocation_blocker_name(blocker)));
             }
             result["blockers"] = std::move(blockers);
             return result;
           })
      .def("_graph_task_manifest",
           [](Program &program, const aot::CompiledGraph &graph,
              bool include_external_commands) {
             auto tree_guard =
                 program.acquire_snode_tree_lifecycle_read_guard();
             py::list result;
             for (std::size_t dispatch_index = 0;
                  dispatch_index < graph.dispatches.size();
                  ++dispatch_index) {
               const auto &dispatch = graph.dispatches[dispatch_index];
               if (include_external_commands && dispatch.cuda_capture_command) {
                 // A vendor command has no Forge offload IR. Preserve its
                 // actual position without inventing a kernel/task geometry.
                 py::dict item;
                 item["external_command"] = dispatch.cuda_capture_command->kind();
                 item["dispatch_index"] = dispatch_index;
                 result.append(std::move(item));
                 continue;
               }
               TI_ERROR_IF(
                   dispatch.ti_kernel == nullptr,
                   "Task manifests are unavailable for AOT-only Graph "
                   "dispatch {} ({})",
                   dispatch_index, dispatch.kernel_name);
               const auto &compiled = program.compile_kernel(
                   program.compile_config(), program.get_device_caps(),
                   *dispatch.ti_kernel);
               for (const auto &task : compiled.task_manifest()) {
                 auto item = offloaded_task_manifest_to_python(
                     overlay_execution_plan_launch_requests(
                         task, dispatch.ti_kernel));
                 item["dispatch_index"] = dispatch_index;
                 item["kernel_name"] = dispatch.kernel_name;
                 item["dispatch_label"] = dispatch.dispatch_label;
                 item["indirect"] = dispatch.indirect_dispatch_arg.has_value();
                 item["source_dispatch_count"] =
                     std::max<std::size_t>(
                         1, dispatch.source_dispatches.size());
                 if (dispatch.indirect_dispatch_arg.has_value()) {
                   item["actual_grid_size"] = py::none();
                   item["actual_block_size"] = py::none();
                   item["actual_geometry_kind"] = "runtime_indirect";
                   item["actual_geometry_reason"] =
                       "the device-owned dispatch packet determines geometry "
                       "for each invocation without host readback";
                 }
                 result.append(std::move(item));
               }
             }
             return result;
           },
           py::arg("graph"), py::arg("include_external_commands") = false)
      .def("_graph_gpu_semantics_snapshot",
           [](Program &program, const aot::CompiledGraph &graph) {
             const auto backend = program.compile_config().arch;
             TI_ERROR_IF(backend != Arch::cuda && backend != Arch::vulkan,
                         "GPU executable-plan semantics are supported only "
                         "on CUDA and Vulkan, not {}",
                         arch_name(backend));
             auto tree_guard =
                 program.acquire_snode_tree_lifecycle_read_guard();
             py::dict result;
             result["backend"] = arch_name(backend);
             py::list segments;
             for (std::size_t dispatch_index = 0;
                  dispatch_index < graph.dispatches.size();
                  ++dispatch_index) {
               const auto &dispatch = graph.dispatches[dispatch_index];
               TI_ERROR_IF(
                   dispatch.ti_kernel == nullptr,
                   "GPU executable-plan semantics are unavailable for "
                   "AOT-only Graph dispatch {} ({})",
                   dispatch_index, dispatch.kernel_name);
               const auto &compiled = program.compile_kernel(
                   program.compile_config(), program.get_device_caps(),
                   *dispatch.ti_kernel);
               segments.append(compiled_kernel_gpu_semantics_to_python(
                   compiled, dispatch.ti_kernel->autodiff_mode,
                   dispatch.ti_kernel, &dispatch, dispatch_index));
             }
             result["segments"] = std::move(segments);
             return result;
           })
      .def("set_kernel_profiler_toolkit",
           [](Program *program, const std::string toolkit_name) {
             return program->profiler->set_profiler_toolkit(toolkit_name);
           })
      .def("timeline_clear",
           [](Program *) { Timelines::get_instance().clear(); })
      .def("timeline_save",
           [](Program *, const std::string &fn) {
             Timelines::get_instance().save(fn);
           })
      .def("print_memory_profiler_info", &Program::print_memory_profiler_info)
      .def("finalize", &Program::finalize)
      .def("get_total_compilation_time", &Program::get_total_compilation_time)
      .def("get_snode_num_dynamically_allocated",
           &Program::get_snode_num_dynamically_allocated)
      .def("reset_hash_snode_probe_stats",
           &Program::reset_hash_snode_probe_stats)
      .def("get_hash_snode_probe_stats",
           [](Program *program) {
             auto raw = program->get_hash_snode_probe_stats();
             py::dict ret;
             ret["insert_count"] = raw.size() > 0 ? raw[0] : 0;
             ret["insert_total"] = raw.size() > 1 ? raw[1] : 0;
             ret["insert_max"] = raw.size() > 2 ? raw[2] : 0;
             ret["lookup_count"] = raw.size() > 3 ? raw[3] : 0;
             ret["lookup_total"] = raw.size() > 4 ? raw[4] : 0;
             ret["lookup_max"] = raw.size() > 5 ? raw[5] : 0;
             ret["insert_mean"] =
                 raw.size() > 1 && raw[0] > 0 ? double(raw[1]) / raw[0] : 0.0;
             ret["lookup_mean"] =
                 raw.size() > 4 && raw[3] > 0 ? double(raw[4]) / raw[3] : 0.0;
             return ret;
           })
      .def("synchronize", &Program::synchronize)
      .def("_vulkan_conditional_rendering_available", [](Program &program) {
        Device *device = program.get_graphics_device();
        return device != nullptr && device->supports_conditional_commands();
      })
      .def("_vulkan_max_compute_work_group_count_x", [](Program &program) {
#if defined(TI_WITH_VULKAN)
        auto *device = dynamic_cast<vulkan::VulkanDevice *>(
            program.get_graphics_device());
        TI_ERROR_IF(device == nullptr,
                    "Vulkan compute dispatch limits require an active Vulkan "
                    "device");
        return static_cast<std::uint64_t>(
            device->get_vk_physical_device_props()
                .limits.maxComputeWorkGroupCount[0]);
#else
        TI_ERROR("Vulkan compute dispatch limits are unavailable in this build");
        return std::uint64_t{0};
#endif
      })
      .def("_debug_vulkan_queue_submission_stats", [](Program &program) {
        py::dict result;
#if defined(TI_WITH_VULKAN)
        auto *device = dynamic_cast<vulkan::VulkanDevice *>(
            program.get_graphics_device());
        if (device != nullptr) {
          const auto snapshot = device->queue_submission_snapshot();
          result["supported"] = true;
          result["queue_submit_calls"] = snapshot.queue_submit_calls;
          result["submitted_command_buffers"] =
              snapshot.submitted_command_buffers;
          result["batched_queue_submit_calls"] =
              snapshot.batched_queue_submit_calls;
          result["batched_command_buffers"] =
              snapshot.batched_command_buffers;
          return result;
        }
#endif
        result["supported"] = false;
        result["queue_submit_calls"] = 0;
        result["submitted_command_buffers"] = 0;
        result["batched_queue_submit_calls"] = 0;
        result["batched_command_buffers"] = 0;
        return result;
      })
      .def("_runtime_has_fatal_fault", &Program::runtime_has_fatal_fault)
      .def("_debug_inject_runtime_fault",
           &Program::debug_inject_runtime_fault)
      .def("_debug_runtime_fault_state", [](const Program &program) {
        const RuntimeFaultSnapshot snapshot = program.runtime_fault_snapshot();
        py::dict result;
        result["state"] = runtime_lifecycle_state_name(snapshot.state);
        result["domain"] = snapshot.program_domain;
        result["rejected_submissions"] = snapshot.rejected_submissions;
        if (snapshot.first_fault) {
          const RuntimeFaultRecord &fault = *snapshot.first_fault;
          result["backend"] = arch_name(fault.backend);
          result["backend_code"] = fault.backend_code;
          result["sequence"] = fault.submission_sequence;
          result["operation"] = fault.operation;
          result["message"] = fault.message;
        } else {
          result["backend"] = py::none();
          result["backend_code"] = py::none();
          result["sequence"] = py::none();
          result["operation"] = py::none();
          result["message"] = py::none();
        }
        return result;
      })
      .def("_debug_reset_ordinary_launch_attribution",
           &Program::debug_reset_ordinary_launch_attribution)
      .def("_debug_ordinary_launch_attribution",
           &Program::debug_ordinary_launch_attribution)
      .def("_runtime_statistics_snapshot", [](Program &program) {
        const RuntimeStatisticsSnapshot snapshot =
            program.runtime_statistics_snapshot();
        auto optional_counter =
            [](const RuntimeOptionalCounter &counter) -> py::object {
          if (!counter.available) {
            return py::none();
          }
          return py::int_(counter.value);
        };

        py::dict submission;
        submission["kernel_submissions"] =
            snapshot.submission.kernel_submissions;
        submission["graph_submissions"] =
            snapshot.submission.graph_submissions;
        submission["graph_backend_submissions"] =
            snapshot.submission.graph_backend_submissions;
        submission["backend_graph_launches"] =
            snapshot.submission.backend_graph_launches;
        submission["native_submissions"] =
            snapshot.submission.native_submissions;
        submission["failed_submissions"] =
            snapshot.submission.failed_submissions;

        py::dict synchronization;
        synchronization["program_syncs"] =
            snapshot.synchronization.program_syncs;
        synchronization["program_sync_wait_ns"] =
            snapshot.synchronization.program_sync_wait_ns;
        synchronization["completion_polls"] =
            snapshot.synchronization.completion_polls;
        synchronization["completion_waits"] =
            snapshot.synchronization.completion_waits;
        synchronization["completion_wait_ns"] =
            snapshot.synchronization.completion_wait_ns;
        synchronization["backend_waits"] =
            optional_counter(snapshot.synchronization.backend_waits);
        synchronization["backend_wait_ns"] =
            optional_counter(snapshot.synchronization.backend_wait_ns);
        synchronization["backend_lock_samples"] =
            optional_counter(snapshot.synchronization.backend_lock_samples);
        synchronization["backend_lock_contentions"] = optional_counter(
            snapshot.synchronization.backend_lock_contentions);
        synchronization["backend_lock_sampled_wait_ns"] = optional_counter(
            snapshot.synchronization.backend_lock_sampled_wait_ns);

        py::dict memory;
        memory["live_resources"] = snapshot.memory.live_resources;
        memory["retiring_resources"] = snapshot.memory.retiring_resources;
        memory["inflight_resources"] = snapshot.memory.inflight_resources;
        const auto &host_allocator_snapshot =
            snapshot.memory.host_allocator;
        py::dict host_allocator;
        host_allocator["requested_live_bytes"] = optional_counter(
            host_allocator_snapshot.requested_live_bytes);
        host_allocator["peak_requested_live_bytes"] = optional_counter(
            host_allocator_snapshot.peak_requested_live_bytes);
        host_allocator["reserved_bytes"] = optional_counter(
            host_allocator_snapshot.reserved_bytes);
        host_allocator["committed_bytes"] = optional_counter(
            host_allocator_snapshot.committed_bytes);
        host_allocator["capacity_bytes"] = optional_counter(
            host_allocator_snapshot.capacity_bytes);
        host_allocator["used_bytes"] =
            optional_counter(host_allocator_snapshot.used_bytes);
        host_allocator["available_bytes"] = optional_counter(
            host_allocator_snapshot.available_bytes);
        host_allocator["alignment_waste_bytes"] = optional_counter(
            host_allocator_snapshot.alignment_waste_bytes);
        host_allocator["unreclaimed_released_bytes"] = optional_counter(
            host_allocator_snapshot.unreclaimed_released_bytes);
        host_allocator["wasted_bytes"] =
            optional_counter(host_allocator_snapshot.wasted_bytes);
        host_allocator["chunk_count"] =
            optional_counter(host_allocator_snapshot.chunk_count);
        host_allocator["slab_chunk_count"] = optional_counter(
            host_allocator_snapshot.slab_chunk_count);
        host_allocator["large_chunk_count"] = optional_counter(
            host_allocator_snapshot.large_chunk_count);
        host_allocator["exclusive_chunk_count"] = optional_counter(
            host_allocator_snapshot.exclusive_chunk_count);
        host_allocator["peak_reserved_bytes"] = optional_counter(
            host_allocator_snapshot.peak_reserved_bytes);
        host_allocator["peak_used_bytes"] = optional_counter(
            host_allocator_snapshot.peak_used_bytes);
        host_allocator["peak_wasted_bytes"] = optional_counter(
            host_allocator_snapshot.peak_wasted_bytes);
        host_allocator["peak_chunk_count"] = optional_counter(
            host_allocator_snapshot.peak_chunk_count);
        memory["host_allocator"] = host_allocator;
        memory["host_requested_live_bytes"] =
            optional_counter(snapshot.memory.host_requested_live_bytes);
        memory["host_raw_bytes"] =
            optional_counter(snapshot.memory.host_raw_bytes);
        memory["host_capacity_bytes"] =
            optional_counter(snapshot.memory.host_capacity_bytes);
        memory["device_requested_live_bytes"] =
            optional_counter(snapshot.memory.device_requested_live_bytes);
        memory["device_raw_bytes"] =
            optional_counter(snapshot.memory.device_raw_bytes);
        memory["device_cached_bytes"] =
            optional_counter(snapshot.memory.device_cached_bytes);
        memory["cuda_mempool_reserved_bytes"] =
            optional_counter(snapshot.memory.cuda_mempool_reserved_bytes);
        memory["cuda_mempool_used_bytes"] =
            optional_counter(snapshot.memory.cuda_mempool_used_bytes);

        py::dict transfer;
        transfer["host_to_device_bytes"] =
            snapshot.transfer.host_to_device_bytes;
        transfer["device_to_host_bytes"] =
            snapshot.transfer.device_to_host_bytes;
        transfer["device_to_device_bytes"] =
            snapshot.transfer.device_to_device_bytes;
        transfer["cuda_vulkan_direct_bytes"] =
            snapshot.transfer.cuda_vulkan_direct_bytes;
        transfer["cuda_vulkan_fallback_bytes"] =
            snapshot.transfer.cuda_vulkan_fallback_bytes;

        py::dict graph;
        graph["captures"] = snapshot.graph.captures;
        graph["recaptures"] = snapshot.graph.recaptures;
        graph["replays"] = snapshot.graph.replays;
        graph["ordinary_fallbacks"] =
            snapshot.graph.ordinary_fallbacks;
        graph["replay_slot_saturation_fallbacks"] =
            snapshot.graph.replay_slot_saturation_fallbacks;

        py::dict display;
        display["accepted_frames"] = snapshot.display.accepted_frames;
        display["submitted_frames"] = snapshot.display.submitted_frames;
        display["dropped_frames"] = snapshot.display.dropped_frames;
        display["accepted_frame_bytes"] =
            snapshot.display.accepted_frame_bytes;

        const RuntimeFaultSnapshot fault_snapshot =
            program.runtime_fault_snapshot();
        py::dict fault;
        fault["state"] = runtime_lifecycle_state_name(fault_snapshot.state);
        fault["first_fatal_faults"] =
            snapshot.fault.first_fatal_faults;
        fault["rejected_submissions"] =
            snapshot.fault.rejected_submissions;
        if (fault_snapshot.first_fault) {
          const RuntimeFaultRecord &first = *fault_snapshot.first_fault;
          py::dict first_fault;
          first_fault["backend"] = arch_name(first.backend);
          first_fault["backend_code"] = first.backend_code;
          first_fault["sequence"] = first.submission_sequence;
          first_fault["operation"] = first.operation;
          first_fault["message"] = first.message;
          fault["first_fault"] = std::move(first_fault);
        } else {
          fault["first_fault"] = py::none();
        }

        py::dict trace;
        trace["recorded_events"] = snapshot.trace.recorded_events;
        trace["dropped_events"] = snapshot.trace.dropped_events;

        py::dict result;
        result["schema_version"] = snapshot.schema_version;
        result["backend"] = arch_name(snapshot.backend);
        result["program_domain"] = snapshot.program_domain;
        result["submission"] = std::move(submission);
        result["synchronization"] = std::move(synchronization);
        result["memory"] = std::move(memory);
        result["transfer"] = std::move(transfer);
        result["graph"] = std::move(graph);
        result["display"] = std::move(display);
        result["fault"] = std::move(fault);
        result["trace"] = std::move(trace);
        return result;
      })
      .def("_primitive_workspace_stats", [](const Program &program) {
        return primitive_workspace_snapshot_to_dict(
            program.primitive_workspace_snapshot());
      })
      .def("_primitive_workspace_set_budget_bytes",
           &Program::set_primitive_workspace_budget)
      .def("_primitive_workspace_clear", &Program::clear_primitive_workspaces,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "_runtime_trace_start",
          [](Program &program, std::size_t max_threads,
             std::size_t events_per_thread) {
            RuntimeTraceSnapshot snapshot;
            {
              py::gil_scoped_release release;
              snapshot =
                  program.runtime_trace().start(max_threads, events_per_thread);
            }
            return runtime_trace_snapshot_to_dict(snapshot);
          },
          py::arg("max_threads") = RuntimeTraceRecorder::kDefaultMaxThreads,
          py::arg("events_per_thread") =
              RuntimeTraceRecorder::kDefaultEventsPerThread)
      .def("_runtime_trace_stop", [](Program &program) {
        RuntimeTraceSnapshot snapshot;
        {
          py::gil_scoped_release release;
          snapshot = program.runtime_trace().stop();
        }
        return runtime_trace_snapshot_to_dict(snapshot);
      })
      .def("_runtime_trace_snapshot", [](const Program &program) {
        return runtime_trace_snapshot_to_dict(program.runtime_trace().snapshot());
      })
      .def("_runtime_trace_export", [](const Program &program,
                                        const std::string &path) {
        py::gil_scoped_release release;
        return program.runtime_trace().export_chrome_trace(path);
      })
      .def("_record_runtime_completion",
           [](Program &program) { return program.record_runtime_completion(); },
           py::call_guard<py::gil_scoped_release>())
      .def("_record_runtime_graph_submission", [](Program &program) {
        program.record_runtime_submission_stat(RuntimeSubmissionKind::kGraph);
      })
      .def("_begin_runtime_submission_transaction",
           &Program::begin_runtime_submission_transaction,
           py::call_guard<py::gil_scoped_release>(),
           py::arg("gpu_timing") = false,
           py::arg("cuda_concurrent_batch") = false)
      .def("_begin_external_cuda_submission",
           &Program::begin_external_cuda_submission)
      .def("_debug_runtime_completion_stats",
           &Program::debug_runtime_completion_stats)
      .def("_set_cpu_scheduler_telemetry_enabled",
           &Program::set_cpu_scheduler_telemetry_enabled,
           py::arg("enabled"))
      .def("_debug_cpu_scheduler_telemetry",
           &Program::debug_cpu_scheduler_telemetry,
           py::arg("reset") = false)
      .def("_debug_kernel_definition_count", [](const Program &program) {
        return program.kernels.size();
      })
      .def("_debug_kernel_lifecycle_stats", [](Program &program) {
        std::size_t live_definitions = 0;
        std::size_t retired_shells = 0;
        for (const auto &kernel : program.kernels) {
          if (kernel->ir == nullptr) {
            ++retired_shells;
          } else {
            ++live_definitions;
          }
        }
        py::dict result;
        result["total_slots"] = program.kernels.size();
        result["live_definitions"] = live_definitions;
        result["retired_shells"] = retired_shells;
        result["retired_shell_inline_bytes_lower_bound"] =
            retired_shells * sizeof(Kernel);
        result["retired_shell_total_owned_bytes_reported"] = false;
        result["registered_executables"] =
            program.get_kernel_launcher().debug_registered_kernel_count();
        return result;
      })
      .def("_set_kernel_executable_lifecycle_telemetry_enabled",
           &Program::set_kernel_executable_lifecycle_telemetry_enabled,
           py::arg("enabled"))
      .def("_debug_kernel_executable_lifecycle_stats",
           [](Program &program, bool reset) {
             const auto stats =
                 program.debug_kernel_executable_lifecycle_statistics(reset);
             py::dict result;
             result["schema_version"] = 1;
             result["enabled"] = stats.enabled;
             result["memory_cache_hits"] = stats.memory_cache_hits;
             result["loaded_cache_hits"] = stats.loaded_cache_hits;
             result["disk_loads"] = stats.disk_loads;
             result["compiler_invocations"] = stats.compiler_invocations;
             result["templates_installed"] = stats.templates_installed;
             result["templates_retired"] = stats.templates_retired;
             result["resident_templates"] = stats.resident_templates;
             result["in_progress_compiles"] = stats.in_progress_compiles;
             result["live_handles"] = stats.live_handles;
             result["pinned_handles"] = stats.pinned_handles;
             result["retired_handles"] = stats.retired_handles;
             result["retired_generation_bound_handles"] =
                 stats.retired_generation_bound_handles;
             result["relocatable_templates"] =
                 stats.relocatable_templates;
             result["relocatable_template_hits"] =
                 stats.relocatable_template_hits;
             result["relocatable_bindings_created"] =
                 stats.relocatable_bindings_created;
             result["relocatable_template_reclaims"] =
                 stats.relocatable_template_reclaims;
             result["handle_inline_bytes"] = stats.handle_inline_bytes;
             result["registered_executables"] =
                 program.get_kernel_launcher().debug_registered_kernel_count();
             return result;
           },
           py::arg("reset") = false)
      .def("_debug_kernel_registration_count", [](Program &program) {
        return program.get_kernel_launcher().debug_registered_kernel_count();
      })
      .def("_snode_tree_layout_fingerprint",
           &Program::snode_tree_layout_fingerprint_for)
      .def("_snode_executable_reuse_enabled",
           &Program::snode_executable_reuse_enabled)
      .def("_debug_snode_executable_reuse_capabilities",
           [](Program &program) {
             const Arch backend = program.compile_config().arch;
             const bool backend_qualified =
                 arch_is_cpu(backend) || backend == Arch::cuda ||
                 backend == Arch::vulkan;
             py::dict result;
             result["schema_version"] = 1;
             result["backend"] = arch_name(backend);
             result["rollback_switch"] =
                 "TI_ENABLE_SNODE_EXECUTABLE_REUSE";
             result["enabled"] =
                 program.snode_executable_reuse_enabled();
             result["stale_graph_retarget"] = false;
             result["root_dense"] =
                 backend_qualified ? "partially_relocatable"
                                   : "generation_bound";
             result["pointer"] = "generation_bound";
             result["bitmasked"] = "generation_bound";
             result["dynamic"] = "generation_bound";
             result["hash"] = "generation_bound";
             return result;
           })
      .def("_prepare_relocatable_kernel_template",
           &Program::prepare_relocatable_kernel_template,
           py::call_guard<py::gil_scoped_release>())
      .def("_register_relocatable_kernel_template_candidate",
           &Program::register_relocatable_kernel_template_candidate,
           py::call_guard<py::gil_scoped_release>())
      .def("_reclaim_relocatable_kernel_templates",
           &Program::reclaim_relocatable_kernel_templates,
           py::call_guard<py::gil_scoped_release>())
      .def("_debug_snode_field_mapping_count", [](Program &program) {
        return program.get_snode_to_fields()->size();
      })
      .def("_debug_snode_runtime_directory_stats", [](Program &program) {
        const auto snapshot =
            program.debug_snode_runtime_directory_statistics();
        py::dict result;
        result["available"] = snapshot.available;
        result["host_visible"] = snapshot.host_visible;
        result["capacity"] = snapshot.capacity;
        result["active_tree_count"] = snapshot.active_tree_count;
        result["reserved_bytes"] = snapshot.reserved_bytes;
        result["growth_events"] = snapshot.growth_events;
        return result;
      })
      .def("_debug_snode_metadata_stats", [](const Program &program) {
        const auto snapshot = program.debug_snode_metadata_statistics();
        py::dict result;
        result["tree_slots"] = snapshot.tree_slots;
        result["active_tree_count"] = snapshot.active_tree_count;
        result["retired_tree_shells"] = snapshot.retired_tree_shells;
        result["free_tree_ids"] = snapshot.free_tree_ids;
        result["active_snode_count"] = snapshot.active_snode_count;
        result["retired_snode_count"] = snapshot.retired_snode_count;
        result["tree_inline_bytes_lower_bound"] =
            snapshot.tree_inline_bytes_lower_bound;
        result["snode_inline_bytes_lower_bound"] =
            snapshot.snode_inline_bytes_lower_bound;
        result["generation_table_bytes"] = snapshot.generation_table_bytes;
        result["active_table_bytes"] = snapshot.active_table_bytes;
        result["global_snode_ids_issued"] = snapshot.global_snode_ids_issued;
        result["logical_bytes_are_lower_bounds"] = true;
        return result;
      })
      .def("_debug_sparse_snode_tree_stats",
           [](Program &program, int tree_id) {
             SparseSNodeTreeStatistics snapshot;
             {
               py::gil_scoped_release release;
               snapshot =
                   program.debug_sparse_snode_tree_statistics(tree_id);
             }
             const auto optional_counter =
                 [](const RuntimeOptionalCounter &counter) -> py::object {
               if (!counter.available) {
                 return py::none();
               }
               return py::int_(counter.value);
             };
             const auto &source = snapshot.memory;
             py::dict memory;
             memory["tree_owned_reserved_bytes"] =
                 optional_counter(source.tree_owned_reserved_bytes);
             memory["root_reserved_bytes"] =
                 optional_counter(source.root_reserved_bytes);
             memory["sparse_pool_reserved_bytes"] =
                 optional_counter(source.sparse_pool_reserved_bytes);
             memory["runtime_state_reserved_bytes"] =
                 optional_counter(source.runtime_state_reserved_bytes);
             memory["runtime_metadata_requested_bytes"] =
                 optional_counter(source.runtime_metadata_requested_bytes);
             memory["direct_ambient_requested_bytes"] =
                 optional_counter(source.direct_ambient_requested_bytes);
             memory["allocator_payload_reserved_bytes"] =
                 optional_counter(source.allocator_payload_reserved_bytes);
             memory["allocator_payload_used_bytes"] =
                 optional_counter(source.allocator_payload_used_bytes);
             memory["allocator_bookkeeping_reserved_bytes"] =
                 optional_counter(
                     source.allocator_bookkeeping_reserved_bytes);
             memory["active_list_reserved_bytes"] =
                 optional_counter(source.active_list_reserved_bytes);
             memory["active_list_used_bytes"] =
                 optional_counter(source.active_list_used_bytes);
             memory["allocator_in_use_elements"] =
                 optional_counter(source.allocator_in_use_elements);
             memory["allocator_free_elements"] =
                 optional_counter(source.allocator_free_elements);
             memory["allocator_recycled_elements"] =
                 optional_counter(source.allocator_recycled_elements);
             memory["shared_listgen_workspace_reserved_bytes"] =
                 optional_counter(
                     source.shared_listgen_workspace_reserved_bytes);
             memory["tree_owned_scope"] = source.tree_owned_scope;
             memory["runtime_resource_scope"] =
                 source.runtime_resource_scope;
             memory["shared_listgen_workspace_scope"] =
                 source.shared_listgen_workspace_scope;

             py::list listgen_nodes;
             std::uint64_t requests = 0;
             std::uint64_t rebuilds = 0;
             std::uint64_t reuse_hits = 0;
             std::uint64_t invalidations = 0;
             std::uint64_t resident_evictions = 0;
             std::uint64_t candidate_slots = 0;
             std::uint64_t scanned_elements = 0;
             std::uint64_t emitted_elements = 0;
             std::uint64_t serial_rebuilds = 0;
             std::uint64_t parallel_rebuilds = 0;
             bool candidate_slots_available = true;
             bool resident_evictions_available = true;
             bool scanned_elements_available = true;
             bool emitted_elements_available = true;
             bool serial_rebuilds_available = true;
             bool parallel_rebuilds_available = true;
             for (const auto &node : snapshot.listgen.nodes) {
               py::dict item;
               item["snode_id"] = node.snode_id;
               item["parent_snode_id"] = node.parent_snode_id;
               item["requests"] = node.requests;
               item["rebuilds"] = node.rebuilds;
               item["reuse_hits"] = node.reuse_hits;
               item["invalidations"] = node.invalidations;
               item["resident_evictions"] =
                   optional_counter(node.resident_evictions);
               item["candidate_slots_dispatched"] =
                   optional_counter(node.candidate_slots_dispatched);
               item["scanned_elements"] =
                   optional_counter(node.scanned_elements);
               item["emitted_elements"] =
                   optional_counter(node.emitted_elements);
               item["serial_rebuilds"] =
                   optional_counter(node.serial_rebuilds);
               item["parallel_rebuilds"] =
                   optional_counter(node.parallel_rebuilds);
               item["last_rebuild_reason"] = node.last_rebuild_reason;
               listgen_nodes.append(std::move(item));
               requests += node.requests;
               rebuilds += node.rebuilds;
               reuse_hits += node.reuse_hits;
               invalidations += node.invalidations;
               resident_evictions_available &=
                   node.resident_evictions.available;
               resident_evictions += node.resident_evictions.value;
               candidate_slots_available &=
                   node.candidate_slots_dispatched.available;
               candidate_slots += node.candidate_slots_dispatched.value;
               scanned_elements_available &= node.scanned_elements.available;
               scanned_elements += node.scanned_elements.value;
               emitted_elements_available &= node.emitted_elements.available;
               emitted_elements += node.emitted_elements.value;
               serial_rebuilds_available &= node.serial_rebuilds.available;
               serial_rebuilds += node.serial_rebuilds.value;
               parallel_rebuilds_available &= node.parallel_rebuilds.available;
               parallel_rebuilds += node.parallel_rebuilds.value;
             }
             py::dict listgen_totals;
             listgen_totals["requests"] = requests;
             listgen_totals["rebuilds"] = rebuilds;
             listgen_totals["reuse_hits"] = reuse_hits;
             listgen_totals["invalidations"] = invalidations;
             listgen_totals["resident_evictions"] =
                 resident_evictions_available && !snapshot.listgen.nodes.empty()
                     ? py::cast(resident_evictions)
                     : py::none();
             listgen_totals["candidate_slots_dispatched"] =
                 candidate_slots_available && !snapshot.listgen.nodes.empty()
                     ? py::cast(candidate_slots)
                     : py::none();
             listgen_totals["scanned_elements"] =
                 scanned_elements_available && !snapshot.listgen.nodes.empty()
                     ? py::cast(scanned_elements)
                     : py::none();
             listgen_totals["emitted_elements"] =
                 emitted_elements_available && !snapshot.listgen.nodes.empty()
                     ? py::cast(emitted_elements)
                     : py::none();
             listgen_totals["serial_rebuilds"] =
                 serial_rebuilds_available && !snapshot.listgen.nodes.empty()
                     ? py::cast(serial_rebuilds)
                     : py::none();
             listgen_totals["parallel_rebuilds"] =
                 parallel_rebuilds_available && !snapshot.listgen.nodes.empty()
                     ? py::cast(parallel_rebuilds)
                     : py::none();
             py::dict listgen;
             listgen["available"] = snapshot.listgen.available;
             listgen["totals"] = std::move(listgen_totals);
             listgen["nodes"] = std::move(listgen_nodes);

             py::dict result;
             result["schema_version"] = 1;
             result["tree_id"] = snapshot.tree_id;
             result["generation"] = snapshot.generation;
             result["layout_fingerprint"] = snapshot.layout_fingerprint;
             result["backend"] = arch_name(snapshot.backend);
             result["memory"] = std::move(memory);
             result["listgen"] = std::move(listgen);
             return result;
           })
      .def("_debug_reset_sparse_listgen_stats",
           &Program::debug_reset_sparse_listgen_statistics,
           py::call_guard<py::gil_scoped_release>())
      .def("materialize_runtime", &Program::materialize_runtime)
      .def("make_aot_module_builder", &Program::make_aot_module_builder)
      .def("get_snode_tree_size", &Program::get_snode_tree_size)
      .def("get_active_snode_tree_ids", &Program::get_active_snode_tree_ids)
      .def("get_snode_root", &Program::get_snode_root,
           py::return_value_policy::reference)
      .def(
          "create_kernel",
          [](Program *program, const std::function<void(Kernel *)> &body,
             const std::string &name, AutodiffMode autodiff_mode) -> Kernel * {
            py::gil_scoped_release release;
            return &program->kernel(body, name, autodiff_mode);
          },
          py::return_value_policy::reference)
      .def("create_function", &Program::create_function,
           py::return_value_policy::reference)
       .def("create_sparse_matrix",
           [](Program *program, int n, int m, DataType dtype,
              std::string storage_format) {
             TI_ERROR_IF(!arch_is_cpu(program->compile_config().arch) &&
                             !arch_is_cuda(program->compile_config().arch),
                         "SparseMatrix only supports CPU and CUDA for now.");
             if (arch_is_cpu(program->compile_config().arch))
               return make_sparse_matrix(n, m, dtype, storage_format);
              else
                return make_cu_sparse_matrix(n, m, dtype);
            })
      .def("_create_bsr_pattern",
           [](Program *program, int block_rows, int block_cols, int block_size,
              const Ndarray &row_offsets, const Ndarray &column_indices) {
             return std::make_shared<SparseBsrPattern>(
                 program, block_rows, block_cols, block_size, row_offsets,
                 column_indices);
           },
           py::keep_alive<0, 1>())
      .def("_create_csr_pattern",
           [](Program *program, int rows, int cols,
              const Ndarray &row_offsets, const Ndarray &column_indices) {
             return std::make_shared<SparseCsrPattern>(
                 program, rows, cols, row_offsets, column_indices);
           },
           py::keep_alive<0, 1>())
      .def("_create_compiled_kernel_linear_operator",
           [](Program *program, Kernel &kernel, int size,
              std::uint64_t topology_version,
              std::uint64_t numeric_version,
              const Ndarray &operator_data) -> std::unique_ptr<SparseMatrix> {
             return std::make_unique<CompiledKernelLinearOperator>(
                 program, kernel, size, topology_version, numeric_version,
                 operator_data);
           },
           py::keep_alive<0, 1>())
      .def("_create_compiled_kernel_linear_operator_with_numeric_data",
           [](Program *program, Kernel &kernel, int size,
              std::uint64_t topology_version,
              std::uint64_t numeric_version,
              const Ndarray &topology_data,
              const Ndarray &numeric_data) -> std::unique_ptr<SparseMatrix> {
             return std::make_unique<CompiledKernelLinearOperator>(
                 program, kernel, size, topology_version, numeric_version,
                 topology_data, numeric_data);
            },
            py::keep_alive<0, 1>())
      .def("_create_compiled_graph_linear_operator",
           [](Program *program, const aot::CompiledGraph &graph, int size,
              std::uint64_t topology_version,
              std::uint64_t numeric_version, const py::dict &fixed_i32_args,
              const py::dict &topology_args, const py::dict &numeric_args,
              const py::dict &workspace_args,
              const py::dict &state_args) -> std::unique_ptr<SparseMatrix> {
             CompiledGraphLinearOperator::FixedI32Arguments fixed_i32;
             for (const auto &item : fixed_i32_args) {
               fixed_i32.emplace(py::cast<std::string>(item.first),
                                 py::cast<std::int32_t>(item.second));
             }
             auto parse_ndarrays = [](const py::dict &arguments) {
               CompiledGraphLinearOperator::NdarrayArguments result;
               for (const auto &item : arguments) {
                 result.emplace(py::cast<std::string>(item.first),
                                &py::cast<const Ndarray &>(item.second));
               }
               return result;
             };
             std::vector<SNodeTreeDependency> state_dependencies;
             state_dependencies.reserve(state_args.size());
             for (const auto &item : state_args) {
               const auto name = py::cast<std::string>(item.first);
               TI_ERROR_IF(name.empty(),
                           "Compiled-Graph fixed Field state names must be "
                           "non-empty.");
               const auto &descriptor =
                   py::cast<const storage::DenseStorageDescriptor &>(
                       item.second);
               const auto source_kind = descriptor.source_kind();
               const auto &owner = descriptor.owner();
               TI_ERROR_IF(
                   (source_kind !=
                        storage::StorageSourceKind::kDenseScalarField &&
                    source_kind !=
                        storage::StorageSourceKind::kDensePackedField) ||
                       owner.kind !=
                           storage::StorageOwnerKind::kSNodePayload ||
                       storage::validate_storage_owner(*program, descriptor) !=
                           storage::StorageFailureReason::kNone,
                   "Compiled-Graph fixed Field state '{}' must be a live "
                   "root-dense scalar, vector, or matrix Field owned by the "
                   "same Program.",
                   name);
               state_dependencies.push_back(owner.tree);
             }
             return std::make_unique<CompiledGraphLinearOperator>(
                 program, graph, size, topology_version, numeric_version,
                 std::move(fixed_i32), parse_ndarrays(topology_args),
                 parse_ndarrays(numeric_args),
                 parse_ndarrays(workspace_args),
                 std::move(state_dependencies));
           },
           py::keep_alive<0, 1>(), py::arg("graph"), py::arg("size"),
           py::arg("topology_version"), py::arg("numeric_version"),
           py::arg("fixed_i32_args"), py::arg("topology_args"),
           py::arg("numeric_args"), py::arg("workspace_args"),
           py::arg("state_args") = py::dict())
      .def("_create_csr_matrix_from_pattern",
           [](Program *program, std::shared_ptr<SparseCsrPattern> pattern,
              const Ndarray &values) -> std::unique_ptr<SparseMatrix> {
             TI_ERROR_IF(!pattern || pattern->program() != program,
                         "Internal CSR operator construction requires a "
                         "pattern owned by the same Program.");
             if (arch_is_cpu(program->compile_config().arch)) {
               return std::make_unique<CpuSparseCsrMatrix>(
                   std::move(pattern), values);
             }
             if (arch_is_cuda(program->compile_config().arch)) {
               return std::make_unique<CuSparseMatrix>(
                   std::move(pattern), values);
             }
             if (program->compile_config().arch == Arch::vulkan) {
               return std::make_unique<VulkanSparseMatrix>(
                   std::move(pattern), values);
             }
             TI_ERROR(
                 "Internal CSR operators support CPU, CUDA, and Vulkan "
                 "backends, got {}.",
                 arch_name(program->compile_config().arch));
           },
           py::keep_alive<0, 1>())
      .def("_create_bsr_matrix_from_pattern",
           [](Program *program, std::shared_ptr<SparseBsrPattern> pattern,
              const Ndarray &values) -> std::unique_ptr<SparseMatrix> {
             TI_ERROR_IF(!pattern || pattern->program() != program,
                         "Internal BSR operator construction requires a "
                         "pattern owned by the same Program.");
             if (arch_is_cpu(program->compile_config().arch)) {
               return std::make_unique<CpuSparseBsrMatrix>(
                   std::move(pattern), values);
             }
             if (arch_is_cuda(program->compile_config().arch)) {
               return std::make_unique<CuSparseBsrMatrix>(
                   std::move(pattern), values);
             }
             if (program->compile_config().arch == Arch::vulkan) {
               return std::make_unique<VulkanSparseBsrMatrix>(
                   std::move(pattern), values);
             }
             TI_ERROR(
                 "Internal BSR operators support CPU, CUDA, and Vulkan "
                 "backends, got {}.",
                 arch_name(program->compile_config().arch));
           },
           py::keep_alive<0, 1>())
      .def("_create_cpu_bsr_matrix",
           [](Program *program, int block_rows, int block_cols, int block_size,
              const Ndarray &row_offsets, const Ndarray &column_indices,
              const Ndarray &values) {
             TI_ERROR_IF(!arch_is_cpu(program->compile_config().arch),
                         "Internal CPU BSR matrices require a CPU backend.");
             return std::make_unique<CpuSparseBsrMatrix>(
                 program, block_rows, block_cols, block_size, row_offsets,
                 column_indices, values);
           },
           py::keep_alive<0, 1>())
      .def("_create_cuda_bsr_matrix",
           [](Program *program, int block_rows, int block_cols, int block_size,
              const Ndarray &row_offsets, const Ndarray &column_indices,
              const Ndarray &values) {
             TI_ERROR_IF(!arch_is_cuda(program->compile_config().arch),
                         "Internal BSR matrices require the CUDA backend.");
             return std::make_unique<CuSparseBsrMatrix>(
                 program, block_rows, block_cols, block_size, row_offsets,
                 column_indices, values);
           },
           py::keep_alive<0, 1>())
      .def("_create_vulkan_csr_matrix",
           [](Program *program, int rows, int cols, const Ndarray &row_offsets,
              const Ndarray &column_indices, const Ndarray &values) {
             TI_ERROR_IF(program->compile_config().arch != Arch::vulkan,
                         "Internal Vulkan CSR matrices require the Vulkan "
                         "backend.");
             return std::make_unique<VulkanSparseMatrix>(
                 program, rows, cols, row_offsets, column_indices, values);
           },
           py::keep_alive<0, 1>())
      .def("_create_vulkan_bsr_matrix",
           [](Program *program, int block_rows, int block_cols, int block_size,
              const Ndarray &row_offsets, const Ndarray &column_indices,
              const Ndarray &values) {
             TI_ERROR_IF(program->compile_config().arch != Arch::vulkan,
                         "Internal Vulkan BSR matrices require the Vulkan "
                         "backend.");
             return std::make_unique<VulkanSparseBsrMatrix>(
                 program, block_rows, block_cols, block_size, row_offsets,
                 column_indices, values);
           },
           py::keep_alive<0, 1>())
      .def("_vulkan_sparse_assembly_available",
           &Program::vulkan_sparse_assembly_available)
      .def("_cuda_sparse_assembly_available",
           &Program::cuda_sparse_assembly_available)
      .def("_vulkan_sparse_axpy",
           tracked_native_program_method(&Program::vulkan_sparse_axpy),
           py::arg("x"), py::arg("y"), py::arg("n"),
           py::arg("alpha"))
      .def("_vulkan_sparse_dot",
           tracked_native_program_method(&Program::vulkan_sparse_dot),
           py::arg("x"), py::arg("y"), py::arg("output"),
           py::arg("n"))
      .def("_vulkan_sparse_norm",
           tracked_native_program_method(&Program::vulkan_sparse_norm),
           py::arg("x"), py::arg("output"), py::arg("n"))
      .def("_vulkan_sparse_scalar_divide",
           tracked_native_program_method(
               &Program::vulkan_sparse_scalar_divide),
           py::arg("numerator"), py::arg("denominator"),
           py::arg("quotient"), py::arg("status"))
      .def("_vulkan_sparse_cg_update",
           tracked_native_program_method(&Program::vulkan_sparse_cg_update),
           py::arg("direction"), py::arg("applied_direction"),
           py::arg("alpha"), py::arg("solution"),
           py::arg("residual"), py::arg("n"))
      .def("_vulkan_sparse_cg_direction",
           tracked_native_program_method(
               &Program::vulkan_sparse_cg_direction),
           py::arg("residual"), py::arg("beta"),
           py::arg("direction"), py::arg("n"))
      .def("_vulkan_sparse_convergence",
           tracked_native_program_method(
               &Program::vulkan_sparse_convergence),
           py::arg("residual_squared"), py::arg("status"),
           py::arg("completed_iterations"),
           py::arg("rhs_squared"), py::arg("absolute_tolerance"),
           py::arg("relative_tolerance"), py::arg("iteration"))
      .def("_vulkan_sparse_algebra_clear_workspace",
           &Program::vulkan_sparse_algebra_clear_workspace)
      .def("_vulkan_sparse_algebra_workspace_bytes",
           &Program::vulkan_sparse_algebra_workspace_bytes)
      .def("make_sparse_matrix_from_ndarray",
           [](Program *program, SparseMatrix &sm, const Ndarray &ndarray) {
             TI_ERROR_IF(!arch_is_cpu(program->compile_config().arch) &&
                             !arch_is_cuda(program->compile_config().arch),
                         "SparseMatrix only supports CPU and CUDA for now.");
             return make_sparse_matrix_from_ndarray(program, sm, ndarray);
           })
      .def("make_id_expr",
           [](Program *program, const std::string &name) {
             return Expr::make<IdExpression>(program->get_next_global_id(name));
           })
      .def(
          "create_ndarray",
          [&](Program *program, const DataType &dt,
              const std::vector<int> &shape, ExternalArrayLayout layout,
              bool zero_fill, DebugInfo dbg_info) -> Ndarray * {
            return program->create_ndarray(dt, shape, layout, zero_fill,
                                           dbg_info);
          },
          py::arg("dt"), py::arg("shape"),
          py::arg("layout") = ExternalArrayLayout::kNull,
          py::arg("zero_fill") = false, py::arg("dbg_info") = DebugInfo(),
          py::return_value_policy::reference)
      .def(
          "_create_graph_observation_ndarray",
          [&](Program *program, const DataType &dt,
              const std::vector<int> &shape,
              ExternalArrayLayout layout) -> Ndarray * {
            return program->create_ndarray(dt, shape, layout,
                                           /*zero_fill=*/false, DebugInfo(),
                                           /*host_read=*/true);
          },
          py::arg("dt"), py::arg("shape"),
          py::arg("layout") = ExternalArrayLayout::kNull,
          py::return_value_policy::reference)
      .def("delete_ndarray", &Program::delete_ndarray)
      .def(
          "create_argpack",
          [&](Program *program, const DataType &dt) -> ArgPack * {
            return program->create_argpack(dt);
          },
          py::arg("dt"), py::return_value_policy::reference)
      .def("delete_argpack", &Program::delete_argpack)
      .def("delete_texture", &Program::delete_texture)
      .def("_debug_argpack_resource_stats",
           &Program::debug_argpack_resource_stats)
      .def("_debug_argpack_resource_identity",
           &Program::debug_argpack_resource_identity)
      .def("_debug_ndarray_resource_stats",
           &Program::debug_ndarray_resource_stats)
      .def("_debug_ndarray_resource_identity",
           &Program::debug_ndarray_resource_identity)
      .def("_debug_texture_resource_stats",
           &Program::debug_texture_resource_stats)
      .def("_debug_texture_resource_identity",
           &Program::debug_texture_resource_identity)
      .def("_debug_vulkan_image_sampler_cache_size",
           &Program::debug_vulkan_image_sampler_cache_size)
      .def("_debug_vulkan_graphics_pipeline_count",
           &Program::debug_vulkan_graphics_pipeline_count)
      .def("_debug_vulkan_graphics_resource_stats",
           &Program::debug_vulkan_graphics_resource_stats)
      .def("_debug_vulkan_ray_resource_stats",
           &Program::debug_vulkan_ray_resource_stats)
      .def("_debug_dense_field_staging_stats",
           &Program::debug_dense_field_staging_stats)
      .def("_debug_dense_storage_binding_stats",
           &Program::debug_dense_storage_binding_stats)
      .def("_debug_external_dense_storage_stats",
           &Program::debug_external_dense_storage_stats)
      .def(
          "create_texture",
          [&](Program *program, BufferFormat fmt,
              const std::vector<int> &shape,
              const ImageSamplerConfig &sampler_config) -> Texture * {
            return program->create_texture(fmt, shape, sampler_config);
          },
          py::arg("fmt"), py::arg("shape"), py::arg("sampler_config"),
          py::return_value_policy::reference)
      .def("get_ndarray_data_ptr_as_int",
           [](Program *program, Ndarray *ndarray) {
             return program->get_ndarray_data_ptr_as_int(ndarray);
           })
      .def("fill_float",
           [](Program *program, Ndarray *ndarray, float val) {
             program->fill_ndarray_fast_u32(ndarray,
                                            reinterpret_cast<uint32_t &>(val));
           })
      .def("fill_int",
           [](Program *program, Ndarray *ndarray, int32_t val) {
             program->fill_ndarray_fast_u32(ndarray,
                                            reinterpret_cast<int32_t &>(val));
           })
      .def("fill_uint",
           [](Program *program, Ndarray *ndarray, uint32_t val) {
             program->fill_ndarray_fast_u32(ndarray, val);
           })
      .def("copy_ndarray",
           [](Program *program, Ndarray *dst, Ndarray *src) {
             program->copy_ndarray_fast(dst, src);
           },
           py::call_guard<py::gil_scoped_release>())
      .def("_vulkan_copy_texture",
           tracked_native_program_method(&Program::vulkan_copy_texture),
           py::arg("destination"), py::arg("source"),
           py::call_guard<py::gil_scoped_release>())
      .def("_vulkan_copy_texture_region",
           tracked_native_program_method(
               &Program::vulkan_copy_texture_region),
           py::arg("destination"), py::arg("source"),
           py::arg("source_offset"), py::arg("destination_offset"),
           py::arg("extent"), py::arg("source_mip_level"),
           py::arg("destination_mip_level"), py::arg("source_base_layer"),
           py::arg("destination_base_layer"), py::arg("layer_count"),
           py::call_guard<py::gil_scoped_release>())
      .def("_vulkan_copy_ndarray_to_texture",
           tracked_native_program_method(
               &Program::vulkan_copy_ndarray_to_texture),
           py::arg("destination"), py::arg("source"),
           py::arg("buffer_offset"), py::arg("buffer_row_length"),
           py::arg("buffer_image_height"), py::arg("image_offset"),
           py::arg("image_extent"), py::arg("image_mip_level"),
           py::arg("image_base_layer"), py::arg("image_layer_count"),
           py::call_guard<py::gil_scoped_release>())
      .def("_vulkan_copy_texture_to_ndarray",
           tracked_native_program_method(
               &Program::vulkan_copy_texture_to_ndarray),
           py::arg("destination"), py::arg("source"),
           py::arg("buffer_offset"), py::arg("buffer_row_length"),
           py::arg("buffer_image_height"), py::arg("image_offset"),
           py::arg("image_extent"), py::arg("image_mip_level"),
           py::arg("image_base_layer"), py::arg("image_layer_count"),
           py::call_guard<py::gil_scoped_release>())
      .def("_vulkan_blit_texture",
           tracked_native_program_method(&Program::vulkan_blit_texture),
           py::arg("destination"), py::arg("source"),
           py::arg("source_offset"), py::arg("source_extent"),
           py::arg("destination_offset"), py::arg("destination_extent"),
           py::arg("source_mip_level"),
           py::arg("destination_mip_level"), py::arg("source_base_layer"),
           py::arg("destination_base_layer"), py::arg("layer_count"),
           py::arg("linear_filter"),
           py::call_guard<py::gil_scoped_release>())
      .def(
          "_prepare_vulkan_buffer_commands",
          [](Program *program, const py::sequence &raw_commands) {
            std::vector<VulkanBufferCommand> commands;
            commands.reserve(raw_commands.size());
            for (const py::handle raw : raw_commands) {
              const py::tuple item = py::cast<py::tuple>(raw);
              TI_ERROR_IF(item.size() != 7,
                          "Vulkan buffer commands require seven fields.");
              const std::string kind = py::cast<std::string>(item[0]);
              VulkanBufferCommand command;
              if (kind == "fill_u32") {
                command.kind = VulkanBufferCommandKind::kFillU32;
              } else if (kind == "copy") {
                command.kind = VulkanBufferCommandKind::kCopy;
              } else if (kind == "buffer_barrier") {
                command.kind = VulkanBufferCommandKind::kBufferBarrier;
              } else if (kind == "memory_barrier") {
                command.kind = VulkanBufferCommandKind::kMemoryBarrier;
              } else {
                TI_ERROR("Unsupported Vulkan buffer command kind: {}", kind);
              }
              if (!item[1].is_none()) {
                command.destination_storage =
                    py::cast<const storage::DenseStorageDescriptor *>(item[1]);
              }
              if (!item[2].is_none()) {
                command.source_storage =
                    py::cast<const storage::DenseStorageDescriptor *>(item[2]);
              }
              command.destination_offset = py::cast<std::size_t>(item[3]);
              command.source_offset = py::cast<std::size_t>(item[4]);
              command.bytes = py::cast<std::size_t>(item[5]);
              command.value = py::cast<std::uint32_t>(item[6]);
              commands.push_back(command);
            }
            py::gil_scoped_release release;
            return program->prepare_vulkan_buffer_commands(commands);
          },
          py::arg("commands"))
      .def("_execute_vulkan_buffer_commands",
           tracked_native_program_method(&Program::execute_vulkan_buffer_commands),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_graphics_pipeline_available",
           &Program::vulkan_graphics_pipeline_available)
      .def("vulkan_graphics_indirect_capabilities",
           &Program::vulkan_graphics_indirect_capabilities)
      .def("vulkan_bindless_buffer_capabilities",
           &Program::vulkan_bindless_buffer_capabilities)
      .def("vulkan_mesh_shader_capabilities",
           &Program::vulkan_mesh_shader_capabilities)
      .def(
          "_create_vulkan_graphics_pipeline",
          [](Program *program, py::bytes vertex_bytes,
             py::bytes fragment_bytes, const py::sequence &raw_bindings,
             const py::sequence &raw_attributes, int topology,
             int polygon_mode, bool front_face_cull, bool back_face_cull,
             bool depth_test, bool depth_write, bool blending,
             const std::string &name) {
            auto decode_spirv = [](py::bytes value, const char *stage) {
              const std::string bytes = py::cast<std::string>(value);
              TI_ERROR_IF(bytes.empty() ||
                              bytes.size() % sizeof(std::uint32_t) != 0,
                          "Vulkan graphics {} SPIR-V must be nonempty and "
                          "four-byte aligned.",
                          stage);
              std::vector<std::uint32_t> words(
                  bytes.size() / sizeof(std::uint32_t));
              std::memcpy(words.data(), bytes.data(), bytes.size());
              return words;
            };
            std::vector<VulkanGraphicsVertexBinding> bindings;
            bindings.reserve(raw_bindings.size());
            for (const py::handle raw : raw_bindings) {
              const py::tuple item = py::cast<py::tuple>(raw);
              TI_ERROR_IF(item.size() != 3,
                          "Vulkan graphics vertex bindings require binding, "
                          "stride, and instance fields.");
              bindings.push_back({py::cast<std::uint32_t>(item[0]),
                                  py::cast<std::size_t>(item[1]),
                                  py::cast<bool>(item[2])});
            }
            std::vector<VulkanGraphicsVertexAttribute> attributes;
            attributes.reserve(raw_attributes.size());
            for (const py::handle raw : raw_attributes) {
              const py::tuple item = py::cast<py::tuple>(raw);
              TI_ERROR_IF(item.size() != 4,
                          "Vulkan graphics vertex attributes require "
                          "location, binding, format, and offset fields.");
              attributes.push_back(
                  {py::cast<std::uint32_t>(item[0]),
                   py::cast<std::uint32_t>(item[1]),
                   py::cast<BufferFormat>(item[2]),
                   py::cast<std::uint32_t>(item[3])});
            }
            auto vertex_spirv = decode_spirv(vertex_bytes, "vertex");
            auto fragment_spirv = decode_spirv(fragment_bytes, "fragment");
            py::gil_scoped_release release;
            return program->create_vulkan_graphics_pipeline(
                vertex_spirv, fragment_spirv, bindings, attributes, topology,
                polygon_mode, front_face_cull, back_face_cull, depth_test,
                depth_write, blending, name);
          },
          py::arg("vertex_spirv"), py::arg("fragment_spirv"),
          py::arg("vertex_bindings"), py::arg("vertex_attributes"),
          py::arg("topology"), py::arg("polygon_mode"),
          py::arg("front_face_cull"), py::arg("back_face_cull"),
          py::arg("depth_test"), py::arg("depth_write"),
          py::arg("blending"), py::arg("name"))
      .def(
          "_create_vulkan_mesh_pipeline",
          [](Program *program, py::bytes task_bytes, py::bytes mesh_bytes,
             py::bytes fragment_bytes, int topology, int polygon_mode,
             bool front_face_cull, bool back_face_cull, bool depth_test,
             bool depth_write, bool blending, const std::string &name) {
            auto decode_spirv = [](py::bytes value, const char *stage,
                                   bool optional) {
              const std::string bytes = py::cast<std::string>(value);
              TI_ERROR_IF((!optional && bytes.empty()) ||
                              bytes.size() % sizeof(std::uint32_t) != 0,
                          "Vulkan mesh pipeline {} SPIR-V must be {}and "
                          "four-byte aligned.",
                          stage, optional ? "optional " : "nonempty ");
              std::vector<std::uint32_t> words(
                  bytes.size() / sizeof(std::uint32_t));
              if (!bytes.empty()) {
                std::memcpy(words.data(), bytes.data(), bytes.size());
              }
              return words;
            };
            auto task_spirv = decode_spirv(task_bytes, "task", true);
            auto mesh_spirv = decode_spirv(mesh_bytes, "mesh", false);
            auto fragment_spirv =
                decode_spirv(fragment_bytes, "fragment", false);
            py::gil_scoped_release release;
            return program->create_vulkan_mesh_pipeline(
                task_spirv, mesh_spirv, fragment_spirv, topology,
                polygon_mode, front_face_cull, back_face_cull, depth_test,
                depth_write, blending, name);
          },
          py::arg("task_spirv"), py::arg("mesh_spirv"),
          py::arg("fragment_spirv"), py::arg("topology"),
          py::arg("polygon_mode"), py::arg("front_face_cull"),
          py::arg("back_face_cull"), py::arg("depth_test"),
          py::arg("depth_write"), py::arg("blending"), py::arg("name"))
      .def(
          "_vulkan_graphics_draw",
          [](Program *program, std::uint64_t handle, Texture *color,
             Texture *depth, const py::sequence &raw_vertex_buffers,
             Ndarray *index_buffer, std::uint32_t element_count,
             std::uint32_t instance_count, std::uint32_t first_vertex,
             std::uint32_t first_index, std::uint32_t first_instance,
             std::int32_t vertex_offset, std::uint32_t index_min,
             std::uint32_t index_max,
             bool indexed, const std::array<float, 4> &clear_color,
             const std::array<std::uint32_t, 4> &viewport) {
            std::vector<std::pair<std::uint32_t, Ndarray *>> vertex_buffers;
            vertex_buffers.reserve(raw_vertex_buffers.size());
            for (const py::handle raw : raw_vertex_buffers) {
              const py::tuple item = py::cast<py::tuple>(raw);
              TI_ERROR_IF(item.size() != 2,
                          "Vulkan graphics vertex-buffer bindings require "
                          "binding and ndarray fields.");
              vertex_buffers.emplace_back(
                  py::cast<std::uint32_t>(item[0]),
                  py::cast<Ndarray *>(item[1]));
            }
            VulkanGraphicsDrawInfo draw;
            draw.element_count = element_count;
            draw.instance_count = instance_count;
            draw.first_vertex = first_vertex;
            draw.first_index = first_index;
            draw.first_instance = first_instance;
            draw.vertex_offset = vertex_offset;
            draw.index_min = index_min;
            draw.index_max = index_max;
            draw.indexed = indexed;
            draw.clear_color = clear_color;
            draw.viewport = viewport;
            try {
              py::gil_scoped_release release;
              const auto result = program->vulkan_graphics_draw(
                  handle, color, depth, vertex_buffers, index_buffer, draw);
              program->record_runtime_submission_stat(
                  RuntimeSubmissionKind::kNative);
              return result;
            } catch (...) {
              program->record_runtime_submission_failure();
              throw;
            }
          },
          py::arg("handle"), py::arg("color"), py::arg("depth"),
          py::arg("vertex_buffers"), py::arg("index_buffer"),
          py::arg("element_count"), py::arg("instance_count"),
          py::arg("first_vertex"), py::arg("first_index"),
          py::arg("first_instance"), py::arg("vertex_offset"),
          py::arg("index_min"), py::arg("index_max"), py::arg("indexed"),
          py::arg("clear_color"), py::arg("viewport"))
      .def(
          "_vulkan_graphics_pass",
          [](Program *program, Texture *color, Texture *depth,
             const py::sequence &raw_draws, bool color_clear,
             bool depth_clear, const std::array<float, 4> &clear_color,
             const std::array<std::uint32_t, 4> &viewport,
             bool retained_replay) {
            std::vector<VulkanGraphicsDrawCommand> commands;
            commands.reserve(raw_draws.size());
            for (const py::handle raw : raw_draws) {
              const py::tuple item = py::cast<py::tuple>(raw);
              const bool mesh = item.size() == 8;
              const bool indirect = item.size() == 15;
              TI_ERROR_IF(item.size() != 13 && !indirect && !mesh,
                          "Vulkan graphics pass draws require either the "
                          "13-field direct, 15-field indirect, or 8-field "
                          "mesh ABI.");
              VulkanGraphicsDrawCommand command;
              command.pipeline_handle = py::cast<std::uint64_t>(item[0]);
              const py::sequence raw_vertex_buffers =
                  py::cast<py::sequence>(item[1]);
              command.vertex_buffers.reserve(raw_vertex_buffers.size());
              for (const py::handle raw_vertex : raw_vertex_buffers) {
                const py::tuple vertex = py::cast<py::tuple>(raw_vertex);
                TI_ERROR_IF(
                    vertex.size() != 2,
                    "Vulkan graphics vertex-buffer bindings require binding "
                    "and ndarray fields.");
                command.vertex_buffers.emplace_back(
                    py::cast<std::uint32_t>(vertex[0]),
                    py::cast<Ndarray *>(vertex[1]));
              }
              command.index_buffer = py::cast<Ndarray *>(item[2]);
              const py::sequence raw_shader_buffers =
                  py::cast<py::sequence>(item[3]);
              command.shader_buffers.reserve(raw_shader_buffers.size());
              for (const py::handle raw_shader : raw_shader_buffers) {
                const py::tuple shader = py::cast<py::tuple>(raw_shader);
                TI_ERROR_IF(
                    shader.size() != 4 && shader.size() != 5,
                    "Vulkan graphics shader-buffer bindings require either "
                    "the scalar or fixed-array descriptor ABI.");
                VulkanGraphicsShaderBufferBinding shader_binding;
                shader_binding.set_index =
                    py::cast<std::uint32_t>(shader[0]);
                shader_binding.binding =
                    py::cast<std::uint32_t>(shader[1]);
                shader_binding.storage = py::cast<bool>(shader[3]);
                if (shader.size() == 5) {
                  TI_ERROR_IF(!py::cast<bool>(shader[4]),
                              "Vulkan graphics fixed-array descriptor ABI "
                              "requires its array marker.");
                  const py::sequence elements =
                      py::cast<py::sequence>(shader[2]);
                  shader_binding.array_elements.reserve(elements.size());
                  for (const py::handle element : elements) {
                    shader_binding.array_elements.push_back(
                        py::cast<Ndarray *>(element));
                  }
                } else {
                  shader_binding.array = py::cast<Ndarray *>(shader[2]);
                }
                command.shader_buffers.push_back(
                    std::move(shader_binding));
              }
              if (mesh) {
                TI_ERROR_IF(!py::cast<bool>(item[7]),
                            "Vulkan graphics mesh ABI requires its mesh "
                            "marker.");
                command.mesh = VulkanGraphicsMeshDrawInfo{
                    py::cast<std::uint32_t>(item[4]),
                    py::cast<std::uint32_t>(item[5]),
                    py::cast<std::uint32_t>(item[6])};
              } else if (indirect) {
                VulkanGraphicsIndirectInfo indirect_info;
                indirect_info.command_buffer = py::cast<Ndarray *>(item[4]);
                indirect_info.count_buffer = py::cast<Ndarray *>(item[5]);
                indirect_info.command_offset =
                    py::cast<std::uint32_t>(item[6]);
                indirect_info.count_offset =
                    py::cast<std::uint32_t>(item[7]);
                indirect_info.max_draw_count =
                    py::cast<std::uint32_t>(item[8]);
                indirect_info.stride = py::cast<std::uint32_t>(item[9]);
                indirect_info.vertex_record_limit =
                    py::cast<std::uint32_t>(item[10]);
                indirect_info.instance_record_limit =
                    py::cast<std::uint32_t>(item[11]);
                indirect_info.index_element_limit =
                    py::cast<std::uint32_t>(item[12]);
                indirect_info.first_instance_may_be_nonzero =
                    py::cast<bool>(item[13]);
                command.draw.indexed = py::cast<bool>(item[14]);
                command.indirect = indirect_info;
              } else {
                command.draw.element_count =
                    py::cast<std::uint32_t>(item[4]);
                command.draw.instance_count =
                    py::cast<std::uint32_t>(item[5]);
                command.draw.first_vertex =
                    py::cast<std::uint32_t>(item[6]);
                command.draw.first_index =
                    py::cast<std::uint32_t>(item[7]);
                command.draw.first_instance =
                    py::cast<std::uint32_t>(item[8]);
                command.draw.vertex_offset =
                    py::cast<std::int32_t>(item[9]);
                command.draw.index_min =
                    py::cast<std::uint32_t>(item[10]);
                command.draw.index_max =
                    py::cast<std::uint32_t>(item[11]);
                command.draw.indexed = py::cast<bool>(item[12]);
              }
              commands.push_back(std::move(command));
            }
            VulkanGraphicsPassInfo pass;
            pass.color_clear = color_clear;
            pass.depth_clear = depth_clear;
            pass.retained_replay = retained_replay;
            pass.clear_color = clear_color;
            pass.viewport = viewport;
            try {
              py::gil_scoped_release release;
              const auto result = program->vulkan_graphics_pass(
                  color, depth, commands, pass);
              program->record_runtime_submission_stat(
                  RuntimeSubmissionKind::kNative);
              return result;
            } catch (...) {
              program->record_runtime_submission_failure();
              throw;
            }
          },
          py::arg("color"), py::arg("depth"), py::arg("draws"),
          py::arg("color_clear"), py::arg("depth_clear"),
          py::arg("clear_color"), py::arg("viewport"),
          py::arg("retained_replay"))
      .def("_destroy_vulkan_graphics_pipeline",
           &Program::destroy_vulkan_graphics_pipeline, py::arg("handle"),
           py::call_guard<py::gil_scoped_release>())
      .def("copy_ndarray_from_host",
           [](Program *program, Ndarray *dst, py::buffer src) {
             py::buffer_info info = src.request();
             const auto bytes = static_cast<std::size_t>(info.size) *
                                static_cast<std::size_t>(info.itemsize);
             py::gil_scoped_release release;
             program->copy_ndarray_from_host(dst, info.ptr, bytes);
           })
      .def("copy_ndarray_to_host",
           [](Program *program, Ndarray *src, py::buffer dst) {
             py::buffer_info info = dst.request();
             TI_ERROR_IF(info.readonly,
                         "copy_ndarray_to_host received a read-only buffer.");
             const auto bytes = static_cast<std::size_t>(info.size) *
                                static_cast<std::size_t>(info.itemsize);
             py::gil_scoped_release release;
             program->copy_ndarray_to_host(src, info.ptr, bytes);
           })
      .def("cuda_matrix_mma_f16_f32_available",
           &Program::cuda_matrix_mma_f16_f32_available)
      .def("_cuda_async_tile_status", [](Program *program) {
        py::dict result;
        result["provider_available"] =
            program->cuda_async_tile_available();
        auto statistics = program->cuda_async_tile_statistics();
        result["lowered_specializations"] =
            statistics.at("lowered_specializations");
        result["copy_sites"] = statistics.at("copy_sites");
        for (const auto *key :
             {"candidates", "admitted", "lowered", "fallback", "rejected",
              "below_size", "read_write_bls", "unsupported_width",
              "non_direct_address", "alias_unknown",
              "shared_memory_pressure", "target_capability", "cost_gate"}) {
          result[key] = statistics.at(key);
        }
        result["minimum_bls_bytes"] =
            taichi::lang::cuda::detail::kCudaAsyncTileMinBlsBytes;
#ifdef TI_WITH_CUDA
        if (program->compile_config().arch == Arch::cuda) {
          auto &context = CUDAContext::get_instance();
          result["device_compute_capability"] =
              context.get_compute_capability();
          result["codegen_compute_capability"] =
              context.get_codegen_compute_capability();
          result["ptx_version"] = context.get_ptx_version();
        }
#endif
        return result;
      })
      .def("_cuda_device_identity", [](Program *program) {
        py::dict result;
#ifdef TI_WITH_CUDA
        TI_ERROR_IF(program->compile_config().arch != Arch::cuda,
                    "CUDA device identity requires the CUDA backend");
        auto &context = CUDAContext::get_instance();
        result["device_name"] = context.get_device_name();
        result["device_compute_capability"] =
            context.get_compute_capability();
        result["codegen_compute_capability"] =
            context.get_codegen_compute_capability();
        result["target"] = context.get_mcpu() + "|" + context.get_mattrs();
        int multiprocessor_count = 0;
        CUDADriver::get_instance().device_get_attribute(
            &multiprocessor_count,
            CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            context.get_device());
        result["multiprocessor_count"] = multiprocessor_count;
#else
        TI_NOT_IMPLEMENTED
#endif
        return result;
      })
      .def("_cuda_matrix_mma_f16_f32",
           tracked_native_program_method(
               &Program::cuda_matrix_mma_f16_f32),
           py::arg("a"), py::arg("b"), py::arg("output"),
           py::arg("batch_count"), py::call_guard<py::gil_scoped_release>())
      .def("vulkan_ray_query_available",
           &Program::vulkan_ray_query_available)
      .def("_vulkan_ray_query_properties",
           &Program::vulkan_ray_query_properties)
      .def("vulkan_cooperative_matrix_available",
           &Program::vulkan_cooperative_matrix_available)
      .def("_vulkan_cooperative_matrix_properties",
           &Program::vulkan_cooperative_matrix_properties)
      .def("_create_vulkan_triangle_ray_scene",
           tracked_native_program_method(
               &Program::create_vulkan_triangle_ray_scene),
           py::arg("vertices"), py::arg("indices"), py::arg("vertex_count"),
           py::arg("triangle_count"),
           py::call_guard<py::gil_scoped_release>())
      .def("_vulkan_triangle_ray_query",
           tracked_native_program_method(
               &Program::vulkan_triangle_ray_query),
           py::arg("handle"), py::arg("rays"), py::arg("hits"),
           py::arg("ray_count"), py::arg("hit_indices") = nullptr,
           py::call_guard<py::gil_scoped_release>())
      .def("_prepare_vulkan_typed_ray_query",
           &Program::prepare_vulkan_typed_ray_query,
           py::arg("handle"), py::arg("instance_tlas"),
           py::call_guard<py::gil_scoped_release>())
      .def("_prepare_vulkan_ray_query", &Program::prepare_vulkan_ray_query,
           py::arg("handle"), py::arg("instance_tlas"), py::arg("rays"),
           py::arg("hits"), py::arg("ray_count"), py::arg("hit_indices") = nullptr,
           py::call_guard<py::gil_scoped_release>())
      .def("_execute_vulkan_ray_query",
           tracked_native_program_method(&Program::execute_vulkan_ray_query),
           py::call_guard<py::gil_scoped_release>())
      .def("_vulkan_triangle_ray_refit",
           tracked_native_program_method(
               &Program::vulkan_triangle_ray_refit),
           py::arg("handle"), py::arg("vertices"), py::arg("vertex_count"),
           py::call_guard<py::gil_scoped_release>())
      .def("_vulkan_triangle_ray_scene_memory_stats",
           [](Program &program, std::uint64_t handle) {
             const auto stats =
                 program.vulkan_triangle_ray_scene_memory_statistics(handle);
             py::dict result;
             result["geometry_input_requested_bytes"] =
                 stats.geometry_input_requested_bytes;
             result["acceleration_structure_requested_bytes"] =
                 stats.acceleration_structure_requested_bytes;
             result["build_scratch_requested_bytes"] =
                 stats.build_scratch_requested_bytes;
             result["known_requested_bytes"] = stats.known_requested_bytes;
             result["known_allocation_count"] = stats.known_allocation_count;
             result["opaque_driver_bytes"] = py::none();
             return result;
           },
           py::arg("handle"))
      .def("_destroy_vulkan_triangle_ray_scene",
           tracked_native_program_method(
               &Program::destroy_vulkan_triangle_ray_scene),
           py::arg("handle"), py::call_guard<py::gil_scoped_release>())
      .def("_create_vulkan_triangle_blas_resource",
           tracked_native_program_method(
               &Program::create_vulkan_triangle_blas_resource),
           py::arg("vertex_count"), py::arg("triangle_count"),
           py::call_guard<py::gil_scoped_release>())
      .def("_vulkan_triangle_blas_build",
           tracked_native_program_method(
               &Program::vulkan_triangle_blas_build),
           py::arg("handle"), py::arg("vertices"), py::arg("indices"),
           py::arg("vertex_count"), py::arg("triangle_count"),
           py::arg("update"), py::call_guard<py::gil_scoped_release>())
      .def("_create_vulkan_instance_tlas_resource",
           tracked_native_program_method(
               &Program::create_vulkan_instance_tlas_resource),
           py::arg("blas_handles"),
           py::call_guard<py::gil_scoped_release>())
      .def("_vulkan_instance_tlas_build",
           tracked_native_program_method(&Program::vulkan_instance_tlas_build),
           py::arg("handle"), py::arg("instances"), py::arg("update"),
           py::call_guard<py::gil_scoped_release>())
      .def("_vulkan_instance_tlas_query",
           tracked_native_program_method(&Program::vulkan_instance_tlas_query),
           py::arg("handle"), py::arg("rays"), py::arg("hits"),
           py::arg("ray_count"), py::arg("hit_indices") = nullptr,
           py::call_guard<py::gil_scoped_release>())
      .def("_vulkan_ray_resource_memory_stats",
           [](Program &program, std::uint64_t handle) {
             const auto stats =
                 program.vulkan_ray_resource_memory_statistics(handle);
             py::dict result;
             result["geometry_input_requested_bytes"] =
                 stats.geometry_input_requested_bytes;
             result["acceleration_structure_requested_bytes"] =
                 stats.acceleration_structure_requested_bytes;
             result["build_scratch_requested_bytes"] =
                 stats.build_scratch_requested_bytes;
             result["known_requested_bytes"] = stats.known_requested_bytes;
             result["known_allocation_count"] = stats.known_allocation_count;
             result["opaque_driver_bytes"] = py::none();
             return result;
           },
           py::arg("handle"))
      .def("_vulkan_ray_kernel_resource_properties",
           &Program::vulkan_ray_kernel_resource_properties,
           py::arg("handle"))
      .def("_destroy_vulkan_ray_resource",
           tracked_native_program_method(&Program::destroy_vulkan_ray_resource),
           py::arg("handle"), py::call_guard<py::gil_scoped_release>())
      .def("_create_vulkan_fft_plan",
           tracked_native_program_method(&Program::create_vulkan_fft_plan),
           py::arg("adapter_path"), py::arg("data"), py::arg("dimensions"),
           py::arg("batches"), py::arg("direction"), py::arg("normalize_inverse"),
           py::call_guard<py::gil_scoped_release>())
      .def("_create_vulkan_parallel_sort_plan",
           tracked_native_program_method(&Program::create_vulkan_parallel_sort_plan),
           py::arg("keys"), py::arg("values"), py::arg("shaders"),
           py::call_guard<py::gil_scoped_release>())
      .def("_create_vulkan_parallel_sort_plan_fused",
           tracked_native_program_method(&Program::create_vulkan_parallel_sort_plan),
           py::arg("keys"), py::arg("values"), py::arg("shaders"),
           py::call_guard<py::gil_scoped_release>())
      .def("_vulkan_parallel_sort_execute",
           &Program::vulkan_parallel_sort_execute,
           py::arg("handle"), py::call_guard<py::gil_scoped_release>())
      .def("_vulkan_parallel_sort_plan_statistics",
           &Program::vulkan_parallel_sort_plan_statistics, py::arg("handle"))
      .def("_destroy_vulkan_parallel_sort_plan",
           tracked_native_program_method(&Program::destroy_vulkan_parallel_sort_plan),
           py::arg("handle"), py::call_guard<py::gil_scoped_release>())
      .def("_vulkan_fft_execute",
           tracked_native_program_method(&Program::vulkan_fft_execute),
           py::arg("handle"), py::call_guard<py::gil_scoped_release>())
      .def("_create_vulkan_fft_recipe_plan",
           tracked_native_program_method(&Program::create_vulkan_fft_recipe_plan),
           py::arg("adapter_path"), py::arg("data"), py::arg("dimensions"),
           py::arg("batches"), py::arg("direction"), py::arg("normalize_inverse"),
           py::arg("batch_tile"), py::call_guard<py::gil_scoped_release>())
      .def("_vulkan_fft_plan_statistics", &Program::vulkan_fft_plan_statistics,
           py::arg("handle"))
      .def("_vulkan_fft_graph_command", &Program::vulkan_fft_graph_command,
           py::arg("handle"), py::arg("binding_name"))
      .def("_destroy_vulkan_fft_plan",
           tracked_native_program_method(&Program::destroy_vulkan_fft_plan),
           py::arg("handle"), py::call_guard<py::gil_scoped_release>())
      .def("_create_cuda_cufft_plan_1d",
           tracked_native_program_method(&Program::create_cuda_cufft_plan_1d),
           py::arg("length"), py::arg("batch_count"),
           py::arg("transform_kind") = 0,
           py::call_guard<py::gil_scoped_release>())
      .def("_create_cuda_cufft_plan_many",
           tracked_native_program_method(
               &Program::create_cuda_cufft_plan_many),
           py::arg("dimensions"), py::arg("input_embed"),
           py::arg("input_stride"), py::arg("input_distance"),
           py::arg("output_embed"), py::arg("output_stride"),
           py::arg("output_distance"), py::arg("batch_count"),
           py::arg("transform_kind"), py::arg("separable") = false,
           py::call_guard<py::gil_scoped_release>())
      .def("_create_cuda_cufft_cross_batch_plan",
           tracked_native_program_method(&Program::create_cuda_cufft_cross_batch_plan),
           py::arg("dimensions"), py::arg("batch_count"),
           py::call_guard<py::gil_scoped_release>())
      .def("_create_cuda_cufft_store_callback_plan",
           tracked_native_program_method(&Program::create_cuda_cufft_store_callback_plan),
           py::arg("dimensions"), py::arg("batch_count"),
           py::arg("lto_ir"), py::arg("symbol"),
           py::call_guard<py::gil_scoped_release>())
      .def("_cuda_cufft_execute",
           tracked_native_program_method(&Program::cuda_cufft_execute),
           py::arg("handle"), py::arg("input"), py::arg("output"),
           py::arg("direction"), py::call_guard<py::gil_scoped_release>())
      .def("_cuda_cufft_execute_c2c",
           tracked_native_program_method(&Program::cuda_cufft_execute_c2c),
           py::arg("handle"), py::arg("input"), py::arg("output"),
           py::arg("direction"), py::call_guard<py::gil_scoped_release>())
      .def("_destroy_cuda_cufft_plan",
           tracked_native_program_method(&Program::destroy_cuda_cufft_plan),
           py::arg("handle"), py::call_guard<py::gil_scoped_release>())
      .def("_cuda_cufft_plan_memory_statistics",
           &Program::cuda_cufft_plan_memory_statistics, py::arg("handle"),
           py::call_guard<py::gil_scoped_release>())
      .def("_cuda_cufft_plan_cache_statistics",
           &Program::cuda_cufft_plan_cache_statistics,
           py::call_guard<py::gil_scoped_release>())
      .def("_cuda_cublas_gemm_f32",
           tracked_native_program_method(&Program::cuda_cublas_gemm_f32),
           py::arg("a"), py::arg("b"), py::arg("output"), py::arg("rows"),
           py::arg("columns"), py::arg("inner"), py::arg("alpha"),
           py::arg("beta"), py::call_guard<py::gil_scoped_release>())
      .def("_create_cuda_cudss_plan",
           tracked_native_program_method(&Program::create_cuda_cudss_plan),
           py::arg("matrix"), py::arg("matrix_type"),
           py::arg("matrix_view"), py::arg("adapter_path"),
           py::arg("runtime_library_path") = "",
           py::call_guard<py::gil_scoped_release>())
      .def("_cuda_cudss_analyze",
           tracked_native_program_method(&Program::cuda_cudss_analyze),
           py::arg("handle"), py::arg("matrix"),
           py::call_guard<py::gil_scoped_release>())
      .def("_create_cuda_cudss_configured_plan",
           tracked_native_program_method(
               &Program::create_cuda_cudss_configured_plan),
           py::arg("matrix"), py::arg("matrix_type"), py::arg("matrix_view"),
           py::arg("adapter_path"), py::arg("runtime_library_path"),
           py::arg("configuration"), py::arg("graph_owned") = false,
           py::call_guard<py::gil_scoped_release>())
      .def("_cuda_cudss_plan_configuration",
           &Program::cuda_cudss_plan_configuration, py::arg("handle"),
           py::call_guard<py::gil_scoped_release>())
      .def("_cuda_cudss_factorize",
           tracked_native_program_method(&Program::cuda_cudss_factorize),
           py::arg("handle"), py::arg("matrix"),
           py::arg("refactorize") = false,
           py::call_guard<py::gil_scoped_release>())
      .def("_cuda_cudss_solve",
           tracked_native_program_method(&Program::cuda_cudss_solve),
           py::arg("handle"), py::arg("matrix"), py::arg("rhs"),
           py::arg("solution"),
           py::call_guard<py::gil_scoped_release>())
      .def("_cuda_cudss_refactor_solve",
           tracked_native_program_method(
               &Program::cuda_cudss_refactor_solve),
           py::arg("handle"), py::arg("values"), py::arg("rhs"),
           py::arg("solution"),
           py::call_guard<py::gil_scoped_release>())
      .def("_cuda_cudss_plan_statistics",
           &Program::cuda_cudss_plan_statistics, py::arg("handle"),
           py::call_guard<py::gil_scoped_release>())
      .def("_debug_cuda_cudss_fail_next_refactor_solve",
           &Program::debug_cuda_cudss_fail_next_refactor_solve,
           py::arg("handle"), py::call_guard<py::gil_scoped_release>())
      .def("_destroy_cuda_cudss_plan",
           tracked_native_program_method(&Program::destroy_cuda_cudss_plan),
           py::arg("handle"), py::call_guard<py::gil_scoped_release>())
      .def("copy_ndarrays_to_host",
           [](Program *program, const std::vector<Ndarray *> &srcs,
              const py::sequence &dsts) {
             TI_ERROR_IF(
                 srcs.size() != static_cast<std::size_t>(py::len(dsts)),
                 "copy_ndarrays_to_host received {} sources and {} "
                 "destinations.",
                 srcs.size(), py::len(dsts));
             std::vector<const Ndarray *> const_srcs(srcs.begin(), srcs.end());
             std::vector<py::buffer_info> infos;
             std::vector<void *> host_ptrs;
             std::vector<std::size_t> bytes;
             infos.reserve(srcs.size());
             host_ptrs.reserve(srcs.size());
             bytes.reserve(srcs.size());
             for (std::size_t i = 0; i < srcs.size(); ++i) {
               py::buffer dst =
                   py::reinterpret_borrow<py::buffer>(dsts[py::int_(i)]);
               infos.push_back(dst.request());
               const py::buffer_info &info = infos.back();
               TI_ERROR_IF(
                   info.readonly,
                   "copy_ndarrays_to_host received a read-only buffer at "
                   "index {}.",
                   i);
               TI_ERROR_IF(
                   info.size < 0 || info.itemsize < 0,
                   "copy_ndarrays_to_host received an invalid buffer at "
                   "index {}.",
                   i);
               host_ptrs.push_back(info.ptr);
               bytes.push_back(static_cast<std::size_t>(info.size) *
                               static_cast<std::size_t>(info.itemsize));
             }
             py::gil_scoped_release release;
             program->copy_ndarrays_to_host(
                 const_srcs.data(), host_ptrs.data(), bytes.data(),
                 const_srcs.size());
           })
      .def("copy_graph_observations_to_host",
           [](Program *program, const std::vector<Ndarray *> &srcs,
              const py::sequence &dsts) {
             TI_ERROR_IF(
                 srcs.size() != static_cast<std::size_t>(py::len(dsts)),
                 "copy_graph_observations_to_host received {} sources and {} "
                 "destinations.",
                 srcs.size(), py::len(dsts));
             std::vector<const Ndarray *> const_srcs(srcs.begin(), srcs.end());
             std::vector<py::buffer_info> infos;
             std::vector<void *> host_ptrs;
             std::vector<std::size_t> bytes;
             infos.reserve(srcs.size());
             host_ptrs.reserve(srcs.size());
             bytes.reserve(srcs.size());
             for (std::size_t i = 0; i < srcs.size(); ++i) {
               py::buffer dst =
                   py::reinterpret_borrow<py::buffer>(dsts[py::int_(i)]);
               infos.push_back(dst.request());
               const py::buffer_info &info = infos.back();
               TI_ERROR_IF(
                   info.readonly,
                   "copy_graph_observations_to_host received a read-only "
                   "buffer at index {}.",
                   i);
               TI_ERROR_IF(
                   info.size < 0 || info.itemsize < 0,
                   "copy_graph_observations_to_host received an invalid "
                   "buffer at index {}.",
                   i);
               host_ptrs.push_back(info.ptr);
               bytes.push_back(static_cast<std::size_t>(info.size) *
                               static_cast<std::size_t>(info.itemsize));
             }
             py::gil_scoped_release release;
             program->copy_graph_observations_to_host(
                 const_srcs.data(), host_ptrs.data(), bytes.data(),
                 const_srcs.size());
           })
      .def("_copy_host_readable_graph_observations_to_host",
           [](Program *program, const std::vector<Ndarray *> &srcs,
              const py::sequence &dsts) {
             TI_ERROR_IF(
                 srcs.size() != static_cast<std::size_t>(py::len(dsts)),
                 "Host-readable Graph observation received {} sources and {} "
                 "destinations.",
                 srcs.size(), py::len(dsts));
             std::vector<const Ndarray *> const_srcs(srcs.begin(), srcs.end());
             std::vector<py::buffer_info> infos;
             std::vector<void *> host_ptrs;
             std::vector<std::size_t> bytes;
             infos.reserve(srcs.size());
             host_ptrs.reserve(srcs.size());
             bytes.reserve(srcs.size());
             for (std::size_t i = 0; i < srcs.size(); ++i) {
               py::buffer dst =
                   py::reinterpret_borrow<py::buffer>(dsts[py::int_(i)]);
               infos.push_back(dst.request());
               const py::buffer_info &info = infos.back();
               TI_ERROR_IF(
                   info.readonly,
                   "Host-readable Graph observation received a read-only "
                   "buffer at index {}.",
                   i);
               TI_ERROR_IF(
                   info.size < 0 || info.itemsize < 0,
                   "Host-readable Graph observation received an invalid "
                   "buffer at index {}.",
                   i);
               host_ptrs.push_back(info.ptr);
               bytes.push_back(static_cast<std::size_t>(info.size) *
                               static_cast<std::size_t>(info.itemsize));
             }
             py::gil_scoped_release release;
             program->copy_host_readable_graph_observations_to_host(
                 const_srcs.data(), host_ptrs.data(), bytes.data(),
                 const_srcs.size());
           })
      .def("_create_cuda_graph_observation_readback",
           &Program::create_cuda_graph_observation_readback,
           py::arg("bytes"))
      .def("_destroy_cuda_graph_observation_readback",
           &Program::destroy_cuda_graph_observation_readback,
           py::arg("handle"))
      .def(
          "_enqueue_cuda_graph_observation_readback",
          [](Program *program, std::uint64_t handle,
             const std::vector<Ndarray *> &srcs) {
            std::vector<const Ndarray *> const_srcs(srcs.begin(), srcs.end());
            std::vector<std::size_t> bytes;
            bytes.reserve(srcs.size());
            for (const Ndarray *src : srcs) {
              TI_ERROR_IF(!src,
                          "CUDA Graph observation received a null source.");
              bytes.push_back(src->get_nelement() * src->get_element_size());
            }
            py::gil_scoped_release release;
            program->enqueue_cuda_graph_observation_readback(
                handle, const_srcs.data(), bytes.data(), const_srcs.size());
          },
          py::arg("handle"), py::arg("sources"))
      .def(
          "_copy_cuda_graph_observation_readback_to_host",
          [](Program *program, std::uint64_t handle,
             const py::sequence &dsts) {
            std::vector<py::buffer_info> infos;
            std::vector<void *> host_ptrs;
            std::vector<std::size_t> bytes;
            const std::size_t count = static_cast<std::size_t>(py::len(dsts));
            infos.reserve(count);
            host_ptrs.reserve(count);
            bytes.reserve(count);
            for (std::size_t i = 0; i < count; ++i) {
              py::buffer dst =
                  py::reinterpret_borrow<py::buffer>(dsts[py::int_(i)]);
              infos.push_back(dst.request());
              const py::buffer_info &info = infos.back();
              TI_ERROR_IF(info.readonly,
                          "CUDA Graph observation received a read-only "
                          "destination at index {}.",
                          i);
              TI_ERROR_IF(info.size < 0 || info.itemsize < 0,
                          "CUDA Graph observation received an invalid "
                          "destination at index {}.",
                          i);
              host_ptrs.push_back(info.ptr);
              bytes.push_back(static_cast<std::size_t>(info.size) *
                              static_cast<std::size_t>(info.itemsize));
            }
            py::gil_scoped_release release;
            program->copy_cuda_graph_observation_readback_to_host(
                handle, host_ptrs.data(), bytes.data(), count);
          },
          py::arg("handle"), py::arg("destinations"))
      .def("_graph_observation_staging_stats", [](Program *program) {
        const auto stats = program->graph_observation_staging_statistics();
        py::dict result;
        result["persistent_bytes"] = stats.persistent_bytes;
        result["allocations"] = stats.allocations;
        result["reuses"] = stats.reuses;
        result["packed_batches"] = stats.packed_batches;
        result["direct_batches"] = stats.direct_batches;
        result["fallback_batches"] = stats.fallback_batches;
        result["packed_payload_bytes"] = stats.packed_payload_bytes;
        result["completion_attached_batches"] =
            stats.completion_attached_batches;
        result["completion_attached_bytes"] =
            stats.completion_attached_bytes;
        return result;
      })
      .def("cuda_device_transform_available",
           &Program::cuda_device_transform_available)
      .def("cuda_toolkit_transform_available",
           &Program::cuda_toolkit_transform_available)
      .def("cuda_device_transform_affine_ndarray",
           tracked_native_program_method(&Program::cuda_device_transform_affine_ndarray), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("scale"),
           py::arg("bias"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_transform_affine_member_ndarray",
           tracked_native_program_method(&Program::cuda_device_transform_affine_member_ndarray),
           py::arg("src"), py::arg("dst"), py::arg("value_type"),
           py::arg("offset"), py::arg("stride"), py::arg("scale"),
           py::arg("bias"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_transform_affine_strided_ndarray",
           tracked_native_program_method(&Program::cuda_device_transform_affine_strided_ndarray),
           py::arg("src"), py::arg("dst"), py::arg("value_type"),
           py::arg("src_offset"), py::arg("src_stride"),
           py::arg("dst_offset"), py::arg("dst_stride"), py::arg("scale"),
           py::arg("bias"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_transform_affine_packed_strided_ndarray",
           tracked_native_program_method(&Program::cuda_device_transform_affine_packed_strided_ndarray),
           py::arg("src"), py::arg("dst"), py::arg("value_type"),
           py::arg("lane_count"), py::arg("src_offset"),
           py::arg("src_stride"), py::arg("dst_offset"),
           py::arg("dst_stride"), py::arg("scale"), py::arg("bias"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_transform_affine_dense_field",
           tracked_native_program_method(&Program::cuda_device_transform_affine_dense_field), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("n"),
           py::arg("scale"), py::arg("bias"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_zero_dense_field",
           tracked_native_program_method(&Program::cuda_device_zero_dense_field), py::arg("dst"),
           py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_add_merge_available",
           &Program::cuda_device_add_merge_available)
      .def("cuda_device_add_merge_ndarray",
           tracked_native_program_method(&Program::cuda_device_add_merge_ndarray), py::arg("src"),
           py::arg("dst"), py::arg("value_type"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_add_scaled_ndarray",
           tracked_native_program_method(&Program::cuda_device_add_scaled_ndarray), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("scale"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_add_scalar_ndarray_to_ndarray",
           tracked_native_program_method(&Program::cuda_device_add_scalar_ndarray_to_ndarray),
           py::arg("src"), py::arg("dst"), py::arg("value_type"),
           py::arg("scale"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_add_merge_strided_ndarray",
           tracked_native_program_method(&Program::cuda_device_add_merge_strided_ndarray), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("src_offset"),
           py::arg("src_stride"), py::arg("dst_offset"),
           py::arg("dst_stride"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_add_merge_dense_field",
           tracked_native_program_method(&Program::cuda_device_add_merge_dense_field), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_add_scaled_dense_field",
           tracked_native_program_method(&Program::cuda_device_add_scaled_dense_field), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("n"),
           py::arg("scale"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_add_scalar_field_to_dense_field",
           tracked_native_program_method(&Program::cuda_device_add_scalar_field_to_dense_field),
           py::arg("src"), py::arg("dst"), py::arg("value_type"),
           py::arg("n"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_indexed_copy_available",
           &Program::cuda_device_indexed_copy_available)
      .def("cuda_device_indexed_copy_payload_available",
           &Program::cuda_device_indexed_copy_payload_available,
           py::arg("item_bytes"))
      .def("cuda_device_gather_ndarray", tracked_native_program_method(&Program::cuda_device_gather_ndarray),
           py::arg("src"), py::arg("indices"), py::arg("dst"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_gather_strided_ndarray",
           tracked_native_program_method(&Program::cuda_device_gather_strided_ndarray), py::arg("src"),
           py::arg("indices"), py::arg("dst"), py::arg("item_bytes"),
           py::arg("src_offset"), py::arg("src_stride"),
           py::arg("dst_offset"), py::arg("dst_stride"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_gather_dense_field",
           tracked_native_program_method(&Program::cuda_device_gather_dense_field), py::arg("src"),
           py::arg("indices"), py::arg("dst"), py::arg("value_type"),
           py::arg("src_n"), py::arg("dst_n"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_gather_dense_field_packed",
           tracked_native_program_method(&Program::cuda_device_gather_dense_field_packed), py::arg("src"),
           py::arg("indices"), py::arg("dst"), py::arg("value_type"),
           py::arg("src_n"), py::arg("dst_n"), py::arg("lane_count"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_gather_dense_field_packed_indices_field",
           tracked_native_program_method(&Program::cuda_device_gather_dense_field_packed_indices_field),
           py::arg("src"), py::arg("indices"), py::arg("dst"),
           py::arg("value_type"), py::arg("src_n"), py::arg("indices_n"),
           py::arg("dst_n"), py::arg("lane_count"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_gather_dense_field_indices_field",
           tracked_native_program_method(&Program::cuda_device_gather_dense_field_indices_field),
           py::arg("src"), py::arg("indices"), py::arg("dst"),
           py::arg("value_type"), py::arg("src_n"), py::arg("indices_n"),
           py::arg("dst_n"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_gather_add_ndarray",
           tracked_native_program_method(&Program::cuda_device_gather_add_ndarray), py::arg("src"),
           py::arg("indices"), py::arg("dst"), py::arg("value_type"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_gather_add_dense_field",
           tracked_native_program_method(&Program::cuda_device_gather_add_dense_field), py::arg("src"),
           py::arg("indices"), py::arg("dst"), py::arg("value_type"),
           py::arg("src_n"), py::arg("dst_n"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_gather_add_dense_field_indices_field",
           tracked_native_program_method(&Program::cuda_device_gather_add_dense_field_indices_field),
           py::arg("src"), py::arg("indices"), py::arg("dst"),
           py::arg("value_type"), py::arg("src_n"), py::arg("indices_n"),
           py::arg("dst_n"), py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_scatter_ndarray",
            tracked_native_program_method(&Program::cuda_device_scatter_ndarray), py::arg("src"),
            py::arg("indices"), py::arg("dst"),
            py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_scatter_strided_ndarray",
            tracked_native_program_method(&Program::cuda_device_scatter_strided_ndarray), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("item_bytes"),
            py::arg("src_offset"), py::arg("src_stride"),
            py::arg("dst_offset"), py::arg("dst_stride"),
            py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_scatter_dense_field",
            tracked_native_program_method(&Program::cuda_device_scatter_dense_field), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("value_type"),
            py::arg("src_n"), py::arg("dst_n"),
            py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_scatter_dense_field_packed",
            tracked_native_program_method(&Program::cuda_device_scatter_dense_field_packed), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("value_type"),
            py::arg("src_n"), py::arg("dst_n"), py::arg("lane_count"),
            py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_scatter_dense_field_packed_indices_field",
            tracked_native_program_method(&Program::cuda_device_scatter_dense_field_packed_indices_field),
            py::arg("src"), py::arg("indices"), py::arg("dst"),
            py::arg("value_type"), py::arg("src_n"), py::arg("indices_n"),
            py::arg("dst_n"), py::arg("lane_count"),
            py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_scatter_dense_field_indices_field",
            tracked_native_program_method(&Program::cuda_device_scatter_dense_field_indices_field),
            py::arg("src"), py::arg("indices"), py::arg("dst"),
            py::arg("value_type"), py::arg("src_n"), py::arg("indices_n"),
            py::arg("dst_n"), py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_scatter_add_available",
            &Program::cuda_device_scatter_add_available)
       .def("cuda_device_scatter_add_ndarray",
            tracked_native_program_method(&Program::cuda_device_scatter_add_ndarray), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("value_type"),
            py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_scatter_add_member_ndarray",
            tracked_native_program_method(&Program::cuda_device_scatter_add_member_ndarray), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("value_type"),
            py::arg("offset"), py::arg("stride"),
            py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_scatter_add_strided_ndarray",
            tracked_native_program_method(&Program::cuda_device_scatter_add_strided_ndarray), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("value_type"),
            py::arg("src_offset"), py::arg("src_stride"),
            py::arg("dst_offset"), py::arg("dst_stride"),
            py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_scatter_add_dense_field",
            tracked_native_program_method(&Program::cuda_device_scatter_add_dense_field), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("value_type"),
            py::arg("src_n"), py::arg("dst_n"),
            py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_scatter_add_dense_field_indices_field",
            tracked_native_program_method(&Program::cuda_device_scatter_add_dense_field_indices_field),
            py::arg("src"), py::arg("indices"), py::arg("dst"),
            py::arg("value_type"), py::arg("src_n"), py::arg("indices_n"),
            py::arg("dst_n"), py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_bucket_builder_available",
            &Program::cuda_device_bucket_builder_available)
       .def("cuda_device_bucket_builder_i32_ndarray",
            tracked_native_program_method(&Program::cuda_device_bucket_builder_i32_ndarray), py::arg("keys"),
            py::arg("values"), py::arg("offsets"), py::arg("output"),
            py::arg("cursor"), py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_bucket_builder_ndarray",
            tracked_native_program_method(&Program::cuda_device_bucket_builder_ndarray), py::arg("keys"),
            py::arg("values"), py::arg("offsets"), py::arg("output"),
            py::arg("cursor"), py::arg("value_type"),
            py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_bucket_builder_dense_field",
            tracked_native_program_method(&Program::cuda_device_bucket_builder_dense_field), py::arg("keys"),
            py::arg("values"), py::arg("offsets"), py::arg("output"),
            py::arg("cursor"), py::arg("value_type"), py::arg("n"),
            py::arg("num_bins"), py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_grouped_reduce_available",
            &Program::cuda_device_grouped_reduce_available)
       .def("cuda_device_grouped_reduce_atomic_ndarray",
            tracked_native_program_method(&Program::cuda_device_grouped_reduce_atomic_ndarray),
            py::arg("keys"), py::arg("values"), py::arg("output"),
            py::arg("value_type"), py::arg("op"),
            py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_grouped_reduce_atomic_dense_field",
            tracked_native_program_method(&Program::cuda_device_grouped_reduce_atomic_dense_field),
            py::arg("keys"), py::arg("values"), py::arg("output"),
            py::arg("value_type"), py::arg("n"), py::arg("num_groups"),
            py::arg("op"), py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_grouped_reduce_atomic_member_ndarray",
            tracked_native_program_method(&Program::cuda_device_grouped_reduce_atomic_member_ndarray),
            py::arg("keys"), py::arg("values"), py::arg("output"),
            py::arg("value_type"), py::arg("offset"), py::arg("stride"),
            py::arg("op"), py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_grouped_reduce_atomic_strided_ndarray",
            tracked_native_program_method(&Program::cuda_device_grouped_reduce_atomic_strided_ndarray),
            py::arg("keys"), py::arg("values"), py::arg("output"),
            py::arg("value_type"), py::arg("values_offset"),
            py::arg("values_stride"), py::arg("output_offset"),
            py::arg("output_stride"), py::arg("op"),
            py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_grouped_reduce_atomic_strided_keys_ndarray",
            tracked_native_program_method(&Program::cuda_device_grouped_reduce_atomic_strided_keys_ndarray),
            py::arg("keys"), py::arg("values"), py::arg("output"),
            py::arg("value_type"), py::arg("keys_offset"),
            py::arg("keys_stride"), py::arg("values_offset"),
            py::arg("values_stride"), py::arg("output_offset"),
            py::arg("output_stride"), py::arg("op"),
            py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_grouped_reduce_i32_atomic_ndarray",
            tracked_native_program_method(&Program::cuda_device_grouped_reduce_i32_atomic_ndarray),
            py::arg("keys"), py::arg("values"), py::arg("output"),
            py::arg("op"), py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_grouped_reduce_i32_ndarray",
            tracked_native_program_method(&Program::cuda_device_grouped_reduce_i32_ndarray), py::arg("keys"),
            py::arg("values"), py::arg("output"), py::arg("offsets"),
            py::arg("scratch"), py::arg("cursor"), py::arg("op"),
            py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_grouped_reduce_ndarray",
            tracked_native_program_method(&Program::cuda_device_grouped_reduce_ndarray), py::arg("keys"),
            py::arg("values"), py::arg("output"), py::arg("offsets"),
            py::arg("scratch"), py::arg("cursor"), py::arg("value_type"),
            py::arg("op"), py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_grouped_reduce_segmented_strided_keys_ndarray",
            tracked_native_program_method(&Program::cuda_device_grouped_reduce_segmented_strided_keys_ndarray),
            py::arg("keys"), py::arg("values"), py::arg("output"),
            py::arg("offsets"), py::arg("scratch"), py::arg("cursor"),
            py::arg("value_type"), py::arg("keys_offset"),
            py::arg("keys_stride"), py::arg("values_offset"),
            py::arg("values_stride"), py::arg("output_offset"),
            py::arg("output_stride"), py::arg("op"),
            py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_radix_sort_available",
           &Program::cuda_device_radix_sort_available)
      .def("cuda_device_radix_sort_clear_workspace",
           &Program::cuda_device_radix_sort_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_radix_sort_workspace_bytes",
           &Program::cuda_device_radix_sort_workspace_bytes)
      .def("cuda_device_radix_sort_ndarray",
           tracked_native_program_method(&Program::cuda_device_radix_sort_ndarray),
           py::arg("keys"), py::arg("values"), py::arg("key_type"),
           py::arg("value_type"), py::arg("nan_policy"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_radix_sort_keys_ndarray",
           [](Program *program, Ndarray *keys, int key_type, int nan_policy) {
             return program->cuda_device_radix_sort_ndarray(
                 keys, nullptr, key_type, 0, nan_policy);
           },
           py::arg("keys"), py::arg("key_type"), py::arg("nan_policy"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_radix_sort_dense_field",
           tracked_native_program_method(&Program::cuda_device_radix_sort_dense_field),
           py::arg("keys"), py::arg("values"), py::arg("key_type"),
           py::arg("value_type"), py::arg("n"), py::arg("nan_policy"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_radix_sort_keys_dense_field",
           [](Program *program, SNode *keys, int key_type, std::size_t n,
              int nan_policy) {
             return program->cuda_device_radix_sort_dense_field(
                 keys, nullptr, key_type, 0, n, nan_policy);
           },
           py::arg("keys"), py::arg("key_type"), py::arg("n"),
           py::arg("nan_policy"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_radix_sort_available",
           &Program::cuda_cub_radix_sort_available)
      .def("cuda_cub_radix_sort_clear_workspace",
           &Program::cuda_cub_radix_sort_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_radix_sort_workspace_bytes",
           &Program::cuda_cub_radix_sort_workspace_bytes)
      .def("cuda_cub_radix_sort_ndarray",
           tracked_native_program_method(&Program::cuda_cub_radix_sort_ndarray), py::arg("keys"),
           py::arg("values"), py::arg("key_type"), py::arg("value_type"),
           py::arg("mode"), py::arg("nan_policy"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_radix_sort_keys_ndarray",
           [](Program *program, Ndarray *keys, int key_type, int mode,
              int nan_policy) {
             return program->cuda_cub_radix_sort_ndarray(keys, nullptr,
                                                         key_type, 0, mode,
                                                         nan_policy);
           },
           py::arg("keys"), py::arg("key_type"), py::arg("mode"),
           py::arg("nan_policy"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_radix_sort_dense_field",
           tracked_native_program_method(&Program::cuda_cub_radix_sort_dense_field), py::arg("keys"),
           py::arg("values"), py::arg("key_type"), py::arg("value_type"),
           py::arg("n"), py::arg("mode"), py::arg("nan_policy"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_radix_sort_keys_dense_field",
           [](Program *program, SNode *keys, int key_type, std::size_t n,
              int mode, int nan_policy) {
             return program->cuda_cub_radix_sort_dense_field(
                 keys, nullptr, key_type, 0, n, mode, nan_policy);
           },
           py::arg("keys"), py::arg("key_type"), py::arg("n"),
           py::arg("mode"), py::arg("nan_policy"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_stable_sort_available", &Program::cpu_stable_sort_available)
      .def("cpu_stable_sort_ndarray", tracked_native_program_method(&Program::cpu_stable_sort_ndarray),
           py::arg("keys"), py::arg("values"), py::arg("key_type"),
           py::arg("value_type"), py::arg("descending"),
           py::arg("nan_policy"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_stable_sort_keys_ndarray",
           [](Program *program, Ndarray *keys, int key_type, bool descending,
              int nan_policy) {
             return program->cpu_stable_sort_ndarray(
                 keys, nullptr, key_type, 0, descending, nan_policy);
           },
           py::arg("keys"), py::arg("key_type"), py::arg("descending"),
           py::arg("nan_policy"), py::call_guard<py::gil_scoped_release>())
      .def("cpu_stable_sort_dense_field",
           tracked_native_program_method(&Program::cpu_stable_sort_dense_field), py::arg("keys"),
           py::arg("values"), py::arg("key_type"), py::arg("value_type"),
           py::arg("n"), py::arg("descending"), py::arg("nan_policy"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_stable_sort_keys_dense_field",
           [](Program *program, SNode *keys, int key_type, std::size_t n,
              bool descending, int nan_policy) {
             return program->cpu_stable_sort_dense_field(
                 keys, nullptr, key_type, 0, n, descending, nan_policy);
           },
           py::arg("keys"), py::arg("key_type"), py::arg("n"),
           py::arg("descending"), py::arg("nan_policy"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_scan_available", &Program::cuda_device_scan_available)
      .def("cuda_device_scan_clear_workspace",
           &Program::cuda_device_scan_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_scan_workspace_bytes",
           &Program::cuda_device_scan_workspace_bytes)
      .def("cuda_device_inclusive_scan_ndarray",
           tracked_native_program_method(&Program::cuda_device_inclusive_scan_ndarray),
           py::arg("data"), py::arg("value_type"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_inclusive_reverse_scan_ndarray",
           tracked_native_program_method(&Program::cuda_device_inclusive_reverse_scan_ndarray),
           py::arg("data"), py::arg("value_type"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_inclusive_scan_member_ndarray",
           tracked_native_program_method(&Program::cuda_device_inclusive_scan_member_ndarray),
           py::arg("data"), py::arg("value_type"), py::arg("offset"),
           py::arg("stride"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_inclusive_reverse_scan_member_ndarray",
           tracked_native_program_method(&Program::cuda_device_inclusive_reverse_scan_member_ndarray),
           py::arg("data"), py::arg("value_type"), py::arg("offset"),
           py::arg("stride"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_inclusive_scan_dense_field",
           tracked_native_program_method(&Program::cuda_device_inclusive_scan_dense_field),
           py::arg("data"), py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_inclusive_reverse_scan_dense_field",
           tracked_native_program_method(&Program::cuda_device_inclusive_reverse_scan_dense_field),
           py::arg("data"), py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_inclusive_scan_dense_field_packed",
           tracked_native_program_method(&Program::cuda_device_inclusive_scan_dense_field_packed),
           py::arg("data"), py::arg("value_type"), py::arg("n"),
           py::arg("lane_count"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_inclusive_reverse_scan_dense_field_packed",
           tracked_native_program_method(&Program::cuda_device_inclusive_reverse_scan_dense_field_packed),
           py::arg("data"), py::arg("value_type"), py::arg("n"),
           py::arg("lane_count"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_scan_available", &Program::cuda_cub_scan_available)
      .def("cuda_cub_scan_clear_workspace",
           &Program::cuda_cub_scan_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_scan_workspace_bytes",
           &Program::cuda_cub_scan_workspace_bytes)
      .def("cuda_cub_inclusive_scan_ndarray",
           tracked_native_program_method(&Program::cuda_cub_inclusive_scan_ndarray), py::arg("data"),
           py::arg("value_type"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_inclusive_reverse_scan_ndarray",
           tracked_native_program_method(&Program::cuda_cub_inclusive_reverse_scan_ndarray), py::arg("data"),
           py::arg("value_type"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_inclusive_scan_member_ndarray",
           tracked_native_program_method(&Program::cuda_cub_inclusive_scan_member_ndarray), py::arg("data"),
           py::arg("value_type"), py::arg("offset"), py::arg("stride"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_inclusive_reverse_scan_member_ndarray",
           tracked_native_program_method(&Program::cuda_cub_inclusive_reverse_scan_member_ndarray),
           py::arg("data"), py::arg("value_type"), py::arg("offset"),
           py::arg("stride"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_inclusive_scan_dense_field",
           tracked_native_program_method(&Program::cuda_cub_inclusive_scan_dense_field), py::arg("data"),
           py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_inclusive_reverse_scan_dense_field",
           tracked_native_program_method(&Program::cuda_cub_inclusive_reverse_scan_dense_field),
           py::arg("data"), py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_inclusive_scan_dense_field_packed",
           tracked_native_program_method(&Program::cuda_cub_inclusive_scan_dense_field_packed),
           py::arg("data"), py::arg("value_type"), py::arg("n"),
           py::arg("lane_count"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_inclusive_reverse_scan_dense_field_packed",
           tracked_native_program_method(&Program::cuda_cub_inclusive_reverse_scan_dense_field_packed),
           py::arg("data"), py::arg("value_type"), py::arg("n"),
           py::arg("lane_count"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_compact_available",
           &Program::cuda_device_compact_available)
      .def("cuda_device_compact_ndarray",
           tracked_native_program_method(&Program::cuda_device_compact_ndarray),
           py::arg("values"), py::arg("flags"), py::arg("output"),
           py::arg("count"), py::arg("value_type"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_compact_dense_field",
           tracked_native_program_method(
               &Program::cuda_device_compact_dense_field),
           py::arg("values"), py::arg("flags"), py::arg("output"),
           py::arg("count"), py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_compact_clear_workspace",
           &Program::cuda_device_compact_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_compact_workspace_bytes",
           &Program::cuda_device_compact_workspace_bytes)
      .def("cuda_cub_select_available", &Program::cuda_cub_select_available)
      .def("cuda_cub_select_clear_workspace",
           &Program::cuda_cub_select_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_select_workspace_bytes",
           &Program::cuda_cub_select_workspace_bytes)
      .def("cuda_cub_select_ndarray", tracked_native_program_method(&Program::cuda_cub_select_ndarray),
           py::arg("values"), py::arg("flags"), py::arg("output"),
           py::arg("count"), py::arg("value_type"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_select_dense_field",
           tracked_native_program_method(&Program::cuda_cub_select_dense_field), py::arg("values"),
           py::arg("flags"), py::arg("output"), py::arg("count"),
           py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_select_i32_ndarray",
           tracked_native_program_method(&Program::cuda_cub_select_i32_ndarray), py::arg("values"),
           py::arg("flags"), py::arg("output"), py::arg("count"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_histogram_available",
           &Program::cuda_device_histogram_available)
      .def("cuda_device_histogram_ndarray",
           tracked_native_program_method(&Program::cuda_device_histogram_ndarray),
           py::arg("values"), py::arg("bins"), py::arg("value_type"),
           py::arg("bin_type") = 0,
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_histogram_dense_field",
           tracked_native_program_method(&Program::cuda_device_histogram_dense_field),
           py::arg("values"), py::arg("bins"), py::arg("value_type"),
           py::arg("bin_type"), py::arg("n"), py::arg("num_bins"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_histogram_clear_workspace",
           &Program::cuda_device_histogram_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_histogram_workspace_bytes",
           &Program::cuda_device_histogram_workspace_bytes)
      .def("cuda_cub_histogram_available",
           &Program::cuda_cub_histogram_available)
      .def("cuda_cub_histogram_clear_workspace",
           &Program::cuda_cub_histogram_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_histogram_workspace_bytes",
           &Program::cuda_cub_histogram_workspace_bytes)
      .def("cuda_cub_histogram_ndarray",
           tracked_native_program_method(&Program::cuda_cub_histogram_ndarray), py::arg("values"),
           py::arg("bins"), py::arg("value_type"), py::arg("bin_type") = 0,
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_histogram_i32_ndarray",
           tracked_native_program_method(&Program::cuda_cub_histogram_i32_ndarray), py::arg("values"),
           py::arg("bins"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_histogram_dense_field",
           tracked_native_program_method(&Program::cuda_cub_histogram_dense_field), py::arg("values"),
           py::arg("bins"), py::arg("value_type"), py::arg("bin_type"),
           py::arg("n"), py::arg("num_bins"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_reduce_available",
           &Program::cuda_device_reduce_available)
      .def("cuda_device_reduce_clear_workspace",
           &Program::cuda_device_reduce_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_reduce_workspace_bytes",
           &Program::cuda_device_reduce_workspace_bytes)
      .def("cuda_device_reduce_ndarray",
           tracked_native_program_method(&Program::cuda_device_reduce_ndarray),
           py::arg("values"), py::arg("output"), py::arg("value_type"),
           py::arg("op"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_reduce_member_ndarray",
           tracked_native_program_method(&Program::cuda_device_reduce_member_ndarray),
           py::arg("values"), py::arg("output"), py::arg("value_type"),
           py::arg("offset"), py::arg("stride"), py::arg("op"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_reduce_strided_ndarray",
           tracked_native_program_method(&Program::cuda_device_reduce_strided_ndarray),
           py::arg("values"), py::arg("output"), py::arg("value_type"),
           py::arg("values_offset"), py::arg("values_stride"),
           py::arg("output_offset"), py::arg("output_stride"), py::arg("op"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_reduce_dense_field",
           tracked_native_program_method(&Program::cuda_device_reduce_dense_field),
           py::arg("values"), py::arg("output"), py::arg("value_type"),
           py::arg("n"), py::arg("op"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_reduce_dense_field_packed",
           tracked_native_program_method(&Program::cuda_device_reduce_dense_field_packed),
           py::arg("values"), py::arg("output"), py::arg("value_type"),
           py::arg("n"), py::arg("lane_count"), py::arg("op"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_reduce_available", &Program::cuda_cub_reduce_available)
      .def("cuda_cub_reduce_clear_workspace",
           &Program::cuda_cub_reduce_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_reduce_workspace_bytes",
           &Program::cuda_cub_reduce_workspace_bytes)
      .def("cuda_device_check_count_available",
           &Program::cuda_device_check_count_available)
      .def("cuda_device_check_count_clear_workspace",
           &Program::cuda_device_check_count_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_check_count_workspace_bytes",
           &Program::cuda_device_check_count_workspace_bytes)
      .def("cuda_device_check_count_ndarray",
           tracked_native_program_method(&Program::cuda_device_check_count_ndarray), py::arg("values"),
           py::arg("output"), py::arg("value_type"), py::arg("check_op"),
           py::arg("lower"), py::arg("upper"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_check_count_strided_ndarray",
           tracked_native_program_method(&Program::cuda_device_check_count_strided_ndarray), py::arg("values"),
           py::arg("output"), py::arg("value_type"), py::arg("offset"),
           py::arg("stride"), py::arg("check_op"), py::arg("lower"),
           py::arg("upper"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_check_count_dense_field",
           tracked_native_program_method(&Program::cuda_device_check_count_dense_field), py::arg("values"),
           py::arg("output"), py::arg("value_type"), py::arg("n"),
           py::arg("check_op"), py::arg("lower"), py::arg("upper"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_metric_reduce_available",
           &Program::cuda_device_metric_reduce_available)
      .def("cuda_device_metric_reduce_value_type_available",
           &Program::cuda_device_metric_reduce_value_type_available,
           py::arg("value_type"))
      .def("cuda_device_metric_reduce_clear_workspace",
           &Program::cuda_device_metric_reduce_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_metric_reduce_workspace_bytes",
           &Program::cuda_device_metric_reduce_workspace_bytes)
      .def("cuda_device_metric_reduce_ndarray",
           tracked_native_program_method(&Program::cuda_device_metric_reduce_ndarray), py::arg("values"),
           py::arg("other"), py::arg("output"), py::arg("value_type"),
           py::arg("metric_op"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_metric_reduce_strided_ndarray",
           tracked_native_program_method(&Program::cuda_device_metric_reduce_strided_ndarray), py::arg("values"),
           py::arg("other"), py::arg("output"), py::arg("value_type"),
           py::arg("values_offset"), py::arg("values_stride"),
           py::arg("other_offset"), py::arg("other_stride"),
           py::arg("metric_op"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_metric_reduce_dense_field",
           tracked_native_program_method(&Program::cuda_device_metric_reduce_dense_field), py::arg("values"),
           py::arg("other"), py::arg("output"), py::arg("value_type"),
           py::arg("n"), py::arg("metric_op"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_metric_reduce_dense_field_strided_ndarray",
           tracked_native_program_method(&Program::cuda_device_metric_reduce_dense_field_strided_ndarray),
           py::arg("field"), py::arg("array"), py::arg("output"),
           py::arg("value_type"), py::arg("n"), py::arg("array_offset"),
           py::arg("array_stride"), py::arg("field_is_values"),
           py::arg("metric_op"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_check_count_available",
           &Program::cuda_cub_check_count_available)
      .def("cuda_cub_check_count_clear_workspace",
           &Program::cuda_cub_check_count_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_check_count_workspace_bytes",
           &Program::cuda_cub_check_count_workspace_bytes)
      .def("cuda_cub_check_count_ndarray",
           tracked_native_program_method(&Program::cuda_cub_check_count_ndarray), py::arg("values"),
           py::arg("output"), py::arg("value_type"), py::arg("check_op"),
           py::arg("lower"), py::arg("upper"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_check_count_strided_ndarray",
           tracked_native_program_method(&Program::cuda_cub_check_count_strided_ndarray), py::arg("values"),
           py::arg("output"), py::arg("value_type"), py::arg("offset"),
           py::arg("stride"), py::arg("check_op"), py::arg("lower"),
           py::arg("upper"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_check_count_dense_field",
           tracked_native_program_method(&Program::cuda_cub_check_count_dense_field), py::arg("values"),
           py::arg("output"), py::arg("value_type"), py::arg("n"),
           py::arg("check_op"), py::arg("lower"), py::arg("upper"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_metric_reduce_available",
           &Program::cuda_cub_metric_reduce_available)
      .def("cuda_cub_metric_reduce_value_type_available",
           &Program::cuda_cub_metric_reduce_value_type_available,
           py::arg("value_type"))
      .def("cuda_cub_metric_reduce_clear_workspace",
           &Program::cuda_cub_metric_reduce_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_metric_reduce_workspace_bytes",
           &Program::cuda_cub_metric_reduce_workspace_bytes)
      .def("cuda_cub_metric_reduce_ndarray",
           tracked_native_program_method(&Program::cuda_cub_metric_reduce_ndarray), py::arg("values"),
           py::arg("other"), py::arg("output"), py::arg("value_type"),
           py::arg("metric_op"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_metric_reduce_strided_ndarray",
           tracked_native_program_method(&Program::cuda_cub_metric_reduce_strided_ndarray), py::arg("values"),
           py::arg("other"), py::arg("output"), py::arg("value_type"),
           py::arg("values_offset"), py::arg("values_stride"),
           py::arg("other_offset"), py::arg("other_stride"),
           py::arg("metric_op"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_metric_reduce_dense_field",
           tracked_native_program_method(&Program::cuda_cub_metric_reduce_dense_field), py::arg("values"),
           py::arg("other"), py::arg("output"), py::arg("value_type"),
           py::arg("n"), py::arg("metric_op"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_metric_reduce_dense_field_strided_ndarray",
           tracked_native_program_method(&Program::cuda_cub_metric_reduce_dense_field_strided_ndarray),
           py::arg("field"), py::arg("array"), py::arg("output"),
           py::arg("value_type"), py::arg("n"), py::arg("array_offset"),
           py::arg("array_stride"), py::arg("field_is_values"),
           py::arg("metric_op"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_reduce_ndarray", tracked_native_program_method(&Program::cuda_cub_reduce_ndarray),
           py::arg("values"), py::arg("output"), py::arg("value_type"),
           py::arg("op"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_reduce_member_ndarray",
           tracked_native_program_method(&Program::cuda_cub_reduce_member_ndarray), py::arg("values"),
           py::arg("output"), py::arg("value_type"), py::arg("offset"),
           py::arg("stride"), py::arg("op"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_reduce_strided_ndarray",
           tracked_native_program_method(&Program::cuda_cub_reduce_strided_ndarray), py::arg("values"),
           py::arg("output"), py::arg("value_type"),
           py::arg("values_offset"), py::arg("values_stride"),
           py::arg("output_offset"), py::arg("output_stride"), py::arg("op"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_reduce_dense_field", tracked_native_program_method(&Program::cuda_cub_reduce_dense_field),
           py::arg("values"), py::arg("output"), py::arg("value_type"),
           py::arg("n"), py::arg("op"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_reduce_dense_field_packed",
           tracked_native_program_method(&Program::cuda_cub_reduce_dense_field_packed), py::arg("values"),
           py::arg("output"), py::arg("value_type"), py::arg("n"),
           py::arg("lane_count"), py::arg("op"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_scan_available", &Program::cpu_scan_available)
      .def("cpu_scan_workspace_bytes", &Program::cpu_scan_workspace_bytes)
      .def("cpu_inclusive_scan_ndarray",
           tracked_native_program_method(&Program::cpu_inclusive_scan_ndarray), py::arg("data"),
           py::arg("value_type"), py::call_guard<py::gil_scoped_release>())
      .def("cpu_inclusive_reverse_scan_ndarray",
           tracked_native_program_method(&Program::cpu_inclusive_reverse_scan_ndarray), py::arg("data"),
           py::arg("value_type"), py::call_guard<py::gil_scoped_release>())
      .def("cpu_inclusive_scan_member_ndarray",
           tracked_native_program_method(&Program::cpu_inclusive_scan_member_ndarray), py::arg("data"),
           py::arg("value_type"), py::arg("offset"), py::arg("stride"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_inclusive_reverse_scan_member_ndarray",
           tracked_native_program_method(&Program::cpu_inclusive_reverse_scan_member_ndarray),
           py::arg("data"), py::arg("value_type"), py::arg("offset"),
           py::arg("stride"), py::call_guard<py::gil_scoped_release>())
      .def("cpu_inclusive_scan_dense_field",
           tracked_native_program_method(&Program::cpu_inclusive_scan_dense_field), py::arg("data"),
           py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_inclusive_reverse_scan_dense_field",
           tracked_native_program_method(&Program::cpu_inclusive_reverse_scan_dense_field), py::arg("data"),
           py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_inclusive_scan_dense_field_packed",
           tracked_native_program_method(&Program::cpu_inclusive_scan_dense_field_packed), py::arg("data"),
           py::arg("value_type"), py::arg("n"), py::arg("lane_count"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_inclusive_reverse_scan_dense_field_packed",
           tracked_native_program_method(&Program::cpu_inclusive_reverse_scan_dense_field_packed),
           py::arg("data"), py::arg("value_type"), py::arg("n"),
           py::arg("lane_count"), py::call_guard<py::gil_scoped_release>())
      .def("cpu_compact_available", &Program::cpu_compact_available)
      .def("cpu_compact_workspace_bytes", &Program::cpu_compact_workspace_bytes)
      .def("fill_dense_field", &Program::fill_dense_field, py::arg("dst"),
           py::arg("value_type"), py::arg("value_bits"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("fill_dense_field_packed", &Program::fill_dense_field_packed,
           py::arg("dst"), py::arg("value_type"), py::arg("value_bits"),
           py::arg("n"), py::arg("lane_count"),
           py::call_guard<py::gil_scoped_release>())
      .def("transform_affine_dense_field_packed",
           &Program::transform_affine_dense_field_packed, py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("n"),
           py::arg("lane_count"), py::arg("scale"), py::arg("bias"),
           py::call_guard<py::gil_scoped_release>())
      .def("copy_dense_field", &Program::copy_dense_field, py::arg("dst"),
           py::arg("src"), py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("copy_dense_field_packed", &Program::copy_dense_field_packed,
           py::arg("dst"), py::arg("src"), py::arg("value_type"),
           py::arg("n"), py::arg("lane_count"),
           py::call_guard<py::gil_scoped_release>())
      .def("_copy_dense_field_to_ndarray",
           tracked_native_program_method(
               &Program::copy_dense_field_to_ndarray),
           py::arg("dst"), py::arg("src"), py::arg("value_type"),
           py::arg("n"), py::arg("lane_count"),
           py::call_guard<py::gil_scoped_release>())
      .def("_copy_ndarray_to_dense_field",
           tracked_native_program_method(
               &Program::copy_ndarray_to_dense_field),
           py::arg("dst"), py::arg("src"), py::arg("value_type"),
           py::arg("n"), py::arg("lane_count"),
           py::call_guard<py::gil_scoped_release>())
      .def(
          "copy_dense_field_from_host",
          [](Program &program, SNode *dst, py::buffer src, int value_type,
             std::size_t n) {
            py::buffer_info info = src.request();
            const auto src_ptr =
                reinterpret_cast<std::uintptr_t>(info.ptr);
            const std::size_t src_bytes = checked_buffer_nbytes(
                info,
                "Native dense field host copy received an invalid source "
                "buffer.",
                "Native dense field host copy received an oversized source "
                "buffer.");
            py::gil_scoped_release release;
            program.copy_dense_field_from_host(dst, src_ptr, src_bytes,
                                               value_type, n);
          },
          py::arg("dst"), py::arg("src"), py::arg("value_type"), py::arg("n"))
      .def(
          "copy_dense_field_packed_from_host",
          [](Program &program, SNode *dst, py::buffer src, int value_type,
             std::size_t n, int lane_count) {
            py::buffer_info info = src.request();
            const auto src_ptr =
                reinterpret_cast<std::uintptr_t>(info.ptr);
            const std::size_t src_bytes = checked_buffer_nbytes(
                info,
                "Native packed dense field host copy received an invalid "
                "source buffer.",
                "Native packed dense field host copy received an oversized "
                "source buffer.");
            py::gil_scoped_release release;
            program.copy_dense_field_packed_from_host(
                dst, src_ptr, src_bytes, value_type, n, lane_count);
          },
          py::arg("dst"), py::arg("src"), py::arg("value_type"), py::arg("n"),
          py::arg("lane_count"))
      .def(
          "copy_dense_field_to_host",
          [](Program &program, SNode *src, py::buffer dst, int value_type,
             std::size_t n) {
            py::buffer_info info = dst.request();
            const auto dst_ptr = reinterpret_cast<std::uintptr_t>(info.ptr);
            const std::size_t dst_bytes = checked_buffer_nbytes(
                info,
                "Native dense field host readback received an invalid "
                "destination buffer.",
                "Native dense field host readback received an oversized "
                "destination buffer.");
            py::gil_scoped_release release;
            program.copy_dense_field_to_host(src, dst_ptr, dst_bytes,
                                             value_type, n);
          },
          py::arg("src"), py::arg("dst"), py::arg("value_type"), py::arg("n"))
      .def(
          "copy_dense_field_packed_to_host",
          [](Program &program, SNode *src, py::buffer dst, int value_type,
             std::size_t n, int lane_count) {
            py::buffer_info info = dst.request();
            const auto dst_ptr = reinterpret_cast<std::uintptr_t>(info.ptr);
            const std::size_t dst_bytes = checked_buffer_nbytes(
                info,
                "Native packed dense field host readback received an invalid "
                "destination buffer.",
                "Native packed dense field host readback received an "
                "oversized destination buffer.");
            py::gil_scoped_release release;
            program.copy_dense_field_packed_to_host(
                src, dst_ptr, dst_bytes, value_type, n, lane_count);
          },
          py::arg("src"), py::arg("dst"), py::arg("value_type"),
          py::arg("n"), py::arg("lane_count"))
      .def("add_merge_dense_field_packed",
           &Program::add_merge_dense_field_packed, py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("n"),
           py::arg("lane_count"), py::call_guard<py::gil_scoped_release>())
      .def("scatter_add_dense_field_packed",
           &Program::scatter_add_dense_field_packed, py::arg("src"),
           py::arg("indices"), py::arg("dst"), py::arg("value_type"),
           py::arg("src_n"), py::arg("dst_n"), py::arg("lane_count"),
           py::call_guard<py::gil_scoped_release>())
      .def("scatter_add_dense_field_packed_indices_field",
           &Program::scatter_add_dense_field_packed_indices_field,
           py::arg("src"), py::arg("indices"), py::arg("dst"),
           py::arg("value_type"), py::arg("src_n"), py::arg("indices_n"),
           py::arg("dst_n"), py::arg("lane_count"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_compact_ndarray", tracked_native_program_method(&Program::cpu_compact_ndarray),
           py::arg("values"), py::arg("flags"), py::arg("output"),
           py::arg("count"), py::arg("value_type"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_compact_dense_field", tracked_native_program_method(&Program::cpu_compact_dense_field),
           py::arg("values"), py::arg("flags"), py::arg("output"),
           py::arg("count"), py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_compact_i32_ndarray",
           tracked_native_program_method(&Program::cpu_compact_i32_ndarray), py::arg("values"),
           py::arg("flags"), py::arg("output"), py::arg("count"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_histogram_available", &Program::cpu_histogram_available)
      .def("cpu_histogram_workspace_bytes",
           &Program::cpu_histogram_workspace_bytes)
      .def("cpu_histogram_ndarray", tracked_native_program_method(&Program::cpu_histogram_ndarray),
           py::arg("values"), py::arg("bins"), py::arg("value_type"),
           py::arg("bin_type") = 0,
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_histogram_i32_ndarray",
           tracked_native_program_method(&Program::cpu_histogram_i32_ndarray), py::arg("values"),
           py::arg("bins"), py::call_guard<py::gil_scoped_release>())
      .def("cpu_histogram_dense_field", tracked_native_program_method(&Program::cpu_histogram_dense_field),
           py::arg("values"), py::arg("bins"), py::arg("value_type"),
           py::arg("bin_type"), py::arg("n"), py::arg("num_bins"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_reduce_available", &Program::cpu_reduce_available)
      .def("cpu_reduce_workspace_bytes", &Program::cpu_reduce_workspace_bytes)
      .def("cpu_check_count_available", &Program::cpu_check_count_available)
      .def("cpu_check_count_workspace_bytes",
           &Program::cpu_check_count_workspace_bytes)
      .def("cpu_check_count_ndarray", tracked_native_program_method(&Program::cpu_check_count_ndarray),
           py::arg("values"), py::arg("output"), py::arg("value_type"),
           py::arg("check_op"), py::arg("lower"), py::arg("upper"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_check_count_strided_ndarray",
           tracked_native_program_method(&Program::cpu_check_count_strided_ndarray), py::arg("values"),
           py::arg("output"), py::arg("value_type"), py::arg("offset"),
           py::arg("stride"), py::arg("check_op"), py::arg("lower"),
           py::arg("upper"), py::call_guard<py::gil_scoped_release>())
      .def("cpu_check_count_dense_field", tracked_native_program_method(&Program::cpu_check_count_dense_field),
           py::arg("values"), py::arg("output"), py::arg("value_type"),
           py::arg("n"), py::arg("check_op"), py::arg("lower"),
           py::arg("upper"), py::call_guard<py::gil_scoped_release>())
      .def("cpu_metric_reduce_available", &Program::cpu_metric_reduce_available)
      .def("cpu_metric_reduce_value_type_available",
           &Program::cpu_metric_reduce_value_type_available,
           py::arg("value_type"))
      .def("cpu_metric_reduce_workspace_bytes",
           &Program::cpu_metric_reduce_workspace_bytes)
      .def("cpu_metric_reduce_ndarray", tracked_native_program_method(&Program::cpu_metric_reduce_ndarray),
           py::arg("values"), py::arg("other"), py::arg("output"),
           py::arg("value_type"), py::arg("metric_op"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_metric_reduce_strided_ndarray",
           tracked_native_program_method(&Program::cpu_metric_reduce_strided_ndarray), py::arg("values"),
           py::arg("other"), py::arg("output"), py::arg("value_type"),
           py::arg("values_offset"), py::arg("values_stride"),
           py::arg("other_offset"), py::arg("other_stride"),
           py::arg("metric_op"), py::call_guard<py::gil_scoped_release>())
      .def("cpu_metric_reduce_dense_field",
           tracked_native_program_method(&Program::cpu_metric_reduce_dense_field), py::arg("values"),
           py::arg("other"), py::arg("output"), py::arg("value_type"),
           py::arg("n"), py::arg("metric_op"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_metric_reduce_dense_field_strided_ndarray",
           tracked_native_program_method(&Program::cpu_metric_reduce_dense_field_strided_ndarray),
           py::arg("field"), py::arg("array"), py::arg("output"),
           py::arg("value_type"), py::arg("n"), py::arg("array_offset"),
           py::arg("array_stride"), py::arg("field_is_values"),
           py::arg("metric_op"), py::call_guard<py::gil_scoped_release>())
      .def("cpu_reduce_ndarray", tracked_native_program_method(&Program::cpu_reduce_ndarray),
           py::arg("values"), py::arg("output"), py::arg("value_type"),
           py::arg("op"), py::call_guard<py::gil_scoped_release>())
      .def("cpu_reduce_member_ndarray",
           tracked_native_program_method(&Program::cpu_reduce_member_ndarray), py::arg("values"),
           py::arg("output"), py::arg("value_type"), py::arg("offset"),
           py::arg("stride"), py::arg("op"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_reduce_strided_ndarray", tracked_native_program_method(&Program::cpu_reduce_strided_ndarray),
           py::arg("values"), py::arg("output"), py::arg("value_type"),
           py::arg("values_offset"), py::arg("values_stride"),
           py::arg("output_offset"), py::arg("output_stride"), py::arg("op"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_reduce_dense_field", tracked_native_program_method(&Program::cpu_reduce_dense_field),
           py::arg("values"), py::arg("output"), py::arg("value_type"),
           py::arg("n"), py::arg("op"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_reduce_dense_field_packed",
           tracked_native_program_method(&Program::cpu_reduce_dense_field_packed), py::arg("values"),
           py::arg("output"), py::arg("value_type"), py::arg("n"),
           py::arg("lane_count"), py::arg("op"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_transform_available", &Program::cpu_transform_available)
      .def("cpu_transform_workspace_bytes",
           &Program::cpu_transform_workspace_bytes)
      .def("cpu_transform_affine_ndarray",
           tracked_native_program_method(&Program::cpu_transform_affine_ndarray), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("scale"),
           py::arg("bias"), py::call_guard<py::gil_scoped_release>())
      .def("cpu_transform_affine_member_ndarray",
           tracked_native_program_method(&Program::cpu_transform_affine_member_ndarray), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("offset"),
           py::arg("stride"), py::arg("scale"), py::arg("bias"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_transform_affine_strided_ndarray",
           tracked_native_program_method(&Program::cpu_transform_affine_strided_ndarray), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("src_offset"),
           py::arg("src_stride"), py::arg("dst_offset"),
           py::arg("dst_stride"), py::arg("scale"), py::arg("bias"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_transform_affine_packed_strided_ndarray",
           tracked_native_program_method(&Program::cpu_transform_affine_packed_strided_ndarray),
           py::arg("src"), py::arg("dst"), py::arg("value_type"),
           py::arg("lane_count"), py::arg("src_offset"),
           py::arg("src_stride"), py::arg("dst_offset"),
           py::arg("dst_stride"), py::arg("scale"), py::arg("bias"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_transform_affine_dense_field",
           tracked_native_program_method(&Program::cpu_transform_affine_dense_field), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("n"),
           py::arg("scale"), py::arg("bias"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_add_merge_available", &Program::cpu_add_merge_available)
      .def("cpu_add_merge_ndarray", tracked_native_program_method(&Program::cpu_add_merge_ndarray),
           py::arg("src"), py::arg("dst"), py::arg("value_type"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_add_scaled_ndarray", tracked_native_program_method(&Program::cpu_add_scaled_ndarray),
           py::arg("src"), py::arg("dst"), py::arg("value_type"),
           py::arg("scale"), py::call_guard<py::gil_scoped_release>())
      .def("cpu_add_scalar_ndarray_to_ndarray",
           tracked_native_program_method(&Program::cpu_add_scalar_ndarray_to_ndarray), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("scale"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_add_merge_strided_ndarray",
           tracked_native_program_method(&Program::cpu_add_merge_strided_ndarray), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("src_offset"),
           py::arg("src_stride"), py::arg("dst_offset"),
           py::arg("dst_stride"), py::call_guard<py::gil_scoped_release>())
      .def("cpu_add_merge_dense_field",
           tracked_native_program_method(&Program::cpu_add_merge_dense_field), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_add_scaled_dense_field",
           tracked_native_program_method(&Program::cpu_add_scaled_dense_field), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("n"),
           py::arg("scale"), py::call_guard<py::gil_scoped_release>())
      .def("cpu_add_scalar_field_to_dense_field",
           tracked_native_program_method(&Program::cpu_add_scalar_field_to_dense_field), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_indexed_copy_available", &Program::cpu_indexed_copy_available)
      .def("cpu_indexed_copy_workspace_bytes",
           &Program::cpu_indexed_copy_workspace_bytes)
      .def("cpu_gather_ndarray", tracked_native_program_method(&Program::cpu_gather_ndarray),
           py::arg("src"), py::arg("indices"), py::arg("dst"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_gather_strided_ndarray",
           tracked_native_program_method(&Program::cpu_gather_strided_ndarray), py::arg("src"),
           py::arg("indices"), py::arg("dst"), py::arg("item_bytes"),
           py::arg("src_offset"), py::arg("src_stride"),
           py::arg("dst_offset"), py::arg("dst_stride"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_gather_dense_field", tracked_native_program_method(&Program::cpu_gather_dense_field),
           py::arg("src"), py::arg("indices"), py::arg("dst"),
           py::arg("value_type"), py::arg("src_n"), py::arg("dst_n"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_gather_dense_field_packed",
           tracked_native_program_method(&Program::cpu_gather_dense_field_packed), py::arg("src"),
           py::arg("indices"), py::arg("dst"), py::arg("value_type"),
           py::arg("src_n"), py::arg("dst_n"), py::arg("lane_count"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_gather_dense_field_packed_indices_field",
           tracked_native_program_method(&Program::cpu_gather_dense_field_packed_indices_field),
           py::arg("src"), py::arg("indices"), py::arg("dst"),
           py::arg("value_type"), py::arg("src_n"), py::arg("indices_n"),
           py::arg("dst_n"), py::arg("lane_count"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_gather_dense_field_indices_field",
           tracked_native_program_method(&Program::cpu_gather_dense_field_indices_field), py::arg("src"),
           py::arg("indices"), py::arg("dst"), py::arg("value_type"),
           py::arg("src_n"), py::arg("indices_n"), py::arg("dst_n"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_gather_add_ndarray", tracked_native_program_method(&Program::cpu_gather_add_ndarray),
           py::arg("src"), py::arg("indices"), py::arg("dst"),
           py::arg("value_type"), py::call_guard<py::gil_scoped_release>())
      .def("cpu_gather_add_dense_field",
           tracked_native_program_method(&Program::cpu_gather_add_dense_field), py::arg("src"),
           py::arg("indices"), py::arg("dst"), py::arg("value_type"),
           py::arg("src_n"), py::arg("dst_n"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_gather_add_dense_field_indices_field",
           tracked_native_program_method(&Program::cpu_gather_add_dense_field_indices_field),
           py::arg("src"), py::arg("indices"), py::arg("dst"),
           py::arg("value_type"), py::arg("src_n"), py::arg("indices_n"),
           py::arg("dst_n"), py::call_guard<py::gil_scoped_release>())
       .def("cpu_scatter_ndarray", tracked_native_program_method(&Program::cpu_scatter_ndarray),
            py::arg("src"), py::arg("indices"), py::arg("dst"),
            py::call_guard<py::gil_scoped_release>())
       .def("cpu_scatter_strided_ndarray",
            tracked_native_program_method(&Program::cpu_scatter_strided_ndarray), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("item_bytes"),
            py::arg("src_offset"), py::arg("src_stride"),
            py::arg("dst_offset"), py::arg("dst_stride"),
            py::call_guard<py::gil_scoped_release>())
       .def("cpu_scatter_dense_field", tracked_native_program_method(&Program::cpu_scatter_dense_field),
            py::arg("src"), py::arg("indices"), py::arg("dst"),
            py::arg("value_type"), py::arg("src_n"), py::arg("dst_n"),
            py::call_guard<py::gil_scoped_release>())
       .def("cpu_scatter_dense_field_packed",
            tracked_native_program_method(&Program::cpu_scatter_dense_field_packed), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("value_type"),
            py::arg("src_n"), py::arg("dst_n"), py::arg("lane_count"),
            py::call_guard<py::gil_scoped_release>())
       .def("cpu_scatter_dense_field_packed_indices_field",
            tracked_native_program_method(&Program::cpu_scatter_dense_field_packed_indices_field),
            py::arg("src"), py::arg("indices"), py::arg("dst"),
            py::arg("value_type"), py::arg("src_n"), py::arg("indices_n"),
            py::arg("dst_n"), py::arg("lane_count"),
            py::call_guard<py::gil_scoped_release>())
       .def("cpu_scatter_dense_field_indices_field",
            tracked_native_program_method(&Program::cpu_scatter_dense_field_indices_field), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("value_type"),
            py::arg("src_n"), py::arg("indices_n"), py::arg("dst_n"),
            py::call_guard<py::gil_scoped_release>())
       .def("cpu_scatter_add_available", &Program::cpu_scatter_add_available)
       .def("cpu_scatter_add_workspace_bytes",
            &Program::cpu_scatter_add_workspace_bytes)
       .def("cpu_scatter_add_clear_workspace",
            &Program::cpu_scatter_add_clear_workspace,
            py::call_guard<py::gil_scoped_release>())
       .def("cpu_scatter_add_ndarray", tracked_native_program_method(&Program::cpu_scatter_add_ndarray),
            py::arg("src"), py::arg("indices"), py::arg("dst"),
            py::arg("value_type"), py::call_guard<py::gil_scoped_release>())
       .def("cpu_scatter_add_member_ndarray",
            tracked_native_program_method(&Program::cpu_scatter_add_member_ndarray), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("value_type"),
            py::arg("offset"), py::arg("stride"),
            py::call_guard<py::gil_scoped_release>())
       .def("cpu_scatter_add_strided_ndarray",
            tracked_native_program_method(&Program::cpu_scatter_add_strided_ndarray), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("value_type"),
            py::arg("src_offset"), py::arg("src_stride"),
            py::arg("dst_offset"), py::arg("dst_stride"),
            py::call_guard<py::gil_scoped_release>())
       .def("cpu_scatter_add_dense_field",
            tracked_native_program_method(&Program::cpu_scatter_add_dense_field), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("value_type"),
            py::arg("src_n"), py::arg("dst_n"),
            py::call_guard<py::gil_scoped_release>())
       .def("cpu_scatter_add_dense_field_indices_field",
            tracked_native_program_method(&Program::cpu_scatter_add_dense_field_indices_field),
            py::arg("src"), py::arg("indices"), py::arg("dst"),
            py::arg("value_type"), py::arg("src_n"), py::arg("indices_n"),
            py::arg("dst_n"), py::call_guard<py::gil_scoped_release>())
       .def("cpu_bucket_builder_available",
            &Program::cpu_bucket_builder_available)
       .def("cpu_bucket_builder_workspace_bytes",
            &Program::cpu_bucket_builder_workspace_bytes)
       .def("cpu_bucket_builder_i32_ndarray",
            tracked_native_program_method(&Program::cpu_bucket_builder_i32_ndarray), py::arg("keys"),
            py::arg("values"), py::arg("offsets"), py::arg("output"),
            py::call_guard<py::gil_scoped_release>())
       .def("cpu_bucket_builder_ndarray",
            tracked_native_program_method(&Program::cpu_bucket_builder_ndarray), py::arg("keys"),
            py::arg("values"), py::arg("offsets"), py::arg("output"),
            py::arg("value_type"), py::call_guard<py::gil_scoped_release>())
       .def("cpu_bucket_builder_dense_field",
            tracked_native_program_method(&Program::cpu_bucket_builder_dense_field), py::arg("keys"),
            py::arg("values"), py::arg("offsets"), py::arg("output"),
            py::arg("value_type"), py::arg("n"), py::arg("num_bins"),
            py::call_guard<py::gil_scoped_release>())
       .def("cpu_grouped_reduce_available",
            &Program::cpu_grouped_reduce_available)
       .def("cpu_grouped_reduce_ndarray", tracked_native_program_method(&Program::cpu_grouped_reduce_ndarray),
            py::arg("keys"), py::arg("values"), py::arg("output"),
            py::arg("value_type"), py::arg("op"),
            py::call_guard<py::gil_scoped_release>())
       .def("cpu_grouped_reduce_dense_field",
            tracked_native_program_method(&Program::cpu_grouped_reduce_dense_field), py::arg("keys"),
            py::arg("values"), py::arg("output"), py::arg("value_type"),
            py::arg("n"), py::arg("num_groups"), py::arg("op"),
            py::call_guard<py::gil_scoped_release>())
       .def("cpu_grouped_reduce_member_ndarray",
            tracked_native_program_method(&Program::cpu_grouped_reduce_member_ndarray), py::arg("keys"),
            py::arg("values"), py::arg("output"), py::arg("value_type"),
            py::arg("offset"), py::arg("stride"), py::arg("op"),
            py::call_guard<py::gil_scoped_release>())
       .def("cpu_grouped_reduce_strided_ndarray",
            tracked_native_program_method(&Program::cpu_grouped_reduce_strided_ndarray), py::arg("keys"),
            py::arg("values"), py::arg("output"), py::arg("value_type"),
            py::arg("values_offset"), py::arg("values_stride"),
            py::arg("output_offset"), py::arg("output_stride"),
            py::arg("op"), py::call_guard<py::gil_scoped_release>())
       .def("cpu_grouped_reduce_strided_keys_ndarray",
            tracked_native_program_method(&Program::cpu_grouped_reduce_strided_keys_ndarray), py::arg("keys"),
            py::arg("values"), py::arg("output"), py::arg("value_type"),
            py::arg("keys_offset"), py::arg("keys_stride"),
            py::arg("values_offset"), py::arg("values_stride"),
            py::arg("output_offset"), py::arg("output_stride"),
            py::arg("op"), py::call_guard<py::gil_scoped_release>())
       .def("cpu_grouped_reduce_workspace_bytes",
            &Program::cpu_grouped_reduce_workspace_bytes)
       .def("cpu_grouped_reduce_clear_workspace",
            &Program::cpu_grouped_reduce_clear_workspace,
            py::call_guard<py::gil_scoped_release>())
       .def("cpu_grouped_reduce_i32_ndarray",
            tracked_native_program_method(&Program::cpu_grouped_reduce_i32_ndarray), py::arg("keys"),
            py::arg("values"), py::arg("output"), py::arg("op"),
            py::call_guard<py::gil_scoped_release>())
      .def("vulkan_radix_sort_available",
           &Program::vulkan_radix_sort_available)
      .def("vulkan_radix_sort_clear_workspace",
           &Program::vulkan_radix_sort_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_radix_sort_workspace_bytes",
           &Program::vulkan_radix_sort_workspace_bytes)
      .def("vulkan_radix_sort_cpu_profile_clear",
           &Program::vulkan_radix_sort_cpu_profile_clear)
      .def("vulkan_radix_sort_cpu_profile_report",
           &Program::vulkan_radix_sort_cpu_profile_report)
      .def("vulkan_scan_available", &Program::vulkan_scan_available)
      .def("vulkan_scan_value_type_available",
           &Program::vulkan_scan_value_type_available, py::arg("value_type"))
      .def("vulkan_scan_clear_workspace",
           &Program::vulkan_scan_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_scan_workspace_bytes",
           &Program::vulkan_scan_workspace_bytes)
      .def("vulkan_inclusive_scan_ndarray",
           tracked_native_program_method(&Program::vulkan_inclusive_scan_ndarray), py::arg("data"),
           py::arg("value_type"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_inclusive_reverse_scan_ndarray",
           tracked_native_program_method(&Program::vulkan_inclusive_reverse_scan_ndarray), py::arg("data"),
           py::arg("value_type"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_inclusive_scan_member_ndarray",
           tracked_native_program_method(&Program::vulkan_inclusive_scan_member_ndarray), py::arg("data"),
           py::arg("value_type"), py::arg("offset"), py::arg("stride"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_inclusive_reverse_scan_member_ndarray",
           tracked_native_program_method(&Program::vulkan_inclusive_reverse_scan_member_ndarray),
           py::arg("data"), py::arg("value_type"), py::arg("offset"),
           py::arg("stride"), py::call_guard<py::gil_scoped_release>())
      .def("vulkan_inclusive_scan_dense_field",
           tracked_native_program_method(&Program::vulkan_inclusive_scan_dense_field), py::arg("data"),
           py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_inclusive_reverse_scan_dense_field",
           tracked_native_program_method(&Program::vulkan_inclusive_reverse_scan_dense_field),
           py::arg("data"), py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_inclusive_scan_dense_field_packed",
           tracked_native_program_method(&Program::vulkan_inclusive_scan_dense_field_packed), py::arg("data"),
           py::arg("value_type"), py::arg("n"), py::arg("lane_count"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_inclusive_reverse_scan_dense_field_packed",
           tracked_native_program_method(&Program::vulkan_inclusive_reverse_scan_dense_field_packed),
           py::arg("data"), py::arg("value_type"), py::arg("n"),
           py::arg("lane_count"), py::call_guard<py::gil_scoped_release>())
      .def("vulkan_compact_available", &Program::vulkan_compact_available)
      .def("vulkan_compact_clear_workspace",
           &Program::vulkan_compact_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_compact_workspace_bytes",
           &Program::vulkan_compact_workspace_bytes)
      .def("vulkan_compact_ndarray", tracked_native_program_method(&Program::vulkan_compact_ndarray),
           py::arg("values"), py::arg("flags"), py::arg("output"),
           py::arg("count"), py::arg("value_type"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_compact_ndarray_bounded",
           tracked_native_program_method(
               &Program::vulkan_compact_ndarray_bounded),
           py::arg("values"), py::arg("flags"), py::arg("output"),
           py::arg("count"), py::arg("value_type"),
           py::arg("dispatch_packet"), py::arg("block_dim"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_compact_dense_field", tracked_native_program_method(&Program::vulkan_compact_dense_field),
           py::arg("values"), py::arg("flags"), py::arg("output"),
           py::arg("count"), py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_compact_i32_ndarray",
           tracked_native_program_method(&Program::vulkan_compact_i32_ndarray), py::arg("values"),
           py::arg("flags"), py::arg("output"), py::arg("count"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_histogram_available",
           &Program::vulkan_histogram_available)
      .def("vulkan_histogram_clear_workspace",
           &Program::vulkan_histogram_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_histogram_workspace_bytes",
           &Program::vulkan_histogram_workspace_bytes)
      .def("vulkan_histogram_value_type_available",
           &Program::vulkan_histogram_value_type_available,
           py::arg("value_type"), py::arg("bin_type") = 0)
      .def("vulkan_histogram_ndarray", tracked_native_program_method(&Program::vulkan_histogram_ndarray),
           py::arg("values"), py::arg("bins"), py::arg("value_type"),
           py::arg("bin_type") = 0,
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_histogram_i32_ndarray",
           tracked_native_program_method(&Program::vulkan_histogram_i32_ndarray), py::arg("values"),
           py::arg("bins"), py::call_guard<py::gil_scoped_release>())
      .def("vulkan_histogram_dense_field",
           tracked_native_program_method(&Program::vulkan_histogram_dense_field), py::arg("values"),
           py::arg("bins"), py::arg("value_type"), py::arg("bin_type"),
           py::arg("n"), py::arg("num_bins"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_reduce_available", &Program::vulkan_reduce_available)
      .def("vulkan_reduce_clear_workspace",
           &Program::vulkan_reduce_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_reduce_workspace_bytes",
           &Program::vulkan_reduce_workspace_bytes)
      .def("vulkan_check_count_available",
           &Program::vulkan_check_count_available)
      .def("vulkan_check_count_clear_workspace",
           &Program::vulkan_check_count_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_check_count_workspace_bytes",
           &Program::vulkan_check_count_workspace_bytes)
      .def("vulkan_check_count_value_type_available",
           &Program::vulkan_check_count_value_type_available,
           py::arg("value_type"))
      .def("vulkan_check_count_ndarray", tracked_native_program_method(&Program::vulkan_check_count_ndarray),
           py::arg("values"), py::arg("output"), py::arg("value_type"),
           py::arg("check_op"), py::arg("lower"), py::arg("upper"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_check_count_strided_ndarray",
           tracked_native_program_method(&Program::vulkan_check_count_strided_ndarray), py::arg("values"),
           py::arg("output"), py::arg("value_type"), py::arg("offset"),
           py::arg("stride"), py::arg("check_op"), py::arg("lower"),
           py::arg("upper"), py::call_guard<py::gil_scoped_release>())
      .def("vulkan_check_count_dense_field",
           tracked_native_program_method(&Program::vulkan_check_count_dense_field), py::arg("values"),
           py::arg("output"), py::arg("value_type"), py::arg("n"),
           py::arg("check_op"), py::arg("lower"), py::arg("upper"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_metric_reduce_available",
           &Program::vulkan_metric_reduce_available)
      .def("vulkan_metric_reduce_clear_workspace",
           &Program::vulkan_metric_reduce_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_metric_reduce_workspace_bytes",
           &Program::vulkan_metric_reduce_workspace_bytes)
      .def("vulkan_metric_reduce_value_type_available",
           &Program::vulkan_metric_reduce_value_type_available,
           py::arg("value_type"))
      .def("vulkan_metric_reduce_ndarray",
           tracked_native_program_method(&Program::vulkan_metric_reduce_ndarray), py::arg("values"),
           py::arg("other"), py::arg("output"), py::arg("value_type"),
           py::arg("metric_op"), py::call_guard<py::gil_scoped_release>())
      .def("vulkan_metric_reduce_strided_ndarray",
           tracked_native_program_method(&Program::vulkan_metric_reduce_strided_ndarray), py::arg("values"),
           py::arg("other"), py::arg("output"), py::arg("value_type"),
           py::arg("values_offset"), py::arg("values_stride"),
           py::arg("other_offset"), py::arg("other_stride"),
           py::arg("metric_op"), py::call_guard<py::gil_scoped_release>())
      .def("vulkan_metric_reduce_dense_field",
           tracked_native_program_method(&Program::vulkan_metric_reduce_dense_field), py::arg("values"),
           py::arg("other"), py::arg("output"), py::arg("value_type"),
           py::arg("n"), py::arg("metric_op"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_metric_reduce_dense_field_strided_ndarray",
           tracked_native_program_method(&Program::vulkan_metric_reduce_dense_field_strided_ndarray),
           py::arg("field"), py::arg("array"), py::arg("output"),
           py::arg("value_type"), py::arg("n"), py::arg("array_offset"),
           py::arg("array_stride"), py::arg("field_is_values"),
           py::arg("metric_op"), py::call_guard<py::gil_scoped_release>())
      .def("vulkan_reduce_value_type_available",
           &Program::vulkan_reduce_value_type_available,
           py::arg("value_type"))
      .def("vulkan_reduce_ndarray", tracked_native_program_method(&Program::vulkan_reduce_ndarray),
           py::arg("values"), py::arg("output"), py::arg("value_type"),
           py::arg("op"), py::call_guard<py::gil_scoped_release>())
      .def("vulkan_reduce_member_ndarray",
           tracked_native_program_method(&Program::vulkan_reduce_member_ndarray), py::arg("values"),
           py::arg("output"), py::arg("value_type"), py::arg("offset"),
           py::arg("stride"), py::arg("op"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_reduce_strided_ndarray",
           tracked_native_program_method(&Program::vulkan_reduce_strided_ndarray), py::arg("values"),
           py::arg("output"), py::arg("value_type"),
           py::arg("values_offset"), py::arg("values_stride"),
           py::arg("output_offset"), py::arg("output_stride"), py::arg("op"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_reduce_dense_field", tracked_native_program_method(&Program::vulkan_reduce_dense_field),
           py::arg("values"), py::arg("output"), py::arg("value_type"),
           py::arg("n"), py::arg("op"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_reduce_dense_field_packed",
           tracked_native_program_method(&Program::vulkan_reduce_dense_field_packed), py::arg("values"),
           py::arg("output"), py::arg("value_type"), py::arg("n"),
           py::arg("lane_count"), py::arg("op"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_reduce_i32_ndarray",
           tracked_native_program_method(&Program::vulkan_reduce_i32_ndarray), py::arg("values"),
           py::arg("output"), py::arg("op"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_transform_available", &Program::vulkan_transform_available)
      .def("vulkan_transform_value_type_available",
           &Program::vulkan_transform_value_type_available,
           py::arg("value_type"))
      .def("vulkan_transform_clear_workspace",
           &Program::vulkan_transform_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_transform_workspace_bytes",
           &Program::vulkan_transform_workspace_bytes)
      .def("vulkan_transform_affine_ndarray",
           tracked_native_program_method(&Program::vulkan_transform_affine_ndarray), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("scale"),
           py::arg("bias"), py::call_guard<py::gil_scoped_release>())
      .def("vulkan_transform_affine_ndarray_trusted",
           tracked_native_program_method(&Program::vulkan_transform_affine_ndarray_trusted), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("scale"),
           py::arg("bias"), py::call_guard<py::gil_scoped_release>())
      .def("vulkan_transform_indexed_affine_ndarray",
           tracked_native_program_method(&Program::vulkan_transform_indexed_affine_ndarray), py::arg("src"),
           py::arg("indices"), py::arg("dst"), py::arg("value_type"),
           py::arg("scale"), py::arg("bias"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_transform_affine_member_ndarray",
           tracked_native_program_method(&Program::vulkan_transform_affine_member_ndarray), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("offset"),
           py::arg("stride"), py::arg("scale"), py::arg("bias"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_transform_affine_strided_ndarray",
           tracked_native_program_method(&Program::vulkan_transform_affine_strided_ndarray), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("src_offset"),
           py::arg("src_stride"), py::arg("dst_offset"),
           py::arg("dst_stride"), py::arg("scale"), py::arg("bias"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_transform_affine_strided_ndarray_trusted",
           tracked_native_program_method(&Program::vulkan_transform_affine_strided_ndarray_trusted),
           py::arg("src"), py::arg("dst"), py::arg("value_type"),
           py::arg("src_offset"), py::arg("src_stride"),
           py::arg("dst_offset"), py::arg("dst_stride"), py::arg("scale"),
           py::arg("bias"), py::call_guard<py::gil_scoped_release>())
      .def("vulkan_transform_affine_packed_strided_ndarray",
           tracked_native_program_method(&Program::vulkan_transform_affine_packed_strided_ndarray),
           py::arg("src"), py::arg("dst"), py::arg("value_type"),
           py::arg("lane_count"), py::arg("src_offset"),
           py::arg("src_stride"), py::arg("dst_offset"),
           py::arg("dst_stride"), py::arg("scale"), py::arg("bias"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_transform_affine_dense_field",
           tracked_native_program_method(&Program::vulkan_transform_affine_dense_field), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("n"),
           py::arg("scale"), py::arg("bias"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_transform_affine_dense_field_trusted",
           tracked_native_program_method(&Program::vulkan_transform_affine_dense_field_trusted),
           py::arg("src"), py::arg("dst"), py::arg("value_type"),
           py::arg("n"), py::arg("scale"), py::arg("bias"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_zero_dense_field", tracked_native_program_method(&Program::vulkan_zero_dense_field),
           py::arg("dst"), py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_zero_dense_fields", tracked_native_program_method(&Program::vulkan_zero_dense_fields),
           py::arg("dsts"), py::arg("value_types"), py::arg("ns"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_add_merge_available", &Program::vulkan_add_merge_available)
      .def("vulkan_add_merge_value_type_available",
           &Program::vulkan_add_merge_value_type_available,
           py::arg("value_type"))
      .def("vulkan_add_merge_clear_workspace",
           &Program::vulkan_add_merge_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_add_merge_workspace_bytes",
           &Program::vulkan_add_merge_workspace_bytes)
      .def("vulkan_add_merge_ndarray", tracked_native_program_method(&Program::vulkan_add_merge_ndarray),
           py::arg("src"), py::arg("dst"), py::arg("value_type"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_add_merge_strided_ndarray",
           tracked_native_program_method(&Program::vulkan_add_merge_strided_ndarray), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("src_offset"),
           py::arg("src_stride"), py::arg("dst_offset"),
           py::arg("dst_stride"), py::call_guard<py::gil_scoped_release>())
      .def("vulkan_add_merge_dense_field",
           tracked_native_program_method(&Program::vulkan_add_merge_dense_field), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_add_scalar_field_to_dense_field",
           tracked_native_program_method(&Program::vulkan_add_scalar_field_to_dense_field), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_indexed_copy_available",
           &Program::vulkan_indexed_copy_available)
      .def("vulkan_indexed_copy_clear_workspace",
           &Program::vulkan_indexed_copy_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_indexed_copy_workspace_bytes",
           &Program::vulkan_indexed_copy_workspace_bytes)
      .def("vulkan_gather_ndarray", tracked_native_program_method(&Program::vulkan_gather_ndarray),
           py::arg("src"), py::arg("indices"), py::arg("dst"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_gather_strided_ndarray",
           tracked_native_program_method(&Program::vulkan_gather_strided_ndarray), py::arg("src"),
           py::arg("indices"), py::arg("dst"), py::arg("item_bytes"),
           py::arg("src_offset"), py::arg("src_stride"),
           py::arg("dst_offset"), py::arg("dst_stride"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_gather_dense_field", tracked_native_program_method(&Program::vulkan_gather_dense_field),
           py::arg("src"), py::arg("indices"), py::arg("dst"),
           py::arg("value_type"), py::arg("src_n"), py::arg("dst_n"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_gather_dense_field_packed",
           tracked_native_program_method(&Program::vulkan_gather_dense_field_packed), py::arg("src"),
           py::arg("indices"), py::arg("dst"), py::arg("value_type"),
           py::arg("src_n"), py::arg("dst_n"), py::arg("lane_count"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_gather_dense_field_packed_indices_field",
           tracked_native_program_method(&Program::vulkan_gather_dense_field_packed_indices_field),
           py::arg("src"), py::arg("indices"), py::arg("dst"),
           py::arg("value_type"), py::arg("src_n"), py::arg("indices_n"),
           py::arg("dst_n"), py::arg("lane_count"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_gather_dense_field_indices_field",
           tracked_native_program_method(&Program::vulkan_gather_dense_field_indices_field), py::arg("src"),
           py::arg("indices"), py::arg("dst"), py::arg("value_type"),
           py::arg("src_n"), py::arg("indices_n"), py::arg("dst_n"),
           py::call_guard<py::gil_scoped_release>())
       .def("vulkan_scatter_ndarray", tracked_native_program_method(&Program::vulkan_scatter_ndarray),
            py::arg("src"), py::arg("indices"), py::arg("dst"),
            py::call_guard<py::gil_scoped_release>())
       .def("vulkan_scatter_strided_ndarray",
            tracked_native_program_method(&Program::vulkan_scatter_strided_ndarray), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("item_bytes"),
            py::arg("src_offset"), py::arg("src_stride"),
            py::arg("dst_offset"), py::arg("dst_stride"),
            py::call_guard<py::gil_scoped_release>())
       .def("vulkan_scatter_dense_field",
            tracked_native_program_method(&Program::vulkan_scatter_dense_field), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("value_type"),
            py::arg("src_n"), py::arg("dst_n"),
            py::call_guard<py::gil_scoped_release>())
       .def("vulkan_scatter_dense_field_packed",
            tracked_native_program_method(&Program::vulkan_scatter_dense_field_packed), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("value_type"),
            py::arg("src_n"), py::arg("dst_n"), py::arg("lane_count"),
            py::call_guard<py::gil_scoped_release>())
       .def("vulkan_scatter_dense_field_packed_indices_field",
            tracked_native_program_method(&Program::vulkan_scatter_dense_field_packed_indices_field),
            py::arg("src"), py::arg("indices"), py::arg("dst"),
            py::arg("value_type"), py::arg("src_n"), py::arg("indices_n"),
            py::arg("dst_n"), py::arg("lane_count"),
            py::call_guard<py::gil_scoped_release>())
       .def("vulkan_scatter_dense_field_indices_field",
            tracked_native_program_method(&Program::vulkan_scatter_dense_field_indices_field), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("value_type"),
            py::arg("src_n"), py::arg("indices_n"), py::arg("dst_n"),
            py::call_guard<py::gil_scoped_release>())
       .def("vulkan_scatter_add_available",
            &Program::vulkan_scatter_add_available)
       .def("vulkan_scatter_add_value_type_available",
            &Program::vulkan_scatter_add_value_type_available,
            py::arg("value_type"))
       .def("vulkan_scatter_add_clear_workspace",
            &Program::vulkan_scatter_add_clear_workspace,
            py::call_guard<py::gil_scoped_release>())
       .def("vulkan_scatter_add_workspace_bytes",
            &Program::vulkan_scatter_add_workspace_bytes)
       .def("vulkan_scatter_add_ndarray",
            tracked_native_program_method(&Program::vulkan_scatter_add_ndarray), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("value_type"),
            py::call_guard<py::gil_scoped_release>())
       .def("vulkan_scatter_add_member_ndarray",
            tracked_native_program_method(&Program::vulkan_scatter_add_member_ndarray), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("value_type"),
            py::arg("offset"), py::arg("stride"),
            py::call_guard<py::gil_scoped_release>())
       .def("vulkan_scatter_add_strided_ndarray",
            tracked_native_program_method(&Program::vulkan_scatter_add_strided_ndarray), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("value_type"),
            py::arg("src_offset"), py::arg("src_stride"),
            py::arg("dst_offset"), py::arg("dst_stride"),
            py::call_guard<py::gil_scoped_release>())
       .def("vulkan_scatter_add_dense_field",
            tracked_native_program_method(&Program::vulkan_scatter_add_dense_field), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("value_type"),
            py::arg("src_n"), py::arg("dst_n"),
            py::call_guard<py::gil_scoped_release>())
       .def("vulkan_scatter_add_dense_field_indices_field",
            tracked_native_program_method(&Program::vulkan_scatter_add_dense_field_indices_field),
            py::arg("src"), py::arg("indices"), py::arg("dst"),
            py::arg("value_type"), py::arg("src_n"), py::arg("indices_n"),
            py::arg("dst_n"), py::call_guard<py::gil_scoped_release>())
       .def("vulkan_bucket_builder_available",
            &Program::vulkan_bucket_builder_available)
       .def("vulkan_bucket_builder_clear_workspace",
            &Program::vulkan_bucket_builder_clear_workspace,
            py::call_guard<py::gil_scoped_release>())
       .def("vulkan_bucket_builder_workspace_bytes",
            &Program::vulkan_bucket_builder_workspace_bytes)
       .def("vulkan_bucket_builder_value_type_available",
            &Program::vulkan_bucket_builder_value_type_available,
            py::arg("value_type"))
       .def("vulkan_bucket_builder_i32_ndarray",
            tracked_native_program_method(&Program::vulkan_bucket_builder_i32_ndarray), py::arg("keys"),
            py::arg("values"), py::arg("offsets"), py::arg("output"),
            py::arg("cursor"), py::call_guard<py::gil_scoped_release>())
       .def("vulkan_bucket_builder_ndarray",
            tracked_native_program_method(&Program::vulkan_bucket_builder_ndarray), py::arg("keys"),
            py::arg("values"), py::arg("offsets"), py::arg("output"),
            py::arg("cursor"), py::arg("value_type"),
            py::call_guard<py::gil_scoped_release>())
       .def("vulkan_bucket_builder_dense_field",
            tracked_native_program_method(&Program::vulkan_bucket_builder_dense_field), py::arg("keys"),
            py::arg("values"), py::arg("offsets"), py::arg("output"),
            py::arg("cursor"), py::arg("value_type"), py::arg("n"),
            py::arg("num_bins"), py::call_guard<py::gil_scoped_release>())
       .def("vulkan_grouped_reduce_available",
            &Program::vulkan_grouped_reduce_available)
       .def("vulkan_grouped_reduce_value_type_available",
            &Program::vulkan_grouped_reduce_value_type_available,
            py::arg("value_type"))
       .def("vulkan_grouped_reduce_atomic_value_type_available",
            &Program::vulkan_grouped_reduce_atomic_value_type_available,
            py::arg("value_type"))
       .def("vulkan_grouped_reduce_atomic_ndarray",
            tracked_native_program_method(&Program::vulkan_grouped_reduce_atomic_ndarray), py::arg("keys"),
            py::arg("values"), py::arg("output"), py::arg("value_type"),
            py::arg("op"), py::call_guard<py::gil_scoped_release>())
       .def("vulkan_grouped_reduce_atomic_dense_field",
            tracked_native_program_method(&Program::vulkan_grouped_reduce_atomic_dense_field),
            py::arg("keys"), py::arg("values"), py::arg("output"),
            py::arg("value_type"), py::arg("n"), py::arg("num_groups"),
            py::arg("op"), py::call_guard<py::gil_scoped_release>())
       .def("vulkan_grouped_reduce_atomic_member_ndarray",
            tracked_native_program_method(&Program::vulkan_grouped_reduce_atomic_member_ndarray),
            py::arg("keys"), py::arg("values"), py::arg("output"),
            py::arg("value_type"), py::arg("offset"), py::arg("stride"),
            py::arg("op"), py::call_guard<py::gil_scoped_release>())
       .def("vulkan_grouped_reduce_atomic_strided_ndarray",
            tracked_native_program_method(&Program::vulkan_grouped_reduce_atomic_strided_ndarray),
            py::arg("keys"), py::arg("values"), py::arg("output"),
            py::arg("value_type"), py::arg("values_offset"),
            py::arg("values_stride"), py::arg("output_offset"),
            py::arg("output_stride"), py::arg("op"),
            py::call_guard<py::gil_scoped_release>())
       .def("vulkan_grouped_reduce_atomic_strided_keys_ndarray",
            tracked_native_program_method(&Program::vulkan_grouped_reduce_atomic_strided_keys_ndarray),
            py::arg("keys"), py::arg("values"), py::arg("output"),
            py::arg("value_type"), py::arg("keys_offset"),
            py::arg("keys_stride"), py::arg("values_offset"),
            py::arg("values_stride"), py::arg("output_offset"),
            py::arg("output_stride"), py::arg("op"),
            py::call_guard<py::gil_scoped_release>())
       .def("vulkan_grouped_reduce_clear_workspace",
            &Program::vulkan_grouped_reduce_clear_workspace,
            py::call_guard<py::gil_scoped_release>())
       .def("vulkan_grouped_reduce_workspace_bytes",
            &Program::vulkan_grouped_reduce_workspace_bytes)
       .def("vulkan_grouped_reduce_i32_atomic_ndarray",
            tracked_native_program_method(&Program::vulkan_grouped_reduce_i32_atomic_ndarray),
            py::arg("keys"), py::arg("values"), py::arg("output"),
            py::arg("op"), py::call_guard<py::gil_scoped_release>())
       .def("vulkan_grouped_reduce_i32_ndarray",
            tracked_native_program_method(&Program::vulkan_grouped_reduce_i32_ndarray), py::arg("keys"),
            py::arg("values"), py::arg("output"), py::arg("offsets"),
            py::arg("scratch"), py::arg("cursor"), py::arg("op"),
            py::call_guard<py::gil_scoped_release>())
       .def("vulkan_grouped_reduce_ndarray",
            tracked_native_program_method(&Program::vulkan_grouped_reduce_ndarray), py::arg("keys"),
            py::arg("values"), py::arg("output"), py::arg("offsets"),
            py::arg("scratch"), py::arg("cursor"), py::arg("value_type"),
            py::arg("op"), py::call_guard<py::gil_scoped_release>())
       .def("vulkan_radix_sort_u32_dense_field",
            tracked_native_program_method(&Program::vulkan_radix_sort_u32_dense_field), py::arg("keys"),
            py::arg("values"), py::arg("key_type"), py::arg("value_type"),
            py::arg("n"), py::call_guard<py::gil_scoped_release>())
       .def("vulkan_radix_sort_u32_keys_dense_field",
            [](Program *program, SNode *keys, int key_type, std::size_t n) {
              return program->vulkan_radix_sort_u32_dense_field(
                  keys, nullptr, key_type, 0, n);
            },
            py::arg("keys"), py::arg("key_type"), py::arg("n"),
            py::call_guard<py::gil_scoped_release>())
       .def("vulkan_radix_sort_u32_ndarray",
           tracked_native_program_method(&Program::vulkan_radix_sort_u32_ndarray), py::arg("keys"),
           py::arg("values"), py::arg("key_type"), py::arg("value_type"),
           py::arg("key_offset") = 0, py::arg("value_offset") = 0,
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_radix_sort_u32_keys_ndarray",
           [](Program *program, Ndarray *keys, int key_type) {
             return program->vulkan_radix_sort_u32_ndarray(keys, nullptr,
                                                           key_type, 0);
           },
           py::arg("keys"), py::arg("key_type"),
           py::call_guard<py::gil_scoped_release>())
      .def("get_graphics_device",
           [](Program *program) { return program->get_graphics_device(); })
      .def("compile_kernel", &Program::compile_kernel,
           py::return_value_policy::reference)
      .def(
          "compile_kernels",
          [](Program *program, const CompileConfig &cfg,
             const std::vector<Kernel *> &kernels) {
            std::vector<const Kernel *> ks(kernels.begin(), kernels.end());
            program->compile_kernels(cfg, ks);
          },
          py::call_guard<py::gil_scoped_release>())
      .def("launch_kernel", &Program::launch_kernel)
      .def("compile_and_launch_kernel", &Program::compile_and_launch_kernel,
           py::call_guard<py::gil_scoped_release>())
      .def("_register_kernel_execution_plan",
           &Program::register_kernel_execution_plan,
           py::arg("compile_config"), py::arg("device_caps"),
           py::arg("kernel"),
           py::arg("allow_snode_tree_dependencies") = false,
           py::call_guard<py::gil_scoped_release>())
      .def("get_device_caps", &Program::get_device_caps);

  py::class_<AotModuleBuilder>(m, "AotModuleBuilder")
      .def("add_field", &AotModuleBuilder::add_field)
      .def("add", &AotModuleBuilder::add)
      .def("add_kernel_template", &AotModuleBuilder::add_kernel_template)
      .def("add_graph", &AotModuleBuilder::add_graph)
      .def("dump", &AotModuleBuilder::dump);

  py::class_<Axis>(m, "Axis").def(py::init<int>());
  py::class_<SNode>(m, "SNode")
      .def(py::init<>())
      .def_readwrite("parent", &SNode::parent)
      .def_readonly("type", &SNode::type)
      .def_readonly("id", &SNode::id)
      .def_readonly("runtime_local_id", &SNode::runtime_local_id)
      .def("get_snode_tree_id", &SNode::get_snode_tree_id)
      .def_readonly("offset", &SNode::index_offsets)
      .def("dense",
           (SNode & (SNode::*)(const std::vector<Axis> &,
                               const std::vector<int> &,
                               const DebugInfo &))(&SNode::dense),
           py::return_value_policy::reference)
      .def("pointer",
           (SNode & (SNode::*)(const std::vector<Axis> &,
                               const std::vector<int> &,
                               const DebugInfo &))(&SNode::pointer),
           py::return_value_policy::reference)
      // C-1 (2026-05): pointer + per-SNode pool capacity hint。后端在
      // SNodeTree finalize 时读取 vk_max_active_hint 决议池容量，绕过 worst-
      // case 与全局 fraction（详见 SNode_Vulkan_规划.md §11.1）。
      .def(
          "pointer_with_hint",
          [](SNode &self, const std::vector<Axis> &axes,
             const std::vector<int> &sizes, int64_t vk_max_active,
             const DebugInfo &dbg_info) -> SNode & {
            SNode &child = self.pointer(axes, sizes, dbg_info);
            child.vk_max_active_hint = vk_max_active;
            return child;
          },
          py::return_value_policy::reference)
      .def("hash",
           (SNode & (SNode::*)(const std::vector<Axis> &,
                               const std::vector<int> &,
                               const DebugInfo &))(&SNode::hash),
           py::return_value_policy::reference)
      .def(
          "hash_with_capacity",
          [](SNode &self, const std::vector<Axis> &axes,
             const std::vector<int> &sizes, int64_t table_capacity,
             const DebugInfo &dbg_info) -> SNode & {
            SNode &child = self.hash(axes, sizes, dbg_info);
            child.vk_max_active_hint = table_capacity;
            return child;
          },
          py::return_value_policy::reference)
      .def(
          "hash_with_capacity_and_active_hint",
          [](SNode &self, const std::vector<Axis> &axes,
             const std::vector<int> &sizes, int64_t table_capacity,
             int64_t expected_active, const DebugInfo &dbg_info) -> SNode & {
            SNode &child = self.hash(axes, sizes, dbg_info);
            child.vk_max_active_hint = table_capacity;
            child.hash_expected_active_hint = expected_active;
            return child;
          },
          py::return_value_policy::reference)
      .def("dynamic", &SNode::dynamic, py::return_value_policy::reference)
      .def("bitmasked",
           (SNode & (SNode::*)(const std::vector<Axis> &,
                               const std::vector<int> &,
                               const DebugInfo &))(&SNode::bitmasked),
           py::return_value_policy::reference)
      .def("bit_struct", &SNode::bit_struct, py::return_value_policy::reference)
      .def("quant_array", &SNode::quant_array,
           py::return_value_policy::reference)
      .def("place", &SNode::place)
      .def("data_type", [](SNode *snode) { return snode->dt; })
      .def("name", [](SNode *snode) { return snode->name; })
      .def("get_num_ch",
           [](SNode *snode) -> int { return (int)snode->ch.size(); })
      .def(
          "get_ch",
          [](SNode *snode, int i) -> SNode * { return snode->ch[i].get(); },
          py::return_value_policy::reference)
      .def("lazy_grad", &SNode::lazy_grad)
      .def("lazy_dual", &SNode::lazy_dual)
      .def("allocate_adjoint_checkbit", &SNode::allocate_adjoint_checkbit)
      .def("read_int", &SNode::read_int)
      .def("read_uint", &SNode::read_uint)
      .def("read_float", &SNode::read_float)
      .def("has_adjoint", &SNode::has_adjoint)
      .def("has_adjoint_checkbit", &SNode::has_adjoint_checkbit)
      .def("get_snode_grad_type", &SNode::get_snode_grad_type)
      .def("has_dual", &SNode::has_dual)
      .def("is_primal", &SNode::is_primal)
      .def("is_place", &SNode::is_place)
      .def("get_expr", &SNode::get_expr)
      .def("write_int", &SNode::write_int)
      .def("write_uint", &SNode::write_uint)
      .def("write_float", &SNode::write_float)
      .def("get_shape_along_axis", &SNode::shape_along_axis)
      .def("get_physical_index_position",
           [](SNode *snode) {
             return std::vector<int>(
                 snode->physical_index_position,
                 snode->physical_index_position + taichi_max_num_indices);
           })
      .def("num_active_indices",
           [](SNode *snode) { return snode->num_active_indices; })
      .def_readonly("cell_size_bytes", &SNode::cell_size_bytes)
      .def_readonly("offset_bytes_in_parent_cell",
                    &SNode::offset_bytes_in_parent_cell);

  py::class_<SNodeTree>(m, "SNodeTree")
      .def("id", &SNodeTree::id)
      .def("generation", &SNodeTree::generation)
      .def(
          "destroy_snode_tree",
          [](SNodeTree *snode_tree, Program *program) {
            program->destroy_snode_tree(snode_tree);
          },
          py::arg("program"), py::call_guard<py::gil_scoped_release>());

  py::class_<DeviceAllocation>(m, "DeviceAllocation")
      .def(py::init([](uint64_t device, uint64_t alloc_id) -> DeviceAllocation {
             DeviceAllocation alloc;
             alloc.device = (Device *)device;
             alloc.alloc_id = (DeviceAllocationId)alloc_id;
             return alloc;
           }),
           py::arg("device"), py::arg("alloc_id"))
      .def_readonly("device", &DeviceAllocation::device)
      .def_readonly("alloc_id", &DeviceAllocation::alloc_id);

  py::class_<Ndarray>(m, "Ndarray")
      .def("device_allocation_ptr", &Ndarray::get_device_allocation_ptr_as_int)
      .def("device_allocation", &Ndarray::get_device_allocation)
      .def("element_size", &Ndarray::get_element_size)
      .def("nelement", &Ndarray::get_nelement)
      .def("read_int", &Ndarray::read_int)
      .def("read_uint", &Ndarray::read_uint)
      .def("read_float", &Ndarray::read_float)
      .def("write_int", &Ndarray::write_int)
      .def("write_float", &Ndarray::write_float)
      .def("total_shape", &Ndarray::total_shape)
      .def("element_shape", &Ndarray::get_element_shape)
      .def("element_data_type", &Ndarray::get_element_data_type)
      .def_readonly("dtype", &Ndarray::dtype)
      .def_readonly("shape", &Ndarray::shape);

  export_storage_view(m);

  py::class_<ArgPack>(m, "ArgPack")
      .def("device_allocation_ptr", &ArgPack::get_device_allocation_ptr_as_int)
      .def("device_allocation", &ArgPack::get_device_allocation)
      .def("nelement", &ArgPack::get_nelement)
      .def("data_type", &ArgPack::get_data_type)
      .def("set_arg_float", &ArgPack::set_arg_float)
      .def("set_arg_int", &ArgPack::set_arg_int)
      .def("set_arg_uint", &ArgPack::set_arg_uint)
      .def("set_arg_nested_argpack", &ArgPack::set_arg_nested_argpack)
      .def_readonly("dtype", &ArgPack::dtype);

  py::enum_<BufferFormat>(m, "Format")
#define PER_BUFFER_FORMAT(x) .value(#x, BufferFormat::x)
#include "taichi/inc/rhi_constants.inc.h"
#undef PER_EXTENSION
      ;

  py::enum_<ImageFilter>(m, "ImageFilter")
      .value("nearest", ImageFilter::nearest)
      .value("linear", ImageFilter::linear);

  py::enum_<ImageAddressMode>(m, "ImageAddressMode")
      .value("repeat", ImageAddressMode::repeat)
      .value("mirrored_repeat", ImageAddressMode::mirrored_repeat)
      .value("clamp_to_edge", ImageAddressMode::clamp_to_edge);

  py::class_<ImageSamplerConfig>(m, "ImageSamplerConfig")
      .def(py::init<>())
      .def_readwrite("min_filter", &ImageSamplerConfig::min_filter)
      .def_readwrite("mag_filter", &ImageSamplerConfig::mag_filter)
      .def_readwrite("address_mode_u", &ImageSamplerConfig::address_mode_u)
      .def_readwrite("address_mode_v", &ImageSamplerConfig::address_mode_v)
      .def_readwrite("address_mode_w", &ImageSamplerConfig::address_mode_w);

  py::class_<Texture>(m, "Texture")
      .def("device_allocation_ptr", &Texture::get_device_allocation_ptr_as_int)
      .def("from_ndarray", &Texture::from_ndarray)
      .def("from_snode", &Texture::from_snode);

  py::enum_<aot::ArgKind>(m, "ArgKind")
      .value("SCALAR", aot::ArgKind::kScalar)
      .value("NDARRAY", aot::ArgKind::kNdarray)
      // Using this MATRIX as Scalar alias, we can move to native matrix type
      // when supported
      .value("MATRIX", aot::ArgKind::kMatrix)
      .value("TEXTURE", aot::ArgKind::kTexture)
      .value("RWTEXTURE", aot::ArgKind::kRWTexture)
      .export_values();

  py::class_<aot::Arg>(m, "Arg")
      .def(py::init<aot::ArgKind, std::string, DataType &, size_t,
                    std::vector<int>>(),
           py::arg("tag"), py::arg("name"), py::arg("dtype"),
           py::arg("field_dim"), py::arg("element_shape"))
      .def(py::init<aot::ArgKind, std::string, DataType &, size_t,
                    std::vector<int>>(),
           py::arg("tag"), py::arg("name"), py::arg("channel_format"),
           py::arg("num_channels"), py::arg("shape"))
      .def_readonly("tag", &aot::Arg::tag)
      .def_readonly("name", &aot::Arg::name)
      .def_readonly("element_shape", &aot::Arg::element_shape)
      .def_readonly("texture_shape", &aot::Arg::element_shape)
      .def_readonly("field_dim", &aot::Arg::field_dim)
      .def_readonly("num_channels", &aot::Arg::num_channels)
      .def("dtype", &aot::Arg::dtype)
      .def("element_dtype", &aot::Arg::element_dtype)
      .def("channel_format", &aot::Arg::dtype);

  py::class_<Node>(m, "Node");  // NOLINT(bugprone-unused-raii)

  py::class_<Sequential, Node>(m, "Sequential")
      .def(py::init<GraphBuilder *>())
      .def("append", &Sequential::append)
      .def("dispatch", &Sequential::dispatch, py::arg("kernel"),
           py::arg("args"), py::arg("label") = "")
      .def("dispatch_indirect", &Sequential::dispatch_indirect,
           py::arg("kernel"), py::arg("args"),
           py::arg("dispatch_packet"), py::arg("label") = "");

  py::class_<GraphBuilder>(m, "GraphBuilder")
      .def(py::init<>())
      .def("dispatch", &GraphBuilder::dispatch, py::arg("kernel"),
           py::arg("args"), py::arg("label") = "")
      .def("dispatch_indirect", &GraphBuilder::dispatch_indirect,
           py::arg("kernel"), py::arg("args"),
           py::arg("dispatch_packet"), py::arg("label") = "")
      .def("dispatch_cuda_bounded", &GraphBuilder::dispatch_cuda_bounded,
           py::arg("kernel"), py::arg("args"), py::arg("extent"),
           py::arg("capacity"), py::arg("block_dim"),
           py::arg("adaptive_grid"), py::arg("grouped_update") = false,
           py::arg("label") = "")
      .def("dispatch_cpu_bounded", &GraphBuilder::dispatch_cpu_bounded,
           py::arg("kernel"), py::arg("args"), py::arg("extent"),
           py::arg("capacity"), py::arg("label") = "")
      .def(
          "_dispatch_cuda_cusparse_spmv_capture_recipe",
          [](GraphBuilder &builder, const py::object &matrix,
             Program *program, const aot::Arg &input,
             const aot::Arg &output) {
            builder.dispatch_cuda_capture_cusparse_spmv(
                matrix.cast<SparseMatrix *>(), program, input, output);
          },
          py::arg("matrix"), py::arg("program"), py::arg("input"),
          py::arg("output"))
      .def(
          "_dispatch_cuda_cusparse_spmm_capture_recipe",
          [](GraphBuilder &builder, const py::object &matrix,
             Program *program, const aot::Arg &input,
             const aot::Arg &output, int rhs_count, int algorithm) {
            auto *csr = dynamic_cast<CuSparseMatrix *>(
                matrix.cast<SparseMatrix *>());
            TI_ERROR_IF(csr == nullptr,
                        "CUDA cuSPARSE SpMM capture supports scalar CSR "
                        "matrices only.");
            builder.dispatch_cuda_capture_cusparse_spmm(
                csr, program, input, output, rhs_count, algorithm);
          },
          py::arg("matrix"), py::arg("program"), py::arg("input"),
          py::arg("output"), py::arg("rhs_count"), py::arg("algorithm"))
      .def(
          "_dispatch_cuda_cusparse_spmm_owned_capture_recipe",
          [](GraphBuilder &builder, const py::object &matrix, Program *program,
             const aot::Arg &input, const aot::Arg &output, int rhs_count,
             int algorithm, std::int64_t expected_workspace) {
            auto *csr =
                dynamic_cast<CuSparseMatrix *>(matrix.cast<SparseMatrix *>());
            TI_ERROR_IF(csr == nullptr,
                        "CUDA SpMM ownership requires a scalar CSR matrix.");
            builder.dispatch_cuda_capture_cusparse_spmm(
                csr, program, input, output, rhs_count, algorithm, true,
                expected_workspace);
          },
          py::arg("matrix"), py::arg("program"), py::arg("input"),
          py::arg("output"), py::arg("rhs_count"), py::arg("algorithm"),
          py::arg("expected_workspace") = -1)
      .def(
          "_dispatch_cuda_cusparse_triangular_capture_recipe",
          [](GraphBuilder &builder, const py::object &matrix,
             Program *program, const aot::Arg &input,
             const aot::Arg &output, int rhs_count, int fill_mode,
             bool unit_diagonal, bool transpose) {
            auto *csr = dynamic_cast<CuSparseMatrix *>(
                matrix.cast<SparseMatrix *>());
            TI_ERROR_IF(csr == nullptr,
                        "CUDA cuSPARSE triangular capture supports scalar "
                        "CSR matrices only.");
            builder.dispatch_cuda_capture_cusparse_triangular(
                csr, program, input, output, rhs_count, fill_mode,
                unit_diagonal, transpose);
          },
          py::arg("matrix"), py::arg("program"), py::arg("input"),
          py::arg("output"), py::arg("rhs_count"), py::arg("fill_mode"),
          py::arg("unit_diagonal"), py::arg("transpose"))
      .def("_dispatch_cuda_cufft_capture_recipe",
           &GraphBuilder::dispatch_cuda_capture_cufft,
           py::arg("plan_handle"), py::arg("program"), py::arg("input"),
           py::arg("output"), py::arg("direction"),
           py::arg("input_scalars"), py::arg("output_scalars"))
      .def("_dispatch_cuda_capture_description",
           &GraphBuilder::dispatch_cuda_capture_description,
           py::arg("program"), py::arg("kind"), py::arg("arguments"))
      .def("_dispatch_cuda_addon_capture_recipe",
           &GraphBuilder::dispatch_cuda_capture_addon,
           py::arg("program"), py::arg("invoke_address"), py::arg("payload"),
           py::arg("stream_offset"), py::arg("arguments"),
           py::arg("pointer_offsets"), py::arg("scalar_counts"),
           py::arg("writable"), py::arg("error_address") = 0)
      .def("_dispatch_cuda_scan_capture_recipe",
           &GraphBuilder::dispatch_cuda_capture_scan,
           py::arg("program"), py::arg("values"), py::arg("workspace"),
           py::arg("num_items"), py::arg("value_type"))
      .def("_dispatch_cuda_cublaslt_capture_recipe",
           &GraphBuilder::dispatch_cuda_capture_cublaslt,
           py::arg("program"), py::arg("plan"), py::arg("arguments"))
      .def("_dispatch_cuda_cutensor_capture_recipe",
           &GraphBuilder::dispatch_cuda_capture_cutensor,
           py::arg("program"), py::arg("plan"), py::arg("arguments"))
      .def("_dispatch_cuda_cusparselt_capture_recipe",
           &GraphBuilder::dispatch_cuda_capture_cusparselt,
           py::arg("program"), py::arg("plan"), py::arg("arguments"))
      .def("_dispatch_cuda_cudss_capture_recipe",
           &GraphBuilder::dispatch_cuda_capture_cudss,
           py::arg("program"), py::arg("handle"), py::arg("numeric_phase"),
           py::arg("arguments"))
      .def("_dispatch_cuda_cudss_capture_group",
           &GraphBuilder::dispatch_cuda_capture_cudss,
           py::arg("program"), py::arg("handle"), py::arg("numeric_phase"),
           py::arg("arguments"))
      .def("compile", &GraphBuilder::compile)
      .def("_enable_two_map_composer",
           &GraphBuilder::enable_two_map_composer)
      .def("_set_map_composer_max_group_size",
           &GraphBuilder::set_map_composer_max_group_size)
      .def("_set_map_composer_allowed_groups",
           &GraphBuilder::set_map_composer_allowed_groups)
      .def("_set_cuda_parallel_dispatch_groups",
           &GraphBuilder::set_cuda_parallel_dispatch_groups)
      .def("create_sequential", &GraphBuilder::new_sequential_node,
           py::return_value_policy::reference)
      .def("seq", &GraphBuilder::seq, py::return_value_policy::reference);

  struct VulkanNestedGraphRequest {
    Ndarray *outer_predicate{nullptr};
    Ndarray *outer_counter{nullptr};
    Ndarray *outer_status{nullptr};
    std::vector<aot::CompiledGraphNestedInnerControl> inner_controls;
    std::size_t outer_condition_dispatch_count{0};
    int outer_max_iterations{0};
    bool wait_for_result{true};
    aot::CompiledGraphNestedStructuredResult *result{nullptr};
  };

  struct CudaNestedGraphRequest {
    Ndarray *outer_predicate{nullptr};
    Ndarray *outer_counter{nullptr};
    Ndarray *outer_status{nullptr};
    std::vector<aot::CompiledGraphNestedInnerControl> inner_controls;
    std::size_t outer_condition_dispatch_count{0};
    int outer_max_iterations{0};
    bool allow_device_update{true};
  };

  auto make_nested_graph_controls = [](
                                        const py::sequence &predicates,
                                        const py::sequence &counters,
                                        const py::sequence &boundaries,
                                        const py::sequence &max_iterations,
                                        const py::sequence &statuses,
                                        const py::object &chunk_sizes) {
    const py::ssize_t count = py::len(predicates);
    TI_ERROR_IF(count == 0 || count > 8,
                "Nested Graph execution requires between one and eight "
                "ordered inner regions");
    TI_ERROR_IF(py::len(counters) != count || py::len(boundaries) != count ||
                    py::len(max_iterations) != count ||
                    py::len(statuses) != count,
                "Nested Graph inner control sequences must have equal "
                "lengths");
    py::sequence chunks;
    if (!chunk_sizes.is_none()) {
      chunks = chunk_sizes.cast<py::sequence>();
      TI_ERROR_IF(py::len(chunks) != count,
                  "Nested Graph inner chunk sizes must match the number of "
                  "inner regions");
    }
    std::vector<aot::CompiledGraphNestedInnerControl> controls;
    controls.reserve(static_cast<std::size_t>(count));
    for (py::ssize_t i = 0; i < count; ++i) {
      auto boundary = boundaries[i].cast<py::sequence>();
      TI_ERROR_IF(py::len(boundary) != 3,
                  "Nested Graph inner boundaries must contain condition, "
                  "body and end dispatch indices");
      Ndarray *status = nullptr;
      if (!statuses[i].is_none()) {
        status = &statuses[i].cast<Ndarray &>();
      }
      controls.push_back(
          {&predicates[i].cast<Ndarray &>(),
           &counters[i].cast<Ndarray &>(), status,
           boundary[0].cast<std::size_t>(),
           boundary[1].cast<std::size_t>(),
           boundary[2].cast<std::size_t>(),
           max_iterations[i].cast<int>(),
           chunk_sizes.is_none() ? 1 : chunks[i].cast<int>()});
    }
    return controls;
  };

  auto encode_nested_graph_result = [](
                                        const aot::CompiledGraphNestedStructuredResult &result,
                                        bool wait_for_result) {
    py::dict encoded;
    encoded["submitted"] = wait_for_result ? result.submitted : true;
    encoded["inner_region_count"] = result.inner_region_count;
    encoded["outer_logical_iterations"] = result.outer_logical_iterations;
    encoded["outer_encoded_iterations"] = result.outer_encoded_iterations;
    encoded["outer_initial_predicate"] = result.outer_initial_predicate;
    encoded["outer_final_predicate"] = result.outer_final_predicate;
    encoded["outer_initial_counter"] = result.outer_initial_counter;
    encoded["outer_final_counter"] = result.outer_final_counter;
    encoded["outer_initial_status"] = result.outer_initial_status;
    encoded["outer_final_status"] = result.outer_final_status;
    encoded["inner_logical_iterations"] = result.inner_logical_iterations;
    encoded["inner_encoded_iterations"] = result.inner_encoded_iterations;
    encoded["inner_initial_counters"] = result.inner_initial_counters;
    encoded["inner_final_counters"] = result.inner_final_counters;
    encoded["inner_final_predicates"] = result.inner_final_predicates;
    encoded["inner_initial_statuses"] = result.inner_initial_statuses;
    encoded["inner_final_statuses"] = result.inner_final_statuses;
    encoded["indirect_dispatches"] = result.indirect_dispatches;
    encoded["controller_dispatches"] = result.controller_dispatches;
    encoded["controller_invocations"] = result.controller_invocations;
    encoded["zero_dispatches"] = result.zero_dispatches;
    encoded["control_bytes"] = result.control_bytes;
    encoded["observation_bytes"] = result.observation_bytes;
    return encoded;
  };

  auto with_graph_arguments = [](const aot::CompiledGraph *self,
                                 const py::dict &pyargs, auto &&action) {
        std::unordered_map<std::string, aot::IValue> args;
        auto insert_scalar_arg = [&args](std::string arg_name,
                                         DataType expected_dtype,
                                         py::object pyarg) {
          auto type_id = expected_dtype->as<PrimitiveType>()->type;
          switch (type_id) {
#define PER_C_TYPE(type, ctype)                                           \
  case PrimitiveTypeID::type:                                             \
    args.insert({arg_name, aot::IValue::create(py::cast<ctype>(pyarg))}); \
    break;
#include "taichi/inc/data_type_with_c_type.inc.h"
#undef PER_C_TYPE
            default:
              TI_ERROR("Unsupported scalar type {}",
                       expected_dtype->to_string());
          }
        };

        std::vector<std::unique_ptr<char[]>> matrix_buffers;
        matrix_buffers.reserve(self->args.size());
        std::vector<Matrix> matrices;
        // Reserve to avoid changes in element addresses
        matrices.reserve(self->args.size());
        auto matrix_dimensions = [](const py::buffer_info &buffer_info) {
          TI_ERROR_IF(buffer_info.ndim < 1 || buffer_info.ndim > 2,
                      "Graph Matrix runtime arguments must have rank 1 or 2, "
                      "but got rank {}",
                      buffer_info.ndim);
          return std::array<uint32_t, 3>{
              static_cast<uint32_t>(buffer_info.ndim),
              static_cast<uint32_t>(buffer_info.shape[0]),
              buffer_info.ndim == 2
                  ? static_cast<uint32_t>(buffer_info.shape[1])
                  : 0};
        };
        for (const auto &[arg_name, arg] : self->args) {
          auto tag = arg.tag;
          TI_ASSERT(pyargs.contains(arg_name.c_str()));
          auto pyarg = pyargs[arg_name.c_str()];
          if (tag == aot::ArgKind::kNdarray) {
            if (py::isinstance<py::tuple>(pyarg)) {
              auto runtime_pair = pyarg.template cast<py::tuple>();
              TI_ERROR_IF(runtime_pair.size() != 2,
                          "Graph ndarray runtime storage tuple must contain "
                          "the owner and argument");
              auto &runtime_argument =
                  runtime_pair[1]
                      .template cast<storage::RuntimeStorageArgument &>();
              if (py::isinstance<Ndarray>(runtime_pair[0])) {
                auto &val = runtime_pair[0].template cast<Ndarray &>();
                args.insert(
                    {arg_name, aot::IValue::create(val, runtime_argument)});
              } else {
                // The first tuple item is a Python keepalive for a dense
                // Field/view. Its immutable descriptor is the complete C++
                // runtime value; no owning Ndarray is fabricated.
                args.insert(
                    {arg_name, aot::IValue::create(runtime_argument)});
              }
            } else {
              auto &val = pyarg.template cast<Ndarray &>();
              args.insert({arg_name, aot::IValue::create(val)});
            }
          } else if (tag == aot::ArgKind::kTexture ||
                     tag == aot::ArgKind::kRWTexture) {
            auto &val = pyarg.template cast<Texture &>();
            args.insert({arg_name, aot::IValue::create(val)});
          } else if (tag == aot::ArgKind::kScalar) {
            auto expected_dtype = arg.dtype();
            insert_scalar_arg(arg_name, expected_dtype, pyarg);
          } else if (tag == aot::ArgKind::kMatrix) {
            auto type_id = arg.dtype()->template as<PrimitiveType>()->type;
            switch (type_id) {
              case PrimitiveTypeID::f16: {
                auto arr = pyarg.template cast<py::array_t<
                    float32, py::array::c_style | py::array::forcecast>>();
                py::buffer_info buffer_info = arr.request();
                auto length = buffer_info.size;
                auto ptr = reinterpret_cast<intptr_t>(buffer_info.ptr);
                auto dimensions = matrix_dimensions(buffer_info);
                auto byte_size = sizeof(uint16) * length;
                TI_ERROR_IF(byte_size > 128,
                            "Graph Matrix runtime argument {} uses {} bytes; "
                            "the limit is 128 bytes",
                            arg_name, byte_size);

                std::unique_ptr<char[]> data(new char[byte_size]);
                for (uint32_t i = 0; i < length; i++) {
                  uint16 half = fp16_ieee_from_fp32_value(
                      reinterpret_cast<float32 *>(ptr)[i]);
                  reinterpret_cast<uint16 *>(data.get())[i] = half;
                }
                matrix_buffers.emplace_back(std::move(data));

                matrices.emplace_back(Matrix(
                    length, arg.dtype(),
                    reinterpret_cast<intptr_t>(matrix_buffers.back().get()),
                    dimensions[0], dimensions[1], dimensions[2]));
                args.insert({arg_name, aot::IValue::create(matrices.back())});
                break;
              }
#define PER_C_TYPE(type, ctype)                                           \
  case PrimitiveTypeID::type: {                                           \
    auto arr = pyarg.template cast<py::array_t<                          \
        ctype, py::array::c_style | py::array::forcecast>>();            \
    py::buffer_info buffer_info = arr.request();                          \
    auto length = buffer_info.size;                                       \
    auto ptr = reinterpret_cast<intptr_t>(buffer_info.ptr);               \
    auto dimensions = matrix_dimensions(buffer_info);                     \
    auto byte_size = sizeof(ctype) * length;                              \
    TI_ERROR_IF(byte_size > 128,                                          \
                "Graph Matrix runtime argument {} uses {} bytes; the "    \
                "limit is 128 bytes",                                    \
                arg_name, byte_size);                                     \
                                                                           \
    std::unique_ptr<char[]> data(new char[byte_size]);                    \
    std::memcpy(data.get(), reinterpret_cast<char *>(ptr),                \
                byte_size);                                               \
    matrix_buffers.emplace_back(std::move(data));                         \
                                                                          \
    matrices.emplace_back(                                                \
        Matrix(length, arg.dtype(),                                       \
               reinterpret_cast<intptr_t>(matrix_buffers.back().get()),   \
               dimensions[0], dimensions[1], dimensions[2]));            \
    args.insert({arg_name, aot::IValue::create(matrices.back())});        \
    break;                                                                \
  }
#include "taichi/inc/data_type_with_c_type.inc.h"
#undef PER_C_TYPE
              default:
                TI_ERROR("Unsupported scalar type {}",
                         arg.dtype()->to_string());
            }
          } else {
            TI_NOT_IMPLEMENTED;
          }
        }
        // Argument conversion above touches Python objects and must keep the
        // GIL. Once it is complete, graph execution is entirely native. In
        // particular, releasing the GIL here lets independent Python callers
        // reach CompiledGraphJITCache::run_mutex instead of being accidentally
        // serialized by Python. The cache owns the C++ transaction boundary;
        // Python objects passed in pyargs remain alive for this call.
        py::gil_scoped_release release;
        return action(args);
      };

  auto jit_run_graph =
      [with_graph_arguments](
          aot::CompiledGraph *self, const CompileConfig &compile_config,
          const py::dict &pyargs, aot::CompiledGraphJITCache *cache,
          Ndarray *bounded_predicate = nullptr, int bounded_max_iterations = 0,
          bool continue_while_nonzero = true,
          Ndarray *conditional_selector = nullptr,
          const std::vector<int> *branch_dispatch_counts = nullptr,
          int conditional_type = -1, int default_branch = -1,
          Ndarray *vulkan_predicate = nullptr,
          Ndarray *vulkan_counter = nullptr, Ndarray *vulkan_status = nullptr,
          std::size_t vulkan_initial_dispatch_count = 0,
          bool vulkan_execute_initial_dispatches = true,
          std::uint32_t vulkan_strategy = 0,
          aot::CompiledGraphStructuredResult *vulkan_result = nullptr,
          bool vulkan_wait_for_result = true,
          const std::vector<int> *vulkan_chunk_iterations = nullptr,
          const std::vector<std::uint32_t> *vulkan_chunk_strategies = nullptr,
          const VulkanNestedGraphRequest *vulkan_nested = nullptr,
          bool cuda_masked_control = false,
          const CudaNestedGraphRequest *cuda_nested = nullptr,
          bool cuda_concurrent_batch_lane = false) -> bool {
    return with_graph_arguments(self, pyargs, [&](const auto &args) -> bool {
      if (vulkan_nested != nullptr) {
        TI_ASSERT(cache != nullptr);
        auto result = self->jit_run_bounded_vulkan_nested_sequence_cached(
            compile_config, args, *cache, vulkan_nested->outer_predicate,
            vulkan_nested->outer_counter, vulkan_nested->outer_status,
            vulkan_nested->inner_controls,
            vulkan_nested->outer_condition_dispatch_count,
            vulkan_nested->outer_max_iterations,
            vulkan_nested->wait_for_result);
        if (vulkan_nested->result != nullptr) {
          *vulkan_nested->result = std::move(result);
          return vulkan_nested->result->submitted;
        }
        return result.submitted;
      }
      if (cuda_nested != nullptr) {
        TI_ASSERT(cache != nullptr);
        return self->jit_submit_bounded_cuda_nested_sequence_cached(
            compile_config, args, *cache, cuda_nested->outer_predicate,
            cuda_nested->outer_counter, cuda_nested->outer_status,
            cuda_nested->inner_controls,
            cuda_nested->outer_condition_dispatch_count,
            cuda_nested->outer_max_iterations,
            cuda_nested->allow_device_update);
      }
      if (vulkan_chunk_iterations != nullptr) {
        TI_ASSERT(cache != nullptr);
        TI_ASSERT(vulkan_chunk_strategies != nullptr);
        return self->jit_submit_bounded_vulkan_compound_cached(
            compile_config, args, *cache, vulkan_predicate, vulkan_counter,
            vulkan_status, vulkan_initial_dispatch_count,
            *vulkan_chunk_iterations, *vulkan_chunk_strategies);
      }
      if (bounded_predicate != nullptr) {
        TI_ASSERT(cache != nullptr);
        return cuda_masked_control
                   ? self->jit_run_bounded_cuda_masked_cached(
                         compile_config, args, *cache, bounded_predicate,
                         bounded_max_iterations, continue_while_nonzero)
                   : self->jit_run_bounded_cuda_cached(
                         compile_config, args, *cache, bounded_predicate,
                         bounded_max_iterations, continue_while_nonzero);
      }
      if (conditional_selector != nullptr) {
        TI_ASSERT(cache != nullptr);
        TI_ASSERT(branch_dispatch_counts != nullptr);
        return cuda_masked_control
                   ? self->jit_run_conditional_cuda_masked_cached(
                         compile_config, args, *cache, conditional_selector,
                         *branch_dispatch_counts, conditional_type,
                         default_branch)
                   : self->jit_run_conditional_cuda_cached(
                         compile_config, args, *cache, conditional_selector,
                         *branch_dispatch_counts, conditional_type,
                         default_branch);
      }
      if (vulkan_predicate != nullptr) {
        TI_ASSERT(cache != nullptr);
        TI_ASSERT(vulkan_counter != nullptr);
        auto result = self->jit_run_bounded_vulkan_cached(
            compile_config, args, *cache, vulkan_predicate, vulkan_counter,
            vulkan_status, vulkan_initial_dispatch_count,
            bounded_max_iterations, vulkan_execute_initial_dispatches,
            vulkan_strategy, vulkan_wait_for_result);
        if (vulkan_result != nullptr) {
          *vulkan_result = result;
        }
        return result.submitted;
      }
      if (cache) {
        self->jit_run_cached(compile_config, args, *cache,
                             cuda_concurrent_batch_lane);
      } else {
        self->jit_run(compile_config, args);
      }
      return true;
    });
  };

#if defined(TI_WITH_VULKAN)
  py::class_<gfx::ExternalGraphCommand, std::shared_ptr<gfx::ExternalGraphCommand>>(
      m, "_GfxExternalGraphCommand");
  py::class_<gfx::FixedGraphRecording, std::shared_ptr<gfx::FixedGraphRecording>>(
      m, "_VulkanFixedGraphRecording")
      .def("run", &gfx::FixedGraphRecording::run, py::call_guard<py::gil_scoped_release>())
      .def("close", &gfx::FixedGraphRecording::close, py::call_guard<py::gil_scoped_release>())
      .def("argument_bytes", &gfx::FixedGraphRecording::argument_bytes,
           py::call_guard<py::gil_scoped_release>())
      .def("uses_secondary_commands", &gfx::FixedGraphRecording::uses_secondary_commands,
           py::call_guard<py::gil_scoped_release>());
  m.def("_prepare_vulkan_graph_recording",
        [with_graph_arguments](Program &program, const py::list &sources, const py::dict &args) {
          std::vector<gfx::GraphRecordingSource> native_sources;
          for (const auto &source : sources) {
            if (py::isinstance<aot::CompiledGraph>(source)) {
              native_sources.push_back({source.cast<aot::CompiledGraph *>()});
            } else {
              native_sources.push_back({source.cast<std::shared_ptr<gfx::ExternalGraphCommand>>()});
            }
          }
          auto schema = gfx::graph_recording_argument_schema(native_sources);
          return with_graph_arguments(&schema, args, [&](const auto &converted) {
            return program.create_vulkan_graph_recording(native_sources, converted);
          });
        }, py::keep_alive<0, 1>());
#endif

#if defined(TI_WITH_CUDA)
  py::class_<cuda::GraphBindingFrame, std::shared_ptr<cuda::GraphBindingFrame>>(
      m, "_CudaGraphBindingFrame");
  py::class_<cuda::GraphBindingExecutor>(m, "_CudaGraphBindingExecutor")
      .def_static("available", &cuda::GraphBindingExecutor::available)
      .def_static("supports_capture_commands", [] { return true; })
      .def_static("retains_completion_events_until_close", [] { return true; })
      .def(py::init<const aot::CompiledGraph &, const CompileConfig &,
                    Program &>(),
           py::keep_alive<1, 2>(), py::keep_alive<1, 4>())
      .def("prepare",
           [with_graph_arguments](cuda::GraphBindingExecutor &self,
                                  const py::dict &args) {
             return with_graph_arguments(&self.graph(), args,
                                         [&](const auto &converted) {
                                           return self.prepare(converted);
                                         });
           })
      .def("run", &cuda::GraphBindingExecutor::run,
           py::call_guard<py::gil_scoped_release>())
      .def("close", &cuda::GraphBindingExecutor::close,
           py::call_guard<py::gil_scoped_release>())
      .def("snapshot", &cuda::GraphBindingExecutor::snapshot,
           py::call_guard<py::gil_scoped_release>());

  py::class_<cuda::GraphMemoryPool>(m, "_CudaGraphMemoryPool")
      .def_static("available", &cuda::GraphMemoryPool::available)
      .def(py::init<Program &, std::uint64_t>(), py::keep_alive<1, 2>())
      .def("create_ndarray", &cuda::GraphMemoryPool::create_ndarray,
           py::return_value_policy::reference,
           py::call_guard<py::gil_scoped_release>())
      .def("trim", &cuda::GraphMemoryPool::trim,
           py::call_guard<py::gil_scoped_release>())
      .def("close", &cuda::GraphMemoryPool::close,
           py::call_guard<py::gil_scoped_release>())
      .def("snapshot", &cuda::GraphMemoryPool::snapshot,
           py::call_guard<py::gil_scoped_release>());
#endif

  py::class_<aot::CompiledGraphJITCache>(m, "CompiledGraphJITCache")
      .def(py::init<>())
      .def("clear_runtime_state",
           &aot::CompiledGraphJITCache::clear_runtime_state)
      .def("retire_snode_tree_runtime_state",
           &aot::CompiledGraphJITCache::retire_snode_tree_runtime_state)
      .def("_set_stable_replay_optimization",
           [](aot::CompiledGraphJITCache &cache, bool enabled) {
             const bool previous =
                 cache.stable_replay_optimization_enabled.exchange(
                     enabled, std::memory_order_relaxed);
             if (previous != enabled) {
               cache.clear_runtime_state();
             }
           })
      .def("_debug_graph_stats", [](aot::CompiledGraphJITCache &cache,
                                     bool enable_diagnostics) {
        if (enable_diagnostics && !cache.graph_diagnostics_enabled) {
          const auto previous = cache.debug_graph_stats();
          if (previous.stats.last_path !=
                  aot::CompiledGraphExecutionPath::none ||
              previous.stats.last_fallback_reason !=
                  aot::CompiledGraphFallbackReason::none) {
            cache.graph_diagnostics_counters_complete = false;
          }
          // This is a private test/debug opt-in. Public execution_stats()
          // passes false and therefore cannot mutate the production replay
          // mode or its future cost.
          cache.graph_diagnostics_enabled = true;
        }
        const auto snapshot = cache.debug_graph_stats();
        const auto &stats = snapshot.stats;
        auto backend_name = [](aot::CompiledGraphBackend backend) {
          switch (backend) {
            case aot::CompiledGraphBackend::cuda:
              return "cuda";
            case aot::CompiledGraphBackend::vulkan:
              return "vulkan";
            case aot::CompiledGraphBackend::none:
              return "none";
          }
          return "unknown";
        };
        auto path_name = [](aot::CompiledGraphExecutionPath path) {
          switch (path) {
            case aot::CompiledGraphExecutionPath::ordinary_fallback:
              return "ordinary_fallback";
            case aot::CompiledGraphExecutionPath::cuda_capture:
              return "cuda_capture";
            case aot::CompiledGraphExecutionPath::cuda_exact_replay:
              return "cuda_exact_replay";
            case aot::CompiledGraphExecutionPath::cuda_patched_replay:
              return "cuda_patched_replay";
            case aot::CompiledGraphExecutionPath::cuda_masked_capture:
              return "cuda_masked_capture";
            case aot::CompiledGraphExecutionPath::cuda_masked_replay:
              return "cuda_masked_replay";
            case aot::CompiledGraphExecutionPath::cuda_masked_patched_replay:
              return "cuda_masked_patched_replay";
            case aot::CompiledGraphExecutionPath::
                cuda_device_update_nested_capture:
              return "cuda_device_update_nested_capture";
            case aot::CompiledGraphExecutionPath::
                cuda_device_update_nested_replay:
              return "cuda_device_update_nested_replay";
            case aot::CompiledGraphExecutionPath::
                cuda_device_update_nested_patched_replay:
              return "cuda_device_update_nested_patched_replay";
            case aot::CompiledGraphExecutionPath::vulkan_record:
              return "vulkan_record";
            case aot::CompiledGraphExecutionPath::vulkan_replay:
              return "vulkan_replay";
            case aot::CompiledGraphExecutionPath::vulkan_patched_replay:
              return "vulkan_patched_replay";
            case aot::CompiledGraphExecutionPath::none:
              return "none";
          }
          return "unknown";
        };
        auto fallback_name = [](aot::CompiledGraphFallbackReason reason) {
          switch (reason) {
            case aot::CompiledGraphFallbackReason::debug_mode:
              return "debug_mode";
            case aot::CompiledGraphFallbackReason::insufficient_dispatches:
              return "insufficient_dispatches";
            case aot::CompiledGraphFallbackReason::unsupported_arguments:
              return "unsupported_arguments";
            case aot::CompiledGraphFallbackReason::resource_unavailable:
              return "resource_unavailable";
            case aot::CompiledGraphFallbackReason::structural_unsupported:
              return "structural_unsupported";
            case aot::CompiledGraphFallbackReason::transient_driver_failure:
              return "transient_driver_failure";
            case aot::CompiledGraphFallbackReason::fatal_driver_failure:
              return "fatal_driver_failure";
            case aot::CompiledGraphFallbackReason::retry_backoff:
              return "retry_backoff";
            case aot::CompiledGraphFallbackReason::runtime_mode:
              return "runtime_mode";
            case aot::CompiledGraphFallbackReason::replay_slot_saturated:
              return "replay_slot_saturated";
            case aot::CompiledGraphFallbackReason::none:
              return "none";
          }
          return "unknown";
        };
        py::dict result;
        result["backend"] = backend_name(stats.backend);
        result["last_path"] = path_name(stats.last_path);
        result["last_fallback_reason"] =
            fallback_name(stats.last_fallback_reason);
        result["attempts"] = stats.attempts;
        result["ordinary_fallbacks"] = stats.ordinary_fallbacks;
        result["capture_attempts"] = stats.capture_attempts;
        result["captures"] = stats.captures;
        result["exact_replays"] = stats.exact_replays;
        result["patched_replays"] = stats.patched_replays;
        result["masked_captures"] = stats.masked_captures;
        result["masked_replays"] = stats.masked_replays;
        result["masked_patched_replays"] = stats.masked_patched_replays;
        result["recaptures"] = stats.recaptures;
        result["records"] = stats.records;
        result["replays"] = stats.replays;
        result["structural_fallbacks"] = stats.structural_fallbacks;
        result["transient_failures"] = stats.transient_failures;
        result["retry_backoff_fallbacks"] =
            stats.retry_backoff_fallbacks;
        result["replay_slot_saturation_fallbacks"] =
            stats.replay_slot_saturation_fallbacks;
        result["capture_exceptions"] = stats.capture_exceptions;
        result["zero_arg_captures"] = stats.zero_arg_captures;
        result["asynchronous_control_updates"] =
            stats.asynchronous_control_updates;
        result["deferred_replay_waits"] = stats.deferred_replay_waits;
        result["peak_deferred_replay_batches"] =
            stats.peak_deferred_replay_batches;
        result["zero_arg_eligible"] = stats.zero_arg_eligible;
        result["known_persistent_argument_bytes"] =
            stats.known_persistent_argument_bytes;
        result["known_bounded_control_bytes"] =
            stats.known_bounded_control_bytes;
        result["known_bounded_update_groups"] =
            stats.known_bounded_update_groups;
        result["known_bounded_updater_dispatches"] =
            stats.known_bounded_updater_dispatches;
        result["known_bounded_grouped_payloads"] =
            stats.known_bounded_grouped_payloads;
        result["known_bounded_producer_fused_groups"] =
            stats.known_bounded_producer_fused_groups;
        result["known_bounded_payloads"] = stats.known_bounded_payloads;
        result["last_bounded_useful_lanes"] =
            stats.last_bounded_useful_lanes;
        result["last_bounded_physical_blocks"] =
            stats.last_bounded_physical_blocks;
        result["last_bounded_physical_threads"] =
            stats.last_bounded_physical_threads;
        result["last_bounded_baseline_blocks"] =
            stats.last_bounded_baseline_blocks;
        result["last_bounded_zero_payloads"] =
            stats.last_bounded_zero_payloads;
        result["bounded_physical_observation_available"] =
            stats.bounded_physical_observation_available;
        result["bounded_update_replays"] =
            stats.bounded_update_replays;
        result["bounded_update_state_changes"] =
            stats.bounded_update_state_changes;
        result["bounded_update_cache_hits"] =
            stats.bounded_update_cache_hits;
        result["bounded_node_api_calls"] =
            stats.bounded_node_api_calls;
        result["known_bounded_max_group_size"] =
            stats.known_bounded_max_group_size;
        result["known_compiled_tasks"] =
            snapshot.known_compiled_tasks;
        result["known_compiled_dispatches"] =
            snapshot.known_compiled_dispatches;
        result["runtime_binding_plan_slots"] =
            snapshot.runtime_binding_plan_slots;
        result["backend_replay_signature_slots"] =
            snapshot.backend_replay_signature_slots;
        result["backend_replay_signature_slot_capacity"] =
            snapshot.backend_replay_signature_slot_capacity;
        result["last_driver_error"] = stats.last_driver_error;
        result["retry_backoff_remaining"] =
            stats.retry_backoff_remaining;
        result["effect_reads"] = stats.effect_reads;
        result["effect_writes"] = stats.effect_writes;
        result["dependency_barriers"] = stats.dependency_barriers;
        result["exit_barriers"] = stats.exit_barriers;
        result["barrier_deferrals"] = stats.barrier_deferrals;
        result["rar_elisions"] = stats.rar_elisions;
        result["consecutive_transient_failures"] =
            stats.consecutive_transient_failures;
        result["diagnostics_previously_enabled"] =
            snapshot.diagnostics_previously_enabled;
        result["diagnostics_counters_complete"] =
            snapshot.diagnostics_counters_complete;
        result["replay_attribution_enabled"] =
            snapshot.replay_attribution_enabled;
        result["replay_calls"] = snapshot.replay_calls;
        result["replay_total_ns"] = snapshot.replay_total_ns;
        result["replay_snode_guard_wait_ns"] =
            snapshot.replay_snode_guard_ns;
        result["replay_resource_guard_wait_ns"] =
            snapshot.replay_resource_guard_ns;
        result["replay_cuda_submission_wait_ns"] =
            snapshot.replay_cuda_submission_lock_ns;
        result["replay_cache_wait_ns"] = snapshot.replay_cache_wait_ns;
        result["replay_binding_plan_ns"] =
            snapshot.replay_binding_plan_ns;
        result["replay_resource_retain_ns"] =
            snapshot.replay_resource_retain_ns;
        result["replay_snode_validation_ns"] =
            snapshot.replay_snode_validation_ns;
        result["replay_backend_ns"] = snapshot.replay_backend_ns;
        result["replay_signature_ns"] = snapshot.replay_signature_ns;
        result["replay_binding_plan_hits"] =
            snapshot.replay_binding_plan_hits;
        result["replay_binding_plan_misses"] =
            snapshot.replay_binding_plan_misses;
        result["replay_signature_fast_hits"] =
            snapshot.replay_signature_hits;
        result["replay_signature_fast_misses"] =
            snapshot.replay_signature_misses;
        result["replay_snode_guard_acquisitions"] =
            snapshot.replay_snode_guard_acquisitions;
        result["replay_snode_guard_elisions"] =
            snapshot.replay_snode_guard_elisions;
        return result;
      }, py::arg("enable_diagnostics") = true);

  py::class_<aot::CompiledGraph>(m, "CompiledGraph")
      .def_property_readonly(
          "_owned_jit_dispatch_sources", [](const py::object &owner) {
            const auto &graph = owner.cast<const aot::CompiledGraph &>();
            py::list sources;
            for (const auto &dispatch : graph.dispatches) {
              const bool owned = std::any_of(
                  graph.owned_jit_kernels.begin(), graph.owned_jit_kernels.end(),
                  [&](const auto &kernel) { return kernel.get() == dispatch.ti_kernel; });
              TI_ERROR_IF(!owned, "Graph dispatch does not own a synthetic JIT kernel");
              // Recording handles retain the original CompiledGraph, which
              // owns the synthetic IR. No executable/Python deserialization.
              sources.append(py::make_tuple(
                  py::cast(dispatch.ti_kernel,
                           py::return_value_policy::reference_internal, owner),
                  dispatch.symbolic_args));
            }
            return sources;
          })
      .def_property_readonly(
          "_has_dispatch_labels",
          [](const aot::CompiledGraph &graph) {
            return graph.has_dispatch_labels();
          })
      .def_property_readonly(
          "_dispatch_metadata",
          [](const aot::CompiledGraph &graph) {
            py::list result;
            auto append = [&](const std::string &kernel_name,
                              const std::string &dispatch_label,
                              const std::vector<aot::Arg> &args,
                              const GraphKernelMetadata &metadata,
                              std::uint64_t logical_dispatch_id,
                              const std::string &logical_kernel_identity) {
              py::dict item = graph_kernel_metadata_to_python(metadata);
              item["kernel_name"] = kernel_name;
              item["dispatch_label"] = dispatch_label;
              item["logical_kernel_identity"] = logical_kernel_identity;
              item["logical_dispatch_id"] =
                  logical_dispatch_id ==
                          std::numeric_limits<std::uint64_t>::max()
                      ? py::none()
                      : py::cast(logical_dispatch_id);
              py::list symbolic_args;
              for (const auto &arg : args) {
                py::dict encoded;
                encoded["name"] = arg.name;
                encoded["tag"] = static_cast<int>(arg.tag);
                symbolic_args.append(std::move(encoded));
              }
              item["symbolic_args"] = std::move(symbolic_args);
              result.append(std::move(item));
            };
            for (const auto &dispatch : graph.dispatches) {
              if (dispatch.source_dispatches.empty()) {
                append(dispatch.kernel_name, dispatch.dispatch_label,
                       dispatch.symbolic_args,
                       dispatch.graph_metadata,
                       std::numeric_limits<std::uint64_t>::max(), "");
                continue;
              }
              for (const auto &source : dispatch.source_dispatches) {
                append(source.kernel_name, source.dispatch_label,
                       source.symbolic_args,
                       source.graph_metadata, source.logical_dispatch_id,
                       source.logical_kernel_identity);
              }
            }
            return result;
          })
      .def_property_readonly(
          "_composer_stats",
          [](const aot::CompiledGraph &graph) {
            py::dict result;
            std::uint32_t source_dispatches = 0;
            std::uint32_t applied_groups = 0;
            std::uint64_t compiled_tasks = 0;
            bool tasks_known = true;
            py::list source_groups;
            for (const auto &dispatch : graph.dispatches) {
              const auto source_count = static_cast<std::uint32_t>(
                  std::max<std::size_t>(1,
                      dispatch.source_dispatches.size()));
              source_dispatches += source_count;
              applied_groups += source_count > 1 ? 1 : 0;
              if (source_count > 1) {
                py::list group;
                for (const auto &source : dispatch.source_dispatches) {
                  if (source.logical_dispatch_id ==
                      std::numeric_limits<std::uint64_t>::max()) {
                    group.append(py::none());
                  } else {
                    group.append(source.logical_dispatch_id);
                  }
                }
                source_groups.append(std::move(group));
              }
              if (dispatch.compiled_task_count ==
                  std::numeric_limits<std::uint32_t>::max()) {
                tasks_known = false;
              } else {
                compiled_tasks += dispatch.compiled_task_count;
              }
            }
            result["source_dispatches"] = source_dispatches;
            result["physical_dispatches"] = graph.dispatches.size();
            result["applied_groups"] = applied_groups;
            result["source_groups"] = std::move(source_groups);
            result["compiled_tasks"] =
                tasks_known ? py::cast(compiled_tasks) : py::none();
            result["lowering_available"] = true;
            return result;
          })
      .def_property_readonly(
          "_snode_tree_dependencies",
          [](const aot::CompiledGraph &graph) {
            py::list dependencies;
            for (const auto &dependency : graph.snode_tree_dependencies) {
              dependencies.append(
                  py::make_tuple(dependency.tree_id, dependency.generation));
            }
            return dependencies;
          })
      .def_property_readonly(
          "_snode_tree_dependency_info",
          [](const aot::CompiledGraph &graph) {
            py::list dependencies;
            for (const auto &dependency : graph.snode_tree_dependencies) {
              dependencies.append(py::make_tuple(
                  dependency.tree_id, dependency.generation,
                  dependency.layout_fingerprint));
            }
            return dependencies;
          })
      .def("jit_run",
           [jit_run_graph](aot::CompiledGraph *self,
                           const CompileConfig &compile_config,
                           const py::dict &pyargs) {
             jit_run_graph(self, compile_config, pyargs, nullptr);
           })
      .def("jit_run_cached",
           [jit_run_graph](aot::CompiledGraph *self,
                           const CompileConfig &compile_config,
                           const py::dict &pyargs,
                           aot::CompiledGraphJITCache &cache) {
             jit_run_graph(self, compile_config, pyargs, &cache);
           })
      .def("jit_run_cached_cuda_concurrent_batch",
           [jit_run_graph](aot::CompiledGraph *self,
                           const CompileConfig &compile_config,
                           const py::dict &pyargs,
                           aot::CompiledGraphJITCache &cache) {
             jit_run_graph(self, compile_config, pyargs, &cache,
                           /*predicate=*/nullptr,
                           /*max_iterations=*/0,
                           /*continue_while_nonzero=*/true,
                           /*conditional_selector=*/nullptr,
                           /*branch_dispatch_counts=*/nullptr,
                           /*conditional_type=*/-1,
                           /*default_branch=*/-1,
                           /*vulkan_predicate=*/nullptr,
                           /*vulkan_counter=*/nullptr,
                           /*vulkan_status=*/nullptr,
                           /*vulkan_initial_dispatch_count=*/0,
                           /*vulkan_execute_initial_dispatches=*/true,
                           /*vulkan_strategy=*/0,
                           /*vulkan_result=*/nullptr,
                           /*vulkan_wait_for_result=*/true,
                           /*vulkan_chunk_iterations=*/nullptr,
                           /*vulkan_chunk_strategies=*/nullptr,
                           /*vulkan_nested=*/nullptr,
                           /*cuda_masked_control=*/false,
                           /*cuda_nested=*/nullptr,
                           /*cuda_concurrent_batch_lane=*/true);
           })
      .def("jit_run_bounded_cuda_cached",
           [jit_run_graph](aot::CompiledGraph *self,
                           const CompileConfig &compile_config,
                           const py::dict &pyargs,
                           aot::CompiledGraphJITCache &cache,
                           Ndarray &predicate, int max_iterations,
                           bool continue_while_nonzero) {
             return jit_run_graph(
                 self, compile_config, pyargs, &cache, &predicate,
                 max_iterations, continue_while_nonzero);
           },
           py::arg("compile_config"), py::arg("args"),
           py::arg("cache"), py::arg("predicate"),
           py::arg("max_iterations"),
           py::arg("continue_while_nonzero"))
      .def("jit_run_bounded_cuda_masked_cached",
           [jit_run_graph](aot::CompiledGraph *self,
                           const CompileConfig &compile_config,
                           const py::dict &pyargs,
                           aot::CompiledGraphJITCache &cache,
                           Ndarray &predicate, int max_iterations,
                           bool continue_while_nonzero) {
             return jit_run_graph(
                 self, compile_config, pyargs, &cache, &predicate,
                 max_iterations, continue_while_nonzero, nullptr, nullptr,
                 -1, -1, nullptr, nullptr, nullptr, 0, true, 0, nullptr,
                 true, nullptr, nullptr, nullptr,
                 /*cuda_masked_control=*/true);
           },
           py::arg("compile_config"), py::arg("args"),
           py::arg("cache"), py::arg("predicate"),
           py::arg("max_iterations"),
           py::arg("continue_while_nonzero"))
      .def("jit_run_bounded_vulkan_cached",
           [jit_run_graph](aot::CompiledGraph *self,
                           const CompileConfig &compile_config,
                           const py::dict &pyargs,
                           aot::CompiledGraphJITCache &cache,
                           Ndarray &predicate, Ndarray &counter,
                           std::size_t initial_dispatch_count,
                           int max_iterations, Ndarray *status,
                           bool execute_initial_dispatches,
                           std::uint32_t strategy) {
             aot::CompiledGraphStructuredResult result;
             jit_run_graph(
                 self, compile_config, pyargs, &cache, nullptr,
                 max_iterations, true, nullptr, nullptr, -1, -1,
                 &predicate, &counter, status,
                 initial_dispatch_count, execute_initial_dispatches,
                 strategy, &result);
             py::dict encoded;
             encoded["submitted"] = result.submitted;
             encoded["strategy"] = result.strategy;
             encoded["logical_iterations"] = result.logical_iterations;
             encoded["predicate"] = result.predicate;
             encoded["counter"] = result.counter;
             encoded["status"] = result.status;
             encoded["initial_status"] = result.initial_status;
             encoded["encoded_iterations"] = result.encoded_iterations;
             encoded["indirect_dispatches"] =
                 result.indirect_dispatches;
             encoded["controller_dispatches"] =
                 result.controller_dispatches;
             encoded["controller_invocations"] =
                 result.controller_invocations;
             encoded["zero_dispatches"] = result.zero_dispatches;
             encoded["control_bytes"] = result.control_bytes;
             encoded["observation_bytes"] = result.observation_bytes;
             return encoded;
           },
           py::arg("compile_config"), py::arg("args"), py::arg("cache"),
           py::arg("predicate"), py::arg("counter"),
           py::arg("initial_dispatch_count"),
           py::arg("max_iterations"), py::arg("status") = nullptr,
           py::arg("execute_initial_dispatches") = true,
           py::arg("strategy") = 0)
      .def("jit_submit_bounded_vulkan_cached",
           [jit_run_graph](aot::CompiledGraph *self,
                           const CompileConfig &compile_config,
                           const py::dict &pyargs,
                           aot::CompiledGraphJITCache &cache,
                           Ndarray &predicate, Ndarray &counter,
                           std::size_t initial_dispatch_count,
                           int max_iterations, Ndarray *status,
                           bool execute_initial_dispatches,
                           std::uint32_t strategy) {
             return jit_run_graph(
                 self, compile_config, pyargs, &cache, nullptr,
                 max_iterations, true, nullptr, nullptr, -1, -1,
                 &predicate, &counter, status,
                 initial_dispatch_count, execute_initial_dispatches,
                 strategy, nullptr, false);
           },
           py::arg("compile_config"), py::arg("args"), py::arg("cache"),
           py::arg("predicate"), py::arg("counter"),
           py::arg("initial_dispatch_count"),
           py::arg("max_iterations"), py::arg("status") = nullptr,
           py::arg("execute_initial_dispatches") = true,
           py::arg("strategy") = 0)
      .def("jit_submit_bounded_vulkan_compound_cached",
           [jit_run_graph](
               aot::CompiledGraph *self,
               const CompileConfig &compile_config,
               const py::dict &pyargs,
               aot::CompiledGraphJITCache &cache,
               Ndarray &predicate, Ndarray &counter,
               std::size_t initial_dispatch_count,
               const std::vector<int> &chunk_iterations,
               Ndarray *status,
               const std::vector<std::uint32_t> &strategies) {
             TI_ERROR_IF(
                 chunk_iterations.empty() ||
                     chunk_iterations.size() != strategies.size(),
                 "Vulkan structured compound submission requires one "
                 "strategy for every non-empty chunk");
             if (get_environ_config(
                     "TI_VULKAN_COMPOUND_SINGLE_PREPARATION", 1) == 0) {
               bool execute_initial_dispatches = true;
               for (std::size_t chunk = 0;
                    chunk < chunk_iterations.size(); ++chunk) {
                 TI_ERROR_IF(
                     chunk_iterations[chunk] <= 0,
                     "Vulkan structured compound chunk sizes must be "
                     "positive");
                 if (!jit_run_graph(
                         self, compile_config, pyargs, &cache, nullptr,
                         chunk_iterations[chunk], true, nullptr, nullptr,
                         -1, -1, &predicate, &counter, status,
                         initial_dispatch_count,
                         execute_initial_dispatches, strategies[chunk],
                         nullptr, false)) {
                   return false;
                 }
                 execute_initial_dispatches = false;
               }
               return true;
             }
             return jit_run_graph(
                 self, compile_config, pyargs, &cache, nullptr, 0, true,
                 nullptr, nullptr, -1, -1, &predicate, &counter, status,
                 initial_dispatch_count,
                 /*vulkan_execute_initial_dispatches=*/true,
                 /*vulkan_strategy=*/0, nullptr,
                 /*vulkan_wait_for_result=*/false, &chunk_iterations,
                 &strategies);
           },
           py::arg("compile_config"), py::arg("args"), py::arg("cache"),
           py::arg("predicate"), py::arg("counter"),
           py::arg("initial_dispatch_count"),
           py::arg("chunk_iterations"), py::arg("status"),
           py::arg("strategies"))
      .def(
          "jit_run_bounded_vulkan_nested_cached",
          [jit_run_graph](
              aot::CompiledGraph *self,
              const CompileConfig &compile_config,
              const py::dict &pyargs,
              aot::CompiledGraphJITCache &cache,
              Ndarray &outer_predicate, Ndarray &outer_counter,
              Ndarray &inner_predicate, Ndarray &inner_counter,
              std::size_t outer_condition_dispatch_count,
              std::size_t inner_condition_dispatch_begin,
              std::size_t inner_body_dispatch_begin,
              std::size_t outer_suffix_dispatch_begin,
              int outer_max_iterations, int inner_max_iterations,
              int inner_chunk_size, Ndarray *outer_status,
              Ndarray *inner_status,
              bool wait_for_result) {
            aot::CompiledGraphNestedStructuredResult result;
            VulkanNestedGraphRequest request;
            request.outer_predicate = &outer_predicate;
            request.outer_counter = &outer_counter;
            request.outer_status = outer_status;
            request.outer_condition_dispatch_count =
                outer_condition_dispatch_count;
            request.outer_max_iterations = outer_max_iterations;
            request.inner_controls.push_back(
                {&inner_predicate, &inner_counter, inner_status,
                 inner_condition_dispatch_begin, inner_body_dispatch_begin,
                 outer_suffix_dispatch_begin, inner_max_iterations,
                 inner_chunk_size});
            request.wait_for_result = wait_for_result;
            request.result = wait_for_result ? &result : nullptr;
            jit_run_graph(
                self, compile_config, pyargs, &cache, nullptr, 0,
                true, nullptr, nullptr, -1, -1, nullptr, nullptr,
                nullptr, 0, true, 0, nullptr, true, nullptr,
                nullptr, &request);
            py::dict encoded;
            encoded["submitted"] = wait_for_result ? result.submitted : true;
            encoded["inner_region_count"] = result.inner_region_count;
            encoded["outer_logical_iterations"] =
                result.outer_logical_iterations;
            encoded["outer_encoded_iterations"] =
                result.outer_encoded_iterations;
            encoded["outer_initial_predicate"] =
                result.outer_initial_predicate;
            encoded["outer_final_predicate"] =
                result.outer_final_predicate;
            encoded["outer_initial_counter"] =
                result.outer_initial_counter;
            encoded["outer_final_counter"] =
                result.outer_final_counter;
            encoded["outer_initial_status"] =
                result.outer_initial_status;
            encoded["outer_final_status"] =
                result.outer_final_status;
            encoded["inner_logical_iterations"] =
                result.inner_logical_iterations;
            encoded["inner_encoded_iterations"] =
                result.inner_encoded_iterations;
            encoded["inner_initial_counters"] =
                result.inner_initial_counters;
            encoded["inner_final_counters"] =
                result.inner_final_counters;
            encoded["inner_final_predicates"] =
                result.inner_final_predicates;
            encoded["inner_initial_statuses"] =
                result.inner_initial_statuses;
            encoded["inner_final_statuses"] =
                result.inner_final_statuses;
            encoded["indirect_dispatches"] =
                result.indirect_dispatches;
            encoded["controller_dispatches"] =
                result.controller_dispatches;
            encoded["controller_invocations"] =
                result.controller_invocations;
            encoded["zero_dispatches"] = result.zero_dispatches;
            encoded["control_bytes"] = result.control_bytes;
            encoded["observation_bytes"] =
                result.observation_bytes;
            return encoded;
          },
          py::arg("compile_config"), py::arg("args"),
          py::arg("cache"), py::arg("outer_predicate"),
          py::arg("outer_counter"), py::arg("inner_predicate"),
          py::arg("inner_counter"),
          py::arg("outer_condition_dispatch_count"),
          py::arg("inner_condition_dispatch_begin"),
          py::arg("inner_body_dispatch_begin"),
          py::arg("outer_suffix_dispatch_begin"),
          py::arg("outer_max_iterations"),
          py::arg("inner_max_iterations"),
          py::arg("inner_chunk_size"),
          py::arg("outer_status") = nullptr,
          py::arg("inner_status") = nullptr, py::arg("wait_for_result") = true)
      .def(
          "jit_run_bounded_vulkan_nested_sequence_cached",
          [jit_run_graph, make_nested_graph_controls,
           encode_nested_graph_result](
              aot::CompiledGraph *self,
              const CompileConfig &compile_config, const py::dict &pyargs,
              aot::CompiledGraphJITCache &cache,
              Ndarray &outer_predicate, Ndarray &outer_counter,
              const py::sequence &inner_predicates,
              const py::sequence &inner_counters,
              std::size_t outer_condition_dispatch_count,
              const py::sequence &inner_boundaries,
              int outer_max_iterations,
              const py::sequence &inner_max_iterations,
              const py::sequence &inner_chunk_sizes,
              Ndarray *outer_status, const py::sequence &inner_statuses,
              bool wait_for_result) {
            aot::CompiledGraphNestedStructuredResult result;
            VulkanNestedGraphRequest request;
            request.outer_predicate = &outer_predicate;
            request.outer_counter = &outer_counter;
            request.outer_status = outer_status;
            request.inner_controls = make_nested_graph_controls(
                inner_predicates, inner_counters, inner_boundaries,
                inner_max_iterations, inner_statuses, inner_chunk_sizes);
            request.outer_condition_dispatch_count =
                outer_condition_dispatch_count;
            request.outer_max_iterations = outer_max_iterations;
            request.wait_for_result = wait_for_result;
            request.result = wait_for_result ? &result : nullptr;
            jit_run_graph(self, compile_config, pyargs, &cache, nullptr, 0,
                          true, nullptr, nullptr, -1, -1, nullptr, nullptr,
                          nullptr, 0, true, 0, nullptr, true, nullptr, nullptr,
                          &request);
            return encode_nested_graph_result(result, wait_for_result);
          },
          py::arg("compile_config"), py::arg("args"), py::arg("cache"),
          py::arg("outer_predicate"), py::arg("outer_counter"),
          py::arg("inner_predicates"), py::arg("inner_counters"),
          py::arg("outer_condition_dispatch_count"),
          py::arg("inner_boundaries"), py::arg("outer_max_iterations"),
          py::arg("inner_max_iterations"), py::arg("inner_chunk_sizes"),
          py::arg("outer_status"), py::arg("inner_statuses"),
          py::arg("wait_for_result") = true)
      .def(
          "jit_submit_bounded_cuda_nested_cached",
          [jit_run_graph](
              aot::CompiledGraph *self, const CompileConfig &compile_config,
              const py::dict &pyargs, aot::CompiledGraphJITCache &cache,
              Ndarray &outer_predicate, Ndarray &outer_counter,
              Ndarray &inner_predicate, Ndarray &inner_counter,
              std::size_t outer_condition_dispatch_count,
              std::size_t inner_condition_dispatch_begin,
              std::size_t inner_body_dispatch_begin,
              std::size_t outer_suffix_dispatch_begin, int outer_max_iterations,
              int inner_max_iterations, Ndarray *outer_status,
              Ndarray *inner_status, bool allow_device_update) {
            CudaNestedGraphRequest request;
            request.outer_predicate = &outer_predicate;
            request.outer_counter = &outer_counter;
            request.outer_status = outer_status;
            request.outer_condition_dispatch_count =
                outer_condition_dispatch_count;
            request.outer_max_iterations = outer_max_iterations;
            request.inner_controls.push_back(
                {&inner_predicate, &inner_counter, inner_status,
                 inner_condition_dispatch_begin, inner_body_dispatch_begin,
                 outer_suffix_dispatch_begin, inner_max_iterations, 1});
            request.allow_device_update = allow_device_update;
            return jit_run_graph(self, compile_config, pyargs, &cache, nullptr,
                                 0, true, nullptr, nullptr, -1, -1, nullptr,
                                 nullptr, nullptr, 0, true, 0, nullptr, true,
                                 nullptr, nullptr, nullptr, false, &request);
          },
          py::arg("compile_config"), py::arg("args"), py::arg("cache"),
          py::arg("outer_predicate"), py::arg("outer_counter"),
          py::arg("inner_predicate"), py::arg("inner_counter"),
          py::arg("outer_condition_dispatch_count"),
          py::arg("inner_condition_dispatch_begin"),
          py::arg("inner_body_dispatch_begin"),
          py::arg("outer_suffix_dispatch_begin"),
          py::arg("outer_max_iterations"), py::arg("inner_max_iterations"),
          py::arg("outer_status") = nullptr, py::arg("inner_status") = nullptr,
          py::arg("allow_device_update") = true)
      .def(
          "jit_submit_bounded_cuda_nested_sequence_cached",
          [jit_run_graph, make_nested_graph_controls](
              aot::CompiledGraph *self,
              const CompileConfig &compile_config, const py::dict &pyargs,
              aot::CompiledGraphJITCache &cache,
              Ndarray &outer_predicate, Ndarray &outer_counter,
              const py::sequence &inner_predicates,
              const py::sequence &inner_counters,
              std::size_t outer_condition_dispatch_count,
              const py::sequence &inner_boundaries,
              int outer_max_iterations,
              const py::sequence &inner_max_iterations,
              Ndarray *outer_status, const py::sequence &inner_statuses,
              bool allow_device_update) {
            CudaNestedGraphRequest request;
            request.outer_predicate = &outer_predicate;
            request.outer_counter = &outer_counter;
            request.outer_status = outer_status;
            request.inner_controls = make_nested_graph_controls(
                inner_predicates, inner_counters, inner_boundaries,
                inner_max_iterations, inner_statuses, py::none());
            request.outer_condition_dispatch_count =
                outer_condition_dispatch_count;
            request.outer_max_iterations = outer_max_iterations;
            request.allow_device_update = allow_device_update;
            return jit_run_graph(
                self, compile_config, pyargs, &cache, nullptr, 0, true,
                nullptr, nullptr, -1, -1, nullptr, nullptr, nullptr, 0, true,
                0, nullptr, true, nullptr, nullptr, nullptr, false, &request);
          },
          py::arg("compile_config"), py::arg("args"), py::arg("cache"),
          py::arg("outer_predicate"), py::arg("outer_counter"),
          py::arg("inner_predicates"), py::arg("inner_counters"),
          py::arg("outer_condition_dispatch_count"),
          py::arg("inner_boundaries"), py::arg("outer_max_iterations"),
          py::arg("inner_max_iterations"), py::arg("outer_status"),
          py::arg("inner_statuses"),
          py::arg("allow_device_update") = true)
      .def("jit_run_conditional_cuda_cached",
           [jit_run_graph](aot::CompiledGraph *self,
                           const CompileConfig &compile_config,
                           const py::dict &pyargs,
                           aot::CompiledGraphJITCache &cache,
                           Ndarray &selector,
                           const std::vector<int> &branch_dispatch_counts,
                           int conditional_type, int default_branch) {
             return jit_run_graph(
                 self, compile_config, pyargs, &cache, nullptr, 0, true,
                 &selector, &branch_dispatch_counts, conditional_type,
                 default_branch);
           },
           py::arg("compile_config"), py::arg("args"), py::arg("cache"),
           py::arg("selector"), py::arg("branch_dispatch_counts"),
           py::arg("conditional_type"), py::arg("default_branch") = -1)
      .def("jit_run_conditional_cuda_masked_cached",
           [jit_run_graph](aot::CompiledGraph *self,
                           const CompileConfig &compile_config,
                           const py::dict &pyargs,
                           aot::CompiledGraphJITCache &cache,
                           Ndarray &selector,
                           const std::vector<int> &branch_dispatch_counts,
                           int conditional_type, int default_branch) {
             return jit_run_graph(
                 self, compile_config, pyargs, &cache, nullptr, 0, true,
                 &selector, &branch_dispatch_counts, conditional_type,
                 default_branch, nullptr, nullptr, nullptr, 0, true, 0,
                 nullptr, true, nullptr, nullptr, nullptr,
                 /*cuda_masked_control=*/true);
           },
           py::arg("compile_config"), py::arg("args"), py::arg("cache"),
           py::arg("selector"), py::arg("branch_dispatch_counts"),
           py::arg("conditional_type"), py::arg("default_branch") = -1);

  py::class_<Kernel>(m, "Kernel")
      .def("no_activate",
           [](Kernel *self, SNode *snode) {
             // TODO(#2193): Also apply to @ti.func?
             self->no_activate.push_back(snode);
           })
      .def("insert_scalar_param", &Kernel::insert_scalar_param)
      .def("insert_arr_param", &Kernel::insert_arr_param)
      .def("insert_ndarray_param", &Kernel::insert_ndarray_param)
      .def("insert_texture_param", &Kernel::insert_texture_param)
      .def("insert_pointer_param", &Kernel::insert_pointer_param)
      .def("insert_rw_texture_param", &Kernel::insert_rw_texture_param)
      .def("insert_acceleration_structure_param",
           &Kernel::insert_acceleration_structure_param)
      .def("insert_argpack_param_and_push",
           &Kernel::insert_argpack_param_and_push)
      .def("pop_argpack_stack", &Kernel::pop_argpack_stack)
      .def("insert_ret", &Kernel::insert_ret)
      .def("finalize_rets", &Kernel::finalize_rets)
      .def("finalize_params", &Kernel::finalize_params)
      .def("make_launch_context", &Kernel::make_launch_context,
           py::arg("cpu_bounded_range") = false)
      .def("definition_retired", &Kernel::definition_retired)
      .def("set_compile_tier_override",
           &Kernel::set_compile_tier_override)
      .def("set_task_launch_policy", &Kernel::set_task_launch_policy)
      .def("set_kernel_optimization_spec",
           &Kernel::set_kernel_optimization_spec)
      .def("set_offload_execution_plan",
           &Kernel::set_offload_execution_plan)
      .def("clear_compile_tier_override",
           &Kernel::clear_compile_tier_override)
      .def("get_compile_tier_override",
           [](Kernel *self) -> py::object {
             const auto &v = self->get_compile_tier_override();
             if (v.has_value()) {
               return py::cast(*v);
             }
             return py::none();
           })
      .def(
          "ast_builder",
          [](Kernel *self) -> ASTBuilder * {
            return &self->context->builder();
          },
          py::return_value_policy::reference);

  py::class_<LaunchContextBuilder>(m, "KernelLaunchContext")
      .def("set_arg_int", &LaunchContextBuilder::set_arg_int)
      .def("set_arg_uint", &LaunchContextBuilder::set_arg_uint)
      .def("set_arg_float", &LaunchContextBuilder::set_arg_float)
      .def("set_struct_arg_int", &LaunchContextBuilder::set_struct_arg<int64>)
      .def("set_struct_arg_uint", &LaunchContextBuilder::set_struct_arg<uint64>)
      .def("set_struct_arg_float",
           &LaunchContextBuilder::set_struct_arg<double>)
      .def("set_arg_external_array_with_shape",
           &LaunchContextBuilder::set_arg_external_array_with_shape)
      .def("set_arg_argpack", &LaunchContextBuilder::set_arg_argpack)
      .def("_debug_set_argpack_resource_handle",
           [](LaunchContextBuilder &ctx, const std::vector<int> &arg_id,
              std::uint64_t domain, std::uint32_t kind, std::uint32_t index,
              std::uint32_t generation) {
             ctx.debug_set_argpack_resource_handle(
                 arg_id, RuntimeResourceHandle{domain, kind, index,
                                               generation});
           })
      .def("set_arg_ndarray", &LaunchContextBuilder::set_arg_ndarray)
      .def("set_arg_ndarray_with_grad",
           &LaunchContextBuilder::set_arg_ndarray_with_grad)
      .def("set_arg_dense_storage",
           &LaunchContextBuilder::set_arg_dense_storage)
      .def("set_arg_runtime_storage",
           &LaunchContextBuilder::set_arg_runtime_storage)
      .def("_debug_set_ndarray_resource_handle",
           [](LaunchContextBuilder &ctx, const std::vector<int> &arg_id,
              std::uint64_t domain, std::uint32_t kind, std::uint32_t index,
              std::uint32_t generation) {
             ctx.debug_set_ndarray_resource_handle(
                 arg_id, RuntimeResourceHandle{domain, kind, index,
                                               generation});
           })
      .def("set_arg_texture", &LaunchContextBuilder::set_arg_texture)
      .def("set_arg_acceleration_structure",
           &LaunchContextBuilder::set_arg_acceleration_structure)
      .def("_debug_set_texture_resource_handle",
           [](LaunchContextBuilder &ctx, const std::vector<int> &arg_id,
              std::uint64_t domain, std::uint32_t kind, std::uint32_t index,
              std::uint32_t generation) {
             ctx.debug_set_texture_resource_handle(
                 arg_id, RuntimeResourceHandle{domain, kind, index,
                                               generation});
           })
      .def("set_arg_rw_texture", &LaunchContextBuilder::set_arg_rw_texture)
      .def("_set_cuda_grid_residency_waves",
           &LaunchContextBuilder::set_cuda_grid_residency_waves)
      .def("_set_cuda_range_work_per_thread_target",
           &LaunchContextBuilder::set_cuda_range_work_per_thread_target)
      .def("get_struct_ret_int", &LaunchContextBuilder::get_struct_ret_int)
      .def("get_struct_ret_uint", &LaunchContextBuilder::get_struct_ret_uint)
      .def("get_struct_ret_float", &LaunchContextBuilder::get_struct_ret_float);

  py::class_<Function>(m, "Function")
      .def("insert_scalar_param", &Function::insert_scalar_param)
      .def("insert_arr_param", &Function::insert_arr_param)
      .def("insert_ndarray_param", &Function::insert_ndarray_param)
      .def("insert_texture_param", &Function::insert_texture_param)
      .def("insert_pointer_param", &Function::insert_pointer_param)
      .def("insert_rw_texture_param", &Function::insert_rw_texture_param)
      .def("insert_acceleration_structure_param",
           &Function::insert_acceleration_structure_param)
      .def("insert_ret", &Function::insert_ret)
      .def("set_function_body",
           py::overload_cast<const std::function<void()> &>(
               &Function::set_function_body))
      .def("finalize_rets", &Function::finalize_rets)
      .def("finalize_params", &Function::finalize_params)
      .def(
          "ast_builder",
          [](Function *self) -> ASTBuilder * {
            return &self->context->builder();
          },
          py::return_value_policy::reference);

  py::class_<Expr> expr(m, "Expr");
  expr.def("snode", &Expr::snode, py::return_value_policy::reference)
      .def("is_external_tensor_expr",
           [](Expr *expr) { return expr->is<ExternalTensorExpression>(); })
      .def("is_index_expr",
           [](Expr *expr) { return expr->is<IndexExpression>(); })
      .def("is_primal",
           [](Expr *expr) {
             return expr->cast<FieldExpression>()->snode_grad_type ==
                    SNodeGradType::kPrimal;
           })
      .def("is_lvalue", [](Expr *expr) { return expr->expr->is_lvalue(); })
      .def("set_dbg_info", &Expr::set_dbg_info)
      .def("get_dbg_info", [](Expr *expr) { return expr->expr->dbg_info; })
      .def("set_name",
           [&](Expr *expr, std::string na) {
             expr->cast<FieldExpression>()->name = na;
           })
      .def("set_grad_type",
           [&](Expr *expr, SNodeGradType t) {
             expr->cast<FieldExpression>()->snode_grad_type = t;
           })
      .def("set_adjoint", &Expr::set_adjoint)
      .def("set_adjoint_checkbit", &Expr::set_adjoint_checkbit)
      .def("set_dual", &Expr::set_dual)
      .def("set_dynamic_index_stride",
           [&](Expr *expr, int dynamic_index_stride) {
             auto matrix_field = expr->cast<MatrixFieldExpression>();
             matrix_field->dynamic_indexable = true;
             matrix_field->dynamic_index_stride = dynamic_index_stride;
           })
      .def("get_dynamic_indexable",
           [&](Expr *expr) -> bool {
             return expr->cast<MatrixFieldExpression>()->dynamic_indexable;
           })
      .def("get_dynamic_index_stride",
           [&](Expr *expr) -> int {
             return expr->cast<MatrixFieldExpression>()->dynamic_index_stride;
           })
      .def(
          "get_dt",
          [&](Expr *expr) -> const Type * {
            return expr->cast<FieldExpression>()->dt;
          },
          py::return_value_policy::reference)
      .def("get_ret_type", &Expr::get_ret_type)
      .def("get_rvalue_type",
           [](Expr *expr) { return expr->get_rvalue_type(); })
      .def("is_tensor",
           [](Expr *expr) { return expr->get_rvalue_type()->is<TensorType>(); })
      .def("is_struct",
           [](Expr *expr) { return expr->get_rvalue_type()->is<StructType>(); })
      .def("get_shape",
           [](Expr *expr) -> std::optional<std::vector<int>> {
             auto tensor_type = expr->get_rvalue_type()->cast<TensorType>();
             if (tensor_type) {
               return std::optional<std::vector<int>>(tensor_type->get_shape());
             }
             return std::nullopt;
           })
      .def("type_check", &Expr::type_check)
      .def("get_expr_name",
           [](Expr *expr) { return expr->cast<FieldExpression>()->name; })
      .def("get_raw_address", [](Expr *expr) { return (uint64)expr; })
      .def("get_underlying_ptr_address", [](Expr *e) {
        // The reason that there are both get_raw_address() and
        // get_underlying_ptr_address() is that Expr itself is mostly wrapper
        // around its underlying |expr| (of type Expression). Expr |e| can be
        // temporary, while the underlying |expr| is mostly persistent.
        //
        // Same get_raw_address() implies that get_underlying_ptr_address() are
        // also the same. The reverse is not true.
        return (uint64)e->expr.get();
      });

  py::class_<ExprGroup>(m, "ExprGroup")
      .def(py::init<>())
      .def("size", [](ExprGroup *eg) { return eg->exprs.size(); })
      .def("push_back", &ExprGroup::push_back);

  py::class_<Stmt>(m, "Stmt");  // NOLINT(bugprone-unused-raii)

  m.def("insert_internal_func_call", [&](Operation *op, const ExprGroup &args) {
    return Expr::make<InternalFuncCallExpression>(op, args.exprs);
  });

  m.def("insert_materialized_internal_func_call",
        [&](Operation *op, const ExprGroup &args) {
          return Expr::make<InternalFuncCallExpression>(op, args.exprs, true);
        });

  m.def("make_get_element_expr",
        Expr::make<GetElementExpression, const Expr &, std::vector<int>,
                   const DebugInfo &>);

  m.def("value_cast", static_cast<Expr (*)(const Expr &expr, DataType)>(cast));
  m.def("bits_cast",
        static_cast<Expr (*)(const Expr &expr, DataType)>(bit_cast));

  m.def("expr_atomic_add", [&](const Expr &a, const Expr &b) {
    return Expr::make<AtomicOpExpression>(AtomicOpType::add, a, b);
  });

  m.def("expr_atomic_sub", [&](const Expr &a, const Expr &b) {
    return Expr::make<AtomicOpExpression>(AtomicOpType::sub, a, b);
  });

  m.def("expr_atomic_min", [&](const Expr &a, const Expr &b) {
    return Expr::make<AtomicOpExpression>(AtomicOpType::min, a, b);
  });

  m.def("expr_atomic_max", [&](const Expr &a, const Expr &b) {
    return Expr::make<AtomicOpExpression>(AtomicOpType::max, a, b);
  });

  m.def("expr_atomic_bit_and", [&](const Expr &a, const Expr &b) {
    return Expr::make<AtomicOpExpression>(AtomicOpType::bit_and, a, b);
  });

  m.def("expr_atomic_bit_or", [&](const Expr &a, const Expr &b) {
    return Expr::make<AtomicOpExpression>(AtomicOpType::bit_or, a, b);
  });

  m.def("expr_atomic_bit_xor", [&](const Expr &a, const Expr &b) {
    return Expr::make<AtomicOpExpression>(AtomicOpType::bit_xor, a, b);
  });

  m.def("expr_atomic_mul", [&](const Expr &a, const Expr &b) {
    return Expr::make<AtomicOpExpression>(AtomicOpType::mul, a, b);
  });

  m.def("expr_assume_in_range", assume_range);

  m.def("expr_loop_unique", loop_unique);

  m.def("expr_field", expr_field);

  m.def("expr_matrix_field", expr_matrix_field);

#define DEFINE_EXPRESSION_OP(x) m.def("expr_" #x, expr_##x);

  DEFINE_EXPRESSION_OP(neg)
  DEFINE_EXPRESSION_OP(sqrt)
  DEFINE_EXPRESSION_OP(round)
  DEFINE_EXPRESSION_OP(floor)
  DEFINE_EXPRESSION_OP(frexp)
  DEFINE_EXPRESSION_OP(ceil)
  DEFINE_EXPRESSION_OP(abs)
  DEFINE_EXPRESSION_OP(sin)
  DEFINE_EXPRESSION_OP(asin)
  DEFINE_EXPRESSION_OP(cos)
  DEFINE_EXPRESSION_OP(acos)
  DEFINE_EXPRESSION_OP(tan)
  DEFINE_EXPRESSION_OP(tanh)
  DEFINE_EXPRESSION_OP(inv)
  DEFINE_EXPRESSION_OP(rcp)
  DEFINE_EXPRESSION_OP(rsqrt)
  DEFINE_EXPRESSION_OP(exp)
  DEFINE_EXPRESSION_OP(log)
  DEFINE_EXPRESSION_OP(popcnt)
  DEFINE_EXPRESSION_OP(clz)

  DEFINE_EXPRESSION_OP(select)
  DEFINE_EXPRESSION_OP(ifte)

  DEFINE_EXPRESSION_OP(cmp_le)
  DEFINE_EXPRESSION_OP(cmp_lt)
  DEFINE_EXPRESSION_OP(cmp_ge)
  DEFINE_EXPRESSION_OP(cmp_gt)
  DEFINE_EXPRESSION_OP(cmp_ne)
  DEFINE_EXPRESSION_OP(cmp_eq)

  DEFINE_EXPRESSION_OP(bit_and)
  DEFINE_EXPRESSION_OP(bit_or)
  DEFINE_EXPRESSION_OP(bit_xor)
  DEFINE_EXPRESSION_OP(bit_shl)
  DEFINE_EXPRESSION_OP(bit_shr)
  DEFINE_EXPRESSION_OP(bit_sar)
  DEFINE_EXPRESSION_OP(bit_not)

  DEFINE_EXPRESSION_OP(logic_not)
  DEFINE_EXPRESSION_OP(logical_and)
  DEFINE_EXPRESSION_OP(logical_or)

  DEFINE_EXPRESSION_OP(add)
  DEFINE_EXPRESSION_OP(sub)
  DEFINE_EXPRESSION_OP(mul)
  DEFINE_EXPRESSION_OP(div)
  DEFINE_EXPRESSION_OP(truediv)
  DEFINE_EXPRESSION_OP(floordiv)
  DEFINE_EXPRESSION_OP(mod)
  DEFINE_EXPRESSION_OP(max)
  DEFINE_EXPRESSION_OP(min)
  DEFINE_EXPRESSION_OP(atan2)
  DEFINE_EXPRESSION_OP(pow)

#undef DEFINE_EXPRESSION_OP

  m.def("make_global_load_stmt", Stmt::make<GlobalLoadStmt, Stmt *>);
  m.def("make_global_store_stmt", Stmt::make<GlobalStoreStmt, Stmt *, Stmt *>);
  m.def("make_frontend_assign_stmt",
        Stmt::make<FrontendAssignStmt, const Expr &, const Expr &,
                   const DebugInfo &>);

  m.def("make_arg_load_expr",
        Expr::make<ArgLoadExpression, const std::vector<int> &,
                   const DataType &, bool, bool, int, const DebugInfo &>,
        "arg_id"_a, "dt"_a, "is_ptr"_a = false, "create_load"_a = true,
        "arg_depth"_a = 0, "dbg_info"_a = DebugInfo());

  m.def("make_reference",
        Expr::make<ReferenceExpression, const Expr &, const DebugInfo &>);

  m.def(
      "make_external_tensor_expr",
      [](const DataType &dt, int ndim, const std::vector<int> &arg_id,
         bool needs_grad, int arg_depth, const BoundaryMode &boundary,
         bool runtime_affine) {
        return Expr::make<ExternalTensorExpression>(
            dt, ndim, arg_id, needs_grad, arg_depth, boundary,
            /*byte_offset=*/0, /*byte_stride=*/0, runtime_affine);
      },
      "dt"_a, "ndim"_a, "arg_id"_a, "needs_grad"_a, "arg_depth"_a,
      "boundary"_a, "runtime_affine"_a = true);
  m.def("make_external_tensor_strided_expr",
        Expr::make<ExternalTensorExpression, const DataType &, int,
                   const std::vector<int> &, bool, int, const BoundaryMode &,
                   std::size_t, std::size_t>);
  m.def("make_external_tensor_member_expr",
        [](const Expr &expr, const DataType &dt, std::size_t byte_offset,
           std::size_t byte_stride) {
          TI_ASSERT(expr.is<ExternalTensorExpression>());
          auto external_tensor_expr = expr.cast<ExternalTensorExpression>();
          return Expr::make<ExternalTensorExpression>(
              dt, external_tensor_expr->ndim, external_tensor_expr->arg_id,
              external_tensor_expr->needs_grad,
              external_tensor_expr->arg_depth, external_tensor_expr->boundary,
              byte_offset, byte_stride,
              external_tensor_expr->runtime_affine);
        });

  m.def("make_external_tensor_grad_expr",
        Expr::make<ExternalTensorExpression, Expr *>);

  m.def("make_rand_expr",
        Expr::make<RandExpression, const DataType &, const DebugInfo &>);

  m.def("make_const_expr_bool",
        Expr::make<ConstExpression, const DataType &, uint1>);

  m.def("make_const_expr_int",
        Expr::make<ConstExpression, const DataType &, int64>);

  m.def("make_const_expr_fp",
        Expr::make<ConstExpression, const DataType &, float64>);

  m.def("make_texture_ptr_expr",
        Expr::make<TexturePtrExpression, const std::vector<int> &, int, int,
                   const DebugInfo &>);
  m.def("make_rw_texture_ptr_expr",
        Expr::make<TexturePtrExpression, const std::vector<int> &, int, int,
                   const BufferFormat &, int, const DebugInfo &>);
  m.def("make_acceleration_structure_ptr_expr",
        Expr::make<AccelerationStructurePtrExpression,
                   const std::vector<int> &, int, const DebugInfo &>);

  auto &&texture =
      py::enum_<TextureOpType>(m, "TextureOpType", py::arithmetic());
  for (int t = 0; t <= (int)TextureOpType::kStore; t++)
    texture.value(texture_op_type_name(TextureOpType(t)).c_str(),
                  TextureOpType(t));
  texture.export_values();

  auto &&bin = py::enum_<BinaryOpType>(m, "BinaryOpType", py::arithmetic());
  for (int t = 0; t <= (int)BinaryOpType::undefined; t++)
    bin.value(binary_op_type_name(BinaryOpType(t)).c_str(), BinaryOpType(t));
  bin.export_values();
  m.def("make_binary_op_expr",
        Expr::make<BinaryOpExpression, const BinaryOpType &, const Expr &,
                   const Expr &>);

  auto &&unary = py::enum_<UnaryOpType>(m, "UnaryOpType", py::arithmetic());
  for (int t = 0; t <= (int)UnaryOpType::undefined; t++)
    unary.value(unary_op_type_name(UnaryOpType(t)).c_str(), UnaryOpType(t));
  unary.export_values();
  m.def("make_unary_op_expr",
        Expr::make<UnaryOpExpression, const UnaryOpType &, const Expr &>);
#define PER_TYPE(x)                                                  \
  m.attr(("DataType_" + data_type_name(PrimitiveType::x)).c_str()) = \
      PrimitiveType::x;
#include "taichi/inc/data_type.inc.h"
#undef PER_TYPE

  m.def("data_type_size", data_type_size);
  m.def("data_type_alignment", data_type_alignment);
  m.def("data_type_element_offset",
        [](DataType dt, const std::vector<int> &indices) {
          return dt.ptr_removed()->as<StructType>()->get_element_offset(indices);
        });
  m.def("is_quant", is_quant);
  m.def("is_integral", is_integral);
  m.def("is_signed", is_signed);
  m.def("is_real", is_real);
  m.def("is_unsigned", is_unsigned);
  m.def("is_tensor", is_tensor);

  m.def("data_type_name", data_type_name);

  m.def(
      "subscript_with_multiple_indices",
      Expr::make<IndexExpression, const Expr &, const std::vector<ExprGroup> &,
                 const std::vector<int> &, const DebugInfo &>);

  m.def("get_external_tensor_element_dim", [](const Expr &expr) {
    TI_ASSERT(expr.is<ExternalTensorExpression>());
    // FIXME: no need to make it negative since we don't support SOA
    auto dtype = expr.cast<ExternalTensorExpression>()->dt;
    return dtype->is<TensorType>()
               ? -dtype->cast<TensorType>()->get_shape().size()
               : 0;
  });

  m.def("get_external_tensor_needs_grad", [](const Expr &expr) {
    TI_ASSERT(expr.is<ExternalTensorExpression>());
    return expr.cast<ExternalTensorExpression>()->needs_grad;
  });

  m.def("get_external_tensor_arg_id", [](const Expr &expr) {
    TI_ASSERT(expr.is<ExternalTensorExpression>());
    return expr.cast<ExternalTensorExpression>()->arg_id;
  });

  m.def("get_external_tensor_element_type", [](const Expr &expr) {
    TI_ASSERT(expr.is<ExternalTensorExpression>());
    auto external_tensor_expr = expr.cast<ExternalTensorExpression>();
    return external_tensor_expr->dt;
  });

  m.def("get_external_tensor_element_shape", [](const Expr &expr) {
    TI_ASSERT(expr.is<ExternalTensorExpression>());
    auto external_tensor_expr = expr.cast<ExternalTensorExpression>();
    return external_tensor_expr->dt.get_shape();
  });

  m.def("get_external_tensor_dim", [](const Expr &expr) {
    if (expr.is<ExternalTensorExpression>()) {
      return expr.cast<ExternalTensorExpression>()->ndim;
    } else if (expr.is<TexturePtrExpression>()) {
      return expr.cast<TexturePtrExpression>()->num_dims;
    } else {
      TI_ASSERT(false);
      return 0;
    }
  });

  m.def("get_external_tensor_shape_along_axis",
        Expr::make<ExternalTensorShapeAlongAxisExpression, const Expr &, int,
                   const DebugInfo &>);

  m.def("get_external_tensor_real_func_args",
        [](const Expr &expr, const DebugInfo &dbg_info = DebugInfo()) {
          TI_ASSERT(expr.is<ExternalTensorExpression>());
          auto external_tensor_expr = expr.cast<ExternalTensorExpression>();

          std::vector<Expr> args;
          for (int i = 0; i < external_tensor_expr->ndim; i++) {
            args.push_back(Expr::make<ExternalTensorShapeAlongAxisExpression>(
                expr, i, expr->dbg_info));
            args.back()->type_check(nullptr);
          }
          for (int i = 0; i < external_tensor_expr->ndim; i++) {
            args.push_back(Expr::make<GetElementExpression>(
                expr,
                std::vector<int>{
                    TypeFactory::SHAPE_POS_IN_NDARRAY,
                    TypeFactory::stride_pos_in_ndarray(
                        external_tensor_expr->ndim, i)},
                dbg_info));
            args.back()->type_check(nullptr);
          }
          args.push_back(Expr::make<GetElementExpression>(
              expr,
              std::vector<int>{
                  TypeFactory::SHAPE_POS_IN_NDARRAY,
                  TypeFactory::affine_mode_pos_in_ndarray(
                      external_tensor_expr->ndim)},
              dbg_info));
          args.back()->type_check(nullptr);

          args.push_back(Expr::make<ExternalTensorBasePtrExpression>(
              expr, /*is_grad=*/false, dbg_info));
          args.back()->type_check(nullptr);

          if (external_tensor_expr->needs_grad) {
            args.push_back(Expr::make<ExternalTensorBasePtrExpression>(
                expr, /*is_grad=*/true, dbg_info));
            args.back()->type_check(nullptr);
          }

          return args;
        });

  // Mesh related.
  m.def("get_relation_size", [](mesh::MeshPtr mesh_ptr, const Expr &mesh_idx,
                                mesh::MeshElementType to_type,
                                const DebugInfo &dbg_info = DebugInfo()) {
    return Expr::make<MeshRelationAccessExpression>(
        mesh_ptr.ptr.get(), mesh_idx, to_type, dbg_info);
  });

  m.def("get_relation_access",
        [](mesh::MeshPtr mesh_ptr, const Expr &mesh_idx,
           mesh::MeshElementType to_type, const Expr &neighbor_idx,
           const DebugInfo &dbg_info = DebugInfo()) {
          return Expr::make<MeshRelationAccessExpression>(
              mesh_ptr.ptr.get(), mesh_idx, to_type, neighbor_idx, dbg_info);
        });

  py::class_<FunctionKey>(m, "FunctionKey")
      .def(py::init<const std::string &, int, int>())
      .def_readonly("instance_id", &FunctionKey::instance_id);

  m.def("test_throw", [] {
    try {
      throw IRModified();
    } catch (IRModified) {
      TI_INFO("caught");
    }
  });

  m.def("test_throw", [] { throw IRModified(); });

#if TI_WITH_LLVM
  m.def("libdevice_path", libdevice_path);
#endif

  m.def("host_arch", host_arch);
  m.def("arch_uses_llvm", arch_uses_llvm);

  m.def("set_lib_dir", [&](const std::string &dir) { compiled_lib_dir = dir; });
  m.def("set_tmp_dir", [&](const std::string &dir) { runtime_tmp_dir = dir; });

  m.def("get_commit_hash", get_commit_hash);
  m.def("get_version_string", get_version_string);
  m.def("get_version_major", get_version_major);
  m.def("get_version_minor", get_version_minor);
  m.def("get_version_patch", get_version_patch);
  m.def("get_runtime_contract_manifest", [] {
    py::dict result;
    result["schema_version"] =
        get_forge_contract_manifest_schema_version();
    result["native_abi_revision"] = get_forge_native_abi_revision();
    result["runtime_statistics_schema"] =
        get_forge_runtime_statistics_schema_version();
    result["source_id"] = get_commit_hash();
    result["shim_source_id"] = TI_COMMIT_HASH;
    result["runtime_compiler_abi"] = get_forge_native_compiler_abi();
    result["shim_compiler_abi"] = forge_compiler_abi_identity();
    const auto bitmap = get_forge_native_feature_bitmap();
    py::dict profile;
    profile["schema_version"] = 1;
    const bool profile_known = bitmap & kForgeBuildProfileKnown;
    const bool toolkit_reference = bitmap & kForgeBuildToolkitReference;
    const bool cupti = bitmap & kForgeBuildCupti;
    profile["kind"] = !profile_known ? "unknown"
                      : (toolkit_reference || cupti)
                          ? "cuda-toolkit-specialized-runtime"
                          : "portable-runtime";
    profile["cuda_toolkit_primitive_reference"] = toolkit_reference;
    profile["cupti"] = cupti;
    result["build_profile"] = std::move(profile);
    py::dict features;
    features["cpu"] = bool(bitmap & kForgeFeatureCpu);
    features["llvm"] = bool(bitmap & kForgeFeatureLlvm);
    features["cuda"] = bool(bitmap & kForgeFeatureCuda);
    features["vulkan"] = bool(bitmap & kForgeFeatureVulkan);
    features["ggui"] = bool(bitmap & kForgeFeatureGgui);
    features["opengl"] = bool(bitmap & kForgeFeatureOpenGl);
    result["features"] = std::move(features);
    result["legacy_runtime"] = false;
    return result;
  });
  m.def("get_llvm_target_support", [] {
#if defined(TI_WITH_LLVM)
    return LLVM_VERSION_STRING;
#else
    return "targets unsupported";
#endif
  });
  m.def("test_printf", [] { printf("test_printf\n"); });
  m.def("test_logging", [] { TI_INFO("test_logging"); });
  m.def("trigger_crash", [] { *(int *)(1) = 0; });
  m.def("get_max_num_indices", [] { return taichi_max_num_indices; });
  m.def("get_max_num_args", [] { return taichi_max_num_args; });
  m.def("test_threading", test_threading);
  m.def("is_extension_supported", is_extension_supported);

  m.def("query_int64", [](const std::string &key) {
    if (key == "cuda_compute_capability") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          CUDAContext::get_instance().get_compute_capability());
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_driver_lock_sampled_acquisitions") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(CUDADriver::get_instance()
                                      .get_telemetry_snapshot()
                                      .lock.sampled_acquisitions);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_driver_lock_contended_acquisitions") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(CUDADriver::get_instance()
                                      .get_telemetry_snapshot()
                                      .lock.contended_acquisitions);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_context_lock_sampled_acquisitions") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          CUDAContext::get_instance()
              .get_lock_telemetry_snapshot()
              .sampled_acquisitions);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_context_lock_contended_acquisitions") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          CUDAContext::get_instance()
              .get_lock_telemetry_snapshot()
              .contended_acquisitions);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_async_allocation_calls") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          CUDADriver::get_instance().get_telemetry_snapshot().async_allocation_calls);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_sync_allocation_fallback_calls") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(CUDADriver::get_instance()
                                      .get_telemetry_snapshot()
                                      .sync_allocation_fallback_calls);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_async_free_calls") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          CUDADriver::get_instance().get_telemetry_snapshot().async_free_calls);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_sync_free_fallback_calls") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(CUDADriver::get_instance()
                                      .get_telemetry_snapshot()
                                      .sync_free_fallback_calls);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_jit_module_load_calls") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(CUDADriver::get_instance()
                                      .get_telemetry_snapshot()
                                      .jit_module_load_calls);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_jit_ptx_bytes") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          CUDADriver::get_instance().get_telemetry_snapshot().jit_ptx_bytes);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_jit_host_wall_ns") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(CUDADriver::get_instance()
                                      .get_telemetry_snapshot()
                                      .jit_host_wall_ns);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_jit_driver_wall_us") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(CUDADriver::get_instance()
                                      .get_telemetry_snapshot()
                                      .jit_driver_wall_us);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_jit_diagnostic_loads") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(CUDADriver::get_instance()
                                      .get_telemetry_snapshot()
                                      .jit_diagnostic_loads);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_jit_info_log_bytes") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(CUDADriver::get_instance()
                                      .get_telemetry_snapshot()
                                      .jit_info_log_bytes);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_jit_error_log_bytes") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(CUDADriver::get_instance()
                                      .get_telemetry_snapshot()
                                      .jit_error_log_bytes);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_artifact_external_requests") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          cuda::get_cuda_artifact_provider_telemetry_snapshot()
              .external_requests);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_artifact_cache_hits") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          cuda::get_cuda_artifact_provider_telemetry_snapshot().cache_hits);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_artifact_cache_misses") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          cuda::get_cuda_artifact_provider_telemetry_snapshot().cache_misses);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_artifact_compile_calls") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          cuda::get_cuda_artifact_provider_telemetry_snapshot().compile_calls);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_artifact_compile_failures") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          cuda::get_cuda_artifact_provider_telemetry_snapshot()
              .compile_failures);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_artifact_compile_wall_ns") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          cuda::get_cuda_artifact_provider_telemetry_snapshot()
              .compile_wall_ns);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_artifact_cubin_loads") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          cuda::get_cuda_artifact_provider_telemetry_snapshot().cubin_loads);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_artifact_cubin_bytes") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          cuda::get_cuda_artifact_provider_telemetry_snapshot().cubin_bytes);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_artifact_cubin_unloads") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          cuda::get_cuda_artifact_provider_telemetry_snapshot().cubin_unloads);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_artifact_cubin_current_bytes") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          cuda::get_cuda_artifact_provider_telemetry_snapshot()
              .cubin_current_bytes);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_artifact_cubin_peak_bytes") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          cuda::get_cuda_artifact_provider_telemetry_snapshot()
              .cubin_peak_bytes);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_artifact_entry_points_loaded") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          cuda::get_cuda_artifact_provider_telemetry_snapshot()
              .entry_points_loaded);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_artifact_multi_entry_artifacts") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          cuda::get_cuda_artifact_provider_telemetry_snapshot()
              .multi_entry_artifacts);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_artifact_advanced_controls_skipped_non_user") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          cuda::get_cuda_artifact_provider_telemetry_snapshot()
              .advanced_controls_skipped_non_user);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_artifact_advanced_controls_fallbacks") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          cuda::get_cuda_artifact_provider_telemetry_snapshot()
              .advanced_controls_fallbacks);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_artifact_driver_ptx_fallbacks") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          cuda::get_cuda_artifact_provider_telemetry_snapshot()
              .driver_ptx_fallbacks);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_compileiq_protocol_requests") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          cuda::get_cuda_compileiq_protocol_telemetry_snapshot().requests);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_compileiq_protocol_cache_hits") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          cuda::get_cuda_compileiq_protocol_telemetry_snapshot().cache_hits);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_compileiq_worker_calls") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          cuda::get_cuda_compileiq_protocol_telemetry_snapshot().worker_calls);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_compileiq_worker_failures") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          cuda::get_cuda_compileiq_protocol_telemetry_snapshot()
              .worker_failures);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_compileiq_worker_wall_ns") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          cuda::get_cuda_compileiq_protocol_telemetry_snapshot()
              .worker_wall_ns);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_compileiq_acf_responses") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          cuda::get_cuda_compileiq_protocol_telemetry_snapshot().acf_responses);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_compileiq_pass_responses") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          cuda::get_cuda_compileiq_protocol_telemetry_snapshot()
              .pass_responses);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_compileiq_fail_open_responses") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          cuda::get_cuda_compileiq_protocol_telemetry_snapshot()
              .fail_open_responses);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_compileiq_nested_requests_rejected") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          cuda::get_cuda_compileiq_protocol_telemetry_snapshot()
              .nested_requests_rejected);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_retained_launch_current_bytes") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          cuda::get_retained_launch_buffer_telemetry_snapshot().current_bytes);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_retained_launch_peak_bytes") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          cuda::get_retained_launch_buffer_telemetry_snapshot().peak_bytes);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_retained_launch_allocation_calls") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          cuda::get_retained_launch_buffer_telemetry_snapshot()
              .allocation_calls);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_retained_launch_release_calls") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          cuda::get_retained_launch_buffer_telemetry_snapshot().release_calls);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_grid_residency_resolution_calls") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          cuda::get_grid_residency_telemetry_snapshot().resolution_calls);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_grid_residency_resolution_failures") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          cuda::get_grid_residency_telemetry_snapshot().resolution_failures);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_grid_residency_last_requested_waves") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(cuda::get_grid_residency_telemetry_snapshot()
                                      .last_requested_waves);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_grid_residency_last_baseline_grid") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(cuda::get_grid_residency_telemetry_snapshot()
                                      .last_baseline_grid);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_grid_residency_last_resolved_grid") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(cuda::get_grid_residency_telemetry_snapshot()
                                      .last_resolved_grid);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (
        key ==
        "cuda_grid_residency_last_active_blocks_per_multiprocessor") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(cuda::get_grid_residency_telemetry_snapshot()
                                      .last_active_blocks_per_multiprocessor);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_grid_residency_last_multiprocessor_count") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(cuda::get_grid_residency_telemetry_snapshot()
                                      .last_multiprocessor_count);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_range_coarsening_resolution_calls") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(cuda::get_grid_residency_telemetry_snapshot()
                                      .coarsening_resolution_calls);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_range_coarsening_resolution_failures") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(cuda::get_grid_residency_telemetry_snapshot()
                                      .coarsening_resolution_failures);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_range_coarsening_last_work_per_thread_target") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(cuda::get_grid_residency_telemetry_snapshot()
                                      .last_work_per_thread_target);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_range_coarsening_last_baseline_grid") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(cuda::get_grid_residency_telemetry_snapshot()
                                      .last_coarsening_baseline_grid);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_range_coarsening_last_resolved_grid") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(cuda::get_grid_residency_telemetry_snapshot()
                                      .last_coarsened_grid);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "vulkan_graph_replay_slot_saturation_fallbacks") {
#if defined(TI_WITH_VULKAN)
      return static_cast<int64_t>(
          gfx::get_graph_replay_slot_saturation_fallbacks());
#else
      TI_NOT_IMPLEMENTED
#endif
    } else {
      TI_ERROR("Key {} not supported in query_int64", key);
    }
  });

  // Type system

  py::class_<Type>(m, "Type").def("to_string", &Type::to_string);

  m.def("promoted_type", promoted_type);

  // Note that it is important to specify py::return_value_policy::reference for
  // the factory methods, otherwise pybind11 will delete the Types owned by
  // TypeFactory on Python-scope pointer destruction.
  py::class_<TypeFactory>(m, "TypeFactory")
      .def("get_quant_int_type", &TypeFactory::get_quant_int_type,
           py::arg("num_bits"), py::arg("is_signed"), py::arg("compute_type"),
           py::return_value_policy::reference)
      .def("get_quant_fixed_type", &TypeFactory::get_quant_fixed_type,
           py::arg("digits_type"), py::arg("compute_type"), py::arg("scale"),
           py::return_value_policy::reference)
      .def("get_quant_float_type", &TypeFactory::get_quant_float_type,
           py::arg("digits_type"), py::arg("exponent_type"),
           py::arg("compute_type"), py::return_value_policy::reference)
      .def(
          "get_tensor_type",
          [&](TypeFactory *factory, std::vector<int> shape,
              const DataType &element_type) {
            return factory->create_tensor_type(shape, element_type);
          },
          py::return_value_policy::reference)
      .def(
          "get_struct_type",
          [&](TypeFactory *factory,
              std::vector<std::pair<DataType, std::string>> elements) {
            std::vector<AbstractDictionaryMember> members;
            for (auto &[type, name] : elements) {
              members.push_back({type, name});
            }
            return DataType(factory->get_struct_type(members));
          },
          py::return_value_policy::reference)
      .def("get_rwtexture_struct_type", &TypeFactory::get_rwtexture_struct_type,
           py::return_value_policy::reference)
      .def("get_ndarray_struct_type", &TypeFactory::get_ndarray_struct_type,
           py::arg("dt"), py::arg("ndim"), py::arg("needs_grad"),
           py::return_value_policy::reference)
      .def("get_struct_type_for_argpack_ptr",
           &TypeFactory::get_struct_type_for_argpack_ptr, py::arg("dt"),
           py::arg("layout") = "none", py::return_value_policy::reference)
      .def(
          "get_argpack_type",
          [&](TypeFactory *factory,
              std::vector<std::pair<DataType, std::string>> elements) {
            std::vector<AbstractDictionaryMember> members;
            size_t pos = 0;
            for (auto &[type, name] : elements) {
              members.push_back({type, name, ++pos});
            }
            return DataType(factory->get_argpack_type(members));
          },
          py::return_value_policy::reference);

  m.def("get_type_factory_instance", TypeFactory::get_instance,
        py::return_value_policy::reference);

  // NOLINTNEXTLINE(bugprone-unused-raii)
  py::class_<BitStructType>(m, "BitStructType");
  py::class_<BitStructTypeBuilder>(m, "BitStructTypeBuilder")
      .def(py::init<int>())
      .def("begin_placing_shared_exponent",
           &BitStructTypeBuilder::begin_placing_shared_exponent)
      .def("end_placing_shared_exponent",
           &BitStructTypeBuilder::end_placing_shared_exponent)
      .def("add_member", &BitStructTypeBuilder::add_member)
      .def("build", &BitStructTypeBuilder::build,
           py::return_value_policy::reference);

  py::class_<SNodeRegistry>(m, "SNodeRegistry")
      .def(py::init<>())
      .def("create_root", &SNodeRegistry::create_root,
           py::return_value_policy::reference);

  m.def(
      "finalize_snode_tree",
      [](SNodeRegistry *registry, const SNode *root, Program *program,
         bool compile_only) -> SNodeTree * {
        return program->add_snode_tree(registry->finalize(root), compile_only);
      },
      py::return_value_policy::reference);

  // Sparse Matrix
  py::class_<SparseMatrixBuilder>(m, "SparseMatrixBuilder")
      .def(py::init<int, int, int, DataType, const std::string &>(),
           py::arg("rows"), py::arg("cols"), py::arg("max_num_triplets"),
           py::arg("dt") = PrimitiveType::f32,
           py::arg("storage_format") = "col_major")
      .def("print_triplets_eigen", &SparseMatrixBuilder::print_triplets_eigen)
      .def("print_triplets_cuda", &SparseMatrixBuilder::print_triplets_cuda)
      .def("create_ndarray",
           [&](SparseMatrixBuilder *builder, Program *prog) {
             return builder->create_ndarray(prog);
           })
      .def("delete_ndarray",
           [&](SparseMatrixBuilder *builder, Program *prog) {
             return builder->delete_ndarray(prog);
           })
      .def("get_ndarray_data_ptr", &SparseMatrixBuilder::get_ndarray_data_ptr)
      .def("get_ndarray", &SparseMatrixBuilder::get_ndarray,
           py::return_value_policy::reference)
      .def("build", &SparseMatrixBuilder::build)
      .def("build_cuda", &SparseMatrixBuilder::build_cuda)
      .def("build_vulkan", &SparseMatrixBuilder::build_vulkan)
      .def("get_addr", [](SparseMatrixBuilder *mat) { return uint64(mat); });

  py::class_<SparsePattern, std::shared_ptr<SparsePattern>>(m,
                                                            "SparsePattern");
  auto sparse_pattern_runtime_stats =
      [](const SparsePattern &pattern) {
        const auto stats = pattern.debug_runtime_statistics();
        py::dict identity;
        identity["backend_family"] = stats.backend_family;
        identity["storage_format"] = stats.storage_format;
        identity["index_dtype"] = stats.index_dtype;
        identity["value_order"] = stats.value_order;
        identity["topology_fingerprint"] = stats.topology_fingerprint;
        identity["rows"] = stats.rows;
        identity["cols"] = stats.cols;
        identity["nnz"] = stats.nnz;
        identity["block_rows"] = stats.block_rows;
        identity["block_cols"] = stats.block_cols;
        identity["block_size"] = stats.block_size;
        identity["block_nnz"] = stats.block_nnz;
        identity["pattern_id"] = stats.pattern_id;
        identity["pattern_version"] = stats.pattern_version;

        py::dict lifecycle;
        lifecycle["immutable"] = stats.immutable;
        lifecycle["pattern_builds"] = stats.pattern_builds;
        lifecycle["operator_references"] = stats.operator_references;
        lifecycle["program_bound"] = true;

        py::dict resources;
        resources["pattern_reserved_bytes"] =
            stats.pattern_reserved_bytes;
        resources["ownership_scope"] =
            "shared_immutable_pattern_storage";
        resources["sum_once_across_operators"] = true;

        py::dict transfers;
        transfers["host_to_device_bytes"] = stats.host_to_device_bytes;
        transfers["device_to_host_bytes"] = stats.device_to_host_bytes;
        transfers["device_to_device_bytes"] =
            stats.device_to_device_bytes;
        transfers["scope"] = "pattern_creation_only";

        py::dict result;
        result["schema_version"] = 1;
        result["identity"] = std::move(identity);
        result["lifecycle"] = std::move(lifecycle);
        result["resources"] = std::move(resources);
        result["transfers"] = std::move(transfers);
        return result;
      };
  py::class_<SparseCsrPattern, SparsePattern,
             std::shared_ptr<SparseCsrPattern>>(m, "SparseCsrPattern")
      .def("_debug_runtime_stats", sparse_pattern_runtime_stats);
  py::class_<SparseBsrPattern, SparsePattern,
             std::shared_ptr<SparseBsrPattern>>(m, "SparseBsrPattern")
      .def("_debug_runtime_stats", sparse_pattern_runtime_stats);

  py::class_<CuSparseMatrix::SpmmPlanLease>(m, "_CudaSpmmPlanLease");
  py::class_<SparseMatrix>(m, "SparseMatrix")
      .def(py::init<>())
      .def(py::init<int, int, DataType>(), py::arg("rows"), py::arg("cols"),
           py::arg("dt") = PrimitiveType::f32)
      .def(py::init<SparseMatrix &>())
      .def("to_string", &SparseMatrix::to_string)
      .def("get_element", &SparseMatrix::get_element<float32>)
      .def("set_element", &SparseMatrix::set_element<float32>)
      .def("mmwrite", &SparseMatrix::mmwrite)
      .def("num_rows", &SparseMatrix::num_rows)
      .def("num_cols", &SparseMatrix::num_cols)
      .def("num_nonzero", &SparseMatrix::num_nonzero)
      .def("update_values", &SparseMatrix::update_values)
      .def(
          "_cuda_cusparse_retain_spmm_plan",
          [](SparseMatrix &matrix, int rhs_count, int algorithm,
             bool release_cache) {
            auto *csr = dynamic_cast<CuSparseMatrix *>(&matrix);
            TI_ERROR_IF(csr == nullptr,
                        "CUDA SpMM ownership requires a scalar CSR matrix.");
            return csr->retain_spmm_plan(rhs_count, algorithm, release_cache);
          },
          py::arg("rhs_count"), py::arg("algorithm"),
          py::arg("release_cache") = false, py::keep_alive<0, 1>())
      .def(
          "_cuda_cusparse_prepare_spmm",
          [](SparseMatrix &matrix, Program *program, const Ndarray &input,
             const Ndarray &output, int rhs_count, int algorithm) {
            auto *csr = dynamic_cast<CuSparseMatrix *>(&matrix);
            TI_ERROR_IF(csr == nullptr,
                        "CUDA cuSPARSE SpMM preparation requires a scalar CSR "
                        "matrix.");
            csr->nd_spmm(program, input, output, rhs_count, algorithm, true);
          },
          py::arg("program"), py::arg("input"), py::arg("output"),
          py::arg("rhs_count"), py::arg("algorithm"))
      .def(
          "_cuda_cusparse_spmm_plan_info",
          [](const SparseMatrix &matrix, int rhs_count, int algorithm) {
            const auto *csr = dynamic_cast<const CuSparseMatrix *>(&matrix);
            TI_ERROR_IF(csr == nullptr,
                        "CUDA cuSPARSE SpMM plan info requires a scalar CSR "
                        "matrix.");
            const auto info = csr->spmm_plan_info(rhs_count, algorithm);
            py::dict result;
            result["prepared"] = info.prepared;
            result["preprocess_attempted"] = info.preprocess_attempted;
            result["preprocessed"] = info.preprocessed;
            result["preprocess_error"] = info.preprocess_error;
            result["workspace_bytes"] = info.workspace_bytes;
            return result;
          },
          py::arg("rhs_count"), py::arg("algorithm"))
      .def(
          "_cuda_cusparse_spmm_f32",
          [](SparseMatrix &matrix, Program *program, const Ndarray &input,
             const Ndarray &output, int rhs_count, int algorithm) {
            auto *csr = dynamic_cast<CuSparseMatrix *>(&matrix);
            TI_ERROR_IF(csr == nullptr,
                        "CUDA cuSPARSE SpMM supports scalar CSR matrices "
                        "only.");
            csr->nd_spmm(program, input, output, rhs_count, algorithm);
          },
          py::arg("program"), py::arg("input"), py::arg("output"),
          py::arg("rhs_count"), py::arg("algorithm"))
      .def(
          "_cuda_cusparse_spsv_f32",
          [](SparseMatrix &matrix, Program *program, const Ndarray &input,
             const Ndarray &output, int fill_mode, bool unit_diagonal,
             bool transpose) {
            auto *csr = dynamic_cast<CuSparseMatrix *>(&matrix);
            TI_ERROR_IF(csr == nullptr,
                        "CUDA cuSPARSE SpSV supports scalar CSR matrices "
                        "only.");
            csr->nd_spsv(program, input, output, fill_mode, unit_diagonal,
                         transpose);
          },
          py::arg("program"), py::arg("input"), py::arg("output"),
          py::arg("fill_mode"), py::arg("unit_diagonal"),
          py::arg("transpose"))
      .def(
          "_cuda_cusparse_spsm_f32",
          [](SparseMatrix &matrix, Program *program, const Ndarray &input,
             const Ndarray &output, int rhs_count, int fill_mode,
             bool unit_diagonal, bool transpose) {
            auto *csr = dynamic_cast<CuSparseMatrix *>(&matrix);
            TI_ERROR_IF(csr == nullptr,
                        "CUDA cuSPARSE SpSM supports scalar CSR matrices "
                        "only.");
            csr->nd_spsm(program, input, output, rhs_count, fill_mode,
                         unit_diagonal, transpose);
          },
          py::arg("program"), py::arg("input"), py::arg("output"),
          py::arg("rhs_count"), py::arg("fill_mode"),
          py::arg("unit_diagonal"), py::arg("transpose"))
      .def("_debug_runtime_stats", [](const SparseMatrix &matrix) {
        const auto stats = matrix.debug_runtime_statistics();
        py::dict identity;
        identity["backend_family"] = stats.backend_family;
        identity["storage_format"] = stats.storage_format;
        identity["dtype"] = stats.dtype;
        identity["rows"] = stats.rows;
        identity["cols"] = stats.cols;
        identity["nnz"] = stats.nnz;
        if (stats.block_size > 0) {
          identity["block_rows"] = stats.block_rows;
          identity["block_cols"] = stats.block_cols;
          identity["block_size"] = stats.block_size;
          identity["block_nnz"] = stats.block_nnz;
        } else {
          identity["block_rows"] = py::none();
          identity["block_cols"] = py::none();
          identity["block_size"] = py::none();
          identity["block_nnz"] = py::none();
        }
        identity["pattern_version"] = stats.pattern_version;
        identity["numeric_version"] = stats.numeric_version;
        if (stats.pattern_storage_shared) {
          identity["pattern_id"] = stats.shared_pattern_id;
        } else {
          identity["pattern_id"] = py::none();
        }

        py::dict operations;
        operations["pattern_builds"] = stats.pattern_builds;
        operations["numeric_updates"] = stats.numeric_updates;
        operations["numeric_update_bytes"] = stats.numeric_update_bytes;
        operations["spmv_calls"] = stats.spmv_calls;
        operations["spmv_plan_builds"] = stats.spmv_plan_builds;
        operations["spmv_plan_reuses"] = stats.spmv_plan_reuses;
        operations["spmv_handle_creations"] =
            stats.spmv_handle_creations;
        operations["dense_vector_descriptor_creations"] =
            stats.dense_vector_descriptor_creations;
        operations["dense_vector_descriptor_rebinds"] =
            stats.dense_vector_descriptor_rebinds;
        operations["spmv_workspace_allocations"] =
            stats.spmv_workspace_allocations;
        operations["spmv_preprocess_builds"] =
            stats.spmv_preprocess_builds;
        operations["spmv_preprocess_reuses"] =
            stats.spmv_preprocess_reuses;
        operations["spmv_preprocess_fallbacks"] =
            stats.spmv_preprocess_fallbacks;
        operations["spmm_calls"] = stats.spmm_calls;
        operations["spmm_plan_builds"] = stats.spmm_plan_builds;
        operations["spmm_plan_reuses"] = stats.spmm_plan_reuses;
        operations["spmm_dense_descriptor_rebinds"] =
            stats.spmm_dense_descriptor_rebinds;
        operations["spmm_workspace_allocations"] =
            stats.spmm_workspace_allocations;
        operations["spmm_preprocess_builds"] =
            stats.spmm_preprocess_builds;
        operations["spmm_preprocess_reuses"] =
            stats.spmm_preprocess_reuses;
        operations["spmm_preprocess_fallbacks"] =
            stats.spmm_preprocess_fallbacks;
        operations["spsv_calls"] = stats.spsv_calls;
        operations["spsv_plan_builds"] = stats.spsv_plan_builds;
        operations["spsv_plan_reuses"] = stats.spsv_plan_reuses;
        operations["spsv_analysis_builds"] = stats.spsv_analysis_builds;
        operations["spsv_dense_descriptor_rebinds"] =
            stats.spsv_dense_descriptor_rebinds;
        operations["spsv_value_updates"] = stats.spsv_value_updates;
        operations["spsm_calls"] = stats.spsm_calls;
        operations["spsm_plan_builds"] = stats.spsm_plan_builds;
        operations["spsm_plan_reuses"] = stats.spsm_plan_reuses;
        operations["spsm_analysis_builds"] = stats.spsm_analysis_builds;
        operations["spsm_dense_descriptor_rebinds"] =
            stats.spsm_dense_descriptor_rebinds;
        operations["spsm_value_updates"] = stats.spsm_value_updates;
        operations["resource_generations_published"] =
            stats.resource_generations_published;
        operations["resource_generations_retired"] =
            stats.resource_generations_retired;
        operations["resource_generations_released"] =
            stats.resource_generations_released;

        py::dict resources;
        resources["pattern_reserved_bytes"] =
            stats.pattern_reserved_bytes;
        resources["values_reserved_bytes"] = stats.values_reserved_bytes;
        resources["spmv_workspace_reserved_bytes"] =
            stats.spmv_workspace_reserved_bytes;
        resources["spmm_workspace_reserved_bytes"] =
            stats.spmm_workspace_reserved_bytes;
        resources["spsv_workspace_reserved_bytes"] =
            stats.spsv_workspace_reserved_bytes;
        resources["spsm_workspace_reserved_bytes"] =
            stats.spsm_workspace_reserved_bytes;
        resources["operator_owned_reserved_bytes"] =
            stats.operator_owned_reserved_bytes;
        resources["numeric_update_peak_temporary_bytes"] =
            stats.numeric_update_peak_temporary_bytes;
        resources["resource_generation_active_leases"] =
            stats.resource_generation_active_leases;
        resources["resource_generation_current"] =
            stats.resource_generation_current;
        resources["operator_exclusive_reserved_bytes"] =
            stats.pattern_storage_shared
                ? stats.operator_exclusive_reserved_bytes
                : stats.operator_owned_reserved_bytes;
        resources["pattern_storage_shared"] =
            stats.pattern_storage_shared;
        if (stats.pattern_storage_shared) {
          resources["shared_pattern_operator_references"] =
              stats.shared_pattern_operator_references;
        } else {
          resources["shared_pattern_operator_references"] = py::none();
        }
        resources["sum_operator_owned_bytes_across_operators_safe"] =
            !stats.pattern_storage_shared;
        resources["matrix_descriptor_count"] =
            stats.matrix_descriptor_count;
        resources["dense_vector_descriptor_count"] =
            stats.dense_vector_descriptor_count;
        resources["spmv_handle_count"] = stats.spmv_handle_count;
        resources["spmm_plan_count"] = stats.spmm_plan_count;
        resources["spmm_dense_matrix_descriptor_count"] =
            stats.spmm_dense_matrix_descriptor_count;
        resources["spsv_plan_count"] = stats.spsv_plan_count;
        resources["spsm_plan_count"] = stats.spsm_plan_count;
        resources["triangular_dense_descriptor_count"] =
            stats.triangular_dense_descriptor_count;
        resources["opaque_provider_resource_bytes"] = py::none();
        resources["ownership_scope"] =
            stats.pattern_storage_shared
                ? "matrix_values_plan_and_shared_pattern_reference"
                : "matrix_pattern_values_and_persistent_spmv_plan";
        resources["excluded"] =
            "builder_staging_transients_input_output_vectors_and_solver_"
            "workspace";

        py::dict transfers;
        transfers["host_to_device_bytes"] = stats.host_to_device_bytes;
        transfers["device_to_host_bytes"] = stats.device_to_host_bytes;
        transfers["device_to_device_bytes"] =
            stats.device_to_device_bytes;
        transfers["scope"] = "direct_backend_copies_attributed_to_operator";

        py::dict provider;
        provider["name"] = stats.provider_name;
        if (stats.provider_version_major >= 0 &&
            stats.provider_version_minor >= 0 &&
            stats.provider_version_patch >= 0) {
          py::dict version;
          version["major"] = stats.provider_version_major;
          version["minor"] = stats.provider_version_minor;
          version["patch"] = stats.provider_version_patch;
          provider["library_version"] = std::move(version);
        } else {
          provider["library_version"] = py::none();
        }
        provider["bsr_descriptor_available"] =
            stats.provider_bsr_descriptor_available;
        provider["generic_bsr_spmv_available"] =
            stats.provider_generic_bsr_spmv_available;
        provider["spmv_preprocess_available"] =
            stats.provider_spmv_preprocess_available;
        provider["spmm_f32_available"] =
            stats.provider_spmm_f32_available;
        provider["spmm_preprocess_available"] =
            stats.provider_spmm_preprocess_available;
        provider["spsv_f32_available"] =
            stats.provider_spsv_f32_available;
        provider["spsm_f32_available"] =
            stats.provider_spsm_f32_available;
        provider["spsv_value_update_available"] =
            stats.provider_spsv_value_update_available;
        provider["spsm_value_update_available"] =
            stats.provider_spsm_value_update_available;
        provider["triangular_value_update_available"] =
            stats.provider_triangular_value_update_available;
        provider["spmv_preprocess_active"] =
            stats.spmv_preprocess_active;
        provider["spmv_preprocess_last_error"] =
            stats.spmv_preprocess_last_error;
        provider["selected_storage_format"] = stats.storage_format;
        provider["capability_scope"] =
            "loaded_library_symbols_and_version_not_performance";

        py::dict result;
        result["schema_version"] = 1;
        result["identity"] = std::move(identity);
        result["operations"] = std::move(operations);
        result["resources"] = std::move(resources);
        result["transfers"] = std::move(transfers);
        result["provider"] = std::move(provider);
        return result;
      })
      .def("get_data_type", &SparseMatrix::get_data_type);

#define MAKE_SPARSE_MATRIX(TYPE, STORAGE, VTYPE)                             \
  using STORAGE##TYPE##EigenMatrix =                                         \
      Eigen::SparseMatrix<float##TYPE, Eigen::STORAGE>;                      \
  py::class_<EigenSparseMatrix<STORAGE##TYPE##EigenMatrix>, SparseMatrix>(   \
      m, #VTYPE #STORAGE "_EigenSparseMatrix")                               \
      .def(py::init<int, int, DataType>())                                   \
      .def(py::init<EigenSparseMatrix<STORAGE##TYPE##EigenMatrix> &>())      \
      .def(py::init<const STORAGE##TYPE##EigenMatrix &>())                   \
      .def(py::self += py::self)                                             \
      .def(py::self + py::self)                                              \
      .def(py::self -= py::self)                                             \
      .def(py::self - py::self)                                              \
      .def(py::self *= float##TYPE())                                        \
      .def(py::self *float##TYPE())                                          \
      .def(float##TYPE() * py::self)                                         \
      .def(py::self *py::self)                                               \
      .def("matmul", &EigenSparseMatrix<STORAGE##TYPE##EigenMatrix>::matmul) \
      .def("spmv", &EigenSparseMatrix<STORAGE##TYPE##EigenMatrix>::spmv)     \
      .def("transpose",                                                      \
           &EigenSparseMatrix<STORAGE##TYPE##EigenMatrix>::transpose)        \
      .def("get_element",                                                    \
           &EigenSparseMatrix<STORAGE##TYPE##EigenMatrix>::get_element<      \
               float##TYPE>)                                                 \
      .def("set_element",                                                    \
           &EigenSparseMatrix<STORAGE##TYPE##EigenMatrix>::set_element<      \
               float##TYPE>)                                                 \
      .def("mat_vec_mul",                                                    \
           &EigenSparseMatrix<STORAGE##TYPE##EigenMatrix>::mat_vec_mul<      \
               Eigen::VectorX##VTYPE>);

  MAKE_SPARSE_MATRIX(32, ColMajor, f);
  MAKE_SPARSE_MATRIX(32, RowMajor, f);
  MAKE_SPARSE_MATRIX(64, ColMajor, d);
  MAKE_SPARSE_MATRIX(64, RowMajor, d);

  py::class_<CpuSparseCsrMatrix, SparseMatrix>(m, "CpuSparseCsrMatrix")
      .def("spmv", &CpuSparseCsrMatrix::nd_spmv);
  py::class_<CpuSparseBsrMatrix, SparseMatrix>(m, "CpuSparseBsrMatrix")
      .def("spmv", &CpuSparseBsrMatrix::nd_spmv);

  py::class_<CompiledKernelLinearOperator, SparseMatrix>(
      m, "CompiledKernelLinearOperator")
      .def("spmv", &CompiledKernelLinearOperator::nd_spmv)
      .def("update_numeric_data",
           &CompiledKernelLinearOperator::update_numeric_data);

  py::class_<CompiledGraphLinearOperator, SparseMatrix>(
      m, "CompiledGraphLinearOperator")
      .def("spmv", &CompiledGraphLinearOperator::nd_spmv)
      .def(
          "update_numeric_data",
          [](CompiledGraphLinearOperator &operator_, Program *program,
             const py::dict &numeric_args,
             std::uint64_t expected_topology_version,
             std::uint64_t expected_numeric_version) {
            CompiledGraphLinearOperator::NdarrayArguments numeric;
            for (const auto &item : numeric_args) {
              numeric.emplace(py::cast<std::string>(item.first),
                              &py::cast<const Ndarray &>(item.second));
            }
            operator_.update_numeric_arguments(
                program, std::move(numeric), expected_topology_version,
                expected_numeric_version);
          });

  py::class_<CuSparseMatrix, SparseMatrix>(m, "CuSparseMatrix")
      .def(py::init<int, int, DataType>())
      .def(py::init<const CuSparseMatrix &>())
      .def("spmv", &CuSparseMatrix::nd_spmv)
      .def("spsv", &CuSparseMatrix::nd_spsv)
      .def("spsm", &CuSparseMatrix::nd_spsm)
      .def("spmv_kernel", &CuSparseMatrix::nd_spmv_kernel)
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * float32())
      .def(float32() * py::self)
      .def("matmul", &CuSparseMatrix::matmul)
      .def("transpose", &CuSparseMatrix::transpose)
      .def("get_element", &CuSparseMatrix::get_element)
      .def("to_string", &CuSparseMatrix::to_string);

  py::class_<CuSparseBsrMatrix, SparseMatrix>(m, "CuSparseBsrMatrix")
      .def("spmv", &CuSparseBsrMatrix::nd_spmv)
      .def("spmv_kernel", &CuSparseBsrMatrix::nd_spmv_kernel);

  py::class_<VulkanSparseMatrix, SparseMatrix>(m, "VulkanSparseMatrix")
      .def("spmv", &VulkanSparseMatrix::nd_spmv);
  py::class_<VulkanSparseBsrMatrix, SparseMatrix>(
      m, "VulkanSparseBsrMatrix")
      .def("spmv", &VulkanSparseBsrMatrix::nd_spmv);

  py::class_<VulkanSparseAssemblyPlan>(m, "VulkanSparseAssemblyPlan")
      .def("build", &VulkanSparseAssemblyPlan::build,
           py::call_guard<py::gil_scoped_release>())
      .def("_debug_runtime_stats",
           [](const VulkanSparseAssemblyPlan &plan) {
             const auto stats = plan.debug_runtime_statistics();
             py::dict identity;
             identity["backend_family"] = "vulkan";
             identity["method"] = "radix_sort_segment_reduce_csr";
             identity["rows"] = stats.rows;
             identity["cols"] = stats.cols;
             identity["triplet_capacity"] = stats.capacity;

             py::dict status;
             status["last_status"] = stats.last_status;
             status["last_input_triplets"] = stats.last_input_triplets;
             status["last_unique_nnz"] = stats.last_unique_nnz;
             status["last_duplicate_triplets"] =
                 stats.last_duplicate_triplets;
             status["codes"] = py::dict(
                 "ok"_a = 0, "index_out_of_range"_a = 1,
                 "nonfinite_input"_a = 2,
                 "nonfinite_duplicate_sum"_a = 3,
                 "invalid_device_state"_a = 4,
                 "active_count_exceeds_capacity"_a = 5,
                 "publish_failed"_a = 6);

             py::dict operations;
             operations["build_calls"] = stats.build_calls;
             operations["successful_builds"] = stats.successful_builds;
             operations["failed_builds"] = stats.failed_builds;
             operations["workspace_builds"] = stats.workspace_builds;
             operations["workspace_reuses"] = stats.workspace_reuses;
             operations["workspace_growth_synchronizations"] =
                 stats.workspace_growth_synchronizations;
             operations["host_synchronizations"] =
                 stats.host_synchronizations;
             operations["host_control_readbacks"] =
                 stats.host_control_readbacks;
             operations["host_scalar_readbacks"] =
                 stats.host_scalar_readbacks;

             py::dict resources;
             resources["persistent_workspace_reserved_bytes"] =
                 stats.persistent_workspace_reserved_bytes;
             resources["shared_radix_sort_workspace_reserved_bytes"] =
                 stats.shared_radix_sort_workspace_reserved_bytes;
             resources["shared_scan_workspace_reserved_bytes"] =
                 stats.shared_scan_workspace_reserved_bytes;
             resources["last_output_pattern_bytes"] =
                 stats.last_output_pattern_bytes;
             resources["last_output_value_bytes"] =
                 stats.last_output_value_bytes;
             resources["workspace_ownership"] =
                 "plan_staging_plus_program_shared_sort_scan";

             py::dict transfers;
             transfers["device_to_host_bytes"] =
                 stats.device_to_host_bytes;
             transfers["device_to_device_bytes"] =
                 stats.device_to_device_bytes;
             transfers["host_to_device_bytes"] = 0;
             transfers["device_payload_readback_bytes"] = 0;

             py::dict contract;
             contract["fixed_capacity"] = true;
             contract["device_resident_triplet_payload"] = true;
             contract["transactional_publish"] = true;
             contract["exact_sized_published_csr"] = true;
             contract["sorted_unique_columns_per_row"] = true;
             contract["duplicate_reduce_order"] =
                 "sorted_segment_sequential";
             contract["control_readback_bytes_per_build"] = 8;
             contract["public_sparse_builder"] = false;
             contract["bsr_output"] = false;

             py::dict result;
             result["schema_version"] = 1;
             result["identity"] = std::move(identity);
             result["status"] = std::move(status);
             result["operations"] = std::move(operations);
             result["resources"] = std::move(resources);
             result["transfers"] = std::move(transfers);
             result["contract"] = std::move(contract);
             return result;
           });
  m.def("_make_vulkan_sparse_assembly_plan",
        [](Program *program, int rows, int cols, int capacity) {
          return std::make_unique<VulkanSparseAssemblyPlan>(
              program, rows, cols, capacity);
        });

  py::class_<CudaSparseAssemblyPlan>(m, "CudaSparseAssemblyPlan")
      .def("build", &CudaSparseAssemblyPlan::build,
           py::call_guard<py::gil_scoped_release>())
      .def("_debug_runtime_stats",
           [](const CudaSparseAssemblyPlan &plan) {
             const auto stats = plan.debug_runtime_statistics();
             py::dict identity;
             identity["backend_family"] = "cuda";
             identity["method"] = "radix_sort_segment_reduce_csr";
             identity["rows"] = stats.rows;
             identity["cols"] = stats.cols;
             identity["triplet_capacity"] = stats.capacity;

             py::dict status;
             status["last_status"] = stats.last_status;
             status["last_input_triplets"] = stats.last_input_triplets;
             status["last_unique_nnz"] = stats.last_unique_nnz;
             status["last_duplicate_triplets"] =
                 stats.last_duplicate_triplets;
             status["codes"] = py::dict(
                 "ok"_a = 0, "index_out_of_range"_a = 1,
                 "nonfinite_input"_a = 2,
                 "nonfinite_duplicate_sum"_a = 3,
                 "invalid_device_state"_a = 4,
                 "active_count_out_of_range"_a = 5,
                 "publish_failed"_a = 6);

             py::dict operations;
             operations["build_calls"] = stats.build_calls;
             operations["successful_builds"] = stats.successful_builds;
             operations["failed_builds"] = stats.failed_builds;
             operations["workspace_builds"] = stats.workspace_builds;
             operations["workspace_reuses"] = stats.workspace_reuses;
             operations["workspace_growth_synchronizations"] =
                 stats.workspace_growth_synchronizations;
             operations["host_synchronizations"] =
                 stats.host_synchronizations;
             operations["host_control_readbacks"] =
                 stats.host_control_readbacks;
             operations["host_scalar_readbacks"] =
                 stats.host_scalar_readbacks;

             py::dict resources;
             resources["persistent_workspace_reserved_bytes"] =
                 stats.persistent_workspace_reserved_bytes;
             resources["shared_radix_sort_workspace_reserved_bytes"] =
                 stats.shared_radix_sort_workspace_reserved_bytes;
             resources["shared_scan_workspace_reserved_bytes"] =
                 stats.shared_scan_workspace_reserved_bytes;
             resources["last_output_pattern_bytes"] =
                 stats.last_output_pattern_bytes;
             resources["last_output_value_bytes"] =
                 stats.last_output_value_bytes;
             resources["workspace_ownership"] =
                 "plan_staging_plus_program_shared_sort_scan";

             py::dict transfers;
             transfers["device_to_host_bytes"] =
                 stats.device_to_host_bytes;
             transfers["device_to_device_bytes"] =
                 stats.device_to_device_bytes;
             transfers["host_to_device_bytes"] = 0;
             transfers["device_payload_readback_bytes"] = 0;

             py::dict contract;
             contract["fixed_capacity"] = true;
             contract["device_active_count"] = true;
             contract["empty_matrix"] = true;
             contract["device_resident_triplet_payload"] = true;
             contract["transactional_publish"] = true;
             contract["exact_sized_published_csr"] = true;
             contract["sorted_unique_columns_per_row"] = true;
             contract["duplicate_reduce_order"] =
                 "stable_sorted_segment_sequential";
             contract["control_readback_bytes_per_build"] = 8;
             contract["public_sparse_builder"] = true;
             contract["bsr_output"] = false;
             contract["cuda_toolkit_required"] = false;

             py::dict result;
             result["schema_version"] = 1;
             result["identity"] = std::move(identity);
             result["status"] = std::move(status);
             result["operations"] = std::move(operations);
             result["resources"] = std::move(resources);
             result["transfers"] = std::move(transfers);
             result["contract"] = std::move(contract);
             return result;
           });
  m.def("_make_cuda_sparse_assembly_plan",
        [](Program *program, int rows, int cols, int capacity) {
          return std::make_unique<CudaSparseAssemblyPlan>(
              program, rows, cols, capacity);
        });

  py::class_<SparseSolver>(m, "SparseSolver")
      .def("compute", &SparseSolver::compute)
      .def("analyze_pattern", &SparseSolver::analyze_pattern)
      .def("factorize", &SparseSolver::factorize)
      .def("validate_factorization", &SparseSolver::validate_factorization)
      .def("info", &SparseSolver::info);

#define REGISTER_EIGEN_SOLVER(dt, type, order, fd)                           \
  py::class_<EigenSparseSolver##dt##type##order, SparseSolver>(              \
      m, "EigenSparseSolver" #dt #type #order)                               \
      .def("compute", &EigenSparseSolver##dt##type##order::compute)          \
      .def("analyze_pattern",                                                \
           &EigenSparseSolver##dt##type##order::analyze_pattern)             \
      .def("factorize", &EigenSparseSolver##dt##type##order::factorize)      \
      .def("solve",                                                          \
           &EigenSparseSolver##dt##type##order::solve<Eigen::VectorX##fd>)   \
      .def("solve_rf",                                                       \
           &EigenSparseSolver##dt##type##order::solve_rf<Eigen::VectorX##fd, \
                                                         dt>)                \
      .def("info", &EigenSparseSolver##dt##type##order::info);

  REGISTER_EIGEN_SOLVER(float32, LLT, AMD, f)
  REGISTER_EIGEN_SOLVER(float32, LLT, COLAMD, f)
  REGISTER_EIGEN_SOLVER(float32, LDLT, AMD, f)
  REGISTER_EIGEN_SOLVER(float32, LDLT, COLAMD, f)
  REGISTER_EIGEN_SOLVER(float32, LU, AMD, f)
  REGISTER_EIGEN_SOLVER(float32, LU, COLAMD, f)
  REGISTER_EIGEN_SOLVER(float64, LLT, AMD, d)
  REGISTER_EIGEN_SOLVER(float64, LLT, COLAMD, d)
  REGISTER_EIGEN_SOLVER(float64, LDLT, AMD, d)
  REGISTER_EIGEN_SOLVER(float64, LDLT, COLAMD, d)
  REGISTER_EIGEN_SOLVER(float64, LU, AMD, d)
  REGISTER_EIGEN_SOLVER(float64, LU, COLAMD, d)

  py::class_<CuSparseSolver, SparseSolver>(m, "CuSparseSolver")
      .def("compute", &CuSparseSolver::compute)
      .def("analyze_pattern", &CuSparseSolver::analyze_pattern)
      .def("factorize", &CuSparseSolver::factorize)
      .def("solve_rf", &CuSparseSolver::solve_rf)
      .def("info", &CuSparseSolver::info);

  m.def("make_sparse_solver", &make_sparse_solver);
  m.def("make_cusparse_solver", &make_cusparse_solver);

  // Conjugate Gradient solver
  auto sparse_solve_result_to_dict = [](const SparseSolveResult &result) {
    const char *termination_reason = "not_run";
    switch (result.status) {
      case SparseSolveStatus::kMaxIterations:
        termination_reason = "max_iterations";
        break;
      case SparseSolveStatus::kBreakdown:
        termination_reason = "breakdown";
        break;
      case SparseSolveStatus::kConverged:
        termination_reason = "converged";
        break;
      case SparseSolveStatus::kNotRun:
        break;
    }
    py::dict snapshot;
    snapshot["status_code"] = static_cast<int>(result.status);
    snapshot["termination_reason"] = termination_reason;
    snapshot["converged"] = result.converged();
    snapshot["breakdown"] = result.breakdown();
    snapshot["reached_max_iterations"] =
        result.reached_max_iterations();
    snapshot["iterations"] = result.iterations;
    snapshot["initial_residual_norm"] = result.initial_residual_norm;
    snapshot["residual_norm"] = result.residual_norm;
    snapshot["absolute_tolerance"] = result.absolute_tolerance;
    snapshot["relative_tolerance"] = result.relative_tolerance;
    snapshot["relative_reference_norm"] =
        result.relative_reference_norm;
    snapshot["effective_tolerance"] = result.effective_tolerance;
    snapshot["breakdown_reason"] =
        result.breakdown()
            ? (result.breakdown_reason == SparseSolveBreakdownReason::none
                   ? "unspecified"
                   : sparse_solve_breakdown_reason_name(
                         result.breakdown_reason))
            : "none";
    return snapshot;
  };
  auto sparse_solve_plan_stats_to_dict =
      [](const SparseSolvePlanRuntimeStatistics &stats) {
        py::dict identity;
        identity["backend_family"] = stats.backend_family;
        identity["method"] = stats.method;
        identity["dtype"] = stats.dtype;
        identity["rows"] = stats.rows;
        identity["cols"] = stats.cols;
        identity["max_iterations"] = stats.max_iterations;
        identity["absolute_tolerance"] = stats.absolute_tolerance;
        identity["relative_tolerance"] = stats.relative_tolerance;
        identity["last_relative_reference_norm"] =
            stats.last_relative_reference_norm;
        identity["last_effective_tolerance"] =
            stats.last_effective_tolerance;
        identity["last_breakdown_reason"] = stats.last_breakdown_reason;
        identity["preconditioner_method"] =
            stats.preconditioner_method;
        identity["operator_action_provider"] =
            stats.operator_action_provider;
        identity["operator_asynchronous_submit"] =
            stats.operator_asynchronous_submit;
        identity["operator_execution_kind"] =
            stats.operator_execution_kind;
        identity["operator_backend_execution_path"] =
            stats.operator_backend_execution_path;
        identity["solver_execution_policy"] =
            stats.solver_execution_policy;
        identity["requested_solver_execution_policy"] =
            stats.requested_solver_execution_policy;
        identity["host_check_interval"] = stats.host_check_interval;
        identity["bounded_chunk_limit"] = stats.bounded_chunk_limit;
        identity["bounded_preferred_chunk_size"] =
            stats.bounded_preferred_chunk_size;
        identity["bounded_control_path"] = stats.bounded_control_path;
        identity["bounded_chunk_schedule"] = stats.bounded_chunk_schedule;
        identity["solver_graph_enabled"] =
            stats.solver_graph_enabled;
        identity["solver_replay_unavailable_reason"] =
            stats.solver_replay_unavailable_reason;
        identity["solver_scalar_location"] =
            stats.solver_scalar_location;
        identity["solver_stream_policy"] =
            stats.solver_stream_policy;
        identity["control_readback_strategy"] =
            stats.control_readback_strategy;
        identity["preconditioning_side"] = stats.preconditioning_side;
        identity["preconditioner_action_provider"] =
            stats.preconditioner_action_provider;
        identity["preconditioner_behavior"] =
            stats.preconditioner_behavior;
        identity["preconditioner_asynchronous_submit"] =
            stats.preconditioner_asynchronous_submit;
        identity["preconditioner_action_count"] =
            stats.preconditioner_action_count;
        identity["preconditioner_action_selection"] =
            stats.preconditioner_action_selection;
        identity["operator_pattern_version"] =
            stats.operator_pattern_version;
        identity["operator_numeric_version"] =
            stats.operator_numeric_version;
        identity["last_solve_pattern_version"] =
            stats.last_solve_pattern_version;
        identity["last_solve_numeric_version"] =
            stats.last_solve_numeric_version;
        identity["operator_pattern_changed_since_last_solve"] =
            stats.operator_pattern_changed_since_last_solve;
        identity["operator_numeric_changed_since_last_solve"] =
            stats.operator_numeric_changed_since_last_solve;

        py::dict operations;
        operations["solve_calls"] = stats.solve_calls;
        operations["total_iterations"] = stats.total_iterations;
        operations["workspace_builds"] = stats.workspace_builds;
        operations["workspace_reuses"] = stats.workspace_reuses;
        operations["operator_apply_calls"] =
            stats.operator_apply_calls_available
                ? py::cast(stats.operator_apply_calls)
                : py::none();
        operations["dot_product_calls"] =
            stats.dot_product_calls_available
                ? py::cast(stats.dot_product_calls)
                : py::none();
        operations["multi_dot_calls"] =
            stats.multi_dot_calls_available
                ? py::cast(stats.multi_dot_calls)
                : py::none();
        operations["vector_update_calls"] =
            stats.vector_update_calls_available
                ? py::cast(stats.vector_update_calls)
                : py::none();
        operations["host_scalar_reductions"] =
            stats.host_scalar_reductions;
        operations["device_scalar_operations"] =
            stats.device_scalar_operations;
        operations["host_scalar_readbacks"] =
            stats.host_scalar_readbacks;
        operations["host_readback_batches"] =
            stats.host_readback_batches;
        operations["host_synchronizations"] =
            stats.host_synchronizations;
        operations["logical_iterations"] =
            stats.logical_iterations;
        operations["executed_iterations"] =
            stats.executed_iterations;
        operations["wasted_iterations"] =
            stats.wasted_iterations;
        operations["solver_chunk_builds"] =
            stats.solver_chunk_builds;
        operations["solver_chunk_reuses"] =
            stats.solver_chunk_reuses;
        operations["solver_chunk_direct_submissions"] =
            stats.solver_chunk_direct_submissions;
        operations["solver_chunk_replays"] =
            stats.solver_chunk_replays;
        operations["solver_chunk_rebinds"] =
            stats.solver_chunk_rebinds;
        operations["solver_chunk_invalidations"] =
            stats.solver_chunk_invalidations;
        operations["solver_chunk_submissions"] = stats.solver_chunk_submissions;
        operations["convergence_observations"] = stats.convergence_observations;
        operations["last_logical_iterations"] = stats.last_logical_iterations;
        operations["last_executed_iterations"] = stats.last_executed_iterations;
        operations["last_convergence_observation_boundaries"] =
            stats.last_convergence_observation_boundaries;
        operations["restart_cycles"] = stats.restart_cycles;
        operations["happy_breakdowns"] = stats.happy_breakdowns;
        operations["restart"] = stats.restart;
        operations["orthogonalization_strategy"] =
            stats.orthogonalization_strategy;
        operations["orthogonalization_passes"] =
            stats.orthogonalization_passes;
        operations["fixed_iteration_only"] =
            stats.fixed_iteration_only;
        operations["bounded_masked_execution"] =
            stats.bounded_masked_execution;
        operations["preconditioner_apply_calls"] =
            stats.preconditioner_apply_calls_available
                ? py::cast(stats.preconditioner_apply_calls)
                : py::none();
        operations["preconditioner_action_selections"] =
            stats.preconditioner_action_selections;
        operations["preconditioner_schedule_wraps"] =
            stats.preconditioner_schedule_wraps;
        operations["operator_generation_pins"] =
            stats.operator_generation_pins;
        operations["operator_generation_changes"] =
            stats.operator_generation_changes;
        operations["operator_numeric_generation_changes"] =
            stats.operator_numeric_generation_changes;
        operations["operator_binding_generation_changes"] =
            stats.operator_binding_generation_changes;
        operations["operator_plan_invalidations"] =
            stats.operator_plan_invalidations;
        operations["operator_execution_plan_builds"] =
            stats.operator_execution_plan_builds;
        operations["operator_execution_plan_reuses"] =
            stats.operator_execution_plan_reuses;
        operations["operator_binding_rebinds"] =
            stats.operator_binding_rebinds;
        operations["operator_sequence_submissions"] =
            stats.operator_sequence_submissions;
        operations["operator_compiled_graph_submissions"] =
            stats.operator_compiled_graph_submissions;
        operations["operator_runtime_capture_submissions"] =
            stats.operator_runtime_capture_submissions;
        operations["operator_backend_captures"] =
            stats.operator_backend_captures;
        operations["operator_backend_replays"] =
            stats.operator_backend_replays;
        operations["operator_ordinary_fallbacks"] =
            stats.operator_ordinary_fallbacks;
        operations["operator_cache_invalidations"] =
            stats.operator_cache_invalidations;
        operations["preconditioner_generation_pins"] =
            stats.preconditioner_generation_pins;
        operations["preconditioner_generation_changes"] =
            stats.preconditioner_generation_changes;
        operations["preconditioner_numeric_generation_changes"] =
            stats.preconditioner_numeric_generation_changes;
        operations["preconditioner_binding_generation_changes"] =
            stats.preconditioner_binding_generation_changes;
        operations["preconditioner_plan_invalidations"] =
            stats.preconditioner_plan_invalidations;
        operations["preconditioner_setup_calls"] =
            stats.preconditioner_setup_calls;
        operations["preconditioner_update_calls"] =
            stats.preconditioner_update_calls;
        operations["preconditioner_update_successes"] =
            stats.preconditioner_update_successes;
        operations["preconditioner_update_noops"] =
            stats.preconditioner_update_noops;
        operations["preconditioner_update_failures"] =
            stats.preconditioner_update_failures;

        py::dict resources;
        resources["persistent_vector_count"] =
            stats.persistent_vector_count;
        resources["persistent_vector_reserved_bytes"] =
            stats.persistent_vector_reserved_bytes;
        resources["persistent_scalar_count"] =
            stats.persistent_scalar_count;
        resources["persistent_scalar_reserved_bytes"] =
            stats.persistent_scalar_reserved_bytes;
        resources["solver_library_workspace_reserved_bytes"] =
            stats.solver_library_workspace_reserved_bytes;
        resources["basis_vector_count"] =
            stats.basis_vector_count;
        resources["basis_reserved_bytes"] =
            stats.basis_reserved_bytes;
        resources["preconditioned_basis_vector_count"] =
            stats.preconditioned_basis_vector_count;
        resources["preconditioned_basis_reserved_bytes"] =
            stats.preconditioned_basis_reserved_bytes;
        resources["solver_replay_executable_count"] =
            stats.solver_replay_executable_count;
        resources["solver_replay_opaque_bytes"] = py::none();
        resources["cublas_handle_count"] = stats.cublas_handle_count;
        resources["cublas_stream_bound"] =
            stats.cublas_stream_bound;
        resources["cublas_device_pointer_mode"] =
            stats.cublas_device_pointer_mode;
        resources["external_preconditioner"] =
            stats.external_preconditioner;
        resources["preconditioner_ownership_scope"] =
            stats.preconditioner_ownership_scope;
        resources["opaque_provider_resource_bytes"] = py::none();
        resources["solver_state_rebuilt_each_solve"] =
            stats.solver_state_rebuilt_each_solve;
        resources["transient_solver_workspace_bytes"] =
            stats.transient_solver_workspace_bytes_available
                ? py::cast(stats.transient_solver_workspace_bytes)
                : py::none();
        resources["ownership_scope"] =
            "solve_plan_vectors_scalars_and_provider_handle";
        resources["excluded"] =
            "operator_resources_rhs_solution_and_caller_vectors";

        py::dict transfers;
        transfers["device_to_device_bytes"] =
            stats.device_to_device_bytes;
        transfers["device_to_host_bytes"] =
            stats.device_to_host_bytes;
        transfers["host_to_device_bytes"] =
            stats.host_to_device_bytes;
        transfers["scope"] = "copies_issued_directly_by_solve_plan";

        py::dict result;
        result["schema_version"] = 1;
        result["identity"] = std::move(identity);
        result["operations"] = std::move(operations);
        result["resources"] = std::move(resources);
        result["transfers"] = std::move(transfers);
        return result;
      };
  auto sparse_preconditioner_stats_to_dict =
      [](const SparsePreconditionerPlanRuntimeStatistics &stats) {
        py::dict identity;
        identity["backend_family"] = stats.backend_family;
        identity["method"] = stats.method;
        identity["dtype"] = stats.dtype;
        identity["rows"] = stats.rows;
        if (stats.block_size > 0) {
          identity["block_rows"] = stats.block_rows;
          identity["block_size"] = stats.block_size;
        } else {
          identity["block_rows"] = py::none();
          identity["block_size"] = py::none();
        }
        identity["operator_pattern_version_at_build"] =
            stats.operator_pattern_version_at_build;
        identity["operator_numeric_version_at_build"] =
            stats.operator_numeric_version_at_build;
        identity["operator_pattern_version_current"] =
            stats.operator_pattern_version_current;
        identity["operator_numeric_version_current"] =
            stats.operator_numeric_version_current;
        identity["operator_stale"] = stats.operator_stale;
        identity["preconditioner_pattern_version_at_build"] =
            stats.preconditioner_pattern_version_at_build;
        identity["preconditioner_numeric_version_at_build"] =
            stats.preconditioner_numeric_version_at_build;
        identity["preconditioner_pattern_version_current"] =
            stats.preconditioner_pattern_version_current;
        identity["preconditioner_numeric_version_current"] =
            stats.preconditioner_numeric_version_current;
        identity["preconditioner_stale"] = stats.preconditioner_stale;

        py::dict operations;
        operations["apply_calls"] = stats.apply_calls;
        operations["numeric_refresh_calls"] =
            stats.numeric_refresh_calls;
        operations["numeric_refresh_successes"] =
            stats.numeric_refresh_successes;
        operations["numeric_refresh_noops"] =
            stats.numeric_refresh_noops;
        operations["numeric_refresh_failures"] =
            stats.numeric_refresh_failures;

        py::dict resources;
        resources["persistent_inverse_count"] =
            stats.persistent_inverse_count;
        resources["persistent_inverse_reserved_bytes"] =
            stats.persistent_inverse_reserved_bytes;
        resources["persistent_refresh_reserved_bytes"] =
            stats.persistent_refresh_reserved_bytes;
        resources["refresh_peak_temporary_host_bytes"] =
            stats.refresh_peak_temporary_host_bytes;
        resources["refresh_peak_temporary_device_bytes"] =
            stats.refresh_peak_temporary_device_bytes;
        const bool external_inverse_operator =
            stats.method == "compiled_kernel_inverse_apply" ||
            stats.method == "compiled_graph_inverse_apply";
        resources["ownership_scope"] =
            external_inverse_operator ? "external_inverse_operator"
                                      : "preconditioner_inverse";
        resources["excluded"] =
            external_inverse_operator
                ? "target_inverse_operators_input_output_and_program_shared_"
                  "cache"
                : "operator_input_output_and_program_shared_cache";

        py::dict transfers;
        transfers["construction_device_to_host_bytes"] =
            stats.construction_device_to_host_bytes;
        transfers["construction_host_to_device_bytes"] =
            stats.construction_host_to_device_bytes;
        transfers["construction_host_synchronizations"] =
            stats.construction_host_synchronizations;
        transfers["refresh_device_to_host_bytes"] =
            stats.refresh_device_to_host_bytes;
        transfers["refresh_full_values_device_to_host_bytes"] =
            stats.refresh_full_values_device_to_host_bytes;
        transfers["refresh_status_device_to_host_bytes"] =
            stats.refresh_status_device_to_host_bytes;
        transfers["refresh_host_to_device_bytes"] =
            stats.refresh_host_to_device_bytes;
        transfers["refresh_device_to_device_bytes"] =
            stats.refresh_device_to_device_bytes;
        transfers["refresh_host_synchronizations"] =
            stats.refresh_host_synchronizations;
        transfers["refresh_device_kernel_launches"] =
            stats.refresh_device_kernel_launches;
        transfers["refresh_device_allocations"] =
            stats.refresh_device_allocations;
        transfers["apply_host_transfer_bytes"] = 0;

        py::dict contract;
        contract["fixed_csr_only"] = stats.method == "jacobi";
        contract["fixed_bsr_only"] = stats.method == "block_jacobi";
        contract["in_place_apply_supported"] =
            stats.in_place_apply_supported;
        contract["numeric_refresh_supported"] =
            stats.numeric_refresh_supported;
        contract["device_native_numeric_refresh"] =
            stats.device_native_numeric_refresh;
        contract["stable_refresh_binding"] =
            stats.stable_refresh_binding;
        contract["numeric_update_requires_refresh"] =
            stats.numeric_refresh_supported;
        contract["numeric_update_requires_rebuild"] =
            !stats.numeric_refresh_supported;
        contract["pattern_update_requires_rebuild"] = true;
        contract["public_solver_integration"] = false;

        py::dict result;
        result["schema_version"] = 3;
        result["identity"] = std::move(identity);
        result["operations"] = std::move(operations);
        result["resources"] = std::move(resources);
        result["transfers"] = std::move(transfers);
        result["contract"] = std::move(contract);
        return result;
      };

  py::class_<LinearOperatorSession>(
      m, "LinearOperatorSession")
      .def("_submit", &LinearOperatorSession::submit,
           py::keep_alive<1, 2>(), py::arg("program"), py::arg("input"),
           py::arg("output"))
      .def("_wait", &LinearOperatorSession::wait)
      .def("_mark_synchronized",
           &LinearOperatorSession::mark_synchronized);

  py::class_<ExperimentalPreconditionerSession>(
      m, "ExperimentalPreconditionerSession")
      .def("_apply", &ExperimentalPreconditionerSession::apply,
           py::keep_alive<1, 2>(), py::arg("program"), py::arg("input"),
           py::arg("output"))
      .def("_metadata", [](const ExperimentalPreconditionerSession &session) {
        auto stamp_to_dict = [](const OperatorResourceStamp &stamp) {
          py::dict result;
          result["program_generation"] = stamp.program_generation;
          result["schema_revision"] = stamp.schema_revision;
          result["topology_revision"] = stamp.topology_revision;
          result["numeric_revision"] = stamp.numeric_revision;
          result["binding_revision"] = stamp.binding_revision;
          return result;
        };
        py::dict result;
        result["schema_version"] = 1;
        result["target_stamp"] =
            stamp_to_dict(session.target_stamp());
        result["action_stamp"] =
            stamp_to_dict(session.action_stamp());
        return result;
      });

  py::class_<ExperimentalPreconditionerPlanHandle>(
      m, "ExperimentalPreconditionerPlanHandle")
      .def("_setup", &ExperimentalPreconditionerPlanHandle::setup,
           py::keep_alive<1, 2>(), py::arg("program"))
      .def("_validate_update",
           &ExperimentalPreconditionerPlanHandle::validate_update,
           py::keep_alive<1, 2>(), py::arg("program"),
           py::arg("accept_reuse"))
      .def("_update", &ExperimentalPreconditionerPlanHandle::update,
           py::keep_alive<1, 2>(), py::arg("program"),
           py::arg("accept_reuse"))
      .def("_pin", &ExperimentalPreconditionerPlanHandle::pin,
           py::keep_alive<0, 1>(), py::keep_alive<1, 2>(),
           py::arg("program"))
      .def("_metadata", [](const ExperimentalPreconditionerPlanHandle &plan) {
        auto stamp_to_dict = [](const OperatorResourceStamp &stamp) {
          py::dict result;
          result["program_generation"] = stamp.program_generation;
          result["schema_revision"] = stamp.schema_revision;
          result["topology_revision"] = stamp.topology_revision;
          result["numeric_revision"] = stamp.numeric_revision;
          result["binding_revision"] = stamp.binding_revision;
          return result;
        };
        py::dict result;
        result["schema_version"] = 1;
        result["method"] = plan.method();
        result["behavior"] = "fixed_linear";
        result["supported"] = true;
        result["is_setup"] = plan.is_setup();
        result["built_from_operator_stamp"] =
            stamp_to_dict(plan.built_from_operator_stamp());
        result["accepted_target_stamp"] =
            stamp_to_dict(plan.accepted_target_stamp());
        result["accepted_action_stamp"] =
            stamp_to_dict(plan.accepted_action_stamp());
        return result;
      })
      .def("_debug_runtime_stats",
           [](const ExperimentalPreconditionerPlanHandle &plan) {
             const auto stats = plan.debug_runtime_statistics();
             py::dict result;
             result["schema_version"] = 1;
             result["setup_calls"] = stats.setup_calls;
             result["update_calls"] = stats.update_calls;
             result["update_successes"] = stats.update_successes;
             result["update_noops"] = stats.update_noops;
             result["update_failures"] = stats.update_failures;
             result["target_generation_changes"] =
                 stats.target_generation_changes;
             result["action_generation_changes"] =
                 stats.action_generation_changes;
             result["rebuild_attestations"] =
                 stats.rebuild_attestations;
             result["reuse_attestations"] = stats.reuse_attestations;
             result["pins"] = stats.pins;
             result["apply_calls"] = stats.apply_calls;
             result["stale_rejections"] = stats.stale_rejections;
             result["approved_generations_published"] =
                 stats.approved_generations_published;
             result["approved_generations_retired"] =
                 stats.approved_generations_retired;
             result["approved_generations_released"] =
                 stats.approved_generations_released;
             result["approved_generation_active_leases"] =
                 stats.approved_generation_active_leases;
             result["has_current_approved_generation"] =
                 stats.has_current_approved_generation;
             return result;
           });

  py::class_<LinearOperatorRecordableKernel,
             std::shared_ptr<LinearOperatorRecordableKernel>>(
      m, "LinearOperatorRecordableKernel")
      .def_property_readonly(
          "_kernel", &LinearOperatorRecordableKernel::kernel,
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "_topology", &LinearOperatorRecordableKernel::topology,
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "_numeric", &LinearOperatorRecordableKernel::numeric,
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "_graph_dispatches",
          [](const LinearOperatorRecordableKernel &record) {
            py::list result;
            const auto *graph = record.graph();
            if (!graph) {
              return result;
            }
            for (const auto &dispatch : graph->dispatches) {
              result.append(py::make_tuple(
                  py::cast(dispatch.ti_kernel,
                           py::return_value_policy::reference),
                  dispatch.symbolic_args));
            }
            return result;
          })
      .def_property_readonly(
          "_fixed_i32",
          [](const LinearOperatorRecordableKernel &record) {
            py::dict result;
            for (const auto &[name, value] : record.fixed_i32()) {
              result[py::str(name)] = value;
            }
            return result;
          })
      .def_property_readonly(
          "_fixed_ndarrays",
          [](const LinearOperatorRecordableKernel &record) {
            py::dict result;
            for (const auto &[name, value] : record.fixed_ndarrays()) {
              result[py::str(name)] =
                  py::cast(value, py::return_value_policy::reference);
            }
            return result;
          })
      .def_property_readonly(
          "_state_dependencies",
          [](const LinearOperatorRecordableKernel &record) {
            py::list result;
            for (const auto &dependency : record.state_dependencies()) {
              result.append(py::make_tuple(
                  dependency.tree_id, dependency.generation,
                  dependency.layout_fingerprint));
            }
            return result;
          })
      .def_property_readonly(
          "_recordable_stats",
          [](const LinearOperatorRecordableKernel &record) {
            py::dict result;
            std::unordered_set<const Ndarray *> unique_fixed;
            if (record.topology()) {
              unique_fixed.insert(record.topology());
            }
            if (record.numeric()) {
              unique_fixed.insert(record.numeric());
            }
            for (const auto &[name, value] : record.fixed_ndarrays()) {
              (void)name;
              if (value) {
                unique_fixed.insert(value);
              }
            }
            std::uint64_t fixed_snapshot_bytes = 0;
            for (const auto *value : unique_fixed) {
              fixed_snapshot_bytes +=
                  value->get_nelement() * value->get_element_size();
            }
            result["dispatch_count"] =
                record.graph() ? record.graph()->dispatches.size() : 1;
            result["fixed_ndarray_count"] =
                record.fixed_ndarrays().size() +
                (record.topology() ? 1 : 0) +
                (record.numeric() ? 1 : 0);
            result["provider_fixed_snapshot_reserved_bytes"] =
                fixed_snapshot_bytes;
            result["outer_graph_resource_snapshot_copies"] = 0;
            result["outer_graph_resource_snapshot_reserved_bytes"] = 0;
            result["state_tree_count"] =
                record.state_dependencies().size();
            result["state_snapshot_copies"] = 0;
            result["state_snapshot_reserved_bytes"] = 0;
            return result;
          })
      .def_property_readonly(
          "active_size", &LinearOperatorRecordableKernel::active_size)
      .def("_resource_stamp", [](const LinearOperatorRecordableKernel &record) {
        const auto stamp = record.resource_stamp();
        return py::make_tuple(
            stamp.program_identity, stamp.program_generation,
            stamp.schema_revision, stamp.topology_revision,
            stamp.numeric_revision, stamp.binding_revision);
      });

  py::class_<LinearOperatorAffineParameterSnapshot,
             std::shared_ptr<LinearOperatorAffineParameterSnapshot>>(
      m, "LinearOperatorAffineParameterSnapshot")
      .def_property_readonly(
          "alpha", &LinearOperatorAffineParameterSnapshot::alpha)
      .def_property_readonly(
          "beta", &LinearOperatorAffineParameterSnapshot::beta)
      .def_property_readonly(
          "version", &LinearOperatorAffineParameterSnapshot::version);

  py::class_<LinearOperatorHandle>(
      m, "LinearOperatorHandle")
      .def("_apply", &LinearOperatorHandle::apply,
           py::keep_alive<1, 2>(), py::arg("program"),
           py::arg("input"), py::arg("output"))
      .def("_apply_dense_storage",
           &LinearOperatorHandle::apply_dense_storage,
           py::keep_alive<1, 2>(), py::arg("program"), py::arg("input"),
           py::arg("output"))
      .def(
          "_apply_generalized",
          [](LinearOperatorHandle &handle, Program *program,
             const Ndarray &input, const py::object &addend,
             const Ndarray &output, double alpha, double beta) {
            const Ndarray *native_addend = nullptr;
            if (beta != 0.0) {
              TI_ERROR_IF(addend.is_none(),
                          "LinearOperator generalized apply with nonzero "
                          "beta requires an addend.");
              native_addend = &py::cast<const Ndarray &>(addend);
            }
            handle.apply_generalized(program, input, native_addend, output,
                                     alpha, beta);
          },
          py::keep_alive<1, 2>(), py::arg("program"), py::arg("input"),
          py::arg("addend"), py::arg("output"), py::arg("alpha"),
          py::arg("beta"))
      .def(
          "_update_numeric",
          [](LinearOperatorHandle &handle, Program *program,
             const py::dict &arguments,
             std::uint64_t expected_topology_version,
             std::uint64_t expected_numeric_version) {
            LinearOperatorHandle::NumericUpdateArguments native;
            for (const auto &item : arguments) {
              native.emplace(py::cast<std::string>(item.first),
                             &py::cast<const Ndarray &>(item.second));
            }
            handle.update_numeric(program, native, expected_topology_version,
                                  expected_numeric_version);
          },
          py::keep_alive<1, 2>(), py::arg("program"), py::arg("arguments"),
          py::arg("expected_topology_version"),
          py::arg("expected_numeric_version"))
      .def("_supports_numeric_update",
           &LinearOperatorHandle::supports_numeric_update)
      .def("_update_affine_parameters",
           &LinearOperatorHandle::update_affine_parameters,
           py::arg("alpha"), py::arg("beta"),
           py::arg("expected_version"), py::arg("next_version"))
      .def("_supports_affine_parameter_update",
           &LinearOperatorHandle::supports_affine_parameter_update)
      .def("_affine_parameter_snapshot",
           &LinearOperatorHandle::affine_parameter_snapshot)
      .def("_supports_recordable_kernel",
           &LinearOperatorHandle::supports_recordable_kernel)
      .def(
          "_recordable_kernel",
          [](LinearOperatorHandle &handle, bool adjoint) {
            return handle.recordable_kernel(
                adjoint ? OperatorApplyMode::adjoint
                        : OperatorApplyMode::forward);
          },
          py::arg("adjoint") = false)
      .def("_resource_stamp", [](const LinearOperatorHandle &handle) {
        const auto stamp = handle.resource_stamp();
        return py::make_tuple(
            stamp.program_identity, stamp.program_generation,
            stamp.schema_revision, stamp.topology_revision,
            stamp.numeric_revision, stamp.binding_revision);
      })
      .def("_begin_session",
           &LinearOperatorHandle::begin_session,
           py::keep_alive<0, 1>())
      .def("_metadata", [](const LinearOperatorHandle &handle) {
        const auto &descriptor = handle.descriptor();
        const auto &capabilities = handle.capabilities();
        const auto &traits = handle.mathematical_traits();
        const auto stamp = handle.resource_stamp();
        const auto claim_to_dict = [](const OperatorTraitClaim &claim) {
          py::dict result;
          result["known"] = claim.known();
          result["value"] = claim.known()
                                  ? py::cast(claim.value)
                                  : py::none();
          const char *provenance = "unspecified";
          switch (claim.provenance) {
            case OperatorTraitProvenance::asserted_by_user:
              provenance = "asserted_by_user";
              break;
            case OperatorTraitProvenance::derived_structurally:
              provenance = "derived_structurally";
              break;
            case OperatorTraitProvenance::constructed_by_framework:
              provenance = "constructed_by_framework";
              break;
            case OperatorTraitProvenance::empirically_checked:
              provenance = "empirically_checked";
              break;
            case OperatorTraitProvenance::unspecified:
              break;
          }
          result["provenance"] = provenance;
          result["validity_scope"] = claim.validity_scope;
          return result;
        };
        py::dict capabilities_dict;
        capabilities_dict["forward_apply"] = capabilities.forward_apply;
        capabilities_dict["adjoint_apply"] = capabilities.adjoint_apply;
        capabilities_dict["native_generalized_apply"] =
            capabilities.native_generalized_apply;
        capabilities_dict["asynchronous_submit"] =
            capabilities.asynchronous_submit;
        capabilities_dict["explicit_sequence"] =
            capabilities.explicit_sequence;
        capabilities_dict["compiled_graph"] = capabilities.compiled_graph;
        capabilities_dict["runtime_capture"] = capabilities.runtime_capture;
        capabilities_dict["binding_rebind"] = capabilities.binding_rebind;
        capabilities_dict["persistent_workspace"] =
            capabilities.persistent_workspace;
        capabilities_dict["dense_storage_operands"] =
            capabilities.dense_storage_operands;
        capabilities_dict["dense_storage_affine_operands"] =
            capabilities.dense_storage_affine_operands;
        py::dict traits_dict;
        traits_dict["self_adjoint"] =
            claim_to_dict(traits.self_adjoint);
        traits_dict["positive_definite"] =
            claim_to_dict(traits.positive_definite);
        traits_dict["positive_semidefinite"] =
            claim_to_dict(traits.positive_semidefinite);
        traits_dict["singular"] = claim_to_dict(traits.singular);
        py::dict resource_stamp;
        resource_stamp["program_generation"] = stamp.program_generation;
        resource_stamp["schema_revision"] = stamp.schema_revision;
        resource_stamp["topology_revision"] = stamp.topology_revision;
        resource_stamp["numeric_revision"] = stamp.numeric_revision;
        resource_stamp["binding_revision"] = stamp.binding_revision;
        py::dict result;
        result["schema_version"] = 1;
        result["provider"] = handle.provider_name();
        result["dtype"] = data_type_name(descriptor.domain.scalar_type);
        result["shape"] = py::make_tuple(
            descriptor.range.scalar_extent,
            descriptor.domain.scalar_extent);
        result["entry_shape"] = descriptor.domain.entry_shape;
        result["execution_kind"] =
            operator_execution_kind_name(handle.execution_kind());
        result["capabilities"] = std::move(capabilities_dict);
        result["traits"] = std::move(traits_dict);
        result["resource_stamp"] = std::move(resource_stamp);
        return result;
      })
      .def("_debug_runtime_stats",
           [](const LinearOperatorHandle &handle) {
             const auto stats = handle.debug_runtime_statistics();
             py::dict result;
             result["schema_version"] = 1;
             result["provider"] = handle.provider_name();
             result["execution_kind"] =
                 operator_execution_kind_name(stats.execution_kind);
             result["submissions"] = stats.submissions;
             result["primitive_apply_calls"] =
                 stats.primitive_apply_calls;
             result["generalized_lowerings"] =
                 stats.generalized_lowerings;
             result["scratch_builds"] = stats.scratch_builds;
             result["scratch_reuses"] = stats.scratch_reuses;
             result["scratch_reserved_bytes"] =
                 stats.scratch_reserved_bytes;
             result["generation_pins"] = stats.generation_pins;
             result["generation_changes"] = stats.generation_changes;
             result["invalidations"] = stats.invalidations;
             result["execution_plan_builds"] =
                 stats.execution_plan_builds;
             result["execution_plan_reuses"] =
                 stats.execution_plan_reuses;
             result["binding_rebinds"] = stats.binding_rebinds;
             result["sequence_submissions"] =
                 stats.sequence_submissions;
             result["compiled_graph_submissions"] =
                 stats.compiled_graph_submissions;
             result["runtime_capture_submissions"] =
                 stats.runtime_capture_submissions;
             result["backend_captures"] = stats.backend_captures;
             result["backend_replays"] = stats.backend_replays;
             result["ordinary_fallbacks"] =
                 stats.ordinary_fallbacks;
             result["cache_invalidations"] =
                 stats.cache_invalidations;
             result["backend_path"] =
                 operator_backend_execution_path_name(
                     stats.last_backend_path);
             return result;
           });

  m.def(
      "_make_linear_operator",
      [](Program *program, SparseMatrix &matrix, int self_adjoint,
         int positive_definite, int positive_semidefinite, int singular) {
        return make_linear_operator_handle(
            program, matrix,
            make_asserted_operator_traits(
                self_adjoint, positive_definite,
                positive_semidefinite, singular));
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::arg("program"), py::arg("matrix"),
      py::arg("self_adjoint") = -1,
      py::arg("positive_definite") = -1,
      py::arg("positive_semidefinite") = -1,
      py::arg("singular") = -1);
  m.def("_make_experimental_preconditioner_plan",
        make_experimental_preconditioner_plan_handle,
        py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
        py::keep_alive<0, 3>(), py::arg("program"), py::arg("target"),
        py::arg("action"), py::arg("method"));
  m.def("_make_experimental_preconditioner_action",
        make_experimental_preconditioner_action_handle,
        py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
        py::arg("program"), py::arg("plan"));
  m.def(
      "_make_compiled_kernel_operator",
      [](Program *program, Kernel &forward_kernel,
         const py::object &adjoint_kernel, std::size_t range_extent,
         std::size_t domain_extent, std::uint64_t topology_version,
         std::uint64_t numeric_version, const Ndarray &topology_data,
         const py::object &numeric_data, int self_adjoint,
         int positive_definite, int positive_semidefinite, int singular) {
        Kernel *adjoint = nullptr;
        if (!adjoint_kernel.is_none()) {
          adjoint = &py::cast<Kernel &>(adjoint_kernel);
        }
        const Ndarray *numeric = nullptr;
        if (!numeric_data.is_none()) {
          numeric = &py::cast<const Ndarray &>(numeric_data);
        }
        return make_compiled_kernel_operator_handle(
            program, forward_kernel, adjoint, range_extent, domain_extent,
            topology_version, numeric_version, topology_data, numeric,
            make_asserted_operator_traits(
                self_adjoint, positive_definite, positive_semidefinite,
                singular));
      },
      py::keep_alive<0, 1>(), py::arg("program"),
      py::arg("forward_kernel"), py::arg("adjoint_kernel"),
      py::arg("range_extent"), py::arg("domain_extent"),
      py::arg("topology_version"), py::arg("numeric_version"),
      py::arg("topology_data"), py::arg("numeric_data"),
      py::arg("self_adjoint") = -1,
      py::arg("positive_definite") = -1,
      py::arg("positive_semidefinite") = -1,
      py::arg("singular") = -1);
  m.def(
      "_make_compiled_graph_operator",
      [](Program *program, const aot::CompiledGraph &forward_graph,
         const py::object &adjoint_graph, std::size_t range_extent,
         std::size_t domain_extent, std::uint64_t topology_version,
         std::uint64_t numeric_version, const py::dict &fixed_i32_args,
         const py::dict &topology_args, const py::dict &numeric_args,
         const py::dict &workspace_args, const py::dict &state_args,
         int self_adjoint,
         int positive_definite, int positive_semidefinite, int singular) {
        const aot::CompiledGraph *adjoint = nullptr;
        if (!adjoint_graph.is_none()) {
          adjoint = &py::cast<const aot::CompiledGraph &>(adjoint_graph);
        }
        std::unordered_map<std::string, std::int32_t> fixed_i32;
        for (const auto &item : fixed_i32_args) {
          fixed_i32.emplace(py::cast<std::string>(item.first),
                            py::cast<std::int32_t>(item.second));
        }
        auto parse_ndarrays = [](const py::dict &arguments) {
          std::unordered_map<std::string, const Ndarray *> result;
          for (const auto &item : arguments) {
            result.emplace(py::cast<std::string>(item.first),
                           &py::cast<const Ndarray &>(item.second));
          }
          return result;
        };
        std::vector<SNodeTreeDependency> state_dependencies;
        state_dependencies.reserve(state_args.size());
        for (const auto &item : state_args) {
          const auto name = py::cast<std::string>(item.first);
          TI_ERROR_IF(name.empty(),
                      "Compiled-Graph fixed Field state names must be "
                      "non-empty.");
          const auto &descriptor =
              py::cast<const storage::DenseStorageDescriptor &>(item.second);
          const auto source_kind = descriptor.source_kind();
          const auto &owner = descriptor.owner();
          TI_ERROR_IF(
              (source_kind != storage::StorageSourceKind::kDenseScalarField &&
               source_kind != storage::StorageSourceKind::kDensePackedField) ||
                  owner.kind != storage::StorageOwnerKind::kSNodePayload ||
                  storage::validate_storage_owner(*program, descriptor) !=
                      storage::StorageFailureReason::kNone,
              "Compiled-Graph fixed Field state '{}' must be a live "
              "root-dense scalar, vector, or matrix Field owned by the same "
              "Program.",
              name);
          state_dependencies.push_back(owner.tree);
        }
        return make_compiled_graph_operator_handle(
            program, forward_graph, adjoint, range_extent, domain_extent,
            topology_version, numeric_version, std::move(fixed_i32),
            parse_ndarrays(topology_args), parse_ndarrays(numeric_args),
            parse_ndarrays(workspace_args), std::move(state_dependencies),
            make_asserted_operator_traits(
                self_adjoint, positive_definite, positive_semidefinite,
                singular));
      },
      py::keep_alive<0, 1>(), py::arg("program"),
      py::arg("forward_graph"), py::arg("adjoint_graph"),
      py::arg("range_extent"), py::arg("domain_extent"),
      py::arg("topology_version"), py::arg("numeric_version"),
      py::arg("fixed_i32_arguments"), py::arg("topology_arguments"),
      py::arg("numeric_arguments"), py::arg("workspace_arguments"),
      py::arg("state_arguments") = py::dict(),
      py::arg("self_adjoint") = -1,
      py::arg("positive_definite") = -1,
      py::arg("positive_semidefinite") = -1,
      py::arg("singular") = -1);
  m.def(
      "_make_identity_operator",
      [](Program *program, DataType dtype, std::size_t size) {
        return make_identity_operator_handle(
            program, OperatorSpaceDesc{dtype, size});
      },
      py::keep_alive<0, 1>(), py::arg("program"),
      py::arg("dtype"), py::arg("size"));
  m.def("_make_adjoint_operator",
        make_adjoint_operator_handle,
        py::keep_alive<0, 1>(), py::arg("operand"));
  m.def("_make_scaled_operator",
        make_scaled_operator_handle,
        py::keep_alive<0, 2>(), py::arg("scale"), py::arg("operand"));
  m.def("_make_sum_operator",
        make_sum_operator_handle,
        py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
        py::arg("left"), py::arg("right"));
  m.def("_make_composed_operator",
        make_composed_operator_handle,
        py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
        py::arg("outer"), py::arg("inner"));
  m.def("_make_parameterized_affine_operator",
        make_parameterized_affine_operator_handle,
        py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
        py::arg("left"), py::arg("right"), py::arg("alpha"),
        py::arg("beta"), py::arg("alpha_min"),
        py::arg("alpha_max"), py::arg("beta_min"),
        py::arg("beta_max"));
  m.def(
      "_make_block_diagonal_operator",
      [](const py::iterable &items) {
        std::vector<LinearOperatorHandle *> blocks;
        for (const auto &item : items) {
          blocks.push_back(
              py::cast<LinearOperatorHandle *>(item));
        }
        return make_block_diagonal_operator_handle(blocks);
      },
      py::arg("blocks"));

  py::class_<SparseJacobiPreconditionerPlan>(
      m, "SparseJacobiPreconditionerPlan")
      .def("_refresh_numeric",
           &SparseJacobiPreconditionerPlan::refresh_numeric)
      .def("apply", &SparseJacobiPreconditionerPlan::apply)
      .def("_debug_runtime_stats",
           [sparse_preconditioner_stats_to_dict](
               const SparseJacobiPreconditionerPlan &plan) {
             return sparse_preconditioner_stats_to_dict(
                 plan.debug_runtime_statistics());
           });
  m.def("_make_sparse_jacobi_preconditioner_plan",
        make_sparse_jacobi_preconditioner_plan, py::keep_alive<0, 1>(),
        py::keep_alive<0, 2>());
  py::class_<SparseBlockJacobiPreconditionerPlan>(
      m, "SparseBlockJacobiPreconditionerPlan")
      .def("apply", &SparseBlockJacobiPreconditionerPlan::apply)
      .def("_refresh_numeric",
           &SparseBlockJacobiPreconditionerPlan::refresh_numeric)
      .def("_debug_runtime_stats",
           [sparse_preconditioner_stats_to_dict](
               const SparseBlockJacobiPreconditionerPlan &plan) {
             return sparse_preconditioner_stats_to_dict(
                 plan.debug_runtime_statistics());
           });
  m.def("_make_sparse_block_jacobi_preconditioner_plan",
        make_sparse_block_jacobi_preconditioner_plan,
        py::keep_alive<0, 1>(), py::keep_alive<0, 2>());
  py::class_<CompiledKernelPreconditionerPlan>(
      m, "CompiledKernelPreconditionerPlan")
      .def("apply", &CompiledKernelPreconditionerPlan::apply)
      .def("_debug_runtime_stats",
           [sparse_preconditioner_stats_to_dict](
               const CompiledKernelPreconditionerPlan &plan) {
             return sparse_preconditioner_stats_to_dict(
                 plan.debug_runtime_statistics());
           });
  m.def("_make_compiled_kernel_preconditioner_plan",
        make_compiled_kernel_preconditioner_plan,
        py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
        py::keep_alive<0, 3>(), py::arg("program"),
        py::arg("target_operator"), py::arg("inverse_apply_operator"),
        py::arg("assume_symmetric_positive_definite"));
  py::class_<CG<Eigen::VectorXf, float>>(m, "CGf")
      .def(py::init<SparseMatrix &, int, float, bool>())
      .def("solve", &CG<Eigen::VectorXf, float>::solve)
      .def("set_x", &CG<Eigen::VectorXf, float>::set_x)
      .def("reset_x", &CG<Eigen::VectorXf, float>::reset_x)
      .def("get_x", &CG<Eigen::VectorXf, float>::get_x)
      .def("set_x_ndarray", &CG<Eigen::VectorXf, float>::set_x_ndarray)
      .def("set_b", &CG<Eigen::VectorXf, float>::set_b)
      .def("set_b_ndarray", &CG<Eigen::VectorXf, float>::set_b_ndarray)
      .def("is_success", &CG<Eigen::VectorXf, float>::is_success)
      .def("get_status", &CG<Eigen::VectorXf, float>::get_status)
      .def("get_iterations", &CG<Eigen::VectorXf, float>::get_iterations)
      .def("get_initial_residual_norm",
           &CG<Eigen::VectorXf, float>::get_initial_residual_norm)
      .def("get_residual_norm",
           &CG<Eigen::VectorXf, float>::get_residual_norm)
      .def("_get_last_result", [sparse_solve_result_to_dict](
                                    const CG<Eigen::VectorXf, float> &cg) {
        return sparse_solve_result_to_dict(cg.get_last_result());
      })
      .def("_debug_runtime_stats", [sparse_solve_plan_stats_to_dict](
                                         const CG<Eigen::VectorXf, float> &cg) {
        return sparse_solve_plan_stats_to_dict(
            cg.debug_runtime_statistics());
      });
  py::class_<CG<Eigen::VectorXd, double>>(m, "CGd")
      .def(py::init<SparseMatrix &, int, double, bool>())
      .def("solve", &CG<Eigen::VectorXd, double>::solve)
      .def("set_x", &CG<Eigen::VectorXd, double>::set_x)
      .def("reset_x", &CG<Eigen::VectorXd, double>::reset_x)
      .def("set_x_ndarray", &CG<Eigen::VectorXd, double>::set_x_ndarray)
      .def("get_x", &CG<Eigen::VectorXd, double>::get_x)
      .def("set_b_ndarray", &CG<Eigen::VectorXd, double>::set_b_ndarray)
      .def("set_b", &CG<Eigen::VectorXd, double>::set_b)
      .def("is_success", &CG<Eigen::VectorXd, double>::is_success)
      .def("get_status", &CG<Eigen::VectorXd, double>::get_status)
      .def("get_iterations", &CG<Eigen::VectorXd, double>::get_iterations)
      .def("get_initial_residual_norm",
           &CG<Eigen::VectorXd, double>::get_initial_residual_norm)
      .def("get_residual_norm",
           &CG<Eigen::VectorXd, double>::get_residual_norm)
      .def("_get_last_result", [sparse_solve_result_to_dict](
                                    const CG<Eigen::VectorXd, double> &cg) {
        return sparse_solve_result_to_dict(cg.get_last_result());
      })
      .def("_debug_runtime_stats", [sparse_solve_plan_stats_to_dict](
                                         const CG<Eigen::VectorXd, double> &cg) {
        return sparse_solve_plan_stats_to_dict(
            cg.debug_runtime_statistics());
      });
  m.def(
      "make_float_cg_solver",
      [](SparseMatrix &A, int max_iters, float absolute_tolerance,
         bool verbose, float relative_tolerance) {
        return make_cg_solver<Eigen::VectorXf, float>(
            A, max_iters, absolute_tolerance, verbose,
            relative_tolerance);
      },
      py::arg("matrix"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"), py::arg("verbose"),
      py::arg("relative_tolerance") = 0.0f);
  m.def(
      "make_double_cg_solver",
      [](SparseMatrix &A, int max_iters, double absolute_tolerance,
         bool verbose, double relative_tolerance) {
        return make_cg_solver<Eigen::VectorXd, double>(
            A, max_iters, absolute_tolerance, verbose,
            relative_tolerance);
      },
      py::arg("matrix"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"), py::arg("verbose"),
      py::arg("relative_tolerance") = 0.0);

  py::class_<SparseBiCGSTAB<Eigen::VectorXf, float>>(
      m, "SparseBiCGSTABf")
      .def("solve", &SparseBiCGSTAB<Eigen::VectorXf, float>::solve)
      .def("set_x", &SparseBiCGSTAB<Eigen::VectorXf, float>::set_x)
      .def("reset_x", &SparseBiCGSTAB<Eigen::VectorXf, float>::reset_x)
      .def("get_x", &SparseBiCGSTAB<Eigen::VectorXf, float>::get_x)
      .def("set_x_ndarray",
           &SparseBiCGSTAB<Eigen::VectorXf, float>::set_x_ndarray)
      .def("set_b", &SparseBiCGSTAB<Eigen::VectorXf, float>::set_b)
      .def("set_b_ndarray",
           &SparseBiCGSTAB<Eigen::VectorXf, float>::set_b_ndarray)
      .def("is_success",
           &SparseBiCGSTAB<Eigen::VectorXf, float>::is_success)
      .def("get_status",
           &SparseBiCGSTAB<Eigen::VectorXf, float>::get_status)
      .def("get_iterations",
           &SparseBiCGSTAB<Eigen::VectorXf, float>::get_iterations)
      .def("get_initial_residual_norm",
           &SparseBiCGSTAB<Eigen::VectorXf,
                           float>::get_initial_residual_norm)
      .def("get_residual_norm",
           &SparseBiCGSTAB<Eigen::VectorXf, float>::get_residual_norm)
      .def("_get_last_result",
           [sparse_solve_result_to_dict](
               const SparseBiCGSTAB<Eigen::VectorXf, float> &solver) {
             return sparse_solve_result_to_dict(solver.get_last_result());
           })
      .def("_debug_runtime_stats",
           [sparse_solve_plan_stats_to_dict](
               const SparseBiCGSTAB<Eigen::VectorXf, float> &solver) {
             return sparse_solve_plan_stats_to_dict(
                 solver.debug_runtime_statistics());
           });
  py::class_<SparseBiCGSTAB<Eigen::VectorXd, double>>(
      m, "SparseBiCGSTABd")
      .def("solve", &SparseBiCGSTAB<Eigen::VectorXd, double>::solve)
      .def("set_x", &SparseBiCGSTAB<Eigen::VectorXd, double>::set_x)
      .def("reset_x", &SparseBiCGSTAB<Eigen::VectorXd, double>::reset_x)
      .def("get_x", &SparseBiCGSTAB<Eigen::VectorXd, double>::get_x)
      .def("set_x_ndarray",
           &SparseBiCGSTAB<Eigen::VectorXd, double>::set_x_ndarray)
      .def("set_b", &SparseBiCGSTAB<Eigen::VectorXd, double>::set_b)
      .def("set_b_ndarray",
           &SparseBiCGSTAB<Eigen::VectorXd, double>::set_b_ndarray)
      .def("is_success",
           &SparseBiCGSTAB<Eigen::VectorXd, double>::is_success)
      .def("get_status",
           &SparseBiCGSTAB<Eigen::VectorXd, double>::get_status)
      .def("get_iterations",
           &SparseBiCGSTAB<Eigen::VectorXd, double>::get_iterations)
      .def("get_initial_residual_norm",
           &SparseBiCGSTAB<Eigen::VectorXd,
                           double>::get_initial_residual_norm)
      .def("get_residual_norm",
           &SparseBiCGSTAB<Eigen::VectorXd, double>::get_residual_norm)
      .def("_get_last_result",
           [sparse_solve_result_to_dict](
               const SparseBiCGSTAB<Eigen::VectorXd, double> &solver) {
             return sparse_solve_result_to_dict(solver.get_last_result());
           })
      .def("_debug_runtime_stats",
           [sparse_solve_plan_stats_to_dict](
               const SparseBiCGSTAB<Eigen::VectorXd, double> &solver) {
             return sparse_solve_plan_stats_to_dict(
                 solver.debug_runtime_statistics());
           });
  m.def(
      "make_float_sparse_bicgstab_solver",
      [](SparseMatrix &matrix, int max_iterations,
         float absolute_tolerance, bool verbose,
         float relative_tolerance) {
        return std::make_unique<SparseBiCGSTAB<Eigen::VectorXf, float>>(
            matrix, max_iterations, absolute_tolerance, verbose,
            relative_tolerance);
      },
      py::arg("matrix"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"), py::arg("verbose"),
      py::arg("relative_tolerance") = 0.0f);
  m.def(
      "make_double_sparse_bicgstab_solver",
      [](SparseMatrix &matrix, int max_iterations,
         double absolute_tolerance, bool verbose,
         double relative_tolerance) {
        return std::make_unique<SparseBiCGSTAB<Eigen::VectorXd, double>>(
            matrix, max_iterations, absolute_tolerance, verbose,
            relative_tolerance);
      },
      py::arg("matrix"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"), py::arg("verbose"),
      py::arg("relative_tolerance") = 0.0);

  py::class_<FixedSparseBiCGSTAB<Eigen::VectorXf, float>>(
      m, "FixedSparseBiCGSTABf")
      .def("solve", &FixedSparseBiCGSTAB<Eigen::VectorXf, float>::solve)
      .def("solve_ndarray",
           &FixedSparseBiCGSTAB<Eigen::VectorXf, float>::solve_ndarray)
      .def("set_x", &FixedSparseBiCGSTAB<Eigen::VectorXf, float>::set_x)
      .def("reset_x",
           &FixedSparseBiCGSTAB<Eigen::VectorXf, float>::reset_x)
      .def("get_x", &FixedSparseBiCGSTAB<Eigen::VectorXf, float>::get_x)
      .def("set_x_ndarray",
           &FixedSparseBiCGSTAB<Eigen::VectorXf, float>::set_x_ndarray)
      .def("set_b", &FixedSparseBiCGSTAB<Eigen::VectorXf, float>::set_b)
      .def("set_b_ndarray",
           &FixedSparseBiCGSTAB<Eigen::VectorXf, float>::set_b_ndarray)
      .def("is_success",
           &FixedSparseBiCGSTAB<Eigen::VectorXf, float>::is_success)
      .def("get_status",
           &FixedSparseBiCGSTAB<Eigen::VectorXf, float>::get_status)
      .def("get_iterations",
           &FixedSparseBiCGSTAB<Eigen::VectorXf, float>::get_iterations)
      .def("get_initial_residual_norm",
           &FixedSparseBiCGSTAB<Eigen::VectorXf,
                                float>::get_initial_residual_norm)
      .def("get_residual_norm",
           &FixedSparseBiCGSTAB<Eigen::VectorXf,
                                float>::get_residual_norm)
      .def("_get_last_result",
           [sparse_solve_result_to_dict](
               const FixedSparseBiCGSTAB<Eigen::VectorXf, float> &solver) {
             return sparse_solve_result_to_dict(solver.get_last_result());
           })
      .def("_debug_runtime_stats",
           [sparse_solve_plan_stats_to_dict](
               const FixedSparseBiCGSTAB<Eigen::VectorXf, float> &solver) {
             return sparse_solve_plan_stats_to_dict(
                 solver.debug_runtime_statistics());
           });
  py::class_<FixedSparseBiCGSTAB<Eigen::VectorXd, double>>(
      m, "FixedSparseBiCGSTABd")
      .def("solve", &FixedSparseBiCGSTAB<Eigen::VectorXd, double>::solve)
      .def("solve_ndarray",
           &FixedSparseBiCGSTAB<Eigen::VectorXd, double>::solve_ndarray)
      .def("set_x", &FixedSparseBiCGSTAB<Eigen::VectorXd, double>::set_x)
      .def("reset_x",
           &FixedSparseBiCGSTAB<Eigen::VectorXd, double>::reset_x)
      .def("get_x", &FixedSparseBiCGSTAB<Eigen::VectorXd, double>::get_x)
      .def("set_x_ndarray",
           &FixedSparseBiCGSTAB<Eigen::VectorXd, double>::set_x_ndarray)
      .def("set_b", &FixedSparseBiCGSTAB<Eigen::VectorXd, double>::set_b)
      .def("set_b_ndarray",
           &FixedSparseBiCGSTAB<Eigen::VectorXd, double>::set_b_ndarray)
      .def("is_success",
           &FixedSparseBiCGSTAB<Eigen::VectorXd, double>::is_success)
      .def("get_status",
           &FixedSparseBiCGSTAB<Eigen::VectorXd, double>::get_status)
      .def("get_iterations",
           &FixedSparseBiCGSTAB<Eigen::VectorXd, double>::get_iterations)
      .def("get_initial_residual_norm",
           &FixedSparseBiCGSTAB<Eigen::VectorXd,
                                double>::get_initial_residual_norm)
      .def("get_residual_norm",
           &FixedSparseBiCGSTAB<Eigen::VectorXd,
                                double>::get_residual_norm)
      .def("_get_last_result",
           [sparse_solve_result_to_dict](
               const FixedSparseBiCGSTAB<Eigen::VectorXd, double> &solver) {
             return sparse_solve_result_to_dict(solver.get_last_result());
           })
      .def("_debug_runtime_stats",
           [sparse_solve_plan_stats_to_dict](
               const FixedSparseBiCGSTAB<Eigen::VectorXd, double> &solver) {
             return sparse_solve_plan_stats_to_dict(
                 solver.debug_runtime_statistics());
           });
  m.def(
      "_make_float_cpu_fixed_sparse_bicgstab_solver",
      [](Program *program, SparseMatrix &matrix, int max_iterations,
         float absolute_tolerance, bool verbose,
         float relative_tolerance) {
        return std::make_unique<
            FixedSparseBiCGSTAB<Eigen::VectorXf, float>>(
            program, matrix, max_iterations, absolute_tolerance, verbose,
            relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::arg("program"), py::arg("matrix"),
      py::arg("max_iterations"), py::arg("absolute_tolerance"),
      py::arg("verbose"), py::arg("relative_tolerance") = 0.0f);
  m.def(
      "_make_double_cpu_fixed_sparse_bicgstab_solver",
      [](Program *program, SparseMatrix &matrix, int max_iterations,
         double absolute_tolerance, bool verbose,
         double relative_tolerance) {
        return std::make_unique<
            FixedSparseBiCGSTAB<Eigen::VectorXd, double>>(
            program, matrix, max_iterations, absolute_tolerance, verbose,
            relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::arg("program"), py::arg("matrix"),
      py::arg("max_iterations"), py::arg("absolute_tolerance"),
      py::arg("verbose"), py::arg("relative_tolerance") = 0.0);
  m.def(
      "_make_float_cpu_experimental_bicgstab_solver",
      [](Program *program, LinearOperatorHandle &operator_handle,
         int max_iterations, float absolute_tolerance,
         float relative_tolerance) {
        TI_ERROR_IF(operator_handle.program() != program,
                    "BiCGSTAB operator belongs to a different Program.");
        return std::make_unique<
            FixedSparseBiCGSTAB<Eigen::VectorXf, float>>(
            program, operator_handle.binding(), max_iterations,
            absolute_tolerance, false, relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::arg("program"), py::arg("operator"),
      py::arg("max_iterations"), py::arg("absolute_tolerance"),
      py::arg("relative_tolerance") = 0.0f);
  m.def(
      "_make_double_cpu_experimental_bicgstab_solver",
      [](Program *program, LinearOperatorHandle &operator_handle,
         int max_iterations, double absolute_tolerance,
         double relative_tolerance) {
        TI_ERROR_IF(operator_handle.program() != program,
                    "BiCGSTAB operator belongs to a different Program.");
        return std::make_unique<
            FixedSparseBiCGSTAB<Eigen::VectorXd, double>>(
            program, operator_handle.binding(), max_iterations,
            absolute_tolerance, false, relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::arg("program"), py::arg("operator"),
      py::arg("max_iterations"), py::arg("absolute_tolerance"),
      py::arg("relative_tolerance") = 0.0);
  m.def(
      "_make_float_cpu_preconditioned_experimental_bicgstab_solver",
      [](Program *program, LinearOperatorHandle &operator_handle,
         LinearOperatorHandle &preconditioner,
         int max_iterations, float absolute_tolerance,
         float relative_tolerance) {
        TI_ERROR_IF(operator_handle.program() != program ||
                        preconditioner.program() != program,
                    "BiCGSTAB operator and preconditioner must belong to "
                    "the target Program.");
        return std::make_unique<
            FixedSparseBiCGSTAB<Eigen::VectorXf, float>>(
            program, operator_handle.binding(), preconditioner,
            max_iterations, absolute_tolerance, false,
            relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::keep_alive<0, 3>(), py::arg("program"), py::arg("operator"),
      py::arg("preconditioner"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"), py::arg("relative_tolerance") = 0.0f);
  m.def(
      "_make_double_cpu_preconditioned_experimental_bicgstab_solver",
      [](Program *program, LinearOperatorHandle &operator_handle,
         LinearOperatorHandle &preconditioner,
         int max_iterations, double absolute_tolerance,
         double relative_tolerance) {
        TI_ERROR_IF(operator_handle.program() != program ||
                        preconditioner.program() != program,
                    "BiCGSTAB operator and preconditioner must belong to "
                    "the target Program.");
        return std::make_unique<
            FixedSparseBiCGSTAB<Eigen::VectorXd, double>>(
            program, operator_handle.binding(), preconditioner,
            max_iterations, absolute_tolerance, false,
            relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::keep_alive<0, 3>(), py::arg("program"), py::arg("operator"),
      py::arg("preconditioner"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"), py::arg("relative_tolerance") = 0.0);

  py::class_<FixedSparseGMRES<Eigen::VectorXf, float>>(
      m, "FixedSparseGMRESf")
      .def("solve_ndarray",
           &FixedSparseGMRES<Eigen::VectorXf, float>::solve_ndarray)
      .def("_get_last_result",
           [sparse_solve_result_to_dict](
               const FixedSparseGMRES<Eigen::VectorXf, float> &solver) {
             return sparse_solve_result_to_dict(solver.get_last_result());
           })
      .def("_debug_runtime_stats",
           [sparse_solve_plan_stats_to_dict](
               const FixedSparseGMRES<Eigen::VectorXf, float> &solver) {
             return sparse_solve_plan_stats_to_dict(
                 solver.debug_runtime_statistics());
           });
  py::class_<FixedSparseGMRES<Eigen::VectorXd, double>>(
      m, "FixedSparseGMRESd")
      .def("solve_ndarray",
           &FixedSparseGMRES<Eigen::VectorXd, double>::solve_ndarray)
      .def("_get_last_result",
           [sparse_solve_result_to_dict](
               const FixedSparseGMRES<Eigen::VectorXd, double> &solver) {
             return sparse_solve_result_to_dict(solver.get_last_result());
           })
      .def("_debug_runtime_stats",
           [sparse_solve_plan_stats_to_dict](
               const FixedSparseGMRES<Eigen::VectorXd, double> &solver) {
             return sparse_solve_plan_stats_to_dict(
                 solver.debug_runtime_statistics());
           });
  m.def(
      "_make_float_cpu_experimental_gmres_solver",
      [](Program *program, LinearOperatorHandle &operator_handle,
         int max_iterations, int restart, float absolute_tolerance,
         float relative_tolerance) {
        TI_ERROR_IF(operator_handle.program() != program,
                    "GMRES operator belongs to a different Program.");
        return std::make_unique<
            FixedSparseGMRES<Eigen::VectorXf, float>>(
            program, operator_handle.binding(), max_iterations, restart,
            absolute_tolerance, false, relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::arg("program"), py::arg("operator"),
      py::arg("max_iterations"), py::arg("restart"),
      py::arg("absolute_tolerance"), py::arg("relative_tolerance") = 0.0f);
  m.def(
      "_make_double_cpu_experimental_gmres_solver",
      [](Program *program, LinearOperatorHandle &operator_handle,
         int max_iterations, int restart, double absolute_tolerance,
         double relative_tolerance) {
        TI_ERROR_IF(operator_handle.program() != program,
                    "GMRES operator belongs to a different Program.");
        return std::make_unique<
            FixedSparseGMRES<Eigen::VectorXd, double>>(
            program, operator_handle.binding(), max_iterations, restart,
            absolute_tolerance, false, relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::arg("program"), py::arg("operator"),
      py::arg("max_iterations"), py::arg("restart"),
      py::arg("absolute_tolerance"), py::arg("relative_tolerance") = 0.0);
  m.def(
      "_make_float_cpu_preconditioned_experimental_gmres_solver",
      [](Program *program, LinearOperatorHandle &operator_handle,
         LinearOperatorHandle &preconditioner,
         int max_iterations, int restart, float absolute_tolerance,
         float relative_tolerance) {
        TI_ERROR_IF(operator_handle.program() != program ||
                        preconditioner.program() != program,
                    "GMRES operator and preconditioner must belong to "
                    "the target Program.");
        return std::make_unique<
            FixedSparseGMRES<Eigen::VectorXf, float>>(
            program, operator_handle.binding(), preconditioner,
            max_iterations, restart, absolute_tolerance, false,
            relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::keep_alive<0, 3>(), py::arg("program"), py::arg("operator"),
      py::arg("preconditioner"), py::arg("max_iterations"),
      py::arg("restart"), py::arg("absolute_tolerance"),
      py::arg("relative_tolerance") = 0.0f);
  m.def(
      "_make_double_cpu_preconditioned_experimental_gmres_solver",
      [](Program *program, LinearOperatorHandle &operator_handle,
         LinearOperatorHandle &preconditioner,
         int max_iterations, int restart, double absolute_tolerance,
         double relative_tolerance) {
        TI_ERROR_IF(operator_handle.program() != program ||
                        preconditioner.program() != program,
                    "GMRES operator and preconditioner must belong to "
                    "the target Program.");
        return std::make_unique<
            FixedSparseGMRES<Eigen::VectorXd, double>>(
            program, operator_handle.binding(), preconditioner,
            max_iterations, restart, absolute_tolerance, false,
            relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::keep_alive<0, 3>(), py::arg("program"), py::arg("operator"),
      py::arg("preconditioner"), py::arg("max_iterations"),
      py::arg("restart"), py::arg("absolute_tolerance"),
      py::arg("relative_tolerance") = 0.0);
  m.def(
      "_make_float_cpu_variable_preconditioned_experimental_gmres_solver",
      [](Program *program, LinearOperatorHandle &operator_handle,
         const py::list &preconditioners, int max_iterations, int restart,
         float absolute_tolerance, float relative_tolerance) {
        std::vector<LinearOperatorHandle *> actions;
        actions.reserve(preconditioners.size());
        for (const auto &item : preconditioners) {
          auto *action =
              item.cast<LinearOperatorHandle *>();
          TI_ERROR_IF(!action || action->program() != program,
                      "FGMRES actions must belong to the target Program.");
          actions.push_back(action);
        }
        TI_ERROR_IF(operator_handle.program() != program,
                    "FGMRES operator belongs to a different Program.");
        return std::make_unique<
            FixedSparseGMRES<Eigen::VectorXf, float>>(
            program, operator_handle.binding(), actions, max_iterations,
            restart, absolute_tolerance, false, relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::keep_alive<0, 3>(), py::arg("program"), py::arg("operator"),
      py::arg("preconditioners"), py::arg("max_iterations"),
      py::arg("restart"), py::arg("absolute_tolerance"),
      py::arg("relative_tolerance") = 0.0f);
  m.def(
      "_make_double_cpu_variable_preconditioned_experimental_gmres_solver",
      [](Program *program, LinearOperatorHandle &operator_handle,
         const py::list &preconditioners, int max_iterations, int restart,
         double absolute_tolerance, double relative_tolerance) {
        std::vector<LinearOperatorHandle *> actions;
        actions.reserve(preconditioners.size());
        for (const auto &item : preconditioners) {
          auto *action =
              item.cast<LinearOperatorHandle *>();
          TI_ERROR_IF(!action || action->program() != program,
                      "FGMRES actions must belong to the target Program.");
          actions.push_back(action);
        }
        TI_ERROR_IF(operator_handle.program() != program,
                    "FGMRES operator belongs to a different Program.");
        return std::make_unique<
            FixedSparseGMRES<Eigen::VectorXd, double>>(
            program, operator_handle.binding(), actions, max_iterations,
            restart, absolute_tolerance, false, relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::keep_alive<0, 3>(), py::arg("program"), py::arg("operator"),
      py::arg("preconditioners"), py::arg("max_iterations"),
      py::arg("restart"), py::arg("absolute_tolerance"),
      py::arg("relative_tolerance") = 0.0);

  py::class_<OperatorMINRES<Eigen::VectorXf, float>>(m, "OperatorMINRESf")
      .def("solve_ndarray",
           &OperatorMINRES<Eigen::VectorXf, float>::solve_ndarray)
      .def("_get_last_result",
           [sparse_solve_result_to_dict](
               const OperatorMINRES<Eigen::VectorXf, float> &solver) {
             return sparse_solve_result_to_dict(solver.get_last_result());
           })
      .def("_debug_runtime_stats",
           [sparse_solve_plan_stats_to_dict](
               const OperatorMINRES<Eigen::VectorXf, float> &solver) {
             return sparse_solve_plan_stats_to_dict(
                 solver.debug_runtime_statistics());
           });
  py::class_<OperatorMINRES<Eigen::VectorXd, double>>(m, "OperatorMINRESd")
      .def("solve_ndarray",
           &OperatorMINRES<Eigen::VectorXd, double>::solve_ndarray)
      .def("_get_last_result",
           [sparse_solve_result_to_dict](
               const OperatorMINRES<Eigen::VectorXd, double> &solver) {
             return sparse_solve_result_to_dict(solver.get_last_result());
           })
      .def("_debug_runtime_stats",
           [sparse_solve_plan_stats_to_dict](
               const OperatorMINRES<Eigen::VectorXd, double> &solver) {
             return sparse_solve_plan_stats_to_dict(
                 solver.debug_runtime_statistics());
           });
  m.def(
      "_make_float_cpu_experimental_minres_solver",
      [](Program *program, LinearOperatorHandle &operator_handle,
         int max_iterations, float absolute_tolerance,
         float relative_tolerance) {
        TI_ERROR_IF(operator_handle.program() != program,
                    "MINRES operator belongs to a different Program.");
        return std::make_unique<OperatorMINRES<Eigen::VectorXf, float>>(
            program, operator_handle.binding(), max_iterations,
            absolute_tolerance, relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::arg("program"), py::arg("operator"),
      py::arg("max_iterations"), py::arg("absolute_tolerance"),
      py::arg("relative_tolerance") = 0.0f);
  m.def(
      "_make_double_cpu_experimental_minres_solver",
      [](Program *program, LinearOperatorHandle &operator_handle,
         int max_iterations, double absolute_tolerance,
         double relative_tolerance) {
        TI_ERROR_IF(operator_handle.program() != program,
                    "MINRES operator belongs to a different Program.");
        return std::make_unique<OperatorMINRES<Eigen::VectorXd, double>>(
            program, operator_handle.binding(), max_iterations,
            absolute_tolerance, relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::arg("program"), py::arg("operator"),
      py::arg("max_iterations"), py::arg("absolute_tolerance"),
      py::arg("relative_tolerance") = 0.0);

  py::class_<DeviceMINRES>(m, "DeviceMINRES")
      .def("solve", &DeviceMINRES::solve)
      .def(
          "_configure_execution_policy",
          [](DeviceMINRES &solver, const std::string &policy,
             int check_interval) {
            SparseSolveExecutionPolicy native_policy;
            if (policy == "host_each_iteration") {
              native_policy =
                  SparseSolveExecutionPolicy::host_each_iteration;
            } else if (policy == "host_check_every_k") {
              native_policy =
                  SparseSolveExecutionPolicy::host_check_every_k;
            } else if (policy == "fixed_budget_masked") {
              native_policy =
                  SparseSolveExecutionPolicy::fixed_budget_masked;
            } else {
              TI_ERROR("Unsupported device MINRES execution policy '{}'.",
                       policy);
            }
            solver.configure_execution_policy(native_policy,
                                              check_interval);
          },
          py::arg("policy"), py::arg("check_interval"))
      .def("is_success", &DeviceMINRES::is_success)
      .def("get_status", &DeviceMINRES::get_status)
      .def("get_iterations", &DeviceMINRES::get_iterations)
      .def("get_initial_residual_norm",
           &DeviceMINRES::get_initial_residual_norm)
      .def("get_residual_norm", &DeviceMINRES::get_residual_norm)
      .def("_get_last_result",
           [sparse_solve_result_to_dict](const DeviceMINRES &solver) {
             return sparse_solve_result_to_dict(solver.get_last_result());
           })
      .def("_debug_runtime_stats",
           [sparse_solve_plan_stats_to_dict](const DeviceMINRES &solver) {
             return sparse_solve_plan_stats_to_dict(
                 solver.debug_runtime_statistics());
           });
  m.def(
      "_make_device_experimental_minres_solver",
      [](Program *program, LinearOperatorHandle &operator_handle,
         int max_iterations, float absolute_tolerance,
         float relative_tolerance) {
        return make_device_minres_solver(
            program, operator_handle, nullptr, nullptr, nullptr, nullptr,
            max_iterations, absolute_tolerance, relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::arg("program"), py::arg("operator"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"), py::arg("relative_tolerance") = 0.0f);
  m.def(
      "_make_device_fixed_sparse_minres_solver",
      [](Program *program, LinearOperatorHandle &operator_handle,
         SparseMatrix &matrix, int max_iterations,
         float absolute_tolerance, float relative_tolerance) {
        return make_device_minres_solver(
            program, operator_handle, &matrix, nullptr, nullptr, nullptr,
            max_iterations, absolute_tolerance, relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::keep_alive<0, 3>(), py::arg("program"), py::arg("operator"),
      py::arg("matrix"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"), py::arg("relative_tolerance") = 0.0f);
  m.def(
      "_make_device_jacobi_minres_solver",
      [](Program *program, LinearOperatorHandle &operator_handle,
         SparseMatrix &matrix, SparseJacobiPreconditionerPlan &preconditioner,
         int max_iterations, float absolute_tolerance,
         float relative_tolerance) {
        return make_device_minres_solver(
            program, operator_handle, &matrix, &preconditioner, nullptr,
            nullptr, max_iterations, absolute_tolerance,
            relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::keep_alive<0, 3>(), py::keep_alive<0, 4>(),
      py::arg("program"), py::arg("operator"), py::arg("matrix"),
      py::arg("preconditioner"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"), py::arg("relative_tolerance") = 0.0f);
  m.def(
      "_make_device_block_jacobi_minres_solver",
      [](Program *program, LinearOperatorHandle &operator_handle,
         SparseMatrix &matrix,
         SparseBlockJacobiPreconditionerPlan &preconditioner,
         int max_iterations, float absolute_tolerance,
         float relative_tolerance) {
        return make_device_minres_solver(
            program, operator_handle, &matrix, nullptr, &preconditioner,
            nullptr, max_iterations, absolute_tolerance,
            relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::keep_alive<0, 3>(), py::keep_alive<0, 4>(),
      py::arg("program"), py::arg("operator"), py::arg("matrix"),
      py::arg("preconditioner"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"), py::arg("relative_tolerance") = 0.0f);
  m.def(
      "_make_device_operator_preconditioned_minres_solver",
      [](Program *program, LinearOperatorHandle &operator_handle,
         LinearOperatorHandle &preconditioner,
         int max_iterations, float absolute_tolerance,
         float relative_tolerance) {
        return make_device_minres_solver(
            program, operator_handle, nullptr, nullptr, nullptr,
            &preconditioner, max_iterations, absolute_tolerance,
            relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::keep_alive<0, 3>(), py::arg("program"), py::arg("operator"),
      py::arg("preconditioner"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"), py::arg("relative_tolerance") = 0.0f);

  py::class_<DeviceGMRES>(m, "DeviceGMRES")
      .def("solve", &DeviceGMRES::solve)
      .def(
          "_configure_execution_policy",
          [](DeviceGMRES &solver, const std::string &policy,
             int check_interval) {
            SparseSolveExecutionPolicy native_policy;
            if (policy == "host_check_every_k") {
              native_policy =
                  SparseSolveExecutionPolicy::host_check_every_k;
            } else if (policy == "fixed_budget_masked") {
              native_policy =
                  SparseSolveExecutionPolicy::fixed_budget_masked;
            } else {
              TI_ERROR("Unsupported device GMRES execution policy '{}'.",
                       policy);
            }
            solver.configure_execution_policy(native_policy,
                                              check_interval);
          },
          py::arg("policy"), py::arg("check_interval"))
      .def("is_success", &DeviceGMRES::is_success)
      .def("get_status", &DeviceGMRES::get_status)
      .def("get_iterations", &DeviceGMRES::get_iterations)
      .def("get_initial_residual_norm",
           &DeviceGMRES::get_initial_residual_norm)
      .def("get_residual_norm", &DeviceGMRES::get_residual_norm)
      .def("_get_last_result",
           [sparse_solve_result_to_dict](const DeviceGMRES &solver) {
             return sparse_solve_result_to_dict(solver.get_last_result());
           })
      .def("_debug_runtime_stats",
           [sparse_solve_plan_stats_to_dict](const DeviceGMRES &solver) {
             return sparse_solve_plan_stats_to_dict(
                 solver.debug_runtime_statistics());
           });
  m.def(
      "_make_device_experimental_gmres_solver",
      [](Program *program, LinearOperatorHandle &operator_handle,
         int max_iterations, int restart, float absolute_tolerance,
         float relative_tolerance) {
        return make_device_gmres_solver(
            program, operator_handle, nullptr, nullptr, max_iterations,
            restart, absolute_tolerance, relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::arg("program"), py::arg("operator"), py::arg("max_iterations"),
      py::arg("restart"), py::arg("absolute_tolerance"),
      py::arg("relative_tolerance") = 0.0f);
  m.def(
      "_make_device_fixed_sparse_gmres_solver",
      [](Program *program, LinearOperatorHandle &operator_handle,
         SparseMatrix &matrix, int max_iterations, int restart,
         float absolute_tolerance, float relative_tolerance) {
        return make_device_gmres_solver(
            program, operator_handle, &matrix, nullptr, max_iterations,
            restart, absolute_tolerance, relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::keep_alive<0, 3>(), py::arg("program"), py::arg("operator"),
      py::arg("matrix"), py::arg("max_iterations"), py::arg("restart"),
      py::arg("absolute_tolerance"),
      py::arg("relative_tolerance") = 0.0f);
  m.def(
      "_make_device_operator_preconditioned_gmres_solver",
      [](Program *program, LinearOperatorHandle &operator_handle,
         LinearOperatorHandle &preconditioner,
         int max_iterations, int restart, float absolute_tolerance,
         float relative_tolerance) {
        return make_device_gmres_solver(
            program, operator_handle, nullptr, &preconditioner,
            max_iterations, restart, absolute_tolerance,
            relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::keep_alive<0, 3>(), py::arg("program"), py::arg("operator"),
      py::arg("preconditioner"), py::arg("max_iterations"),
      py::arg("restart"), py::arg("absolute_tolerance"),
      py::arg("relative_tolerance") = 0.0f);
  m.def(
      "_make_device_variable_preconditioned_gmres_solver",
      [](Program *program, LinearOperatorHandle &operator_handle,
         const py::list &preconditioners, int max_iterations, int restart,
         float absolute_tolerance, float relative_tolerance) {
        std::vector<LinearOperatorHandle *> actions;
        actions.reserve(preconditioners.size());
        for (const auto &item : preconditioners) {
          auto *action =
              item.cast<LinearOperatorHandle *>();
          TI_ERROR_IF(!action || action->program() != program,
                      "Device FGMRES actions must belong to the target "
                      "Program.");
          actions.push_back(action);
        }
        TI_ERROR_IF(operator_handle.program() != program,
                    "Device FGMRES operator belongs to a different "
                    "Program.");
        return make_device_fgmres_solver(
            program, operator_handle, std::move(actions), max_iterations,
            restart, absolute_tolerance, relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::keep_alive<0, 3>(), py::arg("program"), py::arg("operator"),
      py::arg("preconditioners"), py::arg("max_iterations"),
      py::arg("restart"), py::arg("absolute_tolerance"),
      py::arg("relative_tolerance") = 0.0f);

  py::class_<DeviceBiCGSTAB>(m, "DeviceBiCGSTAB")
      .def("solve", &DeviceBiCGSTAB::solve)
      .def(
          "_configure_execution_policy",
          [](DeviceBiCGSTAB &solver, const std::string &policy,
             int check_interval) {
            SparseSolveExecutionPolicy native_policy;
            if (policy == "host_each_iteration") {
              native_policy =
                  SparseSolveExecutionPolicy::host_each_iteration;
            } else if (policy == "host_check_every_k") {
              native_policy =
                  SparseSolveExecutionPolicy::host_check_every_k;
            } else if (policy == "fixed_budget_masked") {
              native_policy =
                  SparseSolveExecutionPolicy::fixed_budget_masked;
            } else {
              TI_ERROR(
                  "Unsupported device BiCGSTAB execution policy '{}'.",
                  policy);
            }
            solver.configure_execution_policy(native_policy,
                                              check_interval);
          },
          py::arg("policy"), py::arg("check_interval"))
      .def("is_success", &DeviceBiCGSTAB::is_success)
      .def("get_status", &DeviceBiCGSTAB::get_status)
      .def("get_iterations", &DeviceBiCGSTAB::get_iterations)
      .def("get_initial_residual_norm",
           &DeviceBiCGSTAB::get_initial_residual_norm)
      .def("get_residual_norm", &DeviceBiCGSTAB::get_residual_norm)
      .def("_get_last_result",
           [sparse_solve_result_to_dict](const DeviceBiCGSTAB &solver) {
             return sparse_solve_result_to_dict(solver.get_last_result());
           })
      .def("_debug_runtime_stats",
           [sparse_solve_plan_stats_to_dict](const DeviceBiCGSTAB &solver) {
             return sparse_solve_plan_stats_to_dict(
                 solver.debug_runtime_statistics());
           });
  m.def(
      "_make_device_experimental_bicgstab_solver",
      [](Program *program, LinearOperatorHandle &operator_handle,
         int max_iterations, float absolute_tolerance,
         float relative_tolerance) {
        return make_device_bicgstab_solver(
            program, operator_handle, nullptr, nullptr, max_iterations,
            absolute_tolerance, relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::arg("program"), py::arg("operator"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"), py::arg("relative_tolerance") = 0.0f);
  m.def(
      "_make_device_fixed_sparse_bicgstab_solver",
      [](Program *program, LinearOperatorHandle &operator_handle,
         SparseMatrix &matrix, int max_iterations,
         float absolute_tolerance, float relative_tolerance) {
        return make_device_bicgstab_solver(
            program, operator_handle, &matrix, nullptr, max_iterations,
            absolute_tolerance, relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::keep_alive<0, 3>(), py::arg("program"), py::arg("operator"),
      py::arg("matrix"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"), py::arg("relative_tolerance") = 0.0f);
  m.def(
      "_make_device_operator_preconditioned_bicgstab_solver",
      [](Program *program, LinearOperatorHandle &operator_handle,
         LinearOperatorHandle &preconditioner,
         int max_iterations, float absolute_tolerance,
         float relative_tolerance) {
        return make_device_bicgstab_solver(
            program, operator_handle, nullptr, &preconditioner,
            max_iterations, absolute_tolerance, relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::keep_alive<0, 3>(), py::arg("program"), py::arg("operator"),
      py::arg("preconditioner"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"), py::arg("relative_tolerance") = 0.0f);

  py::class_<SparseMINRES<Eigen::VectorXf, float>>(m, "SparseMINRESf")
      .def("solve", &SparseMINRES<Eigen::VectorXf, float>::solve)
      .def("set_x", &SparseMINRES<Eigen::VectorXf, float>::set_x)
      .def("reset_x", &SparseMINRES<Eigen::VectorXf, float>::reset_x)
      .def("get_x", &SparseMINRES<Eigen::VectorXf, float>::get_x)
      .def("set_x_ndarray",
           &SparseMINRES<Eigen::VectorXf, float>::set_x_ndarray)
      .def("set_b", &SparseMINRES<Eigen::VectorXf, float>::set_b)
      .def("set_b_ndarray",
           &SparseMINRES<Eigen::VectorXf, float>::set_b_ndarray)
      .def("is_success", &SparseMINRES<Eigen::VectorXf, float>::is_success)
      .def("get_status", &SparseMINRES<Eigen::VectorXf, float>::get_status)
      .def("get_iterations",
           &SparseMINRES<Eigen::VectorXf, float>::get_iterations)
      .def("get_initial_residual_norm",
           &SparseMINRES<Eigen::VectorXf,
                         float>::get_initial_residual_norm)
      .def("get_residual_norm",
           &SparseMINRES<Eigen::VectorXf, float>::get_residual_norm)
      .def("_get_last_result",
           [sparse_solve_result_to_dict](
               const SparseMINRES<Eigen::VectorXf, float> &solver) {
             return sparse_solve_result_to_dict(solver.get_last_result());
           })
      .def("_debug_runtime_stats",
           [sparse_solve_plan_stats_to_dict](
               const SparseMINRES<Eigen::VectorXf, float> &solver) {
             return sparse_solve_plan_stats_to_dict(
                 solver.debug_runtime_statistics());
           });
  py::class_<SparseMINRES<Eigen::VectorXd, double>>(m, "SparseMINRESd")
      .def("solve", &SparseMINRES<Eigen::VectorXd, double>::solve)
      .def("set_x", &SparseMINRES<Eigen::VectorXd, double>::set_x)
      .def("reset_x", &SparseMINRES<Eigen::VectorXd, double>::reset_x)
      .def("get_x", &SparseMINRES<Eigen::VectorXd, double>::get_x)
      .def("set_x_ndarray",
           &SparseMINRES<Eigen::VectorXd, double>::set_x_ndarray)
      .def("set_b", &SparseMINRES<Eigen::VectorXd, double>::set_b)
      .def("set_b_ndarray",
           &SparseMINRES<Eigen::VectorXd, double>::set_b_ndarray)
      .def("is_success", &SparseMINRES<Eigen::VectorXd, double>::is_success)
      .def("get_status", &SparseMINRES<Eigen::VectorXd, double>::get_status)
      .def("get_iterations",
           &SparseMINRES<Eigen::VectorXd, double>::get_iterations)
      .def("get_initial_residual_norm",
           &SparseMINRES<Eigen::VectorXd,
                         double>::get_initial_residual_norm)
      .def("get_residual_norm",
           &SparseMINRES<Eigen::VectorXd, double>::get_residual_norm)
      .def("_get_last_result",
           [sparse_solve_result_to_dict](
               const SparseMINRES<Eigen::VectorXd, double> &solver) {
             return sparse_solve_result_to_dict(solver.get_last_result());
           })
      .def("_debug_runtime_stats",
           [sparse_solve_plan_stats_to_dict](
               const SparseMINRES<Eigen::VectorXd, double> &solver) {
             return sparse_solve_plan_stats_to_dict(
                 solver.debug_runtime_statistics());
           });
  m.def(
      "make_float_sparse_minres_solver",
      [](SparseMatrix &matrix, int max_iterations,
         float absolute_tolerance, bool verbose,
         float relative_tolerance) {
        return std::make_unique<SparseMINRES<Eigen::VectorXf, float>>(
            matrix, max_iterations, absolute_tolerance, verbose,
            relative_tolerance);
      },
      py::arg("matrix"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"), py::arg("verbose"),
      py::arg("relative_tolerance") = 0.0f);
  m.def(
      "make_double_sparse_minres_solver",
      [](SparseMatrix &matrix, int max_iterations,
         double absolute_tolerance, bool verbose,
         double relative_tolerance) {
        return std::make_unique<SparseMINRES<Eigen::VectorXd, double>>(
            matrix, max_iterations, absolute_tolerance, verbose,
            relative_tolerance);
      },
      py::arg("matrix"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"), py::arg("verbose"),
      py::arg("relative_tolerance") = 0.0);
  m.def(
      "_make_float_cpu_fixed_sparse_minres_solver",
      [](Program *program, SparseMatrix &matrix, int max_iterations,
         float absolute_tolerance, bool verbose,
         float relative_tolerance) {
        return std::make_unique<SparseMINRES<Eigen::VectorXf, float>>(
            program, matrix, max_iterations, absolute_tolerance, verbose,
            relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::arg("program"), py::arg("matrix"),
      py::arg("max_iterations"), py::arg("absolute_tolerance"),
      py::arg("verbose"), py::arg("relative_tolerance") = 0.0f);
  m.def(
      "_make_double_cpu_fixed_sparse_minres_solver",
      [](Program *program, SparseMatrix &matrix, int max_iterations,
         double absolute_tolerance, bool verbose,
         double relative_tolerance) {
        return std::make_unique<SparseMINRES<Eigen::VectorXd, double>>(
            program, matrix, max_iterations, absolute_tolerance, verbose,
            relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::arg("program"), py::arg("matrix"),
      py::arg("max_iterations"), py::arg("absolute_tolerance"),
      py::arg("verbose"), py::arg("relative_tolerance") = 0.0);

  py::class_<CUCG>(m, "CUCG")
      .def("solve", &CUCG::solve)
      .def(
          "_configure_execution_policy",
          [](CUCG &cg, const std::string &policy, int check_interval,
             bool require_native) {
            SparseSolveExecutionPolicy native_policy;
            if (policy == "host_each_iteration") {
              native_policy =
                  SparseSolveExecutionPolicy::host_each_iteration;
            } else if (policy == "host_check_every_k") {
              native_policy =
                  SparseSolveExecutionPolicy::host_check_every_k;
            } else if (policy == "bounded_convergent") {
              native_policy = SparseSolveExecutionPolicy::bounded_convergent;
            } else if (policy == "device_convergent") {
              native_policy = SparseSolveExecutionPolicy::device_convergent;
            } else {
              TI_ERROR("Unsupported CUDA CG execution policy '{}'.",
                       policy);
            }
            cg.configure_execution_policy(native_policy, check_interval,
                                          require_native);
          },
          py::arg("policy"), py::arg("check_interval"),
          py::arg("require_native") = false)
      .def("is_success", &CUCG::is_success)
      .def("get_status", &CUCG::get_status)
      .def("get_iterations", &CUCG::get_iterations)
      .def("get_initial_residual_norm", &CUCG::get_initial_residual_norm)
      .def("get_residual_norm", &CUCG::get_residual_norm)
      .def("_get_last_result", [sparse_solve_result_to_dict](const CUCG &cg) {
        return sparse_solve_result_to_dict(cg.get_last_result());
      })
      .def("_debug_runtime_stats", [sparse_solve_plan_stats_to_dict](
                                         const CUCG &cg) {
        return sparse_solve_plan_stats_to_dict(
            cg.debug_runtime_statistics());
      });
  m.def(
      "make_cucg_solver",
      [](SparseMatrix &matrix, int max_iterations,
         float absolute_tolerance, bool verbose,
         float relative_tolerance) {
        return make_cucg_solver(matrix, max_iterations,
                                absolute_tolerance, verbose,
                                relative_tolerance);
      },
      py::arg("matrix"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"), py::arg("verbose"),
      py::arg("relative_tolerance") = 0.0f);
  m.def(
      "_make_cuda_jacobi_pcg_solver",
      [](Program *program, SparseMatrix &matrix,
         SparseJacobiPreconditionerPlan &preconditioner,
         int max_iterations, float absolute_tolerance, bool verbose,
         float relative_tolerance) {
        return make_cuda_jacobi_pcg_solver(
            program, matrix, preconditioner, max_iterations,
            absolute_tolerance, verbose, relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::keep_alive<0, 3>(), py::arg("program"), py::arg("matrix"),
      py::arg("preconditioner"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"), py::arg("verbose"),
      py::arg("relative_tolerance") = 0.0f);
  m.def(
      "_make_cuda_block_jacobi_pcg_solver",
      [](Program *program, SparseMatrix &matrix,
         SparseBlockJacobiPreconditionerPlan &preconditioner,
         int max_iterations, float absolute_tolerance, bool verbose,
         float relative_tolerance) {
        return make_cuda_block_jacobi_pcg_solver(
            program, matrix, preconditioner, max_iterations,
            absolute_tolerance, verbose, relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::keep_alive<0, 3>(), py::arg("program"), py::arg("matrix"),
      py::arg("preconditioner"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"), py::arg("verbose"),
      py::arg("relative_tolerance") = 0.0f);
  m.def(
      "_make_cuda_compiled_kernel_cg_solver",
      make_cuda_compiled_kernel_cg_solver,
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::arg("program"), py::arg("matrix"),
      py::arg("max_iterations"), py::arg("absolute_tolerance"),
      py::arg("verbose"), py::arg("relative_tolerance") = 0.0f);
  m.def(
      "_make_cuda_compiled_graph_cg_solver",
      make_cuda_compiled_graph_cg_solver,
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::arg("program"), py::arg("matrix"),
      py::arg("max_iterations"), py::arg("absolute_tolerance"),
      py::arg("verbose"), py::arg("relative_tolerance") = 0.0f);
  m.def(
      "_make_cuda_compiled_kernel_pcg_solver",
      make_cuda_compiled_kernel_pcg_solver,
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::keep_alive<0, 3>(), py::arg("program"), py::arg("matrix"),
      py::arg("preconditioner"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"), py::arg("verbose"),
      py::arg("relative_tolerance") = 0.0f);
  m.def(
      "_make_cuda_experimental_pcg_solver",
      make_cuda_experimental_pcg_solver,
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::keep_alive<0, 3>(), py::arg("program"), py::arg("matrix"),
      py::arg("preconditioner"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"), py::arg("verbose"),
      py::arg("relative_tolerance") = 0.0f);

  py::class_<CpuSparseCGPlan>(m, "CpuSparseCGPlan")
      .def("solve", &CpuSparseCGPlan::solve)
      .def("is_success", &CpuSparseCGPlan::is_success)
      .def("get_status", &CpuSparseCGPlan::get_status)
      .def("get_iterations", &CpuSparseCGPlan::get_iterations)
      .def("get_initial_residual_norm",
           &CpuSparseCGPlan::get_initial_residual_norm)
      .def("get_residual_norm", &CpuSparseCGPlan::get_residual_norm)
      .def("_get_last_result",
           [sparse_solve_result_to_dict](const CpuSparseCGPlan &plan) {
             return sparse_solve_result_to_dict(plan.get_last_result());
           })
      .def("_debug_runtime_stats",
           [sparse_solve_plan_stats_to_dict](const CpuSparseCGPlan &plan) {
             return sparse_solve_plan_stats_to_dict(
                 plan.debug_runtime_statistics());
           });
  m.attr("CpuBsrCGPlan") = m.attr("CpuSparseCGPlan");
  m.def(
      "_make_cpu_operator_cg_solver",
      [](Program *program, SparseMatrix &matrix, int max_iterations,
         double absolute_tolerance, double relative_tolerance) {
        return make_cpu_operator_cg_solver(
            program, matrix, max_iterations, absolute_tolerance,
            relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::arg("program"), py::arg("matrix"),
      py::arg("max_iterations"), py::arg("absolute_tolerance"),
      py::arg("relative_tolerance") = 0.0);
  m.def(
      "_make_cpu_experimental_cg_solver",
      make_cpu_experimental_cg_solver,
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::arg("program"), py::arg("operator"),
      py::arg("max_iterations"), py::arg("absolute_tolerance"),
      py::arg("relative_tolerance") = 0.0);
  m.def(
      "_make_cpu_experimental_pcg_solver",
      make_cpu_experimental_pcg_solver,
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::keep_alive<0, 3>(), py::arg("program"), py::arg("operator"),
      py::arg("preconditioner"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"),
      py::arg("relative_tolerance") = 0.0);
  m.def(
      "_make_cpu_jacobi_pcg_solver",
      [](Program *program, SparseMatrix &matrix,
         SparseJacobiPreconditionerPlan &preconditioner,
         int max_iterations, double absolute_tolerance,
         double relative_tolerance) {
        return make_cpu_jacobi_pcg_solver(
            program, matrix, preconditioner, max_iterations,
            absolute_tolerance, relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::keep_alive<0, 3>(), py::arg("program"), py::arg("matrix"),
      py::arg("preconditioner"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"),
      py::arg("relative_tolerance") = 0.0);
  m.def(
      "_make_cpu_block_jacobi_pcg_solver",
      [](Program *program, SparseMatrix &matrix,
         SparseBlockJacobiPreconditionerPlan &preconditioner,
         int max_iterations, double absolute_tolerance,
         double relative_tolerance) {
        return make_cpu_block_jacobi_pcg_solver(
            program, matrix, preconditioner, max_iterations,
            absolute_tolerance, relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::keep_alive<0, 3>(), py::arg("program"), py::arg("matrix"),
      py::arg("preconditioner"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"),
      py::arg("relative_tolerance") = 0.0);
  m.def(
      "_make_cpu_compiled_kernel_pcg_solver",
      make_cpu_compiled_kernel_pcg_solver,
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::keep_alive<0, 3>(), py::arg("program"), py::arg("matrix"),
      py::arg("preconditioner"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"),
      py::arg("relative_tolerance") = 0.0);

  py::class_<VulkanCGIterationPlan>(m, "VulkanCGIterationPlan")
      .def("solve", &VulkanCGIterationPlan::solve)
      .def(
          "_configure_execution_policy",
          [](VulkanCGIterationPlan &cg, const std::string &policy,
             int check_interval) {
            SparseSolveExecutionPolicy native_policy;
            if (policy == "host_check_every_k") {
              native_policy =
                  SparseSolveExecutionPolicy::host_check_every_k;
            } else if (policy == "bounded_convergent") {
              native_policy = SparseSolveExecutionPolicy::bounded_convergent;
            } else if (policy == "fixed_budget_masked") {
              native_policy =
                  SparseSolveExecutionPolicy::fixed_budget_masked;
            } else {
              TI_ERROR("Unsupported Vulkan CG execution policy '{}'.",
                       policy);
            }
            cg.configure_execution_policy(native_policy, check_interval);
          },
          py::arg("policy"), py::arg("check_interval"))
      .def("is_success", &VulkanCGIterationPlan::is_success)
      .def("get_iterations",
           &VulkanCGIterationPlan::get_iterations)
      .def("get_initial_residual_norm",
           &VulkanCGIterationPlan::get_initial_residual_norm)
      .def("get_residual_norm",
           &VulkanCGIterationPlan::get_residual_norm)
      .def("get_status", &VulkanCGIterationPlan::get_status)
      .def("_get_last_result", [sparse_solve_result_to_dict](
                                    const VulkanCGIterationPlan &plan) {
        return sparse_solve_result_to_dict(plan.get_last_result());
      })
      .def("_debug_runtime_stats",
           [sparse_solve_plan_stats_to_dict](
               const VulkanCGIterationPlan &plan) {
             return sparse_solve_plan_stats_to_dict(
                 plan.debug_runtime_statistics());
           });
  m.def("_make_vulkan_cg_iteration_plan",
        make_vulkan_cg_iteration_plan);
  m.def("_make_vulkan_cg_convergence_plan",
        make_vulkan_cg_convergence_plan, py::keep_alive<0, 1>(),
        py::keep_alive<0, 2>(), py::arg("program"), py::arg("matrix"),
        py::arg("max_iterations"), py::arg("absolute_tolerance"),
        py::arg("relative_tolerance") = 0.0f);
  m.def("_make_vulkan_jacobi_pcg_convergence_plan",
        make_vulkan_jacobi_pcg_convergence_plan,
        py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
        py::keep_alive<0, 3>(), py::arg("program"), py::arg("matrix"),
        py::arg("preconditioner"), py::arg("max_iterations"),
        py::arg("absolute_tolerance"),
        py::arg("relative_tolerance") = 0.0f);
  m.def("_make_vulkan_block_jacobi_pcg_convergence_plan",
        make_vulkan_block_jacobi_pcg_convergence_plan,
        py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
        py::keep_alive<0, 3>(), py::arg("program"), py::arg("matrix"),
        py::arg("preconditioner"), py::arg("max_iterations"),
        py::arg("absolute_tolerance"),
        py::arg("relative_tolerance") = 0.0f);
  m.def("_make_vulkan_compiled_kernel_cg_convergence_plan",
        make_vulkan_compiled_kernel_cg_convergence_plan,
        py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
        py::arg("program"), py::arg("matrix"),
        py::arg("max_iterations"), py::arg("absolute_tolerance"),
        py::arg("relative_tolerance") = 0.0f);
  m.def("_make_vulkan_compiled_graph_cg_convergence_plan",
        make_vulkan_compiled_graph_cg_convergence_plan,
        py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
        py::arg("program"), py::arg("matrix"),
        py::arg("max_iterations"), py::arg("absolute_tolerance"),
        py::arg("relative_tolerance") = 0.0f);
  m.def("_make_vulkan_compiled_kernel_pcg_convergence_plan",
        make_vulkan_compiled_kernel_pcg_convergence_plan,
        py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
        py::keep_alive<0, 3>(), py::arg("program"), py::arg("matrix"),
        py::arg("preconditioner"), py::arg("max_iterations"),
        py::arg("absolute_tolerance"),
        py::arg("relative_tolerance") = 0.0f);
  m.def("_make_vulkan_experimental_pcg_convergence_plan",
        make_vulkan_experimental_pcg_convergence_plan,
        py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
        py::keep_alive<0, 3>(), py::arg("program"), py::arg("matrix"),
        py::arg("preconditioner"), py::arg("max_iterations"),
        py::arg("absolute_tolerance"),
        py::arg("relative_tolerance") = 0.0f);

  // Mesh Class
  // Mesh related.
  py::enum_<mesh::MeshTopology>(m, "MeshTopology", py::arithmetic())
      .value("Triangle", mesh::MeshTopology::Triangle)
      .value("Tetrahedron", mesh::MeshTopology::Tetrahedron)
      .export_values();

  py::enum_<mesh::MeshElementType>(m, "MeshElementType", py::arithmetic())
      .value("Vertex", mesh::MeshElementType::Vertex)
      .value("Edge", mesh::MeshElementType::Edge)
      .value("Face", mesh::MeshElementType::Face)
      .value("Cell", mesh::MeshElementType::Cell)
      .export_values();

  py::enum_<mesh::MeshRelationType>(m, "MeshRelationType", py::arithmetic())
      .value("VV", mesh::MeshRelationType::VV)
      .value("VE", mesh::MeshRelationType::VE)
      .value("VF", mesh::MeshRelationType::VF)
      .value("VC", mesh::MeshRelationType::VC)
      .value("EV", mesh::MeshRelationType::EV)
      .value("EE", mesh::MeshRelationType::EE)
      .value("EF", mesh::MeshRelationType::EF)
      .value("EC", mesh::MeshRelationType::EC)
      .value("FV", mesh::MeshRelationType::FV)
      .value("FE", mesh::MeshRelationType::FE)
      .value("FF", mesh::MeshRelationType::FF)
      .value("FC", mesh::MeshRelationType::FC)
      .value("CV", mesh::MeshRelationType::CV)
      .value("CE", mesh::MeshRelationType::CE)
      .value("CF", mesh::MeshRelationType::CF)
      .value("CC", mesh::MeshRelationType::CC)
      .export_values();

  py::enum_<mesh::ConvType>(m, "ConvType", py::arithmetic())
      .value("l2g", mesh::ConvType::l2g)
      .value("l2r", mesh::ConvType::l2r)
      .value("g2r", mesh::ConvType::g2r)
      .export_values();

  py::class_<mesh::Mesh>(m, "Mesh");        // NOLINT(bugprone-unused-raii)
  py::class_<mesh::MeshPtr>(m, "MeshPtr");  // NOLINT(bugprone-unused-raii)

  m.def("element_order", mesh::element_order);
  m.def("from_end_element_order", mesh::from_end_element_order);
  m.def("to_end_element_order", mesh::to_end_element_order);
  m.def("relation_by_orders", mesh::relation_by_orders);
  m.def("inverse_relation", mesh::inverse_relation);
  m.def("element_type_name", mesh::element_type_name);

  m.def(
      "create_mesh",
      []() {
        auto mesh_shared = std::make_shared<mesh::Mesh>();
        mesh::MeshPtr mesh_ptr = mesh::MeshPtr{mesh_shared};
        return mesh_ptr;
      },
      py::return_value_policy::reference);

  // ad-hoc setters
  m.def("set_owned_offset",
        [](mesh::MeshPtr &mesh_ptr, mesh::MeshElementType type, SNode *snode) {
          mesh_ptr.ptr->owned_offset.insert(std::pair(type, snode));
        });
  m.def("set_total_offset",
        [](mesh::MeshPtr &mesh_ptr, mesh::MeshElementType type, SNode *snode) {
          mesh_ptr.ptr->total_offset.insert(std::pair(type, snode));
        });
  m.def("set_num_patches", [](mesh::MeshPtr &mesh_ptr, int num_patches) {
    mesh_ptr.ptr->num_patches = num_patches;
  });

  m.def("set_num_elements", [](mesh::MeshPtr &mesh_ptr,
                               mesh::MeshElementType type, int num_elements) {
    mesh_ptr.ptr->num_elements.insert(std::pair(type, num_elements));
  });

  m.def("get_num_elements",
        [](mesh::MeshPtr &mesh_ptr, mesh::MeshElementType type) {
          return mesh_ptr.ptr->num_elements.find(type)->second;
        });

  m.def("set_patch_max_element_num",
        [](mesh::MeshPtr &mesh_ptr, mesh::MeshElementType type,
           int max_element_num) {
          mesh_ptr.ptr->patch_max_element_num.insert(
              std::pair(type, max_element_num));
        });

  m.def("set_index_mapping",
        [](mesh::MeshPtr &mesh_ptr, mesh::MeshElementType element_type,
           mesh::ConvType conv_type, SNode *snode) {
          mesh_ptr.ptr->index_mapping.insert(
              std::make_pair(std::make_pair(element_type, conv_type), snode));
        });

  m.def("set_relation_fixed",
        [](mesh::MeshPtr &mesh_ptr, mesh::MeshRelationType type, SNode *value) {
          mesh_ptr.ptr->relations.insert(
              std::pair(type, mesh::MeshLocalRelation(value)));
        });

  m.def("set_relation_dynamic",
        [](mesh::MeshPtr &mesh_ptr, mesh::MeshRelationType type, SNode *value,
           SNode *patch_offset, SNode *offset) {
          mesh_ptr.ptr->relations.insert(std::pair(
              type, mesh::MeshLocalRelation(value, patch_offset, offset)));
        });

  m.def("wait_for_debugger", []() {
#ifdef WIN32
    while (!::IsDebuggerPresent())
      ::Sleep(100);
#endif
  });

  auto operationClass = py::class_<Operation>(m, "Operation");
  auto internalOpClass = py::class_<InternalOp>(m, "InternalOp");

#define PER_INTERNAL_OP(x)                                           \
  internalOpClass.def_property_readonly_static(                      \
      #x, [](py::object) { return Operations::get(InternalOp::x); }, \
      py::return_value_policy::reference);
#include "taichi/inc/internal_ops.inc.h"
#undef PER_INTERNAL_OP
}

}  // namespace taichi
