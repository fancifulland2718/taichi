#include "taichi/aot/graph_data.h"
#include "taichi/program/program.h"
#include "taichi/program/runtime_fault.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/storage_view.h"
#include "taichi/program/texture.h"
#include "taichi/program/kernel.h"
#include "taichi/program/matrix.h"
#include "taichi/system/profiler.h"
#include "taichi/ir/type_factory.h"
#include "taichi/util/environ_config.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <memory>
#include <numeric>
#include <optional>

#if defined(TI_WITH_LLVM)
#include "taichi/codegen/llvm/compiled_kernel_data.h"
#include "taichi/runtime/llvm/kernel_launcher.h"
#endif

#if defined(TI_WITH_CUDA)
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/rhi/cuda/primitives/graph_ptx.h"
#include "taichi/runtime/cuda/kernel_launcher.h"
#endif

#if defined(TI_WITH_VULKAN)
#include "taichi/runtime/gfx/kernel_launcher.h"
#endif

namespace taichi::lang {
namespace aot {

namespace {

std::vector<SNodeTreeDependency> collect_snode_tree_dependencies(
    const std::vector<CompiledDispatch> &dispatches) {
  std::vector<SNodeTreeDependency> dependencies;
  for (const auto &dispatch : dispatches) {
    dependencies.insert(dependencies.end(),
                        dispatch.snode_tree_dependencies.begin(),
                        dispatch.snode_tree_dependencies.end());
  }
  std::sort(dependencies.begin(), dependencies.end());
  dependencies.erase(
      std::unique(dependencies.begin(), dependencies.end()),
      dependencies.end());
  return dependencies;
}

Program *jit_graph_program(const CompiledGraph &graph) {
  Program *program = nullptr;
  for (const auto &dispatch : graph.dispatches) {
    if (dispatch.ti_kernel == nullptr) {
      if (dispatch.cuda_capture_command) {
        auto *provider_program = dispatch.cuda_capture_command->program();
        TI_ERROR_IF(provider_program == nullptr,
                    "CUDA capture command dispatch lost its Program");
        if (program == nullptr) {
          program = provider_program;
        } else {
          TI_ERROR_IF(program != provider_program,
                      "A JIT Graph cannot mix provider commands and kernels "
                      "from multiple Programs.");
        }
      }
      continue;
    }
    if (program == nullptr) {
      program = dispatch.ti_kernel->program;
    } else {
      TI_ERROR_IF(program != dispatch.ti_kernel->program,
                  "A JIT Graph cannot dispatch kernels from multiple "
                  "Programs.");
    }
  }
  return program;
}

template <typename T, std::size_t InlineCapacity = 4>
class InlineUniqueViewList {
 public:
  void add(const T *view) {
    const T *const *current = data();
    for (std::size_t i = 0; i < size(); ++i) {
      if (current[i] == view) {
        return;
      }
    }
    if (overflow_.empty() && inline_count_ < InlineCapacity) {
      inline_[inline_count_++] = view;
      return;
    }
    if (overflow_.empty()) {
      overflow_.reserve(InlineCapacity * 2);
      overflow_.insert(overflow_.end(), inline_.begin(),
                       inline_.begin() + inline_count_);
    }
    overflow_.push_back(view);
  }

  bool empty() const noexcept {
    return inline_count_ == 0 && overflow_.empty();
  }

  std::size_t size() const noexcept {
    return overflow_.empty() ? inline_count_ : overflow_.size();
  }

  const T *const *data() const noexcept {
    return overflow_.empty() ? inline_.data() : overflow_.data();
  }

 private:
  std::array<const T *, InlineCapacity> inline_{};
  std::size_t inline_count_{0};
  std::vector<const T *> overflow_;
};

struct GraphRuntimeResourceViews {
  InlineUniqueViewList<Ndarray> ndarrays;
  InlineUniqueViewList<storage::RuntimeStorageArgument> runtime_storage;
  InlineUniqueViewList<Texture> textures;

  bool empty() const noexcept {
    return ndarrays.empty() && runtime_storage.empty() && textures.empty();
  }
};

GraphRuntimeResourceViews graph_runtime_resource_views(
    const std::unordered_map<std::string, IValue> &args,
    Program *expected_program) {
  GraphRuntimeResourceViews views;
  for (const auto &[name, value] : args) {
    (void)name;
    if (value.tag == ArgKind::kNdarray) {
      if (value.runtime_storage != nullptr) {
        views.runtime_storage.add(value.runtime_storage);
        continue;
      }
      auto *view = reinterpret_cast<const Ndarray *>(value.val);
      TI_ERROR_IF(view == nullptr, "Graph received a null Ndarray runtime arg");
      Program *owner = view->owning_program();
      if (owner == nullptr) {
        // AOT DeviceAllocation-backed views preserve their existing external
        // ownership contract and never enter a Program registry.
        continue;
      }
      TI_ERROR_IF(expected_program != nullptr && owner != expected_program,
                  "Graph Ndarray runtime arguments must belong to the Graph's "
                  "Program");
      views.ndarrays.add(view);
      continue;
    }
    if (value.tag != ArgKind::kTexture) {
      continue;
    }
    auto *view = reinterpret_cast<const Texture *>(value.val);
    TI_ERROR_IF(view == nullptr, "Graph received a null Texture runtime arg");
    Program *owner = view->owning_program();
    if (owner == nullptr) {
      continue;
    }
    TI_ERROR_IF(expected_program != nullptr && owner != expected_program,
                "Graph Texture runtime arguments must belong to the Graph's "
                "Program");
    views.textures.add(view);
  }
  return views;
}

using ReplayClock = std::chrono::steady_clock;

uint64_t replay_elapsed_ns(ReplayClock::time_point begin) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          ReplayClock::now() - begin)
          .count());
}

bool graph_has_runtime_resource_declarations(
    const std::unordered_map<std::string, IValue> &args) {
  return std::any_of(args.begin(), args.end(), [](const auto &entry) {
    return entry.second.tag == ArgKind::kNdarray ||
           entry.second.tag == ArgKind::kTexture;
  });
}

bool graph_runtime_args_require_snode_guard(
    const std::unordered_map<std::string, IValue> &args) {
  for (const auto &[name, value] : args) {
    (void)name;
    if (value.tag != ArgKind::kNdarray || value.runtime_storage == nullptr) {
      continue;
    }
    if (value.runtime_storage->descriptor().owner().kind ==
        storage::StorageOwnerKind::kSNodePayload) {
      return true;
    }
  }
  return false;
}

template <typename T>
void append_unique_resource(std::vector<T *> &resources, T *resource) {
  if (std::find(resources.begin(), resources.end(), resource) ==
      resources.end()) {
    resources.push_back(resource);
  }
}

CompiledGraphRuntimeResourceIdentity runtime_resource_identity(
    const std::string &name,
    const IValue &value) {
  CompiledGraphRuntimeResourceIdentity identity;
  identity.name = name;
  if (value.tag == ArgKind::kNdarray && value.runtime_storage != nullptr) {
    identity.object = value.runtime_storage;
    const auto &owner = value.runtime_storage->descriptor().owner();
    if (owner.kind == storage::StorageOwnerKind::kProgramNdarray) {
      identity.handle = owner.ndarray_handle;
    }
  } else if (value.tag == ArgKind::kNdarray) {
    auto *array = reinterpret_cast<Ndarray *>(value.val);
    identity.object = array;
    if (array != nullptr) {
      identity.handle = array->runtime_resource_handle();
    }
  } else if (value.tag == ArgKind::kTexture) {
    auto *texture = reinterpret_cast<Texture *>(value.val);
    identity.object = texture;
    if (texture != nullptr) {
      identity.handle = texture->runtime_resource_handle();
    }
  }
  return identity;
}

bool runtime_binding_plan_matches(
    const CompiledGraphRuntimeBindingPlan &plan,
    const std::unordered_map<std::string, IValue> &args,
    Program *program) {
  if (!plan.initialized || plan.program != program) {
    return false;
  }
  std::size_t resource_count = 0;
  for (const auto &[name, value] : args) {
    if (value.tag != ArgKind::kNdarray && value.tag != ArgKind::kTexture) {
      continue;
    }
    ++resource_count;
    const auto expected = std::lower_bound(
        plan.identities.begin(), plan.identities.end(), name,
        [](const auto &identity, const std::string &candidate) {
          return identity.name < candidate;
        });
    if (expected == plan.identities.end() || expected->name != name) {
      return false;
    }
    const auto current = runtime_resource_identity({}, value);
    if (current.object != expected->object ||
        current.handle != expected->handle) {
      return false;
    }
  }
  return resource_count == plan.identities.size();
}

void rebuild_runtime_binding_plan(
    CompiledGraphRuntimeBindingPlan &plan,
    const std::unordered_map<std::string, IValue> &args,
    Program *program,
    uint64_t revision) {
  plan.clear();
  plan.program = program;
  plan.initialized = true;
  plan.revision = revision;
  for (const auto &[name, value] : args) {
    if (value.tag != ArgKind::kNdarray && value.tag != ArgKind::kTexture) {
      continue;
    }
    plan.identities.push_back(runtime_resource_identity(name, value));
    if (value.tag == ArgKind::kNdarray && value.runtime_storage != nullptr) {
      append_unique_resource(plan.runtime_storage, value.runtime_storage);
    } else if (value.tag == ArgKind::kNdarray) {
      auto *array = reinterpret_cast<Ndarray *>(value.val);
      if (array != nullptr && array->owning_program() != nullptr) {
        append_unique_resource(plan.ndarrays, array);
      }
    } else {
      auto *texture = reinterpret_cast<Texture *>(value.val);
      if (texture != nullptr && texture->owning_program() != nullptr) {
        append_unique_resource(plan.textures, texture);
      }
    }
  }
  std::sort(plan.identities.begin(), plan.identities.end(),
            [](const auto &lhs, const auto &rhs) {
              return lhs.name < rhs.name;
            });
}

const CompiledGraphRuntimeBindingPlan &prepare_runtime_binding_plan(
    CompiledGraphJITCache &cache,
    const std::unordered_map<std::string, IValue> &args,
    Program *program,
    bool attribute) {
  constexpr std::size_t kBindingPlanCapacity = 4;
  for (std::size_t index = 0; index < cache.runtime_binding_plans.size();
       ++index) {
    if (!runtime_binding_plan_matches(cache.runtime_binding_plans[index], args,
                                      program)) {
      continue;
    }
    if (attribute) {
      ++cache.replay_attribution.binding_plan_hits;
    }
    if (index != 0) {
      auto plan = std::move(cache.runtime_binding_plans[index]);
      cache.runtime_binding_plans.erase(cache.runtime_binding_plans.begin() +
                                        index);
      cache.runtime_binding_plans.insert(cache.runtime_binding_plans.begin(),
                                         std::move(plan));
    }
    return cache.runtime_binding_plans.front();
  }
  if (attribute) {
    ++cache.replay_attribution.binding_plan_misses;
  }
  uint64_t revision = cache.next_runtime_binding_plan_revision++;
  if (revision == 0) {
    revision = cache.next_runtime_binding_plan_revision++;
  }
  CompiledGraphRuntimeBindingPlan plan;
  rebuild_runtime_binding_plan(plan, args, program, revision);
  cache.runtime_binding_plans.insert(cache.runtime_binding_plans.begin(),
                                     std::move(plan));
  if (cache.runtime_binding_plans.size() > kBindingPlanCapacity) {
    cache.runtime_binding_plans.resize(kBindingPlanCapacity);
  }
  return cache.runtime_binding_plans.front();
}

const CompiledKernelData *get_or_compile_cached_kernel(
    const CompiledDispatch &dispatch,
    const CompileConfig &compile_config,
    CompiledGraphJITCachedKernel &cached,
    bool cache_compiled_kernel_data) {
  auto *prog = dispatch.ti_kernel->program;
  auto execution_handle = cached.execution_handle;
  if (execution_handle != nullptr && !execution_handle->active()) {
    execution_handle.reset();
    cached.execution_handle.reset();
  }
  if (execution_handle == nullptr && !cached.kernel_key.empty()) {
    execution_handle = prog->find_cached_kernel_execution_handle(
        compile_config, cached.kernel_key, *dispatch.ti_kernel);
  }
  if (execution_handle == nullptr) {
    execution_handle = prog->compile_kernel_execution_handle(
        compile_config, prog->get_device_caps(), *dispatch.ti_kernel);
    if (cached.kernel_key.empty()) {
      cached.kernel_key = dispatch.ti_kernel->get_cached_kernel_key();
    }
  }
  // All JIT Graph paths retain the stable handle. The historical flag is kept
  // at call sites for source compatibility while Vulkan transitions away from
  // raw registration pointers.
  (void)cache_compiled_kernel_data;
  cached.execution_handle = execution_handle;
  const auto *compiled_kernel_data = &execution_handle->compiled();
  if (cached.task_count == std::numeric_limits<std::uint32_t>::max()) {
    const std::size_t task_count = compiled_kernel_data->task_count();
    TI_ASSERT(task_count < std::numeric_limits<std::uint32_t>::max());
    cached.task_count = static_cast<std::uint32_t>(task_count);
  }
  return compiled_kernel_data;
}

std::vector<int> append_arg_index(const std::vector<int> &arg_id, int index) {
  std::vector<int> result = arg_id;
  result.push_back(index);
  return result;
}

DataType get_primitive_dtype(DataType dtype) {
  if (dtype->is<TensorType>()) {
    return dtype->cast<TensorType>()->get_element_type();
  }
  return dtype;
}

PrimitiveTypeID get_primitive_type_id(DataType dtype) {
  return get_primitive_dtype(dtype)->as<PrimitiveType>()->type;
}

void validate_graph_runtime_storage_argument(
    const storage::RuntimeStorageArgument &argument,
    const std::string &name,
    PrimitiveTypeID expected_dtype_id,
    std::size_t expected_index_rank,
    const std::vector<int> &expected_element_shape) {
  const auto &qualification = argument.qualification();
  TI_ERROR_IF(!qualification.capabilities.bindable ||
                  !qualification.capabilities.replayable ||
                  !qualification.capabilities.zero_copy_qualified,
              "Graph runtime storage argument {} is not replayable: {}", name,
              storage::to_string(qualification.reason));
  const auto &descriptor = argument.descriptor();
  const char *actual_storage =
      descriptor.source_kind() == storage::StorageSourceKind::kNdarray
          ? "an ndarray"
          : "dense storage";
  TI_ERROR_IF(descriptor.index_rank() != expected_index_rank,
              "Dispatch node is compiled for argument {} with field_dim={} "
              "but got {} with field_dim={}",
              name, expected_index_rank, actual_storage,
              descriptor.index_rank());
  const auto element_shape = descriptor.element_shape();
  TI_ERROR_IF(element_shape.size() != expected_element_shape.size(),
              "Mismatched element rank for Graph argument {}", name);
  for (std::size_t i = 0; i < element_shape.size(); ++i) {
    TI_ERROR_IF(element_shape[i] != expected_element_shape[i],
                "Mismatched element shape for Graph argument {}", name);
  }
  const PrimitiveTypeID actual_dtype_id =
      get_primitive_type_id(descriptor.scalar_type());
  TI_ERROR_IF(actual_dtype_id != expected_dtype_id,
              "Dispatch node is compiled for argument {} with dtype={} but "
              "got {} with dtype={}",
              name, PrimitiveType::get(expected_dtype_id).to_string(),
              actual_storage,
              PrimitiveType::get(actual_dtype_id).to_string());
}
template <typename T>
void write_arg_buffer(char *arg_buffer, int offset, uint64 value) {
  T typed_value = taichi_union_cast_with_different_sizes<T>(value);
  std::memcpy(arg_buffer + offset, &typed_value, sizeof(T));
}

void write_scalar_arg(char *arg_buffer,
                      const CompiledGraphRuntimeArgPlan &arg_plan,
                      uint64 value) {
  switch (arg_plan.type_size) {
    case 1:
      write_arg_buffer<int8>(arg_buffer, arg_plan.arg_buffer_offset, value);
      break;
    case 2:
      write_arg_buffer<int16>(arg_buffer, arg_plan.arg_buffer_offset, value);
      break;
    case 4:
      write_arg_buffer<int32>(arg_buffer, arg_plan.arg_buffer_offset, value);
      break;
    case 8:
      write_arg_buffer<int64>(arg_buffer, arg_plan.arg_buffer_offset, value);
      break;
    default:
      TI_ERROR("Unsupported type size {}", arg_plan.type_size);
  }
}

CompiledGraphDispatchRuntimePlan build_cpu_runtime_arg_plan(
    const CompiledDispatch &dispatch) {
  CompiledGraphDispatchRuntimePlan plan;
  plan.cpu_fast_path = true;
  if (dispatch.cpu_bounded_dispatch.has_value()) {
    plan.bounded_extent_name =
        dispatch.cpu_bounded_dispatch->extent_arg.name;
    plan.bounded_capacity = dispatch.cpu_bounded_dispatch->capacity;
  }
  plan.args.reserve(dispatch.symbolic_args.size());
  auto *args_type = dispatch.ti_kernel->args_type;
  for (int i = 0; i < dispatch.symbolic_args.size(); ++i) {
    const auto &symbolic_arg = dispatch.symbolic_args[i];
    CompiledGraphRuntimeArgPlan arg_plan;
    arg_plan.tag = symbolic_arg.tag;
    arg_plan.name = symbolic_arg.name;
    arg_plan.arg_id = {i};
    if (plan.bounded_extent_name.has_value() &&
        symbolic_arg.name == *plan.bounded_extent_name) {
      plan.bounded_extent_arg_id = arg_plan.arg_id;
    }
    arg_plan.arg_buffer_offset =
        args_type->get_element_offset(arg_plan.arg_id);
    arg_plan.dtype_id = get_primitive_type_id(symbolic_arg.dtype());
    arg_plan.field_dim = symbolic_arg.field_dim;
    arg_plan.element_shape = symbolic_arg.element_shape;

    if (symbolic_arg.tag == ArgKind::kScalar) {
      arg_plan.type_size = data_type_size(symbolic_arg.dtype());
      if (arg_plan.type_size != 1 && arg_plan.type_size != 2 &&
          arg_plan.type_size != 4 && arg_plan.type_size != 8) {
        plan.cpu_fast_path = false;
        plan.args.clear();
        return plan;
      }
    } else if (symbolic_arg.tag == ArgKind::kNdarray) {
      arg_plan.ndarray_data_ptr_key = append_arg_index(
          arg_plan.arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY);
      arg_plan.ndarray_grad_ptr_key = append_arg_index(
          arg_plan.arg_id, TypeFactory::GRAD_PTR_POS_IN_NDARRAY);
      arg_plan.ndarray_shape_offsets.reserve(symbolic_arg.field_dim);
      for (int j = 0; j < symbolic_arg.field_dim; ++j) {
        arg_plan.ndarray_shape_offsets.push_back(
            args_type->get_element_offset({i, 0, j}));
      }
    } else {
      plan.cpu_fast_path = false;
      plan.args.clear();
      return plan;
    }
    plan.args.push_back(std::move(arg_plan));
  }
  return plan;
}

void set_cuda_bounded_range_binding(const CompiledDispatch &dispatch,
                                    LaunchContextBuilder &ctx) {
  if (!dispatch.cuda_bounded_dispatch.has_value()) {
    return;
  }
  const auto &metadata = *dispatch.cuda_bounded_dispatch;
  for (int i = 0; i < dispatch.symbolic_args.size(); ++i) {
    const auto &arg = dispatch.symbolic_args[i];
    if (arg.name == metadata.extent_arg.name) {
      TI_ERROR_IF(arg.tag != ArgKind::kNdarray,
                  "CUDA exact bounded extent argument {} is not an Ndarray",
                  arg.name);
      ctx.set_cuda_bounded_range({i},
                                 static_cast<std::int32_t>(metadata.capacity));
      return;
    }
  }
  TI_ERROR("CUDA exact bounded extent argument {} is not bound by the "
           "dispatch",
           metadata.extent_arg.name);
}

bool init_runtime_context_from_plan(
    const CompiledGraphDispatchRuntimePlan &plan,
    const std::unordered_map<std::string, IValue> &args,
    Program *expected_program,
    LaunchContextBuilder &ctx) {
  TI_COMPILE_PROFILER("compiled_graph_init_runtime_context");
  char *arg_buffer = ctx.get_context().arg_buffer;
  void *bounded_extent = nullptr;
  bool bounded_extent_uses_runtime_storage = false;
  for (const auto &arg_plan : plan.args) {
    auto found = args.find(arg_plan.name);
    TI_ERROR_IF(found == args.end(), "Missing runtime value for {}",
                arg_plan.name);
    const IValue &ival = found->second;
    if (arg_plan.tag == ArgKind::kScalar) {
      TI_ASSERT(ival.tag == ArgKind::kScalar);
      write_scalar_arg(arg_buffer, arg_plan, ival.val);
      continue;
    }

    TI_ASSERT(arg_plan.tag == ArgKind::kNdarray);
    TI_ASSERT(ival.tag == ArgKind::kNdarray);
    if (ival.runtime_storage != nullptr) {
      validate_graph_runtime_storage_argument(
          *ival.runtime_storage, arg_plan.name, arg_plan.dtype_id,
          arg_plan.field_dim, arg_plan.element_shape);
      ctx.set_arg_runtime_storage(arg_plan.arg_id, *ival.runtime_storage);
      if (plan.bounded_extent_name.has_value() &&
          arg_plan.name == *plan.bounded_extent_name) {
        auto *arr = reinterpret_cast<Ndarray *>(ival.val);
        TI_ERROR_IF(arr == nullptr || arr->owning_program() != expected_program,
                    "CPU exact bounded dispatch requires a program-owned "
                    "DeviceExtent");
        TI_ERROR_IF(arr->get_nelement() < 2,
                    "CPU exact bounded DeviceExtent must contain two i32 words");
        bounded_extent_uses_runtime_storage = true;
      }
      continue;
    }
    Ndarray *arr = reinterpret_cast<Ndarray *>(ival.val);
    TI_ERROR_IF(arr == nullptr, "Graph received a null Ndarray runtime arg");
    if (plan.bounded_extent_name.has_value() &&
        arg_plan.name == *plan.bounded_extent_name) {
      TI_ERROR_IF(arr->owning_program() != expected_program,
                  "CPU exact bounded dispatch requires a program-owned "
                  "DeviceExtent");
    }
    TI_ERROR_IF(arr->get_element_shape() != arg_plan.element_shape,
                "Mismatched shape information for argument {}",
                arg_plan.name);
    TI_ERROR_IF(arr->shape.size() != arg_plan.field_dim,
                "Dispatch node is compiled for argument {} with field_dim={} "
                "but got an ndarray with field_dim={}",
                arg_plan.name, arg_plan.field_dim, arr->shape.size());

    DataType arr_primitive_dtype = get_primitive_dtype(arr->dtype);
    PrimitiveTypeID arr_dtype_id =
        arr_primitive_dtype->as<PrimitiveType>()->type;
    TI_ERROR_IF(arr_dtype_id != arg_plan.dtype_id,
                "Dispatch node is compiled for argument {} with dtype={} but "
                "got an ndarray with dtype={}",
                arg_plan.name, PrimitiveType::get(arg_plan.dtype_id).to_string(),
                arr_primitive_dtype.to_string());
    intptr_t ptr = arr->get_device_allocation_ptr_as_int();
    if (plan.bounded_extent_name.has_value() &&
        arg_plan.name == *plan.bounded_extent_name) {
      TI_ERROR_IF(arr->get_nelement() < 2,
                  "CPU exact bounded DeviceExtent must contain two i32 words");
      bounded_extent = reinterpret_cast<void *>(ptr);
    }
    ctx.array_ptrs[arg_plan.ndarray_data_ptr_key] = (void *)ptr;
    if (ptr != 0) {
      ctx.array_ptrs[arg_plan.ndarray_grad_ptr_key] = nullptr;
    }
    ctx.device_allocation_type[arg_plan.arg_id] =
        LaunchContextBuilder::DevAllocType::kNdarray;
    if (Program *owner = arr->owning_program()) {
      LaunchContextBuilder::NdarrayResourceRef ref;
      ref.arg_offset = arg_plan.arg_buffer_offset;
      ref.owner = owner;
      ref.data = arr;
      ref.data_handle = arr->runtime_resource_handle();
      TI_ERROR_IF(!ref.data_handle,
                  "Graph received an unregistered Ndarray runtime resource");
      ctx.ndarray_ptrs.push_back(std::move(ref));
    }
    size_t total_size = 1;
    for (int j = 0; j < arr->shape.size(); ++j) {
      int32 shape = (int32)arr->shape[j];
      std::memcpy(arg_buffer + arg_plan.ndarray_shape_offsets[j], &shape,
                  sizeof(shape));
      total_size *= arr->shape[j];
    }
    ctx.array_runtime_sizes[arg_plan.arg_id] = total_size;
  }
  if (plan.bounded_extent_name.has_value()) {
    TI_ERROR_IF(bounded_extent == nullptr &&
                    !bounded_extent_uses_runtime_storage,
                "CPU exact bounded DeviceExtent is unavailable");
    if (bounded_extent != nullptr) {
      ctx.set_cpu_bounded_range(
          bounded_extent, static_cast<std::int32_t>(plan.bounded_capacity));
    }
  }
  return bounded_extent_uses_runtime_storage;
}

#if defined(TI_WITH_LLVM)
bool try_launch_cached_llvm_kernel(Program *prog,
                                   const CompiledKernelData &compiled,
                                   CompiledGraphJITCachedKernel &cached,
                                   LaunchContextBuilder &launch_ctx,
                                   const CompiledGraphDispatchRuntimePlan *plan,
                                   bool bounded_extent_uses_runtime_storage) {
  auto *launcher =
      dynamic_cast<LLVM::KernelLauncher *>(&prog->get_kernel_launcher());
  auto *llvm_compiled =
      dynamic_cast<const LLVM::CompiledKernelData *>(&compiled);
  if (launcher == nullptr || llvm_compiled == nullptr) {
    return false;
  }
  if (cached.llvm_launch_id < 0) {
    auto handle = launcher->register_llvm_kernel(*llvm_compiled);
    cached.llvm_launch_id = handle.get_launch_id();
  }
  KernelLaunchHandle handle;
  handle.set_launch_id(cached.llvm_launch_id);
  TI_ASSERT(launch_ctx.argpack_ptrs.empty());
  if (!launch_ctx.ndarray_ptrs.empty()) {
    prog->resolve_ndarray_launch_context_under_guard(launch_ctx);
  }
  if (!launch_ctx.dense_storage_ptrs.empty()) {
    prog->resolve_runtime_storage_launch_context_under_guard(launch_ctx);
  }
  if (bounded_extent_uses_runtime_storage) {
    TI_ASSERT(plan != nullptr && plan->bounded_extent_name.has_value() &&
              !plan->bounded_extent_arg_id.empty());
    const auto &binding = launch_ctx.get_resolved_dense_storage(
        plan->bounded_extent_arg_id);
    auto *extent = reinterpret_cast<void *>(
        prog->get_dense_storage_data_ptr_as_int(binding));
    TI_ERROR_IF(extent == nullptr,
                "CPU exact bounded DeviceExtent resolved to a null address");
    launch_ctx.set_cpu_bounded_range(
        extent, static_cast<std::int32_t>(plan->bounded_capacity));
  }
  if (!launch_ctx.texture_ptrs.empty()) {
    prog->resolve_texture_launch_context_under_guard(launch_ctx);
  }
  // The surrounding Graph transaction already owns both the SNodeTree read
  // guard and, when needed, the runtime-resource submission guard.
  {
    TI_PROFILER("launch_llvm_kernel");
    launcher->launch_llvm_kernel(handle, launch_ctx);
  }
  prog->mark_runtime_submission();
  prog->check_runtime_error_after_kernel_launch(compiled);
  return true;
}
#endif

}  // namespace

CompiledGraph::CompiledGraph(
    std::vector<CompiledDispatch> compiled_dispatches)
    : dispatches(std::move(compiled_dispatches)),
      snode_tree_dependencies(
          collect_snode_tree_dependencies(dispatches)) {
}

CompiledGraph::CompiledGraph(
    std::vector<CompiledDispatch> compiled_dispatches,
    std::unordered_map<std::string, aot::Arg> graph_args)
    : dispatches(std::move(compiled_dispatches)),
      args(std::move(graph_args)),
      snode_tree_dependencies(
          collect_snode_tree_dependencies(dispatches)) {
}

bool CompiledGraph::has_indirect_dispatches() const {
  return std::any_of(
      dispatches.begin(), dispatches.end(), [](const auto &dispatch) {
        return dispatch.indirect_dispatch_arg.has_value();
      });
}

bool CompiledGraph::has_dispatch_labels() const {
  return std::any_of(dispatches.begin(), dispatches.end(),
                     [](const CompiledDispatch &dispatch) {
                       return !dispatch.dispatch_label.empty();
                     });
}

bool CompiledGraph::has_cuda_capture_commands() const {
  return std::any_of(
      dispatches.begin(), dispatches.end(), [](const auto &dispatch) {
        return dispatch.cuda_capture_command != nullptr;
      });
}

bool CompiledGraph::cuda_capture_commands_require_exact_bindings() const {
  return std::any_of(
      dispatches.begin(), dispatches.end(), [](const auto &dispatch) {
        return dispatch.cuda_capture_command != nullptr &&
               dispatch.cuda_capture_command->requires_exact_bindings();
      });
}

#if defined(TI_WITH_CUDA)
namespace {

struct CudaGraphArgSignatureEntry {
  std::string name;
  ArgKind tag{ArgKind::kUnknown};
  Device *device{nullptr};
  DeviceAllocationId alloc_id{0};
  uint64_t byte_offset{0};
  uint64_t byte_size{0};
  uint64_t runtime_signature{0};
  PrimitiveTypeID dtype_id{PrimitiveTypeID::unknown};
  ExternalArrayLayout layout{ExternalArrayLayout::kNull};
  std::vector<int> shape;
  std::vector<int> element_shape;
  uint64 value{0};
  std::vector<uint8_t> value_bytes;

  bool operator==(const CudaGraphArgSignatureEntry &other) const {
    return structurally_equals(other) && alloc_id == other.alloc_id &&
           byte_offset == other.byte_offset &&
           runtime_signature == other.runtime_signature &&
           value == other.value && value_bytes == other.value_bytes;
  }

  bool structurally_equals(const CudaGraphArgSignatureEntry &other) const {
    return name == other.name && tag == other.tag && device == other.device &&
           byte_size == other.byte_size && dtype_id == other.dtype_id &&
           layout == other.layout && shape == other.shape &&
           element_shape == other.element_shape;
  }
};

struct CudaGraphSignatureCandidate {
  std::vector<CudaGraphArgSignatureEntry> entries;
  std::vector<DeviceAllocation> allocations;
};

bool cuda_graph_signatures_are_structurally_compatible(
    const std::vector<CudaGraphArgSignatureEntry> &lhs,
    const std::vector<CudaGraphArgSignatureEntry> &rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (!lhs[i].structurally_equals(rhs[i])) {
      return false;
    }
  }
  return true;
}

class CudaGraphCaptureStream {
 public:
  CudaGraphCaptureStream() {
    CUDADriver::get_instance().stream_create(&stream_, CU_STREAM_NON_BLOCKING);
  }

  ~CudaGraphCaptureStream() {
    if (stream_ != nullptr) {
      CUDAContext::get_instance().make_current();
      CUDADriver::get_instance().stream_destroy(stream_);
    }
  }

  void *get() const {
    return stream_;
  }

 private:
  void *stream_{nullptr};
};

class CudaGraphHandle {
 public:
  CudaGraphHandle() = default;
  ~CudaGraphHandle() {
    reset();
  }
  CudaGraphHandle(const CudaGraphHandle &) = delete;
  CudaGraphHandle &operator=(const CudaGraphHandle &) = delete;

  CUgraph *put() {
    reset();
    return &graph_;
  }

  CUgraph get() const {
    return graph_;
  }

  explicit operator bool() const {
    return graph_ != nullptr;
  }

  void reset() {
    if (graph_ != nullptr) {
      CUDAContext::get_instance().make_current();
      CUDADriver::get_instance().graph_destroy(graph_);
      graph_ = nullptr;
    }
  }

 private:
  CUgraph graph_{nullptr};
};

class CudaGraphExecHandle {
 public:
  CudaGraphExecHandle() = default;
  ~CudaGraphExecHandle() {
    reset();
  }
  CudaGraphExecHandle(const CudaGraphExecHandle &) = delete;
  CudaGraphExecHandle &operator=(const CudaGraphExecHandle &) = delete;

  CUgraphExec *put() {
    reset();
    return &graph_exec_;
  }

  CUgraphExec get() const {
    return graph_exec_;
  }

  explicit operator bool() const {
    return graph_exec_ != nullptr;
  }

  void reset() {
    if (graph_exec_ != nullptr) {
      auto &context = CUDAContext::get_instance();
      context.make_current();
      // Replay stays on the default stream to preserve the ordering contract
      // visible to ordinary CUDA launches.
      auto &driver = CUDADriver::get_instance();
      driver.stream_synchronize(nullptr);
      driver.graph_exec_destroy(graph_exec_);
      graph_exec_ = nullptr;
    }
  }

 private:
  CUgraphExec graph_exec_{nullptr};
};

class CudaStreamCaptureGuard {
 public:
  explicit CudaStreamCaptureGuard(void *stream) : stream_(stream) {
  }
  ~CudaStreamCaptureGuard() {
    abort();
  }
  CudaStreamCaptureGuard(const CudaStreamCaptureGuard &) = delete;
  CudaStreamCaptureGuard &operator=(const CudaStreamCaptureGuard &) = delete;

  uint32_t end(CUgraph *graph) {
    active_ = false;
    return CUDADriver::get_instance().stream_end_capture.call(stream_, graph);
  }

  void abort() noexcept {
    if (!active_) {
      return;
    }
    active_ = false;
    CUgraph graph = nullptr;
    auto &driver = CUDADriver::get_instance();
    const uint32_t result =
        driver.stream_end_capture.call(stream_, &graph);
    if (result == CUDA_SUCCESS && graph != nullptr) {
      driver.graph_destroy.call(graph);
    }
  }

 private:
  void *stream_{nullptr};
  bool active_{true};
};

class CudaGraphCapturePacket {
 public:
  explicit CudaGraphCapturePacket(void *stream) : stream_(stream) {
  }
  ~CudaGraphCapturePacket() {
    release_and_wait();
  }
  CudaGraphCapturePacket(const CudaGraphCapturePacket &) = delete;
  CudaGraphCapturePacket &operator=(const CudaGraphCapturePacket &) = delete;
  CudaGraphCapturePacket(CudaGraphCapturePacket &&other) noexcept
      : launcher(std::exchange(other.launcher, nullptr)),
        packet(std::move(other.packet)),
        stream_(std::exchange(other.stream_, nullptr)) {
    other.packet.device_arg_buffer = nullptr;
  }
  CudaGraphCapturePacket &operator=(
      CudaGraphCapturePacket &&other) noexcept {
    if (this != &other) {
      release_and_wait();
      launcher = std::exchange(other.launcher, nullptr);
      packet = std::move(other.packet);
      stream_ = std::exchange(other.stream_, nullptr);
      other.packet.device_arg_buffer = nullptr;
    }
    return *this;
  }

  bool retire_argument_buffer() {
    if (packet.device_arg_buffer == nullptr) {
      return false;
    }
    auto &context = CUDAContext::get_instance();
    context.make_current();
    auto &driver = CUDADriver::get_instance();
    bool stream_ordered_free = false;
    if (context.supports_mem_pool()) {
      driver.mem_free_async_impl(packet.device_arg_buffer, stream_);
      stream_ordered_free = true;
    } else {
      driver.mem_free(packet.device_arg_buffer);
    }
    packet.device_arg_buffer = nullptr;
    return stream_ordered_free;
  }

  cuda::KernelLauncher *launcher{nullptr};
  cuda::KernelLauncher::GraphLaunchPacket packet;

 private:
  void release_and_wait() {
    if (retire_argument_buffer()) {
      CUDADriver::get_instance().stream_synchronize(stream_);
    }
  }

  void *stream_{nullptr};
};

class CudaEventHandle {
 public:
  CudaEventHandle() {
    CUDADriver::get_instance().event_create(&event_,
                                            CU_EVENT_DISABLE_TIMING);
  }
  ~CudaEventHandle() {
    reset();
  }
  CudaEventHandle(const CudaEventHandle &) = delete;
  CudaEventHandle &operator=(const CudaEventHandle &) = delete;
  CudaEventHandle(CudaEventHandle &&other) noexcept
      : event_(std::exchange(other.event_, nullptr)),
        recorded_(std::exchange(other.recorded_, false)) {
  }
  CudaEventHandle &operator=(CudaEventHandle &&other) noexcept {
    if (this != &other) {
      reset();
      event_ = std::exchange(other.event_, nullptr);
      recorded_ = std::exchange(other.recorded_, false);
    }
    return *this;
  }

  void record(void *stream) {
    CUDADriver::get_instance().event_record(event_, stream);
    recorded_ = true;
  }

  void *get() const {
    return event_;
  }

  // Capture-only fork/join events are retained by the instantiated Graph.
  // Once graph_exec retirement has synchronized the replay stream, no event
  // operation remains outstanding and destruction must not wait again.
  void mark_complete() {
    recorded_ = false;
  }

  bool ready() const {
    if (!recorded_) {
      return true;
    }
    auto &driver = CUDADriver::get_instance();
    const uint32_t result = driver.event_query.call(event_);
    if (result == CUDA_SUCCESS) {
      return true;
    }
    if (result == CUDA_ERROR_NOT_READY) {
      return false;
    }
    TI_ERROR("{}", driver.event_query.get_error_message(result));
  }

  void wait() {
    if (recorded_) {
      CUDADriver::get_instance().event_synchronize(event_);
      recorded_ = false;
    }
  }

 private:
  void reset() {
    if (event_ != nullptr) {
      wait();
      CUDADriver::get_instance().event_destroy(event_);
      event_ = nullptr;
    }
  }

  void *event_{nullptr};
  bool recorded_{false};
};

struct CudaDeferredReplayResources {
  CudaDeferredReplayResources(
      std::vector<std::unique_ptr<cuda::CudaDevice::AllocationLease>> leases,
      std::vector<std::vector<uint8_t>> host_buffers,
      void *stream = nullptr)
      : allocation_leases(std::move(leases)),
        host_arg_buffers(std::move(host_buffers)) {
    ready_event.record(stream);
  }
  CudaDeferredReplayResources(
      CudaEventHandle event,
      std::vector<std::unique_ptr<cuda::CudaDevice::AllocationLease>> leases,
      std::vector<std::vector<uint8_t>> host_buffers,
      void *stream = nullptr)
      : ready_event(std::move(event)),
        allocation_leases(std::move(leases)),
        host_arg_buffers(std::move(host_buffers)) {
    ready_event.record(stream);
  }
  ~CudaDeferredReplayResources() {
    // Host patch buffers and old allocation leases remain valid until every
    // preceding default-stream replay and the patch that replaced them has
    // completed.
    ready_event.wait();
  }
  CudaDeferredReplayResources(const CudaDeferredReplayResources &) = delete;
  CudaDeferredReplayResources &operator=(
      const CudaDeferredReplayResources &) = delete;
  CudaDeferredReplayResources(CudaDeferredReplayResources &&) noexcept =
      default;
  CudaDeferredReplayResources &operator=(
      CudaDeferredReplayResources &&) noexcept = default;

  CudaEventHandle ready_event;
  std::vector<std::unique_ptr<cuda::CudaDevice::AllocationLease>>
      allocation_leases;
  std::vector<std::vector<uint8_t>> host_arg_buffers;
};

struct CudaGraphBoundedDispatchControl {
  cuda::CudaGraphBoundedExtentControl *device_control{nullptr};
  cuda::CudaGraphBoundedExtentControl host_control;
};

struct CudaGraphBoundedDispatchGroup {
  cuda::CudaGraphBoundedGroupControl *device_control{nullptr};
  std::uintptr_t *device_nodes{nullptr};
  cuda::CudaGraphBoundedGroupControl host_control;
  std::vector<std::uintptr_t> host_nodes;
  std::vector<std::size_t> dispatch_indices;
};

struct CudaGraphBoundedDispatchObservation {
  Arg extent_arg;
  std::uint32_t capacity{0};
  std::uint32_t block_dim{0};
  std::uint32_t baseline_grid_dim{0};
  bool adaptive_grid{false};
};

}  // namespace

struct CompiledGraphCudaState {
  std::vector<CudaGraphArgSignatureEntry> signature;
  std::vector<DeviceAllocation> allocations;
  std::vector<std::unique_ptr<cuda::CudaDevice::AllocationLease>>
      allocation_leases;
  std::unique_ptr<CudaGraphCaptureStream> capture_stream;
  std::vector<std::unique_ptr<CudaGraphCaptureStream>>
      parallel_capture_streams;
  std::vector<CudaEventHandle> parallel_capture_events;
  std::vector<CudaGraphCapturePacket> packets;
  std::vector<std::shared_ptr<void>> provider_capture_resources;
  std::vector<CudaGraphBoundedDispatchControl> bounded_dispatch_controls;
  std::vector<CudaGraphBoundedDispatchGroup> bounded_dispatch_groups;
  // Host-only immutable recipes for opt-in physical launch observation.
  // They are materialized with the Graph and never touched by replay.
  std::vector<CudaGraphBoundedDispatchObservation>
      bounded_dispatch_observations;
  std::vector<std::int32_t> bounded_dispatch_group_indices;
  std::vector<std::int32_t> bounded_dispatch_group_member_indices;
  CudaGraphExecHandle graph_exec;
  cuda::CudaGraphConditionalControl *conditional_control{nullptr};
  std::uint64_t conditional_handle{0};
  bool conditional_mode{false};
  void *masked_gate{nullptr};
  void *masked_inner_gate{nullptr};
  bool masked_mode{false};
  bool masked_nested_mode{false};
  bool device_update_nested_mode{false};
  cuda::CudaGraphPredicateGroupControl *nested_device_controls{nullptr};
  std::uintptr_t *nested_device_nodes{nullptr};
  std::vector<cuda::CudaGraphPredicateGroupControl> nested_host_controls;
  std::vector<std::uintptr_t> nested_host_nodes;
  int masked_control_type{-1};
  int masked_max_iterations{0};
  bool masked_continue_while_nonzero{true};
  int masked_default_branch{-1};
  std::vector<int> masked_branch_dispatch_counts;
  DeviceAllocation masked_selector_allocation{kDeviceNullAllocation};
  DeviceAllocation masked_inner_selector_allocation{kDeviceNullAllocation};
  std::array<std::size_t, 4> masked_nested_boundaries{};
  int masked_nested_outer_max_iterations{0};
  int masked_nested_inner_max_iterations{0};
  std::vector<void *> nested_inner_gates;
  std::vector<DeviceAllocation> nested_inner_selector_allocations;
  std::vector<std::array<std::size_t, 3>> nested_inner_boundaries;
  std::vector<int> nested_inner_max_iterations;
  int conditional_type{-1};
  int conditional_default_branch{-1};
  std::vector<int> conditional_branch_dispatch_counts;
  std::vector<CudaDeferredReplayResources> deferred_resources;
  std::vector<CudaEventHandle> reusable_events;
  CompiledGraphCaptureRetryState retry;
  CompiledGraphCaptureRetryState masked_retry;
  CompiledGraphStats stats;
  bool diagnostics_enabled{false};
  bool has_captured_once{false};

  static constexpr std::size_t kMaxDeferredReplayBatches = 2;

  ~CompiledGraphCudaState() {
    retire();
  }

  void *ensure_capture_stream() {
    if (capture_stream == nullptr) {
      capture_stream = std::make_unique<CudaGraphCaptureStream>();
    }
    return capture_stream->get();
  }

  void ensure_parallel_capture_resources(std::size_t branch_count) {
    if (parallel_capture_streams.size() != branch_count) {
      parallel_capture_streams.clear();
      parallel_capture_streams.reserve(branch_count);
      for (std::size_t i = 0; i < branch_count; ++i) {
        parallel_capture_streams.push_back(
            std::make_unique<CudaGraphCaptureStream>());
      }
    }
    // One fork event plus one tail event per branch.
    if (parallel_capture_events.size() != branch_count + 1) {
      parallel_capture_events.clear();
      parallel_capture_events.resize(branch_count + 1);
    }
  }

  void recycle_front_deferred_resources() {
    deferred_resources.front().ready_event.wait();
    reusable_events.push_back(
        std::move(deferred_resources.front().ready_event));
    deferred_resources.erase(deferred_resources.begin());
  }

  void collect_ready_deferred_resources() {
    while (!deferred_resources.empty() &&
           deferred_resources.front().ready_event.ready()) {
      recycle_front_deferred_resources();
    }
  }

  void defer_replay_resources(
      std::vector<std::unique_ptr<cuda::CudaDevice::AllocationLease>> leases,
      std::vector<std::vector<uint8_t>> host_buffers,
      void *stream = nullptr) {
    collect_ready_deferred_resources();
    if (deferred_resources.size() >= kMaxDeferredReplayBatches) {
      if (diagnostics_enabled) {
        ++stats.deferred_replay_waits;
      }
      recycle_front_deferred_resources();
    }
    if (reusable_events.empty()) {
      deferred_resources.emplace_back(std::move(leases),
                                      std::move(host_buffers), stream);
    } else {
      CudaEventHandle event = std::move(reusable_events.back());
      reusable_events.pop_back();
      deferred_resources.emplace_back(std::move(event), std::move(leases),
                                      std::move(host_buffers), stream);
    }
    if (diagnostics_enabled) {
      stats.peak_deferred_replay_batches = std::max<std::uint64_t>(
          stats.peak_deferred_replay_batches, deferred_resources.size());
    }
  }

  void retire() {
    auto &context = CUDAContext::get_instance();
    context.make_current();
    graph_exec.reset();
    for (auto &event : parallel_capture_events) {
      event.mark_complete();
    }
    if (conditional_control != nullptr) {
      CUDADriver::get_instance().mem_free(conditional_control);
      conditional_control = nullptr;
    }
    if (masked_gate != nullptr) {
      CUDADriver::get_instance().mem_free(masked_gate);
      masked_gate = nullptr;
    }
    if (masked_inner_gate != nullptr) {
      CUDADriver::get_instance().mem_free(masked_inner_gate);
      masked_inner_gate = nullptr;
    }
    for (void *gate : nested_inner_gates) {
      if (gate != nullptr) {
        CUDADriver::get_instance().mem_free(gate);
      }
    }
    nested_inner_gates.clear();
    if (nested_device_controls != nullptr) {
      CUDADriver::get_instance().mem_free(nested_device_controls);
      nested_device_controls = nullptr;
    }
    if (nested_device_nodes != nullptr) {
      CUDADriver::get_instance().mem_free(nested_device_nodes);
      nested_device_nodes = nullptr;
    }
    nested_host_controls.clear();
    nested_host_nodes.clear();
    for (auto &control : bounded_dispatch_controls) {
      if (control.device_control != nullptr) {
        CUDADriver::get_instance().mem_free(control.device_control);
        control.device_control = nullptr;
      }
    }
    bounded_dispatch_controls.clear();
    for (auto &group : bounded_dispatch_groups) {
      if (group.device_control != nullptr) {
        CUDADriver::get_instance().mem_free(group.device_control);
        group.device_control = nullptr;
      }
      if (group.device_nodes != nullptr) {
        CUDADriver::get_instance().mem_free(group.device_nodes);
        group.device_nodes = nullptr;
      }
    }
    bounded_dispatch_groups.clear();
    bounded_dispatch_group_indices.clear();
    bounded_dispatch_group_member_indices.clear();
    conditional_handle = 0;
    conditional_mode = false;
    masked_mode = false;
    masked_nested_mode = false;
    device_update_nested_mode = false;
    masked_control_type = -1;
    masked_max_iterations = 0;
    masked_continue_while_nonzero = true;
    masked_inner_selector_allocation = kDeviceNullAllocation;
    masked_nested_boundaries = {};
    masked_nested_outer_max_iterations = 0;
    masked_nested_inner_max_iterations = 0;
    nested_inner_selector_allocations.clear();
    nested_inner_boundaries.clear();
    nested_inner_max_iterations.clear();
    masked_default_branch = -1;
    masked_branch_dispatch_counts.clear();
    masked_selector_allocation = kDeviceNullAllocation;
    conditional_type = -1;
    conditional_default_branch = -1;
    conditional_branch_dispatch_counts.clear();
    void *stream = capture_stream == nullptr ? nullptr : capture_stream->get();
    bool stream_ordered_free = false;
    for (auto &packet : packets) {
      stream_ordered_free =
          packet.retire_argument_buffer() || stream_ordered_free;
    }
    if (stream_ordered_free) {
      CUDADriver::get_instance().stream_synchronize(stream);
    }
    packets.clear();
    provider_capture_resources.clear();
    // graph_exec.reset() synchronized the default stream, so all deferred
    // events are ready. Recycle their handles for a later recapture/patch.
    while (!deferred_resources.empty()) {
      recycle_front_deferred_resources();
    }
    signature.clear();
    allocations.clear();
    // A lease may be the last owner preventing an Ndarray allocation retired
    // by Python GC from being released. Drop it only after all replay and
    // capture-owned argument buffers that contain the address are retired.
    allocation_leases.clear();
  }

  uint64_t known_bounded_control_bytes() const {
    uint64_t bytes = std::count_if(bounded_dispatch_controls.begin(),
                                   bounded_dispatch_controls.end(),
                                   [](const auto &control) {
                                     return control.device_control != nullptr;
                                   }) *
                     sizeof(cuda::CudaGraphBoundedExtentControl);
    for (const auto &group : bounded_dispatch_groups) {
      if (group.device_control != nullptr) {
        bytes += sizeof(cuda::CudaGraphBoundedGroupControl);
      }
      if (group.device_nodes != nullptr) {
        bytes += group.host_nodes.size() * sizeof(std::uintptr_t);
      }
    }
    if (nested_device_controls != nullptr) {
      bytes += nested_host_controls.size() *
               sizeof(cuda::CudaGraphPredicateGroupControl);
    }
    if (nested_device_nodes != nullptr) {
      bytes += nested_host_nodes.size() * sizeof(std::uintptr_t);
    }
    return bytes;
  }

  uint64_t known_persistent_argument_bytes() const {
    uint64_t bytes = conditional_control == nullptr
                         ? 0
                         : sizeof(cuda::CudaGraphConditionalControl);
    if (masked_gate != nullptr) {
      bytes += sizeof(std::uint32_t);
    }
    if (masked_inner_gate != nullptr) {
      bytes += sizeof(std::uint32_t);
    }
    bytes += known_bounded_control_bytes();
    for (const auto &packet : packets) {
      bytes += packet.packet.device_arg_buffer_size;
    }
    for (const auto &batch : deferred_resources) {
      for (const auto &buffer : batch.host_arg_buffers) {
        bytes += buffer.size();
      }
    }
    return bytes;
  }
};

void CompiledGraphCudaStateDeleter::operator()(
    CompiledGraphCudaState *state) const noexcept {
  delete state;
}

namespace {

CompiledGraphCudaState *get_cuda_graph_state(CompiledGraphJITCache &cache) {
  if (!cache.cuda_graph_state) {
    cache.cuda_graph_state.reset(new CompiledGraphCudaState());
  }
  if (cache.graph_diagnostics_enabled) {
    cache.cuda_graph_state->diagnostics_enabled = true;
  }
  return cache.cuda_graph_state.get();
}

std::optional<CudaGraphSignatureCandidate>
make_cuda_graph_signature(const CompiledGraph &graph,
                          Program &program,
                          const std::unordered_map<std::string, IValue> &args) {
  CudaGraphSignatureCandidate signature;
  signature.entries.reserve(args.size());
  signature.allocations.reserve(args.size());
  for (const auto &kv : args) {
    const auto declared_it = graph.args.find(kv.first);
    if (declared_it == graph.args.end() ||
        declared_it->second.tag != kv.second.tag) {
      return std::nullopt;
    }

    CudaGraphArgSignatureEntry entry;
    entry.name = kv.first;
    entry.tag = kv.second.tag;
    entry.dtype_id = declared_it->second.dtype_id;
    entry.element_shape = declared_it->second.element_shape;

    if (kv.second.tag == ArgKind::kNdarray) {
      DeviceAllocation allocation = kDeviceNullAllocation;
      if (kv.second.runtime_storage != nullptr) {
        const auto &argument = *kv.second.runtime_storage;
        const auto &qualification = argument.qualification();
        const auto &descriptor = argument.descriptor();
        const auto owner_kind = descriptor.owner().kind;
        if (!qualification.capabilities.capturable ||
            (owner_kind != storage::StorageOwnerKind::kProgramNdarray &&
             owner_kind != storage::StorageOwnerKind::kSNodePayload) ||
            argument.synchronization_domain_identity() != 0) {
          return std::nullopt;
        }
        const auto binding =
            program.resolve_runtime_storage_argument_under_graph_guard(
                argument);
        if (!binding.valid || binding.allocation == kDeviceNullAllocation) {
          return std::nullopt;
        }
        allocation = binding.allocation;
        entry.device = allocation.device;
        entry.alloc_id = allocation.alloc_id;
        entry.byte_offset = binding.byte_offset;
        entry.byte_size = binding.byte_size;
        entry.runtime_signature = binding.runtime_signature;
        entry.dtype_id =
            descriptor.scalar_type()->as<PrimitiveType>()->type;
        entry.shape.reserve(descriptor.index_rank());
        for (std::size_t axis = 0; axis < descriptor.index_rank(); ++axis) {
          const std::int64_t extent = descriptor.index_extent(axis);
          if (extent < 0 || extent > (std::numeric_limits<int>::max)()) {
            return std::nullopt;
          }
          entry.shape.push_back(static_cast<int>(extent));
        }
        entry.element_shape.reserve(descriptor.element_rank());
        for (std::size_t axis = 0; axis < descriptor.element_rank(); ++axis) {
          const std::int64_t extent = descriptor.element_extent(axis);
          if (extent < 0 || extent > (std::numeric_limits<int>::max)()) {
            return std::nullopt;
          }
          entry.element_shape.push_back(static_cast<int>(extent));
        }
        switch (descriptor.properties().array_layout) {
          case storage::StorageArrayLayout::kScalar:
            entry.layout = ExternalArrayLayout::kNull;
            break;
          case storage::StorageArrayLayout::kAos:
            entry.layout = ExternalArrayLayout::kAOS;
            break;
          case storage::StorageArrayLayout::kSoa:
            entry.layout = ExternalArrayLayout::kSOA;
            break;
          case storage::StorageArrayLayout::kNone:
            return std::nullopt;
        }
      } else {
        auto *arr = reinterpret_cast<Ndarray *>(kv.second.val);
        if (arr == nullptr) {
          return std::nullopt;
        }
        allocation = arr->get_device_allocation();
        entry.device = allocation.device;
        entry.alloc_id = allocation.alloc_id;
        entry.byte_offset = 0;
        entry.byte_size = arr->get_nelement() * arr->get_element_size();
        entry.dtype_id = arr->get_element_data_type()
                             ->as<PrimitiveType>()
                             ->type;
        entry.layout = arr->layout;
        entry.shape = arr->shape;
        entry.element_shape = arr->get_element_shape();
      }
      auto *device = dynamic_cast<cuda::CudaDevice *>(allocation.device);
      if (device == nullptr) {
        return std::nullopt;
      }
      const bool already_listed =
          std::find(signature.allocations.begin(),
                    signature.allocations.end(),
                    allocation) != signature.allocations.end();
      if (!already_listed) {
        signature.allocations.push_back(allocation);
      }
    } else if (kv.second.tag == ArgKind::kScalar) {
      entry.byte_size = data_type_size(declared_it->second.dtype());
      entry.value = kv.second.val;
    } else if (kv.second.tag == ArgKind::kMatrix) {
      auto *matrix = reinterpret_cast<Matrix *>(kv.second.val);
      entry.byte_size = matrix->length() * data_type_size(matrix->dtype());
      if (entry.byte_size > 128) {
        return std::nullopt;
      }
      entry.value_bytes.resize(entry.byte_size);
      std::memcpy(entry.value_bytes.data(),
                  reinterpret_cast<const void *>(matrix->data()),
                  entry.byte_size);
    } else {
      // Texture handles require a backend-specific lifetime owner and are not
      // eligible for CUDA argument-buffer patching yet.
      return std::nullopt;
    }
    signature.entries.push_back(std::move(entry));
  }
  std::sort(signature.entries.begin(), signature.entries.end(),
            [](const CudaGraphArgSignatureEntry &lhs,
               const CudaGraphArgSignatureEntry &rhs) {
              return lhs.name < rhs.name;
            });
  std::sort(signature.allocations.begin(), signature.allocations.end(),
            [](const DeviceAllocation &lhs, const DeviceAllocation &rhs) {
              const auto lhs_device =
                  reinterpret_cast<std::uintptr_t>(lhs.device);
              const auto rhs_device =
                  reinterpret_cast<std::uintptr_t>(rhs.device);
              return lhs_device == rhs_device ? lhs.alloc_id < rhs.alloc_id
                                              : lhs_device < rhs_device;
            });
  return signature;
}

bool cuda_graph_arguments_match_cached_signature(
    const CompiledGraph &graph,
    Program &program,
    const std::unordered_map<std::string, IValue> &args,
    const std::vector<CudaGraphArgSignatureEntry> &signature) {
  if (args.size() != signature.size()) {
    return false;
  }
  for (const auto &entry : signature) {
    const auto value_it = args.find(entry.name);
    const auto declared_it = graph.args.find(entry.name);
    if (value_it == args.end() || declared_it == graph.args.end() ||
        value_it->second.tag != entry.tag ||
        declared_it->second.tag != entry.tag) {
      return false;
    }
    const IValue &value = value_it->second;
    if (entry.tag == ArgKind::kNdarray) {
      DeviceAllocation allocation = kDeviceNullAllocation;
      uint64_t byte_offset = 0;
      uint64_t byte_size = 0;
      uint64_t runtime_signature = 0;
      PrimitiveTypeID dtype_id = PrimitiveTypeID::unknown;
      ExternalArrayLayout layout = ExternalArrayLayout::kNull;
      if (value.runtime_storage != nullptr) {
        const auto &argument = *value.runtime_storage;
        const auto &descriptor = argument.descriptor();
        const auto binding =
            program.resolve_runtime_storage_argument_under_graph_guard(
                argument);
        if (!binding.valid) {
          return false;
        }
        allocation = binding.allocation;
        byte_offset = binding.byte_offset;
        byte_size = binding.byte_size;
        runtime_signature = binding.runtime_signature;
        dtype_id = descriptor.scalar_type()->as<PrimitiveType>()->type;
        switch (descriptor.properties().array_layout) {
          case storage::StorageArrayLayout::kScalar:
            layout = ExternalArrayLayout::kNull;
            break;
          case storage::StorageArrayLayout::kAos:
            layout = ExternalArrayLayout::kAOS;
            break;
          case storage::StorageArrayLayout::kSoa:
            layout = ExternalArrayLayout::kSOA;
            break;
          case storage::StorageArrayLayout::kNone:
            return false;
        }
      } else {
        auto *array = reinterpret_cast<Ndarray *>(value.val);
        if (array == nullptr) {
          return false;
        }
        allocation = array->get_device_allocation();
        byte_size = array->get_nelement() * array->get_element_size();
        dtype_id =
            array->get_element_data_type()->as<PrimitiveType>()->type;
        layout = array->layout;
      }
      if (allocation.device != entry.device ||
          allocation.alloc_id != entry.alloc_id ||
          byte_offset != entry.byte_offset || byte_size != entry.byte_size ||
          runtime_signature != entry.runtime_signature ||
          dtype_id != entry.dtype_id || layout != entry.layout) {
        return false;
      }
    } else if (entry.tag == ArgKind::kScalar) {
      if (value.val != entry.value) {
        return false;
      }
    } else if (entry.tag == ArgKind::kMatrix) {
      auto *matrix = reinterpret_cast<Matrix *>(value.val);
      if (matrix == nullptr ||
          matrix->length() * data_type_size(matrix->dtype()) !=
              entry.value_bytes.size() ||
          std::memcmp(reinterpret_cast<const void *>(matrix->data()),
                      entry.value_bytes.data(), entry.value_bytes.size()) !=
              0) {
        return false;
      }
    } else {
      return false;
    }
  }
  return true;
}

bool cuda_graph_signatures_share_replay_binding(
    const std::vector<CudaGraphArgSignatureEntry> &lhs,
    const std::vector<CudaGraphArgSignatureEntry> &rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (!lhs[i].structurally_equals(rhs[i])) {
      return false;
    }
    // Executable slots separate stable resource bindings. Scalar and matrix
    // values remain patchable launch data and must not consume another heavy
    // CUgraphExec merely because a coefficient changed.
    if (lhs[i].tag == ArgKind::kNdarray && !(lhs[i] == rhs[i])) {
      return false;
    }
  }
  return true;
}

template <typename TopologyMatches>
CompiledGraphCudaState *select_cuda_graph_replay_slot(
    CompiledGraphJITCache &cache,
    const CompiledGraph &graph,
    Program &program,
    const std::unordered_map<std::string, IValue> &args,
    bool &exact_match,
    TopologyMatches &&topology_matches) {
  exact_match = false;
  auto *state = get_cuda_graph_state(cache);
  if (state->graph_exec && topology_matches(*state)) {
    try {
      exact_match = cuda_graph_arguments_match_cached_signature(
          graph, program, args, state->signature);
    } catch (...) {
      state->retire();
      throw;
    }
    if (exact_match) {
      return state;
    }
  }
  for (std::size_t index = 0;
       index < cache.cuda_graph_state_alternates.size(); ++index) {
    auto &candidate = cache.cuda_graph_state_alternates[index];
    if (!candidate || !candidate->graph_exec ||
        !topology_matches(*candidate)) {
      continue;
    }
    bool candidate_matches = false;
    try {
      candidate_matches = cuda_graph_arguments_match_cached_signature(
          graph, program, args, candidate->signature);
    } catch (...) {
      candidate->retire();
      throw;
    }
    if (!candidate_matches) {
      continue;
    }
    auto selected = std::move(candidate);
    cache.cuda_graph_state_alternates.erase(
        cache.cuda_graph_state_alternates.begin() + index);
    cache.cuda_graph_state_alternates.insert(
        cache.cuda_graph_state_alternates.begin(),
        std::move(cache.cuda_graph_state));
    cache.cuda_graph_state = std::move(selected);
    exact_match = true;
    return cache.cuda_graph_state.get();
  }
  return state;
}

CompiledGraphCudaState *allocate_cuda_replay_slot_for_miss(
    CompiledGraphJITCache &cache,
    CompiledGraphCudaState *state,
    const std::vector<CudaGraphArgSignatureEntry> &signature) {
  constexpr std::size_t kCudaReplaySlotCapacity = 2;
  if (state == nullptr || !state->graph_exec ||
      cuda_graph_signatures_share_replay_binding(state->signature, signature)) {
    return state;
  }
  std::unique_ptr<CompiledGraphCudaState, CompiledGraphCudaStateDeleter>
      reusable;
  if (1 + cache.cuda_graph_state_alternates.size() >=
      kCudaReplaySlotCapacity) {
    // Preserve the current MRU binding and recycle the least-recently-used
    // executable for the miss. Reusing the LRU state keeps the two-object
    // driver bound while avoiding the pathological A/C repatch loop that
    // would result from overwriting the current MRU slot.
    reusable = std::move(cache.cuda_graph_state_alternates.back());
    cache.cuda_graph_state_alternates.pop_back();
  }
  cache.cuda_graph_state_alternates.insert(
      cache.cuda_graph_state_alternates.begin(),
      std::move(cache.cuda_graph_state));
  cache.cuda_graph_state = std::move(reusable);
  if (!cache.cuda_graph_state) {
    cache.cuda_graph_state.reset(new CompiledGraphCudaState());
  }
  if (cache.graph_diagnostics_enabled) {
    cache.cuda_graph_state->diagnostics_enabled = true;
  }
  return cache.cuda_graph_state.get();
}

template <typename TopologyMatches>
CompiledGraphCudaState *select_cuda_graph_replay_slot_by_signature(
    CompiledGraphJITCache &cache,
    const std::vector<CudaGraphArgSignatureEntry> &signature,
    TopologyMatches &&topology_matches) {
  auto *state = get_cuda_graph_state(cache);
  if (state->graph_exec && topology_matches(*state) &&
      state->signature == signature) {
    return state;
  }
  for (std::size_t index = 0;
       index < cache.cuda_graph_state_alternates.size(); ++index) {
    auto &candidate = cache.cuda_graph_state_alternates[index];
    if (!candidate || !candidate->graph_exec ||
        !topology_matches(*candidate) || candidate->signature != signature) {
      continue;
    }
    auto selected = std::move(candidate);
    cache.cuda_graph_state_alternates.erase(
        cache.cuda_graph_state_alternates.begin() + index);
    cache.cuda_graph_state_alternates.insert(
        cache.cuda_graph_state_alternates.begin(),
        std::move(cache.cuda_graph_state));
    cache.cuda_graph_state = std::move(selected);
    return cache.cuda_graph_state.get();
  }
  return state;
}

std::optional<
    std::vector<std::unique_ptr<cuda::CudaDevice::AllocationLease>>>
acquire_cuda_graph_allocation_leases(
    const std::vector<DeviceAllocation> &allocations) {
  std::vector<std::unique_ptr<cuda::CudaDevice::AllocationLease>> leases;
  leases.reserve(allocations.size());
  for (const DeviceAllocation &allocation : allocations) {
    auto *device = dynamic_cast<cuda::CudaDevice *>(allocation.device);
    if (device == nullptr) {
      return std::nullopt;
    }
    auto lease = device->acquire_allocation_lease(allocation);
    if (lease == nullptr) {
      return std::nullopt;
    }
    leases.push_back(std::move(lease));
  }
  return leases;
}

std::optional<std::uintptr_t> cuda_graph_ndarray_address(
    const std::vector<CudaGraphArgSignatureEntry> &signature,
    const Arg &arg) {
  const auto entry = std::find_if(
      signature.begin(), signature.end(), [&](const auto &candidate) {
        return candidate.name == arg.name && candidate.tag == ArgKind::kNdarray;
      });
  if (entry == signature.end() || entry->device == nullptr ||
      entry->byte_size < 2 * sizeof(std::int32_t)) {
    return std::nullopt;
  }
  auto *device = dynamic_cast<cuda::CudaDevice *>(entry->device);
  if (device == nullptr) {
    return std::nullopt;
  }
  void *base = device->get_memory_addr({entry->device, entry->alloc_id});
  if (base == nullptr) {
    return std::nullopt;
  }
  return reinterpret_cast<std::uintptr_t>(base) + entry->byte_offset;
}

bool initialize_cuda_bounded_dispatch_controls(
    const CompiledGraph &graph,
    CompiledGraphCudaState &state,
    std::uint32_t saturation_grid_dim,
    std::uint32_t *driver_error) {
  TI_ASSERT(driver_error != nullptr);
  *driver_error = CUDA_SUCCESS;
  state.bounded_dispatch_controls.clear();
  state.bounded_dispatch_controls.resize(graph.dispatches.size());
  state.bounded_dispatch_groups.clear();
  state.bounded_dispatch_observations.clear();
  state.bounded_dispatch_group_indices.assign(graph.dispatches.size(), -1);
  state.bounded_dispatch_group_member_indices.assign(graph.dispatches.size(),
                                                     -1);
  state.bounded_dispatch_observations.reserve(graph.dispatches.size());
  for (const auto &dispatch : graph.dispatches) {
    const auto &metadata = dispatch.cuda_bounded_dispatch;
    if (!metadata.has_value()) {
      continue;
    }
    auto extent = cuda_graph_ndarray_address(state.signature,
                                             metadata->extent_arg);
    if (!extent.has_value()) {
      *driver_error = CUDA_ERROR_NOT_SUPPORTED;
      return false;
    }
    const auto capacity_grid = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(metadata->capacity) +
         metadata->block_dim - 1u) /
        metadata->block_dim);
    const auto baseline_grid =
        saturation_grid_dim == 0
            ? capacity_grid
            : std::min(capacity_grid, saturation_grid_dim);
    state.bounded_dispatch_observations.push_back(
        {metadata->extent_arg, metadata->capacity, metadata->block_dim, baseline_grid,
         metadata->adaptive_grid});
  }
  if (std::none_of(graph.dispatches.begin(), graph.dispatches.end(),
                   [](const auto &dispatch) {
                     return dispatch.cuda_bounded_dispatch.has_value() &&
                            dispatch.cuda_bounded_dispatch->adaptive_grid;
                   })) {
    return true;
  }
  if (saturation_grid_dim == 0) {
    *driver_error = CUDA_ERROR_NOT_SUPPORTED;
    return false;
  }
  auto &driver = CUDADriver::get_instance();
  if (!driver.launch_kernel_ex.available() || !driver.graph_upload.available()) {
    *driver_error = CUDA_ERROR_NOT_SUPPORTED;
    return false;
  }
  if (!cuda::driver_graph_prepare_bounded_update(driver_error)) {
    return false;
  }
  auto same_group_contract = [](const CudaBoundedDispatchMetadata &lhs,
                                const CudaBoundedDispatchMetadata &rhs) {
    return rhs.adaptive_grid && rhs.grouped_update &&
           lhs.extent_arg == rhs.extent_arg && lhs.capacity == rhs.capacity &&
           lhs.block_dim == rhs.block_dim;
  };
  for (std::size_t i = 0; i < graph.dispatches.size();) {
    const auto &metadata = graph.dispatches[i].cuda_bounded_dispatch;
    if (!metadata.has_value() || !metadata->adaptive_grid) {
      ++i;
      continue;
    }
    auto extent = cuda_graph_ndarray_address(state.signature,
                                             metadata->extent_arg);
    if (!extent.has_value()) {
      *driver_error = CUDA_ERROR_NOT_SUPPORTED;
      return false;
    }
    if (metadata->grouped_update) {
      std::size_t end = i + 1;
      while (end < graph.dispatches.size()) {
        const auto &candidate = graph.dispatches[end].cuda_bounded_dispatch;
        if (!candidate.has_value() ||
            !same_group_contract(*metadata, *candidate)) {
          break;
        }
        ++end;
      }
      if (end - i > 1) {
        CudaGraphBoundedDispatchGroup group;
        group.host_control.extent = *extent;
        group.host_control.node_count = static_cast<std::uint32_t>(end - i);
        group.host_control.capacity = metadata->capacity;
        group.host_control.block_dim = metadata->block_dim;
        group.host_control.max_grid_dim = saturation_grid_dim;
        group.host_control.telemetry_enabled =
            state.diagnostics_enabled ? 1u : 0u;
        group.host_nodes.resize(end - i);
        group.dispatch_indices.reserve(end - i);
        for (std::size_t dispatch_index = i; dispatch_index < end;
             ++dispatch_index) {
          group.dispatch_indices.push_back(dispatch_index);
        }
        *driver_error =
            driver.malloc.call(reinterpret_cast<void **>(&group.device_control),
                               sizeof(cuda::CudaGraphBoundedGroupControl));
        if (*driver_error != CUDA_SUCCESS) {
          return false;
        }
        *driver_error = driver.malloc.call(
            reinterpret_cast<void **>(&group.device_nodes),
            group.host_nodes.size() * sizeof(std::uintptr_t));
        if (*driver_error != CUDA_SUCCESS) {
          driver.mem_free.call(group.device_control);
          group.device_control = nullptr;
          return false;
        }
        group.host_control.device_nodes =
            reinterpret_cast<std::uintptr_t>(group.device_nodes);
        const auto group_index =
            static_cast<std::int32_t>(state.bounded_dispatch_groups.size());
        for (std::size_t member = 0; member < group.dispatch_indices.size();
             ++member) {
          const auto dispatch_index = group.dispatch_indices[member];
          state.bounded_dispatch_group_indices[dispatch_index] = group_index;
          state.bounded_dispatch_group_member_indices[dispatch_index] =
              static_cast<std::int32_t>(member);
        }
        state.bounded_dispatch_groups.push_back(std::move(group));
        i = end;
        continue;
      }
    }
    auto &control = state.bounded_dispatch_controls[i];
    control.host_control.extent = *extent;
    control.host_control.capacity = metadata->capacity;
    control.host_control.block_dim = metadata->block_dim;
    control.host_control.max_grid_dim = saturation_grid_dim;
    *driver_error = driver.malloc.call(
        reinterpret_cast<void **>(&control.device_control),
        sizeof(cuda::CudaGraphBoundedExtentControl));
    if (*driver_error != CUDA_SUCCESS) {
      return false;
    }
    ++i;
  }
  return true;
}

bool patch_cuda_bounded_dispatch_controls(
    const CompiledGraph &graph,
    const std::vector<CudaGraphArgSignatureEntry> &signature,
    CompiledGraphCudaState &state,
    std::vector<std::vector<uint8_t>> &host_buffers,
    void *stream = nullptr) {
  if (state.bounded_dispatch_controls.size() != graph.dispatches.size()) {
    return false;
  }
  if (state.bounded_dispatch_group_indices.size() != graph.dispatches.size() ||
      state.bounded_dispatch_group_member_indices.size() !=
          graph.dispatches.size()) {
    return false;
  }
  auto &driver = CUDADriver::get_instance();
  for (std::size_t i = 0; i < graph.dispatches.size(); ++i) {
    const auto &metadata = graph.dispatches[i].cuda_bounded_dispatch;
    if (!metadata.has_value() || !metadata->adaptive_grid) {
      continue;
    }
    if (state.bounded_dispatch_group_indices[i] >= 0) {
      continue;
    }
    auto &control = state.bounded_dispatch_controls[i];
    auto extent = cuda_graph_ndarray_address(signature, metadata->extent_arg);
    if (!extent.has_value() || control.device_control == nullptr) {
      return false;
    }
    if (control.host_control.extent == *extent) {
      continue;
    }
    control.host_control.extent = *extent;
    control.host_control.driver_status = 0;
    host_buffers.emplace_back(sizeof(control.host_control));
    std::memcpy(host_buffers.back().data(), &control.host_control,
                sizeof(control.host_control));
    driver.memcpy_host_to_device_async(control.device_control,
                                       host_buffers.back().data(),
                                       sizeof(control.host_control), stream);
    if (state.diagnostics_enabled) {
      ++state.stats.asynchronous_control_updates;
    }
  }
  for (auto &group : state.bounded_dispatch_groups) {
    if (group.dispatch_indices.empty() || group.device_control == nullptr ||
        group.device_nodes == nullptr) {
      return false;
    }
    const auto &metadata =
        graph.dispatches[group.dispatch_indices.front()].cuda_bounded_dispatch;
    if (!metadata.has_value() || !metadata->adaptive_grid ||
        !metadata->grouped_update) {
      return false;
    }
    auto extent = cuda_graph_ndarray_address(signature, metadata->extent_arg);
    if (!extent.has_value()) {
      return false;
    }
    if (group.host_control.extent == *extent) {
      continue;
    }
    group.host_control.extent = *extent;
    group.host_control.initialized = 0;
    group.host_control.driver_status = 0;
    host_buffers.emplace_back(sizeof(group.host_control));
    std::memcpy(host_buffers.back().data(), &group.host_control,
                sizeof(group.host_control));
    driver.memcpy_host_to_device_async(group.device_control,
                                       host_buffers.back().data(),
                                       sizeof(group.host_control), stream);
    if (state.diagnostics_enabled) {
      ++state.stats.asynchronous_control_updates;
    }
  }
  return true;
}

void record_cuda_capture_command(
    const CompiledDispatch &dispatch,
    const std::unordered_map<std::string, IValue> &args,
    Program &program,
    void *stream);

std::uint32_t capture_cuda_graph_packets(const CompiledGraph &graph,
                                         CompiledGraphCudaState &state,
                                         void *capture_stream,
                                         const std::unordered_map<std::string,
                                                                  IValue> &args,
                                         Program &program) {
  TI_ASSERT(state.packets.size() == graph.dispatches.size());
  TI_ASSERT(state.bounded_dispatch_controls.size() == graph.dispatches.size());
  TI_ASSERT(state.bounded_dispatch_group_indices.size() ==
            graph.dispatches.size());
  TI_ASSERT(state.bounded_dispatch_group_member_indices.size() ==
            graph.dispatches.size());
  auto capture_dispatch = [&](std::size_t i, void *stream) -> std::uint32_t {
    auto &packet = state.packets[i];
    const auto &dispatch = graph.dispatches[i];
    if (dispatch.cuda_capture_command) {
      record_cuda_capture_command(dispatch, args, program, stream);
      if (auto resource =
              dispatch.cuda_capture_command->take_capture_resources()) {
        state.provider_capture_resources.push_back(std::move(resource));
      }
      return CUDA_SUCCESS;
    }
    const auto &metadata = dispatch.cuda_bounded_dispatch;
    if (!metadata.has_value() || !metadata->adaptive_grid) {
      packet.launcher->capture_cuda_graph_launch(packet.packet, stream);
      return CUDA_SUCCESS;
    }
    const auto group_index = state.bounded_dispatch_group_indices[i];
    if (metadata->grouped_update && group_index >= 0) {
      const auto member_index = state.bounded_dispatch_group_member_indices[i];
      if (member_index < 0 || static_cast<std::size_t>(group_index) >=
                                  state.bounded_dispatch_groups.size()) {
        return CUDA_ERROR_NOT_SUPPORTED;
      }
      auto &group = state.bounded_dispatch_groups[group_index];
      if (static_cast<std::size_t>(member_index) >= group.host_nodes.size() ||
          group.device_control == nullptr || group.device_nodes == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
      }
      if (member_index == 0) {
        cuda::driver_graph_update_bounded_group(group.device_control,
                                                stream);
      }
      void *device_node = nullptr;
      std::uint32_t driver_error = CUDA_SUCCESS;
      if (!packet.launcher->capture_cuda_graph_bounded_launch(
              packet.packet, stream, &device_node, &driver_error)) {
        return driver_error == CUDA_SUCCESS ? CUDA_ERROR_NOT_SUPPORTED
                                            : driver_error;
      }
      group.host_nodes[member_index] =
          reinterpret_cast<std::uintptr_t>(device_node);
      return CUDA_SUCCESS;
    }
    auto &control = state.bounded_dispatch_controls[i];
    if (control.device_control == nullptr) {
      return CUDA_ERROR_NOT_SUPPORTED;
    }
    cuda::driver_graph_update_bounded_extent(control.device_control,
                                             stream);
    void *device_node = nullptr;
    std::uint32_t driver_error = CUDA_SUCCESS;
    if (!packet.launcher->capture_cuda_graph_bounded_launch(
            packet.packet, stream, &device_node, &driver_error)) {
      return driver_error == CUDA_SUCCESS ? CUDA_ERROR_NOT_SUPPORTED
                                          : driver_error;
    }
    control.host_control.device_node =
        reinterpret_cast<std::uintptr_t>(device_node);
    return CUDA_SUCCESS;
  };

  if (!graph.has_cuda_parallel_dispatch_groups()) {
    for (std::size_t i = 0; i < graph.dispatches.size(); ++i) {
      const auto error = capture_dispatch(i, capture_stream);
      if (error != CUDA_SUCCESS) {
        return error;
      }
    }
    return CUDA_SUCCESS;
  }

  const auto &groups = graph.cuda_parallel_dispatch_groups;
  TI_ASSERT(state.parallel_capture_streams.size() == groups.size());
  TI_ASSERT(state.parallel_capture_events.size() == groups.size() + 1);
  const std::size_t parallel_begin = groups.front().front();
  const std::size_t parallel_end = groups.back().back() + 1;
  for (std::size_t i = 0; i < parallel_begin; ++i) {
    const auto error = capture_dispatch(i, capture_stream);
    if (error != CUDA_SUCCESS) {
      return error;
    }
  }

  auto &driver = CUDADriver::get_instance();
  auto &fork_event = state.parallel_capture_events.front();
  fork_event.record(capture_stream);
  for (std::size_t branch = 0; branch < groups.size(); ++branch) {
    void *branch_stream = state.parallel_capture_streams[branch]->get();
    auto error = driver.stream_wait_event.call(branch_stream, fork_event.get(), 0);
    if (error != CUDA_SUCCESS) {
      return error;
    }
    for (const auto dispatch_index : groups[branch]) {
      error = capture_dispatch(dispatch_index, branch_stream);
      if (error != CUDA_SUCCESS) {
        return error;
      }
    }
    state.parallel_capture_events[branch + 1].record(branch_stream);
  }
  for (std::size_t branch = 0; branch < groups.size(); ++branch) {
    const auto error = driver.stream_wait_event.call(
        capture_stream, state.parallel_capture_events[branch + 1].get(), 0);
    if (error != CUDA_SUCCESS) {
      return error;
    }
  }
  for (std::size_t i = parallel_end; i < graph.dispatches.size(); ++i) {
    const auto error = capture_dispatch(i, capture_stream);
    if (error != CUDA_SUCCESS) {
      return error;
    }
  }
  return CUDA_SUCCESS;
}

std::uint32_t upload_cuda_bounded_dispatch_controls(
    CompiledGraphCudaState &state) {
  auto &driver = CUDADriver::get_instance();
  for (const auto &control : state.bounded_dispatch_controls) {
    if (control.device_control == nullptr) {
      continue;
    }
    const auto error = driver.memcpy_host_to_device.call(
        control.device_control,
        const_cast<cuda::CudaGraphBoundedExtentControl *>(
            &control.host_control),
        sizeof(control.host_control));
    if (error != CUDA_SUCCESS) {
      return error;
    }
  }
  for (const auto &group : state.bounded_dispatch_groups) {
    if (group.device_control == nullptr || group.device_nodes == nullptr ||
        group.host_nodes.empty()) {
      return CUDA_ERROR_NOT_SUPPORTED;
    }
    auto error = driver.memcpy_host_to_device.call(
        group.device_nodes,
        const_cast<std::uintptr_t *>(group.host_nodes.data()),
        group.host_nodes.size() * sizeof(std::uintptr_t));
    if (error != CUDA_SUCCESS) {
      return error;
    }
    error = driver.memcpy_host_to_device.call(
        group.device_control,
        const_cast<cuda::CudaGraphBoundedGroupControl *>(&group.host_control),
        sizeof(group.host_control));
    if (error != CUDA_SUCCESS) {
      return error;
    }
  }
  return CUDA_SUCCESS;
}

bool patch_cuda_graph_arguments(
    const CompiledGraph &graph,
    const std::unordered_map<std::string, IValue> &args,
    const std::vector<CudaGraphArgSignatureEntry> &signature,
    CompiledGraphCudaState &state,
    std::vector<std::vector<uint8_t>> &host_arg_buffers,
    void *stream = nullptr) {
  if (state.packets.size() != graph.dispatches.size()) {
    return false;
  }
  host_arg_buffers.clear();
  host_arg_buffers.reserve(graph.dispatches.size());
  for (std::size_t i = 0; i < graph.dispatches.size(); ++i) {
    const auto &dispatch = graph.dispatches[i];
    auto *prog = dispatch.ti_kernel->program;
    auto *launcher = dynamic_cast<cuda::KernelLauncher *>(
        &prog->get_program_impl()->get_kernel_launcher());
    if (launcher == nullptr || launcher != state.packets[i].launcher) {
      return false;
    }

    auto launch_ctx = dispatch.ti_kernel->make_launch_context();
    launch_ctx.append_dispatch_label(dispatch.dispatch_label);
    graph.init_runtime_context(dispatch.symbolic_args, args, launch_ctx);
    prog->resolve_ndarray_launch_context_under_guard(launch_ctx);
    prog->resolve_runtime_storage_launch_context_under_guard(launch_ctx);
    prog->resolve_texture_launch_context_under_guard(launch_ctx);
    host_arg_buffers.emplace_back();
    if (!launcher->update_cuda_graph_launch(
            state.packets[i].packet, launch_ctx, host_arg_buffers.back(),
            stream)) {
      return false;
    }
    const auto &metadata = dispatch.cuda_bounded_dispatch;
    if (metadata.has_value()) {
      auto extent = cuda_graph_ndarray_address(signature, metadata->extent_arg);
      if (!extent.has_value()) {
        return false;
      }
      host_arg_buffers.emplace_back();
      if (!launcher->update_cuda_graph_bounded_range(
              state.packets[i].packet,
              reinterpret_cast<void *>(*extent), metadata->capacity,
              host_arg_buffers.back(), stream)) {
        return false;
      }
    }
  }
  return true;
}

bool is_cuda_graph_structural_driver_error(uint32_t error) {
  return error == CUDA_ERROR_NOT_SUPPORTED ||
         error == CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
}

void mark_cuda_graph_fallback(
    CompiledGraphCudaState &state,
    CompiledGraphFallbackReason reason,
    bool structural = false) {
  state.stats.last_path = CompiledGraphExecutionPath::ordinary_fallback;
  state.stats.last_fallback_reason = reason;
  if (state.diagnostics_enabled) {
    ++state.stats.ordinary_fallbacks;
    if (structural) {
      ++state.stats.structural_fallbacks;
    }
  }
}

bool handle_cuda_graph_driver_failure(
    CompiledGraphCudaState &state,
    uint32_t error,
    const char *stage,
    CompiledGraphCaptureRetryState *retry_override = nullptr) {
  auto &retry = retry_override == nullptr ? state.retry : *retry_override;
  if (state.diagnostics_enabled) {
    state.stats.last_driver_error = error;
  }
  if (classify_cuda_driver_error(error) ==
      BackendErrorClassification::kFatal) {
    state.stats.last_path = CompiledGraphExecutionPath::none;
    state.stats.last_fallback_reason =
        CompiledGraphFallbackReason::fatal_driver_failure;
    state.stats.last_driver_error = error;
    throw BackendRuntimeError(
        Arch::cuda, error, fmt::format("cuda_graph_{}", stage),
        fmt::format("CUDA graph {} failed with a context-fatal error: {}",
                    stage, get_cuda_error_message(error)));
  }

  if (is_cuda_graph_structural_driver_error(error)) {
    retry.record_structural_failure();
    mark_cuda_graph_fallback(
        state, CompiledGraphFallbackReason::structural_unsupported, true);
  } else {
    retry.record_transient_failure();
    if (state.diagnostics_enabled) {
      ++state.stats.transient_failures;
    }
    mark_cuda_graph_fallback(
        state, CompiledGraphFallbackReason::transient_driver_failure);
  }
  state.retire();
  return false;
}

void record_cuda_capture_command(
    const CompiledDispatch &dispatch,
    const std::unordered_map<std::string, IValue> &args,
    Program &program,
    void *stream) {
  TI_ERROR_IF(dispatch.cuda_capture_command == nullptr,
              "CUDA Graph capture dispatch lost its provider command");
  dispatch.cuda_capture_command->record(args, program, stream);
}

bool try_run_cuda_graph(const CompiledGraph &graph,
                        const CompileConfig &compile_config,
                        const std::unordered_map<std::string, IValue> &args,
                        CompiledGraphJITCache &cache,
                        Program &program,
                        RuntimeStatistics *statistics,
                        bool cuda_concurrent_batch_lane = false) {
  const bool stable_replay = cache.stable_replay_optimization_enabled.load(
      std::memory_order_relaxed);
  bool exact_slot = false;
  auto *state = select_cuda_graph_replay_slot(
      cache, graph, program, args, exact_slot,
      [&](const CompiledGraphCudaState &candidate) {
        return stable_replay && !candidate.conditional_mode &&
               !candidate.masked_mode && !candidate.masked_nested_mode &&
               !candidate.device_update_nested_mode;
      });
  if (state->diagnostics_enabled) {
    state->stats.backend = CompiledGraphBackend::cuda;
    ++state->stats.attempts;
    state->stats.last_driver_error = 0;
  }
  if (compile_config.debug) {
    mark_cuda_graph_fallback(*state,
                             CompiledGraphFallbackReason::debug_mode);
    return false;
  }
  if (graph.dispatches.empty()) {
    mark_cuda_graph_fallback(
        *state, CompiledGraphFallbackReason::insufficient_dispatches, true);
    return false;
  }
  if (!stable_replay) {
    mark_cuda_graph_fallback(*state,
                             CompiledGraphFallbackReason::runtime_mode);
    return false;
  }
  const bool has_capture_commands = graph.has_cuda_capture_commands();
  if (graph.has_cuda_parallel_dispatch_groups() &&
      (has_capture_commands ||
       std::any_of(graph.dispatches.begin(), graph.dispatches.end(),
                   [](const auto &dispatch) {
                     return dispatch.cuda_bounded_dispatch.has_value() ||
                            dispatch.indirect_dispatch_arg.has_value();
                   }))) {
    mark_cuda_graph_fallback(
        *state, CompiledGraphFallbackReason::structural_unsupported, true);
    return false;
  }
  if (has_capture_commands) {
    const bool providers_supported = std::all_of(
        graph.dispatches.begin(), graph.dispatches.end(),
        [&](const auto &dispatch) {
          return dispatch.cuda_capture_command == nullptr ||
                 dispatch.cuda_capture_command->supports(args, program);
        });
    // Preparation builds plans/analysis but never executes user operations.
    // Kernel presence is neither necessary nor a proof of safe preparation.
    if (!providers_supported) {
      mark_cuda_graph_fallback(
          *state, CompiledGraphFallbackReason::structural_unsupported, true);
      return false;
    }
  }
  void *replay_stream = nullptr;
  auto select_replay_stream = [&]() {
    replay_stream =
        cuda_concurrent_batch_lane
            ? program.register_runtime_cuda_concurrent_stream(
                  state->ensure_capture_stream())
            : nullptr;
    return replay_stream;
  };
  // Production replay is deliberately attribution-free. Measured execution
  // uses the explicit submission-telemetry path instead of turning clocks and
  // counters on inside the latency-sensitive replay implementation.
  constexpr bool attribute = false;
  if (state->graph_exec && stable_replay) {
    const auto signature_begin =
        attribute ? ReplayClock::now() : ReplayClock::time_point{};
    const bool exact_match = exact_slot;
    if (attribute) {
      cache.replay_attribution.signature_ns +=
          replay_elapsed_ns(signature_begin);
      if (exact_match) {
        ++cache.replay_attribution.signature_hits;
      } else {
        ++cache.replay_attribution.signature_misses;
      }
    }
    if (exact_match) {
      CUDAContext::get_instance().make_current();
      state->collect_ready_deferred_resources();
      CUDADriver::get_instance().graph_launch(state->graph_exec.get(),
                                              select_replay_stream());
      state->stats.last_path = CompiledGraphExecutionPath::cuda_exact_replay;
      state->stats.last_fallback_reason = CompiledGraphFallbackReason::none;
      if (state->diagnostics_enabled) {
        ++state->stats.exact_replays;
      }
      if (statistics != nullptr) {
        statistics->record_graph_replay();
      }
      return true;
    }
  }
  std::optional<CudaGraphSignatureCandidate> signature;
  const auto signature_begin =
      attribute ? ReplayClock::now() : ReplayClock::time_point{};
  try {
    signature = make_cuda_graph_signature(graph, program, args);
  } catch (...) {
    // A runtime Field may have been destroyed after this executable was last
    // used. Retire the cached graph before surfacing the stale-generation
    // error so no later cleanup keeps an executable with a dead root address.
    state->retire();
    throw;
  }
  if (attribute) {
    cache.replay_attribution.signature_ns +=
        replay_elapsed_ns(signature_begin);
  }
  if (!signature.has_value()) {
    mark_cuda_graph_fallback(
        *state, CompiledGraphFallbackReason::unsupported_arguments, true);
    return false;
  }
  if (state->graph_exec && state->signature != signature->entries) {
    state = allocate_cuda_replay_slot_for_miss(cache, state,
                                               signature->entries);
  }
  CUDAContext::get_instance().make_current();
  if (state->retry.structurally_disabled()) {
    mark_cuda_graph_fallback(
        *state, CompiledGraphFallbackReason::structural_unsupported, true);
    return false;
  }
  state->collect_ready_deferred_resources();
  if (state->graph_exec &&
      state->signature == signature->entries) {
    CUDADriver::get_instance().graph_launch(state->graph_exec.get(),
                                            select_replay_stream());
    state->stats.last_path = CompiledGraphExecutionPath::cuda_exact_replay;
    state->stats.last_fallback_reason = CompiledGraphFallbackReason::none;
    if (state->diagnostics_enabled) {
      ++state->stats.exact_replays;
    }
    if (statistics != nullptr) {
      statistics->record_graph_replay();
    }
    return true;
  }

  if (!state->graph_exec && !state->retry.should_attempt()) {
    if (state->diagnostics_enabled) {
      ++state->stats.retry_backoff_fallbacks;
    }
    mark_cuda_graph_fallback(
        *state, CompiledGraphFallbackReason::retry_backoff);
    return false;
  }

  auto allocation_leases =
      acquire_cuda_graph_allocation_leases(signature->allocations);
  if (!allocation_leases.has_value()) {
    mark_cuda_graph_fallback(
        *state, CompiledGraphFallbackReason::resource_unavailable);
    return false;
  }
  // Provider commands may retain descriptors or other state fixed to captured
  // addresses. Kernel-argument patching must never turn an exact-only command
  // graph into a false match; a changed binding uses another bounded slot.
  const bool structurally_compatible =
      !graph.cuda_capture_commands_require_exact_bindings() &&
      state->graph_exec &&
      cuda_graph_signatures_are_structurally_compatible(
          state->signature, signature->entries);
  if (structurally_compatible) {
    select_replay_stream();
    std::vector<std::vector<uint8_t>> host_arg_buffers;
    if (patch_cuda_graph_arguments(graph, args, signature->entries, *state,
                                   host_arg_buffers, replay_stream) &&
        patch_cuda_bounded_dispatch_controls(
            graph, signature->entries, *state, host_arg_buffers,
            replay_stream)) {
      std::vector<std::unique_ptr<cuda::CudaDevice::AllocationLease>>
          old_allocation_leases;
      if (state->allocations != signature->allocations) {
        old_allocation_leases = std::move(state->allocation_leases);
        state->allocation_leases = std::move(*allocation_leases);
        state->allocations = signature->allocations;
      }
      state->signature = std::move(signature->entries);
      // The replay stream orders this event after the previous replay and
      // every argument-buffer upload above. At that point old allocations are
      // no longer referenced and the host staging buffers have been consumed.
      // Record before the new replay so bounded retirement does not wait for
      // one more graph execution than correctness requires.
      state->defer_replay_resources(std::move(old_allocation_leases),
                                    std::move(host_arg_buffers), replay_stream);
      CUDADriver::get_instance().graph_launch(state->graph_exec.get(),
                                              replay_stream);
      state->stats.last_path =
          CompiledGraphExecutionPath::cuda_patched_replay;
      state->stats.last_fallback_reason = CompiledGraphFallbackReason::none;
      if (state->diagnostics_enabled) {
        ++state->stats.patched_replays;
      }
      if (statistics != nullptr) {
        statistics->record_graph_replay();
      }
      return true;
    }
  }
  // Any partial argument-buffer update is ordered on the chosen replay
  // stream. Retirement synchronizes only the default stream, so a failed
  // concurrent patch must drain its private stream before host staging and
  // old allocation leases can be released. This is a fallback-only safety
  // boundary and never runs on exact pair replay.
  if (replay_stream != nullptr) {
    const auto error =
        CUDADriver::get_instance().stream_synchronize.call(replay_stream);
    TI_ERROR_IF(error != CUDA_SUCCESS,
                "CUDA concurrent Graph patch fallback synchronize failed: {}",
                CUDADriver::get_instance()
                    .stream_synchronize.get_error_message(error));
  }
  const bool is_recapture = state->has_captured_once;
  state->retire();
  state->signature = std::move(signature->entries);
  state->allocations = signature->allocations;
  state->allocation_leases = std::move(*allocation_leases);
  if (state->diagnostics_enabled) {
    ++state->stats.capture_attempts;
    if (state->has_captured_once) {
      ++state->stats.recaptures;
    }
  }
  if (cache.kernels.size() != graph.dispatches.size()) {
    cache.kernels.assign(graph.dispatches.size(), {});
  }

  auto &driver = CUDADriver::get_instance();
  void *capture_stream = state->ensure_capture_stream();

  for (std::size_t i = 0; i < graph.dispatches.size(); ++i) {
    const auto &dispatch = graph.dispatches[i];
    if (dispatch.cuda_capture_command) {
      // Build provider descriptors, workspace and optional preprocessing
      // before CUDA capture, without executing the user's mathematical work.
      dispatch.cuda_capture_command->prepare(args, program);
      state->packets.emplace_back(capture_stream);
      continue;
    }
    TI_ASSERT(dispatch.ti_kernel);
    auto *prog = dispatch.ti_kernel->program;
    auto *launcher = dynamic_cast<cuda::KernelLauncher *>(
        &prog->get_program_impl()->get_kernel_launcher());
    if (launcher == nullptr) {
      state->retry.record_structural_failure();
      mark_cuda_graph_fallback(
          *state, CompiledGraphFallbackReason::structural_unsupported, true);
      state->retire();
      return false;
    }

    auto *compiled_kernel_data =
        get_or_compile_cached_kernel(dispatch, compile_config, cache.kernels[i],
                                     /*cache_compiled_kernel_data=*/true);
    const auto &llvm_ckd =
        dynamic_cast<const LLVM::CompiledKernelData &>(*compiled_kernel_data);
    auto handle = launcher->register_llvm_kernel(llvm_ckd);

    auto launch_ctx = dispatch.ti_kernel->make_launch_context();
    launch_ctx.append_dispatch_label(dispatch.dispatch_label);
    graph.init_runtime_context(dispatch.symbolic_args, args, launch_ctx);
    prog->resolve_ndarray_launch_context_under_guard(launch_ctx);
    prog->resolve_runtime_storage_launch_context_under_guard(launch_ctx);
    prog->resolve_texture_launch_context_under_guard(launch_ctx);
    CudaGraphCapturePacket capture_packet(capture_stream);
    capture_packet.launcher = launcher;
    bool prepared = false;
    if (dispatch.cuda_bounded_dispatch.has_value()) {
      const auto &metadata = *dispatch.cuda_bounded_dispatch;
      auto extent =
          cuda_graph_ndarray_address(state->signature, metadata.extent_arg);
      prepared = extent.has_value() &&
                 launcher->prepare_cuda_graph_bounded_range(
                     handle, launch_ctx, capture_packet.packet,
                     reinterpret_cast<void *>(*extent), metadata.capacity,
                     capture_stream);
    } else {
      prepared = launcher->prepare_cuda_graph_launch(
          handle, launch_ctx, capture_packet.packet, capture_stream);
    }
    if (!prepared) {
      driver.stream_synchronize(capture_stream);
      state->retry.record_structural_failure();
      mark_cuda_graph_fallback(
          *state, CompiledGraphFallbackReason::structural_unsupported, true);
      state->retire();
      return false;
    }
    state->packets.push_back(std::move(capture_packet));
  }

  if (has_capture_commands) {
    // Provider preparation may submit on the runtime's legacy default stream;
    // complete it before capturing onto the dedicated nonblocking stream.
    driver.stream_synchronize(nullptr);
  }

  std::uint32_t bounded_setup_error = CUDA_SUCCESS;
  if (!initialize_cuda_bounded_dispatch_controls(
          graph, *state,
          static_cast<std::uint32_t>(compile_config.saturating_grid_dim),
          &bounded_setup_error)) {
    return handle_cuda_graph_driver_failure(*state, bounded_setup_error,
                                            "bounded control setup");
  }

  state->stats.zero_arg_eligible =
      !has_capture_commands && !state->packets.empty() &&
      std::all_of(state->packets.begin(), state->packets.end(),
                  [](const CudaGraphCapturePacket &packet) {
                    return packet.packet.arg_buffer_size == 0 &&
                           packet.packet.device_arg_buffer == nullptr;
                  });

  driver.stream_synchronize(capture_stream);
  if (graph.has_cuda_parallel_dispatch_groups()) {
    state->ensure_parallel_capture_resources(
        graph.cuda_parallel_dispatch_groups.size());
    for (const auto &stream : state->parallel_capture_streams) {
      driver.stream_synchronize(stream->get());
    }
  }
  auto capture_lock = CUDAContext::get_instance().get_graph_capture_lock_guard();
  auto begin_err = driver.stream_begin_capture.call(
      capture_stream, CU_STREAM_CAPTURE_MODE_RELAXED);
  if (begin_err != CUDA_SUCCESS) {
    return handle_cuda_graph_driver_failure(*state, begin_err,
                                            "stream begin capture");
  }
  CudaStreamCaptureGuard capture_guard(capture_stream);
  try {
    const auto capture_error = capture_cuda_graph_packets(
        graph, *state, capture_stream, args, program);
    if (capture_error != CUDA_SUCCESS) {
      capture_guard.abort();
      return handle_cuda_graph_driver_failure(
          *state, capture_error, "graph packet capture");
    }
  } catch (...) {
    if (state->diagnostics_enabled) {
      ++state->stats.capture_exceptions;
    }
    capture_guard.abort();
    throw;
  }
  CudaGraphHandle captured_graph;
  auto end_err = capture_guard.end(captured_graph.put());
  if (end_err != CUDA_SUCCESS || !captured_graph) {
    return handle_cuda_graph_driver_failure(*state, end_err,
                                            "stream end capture");
  }
  auto instantiate_err = driver.graph_instantiate_with_flags.call(
      state->graph_exec.put(), captured_graph.get(), 0);
  captured_graph.reset();
  if (instantiate_err != CUDA_SUCCESS || !state->graph_exec) {
    return handle_cuda_graph_driver_failure(*state, instantiate_err,
                                            "instantiate");
  }
  if (!state->bounded_dispatch_groups.empty() ||
      std::any_of(state->bounded_dispatch_controls.begin(),
                  state->bounded_dispatch_controls.end(),
                  [](const auto &control) {
                    return control.device_control != nullptr;
                  })) {
    const auto upload_error =
        driver.graph_upload.call(state->graph_exec.get(), nullptr);
    if (upload_error != CUDA_SUCCESS) {
      return handle_cuda_graph_driver_failure(*state, upload_error,
                                              "bounded graph upload");
    }
  }
  // Upload Graph-owned controls only after the executable is instantiated and
  // uploaded. Any earlier failure leaves no externally visible node state.
  const auto control_upload_error =
      upload_cuda_bounded_dispatch_controls(*state);
  if (control_upload_error != CUDA_SUCCESS) {
    return handle_cuda_graph_driver_failure(
        *state, control_upload_error, "bounded control upload");
  }
  state->retry.record_success();
  state->has_captured_once = true;
  state->stats.last_path = CompiledGraphExecutionPath::cuda_capture;
  state->stats.last_fallback_reason = CompiledGraphFallbackReason::none;
  if (state->diagnostics_enabled) {
    ++state->stats.captures;
    if (state->stats.zero_arg_eligible) {
      ++state->stats.zero_arg_captures;
    }
  }
  if (statistics != nullptr) {
    statistics->record_graph_capture();
    if (is_recapture) {
      statistics->record_graph_recapture();
    }
  }
  driver.graph_launch(state->graph_exec.get(), select_replay_stream());
  return true;
}

// CUDA conditional nodes are a CUDA 12.8 facility. Older drivers still have
// ordinary CUDA Graph capture, so encode a bounded graph and latch the device
// selector before each logical region. Every captured Taichi task is a private
// compiler-gated variant: inactive work reaches the device but returns before
// any payload side effect. This matches Vulkan's bounded masking semantics
// without a selector readback or per-iteration host submission.
bool try_run_cuda_masked_control_graph(
    const CompiledGraph &graph,
    const CompileConfig &compile_config,
    const std::unordered_map<std::string, IValue> &args,
    CompiledGraphJITCache &cache,
    Program &program,
    Ndarray &selector,
    int control_type,
    int max_iterations,
    bool continue_while_nonzero,
    const std::vector<int> &branch_dispatch_counts,
    int default_branch,
    RuntimeStatistics *statistics) {
  auto *state = get_cuda_graph_state(cache);
  if (state->diagnostics_enabled) {
    state->stats.backend = CompiledGraphBackend::cuda;
    ++state->stats.attempts;
    state->stats.last_driver_error = 0;
  }
  auto structural_fallback = [&]() {
    state->masked_retry.record_structural_failure();
    mark_cuda_graph_fallback(
        *state, CompiledGraphFallbackReason::structural_unsupported, true);
    state->retire();
    return false;
  };

  const bool is_while = control_type == 1;
  const bool is_if = control_type == 0;
  const bool is_switch = control_type == 2;
  const std::size_t branch_dispatch_total =
      std::accumulate(branch_dispatch_counts.begin(),
                      branch_dispatch_counts.end(), std::size_t{0});
  const std::size_t encoded_dispatches =
      is_while ? (max_iterations > 0 &&
                          graph.dispatches.size() <=
                              (std::numeric_limits<std::size_t>::max)() /
                                  static_cast<std::size_t>(max_iterations)
                      ? graph.dispatches.size() *
                            static_cast<std::size_t>(max_iterations)
                      : (std::numeric_limits<std::size_t>::max)())
               : graph.dispatches.size();
  constexpr std::size_t kMaxEncodedDispatches = 4096;
  if (compile_config.debug || graph.dispatches.empty() ||
      (!is_while && !is_if && !is_switch) ||
      (is_while && max_iterations <= 0) ||
      (!is_while && (branch_dispatch_counts.empty() ||
                     branch_dispatch_total != graph.dispatches.size() ||
                     std::any_of(branch_dispatch_counts.begin(),
                                 branch_dispatch_counts.end(),
                                 [](int count) { return count <= 0; }))) ||
      (is_if && branch_dispatch_counts.size() > 2) ||
      (is_switch &&
       default_branch >= static_cast<int>(branch_dispatch_counts.size())) ||
      encoded_dispatches > kMaxEncodedDispatches ||
      selector.get_nelement() != 1 ||
      selector.get_element_data_type() != PrimitiveType::i32 ||
      selector.owning_program() != &program) {
    return structural_fallback();
  }

  const DeviceAllocation selector_allocation = selector.get_device_allocation();
  auto *selector_device =
      dynamic_cast<cuda::CudaDevice *>(selector_allocation.device);
  if (selector_device == nullptr) {
    return structural_fallback();
  }
  auto &driver = CUDADriver::get_instance();
  const bool graph_symbols = driver.stream_begin_capture.available() &&
                             driver.stream_end_capture.available() &&
                             driver.graph_instantiate_with_flags.available() &&
                             driver.graph_launch.available() &&
                             driver.graph_destroy.available() &&
                             driver.graph_exec_destroy.available();
  if (!graph_symbols || !cuda::driver_graph_mask_latch_compiled()) {
    return structural_fallback();
  }

  std::optional<CudaGraphSignatureCandidate> signature;
  try {
    signature = make_cuda_graph_signature(graph, program, args);
  } catch (...) {
    state->retire();
    throw;
  }
  if (!signature.has_value()) {
    return structural_fallback();
  }
  if (std::find(signature->allocations.begin(), signature->allocations.end(),
                selector_allocation) == signature->allocations.end()) {
    signature->allocations.push_back(selector_allocation);
    std::sort(
        signature->allocations.begin(), signature->allocations.end(),
        [](const DeviceAllocation &lhs, const DeviceAllocation &rhs) {
          const auto lhs_device = reinterpret_cast<std::uintptr_t>(lhs.device);
          const auto rhs_device = reinterpret_cast<std::uintptr_t>(rhs.device);
          return lhs_device == rhs_device ? lhs.alloc_id < rhs.alloc_id
                                          : lhs_device < rhs_device;
        });
  }
  state = select_cuda_graph_replay_slot_by_signature(
      cache, signature->entries,
      [&](const CompiledGraphCudaState &candidate) {
        return candidate.masked_mode && !candidate.masked_nested_mode &&
               candidate.masked_control_type == control_type &&
               candidate.masked_max_iterations ==
                   (is_while ? max_iterations : 0) &&
               candidate.masked_continue_while_nonzero ==
                   continue_while_nonzero &&
               candidate.masked_default_branch == default_branch &&
               candidate.masked_branch_dispatch_counts ==
                   branch_dispatch_counts &&
               candidate.masked_selector_allocation == selector_allocation;
      });
  if (state->graph_exec && state->signature != signature->entries) {
    state = allocate_cuda_replay_slot_for_miss(cache, state,
                                               signature->entries);
  }
  CUDAContext::get_instance().make_current();
  state->collect_ready_deferred_resources();

  const bool topology_matches =
      state->masked_mode && state->masked_control_type == control_type &&
      state->masked_max_iterations == (is_while ? max_iterations : 0) &&
      state->masked_continue_while_nonzero == continue_while_nonzero &&
      state->masked_default_branch == default_branch &&
      state->masked_branch_dispatch_counts == branch_dispatch_counts &&
      state->masked_selector_allocation == selector_allocation;
  if (topology_matches && state->graph_exec &&
      state->signature == signature->entries &&
      state->allocations == signature->allocations) {
    driver.graph_launch(state->graph_exec.get(), nullptr);
    state->stats.last_path = CompiledGraphExecutionPath::cuda_masked_replay;
    state->stats.last_fallback_reason = CompiledGraphFallbackReason::none;
    if (state->diagnostics_enabled) {
      ++state->stats.masked_replays;
    }
    if (statistics != nullptr) {
      statistics->record_graph_replay();
    }
    return true;
  }
  if (!state->graph_exec && !state->masked_retry.should_attempt()) {
    if (state->diagnostics_enabled) {
      ++state->stats.retry_backoff_fallbacks;
    }
    mark_cuda_graph_fallback(*state,
                             CompiledGraphFallbackReason::retry_backoff);
    return false;
  }

  auto allocation_leases =
      acquire_cuda_graph_allocation_leases(signature->allocations);
  if (!allocation_leases.has_value()) {
    mark_cuda_graph_fallback(*state,
                             CompiledGraphFallbackReason::resource_unavailable);
    return false;
  }
  const bool structurally_compatible =
      topology_matches && state->graph_exec &&
      cuda_graph_signatures_are_structurally_compatible(state->signature,
                                                        signature->entries);
  if (structurally_compatible) {
    std::vector<std::vector<uint8_t>> host_arg_buffers;
    if (patch_cuda_graph_arguments(graph, args, signature->entries, *state,
                                   host_arg_buffers)) {
      std::vector<std::unique_ptr<cuda::CudaDevice::AllocationLease>>
          old_allocation_leases;
      if (state->allocations != signature->allocations) {
        old_allocation_leases = std::move(state->allocation_leases);
        state->allocation_leases = std::move(*allocation_leases);
        state->allocations = signature->allocations;
      }
      state->signature = std::move(signature->entries);
      state->defer_replay_resources(std::move(old_allocation_leases),
                                    std::move(host_arg_buffers));
      driver.graph_launch(state->graph_exec.get(), nullptr);
      state->stats.last_path =
          CompiledGraphExecutionPath::cuda_masked_patched_replay;
      state->stats.last_fallback_reason = CompiledGraphFallbackReason::none;
      if (state->diagnostics_enabled) {
        ++state->stats.masked_patched_replays;
      }
      if (statistics != nullptr) {
        statistics->record_graph_replay();
      }
      return true;
    }
  }

  const bool is_recapture = state->has_captured_once;
  state->retire();
  state->signature = std::move(signature->entries);
  state->allocations = signature->allocations;
  state->allocation_leases = std::move(*allocation_leases);
  state->masked_mode = true;
  state->masked_control_type = control_type;
  state->masked_max_iterations = is_while ? max_iterations : 0;
  state->masked_continue_while_nonzero = continue_while_nonzero;
  state->masked_default_branch = default_branch;
  state->masked_branch_dispatch_counts = branch_dispatch_counts;
  state->masked_selector_allocation = selector_allocation;
  if (state->diagnostics_enabled) {
    ++state->stats.capture_attempts;
    if (is_recapture) {
      ++state->stats.recaptures;
    }
  }
  if (cache.kernels.size() != graph.dispatches.size()) {
    cache.kernels.assign(graph.dispatches.size(), {});
  }

  void *capture_stream = state->ensure_capture_stream();
  try {
    cuda::driver_graph_prepare_mask_latch();
  } catch (...) {
    return structural_fallback();
  }
  driver.malloc(&state->masked_gate, sizeof(std::uint32_t));
  std::size_t branch_index = 0;
  std::size_t branch_end =
      is_while ? graph.dispatches.size()
               : static_cast<std::size_t>(branch_dispatch_counts.front());
  for (std::size_t i = 0; i < graph.dispatches.size(); ++i) {
    while (!is_while && i >= branch_end) {
      ++branch_index;
      branch_end +=
          static_cast<std::size_t>(branch_dispatch_counts[branch_index]);
    }
    const auto &dispatch = graph.dispatches[i];
    TI_ASSERT(dispatch.ti_kernel);
    auto *prog = dispatch.ti_kernel->program;
    auto *launcher = dynamic_cast<cuda::KernelLauncher *>(
        &prog->get_program_impl()->get_kernel_launcher());
    if (launcher == nullptr) {
      return structural_fallback();
    }
    auto *compiled_kernel_data =
        get_or_compile_cached_kernel(dispatch, compile_config, cache.kernels[i],
                                     /*cache_compiled_kernel_data=*/true);
    const auto &llvm_ckd =
        dynamic_cast<const LLVM::CompiledKernelData &>(*compiled_kernel_data);
    auto handle = launcher->register_llvm_kernel_graph_gated(llvm_ckd);
    auto launch_ctx = dispatch.ti_kernel->make_launch_context();
    launch_ctx.append_dispatch_label(dispatch.dispatch_label);
    graph.init_runtime_context(dispatch.symbolic_args, args, launch_ctx);
    prog->resolve_ndarray_launch_context_under_guard(launch_ctx);
    prog->resolve_runtime_storage_launch_context_under_guard(launch_ctx);
    prog->resolve_texture_launch_context_under_guard(launch_ctx);
    CudaGraphCapturePacket capture_packet(capture_stream);
    capture_packet.launcher = launcher;
    const std::uint32_t expected =
        is_while ? 1u : static_cast<std::uint32_t>(branch_index + 1);
    if (!launcher->prepare_cuda_graph_gated_launch(
            handle, launch_ctx, capture_packet.packet, state->masked_gate,
            expected, capture_stream)) {
      driver.stream_synchronize(capture_stream);
      return structural_fallback();
    }
    state->packets.push_back(std::move(capture_packet));
  }

  auto *selector_ptr = selector_device->get_alloc_info(selector_allocation).ptr;
  driver.stream_synchronize(capture_stream);
  auto capture_lock =
      CUDAContext::get_instance().get_graph_capture_lock_guard();
  const auto begin_error = driver.stream_begin_capture.call(
      capture_stream, CU_STREAM_CAPTURE_MODE_RELAXED);
  if (begin_error != CUDA_SUCCESS) {
    return handle_cuda_graph_driver_failure(*state, begin_error,
                                            "masked stream begin capture",
                                            &state->masked_retry);
  }
  CudaStreamCaptureGuard capture_guard(capture_stream);
  try {
    if (is_while) {
      for (int iteration = 0; iteration < max_iterations; ++iteration) {
        cuda::driver_graph_latch_while(selector_ptr, state->masked_gate,
                                       continue_while_nonzero, capture_stream);
        for (const auto &packet : state->packets) {
          packet.launcher->capture_cuda_graph_launch(packet.packet,
                                                     capture_stream);
        }
      }
    } else {
      cuda::driver_graph_latch_branch(
          selector_ptr, state->masked_gate,
          static_cast<std::uint32_t>(control_type),
          static_cast<std::uint32_t>(branch_dispatch_counts.size()),
          default_branch < 0 ? (std::numeric_limits<std::uint32_t>::max)()
                             : static_cast<std::uint32_t>(default_branch),
          capture_stream);
      for (const auto &packet : state->packets) {
        packet.launcher->capture_cuda_graph_launch(packet.packet,
                                                   capture_stream);
      }
    }
  } catch (...) {
    if (state->diagnostics_enabled) {
      ++state->stats.capture_exceptions;
    }
    capture_guard.abort();
    state->retire();
    throw;
  }
  CudaGraphHandle captured_graph;
  const auto end_error = capture_guard.end(captured_graph.put());
  if (end_error != CUDA_SUCCESS || !captured_graph) {
    return handle_cuda_graph_driver_failure(
        *state, end_error, "masked stream end capture", &state->masked_retry);
  }
  const auto instantiate_error = driver.graph_instantiate_with_flags.call(
      state->graph_exec.put(), captured_graph.get(), 0);
  captured_graph.reset();
  if (instantiate_error != CUDA_SUCCESS || !state->graph_exec) {
    return handle_cuda_graph_driver_failure(
        *state, instantiate_error, "masked instantiate", &state->masked_retry);
  }

  state->masked_retry.record_success();
  state->has_captured_once = true;
  state->stats.last_path = CompiledGraphExecutionPath::cuda_masked_capture;
  state->stats.last_fallback_reason = CompiledGraphFallbackReason::none;
  if (state->diagnostics_enabled) {
    ++state->stats.captures;
    ++state->stats.masked_captures;
  }
  if (statistics != nullptr) {
    statistics->record_graph_capture();
    if (is_recapture) {
      statistics->record_graph_recapture();
    }
  }
  driver.graph_launch(state->graph_exec.get(), nullptr);
  return true;
}

// Encode a strict while -> while hierarchy with device-updatable CUDA Graph
// kernel nodes. Each static business dispatch is compiled once. The bounded
// topology repeats only lightweight updater nodes and ordinary payload nodes,
// avoiding the per-dispatch compiler-gated variants used by the legacy route.
bool try_run_cuda_device_update_nested_control_graph(
    const CompiledGraph &graph,
    const CompileConfig &compile_config,
    const std::unordered_map<std::string, IValue> &args,
    CompiledGraphJITCache &cache,
    Program &program,
    Ndarray &outer_selector,
    const std::vector<CompiledGraphNestedInnerControl> &inner_controls,
    std::size_t outer_condition_dispatch_count,
    int outer_max_iterations,
    RuntimeStatistics *statistics) {
  auto *state = get_cuda_graph_state(cache);
  if (state->diagnostics_enabled) {
    state->stats.backend = CompiledGraphBackend::cuda;
    ++state->stats.attempts;
    state->stats.last_driver_error = 0;
  }
  auto structural_fallback = [&]() {
    state->retry.record_structural_failure();
    mark_cuda_graph_fallback(
        *state, CompiledGraphFallbackReason::structural_unsupported, true);
    state->retire();
    return false;
  };

  const std::size_t dispatch_count = graph.dispatches.size();
  const std::size_t final_outer_condition_begin =
      dispatch_count >= outer_condition_dispatch_count
          ? dispatch_count - outer_condition_dispatch_count
          : 0;
  bool ordered_boundaries = outer_condition_dispatch_count != 0 &&
                            outer_condition_dispatch_count < dispatch_count &&
                            !inner_controls.empty() &&
                            inner_controls.size() <= 8;
  bool valid_iteration_limits =
      outer_max_iterations > 0 && outer_max_iterations <= 64;
  std::size_t boundary_cursor = outer_condition_dispatch_count;
  std::size_t single_copy_repeated_dispatches = 0;
  std::size_t repeated_dispatches = 0;
  std::size_t controls_per_outer = 1;
  std::vector<std::array<std::size_t, 3>> boundaries;
  boundaries.reserve(inner_controls.size());
  for (const auto &inner : inner_controls) {
    const std::size_t condition_count =
        inner.body_dispatch_begin >= inner.condition_dispatch_begin
            ? inner.body_dispatch_begin - inner.condition_dispatch_begin
            : 0;
    const std::size_t terminal_condition_begin =
        inner.dispatch_end >= condition_count
            ? inner.dispatch_end - condition_count
            : 0;
    ordered_boundaries =
        ordered_boundaries &&
        boundary_cursor <= inner.condition_dispatch_begin &&
        inner.condition_dispatch_begin < inner.body_dispatch_begin &&
        inner.body_dispatch_begin < terminal_condition_begin &&
        terminal_condition_begin < inner.dispatch_end &&
        inner.dispatch_end <= final_outer_condition_begin;
    valid_iteration_limits =
        valid_iteration_limits && inner.max_iterations > 0 &&
        inner.max_iterations <= 64;
    const std::size_t repeated =
        inner.dispatch_end >= inner.body_dispatch_begin
            ? inner.dispatch_end - inner.body_dispatch_begin
            : 0;
    if (repeated > (std::numeric_limits<std::size_t>::max)() /
                       std::max(inner.max_iterations, 1)) {
      valid_iteration_limits = false;
    } else {
      single_copy_repeated_dispatches += repeated;
      repeated_dispatches +=
          repeated * static_cast<std::size_t>(inner.max_iterations);
      controls_per_outer += static_cast<std::size_t>(inner.max_iterations);
    }
    boundaries.push_back({inner.condition_dispatch_begin,
                          inner.body_dispatch_begin, inner.dispatch_end});
    boundary_cursor = inner.dispatch_end;
  }
  const std::size_t static_dispatches =
      dispatch_count >= outer_condition_dispatch_count +
                            single_copy_repeated_dispatches
          ? dispatch_count - outer_condition_dispatch_count -
                single_copy_repeated_dispatches
          : (std::numeric_limits<std::size_t>::max)();
  const std::size_t per_outer_dispatches =
      static_dispatches <= (std::numeric_limits<std::size_t>::max)() -
                               repeated_dispatches
          ? static_dispatches + repeated_dispatches
          : (std::numeric_limits<std::size_t>::max)();
  const std::size_t encoded_dispatches =
      valid_iteration_limits &&
              per_outer_dispatches <=
                  ((std::numeric_limits<std::size_t>::max)() -
                   outer_condition_dispatch_count) /
                      static_cast<std::size_t>(outer_max_iterations)
          ? outer_condition_dispatch_count +
                static_cast<std::size_t>(outer_max_iterations) *
                    per_outer_dispatches
          : (std::numeric_limits<std::size_t>::max)();
  const std::size_t control_count =
      valid_iteration_limits &&
              controls_per_outer <=
                  (std::numeric_limits<std::size_t>::max)() /
                      static_cast<std::size_t>(outer_max_iterations)
          ? static_cast<std::size_t>(outer_max_iterations) *
                controls_per_outer
          : (std::numeric_limits<std::size_t>::max)();
  constexpr std::size_t kMaxEncodedDispatches = 4096;
  constexpr std::size_t kMaxPredicateGroups = 4096;
  if (compile_config.debug || dispatch_count == 0 || !valid_iteration_limits ||
      !ordered_boundaries ||
      final_outer_condition_begin >= dispatch_count ||
      encoded_dispatches > kMaxEncodedDispatches ||
      control_count > kMaxPredicateGroups ||
      outer_selector.get_nelement() != 1 ||
      outer_selector.get_element_data_type() != PrimitiveType::i32 ||
      outer_selector.owning_program() != &program) {
    return structural_fallback();
  }

  const DeviceAllocation outer_allocation =
      outer_selector.get_device_allocation();
  std::vector<DeviceAllocation> inner_allocations;
  inner_allocations.reserve(inner_controls.size());
  for (const auto &inner : inner_controls) {
    if (inner.predicate == nullptr || inner.predicate->get_nelement() != 1 ||
        inner.predicate->get_element_data_type() != PrimitiveType::i32 ||
        inner.predicate->owning_program() != &program) {
      return structural_fallback();
    }
    const DeviceAllocation allocation =
        inner.predicate->get_device_allocation();
    if (allocation == outer_allocation ||
        std::find(inner_allocations.begin(), inner_allocations.end(),
                  allocation) != inner_allocations.end()) {
      return structural_fallback();
    }
    inner_allocations.push_back(allocation);
  }
  auto *selector_device =
      dynamic_cast<cuda::CudaDevice *>(outer_allocation.device);
  if (selector_device == nullptr ||
      std::any_of(inner_allocations.begin(), inner_allocations.end(),
                  [&](const DeviceAllocation &allocation) {
                    return allocation.device != selector_device;
                  })) {
    return structural_fallback();
  }
  auto &driver = CUDADriver::get_instance();
  const bool graph_symbols = driver.stream_begin_capture.available() &&
                             driver.stream_end_capture.available() &&
                             driver.graph_instantiate_with_flags.available() &&
                             driver.graph_launch.available() &&
                             driver.graph_upload.available() &&
                             driver.graph_destroy.available() &&
                             driver.graph_exec_destroy.available() &&
                             driver.launch_kernel_ex.available();
  std::uint32_t updater_error = CUDA_SUCCESS;
  if (!graph_symbols ||
      !cuda::driver_graph_prepare_bounded_update(&updater_error)) {
    return structural_fallback();
  }

  std::optional<CudaGraphSignatureCandidate> signature;
  try {
    signature = make_cuda_graph_signature(graph, program, args);
  } catch (...) {
    state->retire();
    throw;
  }
  if (!signature.has_value()) {
    return structural_fallback();
  }
  auto add_control_allocation = [&](DeviceAllocation allocation) {
    if (std::find(signature->allocations.begin(), signature->allocations.end(),
                  allocation) == signature->allocations.end()) {
      signature->allocations.push_back(allocation);
    }
  };
  add_control_allocation(outer_allocation);
  for (const DeviceAllocation &allocation : inner_allocations) {
    add_control_allocation(allocation);
  }
  std::sort(
      signature->allocations.begin(), signature->allocations.end(),
      [](const DeviceAllocation &lhs, const DeviceAllocation &rhs) {
        const auto lhs_device = reinterpret_cast<std::uintptr_t>(lhs.device);
        const auto rhs_device = reinterpret_cast<std::uintptr_t>(rhs.device);
        return lhs_device == rhs_device ? lhs.alloc_id < rhs.alloc_id
                                        : lhs_device < rhs_device;
      });
  std::vector<int> inner_max_iterations;
  inner_max_iterations.reserve(inner_controls.size());
  for (const auto &inner : inner_controls) {
    inner_max_iterations.push_back(inner.max_iterations);
  }
  state = select_cuda_graph_replay_slot_by_signature(
      cache, signature->entries,
      [&](const CompiledGraphCudaState &candidate) {
        return candidate.device_update_nested_mode &&
               candidate.nested_inner_boundaries == boundaries &&
               candidate.masked_nested_outer_max_iterations ==
                   outer_max_iterations &&
               candidate.nested_inner_max_iterations == inner_max_iterations &&
               candidate.masked_selector_allocation == outer_allocation &&
               candidate.nested_inner_selector_allocations == inner_allocations;
      });
  if (state->graph_exec && state->signature != signature->entries) {
    state = allocate_cuda_replay_slot_for_miss(cache, state,
                                               signature->entries);
  }
  CUDAContext::get_instance().make_current();
  state->collect_ready_deferred_resources();
  const bool topology_matches =
      state->device_update_nested_mode &&
      state->nested_inner_boundaries == boundaries &&
      state->masked_nested_outer_max_iterations == outer_max_iterations &&
      state->nested_inner_max_iterations == inner_max_iterations &&
      state->masked_selector_allocation == outer_allocation &&
      state->nested_inner_selector_allocations == inner_allocations;
  if (topology_matches && state->graph_exec &&
      state->signature == signature->entries &&
      state->allocations == signature->allocations) {
    driver.graph_launch(state->graph_exec.get(), nullptr);
    state->stats.last_path =
        CompiledGraphExecutionPath::cuda_device_update_nested_replay;
    state->stats.last_fallback_reason = CompiledGraphFallbackReason::none;
    if (state->diagnostics_enabled) {
      ++state->stats.exact_replays;
    }
    if (statistics != nullptr) {
      statistics->record_graph_replay();
    }
    return true;
  }
  // The device-update and masked routes share the executable cache but keep
  // independent retry state. Once the device-update route is disabled, do
  // not retire a healthy masked executable merely because graph_exec is set.
  if ((!topology_matches || !state->graph_exec) &&
      !state->retry.should_attempt()) {
    if (state->diagnostics_enabled) {
      ++state->stats.retry_backoff_fallbacks;
    }
    mark_cuda_graph_fallback(*state,
                             CompiledGraphFallbackReason::retry_backoff);
    return false;
  }

  auto allocation_leases =
      acquire_cuda_graph_allocation_leases(signature->allocations);
  if (!allocation_leases.has_value()) {
    mark_cuda_graph_fallback(*state,
                             CompiledGraphFallbackReason::resource_unavailable);
    return false;
  }
  const bool structurally_compatible =
      topology_matches && state->graph_exec &&
      cuda_graph_signatures_are_structurally_compatible(state->signature,
                                                        signature->entries);
  if (structurally_compatible) {
    std::vector<std::vector<uint8_t>> host_arg_buffers;
    if (patch_cuda_graph_arguments(graph, args, signature->entries, *state,
                                   host_arg_buffers)) {
      std::vector<std::unique_ptr<cuda::CudaDevice::AllocationLease>>
          old_allocation_leases;
      if (state->allocations != signature->allocations) {
        old_allocation_leases = std::move(state->allocation_leases);
        state->allocation_leases = std::move(*allocation_leases);
        state->allocations = signature->allocations;
      }
      state->signature = std::move(signature->entries);
      state->defer_replay_resources(std::move(old_allocation_leases),
                                    std::move(host_arg_buffers));
      driver.graph_launch(state->graph_exec.get(), nullptr);
      state->stats.last_path =
          CompiledGraphExecutionPath::cuda_device_update_nested_patched_replay;
      state->stats.last_fallback_reason = CompiledGraphFallbackReason::none;
      if (state->diagnostics_enabled) {
        ++state->stats.patched_replays;
      }
      if (statistics != nullptr) {
        statistics->record_graph_replay();
      }
      return true;
    }
  }

  const bool is_recapture = state->has_captured_once;
  state->retire();
  state->signature = std::move(signature->entries);
  state->allocations = signature->allocations;
  state->allocation_leases = std::move(*allocation_leases);
  state->device_update_nested_mode = true;
  state->masked_control_type = 3;
  state->masked_selector_allocation = outer_allocation;
  state->nested_inner_selector_allocations = inner_allocations;
  state->nested_inner_boundaries = boundaries;
  state->masked_nested_outer_max_iterations = outer_max_iterations;
  state->nested_inner_max_iterations = inner_max_iterations;
  if (state->diagnostics_enabled) {
    ++state->stats.capture_attempts;
    if (is_recapture) {
      ++state->stats.recaptures;
    }
  }
  if (cache.kernels.size() != dispatch_count) {
    cache.kernels.assign(dispatch_count, {});
  }

  void *capture_stream = state->ensure_capture_stream();
  driver.malloc(&state->masked_gate, sizeof(std::uint32_t));
  state->nested_inner_gates.assign(inner_controls.size(), nullptr);
  for (void *&gate : state->nested_inner_gates) {
    driver.malloc(&gate, sizeof(std::uint32_t));
  }
  driver.malloc(reinterpret_cast<void **>(&state->nested_device_controls),
                control_count * sizeof(cuda::CudaGraphPredicateGroupControl));
  state->nested_host_controls.assign(control_count, {});
  std::vector<std::vector<std::uintptr_t>> control_nodes(control_count);
  auto *outer_ptr = selector_device->get_alloc_info(outer_allocation).ptr;
  std::vector<void *> inner_ptrs;
  inner_ptrs.reserve(inner_allocations.size());
  for (const DeviceAllocation &allocation : inner_allocations) {
    inner_ptrs.push_back(selector_device->get_alloc_info(allocation).ptr);
  }
  std::size_t initialized_control_count = 0;
  for (int outer_iteration = 0; outer_iteration < outer_max_iterations;
       ++outer_iteration) {
    auto &outer_control =
        state->nested_host_controls[initialized_control_count++];
    outer_control.predicate = reinterpret_cast<std::uintptr_t>(outer_ptr);
    outer_control.parent_gate = 0;
    outer_control.gate =
        reinterpret_cast<std::uintptr_t>(state->masked_gate);
    outer_control.continue_while_nonzero = 1;
    outer_control.telemetry_enabled = state->diagnostics_enabled ? 1u : 0u;
    for (std::size_t child = 0; child < inner_controls.size(); ++child) {
      for (int inner_iteration = 0;
           inner_iteration < inner_controls[child].max_iterations;
           ++inner_iteration) {
        auto &control =
            state->nested_host_controls[initialized_control_count++];
        control.predicate =
            reinterpret_cast<std::uintptr_t>(inner_ptrs[child]);
        control.parent_gate =
            reinterpret_cast<std::uintptr_t>(state->masked_gate);
        control.gate = reinterpret_cast<std::uintptr_t>(
            state->nested_inner_gates[child]);
        control.continue_while_nonzero = 1;
        control.telemetry_enabled = state->diagnostics_enabled ? 1u : 0u;
      }
    }
  }
  TI_ASSERT(initialized_control_count == control_count);

  for (std::size_t i = 0; i < dispatch_count; ++i) {
    const auto &dispatch = graph.dispatches[i];
    TI_ASSERT(dispatch.ti_kernel);
    auto *prog = dispatch.ti_kernel->program;
    auto *launcher = dynamic_cast<cuda::KernelLauncher *>(
        &prog->get_program_impl()->get_kernel_launcher());
    if (launcher == nullptr) {
      return structural_fallback();
    }
    auto *compiled_kernel_data =
        get_or_compile_cached_kernel(dispatch, compile_config, cache.kernels[i],
                                     /*cache_compiled_kernel_data=*/true);
    const auto &llvm_ckd =
        dynamic_cast<const LLVM::CompiledKernelData &>(*compiled_kernel_data);
    auto handle = launcher->register_llvm_kernel(llvm_ckd);
    auto launch_ctx = dispatch.ti_kernel->make_launch_context();
    launch_ctx.append_dispatch_label(dispatch.dispatch_label);
    graph.init_runtime_context(dispatch.symbolic_args, args, launch_ctx);
    prog->resolve_ndarray_launch_context_under_guard(launch_ctx);
    prog->resolve_runtime_storage_launch_context_under_guard(launch_ctx);
    prog->resolve_texture_launch_context_under_guard(launch_ctx);
    CudaGraphCapturePacket capture_packet(capture_stream);
    capture_packet.launcher = launcher;
    if (!launcher->prepare_cuda_graph_launch(
            handle, launch_ctx, capture_packet.packet, capture_stream)) {
      driver.stream_synchronize(capture_stream);
      return structural_fallback();
    }
    state->packets.push_back(std::move(capture_packet));
  }
  state->stats.zero_arg_eligible =
      !state->packets.empty() &&
      std::all_of(state->packets.begin(), state->packets.end(),
                  [](const CudaGraphCapturePacket &packet) {
                    return packet.packet.arg_buffer_size == 0 &&
                           packet.packet.device_arg_buffer == nullptr;
                  });

  driver.stream_synchronize(capture_stream);
  auto capture_lock =
      CUDAContext::get_instance().get_graph_capture_lock_guard();
  const auto begin_error = driver.stream_begin_capture.call(
      capture_stream, CU_STREAM_CAPTURE_MODE_RELAXED);
  if (begin_error != CUDA_SUCCESS) {
    return handle_cuda_graph_driver_failure(
        *state, begin_error, "nested device-update stream begin capture");
  }
  CudaStreamCaptureGuard capture_guard(capture_stream);
  auto capture_payload = [&](std::size_t packet_index,
                             std::size_t control_index) {
    std::vector<void *> device_nodes;
    std::uint32_t capture_error = CUDA_SUCCESS;
    if (!state->packets[packet_index]
             .launcher->capture_cuda_graph_updatable_launch(
                 state->packets[packet_index].packet, capture_stream,
                 &device_nodes, &capture_error)) {
      return capture_error == CUDA_SUCCESS ? CUDA_ERROR_NOT_SUPPORTED
                                           : capture_error;
    }
    auto &destination = control_nodes[control_index];
    destination.reserve(destination.size() + device_nodes.size());
    for (void *device_node : device_nodes) {
      destination.push_back(reinterpret_cast<std::uintptr_t>(device_node));
    }
    return static_cast<std::uint32_t>(CUDA_SUCCESS);
  };
  std::uint32_t payload_error = CUDA_SUCCESS;
  try {
    for (std::size_t i = 0; i < outer_condition_dispatch_count; ++i) {
      state->packets[i].launcher->capture_cuda_graph_launch(
          state->packets[i].packet, capture_stream);
    }
    std::size_t control_index = 0;
    for (int outer_iteration = 0; outer_iteration < outer_max_iterations;
         ++outer_iteration) {
      const std::size_t outer_control_index = control_index++;
      cuda::driver_graph_update_predicate_group(
          state->nested_device_controls + outer_control_index, capture_stream);
      std::size_t payload_cursor = outer_condition_dispatch_count;
      for (const auto &inner : inner_controls) {
        for (std::size_t i = payload_cursor;
             i < inner.body_dispatch_begin &&
             payload_error == CUDA_SUCCESS;
             ++i) {
          payload_error = capture_payload(i, outer_control_index);
        }
        for (int inner_iteration = 0;
             inner_iteration < inner.max_iterations &&
             payload_error == CUDA_SUCCESS;
             ++inner_iteration) {
          const std::size_t inner_control_index = control_index++;
          cuda::driver_graph_update_predicate_group(
              state->nested_device_controls + inner_control_index,
              capture_stream);
          for (std::size_t i = inner.body_dispatch_begin;
               i < inner.dispatch_end && payload_error == CUDA_SUCCESS;
               ++i) {
            payload_error = capture_payload(i, inner_control_index);
          }
        }
        payload_cursor = inner.dispatch_end;
      }
      for (std::size_t i = payload_cursor;
           i < dispatch_count && payload_error == CUDA_SUCCESS; ++i) {
        payload_error = capture_payload(i, outer_control_index);
      }
      if (payload_error != CUDA_SUCCESS) {
        break;
      }
    }
  } catch (...) {
    if (state->diagnostics_enabled) {
      ++state->stats.capture_exceptions;
    }
    capture_guard.abort();
    state->retire();
    throw;
  }
  if (payload_error != CUDA_SUCCESS) {
    capture_guard.abort();
    return handle_cuda_graph_driver_failure(
        *state, payload_error, "nested device-update payload capture");
  }
  CudaGraphHandle captured_graph;
  const auto end_error = capture_guard.end(captured_graph.put());
  if (end_error != CUDA_SUCCESS || !captured_graph) {
    return handle_cuda_graph_driver_failure(
        *state, end_error, "nested device-update stream end capture");
  }

  std::size_t node_count = 0;
  for (const auto &nodes : control_nodes) {
    if (nodes.empty() ||
        nodes.size() > (std::numeric_limits<std::uint32_t>::max)() ||
        node_count > (std::numeric_limits<std::size_t>::max)() - nodes.size()) {
      return structural_fallback();
    }
    node_count += nodes.size();
  }
  driver.malloc(reinterpret_cast<void **>(&state->nested_device_nodes),
                node_count * sizeof(std::uintptr_t));
  state->nested_host_nodes.reserve(node_count);
  std::size_t node_offset = 0;
  for (std::size_t i = 0; i < control_nodes.size(); ++i) {
    auto &control = state->nested_host_controls[i];
    control.device_nodes = reinterpret_cast<std::uintptr_t>(
        state->nested_device_nodes + node_offset);
    control.node_count = static_cast<std::uint32_t>(control_nodes[i].size());
    state->nested_host_nodes.insert(state->nested_host_nodes.end(),
                                    control_nodes[i].begin(),
                                    control_nodes[i].end());
    node_offset += control_nodes[i].size();
  }

  const auto instantiate_error = driver.graph_instantiate_with_flags.call(
      state->graph_exec.put(), captured_graph.get(), 0);
  captured_graph.reset();
  if (instantiate_error != CUDA_SUCCESS || !state->graph_exec) {
    return handle_cuda_graph_driver_failure(*state, instantiate_error,
                                            "nested device-update instantiate");
  }
  const auto upload_error =
      driver.graph_upload.call(state->graph_exec.get(), nullptr);
  if (upload_error != CUDA_SUCCESS) {
    return handle_cuda_graph_driver_failure(
        *state, upload_error, "nested device-update graph upload");
  }
  auto control_upload_error = driver.memcpy_host_to_device.call(
      state->nested_device_nodes, state->nested_host_nodes.data(),
      state->nested_host_nodes.size() * sizeof(std::uintptr_t));
  if (control_upload_error == CUDA_SUCCESS) {
    control_upload_error = driver.memcpy_host_to_device.call(
        state->nested_device_controls, state->nested_host_controls.data(),
        state->nested_host_controls.size() *
            sizeof(cuda::CudaGraphPredicateGroupControl));
  }
  if (control_upload_error != CUDA_SUCCESS) {
    return handle_cuda_graph_driver_failure(
        *state, control_upload_error, "nested device-update control upload");
  }

  state->retry.record_success();
  state->has_captured_once = true;
  state->stats.last_path =
      CompiledGraphExecutionPath::cuda_device_update_nested_capture;
  state->stats.last_fallback_reason = CompiledGraphFallbackReason::none;
  if (state->diagnostics_enabled) {
    ++state->stats.captures;
  }
  if (statistics != nullptr) {
    statistics->record_graph_capture();
    if (is_recapture) {
      statistics->record_graph_recapture();
    }
  }
  driver.graph_launch(state->graph_exec.get(), nullptr);
  return true;
}

// Encode a strict while -> while hierarchy into one ordinary CUDA Graph.
// Two device gates preserve parent/child ownership: the outer gate masks the
// complete outer payload, while the inner gate is the conjunction of the
// outer gate and the current inner predicate. This route intentionally uses
// only ordinary graph capture and PTX latches, so it is available before CUDA
// 12.8 and has the same bounded-tail contract as Vulkan nested replay.
bool try_run_cuda_masked_nested_control_graph(
    const CompiledGraph &graph,
    const CompileConfig &compile_config,
    const std::unordered_map<std::string, IValue> &args,
    CompiledGraphJITCache &cache,
    Program &program,
    Ndarray &outer_selector,
    const std::vector<CompiledGraphNestedInnerControl> &inner_controls,
    std::size_t outer_condition_dispatch_count,
    int outer_max_iterations,
    RuntimeStatistics *statistics) {
  auto *state = get_cuda_graph_state(cache);
  if (state->diagnostics_enabled) {
    state->stats.backend = CompiledGraphBackend::cuda;
    ++state->stats.attempts;
    state->stats.last_driver_error = 0;
  }
  auto structural_fallback = [&]() {
    state->masked_retry.record_structural_failure();
    mark_cuda_graph_fallback(
        *state, CompiledGraphFallbackReason::structural_unsupported, true);
    state->retire();
    return false;
  };

  const std::size_t dispatch_count = graph.dispatches.size();
  const std::size_t final_outer_condition_begin =
      dispatch_count >= outer_condition_dispatch_count
          ? dispatch_count - outer_condition_dispatch_count
          : 0;
  bool ordered_boundaries = outer_condition_dispatch_count != 0 &&
                            outer_condition_dispatch_count < dispatch_count &&
                            !inner_controls.empty() &&
                            inner_controls.size() <= 8;
  bool valid_iteration_limits =
      outer_max_iterations > 0 && outer_max_iterations <= 64;
  std::size_t boundary_cursor = outer_condition_dispatch_count;
  std::size_t single_copy_repeated_dispatches = 0;
  std::size_t repeated_dispatches = 0;
  std::vector<std::array<std::size_t, 3>> boundaries;
  boundaries.reserve(inner_controls.size());
  for (const auto &inner : inner_controls) {
    const std::size_t condition_count =
        inner.body_dispatch_begin >= inner.condition_dispatch_begin
            ? inner.body_dispatch_begin - inner.condition_dispatch_begin
            : 0;
    const std::size_t terminal_condition_begin =
        inner.dispatch_end >= condition_count
            ? inner.dispatch_end - condition_count
            : 0;
    ordered_boundaries =
        ordered_boundaries &&
        boundary_cursor <= inner.condition_dispatch_begin &&
        inner.condition_dispatch_begin < inner.body_dispatch_begin &&
        inner.body_dispatch_begin < terminal_condition_begin &&
        terminal_condition_begin < inner.dispatch_end &&
        inner.dispatch_end <= final_outer_condition_begin;
    valid_iteration_limits =
        valid_iteration_limits && inner.max_iterations > 0 &&
        inner.max_iterations <= 64;
    const std::size_t repeated =
        inner.dispatch_end >= inner.body_dispatch_begin
            ? inner.dispatch_end - inner.body_dispatch_begin
            : 0;
    single_copy_repeated_dispatches += repeated;
    repeated_dispatches +=
        repeated * static_cast<std::size_t>(std::max(inner.max_iterations, 1));
    boundaries.push_back({inner.condition_dispatch_begin,
                          inner.body_dispatch_begin, inner.dispatch_end});
    boundary_cursor = inner.dispatch_end;
  }
  const std::size_t static_dispatches =
      dispatch_count >= outer_condition_dispatch_count +
                            single_copy_repeated_dispatches
          ? dispatch_count - outer_condition_dispatch_count -
                single_copy_repeated_dispatches
          : (std::numeric_limits<std::size_t>::max)();
  const std::size_t encoded_dispatches =
      valid_iteration_limits &&
              static_dispatches <= 4096 && repeated_dispatches <= 4096
          ? outer_condition_dispatch_count +
                static_cast<std::size_t>(outer_max_iterations) *
                    (static_dispatches + repeated_dispatches)
          : (std::numeric_limits<std::size_t>::max)();
  constexpr std::size_t kMaxEncodedDispatches = 4096;
  if (compile_config.debug || dispatch_count == 0 || !valid_iteration_limits ||
      !ordered_boundaries ||
      final_outer_condition_begin >= dispatch_count ||
      encoded_dispatches > kMaxEncodedDispatches ||
      outer_selector.get_nelement() != 1 ||
      outer_selector.get_element_data_type() != PrimitiveType::i32 ||
      outer_selector.owning_program() != &program) {
    return structural_fallback();
  }

  const DeviceAllocation outer_allocation =
      outer_selector.get_device_allocation();
  std::vector<DeviceAllocation> inner_allocations;
  inner_allocations.reserve(inner_controls.size());
  for (const auto &inner : inner_controls) {
    if (inner.predicate == nullptr || inner.predicate->get_nelement() != 1 ||
        inner.predicate->get_element_data_type() != PrimitiveType::i32 ||
        inner.predicate->owning_program() != &program) {
      return structural_fallback();
    }
    const DeviceAllocation allocation =
        inner.predicate->get_device_allocation();
    if (allocation == outer_allocation ||
        std::find(inner_allocations.begin(), inner_allocations.end(),
                  allocation) != inner_allocations.end()) {
      return structural_fallback();
    }
    inner_allocations.push_back(allocation);
  }
  auto *selector_device =
      dynamic_cast<cuda::CudaDevice *>(outer_allocation.device);
  if (selector_device == nullptr ||
      std::any_of(inner_allocations.begin(), inner_allocations.end(),
                  [&](const DeviceAllocation &allocation) {
                    return allocation.device != selector_device;
                  })) {
    return structural_fallback();
  }
  auto &driver = CUDADriver::get_instance();
  const bool graph_symbols = driver.stream_begin_capture.available() &&
                             driver.stream_end_capture.available() &&
                             driver.graph_instantiate_with_flags.available() &&
                             driver.graph_launch.available() &&
                             driver.graph_destroy.available() &&
                             driver.graph_exec_destroy.available();
  if (!graph_symbols || !cuda::driver_graph_mask_latch_compiled()) {
    return structural_fallback();
  }

  std::optional<CudaGraphSignatureCandidate> signature;
  try {
    signature = make_cuda_graph_signature(graph, program, args);
  } catch (...) {
    state->retire();
    throw;
  }
  if (!signature.has_value()) {
    return structural_fallback();
  }
  auto add_control_allocation = [&](DeviceAllocation allocation) {
    if (std::find(signature->allocations.begin(), signature->allocations.end(),
                  allocation) == signature->allocations.end()) {
      signature->allocations.push_back(allocation);
    }
  };
  add_control_allocation(outer_allocation);
  for (const DeviceAllocation &allocation : inner_allocations) {
    add_control_allocation(allocation);
  }
  std::sort(
      signature->allocations.begin(), signature->allocations.end(),
      [](const DeviceAllocation &lhs, const DeviceAllocation &rhs) {
        const auto lhs_device = reinterpret_cast<std::uintptr_t>(lhs.device);
        const auto rhs_device = reinterpret_cast<std::uintptr_t>(rhs.device);
        return lhs_device == rhs_device ? lhs.alloc_id < rhs.alloc_id
                                        : lhs_device < rhs_device;
      });
  std::vector<int> inner_max_iterations;
  inner_max_iterations.reserve(inner_controls.size());
  for (const auto &inner : inner_controls) {
    inner_max_iterations.push_back(inner.max_iterations);
  }
  state = select_cuda_graph_replay_slot_by_signature(
      cache, signature->entries,
      [&](const CompiledGraphCudaState &candidate) {
        return candidate.masked_mode && candidate.masked_nested_mode &&
               candidate.nested_inner_boundaries == boundaries &&
               candidate.masked_nested_outer_max_iterations ==
                   outer_max_iterations &&
               candidate.nested_inner_max_iterations == inner_max_iterations &&
               candidate.masked_selector_allocation == outer_allocation &&
               candidate.nested_inner_selector_allocations == inner_allocations;
      });
  if (state->graph_exec && state->signature != signature->entries) {
    state = allocate_cuda_replay_slot_for_miss(cache, state,
                                               signature->entries);
  }
  CUDAContext::get_instance().make_current();
  state->collect_ready_deferred_resources();
  const bool topology_matches =
      state->masked_mode && state->masked_nested_mode &&
      state->nested_inner_boundaries == boundaries &&
      state->masked_nested_outer_max_iterations == outer_max_iterations &&
      state->nested_inner_max_iterations == inner_max_iterations &&
      state->masked_selector_allocation == outer_allocation &&
      state->nested_inner_selector_allocations == inner_allocations;
  if (topology_matches && state->graph_exec &&
      state->signature == signature->entries &&
      state->allocations == signature->allocations) {
    driver.graph_launch(state->graph_exec.get(), nullptr);
    state->stats.last_path = CompiledGraphExecutionPath::cuda_masked_replay;
    state->stats.last_fallback_reason = CompiledGraphFallbackReason::none;
    if (state->diagnostics_enabled) {
      ++state->stats.masked_replays;
    }
    if (statistics != nullptr) {
      statistics->record_graph_replay();
    }
    return true;
  }
  // The high and fallback routes share graph_exec. Respect the fallback
  // route's own retry state even when that slot currently holds a high-route
  // executable.
  if ((!topology_matches || !state->graph_exec) &&
      !state->masked_retry.should_attempt()) {
    if (state->diagnostics_enabled) {
      ++state->stats.retry_backoff_fallbacks;
    }
    mark_cuda_graph_fallback(*state,
                             CompiledGraphFallbackReason::retry_backoff);
    return false;
  }

  auto allocation_leases =
      acquire_cuda_graph_allocation_leases(signature->allocations);
  if (!allocation_leases.has_value()) {
    mark_cuda_graph_fallback(*state,
                             CompiledGraphFallbackReason::resource_unavailable);
    return false;
  }
  const bool structurally_compatible =
      topology_matches && state->graph_exec &&
      cuda_graph_signatures_are_structurally_compatible(state->signature,
                                                        signature->entries);
  if (structurally_compatible) {
    std::vector<std::vector<uint8_t>> host_arg_buffers;
    if (patch_cuda_graph_arguments(graph, args, signature->entries, *state,
                                   host_arg_buffers)) {
      std::vector<std::unique_ptr<cuda::CudaDevice::AllocationLease>>
          old_allocation_leases;
      if (state->allocations != signature->allocations) {
        old_allocation_leases = std::move(state->allocation_leases);
        state->allocation_leases = std::move(*allocation_leases);
        state->allocations = signature->allocations;
      }
      state->signature = std::move(signature->entries);
      state->defer_replay_resources(std::move(old_allocation_leases),
                                    std::move(host_arg_buffers));
      driver.graph_launch(state->graph_exec.get(), nullptr);
      state->stats.last_path =
          CompiledGraphExecutionPath::cuda_masked_patched_replay;
      state->stats.last_fallback_reason = CompiledGraphFallbackReason::none;
      if (state->diagnostics_enabled) {
        ++state->stats.masked_patched_replays;
      }
      if (statistics != nullptr) {
        statistics->record_graph_replay();
      }
      return true;
    }
  }

  const bool is_recapture = state->has_captured_once;
  state->retire();
  state->signature = std::move(signature->entries);
  state->allocations = signature->allocations;
  state->allocation_leases = std::move(*allocation_leases);
  state->masked_mode = true;
  state->masked_nested_mode = true;
  state->masked_control_type = 3;
  state->masked_selector_allocation = outer_allocation;
  state->nested_inner_selector_allocations = inner_allocations;
  state->nested_inner_boundaries = boundaries;
  state->masked_nested_outer_max_iterations = outer_max_iterations;
  state->nested_inner_max_iterations = inner_max_iterations;
  if (state->diagnostics_enabled) {
    ++state->stats.capture_attempts;
    if (is_recapture) {
      ++state->stats.recaptures;
    }
  }
  if (cache.kernels.size() != dispatch_count) {
    cache.kernels.assign(dispatch_count, {});
  }

  void *capture_stream = state->ensure_capture_stream();
  try {
    cuda::driver_graph_prepare_mask_latch();
  } catch (...) {
    return structural_fallback();
  }
  driver.malloc(&state->masked_gate, sizeof(std::uint32_t));
  state->nested_inner_gates.assign(inner_controls.size(), nullptr);
  for (void *&gate : state->nested_inner_gates) {
    driver.malloc(&gate, sizeof(std::uint32_t));
  }
  for (std::size_t i = 0; i < dispatch_count; ++i) {
    const auto &dispatch = graph.dispatches[i];
    TI_ASSERT(dispatch.ti_kernel);
    auto *prog = dispatch.ti_kernel->program;
    auto *launcher = dynamic_cast<cuda::KernelLauncher *>(
        &prog->get_program_impl()->get_kernel_launcher());
    if (launcher == nullptr) {
      return structural_fallback();
    }
    auto *compiled_kernel_data =
        get_or_compile_cached_kernel(dispatch, compile_config, cache.kernels[i],
                                     /*cache_compiled_kernel_data=*/true);
    const auto &llvm_ckd =
        dynamic_cast<const LLVM::CompiledKernelData &>(*compiled_kernel_data);
    const bool initial_outer_condition = i < outer_condition_dispatch_count;
    const auto inner_gate = [&]() -> void * {
      for (std::size_t child = 0; child < inner_controls.size(); ++child) {
        if (i >= inner_controls[child].body_dispatch_begin &&
            i < inner_controls[child].dispatch_end) {
          return state->nested_inner_gates[child];
        }
      }
      return nullptr;
    }();
    auto handle = initial_outer_condition
                      ? launcher->register_llvm_kernel(llvm_ckd)
                      : launcher->register_llvm_kernel_graph_gated(llvm_ckd);
    auto launch_ctx = dispatch.ti_kernel->make_launch_context();
    launch_ctx.append_dispatch_label(dispatch.dispatch_label);
    graph.init_runtime_context(dispatch.symbolic_args, args, launch_ctx);
    prog->resolve_ndarray_launch_context_under_guard(launch_ctx);
    prog->resolve_runtime_storage_launch_context_under_guard(launch_ctx);
    prog->resolve_texture_launch_context_under_guard(launch_ctx);
    CudaGraphCapturePacket capture_packet(capture_stream);
    capture_packet.launcher = launcher;
    const bool prepared =
        initial_outer_condition
            ? launcher->prepare_cuda_graph_launch(
                  handle, launch_ctx, capture_packet.packet, capture_stream)
            : launcher->prepare_cuda_graph_gated_launch(
                  handle, launch_ctx, capture_packet.packet,
                  inner_gate != nullptr ? inner_gate : state->masked_gate,
                  1u, capture_stream);
    if (!prepared) {
      driver.stream_synchronize(capture_stream);
      return structural_fallback();
    }
    state->packets.push_back(std::move(capture_packet));
  }

  auto *outer_ptr = selector_device->get_alloc_info(outer_allocation).ptr;
  std::vector<void *> inner_ptrs;
  inner_ptrs.reserve(inner_allocations.size());
  for (const DeviceAllocation &allocation : inner_allocations) {
    inner_ptrs.push_back(selector_device->get_alloc_info(allocation).ptr);
  }
  driver.stream_synchronize(capture_stream);
  auto capture_lock =
      CUDAContext::get_instance().get_graph_capture_lock_guard();
  const auto begin_error = driver.stream_begin_capture.call(
      capture_stream, CU_STREAM_CAPTURE_MODE_RELAXED);
  if (begin_error != CUDA_SUCCESS) {
    return handle_cuda_graph_driver_failure(
        *state, begin_error, "nested masked stream begin capture",
        &state->masked_retry);
  }
  CudaStreamCaptureGuard capture_guard(capture_stream);
  try {
    for (std::size_t i = 0; i < outer_condition_dispatch_count; ++i) {
      state->packets[i].launcher->capture_cuda_graph_launch(
          state->packets[i].packet, capture_stream);
    }
    for (int outer_iteration = 0; outer_iteration < outer_max_iterations;
         ++outer_iteration) {
      cuda::driver_graph_latch_while(outer_ptr, state->masked_gate, true,
                                     capture_stream);
      std::size_t payload_cursor = outer_condition_dispatch_count;
      for (std::size_t child = 0; child < inner_controls.size(); ++child) {
        const auto &inner = inner_controls[child];
        for (std::size_t i = payload_cursor;
             i < inner.body_dispatch_begin; ++i) {
          state->packets[i].launcher->capture_cuda_graph_launch(
              state->packets[i].packet, capture_stream);
        }
        for (int inner_iteration = 0;
             inner_iteration < inner.max_iterations; ++inner_iteration) {
          cuda::driver_graph_latch_nested_while(
              state->masked_gate, inner_ptrs[child],
              state->nested_inner_gates[child], true, capture_stream);
          for (std::size_t i = inner.body_dispatch_begin;
               i < inner.dispatch_end; ++i) {
            state->packets[i].launcher->capture_cuda_graph_launch(
                state->packets[i].packet, capture_stream);
          }
        }
        payload_cursor = inner.dispatch_end;
      }
      for (std::size_t i = payload_cursor; i < dispatch_count; ++i) {
        state->packets[i].launcher->capture_cuda_graph_launch(
            state->packets[i].packet, capture_stream);
      }
    }
  } catch (...) {
    if (state->diagnostics_enabled) {
      ++state->stats.capture_exceptions;
    }
    capture_guard.abort();
    state->retire();
    throw;
  }
  CudaGraphHandle captured_graph;
  const auto end_error = capture_guard.end(captured_graph.put());
  if (end_error != CUDA_SUCCESS || !captured_graph) {
    return handle_cuda_graph_driver_failure(*state, end_error,
                                            "nested masked stream end capture",
                                            &state->masked_retry);
  }
  const auto instantiate_error = driver.graph_instantiate_with_flags.call(
      state->graph_exec.put(), captured_graph.get(), 0);
  captured_graph.reset();
  if (instantiate_error != CUDA_SUCCESS || !state->graph_exec) {
    return handle_cuda_graph_driver_failure(*state, instantiate_error,
                                            "nested masked instantiate",
                                            &state->masked_retry);
  }

  state->masked_retry.record_success();
  state->has_captured_once = true;
  state->stats.last_path = CompiledGraphExecutionPath::cuda_masked_capture;
  state->stats.last_fallback_reason = CompiledGraphFallbackReason::none;
  if (state->diagnostics_enabled) {
    ++state->stats.captures;
    ++state->stats.masked_captures;
  }
  if (statistics != nullptr) {
    statistics->record_graph_capture();
    if (is_recapture) {
      statistics->record_graph_recapture();
    }
  }
  driver.graph_launch(state->graph_exec.get(), nullptr);
  return true;
}

bool try_run_cuda_bounded_graph(
    const CompiledGraph &graph,
    const CompileConfig &compile_config,
    const std::unordered_map<std::string, IValue> &args,
    CompiledGraphJITCache &cache,
    Program &program,
    Ndarray &predicate,
    int max_iterations,
    bool continue_while_nonzero,
    RuntimeStatistics *statistics) {
  auto *state = get_cuda_graph_state(cache);
  if (state->diagnostics_enabled) {
    state->stats.backend = CompiledGraphBackend::cuda;
    ++state->stats.attempts;
    state->stats.last_driver_error = 0;
  }
  auto structural_fallback = [&]() {
    state->retry.record_structural_failure();
    mark_cuda_graph_fallback(
        *state, CompiledGraphFallbackReason::structural_unsupported, true);
    state->retire();
    return false;
  };
  if (compile_config.debug || graph.dispatches.empty() ||
      max_iterations <= 0 || predicate.get_nelement() != 1 ||
      predicate.get_element_data_type() != PrimitiveType::i32 ||
      predicate.owning_program() != &program) {
    return structural_fallback();
  }

  const DeviceAllocation predicate_allocation =
      predicate.get_device_allocation();
  auto *predicate_device = dynamic_cast<cuda::CudaDevice *>(
      predicate_allocation.device);
  if (predicate_device == nullptr) {
    return structural_fallback();
  }
  auto &driver = CUDADriver::get_instance();
  const bool conditional_symbols =
      driver.stream_begin_capture_to_graph.available() &&
      driver.stream_end_capture.available() &&
      driver.graph_create.available() &&
      driver.graph_conditional_handle_create.available() &&
      driver.graph_add_node.available() &&
      driver.graph_get_nodes.available() &&
      driver.graph_instantiate_with_flags.available() &&
      driver.graph_launch.available() && driver.graph_destroy.available() &&
      driver.graph_exec_destroy.available();
  if (!conditional_symbols ||
      !cuda::driver_graph_conditional_setter_compiled()) {
    return structural_fallback();
  }

  std::optional<CudaGraphSignatureCandidate> signature;
  try {
    signature = make_cuda_graph_signature(graph, program, args);
  } catch (...) {
    state->retire();
    throw;
  }
  if (!signature.has_value()) {
    return structural_fallback();
  }
  state = select_cuda_graph_replay_slot_by_signature(
      cache, signature->entries,
      [](const CompiledGraphCudaState &candidate) {
        return candidate.conditional_mode && candidate.conditional_type == -1;
      });
  if (state->graph_exec && state->signature != signature->entries) {
    state = allocate_cuda_replay_slot_for_miss(cache, state,
                                               signature->entries);
  }
  CUDAContext::get_instance().make_current();
  state->collect_ready_deferred_resources();

  auto update_control = [&]() {
    TI_ASSERT(state->conditional_control != nullptr);
    cuda::CudaGraphConditionalControl control;
    control.predicate = reinterpret_cast<std::uintptr_t>(
        predicate_device->get_alloc_info(predicate_allocation).ptr);
    // The parent-graph setter increments before checking the iteration cap.
    // Wrapping UINT32_MAX to zero makes it the launch-time predicate check;
    // setters captured at the end of the body then advance iterations 1..N.
    control.iteration = ~std::uint32_t{0};
    control.max_iterations = static_cast<std::uint32_t>(max_iterations);
    control.continue_while_nonzero = continue_while_nonzero ? 1u : 0u;
    std::vector<std::vector<uint8_t>> host_buffers(1);
    host_buffers.front().resize(sizeof(control));
    std::memcpy(host_buffers.front().data(), &control, sizeof(control));
    driver.memcpy_host_to_device_async(state->conditional_control,
                                       host_buffers.front().data(),
                                       sizeof(control), nullptr);
    if (state->diagnostics_enabled) {
      ++state->stats.asynchronous_control_updates;
    }
    return host_buffers;
  };
  auto launch = [&](CompiledGraphExecutionPath path) {
    auto host_buffers = update_control();
    try {
      driver.graph_launch(state->graph_exec.get(), nullptr);
    } catch (...) {
      state->defer_replay_resources({}, std::move(host_buffers));
      throw;
    }
    state->defer_replay_resources({}, std::move(host_buffers));
    state->stats.last_path = path;
    state->stats.last_fallback_reason = CompiledGraphFallbackReason::none;
  };

  if (state->conditional_mode && state->graph_exec &&
      state->signature == signature->entries) {
    launch(CompiledGraphExecutionPath::cuda_exact_replay);
    if (state->diagnostics_enabled) {
      ++state->stats.exact_replays;
    }
    if (statistics != nullptr) {
      statistics->record_graph_replay();
    }
    return true;
  }
  if (!state->graph_exec && !state->retry.should_attempt()) {
    if (state->diagnostics_enabled) {
      ++state->stats.retry_backoff_fallbacks;
    }
    mark_cuda_graph_fallback(
        *state, CompiledGraphFallbackReason::retry_backoff);
    return false;
  }

  auto allocation_leases =
      acquire_cuda_graph_allocation_leases(signature->allocations);
  if (!allocation_leases.has_value()) {
    mark_cuda_graph_fallback(
        *state, CompiledGraphFallbackReason::resource_unavailable);
    return false;
  }
  const bool structurally_compatible =
      state->conditional_mode && state->graph_exec &&
      cuda_graph_signatures_are_structurally_compatible(
          state->signature, signature->entries);
  if (structurally_compatible) {
    std::vector<std::vector<uint8_t>> host_arg_buffers;
    if (patch_cuda_graph_arguments(graph, args, signature->entries, *state,
                                   host_arg_buffers)) {
      std::vector<std::unique_ptr<cuda::CudaDevice::AllocationLease>>
          old_allocation_leases;
      if (state->allocations != signature->allocations) {
        old_allocation_leases = std::move(state->allocation_leases);
        state->allocation_leases = std::move(*allocation_leases);
        state->allocations = signature->allocations;
      }
      state->signature = std::move(signature->entries);
      state->defer_replay_resources(std::move(old_allocation_leases),
                                    std::move(host_arg_buffers));
      launch(CompiledGraphExecutionPath::cuda_patched_replay);
      if (state->diagnostics_enabled) {
        ++state->stats.patched_replays;
      }
      if (statistics != nullptr) {
        statistics->record_graph_replay();
      }
      return true;
    }
  }

  const bool is_recapture = state->has_captured_once;
  state->retire();
  state->signature = std::move(signature->entries);
  state->allocations = signature->allocations;
  state->allocation_leases = std::move(*allocation_leases);
  if (state->diagnostics_enabled) {
    ++state->stats.capture_attempts;
    if (is_recapture) {
      ++state->stats.recaptures;
    }
  }
  if (cache.kernels.size() != graph.dispatches.size()) {
    cache.kernels.assign(graph.dispatches.size(), {});
  }

  void *capture_stream = state->ensure_capture_stream();
  for (std::size_t i = 0; i < graph.dispatches.size(); ++i) {
    const auto &dispatch = graph.dispatches[i];
    TI_ASSERT(dispatch.ti_kernel);
    auto *prog = dispatch.ti_kernel->program;
    auto *launcher = dynamic_cast<cuda::KernelLauncher *>(
        &prog->get_program_impl()->get_kernel_launcher());
    if (launcher == nullptr) {
      return structural_fallback();
    }
    auto *compiled_kernel_data =
        get_or_compile_cached_kernel(dispatch, compile_config, cache.kernels[i],
                                     /*cache_compiled_kernel_data=*/true);
    const auto &llvm_ckd =
        dynamic_cast<const LLVM::CompiledKernelData &>(*compiled_kernel_data);
    auto handle = launcher->register_llvm_kernel(llvm_ckd);
    auto launch_ctx = dispatch.ti_kernel->make_launch_context();
    launch_ctx.append_dispatch_label(dispatch.dispatch_label);
    graph.init_runtime_context(dispatch.symbolic_args, args, launch_ctx);
    prog->resolve_ndarray_launch_context_under_guard(launch_ctx);
    prog->resolve_runtime_storage_launch_context_under_guard(launch_ctx);
    prog->resolve_texture_launch_context_under_guard(launch_ctx);
    CudaGraphCapturePacket capture_packet(capture_stream);
    capture_packet.launcher = launcher;
    if (!launcher->prepare_cuda_graph_launch(
            handle, launch_ctx, capture_packet.packet, capture_stream)) {
      driver.stream_synchronize(capture_stream);
      return structural_fallback();
    }
    state->packets.push_back(std::move(capture_packet));
  }
  state->stats.zero_arg_eligible =
      !state->packets.empty() &&
      std::all_of(state->packets.begin(), state->packets.end(),
                  [](const CudaGraphCapturePacket &packet) {
                    return packet.packet.arg_buffer_size == 0 &&
                           packet.packet.device_arg_buffer == nullptr;
                  });

  try {
    cuda::driver_graph_prepare_conditional_setter();
  } catch (...) {
    return structural_fallback();
  }
  driver.malloc(reinterpret_cast<void **>(&state->conditional_control),
                sizeof(cuda::CudaGraphConditionalControl));
  CudaGraphHandle parent_graph;
  const auto create_error = driver.graph_create.call(parent_graph.put(), 0);
  if (create_error != CUDA_SUCCESS || !parent_graph) {
    return handle_cuda_graph_driver_failure(
        *state, create_error, "conditional graph create");
  }
  void *current_context = nullptr;
  const auto context_error =
      driver.context_get_current.call(&current_context);
  if (context_error != CUDA_SUCCESS || current_context == nullptr) {
    return handle_cuda_graph_driver_failure(
        *state, context_error, "conditional context query");
  }

  constexpr unsigned int kAssignDefaultValue = 1;
  std::uint64_t conditional_handle = 0;
  const auto handle_error = driver.graph_conditional_handle_create.call(
      &conditional_handle, parent_graph.get(), current_context, 0,
      kAssignDefaultValue);
  if (handle_error != CUDA_SUCCESS || conditional_handle == 0) {
    return handle_cuda_graph_driver_failure(
        *state, handle_error, "conditional handle create");
  }
  driver.stream_synchronize(capture_stream);
  {
    auto capture_lock =
        CUDAContext::get_instance().get_graph_capture_lock_guard();
    const auto begin_error = driver.stream_begin_capture_to_graph.call(
        capture_stream, parent_graph.get(), nullptr, nullptr, 0,
        CU_STREAM_CAPTURE_MODE_RELAXED);
    if (begin_error != CUDA_SUCCESS) {
      return handle_cuda_graph_driver_failure(
          *state, begin_error, "conditional setter capture begin");
    }
    CudaStreamCaptureGuard capture_guard(capture_stream);
    try {
      cuda::driver_graph_set_conditional(
          state->conditional_control, conditional_handle, capture_stream);
    } catch (...) {
      if (state->diagnostics_enabled) {
        ++state->stats.capture_exceptions;
      }
      capture_guard.abort();
      state->retire();
      throw;
    }
    CUgraph captured_parent = nullptr;
    const auto end_error = capture_guard.end(&captured_parent);
    if (end_error != CUDA_SUCCESS || captured_parent == nullptr) {
      return handle_cuda_graph_driver_failure(
          *state, end_error, "conditional setter capture end");
    }
  }

  std::size_t parent_node_count = 0;
  const auto count_error = driver.graph_get_nodes.call(
      parent_graph.get(), nullptr, &parent_node_count);
  if (count_error != CUDA_SUCCESS || parent_node_count != 1) {
    return handle_cuda_graph_driver_failure(
        *state, count_error, "conditional setter node query");
  }
  void *setter_node = nullptr;
  std::size_t setter_node_count = 1;
  const auto nodes_error = driver.graph_get_nodes.call(
      parent_graph.get(), &setter_node, &setter_node_count);
  if (nodes_error != CUDA_SUCCESS || setter_node_count != 1 ||
      setter_node == nullptr) {
    return handle_cuda_graph_driver_failure(
        *state, nodes_error, "conditional setter node fetch");
  }

  TaichiCudaGraphNodeParams node_params{};
  constexpr std::uint32_t kConditionalNodeType = 13;
  constexpr std::uint32_t kWhileConditionalType = 1;
  node_params.type = kConditionalNodeType;
  node_params.parameters.conditional.handle = conditional_handle;
  node_params.parameters.conditional.type = kWhileConditionalType;
  node_params.parameters.conditional.size = 1;
  node_params.parameters.conditional.ph_graph_out = nullptr;
  node_params.parameters.conditional.context = current_context;
  void *conditional_node = nullptr;
  const auto add_error = driver.graph_add_node.call(
      &conditional_node, parent_graph.get(), &setter_node, 1, &node_params);
  if (add_error != CUDA_SUCCESS || conditional_node == nullptr ||
      node_params.parameters.conditional.ph_graph_out == nullptr ||
      node_params.parameters.conditional.ph_graph_out[0] == nullptr) {
    return handle_cuda_graph_driver_failure(
        *state, add_error, "conditional node create");
  }

  auto capture_lock = CUDAContext::get_instance().get_graph_capture_lock_guard();
  const auto begin_error = driver.stream_begin_capture_to_graph.call(
      capture_stream, node_params.parameters.conditional.ph_graph_out[0],
      nullptr, nullptr, 0, CU_STREAM_CAPTURE_MODE_RELAXED);
  if (begin_error != CUDA_SUCCESS) {
    return handle_cuda_graph_driver_failure(
        *state, begin_error, "conditional body capture begin");
  }
  CudaStreamCaptureGuard capture_guard(capture_stream);
  try {
    for (const auto &packet : state->packets) {
      packet.launcher->capture_cuda_graph_launch(packet.packet,
                                                 capture_stream);
    }
    cuda::driver_graph_set_conditional(
        state->conditional_control, conditional_handle, capture_stream);
  } catch (...) {
    if (state->diagnostics_enabled) {
      ++state->stats.capture_exceptions;
    }
    capture_guard.abort();
    state->retire();
    throw;
  }
  CUgraph captured_child = nullptr;
  const auto end_error = capture_guard.end(&captured_child);
  if (end_error != CUDA_SUCCESS || captured_child == nullptr) {
    return handle_cuda_graph_driver_failure(
        *state, end_error, "conditional body capture end");
  }
  const auto instantiate_error = driver.graph_instantiate_with_flags.call(
      state->graph_exec.put(), parent_graph.get(), 0);
  if (instantiate_error != CUDA_SUCCESS || !state->graph_exec) {
    return handle_cuda_graph_driver_failure(
        *state, instantiate_error, "conditional graph instantiate");
  }

  state->conditional_handle = conditional_handle;
  state->conditional_mode = true;
  state->retry.record_success();
  state->has_captured_once = true;
  state->stats.last_path = CompiledGraphExecutionPath::cuda_capture;
  state->stats.last_fallback_reason = CompiledGraphFallbackReason::none;
  if (state->diagnostics_enabled) {
    ++state->stats.captures;
    if (state->stats.zero_arg_eligible) {
      ++state->stats.zero_arg_captures;
    }
  }
  if (statistics != nullptr) {
    statistics->record_graph_capture();
    if (is_recapture) {
      statistics->record_graph_recapture();
    }
  }
  launch(CompiledGraphExecutionPath::cuda_capture);
  return true;
}

bool try_run_cuda_conditional_graph(
    const CompiledGraph &graph,
    const CompileConfig &compile_config,
    const std::unordered_map<std::string, IValue> &args,
    CompiledGraphJITCache &cache,
    Program &program,
    Ndarray &selector,
    const std::vector<int> &branch_dispatch_counts,
    int conditional_type,
    int default_branch,
    RuntimeStatistics *statistics) {
  auto *state = get_cuda_graph_state(cache);
  if (state->diagnostics_enabled) {
    state->stats.backend = CompiledGraphBackend::cuda;
    ++state->stats.attempts;
    state->stats.last_driver_error = 0;
  }
  auto structural_fallback = [&]() {
    state->retry.record_structural_failure();
    mark_cuda_graph_fallback(
        *state, CompiledGraphFallbackReason::structural_unsupported, true);
    state->retire();
    return false;
  };
  const bool is_if = conditional_type == 0;
  const bool is_switch = conditional_type == 2;
  const std::size_t dispatch_total = std::accumulate(
      branch_dispatch_counts.begin(), branch_dispatch_counts.end(),
      std::size_t{0});
  if (compile_config.debug || graph.dispatches.empty() ||
      branch_dispatch_counts.empty() || dispatch_total != graph.dispatches.size() ||
      std::any_of(branch_dispatch_counts.begin(),
                  branch_dispatch_counts.end(),
                  [](int count) { return count <= 0; }) ||
      (!is_if && !is_switch) ||
      (is_if && branch_dispatch_counts.size() > 2) ||
      (is_switch && default_branch >=
                        static_cast<int>(branch_dispatch_counts.size())) ||
      selector.get_nelement() != 1 ||
      selector.get_element_data_type() != PrimitiveType::i32 ||
      selector.owning_program() != &program) {
    return structural_fallback();
  }

  const DeviceAllocation selector_allocation = selector.get_device_allocation();
  auto *selector_device = dynamic_cast<cuda::CudaDevice *>(
      selector_allocation.device);
  if (selector_device == nullptr) {
    return structural_fallback();
  }
  auto &driver = CUDADriver::get_instance();
  const bool conditional_symbols =
      driver.stream_begin_capture_to_graph.available() &&
      driver.stream_end_capture.available() && driver.graph_create.available() &&
      driver.graph_conditional_handle_create.available() &&
      driver.graph_add_node.available() && driver.graph_get_nodes.available() &&
      driver.graph_instantiate_with_flags.available() &&
      driver.graph_launch.available() && driver.graph_destroy.available() &&
      driver.graph_exec_destroy.available();
  if (!conditional_symbols ||
      !cuda::driver_graph_conditional_setter_compiled()) {
    return structural_fallback();
  }

  std::optional<CudaGraphSignatureCandidate> signature;
  try {
    signature = make_cuda_graph_signature(graph, program, args);
  } catch (...) {
    state->retire();
    throw;
  }
  if (!signature.has_value()) {
    return structural_fallback();
  }
  if (std::find(signature->allocations.begin(), signature->allocations.end(),
                selector_allocation) == signature->allocations.end()) {
    signature->allocations.push_back(selector_allocation);
    std::sort(signature->allocations.begin(), signature->allocations.end(),
              [](const DeviceAllocation &lhs,
                 const DeviceAllocation &rhs) {
                const auto lhs_device =
                    reinterpret_cast<std::uintptr_t>(lhs.device);
                const auto rhs_device =
                    reinterpret_cast<std::uintptr_t>(rhs.device);
                return lhs_device == rhs_device
                           ? lhs.alloc_id < rhs.alloc_id
                           : lhs_device < rhs_device;
              });
  }
  state = select_cuda_graph_replay_slot_by_signature(
      cache, signature->entries,
      [&](const CompiledGraphCudaState &candidate) {
        return candidate.conditional_mode &&
               candidate.conditional_type == conditional_type &&
               candidate.conditional_default_branch == default_branch &&
               candidate.conditional_branch_dispatch_counts ==
                   branch_dispatch_counts;
      });
  if (state->graph_exec && state->signature != signature->entries) {
    state = allocate_cuda_replay_slot_for_miss(cache, state,
                                               signature->entries);
  }
  CUDAContext::get_instance().make_current();
  state->collect_ready_deferred_resources();

  const bool topology_matches =
      state->conditional_mode && state->conditional_type == conditional_type &&
      state->conditional_default_branch == default_branch &&
      state->conditional_branch_dispatch_counts == branch_dispatch_counts;
  auto update_control = [&]() {
    TI_ASSERT(state->conditional_control != nullptr);
    cuda::CudaGraphConditionalControl control;
    control.predicate = reinterpret_cast<std::uintptr_t>(
        selector_device->get_alloc_info(selector_allocation).ptr);
    control.iteration = static_cast<std::uint32_t>(conditional_type);
    control.max_iterations =
        static_cast<std::uint32_t>(branch_dispatch_counts.size());
    control.continue_while_nonzero =
        default_branch < 0 ? ~std::uint32_t{0}
                           : static_cast<std::uint32_t>(default_branch);
    std::vector<std::vector<uint8_t>> host_buffers(1);
    host_buffers.front().resize(sizeof(control));
    std::memcpy(host_buffers.front().data(), &control, sizeof(control));
    driver.memcpy_host_to_device_async(state->conditional_control,
                                       host_buffers.front().data(),
                                       sizeof(control), nullptr);
    if (state->diagnostics_enabled) {
      ++state->stats.asynchronous_control_updates;
    }
    return host_buffers;
  };
  auto launch = [&](CompiledGraphExecutionPath path,
                    bool control_changed = true) {
    if (control_changed) {
      auto host_buffers = update_control();
      try {
        driver.graph_launch(state->graph_exec.get(), nullptr);
      } catch (...) {
        state->defer_replay_resources({}, std::move(host_buffers));
        throw;
      }
      state->defer_replay_resources({}, std::move(host_buffers));
    } else {
      driver.graph_launch(state->graph_exec.get(), nullptr);
    }
    state->stats.last_path = path;
    state->stats.last_fallback_reason = CompiledGraphFallbackReason::none;
  };

  if (topology_matches && state->graph_exec &&
      state->signature == signature->entries &&
      state->allocations == signature->allocations) {
    launch(CompiledGraphExecutionPath::cuda_exact_replay, false);
    if (state->diagnostics_enabled) {
      ++state->stats.exact_replays;
    }
    if (statistics != nullptr) {
      statistics->record_graph_replay();
    }
    return true;
  }
  if (!state->graph_exec && !state->retry.should_attempt()) {
    if (state->diagnostics_enabled) {
      ++state->stats.retry_backoff_fallbacks;
    }
    mark_cuda_graph_fallback(
        *state, CompiledGraphFallbackReason::retry_backoff);
    return false;
  }

  auto allocation_leases =
      acquire_cuda_graph_allocation_leases(signature->allocations);
  if (!allocation_leases.has_value()) {
    mark_cuda_graph_fallback(
        *state, CompiledGraphFallbackReason::resource_unavailable);
    return false;
  }
  const bool structurally_compatible =
      topology_matches && state->graph_exec &&
      cuda_graph_signatures_are_structurally_compatible(
          state->signature, signature->entries);
  if (structurally_compatible) {
    std::vector<std::vector<uint8_t>> host_arg_buffers;
    if (patch_cuda_graph_arguments(graph, args, signature->entries, *state,
                                   host_arg_buffers)) {
      std::vector<std::unique_ptr<cuda::CudaDevice::AllocationLease>>
          old_allocation_leases;
      if (state->allocations != signature->allocations) {
        old_allocation_leases = std::move(state->allocation_leases);
        state->allocation_leases = std::move(*allocation_leases);
        state->allocations = signature->allocations;
      }
      state->signature = std::move(signature->entries);
      state->defer_replay_resources(std::move(old_allocation_leases),
                                    std::move(host_arg_buffers));
      launch(CompiledGraphExecutionPath::cuda_patched_replay);
      if (state->diagnostics_enabled) {
        ++state->stats.patched_replays;
      }
      if (statistics != nullptr) {
        statistics->record_graph_replay();
      }
      return true;
    }
  }

  const bool is_recapture = state->has_captured_once;
  state->retire();
  state->signature = std::move(signature->entries);
  state->allocations = signature->allocations;
  state->allocation_leases = std::move(*allocation_leases);
  if (state->diagnostics_enabled) {
    ++state->stats.capture_attempts;
    if (is_recapture) {
      ++state->stats.recaptures;
    }
  }
  if (cache.kernels.size() != graph.dispatches.size()) {
    cache.kernels.assign(graph.dispatches.size(), {});
  }

  void *capture_stream = state->ensure_capture_stream();
  for (std::size_t i = 0; i < graph.dispatches.size(); ++i) {
    const auto &dispatch = graph.dispatches[i];
    TI_ASSERT(dispatch.ti_kernel);
    auto *prog = dispatch.ti_kernel->program;
    auto *launcher = dynamic_cast<cuda::KernelLauncher *>(
        &prog->get_program_impl()->get_kernel_launcher());
    if (launcher == nullptr) {
      return structural_fallback();
    }
    auto *compiled_kernel_data =
        get_or_compile_cached_kernel(dispatch, compile_config, cache.kernels[i],
                                     /*cache_compiled_kernel_data=*/true);
    const auto &llvm_ckd =
        dynamic_cast<const LLVM::CompiledKernelData &>(*compiled_kernel_data);
    auto handle = launcher->register_llvm_kernel(llvm_ckd);
    auto launch_ctx = dispatch.ti_kernel->make_launch_context();
    launch_ctx.append_dispatch_label(dispatch.dispatch_label);
    graph.init_runtime_context(dispatch.symbolic_args, args, launch_ctx);
    prog->resolve_ndarray_launch_context_under_guard(launch_ctx);
    prog->resolve_runtime_storage_launch_context_under_guard(launch_ctx);
    prog->resolve_texture_launch_context_under_guard(launch_ctx);
    CudaGraphCapturePacket capture_packet(capture_stream);
    capture_packet.launcher = launcher;
    if (!launcher->prepare_cuda_graph_launch(
            handle, launch_ctx, capture_packet.packet, capture_stream)) {
      driver.stream_synchronize(capture_stream);
      return structural_fallback();
    }
    state->packets.push_back(std::move(capture_packet));
  }
  state->stats.zero_arg_eligible =
      !state->packets.empty() &&
      std::all_of(state->packets.begin(), state->packets.end(),
                  [](const CudaGraphCapturePacket &packet) {
                    return packet.packet.arg_buffer_size == 0 &&
                           packet.packet.device_arg_buffer == nullptr;
                  });

  try {
    cuda::driver_graph_prepare_conditional_setter();
  } catch (...) {
    return structural_fallback();
  }
  driver.malloc(reinterpret_cast<void **>(&state->conditional_control),
                sizeof(cuda::CudaGraphConditionalControl));
  CudaGraphHandle parent_graph;
  const auto create_error = driver.graph_create.call(parent_graph.put(), 0);
  if (create_error != CUDA_SUCCESS || !parent_graph) {
    return handle_cuda_graph_driver_failure(
        *state, create_error, "branch graph create");
  }
  void *current_context = nullptr;
  const auto context_error =
      driver.context_get_current.call(&current_context);
  if (context_error != CUDA_SUCCESS || current_context == nullptr) {
    return handle_cuda_graph_driver_failure(
        *state, context_error, "branch context query");
  }

  constexpr unsigned int kAssignDefaultValue = 1;
  std::uint64_t conditional_handle = 0;
  const auto handle_error = driver.graph_conditional_handle_create.call(
      &conditional_handle, parent_graph.get(), current_context, 0,
      kAssignDefaultValue);
  if (handle_error != CUDA_SUCCESS || conditional_handle == 0) {
    return handle_cuda_graph_driver_failure(
        *state, handle_error, "branch handle create");
  }
  driver.stream_synchronize(capture_stream);
  {
    auto capture_lock =
        CUDAContext::get_instance().get_graph_capture_lock_guard();
    const auto begin_error = driver.stream_begin_capture_to_graph.call(
        capture_stream, parent_graph.get(), nullptr, nullptr, 0,
        CU_STREAM_CAPTURE_MODE_RELAXED);
    if (begin_error != CUDA_SUCCESS) {
      return handle_cuda_graph_driver_failure(
          *state, begin_error, "branch setter capture begin");
    }
    CudaStreamCaptureGuard capture_guard(capture_stream);
    try {
      cuda::driver_graph_set_branch_conditional(
          state->conditional_control, conditional_handle, capture_stream);
    } catch (...) {
      if (state->diagnostics_enabled) {
        ++state->stats.capture_exceptions;
      }
      capture_guard.abort();
      state->retire();
      throw;
    }
    CUgraph captured_parent = nullptr;
    const auto end_error = capture_guard.end(&captured_parent);
    if (end_error != CUDA_SUCCESS || captured_parent == nullptr) {
      return handle_cuda_graph_driver_failure(
          *state, end_error, "branch setter capture end");
    }
  }

  std::size_t parent_node_count = 0;
  const auto count_error = driver.graph_get_nodes.call(
      parent_graph.get(), nullptr, &parent_node_count);
  if (count_error != CUDA_SUCCESS || parent_node_count != 1) {
    return handle_cuda_graph_driver_failure(
        *state, count_error, "branch setter node query");
  }
  void *setter_node = nullptr;
  std::size_t setter_node_count = 1;
  const auto nodes_error = driver.graph_get_nodes.call(
      parent_graph.get(), &setter_node, &setter_node_count);
  if (nodes_error != CUDA_SUCCESS || setter_node_count != 1 ||
      setter_node == nullptr) {
    return handle_cuda_graph_driver_failure(
        *state, nodes_error, "branch setter node fetch");
  }

  TaichiCudaGraphNodeParams node_params{};
  constexpr std::uint32_t kConditionalNodeType = 13;
  node_params.type = kConditionalNodeType;
  node_params.parameters.conditional.handle = conditional_handle;
  node_params.parameters.conditional.type =
      static_cast<std::uint32_t>(conditional_type);
  node_params.parameters.conditional.size =
      static_cast<std::uint32_t>(branch_dispatch_counts.size());
  node_params.parameters.conditional.ph_graph_out = nullptr;
  node_params.parameters.conditional.context = current_context;
  void *conditional_node = nullptr;
  const auto add_error = driver.graph_add_node.call(
      &conditional_node, parent_graph.get(), &setter_node, 1, &node_params);
  if (add_error != CUDA_SUCCESS || conditional_node == nullptr ||
      node_params.parameters.conditional.ph_graph_out == nullptr) {
    return handle_cuda_graph_driver_failure(
        *state, add_error, "branch conditional node create");
  }

  std::size_t packet_offset = 0;
  for (std::size_t branch = 0; branch < branch_dispatch_counts.size();
       ++branch) {
    CUgraph child = node_params.parameters.conditional.ph_graph_out[branch];
    if (child == nullptr) {
      return structural_fallback();
    }
    auto capture_lock =
        CUDAContext::get_instance().get_graph_capture_lock_guard();
    const auto begin_error = driver.stream_begin_capture_to_graph.call(
        capture_stream, child, nullptr, nullptr, 0,
        CU_STREAM_CAPTURE_MODE_RELAXED);
    if (begin_error != CUDA_SUCCESS) {
      return handle_cuda_graph_driver_failure(
          *state, begin_error, "branch body capture begin");
    }
    CudaStreamCaptureGuard capture_guard(capture_stream);
    try {
      const std::size_t packet_end =
          packet_offset + branch_dispatch_counts[branch];
      for (; packet_offset < packet_end; ++packet_offset) {
        const auto &packet = state->packets[packet_offset];
        packet.launcher->capture_cuda_graph_launch(packet.packet,
                                                   capture_stream);
      }
    } catch (...) {
      if (state->diagnostics_enabled) {
        ++state->stats.capture_exceptions;
      }
      capture_guard.abort();
      state->retire();
      throw;
    }
    CUgraph captured_child = nullptr;
    const auto end_error = capture_guard.end(&captured_child);
    if (end_error != CUDA_SUCCESS || captured_child == nullptr) {
      return handle_cuda_graph_driver_failure(
          *state, end_error, "branch body capture end");
    }
  }
  const auto instantiate_error = driver.graph_instantiate_with_flags.call(
      state->graph_exec.put(), parent_graph.get(), 0);
  if (instantiate_error != CUDA_SUCCESS || !state->graph_exec) {
    return handle_cuda_graph_driver_failure(
        *state, instantiate_error, "branch graph instantiate");
  }

  state->conditional_handle = conditional_handle;
  state->conditional_mode = true;
  state->conditional_type = conditional_type;
  state->conditional_default_branch = default_branch;
  state->conditional_branch_dispatch_counts = branch_dispatch_counts;
  state->retry.record_success();
  state->has_captured_once = true;
  state->stats.last_path = CompiledGraphExecutionPath::cuda_capture;
  state->stats.last_fallback_reason = CompiledGraphFallbackReason::none;
  if (state->diagnostics_enabled) {
    ++state->stats.captures;
    if (state->stats.zero_arg_eligible) {
      ++state->stats.zero_arg_captures;
    }
  }
  if (statistics != nullptr) {
    statistics->record_graph_capture();
    if (is_recapture) {
      statistics->record_graph_recapture();
    }
  }
  launch(CompiledGraphExecutionPath::cuda_capture);
  return true;
}

}  // namespace

#endif

#if !defined(TI_WITH_CUDA)
void CompiledGraphCudaStateDeleter::operator()(
    CompiledGraphCudaState *state) const noexcept {
  TI_ASSERT(state == nullptr);
}
#endif

#if defined(TI_WITH_VULKAN)
struct VulkanGraphArgumentSignatureEntry {
  std::string name;
  ArgKind tag{ArgKind::kUnknown};
  uint64_t scalar{0};
  std::vector<uint8_t> matrix;
};

struct VulkanGraphLaunchSlot {
  uint64_t binding_plan_revision{0};
  std::size_t argument_count{0};
  std::vector<VulkanGraphArgumentSignatureEntry> argument_signature;
  std::vector<std::unique_ptr<LaunchContextBuilder>> launch_contexts;
  std::vector<gfx::GfxRuntime::GraphDispatch> dispatches;
};

struct CompiledGraphVulkanState {
  std::unique_ptr<gfx::GraphReplayRegistration> registration;
  gfx::GfxRuntime *runtime{nullptr};
  // Host-side launch contexts are much cheaper than backend command slots.
  // Four lazy entries cover ping-pong and triple-buffer binding sets while
  // preserving a hard ownership bound.
  std::vector<VulkanGraphLaunchSlot> launch_slots;
  bool diagnostics_enabled{false};
};

void CompiledGraphVulkanStateDeleter::operator()(
    CompiledGraphVulkanState *state) const noexcept {
  delete state;
}

namespace {

CompiledGraphVulkanState *get_vulkan_graph_state(
    CompiledGraphJITCache &cache,
    gfx::GfxRuntime *runtime) {
  if (cache.vulkan_graph_state &&
      !runtime->owns_graph_replay_registration(
          *cache.vulkan_graph_state->registration)) {
    cache.vulkan_graph_state.reset();
  }
  if (!cache.vulkan_graph_state) {
    auto state = std::make_unique<CompiledGraphVulkanState>();
    state->registration =
        runtime->register_graph_replay(cache.graph_replay_token());
    state->runtime = runtime;
    cache.vulkan_graph_state.reset(state.release());
  }
  if (cache.graph_diagnostics_enabled &&
      !cache.vulkan_graph_state->diagnostics_enabled) {
    // A report may be requested before the first Vulkan run, before a runtime
    // replay state exists. Touching the registration here enables detailed
    // counters before this very launch instead of dropping the first sample.
    cache.vulkan_graph_state->registration->debug_stats();
    cache.vulkan_graph_state->diagnostics_enabled = true;
  }
  return cache.vulkan_graph_state.get();
}

DevicePtr resolve_vulkan_indirect_dispatch_packet(
    const CompiledDispatch &dispatch,
    const std::unordered_map<std::string, IValue> &args,
    Program *program) {
  TI_ASSERT(dispatch.indirect_dispatch_arg.has_value());
  const auto &symbolic = *dispatch.indirect_dispatch_arg;
  const auto found = args.find(symbolic.name);
  TI_ERROR_IF(found == args.end(),
              "Missing runtime value for indirect dispatch packet {}",
              symbolic.name);
  const IValue &value = found->second;
  TI_ERROR_IF(value.tag != ArgKind::kNdarray || value.val == 0,
              "Graph indirect dispatch packet {} must be backed by a Taichi "
              "u32 ndarray; field and external storage views are not "
              "supported",
              symbolic.name);
  auto *packet = reinterpret_cast<Ndarray *>(value.val);
  TI_ERROR_IF(packet == nullptr,
              "Graph received a null indirect dispatch packet {}",
              symbolic.name);
  TI_ERROR_IF(packet->owning_program() != program,
              "Graph indirect dispatch packet {} belongs to a different "
              "Taichi runtime",
              symbolic.name);
  TI_ERROR_IF(
      packet->shape.size() != 1 || packet->get_nelement() < 3 ||
          !packet->get_element_shape().empty() ||
          !packet->get_element_data_type()->is_primitive(
              PrimitiveTypeID::u32),
      "Graph indirect dispatch packet {} must contain at least three scalar "
      "u32 values",
      symbolic.name);
  return packet->get_device_allocation().get_ptr();
}

struct PreparedVulkanGraphLaunch {
  std::vector<std::unique_ptr<LaunchContextBuilder>> launch_contexts;
  std::vector<gfx::GfxRuntime::GraphDispatch> dispatches;
  const std::vector<gfx::GfxRuntime::GraphDispatch> *dispatch_view{nullptr};
  gfx::GfxRuntime *runtime{nullptr};
  std::uint64_t replay_key{0};
};

std::optional<std::vector<VulkanGraphArgumentSignatureEntry>>
make_vulkan_graph_argument_signature(
    const CompiledGraph &graph,
    const std::unordered_map<std::string, IValue> &args) {
  std::vector<VulkanGraphArgumentSignatureEntry> signature;
  for (const auto &[name, value] : args) {
    if (value.tag == ArgKind::kNdarray || value.tag == ArgKind::kTexture) {
      continue;
    }
    const auto declared = graph.args.find(name);
    if (declared == graph.args.end() || declared->second.tag != value.tag) {
      return std::nullopt;
    }
    VulkanGraphArgumentSignatureEntry entry;
    entry.name = name;
    entry.tag = value.tag;
    if (value.tag == ArgKind::kScalar) {
      entry.scalar = value.val;
    } else if (value.tag == ArgKind::kMatrix) {
      auto *matrix = reinterpret_cast<Matrix *>(value.val);
      if (matrix == nullptr) {
        return std::nullopt;
      }
      const std::size_t bytes =
          matrix->length() * data_type_size(matrix->dtype());
      entry.matrix.resize(bytes);
      std::memcpy(entry.matrix.data(),
                  reinterpret_cast<const void *>(matrix->data()), bytes);
    } else {
      return std::nullopt;
    }
    signature.push_back(std::move(entry));
  }
  std::sort(signature.begin(), signature.end(), [](const auto &lhs,
                                                    const auto &rhs) {
    return lhs.name < rhs.name;
  });
  return signature;
}

bool vulkan_graph_arguments_match_cached_signature(
    const CompiledGraph &graph,
    const std::unordered_map<std::string, IValue> &args,
    const VulkanGraphLaunchSlot &slot) {
  if (args.size() != slot.argument_count) {
    return false;
  }
  for (const auto &entry : slot.argument_signature) {
    const auto value = args.find(entry.name);
    const auto declared = graph.args.find(entry.name);
    if (value == args.end() || declared == graph.args.end() ||
        value->second.tag != entry.tag ||
        declared->second.tag != entry.tag) {
      return false;
    }
    if (entry.tag == ArgKind::kScalar) {
      if (value->second.val != entry.scalar) {
        return false;
      }
    } else if (entry.tag == ArgKind::kMatrix) {
      auto *matrix = reinterpret_cast<Matrix *>(value->second.val);
      const std::size_t bytes =
          matrix == nullptr
              ? 0
              : matrix->length() * data_type_size(matrix->dtype());
      if (matrix == nullptr || bytes != entry.matrix.size() ||
          std::memcmp(reinterpret_cast<const void *>(matrix->data()),
                      entry.matrix.data(), bytes) != 0) {
        return false;
      }
    }
  }
  return true;
}

bool vulkan_binding_plan_supports_stable_launch_context(
    const std::unordered_map<std::string, IValue> &args) {
  for (const auto &[name, value] : args) {
    (void)name;
    if (value.tag != ArgKind::kNdarray || value.runtime_storage == nullptr) {
      continue;
    }
    if (value.runtime_storage->descriptor().owner().kind !=
        storage::StorageOwnerKind::kProgramNdarray) {
      return false;
    }
  }
  return true;
}

bool prepare_vulkan_graph_launch(
    const CompiledGraph &graph,
    const CompileConfig &compile_config,
    const std::unordered_map<std::string, IValue> &args,
    CompiledGraphJITCache &cache,
    PreparedVulkanGraphLaunch &prepared) {
  if (cache.kernels.size() != graph.dispatches.size()) {
    cache.kernels.assign(graph.dispatches.size(), {});
  }
  // Keep signature lookup on the production hot path free of diagnostic
  // clocks. Explicit submission telemetry owns measured executions.
  constexpr bool attribute = false;
  auto *existing_state = cache.vulkan_graph_state.get();
  const bool stable_replay = cache.stable_replay_optimization_enabled.load(
      std::memory_order_relaxed);
  Program *graph_program = jit_graph_program(graph);
  uint64_t binding_plan_revision = 0;
  if (stable_replay && graph_program != nullptr &&
      graph_has_runtime_resource_declarations(args)) {
    // Structured Vulkan callers share this preparation path but do not pass
    // through jit_run_cached(). Keep their resource generation in the same
    // revision domain without double-counting ordinary replay attribution.
    binding_plan_revision =
        prepare_runtime_binding_plan(cache, args, graph_program,
                                     /*attribute=*/false)
            .revision;
  }
  if (stable_replay && existing_state != nullptr) {
    const auto signature_begin =
        attribute ? ReplayClock::now() : ReplayClock::time_point{};
    std::size_t matching_slot = existing_state->launch_slots.size();
    for (std::size_t index = 0; index < existing_state->launch_slots.size();
         ++index) {
      const auto &slot = existing_state->launch_slots[index];
      if (slot.binding_plan_revision == binding_plan_revision &&
          !slot.dispatches.empty() &&
          vulkan_graph_arguments_match_cached_signature(graph, args, slot)) {
        matching_slot = index;
        break;
      }
    }
    if (attribute) {
      cache.replay_attribution.signature_ns +=
          replay_elapsed_ns(signature_begin);
      if (matching_slot != existing_state->launch_slots.size()) {
        ++cache.replay_attribution.signature_hits;
      } else {
        ++cache.replay_attribution.signature_misses;
      }
    }
    if (matching_slot != existing_state->launch_slots.size()) {
      if (matching_slot != 0) {
        auto slot = std::move(existing_state->launch_slots[matching_slot]);
        existing_state->launch_slots.erase(
            existing_state->launch_slots.begin() + matching_slot);
        existing_state->launch_slots.insert(
            existing_state->launch_slots.begin(), std::move(slot));
      }
      prepared.runtime = existing_state->runtime;
      prepared.replay_key = existing_state->registration->replay_key();
      prepared.dispatch_view = &existing_state->launch_slots.front().dispatches;
      return true;
    }
  }
  prepared.launch_contexts.reserve(graph.dispatches.size());
  prepared.dispatches.reserve(graph.dispatches.size());

  gfx::KernelLauncher *gfx_launcher = nullptr;
  for (std::size_t i = 0; i < graph.dispatches.size(); ++i) {
    const auto &dispatch = graph.dispatches[i];
    TI_ASSERT(dispatch.ti_kernel);
    auto *prog = dispatch.ti_kernel->program;
    auto *launcher = dynamic_cast<gfx::KernelLauncher *>(
        &prog->get_kernel_launcher());
    if (launcher == nullptr) {
      return false;
    }
    if (gfx_launcher == nullptr) {
      gfx_launcher = launcher;
    } else if (gfx_launcher != launcher) {
      return false;
    }

    const CompiledKernelData *compiled_kernel_data =
        get_or_compile_cached_kernel(dispatch, compile_config, cache.kernels[i],
                                     /*cache_compiled_kernel_data=*/false);
    auto handle = launcher->get_or_register_kernel(*compiled_kernel_data);
    prepared.launch_contexts.push_back(
        std::make_unique<LaunchContextBuilder>(
            dispatch.ti_kernel->make_launch_context()));
    prepared.launch_contexts.back()->append_dispatch_label(
        dispatch.dispatch_label);
    graph.init_runtime_context(dispatch.symbolic_args, args,
                               *prepared.launch_contexts.back());
    prog->resolve_ndarray_launch_context_under_guard(
        *prepared.launch_contexts.back());
    prog->resolve_runtime_storage_launch_context_under_guard(
        *prepared.launch_contexts.back());
    prog->resolve_texture_launch_context_under_guard(
        *prepared.launch_contexts.back());
    DevicePtr indirect_dispatch = kDeviceNullPtr;
    if (dispatch.indirect_dispatch_arg.has_value()) {
      indirect_dispatch = resolve_vulkan_indirect_dispatch_packet(
          dispatch, args, prog);
    }
    prepared.dispatches.push_back(
        {handle, prepared.launch_contexts.back().get(), indirect_dispatch,
         static_cast<std::uint32_t>(
             std::max<std::size_t>(1, dispatch.source_dispatches.size()))});
  }

  if (gfx_launcher == nullptr) {
    return false;
  }
  prepared.runtime = gfx_launcher->runtime();
  auto *state = get_vulkan_graph_state(cache, prepared.runtime);
  prepared.replay_key = state->registration->replay_key();
  const bool cache_launch =
      stable_replay &&
      vulkan_binding_plan_supports_stable_launch_context(args);
  if (cache_launch) {
    const auto signature_begin =
        attribute ? ReplayClock::now() : ReplayClock::time_point{};
    auto signature = make_vulkan_graph_argument_signature(graph, args);
    if (attribute) {
      cache.replay_attribution.signature_ns +=
          replay_elapsed_ns(signature_begin);
    }
    if (signature.has_value()) {
      VulkanGraphLaunchSlot slot;
      slot.binding_plan_revision = binding_plan_revision;
      slot.argument_count = args.size();
      slot.argument_signature = std::move(*signature);
      slot.launch_contexts = std::move(prepared.launch_contexts);
      slot.dispatches = std::move(prepared.dispatches);
      state->launch_slots.insert(state->launch_slots.begin(), std::move(slot));
      constexpr std::size_t kVulkanLaunchSlotCapacity = 4;
      if (state->launch_slots.size() > kVulkanLaunchSlotCapacity) {
        state->launch_slots.resize(kVulkanLaunchSlotCapacity);
      }
      prepared.dispatch_view = &state->launch_slots.front().dispatches;
      return true;
    }
  }
  prepared.dispatch_view = &prepared.dispatches;
  return true;
}

bool try_run_vulkan_graph(const CompiledGraph &graph,
                          const CompileConfig &compile_config,
                          const std::unordered_map<std::string, IValue> &args,
                          CompiledGraphJITCache &cache,
                          RuntimeStatistics *statistics,
                          const gfx::GfxRuntime::GraphStructuredControl
                              *structured_control = nullptr,
                          gfx::GfxRuntime::GraphStructuredResult
                              *structured_result = nullptr,
                          const gfx::GfxRuntime::GraphNestedStructuredControl
                              *nested_control = nullptr,
                          gfx::GfxRuntime::GraphNestedStructuredResult
                              *nested_result = nullptr) {
  if (compile_config.debug) {
    auto &stats = cache.vulkan_inline_stats;
    stats.backend = CompiledGraphBackend::vulkan;
    stats.last_path = CompiledGraphExecutionPath::ordinary_fallback;
    stats.last_fallback_reason = CompiledGraphFallbackReason::debug_mode;
    if (cache.graph_diagnostics_enabled) {
      ++stats.attempts;
      ++stats.ordinary_fallbacks;
    }
    return false;
  }
  const bool composed_single_dispatch =
      graph.dispatches.size() == 1 &&
      graph.dispatches.front().source_dispatches.size() > 1;
  if (graph.dispatches.size() <= 1 &&
      !graph.has_indirect_dispatches() && !composed_single_dispatch) {
    auto &stats = cache.vulkan_inline_stats;
    stats.backend = CompiledGraphBackend::vulkan;
    stats.last_path = CompiledGraphExecutionPath::ordinary_fallback;
    stats.last_fallback_reason =
        CompiledGraphFallbackReason::insufficient_dispatches;
    if (cache.graph_diagnostics_enabled) {
      ++stats.attempts;
      ++stats.ordinary_fallbacks;
    }
    return false;
  }
  PreparedVulkanGraphLaunch prepared;
  if (!prepare_vulkan_graph_launch(
          graph, compile_config, args, cache, prepared)) {
    return false;
  }
  return prepared.runtime->try_launch_graph(
      *prepared.dispatch_view, prepared.replay_key, statistics,
      structured_control, structured_result, nested_control, nested_result);
}

}  // namespace
#endif

#if !defined(TI_WITH_VULKAN)
void CompiledGraphVulkanStateDeleter::operator()(
    CompiledGraphVulkanState *state) const noexcept {
  TI_ASSERT(state == nullptr);
}
#endif

CompiledGraphDebugSnapshot CompiledGraphJITCache::debug_graph_stats() {
  std::lock_guard<std::mutex> lock(run_mutex);
  const bool diagnostics_previously_enabled = graph_diagnostics_enabled;
  std::uint32_t backend_replay_signature_slots = 0;
  std::uint32_t backend_replay_signature_slot_capacity = 0;
  auto finalize = [&](CompiledGraphStats result) {
    CompiledGraphDebugSnapshot snapshot;
    snapshot.stats = result;
    snapshot.diagnostics_previously_enabled =
        diagnostics_previously_enabled;
    const bool execution_observed =
        result.last_path != CompiledGraphExecutionPath::none ||
        result.last_fallback_reason != CompiledGraphFallbackReason::none;
    snapshot.diagnostics_counters_complete =
        graph_diagnostics_enabled ? graph_diagnostics_counters_complete
                                  : !execution_observed;
    snapshot.replay_attribution_enabled = false;
    snapshot.replay_calls = replay_attribution.calls;
    snapshot.replay_total_ns = replay_attribution.total_ns;
    snapshot.replay_snode_guard_ns = replay_attribution.snode_guard_ns;
    snapshot.replay_resource_guard_ns = replay_attribution.resource_guard_ns;
    snapshot.replay_cuda_submission_lock_ns =
        replay_attribution.cuda_submission_lock_ns;
    snapshot.replay_cache_wait_ns = replay_attribution.cache_wait_ns;
    snapshot.replay_binding_plan_ns = replay_attribution.binding_plan_ns;
    snapshot.replay_resource_retain_ns =
        replay_attribution.resource_retain_ns;
    snapshot.replay_snode_validation_ns =
        replay_attribution.snode_validation_ns;
    snapshot.replay_backend_ns = replay_attribution.backend_ns;
    snapshot.replay_signature_ns = replay_attribution.signature_ns;
    snapshot.replay_binding_plan_hits =
        replay_attribution.binding_plan_hits;
    snapshot.replay_binding_plan_misses =
        replay_attribution.binding_plan_misses;
    snapshot.replay_signature_hits = replay_attribution.signature_hits;
    snapshot.replay_signature_misses = replay_attribution.signature_misses;
    snapshot.replay_snode_guard_acquisitions =
        replay_attribution.snode_guard_acquisitions;
    snapshot.replay_snode_guard_elisions =
        replay_attribution.snode_guard_elisions;
    snapshot.runtime_binding_plan_slots =
        static_cast<std::uint32_t>(runtime_binding_plans.size());
    snapshot.backend_replay_signature_slots =
        backend_replay_signature_slots;
    snapshot.backend_replay_signature_slot_capacity =
        backend_replay_signature_slot_capacity;
    for (const auto &kernel : kernels) {
      if (kernel.task_count ==
          std::numeric_limits<std::uint32_t>::max()) {
        continue;
      }
      ++snapshot.known_compiled_dispatches;
      snapshot.known_compiled_tasks += kernel.task_count;
    }
    return snapshot;
  };
#if defined(TI_WITH_CUDA)
  if (cuda_graph_state) {
    if (graph_diagnostics_enabled) {
      cuda_graph_state->diagnostics_enabled = true;
      for (const auto &alternate : cuda_graph_state_alternates) {
        if (alternate) {
          alternate->diagnostics_enabled = true;
        }
      }
    }
    backend_replay_signature_slots = static_cast<std::uint32_t>(
        1 + cuda_graph_state_alternates.size());
    backend_replay_signature_slot_capacity = 2;
    cuda_graph_state->stats.backend = CompiledGraphBackend::cuda;
    CompiledGraphStats result = cuda_graph_state->stats;
    if (graph_diagnostics_enabled &&
        (!cuda_graph_state->bounded_dispatch_controls.empty() ||
         !cuda_graph_state->bounded_dispatch_groups.empty() ||
         !cuda_graph_state->bounded_dispatch_observations.empty() ||
         cuda_graph_state->nested_device_controls != nullptr)) {
      auto &driver = CUDADriver::get_instance();
      driver.stream_synchronize(nullptr);
      result.known_bounded_payloads = static_cast<std::uint32_t>(
          cuda_graph_state->bounded_dispatch_observations.size());
      result.bounded_physical_observation_available =
          !cuda_graph_state->bounded_dispatch_observations.empty();
      for (const auto &observation :
           cuda_graph_state->bounded_dispatch_observations) {
        const auto extent = cuda_graph_ndarray_address(
            cuda_graph_state->signature, observation.extent_arg);
        if (!extent.has_value()) {
          result.bounded_physical_observation_available = false;
          continue;
        }
        std::array<std::int32_t, 2> state_words{};
        driver.memcpy_device_to_host(
            state_words.data(), reinterpret_cast<void *>(*extent),
            sizeof(state_words));
        const auto useful = static_cast<std::uint32_t>(std::clamp(
            state_words[0], std::int32_t{0},
            static_cast<std::int32_t>(observation.capacity)));
        const auto logical_blocks = static_cast<std::uint32_t>(std::min<
            std::uint64_t>(
            (static_cast<std::uint64_t>(useful) + observation.block_dim - 1u) /
                observation.block_dim,
            observation.baseline_grid_dim));
        const auto physical_blocks =
            observation.adaptive_grid ? logical_blocks
                                      : observation.baseline_grid_dim;
        result.last_bounded_useful_lanes += useful;
        result.last_bounded_physical_blocks += physical_blocks;
        result.last_bounded_physical_threads +=
            static_cast<std::uint64_t>(physical_blocks) *
            observation.block_dim;
        result.last_bounded_baseline_blocks +=
            observation.baseline_grid_dim;
        result.last_bounded_zero_payloads += useful == 0 ? 1u : 0u;
      }
      for (const auto &control :
           cuda_graph_state->bounded_dispatch_controls) {
        if (control.device_control == nullptr) {
          continue;
        }
        cuda::CudaGraphBoundedExtentControl observed;
        driver.memcpy_device_to_host(
            &observed, control.device_control, sizeof(observed));
        if (observed.driver_status != CUDA_SUCCESS) {
          result.last_driver_error = observed.driver_status;
          break;
        }
      }
      for (auto &group : cuda_graph_state->bounded_dispatch_groups) {
        if (group.device_control == nullptr) {
          continue;
        }
        cuda::CudaGraphBoundedGroupControl observed;
        driver.memcpy_device_to_host(&observed, group.device_control,
                                     sizeof(observed));
        result.bounded_update_replays += observed.replay_count;
        result.bounded_update_state_changes += observed.state_change_count;
        result.bounded_node_api_calls += observed.node_api_call_count;
        if (observed.driver_status != CUDA_SUCCESS &&
            result.last_driver_error == 0) {
          result.last_driver_error = observed.driver_status;
        }
        if (group.host_control.telemetry_enabled == 0) {
          std::uint32_t enabled = 1;
          auto *device_flag =
              reinterpret_cast<std::uint8_t *>(group.device_control) +
              offsetof(cuda::CudaGraphBoundedGroupControl,
                       telemetry_enabled);
          const auto telemetry_error = driver.memcpy_host_to_device.call(
              device_flag, &enabled, sizeof(enabled));
          if (telemetry_error == CUDA_SUCCESS) {
            group.host_control.telemetry_enabled = enabled;
          } else if (result.last_driver_error == 0) {
            result.last_driver_error = telemetry_error;
          }
        }
      }
      if (cuda_graph_state->nested_device_controls != nullptr &&
          !cuda_graph_state->nested_host_controls.empty()) {
        std::vector<cuda::CudaGraphPredicateGroupControl> observed(
            cuda_graph_state->nested_host_controls.size());
        driver.memcpy_device_to_host(
            observed.data(), cuda_graph_state->nested_device_controls,
            observed.size() * sizeof(cuda::CudaGraphPredicateGroupControl));
        for (const auto &control : observed) {
          result.bounded_update_replays += control.replay_count;
          result.bounded_update_state_changes += control.state_change_count;
          result.bounded_node_api_calls += control.node_api_call_count;
          if (control.driver_status != CUDA_SUCCESS &&
              result.last_driver_error == 0) {
            result.last_driver_error = control.driver_status;
          }
        }
      }
      result.bounded_update_cache_hits =
          result.bounded_update_replays -
          std::min(result.bounded_update_replays,
                   result.bounded_update_state_changes);
    }
    result.known_persistent_argument_bytes =
        cuda_graph_state->known_persistent_argument_bytes();
    result.known_bounded_control_bytes =
        cuda_graph_state->known_bounded_control_bytes();
    for (const auto &alternate : cuda_graph_state_alternates) {
      if (alternate == nullptr) {
        continue;
      }
      result.known_persistent_argument_bytes +=
          alternate->known_persistent_argument_bytes();
      result.known_bounded_control_bytes +=
          alternate->known_bounded_control_bytes();
    }
    const auto per_node_update_count = static_cast<std::uint32_t>(std::count_if(
        cuda_graph_state->bounded_dispatch_controls.begin(),
        cuda_graph_state->bounded_dispatch_controls.end(),
        [](const auto &control) { return control.device_control != nullptr; }));
    const auto grouped_update_count = static_cast<std::uint32_t>(
        cuda_graph_state->bounded_dispatch_groups.size());
    const auto nested_update_count = static_cast<std::uint32_t>(
        cuda_graph_state->nested_host_controls.size());
    result.known_bounded_update_groups =
        per_node_update_count + grouped_update_count + nested_update_count;
    result.known_bounded_updater_dispatches =
        per_node_update_count + grouped_update_count + nested_update_count;
    result.known_bounded_grouped_payloads = std::accumulate(
        cuda_graph_state->bounded_dispatch_groups.begin(),
        cuda_graph_state->bounded_dispatch_groups.end(), 0u,
        [](std::uint32_t total, const auto &group) {
          return total +
                 (group.host_nodes.size() > 1
                      ? static_cast<std::uint32_t>(group.host_nodes.size())
                      : 0u);
        });
    if (nested_update_count != 0) {
      result.known_bounded_grouped_payloads += static_cast<std::uint32_t>(
          cuda_graph_state->nested_host_nodes.size());
    }
    result.known_bounded_max_group_size =
        per_node_update_count == 0 ? 0u : 1u;
    for (const auto &group : cuda_graph_state->bounded_dispatch_groups) {
      result.known_bounded_max_group_size = std::max(
          result.known_bounded_max_group_size,
          static_cast<std::uint32_t>(group.host_nodes.size()));
    }
    for (const auto &control : cuda_graph_state->nested_host_controls) {
      result.known_bounded_max_group_size =
          std::max(result.known_bounded_max_group_size, control.node_count);
    }
    result.known_bounded_producer_fused_groups = 0;
    const auto &active_retry = cuda_graph_state->masked_mode
                                   ? cuda_graph_state->masked_retry
                                   : cuda_graph_state->retry;
    result.retry_backoff_remaining = active_retry.retry_backoff_remaining();
    result.consecutive_transient_failures =
        active_retry.consecutive_transient_failures();
    return finalize(result);
  }
#endif
#if defined(TI_WITH_VULKAN)
  if (vulkan_graph_state && vulkan_graph_state->registration) {
    backend_replay_signature_slots = static_cast<std::uint32_t>(
        vulkan_graph_state->launch_slots.size());
    backend_replay_signature_slot_capacity = 4;
    const auto source = graph_diagnostics_enabled
                            ? vulkan_graph_state->registration->debug_stats()
                            : vulkan_graph_state->registration->snapshot_stats();
    vulkan_graph_state->diagnostics_enabled = graph_diagnostics_enabled;
    CompiledGraphStats result;
    result.backend = CompiledGraphBackend::vulkan;
    result.attempts = source.attempts;
    result.ordinary_fallbacks = source.fallbacks;
    result.records = source.recorded;
    result.replays = source.replayed;
    result.patched_replays = source.patched;
    result.structural_fallbacks = source.structural_fallbacks;
    result.replay_slot_saturation_fallbacks =
        source.slot_saturation_fallbacks;
    result.known_persistent_argument_bytes =
        source.known_persistent_argument_bytes;
    result.effect_reads = source.effect_reads;
    result.effect_writes = source.effect_writes;
    result.dependency_barriers = source.dependency_barriers;
    result.exit_barriers = source.exit_barriers;
    result.barrier_deferrals = source.barrier_deferrals;
    result.rar_elisions = source.rar_elisions;
    switch (source.last_path) {
      case gfx::GraphReplayLastPath::fallback:
        result.last_path = CompiledGraphExecutionPath::ordinary_fallback;
        break;
      case gfx::GraphReplayLastPath::record:
        result.last_path = CompiledGraphExecutionPath::vulkan_record;
        break;
      case gfx::GraphReplayLastPath::replay:
        result.last_path = CompiledGraphExecutionPath::vulkan_replay;
        break;
      case gfx::GraphReplayLastPath::patched_replay:
        result.last_path =
            CompiledGraphExecutionPath::vulkan_patched_replay;
        break;
      case gfx::GraphReplayLastPath::none:
        break;
    }
    switch (source.last_fallback_reason) {
      case gfx::GraphReplayFallbackReason::runtime_mode:
        result.last_fallback_reason =
            CompiledGraphFallbackReason::runtime_mode;
        break;
      case gfx::GraphReplayFallbackReason::insufficient_tasks:
        result.last_fallback_reason =
            CompiledGraphFallbackReason::insufficient_dispatches;
        break;
      case gfx::GraphReplayFallbackReason::structural_unsupported:
        result.last_fallback_reason =
            CompiledGraphFallbackReason::structural_unsupported;
        break;
      case gfx::GraphReplayFallbackReason::slot_saturated:
        result.last_fallback_reason =
            CompiledGraphFallbackReason::replay_slot_saturated;
        break;
      case gfx::GraphReplayFallbackReason::none:
        break;
    }
    return finalize(result);
  }
#endif
  if (vulkan_inline_stats.attempts > 0) {
    return finalize(vulkan_inline_stats);
  }
  if (vulkan_inline_stats.backend != CompiledGraphBackend::none) {
    return finalize(vulkan_inline_stats);
  }
  return finalize({});
}

namespace {

void retain_graph_execution_handles(
    std::vector<std::shared_ptr<KernelExecutionHandle>> &retired,
    const std::vector<CompiledGraphJITCachedKernel> &kernels) {
  for (const auto &kernel : kernels) {
    if (kernel.execution_handle == nullptr) {
      continue;
    }
    const auto identity = kernel.execution_handle->identity();
    const bool already_retained = std::any_of(
        retired.begin(), retired.end(), [&](const auto &handle) {
          return handle != nullptr && handle->identity() == identity;
        });
    if (!already_retained) {
      retired.push_back(kernel.execution_handle);
    }
  }
}

}  // namespace

void CompiledGraphJITCache::clear_runtime_state() {
  auto clear_locked = [this]() {
    cuda_graph_state.reset();
    cuda_graph_state_alternates.clear();
    vulkan_graph_state.reset();
    vulkan_inline_stats = {};
    graph_diagnostics_counters_complete = true;
    kernels.clear();
    retired_execution_handles.clear();
    runtime_arg_plans.clear();
    runtime_binding_plans.clear();
    next_runtime_binding_plan_revision = 1;
    replay_attribution = {};
    validated_snode_tree_program = nullptr;
    validated_snode_tree_epoch = 0;
  };

#if defined(TI_WITH_CUDA)
  // A CUDA-enabled build may run only CPU/Vulkan graphs on a machine without
  // an NVIDIA driver. Do not construct CUDAContext unless this cache actually
  // owns CUDA state. Probe under run_mutex, which is also the state-creation
  // lock used by jit_run_cached().
  std::unique_lock<std::mutex> run_lock(run_mutex);
  if (cuda_graph_state == nullptr && cuda_graph_state_alternates.empty()) {
    clear_locked();
    return;
  }

  // Match jit_run_cached()'s CUDA-submission-lock -> run_mutex ordering so
  // reset/destruction cannot retire an executable during a submission. Drop
  // run_mutex before taking the outer lock, then reacquire it and clear the
  // current state. A concurrent run is therefore either completed first or
  // starts from an empty cache after this reset.
  run_lock.unlock();
  auto cuda_submission_lock =
      CUDAContext::get_instance().get_submission_lock_guard();
  run_lock.lock();
  clear_locked();
#else
  std::lock_guard<std::mutex> run_lock(run_mutex);
  clear_locked();
#endif
}

void CompiledGraphJITCache::retire_snode_tree_runtime_state() {
  auto retire_locked = [this]() {
    cuda_graph_state.reset();
    cuda_graph_state_alternates.clear();
    vulkan_graph_state.reset();
    vulkan_inline_stats = {};
    graph_diagnostics_counters_complete = true;
    retain_graph_execution_handles(retired_execution_handles, kernels);
    kernels.clear();
    runtime_arg_plans.clear();
    runtime_binding_plans.clear();
    next_runtime_binding_plan_revision = 1;
    replay_attribution = {};
    validated_snode_tree_program = nullptr;
    validated_snode_tree_epoch = 0;
  };

#if defined(TI_WITH_CUDA)
  std::unique_lock<std::mutex> run_lock(run_mutex);
  if (cuda_graph_state == nullptr && cuda_graph_state_alternates.empty()) {
    retire_locked();
    return;
  }
  run_lock.unlock();
  auto cuda_submission_lock =
      CUDAContext::get_instance().get_submission_lock_guard();
  run_lock.lock();
  retire_locked();
#else
  std::lock_guard<std::mutex> run_lock(run_mutex);
  retire_locked();
#endif
}

CompiledGraphJITCache::~CompiledGraphJITCache() {
  clear_runtime_state();
}

void CompiledGraph::run(
    const std::unordered_map<std::string, IValue> &args) const {
  for (const auto &dispatch : dispatches) {
    TI_ASSERT(dispatch.compiled_kernel);
    LaunchContextBuilder launch_ctx(dispatch.compiled_kernel);
    set_cuda_bounded_range_binding(dispatch, launch_ctx);
    launch_ctx.append_dispatch_label(dispatch.dispatch_label);
    init_runtime_context(dispatch.symbolic_args, args, launch_ctx);
    TI_ERROR_IF(
        !launch_ctx.dense_storage_ptrs.empty(),
        "AOT Graph does not support runtime dense storage arguments; use an "
        "owning Ndarray or a JIT Graph");
    Program *program = nullptr;
    std::vector<const Ndarray *> ndarray_views;
    std::vector<const Texture *> texture_views;
    for (const auto &ref : launch_ctx.ndarray_ptrs) {
      Program *owner = ref.owner;
      TI_ASSERT(owner != nullptr);
      TI_ERROR_IF(program != nullptr && owner != program,
                  "AOT Graph Ndarray arguments span multiple Programs");
      program = owner;
      for (const Ndarray *view : {ref.data, ref.grad}) {
        if (view == nullptr) {
          continue;
        }
        if (std::find(ndarray_views.begin(), ndarray_views.end(), view) ==
            ndarray_views.end()) {
          ndarray_views.push_back(view);
        }
      }
    }
    for (const auto &ref : launch_ctx.texture_ptrs) {
      Program *owner = ref.owner;
      TI_ASSERT(owner != nullptr);
      TI_ERROR_IF(program != nullptr && owner != program,
                  "AOT Graph runtime resources span multiple Programs");
      program = owner;
      if (std::find(texture_views.begin(), texture_views.end(), ref.texture) ==
          texture_views.end()) {
        texture_views.push_back(ref.texture);
      }
    }
    if (program != nullptr) {
      program->ensure_runtime_submission_allowed("AOT Graph launch");
    }
    std::optional<Program::RuntimeResourceSubmissionGuard> resource_guard;
    if (program != nullptr) {
      resource_guard.emplace(
          program->acquire_runtime_resource_submission_guard());
      program->retain_ndarrays_for_external_submission(ndarray_views);
      program->retain_textures_for_external_submission(texture_views);
      program->resolve_ndarray_launch_context(launch_ctx);
      program->resolve_texture_launch_context(launch_ctx);
    }
    // Run cgraph loaded from AOT module
    dispatch.compiled_kernel->launch(launch_ctx);
  }
}

void CompiledGraph::jit_run(
    const CompileConfig &compile_config,
    const std::unordered_map<std::string, IValue> &args) const try {
  TI_ERROR_IF(has_indirect_dispatches(),
              "Graph indirect dispatch requires the cached Vulkan JIT "
              "execution path");
  Program *program = jit_graph_program(*this);
  if (program != nullptr) {
    program->ensure_runtime_submission_allowed("Graph launch");
  }
  std::optional<Program::SNodeTreeLifecycleReadGuard> tree_lifecycle_guard;
  std::optional<Program::RuntimeResourceGraphScope> resource_guard;
  std::optional<Program::RuntimeSubmissionScope> completion_scope;
  GraphRuntimeResourceViews resource_views;
  if (program != nullptr) {
    tree_lifecycle_guard.emplace(
        program->acquire_snode_tree_lifecycle_read_guard());
    program->validate_snode_tree_dependencies(snode_tree_dependencies);
    resource_views = graph_runtime_resource_views(args, program);
    if (!resource_views.empty()) {
      resource_guard.emplace(program->acquire_runtime_resource_graph_scope());
      if (!resource_views.ndarrays.empty()) {
        program->retain_ndarrays_for_external_submission(
            resource_views.ndarrays.data(), resource_views.ndarrays.size());
      }
      if (!resource_views.runtime_storage.empty()) {
        program->retain_runtime_storage_for_graph_submission(
            resource_views.runtime_storage.data(),
            resource_views.runtime_storage.size());
      }
      if (!resource_views.textures.empty()) {
        program->retain_textures_for_external_submission(
            resource_views.textures.data(), resource_views.textures.size());
      }
    }
    completion_scope.emplace(program->acquire_runtime_submission_scope());
  }
#if defined(TI_WITH_CUDA)
  std::unique_lock<std::recursive_mutex> cuda_submission_lock;
  if (compile_config.arch == Arch::cuda) {
    cuda_submission_lock =
        CUDAContext::get_instance().get_submission_lock_guard();
  }
#endif
  for (const auto &dispatch : dispatches) {
    if (dispatch.cuda_capture_command) {
#if defined(TI_WITH_CUDA)
      TI_ERROR_IF(compile_config.arch != Arch::cuda,
                  "CUDA capture command requires the CUDA backend");
      record_cuda_capture_command(dispatch, args, *program, nullptr);
      continue;
#else
      TI_NOT_IMPLEMENTED;
#endif
    }
    TI_ASSERT(dispatch.ti_kernel);
    auto launch_ctx = dispatch.ti_kernel->make_launch_context();
    launch_ctx.append_dispatch_label(dispatch.dispatch_label);
    init_runtime_context(dispatch.symbolic_args, args, launch_ctx);
    // Compile & Run (JIT): The compilation result will be cached, so don't
    // worry that the kernels dispatched by this cgraph will be compiled
    // repeatedly.
    auto *prog = dispatch.ti_kernel->program;
    const auto &compiled_kernel_data = prog->compile_kernel(
        compile_config, prog->get_device_caps(), *dispatch.ti_kernel);
    prog->launch_kernel(compiled_kernel_data, launch_ctx);
  }
  if (resource_guard) {
    resource_guard->finish_external_access_epoch();
  }
} catch (const BackendRuntimeError &error) {
  Program *program = jit_graph_program(*this);
  if (program != nullptr) {
    program->record_runtime_submission_failure();
    program->report_backend_runtime_error(error);
  }
  throw;
} catch (...) {
  Program *program = jit_graph_program(*this);
  if (program != nullptr) {
    program->record_runtime_submission_failure();
  }
  throw;
}

void CompiledGraph::jit_run_cached(
    const CompileConfig &compile_config,
    const std::unordered_map<std::string, IValue> &args,
    CompiledGraphJITCache &cache,
    bool cuda_concurrent_batch_lane) const try {
  const bool has_indirect_dispatch = has_indirect_dispatches();
  TI_ERROR_IF(has_indirect_dispatch &&
                  compile_config.arch != Arch::vulkan,
              "Graph indirect dispatch is currently supported only by the "
              "Vulkan backend");
  Program *program = jit_graph_program(*this);
  if (program != nullptr) {
    program->ensure_runtime_submission_allowed("cached Graph launch");
  }
  // Replay attribution used to be enabled implicitly by a statistics query.
  // Statistics are now side-effect free, and opt-in ticket telemetry is the
  // only measured path, so the production launch is compiled without clocks.
  constexpr bool attribute = false;
  const bool stable_replay = cache.stable_replay_optimization_enabled.load(
      std::memory_order_relaxed);
  const auto replay_begin =
      attribute ? ReplayClock::now() : ReplayClock::time_point{};
  auto finish_attribution = [&]() {
    if (attribute) {
      ++cache.replay_attribution.calls;
      cache.replay_attribution.total_ns += replay_elapsed_ns(replay_begin);
    }
  };
  std::optional<Program::SNodeTreeLifecycleReadGuard> tree_lifecycle_guard;
  std::optional<Program::RuntimeResourceGraphScope> resource_guard;
  std::optional<Program::RuntimeSubmissionScope> completion_scope;
  const bool requires_snode_guard =
      program != nullptr &&
      (!stable_replay || !snode_tree_dependencies.empty() ||
       graph_runtime_args_require_snode_guard(args));
  if (program != nullptr) {
    if (requires_snode_guard) {
      const auto guard_begin =
          attribute ? ReplayClock::now() : ReplayClock::time_point{};
      tree_lifecycle_guard.emplace(
          program->acquire_snode_tree_lifecycle_read_guard());
      if (attribute) {
        cache.replay_attribution.snode_guard_ns +=
            replay_elapsed_ns(guard_begin);
        ++cache.replay_attribution.snode_guard_acquisitions;
      }
    } else if (attribute) {
      ++cache.replay_attribution.snode_guard_elisions;
    }
    if (graph_has_runtime_resource_declarations(args)) {
      const auto guard_begin =
          attribute ? ReplayClock::now() : ReplayClock::time_point{};
      resource_guard.emplace(program->acquire_runtime_resource_graph_scope());
      if (attribute) {
        cache.replay_attribution.resource_guard_ns +=
            replay_elapsed_ns(guard_begin);
      }
    }
  }
#if defined(TI_WITH_CUDA)
  // A graph is one submission transaction. This is required not only while
  // capturing: replaying a CUDA graph concurrently with an ordinary kernel on
  // the shared legacy default stream exposed invalid runtime state to both
  // callers once Python graph execution started releasing the GIL.
  std::unique_lock<std::recursive_mutex> cuda_submission_lock;
  if (compile_config.arch == Arch::cuda) {
    const auto lock_begin =
        attribute ? ReplayClock::now() : ReplayClock::time_point{};
    cuda_submission_lock =
        CUDAContext::get_instance().get_submission_lock_guard();
    if (attribute) {
      cache.replay_attribution.cuda_submission_lock_ns +=
          replay_elapsed_ns(lock_begin);
    }
  }
#endif
  const auto cache_wait_begin =
      attribute ? ReplayClock::now() : ReplayClock::time_point{};
  std::lock_guard<std::mutex> lock(cache.run_mutex);
  if (attribute) {
    cache.replay_attribution.cache_wait_ns +=
        replay_elapsed_ns(cache_wait_begin);
  }
  if (program != nullptr && tree_lifecycle_guard &&
      (cache.validated_snode_tree_program != program ||
       cache.validated_snode_tree_epoch != tree_lifecycle_guard->epoch())) {
    const auto validation_begin =
        attribute ? ReplayClock::now() : ReplayClock::time_point{};
    try {
      program->validate_snode_tree_dependencies(snode_tree_dependencies);
    } catch (...) {
      // The dependency is stale. Retire replay/cached launch state before
      // surfacing the rebuild requirement so no backend object keeps an
      // executable containing the old root binding.
      cache.cuda_graph_state.reset();
      cache.cuda_graph_state_alternates.clear();
      cache.vulkan_graph_state.reset();
      cache.vulkan_inline_stats = {};
      cache.kernels.clear();
      cache.runtime_arg_plans.clear();
      cache.runtime_binding_plans.clear();
      cache.next_runtime_binding_plan_revision = 1;
      cache.validated_snode_tree_program = nullptr;
      cache.validated_snode_tree_epoch = 0;
      throw;
    }
    cache.validated_snode_tree_program = program;
    cache.validated_snode_tree_epoch = tree_lifecycle_guard->epoch();
    if (attribute) {
      cache.replay_attribution.snode_validation_ns +=
          replay_elapsed_ns(validation_begin);
    }
  }
  if (program != nullptr && resource_guard) {
    const auto plan_begin =
        attribute ? ReplayClock::now() : ReplayClock::time_point{};
    if (stable_replay) {
      const auto &plan =
          prepare_runtime_binding_plan(cache, args, program, attribute);
      if (attribute) {
        cache.replay_attribution.binding_plan_ns +=
            replay_elapsed_ns(plan_begin);
      }
      const auto retain_begin =
          attribute ? ReplayClock::now() : ReplayClock::time_point{};
      if (!plan.ndarrays.empty()) {
        program->retain_ndarrays_for_external_submission(plan.ndarrays.data(),
                                                         plan.ndarrays.size());
      }
      if (!plan.runtime_storage.empty()) {
        program->retain_runtime_storage_for_graph_submission(
            plan.runtime_storage.data(), plan.runtime_storage.size());
      }
      if (!plan.textures.empty()) {
        program->retain_textures_for_external_submission(plan.textures.data(),
                                                         plan.textures.size());
      }
      if (attribute) {
        cache.replay_attribution.resource_retain_ns +=
            replay_elapsed_ns(retain_begin);
      }
    } else {
      const auto resource_views = graph_runtime_resource_views(args, program);
      if (attribute) {
        cache.replay_attribution.binding_plan_ns +=
            replay_elapsed_ns(plan_begin);
        ++cache.replay_attribution.binding_plan_misses;
      }
      const auto retain_begin =
          attribute ? ReplayClock::now() : ReplayClock::time_point{};
      if (!resource_views.ndarrays.empty()) {
        program->retain_ndarrays_for_external_submission(
            resource_views.ndarrays.data(), resource_views.ndarrays.size());
      }
      if (!resource_views.runtime_storage.empty()) {
        program->retain_runtime_storage_for_graph_submission(
            resource_views.runtime_storage.data(),
            resource_views.runtime_storage.size());
      }
      if (!resource_views.textures.empty()) {
        program->retain_textures_for_external_submission(
            resource_views.textures.data(), resource_views.textures.size());
      }
      if (attribute) {
        cache.replay_attribution.resource_retain_ns +=
            replay_elapsed_ns(retain_begin);
      }
    }
    completion_scope.emplace(program->acquire_runtime_submission_scope());
  }
  if (program != nullptr && !completion_scope) {
    completion_scope.emplace(program->acquire_runtime_submission_scope());
  }
  const auto backend_begin =
      attribute ? ReplayClock::now() : ReplayClock::time_point{};
#if defined(TI_WITH_CUDA)
  if (compile_config.arch == Arch::cuda) {
    TI_ASSERT(program != nullptr);
    // A dispatch label describes one physical kernel launch. Native CUDA
    // replay can expose that host-side annotation only while capturing, not on
    // every replay, so labeled graphs deliberately keep the ordinary cached
    // launch path below. Unlabeled graphs remain replay-first.
    if (!has_dispatch_labels() &&
        try_run_cuda_graph(*this, compile_config, args, cache, *program,
                           &program->runtime_statistics(),
                           cuda_concurrent_batch_lane)) {
      program->mark_runtime_submission(
          RuntimeSubmissionKind::kGraphBackendSubmission);
      if (resource_guard) {
        resource_guard->finish_external_access_epoch();
      }
      if (attribute) {
        cache.replay_attribution.backend_ns +=
            replay_elapsed_ns(backend_begin);
      }
      finish_attribution();
      return;
    }
    TI_ERROR_IF(cuda_concurrent_batch_lane,
                "Concurrent CUDA Graph batch requires native unlabeled "
                "Graph replay");
    if (program != nullptr) {
      program->runtime_statistics().record_graph_ordinary_fallback();
    }
  }
#endif
#if defined(TI_WITH_VULKAN)
  if (compile_config.arch == Arch::vulkan) {
    if (try_run_vulkan_graph(*this, compile_config, args, cache,
                             program != nullptr
                                 ? &program->runtime_statistics()
                                 : nullptr)) {
      TI_ASSERT(program != nullptr);
      program->mark_runtime_submission(
          RuntimeSubmissionKind::kGraphBackendSubmission);
      if (resource_guard) {
        resource_guard->finish_external_access_epoch();
      }
      if (attribute) {
        cache.replay_attribution.backend_ns +=
            replay_elapsed_ns(backend_begin);
      }
      finish_attribution();
      return;
    }
    TI_ERROR_IF(
        has_indirect_dispatch,
        "Graph indirect dispatch requires native Vulkan Graph replay, but "
        "the active runtime mode or graph structure is unsupported");
    if (program != nullptr) {
      program->runtime_statistics().record_graph_ordinary_fallback();
    }
  }
#endif
  if (cache.kernels.size() != dispatches.size()) {
    cache.kernels.assign(dispatches.size(), {});
    cache.runtime_arg_plans.clear();
  }
  const bool use_cpu_runtime_arg_plan = arch_is_cpu(compile_config.arch);
  if (use_cpu_runtime_arg_plan &&
      cache.runtime_arg_plans.size() != dispatches.size()) {
    cache.runtime_arg_plans.clear();
    cache.runtime_arg_plans.reserve(dispatches.size());
    for (const auto &dispatch : dispatches) {
      cache.runtime_arg_plans.push_back(build_cpu_runtime_arg_plan(dispatch));
    }
  }
  if (use_cpu_runtime_arg_plan) {
    for (std::size_t i = 0; i < dispatches.size(); ++i) {
      TI_ERROR_IF(dispatches[i].cpu_bounded_dispatch.has_value() &&
                      !cache.runtime_arg_plans[i].cpu_fast_path,
                  "CPU exact bounded dispatch requires the cached LLVM "
                  "runtime argument path");
    }
  }
  const bool cache_compiled_kernel_data =
      arch_is_cpu(compile_config.arch) || compile_config.arch == Arch::cuda;
  for (std::size_t i = 0; i < dispatches.size(); ++i) {
    const auto &dispatch = dispatches[i];
    if (dispatch.cuda_capture_command) {
#if defined(TI_WITH_CUDA)
      TI_ERROR_IF(compile_config.arch != Arch::cuda,
                  "CUDA capture command requires the CUDA backend");
      record_cuda_capture_command(dispatch, args, *program, nullptr);
      continue;
#else
      TI_NOT_IMPLEMENTED;
#endif
    }
    TI_ASSERT(dispatch.ti_kernel);
    auto *prog = dispatch.ti_kernel->program;
    auto launch_ctx = dispatch.ti_kernel->make_launch_context(
        dispatch.cpu_bounded_dispatch.has_value());
    if (compile_config.arch == Arch::cuda) {
      set_cuda_bounded_range_binding(dispatch, launch_ctx);
    }
    launch_ctx.append_dispatch_label(dispatch.dispatch_label);
    bool bounded_extent_uses_runtime_storage = false;
    if (use_cpu_runtime_arg_plan && cache.runtime_arg_plans[i].cpu_fast_path) {
      bounded_extent_uses_runtime_storage = init_runtime_context_from_plan(
          cache.runtime_arg_plans[i], args, prog, launch_ctx);
    } else {
      init_runtime_context(dispatch.symbolic_args, args, launch_ctx);
    }
    auto &cached = cache.kernels[i];
    const CompiledKernelData *compiled_kernel_data =
        get_or_compile_cached_kernel(dispatch, compile_config, cached,
                                     cache_compiled_kernel_data);
#if defined(TI_WITH_LLVM)
    if (arch_is_cpu(compile_config.arch) &&
        try_launch_cached_llvm_kernel(prog, *compiled_kernel_data, cached,
                                      launch_ctx,
                                      &cache.runtime_arg_plans[i],
                                      bounded_extent_uses_runtime_storage)) {
      continue;
    }
#endif
    TI_ERROR_IF(dispatch.cpu_bounded_dispatch.has_value(),
                "CPU exact bounded dispatch requires the LLVM fast launcher");
    prog->launch_kernel(*compiled_kernel_data, launch_ctx);
  }
  if (resource_guard) {
    resource_guard->finish_external_access_epoch();
  }
  if (attribute) {
    cache.replay_attribution.backend_ns += replay_elapsed_ns(backend_begin);
  }
  finish_attribution();
} catch (const BackendRuntimeError &error) {
  Program *program = jit_graph_program(*this);
  if (program != nullptr) {
    program->record_runtime_submission_failure();
    program->report_backend_runtime_error(error);
  }
  throw;
} catch (...) {
  Program *program = jit_graph_program(*this);
  if (program != nullptr) {
    program->record_runtime_submission_failure();
  }
  throw;
}

bool CompiledGraph::jit_run_bounded_cuda_cached(
    const CompileConfig &compile_config,
    const std::unordered_map<std::string, IValue> &args,
    CompiledGraphJITCache &cache,
    Ndarray *predicate,
    int max_iterations,
    bool continue_while_nonzero) const try {
#if defined(TI_WITH_CUDA)
  if (compile_config.arch != Arch::cuda || predicate == nullptr) {
    return false;
  }
  Program *program = jit_graph_program(*this);
  if (program == nullptr) {
    return false;
  }
  program->ensure_runtime_submission_allowed(
      "bounded CUDA Graph launch");
  auto tree_lifecycle_guard =
      program->acquire_snode_tree_lifecycle_read_guard();
  auto resource_views = graph_runtime_resource_views(args, program);
  std::optional<Program::RuntimeResourceGraphScope> resource_guard;
  if (!resource_views.empty()) {
    resource_guard.emplace(program->acquire_runtime_resource_graph_scope());
    if (!resource_views.ndarrays.empty()) {
      program->retain_ndarrays_for_external_submission(
          resource_views.ndarrays.data(), resource_views.ndarrays.size());
    }
    if (!resource_views.runtime_storage.empty()) {
      program->retain_runtime_storage_for_graph_submission(
          resource_views.runtime_storage.data(),
          resource_views.runtime_storage.size());
    }
    if (!resource_views.textures.empty()) {
      program->retain_textures_for_external_submission(
          resource_views.textures.data(), resource_views.textures.size());
    }
  }
  auto completion_scope = program->acquire_runtime_submission_scope();
  auto cuda_submission_lock =
      CUDAContext::get_instance().get_submission_lock_guard();
  std::lock_guard<std::mutex> lock(cache.run_mutex);
  if (cache.validated_snode_tree_program != program ||
      cache.validated_snode_tree_epoch != tree_lifecycle_guard.epoch()) {
    try {
      program->validate_snode_tree_dependencies(snode_tree_dependencies);
    } catch (...) {
      cache.cuda_graph_state.reset();
      cache.cuda_graph_state_alternates.clear();
      cache.kernels.clear();
      cache.runtime_arg_plans.clear();
      cache.validated_snode_tree_program = nullptr;
      cache.validated_snode_tree_epoch = 0;
      throw;
    }
    cache.validated_snode_tree_program = program;
    cache.validated_snode_tree_epoch = tree_lifecycle_guard.epoch();
  }
  if (!try_run_cuda_bounded_graph(
          *this, compile_config, args, cache, *program, *predicate,
          max_iterations, continue_while_nonzero,
          &program->runtime_statistics())) {
    return false;
  }
  program->mark_runtime_submission(
      RuntimeSubmissionKind::kGraphBackendSubmission);
  return true;
#else
  return false;
#endif
} catch (const BackendRuntimeError &error) {
  Program *program = jit_graph_program(*this);
  if (program != nullptr) {
    program->record_runtime_submission_failure();
    program->report_backend_runtime_error(error);
  }
  throw;
} catch (...) {
  Program *program = jit_graph_program(*this);
  if (program != nullptr) {
    program->record_runtime_submission_failure();
  }
  throw;
}

bool CompiledGraph::jit_run_bounded_cuda_masked_cached(
    const CompileConfig &compile_config,
    const std::unordered_map<std::string, IValue> &args,
    CompiledGraphJITCache &cache,
    Ndarray *predicate,
    int max_iterations,
    bool continue_while_nonzero) const try {
#if defined(TI_WITH_CUDA)
  if (compile_config.arch != Arch::cuda || predicate == nullptr) {
    return false;
  }
  Program *program = jit_graph_program(*this);
  if (program == nullptr) {
    return false;
  }
  program->ensure_runtime_submission_allowed(
      "masked bounded CUDA Graph launch");
  auto tree_lifecycle_guard =
      program->acquire_snode_tree_lifecycle_read_guard();
  auto resource_views = graph_runtime_resource_views(args, program);
  std::optional<Program::RuntimeResourceGraphScope> resource_guard;
  if (!resource_views.empty()) {
    resource_guard.emplace(program->acquire_runtime_resource_graph_scope());
    if (!resource_views.ndarrays.empty()) {
      program->retain_ndarrays_for_external_submission(
          resource_views.ndarrays.data(), resource_views.ndarrays.size());
    }
    if (!resource_views.runtime_storage.empty()) {
      program->retain_runtime_storage_for_graph_submission(
          resource_views.runtime_storage.data(),
          resource_views.runtime_storage.size());
    }
    if (!resource_views.textures.empty()) {
      program->retain_textures_for_external_submission(
          resource_views.textures.data(), resource_views.textures.size());
    }
  }
  auto completion_scope = program->acquire_runtime_submission_scope();
  auto cuda_submission_lock =
      CUDAContext::get_instance().get_submission_lock_guard();
  std::lock_guard<std::mutex> lock(cache.run_mutex);
  if (cache.validated_snode_tree_program != program ||
      cache.validated_snode_tree_epoch != tree_lifecycle_guard.epoch()) {
    try {
      program->validate_snode_tree_dependencies(snode_tree_dependencies);
    } catch (...) {
      cache.cuda_graph_state.reset();
      cache.cuda_graph_state_alternates.clear();
      cache.kernels.clear();
      cache.runtime_arg_plans.clear();
      cache.validated_snode_tree_program = nullptr;
      cache.validated_snode_tree_epoch = 0;
      throw;
    }
    cache.validated_snode_tree_program = program;
    cache.validated_snode_tree_epoch = tree_lifecycle_guard.epoch();
  }
  if (!try_run_cuda_masked_control_graph(
          *this, compile_config, args, cache, *program, *predicate,
          /*control_type=*/1, max_iterations, continue_while_nonzero,
          /*branch_dispatch_counts=*/{}, /*default_branch=*/-1,
          &program->runtime_statistics())) {
    return false;
  }
  program->mark_runtime_submission(
      RuntimeSubmissionKind::kGraphBackendSubmission);
  return true;
#else
  return false;
#endif
} catch (const BackendRuntimeError &error) {
  Program *program = jit_graph_program(*this);
  if (program != nullptr) {
    program->record_runtime_submission_failure();
    program->report_backend_runtime_error(error);
  }
  throw;
} catch (...) {
  Program *program = jit_graph_program(*this);
  if (program != nullptr) {
    program->record_runtime_submission_failure();
  }
  throw;
}

bool CompiledGraph::jit_submit_bounded_cuda_nested_sequence_cached(
    const CompileConfig &compile_config,
    const std::unordered_map<std::string, IValue> &args,
    CompiledGraphJITCache &cache,
    Ndarray *outer_predicate,
    Ndarray *outer_counter,
    Ndarray *outer_status,
    const std::vector<CompiledGraphNestedInnerControl> &inner_controls,
    std::size_t outer_condition_dispatch_count,
    int outer_max_iterations,
    bool allow_device_update) const try {
#if defined(TI_WITH_CUDA)
  if (compile_config.arch != Arch::cuda || outer_predicate == nullptr ||
      outer_counter == nullptr || inner_controls.empty() ||
      inner_controls.size() > 8) {
    return false;
  }
  Program *program = jit_graph_program(*this);
  if (program == nullptr) {
    return false;
  }
  const auto valid_control = [&](const Ndarray *control) {
    return control != nullptr && control->get_nelement() == 1 &&
           control->get_element_data_type() == PrimitiveType::i32 &&
           control->owning_program() == program;
  };
  if (!valid_control(outer_predicate) || !valid_control(outer_counter) ||
      (outer_status != nullptr && !valid_control(outer_status))) {
    return false;
  }
  std::vector<DeviceAllocation> controls{
      outer_predicate->get_device_allocation(),
      outer_counter->get_device_allocation()};
  if (outer_status != nullptr) {
    controls.push_back(outer_status->get_device_allocation());
  }
  for (const auto &inner : inner_controls) {
    if (!valid_control(inner.predicate) || !valid_control(inner.counter) ||
        (inner.status != nullptr && !valid_control(inner.status))) {
      return false;
    }
    controls.push_back(inner.predicate->get_device_allocation());
    controls.push_back(inner.counter->get_device_allocation());
    if (inner.status != nullptr) {
      controls.push_back(inner.status->get_device_allocation());
    }
  }
  std::sort(controls.begin(), controls.end(),
            [](const DeviceAllocation &lhs, const DeviceAllocation &rhs) {
              if (lhs.device != rhs.device) {
                return reinterpret_cast<std::uintptr_t>(lhs.device) <
                       reinterpret_cast<std::uintptr_t>(rhs.device);
              }
              return lhs.alloc_id < rhs.alloc_id;
            });
  if (std::adjacent_find(controls.begin(), controls.end()) != controls.end()) {
    return false;
  }

  program->ensure_runtime_submission_allowed(
      "nested bounded CUDA Graph launch");
  auto tree_lifecycle_guard =
      program->acquire_snode_tree_lifecycle_read_guard();
  auto resource_views = graph_runtime_resource_views(args, program);
  resource_views.ndarrays.add(outer_predicate);
  resource_views.ndarrays.add(outer_counter);
  if (outer_status != nullptr) {
    resource_views.ndarrays.add(outer_status);
  }
  for (const auto &inner : inner_controls) {
    resource_views.ndarrays.add(inner.predicate);
    resource_views.ndarrays.add(inner.counter);
    if (inner.status != nullptr) {
      resource_views.ndarrays.add(inner.status);
    }
  }
  std::optional<Program::RuntimeResourceGraphScope> resource_guard;
  if (!resource_views.empty()) {
    resource_guard.emplace(program->acquire_runtime_resource_graph_scope());
    if (!resource_views.ndarrays.empty()) {
      program->retain_ndarrays_for_external_submission(
          resource_views.ndarrays.data(), resource_views.ndarrays.size());
    }
    if (!resource_views.runtime_storage.empty()) {
      program->retain_runtime_storage_for_graph_submission(
          resource_views.runtime_storage.data(),
          resource_views.runtime_storage.size());
    }
    if (!resource_views.textures.empty()) {
      program->retain_textures_for_external_submission(
          resource_views.textures.data(), resource_views.textures.size());
    }
  }
  auto completion_scope = program->acquire_runtime_submission_scope();
  auto cuda_submission_lock =
      CUDAContext::get_instance().get_submission_lock_guard();
  std::lock_guard<std::mutex> lock(cache.run_mutex);
  if (cache.validated_snode_tree_program != program ||
      cache.validated_snode_tree_epoch != tree_lifecycle_guard.epoch()) {
    try {
      program->validate_snode_tree_dependencies(snode_tree_dependencies);
    } catch (...) {
      cache.cuda_graph_state.reset();
      cache.cuda_graph_state_alternates.clear();
      cache.kernels.clear();
      cache.runtime_arg_plans.clear();
      cache.validated_snode_tree_program = nullptr;
      cache.validated_snode_tree_epoch = 0;
      throw;
    }
    cache.validated_snode_tree_program = program;
    cache.validated_snode_tree_epoch = tree_lifecycle_guard.epoch();
  }
  const bool device_update_submitted =
      allow_device_update &&
      try_run_cuda_device_update_nested_control_graph(
          *this, compile_config, args, cache, *program, *outer_predicate,
          inner_controls, outer_condition_dispatch_count,
          outer_max_iterations, &program->runtime_statistics());
  if (!device_update_submitted &&
      !try_run_cuda_masked_nested_control_graph(
          *this, compile_config, args, cache, *program, *outer_predicate,
          inner_controls, outer_condition_dispatch_count,
          outer_max_iterations, &program->runtime_statistics())) {
    return false;
  }
  program->mark_runtime_submission(
      RuntimeSubmissionKind::kGraphBackendSubmission);
  if (resource_guard) {
    resource_guard->finish_external_access_epoch();
  }
  return true;
#else
  return false;
#endif
} catch (const BackendRuntimeError &error) {
  Program *program = jit_graph_program(*this);
  if (program != nullptr) {
    program->record_runtime_submission_failure();
    program->report_backend_runtime_error(error);
  }
  throw;
} catch (...) {
  Program *program = jit_graph_program(*this);
  if (program != nullptr) {
    program->record_runtime_submission_failure();
  }
  throw;
}

bool CompiledGraph::jit_submit_bounded_cuda_nested_cached(
    const CompileConfig &compile_config,
    const std::unordered_map<std::string, IValue> &args,
    CompiledGraphJITCache &cache,
    Ndarray *outer_predicate,
    Ndarray *outer_counter,
    Ndarray *outer_status,
    Ndarray *inner_predicate,
    Ndarray *inner_counter,
    Ndarray *inner_status,
    std::size_t outer_condition_dispatch_count,
    std::size_t inner_condition_dispatch_begin,
    std::size_t inner_body_dispatch_begin,
    std::size_t outer_suffix_dispatch_begin,
    int outer_max_iterations,
    int inner_max_iterations,
    bool allow_device_update) const {
  CompiledGraphNestedInnerControl inner;
  inner.predicate = inner_predicate;
  inner.counter = inner_counter;
  inner.status = inner_status;
  inner.condition_dispatch_begin = inner_condition_dispatch_begin;
  inner.body_dispatch_begin = inner_body_dispatch_begin;
  inner.dispatch_end = outer_suffix_dispatch_begin;
  inner.max_iterations = inner_max_iterations;
  inner.chunk_size = inner_max_iterations;
  return jit_submit_bounded_cuda_nested_sequence_cached(
      compile_config, args, cache, outer_predicate, outer_counter,
      outer_status, {inner}, outer_condition_dispatch_count,
      outer_max_iterations, allow_device_update);
}

CompiledGraphStructuredResult
CompiledGraph::jit_run_bounded_vulkan_cached(
    const CompileConfig &compile_config,
    const std::unordered_map<std::string, IValue> &args,
    CompiledGraphJITCache &cache,
    Ndarray *predicate,
    Ndarray *counter,
    Ndarray *status,
    std::size_t initial_dispatch_count,
    int max_iterations,
    bool execute_initial_dispatches,
    std::uint32_t strategy,
    bool wait_for_result) const try {
  CompiledGraphStructuredResult result;
#if defined(TI_WITH_VULKAN)
  if (compile_config.arch != Arch::vulkan || predicate == nullptr ||
      counter == nullptr || max_iterations < 0 ||
      initial_dispatch_count == 0 ||
      initial_dispatch_count >= dispatches.size() ||
      strategy > static_cast<std::uint32_t>(
                     gfx::GfxRuntime::GraphStructuredStrategy::
                         coarse_conditional)) {
    return result;
  }
  Program *program = jit_graph_program(*this);
  if (program == nullptr) {
    return result;
  }
  const auto valid_control = [&](const Ndarray *control) {
    return control != nullptr && control->get_nelement() == 1 &&
           control->get_element_data_type() == PrimitiveType::i32 &&
           control->owning_program() == program;
  };
  if (!valid_control(predicate) || !valid_control(counter) ||
      (status != nullptr && !valid_control(status)) ||
      static_cast<std::uint64_t>(max_iterations) >
          std::numeric_limits<std::uint32_t>::max()) {
    return result;
  }

  program->ensure_runtime_submission_allowed(
      "bounded Vulkan Graph launch");
  auto tree_lifecycle_guard =
      program->acquire_snode_tree_lifecycle_read_guard();
  auto resource_views = graph_runtime_resource_views(args, program);
  resource_views.ndarrays.add(predicate);
  resource_views.ndarrays.add(counter);
  if (status != nullptr) {
    resource_views.ndarrays.add(status);
  }
  std::optional<Program::RuntimeResourceGraphScope> resource_guard;
  if (!resource_views.empty()) {
    resource_guard.emplace(program->acquire_runtime_resource_graph_scope());
    if (!resource_views.ndarrays.empty()) {
      program->retain_ndarrays_for_external_submission(
          resource_views.ndarrays.data(), resource_views.ndarrays.size());
    }
    if (!resource_views.runtime_storage.empty()) {
      program->retain_runtime_storage_for_graph_submission(
          resource_views.runtime_storage.data(),
          resource_views.runtime_storage.size());
    }
    if (!resource_views.textures.empty()) {
      program->retain_textures_for_external_submission(
          resource_views.textures.data(), resource_views.textures.size());
    }
  }
  auto completion_scope = program->acquire_runtime_submission_scope();
  std::lock_guard<std::mutex> lock(cache.run_mutex);
  if (cache.validated_snode_tree_program != program ||
      cache.validated_snode_tree_epoch != tree_lifecycle_guard.epoch()) {
    try {
      program->validate_snode_tree_dependencies(snode_tree_dependencies);
    } catch (...) {
      cache.vulkan_graph_state.reset();
      cache.vulkan_inline_stats = {};
      cache.kernels.clear();
      cache.runtime_arg_plans.clear();
      cache.validated_snode_tree_program = nullptr;
      cache.validated_snode_tree_epoch = 0;
      throw;
    }
    cache.validated_snode_tree_program = program;
    cache.validated_snode_tree_epoch = tree_lifecycle_guard.epoch();
  }

  gfx::GfxRuntime::GraphStructuredControl control;
  control.predicate = predicate->get_device_allocation().get_ptr();
  control.counter = counter->get_device_allocation().get_ptr();
  control.initial_dispatch_count = initial_dispatch_count;
  control.max_iterations = static_cast<std::uint32_t>(max_iterations);
  control.has_status = status != nullptr;
  control.execute_initial_dispatches = execute_initial_dispatches;
  control.strategy =
      static_cast<gfx::GfxRuntime::GraphStructuredStrategy>(strategy);
  if (status != nullptr) {
    control.status = status->get_device_allocation().get_ptr();
  }
  gfx::GfxRuntime::GraphStructuredResult gfx_result;
  if (!try_run_vulkan_graph(*this, compile_config, args, cache,
                            &program->runtime_statistics(), &control,
                            wait_for_result ? &gfx_result : nullptr)) {
    return result;
  }
  program->mark_runtime_submission(
      RuntimeSubmissionKind::kGraphBackendSubmission);
  if (resource_guard) {
    resource_guard->finish_external_access_epoch();
  }
  result.submitted = wait_for_result ? gfx_result.submitted : true;
  if (!wait_for_result) {
    return result;
  }
  result.strategy = static_cast<std::uint32_t>(gfx_result.strategy);
  result.logical_iterations = gfx_result.logical_iterations;
  result.predicate = gfx_result.predicate;
  result.counter = gfx_result.counter;
  result.status = gfx_result.status;
  result.initial_status = gfx_result.initial_status;
  result.encoded_iterations = gfx_result.encoded_iterations;
  result.indirect_dispatches = gfx_result.indirect_dispatches;
  result.controller_dispatches = gfx_result.controller_dispatches;
  result.controller_invocations = gfx_result.controller_invocations;
  result.zero_dispatches = gfx_result.zero_dispatches;
  result.control_bytes = gfx_result.control_bytes;
  result.observation_bytes = gfx_result.observation_bytes;
#endif
  return result;
} catch (const BackendRuntimeError &error) {
  Program *program = jit_graph_program(*this);
  if (program != nullptr) {
    program->record_runtime_submission_failure();
    program->report_backend_runtime_error(error);
  }
  throw;
} catch (...) {
  Program *program = jit_graph_program(*this);
  if (program != nullptr) {
    program->record_runtime_submission_failure();
  }
  throw;
}

bool CompiledGraph::jit_submit_bounded_vulkan_compound_cached(
    const CompileConfig &compile_config,
    const std::unordered_map<std::string, IValue> &args,
    CompiledGraphJITCache &cache,
    Ndarray *predicate,
    Ndarray *counter,
    Ndarray *status,
    std::size_t initial_dispatch_count,
    const std::vector<int> &chunk_iterations,
    const std::vector<std::uint32_t> &strategies) const try {
#if defined(TI_WITH_VULKAN)
  if (compile_config.arch != Arch::vulkan || predicate == nullptr ||
      counter == nullptr || chunk_iterations.empty() ||
      chunk_iterations.size() != strategies.size() ||
      chunk_iterations.size() > 8 || initial_dispatch_count == 0 ||
      initial_dispatch_count >= dispatches.size()) {
    return false;
  }
  Program *program = jit_graph_program(*this);
  if (program == nullptr) {
    return false;
  }
  const auto valid_control = [&](const Ndarray *control) {
    return control != nullptr && control->get_nelement() == 1 &&
           control->get_element_data_type() == PrimitiveType::i32 &&
           control->owning_program() == program;
  };
  if (!valid_control(predicate) || !valid_control(counter) ||
      (status != nullptr && !valid_control(status))) {
    return false;
  }
  for (std::size_t chunk = 0; chunk < chunk_iterations.size(); ++chunk) {
    if (chunk_iterations[chunk] <= 0 ||
        static_cast<std::uint64_t>(chunk_iterations[chunk]) >
            std::numeric_limits<std::uint32_t>::max() ||
        strategies[chunk] >
            static_cast<std::uint32_t>(
                gfx::GfxRuntime::GraphStructuredStrategy::
                    coarse_conditional)) {
      return false;
    }
  }

  program->ensure_runtime_submission_allowed(
      "compound bounded Vulkan Graph submission");
  auto tree_lifecycle_guard =
      program->acquire_snode_tree_lifecycle_read_guard();
  auto resource_views = graph_runtime_resource_views(args, program);
  resource_views.ndarrays.add(predicate);
  resource_views.ndarrays.add(counter);
  if (status != nullptr) {
    resource_views.ndarrays.add(status);
  }
  std::optional<Program::RuntimeResourceGraphScope> resource_guard;
  if (!resource_views.empty()) {
    resource_guard.emplace(program->acquire_runtime_resource_graph_scope());
    if (!resource_views.ndarrays.empty()) {
      program->retain_ndarrays_for_external_submission(
          resource_views.ndarrays.data(), resource_views.ndarrays.size());
    }
    if (!resource_views.runtime_storage.empty()) {
      program->retain_runtime_storage_for_graph_submission(
          resource_views.runtime_storage.data(),
          resource_views.runtime_storage.size());
    }
    if (!resource_views.textures.empty()) {
      program->retain_textures_for_external_submission(
          resource_views.textures.data(), resource_views.textures.size());
    }
  }
  auto completion_scope = program->acquire_runtime_submission_scope();
  std::lock_guard<std::mutex> lock(cache.run_mutex);
  if (cache.validated_snode_tree_program != program ||
      cache.validated_snode_tree_epoch != tree_lifecycle_guard.epoch()) {
    try {
      program->validate_snode_tree_dependencies(snode_tree_dependencies);
    } catch (...) {
      cache.vulkan_graph_state.reset();
      cache.vulkan_inline_stats = {};
      cache.kernels.clear();
      cache.runtime_arg_plans.clear();
      cache.validated_snode_tree_program = nullptr;
      cache.validated_snode_tree_epoch = 0;
      throw;
    }
    cache.validated_snode_tree_program = program;
    cache.validated_snode_tree_epoch = tree_lifecycle_guard.epoch();
  }
  if (compile_config.debug ||
      (dispatches.size() <= 1 && !has_indirect_dispatches())) {
    return false;
  }

  PreparedVulkanGraphLaunch prepared;
  if (!prepare_vulkan_graph_launch(
          *this, compile_config, args, cache, prepared)) {
    return false;
  }

  bool execute_initial_dispatches = true;
  for (std::size_t chunk = 0; chunk < chunk_iterations.size(); ++chunk) {
    gfx::GfxRuntime::GraphStructuredControl control;
    control.predicate = predicate->get_device_allocation().get_ptr();
    control.counter = counter->get_device_allocation().get_ptr();
    control.initial_dispatch_count = initial_dispatch_count;
    control.max_iterations =
        static_cast<std::uint32_t>(chunk_iterations[chunk]);
    control.has_status = status != nullptr;
    control.execute_initial_dispatches = execute_initial_dispatches;
    control.strategy =
        static_cast<gfx::GfxRuntime::GraphStructuredStrategy>(
            strategies[chunk]);
    if (status != nullptr) {
      control.status = status->get_device_allocation().get_ptr();
    }
    if (!prepared.runtime->try_launch_graph(
            *prepared.dispatch_view, prepared.replay_key,
            &program->runtime_statistics(), &control,
            /*structured_result=*/nullptr)) {
      return false;
    }
    execute_initial_dispatches = false;
  }
  program->mark_runtime_submission(
      RuntimeSubmissionKind::kGraphBackendSubmission);
  if (resource_guard) {
    resource_guard->finish_external_access_epoch();
  }
  return true;
#else
  return false;
#endif
} catch (const BackendRuntimeError &error) {
  Program *program = jit_graph_program(*this);
  if (program != nullptr) {
    program->record_runtime_submission_failure();
    program->report_backend_runtime_error(error);
  }
  throw;
} catch (...) {
  Program *program = jit_graph_program(*this);
  if (program != nullptr) {
    program->record_runtime_submission_failure();
  }
  throw;
}

CompiledGraphNestedStructuredResult
CompiledGraph::jit_run_bounded_vulkan_nested_cached(
    const CompileConfig &compile_config,
    const std::unordered_map<std::string, IValue> &args,
    CompiledGraphJITCache &cache,
    Ndarray *outer_predicate,
    Ndarray *outer_counter,
    Ndarray *outer_status,
    Ndarray *inner_predicate,
    Ndarray *inner_counter,
    Ndarray *inner_status,
    std::size_t outer_condition_dispatch_count,
    std::size_t inner_condition_dispatch_begin,
    std::size_t inner_body_dispatch_begin,
    std::size_t outer_suffix_dispatch_begin,
    int outer_max_iterations,
    int inner_max_iterations,
    int inner_chunk_size,
    bool wait_for_result) const {
  CompiledGraphNestedInnerControl inner;
  inner.predicate = inner_predicate;
  inner.counter = inner_counter;
  inner.status = inner_status;
  inner.condition_dispatch_begin = inner_condition_dispatch_begin;
  inner.body_dispatch_begin = inner_body_dispatch_begin;
  inner.dispatch_end = outer_suffix_dispatch_begin;
  inner.max_iterations = inner_max_iterations;
  inner.chunk_size = inner_chunk_size;
  return jit_run_bounded_vulkan_nested_sequence_cached(
      compile_config, args, cache, outer_predicate, outer_counter,
      outer_status, {inner}, outer_condition_dispatch_count,
      outer_max_iterations, wait_for_result);
}

CompiledGraphNestedStructuredResult
CompiledGraph::jit_run_bounded_vulkan_nested_sequence_cached(
    const CompileConfig &compile_config,
    const std::unordered_map<std::string, IValue> &args,
    CompiledGraphJITCache &cache,
    Ndarray *outer_predicate,
    Ndarray *outer_counter,
    Ndarray *outer_status,
    const std::vector<CompiledGraphNestedInnerControl> &inner_controls,
    std::size_t outer_condition_dispatch_count,
    int outer_max_iterations,
    bool wait_for_result) const try {
  CompiledGraphNestedStructuredResult result;
#if defined(TI_WITH_VULKAN)
  constexpr int kNestedMaximumOuterIterations = 64;
  constexpr int kNestedMaximumChunkIterations = 64;
  constexpr std::size_t kNestedMaximumInnerRegions = 8;
  if (compile_config.arch != Arch::vulkan || outer_predicate == nullptr ||
      outer_counter == nullptr || inner_controls.empty() ||
      inner_controls.size() > kNestedMaximumInnerRegions ||
      outer_max_iterations <= 0 ||
      outer_max_iterations > kNestedMaximumOuterIterations ||
      outer_condition_dispatch_count == 0) {
    return result;
  }
  Program *program = jit_graph_program(*this);
  if (program == nullptr) {
    return result;
  }
  const auto valid_control = [&](const Ndarray *control) {
    return control != nullptr && control->get_nelement() == 1 &&
           control->get_element_data_type() == PrimitiveType::i32 &&
           control->owning_program() == program;
  };
  if (!valid_control(outer_predicate) || !valid_control(outer_counter) ||
      (outer_status != nullptr && !valid_control(outer_status))) {
    return result;
  }
  std::vector<DevicePtr> control_ptrs{
      outer_predicate->get_device_allocation().get_ptr(),
      outer_counter->get_device_allocation().get_ptr()};
  if (outer_status != nullptr) {
    control_ptrs.push_back(outer_status->get_device_allocation().get_ptr());
  }
  std::size_t boundary_cursor = outer_condition_dispatch_count;
  for (const auto &inner : inner_controls) {
    if (!valid_control(inner.predicate) || !valid_control(inner.counter) ||
        (inner.status != nullptr && !valid_control(inner.status)) ||
        inner.max_iterations <= 0 || inner.chunk_size <= 0 ||
        inner.max_iterations > kNestedMaximumChunkIterations ||
        inner.chunk_size > kNestedMaximumChunkIterations ||
        inner.chunk_size > inner.max_iterations ||
        boundary_cursor > inner.condition_dispatch_begin ||
        inner.condition_dispatch_begin >= inner.body_dispatch_begin ||
        inner.body_dispatch_begin >= inner.dispatch_end ||
        inner.dispatch_end >= dispatches.size()) {
      return result;
    }
    control_ptrs.push_back(
        inner.predicate->get_device_allocation().get_ptr());
    control_ptrs.push_back(inner.counter->get_device_allocation().get_ptr());
    if (inner.status != nullptr) {
      control_ptrs.push_back(inner.status->get_device_allocation().get_ptr());
    }
    boundary_cursor = inner.dispatch_end;
  }
  for (std::size_t i = 0; i < control_ptrs.size(); ++i) {
    for (std::size_t j = i + 1; j < control_ptrs.size(); ++j) {
      if (control_ptrs[i].alloc_id == control_ptrs[j].alloc_id &&
          control_ptrs[i].offset == control_ptrs[j].offset) {
        return result;
      }
    }
  }

  program->ensure_runtime_submission_allowed(
      "nested bounded Vulkan Graph launch");
  auto tree_lifecycle_guard =
      program->acquire_snode_tree_lifecycle_read_guard();
  auto resource_views = graph_runtime_resource_views(args, program);
  resource_views.ndarrays.add(outer_predicate);
  resource_views.ndarrays.add(outer_counter);
  if (outer_status != nullptr) {
    resource_views.ndarrays.add(outer_status);
  }
  for (const auto &inner : inner_controls) {
    resource_views.ndarrays.add(inner.predicate);
    resource_views.ndarrays.add(inner.counter);
    if (inner.status != nullptr) {
      resource_views.ndarrays.add(inner.status);
    }
  }
  std::optional<Program::RuntimeResourceGraphScope> resource_guard;
  if (!resource_views.empty()) {
    resource_guard.emplace(program->acquire_runtime_resource_graph_scope());
    if (!resource_views.ndarrays.empty()) {
      program->retain_ndarrays_for_external_submission(
          resource_views.ndarrays.data(), resource_views.ndarrays.size());
    }
    if (!resource_views.runtime_storage.empty()) {
      program->retain_runtime_storage_for_graph_submission(
          resource_views.runtime_storage.data(),
          resource_views.runtime_storage.size());
    }
    if (!resource_views.textures.empty()) {
      program->retain_textures_for_external_submission(
          resource_views.textures.data(), resource_views.textures.size());
    }
  }
  auto completion_scope = program->acquire_runtime_submission_scope();
  std::lock_guard<std::mutex> lock(cache.run_mutex);
  if (cache.validated_snode_tree_program != program ||
      cache.validated_snode_tree_epoch != tree_lifecycle_guard.epoch()) {
    try {
      program->validate_snode_tree_dependencies(snode_tree_dependencies);
    } catch (...) {
      cache.vulkan_graph_state.reset();
      cache.vulkan_inline_stats = {};
      cache.kernels.clear();
      cache.runtime_arg_plans.clear();
      cache.validated_snode_tree_program = nullptr;
      cache.validated_snode_tree_epoch = 0;
      throw;
    }
    cache.validated_snode_tree_program = program;
    cache.validated_snode_tree_epoch = tree_lifecycle_guard.epoch();
  }

  gfx::GfxRuntime::GraphNestedStructuredControl control;
  control.outer_predicate =
      outer_predicate->get_device_allocation().get_ptr();
  control.outer_counter = outer_counter->get_device_allocation().get_ptr();
  control.outer_has_status = outer_status != nullptr;
  if (outer_status != nullptr) {
    control.outer_status = outer_status->get_device_allocation().get_ptr();
  }
  control.outer_condition_dispatch_count =
      outer_condition_dispatch_count;
  control.outer_max_iterations =
      static_cast<std::uint32_t>(outer_max_iterations);
  control.inner_controls.reserve(inner_controls.size());
  for (const auto &inner : inner_controls) {
    gfx::GfxRuntime::GraphNestedStructuredInnerControl gfx_inner;
    gfx_inner.inner_predicate =
        inner.predicate->get_device_allocation().get_ptr();
    gfx_inner.inner_counter =
        inner.counter->get_device_allocation().get_ptr();
    gfx_inner.inner_has_status = inner.status != nullptr;
    if (inner.status != nullptr) {
      gfx_inner.inner_status =
          inner.status->get_device_allocation().get_ptr();
    }
    gfx_inner.inner_condition_dispatch_begin =
        inner.condition_dispatch_begin;
    gfx_inner.inner_body_dispatch_begin = inner.body_dispatch_begin;
    gfx_inner.inner_dispatch_end = inner.dispatch_end;
    gfx_inner.inner_max_iterations =
        static_cast<std::uint32_t>(inner.max_iterations);
    gfx_inner.inner_chunk_size =
        static_cast<std::uint32_t>(inner.chunk_size);
    control.inner_controls.push_back(gfx_inner);
  }

  gfx::GfxRuntime::GraphNestedStructuredResult gfx_result;
  if (!try_run_vulkan_graph(*this, compile_config, args, cache,
                            &program->runtime_statistics(), nullptr, nullptr,
                            &control,
                            wait_for_result ? &gfx_result : nullptr)) {
    return result;
  }
  program->mark_runtime_submission(
      RuntimeSubmissionKind::kGraphBackendSubmission);
  if (resource_guard) {
    resource_guard->finish_external_access_epoch();
  }
  if (!wait_for_result) {
    result.submitted = true;
    return result;
  }
  result.submitted = gfx_result.submitted;
  result.inner_region_count = gfx_result.inner_region_count;
  result.outer_logical_iterations =
      gfx_result.outer_logical_iterations;
  result.outer_encoded_iterations =
      gfx_result.outer_encoded_iterations;
  result.outer_initial_predicate =
      gfx_result.outer_initial_predicate;
  result.outer_final_predicate = gfx_result.outer_final_predicate;
  result.outer_initial_counter = gfx_result.outer_initial_counter;
  result.outer_final_counter = gfx_result.outer_final_counter;
  result.outer_initial_status = gfx_result.outer_initial_status;
  result.outer_final_status = gfx_result.outer_final_status;
  result.inner_logical_iterations =
      std::move(gfx_result.inner_logical_iterations);
  result.inner_encoded_iterations =
      std::move(gfx_result.inner_encoded_iterations);
  result.inner_initial_counters =
      std::move(gfx_result.inner_initial_counters);
  result.inner_final_counters =
      std::move(gfx_result.inner_final_counters);
  result.inner_final_predicates =
      std::move(gfx_result.inner_final_predicates);
  result.inner_initial_statuses =
      std::move(gfx_result.inner_initial_statuses);
  result.inner_final_statuses =
      std::move(gfx_result.inner_final_statuses);
  result.indirect_dispatches = gfx_result.indirect_dispatches;
  result.controller_dispatches = gfx_result.controller_dispatches;
  result.controller_invocations = gfx_result.controller_invocations;
  result.zero_dispatches = gfx_result.zero_dispatches;
  result.control_bytes = gfx_result.control_bytes;
  result.observation_bytes = gfx_result.observation_bytes;
#endif
  return result;
} catch (const BackendRuntimeError &error) {
  Program *program = jit_graph_program(*this);
  if (program != nullptr) {
    program->record_runtime_submission_failure();
    program->report_backend_runtime_error(error);
  }
  throw;
} catch (...) {
  Program *program = jit_graph_program(*this);
  if (program != nullptr) {
    program->record_runtime_submission_failure();
  }
  throw;
}

bool CompiledGraph::jit_run_conditional_cuda_cached(
    const CompileConfig &compile_config,
    const std::unordered_map<std::string, IValue> &args,
    CompiledGraphJITCache &cache,
    Ndarray *selector,
    const std::vector<int> &branch_dispatch_counts,
    int conditional_type,
    int default_branch) const try {
#if defined(TI_WITH_CUDA)
  if (compile_config.arch != Arch::cuda || selector == nullptr) {
    return false;
  }
  Program *program = jit_graph_program(*this);
  if (program == nullptr) {
    return false;
  }
  program->ensure_runtime_submission_allowed(
      "conditional CUDA Graph launch");
  auto tree_lifecycle_guard =
      program->acquire_snode_tree_lifecycle_read_guard();
  auto resource_views = graph_runtime_resource_views(args, program);
  std::optional<Program::RuntimeResourceGraphScope> resource_guard;
  if (!resource_views.empty()) {
    resource_guard.emplace(program->acquire_runtime_resource_graph_scope());
    if (!resource_views.ndarrays.empty()) {
      program->retain_ndarrays_for_external_submission(
          resource_views.ndarrays.data(), resource_views.ndarrays.size());
    }
    if (!resource_views.runtime_storage.empty()) {
      program->retain_runtime_storage_for_graph_submission(
          resource_views.runtime_storage.data(),
          resource_views.runtime_storage.size());
    }
    if (!resource_views.textures.empty()) {
      program->retain_textures_for_external_submission(
          resource_views.textures.data(), resource_views.textures.size());
    }
  }
  auto completion_scope = program->acquire_runtime_submission_scope();
  auto cuda_submission_lock =
      CUDAContext::get_instance().get_submission_lock_guard();
  std::lock_guard<std::mutex> lock(cache.run_mutex);
  if (cache.validated_snode_tree_program != program ||
      cache.validated_snode_tree_epoch != tree_lifecycle_guard.epoch()) {
    try {
      program->validate_snode_tree_dependencies(snode_tree_dependencies);
    } catch (...) {
      cache.cuda_graph_state.reset();
      cache.cuda_graph_state_alternates.clear();
      cache.kernels.clear();
      cache.runtime_arg_plans.clear();
      cache.validated_snode_tree_program = nullptr;
      cache.validated_snode_tree_epoch = 0;
      throw;
    }
    cache.validated_snode_tree_program = program;
    cache.validated_snode_tree_epoch = tree_lifecycle_guard.epoch();
  }
  if (!try_run_cuda_conditional_graph(
          *this, compile_config, args, cache, *program, *selector,
          branch_dispatch_counts, conditional_type, default_branch,
          &program->runtime_statistics())) {
    return false;
  }
  program->mark_runtime_submission(
      RuntimeSubmissionKind::kGraphBackendSubmission);
  return true;
#else
  return false;
#endif
} catch (const BackendRuntimeError &error) {
  Program *program = jit_graph_program(*this);
  if (program != nullptr) {
    program->record_runtime_submission_failure();
    program->report_backend_runtime_error(error);
  }
  throw;
} catch (...) {
  Program *program = jit_graph_program(*this);
  if (program != nullptr) {
    program->record_runtime_submission_failure();
  }
  throw;
}

bool CompiledGraph::jit_run_conditional_cuda_masked_cached(
    const CompileConfig &compile_config,
    const std::unordered_map<std::string, IValue> &args,
    CompiledGraphJITCache &cache,
    Ndarray *selector,
    const std::vector<int> &branch_dispatch_counts,
    int conditional_type,
    int default_branch) const try {
#if defined(TI_WITH_CUDA)
  if (compile_config.arch != Arch::cuda || selector == nullptr) {
    return false;
  }
  Program *program = jit_graph_program(*this);
  if (program == nullptr) {
    return false;
  }
  program->ensure_runtime_submission_allowed(
      "masked conditional CUDA Graph launch");
  auto tree_lifecycle_guard =
      program->acquire_snode_tree_lifecycle_read_guard();
  auto resource_views = graph_runtime_resource_views(args, program);
  std::optional<Program::RuntimeResourceGraphScope> resource_guard;
  if (!resource_views.empty()) {
    resource_guard.emplace(program->acquire_runtime_resource_graph_scope());
    if (!resource_views.ndarrays.empty()) {
      program->retain_ndarrays_for_external_submission(
          resource_views.ndarrays.data(), resource_views.ndarrays.size());
    }
    if (!resource_views.runtime_storage.empty()) {
      program->retain_runtime_storage_for_graph_submission(
          resource_views.runtime_storage.data(),
          resource_views.runtime_storage.size());
    }
    if (!resource_views.textures.empty()) {
      program->retain_textures_for_external_submission(
          resource_views.textures.data(), resource_views.textures.size());
    }
  }
  auto completion_scope = program->acquire_runtime_submission_scope();
  auto cuda_submission_lock =
      CUDAContext::get_instance().get_submission_lock_guard();
  std::lock_guard<std::mutex> lock(cache.run_mutex);
  if (cache.validated_snode_tree_program != program ||
      cache.validated_snode_tree_epoch != tree_lifecycle_guard.epoch()) {
    try {
      program->validate_snode_tree_dependencies(snode_tree_dependencies);
    } catch (...) {
      cache.cuda_graph_state.reset();
      cache.cuda_graph_state_alternates.clear();
      cache.kernels.clear();
      cache.runtime_arg_plans.clear();
      cache.validated_snode_tree_program = nullptr;
      cache.validated_snode_tree_epoch = 0;
      throw;
    }
    cache.validated_snode_tree_program = program;
    cache.validated_snode_tree_epoch = tree_lifecycle_guard.epoch();
  }
  if (!try_run_cuda_masked_control_graph(
          *this, compile_config, args, cache, *program, *selector,
          conditional_type, /*max_iterations=*/0,
          /*continue_while_nonzero=*/true, branch_dispatch_counts,
          default_branch, &program->runtime_statistics())) {
    return false;
  }
  program->mark_runtime_submission(
      RuntimeSubmissionKind::kGraphBackendSubmission);
  return true;
#else
  return false;
#endif
} catch (const BackendRuntimeError &error) {
  Program *program = jit_graph_program(*this);
  if (program != nullptr) {
    program->record_runtime_submission_failure();
    program->report_backend_runtime_error(error);
  }
  throw;
} catch (...) {
  Program *program = jit_graph_program(*this);
  if (program != nullptr) {
    program->record_runtime_submission_failure();
  }
  throw;
}

// static
void CompiledGraph::init_runtime_context(
    const std::vector<Arg> &paramter_list,
    const std::unordered_map<std::string, IValue> &args,
    LaunchContextBuilder &ctx) {
  TI_COMPILE_PROFILER("compiled_graph_init_runtime_context");
  for (int i = 0; i < paramter_list.size(); ++i) {
    auto &symbolic_arg = paramter_list[i];
    const std::vector<int> arg_id{i};
    auto found = args.find(symbolic_arg.name);
    TI_ERROR_IF(found == args.end(), "Missing runtime value for {}",
                symbolic_arg.name);
    const aot::IValue &ival = found->second;
    if (symbolic_arg.tag == aot::ArgKind::kNdarray) {
      TI_ASSERT(ival.tag == aot::ArgKind::kNdarray);
      if (ival.runtime_storage != nullptr) {
        validate_graph_runtime_storage_argument(
            *ival.runtime_storage, symbolic_arg.name, symbolic_arg.dtype_id,
            symbolic_arg.field_dim, symbolic_arg.element_shape);
        ctx.set_arg_runtime_storage(arg_id, *ival.runtime_storage);
        continue;
      }
      Ndarray *arr = reinterpret_cast<Ndarray *>(ival.val);
      TI_ERROR_IF(arr == nullptr, "Graph received a null Ndarray runtime arg");

      TI_ERROR_IF(arr->get_element_shape() != symbolic_arg.element_shape,
                  "Mismatched shape information for argument {}",
                  symbolic_arg.name);
      TI_ERROR_IF(arr->shape.size() != symbolic_arg.field_dim,
                  "Dispatch node is compiled for argument {} with "
                  "field_dim={} but got an ndarray with field_dim={}",
                  symbolic_arg.name, symbolic_arg.field_dim, arr->shape.size());

      DataType symbolic_arg_element_dtype = symbolic_arg.element_dtype();
      DataType symbolic_arg_primitive_dtype = symbolic_arg_element_dtype;
      if (symbolic_arg_element_dtype->is<TensorType>()) {
        symbolic_arg_primitive_dtype =
            symbolic_arg_element_dtype->cast<TensorType>()->get_element_type();
      }

      DataType arr_primitive_dtype = arr->dtype;
      if (arr->dtype->is<TensorType>()) {
        arr_primitive_dtype =
            arr->dtype->cast<TensorType>()->get_element_type();
      }

      TI_ERROR_IF(arr_primitive_dtype != symbolic_arg_primitive_dtype,
                  "Dispatch node is compiled for argument {} with "
                  "dtype={} but got an ndarray with dtype={}",
                  symbolic_arg.name, symbolic_arg_primitive_dtype.to_string(),
                  arr_primitive_dtype.to_string());
      ctx.set_arg_ndarray(arg_id, *arr);
    } else if (symbolic_arg.tag == aot::ArgKind::kScalar) {
      TI_ASSERT(ival.tag == aot::ArgKind::kScalar);
      // Matrix args are flattened so they're same as scalars.
      int type_size = data_type_size(symbolic_arg.dtype());
      switch (type_size) {
        case 1:
          ctx.set_arg(arg_id,
                      taichi_union_cast_with_different_sizes<int8>(ival.val));
          break;
        case 2:
          ctx.set_arg(arg_id,
                      taichi_union_cast_with_different_sizes<int16>(ival.val));
          break;
        case 4:
          ctx.set_arg(arg_id,
                      taichi_union_cast_with_different_sizes<int32>(ival.val));
          break;
        case 8:
          ctx.set_arg(arg_id,
                      taichi_union_cast_with_different_sizes<int64>(ival.val));
          break;
        default:
          TI_ERROR("Unsupported type size {}", type_size);
      }
    } else if (symbolic_arg.tag == aot::ArgKind::kTexture) {
      TI_ASSERT(ival.tag == aot::ArgKind::kTexture);
      Texture *tex = reinterpret_cast<Texture *>(ival.val);
      ctx.set_arg_texture(arg_id, *tex);
    } else if (symbolic_arg.tag == aot::ArgKind::kRWTexture) {
      TI_ASSERT(ival.tag == aot::ArgKind::kTexture);
      Texture *tex = reinterpret_cast<Texture *>(ival.val);
      ctx.set_arg_rw_texture(arg_id, *tex);
    } else if (symbolic_arg.tag == aot::ArgKind::kMatrix) {
      TI_ASSERT(ival.tag == aot::ArgKind::kMatrix);
      Matrix *mat = reinterpret_cast<Matrix *>(ival.val);

      TI_ERROR_IF(symbolic_arg.element_shape.empty() ||
                      symbolic_arg.element_shape.size() > 2,
                  "Matrix argument {} has unsupported element shape {}",
                  symbolic_arg.name, symbolic_arg.element_shape);
      uint64_t symbolic_arg_size = 1;
      for (int dimension : symbolic_arg.element_shape) {
        TI_ERROR_IF(dimension <= 0,
                    "Matrix argument {} has invalid element shape {}",
                    symbolic_arg.name, symbolic_arg.element_shape);
        symbolic_arg_size *= static_cast<uint64_t>(dimension);
      }
      TI_ERROR_IF(symbolic_arg_size != mat->length(),
                  "Dispatch node is compiled for argument {} with "
                  "size={} but got a matrix with size={}",
                  symbolic_arg.name, symbolic_arg_size, mat->length());
      if (mat->ndim() != 0) {
        bool shape_matches = mat->ndim() == symbolic_arg.element_shape.size();
        if (shape_matches) {
          for (uint32_t axis = 0; axis < mat->ndim(); ++axis) {
            shape_matches &=
                mat->shape(axis) == symbolic_arg.element_shape[axis];
          }
        }
        // AOT graphs produced before the native vector descriptor encoded a
        // vector as [N, 1]. Keep that 0.5.x representation readable while new
        // Graph Args use the canonical rank-1 [N] tensor shape.
        const bool legacy_vector_shape =
            mat->ndim() == 1 && symbolic_arg.element_shape.size() == 2 &&
            symbolic_arg.element_shape[1] == 1 &&
            mat->shape(0) == symbolic_arg.element_shape[0];
        // Matrix values historically accepted a flat ti.Matrix([...]) whose
        // length matched an NxM annotation. Keep that source-compatible form,
        // but reject a genuinely rank-2 value with transposed dimensions.
        const bool legacy_flat_matrix =
            mat->ndim() == 1 && symbolic_arg.element_shape.size() == 2 &&
            mat->shape(0) == symbolic_arg_size;
        TI_ERROR_IF(!shape_matches && !legacy_vector_shape &&
                        !legacy_flat_matrix,
                    "Dispatch node is compiled for Matrix argument {} with "
                    "shape={} but got rank {} runtime data",
                    symbolic_arg.name, symbolic_arg.element_shape, mat->ndim());
      }
      TI_ERROR_IF(mat->length() * data_type_size(mat->dtype()) > 128,
                  "Matrix size={} is out of bound",
                  mat->length() * data_type_size(mat->dtype()));
      ctx.set_arg_matrix(i, *mat);
    } else {
      TI_ERROR("Error in compiled graph: unknown tag {}", int(ival.tag));
    }
  }
}

}  // namespace aot
}  // namespace taichi::lang
