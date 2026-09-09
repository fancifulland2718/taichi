#include "taichi/program/program.h"
#include "taichi/program/ndarray.h"

#ifdef TI_WITH_VULKAN
#include "taichi/rhi/vulkan/vulkan_device.h"

namespace taichi::lang {
namespace {
// Only this provider understands the shader ABI; no generic Graph schema or
// primitive cache depends on FidelityFX's bins, passes, or workspace layout.
struct SortResources final : vkapi::DeviceObj {
  std::shared_ptr<void> storage_lease;
  std::vector<DeviceAllocationUnique> workspace;
  std::vector<std::unique_ptr<Pipeline>> pipelines;
  std::vector<std::unique_ptr<ShaderResourceSet>> bindings;
};
struct Parameters {
  uint32_t keys;
  int32_t blocks_per_group;
  uint32_t groups;
  uint32_t extra_groups;
  uint32_t reduced_per_bin;
  uint32_t scan_values;
  uint32_t shift;
};
static_assert(sizeof(Parameters) == 28);
}  // namespace

class VulkanParallelSortPlan {
 public:
  std::function<void(CommandList *)> replay;
  std::unordered_map<std::string, std::uint64_t> statistics;
};

std::uint64_t Program::create_vulkan_parallel_sort_plan(
    Ndarray *keys,
    Ndarray *values,
    const std::vector<std::vector<std::uint32_t>> &shaders) {
  auto guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Parallel Sort requires the Vulkan backend.");
  std::vector<const Ndarray *> arrays{keys};
  if (values) {
    arrays.push_back(values);
  }
  auto leases = std::make_shared<NdarrayLaunchLeases>(acquire_ndarray_leases(arrays));
  for (const auto *array : arrays) {
    TI_ERROR_IF(!array || array->owning_program() != this ||
                    array->dtype != PrimitiveType::u32 || array->shape.size() != 1 ||
                    array->shape[0] <= 0,
                "Parallel Sort requires local, nonempty, scalar u32 ndarrays.");
  }
  auto *dev = static_cast<vulkan::VulkanDevice *>(get_graphics_device());
  const auto key = keys->get_device_allocation();
  TI_ERROR_IF(key.device != dev || (values &&
                  (values->shape != keys->shape ||
                   values->get_device_allocation().device != dev ||
                   values->get_device_allocation() == key)),
              "Parallel Sort payload must be distinct local storage of the same shape.");
  const auto &caps = dev->get_caps();
  TI_ERROR_IF(!caps.get(DeviceCapability::spirv_has_subgroup_basic) ||
                  !caps.get(DeviceCapability::spirv_has_subgroup_arithmetic) ||
                  !caps.get(DeviceCapability::spirv_has_subgroup_ballot),
              "Parallel Sort requires compute subgroup basic/arithmetic/ballot support.");
  VkPhysicalDeviceSubgroupProperties subgroup{};
  subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
  VkPhysicalDeviceProperties2 properties2{};
  properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  properties2.pNext = &subgroup;
  TI_ERROR_IF(!vkGetPhysicalDeviceProperties2,
              "Parallel Sort requires Vulkan 1.1 subgroup properties.");
  vkGetPhysicalDeviceProperties2(dev->vk_physical_device(), &properties2);
  TI_ERROR_IF(!(subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) ||
                  !(subgroup.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT),
              "Parallel Sort requires compute subgroup shuffle support.");
  TI_ERROR_IF(shaders.size() != 3 && shaders.size() != 5,
              "Parallel Sort requires three fused or five original provider shaders.");
  const bool fused = shaders.size() == 3;
  const int steps = fused ? 3 : 5;
  auto resources = std::make_shared<SortResources>();
  resources->device = dev->vk_device();
  resources->storage_lease = std::move(leases);
  uint64_t workspace_bytes = 0;
  auto allocate = [&](uint64_t bytes) {
    Device::AllocParams params;
    params.size = bytes;
    params.usage = AllocUsage::Storage;
    auto [allocation, result] = dev->allocate_memory_unique(params);
    TI_ERROR_IF(result != RhiResult::success,
                "Parallel Sort workspace allocation failed: {}", int(result));
    const DeviceAllocation value = *allocation;
    resources->workspace.push_back(std::move(allocation));
    workspace_bytes += bytes;
    return value;
  };
  const uint32_t n = keys->shape[0];
  const uint32_t blocks = (uint64_t(n) + 511) / 512;
  const uint32_t groups = std::min(blocks, 1024u);
  const uint32_t reduced_per_bin = (groups + 511) / 512;
  Parameters parameters{n, int32_t(blocks / groups), groups, blocks % groups,
                        reduced_per_bin, 16 * reduced_per_bin, 0};
  const auto scratch_key = allocate(uint64_t(n) * 4);
  const auto scratch_value = values ? allocate(uint64_t(n) * 4) : key;
  const auto sums = allocate(uint64_t(groups) * 16 * 4);
  const auto reduced = allocate(uint64_t(fused ? groups * 16 : parameters.scan_values) * 4);
  const auto value = values ? values->get_device_allocation() : key;
  constexpr const char *names[] = {"count", "reduce", "scan", "scan_add", "scatter"};
  for (int step = 0; step < steps; ++step) {
    const auto &shader = shaders[step];
    TI_ERROR_IF(shader.empty() || shader.front() != 0x07230203u ||
                    shader.size() > (4u << 20),
                "Parallel Sort requires bounded SPIR-V binaries.");
    PipelineSourceDesc source{PipelineSourceType::spirv_binary, shader.data(),
                              shader.size() * 4, PipelineStageType::compute};
    auto [pipeline, result] = dev->create_pipeline_unique(
        source, std::string("fidelityfx_parallel_sort_") +
                    (fused ? (step == 0 ? "count" : step == 1 ? "prefix" : "scatter") : names[step]));
    TI_ERROR_IF(result != RhiResult::success,
                "Parallel Sort pipeline creation failed: {}", int(result));
    resources->pipelines.push_back(std::move(pipeline));
  }
  // Bind and record once. All eight passes return the final data to the
  // original storage, so neither host staging nor a final device copy exists.
  auto commands = dev->get_compute_stream()->new_secondary_command_list();
  TI_ERROR_IF(!commands, "Parallel Sort requires secondary compute recording.");
  auto *list = static_cast<vulkan::VulkanCommandList *>(commands.get());
  list->memory_barrier();
  const uint32_t dispatches[] = {groups, parameters.scan_values, 1,
                                 parameters.scan_values, groups};
  for (uint32_t pass = 0; pass < 8; ++pass) {
    parameters.shift = pass * 4;
    for (int index = 0; index < steps; ++index) {
      const int step = fused && index == 2 ? 4 : index;
      auto binding = std::unique_ptr<ShaderResourceSet>(dev->create_resource_set());
      if (step == 0 || step == 4) {
        binding->rw_buffer(0, pass % 2 ? scratch_key : key);
        binding->rw_buffer(4, fused && step == 4 ? reduced : sums);
      }
      if (step == 4) {
        binding->rw_buffer(1, pass % 2 ? key : scratch_key);
        if (values) {
          binding->rw_buffer(2, pass % 2 ? scratch_value : value);
          binding->rw_buffer(3, pass % 2 ? value : scratch_value);
        }
      } else if (step == 1) {
        binding->rw_buffer(4, sums);
        binding->rw_buffer(5, reduced);
      } else if (step == 2 || step == 3) {
        binding->rw_buffer(6, step == 2 ? reduced : sums);
        binding->rw_buffer(7, step == 2 ? reduced : sums);
        if (step == 3) {
          binding->rw_buffer(8, reduced);
        }
      }
      list->bind_pipeline(resources->pipelines[index].get());
      TI_ERROR_IF(list->bind_shader_resources(binding.get()) != RhiResult::success,
                  "Parallel Sort shader binding failed.");
      list->push_constants(&parameters, sizeof(parameters));
      TI_ERROR_IF(list->dispatch(fused && step == 1 ? 1 : dispatches[step]) != RhiResult::success,
                  "Parallel Sort dispatch recording failed.");
      list->memory_barrier();
      resources->bindings.push_back(std::move(binding));
    }
  }
  auto plan = std::make_shared<VulkanParallelSortPlan>();
  plan->replay = list->finalize_secondary(resources);
  TI_ERROR_IF(!plan->replay, "Parallel Sort requires secondary compute recording.");
  const auto &properties = dev->get_vk_physical_device_props();
  plan->statistics = {{"key_count", n}, {"payload", values ? 1 : 0},
                      {"dispatch_count", uint64_t(steps * 8)}, {"barrier_count", uint64_t(steps * 8 + 1)},
                      {"fused_prefix", fused ? 1 : 0},
                      {"threadgroups", groups}, {"workspace_bytes", workspace_bytes},
                      {"device_copy_count", 0}, {"device_vendor_id", properties.vendorID},
                      {"device_id", properties.deviceID},
                      {"vulkan_driver_version", properties.driverVersion}};
  const auto handle = next_vulkan_parallel_sort_plan_handle_++;
  vulkan_parallel_sort_plans_.emplace(handle, std::move(plan));
  return handle;
}

void Program::vulkan_parallel_sort_execute(std::uint64_t handle) {
  auto guard = acquire_runtime_resource_submission_guard();
  const auto found = vulkan_parallel_sort_plans_.find(handle);
  TI_ERROR_IF(found == vulkan_parallel_sort_plans_.end(), "Parallel Sort plan is closed.");
  auto plan = found->second;
  enqueue_compute_op_lambda([plan](Device *, CommandList *commands) {
    plan->replay(commands);
  }, {});
  mark_runtime_submission_pending();
}

std::unordered_map<std::string, std::uint64_t>
Program::vulkan_parallel_sort_plan_statistics(std::uint64_t handle) {
  auto guard = acquire_runtime_resource_submission_guard();
  const auto found = vulkan_parallel_sort_plans_.find(handle);
  TI_ERROR_IF(found == vulkan_parallel_sort_plans_.end(), "Parallel Sort plan is closed.");
  return found->second->statistics;
}

void Program::destroy_vulkan_parallel_sort_plan(std::uint64_t handle) {
  std::lock_guard<std::recursive_mutex> lock(runtime_resource_submission_mutex_);
  vulkan_parallel_sort_plans_.erase(handle);
}
void Program::vulkan_clear_parallel_sort_plans() {
  // Teardown runs after submissions close, but before destroying the device.
  // Do not use the live-submission guard on this retirement boundary.
  std::lock_guard<std::recursive_mutex> lock(runtime_resource_submission_mutex_);
  vulkan_parallel_sort_plans_.clear();
}
}  // namespace taichi::lang
#else
namespace taichi::lang {
std::uint64_t Program::create_vulkan_parallel_sort_plan(
    Ndarray *, Ndarray *, const std::vector<std::vector<std::uint32_t>> &) {
  TI_ERROR("Parallel Sort is unavailable in this build.");
}
void Program::vulkan_parallel_sort_execute(std::uint64_t) {
  TI_ERROR("Parallel Sort is unavailable in this build.");
}
std::unordered_map<std::string, std::uint64_t>
Program::vulkan_parallel_sort_plan_statistics(std::uint64_t) {
  TI_ERROR("Parallel Sort is unavailable in this build.");
}
void Program::destroy_vulkan_parallel_sort_plan(std::uint64_t) {}
void Program::vulkan_clear_parallel_sort_plans() {}
}  // namespace taichi::lang
#endif
