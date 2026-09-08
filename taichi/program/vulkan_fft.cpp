#include "taichi/program/program.h"

#include "taichi/program/ndarray.h"

#ifdef TI_WITH_VULKAN
#include "taichi/common/dynamic_loader.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/vkfft/forge_vkfft_provider.h"

namespace taichi::lang {

class VulkanFftPlan : public vkapi::DeviceObj {
 public:
  explicit VulkanFftPlan(const std::string &path) : library(path) {
    TI_ERROR_IF(!library.loaded(), "Vulkan FFT adapter is unavailable: {}",
                path);
    TiForgeVkfftQueryFn query{};
    library.load_function(TI_FORGE_VKFFT_QUERY_SYMBOL, query);
    TI_ERROR_IF(query(TI_FORGE_VKFFT_ABI_VERSION, sizeof(api), &api) != 0 ||
                    api.struct_size != sizeof(api) ||
                    api.abi_version != TI_FORGE_VKFFT_ABI_VERSION ||
                    api.vkfft_version != 10304 || !api.create || !api.append ||
                    !api.memory || !api.destroy || !api.last_error,
                "Vulkan FFT adapter ABI is incompatible: {}", path);
    auto recipe_query = reinterpret_cast<TiForgeVkfftRecipeQueryFn>(
        library.load_function_optional(TI_FORGE_VKFFT_RECIPE_QUERY_SYMBOL));
    if (recipe_query) {
      TI_ERROR_IF(
          recipe_query(TI_FORGE_VKFFT_RECIPE_ABI_VERSION, sizeof(recipe_api),
                       &recipe_api) != 0 ||
              recipe_api.struct_size != sizeof(recipe_api) ||
              recipe_api.abi_version != TI_FORGE_VKFFT_RECIPE_ABI_VERSION ||
              !recipe_api.create || !recipe_api.describe,
          "Vulkan FFT recipe extension ABI is incompatible: {}", path);
    }
  }

  ~VulkanFftPlan() override {
    if (handle) {
      api.destroy(handle);
    }
  }

  DynamicLoader library;
  TiForgeVkfftApi api{};
  TiForgeVkfftRecipeApi recipe_api{};
  TiForgeVkfftPlan handle{};
  std::shared_ptr<void> storage_lease;
  vkapi::IVkBuffer buffer;
  uint32_t vendor_id{0};
  uint32_t device_id{0};
  uint32_t driver_version{0};
  uint32_t api_version{0};
};

std::uint64_t Program::create_vulkan_fft_plan(
    const std::string &adapter_path,
    Ndarray *data,
    const std::vector<int> &dimensions,
    int batches,
    int direction,
    bool normalize_inverse) {
  return create_vulkan_fft_recipe_plan(adapter_path, data, dimensions, batches,
                                       direction, normalize_inverse, 0);
}

std::uint64_t Program::create_vulkan_fft_recipe_plan(
    const std::string &adapter_path,
    Ndarray *data,
    const std::vector<int> &dimensions,
    int batches,
    int direction,
    bool normalize_inverse,
    int batch_tile) {
  std::lock_guard<std::recursive_mutex> lock(
      runtime_resource_submission_mutex_);
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan FFT requires the Vulkan backend.");
  TI_ERROR_IF(!data || data->owning_program() != this ||
                  data->dtype != PrimitiveType::f32 || dimensions.empty() ||
                  dimensions.size() > 3 || batches <= 0,
              "Vulkan FFT requires a local compact scalar-f32 ndarray and "
              "one to three positive transform dimensions.");
  TI_ERROR_IF(
      batch_tile < 0 || batch_tile > batches,
      "Vulkan FFT batch tile must be zero (legacy) or in [1, batches].");
  std::vector<int> shape;
  if (batches > 1) {
    shape.push_back(batches);
  }
  for (int dimension : dimensions) {
    TI_ERROR_IF(dimension <= 0, "Vulkan FFT dimensions must be positive.");
    shape.push_back(dimension);
  }
  shape.push_back(2);
  TI_ERROR_IF(data->shape != shape,
              "Vulkan FFT ndarray shape must match compact complex batches.");
  auto leases =
      std::make_shared<NdarrayLaunchLeases>(acquire_ndarray_leases({data}));
  auto *device = static_cast<vulkan::VulkanDevice *>(get_graphics_device());
  const auto allocation = data->get_device_allocation();
  TI_ERROR_IF(allocation.device != device,
              "Vulkan FFT storage belongs to another device.");
  auto plan = std::make_shared<VulkanFftPlan>(adapter_path);
  plan->device = device->vk_device();
  const auto &properties = device->get_vk_physical_device_props();
  plan->vendor_id = properties.vendorID;
  plan->device_id = properties.deviceID;
  plan->driver_version = properties.driverVersion;
  plan->api_version = properties.apiVersion;
  plan->storage_lease = std::move(leases);
  plan->buffer = device->get_vkbuffer(allocation);
  TiForgeVkfftConfig config{};
  config.struct_size = sizeof(config);
  config.rank = static_cast<uint32_t>(dimensions.size());
  for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
    config.dimensions[axis] = dimensions[axis];
  }
  config.batches = batches;
  config.physical_device = device->vk_physical_device();
  config.device = device->vk_device();
  config.queue = device->compute_queue();
  config.queue_family = device->compute_queue_family_index();
  config.direction = direction;
  config.normalize_inverse = normalize_inverse;
  config.buffer = plan->buffer->buffer;
  config.buffer_bytes = data->get_nelement() * data->get_element_size();
  {
    auto queue_lock = device->acquire_external_compute_queue_lock();
    int result;
    if (batch_tile) {
      TI_ERROR_IF(!plan->recipe_api.create,
                  "Vulkan FFT adapter has no batch recipe extension.");
      TiForgeVkfftRecipeConfig recipe{sizeof(TiForgeVkfftRecipeConfig), 0,
                                      static_cast<uint64_t>(batch_tile)};
      result = plan->recipe_api.create(&config, &recipe, &plan->handle);
    } else {
      result = plan->api.create(&config, &plan->handle);
    }
    TI_ERROR_IF(result != 0, "Vulkan FFT plan creation failed: {}",
                plan->api.last_error());
  }
  const auto handle = next_vulkan_fft_plan_handle_++;
  vulkan_fft_plans_.emplace(handle, std::move(plan));
  return handle;
}

void Program::vulkan_fft_execute(std::uint64_t handle) {
  std::lock_guard<std::recursive_mutex> lock(
      runtime_resource_submission_mutex_);
  auto entry = vulkan_fft_plans_.find(handle);
  TI_ERROR_IF(entry == vulkan_fft_plans_.end(), "Vulkan FFT plan is closed.");
  auto plan = entry->second;
  enqueue_compute_op_lambda(
      [plan](Device *, CommandList *commands) {
        auto *list = static_cast<vulkan::VulkanCommandList *>(commands);
        list->memory_barrier();
        auto command = list->begin_external_compute(plan);
        TI_ERROR_IF(plan->api.append(plan->handle, command) != 0,
                    "Vulkan FFT recording failed: {}", plan->api.last_error());
        list->memory_barrier();
      },
      {});
  // VkFFT's secondary command sequence is already recorded. The enclosing
  // runtime-ordered Graph call currently retains its ordinary host dispatch;
  // it is not a claim of one enclosing native Graph replay region.
  mark_runtime_submission_pending();
}

std::unordered_map<std::string, std::uint64_t>
Program::vulkan_fft_plan_statistics(std::uint64_t handle) {
  std::lock_guard<std::recursive_mutex> lock(
      runtime_resource_submission_mutex_);
  auto entry = vulkan_fft_plans_.find(handle);
  TI_ERROR_IF(entry == vulkan_fft_plans_.end(), "Vulkan FFT plan is closed.");
  const auto &plan = entry->second;
  TiForgeVkfftMemory memory{};
  plan->api.memory(plan->handle, &memory);
  std::unordered_map<std::string, std::uint64_t> result{
      {"persistent_allocation_bytes", memory.persistent_allocation_bytes},
      {"initialization_peak_allocation_bytes",
       memory.initialization_peak_allocation_bytes},
      {"persistent_allocation_count", memory.persistent_allocation_count},
      {"temporary_buffer_bytes", memory.temporary_buffer_bytes},
      {"adapter_abi", plan->api.abi_version},
      {"vkfft_version", plan->api.vkfft_version},
      {"glslang_major", plan->api.glslang_major},
      {"glslang_minor", plan->api.glslang_minor},
      {"glslang_patch", plan->api.glslang_patch},
      {"device_vendor_id", plan->vendor_id},
      {"device_id", plan->device_id},
      {"vulkan_driver_version", plan->driver_version},
      {"vulkan_api_version", plan->api_version}};
  if (plan->recipe_api.describe) {
    TiForgeVkfftRecipeFacts facts{};
    plan->recipe_api.describe(plan->handle, &facts);
    result.insert({{"recipe_extension_abi", plan->recipe_api.abi_version},
                   {"batch_tile", facts.batch_tile},
                   {"application_count", facts.application_count},
                   {"shader_module_count", facts.shader_module_count},
                   {"dispatch_count", facts.dispatch_count},
                   {"shader_fingerprint", facts.shader_fingerprint},
                   {"dispatch_fingerprint", facts.dispatch_fingerprint}});
  }
  return result;
}

void Program::destroy_vulkan_fft_plan(std::uint64_t handle) {
  std::lock_guard<std::recursive_mutex> lock(
      runtime_resource_submission_mutex_);
  // In-flight and cached command buffers retain their own plan/storage lease.
  vulkan_fft_plans_.erase(handle);
}

void Program::vulkan_clear_fft_plans() {
  std::lock_guard<std::recursive_mutex> lock(
      runtime_resource_submission_mutex_);
  vulkan_fft_plans_.clear();
}
}  // namespace taichi::lang

#else
namespace taichi::lang {
std::uint64_t Program::create_vulkan_fft_plan(const std::string &,
                                              Ndarray *,
                                              const std::vector<int> &,
                                              int,
                                              int,
                                              bool) {
  TI_ERROR("Vulkan FFT is unavailable in this build.");
}
std::uint64_t Program::create_vulkan_fft_recipe_plan(const std::string &,
                                                     Ndarray *,
                                                     const std::vector<int> &,
                                                     int,
                                                     int,
                                                     bool,
                                                     int) {
  TI_ERROR("Vulkan FFT is unavailable in this build.");
}
void Program::vulkan_fft_execute(std::uint64_t) {
  TI_ERROR("Vulkan FFT is unavailable in this build.");
}
std::unordered_map<std::string, std::uint64_t>
Program::vulkan_fft_plan_statistics(std::uint64_t) {
  TI_ERROR("Vulkan FFT is unavailable in this build.");
}
void Program::destroy_vulkan_fft_plan(std::uint64_t) {
}
void Program::vulkan_clear_fft_plans() {
}
}  // namespace taichi::lang
#endif
