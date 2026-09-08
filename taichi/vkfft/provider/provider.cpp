#include "taichi/vkfft/forge_vkfft_provider.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>
#include <vector>

#include <glslang/build_info.h>

namespace {
// VkFFT allocates LUT and temporary storage itself. Account those actual
// requests at plan creation/destruction, rather than polling device memory
// during replay or claiming the input buffer is the whole plan footprint.
struct Allocations {
  std::unordered_map<VkDeviceMemory, VkDeviceSize> sizes;
  uint64_t live{0};
  uint64_t peak{0};
};
thread_local Allocations *active_allocations = nullptr;
thread_local TiForgeVkfftRecipeFacts *active_facts = nullptr;

void hash_word(uint64_t &hash, uint64_t word) {
  for (int byte = 0; byte < 8; ++byte) {
    hash = (hash ^ (word & 0xff)) * 1099511628211ull;
    word >>= 8;
  }
}

struct FactsScope {
  TiForgeVkfftRecipeFacts *previous;
  explicit FactsScope(TiForgeVkfftRecipeFacts &facts) : previous(active_facts) {
    active_facts = &facts;
  }
  ~FactsScope() {
    active_facts = previous;
  }
};

VkResult create_shader_module(VkDevice device,
                              const VkShaderModuleCreateInfo *info,
                              const VkAllocationCallbacks *callbacks,
                              VkShaderModule *shader) {
  const auto result = vkCreateShaderModule(device, info, callbacks, shader);
  if (result == VK_SUCCESS) {
    ++active_facts->shader_module_count;
    hash_word(active_facts->shader_fingerprint, info->codeSize);
    for (size_t i = 0; i < info->codeSize / sizeof(uint32_t); ++i) {
      hash_word(active_facts->shader_fingerprint, info->pCode[i]);
    }
  }
  return result;
}

void record_dispatch(VkCommandBuffer command,
                     uint32_t x,
                     uint32_t y,
                     uint32_t z) {
  ++active_facts->dispatch_count;
  for (const auto value : {x, y, z}) {
    hash_word(active_facts->dispatch_fingerprint, value);
  }
  vkCmdDispatch(command, x, y, z);
}

struct AllocationScope {
  Allocations *previous;
  explicit AllocationScope(Allocations &allocations)
      : previous(active_allocations) {
    active_allocations = &allocations;
  }
  ~AllocationScope() {
    active_allocations = previous;
  }
};

VkResult allocate_memory(VkDevice device,
                         const VkMemoryAllocateInfo *info,
                         const VkAllocationCallbacks *callbacks,
                         VkDeviceMemory *memory) {
  auto result = vkAllocateMemory(device, info, callbacks, memory);
  if (result == VK_SUCCESS) {
    try {
      active_allocations->sizes.emplace(*memory, info->allocationSize);
    } catch (const std::bad_alloc &) {
      vkFreeMemory(device, *memory, callbacks);
      *memory = VK_NULL_HANDLE;
      return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    active_allocations->live += info->allocationSize;
    active_allocations->peak =
        std::max(active_allocations->peak, active_allocations->live);
  }
  return result;
}

void free_memory(VkDevice device,
                 VkDeviceMemory memory,
                 const VkAllocationCallbacks *callbacks) {
  auto entry = active_allocations->sizes.find(memory);
  if (entry != active_allocations->sizes.end()) {
    active_allocations->live -= entry->second;
    active_allocations->sizes.erase(entry);
  }
  vkFreeMemory(device, memory, callbacks);
}
}  // namespace

// Local to this translation unit, covering upstream allocation calls only.
#define vkAllocateMemory allocate_memory
#define vkFreeMemory free_memory
#define vkCreateShaderModule create_shader_module
#define vkCmdDispatch record_dispatch
#define VKFFT_BACKEND 0
#include <vkFFT.h>
#undef vkAllocateMemory
#undef vkFreeMemory
#undef vkCreateShaderModule
#undef vkCmdDispatch

namespace {
thread_local char error_message[256]{};
std::mutex compiler_mutex;

int fail(const char *operation, int code) {
  std::snprintf(error_message, sizeof(error_message), "%s (%d)", operation,
                code);
  return code == 0 ? -1 : code;
}

struct Plan {
  TiForgeVkfftConfig config{};
  VkCommandPool pool{VK_NULL_HANDLE};
  VkFence fence{VK_NULL_HANDLE};
  VkCommandBuffer executable{VK_NULL_HANDLE};
  struct Application {
    VkFFTApplication fft{};
    bool initialized{false};
  };
  std::vector<std::unique_ptr<Application>> applications;
  Allocations allocations;
  TiForgeVkfftRecipeFacts facts{
      0, 0, 0, 0, 14695981039346656037ull, 14695981039346656037ull};

  ~Plan() {
    AllocationScope scope(allocations);
    for (auto &application : applications) {
      if (application->initialized) {
        deleteVkFFT(&application->fft);
      }
    }
    if (fence != VK_NULL_HANDLE) {
      vkDestroyFence(config.device, fence, nullptr);
    }
    if (pool != VK_NULL_HANDLE) {
      vkDestroyCommandPool(config.device, pool, nullptr);
    }
  }
};

bool supported_size(uint64_t size) {
  if (size == 0) {
    return false;
  }
  for (uint64_t radix : {2, 3, 5, 7, 11, 13}) {
    while (size % radix == 0) {
      size /= radix;
    }
  }
  return size == 1;
}

int record_plan_commands(Plan &plan, VkCommandBuffer command) {
  const auto &config = plan.config;
  const uint64_t tile = plan.facts.batch_tile;
  VkFFTLaunchParams launch{};
  launch.commandBuffer = &command;
  for (uint64_t first = 0; first < config.batches; first += tile) {
    const bool is_tail = config.batches - first < tile;
    auto &application = *plan.applications[is_tail ? 1 : 0];
    launch.bufferOffset = first * (config.buffer_bytes / config.batches);
    const auto result =
        VkFFTAppend(&application.fft, config.direction, &launch);
    if (result != VKFFT_SUCCESS) {
      return fail("VkFFTAppend during plan recording", result);
    }
    // Only full tiles reuse writable scratch. VkFFT already emits its own
    // per-dispatch write-to-read barriers; disjoint slices need no extra one.
    if (application.fft.configuration.allocateTempBuffer &&
        (config.batches - first) / tile > 1) {
      VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      barrier.srcAccessMask =
          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      barrier.dstAccessMask =
          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier,
                           0, nullptr, 0, nullptr);
    }
  }
  return 0;
}

int create_recipe_plan(const TiForgeVkfftConfig *config,
                       const TiForgeVkfftRecipeConfig *recipe,
                       TiForgeVkfftPlan *out) {
  if (!out) {
    return fail("missing plan output", -1);
  }
  *out = nullptr;
  if (!config || config->struct_size != sizeof(*config) || config->rank < 1 ||
      config->rank > 3 || config->batches == 0 || !config->physical_device ||
      !config->device || !config->queue || !config->buffer ||
      config->reserved || (config->direction != -1 && config->direction != 1) ||
      config->normalize_inverse > 1) {
    return fail("invalid compact complex-f32 plan configuration", -1);
  }
  uint64_t bytes = 2 * sizeof(float);
  for (uint32_t axis = 0; axis <= config->rank; ++axis) {
    const auto extent =
        axis == config->rank ? config->batches : config->dimensions[axis];
    if (axis != config->rank && !supported_size(extent)) {
      return fail("unsupported dimension: prime factors must be <= 13", -1);
    }
    if (extent > std::numeric_limits<uint64_t>::max() / bytes) {
      return fail("FFT storage size overflow", -1);
    }
    bytes *= extent;
  }
  if (bytes != config->buffer_bytes) {
    return fail("FFT buffer size must equal compact batched storage", -1);
  }
  if (recipe &&
      (recipe->struct_size != sizeof(*recipe) || recipe->reserved ||
       recipe->batch_tile == 0 || recipe->batch_tile > config->batches)) {
    return fail("invalid FFT batch partition", -1);
  }
  const uint64_t tile = recipe ? recipe->batch_tile : config->batches;
  const uint64_t tail = config->batches % tile;
  try {
    auto plan = std::make_unique<Plan>();
    plan->config = *config;
    plan->facts.batch_tile = tile;
    VkCommandPoolCreateInfo pool_info{
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.queueFamilyIndex = config->queue_family;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    auto result =
        vkCreateCommandPool(config->device, &pool_info, nullptr, &plan->pool);
    if (result != VK_SUCCESS) {
      return fail("vkCreateCommandPool", result);
    }
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    result = vkCreateFence(config->device, &fence_info, nullptr, &plan->fence);
    if (result != VK_SUCCESS) {
      return fail("vkCreateFence", result);
    }
    VkFFTConfiguration parameters{};
    parameters.FFTdim = config->rank;
    for (uint32_t axis = 0; axis < config->rank; ++axis) {
      parameters.size[axis] = config->dimensions[config->rank - axis - 1];
    }
    parameters.numberBatches = tile;
    parameters.physicalDevice = &plan->config.physical_device;
    parameters.device = &plan->config.device;
    parameters.queue = &plan->config.queue;
    parameters.commandPool = &plan->pool;
    parameters.fence = &plan->fence;
    parameters.buffer = &plan->config.buffer;
    parameters.bufferSize = &plan->config.buffer_bytes;
    parameters.normalize = config->normalize_inverse;
    parameters.makeForwardPlanOnly = config->direction == -1;
    parameters.makeInversePlanOnly = config->direction == 1;
    parameters.specifyOffsetsAtLaunch = tile != config->batches;
    // No shared global compiler lifetime races between cold plan builds.
    std::lock_guard<std::mutex> compiler_lock(compiler_mutex);
    AllocationScope scope(plan->allocations);
    FactsScope facts_scope(plan->facts);
    for (const auto count : {tile, tail}) {
      if (count == 0) {
        continue;
      }
      parameters.numberBatches = count;
      // Publish ownership before initialization; a later host allocation
      // failure must not leak a successfully initialized application.
      plan->applications.push_back(std::make_unique<Plan::Application>());
      auto &application = *plan->applications.back();
      const auto fft_result = initializeVkFFT(&application.fft, parameters);
      if (fft_result != VKFFT_SUCCESS) {
        // initializeVkFFT owns cleanup of partially initialized applications.
        return fail("initializeVkFFT", fft_result);
      }
      application.initialized = true;
    }
    plan->facts.application_count = plan->applications.size();
    VkCommandBufferAllocateInfo allocation{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocation.commandPool = plan->pool;
    allocation.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
    allocation.commandBufferCount = 1;
    result = vkAllocateCommandBuffers(config->device, &allocation,
                                      &plan->executable);
    if (result != VK_SUCCESS) {
      return fail("vkAllocateCommandBuffers", result);
    }
    VkCommandBufferInheritanceInfo inheritance{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO};
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    begin.pInheritanceInfo = &inheritance;
    result = vkBeginCommandBuffer(plan->executable, &begin);
    if (result != VK_SUCCESS) {
      return fail("vkBeginCommandBuffer", result);
    }
    if (const auto record_result = record_plan_commands(*plan, plan->executable)) {
      return record_result;
    }
    result = vkEndCommandBuffer(plan->executable);
    if (result != VK_SUCCESS) {
      return fail("vkEndCommandBuffer", result);
    }
    *out = plan.release();
    error_message[0] = '\0';
    return 0;
  } catch (const std::bad_alloc &) {
    return fail("host allocation failed during plan creation", -1);
  } catch (...) {
    return fail("unexpected failure during plan creation", -1);
  }
}

int create_plan(const TiForgeVkfftConfig *config, TiForgeVkfftPlan *out) {
  return create_recipe_plan(config, nullptr, out);
}

int append_plan(TiForgeVkfftPlan handle, VkCommandBuffer command) {
  auto *plan = static_cast<Plan *>(handle);
  // The complete vendor dispatch sequence was recorded once at plan creation.
  // Reuse it inside the caller's ordered primary command list, without a
  // vendor host dispatch loop, descriptor rebinding, JIT or extra submission.
  vkCmdExecuteCommands(command, 1, &plan->executable);
  return 0;
}

void plan_memory(TiForgeVkfftPlan handle, TiForgeVkfftMemory *out) {
  const auto *plan = static_cast<Plan *>(handle);
  uint64_t temporary_bytes = 0;
  for (const auto &application : plan->applications) {
    const auto &config = application->fft.configuration;
    temporary_bytes += config.allocateTempBuffer ? config.tempBufferSize[0] : 0;
  }
  *out = {plan->allocations.live, plan->allocations.peak,
          static_cast<uint64_t>(plan->allocations.sizes.size()),
          temporary_bytes};
}

void describe_plan(TiForgeVkfftPlan handle, TiForgeVkfftRecipeFacts *out) {
  *out = static_cast<Plan *>(handle)->facts;
}

void destroy_plan(TiForgeVkfftPlan handle) {
  delete static_cast<Plan *>(handle);
}

const char *last_error() {
  return error_message;
}
}  // namespace

int taichi_forge_vkfft_provider_query(uint32_t abi,
                                      size_t size,
                                      TiForgeVkfftApi *api) {
  if (abi != TI_FORGE_VKFFT_ABI_VERSION || size != sizeof(*api) || !api) {
    return fail("VkFFT adapter ABI mismatch", -1);
  }
  *api = {sizeof(*api),
          TI_FORGE_VKFFT_ABI_VERSION,
          static_cast<uint32_t>(VkFFTGetVersion()),
          GLSLANG_VERSION_MAJOR,
          GLSLANG_VERSION_MINOR,
          GLSLANG_VERSION_PATCH,
          create_plan,
          append_plan,
          plan_memory,
          destroy_plan,
          last_error};
  return 0;
}

int taichi_forge_vkfft_recipe_query(uint32_t abi,
                                    size_t size,
                                    TiForgeVkfftRecipeApi *api) {
  if (abi != TI_FORGE_VKFFT_RECIPE_ABI_VERSION || size != sizeof(*api) ||
      !api) {
    return fail("VkFFT recipe extension ABI mismatch", -1);
  }
  *api = {sizeof(*api), TI_FORGE_VKFFT_RECIPE_ABI_VERSION, create_recipe_plan,
          describe_plan};
  return 0;
}

int taichi_forge_vkfft_record_inline(TiForgeVkfftPlan handle,
                                     VkCommandBuffer command) {
  // The observation hooks are cold-only. Do not mutate the frozen plan facts
  // when another complete Graph records the same application sequence.
  TiForgeVkfftRecipeFacts observations{};
  FactsScope scope(observations);
  return record_plan_commands(*static_cast<Plan *>(handle), command);
}
