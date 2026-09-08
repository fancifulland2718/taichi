#pragma once

#include <stddef.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

// This interface is loaded explicitly. The core runtime does not link VkFFT,
// glslang, or this adapter. Vulkan handles always belong to the caller's
// device.
#ifdef __cplusplus
extern "C" {
#endif

#define TI_FORGE_VKFFT_ABI_VERSION 1u
#define TI_FORGE_VKFFT_QUERY_SYMBOL "taichi_forge_vkfft_provider_query"

typedef void *TiForgeVkfftPlan;

typedef struct TiForgeVkfftConfig {
  uint32_t struct_size;
  uint32_t rank;
  // Compact row-major dimensions, slowest axis first. Complex f32 scalars
  // are interleaved. Only radices 2, 3, 5, 7, 11 and 13 are supported
  // initially.
  uint64_t dimensions[3];
  uint64_t batches;
  VkPhysicalDevice physical_device;
  VkDevice device;
  VkQueue queue;
  uint32_t queue_family;
  int32_t direction;           // -1 forward, +1 inverse
  uint32_t normalize_inverse;  // 0: neither direction, 1: inverse / volume
  uint32_t reserved;
  VkBuffer buffer;  // frozen in-place storage; no descriptor rebinding
  uint64_t buffer_bytes;
} TiForgeVkfftConfig;

typedef struct TiForgeVkfftMemory {
  uint64_t persistent_allocation_bytes;
  uint64_t initialization_peak_allocation_bytes;
  uint64_t persistent_allocation_count;
  uint64_t temporary_buffer_bytes;
  // Requested VkDeviceMemory allocation sizes, not physical device peak.
  // Excludes caller storage, driver pipeline/descriptor memory and host JIT.
} TiForgeVkfftMemory;

typedef struct TiForgeVkfftApi {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t vkfft_version;
  uint32_t glslang_major;
  uint32_t glslang_minor;
  uint32_t glslang_patch;
  // All functions return 0 on success. Errors are available on the calling
  // thread. Creation may submit/wait for LUT initialization: the caller must
  // hold its existing queue lock. No execution is submitted by append().
  // append() executes a retained secondary command buffer in a primary one.
  // Use the original ordered queue. The caller supplies surrounding memory
  // barriers and retains the plan and its buffer until all recorded command
  // buffers have been retired (not merely until the last append returns).
  int (*create)(const TiForgeVkfftConfig *, TiForgeVkfftPlan *);
  int (*append)(TiForgeVkfftPlan, VkCommandBuffer);
  void (*memory)(TiForgeVkfftPlan, TiForgeVkfftMemory *);
  void (*destroy)(TiForgeVkfftPlan);
  const char *(*last_error)(void);
} TiForgeVkfftApi;

typedef int (*TiForgeVkfftQueryFn)(uint32_t, size_t, TiForgeVkfftApi *);

// Optional materializer extension. The base ABI1 remains usable without this
// symbol. These are complete batched transforms, not runtime tuning knobs.
#define TI_FORGE_VKFFT_RECIPE_ABI_VERSION 1u
#define TI_FORGE_VKFFT_RECIPE_QUERY_SYMBOL "taichi_forge_vkfft_recipe_query"

typedef struct TiForgeVkfftRecipeConfig {
  uint32_t struct_size;
  uint32_t reserved;
  // Independent transforms per retained VkFFT application. Must be in
  // [1, config.batches]. A non-divisible tail owns a second application.
  uint64_t batch_tile;
} TiForgeVkfftRecipeConfig;

typedef struct TiForgeVkfftRecipeFacts {
  uint64_t batch_tile;
  uint64_t application_count;
  uint64_t shader_module_count;
  uint64_t dispatch_count;
  // Deterministic FNV-1a over ordered SPIR-V words / recorded dispatch grids.
  // Diagnostic physical evidence, not cryptographic artifact provenance.
  uint64_t shader_fingerprint;
  uint64_t dispatch_fingerprint;
} TiForgeVkfftRecipeFacts;

typedef struct TiForgeVkfftRecipeApi {
  uint32_t struct_size;
  uint32_t abi_version;
  int (*create)(const TiForgeVkfftConfig *,
                const TiForgeVkfftRecipeConfig *,
                TiForgeVkfftPlan *);
  // Cold immutable observations, valid for plans created by either table.
  void (*describe)(TiForgeVkfftPlan, TiForgeVkfftRecipeFacts *);
} TiForgeVkfftRecipeApi;

typedef int (*TiForgeVkfftRecipeQueryFn)(uint32_t,
                                         size_t,
                                         TiForgeVkfftRecipeApi *);

#if defined(TI_FORGE_VKFFT_PROVIDER_BUILD)
#if defined(_WIN32)
#define TI_FORGE_VKFFT_EXPORT __declspec(dllexport)
#else
#define TI_FORGE_VKFFT_EXPORT __attribute__((visibility("default")))
#endif
TI_FORGE_VKFFT_EXPORT int taichi_forge_vkfft_provider_query(
    uint32_t abi,
    size_t size,
    TiForgeVkfftApi *api);
TI_FORGE_VKFFT_EXPORT int taichi_forge_vkfft_recipe_query(
    uint32_t abi,
    size_t size,
    TiForgeVkfftRecipeApi *api);
#endif

#ifdef __cplusplus
}
#endif
