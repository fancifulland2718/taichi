#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(TI_FORGE_OPTIX_PROVIDER_BUILD)
#define TI_FORGE_OPTIX_EXPORT __declspec(dllexport)
#else
#define TI_FORGE_OPTIX_EXPORT __declspec(dllimport)
#endif
#else
#define TI_FORGE_OPTIX_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define TI_FORGE_OPTIX_PROVIDER_ABI_VERSION 1u
#define TI_FORGE_OPTIX_PROVIDER_QUERY_SYMBOL \
  "taichi_forge_optix_provider_query"

typedef enum TiForgeOptixResult {
  TI_FORGE_OPTIX_SUCCESS = 0,
  TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT = 1,
  TI_FORGE_OPTIX_ERROR_ABI_MISMATCH = 2,
  TI_FORGE_OPTIX_ERROR_CUDA_CONTEXT = 3,
  TI_FORGE_OPTIX_ERROR_OPTIX_UNAVAILABLE = 4,
  TI_FORGE_OPTIX_ERROR_OPTIX_CALL = 5,
  TI_FORGE_OPTIX_ERROR_CUDA_CALL = 6,
  TI_FORGE_OPTIX_ERROR_OUT_OF_MEMORY = 7,
  TI_FORGE_OPTIX_ERROR_LIFETIME = 8,
  TI_FORGE_OPTIX_ERROR_INTERNAL = 9,
} TiForgeOptixResult;

typedef enum TiForgeOptixFeature {
  TI_FORGE_OPTIX_FEATURE_TRIANGLE_GAS = 1ull << 0,
  TI_FORGE_OPTIX_FEATURE_SINGLE_INSTANCE_IAS = 1ull << 1,
  TI_FORGE_OPTIX_FEATURE_GAS_UPDATE = 1ull << 2,
  TI_FORGE_OPTIX_FEATURE_BATCH_CLOSEST_HIT = 1ull << 3,
  TI_FORGE_OPTIX_FEATURE_RUNTIME_ORDERED_STREAM = 1ull << 4,
  TI_FORGE_OPTIX_FEATURE_EXACT_DEVICE_MEMORY = 1ull << 5,
  TI_FORGE_OPTIX_FEATURE_TYPED_HITS = 1ull << 6,
  TI_FORGE_OPTIX_FEATURE_WORD_ALIGNED_QUERY_STORAGE = 1ull << 7,
} TiForgeOptixFeature;

typedef struct TiForgeOptixProviderInfo {
  uint32_t struct_size;
  uint32_t provider_abi_version;
  uint32_t optix_abi_version;
  uint32_t optix_version;
  uint64_t features;
  const char *provider_name;
  const char *build_identity;
} TiForgeOptixProviderInfo;

typedef struct TiForgeOptixContextDesc {
  uint32_t struct_size;
  uint32_t device_ordinal;
  uint64_t cuda_context;
  uint32_t validation_mode;
  uint32_t reserved;
  const char *runtime_library_path;
} TiForgeOptixContextDesc;

typedef struct TiForgeOptixTriangleSceneDesc {
  uint32_t struct_size;
  uint32_t vertex_count;
  uint32_t triangle_count;
  uint32_t allow_update;
  uint64_t vertices;
  uint64_t indices;
  uint64_t cuda_stream;
} TiForgeOptixTriangleSceneDesc;

typedef struct TiForgeOptixTraceDesc {
  uint32_t struct_size;
  uint32_t ray_count;
  uint64_t rays;
  uint64_t hits;
  uint64_t cuda_stream;
} TiForgeOptixTraceDesc;

// hits: float4(t, u, v, 0); hit_indices: uint4(primitive, instance, custom, hit).
// A miss writes (-1, 0, 0, 0) and (UINT32_MAX, UINT32_MAX, UINT32_MAX, 0).
typedef struct TiForgeOptixTypedTraceDesc {
  uint32_t struct_size;
  uint32_t ray_count;
  uint64_t rays;
  uint64_t hits;
  uint64_t hit_indices;
  uint64_t cuda_stream;
} TiForgeOptixTypedTraceDesc;

typedef struct TiForgeOptixSceneMemory {
  uint32_t struct_size;
  uint32_t reserved;
  uint64_t gas_bytes;
  uint64_t ias_bytes;
  uint64_t build_update_scratch_bytes;
  uint64_t instance_bytes;
  uint64_t launch_params_bytes;
  uint64_t shared_pipeline_sbt_bytes;
} TiForgeOptixSceneMemory;

typedef void *TiForgeOptixContext;
typedef void *TiForgeOptixTriangleScene;

typedef TiForgeOptixResult (*TiForgeOptixProbeRuntimeFn)(
    const char *library_path);
typedef TiForgeOptixResult (*TiForgeOptixCreateContextFn)(
    const TiForgeOptixContextDesc *desc,
    TiForgeOptixContext *out_context);
typedef TiForgeOptixResult (*TiForgeOptixDestroyContextFn)(
    TiForgeOptixContext context);
typedef TiForgeOptixResult (*TiForgeOptixCreateTriangleSceneFn)(
    TiForgeOptixContext context,
    const TiForgeOptixTriangleSceneDesc *desc,
    TiForgeOptixTriangleScene *out_scene);
typedef TiForgeOptixResult (*TiForgeOptixUpdateTriangleSceneFn)(
    TiForgeOptixTriangleScene scene,
    const TiForgeOptixTriangleSceneDesc *desc);
typedef TiForgeOptixResult (*TiForgeOptixTraceFn)(
    TiForgeOptixTriangleScene scene,
    const TiForgeOptixTraceDesc *desc);
typedef TiForgeOptixResult (*TiForgeOptixGetSceneMemoryFn)(
    TiForgeOptixTriangleScene scene,
    TiForgeOptixSceneMemory *out_memory);
typedef TiForgeOptixResult (*TiForgeOptixDestroyTriangleSceneFn)(
    TiForgeOptixTriangleScene scene);
typedef size_t (*TiForgeOptixGetLastErrorFn)(char *destination,
                                             size_t destination_size);
typedef TiForgeOptixResult (*TiForgeOptixPrepareTypedFn)(
    TiForgeOptixContext context);
typedef TiForgeOptixResult (*TiForgeOptixTraceTypedFn)(
    TiForgeOptixTriangleScene scene,
    const TiForgeOptixTypedTraceDesc *desc);

typedef struct TiForgeOptixProviderApi {
  uint32_t struct_size;
  uint32_t provider_abi_version;
  TiForgeOptixProviderInfo info;
  TiForgeOptixProbeRuntimeFn probe_runtime;
  TiForgeOptixCreateContextFn create_context;
  TiForgeOptixDestroyContextFn destroy_context;
  TiForgeOptixCreateTriangleSceneFn create_triangle_scene;
  TiForgeOptixUpdateTriangleSceneFn update_triangle_scene;
  TiForgeOptixTraceFn trace;
  TiForgeOptixGetSceneMemoryFn get_scene_memory;
  TiForgeOptixDestroyTriangleSceneFn destroy_triangle_scene;
  TiForgeOptixGetLastErrorFn get_last_error;
  // Optional ABI-1 suffix. Callers negotiate via struct_size + feature bits.
  TiForgeOptixPrepareTypedFn prepare_typed;
  TiForgeOptixTraceTypedFn trace_typed;
} TiForgeOptixProviderApi;

typedef TiForgeOptixResult (*TiForgeOptixProviderQueryFn)(
    uint32_t requested_abi_version,
    size_t api_size,
    TiForgeOptixProviderApi *out_api);

TI_FORGE_OPTIX_EXPORT TiForgeOptixResult
taichi_forge_optix_provider_query(uint32_t requested_abi_version,
                                  size_t api_size,
                                  TiForgeOptixProviderApi *out_api);

#ifdef __cplusplus
}
#endif
