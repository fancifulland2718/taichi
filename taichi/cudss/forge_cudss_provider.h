#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(TI_FORGE_CUDSS_PROVIDER_BUILD)
#define TI_FORGE_CUDSS_EXPORT __declspec(dllexport)
#else
#define TI_FORGE_CUDSS_EXPORT __declspec(dllimport)
#endif
#else
#define TI_FORGE_CUDSS_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define TI_FORGE_CUDSS_PROVIDER_ABI_VERSION 1u
#define TI_FORGE_CUDSS_PROVIDER_QUERY_SYMBOL "taichi_forge_cudss_provider_query"

typedef enum TiForgeCudssResult {
  TI_FORGE_CUDSS_SUCCESS = 0,
  TI_FORGE_CUDSS_ERROR_INVALID_ARGUMENT = 1,
  TI_FORGE_CUDSS_ERROR_ABI_MISMATCH = 2,
  TI_FORGE_CUDSS_ERROR_RUNTIME_UNAVAILABLE = 3,
  TI_FORGE_CUDSS_ERROR_RUNTIME_INCOMPATIBLE = 4,
  TI_FORGE_CUDSS_ERROR_OUT_OF_MEMORY = 5,
  TI_FORGE_CUDSS_ERROR_INTERNAL = 6,
} TiForgeCudssResult;

typedef enum TiForgeCudssFeature {
  TI_FORGE_CUDSS_FEATURE_CSR = 1ull << 0,
  TI_FORGE_CUDSS_FEATURE_DENSE_VECTOR = 1ull << 1,
  TI_FORGE_CUDSS_FEATURE_STAGED_EXECUTION = 1ull << 2,
  TI_FORGE_CUDSS_FEATURE_VALUE_REBIND = 1ull << 3,
  TI_FORGE_CUDSS_FEATURE_EXPLICIT_STREAM = 1ull << 4,
} TiForgeCudssFeature;

typedef struct TiForgeCudssProviderInfo {
  uint32_t struct_size;
  uint32_t provider_abi_version;
  uint32_t cudss_header_version;
  uint32_t reserved;
  uint64_t features;
  const char *provider_name;
  const char *build_identity;
} TiForgeCudssProviderInfo;

typedef struct TiForgeCudssRuntimeInfo {
  uint32_t struct_size;
  uint32_t version_major;
  uint32_t version_minor;
  uint32_t version_patch;
  const char *library_path;
} TiForgeCudssRuntimeInfo;

typedef void *TiForgeCudssRuntime;

typedef TiForgeCudssResult (*TiForgeCudssProbeRuntimeFn)(
    const char *library_path,
    TiForgeCudssRuntimeInfo *out_info);
typedef TiForgeCudssResult (*TiForgeCudssCreateRuntimeFn)(
    const char *library_path,
    TiForgeCudssRuntime *out_runtime,
    TiForgeCudssRuntimeInfo *out_info);
typedef TiForgeCudssResult (*TiForgeCudssDestroyRuntimeFn)(
    TiForgeCudssRuntime runtime);

typedef uint32_t (*TiForgeCudssCreateFn)(TiForgeCudssRuntime runtime,
                                         void **handle);
typedef uint32_t (*TiForgeCudssDestroyFn)(TiForgeCudssRuntime runtime,
                                          void *handle);
typedef uint32_t (*TiForgeCudssSetStreamFn)(TiForgeCudssRuntime runtime,
                                            void *handle,
                                            void *stream);
typedef uint32_t (*TiForgeCudssConfigCreateFn)(TiForgeCudssRuntime runtime,
                                               void **config);
typedef uint32_t (*TiForgeCudssConfigDestroyFn)(TiForgeCudssRuntime runtime,
                                                void *config);
typedef uint32_t (*TiForgeCudssDataCreateFn)(TiForgeCudssRuntime runtime,
                                             const void *handle,
                                             void **data);
typedef uint32_t (*TiForgeCudssDataDestroyFn)(TiForgeCudssRuntime runtime,
                                              void *handle,
                                              void *data);
typedef uint32_t (*TiForgeCudssMatrixCreateCsrFn)(TiForgeCudssRuntime runtime,
                                                  void **matrix,
                                                  int64_t rows,
                                                  int64_t columns,
                                                  int64_t nonzeros,
                                                  const void *row_start,
                                                  const void *row_end,
                                                  const void *column_indices,
                                                  const void *values,
                                                  int offset_type,
                                                  int index_type,
                                                  int value_type,
                                                  int matrix_type,
                                                  int matrix_view,
                                                  int index_base);
typedef uint32_t (*TiForgeCudssMatrixCreateDnFn)(TiForgeCudssRuntime runtime,
                                                 void **matrix,
                                                 int64_t rows,
                                                 int64_t columns,
                                                 int64_t leading_dimension,
                                                 const void *values,
                                                 int value_type,
                                                 int layout);
typedef uint32_t (*TiForgeCudssMatrixDestroyFn)(TiForgeCudssRuntime runtime,
                                                void *matrix);
typedef uint32_t (*TiForgeCudssMatrixSetValuesFn)(TiForgeCudssRuntime runtime,
                                                  void *matrix,
                                                  const void *values);
typedef uint32_t (*TiForgeCudssMatrixSetCsrPointersFn)(
    TiForgeCudssRuntime runtime,
    void *matrix,
    const void *row_start,
    const void *row_end,
    const void *column_indices,
    const void *values);
typedef uint32_t (*TiForgeCudssExecuteFn)(TiForgeCudssRuntime runtime,
                                          void *handle,
                                          int phase,
                                          const void *config,
                                          void *data,
                                          const void *matrix,
                                          void *solution,
                                          const void *rhs);
typedef size_t (*TiForgeCudssGetLastErrorFn)(char *destination,
                                             size_t destination_size);

typedef struct TiForgeCudssProviderApi {
  uint32_t struct_size;
  uint32_t provider_abi_version;
  TiForgeCudssProviderInfo info;
  TiForgeCudssProbeRuntimeFn probe_runtime;
  TiForgeCudssCreateRuntimeFn create_runtime;
  TiForgeCudssDestroyRuntimeFn destroy_runtime;
  TiForgeCudssCreateFn create;
  TiForgeCudssDestroyFn destroy;
  TiForgeCudssSetStreamFn set_stream;
  TiForgeCudssConfigCreateFn config_create;
  TiForgeCudssConfigDestroyFn config_destroy;
  TiForgeCudssDataCreateFn data_create;
  TiForgeCudssDataDestroyFn data_destroy;
  TiForgeCudssMatrixCreateCsrFn matrix_create_csr;
  TiForgeCudssMatrixCreateDnFn matrix_create_dn;
  TiForgeCudssMatrixDestroyFn matrix_destroy;
  TiForgeCudssMatrixSetValuesFn matrix_set_values;
  TiForgeCudssMatrixSetCsrPointersFn matrix_set_csr_pointers;
  TiForgeCudssExecuteFn execute;
  TiForgeCudssGetLastErrorFn get_last_error;
} TiForgeCudssProviderApi;

typedef TiForgeCudssResult (*TiForgeCudssProviderQueryFn)(
    uint32_t requested_abi_version,
    size_t api_size,
    TiForgeCudssProviderApi *out_api);

TI_FORGE_CUDSS_EXPORT TiForgeCudssResult
taichi_forge_cudss_provider_query(uint32_t requested_abi_version,
                                  size_t api_size,
                                  TiForgeCudssProviderApi *out_api);

// Optional preparation-only extension. Keep the execution ABI1 table unchanged
// so older adapters remain usable by the ordinary explicit plan path.
#define TI_FORGE_CUDSS_CONFIGURATION_ABI_VERSION 1u
#define TI_FORGE_CUDSS_CONFIGURATION_QUERY_SYMBOL \
  "taichi_forge_cudss_configuration_query"

// Optional Graph allocator extension, separate from both published tables.
#define TI_FORGE_CUDSS_ALLOCATOR_QUERY_SYMBOL \
  "taichi_forge_cudss_allocator_query"
typedef struct TiForgeCudssAllocatorApi {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t (*set_allocator)(TiForgeCudssRuntime runtime,
                            void *handle,
                            void *owner,
                            int (*allocate)(void *, void **, size_t, void *),
                            int (*deallocate)(void *, void *, size_t, void *));
} TiForgeCudssAllocatorApi;
typedef TiForgeCudssResult (*TiForgeCudssAllocatorQueryFn)(
    uint32_t,
    size_t,
    TiForgeCudssAllocatorApi *);

TI_FORGE_CUDSS_EXPORT TiForgeCudssResult
taichi_forge_cudss_allocator_query(uint32_t requested_abi_version,
                                   size_t api_size,
                                   TiForgeCudssAllocatorApi *out_api);

typedef enum TiForgeCudssReordering {
  TI_FORGE_CUDSS_REORDER_DEFAULT = 0,
  TI_FORGE_CUDSS_REORDER_AMD = 1,
  TI_FORGE_CUDSS_REORDER_ND = 2,
  TI_FORGE_CUDSS_REORDER_NATURAL = 3,
} TiForgeCudssReordering;

typedef struct TiForgeCudssConfigurationApi {
  uint32_t struct_size;
  uint32_t abi_version;
  // reordering uses the Forge enum above; solve is 0 (default) or 1 (general).
  // Returns the vendor status. Both requested settings must round-trip exactly;
  // default denotes vendor policy, not an observed resolved internal algorithm.
  uint32_t (*configure)(TiForgeCudssRuntime runtime,
                        void *config,
                        int reordering,
                        int solve);
  // Call only after analysis, for its single-RHS problem. The 16-entry vendor
  // estimate is not a measurement and must not enter observed-memory metrics.
  uint32_t (*analysis_memory_estimates)(TiForgeCudssRuntime runtime,
                                       void *handle,
                                       void *data,
                                       int64_t *estimates,
                                       size_t capacity_bytes,
                                       size_t *written_bytes);
} TiForgeCudssConfigurationApi;

typedef TiForgeCudssResult (*TiForgeCudssConfigurationQueryFn)(
    uint32_t requested_abi_version,
    size_t api_size,
    TiForgeCudssConfigurationApi *out_api);

TI_FORGE_CUDSS_EXPORT TiForgeCudssResult
taichi_forge_cudss_configuration_query(uint32_t requested_abi_version,
                                       size_t api_size,
                                       TiForgeCudssConfigurationApi *out_api);

// Optional cold diagnostics. Do not extend the published execution/config
// tables.
#define TI_FORGE_CUDSS_FACTOR_STATISTICS_QUERY_SYMBOL \
  "taichi_forge_cudss_factor_statistics_query"

typedef struct TiForgeCudssFactorStatistic {
  int64_t value;
  uint32_t vendor_status;
  uint32_t written_bytes;
} TiForgeCudssFactorStatistic;

typedef struct TiForgeCudssFactorStatistics {
  TiForgeCudssFactorStatistic lu_nonzeros;
  TiForgeCudssFactorStatistic superpanels;
  TiForgeCudssFactorStatistic factor_flops;
} TiForgeCudssFactorStatistics;

typedef struct TiForgeCudssFactorStatisticsApi {
  uint32_t struct_size;
  uint32_t abi_version;
  // Synchronous vendor queries, only after private preparation completes.
  // The sparse matrix must have i32 indices. Each field has its own status;
  // values are unavailable (-1) on vendor failure or a returned-size mismatch.
  TiForgeCudssResult (*read_i32)(TiForgeCudssRuntime runtime,
                                 void *handle,
                                 void *data,
                                 TiForgeCudssFactorStatistics *out_statistics);
} TiForgeCudssFactorStatisticsApi;

typedef TiForgeCudssResult (*TiForgeCudssFactorStatisticsQueryFn)(
    uint32_t,
    size_t,
    TiForgeCudssFactorStatisticsApi *);

TI_FORGE_CUDSS_EXPORT TiForgeCudssResult
taichi_forge_cudss_factor_statistics_query(
    uint32_t requested_abi_version,
    size_t api_size,
    TiForgeCudssFactorStatisticsApi *out_api);

#ifdef __cplusplus
}
#endif
