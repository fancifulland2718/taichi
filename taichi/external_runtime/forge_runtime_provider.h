#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(TI_FORGE_RUNTIME_PROVIDER_BUILD)
#define TI_FORGE_RUNTIME_PROVIDER_EXPORT __declspec(dllexport)
#else
#define TI_FORGE_RUNTIME_PROVIDER_EXPORT __declspec(dllimport)
#endif
#else
#define TI_FORGE_RUNTIME_PROVIDER_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define TI_FORGE_RUNTIME_PROVIDER_ABI_VERSION 2u
#define TI_FORGE_RUNTIME_PROVIDER_EXECUTION_ABI_VERSION 1u

#define TI_FORGE_CUSPARSELT_PROVIDER_QUERY_SYMBOL \
  "taichi_forge_cusparselt_provider_query"
#define TI_FORGE_CUTENSOR_PROVIDER_QUERY_SYMBOL \
  "taichi_forge_cutensor_provider_query"
#define TI_FORGE_AMGX_PROVIDER_QUERY_SYMBOL "taichi_forge_amgx_provider_query"

typedef enum TiForgeRuntimeProviderResult {
  TI_FORGE_RUNTIME_PROVIDER_SUCCESS = 0,
  TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT = 1,
  TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH = 2,
  TI_FORGE_RUNTIME_PROVIDER_ERROR_RUNTIME_UNAVAILABLE = 3,
  TI_FORGE_RUNTIME_PROVIDER_ERROR_RUNTIME_INCOMPATIBLE = 4,
  TI_FORGE_RUNTIME_PROVIDER_ERROR_OUT_OF_MEMORY = 5,
  TI_FORGE_RUNTIME_PROVIDER_ERROR_INTERNAL = 6,
  TI_FORGE_RUNTIME_PROVIDER_ERROR_VENDOR_CALL = 7,
  TI_FORGE_RUNTIME_PROVIDER_ERROR_LIFETIME = 8,
  TI_FORGE_RUNTIME_PROVIDER_ERROR_EXECUTION_UNSUPPORTED = 9,
} TiForgeRuntimeProviderResult;

typedef enum TiForgeRuntimeProviderFeature {
  TI_FORGE_RUNTIME_PROVIDER_FEATURE_VERSION_QUERY = 1ull << 0,
  TI_FORGE_RUNTIME_PROVIDER_FEATURE_REQUIRED_SYMBOL_AUDIT = 1ull << 1,
  TI_FORGE_RUNTIME_PROVIDER_FEATURE_TRANSIENT_PROBE = 1ull << 2,
  TI_FORGE_RUNTIME_PROVIDER_FEATURE_EXECUTION_API = 1ull << 3,
} TiForgeRuntimeProviderFeature;

typedef struct TiForgeRuntimeProviderInfo {
  uint32_t struct_size;
  uint32_t provider_abi_version;
  uint64_t features;
  uint32_t required_symbol_count;
  uint32_t reserved;
  const char *provider_id;
  const char *provider_name;
  const char *supported_version_family;
  const char *build_identity;
} TiForgeRuntimeProviderInfo;

typedef struct TiForgeRuntimeInfo {
  uint32_t struct_size;
  uint32_t version_major;
  uint32_t version_minor;
  uint32_t version_patch;
  uint32_t cuda_runtime_version;
  uint32_t reserved;
  const char *library_path;
  const char *build_version;
} TiForgeRuntimeInfo;

typedef void *TiForgeRuntimeProviderRuntime;

typedef TiForgeRuntimeProviderResult (*TiForgeRuntimeProviderProbeRuntimeFn)(
    const char *library_path,
    TiForgeRuntimeInfo *out_info);
typedef TiForgeRuntimeProviderResult (*TiForgeRuntimeProviderCreateRuntimeFn)(
    const char *library_path,
    TiForgeRuntimeProviderRuntime *out_runtime,
    TiForgeRuntimeInfo *out_info);
typedef TiForgeRuntimeProviderResult (*TiForgeRuntimeProviderDestroyRuntimeFn)(
    TiForgeRuntimeProviderRuntime runtime);
typedef TiForgeRuntimeProviderResult (
    *TiForgeRuntimeProviderQueryExecutionApiFn)(
    TiForgeRuntimeProviderRuntime runtime,
    uint32_t requested_execution_abi_version,
    size_t api_size,
    void *out_api);
typedef size_t (*TiForgeRuntimeProviderGetLastErrorFn)(char *destination,
                                                       size_t destination_size);

typedef struct TiForgeRuntimeProviderApi {
  uint32_t struct_size;
  uint32_t provider_abi_version;
  TiForgeRuntimeProviderInfo info;
  TiForgeRuntimeProviderProbeRuntimeFn probe_runtime;
  TiForgeRuntimeProviderCreateRuntimeFn create_runtime;
  TiForgeRuntimeProviderDestroyRuntimeFn destroy_runtime;
  TiForgeRuntimeProviderQueryExecutionApiFn query_execution_api;
  TiForgeRuntimeProviderGetLastErrorFn get_last_error;
} TiForgeRuntimeProviderApi;

typedef void *TiForgeCutensorContractionPlan;

typedef enum TiForgeCutensorComputeMode {
  TI_FORGE_CUTENSOR_COMPUTE_F32 = 0,
  TI_FORGE_CUTENSOR_COMPUTE_TF32 = 1,
} TiForgeCutensorComputeMode;

typedef enum TiForgeCutensorWorkspacePreference {
  TI_FORGE_CUTENSOR_WORKSPACE_MIN = 1,
  TI_FORGE_CUTENSOR_WORKSPACE_DEFAULT = 2,
  TI_FORGE_CUTENSOR_WORKSPACE_MAX = 3,
} TiForgeCutensorWorkspacePreference;

typedef struct TiForgeCutensorTensorDesc {
  uint32_t struct_size;
  uint32_t rank;
  const int64_t *extents;
  const int64_t *strides;
  const int32_t *modes;
} TiForgeCutensorTensorDesc;

typedef struct TiForgeCutensorContractionPlanDesc {
  uint32_t struct_size;
  uint32_t compute_mode;
  uint32_t alignment_bytes;
  uint32_t workspace_preference;
  uint64_t workspace_limit_bytes;
  TiForgeCutensorTensorDesc a;
  TiForgeCutensorTensorDesc b;
  TiForgeCutensorTensorDesc c;
  TiForgeCutensorTensorDesc d;
} TiForgeCutensorContractionPlanDesc;

typedef struct TiForgeCutensorContractionPlanInfo {
  uint32_t struct_size;
  uint32_t reserved;
  uint64_t workspace_estimate_bytes;
  uint64_t workspace_required_bytes;
} TiForgeCutensorContractionPlanInfo;

typedef struct TiForgeCutensorContractionExecDesc {
  uint32_t struct_size;
  uint32_t reserved;
  float alpha;
  float beta;
  uint64_t a;
  uint64_t b;
  uint64_t c;
  uint64_t d;
  uint64_t workspace;
  uint64_t workspace_bytes;
  uint64_t cuda_stream;
} TiForgeCutensorContractionExecDesc;

typedef TiForgeRuntimeProviderResult (*TiForgeCutensorCreatePlanFn)(
    TiForgeRuntimeProviderRuntime runtime,
    const TiForgeCutensorContractionPlanDesc *desc,
    TiForgeCutensorContractionPlan *out_plan,
    TiForgeCutensorContractionPlanInfo *out_info);
typedef TiForgeRuntimeProviderResult (*TiForgeCutensorExecuteFn)(
    TiForgeCutensorContractionPlan plan,
    const TiForgeCutensorContractionExecDesc *desc);
typedef TiForgeRuntimeProviderResult (*TiForgeCutensorDestroyPlanFn)(
    TiForgeCutensorContractionPlan plan);

typedef struct TiForgeCutensorExecutionApi {
  uint32_t struct_size;
  uint32_t execution_abi_version;
  TiForgeCutensorCreatePlanFn create_contraction_plan;
  TiForgeCutensorExecuteFn execute_contraction;
  TiForgeCutensorDestroyPlanFn destroy_contraction_plan;
} TiForgeCutensorExecutionApi;

typedef void *TiForgeCusparseLtMatmulPlan;

typedef struct TiForgeCusparseLtMatmulPlanDesc {
  uint32_t struct_size;
  uint32_t alignment_bytes;
  int64_t m;
  int64_t n;
  int64_t k;
} TiForgeCusparseLtMatmulPlanDesc;

typedef struct TiForgeCusparseLtMatmulPlanInfo {
  uint32_t struct_size;
  uint32_t reserved;
  uint64_t compressed_bytes;
  uint64_t compression_buffer_bytes;
  uint64_t workspace_bytes;
} TiForgeCusparseLtMatmulPlanInfo;

typedef struct TiForgeCusparseLtCompressDesc {
  uint32_t struct_size;
  uint32_t reserved;
  uint64_t dense_a;
  uint64_t compressed_a;
  uint64_t compression_buffer;
  uint64_t compression_buffer_bytes;
  uint64_t cuda_stream;
} TiForgeCusparseLtCompressDesc;

typedef struct TiForgeCusparseLtMatmulExecDesc {
  uint32_t struct_size;
  uint32_t reserved;
  float alpha;
  float beta;
  uint64_t compressed_a;
  uint64_t b;
  uint64_t c;
  uint64_t d;
  uint64_t workspace;
  uint64_t workspace_bytes;
  uint64_t cuda_stream;
} TiForgeCusparseLtMatmulExecDesc;

typedef TiForgeRuntimeProviderResult (*TiForgeCusparseLtCreatePlanFn)(
    TiForgeRuntimeProviderRuntime runtime,
    const TiForgeCusparseLtMatmulPlanDesc *desc,
    TiForgeCusparseLtMatmulPlan *out_plan,
    TiForgeCusparseLtMatmulPlanInfo *out_info);
typedef TiForgeRuntimeProviderResult (*TiForgeCusparseLtCompressFn)(
    TiForgeCusparseLtMatmulPlan plan,
    const TiForgeCusparseLtCompressDesc *desc);
typedef TiForgeRuntimeProviderResult (*TiForgeCusparseLtExecuteFn)(
    TiForgeCusparseLtMatmulPlan plan,
    const TiForgeCusparseLtMatmulExecDesc *desc);
typedef TiForgeRuntimeProviderResult (*TiForgeCusparseLtDestroyPlanFn)(
    TiForgeCusparseLtMatmulPlan plan);

typedef struct TiForgeCusparseLtExecutionApi {
  uint32_t struct_size;
  uint32_t execution_abi_version;
  TiForgeCusparseLtCreatePlanFn create_matmul_plan;
  TiForgeCusparseLtCompressFn compress_sparse_a;
  TiForgeCusparseLtExecuteFn execute_matmul;
  TiForgeCusparseLtDestroyPlanFn destroy_matmul_plan;
} TiForgeCusparseLtExecutionApi;

// Optional cuSPARSELt-specific extension. ABI 1 remains available unchanged.
// These are provider-private materializer facts, not public optimizer axes.
#define TI_FORGE_CUSPARSELT_CONFIG_EXECUTION_ABI_VERSION 2u
#define TI_FORGE_CUSPARSELT_ATTRIBUTE_DEFAULT INT32_MIN

typedef struct TiForgeCusparseLtMatmulConfig {
  uint32_t struct_size;
  int32_t algorithm_id;  // -1 selects the vendor default at creation.
  int32_t split_k;       // ATTRIBUTE_DEFAULT leaves an attribute untouched.
  int32_t split_k_mode;
  int32_t split_k_buffers;
  uint32_t relu;
} TiForgeCusparseLtMatmulConfig;

typedef TiForgeRuntimeProviderResult (*TiForgeCusparseLtCreateConfiguredPlanFn)(
    TiForgeRuntimeProviderRuntime runtime,
    const TiForgeCusparseLtMatmulPlanDesc *desc,
    const TiForgeCusparseLtMatmulConfig *configuration,
    TiForgeCusparseLtMatmulPlan *out_plan,
    TiForgeCusparseLtMatmulPlanInfo *out_info);
typedef TiForgeRuntimeProviderResult (*TiForgeCusparseLtGetConfigurationFn)(
    TiForgeCusparseLtMatmulPlan plan,
    TiForgeCusparseLtMatmulConfig *out_configuration,
    int32_t *out_algorithm_count);

typedef struct TiForgeCusparseLtConfiguredExecutionApi {
  uint32_t struct_size;
  uint32_t execution_abi_version;
  TiForgeCusparseLtCreateConfiguredPlanFn create_matmul_plan;
  TiForgeCusparseLtGetConfigurationFn get_configuration;
} TiForgeCusparseLtConfiguredExecutionApi;

typedef void *TiForgeAmgxSolver;

typedef enum TiForgeAmgxValueType {
  TI_FORGE_AMGX_VALUE_F32 = 1,
  TI_FORGE_AMGX_VALUE_F64 = 2,
} TiForgeAmgxValueType;

typedef enum TiForgeAmgxConfigSource {
  TI_FORGE_AMGX_CONFIG_STRING = 0,
  TI_FORGE_AMGX_CONFIG_FILE = 1,
} TiForgeAmgxConfigSource;

typedef struct TiForgeAmgxSolverDesc {
  uint32_t struct_size;
  uint32_t value_type;
  uint32_t config_source;
  uint32_t reserved;
  int32_t rows;
  int32_t nonzeros;
  const int32_t *row_offsets;
  const int32_t *column_indices;
  const void *values;
  const char *config;
} TiForgeAmgxSolverDesc;

typedef struct TiForgeAmgxSolveDesc {
  uint32_t struct_size;
  uint32_t zero_initial_guess;
  const void *rhs;
  void *solution;
} TiForgeAmgxSolveDesc;

typedef struct TiForgeAmgxSolveInfo {
  uint32_t struct_size;
  uint32_t solve_status;
  int32_t iterations;
  uint32_t reserved;
  double residual_norm;
} TiForgeAmgxSolveInfo;

typedef TiForgeRuntimeProviderResult (*TiForgeAmgxCreateSolverFn)(
    TiForgeRuntimeProviderRuntime runtime,
    const TiForgeAmgxSolverDesc *desc,
    TiForgeAmgxSolver *out_solver);
typedef TiForgeRuntimeProviderResult (*TiForgeAmgxReplaceCoefficientsFn)(
    TiForgeAmgxSolver solver,
    const void *values,
    int32_t nonzeros);
typedef TiForgeRuntimeProviderResult (*TiForgeAmgxSolveFn)(
    TiForgeAmgxSolver solver,
    const TiForgeAmgxSolveDesc *desc,
    TiForgeAmgxSolveInfo *out_info);
typedef TiForgeRuntimeProviderResult (*TiForgeAmgxDestroySolverFn)(
    TiForgeAmgxSolver solver);

typedef struct TiForgeAmgxExecutionApi {
  uint32_t struct_size;
  uint32_t execution_abi_version;
  TiForgeAmgxCreateSolverFn create_solver;
  TiForgeAmgxReplaceCoefficientsFn replace_coefficients;
  TiForgeAmgxSolveFn solve;
  TiForgeAmgxDestroySolverFn destroy_solver;
} TiForgeAmgxExecutionApi;

typedef TiForgeRuntimeProviderResult (*TiForgeRuntimeProviderQueryFn)(
    uint32_t requested_abi_version,
    size_t api_size,
    TiForgeRuntimeProviderApi *out_api);

#ifdef __cplusplus
}
#endif
