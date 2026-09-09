#include "taichi/external_runtime/forge_runtime_provider.h"

#if defined(_WIN32) && !defined(NOMINMAX)
#define NOMINMAX
#endif

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <mutex>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#ifndef TI_FORGE_RUNTIME_PROVIDER_KIND
#error "TI_FORGE_RUNTIME_PROVIDER_KIND must select one vendor adapter"
#endif

namespace {

thread_local std::string last_error;
thread_local std::string probed_library_path;
thread_local std::string probed_build_version;

constexpr uint64_t kFeatures =
    TI_FORGE_RUNTIME_PROVIDER_FEATURE_VERSION_QUERY |
    TI_FORGE_RUNTIME_PROVIDER_FEATURE_REQUIRED_SYMBOL_AUDIT |
    TI_FORGE_RUNTIME_PROVIDER_FEATURE_TRANSIENT_PROBE |
    TI_FORGE_RUNTIME_PROVIDER_FEATURE_EXECUTION_API |
    (TI_FORGE_RUNTIME_PROVIDER_KIND == 3
         ? TI_FORGE_RUNTIME_PROVIDER_FEATURE_AMGX_OPTIONAL_RESIDUAL
         : 0ull);

#if TI_FORGE_RUNTIME_PROVIDER_KIND == 1
constexpr char kProviderId[] = "cusparselt";
constexpr char kProviderName[] = "NVIDIA cuSPARSELt";
constexpr char kSupportedVersionFamily[] = "0.8.x-0.9.x";
constexpr char kBuildIdentity[] =
    "forge-runtime-provider-abi2-cusparselt-api-0.8-0.9-configured-plan-v1";
constexpr const char *kWindowsCandidates[] = {"cusparseLt64_0.dll",
                                              "cusparseLt.dll"};
constexpr const char *kLinuxCandidates[] = {"libcusparseLt.so.0",
                                            "libcusparseLt.so"};
constexpr const char *kRequiredSymbols[] = {
    "cusparseLtGetProperty",
    "cusparseLtInit",
    "cusparseLtDestroy",
    "cusparseLtDenseDescriptorInit",
    "cusparseLtStructuredDescriptorInit",
    "cusparseLtMatDescriptorDestroy",
    "cusparseLtMatmulDescriptorInit",
    "cusparseLtMatmulAlgSelectionInit",
    "cusparseLtMatmulPlanInit",
    "cusparseLtMatmulPlanDestroy",
    "cusparseLtMatmulGetWorkspace",
    "cusparseLtMatmul",
    "cusparseLtMatmulSearch",
    "cusparseLtMatmulAlgSelectionDestroy",
    "cusparseLtSpMMACompressedSize",
    "cusparseLtSpMMACompress",
    "cusparseLtGetErrorString",
};
#define TI_FORGE_RUNTIME_PROVIDER_QUERY_FUNCTION \
  taichi_forge_cusparselt_provider_query
#elif TI_FORGE_RUNTIME_PROVIDER_KIND == 2
constexpr char kProviderId[] = "cutensor";
constexpr char kProviderName[] = "NVIDIA cuTENSOR";
constexpr char kSupportedVersionFamily[] = "2.0.x-2.7.x";
constexpr char kBuildIdentity[] =
    "forge-runtime-provider-abi2-cutensor-api-2.0-2.7";
constexpr const char *kWindowsCandidates[] = {"cutensor64_2.dll",
                                              "cutensor.dll"};
constexpr const char *kLinuxCandidates[] = {"libcutensor.so.2",
                                            "libcutensor.so"};
constexpr const char *kRequiredSymbols[] = {
    "cutensorGetVersion",
    "cutensorGetCudartVersion",
    "cutensorCreate",
    "cutensorDestroy",
    "cutensorCreateTensorDescriptor",
    "cutensorDestroyTensorDescriptor",
    "cutensorCreateContraction",
    "cutensorCreateReduction",
    "cutensorCreatePlanPreference",
    "cutensorDestroyPlanPreference",
    "cutensorEstimateWorkspaceSize",
    "cutensorCreatePlan",
    "cutensorDestroyPlan",
    "cutensorPlanGetAttribute",
    "cutensorContract",
    "cutensorReduce",
    "cutensorPermute",
    "cutensorGetErrorString",
    "CUTENSOR_COMPUTE_DESC_32F",
    "CUTENSOR_COMPUTE_DESC_TF32",
};
#define TI_FORGE_RUNTIME_PROVIDER_QUERY_FUNCTION \
  taichi_forge_cutensor_provider_query
#elif TI_FORGE_RUNTIME_PROVIDER_KIND == 3
constexpr char kProviderId[] = "amgx";
constexpr char kProviderName[] = "NVIDIA AmgX";
constexpr char kSupportedVersionFamily[] = "stable C API";
constexpr char kBuildIdentity[] =
    "forge-runtime-provider-abi2-amgx-stable-c-api-residual-policy";
constexpr const char *kWindowsCandidates[] = {"amgxsh.dll"};
constexpr const char *kLinuxCandidates[] = {"libamgxsh.so"};
constexpr const char *kRequiredSymbols[] = {
    "AMGX_get_api_version",
    "AMGX_get_build_info_strings",
    "AMGX_get_error_string",
    "AMGX_initialize",
    "AMGX_finalize",
    "AMGX_config_create",
    "AMGX_config_create_from_file",
    "AMGX_config_destroy",
    "AMGX_resources_create_simple",
    "AMGX_resources_destroy",
    "AMGX_matrix_create",
    "AMGX_matrix_destroy",
    "AMGX_matrix_upload_all",
    "AMGX_matrix_replace_coefficients",
    "AMGX_vector_create",
    "AMGX_vector_destroy",
    "AMGX_vector_upload",
    "AMGX_vector_download",
    "AMGX_vector_bind",
    "AMGX_vector_set_zero",
    "AMGX_solver_create",
    "AMGX_solver_destroy",
    "AMGX_solver_setup",
    "AMGX_solver_solve",
    "AMGX_solver_solve_with_0_initial_guess",
    "AMGX_solver_get_status",
    "AMGX_solver_get_iterations_number",
    "AMGX_solver_calculate_residual_norm",
};
#define TI_FORGE_RUNTIME_PROVIDER_QUERY_FUNCTION \
  taichi_forge_amgx_provider_query
#else
#error "Unsupported TI_FORGE_RUNTIME_PROVIDER_KIND"
#endif

constexpr uint32_t kRequiredSymbolCount = static_cast<uint32_t>(
    sizeof(kRequiredSymbols) / sizeof(kRequiredSymbols[0]));

TiForgeRuntimeProviderResult fail(TiForgeRuntimeProviderResult result,
                                  std::string message) {
  last_error = std::move(message);
  return result;
}

#if defined(_WIN32)
using LibraryHandle = HMODULE;

LibraryHandle open_library(const std::string &path) {
  const int wide_size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                            path.c_str(), -1, nullptr, 0);
  if (wide_size <= 0) {
    return nullptr;
  }
  std::wstring wide_path(static_cast<std::size_t>(wide_size), L'\0');
  if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path.c_str(), -1,
                          wide_path.data(), wide_size) <= 0) {
    return nullptr;
  }
  return LoadLibraryW(wide_path.c_str());
}

void close_library(LibraryHandle handle) {
  if (handle != nullptr) {
    FreeLibrary(handle);
  }
}

void *load_symbol(LibraryHandle handle, const char *name) {
  return reinterpret_cast<void *>(GetProcAddress(handle, name));
}
#else
using LibraryHandle = void *;

LibraryHandle open_library(const std::string &path) {
  return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
}

void close_library(LibraryHandle handle) {
  if (handle != nullptr) {
    dlclose(handle);
  }
}

void *load_symbol(LibraryHandle handle, const char *name) {
  return dlsym(handle, name);
}
#endif

struct Runtime {
  LibraryHandle library{nullptr};
  std::string library_path;
  std::string build_version;
  uint32_t version_major{0};
  uint32_t version_minor{0};
  uint32_t version_patch{0};
  uint32_t cuda_runtime_version{0};
  std::atomic<uint32_t> live_execution_resources{0};
  bool execution_runtime_initialized{false};

  ~Runtime() {
    close_library(library);
  }
};

TiForgeRuntimeProviderResult initialize_execution_runtime(Runtime &runtime);
TiForgeRuntimeProviderResult shutdown_execution_runtime(Runtime &runtime);
TiForgeRuntimeProviderResult query_execution_api(
    TiForgeRuntimeProviderRuntime runtime,
    uint32_t requested_execution_abi_version,
    size_t api_size,
    void *out_api);

bool audit_required_symbols(Runtime &runtime) {
  for (const char *symbol : kRequiredSymbols) {
    if (load_symbol(runtime.library, symbol) == nullptr) {
      last_error = std::string(kProviderName) +
                   " runtime is missing required symbol " + symbol;
      return false;
    }
  }
  return true;
}

bool query_version(Runtime &runtime) {
#if TI_FORGE_RUNTIME_PROVIDER_KIND == 1
  using GetProperty = int (*)(int, int *);
  auto get_property = reinterpret_cast<GetProperty>(
      load_symbol(runtime.library, "cusparseLtGetProperty"));
  int major = -1;
  int minor = -1;
  int patch = -1;
  if (get_property == nullptr || get_property(0, &major) != 0 ||
      get_property(1, &minor) != 0 || get_property(2, &patch) != 0) {
    last_error = "cuSPARSELt runtime version query failed";
    return false;
  }
  if (major != 0 || minor < 8 || minor > 9) {
    last_error = "cuSPARSELt adapter supports runtime versions 0.8.x-0.9.x";
    return false;
  }
  runtime.version_major = static_cast<uint32_t>(major);
  runtime.version_minor = static_cast<uint32_t>(minor);
  runtime.version_patch = static_cast<uint32_t>(patch);
#elif TI_FORGE_RUNTIME_PROVIDER_KIND == 2
  using GetVersion = size_t (*)();
  auto get_version = reinterpret_cast<GetVersion>(
      load_symbol(runtime.library, "cutensorGetVersion"));
  auto get_cudart_version = reinterpret_cast<GetVersion>(
      load_symbol(runtime.library, "cutensorGetCudartVersion"));
  if (get_version == nullptr || get_cudart_version == nullptr) {
    last_error = "cuTENSOR runtime version query symbols are missing";
    return false;
  }
  const size_t version = get_version();
  const uint32_t major = static_cast<uint32_t>(version / 10000);
  const uint32_t minor = static_cast<uint32_t>((version / 100) % 100);
  const uint32_t patch = static_cast<uint32_t>(version % 100);
  if (major != 2 || minor > 7) {
    last_error = "cuTENSOR adapter supports runtime versions 2.0.x-2.7.x";
    return false;
  }
  runtime.version_major = major;
  runtime.version_minor = minor;
  runtime.version_patch = patch;
  runtime.cuda_runtime_version = static_cast<uint32_t>(get_cudart_version());
#else
  using GetApiVersion = int (*)(int *, int *);
  using GetBuildInfoStrings = int (*)(char **, char **, char **);
  auto get_api_version = reinterpret_cast<GetApiVersion>(
      load_symbol(runtime.library, "AMGX_get_api_version"));
  auto get_build_info = reinterpret_cast<GetBuildInfoStrings>(
      load_symbol(runtime.library, "AMGX_get_build_info_strings"));
  int major = 0;
  int minor = 0;
  if (get_api_version == nullptr || get_api_version(&major, &minor) != 0 ||
      major <= 0 || minor < 0) {
    last_error = "AmgX stable C API version query failed";
    return false;
  }
  runtime.version_major = static_cast<uint32_t>(major);
  runtime.version_minor = static_cast<uint32_t>(minor);
  runtime.version_patch = 0;
  char *version = nullptr;
  char *date = nullptr;
  char *time = nullptr;
  if (get_build_info != nullptr &&
      get_build_info(&version, &date, &time) == 0 && version != nullptr) {
    runtime.build_version = version;
  } else {
    runtime.build_version =
        "api-" + std::to_string(major) + "." + std::to_string(minor);
  }
#endif
  return true;
}

void fill_runtime_info(const Runtime &runtime, TiForgeRuntimeInfo *info) {
  if (info == nullptr) {
    return;
  }
  info->struct_size = sizeof(*info);
  info->version_major = runtime.version_major;
  info->version_minor = runtime.version_minor;
  info->version_patch = runtime.version_patch;
  info->cuda_runtime_version = runtime.cuda_runtime_version;
  info->reserved = 0;
  info->library_path = runtime.library_path.c_str();
  info->build_version =
      runtime.build_version.empty() ? nullptr : runtime.build_version.c_str();
}

TiForgeRuntimeProviderResult make_runtime(const char *library_path,
                                          Runtime **out_runtime) {
  if (out_runtime == nullptr) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                std::string(kProviderName) + " runtime output is null");
  }
  *out_runtime = nullptr;
  last_error.clear();
  std::vector<std::string> candidates;
  if (library_path != nullptr && library_path[0] != '\0') {
    candidates.emplace_back(library_path);
  } else {
#if defined(_WIN32)
    for (const char *candidate : kWindowsCandidates) {
      candidates.emplace_back(candidate);
    }
#else
    for (const char *candidate : kLinuxCandidates) {
      candidates.emplace_back(candidate);
    }
#endif
  }
  std::string attempted;
  for (const auto &candidate : candidates) {
    auto runtime = std::unique_ptr<Runtime>(new (std::nothrow) Runtime());
    if (!runtime) {
      return fail(
          TI_FORGE_RUNTIME_PROVIDER_ERROR_OUT_OF_MEMORY,
          std::string(kProviderName) + " adapter runtime allocation failed");
    }
    runtime->library_path = candidate;
    runtime->library = open_library(candidate);
    if (runtime->library == nullptr) {
      if (!attempted.empty()) {
        attempted += ", ";
      }
      attempted += candidate;
      continue;
    }
    if (!audit_required_symbols(*runtime) || !query_version(*runtime)) {
      return TI_FORGE_RUNTIME_PROVIDER_ERROR_RUNTIME_INCOMPATIBLE;
    }
    *out_runtime = runtime.release();
    return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
  }
  return fail(
      TI_FORGE_RUNTIME_PROVIDER_ERROR_RUNTIME_UNAVAILABLE,
      std::string(kProviderName) +
          " vendor runtime could not be loaded; attempted: " + attempted);
}

TiForgeRuntimeProviderResult probe_runtime(const char *library_path,
                                           TiForgeRuntimeInfo *out_info) {
  if (out_info != nullptr && out_info->struct_size < sizeof(*out_info)) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "runtime info output is truncated");
  }
  Runtime *runtime = nullptr;
  const auto result = make_runtime(library_path, &runtime);
  if (result != TI_FORGE_RUNTIME_PROVIDER_SUCCESS) {
    return result;
  }
  probed_library_path = runtime->library_path;
  probed_build_version = runtime->build_version;
  fill_runtime_info(*runtime, out_info);
  if (out_info != nullptr) {
    out_info->library_path = probed_library_path.c_str();
    out_info->build_version =
        probed_build_version.empty() ? nullptr : probed_build_version.c_str();
  }
  delete runtime;
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}

TiForgeRuntimeProviderResult create_runtime(
    const char *library_path,
    TiForgeRuntimeProviderRuntime *out_runtime,
    TiForgeRuntimeInfo *out_info) {
  if (out_runtime == nullptr ||
      (out_info != nullptr && out_info->struct_size < sizeof(*out_info))) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "runtime output is null or truncated");
  }
  *out_runtime = nullptr;
  Runtime *runtime = nullptr;
  const auto result = make_runtime(library_path, &runtime);
  if (result != TI_FORGE_RUNTIME_PROVIDER_SUCCESS) {
    return result;
  }
  const auto initialization_result = initialize_execution_runtime(*runtime);
  if (initialization_result != TI_FORGE_RUNTIME_PROVIDER_SUCCESS) {
    delete runtime;
    return initialization_result;
  }
  fill_runtime_info(*runtime, out_info);
  *out_runtime = runtime;
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}

TiForgeRuntimeProviderResult destroy_runtime(
    TiForgeRuntimeProviderRuntime runtime) {
  if (runtime == nullptr) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "adapter runtime is null");
  }
  auto *typed_runtime = static_cast<Runtime *>(runtime);
  if (typed_runtime->live_execution_resources.load() != 0) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_LIFETIME,
                std::string(kProviderName) +
                    " runtime still owns live execution resources");
  }
  const auto shutdown_result = shutdown_execution_runtime(*typed_runtime);
  if (shutdown_result != TI_FORGE_RUNTIME_PROVIDER_SUCCESS) {
    return shutdown_result;
  }
  delete typed_runtime;
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}

#if TI_FORGE_RUNTIME_PROVIDER_KIND == 1

#if defined(_WIN32)
#define TI_FORGE_CUSPARSELT_CALL __stdcall
#else
#define TI_FORGE_CUSPARSELT_CALL
#endif

struct alignas(16) CusparseLtOpaque {
  uint8_t data[512];
};

using CusparseLtInitFn = int(TI_FORGE_CUSPARSELT_CALL *)(CusparseLtOpaque *);
using CusparseLtDestroyFn =
    int(TI_FORGE_CUSPARSELT_CALL *)(const CusparseLtOpaque *);
using CusparseLtDenseDescriptorInitFn =
    int(TI_FORGE_CUSPARSELT_CALL *)(const CusparseLtOpaque *,
                                    CusparseLtOpaque *,
                                    int64_t,
                                    int64_t,
                                    int64_t,
                                    uint32_t,
                                    int,
                                    int);
using CusparseLtStructuredDescriptorInitFn =
    int(TI_FORGE_CUSPARSELT_CALL *)(const CusparseLtOpaque *,
                                    CusparseLtOpaque *,
                                    int64_t,
                                    int64_t,
                                    int64_t,
                                    uint32_t,
                                    int,
                                    int,
                                    int);
using CusparseLtDescriptorDestroyFn =
    int(TI_FORGE_CUSPARSELT_CALL *)(const CusparseLtOpaque *);
using CusparseLtMatmulDescriptorInitFn =
    int(TI_FORGE_CUSPARSELT_CALL *)(const CusparseLtOpaque *,
                                    CusparseLtOpaque *,
                                    int,
                                    int,
                                    const CusparseLtOpaque *,
                                    const CusparseLtOpaque *,
                                    const CusparseLtOpaque *,
                                    const CusparseLtOpaque *,
                                    int);
using CusparseLtAlgSelectionInitFn =
    int(TI_FORGE_CUSPARSELT_CALL *)(const CusparseLtOpaque *,
                                    CusparseLtOpaque *,
                                    const CusparseLtOpaque *,
                                    int);
using CusparseLtPlanInitFn =
    int(TI_FORGE_CUSPARSELT_CALL *)(const CusparseLtOpaque *,
                                    CusparseLtOpaque *,
                                    const CusparseLtOpaque *,
                                    const CusparseLtOpaque *);
using CusparseLtCompressedSizeFn =
    int(TI_FORGE_CUSPARSELT_CALL *)(const CusparseLtOpaque *,
                                    const CusparseLtOpaque *,
                                    size_t *,
                                    size_t *);
using CusparseLtWorkspaceFn =
    int(TI_FORGE_CUSPARSELT_CALL *)(const CusparseLtOpaque *,
                                    const CusparseLtOpaque *,
                                    size_t *);
using CusparseLtCompressFn =
    int(TI_FORGE_CUSPARSELT_CALL *)(const CusparseLtOpaque *,
                                    const CusparseLtOpaque *,
                                    const void *,
                                    void *,
                                    void *,
                                    void *);
using CusparseLtMatmulFn =
    int(TI_FORGE_CUSPARSELT_CALL *)(const CusparseLtOpaque *,
                                    const CusparseLtOpaque *,
                                    const void *,
                                    const void *,
                                    const void *,
                                    const void *,
                                    const void *,
                                    void *,
                                    void *,
                                    void **,
                                    int32_t);
using CusparseLtGetErrorStringFn =
    const char *(TI_FORGE_CUSPARSELT_CALL *)(int);
using CusparseLtSetAttributeFn =
    int(TI_FORGE_CUSPARSELT_CALL *)(const CusparseLtOpaque *,
                                    CusparseLtOpaque *,
                                    int,
                                    const void *,
                                    size_t);
using CusparseLtGetAttributeFn =
    int(TI_FORGE_CUSPARSELT_CALL *)(const CusparseLtOpaque *,
                                    const CusparseLtOpaque *,
                                    int,
                                    void *,
                                    size_t);

struct CusparseLtPlan {
  Runtime *runtime{nullptr};
  CusparseLtOpaque handle{};
  CusparseLtOpaque a{};
  CusparseLtOpaque b{};
  CusparseLtOpaque c{};
  CusparseLtOpaque matmul{};
  CusparseLtOpaque algorithm{};
  CusparseLtOpaque plan{};
  bool handle_live{false};
  bool a_live{false};
  bool b_live{false};
  bool c_live{false};
  bool algorithm_live{false};
  bool plan_live{false};
  uint64_t compressed_bytes{0};
  uint64_t compression_buffer_bytes{0};
  uint64_t workspace_bytes{0};
  CusparseLtDestroyFn destroy_handle{nullptr};
  CusparseLtDescriptorDestroyFn destroy_descriptor{nullptr};
  CusparseLtDescriptorDestroyFn destroy_algorithm{nullptr};
  CusparseLtDescriptorDestroyFn destroy_plan{nullptr};
  CusparseLtCompressFn compress{nullptr};
  CusparseLtMatmulFn execute{nullptr};
  CusparseLtGetErrorStringFn error_string{nullptr};
  CusparseLtGetAttributeFn get_algorithm_attribute{nullptr};
  bool relu{false};
  std::mutex mutex;
};

TiForgeRuntimeProviderResult cusparselt_fail(CusparseLtPlan *plan,
                                             int status,
                                             const char *operation) {
  std::string message = std::string("cuSPARSELt ") + operation + " failed";
  if (plan != nullptr && plan->error_string != nullptr) {
    const char *detail = plan->error_string(status);
    if (detail != nullptr && detail[0] != '\0') {
      message += ": ";
      message += detail;
    }
  }
  return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_VENDOR_CALL, std::move(message));
}

void cleanup_cusparselt_plan(CusparseLtPlan &plan) {
  if (plan.plan_live && plan.destroy_plan != nullptr) {
    plan.destroy_plan(&plan.plan);
    plan.plan_live = false;
  }
  if (plan.algorithm_live && plan.destroy_algorithm != nullptr) {
    plan.destroy_algorithm(&plan.algorithm);
    plan.algorithm_live = false;
  }
  for (auto item :
       {std::pair{&plan.c, &plan.c_live}, std::pair{&plan.b, &plan.b_live},
        std::pair{&plan.a, &plan.a_live}}) {
    if (*item.second && plan.destroy_descriptor != nullptr) {
      plan.destroy_descriptor(item.first);
      *item.second = false;
    }
  }
  if (plan.handle_live && plan.destroy_handle != nullptr) {
    plan.destroy_handle(&plan.handle);
    plan.handle_live = false;
  }
}

TiForgeRuntimeProviderResult cusparselt_create_plan_impl(
    TiForgeRuntimeProviderRuntime runtime_value,
    const TiForgeCusparseLtMatmulPlanDesc *desc,
    const TiForgeCusparseLtMatmulConfig *configuration,
    TiForgeCusparseLtMatmulPlan *out_plan,
    TiForgeCusparseLtMatmulPlanInfo *out_info) {
  if (runtime_value == nullptr || desc == nullptr || out_plan == nullptr ||
      out_info == nullptr || desc->struct_size < sizeof(*desc) ||
      out_info->struct_size < sizeof(*out_info) || desc->m <= 0 ||
      desc->n <= 0 || desc->k <= 0 || desc->alignment_bytes == 0 ||
      desc->m % 16 != 0 || desc->n % 16 != 0 || desc->k % 16 != 0 ||
      (desc->alignment_bytes & (desc->alignment_bytes - 1)) != 0) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "invalid cuSPARSELt FP16 2:4 matmul plan description");
  }
  *out_plan = nullptr;
  if (configuration &&
      (configuration->struct_size < sizeof(*configuration) ||
       configuration->algorithm_id < -1 || configuration->relu > 1)) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "invalid cuSPARSELt physical configuration");
  }
  auto *runtime = static_cast<Runtime *>(runtime_value);
  auto plan =
      std::unique_ptr<CusparseLtPlan>(new (std::nothrow) CusparseLtPlan());
  if (!plan) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_OUT_OF_MEMORY,
                "cuSPARSELt plan allocation failed");
  }
  plan->runtime = runtime;
  auto symbol = [&](const char *name) {
    return load_symbol(runtime->library, name);
  };
  auto init = reinterpret_cast<CusparseLtInitFn>(symbol("cusparseLtInit"));
  auto dense = reinterpret_cast<CusparseLtDenseDescriptorInitFn>(
      symbol("cusparseLtDenseDescriptorInit"));
  auto structured = reinterpret_cast<CusparseLtStructuredDescriptorInitFn>(
      symbol("cusparseLtStructuredDescriptorInit"));
  auto matmul_init = reinterpret_cast<CusparseLtMatmulDescriptorInitFn>(
      symbol("cusparseLtMatmulDescriptorInit"));
  auto algorithm_init = reinterpret_cast<CusparseLtAlgSelectionInitFn>(
      symbol("cusparseLtMatmulAlgSelectionInit"));
  auto plan_init = reinterpret_cast<CusparseLtPlanInitFn>(
      symbol("cusparseLtMatmulPlanInit"));
  auto compressed_size = reinterpret_cast<CusparseLtCompressedSizeFn>(
      symbol("cusparseLtSpMMACompressedSize"));
  auto workspace = reinterpret_cast<CusparseLtWorkspaceFn>(
      symbol("cusparseLtMatmulGetWorkspace"));
  plan->destroy_handle =
      reinterpret_cast<CusparseLtDestroyFn>(symbol("cusparseLtDestroy"));
  plan->destroy_descriptor = reinterpret_cast<CusparseLtDescriptorDestroyFn>(
      symbol("cusparseLtMatDescriptorDestroy"));
  plan->destroy_algorithm = reinterpret_cast<CusparseLtDescriptorDestroyFn>(
      symbol("cusparseLtMatmulAlgSelectionDestroy"));
  plan->destroy_plan = reinterpret_cast<CusparseLtDescriptorDestroyFn>(
      symbol("cusparseLtMatmulPlanDestroy"));
  plan->compress =
      reinterpret_cast<CusparseLtCompressFn>(symbol("cusparseLtSpMMACompress"));
  plan->execute =
      reinterpret_cast<CusparseLtMatmulFn>(symbol("cusparseLtMatmul"));
  plan->error_string = reinterpret_cast<CusparseLtGetErrorStringFn>(
      symbol("cusparseLtGetErrorString"));
  auto set_algorithm = reinterpret_cast<CusparseLtSetAttributeFn>(
      symbol("cusparseLtMatmulAlgSetAttribute"));
  auto set_matmul = reinterpret_cast<CusparseLtSetAttributeFn>(
      symbol("cusparseLtMatmulDescSetAttribute"));
  plan->get_algorithm_attribute = reinterpret_cast<CusparseLtGetAttributeFn>(
      symbol("cusparseLtMatmulAlgGetAttribute"));
  if (configuration &&
      (!set_algorithm || !set_matmul || !plan->get_algorithm_attribute)) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_EXECUTION_UNSUPPORTED,
                "cuSPARSELt configured-plan attributes are unavailable");
  }

  auto checked = [&](int status, const char *operation) {
    return status == 0 ? TI_FORGE_RUNTIME_PROVIDER_SUCCESS
                       : cusparselt_fail(plan.get(), status, operation);
  };
  int status = init(&plan->handle);
  if (status != 0) {
    return cusparselt_fail(plan.get(), status, "handle creation");
  }
  plan->handle_live = true;
  constexpr int kCudaR16F = 2;
  constexpr int kOrderRow = 2;
  constexpr int kNonTranspose = 0;
  constexpr int kTranspose = 1;
  constexpr int kCompute32F = 2;
  const uint32_t alignment = desc->alignment_bytes;
  if (checked(structured(&plan->handle, &plan->a, desc->m, desc->k, desc->k,
                         alignment, kCudaR16F, kOrderRow, 0),
              "structured A descriptor creation") !=
      TI_FORGE_RUNTIME_PROVIDER_SUCCESS) {
    cleanup_cusparselt_plan(*plan);
    return TI_FORGE_RUNTIME_PROVIDER_ERROR_VENDOR_CALL;
  }
  plan->a_live = true;
  if (checked(dense(&plan->handle, &plan->b, desc->n, desc->k, desc->k,
                    alignment, kCudaR16F, kOrderRow),
              "dense B descriptor creation") !=
      TI_FORGE_RUNTIME_PROVIDER_SUCCESS) {
    cleanup_cusparselt_plan(*plan);
    return TI_FORGE_RUNTIME_PROVIDER_ERROR_VENDOR_CALL;
  }
  plan->b_live = true;
  if (checked(dense(&plan->handle, &plan->c, desc->m, desc->n, desc->n,
                    alignment, kCudaR16F, kOrderRow),
              "dense C/D descriptor creation") !=
      TI_FORGE_RUNTIME_PROVIDER_SUCCESS) {
    cleanup_cusparselt_plan(*plan);
    return TI_FORGE_RUNTIME_PROVIDER_ERROR_VENDOR_CALL;
  }
  plan->c_live = true;
  if (checked(
          matmul_init(&plan->handle, &plan->matmul, kNonTranspose, kTranspose,
                      &plan->a, &plan->b, &plan->c, &plan->c, kCompute32F),
          "matmul descriptor creation") != TI_FORGE_RUNTIME_PROVIDER_SUCCESS) {
    cleanup_cusparselt_plan(*plan);
    return TI_FORGE_RUNTIME_PROVIDER_ERROR_VENDOR_CALL;
  }
  if (configuration && configuration->relu) {
    const int32_t enabled = 1;
    // CUSPARSELT_MATMUL_ACTIVATION_RELU = 0, int32, in supported 0.8/0.9
    // headers.
    if (checked(set_matmul(&plan->handle, &plan->matmul, 0, &enabled,
                           sizeof(enabled)),
                "ReLU epilogue configuration") !=
        TI_FORGE_RUNTIME_PROVIDER_SUCCESS) {
      cleanup_cusparselt_plan(*plan);
      return TI_FORGE_RUNTIME_PROVIDER_ERROR_VENDOR_CALL;
    }
    plan->relu = true;
  }
  if (checked(algorithm_init(&plan->handle, &plan->algorithm, &plan->matmul, 0),
              "algorithm selection creation") !=
      TI_FORGE_RUNTIME_PROVIDER_SUCCESS) {
    cleanup_cusparselt_plan(*plan);
    return TI_FORGE_RUNTIME_PROVIDER_ERROR_VENDOR_CALL;
  }
  plan->algorithm_live = true;
  if (configuration) {
    // Algorithm attributes 0/3/4/5 are stable across the supported 0.8/0.9 ABI.
    // No vendor search is performed, and a live plan is never mutated.
    const std::pair<int, int32_t> attributes[] = {
        {0, configuration->algorithm_id < 0
                ? TI_FORGE_CUSPARSELT_ATTRIBUTE_DEFAULT
                : configuration->algorithm_id},
        {3, configuration->split_k},
        {4, configuration->split_k_mode},
        {5, configuration->split_k_buffers}};
    for (const auto &attribute : attributes) {
      if (attribute.second == TI_FORGE_CUSPARSELT_ATTRIBUTE_DEFAULT)
        continue;
      if (checked(
              set_algorithm(&plan->handle, &plan->algorithm, attribute.first,
                            &attribute.second, sizeof(attribute.second)),
              "algorithm configuration") != TI_FORGE_RUNTIME_PROVIDER_SUCCESS) {
        cleanup_cusparselt_plan(*plan);
        return TI_FORGE_RUNTIME_PROVIDER_ERROR_VENDOR_CALL;
      }
    }
  }
  if (checked(plan_init(&plan->handle, &plan->plan, &plan->matmul,
                        &plan->algorithm),
              "matmul plan creation") != TI_FORGE_RUNTIME_PROVIDER_SUCCESS) {
    cleanup_cusparselt_plan(*plan);
    return TI_FORGE_RUNTIME_PROVIDER_ERROR_VENDOR_CALL;
  }
  plan->plan_live = true;
  size_t compressed_bytes = 0;
  size_t compression_buffer_bytes = 0;
  size_t workspace_bytes = 0;
  if (checked(compressed_size(&plan->handle, &plan->plan, &compressed_bytes,
                              &compression_buffer_bytes),
              "compressed-size query") != TI_FORGE_RUNTIME_PROVIDER_SUCCESS ||
      checked(workspace(&plan->handle, &plan->plan, &workspace_bytes),
              "workspace-size query") != TI_FORGE_RUNTIME_PROVIDER_SUCCESS) {
    cleanup_cusparselt_plan(*plan);
    return TI_FORGE_RUNTIME_PROVIDER_ERROR_VENDOR_CALL;
  }
  plan->compressed_bytes = compressed_bytes;
  plan->compression_buffer_bytes = compression_buffer_bytes;
  plan->workspace_bytes = workspace_bytes;
  *out_info = {sizeof(*out_info), 0, compressed_bytes, compression_buffer_bytes,
               workspace_bytes};
  runtime->live_execution_resources.fetch_add(1);
  *out_plan = plan.release();
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}

TiForgeRuntimeProviderResult cusparselt_create_plan(
    TiForgeRuntimeProviderRuntime runtime,
    const TiForgeCusparseLtMatmulPlanDesc *desc,
    TiForgeCusparseLtMatmulPlan *out_plan,
    TiForgeCusparseLtMatmulPlanInfo *out_info) {
  return cusparselt_create_plan_impl(runtime, desc, nullptr, out_plan,
                                     out_info);
}

TiForgeRuntimeProviderResult cusparselt_create_configured_plan(
    TiForgeRuntimeProviderRuntime runtime,
    const TiForgeCusparseLtMatmulPlanDesc *desc,
    const TiForgeCusparseLtMatmulConfig *configuration,
    TiForgeCusparseLtMatmulPlan *out_plan,
    TiForgeCusparseLtMatmulPlanInfo *out_info) {
  if (!configuration) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "cuSPARSELt configured plan requires configuration");
  }
  return cusparselt_create_plan_impl(runtime, desc, configuration, out_plan,
                                     out_info);
}

TiForgeRuntimeProviderResult cusparselt_get_configuration(
    TiForgeCusparseLtMatmulPlan plan_value,
    TiForgeCusparseLtMatmulConfig *out_configuration,
    int32_t *out_algorithm_count) {
  auto *plan = static_cast<CusparseLtPlan *>(plan_value);
  if (!plan || !out_configuration || !out_algorithm_count ||
      out_configuration->struct_size < sizeof(*out_configuration)) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "cuSPARSELt configuration output is null or truncated");
  }
  if (!plan->get_algorithm_attribute) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_EXECUTION_UNSUPPORTED,
                "cuSPARSELt algorithm configuration query is unavailable");
  }
  std::lock_guard<std::mutex> lock(plan->mutex);
  TiForgeCusparseLtMatmulConfig configuration{};
  configuration.struct_size = sizeof(configuration);
  configuration.relu = plan->relu;
  int32_t count = 0;
  const std::pair<int, int32_t *> attributes[] = {
      {0, &configuration.algorithm_id},
      {1, &count},
      {3, &configuration.split_k},
      {4, &configuration.split_k_mode},
      {5, &configuration.split_k_buffers}};
  for (const auto &attribute : attributes) {
    const auto status = plan->get_algorithm_attribute(
        &plan->handle, &plan->algorithm, attribute.first, attribute.second,
        sizeof(int32_t));
    if (status != 0)
      return cusparselt_fail(plan, status, "algorithm configuration query");
  }
  *out_configuration = configuration;
  *out_algorithm_count = count;
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}

TiForgeRuntimeProviderResult cusparselt_compress(
    TiForgeCusparseLtMatmulPlan plan_value,
    const TiForgeCusparseLtCompressDesc *desc) {
  auto *plan = static_cast<CusparseLtPlan *>(plan_value);
  if (plan == nullptr || desc == nullptr || desc->struct_size < sizeof(*desc) ||
      desc->dense_a == 0 || desc->compressed_a == 0 ||
      (plan->compression_buffer_bytes != 0 &&
       (desc->compression_buffer == 0 ||
        desc->compression_buffer_bytes < plan->compression_buffer_bytes))) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "invalid cuSPARSELt compression buffers");
  }
  std::lock_guard<std::mutex> lock(plan->mutex);
  const int status = plan->compress(
      &plan->handle, &plan->plan,
      reinterpret_cast<const void *>(static_cast<uintptr_t>(desc->dense_a)),
      reinterpret_cast<void *>(static_cast<uintptr_t>(desc->compressed_a)),
      reinterpret_cast<void *>(
          static_cast<uintptr_t>(desc->compression_buffer)),
      reinterpret_cast<void *>(static_cast<uintptr_t>(desc->cuda_stream)));
  return status == 0 ? TI_FORGE_RUNTIME_PROVIDER_SUCCESS
                     : cusparselt_fail(plan, status, "2:4 compression");
}

TiForgeRuntimeProviderResult cusparselt_execute(
    TiForgeCusparseLtMatmulPlan plan_value,
    const TiForgeCusparseLtMatmulExecDesc *desc) {
  auto *plan = static_cast<CusparseLtPlan *>(plan_value);
  if (plan == nullptr || desc == nullptr || desc->struct_size < sizeof(*desc) ||
      desc->compressed_a == 0 || desc->b == 0 || desc->c == 0 || desc->d == 0 ||
      (plan->workspace_bytes != 0 &&
       (desc->workspace == 0 ||
        desc->workspace_bytes < plan->workspace_bytes))) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "invalid cuSPARSELt matmul execution buffers");
  }
  std::lock_guard<std::mutex> lock(plan->mutex);
  void *stream =
      reinterpret_cast<void *>(static_cast<uintptr_t>(desc->cuda_stream));
  const int status = plan->execute(
      &plan->handle, &plan->plan, &desc->alpha,
      reinterpret_cast<const void *>(
          static_cast<uintptr_t>(desc->compressed_a)),
      reinterpret_cast<const void *>(static_cast<uintptr_t>(desc->b)),
      &desc->beta,
      reinterpret_cast<const void *>(static_cast<uintptr_t>(desc->c)),
      reinterpret_cast<void *>(static_cast<uintptr_t>(desc->d)),
      reinterpret_cast<void *>(static_cast<uintptr_t>(desc->workspace)),
      &stream, 1);
  return status == 0 ? TI_FORGE_RUNTIME_PROVIDER_SUCCESS
                     : cusparselt_fail(plan, status, "matmul execution");
}

TiForgeRuntimeProviderResult cusparselt_destroy_plan(
    TiForgeCusparseLtMatmulPlan plan_value) {
  auto *plan = static_cast<CusparseLtPlan *>(plan_value);
  if (plan == nullptr) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "cuSPARSELt plan is null");
  }
  Runtime *runtime = plan->runtime;
  cleanup_cusparselt_plan(*plan);
  delete plan;
  runtime->live_execution_resources.fetch_sub(1);
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}

TiForgeRuntimeProviderResult initialize_execution_runtime(Runtime &) {
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}

TiForgeRuntimeProviderResult shutdown_execution_runtime(Runtime &) {
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}

TiForgeRuntimeProviderResult query_execution_api(
    TiForgeRuntimeProviderRuntime runtime,
    uint32_t requested_execution_abi_version,
    size_t api_size,
    void *out_api) {
  if (runtime == nullptr || out_api == nullptr ||
      api_size < sizeof(uint32_t) * 2) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "cuSPARSELt execution API output is null or truncated");
  }
  if (requested_execution_abi_version ==
      TI_FORGE_CUSPARSELT_CONFIG_EXECUTION_ABI_VERSION) {
    if (api_size < sizeof(TiForgeCusparseLtConfiguredExecutionApi)) {
      return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                  "cuSPARSELt configured execution API is truncated");
    }
    const auto library = static_cast<Runtime *>(runtime)->library;
    for (const auto *name :
         {"cusparseLtMatmulAlgSetAttribute", "cusparseLtMatmulAlgGetAttribute",
          "cusparseLtMatmulDescSetAttribute"}) {
      if (!load_symbol(library, name)) {
        return fail(
            TI_FORGE_RUNTIME_PROVIDER_ERROR_EXECUTION_UNSUPPORTED,
            std::string("cuSPARSELt configured execution is missing ") + name);
      }
    }
    auto *api = static_cast<TiForgeCusparseLtConfiguredExecutionApi *>(out_api);
    *api = {sizeof(*api), TI_FORGE_CUSPARSELT_CONFIG_EXECUTION_ABI_VERSION,
            cusparselt_create_configured_plan, cusparselt_get_configuration};
    return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
  }
  if (requested_execution_abi_version !=
      TI_FORGE_RUNTIME_PROVIDER_EXECUTION_ABI_VERSION) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH,
                "unsupported cuSPARSELt execution ABI");
  }
  if (api_size < sizeof(TiForgeCusparseLtExecutionApi)) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "cuSPARSELt execution API output is truncated");
  }
  auto *api = static_cast<TiForgeCusparseLtExecutionApi *>(out_api);
  *api = {
      sizeof(*api),           TI_FORGE_RUNTIME_PROVIDER_EXECUTION_ABI_VERSION,
      cusparselt_create_plan, cusparselt_compress,
      cusparselt_execute,     cusparselt_destroy_plan};
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}

#undef TI_FORGE_CUSPARSELT_CALL

#elif TI_FORGE_RUNTIME_PROVIDER_KIND == 2

using CutensorCreateFn = int (*)(void **);
using CutensorDestroyFn = int (*)(void *);
using CutensorCreateTensorDescriptorFn = int (*)(void *,
                                                 void **,
                                                 uint32_t,
                                                 const int64_t *,
                                                 const int64_t *,
                                                 int,
                                                 uint32_t);
using CutensorCreateContractionFn = int (*)(void *,
                                            void **,
                                            void *,
                                            const int32_t *,
                                            int,
                                            void *,
                                            const int32_t *,
                                            int,
                                            void *,
                                            const int32_t *,
                                            int,
                                            void *,
                                            const int32_t *,
                                            void *);
using CutensorCreatePreferenceFn = int (*)(void *, void **, int, int);
using CutensorEstimateWorkspaceFn =
    int (*)(void *, void *, void *, int, uint64_t *);
using CutensorCreatePlanFn = int (*)(void *, void **, void *, void *, uint64_t);
using CutensorPlanAttributeFn = int (*)(void *, void *, int, void *, size_t);
using CutensorContractFn = int (*)(void *,
                                   void *,
                                   const void *,
                                   const void *,
                                   const void *,
                                   const void *,
                                   const void *,
                                   void *,
                                   void *,
                                   uint64_t,
                                   void *);
using CutensorGetErrorStringFn = const char *(*)(int);

struct CutensorPlan {
  Runtime *runtime{nullptr};
  void *handle{nullptr};
  void *a{nullptr};
  void *b{nullptr};
  void *c{nullptr};
  void *d{nullptr};
  void *operation{nullptr};
  void *preference{nullptr};
  void *plan{nullptr};
  uint64_t workspace_estimate_bytes{0};
  uint64_t workspace_required_bytes{0};
  CutensorDestroyFn destroy_handle{nullptr};
  CutensorDestroyFn destroy_tensor{nullptr};
  CutensorDestroyFn destroy_operation{nullptr};
  CutensorDestroyFn destroy_preference{nullptr};
  CutensorDestroyFn destroy_plan{nullptr};
  CutensorContractFn execute{nullptr};
  CutensorGetErrorStringFn error_string{nullptr};
  std::mutex mutex;
};

TiForgeRuntimeProviderResult cutensor_fail(CutensorPlan *plan,
                                           int status,
                                           const char *operation) {
  std::string message = std::string("cuTENSOR ") + operation + " failed";
  if (plan != nullptr && plan->error_string != nullptr) {
    const char *detail = plan->error_string(status);
    if (detail != nullptr && detail[0] != '\0') {
      message += ": ";
      message += detail;
    }
  }
  return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_VENDOR_CALL, std::move(message));
}

void cleanup_cutensor_plan(CutensorPlan &plan) {
  if (plan.plan != nullptr && plan.destroy_plan != nullptr) {
    plan.destroy_plan(plan.plan);
    plan.plan = nullptr;
  }
  if (plan.preference != nullptr && plan.destroy_preference != nullptr) {
    plan.destroy_preference(plan.preference);
    plan.preference = nullptr;
  }
  if (plan.operation != nullptr && plan.destroy_operation != nullptr) {
    plan.destroy_operation(plan.operation);
    plan.operation = nullptr;
  }
  for (void **tensor : {&plan.d, &plan.c, &plan.b, &plan.a}) {
    if (*tensor != nullptr && plan.destroy_tensor != nullptr) {
      plan.destroy_tensor(*tensor);
      *tensor = nullptr;
    }
  }
  if (plan.handle != nullptr && plan.destroy_handle != nullptr) {
    plan.destroy_handle(plan.handle);
    plan.handle = nullptr;
  }
}

bool valid_cutensor_tensor(const TiForgeCutensorTensorDesc &desc) {
  if (desc.struct_size < sizeof(desc) || desc.rank == 0 || desc.rank > 32 ||
      desc.extents == nullptr || desc.modes == nullptr) {
    return false;
  }
  for (uint32_t i = 0; i < desc.rank; ++i) {
    if (desc.extents[i] <= 0) {
      return false;
    }
    for (uint32_t j = i + 1; j < desc.rank; ++j) {
      if (desc.modes[i] == desc.modes[j]) {
        return false;
      }
    }
  }
  return true;
}

TiForgeRuntimeProviderResult cutensor_create_plan(
    TiForgeRuntimeProviderRuntime runtime_value,
    const TiForgeCutensorContractionPlanDesc *desc,
    TiForgeCutensorContractionPlan *out_plan,
    TiForgeCutensorContractionPlanInfo *out_info) {
  if (runtime_value == nullptr || desc == nullptr || out_plan == nullptr ||
      out_info == nullptr || desc->struct_size < sizeof(*desc) ||
      out_info->struct_size < sizeof(*out_info) ||
      !valid_cutensor_tensor(desc->a) || !valid_cutensor_tensor(desc->b) ||
      !valid_cutensor_tensor(desc->c) || !valid_cutensor_tensor(desc->d) ||
      desc->compute_mode > TI_FORGE_CUTENSOR_COMPUTE_TF32 ||
      desc->workspace_preference < TI_FORGE_CUTENSOR_WORKSPACE_MIN ||
      desc->workspace_preference > TI_FORGE_CUTENSOR_WORKSPACE_MAX ||
      desc->alignment_bytes == 0 ||
      (desc->alignment_bytes & (desc->alignment_bytes - 1)) != 0) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "invalid cuTENSOR FP32 contraction plan description");
  }
  *out_plan = nullptr;
  auto *runtime = static_cast<Runtime *>(runtime_value);
  auto plan = std::unique_ptr<CutensorPlan>(new (std::nothrow) CutensorPlan());
  if (!plan) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_OUT_OF_MEMORY,
                "cuTENSOR plan allocation failed");
  }
  plan->runtime = runtime;
  auto symbol = [&](const char *name) {
    return load_symbol(runtime->library, name);
  };
  auto create = reinterpret_cast<CutensorCreateFn>(symbol("cutensorCreate"));
  auto create_tensor = reinterpret_cast<CutensorCreateTensorDescriptorFn>(
      symbol("cutensorCreateTensorDescriptor"));
  auto create_contraction = reinterpret_cast<CutensorCreateContractionFn>(
      symbol("cutensorCreateContraction"));
  auto create_preference = reinterpret_cast<CutensorCreatePreferenceFn>(
      symbol("cutensorCreatePlanPreference"));
  auto estimate = reinterpret_cast<CutensorEstimateWorkspaceFn>(
      symbol("cutensorEstimateWorkspaceSize"));
  auto create_plan =
      reinterpret_cast<CutensorCreatePlanFn>(symbol("cutensorCreatePlan"));
  auto plan_attribute = reinterpret_cast<CutensorPlanAttributeFn>(
      symbol("cutensorPlanGetAttribute"));
  plan->destroy_handle =
      reinterpret_cast<CutensorDestroyFn>(symbol("cutensorDestroy"));
  plan->destroy_tensor = reinterpret_cast<CutensorDestroyFn>(
      symbol("cutensorDestroyTensorDescriptor"));
  plan->destroy_operation = reinterpret_cast<CutensorDestroyFn>(
      symbol("cutensorDestroyOperationDescriptor"));
  plan->destroy_preference = reinterpret_cast<CutensorDestroyFn>(
      symbol("cutensorDestroyPlanPreference"));
  plan->destroy_plan =
      reinterpret_cast<CutensorDestroyFn>(symbol("cutensorDestroyPlan"));
  plan->execute =
      reinterpret_cast<CutensorContractFn>(symbol("cutensorContract"));
  plan->error_string = reinterpret_cast<CutensorGetErrorStringFn>(
      symbol("cutensorGetErrorString"));
  const char *compute_symbol =
      desc->compute_mode == TI_FORGE_CUTENSOR_COMPUTE_TF32
          ? "CUTENSOR_COMPUTE_DESC_TF32"
          : "CUTENSOR_COMPUTE_DESC_32F";
  auto *compute_export = static_cast<void **>(symbol(compute_symbol));
  if (compute_export == nullptr || *compute_export == nullptr) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_RUNTIME_INCOMPATIBLE,
                std::string("cuTENSOR runtime is missing ") + compute_symbol);
  }
  auto checked = [&](int status, const char *operation) {
    return status == 0 ? TI_FORGE_RUNTIME_PROVIDER_SUCCESS
                       : cutensor_fail(plan.get(), status, operation);
  };
  if (checked(create(&plan->handle), "handle creation") !=
      TI_FORGE_RUNTIME_PROVIDER_SUCCESS) {
    return TI_FORGE_RUNTIME_PROVIDER_ERROR_VENDOR_CALL;
  }
  auto make_tensor = [&](const TiForgeCutensorTensorDesc &tensor, void **output,
                         const char *name) {
    return checked(
        create_tensor(plan->handle, output, tensor.rank, tensor.extents,
                      tensor.strides, 0, desc->alignment_bytes),
        name);
  };
  if (make_tensor(desc->a, &plan->a, "A descriptor creation") !=
          TI_FORGE_RUNTIME_PROVIDER_SUCCESS ||
      make_tensor(desc->b, &plan->b, "B descriptor creation") !=
          TI_FORGE_RUNTIME_PROVIDER_SUCCESS ||
      make_tensor(desc->c, &plan->c, "C descriptor creation") !=
          TI_FORGE_RUNTIME_PROVIDER_SUCCESS ||
      make_tensor(desc->d, &plan->d, "D descriptor creation") !=
          TI_FORGE_RUNTIME_PROVIDER_SUCCESS) {
    cleanup_cutensor_plan(*plan);
    return TI_FORGE_RUNTIME_PROVIDER_ERROR_VENDOR_CALL;
  }
  constexpr int kIdentity = 1;
  if (checked(
          create_contraction(plan->handle, &plan->operation, plan->a,
                             desc->a.modes, kIdentity, plan->b, desc->b.modes,
                             kIdentity, plan->c, desc->c.modes, kIdentity,
                             plan->d, desc->d.modes, *compute_export),
          "contraction descriptor creation") !=
          TI_FORGE_RUNTIME_PROVIDER_SUCCESS ||
      checked(create_preference(plan->handle, &plan->preference, -1, 0),
              "plan preference creation") !=
          TI_FORGE_RUNTIME_PROVIDER_SUCCESS ||
      checked(estimate(plan->handle, plan->operation, plan->preference,
                       static_cast<int>(desc->workspace_preference),
                       &plan->workspace_estimate_bytes),
              "workspace estimate") != TI_FORGE_RUNTIME_PROVIDER_SUCCESS) {
    cleanup_cutensor_plan(*plan);
    return TI_FORGE_RUNTIME_PROVIDER_ERROR_VENDOR_CALL;
  }
  const uint64_t workspace_limit =
      desc->workspace_limit_bytes == 0
          ? plan->workspace_estimate_bytes
          : std::min(desc->workspace_limit_bytes,
                     plan->workspace_estimate_bytes);
  if (checked(create_plan(plan->handle, &plan->plan, plan->operation,
                          plan->preference, workspace_limit),
              "plan creation") != TI_FORGE_RUNTIME_PROVIDER_SUCCESS ||
      checked(plan_attribute(plan->handle, plan->plan, 0,
                             &plan->workspace_required_bytes,
                             sizeof(plan->workspace_required_bytes)),
              "required workspace query") !=
          TI_FORGE_RUNTIME_PROVIDER_SUCCESS) {
    cleanup_cutensor_plan(*plan);
    return TI_FORGE_RUNTIME_PROVIDER_ERROR_VENDOR_CALL;
  }
  *out_info = {sizeof(*out_info), 0, plan->workspace_estimate_bytes,
               plan->workspace_required_bytes};
  runtime->live_execution_resources.fetch_add(1);
  *out_plan = plan.release();
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}

TiForgeRuntimeProviderResult cutensor_execute(
    TiForgeCutensorContractionPlan plan_value,
    const TiForgeCutensorContractionExecDesc *desc) {
  auto *plan = static_cast<CutensorPlan *>(plan_value);
  if (plan == nullptr || desc == nullptr || desc->struct_size < sizeof(*desc) ||
      desc->a == 0 || desc->b == 0 || desc->c == 0 || desc->d == 0 ||
      (plan->workspace_required_bytes != 0 &&
       (desc->workspace == 0 ||
        desc->workspace_bytes < plan->workspace_required_bytes))) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "invalid cuTENSOR contraction execution buffers");
  }
  std::lock_guard<std::mutex> lock(plan->mutex);
  const int status = plan->execute(
      plan->handle, plan->plan, &desc->alpha,
      reinterpret_cast<const void *>(static_cast<uintptr_t>(desc->a)),
      reinterpret_cast<const void *>(static_cast<uintptr_t>(desc->b)),
      &desc->beta,
      reinterpret_cast<const void *>(static_cast<uintptr_t>(desc->c)),
      reinterpret_cast<void *>(static_cast<uintptr_t>(desc->d)),
      reinterpret_cast<void *>(static_cast<uintptr_t>(desc->workspace)),
      desc->workspace_bytes,
      reinterpret_cast<void *>(static_cast<uintptr_t>(desc->cuda_stream)));
  return status == 0 ? TI_FORGE_RUNTIME_PROVIDER_SUCCESS
                     : cutensor_fail(plan, status, "contraction execution");
}

TiForgeRuntimeProviderResult cutensor_destroy_plan(
    TiForgeCutensorContractionPlan plan_value) {
  auto *plan = static_cast<CutensorPlan *>(plan_value);
  if (plan == nullptr) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "cuTENSOR plan is null");
  }
  Runtime *runtime = plan->runtime;
  cleanup_cutensor_plan(*plan);
  delete plan;
  runtime->live_execution_resources.fetch_sub(1);
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}

TiForgeRuntimeProviderResult initialize_execution_runtime(Runtime &) {
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}

TiForgeRuntimeProviderResult shutdown_execution_runtime(Runtime &) {
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}

TiForgeRuntimeProviderResult query_execution_api(
    TiForgeRuntimeProviderRuntime runtime,
    uint32_t requested_execution_abi_version,
    size_t api_size,
    void *out_api) {
  if (runtime == nullptr || out_api == nullptr ||
      api_size < sizeof(TiForgeCutensorExecutionApi)) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "cuTENSOR execution API output is null or truncated");
  }
  if (requested_execution_abi_version !=
      TI_FORGE_RUNTIME_PROVIDER_EXECUTION_ABI_VERSION) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH,
                "unsupported cuTENSOR execution ABI");
  }
  auto *api = static_cast<TiForgeCutensorExecutionApi *>(out_api);
  *api = {sizeof(*api), TI_FORGE_RUNTIME_PROVIDER_EXECUTION_ABI_VERSION,
          cutensor_create_plan, cutensor_execute, cutensor_destroy_plan};
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}

#else

using AmgxSimpleFn = int (*)();
using AmgxConfigCreateFn = int (*)(void **, const char *);
using AmgxDestroyFn = int (*)(void *);
using AmgxResourcesCreateFn = int (*)(void **, void *);
using AmgxObjectCreateFn = int (*)(void **, void *, int);
using AmgxSolverCreateFn = int (*)(void **, void *, int, void *);
using AmgxMatrixUploadFn = int (*)(void *,
                                   int,
                                   int,
                                   int,
                                   int,
                                   const int *,
                                   const int *,
                                   const void *,
                                   const void *);
using AmgxMatrixReplaceFn =
    int (*)(void *, int, int, const void *, const void *);
using AmgxVectorBindFn = int (*)(void *, const void *);
using AmgxVectorUploadFn = int (*)(void *, int, int, const void *);
using AmgxVectorDownloadFn = int (*)(const void *, void *);
using AmgxVectorSetZeroFn = int (*)(void *, int, int);
using AmgxSolverMatrixFn = int (*)(void *, void *);
using AmgxSolverSolveFn = int (*)(void *, void *, void *);
using AmgxSolverStatusFn = int (*)(void *, int *);
using AmgxResidualFn = int (*)(void *, void *, void *, void *, void *);
using AmgxGetErrorStringFn = int (*)(int, char *, int);

std::mutex amgx_lifecycle_mutex;
uint32_t amgx_runtime_refcount = 0;
std::string amgx_runtime_library_path;

TiForgeRuntimeProviderResult amgx_runtime_fail(Runtime &runtime,
                                               int status,
                                               const char *operation) {
  std::string message = std::string("AmgX ") + operation + " failed";
  auto error_string = reinterpret_cast<AmgxGetErrorStringFn>(
      load_symbol(runtime.library, "AMGX_get_error_string"));
  if (error_string != nullptr) {
    char buffer[4096]{};
    if (error_string(status, buffer, sizeof(buffer)) == 0 &&
        buffer[0] != '\0') {
      message += ": ";
      message += buffer;
    }
  }
  return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_VENDOR_CALL, std::move(message));
}

TiForgeRuntimeProviderResult initialize_execution_runtime(Runtime &runtime) {
  std::lock_guard<std::mutex> lock(amgx_lifecycle_mutex);
  if (runtime.execution_runtime_initialized) {
    return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
  }
  if (runtime.version_major != 1) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_EXECUTION_UNSUPPORTED,
                "AmgX execution adapter requires stable C API major 1");
  }
  if (amgx_runtime_refcount != 0 &&
      amgx_runtime_library_path != runtime.library_path) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_LIFETIME,
                "AmgX is already initialized from a different runtime "
                "library in this process");
  }
  if (amgx_runtime_refcount == 0) {
    auto initialize = reinterpret_cast<AmgxSimpleFn>(
        load_symbol(runtime.library, "AMGX_initialize"));
    const int status = initialize();
    if (status != 0) {
      return amgx_runtime_fail(runtime, status, "initialization");
    }
    amgx_runtime_library_path = runtime.library_path;
  }
  ++amgx_runtime_refcount;
  runtime.execution_runtime_initialized = true;
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}

TiForgeRuntimeProviderResult shutdown_execution_runtime(Runtime &runtime) {
  std::lock_guard<std::mutex> lock(amgx_lifecycle_mutex);
  if (!runtime.execution_runtime_initialized) {
    return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
  }
  if (amgx_runtime_refcount == 0) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INTERNAL,
                "AmgX runtime reference count underflow");
  }
  --amgx_runtime_refcount;
  runtime.execution_runtime_initialized = false;
  if (amgx_runtime_refcount == 0) {
    auto finalize = reinterpret_cast<AmgxSimpleFn>(
        load_symbol(runtime.library, "AMGX_finalize"));
    const int status = finalize();
    amgx_runtime_library_path.clear();
    if (status != 0) {
      return amgx_runtime_fail(runtime, status, "finalization");
    }
  }
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}

struct AmgxSolver {
  Runtime *runtime{nullptr};
  int rows{0};
  int nonzeros{0};
  int value_type{0};
  int mode{0};
  bool setup_valid{false};
  bool compute_residual{true};
  void *config{nullptr};
  void *resources{nullptr};
  void *matrix{nullptr};
  void *rhs{nullptr};
  void *solution{nullptr};
  void *solver{nullptr};
  AmgxDestroyFn destroy_config{nullptr};
  AmgxDestroyFn destroy_resources{nullptr};
  AmgxDestroyFn destroy_matrix{nullptr};
  AmgxDestroyFn destroy_vector{nullptr};
  AmgxDestroyFn destroy_solver{nullptr};
  AmgxMatrixReplaceFn replace{nullptr};
  AmgxSolverMatrixFn setup{nullptr};
  AmgxSolverMatrixFn resetup{nullptr};
  AmgxVectorUploadFn vector_upload{nullptr};
  AmgxVectorDownloadFn vector_download{nullptr};
  AmgxVectorSetZeroFn vector_zero{nullptr};
  AmgxSolverSolveFn solve{nullptr};
  AmgxSolverSolveFn solve_zero{nullptr};
  AmgxSolverStatusFn get_iterations{nullptr};
  AmgxSolverStatusFn get_status{nullptr};
  AmgxResidualFn residual{nullptr};
  std::mutex mutex;
};

void cleanup_amgx_solver(AmgxSolver &solver) {
  if (solver.solver != nullptr && solver.destroy_solver != nullptr) {
    solver.destroy_solver(solver.solver);
    solver.solver = nullptr;
  }
  if (solver.solution != nullptr && solver.destroy_vector != nullptr) {
    solver.destroy_vector(solver.solution);
    solver.solution = nullptr;
  }
  if (solver.rhs != nullptr && solver.destroy_vector != nullptr) {
    solver.destroy_vector(solver.rhs);
    solver.rhs = nullptr;
  }
  if (solver.matrix != nullptr && solver.destroy_matrix != nullptr) {
    solver.destroy_matrix(solver.matrix);
    solver.matrix = nullptr;
  }
  if (solver.resources != nullptr && solver.destroy_resources != nullptr) {
    solver.destroy_resources(solver.resources);
    solver.resources = nullptr;
  }
  if (solver.config != nullptr && solver.destroy_config != nullptr) {
    solver.destroy_config(solver.config);
    solver.config = nullptr;
  }
}

TiForgeRuntimeProviderResult amgx_create_solver(
    TiForgeRuntimeProviderRuntime runtime_value,
    const TiForgeAmgxSolverDesc *desc,
    TiForgeAmgxSolver *out_solver) {
  if (runtime_value == nullptr || desc == nullptr || out_solver == nullptr ||
      desc->struct_size < sizeof(*desc) || desc->rows <= 0 ||
      desc->nonzeros <= 0 || desc->row_offsets == nullptr ||
      desc->column_indices == nullptr || desc->values == nullptr ||
      desc->config == nullptr || desc->config[0] == '\0' ||
      (desc->value_type != TI_FORGE_AMGX_VALUE_F32 &&
       desc->value_type != TI_FORGE_AMGX_VALUE_F64) ||
      desc->config_source > TI_FORGE_AMGX_CONFIG_FILE ||
      (desc->reserved & ~TI_FORGE_AMGX_SOLVER_SKIP_RESIDUAL_NORM) != 0 ||
      desc->row_offsets[0] != 0 ||
      desc->row_offsets[desc->rows] != desc->nonzeros) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "invalid AmgX scalar CSR solver description");
  }
  for (int i = 0; i < desc->rows; ++i) {
    if (desc->row_offsets[i] > desc->row_offsets[i + 1]) {
      return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                  "AmgX CSR row offsets are not monotonic");
    }
  }
  for (int i = 0; i < desc->nonzeros; ++i) {
    if (desc->column_indices[i] < 0 || desc->column_indices[i] >= desc->rows) {
      return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                  "AmgX CSR column index is out of range");
    }
  }
  *out_solver = nullptr;
  auto *runtime = static_cast<Runtime *>(runtime_value);
  if (!runtime->execution_runtime_initialized) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_LIFETIME,
                "AmgX runtime is not initialized for execution");
  }
  auto solver = std::unique_ptr<AmgxSolver>(new (std::nothrow) AmgxSolver());
  if (!solver) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_OUT_OF_MEMORY,
                "AmgX solver allocation failed");
  }
  solver->runtime = runtime;
  solver->rows = desc->rows;
  solver->nonzeros = desc->nonzeros;
  solver->value_type = desc->value_type;
  solver->mode = desc->value_type == TI_FORGE_AMGX_VALUE_F64 ? 8193 : 8465;
  solver->compute_residual =
      (desc->reserved & TI_FORGE_AMGX_SOLVER_SKIP_RESIDUAL_NORM) == 0;
  auto symbol = [&](const char *name) {
    return load_symbol(runtime->library, name);
  };
  auto config_create = reinterpret_cast<AmgxConfigCreateFn>(
      symbol(desc->config_source == TI_FORGE_AMGX_CONFIG_FILE
                 ? "AMGX_config_create_from_file"
                 : "AMGX_config_create"));
  auto resources_create = reinterpret_cast<AmgxResourcesCreateFn>(
      symbol("AMGX_resources_create_simple"));
  auto matrix_create =
      reinterpret_cast<AmgxObjectCreateFn>(symbol("AMGX_matrix_create"));
  auto vector_create =
      reinterpret_cast<AmgxObjectCreateFn>(symbol("AMGX_vector_create"));
  auto solver_create =
      reinterpret_cast<AmgxSolverCreateFn>(symbol("AMGX_solver_create"));
  auto matrix_upload =
      reinterpret_cast<AmgxMatrixUploadFn>(symbol("AMGX_matrix_upload_all"));
  auto vector_bind =
      reinterpret_cast<AmgxVectorBindFn>(symbol("AMGX_vector_bind"));
  solver->setup =
      reinterpret_cast<AmgxSolverMatrixFn>(symbol("AMGX_solver_setup"));
  solver->destroy_config =
      reinterpret_cast<AmgxDestroyFn>(symbol("AMGX_config_destroy"));
  solver->destroy_resources =
      reinterpret_cast<AmgxDestroyFn>(symbol("AMGX_resources_destroy"));
  solver->destroy_matrix =
      reinterpret_cast<AmgxDestroyFn>(symbol("AMGX_matrix_destroy"));
  solver->destroy_vector =
      reinterpret_cast<AmgxDestroyFn>(symbol("AMGX_vector_destroy"));
  solver->destroy_solver =
      reinterpret_cast<AmgxDestroyFn>(symbol("AMGX_solver_destroy"));
  solver->replace = reinterpret_cast<AmgxMatrixReplaceFn>(
      symbol("AMGX_matrix_replace_coefficients"));
  solver->resetup =
      reinterpret_cast<AmgxSolverMatrixFn>(symbol("AMGX_solver_resetup"));
  solver->vector_upload =
      reinterpret_cast<AmgxVectorUploadFn>(symbol("AMGX_vector_upload"));
  solver->vector_download =
      reinterpret_cast<AmgxVectorDownloadFn>(symbol("AMGX_vector_download"));
  solver->vector_zero =
      reinterpret_cast<AmgxVectorSetZeroFn>(symbol("AMGX_vector_set_zero"));
  solver->solve =
      reinterpret_cast<AmgxSolverSolveFn>(symbol("AMGX_solver_solve"));
  solver->solve_zero = reinterpret_cast<AmgxSolverSolveFn>(
      symbol("AMGX_solver_solve_with_0_initial_guess"));
  solver->get_iterations = reinterpret_cast<AmgxSolverStatusFn>(
      symbol("AMGX_solver_get_iterations_number"));
  solver->get_status =
      reinterpret_cast<AmgxSolverStatusFn>(symbol("AMGX_solver_get_status"));
  solver->residual = reinterpret_cast<AmgxResidualFn>(
      symbol("AMGX_solver_calculate_residual_norm"));
  auto checked = [&](int status, const char *operation) {
    return status == 0 ? TI_FORGE_RUNTIME_PROVIDER_SUCCESS
                       : amgx_runtime_fail(*runtime, status, operation);
  };
  if (checked(config_create(&solver->config, desc->config),
              "config creation") != TI_FORGE_RUNTIME_PROVIDER_SUCCESS ||
      checked(resources_create(&solver->resources, solver->config),
              "resource creation") != TI_FORGE_RUNTIME_PROVIDER_SUCCESS ||
      checked(matrix_create(&solver->matrix, solver->resources, solver->mode),
              "matrix creation") != TI_FORGE_RUNTIME_PROVIDER_SUCCESS ||
      checked(vector_create(&solver->rhs, solver->resources, solver->mode),
              "RHS vector creation") != TI_FORGE_RUNTIME_PROVIDER_SUCCESS ||
      checked(vector_create(&solver->solution, solver->resources, solver->mode),
              "solution vector creation") !=
          TI_FORGE_RUNTIME_PROVIDER_SUCCESS ||
      checked(solver_create(&solver->solver, solver->resources, solver->mode,
                            solver->config),
              "solver creation") != TI_FORGE_RUNTIME_PROVIDER_SUCCESS ||
      checked(matrix_upload(solver->matrix, desc->rows, desc->nonzeros, 1, 1,
                            desc->row_offsets, desc->column_indices,
                            desc->values, nullptr),
              "matrix upload") != TI_FORGE_RUNTIME_PROVIDER_SUCCESS ||
      checked(vector_bind(solver->rhs, solver->matrix), "RHS binding") !=
          TI_FORGE_RUNTIME_PROVIDER_SUCCESS ||
      checked(vector_bind(solver->solution, solver->matrix),
              "solution binding") != TI_FORGE_RUNTIME_PROVIDER_SUCCESS ||
      checked(solver->setup(solver->solver, solver->matrix), "solver setup") !=
          TI_FORGE_RUNTIME_PROVIDER_SUCCESS) {
    cleanup_amgx_solver(*solver);
    return TI_FORGE_RUNTIME_PROVIDER_ERROR_VENDOR_CALL;
  }
  runtime->live_execution_resources.fetch_add(1);
  solver->setup_valid = true;
  *out_solver = solver.release();
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}

TiForgeRuntimeProviderResult amgx_replace_coefficients(
    TiForgeAmgxSolver solver_value,
    const void *values,
    int32_t nonzeros) {
  auto *solver = static_cast<AmgxSolver *>(solver_value);
  if (solver == nullptr || values == nullptr || nonzeros != solver->nonzeros) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "invalid AmgX replacement coefficients");
  }
  std::lock_guard<std::mutex> lock(solver->mutex);
  solver->setup_valid = false;
  int status =
      solver->replace(solver->matrix, solver->rows, nonzeros, values, nullptr);
  if (status != 0) {
    return amgx_runtime_fail(*solver->runtime, status,
                             "coefficient replacement");
  }
  const bool use_resetup = solver->resetup != nullptr;
  status = (use_resetup ? solver->resetup : solver->setup)(solver->solver,
                                                           solver->matrix);
  if (status != 0) {
    return amgx_runtime_fail(*solver->runtime, status,
                             use_resetup ? "solver resetup"
                                         : "solver setup fallback");
  }
  solver->setup_valid = true;
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}

TiForgeRuntimeProviderResult amgx_solve(TiForgeAmgxSolver solver_value,
                                        const TiForgeAmgxSolveDesc *desc,
                                        TiForgeAmgxSolveInfo *out_info) {
  auto *solver = static_cast<AmgxSolver *>(solver_value);
  if (solver == nullptr || desc == nullptr || out_info == nullptr ||
      desc->struct_size < sizeof(*desc) ||
      out_info->struct_size < sizeof(*out_info) || desc->rhs == nullptr ||
      desc->solution == nullptr || desc->zero_initial_guess > 1) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "invalid AmgX solve buffers");
  }
  std::lock_guard<std::mutex> lock(solver->mutex);
  if (!solver->setup_valid) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_LIFETIME,
                "AmgX solver setup is invalid after coefficient update "
                "failure");
  }
  auto checked = [&](int status, const char *operation) {
    return status == 0 ? TI_FORGE_RUNTIME_PROVIDER_SUCCESS
                       : amgx_runtime_fail(*solver->runtime, status, operation);
  };
  if (checked(solver->vector_upload(solver->rhs, solver->rows, 1, desc->rhs),
              "RHS upload") != TI_FORGE_RUNTIME_PROVIDER_SUCCESS) {
    return TI_FORGE_RUNTIME_PROVIDER_ERROR_VENDOR_CALL;
  }
  int status = 0;
  if (desc->zero_initial_guess != 0) {
    // xIsZero allows the solver to skip forming the initial residual; it does
    // not guarantee that PCG's first axpy initializes the solution vector.
    status = solver->vector_zero(solver->solution, solver->rows, 1);
    if (status == 0) {
      status =
          solver->solve_zero(solver->solver, solver->rhs, solver->solution);
    }
  } else {
    status = solver->vector_upload(solver->solution, solver->rows, 1,
                                   desc->solution);
    if (status == 0) {
      status = solver->solve(solver->solver, solver->rhs, solver->solution);
    }
  }
  if (status != 0) {
    return amgx_runtime_fail(*solver->runtime, status, "solve");
  }
  int iterations = 0;
  int solve_status = 1;
  float residual_norm_f32 = std::numeric_limits<float>::quiet_NaN();
  double residual_norm_f64 = std::numeric_limits<double>::quiet_NaN();
  void *residual_norm = solver->value_type == TI_FORGE_AMGX_VALUE_F32
                            ? static_cast<void *>(&residual_norm_f32)
                            : static_cast<void *>(&residual_norm_f64);
  if (checked(solver->get_iterations(solver->solver, &iterations),
              "iteration query") != TI_FORGE_RUNTIME_PROVIDER_SUCCESS ||
      checked(solver->get_status(solver->solver, &solve_status),
              "status query") != TI_FORGE_RUNTIME_PROVIDER_SUCCESS ||
      (solver->compute_residual &&
       checked(solver->residual(solver->solver, solver->matrix, solver->rhs,
                               solver->solution, residual_norm),
               "residual query") != TI_FORGE_RUNTIME_PROVIDER_SUCCESS) ||
      checked(solver->vector_download(solver->solution, desc->solution),
              "solution download") != TI_FORGE_RUNTIME_PROVIDER_SUCCESS) {
    return TI_FORGE_RUNTIME_PROVIDER_ERROR_VENDOR_CALL;
  }
  const double reported_residual = solver->value_type == TI_FORGE_AMGX_VALUE_F32
                                       ? static_cast<double>(residual_norm_f32)
                                       : residual_norm_f64;
  *out_info = {sizeof(*out_info), static_cast<uint32_t>(solve_status),
               iterations,
               solver->compute_residual
                   ? 0u
                   : static_cast<uint32_t>(TI_FORGE_AMGX_RESIDUAL_NOT_COMPUTED),
               reported_residual};
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}

TiForgeRuntimeProviderResult amgx_destroy_solver(
    TiForgeAmgxSolver solver_value) {
  auto *solver = static_cast<AmgxSolver *>(solver_value);
  if (solver == nullptr) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "AmgX solver is null");
  }
  Runtime *runtime = solver->runtime;
  cleanup_amgx_solver(*solver);
  delete solver;
  runtime->live_execution_resources.fetch_sub(1);
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}

TiForgeRuntimeProviderResult query_execution_api(
    TiForgeRuntimeProviderRuntime runtime,
    uint32_t requested_execution_abi_version,
    size_t api_size,
    void *out_api) {
  if (runtime == nullptr || out_api == nullptr ||
      api_size < sizeof(TiForgeAmgxExecutionApi)) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "AmgX execution API output is null or truncated");
  }
  if (requested_execution_abi_version !=
      TI_FORGE_RUNTIME_PROVIDER_EXECUTION_ABI_VERSION) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH,
                "unsupported AmgX execution ABI");
  }
  auto *api = static_cast<TiForgeAmgxExecutionApi *>(out_api);
  *api = {sizeof(*api),       TI_FORGE_RUNTIME_PROVIDER_EXECUTION_ABI_VERSION,
          amgx_create_solver, amgx_replace_coefficients,
          amgx_solve,         amgx_destroy_solver};
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}

#endif

size_t get_last_error(char *destination, size_t destination_size) {
  const std::size_t required = last_error.size() + 1;
  if (destination != nullptr && destination_size > 0) {
    const std::size_t count = std::min(last_error.size(), destination_size - 1);
    std::memcpy(destination, last_error.data(), count);
    destination[count] = '\0';
  }
  return required;
}

}  // namespace

extern "C" TI_FORGE_RUNTIME_PROVIDER_EXPORT TiForgeRuntimeProviderResult
TI_FORGE_RUNTIME_PROVIDER_QUERY_FUNCTION(uint32_t requested_abi_version,
                                         size_t api_size,
                                         TiForgeRuntimeProviderApi *out_api) {
  if (out_api == nullptr || api_size < sizeof(TiForgeRuntimeProviderApi)) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "provider API output is null or truncated");
  }
  if (requested_abi_version != TI_FORGE_RUNTIME_PROVIDER_ABI_VERSION) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH,
                "unsupported Forge runtime-provider ABI");
  }
  std::memset(out_api, 0, sizeof(*out_api));
  out_api->struct_size = sizeof(*out_api);
  out_api->provider_abi_version = TI_FORGE_RUNTIME_PROVIDER_ABI_VERSION;
  out_api->info = {sizeof(TiForgeRuntimeProviderInfo),
                   TI_FORGE_RUNTIME_PROVIDER_ABI_VERSION,
                   kFeatures,
                   kRequiredSymbolCount,
                   0,
                   kProviderId,
                   kProviderName,
                   kSupportedVersionFamily,
                   kBuildIdentity};
  out_api->probe_runtime = probe_runtime;
  out_api->create_runtime = create_runtime;
  out_api->destroy_runtime = destroy_runtime;
  out_api->query_execution_api = query_execution_api;
  out_api->get_last_error = get_last_error;
  last_error.clear();
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}
