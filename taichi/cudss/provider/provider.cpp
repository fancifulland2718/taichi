#include "taichi/cudss/forge_cudss_provider.h"

#if defined(_WIN32) && !defined(NOMINMAX)
#define NOMINMAX
#endif

#include <algorithm>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include <cudss.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#if CUDSS_VERSION_MAJOR != 0 || CUDSS_VERSION_MINOR != 8
#error "Forge cuDSS provider requires cuDSS 0.8.x headers"
#endif

namespace {

thread_local std::string last_error;
thread_local std::string probed_library_path;

constexpr char kProviderName[] = "taichi-forge-cudss";
constexpr char kBuildIdentity[] =
    "forge-cudss-provider-abi1-cudss-0.8-configured-plan-factor-statistics-v1";
constexpr uint64_t kFeatures = TI_FORGE_CUDSS_FEATURE_CSR |
                               TI_FORGE_CUDSS_FEATURE_DENSE_VECTOR |
                               TI_FORGE_CUDSS_FEATURE_STAGED_EXECUTION |
                               TI_FORGE_CUDSS_FEATURE_VALUE_REBIND |
                               TI_FORGE_CUDSS_FEATURE_EXPLICIT_STREAM;

TiForgeCudssResult fail(TiForgeCudssResult result, std::string message) {
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
  int version_major{-1};
  int version_minor{-1};
  int version_patch{-1};

  decltype(&cudssGetProperty) get_property{nullptr};
  decltype(&cudssCreate) create{nullptr};
  decltype(&cudssDestroy) destroy{nullptr};
  decltype(&cudssSetStream) set_stream{nullptr};
  decltype(&cudssConfigCreate) config_create{nullptr};
  decltype(&cudssConfigDestroy) config_destroy{nullptr};
  decltype(&cudssConfigSet) config_set{nullptr};
  decltype(&cudssConfigGet) config_get{nullptr};
  decltype(&cudssDataGet) data_get{nullptr};
  decltype(&cudssSetDeviceMemHandler) set_mem_handler{nullptr};
  decltype(&cudssDataCreate) data_create{nullptr};
  decltype(&cudssDataDestroy) data_destroy{nullptr};
  decltype(&cudssMatrixCreateCsr) matrix_create_csr{nullptr};
  decltype(&cudssMatrixCreateDn) matrix_create_dn{nullptr};
  decltype(&cudssMatrixDestroy) matrix_destroy{nullptr};
  decltype(&cudssMatrixSetValues) matrix_set_values{nullptr};
  decltype(&cudssMatrixSetCsrPointers) matrix_set_csr_pointers{nullptr};
  decltype(&cudssExecute) execute{nullptr};

  ~Runtime() {
    close_library(library);
  }
};

template <typename T>
bool bind_symbol(Runtime &runtime, T &destination, const char *name) {
  destination = reinterpret_cast<T>(load_symbol(runtime.library, name));
  if (destination == nullptr) {
    last_error =
        std::string("cuDSS runtime is missing required symbol ") + name;
    return false;
  }
  return true;
}

void fill_runtime_info(const Runtime &runtime, TiForgeCudssRuntimeInfo *info) {
  if (info == nullptr) {
    return;
  }
  info->struct_size = sizeof(*info);
  info->version_major = static_cast<uint32_t>(runtime.version_major);
  info->version_minor = static_cast<uint32_t>(runtime.version_minor);
  info->version_patch = static_cast<uint32_t>(runtime.version_patch);
  info->library_path = runtime.library_path.c_str();
}

TiForgeCudssResult make_runtime(const char *library_path,
                                Runtime **out_runtime) {
  if (out_runtime == nullptr) {
    return fail(TI_FORGE_CUDSS_ERROR_INVALID_ARGUMENT,
                "cuDSS runtime output is null");
  }
  *out_runtime = nullptr;
  last_error.clear();
  std::vector<std::string> candidates;
  if (library_path != nullptr && library_path[0] != '\0') {
    candidates.emplace_back(library_path);
  } else {
#if defined(_WIN32)
    candidates.emplace_back("cudss64_0.dll");
#else
    candidates.emplace_back("libcudss.so.0");
    candidates.emplace_back("libcudss.so");
#endif
  }
  std::string attempted;
  for (const auto &candidate : candidates) {
    auto runtime = std::unique_ptr<Runtime>(new (std::nothrow) Runtime());
    if (!runtime) {
      return fail(TI_FORGE_CUDSS_ERROR_OUT_OF_MEMORY,
                  "cuDSS adapter runtime allocation failed");
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
#define BIND_CUDSS(member, symbol)                                \
  if (!bind_symbol(*runtime, runtime->member, "cudss" #symbol)) { \
    return TI_FORGE_CUDSS_ERROR_RUNTIME_INCOMPATIBLE;             \
  }
    BIND_CUDSS(get_property, GetProperty);
    BIND_CUDSS(create, Create);
    BIND_CUDSS(destroy, Destroy);
    BIND_CUDSS(set_stream, SetStream);
    BIND_CUDSS(config_create, ConfigCreate);
    BIND_CUDSS(config_destroy, ConfigDestroy);
    // Optional for the original execution contract; configured plans check
    // these once during preparation, never in solve/refactor replay.
    runtime->config_set = reinterpret_cast<decltype(runtime->config_set)>(
        load_symbol(runtime->library, "cudssConfigSet"));
    runtime->config_get = reinterpret_cast<decltype(runtime->config_get)>(
        load_symbol(runtime->library, "cudssConfigGet"));
    runtime->data_get = reinterpret_cast<decltype(runtime->data_get)>(
        load_symbol(runtime->library, "cudssDataGet"));
    runtime->set_mem_handler =
        reinterpret_cast<decltype(runtime->set_mem_handler)>(
            load_symbol(runtime->library, "cudssSetDeviceMemHandler"));
    BIND_CUDSS(data_create, DataCreate);
    BIND_CUDSS(data_destroy, DataDestroy);
    BIND_CUDSS(matrix_create_csr, MatrixCreateCsr);
    BIND_CUDSS(matrix_create_dn, MatrixCreateDn);
    BIND_CUDSS(matrix_destroy, MatrixDestroy);
    BIND_CUDSS(matrix_set_values, MatrixSetValues);
    BIND_CUDSS(matrix_set_csr_pointers, MatrixSetCsrPointers);
    BIND_CUDSS(execute, Execute);
#undef BIND_CUDSS
    if (runtime->get_property(MAJOR_VERSION, &runtime->version_major) !=
            CUDSS_STATUS_SUCCESS ||
        runtime->get_property(MINOR_VERSION, &runtime->version_minor) !=
            CUDSS_STATUS_SUCCESS ||
        runtime->get_property(PATCH_LEVEL, &runtime->version_patch) !=
            CUDSS_STATUS_SUCCESS) {
      return fail(TI_FORGE_CUDSS_ERROR_RUNTIME_INCOMPATIBLE,
                  "cuDSS runtime version query failed");
    }
    if (runtime->version_major != 0 || runtime->version_minor != 8) {
      return fail(TI_FORGE_CUDSS_ERROR_RUNTIME_INCOMPATIBLE,
                  "cuDSS adapter 0.8 requires a cuDSS 0.8.x runtime");
    }
    *out_runtime = runtime.release();
    return TI_FORGE_CUDSS_SUCCESS;
  }
  return fail(
      TI_FORGE_CUDSS_ERROR_RUNTIME_UNAVAILABLE,
      "cuDSS vendor runtime could not be loaded; attempted: " + attempted);
}

Runtime *checked(TiForgeCudssRuntime runtime) {
  return static_cast<Runtime *>(runtime);
}

TiForgeCudssResult probe_runtime(const char *library_path,
                                 TiForgeCudssRuntimeInfo *out_info) {
  if (out_info != nullptr && out_info->struct_size < sizeof(*out_info)) {
    return fail(TI_FORGE_CUDSS_ERROR_INVALID_ARGUMENT,
                "cuDSS runtime info output is truncated");
  }
  Runtime *runtime = nullptr;
  const auto result = make_runtime(library_path, &runtime);
  if (result != TI_FORGE_CUDSS_SUCCESS) {
    return result;
  }
  probed_library_path = runtime->library_path;
  fill_runtime_info(*runtime, out_info);
  if (out_info != nullptr) {
    out_info->library_path = probed_library_path.c_str();
  }
  delete runtime;
  return TI_FORGE_CUDSS_SUCCESS;
}

TiForgeCudssResult create_runtime(const char *library_path,
                                  TiForgeCudssRuntime *out_runtime,
                                  TiForgeCudssRuntimeInfo *out_info) {
  if (out_runtime == nullptr ||
      (out_info != nullptr && out_info->struct_size < sizeof(*out_info))) {
    return fail(TI_FORGE_CUDSS_ERROR_INVALID_ARGUMENT,
                "cuDSS runtime output is null or truncated");
  }
  *out_runtime = nullptr;
  Runtime *runtime = nullptr;
  const auto result = make_runtime(library_path, &runtime);
  if (result != TI_FORGE_CUDSS_SUCCESS) {
    return result;
  }
  fill_runtime_info(*runtime, out_info);
  *out_runtime = runtime;
  return TI_FORGE_CUDSS_SUCCESS;
}

TiForgeCudssResult destroy_runtime(TiForgeCudssRuntime runtime) {
  if (runtime == nullptr) {
    return fail(TI_FORGE_CUDSS_ERROR_INVALID_ARGUMENT,
                "cuDSS adapter runtime is null");
  }
  delete checked(runtime);
  return TI_FORGE_CUDSS_SUCCESS;
}

uint32_t create(TiForgeCudssRuntime runtime, void **handle) {
  return static_cast<uint32_t>(
      checked(runtime)->create(reinterpret_cast<cudssHandle_t *>(handle)));
}

uint32_t destroy(TiForgeCudssRuntime runtime, void *handle) {
  return static_cast<uint32_t>(
      checked(runtime)->destroy(static_cast<cudssHandle_t>(handle)));
}

uint32_t set_stream(TiForgeCudssRuntime runtime, void *handle, void *stream) {
  return static_cast<uint32_t>(checked(runtime)->set_stream(
      static_cast<cudssHandle_t>(handle), static_cast<cudaStream_t>(stream)));
}

uint32_t config_create(TiForgeCudssRuntime runtime, void **config) {
  return static_cast<uint32_t>(checked(runtime)->config_create(
      reinterpret_cast<cudssConfig_t *>(config)));
}

uint32_t config_destroy(TiForgeCudssRuntime runtime, void *config) {
  return static_cast<uint32_t>(
      checked(runtime)->config_destroy(static_cast<cudssConfig_t>(config)));
}

uint32_t data_create(TiForgeCudssRuntime runtime,
                     const void *handle,
                     void **data) {
  return static_cast<uint32_t>(checked(runtime)->data_create(
      static_cast<cudssHandle_t>(const_cast<void *>(handle)),
      reinterpret_cast<cudssData_t *>(data)));
}

uint32_t data_destroy(TiForgeCudssRuntime runtime, void *handle, void *data) {
  return static_cast<uint32_t>(checked(runtime)->data_destroy(
      static_cast<cudssHandle_t>(handle), static_cast<cudssData_t>(data)));
}

uint32_t matrix_create_csr(TiForgeCudssRuntime runtime,
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
                           int index_base) {
  return static_cast<uint32_t>(checked(runtime)->matrix_create_csr(
      reinterpret_cast<cudssMatrix_t *>(matrix), rows, columns, nonzeros,
      row_start, row_end, column_indices, values,
      static_cast<cudssDataType_t>(offset_type),
      static_cast<cudssDataType_t>(index_type),
      static_cast<cudssDataType_t>(value_type),
      static_cast<cudssMatrixType_t>(matrix_type),
      static_cast<cudssMatrixViewType_t>(matrix_view),
      static_cast<cudssIndexBase_t>(index_base)));
}

uint32_t matrix_create_dn(TiForgeCudssRuntime runtime,
                          void **matrix,
                          int64_t rows,
                          int64_t columns,
                          int64_t leading_dimension,
                          const void *values,
                          int value_type,
                          int layout) {
  return static_cast<uint32_t>(checked(runtime)->matrix_create_dn(
      reinterpret_cast<cudssMatrix_t *>(matrix), rows, columns,
      leading_dimension, const_cast<void *>(values),
      static_cast<cudssDataType_t>(value_type),
      static_cast<cudssLayout_t>(layout)));
}

uint32_t matrix_destroy(TiForgeCudssRuntime runtime, void *matrix) {
  return static_cast<uint32_t>(
      checked(runtime)->matrix_destroy(static_cast<cudssMatrix_t>(matrix)));
}

uint32_t matrix_set_values(TiForgeCudssRuntime runtime,
                           void *matrix,
                           const void *values) {
  return static_cast<uint32_t>(checked(runtime)->matrix_set_values(
      static_cast<cudssMatrix_t>(matrix), const_cast<void *>(values)));
}

uint32_t matrix_set_csr_pointers(TiForgeCudssRuntime runtime,
                                 void *matrix,
                                 const void *row_start,
                                 const void *row_end,
                                 const void *column_indices,
                                 const void *values) {
  return static_cast<uint32_t>(checked(runtime)->matrix_set_csr_pointers(
      static_cast<cudssMatrix_t>(matrix), const_cast<void *>(row_start),
      const_cast<void *>(row_end), const_cast<void *>(column_indices),
      const_cast<void *>(values)));
}

uint32_t execute(TiForgeCudssRuntime runtime,
                 void *handle,
                 int phase,
                 const void *config,
                 void *data,
                 const void *matrix,
                 void *solution,
                 const void *rhs) {
  return static_cast<uint32_t>(checked(runtime)->execute(
      static_cast<cudssHandle_t>(handle), phase,
      static_cast<cudssConfig_t>(const_cast<void *>(config)),
      static_cast<cudssData_t>(data),
      static_cast<cudssMatrix_t>(const_cast<void *>(matrix)),
      static_cast<cudssMatrix_t>(solution),
      static_cast<cudssMatrix_t>(const_cast<void *>(rhs))));
}

template <typename T>
uint32_t configure_exact(Runtime &runtime,
                         cudssConfig_t config,
                         cudssConfigParam_t parameter,
                         T value) {
  auto status = runtime.config_set(config, parameter, &value, sizeof(value));
  if (status != CUDSS_STATUS_SUCCESS) {
    return status;
  }
  T observed{};
  size_t written = 0;
  status = runtime.config_get(config, parameter, &observed, sizeof(observed),
                              &written);
  if (status != CUDSS_STATUS_SUCCESS) {
    return status;
  }
  return written == sizeof(observed) && observed == value
             ? CUDSS_STATUS_SUCCESS
             : CUDSS_STATUS_INTERNAL_ERROR;
}

uint32_t configure(TiForgeCudssRuntime runtime,
                   void *config,
                   int reordering,
                   int solve) {
  auto *owner = checked(runtime);
  if (!owner || !config || !owner->config_set || !owner->config_get) {
    return CUDSS_STATUS_NOT_SUPPORTED;
  }
  const cudssReorderingAlg_t reorderings[] = {
      CUDSS_REORDERING_ALG_DEFAULT, CUDSS_REORDERING_ALG_AMD,
      CUDSS_REORDERING_ALG_NESTED_DISSECTION, CUDSS_REORDERING_ALG_NONE};
  if (reordering < 0 || reordering >= 4 || solve < 0 || solve > 1) {
    return CUDSS_STATUS_INVALID_VALUE;
  }
  auto status =
      configure_exact(*owner, static_cast<cudssConfig_t>(config),
                      CUDSS_CONFIG_REORDERING_ALG, reorderings[reordering]);
  if (status != CUDSS_STATUS_SUCCESS) {
    return status;
  }
  return configure_exact(
      *owner, static_cast<cudssConfig_t>(config), CUDSS_CONFIG_SOLVE_ALG,
      solve == 0 ? CUDSS_SOLVE_ALG_DEFAULT : CUDSS_SOLVE_ALG_GENERAL);
}

uint32_t analysis_memory_estimates(TiForgeCudssRuntime runtime,
                                   void *handle,
                                   void *data,
                                   int64_t *estimates,
                                   size_t capacity_bytes,
                                   size_t *written_bytes) {
  auto *owner = checked(runtime);
  if (!owner || !owner->data_get) {
    return CUDSS_STATUS_NOT_SUPPORTED;
  }
  if (!handle || !data || !estimates || !written_bytes ||
      capacity_bytes < 16 * sizeof(int64_t)) {
    return CUDSS_STATUS_INVALID_VALUE;
  }
  return owner->data_get(
      static_cast<cudssHandle_t>(handle), static_cast<cudssData_t>(data),
      CUDSS_DATA_MEMORY_ESTIMATES, estimates, capacity_bytes, written_bytes);
}

template <typename T>
TiForgeCudssFactorStatistic read_factor_statistic(Runtime &runtime,
                                                  void *handle,
                                                  void *data,
                                                  cudssDataParam_t parameter) {
  T value{-1};
  size_t written = 0;
  const auto status = runtime.data_get(
      static_cast<cudssHandle_t>(handle), static_cast<cudssData_t>(data),
      parameter, &value, sizeof(value), &written);
  return {status == CUDSS_STATUS_SUCCESS && written == sizeof(value)
              ? static_cast<int64_t>(value)
              : -1,
          static_cast<uint32_t>(status), static_cast<uint32_t>(written)};
}

TiForgeCudssResult read_factor_statistics_i32(
    TiForgeCudssRuntime runtime,
    void *handle,
    void *data,
    TiForgeCudssFactorStatistics *out_statistics) {
  auto *owner = checked(runtime);
  if (!owner || !handle || !data || !out_statistics) {
    return TI_FORGE_CUDSS_ERROR_INVALID_ARGUMENT;
  }
  if (!owner->data_get) {
    return TI_FORGE_CUDSS_ERROR_RUNTIME_INCOMPATIBLE;
  }
  *out_statistics = {
      read_factor_statistic<int64_t>(*owner, handle, data, CUDSS_DATA_LU_NNZ),
      read_factor_statistic<int32_t>(*owner, handle, data,
                                     CUDSS_DATA_NSUPERPANELS),
      read_factor_statistic<int64_t>(*owner, handle, data, CUDSS_DATA_FLOPS)};
  return TI_FORGE_CUDSS_SUCCESS;
}

size_t get_last_error(char *destination, size_t destination_size) {
  const std::size_t required = last_error.size() + 1;
  if (destination != nullptr && destination_size > 0) {
    const std::size_t count = std::min(last_error.size(), destination_size - 1);
    std::memcpy(destination, last_error.data(), count);
    destination[count] = '\0';
  }
  return required;
}

uint32_t set_allocator(TiForgeCudssRuntime runtime,
                       void *handle,
                       void *owner,
                       int (*allocate)(void *, void **, size_t, void *),
                       int (*deallocate)(void *, void *, size_t, void *)) {
  auto *instance = checked(runtime);
  if (!instance || !instance->set_mem_handler) {
    return CUDSS_STATUS_NOT_SUPPORTED;
  }
  if (!handle || !owner || !allocate || !deallocate) {
    return CUDSS_STATUS_INVALID_VALUE;
  }
  // cuDSS copies this handler; callback context belongs to the retained native
  // owner. cudaStream_t is an opaque pointer, not a Toolkit runtime dependency.
  cudssDeviceMemHandler_t handler{};
  handler.ctx = owner;
  handler.device_alloc =
      reinterpret_cast<decltype(handler.device_alloc)>(allocate);
  handler.device_free =
      reinterpret_cast<decltype(handler.device_free)>(deallocate);
  std::strncpy(handler.name, "forge-driver-async-retained",
               sizeof(handler.name) - 1);
  return instance->set_mem_handler(static_cast<cudssHandle_t>(handle),
                                   &handler);
}

}  // namespace

extern "C" TI_FORGE_CUDSS_EXPORT TiForgeCudssResult
taichi_forge_cudss_provider_query(uint32_t requested_abi_version,
                                  size_t api_size,
                                  TiForgeCudssProviderApi *out_api) {
  if (out_api == nullptr || api_size < sizeof(TiForgeCudssProviderApi)) {
    return fail(TI_FORGE_CUDSS_ERROR_INVALID_ARGUMENT,
                "cuDSS provider API output is null or truncated");
  }
  if (requested_abi_version != TI_FORGE_CUDSS_PROVIDER_ABI_VERSION) {
    return fail(TI_FORGE_CUDSS_ERROR_ABI_MISMATCH,
                "unsupported Forge cuDSS provider ABI");
  }
  std::memset(out_api, 0, sizeof(*out_api));
  out_api->struct_size = sizeof(*out_api);
  out_api->provider_abi_version = TI_FORGE_CUDSS_PROVIDER_ABI_VERSION;
  out_api->info = {sizeof(TiForgeCudssProviderInfo),
                   TI_FORGE_CUDSS_PROVIDER_ABI_VERSION,
                   CUDSS_VERSION,
                   0,
                   kFeatures,
                   kProviderName,
                   kBuildIdentity};
  out_api->probe_runtime = probe_runtime;
  out_api->create_runtime = create_runtime;
  out_api->destroy_runtime = destroy_runtime;
  out_api->create = create;
  out_api->destroy = destroy;
  out_api->set_stream = set_stream;
  out_api->config_create = config_create;
  out_api->config_destroy = config_destroy;
  out_api->data_create = data_create;
  out_api->data_destroy = data_destroy;
  out_api->matrix_create_csr = matrix_create_csr;
  out_api->matrix_create_dn = matrix_create_dn;
  out_api->matrix_destroy = matrix_destroy;
  out_api->matrix_set_values = matrix_set_values;
  out_api->matrix_set_csr_pointers = matrix_set_csr_pointers;
  out_api->execute = execute;
  out_api->get_last_error = get_last_error;
  last_error.clear();
  return TI_FORGE_CUDSS_SUCCESS;
}

extern "C" TI_FORGE_CUDSS_EXPORT TiForgeCudssResult
taichi_forge_cudss_configuration_query(uint32_t requested_abi_version,
                                       size_t api_size,
                                       TiForgeCudssConfigurationApi *out_api) {
  if (!out_api || api_size < sizeof(*out_api)) {
    return fail(TI_FORGE_CUDSS_ERROR_INVALID_ARGUMENT,
                "cuDSS configuration API output is null or truncated");
  }
  if (requested_abi_version != TI_FORGE_CUDSS_CONFIGURATION_ABI_VERSION) {
    return fail(TI_FORGE_CUDSS_ERROR_ABI_MISMATCH,
                "unsupported cuDSS configuration ABI");
  }
  *out_api = {sizeof(*out_api), TI_FORGE_CUDSS_CONFIGURATION_ABI_VERSION,
              configure, analysis_memory_estimates};
  return TI_FORGE_CUDSS_SUCCESS;
}

extern "C" TI_FORGE_CUDSS_EXPORT TiForgeCudssResult
taichi_forge_cudss_allocator_query(uint32_t requested_abi_version,
                                   size_t api_size,
                                   TiForgeCudssAllocatorApi *out_api) {
  if (!out_api || api_size < sizeof(*out_api)) {
    return TI_FORGE_CUDSS_ERROR_INVALID_ARGUMENT;
  }
  if (requested_abi_version != 1) {
    return TI_FORGE_CUDSS_ERROR_ABI_MISMATCH;
  }
  *out_api = {sizeof(*out_api), 1, set_allocator};
  return TI_FORGE_CUDSS_SUCCESS;
}

extern "C" TI_FORGE_CUDSS_EXPORT TiForgeCudssResult
taichi_forge_cudss_factor_statistics_query(
    uint32_t requested_abi_version,
    size_t api_size,
    TiForgeCudssFactorStatisticsApi *out_api) {
  if (!out_api || api_size < sizeof(*out_api)) {
    return TI_FORGE_CUDSS_ERROR_INVALID_ARGUMENT;
  }
  if (requested_abi_version != 1) {
    return TI_FORGE_CUDSS_ERROR_ABI_MISMATCH;
  }
  *out_api = {sizeof(*out_api), 1, read_factor_statistics_i32};
  return TI_FORGE_CUDSS_SUCCESS;
}
