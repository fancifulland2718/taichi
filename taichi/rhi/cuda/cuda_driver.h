#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "taichi/common/dynamic_loader.h"
#include "taichi/common/core.h"
#include "taichi/rhi/backend_error.h"
#include "taichi/rhi/common/runtime_telemetry.h"
#include "taichi/rhi/cuda/cuda_types.h"

#if (0)
// Turn on to check for compatibility
namespace taichi {
static_assert(sizeof(CUresult) == sizeof(uint32));
static_assert(sizeof(CUmem_advise) == sizeof(uint32));
static_assert(sizeof(CUdevice) == sizeof(uint32));
static_assert(sizeof(CUdevice_attribute) == sizeof(uint32));
static_assert(sizeof(CUfunction) == sizeof(void *));
static_assert(sizeof(CUmodule) == sizeof(void *));
static_assert(sizeof(CUstream) == sizeof(void *));
static_assert(sizeof(CUevent) == sizeof(void *));
static_assert(sizeof(CUjit_option) == sizeof(uint32));
}  // namespace taichi
#endif

namespace taichi::lang {

// Driver constants from cuda.h

constexpr uint32 CU_EVENT_DEFAULT = 0x0;
constexpr uint32 CU_EVENT_DISABLE_TIMING = 0x2;
constexpr uint32 CU_STREAM_DEFAULT = 0x0;
constexpr uint32 CU_STREAM_NON_BLOCKING = 0x1;
constexpr uint32 CU_MEM_ATTACH_GLOBAL = 0x1;
constexpr uint32 CU_MEM_ADVISE_SET_PREFERRED_LOCATION = 3;
constexpr uint32 CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2;
constexpr uint32 CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8;
constexpr uint32 CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10;
constexpr uint32 CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97;
constexpr uint32 CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106;
constexpr uint32 CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16;
constexpr uint32 CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
constexpr uint32 CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115;
constexpr uint32 CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;
constexpr uint32 CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = 4;
constexpr uint32 CUDA_ERROR_ILLEGAL_ADDRESS = 700;
constexpr uint32 CUDA_ERROR_LAUNCH_TIMEOUT = 702;
constexpr uint32 CUDA_ERROR_ASSERT = 710;
constexpr uint32 CUDA_ERROR_HARDWARE_STACK_ERROR = 714;
constexpr uint32 CUDA_ERROR_ILLEGAL_INSTRUCTION = 715;
constexpr uint32 CUDA_ERROR_MISALIGNED_ADDRESS = 716;
constexpr uint32 CUDA_ERROR_INVALID_ADDRESS_SPACE = 717;
constexpr uint32 CUDA_ERROR_INVALID_PC = 718;
constexpr uint32 CUDA_ERROR_LAUNCH_FAILED = 719;
constexpr uint32 CUDA_ERROR_NOT_SUPPORTED = 801;
constexpr uint32 CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900;
constexpr uint32 CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901;
constexpr uint32 CUDA_ERROR_NOT_READY = 600;
constexpr uint32 CU_JIT_MAX_REGISTERS = 0;
constexpr uint32 CU_JIT_WALL_TIME = 2;
constexpr uint32 CU_JIT_INFO_LOG_BUFFER = 3;
constexpr uint32 CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4;
constexpr uint32 CU_JIT_ERROR_LOG_BUFFER = 5;
constexpr uint32 CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6;
constexpr uint32 CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2;
constexpr uint32 CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41;
constexpr uint32 CUDA_SUCCESS = 0;
constexpr uint32 CU_MEMORYTYPE_DEVICE = 2;
constexpr uint32 CU_LIMIT_STACK_SIZE = 0;
constexpr uint32 CU_TRSF_NORMALIZED_COORDINATES = 0x02;

std::string get_cuda_error_message(uint32 err);

using CUDASampledLockTelemetry = SampledLockTelemetry<std::mutex>;
using CUDASampledRecursiveLockTelemetry =
    SampledLockTelemetry<std::recursive_mutex>;

enum class CUDADriverProvider : uint8_t {
  nvidia_cuda,
  musa,
};

namespace cuda::detail {

inline const char *driver_provider_name(CUDADriverProvider provider) {
  return provider == CUDADriverProvider::musa ? "musa" : "nvidia_cuda";
}

inline std::string driver_symbol_name(CUDADriverProvider provider,
                                      const std::string &cuda_symbol) {
  if (provider == CUDADriverProvider::musa &&
      cuda_symbol.rfind("cu", 0) == 0) {
    return "mu" + cuda_symbol.substr(2);
  }
  return cuda_symbol;
}

inline bool driver_symbol_enabled(CUDADriverProvider provider,
                                  const std::string &cuda_symbol) {
  if (provider != CUDADriverProvider::musa) {
    return true;
  }
  return cuda_symbol.rfind("cuGraph", 0) != 0 &&
         cuda_symbol.rfind("cuStreamBeginCapture", 0) != 0 &&
         cuda_symbol != "cuStreamEndCapture" &&
         cuda_symbol != "cuLaunchKernelEx" &&
         cuda_symbol.rfind("cuImportExternal", 0) != 0 &&
         cuda_symbol.rfind("cuDestroyExternal", 0) != 0 &&
         cuda_symbol.rfind("cuExternalMemory", 0) != 0 &&
         cuda_symbol.rfind("cuMipmappedArray", 0) != 0 &&
         cuda_symbol.rfind("cuArray", 0) != 0 &&
         cuda_symbol.rfind("cuTexObject", 0) != 0 &&
         cuda_symbol.rfind("cuMemcpy3D", 0) != 0 &&
         cuda_symbol.rfind("cuSurfObject", 0) != 0 &&
         cuda_symbol.rfind("cuSignalExternal", 0) != 0 &&
         cuda_symbol.rfind("cuWaitExternal", 0) != 0;
}

inline bool driver_version_supported(CUDADriverProvider provider,
                                     int version) {
  return provider == CUDADriverProvider::musa ? version > 0 : version >= 10000;
}

inline bool driver_warp_size_supported(CUDADriverProvider provider,
                                       int warp_size) {
  return provider != CUDADriverProvider::musa || warp_size == 32;
}

}  // namespace cuda::detail

struct CUDADriverTelemetrySnapshot {
  CUDASampledLockTelemetry::Snapshot lock;
  BackendWaitTelemetry::Snapshot wait;
  uint64_t async_allocation_calls;
  uint64_t sync_allocation_fallback_calls;
  uint64_t async_free_calls;
  uint64_t sync_free_fallback_calls;
  uint64_t jit_module_load_calls;
  uint64_t jit_ptx_bytes;
  uint64_t jit_host_wall_ns;
  uint64_t jit_driver_wall_us;
  uint64_t jit_diagnostic_loads;
  uint64_t jit_info_log_bytes;
  uint64_t jit_error_log_bytes;
};

template <typename... Args>
class CUDADriverFunction {
 public:
  CUDADriverFunction() {
    function_ = nullptr;
  }

  void set(void *func_ptr) {
    function_ = (func_type *)func_ptr;
  }

  bool available() const {
    return function_ != nullptr;
  }

  uint32 call(Args... args) {
    TI_ASSERT(function_ != nullptr);
    TI_ASSERT(driver_lock_ != nullptr);
    auto lock = driver_lock_telemetry_
                    ? driver_lock_telemetry_->acquire(*driver_lock_)
                    : std::unique_lock<std::mutex>(*driver_lock_);
    if (wait_telemetry_ == nullptr) {
      return (uint32)function_(args...);
    }
    ScopedBackendWaitTelemetry wait_scope(wait_telemetry_);
    return (uint32)function_(args...);
  }

  void set_names(const std::string &name, const std::string &symbol_name) {
    name_ = name;
    symbol_name_ = symbol_name;
  }

  void set_lock(std::mutex *lock) {
    driver_lock_ = lock;
  }

  void set_lock_telemetry(CUDASampledLockTelemetry *telemetry) {
    driver_lock_telemetry_ = telemetry;
  }

  void set_wait_telemetry(BackendWaitTelemetry *telemetry) {
    wait_telemetry_ = telemetry;
  }

  void set_fault_reporter_slot(
      std::shared_ptr<BackendFaultReporter> *fault_reporter_slot) {
    fault_reporter_slot_ = fault_reporter_slot;
  }

  std::string get_error_message(uint32 err) {
    return get_cuda_error_message(err) +
           fmt::format(" while calling {} ({})", name_, symbol_name_);
  }

  uint32 call_with_warning(Args... args) {
    auto err = call(args...);
    TI_WARN_IF(err, "{}", get_error_message(err));
    return err;
  }

  // Note: CUDA driver API passes everything as value
  void operator()(Args... args) {
    auto err = call(args...);
    if (!err) {
      return;
    }
    if (fault_reporter_slot_ == nullptr) {
      TI_ERROR("{}", get_error_message(err));
    }
    BackendRuntimeError error(Arch::cuda, err, name_, get_error_message(err));
    auto reporter = std::atomic_load_explicit(fault_reporter_slot_,
                                              std::memory_order_acquire);
    if (reporter) {
      reporter->report_backend_error(error, 0);
    }
    throw error;
  }

 private:
  using func_type = uint32_t(Args...);

  func_type *function_{nullptr};
  std::string name_, symbol_name_;
  std::mutex *driver_lock_{nullptr};
  CUDASampledLockTelemetry *driver_lock_telemetry_{nullptr};
  BackendWaitTelemetry *wait_telemetry_{nullptr};
  std::shared_ptr<BackendFaultReporter> *fault_reporter_slot_{nullptr};
};

class CUDADriverBase {
 public:
  ~CUDADriverBase() = default;

 protected:
  std::unique_ptr<DynamicLoader> loader_;
  std::string loaded_library_name_;
  CUDADriverBase();

  bool load_lib(std::string lib_linux, std::string lib_windows);

  bool check_lib_loaded(std::string lib_linux, std::string lib_windows);

  bool try_load_lib_any_version(const std::string &lib_name,
                                const std::string &win_arch_name,
                                const std::vector<int> &versions_to_try);

  bool disabled_by_env_{false};
};

class CUDADriver : protected CUDADriverBase {
 public:
#define PER_CUDA_FUNCTION(name, symbol_name, ...) \
  CUDADriverFunction<__VA_ARGS__> name;
#include "taichi/rhi/cuda/cuda_driver_functions.inc.h"
#undef PER_CUDA_FUNCTION

#define PER_CUDA_OPTIONAL_FUNCTION(name, symbol_name, ...) \
  CUDADriverFunction<__VA_ARGS__> name;
#include "taichi/rhi/cuda/cuda_optional_driver_functions.inc.h"
#undef PER_CUDA_OPTIONAL_FUNCTION

  void (*get_error_name)(uint32, const char **);

  void (*get_error_string)(uint32, const char **);

  void (*driver_get_version)(int *);

  void malloc_async(void **ptr, size_t size, CUstream stream);
  void malloc_async_from_pool(void **ptr,
                              size_t size,
                              void *pool,
                              CUstream stream);

  void mem_free_async(void *ptr, CUstream stream);

  bool detected();

  static CUDADriver &get_instance();

  static CUDADriver &get_instance_without_context();

  void set_fault_reporter(
      std::shared_ptr<BackendFaultReporter> reporter) noexcept;
  void clear_fault_reporter(
      const std::shared_ptr<BackendFaultReporter> &reporter) noexcept;

  int get_version_major() {
    return version_major_;
  }

  int get_version_minor() {
    return version_minor_;
  }

  CUDADriverProvider get_provider() const {
    return provider_;
  }

  bool is_musa() const {
    return provider_ == CUDADriverProvider::musa;
  }

  bool nvidia_extensions_available() {
    return detected() && !is_musa();
  }

  void record_jit_module_load(uint64_t ptx_bytes,
                              uint64_t host_wall_ns,
                              uint64_t driver_wall_us,
                              bool diagnostics_enabled,
                              uint64_t info_log_bytes,
                              uint64_t error_log_bytes) noexcept {
    jit_module_load_calls_.fetch_add(1, std::memory_order_relaxed);
    jit_ptx_bytes_.fetch_add(ptx_bytes, std::memory_order_relaxed);
    jit_host_wall_ns_.fetch_add(host_wall_ns, std::memory_order_relaxed);
    jit_driver_wall_us_.fetch_add(driver_wall_us, std::memory_order_relaxed);
    if (diagnostics_enabled) {
      jit_diagnostic_loads_.fetch_add(1, std::memory_order_relaxed);
    }
    jit_info_log_bytes_.fetch_add(info_log_bytes, std::memory_order_relaxed);
    jit_error_log_bytes_.fetch_add(error_log_bytes, std::memory_order_relaxed);
  }

  CUDADriverTelemetrySnapshot get_telemetry_snapshot() const {
    return {lock_telemetry_.snapshot(),
            wait_telemetry_.snapshot(),
            async_allocation_calls_.load(std::memory_order_relaxed),
            sync_allocation_fallback_calls_.load(std::memory_order_relaxed),
            async_free_calls_.load(std::memory_order_relaxed),
            sync_free_fallback_calls_.load(std::memory_order_relaxed),
            jit_module_load_calls_.load(std::memory_order_relaxed),
            jit_ptx_bytes_.load(std::memory_order_relaxed),
            jit_host_wall_ns_.load(std::memory_order_relaxed),
            jit_driver_wall_us_.load(std::memory_order_relaxed),
            jit_diagnostic_loads_.load(std::memory_order_relaxed),
            jit_info_log_bytes_.load(std::memory_order_relaxed),
            jit_error_log_bytes_.load(std::memory_order_relaxed)};
  }

 private:
  CUDADriver();

  std::mutex lock_;
  CUDASampledLockTelemetry lock_telemetry_;
  BackendWaitTelemetry wait_telemetry_;
  std::shared_ptr<BackendFaultReporter> fault_reporter_;
  std::atomic<uint64_t> async_allocation_calls_{0};
  std::atomic<uint64_t> sync_allocation_fallback_calls_{0};
  std::atomic<uint64_t> async_free_calls_{0};
  std::atomic<uint64_t> sync_free_fallback_calls_{0};
  std::atomic<uint64_t> jit_module_load_calls_{0};
  std::atomic<uint64_t> jit_ptx_bytes_{0};
  std::atomic<uint64_t> jit_host_wall_ns_{0};
  std::atomic<uint64_t> jit_driver_wall_us_{0};
  std::atomic<uint64_t> jit_diagnostic_loads_{0};
  std::atomic<uint64_t> jit_info_log_bytes_{0};
  std::atomic<uint64_t> jit_error_log_bytes_{0};

  bool cuda_version_valid_{false};
  CUDADriverProvider provider_{CUDADriverProvider::nvidia_cuda};

  int version_major_{0};
  int version_minor_{0};
};

struct CUSPARSEProviderCapabilities {
  int library_version_major{-1};
  int library_version_minor{-1};
  int library_version_patch{-1};
  bool bsr_descriptor_available{false};
  bool generic_bsr_spmv_available{false};
  bool spmv_preprocess_available{false};
  bool scalar_spmv_available{false};
  bool spmm_f32_available{false};
  bool spmm_preprocess_available{false};
  bool spsv_f32_available{false};
  bool spsm_f32_available{false};
  bool spsv_value_update_available{false};
  bool spsm_value_update_available{false};
  bool triangular_value_update_available{false};
};

class CUSPARSEDriver : protected CUDADriverBase {
 public:
  static CUSPARSEDriver &get_instance();

#define PER_CUSPARSE_FUNCTION(name, symbol_name, ...) \
  CUDADriverFunction<__VA_ARGS__> name;
#include "taichi/rhi/cuda/cusparse_functions.inc.h"
#undef PER_CUSPARSE_FUNCTION

  // BSR is an optional generic API capability. Keeping this out of the
  // mandatory function table preserves compatibility with older providers.
  CUDADriverFunction<cusparseSpMatDescr_t *,
                     int64_t,
                     int64_t,
                     int64_t,
                     int64_t,
                     int64_t,
                     void *,
                     void *,
                     void *,
                     cusparseIndexType_t,
                     cusparseIndexType_t,
                     cusparseIndexBase_t,
                     cudaDataType,
                     cusparseOrder_t>
      cpCreateBsr;
  CUDADriverFunction<cusparseHandle_t,
                     cusparseOperation_t,
                     const void *,
                     cusparseSpMatDescr_t,
                     cusparseDnVecDescr_t,
                     const void *,
                     cusparseDnVecDescr_t,
                     cudaDataType,
                     cusparseSpMVAlg_t,
                     void *>
      cpSpMVPreprocess;
  CUDADriverFunction<cusparseDnMatDescr_t *,
                     int64_t,
                     int64_t,
                     int64_t,
                     void *,
                     cudaDataType,
                     cusparseOrder_t>
      cpCreateDnMat;
  CUDADriverFunction<cusparseConstDnMatDescr_t> cpDestroyDnMat;
  CUDADriverFunction<cusparseDnMatDescr_t, void *> cpDnMatSetValues;
  CUDADriverFunction<cusparseDnVecDescr_t, void *> cpDnVecSetValues;
  CUDADriverFunction<cusparseSpMatDescr_t,
                     cusparseSpMatAttribute_t,
                     const void *,
                     size_t>
      cpSpMatSetAttribute;
  CUDADriverFunction<cusparseHandle_t,
                     cusparseOperation_t,
                     cusparseOperation_t,
                     const void *,
                     cusparseSpMatDescr_t,
                     cusparseConstDnMatDescr_t,
                     const void *,
                     cusparseDnMatDescr_t,
                     cudaDataType,
                     cusparseSpMMAlg_t,
                     size_t *>
      cpSpMMBufferSize;
  CUDADriverFunction<cusparseHandle_t,
                     cusparseOperation_t,
                     cusparseOperation_t,
                     const void *,
                     cusparseSpMatDescr_t,
                     cusparseConstDnMatDescr_t,
                     const void *,
                     cusparseDnMatDescr_t,
                     cudaDataType,
                     cusparseSpMMAlg_t,
                     void *>
      cpSpMMPreprocess;
  CUDADriverFunction<cusparseHandle_t,
                     cusparseOperation_t,
                     cusparseOperation_t,
                     const void *,
                     cusparseSpMatDescr_t,
                     cusparseConstDnMatDescr_t,
                     const void *,
                     cusparseDnMatDescr_t,
                     cudaDataType,
                     cusparseSpMMAlg_t,
                     void *>
      cpSpMM;
  CUDADriverFunction<cusparseSpSVDescr_t *> cpSpSVCreateDescr;
  CUDADriverFunction<cusparseSpSVDescr_t> cpSpSVDestroyDescr;
  CUDADriverFunction<cusparseHandle_t,
                     cusparseOperation_t,
                     const void *,
                     cusparseSpMatDescr_t,
                     cusparseDnVecDescr_t,
                     cusparseDnVecDescr_t,
                     cudaDataType,
                     cusparseSpSVAlg_t,
                     cusparseSpSVDescr_t,
                     size_t *>
      cpSpSVBufferSize;
  CUDADriverFunction<cusparseHandle_t,
                     cusparseOperation_t,
                     const void *,
                     cusparseSpMatDescr_t,
                     cusparseDnVecDescr_t,
                     cusparseDnVecDescr_t,
                     cudaDataType,
                     cusparseSpSVAlg_t,
                     cusparseSpSVDescr_t,
                     void *>
      cpSpSVAnalysis;
  CUDADriverFunction<cusparseHandle_t,
                     cusparseOperation_t,
                     const void *,
                     cusparseSpMatDescr_t,
                     cusparseDnVecDescr_t,
                     cusparseDnVecDescr_t,
                     cudaDataType,
                     cusparseSpSVAlg_t,
                     cusparseSpSVDescr_t>
      cpSpSVSolve;
  CUDADriverFunction<cusparseHandle_t,
                     cusparseSpSVDescr_t,
                     void *,
                     cusparseSpSVUpdate_t>
      cpSpSVUpdateMatrix;
  CUDADriverFunction<cusparseSpSMDescr_t *> cpSpSMCreateDescr;
  CUDADriverFunction<cusparseSpSMDescr_t> cpSpSMDestroyDescr;
  CUDADriverFunction<cusparseHandle_t,
                     cusparseOperation_t,
                     cusparseOperation_t,
                     const void *,
                     cusparseSpMatDescr_t,
                     cusparseDnMatDescr_t,
                     cusparseDnMatDescr_t,
                     cudaDataType,
                     cusparseSpSMAlg_t,
                     cusparseSpSMDescr_t,
                     size_t *>
      cpSpSMBufferSize;
  CUDADriverFunction<cusparseHandle_t,
                     cusparseOperation_t,
                     cusparseOperation_t,
                     const void *,
                     cusparseSpMatDescr_t,
                     cusparseDnMatDescr_t,
                     cusparseDnMatDescr_t,
                     cudaDataType,
                     cusparseSpSMAlg_t,
                     cusparseSpSMDescr_t,
                     void *>
      cpSpSMAnalysis;
  CUDADriverFunction<cusparseHandle_t,
                     cusparseOperation_t,
                     cusparseOperation_t,
                     const void *,
                     cusparseSpMatDescr_t,
                     cusparseDnMatDescr_t,
                     cusparseDnMatDescr_t,
                     cudaDataType,
                     cusparseSpSMAlg_t,
                     cusparseSpSMDescr_t>
      cpSpSMSolve;
  CUDADriverFunction<cusparseHandle_t,
                     cusparseSpSMDescr_t,
                     void *,
                     cusparseSpSMUpdate_t>
      cpSpSMUpdateMatrix;

  bool load_cusparse();

  CUSPARSEProviderCapabilities capabilities() const {
    return capabilities_;
  }

  inline bool is_loaded() {
    return cusparse_loaded_;
  }

  const std::string &loaded_library_name() const {
    return loaded_library_name_;
  }

 private:
  CUSPARSEDriver();
  std::mutex lock_;
  bool cusparse_loaded_{false};
  CUDADriverFunction<int, int *> cp_get_property_;
  CUSPARSEProviderCapabilities capabilities_;
};

class CUSOLVERDriver : protected CUDADriverBase {
 public:
  // TODO: Add cusolver function APIs
  static CUSOLVERDriver &get_instance();

#define PER_CUSOLVER_FUNCTION(name, symbol_name, ...) \
  CUDADriverFunction<__VA_ARGS__> name;
#include "taichi/rhi/cuda/cusolver_functions.inc.h"
#undef PER_CUSOLVER_FUNCTION

  bool load_cusolver();

  inline bool is_loaded() {
    return cusolver_loaded_;
  }

 private:
  CUSOLVERDriver();
  std::mutex lock_;
  bool cusolver_loaded_{false};
};

struct CUBLASProviderCapabilities {
  int library_version_major{-1};
  int library_version_minor{-1};
  int library_version_patch{-1};
  bool gemm_f32_available{false};
};

class CUBLASDriver : protected CUDADriverBase {
 public:
  static CUBLASDriver &get_instance();

  CUDADriverFunction<cublasHandle_t, void *, size_t> cubSetWorkspace;

#define PER_CUBLAS_FUNCTION(name, symbol_name, ...) \
  CUDADriverFunction<__VA_ARGS__> name;
#include "taichi/rhi/cuda/cublas_functions.inc.h"
#undef PER_CUBLAS_FUNCTION

  bool load_cublas();

  inline bool is_loaded() {
    return cublas_loaded_;
  }

  CUBLASProviderCapabilities capabilities() const {
    return capabilities_;
  }

  const std::string &loaded_library_name() const {
    return loaded_library_name_;
  }

 private:
  CUBLASDriver();
  std::mutex lock_;
  bool cublas_loaded_{false};
  CUDADriverFunction<int, int *> cub_get_property_;
  CUBLASProviderCapabilities capabilities_;
};

struct CUFFTProviderCapabilities {
  int library_version{0};
};

class CUFFTDriver : protected CUDADriverBase {
 public:
  static CUFFTDriver &get_instance();

#define PER_CUFFT_FUNCTION(name, symbol_name, ...) \
  CUDADriverFunction<__VA_ARGS__> name;
#include "taichi/rhi/cuda/cufft_functions.inc.h"
#undef PER_CUFFT_FUNCTION

  // Optional extensible-plan/LTO symbols. Basic cuFFT remains usable without
  // these symbols or a runtime compiler/linker installation.
  CUDADriverFunction<int *> create;
  CUDADriverFunction<int, int, int *, int *, int, int, int *, int, int,
                     int, int, std::size_t *> make_plan_many;
  CUDADriverFunction<int, const char *, const void *, std::size_t, int,
                     void **> set_jit_callback;

  bool load_cufft();

  CUFFTProviderCapabilities capabilities() const {
    return capabilities_;
  }

  inline bool is_loaded() {
    return cufft_loaded_.load(std::memory_order_acquire);
  }

  const std::string &loaded_library_name() const {
    return loaded_library_name_;
  }

 private:
  CUFFTDriver();
  std::mutex load_lock_;
  std::mutex lock_;
  std::atomic<bool> cufft_loaded_{false};
  CUFFTProviderCapabilities capabilities_;
};

}  // namespace taichi::lang
