#include "taichi/rhi/cuda/cuda_driver.h"

#include <algorithm>
#include <cstdlib>
#include <tuple>

#include "taichi/common/dynamic_loader.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/util/environ_config.h"

namespace taichi::lang {

std::string get_cuda_error_message(uint32 err) {
  const char *err_name_ptr;
  const char *err_string_ptr;
  auto &driver = CUDADriver::get_instance_without_context();
  driver.get_error_name(err, &err_name_ptr);
  driver.get_error_string(err, &err_string_ptr);
  return fmt::format("{} driver error {}: {}",
                     cuda::detail::driver_provider_name(driver.get_provider()),
                     err_name_ptr, err_string_ptr);
}

CUDADriverBase::CUDADriverBase() {
  disabled_by_env_ = (get_environ_config("TI_ENABLE_CUDA", 1) == 0);
  if (disabled_by_env_) {
    TI_TRACE("CUDA driver disabled by enviroment variable \"TI_ENABLE_CUDA\".");
  }
}

static void maybe_set_cuda_lazy_loading(int driver_version) {
  // P-Sparse-Mem-2-VRAM (2026-05-04, refined 2026-05-05):
  // CUDA_MODULE_LOADING=LAZY defers cublas/cusparse/cufft/etc. symbol mapping
  // until first reference, saving 200-500 MiB of VRAM that the Taichi LLVM
  // backend never uses.
  //
  // Driver-version timeline:
  //   * < 11070  : env var unsupported, driver default EAGER. setenv is a
  //                no-op (driver ignores unknown env vars), so harmless.
  //   * 11070..12009 : env var honored, driver default EAGER. setenv saves
  //                    real VRAM here. THIS IS THE TARGET WINDOW.
  //   * >= 12010 : driver default already LAZY. setenv is a no-op.
  //
  // We only setenv when the user hasn't chosen a value, so explicit `EAGER`
  // overrides for debugging remain honored. We also avoid touching env in the
  // >=12010 case to keep environment minimal on modern stacks.
  if (std::getenv("CUDA_MODULE_LOADING") != nullptr) {
    return;
  }
  if (driver_version < 11070) {
    TI_TRACE(
        "CUDA driver v{}.{} predates CUDA_MODULE_LOADING (introduced in "
        "11.7); skipping LAZY override.",
        driver_version / 1000, driver_version % 1000 / 10);
    return;
  }
  if (driver_version >= 12010) {
    TI_TRACE(
        "CUDA driver v{}.{} already defaults to LAZY module loading; "
        "skipping explicit override.",
        driver_version / 1000, driver_version % 1000 / 10);
    return;
  }
#if defined(TI_PLATFORM_WINDOWS)
  _putenv_s("CUDA_MODULE_LOADING", "LAZY");
#else
  setenv("CUDA_MODULE_LOADING", "LAZY", 1);
#endif
  TI_TRACE(
      "Set CUDA_MODULE_LOADING=LAZY for driver v{}.{} (saves VRAM by "
      "deferring unused device-library mapping).",
      driver_version / 1000, driver_version % 1000 / 10);
}

bool CUDADriverBase::load_lib(std::string lib_linux, std::string lib_windows) {
#if defined(TI_PLATFORM_LINUX)
  auto lib_name = lib_linux;
#elif defined(TI_PLATFORM_WINDOWS)
  auto lib_name = lib_windows;
#else
  static_assert(false, "Taichi CUDA driver supports only Windows and Linux.");
#endif

  loader_ = std::make_unique<DynamicLoader>(lib_name);
  if (!loader_->loaded()) {
    TI_WARN("{} lib not found.", lib_name);
    return false;
  } else {
    loaded_library_name_ = lib_name;
    TI_TRACE("{} loaded!", lib_name);
    return true;
  }
}

bool CUDADriverBase::check_lib_loaded(std::string lib_linux,
                                      std::string lib_windows) {
#if defined(TI_PLATFORM_LINUX)
  auto lib_name = lib_linux;
#elif defined(TI_PLATFORM_WINDOWS)
  auto lib_name = lib_windows;
#else
  static_assert(false, "Taichi CUDA driver supports only Windows and Linux.");
#endif

  return DynamicLoader::check_lib_loaded(lib_name);
}

std::string get_lib_name_linux(const std::string &lib_name, int version) {
  return "lib" + lib_name + ".so." + std::to_string(version);
}

std::string get_lib_name_windows(const std::string &lib_name,
                                 const std::string &win_arch_name,
                                 int version) {
  return lib_name + win_arch_name + std::to_string(version) + ".dll";
}

bool CUDADriverBase::try_load_lib_any_version(
    const std::string &lib_name,
    const std::string &win_arch_name,
    const std::vector<int> &versions_to_try) {
  // Check if any versions of this lib are already loaded.
  for (auto version : versions_to_try) {
    std::string lib_name_linux = get_lib_name_linux(lib_name, version);
    std::string lib_name_windows =
        get_lib_name_windows(lib_name, win_arch_name, version);
    if (check_lib_loaded(lib_name_linux, lib_name_windows)) {
      load_lib(lib_name_linux, lib_name_windows);
      return true;
    }
  }

  // Try load any version of this lib if none of them are loaded.
  bool loaded = false;
  if (!loaded) {
#ifdef WIN32
    for (auto version : versions_to_try) {
      std::string lib_name_windows =
          get_lib_name_windows(lib_name, win_arch_name, version);
      loader_ = std::make_unique<DynamicLoader>(lib_name_windows);
      loaded = loader_->loaded();
      if (loaded) {
        loaded_library_name_ = lib_name_windows;
        break;
      }
    }
#else
    for (auto version : versions_to_try) {
      std::string lib_name_linux = get_lib_name_linux(lib_name, version);
      loader_ = std::make_unique<DynamicLoader>(lib_name_linux);
      loaded = loader_->loaded();
      if (loaded) {
        loaded_library_name_ = lib_name_linux;
        break;
      }
    }
    if (!loaded) {
      // Use the default version on linux.
      std::string lib_name_linux = "lib" + lib_name + ".so";
      loader_ = std::make_unique<DynamicLoader>(lib_name_linux);
      loaded = loader_->loaded();
      if (loaded) {
        loaded_library_name_ = lib_name_linux;
      }
    }
#endif
  }
  return loaded;
}

bool CUDADriver::detected() {
  return !disabled_by_env_ && cuda_version_valid_ && loader_ &&
         loader_->loaded();
}

CUDADriver::CUDADriver() {
  enum class ProviderPreference {
    automatic,
    nvidia_cuda,
    musa,
  };

  ProviderPreference preference = ProviderPreference::automatic;
  if (const char *configured = std::getenv("TI_CUDA_DRIVER_PROVIDER")) {
    const std::string value(configured);
    if (value == "cuda" || value == "nvidia" || value == "nvidia_cuda") {
      preference = ProviderPreference::nvidia_cuda;
    } else if (value == "musa") {
      preference = ProviderPreference::musa;
    } else if (value != "auto") {
      TI_WARN(
          "Ignoring unsupported TI_CUDA_DRIVER_PROVIDER='{}'. Expected auto, "
          "cuda, nvidia, nvidia_cuda, or musa.",
          value);
    }
  }

  auto try_load_provider = [&](const char *library,
                               CUDADriverProvider provider) {
    auto candidate = std::make_unique<DynamicLoader>(library);
    if (!candidate->loaded()) {
      return false;
    }
    loader_ = std::move(candidate);
    provider_ = provider;
    TI_TRACE("{} loaded as the {} driver provider.", library,
             cuda::detail::driver_provider_name(provider_));
    return true;
  };

  bool loaded = false;
#if defined(TI_PLATFORM_LINUX)
  if (preference != ProviderPreference::musa) {
    loaded = try_load_provider("libcuda.so",
                               CUDADriverProvider::nvidia_cuda);
  }
  if (!loaded && preference != ProviderPreference::nvidia_cuda) {
    loaded = try_load_provider("libmusa.so.1", CUDADriverProvider::musa) ||
             try_load_provider("libmusa.so", CUDADriverProvider::musa);
  }
#elif defined(TI_PLATFORM_WINDOWS)
  if (preference != ProviderPreference::musa) {
    loaded =
        try_load_provider("nvcuda.dll", CUDADriverProvider::nvidia_cuda);
  }
  // Windows uses the ordinary DLL search path for these SDK-provided names.
  // Require an explicit selection so an unrelated musa.dll next to the
  // application can never become an automatic compute provider.
  if (!loaded && preference == ProviderPreference::musa) {
    loaded = try_load_provider("musa.dll", CUDADriverProvider::musa) ||
             try_load_provider("musa_driver.dll", CUDADriverProvider::musa);
  }
#else
  static_assert(false, "Taichi CUDA driver supports only Windows and Linux.");
#endif
  if (!loaded) {
    TI_WARN(
        "No compatible CUDA driver library was found (NVIDIA CUDA or MUSA).");
    return;
  }

  auto load_symbol = [&](const char *cuda_symbol, bool required_for_nvidia) {
    if (!cuda::detail::driver_symbol_enabled(provider_, cuda_symbol)) {
      return static_cast<void *>(nullptr);
    }
    const auto symbol =
        cuda::detail::driver_symbol_name(provider_, cuda_symbol);
    if (provider_ == CUDADriverProvider::nvidia_cuda &&
        required_for_nvidia) {
      return loader_->load_function(symbol);
    }
    return loader_->load_function_optional(symbol);
  };

  const auto error_name_symbol =
      cuda::detail::driver_symbol_name(provider_, "cuGetErrorName");
  const auto error_string_symbol =
      cuda::detail::driver_symbol_name(provider_, "cuGetErrorString");
  const auto version_symbol =
      cuda::detail::driver_symbol_name(provider_, "cuDriverGetVersion");
  get_error_name =
      (decltype(get_error_name))load_symbol("cuGetErrorName", true);
  get_error_string =
      (decltype(get_error_string))load_symbol("cuGetErrorString", true);
  driver_get_version =
      (decltype(driver_get_version))load_symbol("cuDriverGetVersion", true);
  if (!get_error_name || !get_error_string || !driver_get_version) {
    TI_WARN(
        "The {} provider is missing bootstrap Driver API symbols ({}, {}, "
        "{}).",
        cuda::detail::driver_provider_name(provider_), error_name_symbol,
        error_string_symbol, version_symbol);
    return;
  }

  int version = 0;
  driver_get_version(&version);
  TI_TRACE("{} driver API (v{}.{}) loaded.",
           cuda::detail::driver_provider_name(provider_), version / 1000,
           version % 1000 / 10);

  if (provider_ == CUDADriverProvider::nvidia_cuda) {
    // Set CUDA_MODULE_LOADING=LAZY based on driver version, before any other
    // cu* call. driver_get_version itself does not require cuInit.
    maybe_set_cuda_lazy_loading(version);
  }

  if (!cuda::detail::driver_version_supported(provider_, version)) {
    TI_WARN("Unsupported {} driver API version v{}.{}.",
            cuda::detail::driver_provider_name(provider_), version / 1000,
            version % 1000 / 10);
    return;
  }

  version_major_ = version / 1000;
  version_minor_ = version % 1000 / 10;

#define PER_CUDA_FUNCTION(name, symbol_name, ...) \
  name.set(load_symbol(#symbol_name, true));       \
  name.set_lock(&lock_);                          \
  name.set_lock_telemetry(&lock_telemetry_);      \
  name.set_fault_reporter_slot(&fault_reporter_); \
  name.set_names(                                  \
      #name, cuda::detail::driver_symbol_name(provider_, #symbol_name));
#include "taichi/rhi/cuda/cuda_driver_functions.inc.h"
#undef PER_CUDA_FUNCTION

#define PER_CUDA_OPTIONAL_FUNCTION(name, symbol_name, ...) \
  name.set(load_symbol(#symbol_name, false));               \
  name.set_lock(&lock_);                                   \
  name.set_lock_telemetry(&lock_telemetry_);               \
  name.set_fault_reporter_slot(&fault_reporter_);          \
  name.set_names(                                           \
      #name, cuda::detail::driver_symbol_name(provider_, #symbol_name));
#include "taichi/rhi/cuda/cuda_optional_driver_functions.inc.h"
#undef PER_CUDA_OPTIONAL_FUNCTION

  if (provider_ == CUDADriverProvider::musa) {
    std::string missing;
    auto require = [&](bool available, const char *cuda_symbol) {
      if (available) {
        return;
      }
      if (!missing.empty()) {
        missing += ", ";
      }
      missing += cuda::detail::driver_symbol_name(provider_, cuda_symbol);
    };
#define REQUIRE_MUSA_FUNCTION(name, symbol_name) \
  require(name.available(), #symbol_name)
    REQUIRE_MUSA_FUNCTION(init, cuInit);
    REQUIRE_MUSA_FUNCTION(device_get_count, cuDeviceGetCount);
    REQUIRE_MUSA_FUNCTION(device_get, cuDeviceGet);
    REQUIRE_MUSA_FUNCTION(device_get_name, cuDeviceGetName);
    REQUIRE_MUSA_FUNCTION(device_get_attribute, cuDeviceGetAttribute);
    REQUIRE_MUSA_FUNCTION(context_set_current, cuCtxSetCurrent);
    REQUIRE_MUSA_FUNCTION(context_get_current, cuCtxGetCurrent);
    REQUIRE_MUSA_FUNCTION(primary_context_retain, cuDevicePrimaryCtxRetain);
    REQUIRE_MUSA_FUNCTION(context_set_limit, cuCtxSetLimit);
    REQUIRE_MUSA_FUNCTION(memcpy_host_to_device, cuMemcpyHtoD_v2);
    REQUIRE_MUSA_FUNCTION(memcpy_device_to_host, cuMemcpyDtoH_v2);
    REQUIRE_MUSA_FUNCTION(memcpy_device_to_device, cuMemcpyDtoD_v2);
    REQUIRE_MUSA_FUNCTION(memcpy_host_to_device_async, cuMemcpyHtoDAsync_v2);
    REQUIRE_MUSA_FUNCTION(memcpy_device_to_host_async, cuMemcpyDtoHAsync_v2);
    REQUIRE_MUSA_FUNCTION(malloc, cuMemAlloc_v2);
    REQUIRE_MUSA_FUNCTION(malloc_managed, cuMemAllocManaged);
    REQUIRE_MUSA_FUNCTION(memset, cuMemsetD8_v2);
    REQUIRE_MUSA_FUNCTION(memsetd32, cuMemsetD32_v2);
    REQUIRE_MUSA_FUNCTION(mem_free, cuMemFree_v2);
    REQUIRE_MUSA_FUNCTION(mem_get_info, cuMemGetInfo_v2);
    REQUIRE_MUSA_FUNCTION(module_get_function, cuModuleGetFunction);
    REQUIRE_MUSA_FUNCTION(module_load_data_ex, cuModuleLoadDataEx);
    REQUIRE_MUSA_FUNCTION(module_unload, cuModuleUnload);
    REQUIRE_MUSA_FUNCTION(launch_kernel, cuLaunchKernel);
    REQUIRE_MUSA_FUNCTION(stream_synchronize, cuStreamSynchronize);
#undef REQUIRE_MUSA_FUNCTION
    if (!missing.empty()) {
      TI_WARN(
          "The MUSA driver is missing Driver API symbols required by the "
          "basic Taichi CUDA execution path: {}.",
          missing);
      return;
    }
  }

  cuda_version_valid_ = true;

  // Only APIs that can block on device progress contribute backend wait time.
  // The timer starts after acquiring the driver host lock, keeping host-lock
  // contention and device waiting as separate metrics.
  stream_synchronize.set_wait_telemetry(&wait_telemetry_);
  event_synchronize.set_wait_telemetry(&wait_telemetry_);
}

// This is for initializing the CUDA driver itself
CUDADriver &CUDADriver::get_instance_without_context() {
  // Thread safety guaranteed by C++ compiler
  // Note this is never deleted until the process finishes
  static CUDADriver *instance = new CUDADriver();
  return *instance;
}

CUDADriver &CUDADriver::get_instance() {
  // initialize the CUDA context so that the driver APIs can be called later
  CUDAContext::get_instance();
  return get_instance_without_context();
}

void CUDADriver::set_fault_reporter(
    std::shared_ptr<BackendFaultReporter> reporter) noexcept {
  std::atomic_store_explicit(&fault_reporter_, std::move(reporter),
                             std::memory_order_release);
}

void CUDADriver::clear_fault_reporter(
    const std::shared_ptr<BackendFaultReporter> &reporter) noexcept {
  auto current =
      std::atomic_load_explicit(&fault_reporter_, std::memory_order_acquire);
  if (current == reporter) {
    std::atomic_store_explicit(&fault_reporter_,
                               std::shared_ptr<BackendFaultReporter>{},
                               std::memory_order_release);
  }
}

void CUDADriver::malloc_async(void **dev_ptr, size_t size, CUstream stream) {
  if (cuda::detail::memory_allocation_route(
          CUDAContext::get_instance().supports_mem_pool()) ==
      cuda::detail::MemoryAllocationRoute::kAsyncMemoryPool) {
    async_allocation_calls_.fetch_add(1, std::memory_order_relaxed);
    malloc_async_impl(dev_ptr, size, stream);
  } else {
    sync_allocation_fallback_calls_.fetch_add(1, std::memory_order_relaxed);
    malloc(dev_ptr, size);
  }
}

void CUDADriver::malloc_async_from_pool(void **dev_ptr,
                                        size_t size,
                                        void *pool,
                                        CUstream stream) {
  async_allocation_calls_.fetch_add(1, std::memory_order_relaxed);
  mem_alloc_from_pool_async(dev_ptr, size, pool, stream);
}

void CUDADriver::mem_free_async(void *dev_ptr, CUstream stream) {
  if (cuda::detail::memory_allocation_route(
          CUDAContext::get_instance().supports_mem_pool()) ==
      cuda::detail::MemoryAllocationRoute::kAsyncMemoryPool) {
    async_free_calls_.fetch_add(1, std::memory_order_relaxed);
    mem_free_async_impl(dev_ptr, stream);
  } else {
    sync_free_fallback_calls_.fetch_add(1, std::memory_order_relaxed);
    mem_free(dev_ptr);
  }
}

CUSPARSEDriver::CUSPARSEDriver() {
}

CUSPARSEDriver &CUSPARSEDriver::get_instance() {
  static CUSPARSEDriver *instance = new CUSPARSEDriver();
  return *instance;
}

bool CUSPARSEDriver::load_cusparse() {
  /*
  Load the cuSparse lib whose version follows the CUDA driver's version.
  See load_cusolver() for more information.
  */
  // Get the CUDA Driver's version
  int cuda_version = CUDADriver::get_instance().get_version_major();
  // Try to load the cusparse lib whose version is derived from the CUDA driver
  cusparse_loaded_ = try_load_lib_any_version("cusparse", "64_",
                                              {cuda_version, cuda_version - 1});
  if (!cusparse_loaded_) {
    return false;
  }
#define PER_CUSPARSE_FUNCTION(name, symbol_name, ...) \
  name.set(loader_->load_function(#symbol_name));     \
  name.set_lock(&lock_);                              \
  name.set_names(#name, #symbol_name);
#include "taichi/rhi/cuda/cusparse_functions.inc.h"
#undef PER_CUSPARSE_FUNCTION

  // New cuSPARSE matrix formats and operations are introduced independently
  // from the CUDA driver ABI. Probe them as optional capabilities so that an
  // older, otherwise fully usable cuSPARSE library still loads.
  cp_get_property_.set(loader_->load_function_optional("cusparseGetProperty"));
  cp_get_property_.set_lock(&lock_);
  cp_get_property_.set_names("cp_get_property_", "cusparseGetProperty");
  cpSpMVPreprocess.set(
      loader_->load_function_optional("cusparseSpMV_preprocess"));
  cpSpMVPreprocess.set_lock(&lock_);
  cpSpMVPreprocess.set_names("cpSpMVPreprocess", "cusparseSpMV_preprocess");
  cpCreateDnMat.set(loader_->load_function_optional("cusparseCreateDnMat"));
  cpCreateDnMat.set_lock(&lock_);
  cpCreateDnMat.set_names("cpCreateDnMat", "cusparseCreateDnMat");
  cpDestroyDnMat.set(
      loader_->load_function_optional("cusparseDestroyDnMat"));
  cpDestroyDnMat.set_lock(&lock_);
  cpDestroyDnMat.set_names("cpDestroyDnMat", "cusparseDestroyDnMat");
  cpDnMatSetValues.set(
      loader_->load_function_optional("cusparseDnMatSetValues"));
  cpDnMatSetValues.set_lock(&lock_);
  cpDnMatSetValues.set_names("cpDnMatSetValues", "cusparseDnMatSetValues");
  cpDnVecSetValues.set(
      loader_->load_function_optional("cusparseDnVecSetValues"));
  cpDnVecSetValues.set_lock(&lock_);
  cpDnVecSetValues.set_names("cpDnVecSetValues", "cusparseDnVecSetValues");
  cpSpMatSetAttribute.set(
      loader_->load_function_optional("cusparseSpMatSetAttribute"));
  cpSpMatSetAttribute.set_lock(&lock_);
  cpSpMatSetAttribute.set_names("cpSpMatSetAttribute",
                                "cusparseSpMatSetAttribute");
  cpSpMMBufferSize.set(
      loader_->load_function_optional("cusparseSpMM_bufferSize"));
  cpSpMMBufferSize.set_lock(&lock_);
  cpSpMMBufferSize.set_names("cpSpMMBufferSize", "cusparseSpMM_bufferSize");
  cpSpMMPreprocess.set(
      loader_->load_function_optional("cusparseSpMM_preprocess"));
  cpSpMMPreprocess.set_lock(&lock_);
  cpSpMMPreprocess.set_names("cpSpMMPreprocess", "cusparseSpMM_preprocess");
  cpSpMM.set(loader_->load_function_optional("cusparseSpMM"));
  cpSpMM.set_lock(&lock_);
  cpSpMM.set_names("cpSpMM", "cusparseSpMM");
  cpSpSVCreateDescr.set(
      loader_->load_function_optional("cusparseSpSV_createDescr"));
  cpSpSVCreateDescr.set_lock(&lock_);
  cpSpSVCreateDescr.set_names("cpSpSVCreateDescr",
                              "cusparseSpSV_createDescr");
  cpSpSVDestroyDescr.set(
      loader_->load_function_optional("cusparseSpSV_destroyDescr"));
  cpSpSVDestroyDescr.set_lock(&lock_);
  cpSpSVDestroyDescr.set_names("cpSpSVDestroyDescr",
                               "cusparseSpSV_destroyDescr");
  cpSpSVBufferSize.set(
      loader_->load_function_optional("cusparseSpSV_bufferSize"));
  cpSpSVBufferSize.set_lock(&lock_);
  cpSpSVBufferSize.set_names("cpSpSVBufferSize", "cusparseSpSV_bufferSize");
  cpSpSVAnalysis.set(
      loader_->load_function_optional("cusparseSpSV_analysis"));
  cpSpSVAnalysis.set_lock(&lock_);
  cpSpSVAnalysis.set_names("cpSpSVAnalysis", "cusparseSpSV_analysis");
  cpSpSVSolve.set(loader_->load_function_optional("cusparseSpSV_solve"));
  cpSpSVSolve.set_lock(&lock_);
  cpSpSVSolve.set_names("cpSpSVSolve", "cusparseSpSV_solve");
  cpSpSVUpdateMatrix.set(
      loader_->load_function_optional("cusparseSpSV_updateMatrix"));
  cpSpSVUpdateMatrix.set_lock(&lock_);
  cpSpSVUpdateMatrix.set_names("cpSpSVUpdateMatrix",
                               "cusparseSpSV_updateMatrix");
  cpSpSMCreateDescr.set(
      loader_->load_function_optional("cusparseSpSM_createDescr"));
  cpSpSMCreateDescr.set_lock(&lock_);
  cpSpSMCreateDescr.set_names("cpSpSMCreateDescr",
                              "cusparseSpSM_createDescr");
  cpSpSMDestroyDescr.set(
      loader_->load_function_optional("cusparseSpSM_destroyDescr"));
  cpSpSMDestroyDescr.set_lock(&lock_);
  cpSpSMDestroyDescr.set_names("cpSpSMDestroyDescr",
                               "cusparseSpSM_destroyDescr");
  cpSpSMBufferSize.set(
      loader_->load_function_optional("cusparseSpSM_bufferSize"));
  cpSpSMBufferSize.set_lock(&lock_);
  cpSpSMBufferSize.set_names("cpSpSMBufferSize", "cusparseSpSM_bufferSize");
  cpSpSMAnalysis.set(
      loader_->load_function_optional("cusparseSpSM_analysis"));
  cpSpSMAnalysis.set_lock(&lock_);
  cpSpSMAnalysis.set_names("cpSpSMAnalysis", "cusparseSpSM_analysis");
  cpSpSMSolve.set(loader_->load_function_optional("cusparseSpSM_solve"));
  cpSpSMSolve.set_lock(&lock_);
  cpSpSMSolve.set_names("cpSpSMSolve", "cusparseSpSM_solve");
  cpSpSMUpdateMatrix.set(
      loader_->load_function_optional("cusparseSpSM_updateMatrix"));
  cpSpSMUpdateMatrix.set_lock(&lock_);
  cpSpSMUpdateMatrix.set_names("cpSpSMUpdateMatrix",
                               "cusparseSpSM_updateMatrix");

  capabilities_ = {};
  if (cp_get_property_.available()) {
    // libraryPropertyType is a shared CUDA-library enum with stable values:
    // MAJOR_VERSION=0, MINOR_VERSION=1, PATCH_LEVEL=2.
    constexpr int kMajorVersion = 0;
    constexpr int kMinorVersion = 1;
    constexpr int kPatchLevel = 2;
    const auto query_property = [&](int property, int &value) {
      if (cp_get_property_.call(property, &value) != 0) {
        value = -1;
      }
    };
    query_property(kMajorVersion, capabilities_.library_version_major);
    query_property(kMinorVersion, capabilities_.library_version_minor);
    query_property(kPatchLevel, capabilities_.library_version_patch);
  }

  cpCreateBsr.set(loader_->load_function_optional("cusparseCreateBsr"));
  cpCreateBsr.set_lock(&lock_);
  cpCreateBsr.set_names("cpCreateBsr", "cusparseCreateBsr");
  capabilities_.bsr_descriptor_available = cpCreateBsr.available();
  capabilities_.spmv_preprocess_available = cpSpMVPreprocess.available();
  capabilities_.scalar_spmv_available =
      cpCreate.available() && cpDestroy.available() &&
      cpSetStream.available() && cpGetStream.available() &&
      cpCreateCsr.available() && cpDestroySpMat.available() &&
      cpCreateDnVec.available() && cpDestroyDnVec.available() &&
      cpSpMV_bufferSize.available() && cpSpMV.available();
  capabilities_.spmm_f32_available =
      capabilities_.scalar_spmv_available && cpCreateDnMat.available() &&
      cpDestroyDnMat.available() && cpDnMatSetValues.available() &&
      cpSpMMBufferSize.available() && cpSpMM.available();
  capabilities_.spmm_preprocess_available =
      capabilities_.spmm_f32_available && cpSpMMPreprocess.available();
  const bool triangular_common =
      capabilities_.scalar_spmv_available && cpSpMatSetAttribute.available();
  capabilities_.spsv_f32_available =
      triangular_common && cpDnVecSetValues.available() &&
      cpSpSVCreateDescr.available() && cpSpSVDestroyDescr.available() &&
      cpSpSVBufferSize.available() && cpSpSVAnalysis.available() &&
      cpSpSVSolve.available();
  capabilities_.spsm_f32_available =
      triangular_common && cpCreateDnMat.available() &&
      cpDestroyDnMat.available() && cpDnMatSetValues.available() &&
      cpSpSMCreateDescr.available() && cpSpSMDestroyDescr.available() &&
      cpSpSMBufferSize.available() && cpSpSMAnalysis.available() &&
      cpSpSMSolve.available();
  capabilities_.spsv_value_update_available =
      capabilities_.spsv_f32_available && cpSpSVUpdateMatrix.available();
  capabilities_.spsm_value_update_available =
      capabilities_.spsm_f32_available && cpSpSMUpdateMatrix.available();
  capabilities_.triangular_value_update_available =
      capabilities_.spsv_value_update_available &&
      capabilities_.spsm_value_update_available;
  const auto version_at_least = [&](int major, int minor, int patch) {
    const auto actual = std::make_tuple(capabilities_.library_version_major,
                                        capabilities_.library_version_minor,
                                        capabilities_.library_version_patch);
    return actual >= std::make_tuple(major, minor, patch);
  };
  // Generic SpMV did not accept a BSR descriptor until CUDA Toolkit 13.0
  // Update 1, whose independently versioned cuSPARSE component is 12.6.3.
  // Descriptor construction alone is therefore not an execution capability
  // on an older cuSPARSE provider.
  capabilities_.generic_bsr_spmv_available =
      capabilities_.bsr_descriptor_available && cpSpMV.available() &&
      version_at_least(12, 6, 3);
  return cusparse_loaded_;
}

CUSOLVERDriver::CUSOLVERDriver() {
}

CUSOLVERDriver &CUSOLVERDriver::get_instance() {
  static CUSOLVERDriver *instance = new CUSOLVERDriver();
  return *instance;
}

bool CUSOLVERDriver::load_cusolver() {
  /*
  Load the cuSolver lib whose version follows the CUDA driver's version.
  Note that cusolver's filename is NOT necessarily the same with CUDA Toolkit
  (on Windows). For instance, CUDA Toolkit 12.2 ships a cusolver64_11.dll
  (checked on 2023.7.13) Therefore, the following function attempts to load a
  cusolver lib which is one version backward from the CUDA Driver's version.
  */
  // Get the CUDA Driver's version
  int cuda_version = CUDADriver::get_instance().get_version_major();
  // Try to load the cusolver lib whose version is derived from the CUDA driver
  cusolver_loaded_ = try_load_lib_any_version("cusolver", "64_",
                                              {cuda_version, cuda_version - 1});
  if (!cusolver_loaded_) {
    return false;
  }
#define PER_CUSOLVER_FUNCTION(name, symbol_name, ...) \
  name.set(loader_->load_function(#symbol_name));     \
  name.set_lock(&lock_);                              \
  name.set_names(#name, #symbol_name);
#include "taichi/rhi/cuda/cusolver_functions.inc.h"
#undef PER_CUSOLVER_FUNCTION
  return cusolver_loaded_;
}

CUBLASDriver::CUBLASDriver() {
}

CUBLASDriver &CUBLASDriver::get_instance() {
  static CUBLASDriver *instance = new CUBLASDriver();
  return *instance;
}

bool CUBLASDriver::load_cublas() {
  /* To be compatible with torch environment, please libcublas.so.11 other than
   * libcublas.so. When using libcublas.so, the system cublas will be loaded and
   * it would confict with torch's cublas. When using libcublas.so.11, the
   * torch's cublas will be loaded.
   */
  const int cuda_version = CUDADriver::get_instance().get_version_major();
  cublas_loaded_ = try_load_lib_any_version(
      "cublas", "64_", {cuda_version, cuda_version - 1, 11, 10});
  if (!cublas_loaded_) {
    return false;
  }
#define PER_CUBLAS_FUNCTION(name, symbol_name, ...) \
  name.set(loader_->load_function(#symbol_name));   \
  name.set_lock(&lock_);                            \
  name.set_names(#name, #symbol_name);
#include "taichi/rhi/cuda/cublas_functions.inc.h"
#undef PER_CUBLAS_FUNCTION
  cub_get_property_.set(loader_->load_function_optional("cublasGetProperty"));
  cub_get_property_.set_lock(&lock_);
  cub_get_property_.set_names("cub_get_property_", "cublasGetProperty");
  auto *set_workspace =
      loader_->load_function_optional("cublasSetWorkspace_v2");
  const char *set_workspace_symbol = "cublasSetWorkspace_v2";
  if (!set_workspace) {
    set_workspace = loader_->load_function_optional("cublasSetWorkspace");
    set_workspace_symbol = "cublasSetWorkspace";
  }
  cubSetWorkspace.set(set_workspace);
  cubSetWorkspace.set_lock(&lock_);
  cubSetWorkspace.set_names("cubSetWorkspace", set_workspace_symbol);
  capabilities_ = {};
  if (cub_get_property_.available()) {
    const auto query_property = [&](int property, int &value) {
      if (cub_get_property_.call(property, &value) != 0) {
        value = -1;
      }
    };
    query_property(0, capabilities_.library_version_major);
    query_property(1, capabilities_.library_version_minor);
    query_property(2, capabilities_.library_version_patch);
  }
  capabilities_.gemm_f32_available =
      cubCreate.available() && cubDestroy.available() &&
      cubSetStream.available() && cubSetPointerMode.available() &&
      cubSgemm.available();
  return cublas_loaded_;
}

CUFFTDriver::CUFFTDriver() {
}

CUFFTDriver &CUFFTDriver::get_instance() {
  static CUFFTDriver *instance = new CUFFTDriver();
  return *instance;
}

bool CUFFTDriver::load_cufft() {
  std::lock_guard<std::mutex> load_guard(load_lock_);
  if (cufft_loaded_) {
    return true;
  }
  const int cuda_version = CUDADriver::get_instance().get_version_major();
  if (!try_load_lib_any_version(
          "cufft", "64_", {cuda_version, cuda_version - 1, 12, 11, 10})) {
    return false;
  }

  bool symbols_available = true;
#define PER_CUFFT_FUNCTION(name, symbol_name, ...)                 \
  name.set(loader_->load_function_optional(#symbol_name));        \
  name.set_lock(&lock_);                                          \
  name.set_names(#name, #symbol_name);                            \
  symbols_available = symbols_available && name.available();
#include "taichi/rhi/cuda/cufft_functions.inc.h"
#undef PER_CUFFT_FUNCTION
  create.set(loader_->load_function_optional("cufftCreate"));
  create.set_lock(&lock_);
  create.set_names("create", "cufftCreate");
  make_plan_many.set(loader_->load_function_optional("cufftMakePlanMany"));
  make_plan_many.set_lock(&lock_);
  make_plan_many.set_names("make_plan_many", "cufftMakePlanMany");
  set_jit_callback.set(loader_->load_function_optional("cufftXtSetJITCallback"));
  set_jit_callback.set_lock(&lock_);
  set_jit_callback.set_names("set_jit_callback", "cufftXtSetJITCallback");
  capabilities_ = {};
  if (!symbols_available) {
    cufft_loaded_.store(false, std::memory_order_release);
    return false;
  }
  int version = 0;
  if (get_version.call(&version) == 0) {
    capabilities_.library_version = version;
  }
  cufft_loaded_.store(true, std::memory_order_release);
  return true;
}

}  // namespace taichi::lang
