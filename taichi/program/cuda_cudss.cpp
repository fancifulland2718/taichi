#include "taichi/program/program.h"
#include "taichi/program/cuda_cudss_capture.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(TI_WITH_CUDA)
#include "taichi/common/dynamic_loader.h"
#include "taichi/cudss/forge_cudss_provider.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_driver.h"

namespace taichi::lang {
namespace {

constexpr int kCudssStatusSuccess = 0;
constexpr int kCudssPhaseAnalysis = 3;
constexpr int kCudssPhaseFactorization = 4;
constexpr int kCudssPhaseRefactorization = 8;
constexpr int kCudssPhaseSolve = 1008;
constexpr int kCudssDataTypeF32 = 0;
constexpr int kCudssDataTypeI32 = 10;
constexpr int kCudssMatrixGeneral = 0;
constexpr int kCudssMatrixSymmetric = 1;
constexpr int kCudssMatrixSpd = 3;
constexpr int kCudssViewFull = 0;
constexpr int kCudssViewLower = 1;
constexpr int kCudssViewUpper = 2;
constexpr int kCudssBaseZero = 0;
constexpr int kCudssLayoutColumnMajor = 0;

void require_cudss_success(std::uint32_t status, const char *operation) {
  TI_ERROR_IF(status != kCudssStatusSuccess,
              "CUDA cuDSS {} failed (status {}).", operation, status);
}

void warn_cudss_failure(std::uint32_t status, const char *operation) {
  if (status != kCudssStatusSuccess) {
    TI_WARN("CUDA cuDSS {} failed during cleanup (status {}).", operation,
            status);
  }
}

const CuSparseMatrix &require_cudss_matrix(SparseMatrix *matrix,
                                           Program *program) {
  TI_ERROR_IF(!matrix, "CUDA cuDSS received a null sparse matrix.");
  const auto *csr = dynamic_cast<const CuSparseMatrix *>(matrix);
  TI_ERROR_IF(!csr,
              "CUDA cuDSS requires a CUDA scalar CSR SparseMatrix; no "
              "format conversion or host fallback was performed.");
  TI_ERROR_IF(csr->num_rows() <= 0 || csr->num_rows() != csr->num_cols(),
              "CUDA cuDSS requires a non-empty square sparse matrix.");
  TI_ERROR_IF(csr->get_data_type() != PrimitiveType::f32,
              "The first CUDA cuDSS provider slice supports f32 only.");
  TI_ERROR_IF(csr->get_nnz() <= 0 || !csr->get_row_ptr() ||
                  !csr->get_col_ind() || !csr->get_val_ptr(),
              "CUDA cuDSS requires a materialized non-empty CSR matrix.");
  (void)program;
  return *csr;
}

void validate_cudss_matrix_contract(int matrix_type, int matrix_view) {
  TI_ERROR_IF(matrix_type != kCudssMatrixGeneral &&
                  matrix_type != kCudssMatrixSymmetric &&
                  matrix_type != kCudssMatrixSpd,
              "CUDA cuDSS matrix_type must be general, symmetric, or spd.");
  TI_ERROR_IF(matrix_view != kCudssViewFull && matrix_view != kCudssViewLower &&
                  matrix_view != kCudssViewUpper,
              "CUDA cuDSS matrix_view must be full, lower, or upper.");
  TI_ERROR_IF(
      matrix_type == kCudssMatrixGeneral && matrix_view != kCudssViewFull,
      "CUDA cuDSS general matrices require the full matrix view.");
}

void validate_cudss_vector(Ndarray *array,
                           std::size_t expected_elements,
                           const char *name,
                           Program *program) {
  TI_ERROR_IF(!array, "CUDA cuDSS {} received a null ndarray.", name);
  TI_ERROR_IF(!array->get_element_shape().empty() ||
                  array->get_element_data_type() != PrimitiveType::f32 ||
                  array->get_nelement() != expected_elements ||
                  array->get_element_size() != sizeof(float32),
              "CUDA cuDSS {} must be a compact scalar f32 ndarray with {} "
              "entries.",
              name, expected_elements);
  TI_ERROR_IF(array->owning_program() != program,
              "CUDA cuDSS {} must belong to the active runtime.", name);
}

void validate_cudss_values(Ndarray *array,
                           std::size_t expected_elements,
                           Program *program) {
  TI_ERROR_IF(!array, "CUDA cuDSS matrix values received a null ndarray.");
  TI_ERROR_IF(!array->get_element_shape().empty() ||
                  array->get_element_data_type() != PrimitiveType::f32 ||
                  array->get_nelement() != expected_elements ||
                  array->get_element_size() != sizeof(float32),
              "CUDA cuDSS matrix values must be a compact scalar f32 ndarray "
              "with {} entries.",
              expected_elements);
  TI_ERROR_IF(array->owning_program() != program,
              "CUDA cuDSS matrix values must belong to the active runtime.");
}

}  // namespace

class CudssProviderRuntime {
 public:
  CudssProviderRuntime(const std::string &adapter_path,
                       const std::string &runtime_library_path) {
    TI_ERROR_IF(adapter_path.empty(),
                "CUDA cuDSS requires a bundled Forge provider adapter.");
    auto &cuda_driver = CUDADriver::get_instance_without_context();
    TI_ERROR_IF(!cuda_driver.nvidia_extensions_available() ||
                    cuda_driver.get_version_major() < 12,
                "CUDA cuDSS requires the NVIDIA CUDA driver API 12.0 or "
                "newer.");
    auto &cublas = CUBLASDriver::get_instance();
    TI_ERROR_IF(!cublas.is_loaded() && !cublas.load_cublas(),
                "CUDA cuDSS requires a compatible user-managed cuBLAS "
                "runtime.");

    loader_ = std::make_unique<DynamicLoader>(adapter_path);
    TI_ERROR_IF(!loader_->loaded(),
                "CUDA cuDSS could not load bundled adapter {}.", adapter_path);
    auto *query_symbol =
        loader_->load_function_optional(TI_FORGE_CUDSS_PROVIDER_QUERY_SYMBOL);
    TI_ERROR_IF(!query_symbol,
                "CUDA cuDSS bundled adapter is missing its query symbol.");
    auto query = reinterpret_cast<TiForgeCudssProviderQueryFn>(query_symbol);
    const auto query_result =
        query(TI_FORGE_CUDSS_PROVIDER_ABI_VERSION, sizeof(api_), &api_);
    TI_ERROR_IF(query_result != TI_FORGE_CUDSS_SUCCESS,
                "CUDA cuDSS bundled adapter rejected Forge provider ABI {} "
                "(result {}).",
                TI_FORGE_CUDSS_PROVIDER_ABI_VERSION,
                static_cast<int>(query_result));
    TI_ERROR_IF(
        api_.struct_size < sizeof(api_) ||
            api_.provider_abi_version != TI_FORGE_CUDSS_PROVIDER_ABI_VERSION ||
            api_.info.struct_size < sizeof(api_.info) ||
            api_.info.provider_abi_version !=
                TI_FORGE_CUDSS_PROVIDER_ABI_VERSION ||
            api_.info.cudss_header_version / 100 != 8,
        "CUDA cuDSS bundled adapter identity is incompatible.");
    constexpr uint64_t required_features =
        TI_FORGE_CUDSS_FEATURE_CSR | TI_FORGE_CUDSS_FEATURE_DENSE_VECTOR |
        TI_FORGE_CUDSS_FEATURE_STAGED_EXECUTION |
        TI_FORGE_CUDSS_FEATURE_VALUE_REBIND |
        TI_FORGE_CUDSS_FEATURE_EXPLICIT_STREAM;
    TI_ERROR_IF((api_.info.features & required_features) != required_features,
                "CUDA cuDSS bundled adapter lacks required features.");
    TI_ERROR_IF(!api_.create_runtime || !api_.destroy_runtime || !api_.create ||
                    !api_.destroy || !api_.set_stream || !api_.config_create ||
                    !api_.config_destroy || !api_.data_create ||
                    !api_.data_destroy || !api_.matrix_create_csr ||
                    !api_.matrix_create_dn || !api_.matrix_destroy ||
                    !api_.matrix_set_values || !api_.matrix_set_csr_pointers ||
                    !api_.execute || !api_.get_last_error,
                "CUDA cuDSS bundled adapter API table is incomplete.");
    TiForgeCudssRuntime candidate_runtime = nullptr;
    TiForgeCudssRuntimeInfo candidate_info{};
    candidate_info.struct_size = sizeof(candidate_info);
    const char *runtime_path =
        runtime_library_path.empty() ? nullptr : runtime_library_path.c_str();
    const auto runtime_result =
        api_.create_runtime(runtime_path, &candidate_runtime, &candidate_info);
    if (runtime_result != TI_FORGE_CUDSS_SUCCESS || !candidate_runtime) {
      if (candidate_runtime) {
        api_.destroy_runtime(candidate_runtime);
      }
      TI_ERROR("CUDA cuDSS vendor runtime initialization failed: {}",
               adapter_error());
    }
    if (candidate_info.version_major != 0 ||
        candidate_info.version_minor != 8) {
      api_.destroy_runtime(candidate_runtime);
      TI_ERROR(
          "CUDA cuDSS adapter loaded an unsupported vendor runtime "
          "version {}.{}.{}.",
          candidate_info.version_major, candidate_info.version_minor,
          candidate_info.version_patch);
    }
    runtime_ = candidate_runtime;
    runtime_info_ = candidate_info;
  }

  CudssProviderRuntime(const CudssProviderRuntime &) = delete;
  CudssProviderRuntime &operator=(const CudssProviderRuntime &) = delete;

  ~CudssProviderRuntime() {
    if (runtime_ && api_.destroy_runtime) {
      const auto result = api_.destroy_runtime(runtime_);
      if (result != TI_FORGE_CUDSS_SUCCESS) {
        TI_WARN("CUDA cuDSS adapter runtime cleanup failed (result {}).",
                static_cast<int>(result));
      }
      runtime_ = nullptr;
    }
  }

  const TiForgeCudssProviderApi &api() const {
    return api_;
  }

  TiForgeCudssRuntime runtime() const {
    return runtime_;
  }

  TiForgeCudssConfigurationApi configuration_api() const {
    auto *symbol = loader_->load_function_optional(
        TI_FORGE_CUDSS_CONFIGURATION_QUERY_SYMBOL);
    TI_ERROR_IF(!symbol,
                "CUDA cuDSS configured recipes require the optional "
                "configuration adapter extension.");
    const auto query =
        reinterpret_cast<TiForgeCudssConfigurationQueryFn>(symbol);
    TiForgeCudssConfigurationApi result{};
    TI_ERROR_IF(
        query(TI_FORGE_CUDSS_CONFIGURATION_ABI_VERSION, sizeof(result),
              &result) != TI_FORGE_CUDSS_SUCCESS ||
            result.struct_size < sizeof(result) ||
            result.abi_version != TI_FORGE_CUDSS_CONFIGURATION_ABI_VERSION ||
            !result.configure || !result.analysis_memory_estimates,
        "CUDA cuDSS configuration extension is incompatible.");
    return result;
  }

  const TiForgeCudssRuntimeInfo &runtime_info() const {
    return runtime_info_;
  }

  TiForgeCudssAllocatorApi allocator_api() const {
    auto *symbol =
        loader_->load_function_optional(TI_FORGE_CUDSS_ALLOCATOR_QUERY_SYMBOL);
    TI_ERROR_IF(!symbol,
                "CUDA cuDSS Graph allocator extension is unavailable.");
    auto query = reinterpret_cast<TiForgeCudssAllocatorQueryFn>(symbol);
    TiForgeCudssAllocatorApi result{};
    TI_ERROR_IF(query(1, sizeof(result), &result) != TI_FORGE_CUDSS_SUCCESS ||
                    result.struct_size < sizeof(result) ||
                    result.abi_version != 1 || !result.set_allocator,
                "CUDA cuDSS Graph allocator extension is incompatible.");
    return result;
  }

 private:
  std::string adapter_error() const {
    if (!api_.get_last_error) {
      return "adapter error unavailable";
    }
    const auto required = api_.get_last_error(nullptr, 0);
    if (required <= 1) {
      return "adapter call failed without detail";
    }
    std::vector<char> message(required, '\0');
    api_.get_last_error(message.data(), message.size());
    return message.data();
  }

  std::unique_ptr<DynamicLoader> loader_;
  TiForgeCudssProviderApi api_{};
  TiForgeCudssRuntime runtime_{nullptr};
  TiForgeCudssRuntimeInfo runtime_info_{};
};

// Only preparation/capture/retirement calls this allocator. After warming the
// private snapshot it is sealed: captured work cannot create/free persistent
// vendor pointers that would escape a particular CUDA graph's lifetime.
class CudssGraphAllocator {
 public:
  static int allocate(void *context,
                      void **pointer,
                      size_t bytes,
                      void *stream) noexcept {
    auto &owner = *static_cast<CudssGraphAllocator *>(context);
    std::lock_guard<std::mutex> lock(owner.mutex_);
    *pointer = nullptr;
    if (bytes == 0) {
      return 0;
    }
    if (owner.sealed_) {
      ++owner.sealed_allocation_rejections_;
      return 1;
    }
    try {
      CUDADriver::get_instance().malloc_async_impl(pointer, bytes, stream);
      owner.allocations_.emplace(*pointer, bytes);
      owner.live_bytes_ += bytes;
      owner.peak_bytes_ = std::max(owner.peak_bytes_, owner.live_bytes_);
      return 0;
    } catch (...) {
      if (*pointer) {
        try {
          CUDADriver::get_instance().mem_free_async_impl(*pointer, stream);
        } catch (...) {
        }
        *pointer = nullptr;
      }
      return 1;
    }
  }

  static int deallocate(void *context,
                        void *pointer,
                        size_t,
                        void *stream) noexcept {
    auto &owner = *static_cast<CudssGraphAllocator *>(context);
    std::lock_guard<std::mutex> lock(owner.mutex_);
    if (!pointer) {
      return 0;
    }
    if (owner.sealed_) {
      ++owner.sealed_allocation_rejections_;
      return 1;
    }
    const auto found = owner.allocations_.find(pointer);
    if (found == owner.allocations_.end()) {
      return 1;
    }
    try {
      CUDADriver::get_instance().mem_free_async_impl(pointer, stream);
      owner.live_bytes_ -= found->second;
      owner.allocations_.erase(found);
      return 0;
    } catch (...) {
      return 1;
    }
  }

  void seal(bool value) {
    std::lock_guard<std::mutex> lock(mutex_);
    sealed_ = value;
  }

  std::array<std::uint64_t, 3> observation() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {live_bytes_, peak_bytes_, sealed_allocation_rejections_};
  }

 private:
  mutable std::mutex mutex_;
  bool sealed_{false};
  std::unordered_map<void *, std::size_t> allocations_;
  std::uint64_t live_bytes_{0};
  std::uint64_t peak_bytes_{0};
  std::uint64_t sealed_allocation_rejections_{0};
};

// cuDSS emits small host-to-device parameter copies while recording solves.
// Snapshot each RHS immediately, then upload the packed immutable bytes once
// after capture. Graph nodes read device storage owned by that graph/frame;
// later bindings cannot overwrite it, and replay needs no host transfer.
class CudssCaptureInputs final : public aot::CudaGraphCaptureResources {
 public:
  explicit CudssCaptureInputs(std::shared_ptr<RuntimeFaultDomain> fault_domain)
      : fault_domain_(std::move(fault_domain)) {
  }

  ~CudssCaptureInputs() override {
    release(true);
  }

  void snapshot(void *node, CUDA_MEMCPY3D params) {
    auto &input = inputs_.emplace_back();
    input.node = node;
    input.params = params;
    input.bytes.resize(params.WidthInBytes);
    std::memcpy(
        input.bytes.data(),
        static_cast<const std::uint8_t *>(params.srcHost) + params.srcXInBytes,
        input.bytes.size());
    // The vendor may reuse its host buffer during the very next RHS capture.
    params.srcHost = input.bytes.data();
    params.srcXInBytes = 0;
    CUDADriver::get_instance().graph_memcpy_node_set_params(node, &params);
  }

  void finalize() override {
    if (inputs_.empty())
      return;
    std::vector<std::uint8_t> packed;
    for (auto &input : inputs_) {
      const auto size = packed.size();
      TI_ERROR_IF(size > std::numeric_limits<std::size_t>::max() - 7,
                  "cuDSS capture parameter storage is too large.");
      input.offset = (size + 7) & ~std::size_t(7);
      TI_ERROR_IF(input.bytes.size() >
                      std::numeric_limits<std::size_t>::max() - input.offset,
                  "cuDSS capture parameter storage is too large.");
      packed.resize(input.offset + input.bytes.size());
      std::memcpy(packed.data() + input.offset, input.bytes.data(),
                  input.bytes.size());
    }
    auto &driver = CUDADriver::get_instance();
    driver.malloc(&device_, packed.size());
    bytes_ = packed.size();
    // Synchronous cold upload keeps rollback/lifetime local. It executes no
    // mathematical work, and does not impose synchronization on graph replay.
    driver.memcpy_host_to_device(device_, packed.data(), packed.size());
    for (const auto &input : inputs_) {
      auto params = input.params;
      params.srcMemoryType = static_cast<CUmemorytype>(CU_MEMORYTYPE_DEVICE);
      params.srcHost = nullptr;
      params.srcDevice = static_cast<char *>(device_) + input.offset;
      params.srcXInBytes = 0;
      params.srcPitch = params.WidthInBytes;
      driver.graph_memcpy_node_set_params(input.node, &params);
    }
    inputs_.clear();
  }

  std::uint64_t requested_device_bytes() const override {
    return bytes_;
  }

  void release(bool backend_safe) noexcept override {
    auto *device = std::exchange(device_, nullptr);
    bytes_ = 0;
    if (!device || !backend_safe || !fault_domain_ ||
        !fault_domain_->backend_calls_safe() ||
        fault_domain_->state() == RuntimeLifecycleState::kFinalized)
      return;
    try {
      auto context = CUDAContext::get_instance().get_guard();
      CUDADriver::get_instance().mem_free(device);
    } catch (...) {
      // Failed contexts must not cause another backend call during unwinding.
    }
  }

 private:
  struct Input {
    void *node{nullptr};
    CUDA_MEMCPY3D params{};
    std::vector<std::uint8_t> bytes;
    std::size_t offset{0};
  };
  std::vector<Input> inputs_;
  void *device_{nullptr};
  std::uint64_t bytes_{0};
  std::shared_ptr<RuntimeFaultDomain> fault_domain_;
};

class CudaCudssPlan final : public CudaProviderCompletionResource {
 public:
  CudaCudssPlan(const CuSparseMatrix &matrix,
                int matrix_type,
                int matrix_view,
                const std::string &adapter_path,
                const std::string &runtime_library_path,
                std::shared_ptr<RuntimeFaultDomain> fault_domain,
                const std::vector<int> &configuration,
                bool graph_owned)
      : rows_(static_cast<std::size_t>(matrix.num_rows())),
        nonzeros_(static_cast<std::size_t>(matrix.get_nnz())),
        graph_owned_(graph_owned),
        provider_(std::make_unique<CudssProviderRuntime>(adapter_path,
                                                         runtime_library_path)),
        fault_domain_(std::move(fault_domain)) {
    validate_cudss_matrix_contract(matrix_type, matrix_view);
    TI_ERROR_IF(graph_owned && configuration.empty(),
                "CUDA cuDSS Graph plans require a frozen configuration.");
    TI_ERROR_IF(!configuration.empty() &&
                    (configuration.size() != 2 || configuration[0] < 0 ||
                     configuration[0] > 3 || configuration[1] < 0 ||
                     configuration[1] > 1),
                "CUDA cuDSS private configuration must contain a supported "
                "reordering and solve policy.");
    if (!configuration.empty()) {
      configuration_api_ = provider_->configuration_api();
      reordering_ = configuration[0];
      solve_algorithm_ = configuration[1];
    }
    const auto &api = provider_->api();
    auto runtime = provider_->runtime();
    try {
      require_cudss_success(api.create(runtime, &context_), "handle creation");
      TI_ERROR_IF(!context_, "CUDA cuDSS returned a null handle.");
      require_cudss_success(api.set_stream(runtime, context_, nullptr),
                            "runtime stream binding");
      if (graph_owned_) {
        auto &driver = CUDADriver::get_instance();
        TI_ERROR_IF(
            !driver.malloc_async_impl.available() ||
                !driver.mem_free_async_impl.available() ||
                !CUDAContext::get_instance().supports_mem_pool(),
            "CUDA cuDSS Graph plans require stream-ordered allocation.");
        // cuDSS analysis/factors retain stream-local state. Keep one stream
        // for the whole owner lifetime, including capture and destruction.
        driver.stream_create(&graph_stream_, CU_STREAM_NON_BLOCKING);
        driver.event_create(&graph_fork_event_, CU_EVENT_DISABLE_TIMING);
        driver.event_create(&graph_join_event_, CU_EVENT_DISABLE_TIMING);
        require_cudss_success(api.set_stream(runtime, context_, graph_stream_),
                              "Graph owner stream binding");
        allocator_ = std::make_unique<CudssGraphAllocator>();
        require_cudss_success(
            provider_->allocator_api().set_allocator(
                runtime, context_, allocator_.get(),
                CudssGraphAllocator::allocate, CudssGraphAllocator::deallocate),
            "Graph allocator binding");
        create_graph_snapshot(matrix);
      }
      require_cudss_success(api.config_create(runtime, &config_),
                            "configuration creation");
      if (configuration_api_.configure) {
        require_cudss_success(
            configuration_api_.configure(runtime, config_, reordering_,
                                         solve_algorithm_),
            "frozen configuration round-trip");
      }
      require_cudss_success(api.data_create(runtime, context_, &data_),
                            "solver-data creation");
      const void *row_start =
          graph_owned_ ? seed_buffers_[0] : matrix.get_row_ptr();
      // cuDSS accepts the canonical three-array CSR form when rowEnd is null.
      // Passing rowOffsets + 1 selects its unsupported four-array CSR form.
      require_cudss_success(
          api.matrix_create_csr(
              runtime, &matrix_, static_cast<std::int64_t>(matrix.num_rows()),
              static_cast<std::int64_t>(matrix.num_cols()),
              static_cast<std::int64_t>(matrix.get_nnz()), row_start, nullptr,
              graph_owned_ ? seed_buffers_[1] : matrix.get_col_ind(),
              graph_owned_ ? seed_buffers_[2] : matrix.get_val_ptr(),
              kCudssDataTypeI32, kCudssDataTypeI32, kCudssDataTypeF32,
              matrix_type, matrix_view, kCudssBaseZero),
          "CSR descriptor creation");
      if (graph_owned_) {
        prepare_graph_snapshot();
      }
    } catch (...) {
      destroy(true);
      throw;
    }
  }

  ~CudaCudssPlan() {
    const bool provider_calls_safe =
        fault_domain_ && !fault_domain_->has_fatal_fault();
    if (provider_calls_safe) {
      try {
        auto cuda_submission_guard =
            CUDAContext::get_instance().get_submission_lock_guard();
        auto context_guard = CUDAContext::get_instance().get_guard();
        destroy(true);
        return;
      } catch (...) {
      }
    }
    destroy(false);
  }

  std::shared_ptr<CudssCaptureInputs> capture_inputs() const {
    return std::make_shared<CudssCaptureInputs>(fault_domain_);
  }

  void analyze(const CuSparseMatrix &matrix) {
    std::lock_guard<std::mutex> lock(mutex_);
    TI_ERROR_IF(closed_, "CUDA cuDSS plan is closed.");
    require_no_refactor_solve_inflight("analyze");
    TI_ERROR_IF(static_cast<std::size_t>(matrix.num_rows()) != rows_ ||
                    matrix.num_rows() != matrix.num_cols() ||
                    matrix.get_data_type() != PrimitiveType::f32,
                "CUDA cuDSS analyze received a matrix that does not match "
                "the plan shape or dtype.");
    const auto &api = provider_->api();
    auto runtime = provider_->runtime();
    // Invalidate cached analysis facts before a potentially failing reanalysis.
    // Only configured recipes request these estimates; ordinary plans do not
    // acquire additional vendor queries.
    memory_estimates_status_ = -1;
    memory_estimates_written_ = 0;
    memory_estimates_.fill(-1);
    require_cudss_success(
        api.execute(runtime, context_, kCudssPhaseAnalysis, config_, data_,
                    matrix_, nullptr, nullptr),
        "analysis");
    if (configuration_api_.analysis_memory_estimates) {
      memory_estimates_status_ = configuration_api_.analysis_memory_estimates(
          runtime, context_, data_, memory_estimates_.data(),
          sizeof(memory_estimates_), &memory_estimates_written_);
    }
    analyzed_csr_row_ptr_.resize(rows_ + 1);
    analyzed_csr_col_ind_.resize(matrix.get_nnz());
    CUDADriver::get_instance().memcpy_device_to_host(
        analyzed_csr_row_ptr_.data(), matrix.get_row_ptr(),
        sizeof(int) * analyzed_csr_row_ptr_.size());
    CUDADriver::get_instance().memcpy_device_to_host(
        analyzed_csr_col_ind_.data(), matrix.get_col_ind(),
        sizeof(int) * analyzed_csr_col_ind_.size());
    const auto stats = matrix.debug_runtime_statistics();
    analyzed_matrix_id_ = matrix.matrix_id();
    analyzed_pattern_version_ = matrix.pattern_version();
    analyzed_shared_pattern_id_ = stats.shared_pattern_id;
    analyzed_ = true;
    factorized_ = false;
    factorized_from_explicit_values_ = false;
  }

  void factorize(const CuSparseMatrix &matrix, bool refactorize) {
    std::lock_guard<std::mutex> lock(mutex_);
    TI_ERROR_IF(closed_, "CUDA cuDSS plan is closed.");
    require_no_refactor_solve_inflight("factorize");
    TI_ERROR_IF(!analyzed_,
                "CUDA cuDSS factorization requires analyze() first.");
    TI_ERROR_IF(refactorize && !factorized_,
                "CUDA cuDSS refactorization requires a prior successful "
                "factorization.");
    validate_analyzed_pattern(matrix);
    const auto &api = provider_->api();
    auto runtime = provider_->runtime();
    require_cudss_success(api.matrix_set_csr_pointers(
                              runtime, matrix_, matrix.get_row_ptr(), nullptr,
                              matrix.get_col_ind(), matrix.get_val_ptr()),
                          "CSR descriptor rebinding");
    factorized_ = false;
    factorized_from_explicit_values_ = false;
    ++factor_invalidations_;
    require_cudss_success(
        api.execute(
            runtime, context_,
            refactorize ? kCudssPhaseRefactorization : kCudssPhaseFactorization,
            config_, data_, matrix_, nullptr, nullptr),
        refactorize ? "refactorization" : "factorization");
    factorized_ = true;
    factorized_matrix_id_ = matrix.matrix_id();
    factorized_pattern_version_ = matrix.pattern_version();
    factorized_numeric_version_ = matrix.numeric_version();
    ++factor_generation_;
  }

  void solve(const CuSparseMatrix &matrix, void *rhs, void *solution) {
    std::lock_guard<std::mutex> lock(mutex_);
    TI_ERROR_IF(closed_, "CUDA cuDSS plan is closed.");
    require_no_refactor_solve_inflight("solve");
    TI_ERROR_IF(!factorized_,
                "CUDA cuDSS solve requires a successful factorization.");
    TI_ERROR_IF(factorized_from_explicit_values_,
                "CUDA cuDSS factors came from explicit Graph values. Use "
                "record_refactor_solve() again or factorize the stored "
                "matrix before a standalone solve.");
    TI_ERROR_IF(
        factorized_matrix_id_ != matrix.matrix_id() ||
            factorized_pattern_version_ != matrix.pattern_version() ||
            factorized_numeric_version_ != matrix.numeric_version(),
        "CUDA cuDSS factorization is stale because the matrix or its "
        "pattern/numeric version changed. Call factorize() again before "
        "solve().");
    bind_dense_vectors(rhs, solution);
    execute_solve();
  }

  std::size_t rows() const noexcept {
    return rows_;
  }

  std::size_t nonzeros() const noexcept {
    return nonzeros_;
  }

  void reserve_refactor_solve() {
    std::lock_guard<std::mutex> lock(mutex_);
    TI_ERROR_IF(closed_, "CUDA cuDSS plan is closed.");
    require_no_refactor_solve_inflight("refactorize+solve");
    TI_ERROR_IF(!analyzed_,
                "CUDA cuDSS refactorize+solve requires a prior successful "
                "analysis.");
    TI_ERROR_IF(next_refactor_solve_generation_ ==
                    (std::numeric_limits<std::uint64_t>::max)(),
                "CUDA cuDSS refactorize+solve transaction generation "
                "space exhausted.");
    refactor_solve_inflight_ = true;
    refactor_solve_provider_started_ = false;
    refactor_solve_uses_full_factorization_ = !factorized_;
    active_refactor_solve_generation_ = next_refactor_solve_generation_++;
    factorized_ = false;
    factorized_from_explicit_values_ = false;
    factorized_matrix_id_ = 0;
    factorized_pattern_version_ = 0;
    factorized_numeric_version_ = 0;
    ++factor_invalidations_;
    ++refactor_solve_attempts_;
  }

  void cancel_unsubmitted_refactor_solve() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (refactor_solve_inflight_ && !refactor_solve_provider_started_) {
      refactor_solve_inflight_ = false;
      refactor_solve_uses_full_factorization_ = false;
      active_refactor_solve_generation_ = 0;
      ++refactor_solve_failures_;
    }
  }

  void execute_reserved_refactor_solve(void *values,
                                       void *rhs,
                                       void *solution) {
    std::lock_guard<std::mutex> lock(mutex_);
    TI_ERROR_IF(closed_, "CUDA cuDSS plan is closed.");
    TI_ERROR_IF(!refactor_solve_inflight_,
                "CUDA cuDSS refactorize+solve has no reserved transaction.");
    const auto &api = provider_->api();
    auto runtime = provider_->runtime();
    try {
      require_cudss_success(api.matrix_set_values(runtime, matrix_, values),
                            "explicit matrix-values rebinding");
      bind_dense_vectors(rhs, solution);
      refactor_solve_provider_started_ = true;
      const auto factorization_phase = refactor_solve_uses_full_factorization_
                                           ? kCudssPhaseFactorization
                                           : kCudssPhaseRefactorization;
      const auto refactor_status =
          api.execute(runtime, context_, factorization_phase, config_, data_,
                      matrix_, nullptr, nullptr);
      const bool inject_failure =
          debug_fail_next_refactor_solve_after_provider_call_;
      debug_fail_next_refactor_solve_after_provider_call_ = false;
      require_cudss_success(refactor_status, "transactional refactorization");
      TI_ERROR_IF(
          inject_failure,
          "Injected CUDA cuDSS transactional refactorization failure after "
          "the provider call.");
      execute_solve();
      factorized_ = true;
      factorized_from_explicit_values_ = true;
      ++factor_generation_;
      ++refactor_solve_successes_;
    } catch (...) {
      factorized_ = false;
      factorized_from_explicit_values_ = false;
      ++refactor_solve_failures_;
      throw;
    }
  }

  void debug_fail_next_refactor_solve() {
    std::lock_guard<std::mutex> lock(mutex_);
    TI_ERROR_IF(closed_, "CUDA cuDSS plan is closed.");
    require_no_refactor_solve_inflight("failure injection");
    debug_fail_next_refactor_solve_after_provider_call_ = true;
  }

  std::uint64_t submission_retirement_token() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return refactor_solve_inflight_ ? active_refactor_solve_generation_ : 0;
  }

  void on_submission_retired(std::uint64_t token) noexcept override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (refactor_solve_inflight_ && token != 0 &&
        token == active_refactor_solve_generation_) {
      refactor_solve_inflight_ = false;
      refactor_solve_provider_started_ = false;
      refactor_solve_uses_full_factorization_ = false;
      active_refactor_solve_generation_ = 0;
      ++refactor_solve_retirements_;
    }
  }

  std::unordered_map<std::string, std::uint64_t> statistics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto &runtime_info = provider_->runtime_info();
    return {{"rows", static_cast<std::uint64_t>(rows_)},
            {"provider_abi_version", TI_FORGE_CUDSS_PROVIDER_ABI_VERSION},
            {"provider_version_major", runtime_info.version_major},
            {"provider_version_minor", runtime_info.version_minor},
            {"provider_version_patch", runtime_info.version_patch},
            {"analyzed", analyzed_ ? 1u : 0u},
            {"factorized", factorized_ ? 1u : 0u},
            {"factorized_from_explicit_values",
             factorized_from_explicit_values_ ? 1u : 0u},
            {"factor_generation", factor_generation_},
            {"factor_invalidations", factor_invalidations_},
            {"refactor_solve_inflight", refactor_solve_inflight_ ? 1u : 0u},
            {"refactor_solve_transaction_generation",
             active_refactor_solve_generation_},
            {"refactor_solve_attempts", refactor_solve_attempts_},
            {"refactor_solve_successes", refactor_solve_successes_},
            {"refactor_solve_failures", refactor_solve_failures_},
            {"refactor_solve_retirements", refactor_solve_retirements_},
            {"closed", closed_ ? 1u : 0u}};
  }

  std::unordered_map<std::string, std::int64_t> configuration() const {
    std::lock_guard<std::mutex> lock(mutex_);
    const bool estimates_valid =
        analyzed_ && memory_estimates_status_ == kCudssStatusSuccess &&
        memory_estimates_written_ >= 6 * sizeof(std::int64_t);
    const auto allocation =
        allocator_ ? allocator_->observation() : std::array<std::uint64_t, 3>{};
    return {{"configuration_abi", configuration_api_.abi_version},
            {"graph_owned", graph_owned_ ? 1 : 0},
            {"capture_parameters_device_resident", graph_owned_ ? 1 : 0},
            {"allocator_live_requested_bytes",
             static_cast<std::int64_t>(allocation[0])},
            {"allocator_peak_requested_bytes",
             static_cast<std::int64_t>(allocation[1])},
            {"sealed_allocation_rejections",
             static_cast<std::int64_t>(allocation[2])},
            {"graph_snapshot_bytes",
             static_cast<std::int64_t>(graph_snapshot_bytes_)},
            {"reordering", reordering_},
            {"solve", solve_algorithm_},
            {"memory_estimates_status", memory_estimates_status_},
            {"memory_estimates_written_bytes",
             static_cast<std::int64_t>(memory_estimates_written_)},
            {"estimated_device_persistent_bytes",
             estimates_valid ? memory_estimates_[0] : -1},
            {"estimated_device_peak_bytes",
             estimates_valid ? memory_estimates_[1] : -1},
            {"estimated_host_persistent_bytes",
             estimates_valid ? memory_estimates_[2] : -1},
            {"estimated_host_peak_bytes",
             estimates_valid ? memory_estimates_[3] : -1}};
  }

  void claim_graph_phase(int phase) {
    std::lock_guard<std::mutex> lock(mutex_);
    TI_ERROR_IF(closed_ || !graph_owned_ || !factorized_,
                "CUDA cuDSS capture requires a prepared Graph-owned snapshot.");
    TI_ERROR_IF(phase != 0 && phase != kCudssPhaseFactorization &&
                    phase != kCudssPhaseRefactorization,
                "CUDA cuDSS capture numerical phase is unsupported.");
    TI_ERROR_IF(graph_numeric_phase_ != -1 && graph_numeric_phase_ != phase,
                "A cuDSS Graph owner cannot mix fixed factors and numerical "
                "update policies; materialize an independent owner.");
    graph_numeric_phase_ = phase;
  }

  void record_graph(int phase,
                    void *values,
                    void *rhs,
                    void *solution,
                    void *stream,
                    bool update_factors = true) {
    std::lock_guard<std::mutex> lock(mutex_);
    TI_ERROR_IF(closed_ || !graph_owned_ || !factorized_ ||
                    graph_numeric_phase_ != phase,
                "CUDA cuDSS captured owner is unavailable.");
    const auto &api = provider_->api();
    auto runtime = provider_->runtime();
    auto &driver = CUDADriver::get_instance();
    // Capture joins the retained owner stream instead of rebinding warmed
    // cuDSS state to a transient parent stream. These event operations become
    // Graph dependency edges; replay does not call cuDSS or host-synchronize.
    driver.event_record(graph_fork_event_, stream);
    driver.stream_wait_event(graph_stream_, graph_fork_event_, 0);
    try {
      require_cudss_success(
          api.matrix_set_values(runtime, matrix_,
                                phase ? values : seed_buffers_[2]),
          "capture matrix binding");
      bind_dense_vectors(rhs, solution);
      if (phase && update_factors) {
        require_cudss_success(api.execute(runtime, context_, phase, config_,
                                          data_, matrix_, solution_, rhs_),
                              "captured numerical factorization");
      }
      execute_solve();
      driver.event_record(graph_join_event_, graph_stream_);
      driver.stream_wait_event(stream, graph_join_event_, 0);
    } catch (...) {
      factorized_ = false;
      throw;
    }
  }

  void destroy(bool provider_calls_safe) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
      return;
    }
    closed_ = true;
    if (!provider_calls_safe || !provider_) {
      return;
    }
    if (allocator_) {
      allocator_->seal(false);
    }
    const auto &api = provider_->api();
    auto runtime = provider_->runtime();
    if (graph_stream_) {
      warn_cudss_failure(
          CUDADriver::get_instance().stream_synchronize.call(graph_stream_),
          "Graph owner retirement");
    }
    if (solution_) {
      warn_cudss_failure(api.matrix_destroy(runtime, solution_),
                         "solution descriptor destruction");
      solution_ = nullptr;
    }
    if (rhs_) {
      warn_cudss_failure(api.matrix_destroy(runtime, rhs_),
                         "right-hand-side descriptor destruction");
      rhs_ = nullptr;
    }
    if (matrix_) {
      warn_cudss_failure(api.matrix_destroy(runtime, matrix_),
                         "CSR descriptor destruction");
      matrix_ = nullptr;
    }
    if (data_) {
      warn_cudss_failure(api.data_destroy(runtime, context_, data_),
                         "solver-data destruction");
      data_ = nullptr;
    }
    for (auto &buffer : seed_buffers_) {
      if (buffer) {
        warn_cudss_failure(CudssGraphAllocator::deallocate(
                               allocator_.get(), buffer, 0, graph_stream_),
                           "private snapshot retirement");
        buffer = nullptr;
      }
    }
    if (config_) {
      warn_cudss_failure(api.config_destroy(runtime, config_),
                         "configuration destruction");
      config_ = nullptr;
    }
    if (context_) {
      warn_cudss_failure(api.destroy(runtime, context_), "handle destruction");
      context_ = nullptr;
    }
    auto &driver = CUDADriver::get_instance();
    if (graph_fork_event_) {
      driver.event_destroy.call(graph_fork_event_);
      graph_fork_event_ = nullptr;
    }
    if (graph_join_event_) {
      driver.event_destroy.call(graph_join_event_);
      graph_join_event_ = nullptr;
    }
    if (graph_stream_) {
      driver.stream_destroy.call(graph_stream_);
      graph_stream_ = nullptr;
    }
  }

 private:
  void create_graph_snapshot(const CuSparseMatrix &matrix) {
    const std::array<std::size_t, 5> sizes = {
        (rows_ + 1) * sizeof(int), nonzeros_ * sizeof(int),
        nonzeros_ * sizeof(float), rows_ * sizeof(float),
        rows_ * sizeof(float)};
    const std::array<void *, 3> sources = {
        matrix.get_row_ptr(), matrix.get_col_ind(), matrix.get_val_ptr()};
    auto &driver = CUDADriver::get_instance();
    for (std::size_t i = 0; i < sizes.size(); ++i) {
      require_cudss_success(
          CudssGraphAllocator::allocate(allocator_.get(), &seed_buffers_[i],
                                        sizes[i], nullptr),
          "private snapshot allocation");
      graph_snapshot_bytes_ += sizes[i];
      if (i < sources.size()) {
        driver.memcpy_device_to_device(seed_buffers_[i], sources[i], sizes[i]);
      } else {
        driver.memset(seed_buffers_[i], 0, sizes[i]);
      }
    }
  }

  void prepare_graph_snapshot() {
    const auto &api = provider_->api();
    auto runtime = provider_->runtime();
    // The private copy was made on the runtime stream. Publish it to the
    // owner's stream at this cold materialization boundary only.
    CUDADriver::get_instance().stream_synchronize(nullptr);
    bind_dense_vectors(seed_buffers_[3], seed_buffers_[4]);
    require_cudss_success(api.execute(runtime, context_, kCudssPhaseAnalysis,
                                      config_, data_, matrix_, solution_, rhs_),
                          "Graph snapshot analysis");
    analyzed_ = true;
    memory_estimates_status_ = configuration_api_.analysis_memory_estimates(
        runtime, context_, data_, memory_estimates_.data(),
        sizeof(memory_estimates_), &memory_estimates_written_);
    require_cudss_success(
        api.execute(runtime, context_, kCudssPhaseFactorization, config_, data_,
                    matrix_, solution_, rhs_),
        "Graph snapshot factorization");
    execute_solve();
    // This is materialization of private data, never Graph prepare() or replay.
    // Finish initial factors before publishing a captured owner.
    CUDADriver::get_instance().stream_synchronize(graph_stream_);
    factorized_ = true;
    allocator_->seal(true);
  }

  void require_no_refactor_solve_inflight(const char *operation) const {
    TI_ERROR_IF(refactor_solve_inflight_,
                "CUDA cuDSS {} cannot run while a refactorize+solve "
                "transaction is in flight. Wait for its completion before "
                "reusing this plan.",
                operation);
  }

  void bind_dense_vectors(void *rhs, void *solution) {
    const auto &api = provider_->api();
    auto runtime = provider_->runtime();
    if (!rhs_) {
      try {
        require_cudss_success(
            api.matrix_create_dn(runtime, &rhs_,
                                 static_cast<std::int64_t>(rows_), 1,
                                 static_cast<std::int64_t>(rows_), rhs,
                                 kCudssDataTypeF32, kCudssLayoutColumnMajor),
            "right-hand-side descriptor creation");
        require_cudss_success(
            api.matrix_create_dn(runtime, &solution_,
                                 static_cast<std::int64_t>(rows_), 1,
                                 static_cast<std::int64_t>(rows_), solution,
                                 kCudssDataTypeF32, kCudssLayoutColumnMajor),
            "solution descriptor creation");
      } catch (...) {
        if (solution_) {
          warn_cudss_failure(api.matrix_destroy(runtime, solution_),
                             "solution descriptor rollback");
          solution_ = nullptr;
        }
        if (rhs_) {
          warn_cudss_failure(api.matrix_destroy(runtime, rhs_),
                             "right-hand-side descriptor rollback");
          rhs_ = nullptr;
        }
        throw;
      }
    } else {
      require_cudss_success(api.matrix_set_values(runtime, rhs_, rhs),
                            "right-hand-side rebinding");
      require_cudss_success(api.matrix_set_values(runtime, solution_, solution),
                            "solution rebinding");
    }
  }

  void execute_solve() {
    const auto &api = provider_->api();
    require_cudss_success(
        api.execute(provider_->runtime(), context_, kCudssPhaseSolve, config_,
                    data_, matrix_, solution_, rhs_),
        "solve");
  }

  void validate_analyzed_pattern(const CuSparseMatrix &matrix) const {
    TI_ERROR_IF(
        static_cast<std::size_t>(matrix.num_rows()) != rows_ ||
            matrix.num_rows() != matrix.num_cols() ||
            matrix.get_data_type() != PrimitiveType::f32 ||
            matrix.get_nnz() != static_cast<int>(analyzed_csr_col_ind_.size()),
        "CUDA cuDSS factorize() requires the same sparse pattern "
        "that was passed to analyze(); shape, dtype, or nonzero "
        "count changed.");
    const auto stats = matrix.debug_runtime_statistics();
    if ((matrix.matrix_id() == analyzed_matrix_id_ &&
         matrix.pattern_version() == analyzed_pattern_version_) ||
        (analyzed_shared_pattern_id_ != 0 &&
         stats.shared_pattern_id == analyzed_shared_pattern_id_)) {
      return;
    }
    std::vector<int> row_ptr(analyzed_csr_row_ptr_.size());
    std::vector<int> col_ind(analyzed_csr_col_ind_.size());
    CUDADriver::get_instance().memcpy_device_to_host(
        row_ptr.data(), matrix.get_row_ptr(), sizeof(int) * row_ptr.size());
    CUDADriver::get_instance().memcpy_device_to_host(
        col_ind.data(), matrix.get_col_ind(), sizeof(int) * col_ind.size());
    TI_ERROR_IF(
        row_ptr != analyzed_csr_row_ptr_ || col_ind != analyzed_csr_col_ind_,
        "CUDA cuDSS factorize() requires the same sparse pattern "
        "that was passed to analyze(); a CSR index changed.");
  }

  std::size_t rows_{0};
  TiForgeCudssConfigurationApi configuration_api_{};
  int reordering_{-1};
  int solve_algorithm_{-1};
  std::array<std::int64_t, 16> memory_estimates_{};
  std::int64_t memory_estimates_status_{-1};
  std::size_t memory_estimates_written_{0};
  std::size_t nonzeros_{0};
  const bool graph_owned_{false};
  int graph_numeric_phase_{-1};
  void *graph_stream_{nullptr};
  void *graph_fork_event_{nullptr};
  void *graph_join_event_{nullptr};
  std::unique_ptr<CudssGraphAllocator> allocator_;
  std::array<void *, 5> seed_buffers_{};
  std::size_t graph_snapshot_bytes_{0};
  void *context_{nullptr};
  void *config_{nullptr};
  void *data_{nullptr};
  void *matrix_{nullptr};
  void *rhs_{nullptr};
  void *solution_{nullptr};
  bool analyzed_{false};
  bool factorized_{false};
  bool factorized_from_explicit_values_{false};
  bool refactor_solve_inflight_{false};
  bool refactor_solve_provider_started_{false};
  bool refactor_solve_uses_full_factorization_{false};
  bool debug_fail_next_refactor_solve_after_provider_call_{false};
  bool closed_{false};
  std::vector<int> analyzed_csr_row_ptr_;
  std::vector<int> analyzed_csr_col_ind_;
  std::uint64_t analyzed_matrix_id_{0};
  std::uint64_t analyzed_pattern_version_{0};
  std::uint64_t analyzed_shared_pattern_id_{0};
  std::uint64_t factorized_matrix_id_{0};
  std::uint64_t factorized_pattern_version_{0};
  std::uint64_t factorized_numeric_version_{0};
  std::uint64_t factor_generation_{0};
  std::uint64_t factor_invalidations_{0};
  std::uint64_t refactor_solve_attempts_{0};
  std::uint64_t refactor_solve_successes_{0};
  std::uint64_t refactor_solve_failures_{0};
  std::uint64_t refactor_solve_retirements_{0};
  std::uint64_t next_refactor_solve_generation_{1};
  std::uint64_t active_refactor_solve_generation_{0};
  std::unique_ptr<CudssProviderRuntime> provider_;
  std::shared_ptr<RuntimeFaultDomain> fault_domain_;
  mutable std::mutex mutex_;
};

namespace {
class CudaCudssCaptureCommand final : public aot::CudaGraphCaptureCommand {
 public:
  CudaCudssCaptureCommand(Program *program,
                          std::uint64_t handle,
                          int phase,
                          const std::vector<aot::Arg> &arguments)
      : program_(program), phase_(phase), arguments_(arguments) {
    TI_ERROR_IF(!program || program->compile_config().arch != Arch::cuda,
                "cuDSS capture requires a CUDA Program.");
    const std::size_t offset = phase ? 1 : 0;
    TI_ERROR_IF(
        arguments.size() < offset + 2 || (arguments.size() - offset) % 2,
        "cuDSS capture bindings do not match the numerical phase.");
    for (std::size_t i = 0; i < arguments.size(); ++i) {
      const auto &arg = arguments[i];
      TI_ERROR_IF(arg.tag != aot::ArgKind::kNdarray ||
                      arg.dtype_id != PrimitiveTypeID::f32 ||
                      arg.field_dim != 1 || !arg.element_shape.empty() ||
                      arg.name.empty(),
                  "cuDSS capture requires named scalar f32 vectors.");
      for (std::size_t j = 0; j < i; ++j) {
        TI_ERROR_IF(arg.name == arguments[j].name,
                    "cuDSS capture binding names must be distinct.");
      }
    }
    owner_ = program->retain_cuda_cudss_capture_plan(handle);
    auto &driver = CUDADriver::get_instance();
    TI_ERROR_IF(
        !driver.stream_get_capture_info_v2.available() ||
            !driver.graph_node_get_type.available() ||
            !driver.graph_memcpy_node_get_params.available() ||
            !driver.graph_memcpy_node_set_params.available(),
        "cuDSS capture requires immutable host-input snapshot support.");
    owner_->claim_graph_phase(phase);
  }

  const char *kind() const override {
    return "cudss_retained_solve_f32";
  }
  Program *program() const override {
    return program_;
  }
  bool supports_binding_frames() const override {
    return true;
  }
  std::shared_ptr<void> retain_binding_frame_plan(Program &) override {
    return owner_;
  }
  std::shared_ptr<aot::CudaGraphCaptureResources> take_capture_resources() override {
    return std::exchange(capture_inputs_, {});
  }

  bool supports(const std::unordered_map<std::string, aot::IValue> &args,
                Program &program) const override {
    if (&program != program_)
      return false;
    for (std::size_t i = 0; i < arguments_.size(); ++i) {
      auto *value = array(i, args);
      if (!value)
        return false;
      for (std::size_t j = 0; j < i; ++j) {
        if (value->get_device_allocation() ==
            array(j, args)->get_device_allocation())
          return false;
      }
    }
    return true;
  }

  void prepare(const std::unordered_map<std::string, aot::IValue> &args,
               Program &program) override {
    TI_ERROR_IF(!supports(args, program),
                "cuDSS capture bindings are incompatible.");
    // Owner creation already warmed a private numerical snapshot. Never run
    // mathematical work with caller bindings before the first Graph submission.
  }

  void record(const std::unordered_map<std::string, aot::IValue> &args,
              Program &program,
              void *stream) override {
    TI_ERROR_IF(!supports(args, program),
                "cuDSS capture bindings are incompatible.");
    capture_inputs_.reset();
    std::uint32_t capture_status = 0;
    CUgraph graph = nullptr;
    auto &driver = CUDADriver::get_instance();
    driver.stream_get_capture_info_v2(stream, &capture_status, nullptr, &graph,
                                      nullptr, nullptr);
    const auto pointer = [&](std::size_t i) {
      return reinterpret_cast<void *>(
          program.get_ndarray_data_ptr_as_int(array(i, args)));
    };
    if (capture_status == 1) {
      capture_inputs_ = owner_->capture_inputs();
    }
    const std::size_t offset = phase_ ? 1 : 0;
    for (std::size_t i = offset; i < arguments_.size(); i += 2) {
      const auto previous =
          capture_status == 1 ? graph_nodes(graph) : std::vector<void *>{};
      const std::unordered_set<void *> previous_nodes(previous.begin(),
                                                      previous.end());
      owner_->record_graph(phase_, phase_ ? pointer(0) : nullptr, pointer(i),
                           pointer(i + 1), stream, i == offset);
      if (capture_status != 1)
        continue;
      // Freeze each RHS's vendor staging before recording the next one.
      // One numerical update serves every RHS in this complete region.
      for (auto *node : graph_nodes(graph)) {
        if (previous_nodes.count(node))
          continue;
        std::uint32_t type = 0;
        driver.graph_node_get_type(node, &type);
        if (type != 1 /* memcpy */)
          continue;
        CUDA_MEMCPY3D params{};
        driver.graph_memcpy_node_get_params(node, &params);
        TI_ERROR_IF(params.dstMemoryType != CU_MEMORYTYPE_DEVICE,
                    "cuDSS capture contains a non-device copy destination.");
        if (params.srcMemoryType == CU_MEMORYTYPE_DEVICE)
          continue;
        TI_ERROR_IF(params.srcMemoryType != 1 /* host */ || !params.srcHost ||
                        params.Height != 1 || params.Depth != 1 ||
                        params.srcY || params.srcZ || params.srcLOD,
                    "cuDSS capture host inputs require contiguous 1D storage.");
        capture_inputs_->snapshot(node, params);
      }
    }
  }

 private:
  static std::vector<void *> graph_nodes(CUgraph graph) {
    auto &driver = CUDADriver::get_instance();
    std::size_t count = 0;
    driver.graph_get_nodes(graph, nullptr, &count);
    std::vector<void *> nodes(count);
    driver.graph_get_nodes(graph, nodes.data(), &count);
    nodes.resize(count);
    return nodes;
  }
  Ndarray *array(
      std::size_t index,
      const std::unordered_map<std::string, aot::IValue> &args) const {
    const auto found = args.find(arguments_[index].name);
    if (found == args.end() || found->second.tag != aot::ArgKind::kNdarray)
      return nullptr;
    auto *array = reinterpret_cast<Ndarray *>(found->second.val);
    if (!array || array->owning_program() != program_ ||
        array->get_element_data_type() != PrimitiveType::f32 ||
        !array->get_element_shape().empty() || array->shape.size() != 1 ||
        array->get_nelement() !=
            (phase_ && index == 0 ? owner_->nonzeros() : owner_->rows()))
      return nullptr;
    return array;
  }
  Program *program_;
  const int phase_;
  const std::vector<aot::Arg> arguments_;
  std::shared_ptr<CudaCudssPlan> owner_;
  std::shared_ptr<CudssCaptureInputs> capture_inputs_;
};
}  // namespace

std::shared_ptr<aot::CudaGraphCaptureCommand> make_cuda_cudss_capture_command(
    Program *program,
    std::uint64_t handle,
    int numeric_phase,
    const std::vector<aot::Arg> &arguments) {
  return std::make_shared<CudaCudssCaptureCommand>(program, handle,
                                                   numeric_phase, arguments);
}

std::shared_ptr<CudaCudssPlan> Program::retain_cuda_cudss_capture_plan(
    std::uint64_t handle) {
  std::lock_guard<std::mutex> lock(cuda_cudss_plan_mutex_);
  const auto found = cuda_cudss_graph_plans_.find(handle);
  TI_ERROR_IF(found == cuda_cudss_graph_plans_.end(),
              "CUDA cuDSS Graph owner is stale, closed, or not Graph-owned.");
  return found->second;
}

std::uint64_t Program::create_cuda_cudss_plan(
    SparseMatrix *matrix,
    int matrix_type,
    int matrix_view,
    const std::string &adapter_path,
    const std::string &runtime_library_path) {
  return create_cuda_cudss_configured_plan(
      matrix, matrix_type, matrix_view, adapter_path, runtime_library_path, {});
}

std::uint64_t Program::create_cuda_cudss_configured_plan(
    SparseMatrix *matrix,
    int matrix_type,
    int matrix_view,
    const std::string &adapter_path,
    const std::string &runtime_library_path,
    const std::vector<int> &configuration,
    bool graph_owned) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA cuDSS plans require the CUDA backend.");
  TI_ERROR_IF(
      !CUDADriver::get_instance_without_context().nvidia_extensions_available(),
      "CUDA cuDSS requires the NVIDIA CUDA provider.");
  const auto &csr = require_cudss_matrix(matrix, this);
  auto cuda_submission_guard =
      CUDAContext::get_instance().get_submission_lock_guard();
  auto context_guard = CUDAContext::get_instance().get_guard();
  auto plan = std::make_shared<CudaCudssPlan>(
      csr, matrix_type, matrix_view, adapter_path, runtime_library_path,
      runtime_fault_domain_, configuration, graph_owned);
  std::lock_guard<std::mutex> lock(cuda_cudss_plan_mutex_);
  TI_ERROR_IF(next_cuda_cudss_plan_handle_ == 0,
              "CUDA cuDSS plan handle space exhausted.");
  const auto handle = next_cuda_cudss_plan_handle_++;
  auto &owners = graph_owned ? cuda_cudss_graph_plans_ : cuda_cudss_plans_;
  owners.emplace(handle, std::move(plan));
  return handle;
}

void Program::cuda_cudss_analyze(std::uint64_t handle, SparseMatrix *matrix) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  std::shared_ptr<CudaCudssPlan> plan;
  {
    std::lock_guard<std::mutex> lock(cuda_cudss_plan_mutex_);
    const auto found = cuda_cudss_plans_.find(handle);
    TI_ERROR_IF(found == cuda_cudss_plans_.end(),
                "CUDA cuDSS plan handle is stale or closed.");
    plan = found->second;
  }
  auto cuda_submission_guard =
      CUDAContext::get_instance().get_submission_lock_guard();
  auto context_guard = CUDAContext::get_instance().get_guard();
  const auto &csr = require_cudss_matrix(matrix, this);
  plan->analyze(csr);
  pin_cuda_provider_plan(plan);
  mark_runtime_submission_pending();
}

void Program::cuda_cudss_factorize(std::uint64_t handle,
                                   SparseMatrix *matrix,
                                   bool refactorize) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  std::shared_ptr<CudaCudssPlan> plan;
  {
    std::lock_guard<std::mutex> lock(cuda_cudss_plan_mutex_);
    const auto found = cuda_cudss_plans_.find(handle);
    TI_ERROR_IF(found == cuda_cudss_plans_.end(),
                "CUDA cuDSS plan handle is stale or closed.");
    plan = found->second;
  }
  auto cuda_submission_guard =
      CUDAContext::get_instance().get_submission_lock_guard();
  auto context_guard = CUDAContext::get_instance().get_guard();
  const auto &csr = require_cudss_matrix(matrix, this);
  plan->factorize(csr, refactorize);
  pin_cuda_provider_plan(plan);
  mark_runtime_submission_pending();
}

std::size_t Program::cuda_cudss_solve(std::uint64_t handle,
                                      SparseMatrix *matrix,
                                      Ndarray *rhs,
                                      Ndarray *solution) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA cuDSS solve requires the CUDA backend.");
  std::shared_ptr<CudaCudssPlan> plan;
  {
    std::lock_guard<std::mutex> lock(cuda_cudss_plan_mutex_);
    const auto found = cuda_cudss_plans_.find(handle);
    TI_ERROR_IF(found == cuda_cudss_plans_.end(),
                "CUDA cuDSS plan handle is stale or closed.");
    plan = found->second;
  }
  const auto stats = plan->statistics();
  const auto rows = static_cast<std::size_t>(stats.at("rows"));
  validate_cudss_vector(rhs, rows, "right-hand side", this);
  validate_cudss_vector(solution, rows, "solution", this);
  TI_ERROR_IF(rhs->get_device_allocation() == solution->get_device_allocation(),
              "The first CUDA cuDSS slice requires distinct right-hand-side "
              "and solution allocations.");
  auto cuda_submission_guard =
      CUDAContext::get_instance().get_submission_lock_guard();
  auto context_guard = CUDAContext::get_instance().get_guard();
  const auto &csr = require_cudss_matrix(matrix, this);
  auto *rhs_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(rhs));
  auto *solution_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(solution));
  TI_ERROR_IF(!rhs_ptr || !solution_ptr,
              "CUDA cuDSS received a null dense-vector device pointer.");
  plan->solve(csr, rhs_ptr, solution_ptr);
  pin_cuda_provider_plan(plan);
  auto leases = acquire_ndarray_leases({rhs, solution});
  pin_ndarray_launch_leases(leases);
  mark_runtime_submission_pending();
  return 0;
}

std::size_t Program::cuda_cudss_refactor_solve(std::uint64_t handle,
                                               Ndarray *values,
                                               Ndarray *rhs,
                                               Ndarray *solution) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA cuDSS refactorize+solve requires the CUDA backend.");
  std::shared_ptr<CudaCudssPlan> plan;
  {
    std::lock_guard<std::mutex> lock(cuda_cudss_plan_mutex_);
    const auto found = cuda_cudss_plans_.find(handle);
    TI_ERROR_IF(found == cuda_cudss_plans_.end(),
                "CUDA cuDSS plan handle is stale or closed.");
    plan = found->second;
  }
  validate_cudss_values(values, plan->nonzeros(), this);
  validate_cudss_vector(rhs, plan->rows(), "right-hand side", this);
  validate_cudss_vector(solution, plan->rows(), "solution", this);
  const auto values_allocation = values->get_device_allocation();
  const auto rhs_allocation = rhs->get_device_allocation();
  const auto solution_allocation = solution->get_device_allocation();
  TI_ERROR_IF(values_allocation == rhs_allocation ||
                  values_allocation == solution_allocation ||
                  rhs_allocation == solution_allocation,
              "CUDA cuDSS refactorize+solve values, right-hand side, and "
              "solution allocations must be distinct.");
  auto leases = acquire_ndarray_leases({values, rhs, solution});
  auto cuda_submission_guard =
      CUDAContext::get_instance().get_submission_lock_guard();
  auto context_guard = CUDAContext::get_instance().get_guard();
  auto *values_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values));
  auto *rhs_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(rhs));
  auto *solution_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(solution));
  TI_ERROR_IF(!values_ptr || !rhs_ptr || !solution_ptr,
              "CUDA cuDSS refactorize+solve received a null device pointer.");

  plan->reserve_refactor_solve();
  try {
    pin_cuda_provider_plan(plan);
    pin_ndarray_launch_leases(leases);
    mark_runtime_submission_pending();
  } catch (...) {
    plan->cancel_unsubmitted_refactor_solve();
    throw;
  }
  plan->execute_reserved_refactor_solve(values_ptr, rhs_ptr, solution_ptr);
  return 0;
}

std::unordered_map<std::string, std::uint64_t>
Program::cuda_cudss_plan_statistics(std::uint64_t handle) {
  std::lock_guard<std::mutex> lock(cuda_cudss_plan_mutex_);
  const auto found = cuda_cudss_plans_.find(handle);
  TI_ERROR_IF(found == cuda_cudss_plans_.end(),
              "CUDA cuDSS plan handle is stale or closed.");
  return found->second->statistics();
}

std::unordered_map<std::string, std::int64_t>
Program::cuda_cudss_plan_configuration(std::uint64_t handle) {
  std::lock_guard<std::mutex> lock(cuda_cudss_plan_mutex_);
  const auto graph = cuda_cudss_graph_plans_.find(handle);
  if (graph != cuda_cudss_graph_plans_.end()) {
    return graph->second->configuration();
  }
  const auto found = cuda_cudss_plans_.find(handle);
  TI_ERROR_IF(found == cuda_cudss_plans_.end(),
              "CUDA cuDSS plan handle is stale or closed.");
  return found->second->configuration();
}

void Program::debug_cuda_cudss_fail_next_refactor_solve(std::uint64_t handle) {
  std::shared_ptr<CudaCudssPlan> plan;
  {
    std::lock_guard<std::mutex> lock(cuda_cudss_plan_mutex_);
    const auto found = cuda_cudss_plans_.find(handle);
    TI_ERROR_IF(found == cuda_cudss_plans_.end(),
                "CUDA cuDSS plan handle is stale or closed.");
    plan = found->second;
  }
  plan->debug_fail_next_refactor_solve();
}

void Program::destroy_cuda_cudss_plan(std::uint64_t handle) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  std::shared_ptr<CudaCudssPlan> plan;
  {
    std::lock_guard<std::mutex> lock(cuda_cudss_plan_mutex_);
    auto *owners = cuda_cudss_graph_plans_.count(handle)
                       ? &cuda_cudss_graph_plans_
                       : &cuda_cudss_plans_;
    const auto found = owners->find(handle);
    if (found == owners->end()) {
      return;
    }
    plan = std::move(found->second);
    owners->erase(found);
  }
  // RuntimeCompletion owns any in-flight reference. Destruction therefore
  // occurs immediately only when no submitted phase still uses this plan.
  plan.reset();
}

void Program::cuda_clear_cudss_plans() {
  std::vector<std::shared_ptr<CudaCudssPlan>> plans;
  {
    std::lock_guard<std::mutex> lock(cuda_cudss_plan_mutex_);
    plans.reserve(cuda_cudss_plans_.size() + cuda_cudss_graph_plans_.size());
    for (auto &[handle, plan] : cuda_cudss_plans_) {
      plans.push_back(std::move(plan));
    }
    cuda_cudss_plans_.clear();
    for (auto &[handle, plan] : cuda_cudss_graph_plans_) {
      plans.push_back(std::move(plan));
    }
    cuda_cudss_graph_plans_.clear();
  }
  const bool provider_calls_safe = !runtime_has_fatal_fault();
  if (provider_calls_safe && !plans.empty()) {
    auto cuda_submission_guard =
        CUDAContext::get_instance().get_submission_lock_guard();
    auto context_guard = CUDAContext::get_instance().get_guard();
    for (auto &plan : plans) {
      plan->destroy(true);
    }
  } else {
    for (auto &plan : plans) {
      plan->destroy(false);
    }
  }
}

}  // namespace taichi::lang

#else

namespace taichi::lang {

std::shared_ptr<aot::CudaGraphCaptureCommand> make_cuda_cudss_capture_command(
    Program *,
    std::uint64_t,
    int,
    const std::vector<aot::Arg> &) {
  TI_ERROR("CUDA cuDSS requires TI_WITH_CUDA=ON.");
}

std::shared_ptr<CudaCudssPlan> Program::retain_cuda_cudss_capture_plan(
    std::uint64_t) {
  TI_ERROR("CUDA cuDSS requires TI_WITH_CUDA=ON.");
}

std::uint64_t Program::create_cuda_cudss_configured_plan(
    SparseMatrix *,
    int,
    int,
    const std::string &,
    const std::string &,
    const std::vector<int> &,
    bool) {
  TI_ERROR("CUDA cuDSS requires TI_WITH_CUDA=ON.");
}

std::unordered_map<std::string, std::int64_t>
Program::cuda_cudss_plan_configuration(std::uint64_t) {
  TI_ERROR("CUDA cuDSS requires TI_WITH_CUDA=ON.");
}

std::uint64_t Program::create_cuda_cudss_plan(SparseMatrix *,
                                              int,
                                              int,
                                              const std::string &,
                                              const std::string &) {
  TI_ERROR("CUDA cuDSS requires TI_WITH_CUDA=ON.");
}

void Program::cuda_cudss_analyze(std::uint64_t, SparseMatrix *) {
  TI_ERROR("CUDA cuDSS requires TI_WITH_CUDA=ON.");
}

void Program::cuda_cudss_factorize(std::uint64_t, SparseMatrix *, bool) {
  TI_ERROR("CUDA cuDSS requires TI_WITH_CUDA=ON.");
}

std::size_t Program::cuda_cudss_solve(std::uint64_t,
                                      SparseMatrix *,
                                      Ndarray *,
                                      Ndarray *) {
  TI_ERROR("CUDA cuDSS requires TI_WITH_CUDA=ON.");
}

std::size_t Program::cuda_cudss_refactor_solve(std::uint64_t,
                                               Ndarray *,
                                               Ndarray *,
                                               Ndarray *) {
  TI_ERROR("CUDA cuDSS requires TI_WITH_CUDA=ON.");
}

std::unordered_map<std::string, std::uint64_t>
Program::cuda_cudss_plan_statistics(std::uint64_t) {
  TI_ERROR("CUDA cuDSS requires TI_WITH_CUDA=ON.");
}

void Program::debug_cuda_cudss_fail_next_refactor_solve(std::uint64_t) {
  TI_ERROR("CUDA cuDSS requires TI_WITH_CUDA=ON.");
}

void Program::destroy_cuda_cudss_plan(std::uint64_t) {
}

void Program::cuda_clear_cudss_plans() {
}

}  // namespace taichi::lang

#endif
