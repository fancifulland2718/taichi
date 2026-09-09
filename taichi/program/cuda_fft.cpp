#include "taichi/program/program.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(TI_WITH_CUDA)
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_driver.h"

namespace taichi::lang {
namespace {

constexpr int kCufftC2C = 0x29;
constexpr int kCufftR2C = 0x2a;
constexpr int kCufftC2R = 0x2c;
constexpr int kCufftForward = -1;
constexpr int kCufftInverse = 1;

enum class CufftTransformKind : int {
  c2c = 0,
  r2c = 1,
  c2r = 2,
};

struct CufftScalarCounts {
  std::size_t input{0};
  std::size_t output{0};
};

struct CufftPlanDescriptor {
  std::vector<int> dimensions;
  std::vector<int> input_embed;
  int input_stride{1};
  int input_distance{0};
  std::vector<int> output_embed;
  int output_stride{1};
  int output_distance{0};
  int batch_count{1};
  CufftTransformKind transform_kind{CufftTransformKind::c2c};
  bool separable{false};
  bool cross_batch{false};
  std::string store_callback_ir;
  std::string store_callback_symbol;
};

CufftTransformKind validate_transform_kind(int transform_kind) {
  TI_ERROR_IF(transform_kind < static_cast<int>(CufftTransformKind::c2c) ||
                  transform_kind > static_cast<int>(CufftTransformKind::c2r),
              "CUDA cuFFT transform_kind must be C2C, R2C, or C2R.");
  return static_cast<CufftTransformKind>(transform_kind);
}

CufftPlanDescriptor make_cufft_plan_1d_descriptor(
    std::size_t length,
    std::size_t batch_count,
    CufftTransformKind transform_kind) {
  const auto max_int = static_cast<std::size_t>(
      (std::numeric_limits<int>::max)());
  TI_ERROR_IF(length == 0 || length > max_int,
              "CUDA cuFFT length must be in [1, INT_MAX].");
  TI_ERROR_IF(batch_count == 0 || batch_count > max_int,
              "CUDA cuFFT batch_count must be in [1, INT_MAX].");
  const int real_length = static_cast<int>(length);
  const int hermitian_length = static_cast<int>(length / 2 + 1);
  const int input_length = transform_kind == CufftTransformKind::c2r
                               ? hermitian_length
                               : real_length;
  const int output_length = transform_kind == CufftTransformKind::r2c
                                ? hermitian_length
                                : real_length;
  return {{real_length},
          {input_length},
          1,
          input_length,
          {output_length},
          1,
          output_length,
          static_cast<int>(batch_count),
          transform_kind};
}

std::size_t checked_multiply(std::size_t left,
                             std::size_t right,
                             const char *description) {
  const auto max_size = (std::numeric_limits<std::size_t>::max)();
  TI_ERROR_IF(right != 0 && left > max_size / right,
              "CUDA cuFFT {} overflow.", description);
  return left * right;
}

std::size_t checked_add(std::size_t left,
                        std::size_t right,
                        const char *description) {
  TI_ERROR_IF(left > (std::numeric_limits<std::size_t>::max)() - right,
              "CUDA cuFFT {} overflow.", description);
  return left + right;
}

std::size_t cufft_element_span(const std::vector<int> &logical_dimensions,
                               const std::vector<int> &embed,
                               int stride,
                               const char *name) {
  TI_ERROR_IF(logical_dimensions.empty() ||
                  logical_dimensions.size() != embed.size(),
              "CUDA cuFFT {} rank does not match the transform rank.", name);
  TI_ERROR_IF(stride <= 0, "CUDA cuFFT {} stride must be positive.", name);
  std::size_t offset = 0;
  for (std::size_t axis = 0; axis < logical_dimensions.size(); ++axis) {
    TI_ERROR_IF(logical_dimensions[axis] <= 0 || embed[axis] <= 0 ||
                    embed[axis] < logical_dimensions[axis],
                "CUDA cuFFT {} embed must cover every logical dimension.",
                name);
    if (axis != 0) {
      offset = checked_multiply(offset, static_cast<std::size_t>(embed[axis]),
                                "layout offset");
    }
    const auto increment =
        static_cast<std::size_t>(logical_dimensions[axis] - 1);
    offset = checked_add(offset, increment, "layout offset");
  }
  return checked_add(
      checked_multiply(offset, static_cast<std::size_t>(stride),
                       "layout stride"),
      1, "layout span");
}

std::vector<int> cufft_input_logical_dimensions(
    const CufftPlanDescriptor &descriptor) {
  auto result = descriptor.dimensions;
  if (descriptor.transform_kind == CufftTransformKind::c2r) {
    result.back() = result.back() / 2 + 1;
  }
  return result;
}

std::vector<int> cufft_output_logical_dimensions(
    const CufftPlanDescriptor &descriptor) {
  auto result = descriptor.dimensions;
  if (descriptor.transform_kind == CufftTransformKind::r2c) {
    result.back() = result.back() / 2 + 1;
  }
  return result;
}

CufftScalarCounts cufft_scalar_counts(
    const CufftPlanDescriptor &descriptor) {
  const auto rank = descriptor.dimensions.size();
  TI_ERROR_IF(rank == 0 || rank > 3,
              "CUDA cuFFT rank must be in [1, 3].");
  TI_ERROR_IF(descriptor.batch_count <= 0,
              "CUDA cuFFT batch_count must be positive.");
  for (int dimension : descriptor.dimensions) {
    TI_ERROR_IF(dimension <= 0,
                "CUDA cuFFT dimensions must be positive.");
  }
  const auto input_span = cufft_element_span(
      cufft_input_logical_dimensions(descriptor), descriptor.input_embed,
      descriptor.input_stride, "input");
  const auto output_span = cufft_element_span(
      cufft_output_logical_dimensions(descriptor), descriptor.output_embed,
      descriptor.output_stride, "output");
  const auto max_int =
      static_cast<std::size_t>((std::numeric_limits<int>::max)());
  TI_ERROR_IF(input_span > max_int || output_span > max_int,
              "CUDA cuFFT transform storage span exceeds INT_MAX.");
  const auto disjoint_batches = [&](const std::vector<int> &logical,
                                    int stride, int distance,
                                    std::size_t span) {
    if (distance <= 0) {
      return false;
    }
    if (descriptor.batch_count == 1 ||
        static_cast<std::size_t>(distance) >= span) {
      return true;
    }
    if (logical.size() != 1) {
      return false;
    }
    // For rank-one batches, collisions require ds*stride == db*distance.
    // The smallest positive ds/db pair is distance/gcd, stride/gcd.
    const int divisor = std::gcd(stride, distance);
    return logical.front() <= distance / divisor ||
           descriptor.batch_count <= stride / divisor;
  };
  TI_ERROR_IF(!disjoint_batches(cufft_input_logical_dimensions(descriptor),
                               descriptor.input_stride,
                               descriptor.input_distance, input_span) ||
                  !disjoint_batches(cufft_output_logical_dimensions(descriptor),
                                    descriptor.output_stride,
                                    descriptor.output_distance, output_span),
              "CUDA cuFFT batch distance must not overlap transform storage.");
  const auto batch_offset = static_cast<std::size_t>(descriptor.batch_count - 1);
  const auto input_elements = checked_add(
      checked_multiply(batch_offset,
                       static_cast<std::size_t>(descriptor.input_distance),
                       "input batch distance"),
      input_span, "input storage size");
  const auto output_elements = checked_add(
      checked_multiply(batch_offset,
                       static_cast<std::size_t>(descriptor.output_distance),
                       "output batch distance"),
      output_span, "output storage size");
  const std::size_t input_components =
      descriptor.transform_kind == CufftTransformKind::r2c ? 1 : 2;
  const std::size_t output_components =
      descriptor.transform_kind == CufftTransformKind::c2r ? 1 : 2;
  return {checked_multiply(input_elements, input_components,
                           "input scalar count"),
          checked_multiply(output_elements, output_components,
                           "output scalar count")};
}

std::string cufft_plan_cache_key(const CufftPlanDescriptor &descriptor) {
  std::ostringstream key;
  const auto append = [&key](const std::vector<int> &values) {
    key << values.size() << ':';
    for (int value : values) {
      key << value << ',';
    }
  };
  append(descriptor.dimensions);
  append(descriptor.input_embed);
  key << descriptor.input_stride << ':' << descriptor.input_distance << ':';
  append(descriptor.output_embed);
  key << descriptor.output_stride << ':' << descriptor.output_distance << ':'
      << descriptor.batch_count << ':'
      << static_cast<int>(descriptor.transform_kind);
  if (descriptor.separable) {
    key << (descriptor.cross_batch ? ":row-batch-cross-batch-columns"
                                   : ":row-batch-column-inplace");
  }
  if (!descriptor.store_callback_ir.empty()) {
    // Binary-safe, exact identity: never alias a callback plan with a plain
    // transform or with different callback math. This is a cold weak cache.
    key << ":store-lto:" << descriptor.store_callback_symbol.size() << ':'
        << descriptor.store_callback_symbol << ':' << descriptor.store_callback_ir;
  }
  return key.str();
}

}  // namespace

class CudaFftPlan final : public CudaProviderCompletionResource {
 public:
  CudaFftPlan(CufftPlanDescriptor descriptor,
              bool use_plan_many,
              std::shared_ptr<RuntimeFaultDomain> fault_domain)
      : descriptor_(std::move(descriptor)),
        scalar_counts_(cufft_scalar_counts(descriptor_)),
        fault_domain_(std::move(fault_domain)) {
    if (descriptor_.separable) {
      TI_ERROR_IF(descriptor_.dimensions.size() != 2 ||
                      descriptor_.transform_kind != CufftTransformKind::c2c ||
                      descriptor_.input_embed != descriptor_.dimensions ||
                      descriptor_.output_embed != descriptor_.dimensions ||
                      descriptor_.input_stride != 1 ||
                      descriptor_.output_stride != 1,
                  "Separable cuFFT requires compact two-dimensional C2C.");
      const int height = descriptor_.dimensions[0];
      const int width = descriptor_.dimensions[1];
      const auto plane = checked_multiply(height, width, "separable plane");
      const auto row_batches = checked_multiply(
          descriptor_.batch_count, height, "separable row batch");
      TI_ERROR_IF(plane > (std::numeric_limits<int>::max)() ||
                      row_batches > (std::numeric_limits<int>::max)() ||
                      descriptor_.input_distance != static_cast<int>(plane) ||
                      descriptor_.output_distance != static_cast<int>(plane),
                  "Separable cuFFT batch layout exceeds its compact contract.");
      CufftPlanDescriptor rows{{width}, {width}, 1, width,
                               {width}, 1, width,
                               static_cast<int>(row_batches),
                               CufftTransformKind::c2c};
      CufftPlanDescriptor columns{{height}, {height}, width, 1,
                                  {height}, width, 1, width,
                                  CufftTransformKind::c2c};
      if (descriptor_.cross_batch) {
        // At column c: input[b * H * W + r * W + c]. Batch across
        // independent images, rather than launching once per image.
        columns.input_distance = static_cast<int>(plane);
        columns.output_distance = static_cast<int>(plane);
        columns.batch_count = descriptor_.batch_count;
      }
      children_.push_back(std::make_unique<CudaFftPlan>(
          std::move(rows), true, fault_domain_));
      children_.push_back(std::make_unique<CudaFftPlan>(
          std::move(columns), true, fault_domain_));
      workspace_bytes_ = checked_add(children_[0]->workspace_bytes(),
                                     children_[1]->workspace_bytes(),
                                     "separable workspace");
      return;
    }
    auto &driver = CUFFTDriver::get_instance();
    TI_ERROR_IF(!driver.load_cufft(),
                "CUDA cuFFT could not load a compatible shared library and "
                "the required basic-plan symbols.");
    int handle = 0;
    const int transform_type =
        descriptor_.transform_kind == CufftTransformKind::c2c
            ? kCufftC2C
            : descriptor_.transform_kind == CufftTransformKind::r2c
                  ? kCufftR2C
                  : kCufftC2R;
    int plan_status = 0;
    if (!descriptor_.store_callback_ir.empty()) {
      TI_ERROR_IF(!driver.create.available() ||
                      !driver.make_plan_many.available() ||
                      !driver.set_jit_callback.available(),
                  "CUDA cuFFT LTO store callbacks are unavailable; basic FFT "
                  "plans remain supported.");
      plan_status = driver.create.call(&handle);
      if (plan_status == 0 && handle != 0) {
        // cufftXtCallbackType::CUFFT_CB_ST_COMPLEX = 4 (stable cuFFT C ABI).
        plan_status = driver.set_jit_callback.call(
            handle, descriptor_.store_callback_symbol.c_str(),
            descriptor_.store_callback_ir.data(),
            descriptor_.store_callback_ir.size(), 4, nullptr);
        if (plan_status == 0) {
          plan_status = driver.make_plan_many.call(
              handle, static_cast<int>(descriptor_.dimensions.size()),
              descriptor_.dimensions.data(), descriptor_.input_embed.data(),
              descriptor_.input_stride, descriptor_.input_distance,
              descriptor_.output_embed.data(), descriptor_.output_stride,
              descriptor_.output_distance, transform_type,
              descriptor_.batch_count, &workspace_bytes_);
        }
      }
      if (plan_status != 0 || handle == 0) {
        if (handle != 0) {
          driver.destroy.call(handle);
        }
        TI_ERROR("CUDA cuFFT LTO store callback preparation failed (status {}). "
                 "Check compatible cuFFT, NVRTC and nvJitLink runtime libraries; "
                 "the callback was not replaced by a plain transform.", plan_status);
      }
    } else if (use_plan_many) {
      plan_status = driver.plan_many.call(
          &handle, static_cast<int>(descriptor_.dimensions.size()),
          descriptor_.dimensions.data(), descriptor_.input_embed.data(),
          descriptor_.input_stride, descriptor_.input_distance,
          descriptor_.output_embed.data(), descriptor_.output_stride,
          descriptor_.output_distance, transform_type,
          descriptor_.batch_count);
    } else {
      plan_status = driver.plan_1d.call(
          &handle, descriptor_.dimensions.front(), transform_type,
          descriptor_.batch_count);
    }
    TI_ERROR_IF(plan_status != 0 || handle == 0,
                "CUDA cuFFT failed to create a plan (status {}).", plan_status);
    handle_ = handle;
    const auto size_status = driver.get_size.call(handle_, &workspace_bytes_);
    if (size_status != 0) {
      const auto destroy_status = driver.destroy.call(handle_);
      handle_ = 0;
      TI_WARN_IF(destroy_status != 0,
                 "CUDA cuFFT cleanup after workspace query failure returned "
                 "status {}.",
                 destroy_status);
      TI_ERROR("CUDA cuFFT failed to query plan workspace size (status {}).",
               size_status);
    }
    const auto stream_status = driver.set_stream.call(handle_, nullptr);
    if (stream_status != 0) {
      const auto destroy_status = driver.destroy.call(handle_);
      handle_ = 0;
      TI_WARN_IF(destroy_status != 0,
                 "CUDA cuFFT cleanup after stream-binding failure returned "
                 "status {}.",
                 destroy_status);
      TI_ERROR("CUDA cuFFT failed to bind the runtime default stream "
               "(status {}).",
               stream_status);
    }
  }

  ~CudaFftPlan() {
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

  CudaFftPlan(const CudaFftPlan &) = delete;
  CudaFftPlan &operator=(const CudaFftPlan &) = delete;

  std::size_t length() const {
    return static_cast<std::size_t>(descriptor_.dimensions.front());
  }

  std::size_t batch_count() const {
    return static_cast<std::size_t>(descriptor_.batch_count);
  }

  CufftTransformKind transform_kind() const {
    return descriptor_.transform_kind;
  }

  const CufftPlanDescriptor &descriptor() const {
    return descriptor_;
  }

  const CufftScalarCounts &scalar_counts() const {
    return scalar_counts_;
  }

  std::size_t workspace_bytes() const {
    return workspace_bytes_;
  }

  void execute(void *input,
               void *output,
               int direction,
               CUstream stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (descriptor_.separable) {
      TI_ERROR_IF(children_.size() != 2, "Separable cuFFT plan is closed.");
      children_[0]->execute(input, output, direction, stream);
      const auto stride = descriptor_.cross_batch
                              ? 2
                              : scalar_counts_.output / descriptor_.batch_count;
      const int count = descriptor_.cross_batch ? descriptor_.dimensions[1]
                                                : descriptor_.batch_count;
      for (int index = 0; index < count; ++index) {
        auto *plane = static_cast<float *>(output) + index * stride;
        children_[1]->execute(plane, plane, direction, stream);
      }
      return;
    }
    TI_ERROR_IF(handle_ == 0, "CUDA cuFFT plan is closed.");
    auto &driver = CUFFTDriver::get_instance();
    const auto stream_status = driver.set_stream.call(handle_, stream);
    TI_ERROR_IF(stream_status != 0,
                "CUDA cuFFT failed to bind the execution stream (status {}).",
                stream_status);
    int status = 0;
    if (descriptor_.transform_kind == CufftTransformKind::c2c) {
      TI_ERROR_IF(direction != kCufftForward && direction != kCufftInverse,
                  "CUDA cuFFT C2C direction must be -1 (forward) or 1 "
                  "(inverse).");
      status = driver.exec_c2c.call(handle_, input, output, direction);
    } else if (descriptor_.transform_kind == CufftTransformKind::r2c) {
      TI_ERROR_IF(direction != kCufftForward,
                  "CUDA cuFFT R2C requires the forward direction.");
      status = driver.exec_r2c.call(handle_, input, output);
    } else {
      TI_ERROR_IF(direction != kCufftInverse,
                  "CUDA cuFFT C2R requires the inverse direction.");
      status = driver.exec_c2r.call(handle_, input, output);
    }
    TI_ERROR_IF(status != 0, "CUDA cuFFT execution failed (status {}).", status);
  }

  void destroy(bool provider_calls_safe) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &child : children_) {
      child->destroy(provider_calls_safe);
    }
    children_.clear();
    if (handle_ == 0) {
      return;
    }
    if (provider_calls_safe && CUFFTDriver::get_instance().is_loaded()) {
      const auto status = CUFFTDriver::get_instance().destroy.call(handle_);
      TI_WARN_IF(status != 0,
                 "CUDA cuFFT plan destruction returned status {}.", status);
    }
    handle_ = 0;
  }

 private:
  CufftPlanDescriptor descriptor_;
  CufftScalarCounts scalar_counts_;
  std::size_t workspace_bytes_{0};
  int handle_{0};
  std::shared_ptr<RuntimeFaultDomain> fault_domain_;
  std::vector<std::unique_ptr<CudaFftPlan>> children_;
  std::mutex mutex_;
};

std::uint64_t Program::create_cuda_cufft_plan_1d(std::size_t length,
                                                 std::size_t batch_count,
                                                 int transform_kind) {
  const auto transform = validate_transform_kind(transform_kind);
  const auto descriptor =
      make_cufft_plan_1d_descriptor(length, batch_count, transform);
  return create_cuda_cufft_plan_many(
      descriptor.dimensions, descriptor.input_embed,
      descriptor.input_stride, descriptor.input_distance,
      descriptor.output_embed, descriptor.output_stride,
      descriptor.output_distance, descriptor.batch_count, transform_kind);
}

std::uint64_t Program::create_cuda_cufft_plan_many(
    std::vector<int> dimensions,
    std::vector<int> input_embed,
    int input_stride,
    int input_distance,
    std::vector<int> output_embed,
    int output_stride,
    int output_distance,
    int batch_count,
    int transform_kind,
    bool separable) {
  return create_cuda_cufft_plan_many_decomposed(
      std::move(dimensions), std::move(input_embed), input_stride,
      input_distance, std::move(output_embed), output_stride, output_distance,
      batch_count, transform_kind, separable, false);
}

std::uint64_t Program::create_cuda_cufft_cross_batch_plan(
    std::vector<int> dimensions,
    int batch_count) {
  TI_ERROR_IF(
      dimensions.size() != 2 || dimensions[0] <= 0 || dimensions[1] <= 0,
      "Cross-batch cuFFT requires two positive dimensions.");
  const auto plane =
      checked_multiply(dimensions[0], dimensions[1], "cross-batch plane");
  TI_ERROR_IF(plane > (std::numeric_limits<int>::max)(),
              "Cross-batch cuFFT plane exceeds the compact layout contract.");
  return create_cuda_cufft_plan_many_decomposed(
      dimensions, dimensions, 1, static_cast<int>(plane), dimensions, 1,
      static_cast<int>(plane), batch_count,
      static_cast<int>(CufftTransformKind::c2c), true, true);
}

std::uint64_t Program::create_cuda_cufft_store_callback_plan(
    std::vector<int> dimensions,
    int batch_count,
    const std::string &lto_ir,
    const std::string &symbol) {
  TI_ERROR_IF(dimensions.size() != 2 || dimensions[0] <= 0 || dimensions[1] <= 0,
              "CUDA cuFFT store callbacks require two positive dimensions.");
  TI_ERROR_IF(lto_ir.empty() || symbol.empty() ||
                  symbol.find('\0') != std::string::npos,
              "CUDA cuFFT store callback requires LTO IR and a symbol name.");
  const auto plane = checked_multiply(dimensions[0], dimensions[1], "callback plane");
  TI_ERROR_IF(plane > (std::numeric_limits<int>::max)(),
              "CUDA cuFFT callback plane exceeds INT_MAX.");
  return create_cuda_cufft_plan_many_decomposed(
      dimensions, dimensions, 1, static_cast<int>(plane),
      dimensions, 1, static_cast<int>(plane), batch_count, 0, false, false,
      lto_ir, symbol);
}

std::uint64_t Program::create_cuda_cufft_plan_many_decomposed(
    std::vector<int> dimensions,
    std::vector<int> input_embed,
    int input_stride,
    int input_distance,
    std::vector<int> output_embed,
    int output_stride,
    int output_distance,
    int batch_count,
    int transform_kind,
    bool separable,
    bool cross_batch,
    const std::string &store_callback_ir,
    const std::string &store_callback_symbol) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA cuFFT plans require the CUDA backend.");
  const auto validated_transform = validate_transform_kind(transform_kind);
  CufftPlanDescriptor descriptor{std::move(dimensions),
                                 std::move(input_embed),
                                 input_stride,
                                 input_distance,
                                 std::move(output_embed),
                                 output_stride,
                                 output_distance,
                                 batch_count,
                                 validated_transform, separable, cross_batch,
                                 store_callback_ir, store_callback_symbol};
  TI_ERROR_IF(!store_callback_ir.empty() &&
                  (separable || validated_transform != CufftTransformKind::c2c),
              "LTO store callbacks require a whole C2C plan.");
  cufft_scalar_counts(descriptor);
  TI_ERROR_IF(!CUDADriver::get_instance_without_context()
                   .nvidia_extensions_available(),
              "CUDA cuFFT requires the NVIDIA CUDA provider.");
  auto cuda_submission_guard =
      CUDAContext::get_instance().get_submission_lock_guard();
  auto context_guard = CUDAContext::get_instance().get_guard();
  std::lock_guard<std::mutex> lock(cuda_cufft_plan_mutex_);
  ++cuda_cufft_plan_create_requests_;
  const auto cache_key = cufft_plan_cache_key(descriptor);
  std::shared_ptr<CudaFftPlan> plan;
  const auto cached = cuda_cufft_plan_cache_.find(cache_key);
  if (cached != cuda_cufft_plan_cache_.end()) {
    plan = cached->second.lock();
    const bool has_live_handle =
        plan && std::any_of(
                    cuda_cufft_plans_.begin(), cuda_cufft_plans_.end(),
                    [&plan](const auto &item) { return item.second == plan; });
    if (!has_live_handle) {
      plan.reset();
      cuda_cufft_plan_cache_.erase(cached);
    }
  }
  if (plan) {
    ++cuda_cufft_plan_cache_hits_;
  } else {
    ++cuda_cufft_plan_cache_misses_;
    plan = std::make_shared<CudaFftPlan>(descriptor, true,
                                         runtime_fault_domain_);
    cuda_cufft_plan_cache_[cache_key] = plan;
  }
  TI_ERROR_IF(next_cuda_cufft_plan_handle_ == 0,
              "CUDA cuFFT plan handle space exhausted.");
  const auto handle = next_cuda_cufft_plan_handle_++;
  cuda_cufft_plans_.emplace(handle, std::move(plan));
  return handle;
}

std::size_t Program::cuda_cufft_execute(std::uint64_t handle,
                                       Ndarray *input,
                                       Ndarray *output,
                                       int direction) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA cuFFT execution requires the CUDA backend.");
  TI_ERROR_IF(!input || !output,
              "CUDA cuFFT execution received a null ndarray.");

  std::shared_ptr<CudaFftPlan> plan;
  {
    std::lock_guard<std::mutex> lock(cuda_cufft_plan_mutex_);
    const auto found = cuda_cufft_plans_.find(handle);
    TI_ERROR_IF(found == cuda_cufft_plans_.end(),
                "CUDA cuFFT plan handle is stale or closed.");
    plan = found->second;
  }
  const auto expected = plan->scalar_counts();
  const auto validate = [](const char *name,
                           Ndarray *array,
                           std::size_t expected_scalars) {
    TI_ERROR_IF(!array->get_element_shape().empty() ||
                    array->get_element_data_type() != PrimitiveType::f32 ||
                    array->get_nelement() != expected_scalars ||
                    array->get_element_size() != sizeof(float32),
                "CUDA cuFFT {} must be a compact scalar f32 ndarray with "
                "exactly the plan-declared scalar count.", name);
  };
  validate("input", input, expected.input);
  validate("output", output, expected.output);
  TI_ERROR_IF(input->owning_program() != this ||
                  output->owning_program() != this,
              "CUDA cuFFT arrays must belong to the active runtime.");
  TI_ERROR_IF(input->get_device_allocation() ==
                  output->get_device_allocation(),
              "The first CUDA cuFFT slice requires distinct input and output "
              "allocations.");

  auto cuda_submission_guard =
      CUDAContext::get_instance().get_submission_lock_guard();
  auto context_guard = CUDAContext::get_instance().get_guard();
  auto *input_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(input));
  auto *output_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output));
  TI_ERROR_IF(!input_ptr || !output_ptr,
              "CUDA cuFFT received a null device pointer.");
  plan->execute(input_ptr, output_ptr, direction, nullptr);
  pin_cuda_provider_plan(plan);
  mark_runtime_submission_pending();
  return 0;
}

std::size_t Program::cuda_cufft_execute_c2c(std::uint64_t handle,
                                            Ndarray *input,
                                            Ndarray *output,
                                            int direction) {
  {
    std::lock_guard<std::mutex> lock(cuda_cufft_plan_mutex_);
    const auto found = cuda_cufft_plans_.find(handle);
    TI_ERROR_IF(found == cuda_cufft_plans_.end(),
                "CUDA cuFFT plan handle is stale or closed.");
    TI_ERROR_IF(found->second->transform_kind() != CufftTransformKind::c2c,
                "CUDA cuFFT legacy C2C execution requires a C2C plan.");
  }
  return cuda_cufft_execute(handle, input, output, direction);
}

bool Program::cuda_cufft_capture_plan_available(std::uint64_t handle) {
  std::lock_guard<std::mutex> lock(cuda_cufft_plan_mutex_);
  return cuda_cufft_plans_.find(handle) != cuda_cufft_plans_.end();
}

std::shared_ptr<void> Program::retain_cuda_cufft_capture_plan(
    std::uint64_t handle) {
  std::lock_guard<std::mutex> lock(cuda_cufft_plan_mutex_);
  const auto found = cuda_cufft_plans_.find(handle);
  TI_ERROR_IF(found == cuda_cufft_plans_.end(),
              "CUDA cuFFT capture plan handle is stale or closed.");
  return found->second;
}

std::size_t Program::cuda_cufft_capture_record(std::uint64_t handle,
                                               Ndarray *input,
                                               Ndarray *output,
                                               int direction,
                                               void *stream) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA cuFFT capture requires the CUDA backend.");
  TI_ERROR_IF(!input || !output,
              "CUDA cuFFT capture received a null ndarray.");

  std::shared_ptr<CudaFftPlan> plan;
  {
    std::lock_guard<std::mutex> lock(cuda_cufft_plan_mutex_);
    const auto found = cuda_cufft_plans_.find(handle);
    TI_ERROR_IF(found == cuda_cufft_plans_.end(),
                "CUDA cuFFT capture plan handle is stale or closed.");
    plan = found->second;
  }
  const auto expected = plan->scalar_counts();
  const auto validate = [](const char *name,
                           Ndarray *array,
                           std::size_t expected_scalars) {
    TI_ERROR_IF(!array->get_element_shape().empty() ||
                    array->get_element_data_type() != PrimitiveType::f32 ||
                    array->get_nelement() != expected_scalars ||
                    array->get_element_size() != sizeof(float32),
                "CUDA cuFFT capture {} must be a compact scalar f32 ndarray "
                "with exactly the plan-declared scalar count.",
                name);
  };
  validate("input", input, expected.input);
  validate("output", output, expected.output);
  TI_ERROR_IF(input->owning_program() != this ||
                  output->owning_program() != this,
              "CUDA cuFFT capture arrays must belong to the active runtime.");
  TI_ERROR_IF(input->get_device_allocation() ==
                  output->get_device_allocation(),
              "CUDA cuFFT capture input/output alias.");

  auto context_guard = CUDAContext::get_instance().get_guard();
  auto *input_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(input));
  auto *output_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output));
  TI_ERROR_IF(!input_ptr || !output_ptr,
              "CUDA cuFFT capture received a null device pointer.");
  plan->execute(input_ptr, output_ptr, direction,
                reinterpret_cast<CUstream>(stream));
  pin_cuda_provider_plan(plan);
  mark_runtime_submission_pending();
  return 0;
}

std::unordered_map<std::string, std::uint64_t>
Program::cuda_cufft_plan_memory_statistics(std::uint64_t handle) {
  std::lock_guard<std::mutex> lock(cuda_cufft_plan_mutex_);
  const auto found = cuda_cufft_plans_.find(handle);
  TI_ERROR_IF(found == cuda_cufft_plans_.end(),
              "CUDA cuFFT plan handle is stale or closed.");
  const auto *identity = found->second.get();
  std::uint64_t shared_handle_count = 0;
  for (const auto &[other_handle, plan] : cuda_cufft_plans_) {
    if (plan.get() == identity) {
      ++shared_handle_count;
    }
  }
  return {{"workspace_bytes",
           static_cast<std::uint64_t>(found->second->workspace_bytes())},
          {"shared_handle_count", shared_handle_count},
          {"separable", found->second->descriptor().separable ? 1 : 0},
          {"cross_batch", found->second->descriptor().cross_batch ? 1 : 0},
          {"store_callback_lto", found->second->descriptor().store_callback_ir.empty() ? 0 : 1}};
}

std::unordered_map<std::string, std::uint64_t>
Program::cuda_cufft_plan_cache_statistics() {
  std::lock_guard<std::mutex> lock(cuda_cufft_plan_mutex_);
  std::unordered_set<const CudaFftPlan *> unique_plans;
  std::uint64_t workspace_bytes = 0;
  for (const auto &[handle, plan] : cuda_cufft_plans_) {
    if (unique_plans.insert(plan.get()).second) {
      workspace_bytes += static_cast<std::uint64_t>(plan->workspace_bytes());
    }
  }
  return {{"create_requests", cuda_cufft_plan_create_requests_},
          {"cache_hits", cuda_cufft_plan_cache_hits_},
          {"cache_misses", cuda_cufft_plan_cache_misses_},
          {"live_handles",
           static_cast<std::uint64_t>(cuda_cufft_plans_.size())},
          {"live_plans", static_cast<std::uint64_t>(unique_plans.size())},
          {"workspace_bytes_live", workspace_bytes}};
}

void Program::destroy_cuda_cufft_plan(std::uint64_t handle) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  std::shared_ptr<CudaFftPlan> plan;
  {
    std::lock_guard<std::mutex> lock(cuda_cufft_plan_mutex_);
    const auto found = cuda_cufft_plans_.find(handle);
    if (found == cuda_cufft_plans_.end()) {
      return;
    }
    plan = found->second;
    cuda_cufft_plans_.erase(found);
  }
  // The last owner destroys immediately when no submission retains the plan.
  // Otherwise RuntimeCompletion releases the plan after its CUDA event.
  plan.reset();
}

void Program::cuda_clear_cufft_plans() {
  std::vector<std::shared_ptr<CudaFftPlan>> plans;
  {
    std::lock_guard<std::mutex> lock(cuda_cufft_plan_mutex_);
    std::unordered_set<CudaFftPlan *> unique_plans;
    plans.reserve(cuda_cufft_plans_.size());
    for (auto &[handle, plan] : cuda_cufft_plans_) {
      if (unique_plans.insert(plan.get()).second) {
        plans.push_back(std::move(plan));
      }
    }
    cuda_cufft_plans_.clear();
    cuda_cufft_plan_cache_.clear();
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

std::uint64_t Program::create_cuda_cufft_store_callback_plan(
    std::vector<int>, int, const std::string &, const std::string &) {
  TI_ERROR("CUDA cuFFT requires TI_WITH_CUDA=ON.");
}

std::uint64_t Program::create_cuda_cufft_cross_batch_plan(std::vector<int>,
                                                          int) {
  TI_ERROR("CUDA cuFFT requires TI_WITH_CUDA=ON.");
}

std::uint64_t Program::create_cuda_cufft_plan_many_decomposed(std::vector<int>,
                                                              std::vector<int>,
                                                              int,
                                                              int,
                                                              std::vector<int>,
                                                              int,
                                                              int,
                                                              int,
                                                              int,
                                                              bool,
                                                              bool,
                                                              const std::string &,
                                                              const std::string &) {
  TI_ERROR("CUDA cuFFT requires TI_WITH_CUDA=ON.");
}

std::uint64_t Program::create_cuda_cufft_plan_1d(std::size_t,
                                                 std::size_t,
                                                 int) {
  TI_ERROR("CUDA cuFFT requires TI_WITH_CUDA=ON.");
}

std::uint64_t Program::create_cuda_cufft_plan_many(std::vector<int>,
                                                   std::vector<int>,
                                                   int,
                                                   int,
                                                   std::vector<int>,
                                                   int,
                                                   int,
                                                   int,
                                                   int,
                                                   bool) {
  TI_ERROR("CUDA cuFFT requires TI_WITH_CUDA=ON.");
}

std::size_t Program::cuda_cufft_execute(std::uint64_t,
                                       Ndarray *,
                                       Ndarray *,
                                       int) {
  TI_ERROR("CUDA cuFFT requires TI_WITH_CUDA=ON.");
}

std::size_t Program::cuda_cufft_execute_c2c(std::uint64_t,
                                            Ndarray *,
                                            Ndarray *,
                                            int) {
  TI_ERROR("CUDA cuFFT requires TI_WITH_CUDA=ON.");
}

bool Program::cuda_cufft_capture_plan_available(std::uint64_t) {
  return false;
}

std::shared_ptr<void> Program::retain_cuda_cufft_capture_plan(std::uint64_t) {
  TI_ERROR("CUDA cuFFT requires TI_WITH_CUDA=ON.");
}

std::size_t Program::cuda_cufft_capture_record(std::uint64_t,
                                               Ndarray *,
                                               Ndarray *,
                                               int,
                                               void *) {
  TI_ERROR("CUDA cuFFT requires TI_WITH_CUDA=ON.");
}

std::unordered_map<std::string, std::uint64_t>
Program::cuda_cufft_plan_memory_statistics(std::uint64_t) {
  TI_ERROR("CUDA cuFFT requires TI_WITH_CUDA=ON.");
}

std::unordered_map<std::string, std::uint64_t>
Program::cuda_cufft_plan_cache_statistics() {
  return {{"create_requests", 0},
          {"cache_hits", 0},
          {"cache_misses", 0},
          {"live_handles", 0},
          {"live_plans", 0},
          {"workspace_bytes_live", 0}};
}

void Program::destroy_cuda_cufft_plan(std::uint64_t) {
}

void Program::cuda_clear_cufft_plans() {
}

}  // namespace taichi::lang

#endif
