#include "taichi/program/cuda_cusparselt_capture.h"
#include "taichi/external_runtime/forge_runtime_provider.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"

#include <cmath>

namespace taichi::lang {
namespace {

class CudaCusparseLtCaptureCommand final : public aot::CudaGraphCaptureCommand {
 public:
  CudaCusparseLtCaptureCommand(Program *program,
                               const CudaCusparseLtCapturePlan &plan,
                               const std::vector<aot::Arg> &arguments)
      : program_(program), plan_(plan), arguments_(arguments) {
    TI_ERROR_IF(!program || program->compile_config().arch != Arch::cuda,
                "cuSPARSELt capture requires a CUDA Program");
    TI_ERROR_IF(!plan.execute_address || !plan.handle ||
                    (plan.recompress && !plan.compress_address) ||
                    plan.m <= 0 || plan.n <= 0 || plan.k <= 0 || plan.m % 16 ||
                    plan.n % 16 || plan.k % 16 || !plan.compressed_bytes ||
                    plan.alignment_bytes < 16 ||
                    (plan.alignment_bytes & (plan.alignment_bytes - 1)) ||
                    !std::isfinite(plan.alpha) || !std::isfinite(plan.beta),
                "cuSPARSELt capture plan is incomplete");
    shapes_ = {{plan.n, plan.k}, {plan.m, plan.n}, {plan.m, plan.n}, {}};
    sizes_ = {0, 0, 0, plan.compressed_bytes};
    if (plan.workspace_bytes) {
      workspace_index_ = shapes_.size();
      shapes_.push_back({});
      sizes_.push_back(plan.workspace_bytes);
    }
    if (plan.recompress) {
      a_index_ = shapes_.size();
      shapes_.push_back({plan.m, plan.k});
      sizes_.push_back(0);
      if (plan.compression_buffer_bytes) {
        compression_index_ = shapes_.size();
        shapes_.push_back({});
        sizes_.push_back(plan.compression_buffer_bytes);
      }
    }
    TI_ERROR_IF(arguments.size() != shapes_.size(),
                "cuSPARSELt capture bindings are incomplete");
    for (std::size_t i = 0; i < arguments.size(); ++i) {
      const auto &arg = arguments[i];
      TI_ERROR_IF(arg.tag != aot::ArgKind::kNdarray ||
                      !arg.element_shape.empty() ||
                      PrimitiveType::get(arg.dtype_id) != dtype(i) ||
                      arg.field_dim != (sizes_[i] ? 1 : 2),
                  "cuSPARSELt capture requires f16 matrices and byte scratch");
      for (std::size_t j = 0; j < i; ++j) {
        TI_ERROR_IF(arg.name == arguments[j].name,
                    "cuSPARSELt capture binding names must be distinct");
      }
    }
  }

  const char *kind() const override {
    return plan_.recompress ? "cusparselt_compress_matmul_f16"
                            : "cusparselt_snapshot_matmul_f16";
  }

  Program *program() const override {
    return program_;
  }
  bool supports_binding_frames() const override {
    return true;
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
        const bool scratch = sizes_[i] || sizes_[j];
        const bool illegal_output = (i == 2 || j == 2) && !(i == 2 && j == 1);
        if ((scratch || illegal_output) &&
            value->get_device_allocation() ==
                array(j, args)->get_device_allocation()) {
          return false;
        }
      }
    }
    return true;
  }

  void prepare(const std::unordered_map<std::string, aot::IValue> &args,
               Program &program) override {
    TI_ERROR_IF(!supports(args, program),
                "cuSPARSELt capture bindings are incompatible");
    // Do not compress, execute or advance beta*C during capture preparation.
  }

  void record(const std::unordered_map<std::string, aot::IValue> &args,
              Program &program,
              void *stream) override {
    TI_ERROR_IF(!supports(args, program),
                "cuSPARSELt capture bindings are incompatible");
    auto pointer = [&](int i) -> std::uint64_t {
      if (i < 0)
        return 0;
      const auto result = program.get_ndarray_data_ptr_as_int(array(i, args));
      TI_ERROR_IF(result % plan_.alignment_bytes,
                  "cuSPARSELt capture alignment is incompatible");
      return result;
    };
    const auto handle = reinterpret_cast<void *>(plan_.handle);
    const auto stream_value = reinterpret_cast<std::uintptr_t>(stream);
    if (plan_.recompress) {
      TiForgeCusparseLtCompressDesc desc{};
      desc.struct_size = sizeof(desc);
      desc.dense_a = pointer(a_index_);
      desc.compressed_a = pointer(3);
      desc.compression_buffer = pointer(compression_index_);
      desc.compression_buffer_bytes = plan_.compression_buffer_bytes;
      desc.cuda_stream = stream_value;
      const auto status = reinterpret_cast<TiForgeCusparseLtCompressFn>(
          plan_.compress_address)(handle, &desc);
      TI_ERROR_IF(
          status != TI_FORGE_RUNTIME_PROVIDER_SUCCESS,
          "cuSPARSELt compression capture failed with adapter status {}",
          static_cast<int>(status));
    }
    TiForgeCusparseLtMatmulExecDesc desc{};
    desc.struct_size = sizeof(desc);
    desc.alpha = plan_.alpha;
    desc.beta = plan_.beta;
    desc.compressed_a = pointer(3);
    desc.b = pointer(0);
    desc.c = pointer(1);
    desc.d = pointer(2);
    desc.workspace = pointer(workspace_index_);
    desc.workspace_bytes = plan_.workspace_bytes;
    desc.cuda_stream = stream_value;
    const auto status = reinterpret_cast<TiForgeCusparseLtExecuteFn>(
        plan_.execute_address)(handle, &desc);
    TI_ERROR_IF(status != TI_FORGE_RUNTIME_PROVIDER_SUCCESS,
                "cuSPARSELt matmul capture failed with adapter status {}",
                static_cast<int>(status));
  }

 private:
  DataType dtype(std::size_t i) const {
    return sizes_[i] ? PrimitiveType::u8 : PrimitiveType::f16;
  }

  Ndarray *array(
      std::size_t index,
      const std::unordered_map<std::string, aot::IValue> &args) const {
    const auto found = args.find(arguments_[index].name);
    if (found == args.end() || found->second.tag != aot::ArgKind::kNdarray)
      return nullptr;
    auto *value = reinterpret_cast<Ndarray *>(found->second.val);
    if (!value || value->owning_program() != program_ ||
        !value->get_element_shape().empty() ||
        value->get_element_data_type() != dtype(index))
      return nullptr;
    if (!sizes_[index])
      return value->shape == shapes_[index] ? value : nullptr;
    return value->shape.size() == 1 && value->get_nelement() == sizes_[index]
               ? value
               : nullptr;
  }

  Program *program_;
  const CudaCusparseLtCapturePlan plan_;
  const std::vector<aot::Arg> arguments_;
  std::vector<std::vector<int>> shapes_;
  std::vector<std::size_t> sizes_;
  int workspace_index_{-1}, a_index_{-1}, compression_index_{-1};
};

}  // namespace

std::shared_ptr<aot::CudaGraphCaptureCommand>
make_cuda_cusparselt_capture_command(Program *program,
                                     const CudaCusparseLtCapturePlan &plan,
                                     const std::vector<aot::Arg> &arguments) {
  return std::make_shared<CudaCusparseLtCaptureCommand>(program, plan,
                                                        arguments);
}

}  // namespace taichi::lang
