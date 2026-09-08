#include "taichi/program/cuda_cutensor_capture.h"
#include "taichi/external_runtime/forge_runtime_provider.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"

#include <cmath>

namespace taichi::lang {
namespace {

class CudaCutensorCaptureCommand final : public aot::CudaGraphCaptureCommand {
 public:
  CudaCutensorCaptureCommand(Program *program,
                             const CudaCutensorCapturePlan &plan,
                             const std::vector<aot::Arg> &arguments)
      : program_(program), plan_(plan), arguments_(arguments) {
    TI_ERROR_IF(!program || program->compile_config().arch != Arch::cuda,
                "cuTENSOR capture requires a CUDA Program");
    TI_ERROR_IF(!plan.execute_address || !plan.handle ||
                    plan.shapes.size() != 4 ||
                    arguments.size() != (plan.workspace_bytes ? 5 : 4) ||
                    !std::isfinite(plan.alpha) || !std::isfinite(plan.beta) ||
                    !plan.alignment_bytes ||
                    (plan.alignment_bytes & (plan.alignment_bytes - 1)),
                "cuTENSOR capture plan is incomplete");
    for (std::size_t i = 0; i < arguments.size(); ++i) {
      const auto &arg = arguments[i];
      TI_ERROR_IF(
          arg.tag != aot::ArgKind::kNdarray || !arg.element_shape.empty() ||
              PrimitiveType::get(arg.dtype_id) !=
                  (i < 4 ? PrimitiveType::f32 : PrimitiveType::u8),
          "cuTENSOR capture requires scalar f32 tensors and byte workspace");
      if (i < 4) {
        TI_ERROR_IF(plan.shapes[i].empty() || plan.shapes[i].size() > 32 ||
                        arg.field_dim != plan.shapes[i].size(),
                    "cuTENSOR capture tensor rank is incompatible");
        for (auto extent : plan.shapes[i]) {
          TI_ERROR_IF(extent <= 0, "cuTENSOR capture extent is invalid");
        }
      } else {
        TI_ERROR_IF(arg.field_dim != 1,
                    "cuTENSOR workspace must have rank one");
      }
      for (std::size_t j = 0; j < i; ++j) {
        TI_ERROR_IF(arg.name == arguments[j].name,
                    "cuTENSOR capture binding names must be distinct");
      }
    }
  }

  const char *kind() const override {
    return "cutensor_retained_contraction_f32";
  }

  Program *program() const override {
    return program_;
  }

  bool supports_binding_frames() const override {
    // Address changes are captured once per immutable frame. No provider
    // planning, pointer resolution or validation is performed on replay.
    return true;
  }

  bool supports(const std::unordered_map<std::string, aot::IValue> &args,
                Program &program) const override {
    if (&program != program_) {
      return false;
    }
    for (std::size_t i = 0; i < arguments_.size(); ++i) {
      auto *value = array(i, args);
      if (!value) {
        return false;
      }
      for (std::size_t j = 0; j < i; ++j) {
        if (value->get_device_allocation() ==
                array(j, args)->get_device_allocation() &&
            (i == 4 || (i == 3 && (j < 2 || !plan_.output_alias_compatible)))) {
          return false;
        }
      }
    }
    return true;
  }

  void prepare(const std::unordered_map<std::string, aot::IValue> &args,
               Program &program) override {
    TI_ERROR_IF(!supports(args, program),
                "cuTENSOR capture bindings are incompatible");
    // Creating the vendor plan is a provider boundary. Do not execute the
    // contraction here: C and D may alias and beta may carry feedback.
  }

  void record(const std::unordered_map<std::string, aot::IValue> &args,
              Program &program,
              void *stream) override {
    TI_ERROR_IF(!supports(args, program),
                "cuTENSOR capture bindings are incompatible");
    auto pointer = [&](std::size_t i) {
      auto result = program.get_ndarray_data_ptr_as_int(array(i, args));
      TI_ERROR_IF(result % (i < 4 ? plan_.alignment_bytes : 128),
                  "cuTENSOR capture buffer alignment is incompatible");
      return static_cast<std::uint64_t>(result);
    };
    TiForgeCutensorContractionExecDesc desc{};
    desc.struct_size = sizeof(desc);
    desc.alpha = plan_.alpha;
    desc.beta = plan_.beta;
    desc.a = pointer(0);
    desc.b = pointer(1);
    desc.c = pointer(2);
    desc.d = pointer(3);
    desc.workspace = plan_.workspace_bytes ? pointer(4) : 0;
    desc.workspace_bytes = plan_.workspace_bytes;
    desc.cuda_stream = reinterpret_cast<std::uintptr_t>(stream);
    const auto invoke =
        reinterpret_cast<TiForgeCutensorExecuteFn>(plan_.execute_address);
    const auto status = invoke(reinterpret_cast<void *>(plan_.handle), &desc);
    TI_ERROR_IF(status != TI_FORGE_RUNTIME_PROVIDER_SUCCESS,
                "cuTENSOR Graph capture failed with adapter status {}",
                static_cast<int>(status));
  }

 private:
  Ndarray *array(
      std::size_t index,
      const std::unordered_map<std::string, aot::IValue> &args) const {
    const auto found = args.find(arguments_[index].name);
    if (found == args.end() || found->second.tag != aot::ArgKind::kNdarray) {
      return nullptr;
    }
    auto *value = reinterpret_cast<Ndarray *>(found->second.val);
    if (!value || value->owning_program() != program_ ||
        !value->get_element_shape().empty() ||
        value->get_element_data_type() !=
            (index < 4 ? PrimitiveType::f32 : PrimitiveType::u8)) {
      return nullptr;
    }
    if (index < 4) {
      return value->shape == plan_.shapes[index] ? value : nullptr;
    }
    return value->shape.size() == 1 &&
                   value->get_nelement() == plan_.workspace_bytes
               ? value
               : nullptr;
  }

  Program *program_;
  const CudaCutensorCapturePlan plan_;
  const std::vector<aot::Arg> arguments_;
};

}  // namespace

std::shared_ptr<aot::CudaGraphCaptureCommand>
make_cuda_cutensor_capture_command(Program *program,
                                   const CudaCutensorCapturePlan &plan,
                                   const std::vector<aot::Arg> &arguments) {
  return std::make_shared<CudaCutensorCaptureCommand>(program, plan, arguments);
}

}  // namespace taichi::lang
