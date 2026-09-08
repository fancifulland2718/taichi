#include "taichi/program/graph_value_fusion.h"

#include <unordered_map>
#include <utility>

#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/program/compile_config.h"
#include "taichi/program/graph_builder.h"
#include "taichi/program/graph_value_program.h"
#include "taichi/program/kernel.h"

namespace taichi::lang {
namespace {

bool qualified_source(const GraphValueSource &source, const Kernel *reduction) {
  if (source.kernel == nullptr || source.kernel->definition_retired() ||
      source.kernel->program != reduction->program ||
      source.kernel->autodiff_mode != AutodiffMode::kNone ||
      !source.kernel->rets.empty() ||
      source.arguments.size() != source.kernel->parameter_list.size() ||
      source.arguments.size() != source.kernel->nested_parameters.size()) {
    return false;
  }
  for (std::size_t i = 0; i < source.arguments.size(); ++i) {
    const auto &argument = source.arguments[i];
    if (argument.name.empty() ||
        (argument.tag != aot::ArgKind::kScalar &&
         argument.tag != aot::ArgKind::kNdarray) ||
        source.kernel->parameter_list[i].is_argpack ||
        source.kernel->nested_parameters.find({static_cast<int>(i)}) ==
            source.kernel->nested_parameters.end()) {
      return false;
    }
  }
  return true;
}

class ArgumentUnion {
 public:
  std::vector<int> append(const GraphValueSource &source) {
    std::vector<int> remap;
    for (std::size_t i = 0; i < source.arguments.size(); ++i) {
      const auto &argument = source.arguments[i];
      const auto &parameter = source.kernel->parameter_list[i];
      const auto found = indices_.find(argument.name);
      if (found != indices_.end()) {
        const int index = found->second;
        TI_ERROR_IF(
            arguments[index] != argument || !(parameters[index] == parameter),
            "Graph value fusion has conflicting symbolic ABI for {}",
            argument.name);
        remap.push_back(index);
      } else {
        const int index = static_cast<int>(arguments.size());
        arguments.push_back(argument);
        parameters.push_back(parameter);
        indices_.emplace(argument.name, index);
        remap.push_back(index);
      }
    }
    return remap;
  }

  std::vector<aot::Arg> arguments;
  std::vector<Callable::Parameter> parameters;

 private:
  std::unordered_map<std::string, int> indices_;
};

int array_argument(Stmt *stmt) {
  auto *pointer = stmt->cast<ExternalPtrStmt>();
  if (pointer == nullptr || pointer->ndim != 1 ||
      pointer->indices.size() != 1 || !pointer->element_shape.empty() ||
      pointer->is_grad || pointer->boundary != BoundaryMode::kUnsafe ||
      pointer->byte_offset != 0 || pointer->byte_stride != 0) {
    return -1;
  }
  auto *base = pointer->base_ptr->cast<ArgLoadStmt>();
  if (base == nullptr || base->arg_id.size() != 1 || base->arg_depth != 0) {
    return -1;
  }
  return base->arg_id[0];
}

class ReductionSites final : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  void visit(GlobalLoadStmt *stmt) override {
    if (array_argument(stmt->src) == 0) {
      loads.push_back(stmt);
    }
  }

  void visit(GlobalStoreStmt *stmt) override {
    if (array_argument(stmt->dest) == 2) {
      stores.push_back(stmt);
    }
  }

  std::vector<GlobalLoadStmt *> loads;
  std::vector<GlobalStoreStmt *> stores;
};

class ValueEmitter {
 public:
  ValueEmitter(const GraphPointwiseValueProgram &program,
               const std::vector<int> &remap,
               const ArgumentUnion &arguments,
               Stmt *index)
      : program_(program), remap_(remap), arguments_(arguments), index_(index) {
  }

  Stmt *emit(VecStatement *statements,
             int replaced_argument = -1,
             Stmt *replacement = nullptr) {
    std::vector<Stmt *> values;
    for (const auto &node : program_.nodes) {
      Stmt *value = nullptr;
      using Kind = GraphValueNode::Kind;
      switch (node.kind) {
        case Kind::constant:
          value = statements->push_back<ConstStmt>(
              TypedConstant(node.dtype, node.constant_bits));
          break;
        case Kind::index:
          value = index_;
          break;
        case Kind::scalar_argument:
          value = statements->push_back<ArgLoadStmt>(
              std::vector<int>{remap_[node.argument]}, node.dtype,
              /*is_ptr=*/false, /*create_load=*/true, /*arg_depth=*/0);
          break;
        case Kind::array_load:
          if (node.argument == replaced_argument) {
            value = replacement;
          } else {
            value = statements->push_back<GlobalLoadStmt>(
                pointer(statements, node.argument, node.runtime_affine));
          }
          break;
        case Kind::unary: {
          auto *unary = statements->push_back<UnaryOpStmt>(
              node.unary_op, values[node.operands[0]]);
          unary->cast_type = node.dtype;
          value = unary;
          break;
        }
        case Kind::binary:
          value = statements->push_back<BinaryOpStmt>(node.binary_op,
                                                      values[node.operands[0]],
                                                      values[node.operands[1]]);
          break;
      }
      TI_ASSERT(value != nullptr);
      // Index/replacement already belong to the reduction's IR. Their type
      // must agree; never mutate them to accommodate an inconsistent map.
      if (value == index_ || value == replacement) {
        TI_ERROR_IF(value->ret_type != node.dtype,
                    "Graph value fusion substitution dtype mismatch");
      } else {
        value->ret_type = node.dtype;
      }
      values.push_back(value);
    }
    return values[program_.result];
  }

  void preserve_store(VecStatement *statements, Stmt *value) {
    statements->push_back<GlobalStoreStmt>(
        pointer(statements, program_.output_argument,
                program_.output_runtime_affine),
        value);
  }

 private:
  ExternalPtrStmt *pointer(VecStatement *statements,
                           int argument,
                           bool runtime_affine) {
    const int combined = remap_[argument];
    auto descriptor_pointer = arguments_.parameters[combined].get_dtype();
    descriptor_pointer.set_is_pointer(true);
    auto *base = statements->push_back<ArgLoadStmt>(
        std::vector<int>{combined}, descriptor_pointer,
        /*is_ptr=*/true,
        /*create_load=*/false, /*arg_depth=*/0);
    return statements->push_back<ExternalPtrStmt>(
        base, std::vector<Stmt *>{index_}, /*ndim=*/1, std::vector<int>{},
        /*is_grad=*/false, BoundaryMode::kUnsafe, /*byte_offset=*/0,
        /*byte_stride=*/0, runtime_affine);
  }

  const GraphPointwiseValueProgram &program_;
  const std::vector<int> &remap_;
  const ArgumentUnion &arguments_;
  Stmt *index_;
};

}  // namespace

std::unique_ptr<aot::CompiledGraph> compile_graph_segmented_reduce_values(
    const CompileConfig &config,
    const GraphValueSource &reduction,
    const GraphValueSource &producer,
    const GraphValueSource &consumer,
    int consumer_input_argument) {
  TI_ERROR_IF(config.arch != Arch::cuda || config.debug ||
                  config.check_out_of_bound || reduction.kernel == nullptr ||
                  !qualified_source(reduction, reduction.kernel) ||
                  reduction.arguments.size() != 3,
              "Graph value fusion requires a live CUDA reduction kernel");
  TI_ERROR_IF(producer.kernel == nullptr && consumer.kernel == nullptr,
              "Graph value fusion requires a producer or consumer");
  ArgumentUnion arguments;
  const auto reduction_remap = arguments.append(reduction);
  TI_ERROR_IF(reduction_remap != std::vector<int>({0, 1, 2}),
              "Graph reduction arguments must have distinct symbolic names");
  GraphPointwiseValueProgram producer_program, consumer_program;
  std::vector<int> producer_remap, consumer_remap;
  if (producer.kernel != nullptr) {
    TI_ERROR_IF(!qualified_source(producer, reduction.kernel),
                "Graph producer source has an incompatible callable ABI");
    producer_program =
        inspect_graph_pointwise_value_program(config, producer.kernel);
    TI_ERROR_IF(!producer_program.available, "Graph producer value program: {}",
                producer_program.blocker);
    producer_remap = arguments.append(producer);
    TI_ERROR_IF(producer_remap[producer_program.output_argument] != 0,
                "Graph producer output must bind the reduction values symbol");
  }
  if (consumer.kernel != nullptr) {
    TI_ERROR_IF(!qualified_source(consumer, reduction.kernel),
                "Graph consumer source has an incompatible callable ABI");
    consumer_program =
        inspect_graph_pointwise_value_program(config, consumer.kernel);
    TI_ERROR_IF(!consumer_program.available, "Graph consumer value program: {}",
                consumer_program.blocker);
    consumer_remap = arguments.append(consumer);
    TI_ERROR_IF(consumer_input_argument < 0 ||
                    consumer_input_argument >=
                        static_cast<int>(consumer_remap.size()) ||
                    consumer_remap[consumer_input_argument] != 2,
                "Graph consumer input must bind the reduction output symbol");
  }

  auto ir = irpass::analysis::clone(reduction.kernel->ir.get());
  irpass::compile_to_offloads(ir.get(), config, reduction.kernel, false,
                              AutodiffMode::kNone, true,
                              reduction.kernel->ir_is_ast(), nullptr,
                              /*stop_before_offload=*/true);
  ReductionSites sites;
  ir->accept(&sites);
  TI_ERROR_IF(
      sites.loads.size() != 1 || sites.stores.size() != 1,
      "Graph value fusion requires one reduction input load and final store");
  if (producer.kernel != nullptr) {
    auto *load = sites.loads[0];
    auto *index = load->src->as<ExternalPtrStmt>()->indices[0];
    ValueEmitter emitter(producer_program, producer_remap, arguments, index);
    VecStatement statements;
    auto *value = emitter.emit(&statements);
    TI_ERROR_IF(
        value->ret_type != load->ret_type,
        "Graph producer result does not match the reduction input dtype");
    emitter.preserve_store(&statements, value);
    load->parent->insert_before(load, std::move(statements));
    load->replace_usages_with(value);
    load->parent->erase(load);
  }
  if (consumer.kernel != nullptr) {
    auto *store = sites.stores[0];
    auto *index = store->dest->as<ExternalPtrStmt>()->indices[0];
    ValueEmitter emitter(consumer_program, consumer_remap, arguments, index);
    VecStatement statements;
    auto *value =
        emitter.emit(&statements, consumer_input_argument, store->val);
    emitter.preserve_store(&statements, value);
    // Keep the reduction output visible even when its value is forwarded to
    // the consumer. Other consumer inputs observe the post-reduction store.
    store->parent->insert_after(store, std::move(statements));
  }

  auto kernel = std::make_shared<Kernel>(
      *reduction.kernel->program, std::move(ir),
      reduction.kernel->get_name() + "__graph_value_fused");
  kernel->parameter_list = std::move(arguments.parameters);
  for (std::size_t i = 0; i < kernel->parameter_list.size(); ++i) {
    kernel->nested_parameters[{static_cast<int>(i)}] =
        kernel->parameter_list[i];
  }
  kernel->finalize_params();
  irpass::type_check(kernel->ir.get(), config);
  irpass::analysis::verify(kernel->ir.get());
  Dispatch dispatch(kernel.get(), arguments.arguments);
  auto compiled = dispatch.compile_dispatch();
  std::unordered_map<std::string, aot::Arg> graph_args;
  for (const auto &argument : arguments.arguments) {
    graph_args.emplace(argument.name, argument);
  }
  auto result = std::make_unique<aot::CompiledGraph>(
      std::vector<aot::CompiledDispatch>{std::move(compiled)},
      std::move(graph_args));
  result->owned_jit_kernels.push_back(std::move(kernel));
  return result;
}

}  // namespace taichi::lang
