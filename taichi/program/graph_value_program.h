#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "taichi/analysis/graph_kernel_metadata.h"
#include "taichi/ir/stmt_op_types.h"
#include "taichi/ir/type.h"

namespace taichi::lang {

struct CompileConfig;
class Kernel;

// Private, cold compiler input for Graph value substitution. This is not an
// executable, an AOT/cache format, or a certificate about runtime bindings.
struct GraphValueNode {
  enum class Kind {
    constant,
    index,
    scalar_argument,
    array_load,
    unary,
    binary
  };

  Kind kind{Kind::constant};
  DataType dtype{PrimitiveType::unknown};
  std::vector<int> operands;
  int argument{-1};
  std::uint32_t constant_bits{0};
  bool runtime_affine{false};
  UnaryOpType unary_op{UnaryOpType::undefined};
  BinaryOpType binary_op{BinaryOpType::undefined};
};

struct GraphPointwiseValueProgram {
  bool available{false};
  std::string blocker{"value_program_unavailable"};
  GraphKernelMetadata metadata;
  int output_argument{-1};
  bool output_runtime_affine{false};
  int result{-1};
  std::vector<GraphValueNode> nodes;
};

// Describes one unconditional i32/u32 pointwise store from cloned pre-offload
// IR. Array loads are at the same logical index. The materializer must still
// prove domain coverage, binding identity/layout, aliasing and observable-store
// preservation before changing execution topology.
GraphPointwiseValueProgram inspect_graph_pointwise_value_program(
    const CompileConfig &config,
    Kernel *kernel);

}  // namespace taichi::lang
