#pragma once

#include <memory>
#include <vector>

#include "taichi/aot/graph_data.h"

namespace taichi::lang {

struct CompileConfig;
class Kernel;

struct GraphValueSource {
  Kernel *kernel{nullptr};
  std::vector<aot::Arg> arguments;
};

// Private lowering for Forge's segmented-reduction materializer. Reduction
// arguments are values/offsets/output; maps are independently certified from
// their retained IR. The caller proves complete iteration coverage and actual
// storage alias/layout contracts at the materialization/binding boundary.
// Every user-visible producer/reduction/consumer store is preserved.
std::unique_ptr<aot::CompiledGraph> compile_graph_segmented_reduce_values(
    const CompileConfig &config,
    const GraphValueSource &reduction,
    const GraphValueSource &producer,
    const GraphValueSource &consumer,
    int consumer_input_argument);

}  // namespace taichi::lang
