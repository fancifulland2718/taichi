#pragma once

#include "taichi/aot/graph_data.h"

namespace taichi::lang {

// Native owner, immutable pattern, fixed numerical phase. This is JIT-only;
// neither a provider pointer nor vendor binary state enters AOT or a recipe ID.
std::shared_ptr<aot::CudaGraphCaptureCommand> make_cuda_cudss_capture_command(
    Program *program,
    std::uint64_t handle,
    int numeric_phase,
    const std::vector<aot::Arg> &arguments);

}  // namespace taichi::lang
