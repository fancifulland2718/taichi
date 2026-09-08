#pragma once

#include "taichi/aot/graph_data.h"

namespace taichi::lang {

// An already prepared bundled-adapter plan. Its Python recording retains
// the adapter, vendor runtime, descriptors and exact workspace allocation.
// Only the Forge C ABI is needed here, not CUDA Toolkit/cuTENSOR headers.
struct CudaCutensorCapturePlan {
  std::uint64_t execute_address{0};
  std::uint64_t handle{0};
  std::vector<std::vector<int>> shapes;
  std::size_t workspace_bytes{0};
  std::uint32_t alignment_bytes{128};
  bool output_alias_compatible{false};
  float alpha{1.0f};
  float beta{0.0f};
};

std::shared_ptr<aot::CudaGraphCaptureCommand>
make_cuda_cutensor_capture_command(Program *program,
                                   const CudaCutensorCapturePlan &plan,
                                   const std::vector<aot::Arg> &arguments);

}  // namespace taichi::lang
