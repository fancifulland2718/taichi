#pragma once

#include "taichi/aot/graph_data.h"

namespace taichi::lang {

// Private bridge for an already prepared, host-scalar f32 cuBLASLt plan.
// The Python recording retains the library, descriptors and workspace lease.
// This adds no CUDA Toolkit header/link dependency to the portable runtime.
struct CudaCublasLtCapturePlan {
  std::uint64_t matmul_address{0};
  std::uint64_t handle{0};
  std::uint64_t descriptor{0};
  std::vector<std::uint64_t> layouts;
  std::string algorithm;
  std::vector<std::vector<int>> shapes;
  std::size_t workspace_bytes{0};
  float alpha{1.0f};
  float beta{0.0f};
};

std::shared_ptr<aot::CudaGraphCaptureCommand>
make_cuda_cublaslt_capture_command(Program *program,
                                   const CudaCublasLtCapturePlan &plan,
                                   const std::vector<aot::Arg> &arguments);

}  // namespace taichi::lang
