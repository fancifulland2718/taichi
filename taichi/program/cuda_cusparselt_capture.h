#pragma once

#include "taichi/aot/graph_data.h"

namespace taichi::lang {

// Prepared Forge C-ABI plan. Python owns its vendor, compressed data and
// scratch. No CUDA Toolkit or cuSPARSELt headers/imports are needed by the
// runtime.
struct CudaCusparseLtCapturePlan {
  std::uint64_t compress_address{0};
  std::uint64_t execute_address{0};
  std::uint64_t handle{0};
  int m{0}, n{0}, k{0};
  int matmul_count{1};  // One shared compression followed by ordered products.
  std::size_t compressed_bytes{0};
  std::size_t compression_buffer_bytes{0};
  std::size_t workspace_bytes{0};
  std::uint32_t alignment_bytes{16};
  bool recompress{false};
  float alpha{1.0f}, beta{0.0f};
};

std::shared_ptr<aot::CudaGraphCaptureCommand>
make_cuda_cusparselt_capture_command(Program *program,
                                     const CudaCusparseLtCapturePlan &plan,
                                     const std::vector<aot::Arg> &arguments);

}  // namespace taichi::lang
