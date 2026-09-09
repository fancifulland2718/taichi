// Forge-owned C ABI. CUTLASS headers and the resulting addon remain external.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>

#include <cuda_runtime_api.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_splitk_parallel.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>
#include <cutlass/version.h>

#if defined(_WIN32)
#define FORGE_EXPORT extern "C" __declspec(dllexport)
#else
#define FORGE_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {
struct Invocation {
  std::uint32_t struct_size, strategy, transpose_a, transpose_b;
  std::int32_t m, n, k, activation;
  float alpha, beta;
  const float *a, *b;
  float *output;
  void *workspace;
  std::size_t workspace_bytes;
  void *stream;
};
thread_local std::string last_error;

// These are implementation choices, not public tuning axes. Split-K owns
// the complete partial-products -> reduction -> epilogue region.
using Tile = cutlass::gemm::GemmShape<64, 128, 8>;
using Warp = cutlass::gemm::GemmShape<32, 64, 8>;
using Instruction = cutlass::gemm::GemmShape<1, 1, 1>;
using Row = cutlass::layout::RowMajor;
using Col = cutlass::layout::ColumnMajor;

// A wide split-K recipe is a partial-product region followed by one epilogue.
// Cooperating warps reduce the partition dimension, keeping adjacent output
// loads coalesced. No atomics, extra workspace or host work enters replay.
template <bool Relu>
__global__ void reduce_wide(const float *partial, float *output,
                            std::int64_t elements, int partitions,
                            float alpha, float beta) {
  __shared__ float sums[4][32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const std::int64_t index = std::int64_t(blockIdx.x) * 32 + lane;
  float value = 0;
  if (index < elements) {
    for (int part = warp; part < partitions; part += 4)
      value += partial[std::int64_t(part) * elements + index];
  }
  sums[warp][lane] = value;
  __syncthreads();
  if (warp == 0 && index < elements) {
    value = alpha * ((sums[0][lane] + sums[1][lane]) +
                     (sums[2][lane] + sums[3][lane]));
    if (beta != 0) value += beta * output[index];
    if constexpr (Relu) value = value < 0 ? 0 : value;
    output[index] = value;
  }
}

template <typename Parallel, bool Relu>
cutlass::Status run_wide(const Invocation &p,
                         const typename Parallel::Arguments &args) {
  using Kernel = typename Parallel::GemmKernel;
  using Block = typename Parallel::ThreadblockShape;
  typename Parallel::ThreadblockSwizzle swizzle;
  const auto tiled = swizzle.get_tiled_shape(args.problem_size,
      {Block::kM, Block::kN, Block::kK}, args.split_k_slices);
  const std::int64_t stride = std::int64_t(p.m) * p.n;
  cutlass::TensorRef<float, Row> workspace(static_cast<float *>(p.workspace), p.n);
  typename Kernel::Params params{args.problem_size, tiled,
      args.ref_A.non_const_ref(), args.ref_B.non_const_ref(), workspace,
      args.convert, stride};
  auto stream = static_cast<cudaStream_t>(p.stream);
  static_assert(sizeof(typename Kernel::SharedStorage) < (48 << 10));
  cutlass::Kernel<Kernel><<<swizzle.get_grid_shape(tiled), Kernel::kThreadCount,
                           sizeof(typename Kernel::SharedStorage), stream>>>(params);
  if (cudaGetLastError() != cudaSuccess) return cutlass::Status::kErrorInternal;
  reduce_wide<Relu><<<(stride + 31) / 32, 128, 0, stream>>>(
      static_cast<const float *>(p.workspace), p.output, stride, tiled.k(),
      p.alpha, p.beta);
  return cudaGetLastError() == cudaSuccess ? cutlass::Status::kSuccess
                                          : cutlass::Status::kErrorInternal;
}

template <typename A, typename B, bool Relu, bool Split, bool Compact = false>
std::uint32_t invoke(const Invocation &p, std::size_t *query) {
  using Block = std::conditional_t<Compact, cutlass::gemm::GemmShape<64, 64, 8>, Tile>;
  using WarpShape = std::conditional_t<Compact, cutlass::gemm::GemmShape<32, 32, 8>, Warp>;
  using Epilogue = std::conditional_t<Relu,
      cutlass::epilogue::thread::LinearCombinationRelu<float, 1, float, float>,
      cutlass::epilogue::thread::LinearCombination<float, 1, float, float>>;
  using Direct = cutlass::gemm::device::Gemm<
      float, A, float, B, float, Row, float, cutlass::arch::OpClassSimt,
      cutlass::arch::Sm50, Block, WarpShape, Instruction, Epilogue>;
  using Parallel = cutlass::gemm::device::GemmSplitKParallel<
      float, A, float, B, float, Row, float, cutlass::arch::OpClassSimt,
      cutlass::arch::Sm50, Block, WarpShape, Instruction, Epilogue>;
  using Gemm = std::conditional_t<Split, Parallel, Direct>;
  typename Gemm::Arguments args(
      {p.m, p.n, p.k}, {p.a, p.transpose_a ? p.m : p.k},
      {p.b, p.transpose_b ? p.k : p.n}, {p.output, p.n}, {p.output, p.n},
      {p.alpha, p.beta}, Split ? (p.strategy == 2 ? 128 : 16) : 1);
  auto bytes = Gemm::get_workspace_size(args);
  if (query) {
    *query = bytes;
    return 0;
  }
  if (!p.a || !p.b || !p.output || p.a == p.output || p.b == p.output ||
      p.workspace_bytes != bytes || (bytes && !p.workspace)) {
    last_error = "CUTLASS capture pointers or workspace disagree with preparation";
    return 1;
  }
  Gemm operation;
  auto status = operation.can_implement(args);
  if constexpr (Split) {
    if (status == cutlass::Status::kSuccess && p.strategy == 2) {
      status = run_wide<Parallel, Relu>(p, args);
      if (status != cutlass::Status::kSuccess) last_error = cutlassGetStatusString(status);
      return status == cutlass::Status::kSuccess ? 0 : 2;
    }
  }
  if (status == cutlass::Status::kSuccess) {
    if constexpr (Split) {
      status = operation.initialize(args, p.workspace);
    } else {
      status = operation.initialize(args, p.workspace,
                                    static_cast<cudaStream_t>(p.stream));
    }
  }
  if (status == cutlass::Status::kSuccess) {
    status = operation.run(static_cast<cudaStream_t>(p.stream));
  }
  if (status != cutlass::Status::kSuccess) {
    last_error = cutlassGetStatusString(status);
    return 2;
  }
  return 0;
}

template <typename A, typename B>
std::uint32_t dispatch(const Invocation &p, std::size_t *query) {
  if (p.strategy == 2) {
    return p.activation ? invoke<A, B, true, true, true>(p, query)
                        : invoke<A, B, false, true, true>(p, query);
  }
  if (p.strategy == 0) {
    return p.activation ? invoke<A, B, true, false>(p, query)
                        : invoke<A, B, false, false>(p, query);
  }
  return p.activation ? invoke<A, B, true, true>(p, query)
                      : invoke<A, B, false, true>(p, query);
}

std::uint32_t dispatch(const Invocation *p, std::size_t *query) {
  if (!p || p->struct_size != sizeof(Invocation) || p->strategy > 2 ||
      p->transpose_a > 1 || p->transpose_b > 1 || p->activation < 0 ||
      p->activation > 1 || p->m <= 0 || p->n <= 0 || p->k <= 0 ||
      !std::isfinite(p->alpha) || !std::isfinite(p->beta)) {
    last_error = "invalid CUTLASS matmul preparation contract";
    return 1;
  }
  if (p->transpose_a) {
    return p->transpose_b ? dispatch<Col, Col>(*p, query)
                          : dispatch<Col, Row>(*p, query);
  }
  return p->transpose_b ? dispatch<Row, Col>(*p, query)
                        : dispatch<Row, Row>(*p, query);
}
}  // namespace

FORGE_EXPORT std::uint32_t ti_forge_cutlass_abi() {
  return 1;
}
FORGE_EXPORT std::uint32_t ti_forge_cutlass_version() {
  return CUTLASS_MAJOR * 10000 + CUTLASS_MINOR * 100 + CUTLASS_PATCH;
}
FORGE_EXPORT std::uint32_t ti_forge_cutlass_query(const Invocation *p,
                                                std::size_t *bytes) {
  if (!bytes) return 1;
  return dispatch(p, bytes);
}
FORGE_EXPORT std::uint32_t ti_forge_cutlass_capture(const void *p) {
  // Called only at native capture. Replay contains GPU nodes, no C ABI call.
  return dispatch(static_cast<const Invocation *>(p), nullptr);
}
FORGE_EXPORT std::size_t ti_forge_cutlass_error(char *out, std::size_t size) {
  if (out && size) {
    const auto n = std::min(size - 1, last_error.size());
    std::memcpy(out, last_error.data(), n);
    out[n] = '\0';
  }
  return last_error.size() + 1;
}
