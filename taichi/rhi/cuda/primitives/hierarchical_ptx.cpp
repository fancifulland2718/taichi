#include "taichi/rhi/cuda/primitives/hierarchical_ptx.h"

#include "taichi/common/core.h"
#include "taichi/program/primitive_workspace.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/rhi/cuda/primitives/ptx/hierarchical_ptx.inc.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace taichi::lang::cuda {
namespace {

constexpr std::uint32_t kBlockDim = 256;
constexpr std::uint32_t kScanItemsPerThread = 4;
constexpr std::uint32_t kScanTileItems = kBlockDim * kScanItemsPerThread;
constexpr std::uint32_t kScanBlockShift = 8;
constexpr std::uint32_t kScanTileShift = 10;
constexpr std::uint32_t kReduceItemsPerThread = 4;
constexpr std::uint32_t kReduceTileItems = kBlockDim * kReduceItemsPerThread;
constexpr std::uint32_t kRadixBitsPerPass = 4;
constexpr std::uint32_t kRadixDigits = 1u << kRadixBitsPerPass;
constexpr std::uint32_t kRadixItemsPerBlock = 1024;
constexpr std::uint64_t kWorkspaceVariant = 0x4452565054580001ull;

constexpr std::size_t radix_histogram_level_count(std::uint32_t count) {
  std::size_t levels = 1;
  while (count > kScanTileItems) {
    count = (count + kScanTileItems - 1u) / kScanTileItems;
    ++levels;
  }
  return levels;
}

static_assert(radix_histogram_level_count(1) == 1);
static_assert(radix_histogram_level_count(kScanTileItems) == 1);
static_assert(radix_histogram_level_count(kScanTileItems + 1) == 2);
static_assert(radix_histogram_level_count(kScanTileItems *
                                          kScanTileItems) == 2);
static_assert(radix_histogram_level_count(kScanTileItems *
                                              kScanTileItems +
                                          1) == 3);

struct KernelSet {
  void *module{nullptr};
  std::array<void *, 6> scan{};
  std::array<void *, 6> scan_tiled{};
  std::array<void *, 6> uniform_add{};
  std::array<void *, 6> reduce{};
  std::array<void *, 6> add{};
  std::array<void *, 2> add_scaled{};
  std::array<void *, 2> gather_add{};
  std::array<void *, 6> scatter_add{};
  std::array<void *, 6> zero{};
  void *sparse_diagonal_apply_f32{nullptr};
  void *sparse_diagonal_refresh_f32{nullptr};
  void *sparse_block_cholesky_refresh_f32{nullptr};
  void *sparse_block_diagonal_apply_f32{nullptr};
  void *sparse_minres_scalar_f32{nullptr};
  void *sparse_minres_vector_state_f32{nullptr};
  void *sparse_minres_commit_f32{nullptr};
  void *sparse_bicgstab_scalar_f32{nullptr};
  void *sparse_bicgstab_direction_f32{nullptr};
  void *sparse_bicgstab_intermediate_f32{nullptr};
  void *sparse_bicgstab_commit_f32{nullptr};
  void *sparse_bicgstab_reconcile_f32{nullptr};
  void *sparse_gmres_multi_dot_partial_f32{nullptr};
  void *sparse_gmres_multi_dot_final_f32{nullptr};
  void *sparse_gmres_projection_f32{nullptr};
  void *sparse_gmres_basis_f32{nullptr};
  void *sparse_gmres_combine_f32{nullptr};
  void *sparse_gmres_scalar_f32{nullptr};
  std::array<void *, 2> zero_bins{};
  std::array<std::array<void *, 2>, 2> histogram{};
  void *compact_rank_tiles{nullptr};
  void *compact_scatter_tiled{nullptr};
  void *copy_i32{nullptr};
  void *bucket_scatter{nullptr};
  std::array<void *, 6> radix_init{};
  std::array<void *, 2> radix_rank4{};
  void *radix_hist_scan{nullptr};
  void *radix_hist_uniform{nullptr};
  std::array<void *, 2> radix_scatter4{};
  void *radix_gather_words{nullptr};
  void *radix_copy_words{nullptr};
  void *sparse_assembly_pack_validate{nullptr};
  void *sparse_assembly_pack_packed_validate{nullptr};
  void *sparse_assembly_mark_segments{nullptr};
  void *sparse_assembly_scatter_segments{nullptr};
  void *sparse_assembly_reduce_segments{nullptr};
  void *sparse_assembly_emit_csr{nullptr};
  void *sparse_assembly_finalize_control{nullptr};
  void *sparse_csr_spmv_f32{nullptr};
  void *sparse_bsr_spmv_f32{nullptr};
};

std::once_flag kernel_set_once;
KernelSet kernel_set;

std::size_t value_type_index(CudaTransformValueType value_type) {
  const int index = static_cast<int>(value_type);
  TI_ERROR_IF(index < 0 || index >= 6,
              "CUDA Driver hierarchical primitive received an unsupported "
              "value type.");
  return static_cast<std::size_t>(index);
}

std::size_t value_type_size(CudaTransformValueType value_type) {
  switch (value_type) {
    case CudaTransformValueType::i32:
    case CudaTransformValueType::f32:
    case CudaTransformValueType::u32:
      return 4;
    case CudaTransformValueType::u64:
    case CudaTransformValueType::i64:
    case CudaTransformValueType::f64:
      return 8;
  }
  TI_ERROR("Unsupported CUDA hierarchical primitive value type.");
  return 0;
}

std::size_t sort_key_type_size(CudaDriverSortKeyType key_type) {
  switch (key_type) {
    case CudaDriverSortKeyType::u32:
    case CudaDriverSortKeyType::i32:
    case CudaDriverSortKeyType::f32:
      return sizeof(std::uint32_t);
    case CudaDriverSortKeyType::u64:
    case CudaDriverSortKeyType::i64:
    case CudaDriverSortKeyType::f64:
      return sizeof(std::uint64_t);
  }
  TI_ERROR("Unsupported CUDA Driver stable-sort key type.");
  return 0;
}

std::size_t sort_key_type_index(CudaDriverSortKeyType key_type) {
  const int index = static_cast<int>(key_type);
  TI_ERROR_IF(index < 0 || index >= 6,
              "CUDA Driver stable sort received an unsupported key type.");
  return static_cast<std::size_t>(index);
}

void load_kernel_set_once() {
  auto &ctx = CUDAContext::get_instance();
  auto context_guard = ctx.get_guard();
  auto &driver = CUDADriver::get_instance();
  driver.module_load_data_ex(&kernel_set.module, kCudaHierarchicalPtx, 0,
                             nullptr, nullptr);

  constexpr std::array<const char *, 6> suffixes{"i32", "f32", "u32",
                                                 "u64", "i64", "f64"};
  for (std::size_t i = 0; i < suffixes.size(); ++i) {
    const std::string scan_name = std::string("scan_blocks_") + suffixes[i];
    const std::string scan_tiled_name =
        std::string("scan_tiles_") + suffixes[i];
    const std::string uniform_name = std::string("uniform_add_") + suffixes[i];
    const std::string reduce_name = std::string("reduce_blocks_") + suffixes[i];
    const std::string add_name = std::string("add_strided_") + suffixes[i];
    const std::string scatter_add_name =
        std::string("scatter_add_strided_") + suffixes[i];
    const std::string zero_name = std::string("zero_strided_") + suffixes[i];
    driver.module_get_function(&kernel_set.scan[i], kernel_set.module,
                               scan_name.c_str());
    driver.module_get_function(&kernel_set.scan_tiled[i], kernel_set.module,
                               scan_tiled_name.c_str());
    driver.module_get_function(&kernel_set.uniform_add[i], kernel_set.module,
                               uniform_name.c_str());
    driver.module_get_function(&kernel_set.reduce[i], kernel_set.module,
                               reduce_name.c_str());
    driver.module_get_function(&kernel_set.add[i], kernel_set.module,
                               add_name.c_str());
    driver.module_get_function(&kernel_set.scatter_add[i], kernel_set.module,
                               scatter_add_name.c_str());
    driver.module_get_function(&kernel_set.zero[i], kernel_set.module,
                               zero_name.c_str());
  }
  driver.module_get_function(&kernel_set.zero_bins[0], kernel_set.module,
                             "zero_bins_i32");
  driver.module_get_function(&kernel_set.zero_bins[1], kernel_set.module,
                             "zero_bins_i64");
  driver.module_get_function(&kernel_set.histogram[0][0], kernel_set.module,
                             "histogram_i32_i32");
  driver.module_get_function(&kernel_set.histogram[1][0], kernel_set.module,
                             "histogram_u32_i32");
  driver.module_get_function(&kernel_set.histogram[0][1], kernel_set.module,
                             "histogram_i32_i64");
  driver.module_get_function(&kernel_set.histogram[1][1], kernel_set.module,
                             "histogram_u32_i64");
  driver.module_get_function(&kernel_set.compact_rank_tiles,
                             kernel_set.module, "compact_rank_tiles_i32");
  driver.module_get_function(&kernel_set.compact_scatter_tiled,
                             kernel_set.module,
                             "compact_scatter_tiled_words");
  driver.module_get_function(&kernel_set.copy_i32, kernel_set.module,
                             "copy_i32_strided");
  driver.module_get_function(&kernel_set.bucket_scatter, kernel_set.module,
                             "bucket_scatter_words");
  driver.module_get_function(&kernel_set.add_scaled[0], kernel_set.module,
                             "add_scaled_strided_f32");
  driver.module_get_function(&kernel_set.add_scaled[1], kernel_set.module,
                             "add_scaled_strided_f64");
  driver.module_get_function(&kernel_set.sparse_diagonal_apply_f32,
                             kernel_set.module,
                             "sparse_diagonal_apply_f32");
  driver.module_get_function(&kernel_set.sparse_diagonal_refresh_f32,
                             kernel_set.module,
                             "sparse_diagonal_refresh_f32");
  driver.module_get_function(
      &kernel_set.sparse_block_cholesky_refresh_f32, kernel_set.module,
      "sparse_block_cholesky_refresh_f32");
  driver.module_get_function(&kernel_set.sparse_block_diagonal_apply_f32,
                             kernel_set.module,
                             "sparse_block_diagonal_apply_f32");
  driver.module_get_function(&kernel_set.sparse_minres_scalar_f32,
                             kernel_set.module,
                             "sparse_minres_scalar_f32");
  driver.module_get_function(&kernel_set.sparse_minres_vector_state_f32,
                             kernel_set.module,
                             "sparse_minres_vector_state_f32");
  driver.module_get_function(&kernel_set.sparse_minres_commit_f32,
                             kernel_set.module,
                             "sparse_minres_commit_f32");
  driver.module_get_function(&kernel_set.sparse_bicgstab_scalar_f32,
                             kernel_set.module,
                             "sparse_bicgstab_scalar_f32");
  driver.module_get_function(&kernel_set.sparse_bicgstab_direction_f32,
                             kernel_set.module,
                             "sparse_bicgstab_direction_f32");
  driver.module_get_function(&kernel_set.sparse_bicgstab_intermediate_f32,
                             kernel_set.module,
                             "sparse_bicgstab_intermediate_f32");
  driver.module_get_function(&kernel_set.sparse_bicgstab_commit_f32,
                             kernel_set.module,
                             "sparse_bicgstab_commit_f32");
  driver.module_get_function(&kernel_set.sparse_bicgstab_reconcile_f32,
                             kernel_set.module,
                             "sparse_bicgstab_reconcile_f32");
  driver.module_get_function(&kernel_set.sparse_gmres_multi_dot_partial_f32,
                             kernel_set.module,
                             "sparse_gmres_multi_dot_partial_f32");
  driver.module_get_function(&kernel_set.sparse_gmres_multi_dot_final_f32,
                             kernel_set.module,
                             "sparse_gmres_multi_dot_final_f32");
  driver.module_get_function(&kernel_set.sparse_gmres_projection_f32,
                             kernel_set.module,
                             "sparse_gmres_projection_f32");
  driver.module_get_function(&kernel_set.sparse_gmres_basis_f32,
                             kernel_set.module,
                             "sparse_gmres_basis_f32");
  driver.module_get_function(&kernel_set.sparse_gmres_combine_f32,
                             kernel_set.module,
                             "sparse_gmres_combine_f32");
  driver.module_get_function(&kernel_set.sparse_gmres_scalar_f32,
                             kernel_set.module,
                             "sparse_gmres_scalar_f32");
  driver.module_get_function(&kernel_set.gather_add[0], kernel_set.module,
                             "gather_add_strided_f32");
  driver.module_get_function(&kernel_set.gather_add[1], kernel_set.module,
                             "gather_add_strided_f64");
  constexpr std::array<const char *, 6> sort_suffixes{"u32", "i32", "f32",
                                                      "u64", "i64", "f64"};
  for (std::size_t i = 0; i < sort_suffixes.size(); ++i) {
    const std::string radix_init_name =
        std::string("radix_init_") + sort_suffixes[i];
    driver.module_get_function(&kernel_set.radix_init[i], kernel_set.module,
                               radix_init_name.c_str());
  }
  driver.module_get_function(&kernel_set.radix_rank4[0], kernel_set.module,
                             "radix_rank4_u32");
  driver.module_get_function(&kernel_set.radix_rank4[1], kernel_set.module,
                             "radix_rank4_u64");
  driver.module_get_function(&kernel_set.radix_hist_scan, kernel_set.module,
                             "radix_hist_scan");
  driver.module_get_function(&kernel_set.radix_hist_uniform,
                             kernel_set.module, "radix_hist_uniform");
  driver.module_get_function(&kernel_set.radix_scatter4[0], kernel_set.module,
                             "radix_scatter4_u32");
  driver.module_get_function(&kernel_set.radix_scatter4[1], kernel_set.module,
                             "radix_scatter4_u64");
  driver.module_get_function(&kernel_set.radix_gather_words, kernel_set.module,
                             "radix_gather_words");
  driver.module_get_function(&kernel_set.radix_copy_words, kernel_set.module,
                             "radix_copy_words");
  driver.module_get_function(&kernel_set.sparse_assembly_pack_validate,
                             kernel_set.module,
                             "sparse_assembly_pack_validate");
  driver.module_get_function(
      &kernel_set.sparse_assembly_pack_packed_validate, kernel_set.module,
      "sparse_assembly_pack_packed_validate");
  driver.module_get_function(&kernel_set.sparse_assembly_mark_segments,
                             kernel_set.module,
                             "sparse_assembly_mark_segments");
  driver.module_get_function(&kernel_set.sparse_assembly_scatter_segments,
                             kernel_set.module,
                             "sparse_assembly_scatter_segments");
  driver.module_get_function(&kernel_set.sparse_assembly_reduce_segments,
                             kernel_set.module,
                             "sparse_assembly_reduce_segments");
  driver.module_get_function(&kernel_set.sparse_assembly_emit_csr,
                             kernel_set.module, "sparse_assembly_emit_csr");
  driver.module_get_function(&kernel_set.sparse_assembly_finalize_control,
                             kernel_set.module,
                             "sparse_assembly_finalize_control");
  driver.module_get_function(&kernel_set.sparse_csr_spmv_f32,
                             kernel_set.module, "sparse_csr_spmv_f32");
  driver.module_get_function(&kernel_set.sparse_bsr_spmv_f32,
                             kernel_set.module, "sparse_bsr_spmv_f32");
}

KernelSet &kernels() {
  std::call_once(kernel_set_once, load_kernel_set_once);
  return kernel_set;
}

class DriverWorkspace final {
 public:
  DriverWorkspace() = default;
  DriverWorkspace(const DriverWorkspace &) = delete;
  DriverWorkspace &operator=(const DriverWorkspace &) = delete;

  ~DriverWorkspace() {
    release_noexcept();
  }

  void ensure(std::size_t required_bytes) {
    if (required_bytes <= capacity_) {
      return;
    }
    std::size_t new_capacity = std::max<std::size_t>(required_bytes, 4096);
    if (capacity_ != 0 &&
        capacity_ <= std::numeric_limits<std::size_t>::max() / 2) {
      new_capacity = std::max(new_capacity, capacity_ * 2);
    }
    new_capacity = (new_capacity + 255u) & ~std::size_t{255u};

    auto &ctx = CUDAContext::get_instance();
    auto context_guard = ctx.get_guard();
    void *new_block = nullptr;
    CUDADriver::get_instance().malloc(&new_block, new_capacity);
    try {
      allocations_.push_back({new_block, new_capacity});
    } catch (...) {
      CUDADriver::get_instance().mem_free(new_block);
      throw;
    }
    current_ = new_block;
    capacity_ = new_capacity;
    allocated_bytes_ += new_capacity;
  }

  void *data() const noexcept {
    return current_;
  }

  std::size_t allocated_bytes() const noexcept {
    return allocated_bytes_;
  }

 private:
  void release_noexcept() noexcept {
    try {
      auto &ctx = CUDAContext::get_instance();
      auto context_guard = ctx.get_guard();
      auto &driver = CUDADriver::get_instance();
      for (auto it = allocations_.rbegin(); it != allocations_.rend(); ++it) {
        driver.mem_free(it->first);
      }
    } catch (...) {
      // Program teardown already established a synchronization boundary.  A
      // destructor must not mask the original teardown exception.
    }
    allocations_.clear();
    current_ = nullptr;
    capacity_ = 0;
    allocated_bytes_ = 0;
  }

  std::vector<std::pair<void *, std::size_t>> allocations_;
  void *current_{nullptr};
  std::size_t capacity_{0};
  std::size_t allocated_bytes_{0};
};

PrimitiveWorkspaceArena::Lease<DriverWorkspace> acquire_workspace(
    PrimitiveWorkspaceArena *arena,
    PrimitiveWorkspaceFamily family,
    void *stream) {
  TI_ERROR_IF(!arena,
              "CUDA Driver hierarchical primitive requires a Program-owned "
              "workspace arena.");
  PrimitiveWorkspaceKey key{
      PrimitiveWorkspaceBackend::cuda, family,
      static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(stream)),
      kWorkspaceVariant};
  return arena->acquire<DriverWorkspace>(
      key, [] { return std::make_shared<DriverWorkspace>(); });
}

struct LevelLayout {
  std::array<std::uint32_t, 8> counts{};
  std::array<std::size_t, 8> offsets{};
  std::size_t level_count{0};
  std::size_t bytes{0};
};

LevelLayout scan_layout(std::uint32_t num_items, std::size_t value_size) {
  LevelLayout layout;
  std::uint32_t count =
      (num_items + kScanTileItems - 1u) / kScanTileItems;
  while (count > 1u) {
    TI_ASSERT(layout.level_count < layout.counts.size());
    layout.bytes = (layout.bytes + 255u) & ~std::size_t{255u};
    layout.counts[layout.level_count] = count;
    layout.offsets[layout.level_count] = layout.bytes;
    layout.bytes += static_cast<std::size_t>(count) * value_size;
    ++layout.level_count;
    count = (count + kScanTileItems - 1u) / kScanTileItems;
  }
  return layout;
}

LevelLayout reduce_layout(std::uint32_t num_items, std::size_t value_size) {
  LevelLayout layout;
  std::uint32_t count = (num_items + kReduceTileItems - 1u) / kReduceTileItems;
  while (count > 1u) {
    TI_ASSERT(layout.level_count < layout.counts.size());
    layout.bytes = (layout.bytes + 255u) & ~std::size_t{255u};
    layout.counts[layout.level_count] = count;
    layout.offsets[layout.level_count] = layout.bytes;
    layout.bytes += static_cast<std::size_t>(count) * value_size;
    ++layout.level_count;
    if (count <= kReduceTileItems) {
      break;
    }
    count = (count + kReduceTileItems - 1u) / kReduceTileItems;
  }
  return layout;
}

void *byte_offset(void *base, std::size_t offset) {
  return reinterpret_cast<std::uint8_t *>(base) + offset;
}

}  // namespace

bool driver_hierarchical_available() {
  return CUDADriver::get_instance_without_context()
      .nvidia_extensions_available();
}

std::size_t driver_inclusive_scan_strided_for_family(
    void *data,
    int num_items,
    CudaTransformValueType value_type,
    std::size_t offset,
    std::size_t stride,
    bool reverse,
    void *stream,
    PrimitiveWorkspaceArena *workspace_arena,
    PrimitiveWorkspaceFamily workspace_family,
    void *retained_workspace = nullptr) {
  TI_ERROR_IF(num_items < 0,
              "CUDA Driver scan expects non-negative num_items.");
  TI_ERROR_IF(!data, "CUDA Driver scan received a null data pointer.");
  const std::size_t value_size = value_type_size(value_type);
  TI_ERROR_IF(stride < value_size,
              "CUDA Driver scan received a stride smaller than its value.");
  if (num_items <= 1) {
    return 0;
  }

  const std::uint32_t n = static_cast<std::uint32_t>(num_items);
  const auto type_index = value_type_index(value_type);
  auto &kernel = kernels();
  const auto layout = scan_layout(n, value_size);
  std::optional<PrimitiveWorkspaceArena::Lease<DriverWorkspace>> workspace;
  std::array<void *, 8> level_ptrs{};
  if (layout.level_count != 0) {
    if (retained_workspace == nullptr) {
      workspace.emplace(
          acquire_workspace(workspace_arena, workspace_family, stream));
      (*workspace)->ensure(layout.bytes);
      retained_workspace = (*workspace)->data();
    }
    for (std::size_t i = 0; i < layout.level_count; ++i) {
      level_ptrs[i] = byte_offset(retained_workspace, layout.offsets[i]);
    }
  }

  auto launch_scan = [&](void *values, std::uint64_t values_offset,
                         std::uint64_t values_stride, std::uint32_t count,
                         void *sums, std::int32_t reverse_order) {
    void *values_arg = values;
    void *sums_arg = sums;
    std::uint64_t sums_offset = 0;
    std::vector<void *> args{&values_arg,   &values_offset, &values_stride,
                             &count,        &sums_arg,      &sums_offset,
                             &reverse_order};
    const bool use_tiled = count > kBlockDim;
    const std::uint32_t tile_items =
        use_tiled ? kScanTileItems : kBlockDim;
    const unsigned grid = (count + tile_items - 1u) / tile_items;
    CUDAContext::get_instance().launch(
        use_tiled ? kernel.scan_tiled[type_index] : kernel.scan[type_index],
                                       "cuda_driver_scan_blocks", args, {},
                                       grid, kBlockDim, 0, stream);
  };
  auto launch_uniform = [&](void *values, std::uint64_t values_offset,
                            std::uint64_t values_stride, std::uint32_t count,
                            void *sums, std::int32_t reverse_order) {
    void *values_arg = values;
    void *sums_arg = sums;
    std::uint64_t sums_offset = 0;
    std::uint32_t tile_shift =
        count > kBlockDim ? kScanTileShift : kScanBlockShift;
    std::vector<void *> args{&values_arg,   &values_offset, &values_stride,
                             &count,        &sums_arg,      &sums_offset,
                             &reverse_order, &tile_shift};
    const unsigned grid = (count + kBlockDim - 1u) / kBlockDim;
    CUDAContext::get_instance().launch(kernel.uniform_add[type_index],
                                       "cuda_driver_scan_uniform_add", args, {},
                                       grid, kBlockDim, 0, stream);
  };

  launch_scan(data, static_cast<std::uint64_t>(offset),
              static_cast<std::uint64_t>(stride), n, level_ptrs[0],
              reverse ? 1 : 0);
  for (std::size_t i = 0; i < layout.level_count; ++i) {
    void *next = i + 1 < layout.level_count ? level_ptrs[i + 1] : nullptr;
    launch_scan(level_ptrs[i], 0, value_size, layout.counts[i], next, 0);
  }
  for (std::size_t i = layout.level_count; i > 1; --i) {
    const std::size_t level = i - 2;
    launch_uniform(level_ptrs[level], 0, value_size, layout.counts[level],
                   level_ptrs[level + 1], 0);
  }
  if (layout.level_count != 0) {
    launch_uniform(data, static_cast<std::uint64_t>(offset),
                   static_cast<std::uint64_t>(stride), n, level_ptrs[0],
                   reverse ? 1 : 0);
    return workspace ? (*workspace)->allocated_bytes() : layout.bytes;
  }
  return 0;
}

std::size_t driver_inclusive_scan_strided(
    void *data,
    int num_items,
    CudaTransformValueType value_type,
    std::size_t offset,
    std::size_t stride,
    bool reverse,
    void *stream,
    PrimitiveWorkspaceArena *workspace_arena) {
  return driver_inclusive_scan_strided_for_family(
      data, num_items, value_type, offset, stride, reverse, stream,
      workspace_arena, PrimitiveWorkspaceFamily::scan);
}

std::size_t driver_scan_workspace_bytes(int num_items,
                                        CudaTransformValueType value_type) {
  TI_ERROR_IF(num_items < 0, "CUDA scan requires a non-negative size");
  return scan_layout(static_cast<std::uint32_t>(num_items),
                     value_type_size(value_type))
      .bytes;
}

void driver_prepare_scan() {
  (void)kernels();
}

void driver_inclusive_scan_with_workspace(void *data,
                                          int num_items,
                                          CudaTransformValueType value_type,
                                          void *workspace,
                                          void *stream) {
  driver_inclusive_scan_strided_for_family(
      data, num_items, value_type, 0, value_type_size(value_type), false,
      stream, nullptr, PrimitiveWorkspaceFamily::scan, workspace);
}

std::size_t driver_reduce_strided(void *values,
                                  void *output,
                                  int num_items,
                                  CudaTransformValueType value_type,
                                  std::size_t values_offset,
                                  std::size_t values_stride,
                                  std::size_t output_offset,
                                  std::size_t output_stride,
                                  CudaHierarchicalReduceOp op,
                                  void *stream,
                                  PrimitiveWorkspaceArena *workspace_arena) {
  TI_ERROR_IF(num_items <= 0,
              "CUDA Driver reduce expects at least one input item.");
  TI_ERROR_IF(!values || !output,
              "CUDA Driver reduce received a null pointer.");
  const std::size_t value_size = value_type_size(value_type);
  TI_ERROR_IF(values_stride < value_size || output_stride < value_size,
              "CUDA Driver reduce received a stride smaller than its value.");
  const int op_value = static_cast<int>(op);
  TI_ERROR_IF(op_value < 0 || op_value > 2,
              "CUDA Driver reduce received an unsupported operation.");

  const std::uint32_t n = static_cast<std::uint32_t>(num_items);
  const auto layout = reduce_layout(n, value_size);
  std::optional<PrimitiveWorkspaceArena::Lease<DriverWorkspace>> workspace;
  std::array<void *, 8> level_ptrs{};
  if (layout.level_count != 0) {
    workspace.emplace(acquire_workspace(
        workspace_arena, PrimitiveWorkspaceFamily::reduce, stream));
    (*workspace)->ensure(layout.bytes);
    for (std::size_t i = 0; i < layout.level_count; ++i) {
      level_ptrs[i] = byte_offset((*workspace)->data(), layout.offsets[i]);
    }
  }

  const auto type_index = value_type_index(value_type);
  auto &kernel = kernels();
  auto launch_reduce = [&](void *input, std::uint64_t input_offset,
                           std::uint64_t input_stride, std::uint32_t count,
                           void *result, std::uint64_t result_offset,
                           std::uint64_t result_stride) {
    void *input_arg = input;
    void *result_arg = result;
    std::int32_t operation = op_value;
    std::vector<void *> args{&input_arg,     &input_offset, &input_stride,
                             &count,         &result_arg,   &result_offset,
                             &result_stride, &operation};
    const unsigned grid = (count + kBlockDim - 1u) / kBlockDim;
    CUDAContext::get_instance().launch(kernel.reduce[type_index],
                                       "cuda_driver_reduce_blocks", args, {},
                                       grid, kBlockDim, 0, stream);
  };

  if (layout.level_count == 0) {
    launch_reduce(values, static_cast<std::uint64_t>(values_offset),
                  static_cast<std::uint64_t>(values_stride), n, output,
                  static_cast<std::uint64_t>(output_offset),
                  static_cast<std::uint64_t>(output_stride));
    return 0;
  }

  launch_reduce(values, static_cast<std::uint64_t>(values_offset),
                static_cast<std::uint64_t>(values_stride), n, level_ptrs[0], 0,
                value_size);
  for (std::size_t i = 0; i + 1 < layout.level_count; ++i) {
    launch_reduce(level_ptrs[i], 0, value_size, layout.counts[i],
                  level_ptrs[i + 1], 0, value_size);
  }
  const std::size_t final_level = layout.level_count - 1;
  launch_reduce(level_ptrs[final_level], 0, value_size,
                layout.counts[final_level], output,
                static_cast<std::uint64_t>(output_offset),
                static_cast<std::uint64_t>(output_stride));
  return (*workspace)->allocated_bytes();
}

std::size_t driver_histogram_strided(void *values,
                                     void *bins,
                                     int num_items,
                                     int num_bins,
                                     CudaTransformValueType value_type,
                                     CudaTransformValueType bin_type,
                                     std::size_t values_offset,
                                     std::size_t values_stride,
                                     std::size_t bins_offset,
                                     std::size_t bins_stride,
                                     void *stream) {
  TI_ERROR_IF(num_items < 0,
              "CUDA Driver histogram expects non-negative num_items.");
  TI_ERROR_IF(num_bins <= 0, "CUDA Driver histogram expects at least one bin.");
  TI_ERROR_IF((num_items > 0 && !values) || !bins,
              "CUDA Driver histogram received a null pointer.");
  const int value_index = value_type == CudaTransformValueType::i32   ? 0
                          : value_type == CudaTransformValueType::u32 ? 1
                                                                      : -1;
  const int bin_index = bin_type == CudaTransformValueType::i32   ? 0
                        : bin_type == CudaTransformValueType::i64 ? 1
                                                                  : -1;
  TI_ERROR_IF(value_index < 0,
              "CUDA Driver histogram supports only i32/u32 bin ids.");
  TI_ERROR_IF(bin_index < 0,
              "CUDA Driver histogram supports only i32/i64 counters.");
  TI_ERROR_IF(values_stride < value_type_size(value_type) ||
                  bins_stride < value_type_size(bin_type),
              "CUDA Driver histogram received an invalid stride.");

  auto &kernel = kernels();
  void *bins_arg = bins;
  std::uint64_t bins_offset_arg = bins_offset;
  std::uint64_t bins_stride_arg = bins_stride;
  std::uint32_t bins_count_arg = static_cast<std::uint32_t>(num_bins);
  std::vector<void *> zero_args{&bins_arg, &bins_offset_arg, &bins_stride_arg,
                                &bins_count_arg};
  const unsigned bins_grid = (bins_count_arg + kBlockDim - 1u) / kBlockDim;
  CUDAContext::get_instance().launch(kernel.zero_bins[bin_index],
                                     "cuda_driver_histogram_zero", zero_args,
                                     {}, bins_grid, kBlockDim, 0, stream);

  if (num_items == 0) {
    return 0;
  }
  void *values_arg = values;
  std::uint64_t values_offset_arg = values_offset;
  std::uint64_t values_stride_arg = values_stride;
  std::uint32_t item_count_arg = static_cast<std::uint32_t>(num_items);
  std::vector<void *> histogram_args{
      &values_arg, &values_offset_arg, &values_stride_arg, &item_count_arg,
      &bins_arg,   &bins_offset_arg,   &bins_stride_arg,   &bins_count_arg};
  const unsigned item_grid = (item_count_arg + kBlockDim - 1u) / kBlockDim;
  CUDAContext::get_instance().launch(kernel.histogram[value_index][bin_index],
                                     "cuda_driver_histogram", histogram_args,
                                     {}, item_grid, kBlockDim, 0, stream);
  return 0;
}

std::size_t driver_add_strided(void *src,
                               void *dst,
                               int num_items,
                               CudaTransformValueType value_type,
                               std::size_t src_offset,
                               std::size_t src_stride,
                               std::size_t dst_offset,
                               std::size_t dst_stride,
                               void *stream) {
  TI_ERROR_IF(num_items < 0, "CUDA Driver add expects non-negative num_items.");
  const std::size_t value_size = value_type_size(value_type);
  TI_ERROR_IF((num_items > 0 && (!src || !dst)) ||
                  (src_stride != 0 && src_stride < value_size) ||
                  dst_stride < value_size,
              "CUDA Driver add received an invalid pointer or stride.");
  if (num_items == 0) {
    return 0;
  }
  void *src_arg = src;
  void *dst_arg = dst;
  std::uint64_t src_offset_arg = src_offset;
  std::uint64_t src_stride_arg = src_stride;
  std::uint64_t dst_offset_arg = dst_offset;
  std::uint64_t dst_stride_arg = dst_stride;
  std::uint32_t count_arg = static_cast<std::uint32_t>(num_items);
  std::vector<void *> args{&src_arg,  &src_offset_arg, &src_stride_arg,
                           &dst_arg,  &dst_offset_arg, &dst_stride_arg,
                           &count_arg};
  const unsigned grid = (count_arg + kBlockDim - 1u) / kBlockDim;
  CUDAContext::get_instance().launch(
      kernels().add[value_type_index(value_type)], "cuda_driver_add", args, {},
      grid, kBlockDim, 0, stream);
  return 0;
}

std::size_t driver_add_scaled_strided(void *src,
                                      void *dst,
                                      int num_items,
                                      CudaTransformValueType value_type,
                                      std::size_t src_offset,
                                      std::size_t src_stride,
                                      std::size_t dst_offset,
                                      std::size_t dst_stride,
                                      double scale,
                                      void *stream) {
  TI_ERROR_IF(value_type != CudaTransformValueType::f32 &&
                  value_type != CudaTransformValueType::f64,
              "CUDA Driver scaled add supports only f32/f64.");
  TI_ERROR_IF(num_items < 0,
              "CUDA Driver scaled add expects non-negative num_items.");
  const std::size_t value_size = value_type_size(value_type);
  TI_ERROR_IF((num_items > 0 && (!src || !dst)) ||
                  (src_stride != 0 && src_stride < value_size) ||
                  dst_stride < value_size,
              "CUDA Driver scaled add received an invalid pointer or stride.");
  if (num_items == 0) {
    return 0;
  }
  void *src_arg = src;
  void *dst_arg = dst;
  std::uint64_t src_offset_arg = src_offset;
  std::uint64_t src_stride_arg = src_stride;
  std::uint64_t dst_offset_arg = dst_offset;
  std::uint64_t dst_stride_arg = dst_stride;
  std::uint32_t count_arg = static_cast<std::uint32_t>(num_items);
  double scale_arg = scale;
  std::vector<void *> args{&src_arg,   &src_offset_arg, &src_stride_arg,
                           &dst_arg,   &dst_offset_arg, &dst_stride_arg,
                           &count_arg, &scale_arg};
  const unsigned grid = (count_arg + kBlockDim - 1u) / kBlockDim;
  const std::size_t type_index =
      value_type == CudaTransformValueType::f32 ? 0 : 1;
  CUDAContext::get_instance().launch(kernels().add_scaled[type_index],
                                     "cuda_driver_add_scaled", args, {}, grid,
                                     kBlockDim, 0, stream);
  return 0;
}

std::size_t driver_scatter_add_strided(void *src,
                                       void *indices,
                                       void *dst,
                                       int num_items,
                                       int index_bound,
                                       CudaTransformValueType value_type,
                                       std::size_t src_offset,
                                       std::size_t src_stride,
                                       std::size_t indices_offset,
                                       std::size_t indices_stride,
                                       std::size_t dst_offset,
                                       std::size_t dst_stride,
                                       void *stream) {
  TI_ERROR_IF(num_items < 0 || index_bound < 0,
              "CUDA Driver scatter-add received a negative size.");
  const std::size_t value_size = value_type_size(value_type);
  TI_ERROR_IF((num_items > 0 && (!src || !indices || !dst)) ||
                  src_stride < value_size ||
                  indices_stride < sizeof(std::int32_t) ||
                  dst_stride < value_size,
              "CUDA Driver scatter-add received an invalid pointer or stride.");
  if (num_items == 0 || index_bound == 0) {
    return 0;
  }
  void *src_arg = src;
  void *indices_arg = indices;
  void *dst_arg = dst;
  std::uint64_t src_offset_arg = src_offset;
  std::uint64_t src_stride_arg = src_stride;
  std::uint64_t indices_offset_arg = indices_offset;
  std::uint64_t indices_stride_arg = indices_stride;
  std::uint64_t dst_offset_arg = dst_offset;
  std::uint64_t dst_stride_arg = dst_stride;
  std::uint32_t count_arg = static_cast<std::uint32_t>(num_items);
  std::uint32_t bound_arg = static_cast<std::uint32_t>(index_bound);
  std::vector<void *> args{
      &src_arg,     &src_offset_arg,     &src_stride_arg,
      &indices_arg, &indices_offset_arg, &indices_stride_arg,
      &dst_arg,     &dst_offset_arg,     &dst_stride_arg,
      &count_arg,   &bound_arg};
  const unsigned grid = (count_arg + kBlockDim - 1u) / kBlockDim;
  CUDAContext::get_instance().launch(
      kernels().scatter_add[value_type_index(value_type)],
      "cuda_driver_scatter_add", args, {}, grid, kBlockDim, 0, stream);
  return 0;
}

std::size_t driver_gather_add_strided(void *src,
                                      void *indices,
                                      void *dst,
                                      int num_items,
                                      int index_bound,
                                      CudaTransformValueType value_type,
                                      std::size_t src_offset,
                                      std::size_t src_stride,
                                      std::size_t indices_offset,
                                      std::size_t indices_stride,
                                      std::size_t dst_offset,
                                      std::size_t dst_stride,
                                      void *stream) {
  TI_ERROR_IF(value_type != CudaTransformValueType::f32 &&
                  value_type != CudaTransformValueType::f64,
              "CUDA Driver gather-add supports only f32/f64.");
  TI_ERROR_IF(num_items < 0 || index_bound < 0,
              "CUDA Driver gather-add received a negative size.");
  const std::size_t value_size = value_type_size(value_type);
  TI_ERROR_IF((num_items > 0 && (!src || !indices || !dst)) ||
                  src_stride < value_size ||
                  indices_stride < sizeof(std::int32_t) ||
                  dst_stride < value_size,
              "CUDA Driver gather-add received an invalid pointer or stride.");
  if (num_items == 0 || index_bound == 0) {
    return 0;
  }
  void *src_arg = src;
  void *indices_arg = indices;
  void *dst_arg = dst;
  std::uint64_t src_offset_arg = src_offset;
  std::uint64_t src_stride_arg = src_stride;
  std::uint64_t indices_offset_arg = indices_offset;
  std::uint64_t indices_stride_arg = indices_stride;
  std::uint64_t dst_offset_arg = dst_offset;
  std::uint64_t dst_stride_arg = dst_stride;
  std::uint32_t count_arg = static_cast<std::uint32_t>(num_items);
  std::uint32_t bound_arg = static_cast<std::uint32_t>(index_bound);
  std::vector<void *> args{
      &src_arg,     &src_offset_arg,     &src_stride_arg,
      &indices_arg, &indices_offset_arg, &indices_stride_arg,
      &dst_arg,     &dst_offset_arg,     &dst_stride_arg,
      &count_arg,   &bound_arg};
  const unsigned grid = (count_arg + kBlockDim - 1u) / kBlockDim;
  const std::size_t type_index =
      value_type == CudaTransformValueType::f32 ? 0 : 1;
  CUDAContext::get_instance().launch(kernels().gather_add[type_index],
                                     "cuda_driver_gather_add", args, {}, grid,
                                     kBlockDim, 0, stream);
  return 0;
}

std::size_t driver_zero_strided(void *dst,
                                int num_items,
                                CudaTransformValueType value_type,
                                std::size_t dst_offset,
                                std::size_t dst_stride,
                                void *stream) {
  TI_ERROR_IF(num_items < 0,
              "CUDA Driver zero expects non-negative num_items.");
  const std::size_t value_size = value_type_size(value_type);
  TI_ERROR_IF((num_items > 0 && !dst) || dst_stride < value_size,
              "CUDA Driver zero received an invalid pointer or stride.");
  if (num_items == 0) {
    return 0;
  }
  void *dst_arg = dst;
  std::uint64_t dst_offset_arg = dst_offset;
  std::uint64_t dst_stride_arg = dst_stride;
  std::uint32_t count_arg = static_cast<std::uint32_t>(num_items);
  std::vector<void *> args{&dst_arg, &dst_offset_arg, &dst_stride_arg,
                           &count_arg};
  const unsigned grid = (count_arg + kBlockDim - 1u) / kBlockDim;
  CUDAContext::get_instance().launch(
      kernels().zero[value_type_index(value_type)], "cuda_driver_zero", args,
      {}, grid, kBlockDim, 0, stream);
  return 0;
}

std::size_t driver_compact_strided(void *values,
                                   void *flags,
                                   void *output,
                                   void *count,
                                   int num_items,
                                   int item_words,
                                   std::size_t values_offset,
                                   std::size_t values_stride,
                                   std::size_t flags_offset,
                                   std::size_t flags_stride,
                                   std::size_t output_offset,
                                   std::size_t output_stride,
                                   std::size_t count_offset,
                                   void *stream,
                                   PrimitiveWorkspaceArena *workspace_arena,
                                   bool reuse_prefix) {
  TI_ERROR_IF(num_items < 0 || item_words <= 0,
              "CUDA Driver compact received an invalid size.");
  const std::size_t item_bytes =
      static_cast<std::size_t>(item_words) * sizeof(std::uint32_t);
  TI_ERROR_IF(!count || (num_items > 0 && (!values || !flags || !output)) ||
                  values_stride < item_bytes ||
                  flags_stride < sizeof(std::int32_t) ||
                  output_stride < item_bytes,
              "CUDA Driver compact received an invalid pointer or stride.");
  if (num_items == 0) {
    return driver_zero_strided(count, 1, CudaTransformValueType::i32,
                               count_offset, sizeof(std::int32_t), stream);
  }
  const std::size_t prefix_bytes =
      static_cast<std::size_t>(num_items) * sizeof(std::int32_t);
  const std::size_t tile_count =
      (static_cast<std::size_t>(num_items) + kScanTileItems - 1u) /
      kScanTileItems;
  const std::size_t tile_counts_offset =
      (prefix_bytes + 255u) & ~std::size_t{255u};
  TI_ERROR_IF(tile_count >
                  (std::numeric_limits<std::size_t>::max() -
                   tile_counts_offset) /
                      sizeof(std::int32_t),
              "CUDA Driver compact workspace size overflow.");
  const std::size_t required_bytes =
      tile_counts_offset + tile_count * sizeof(std::int32_t);
  auto workspace = acquire_workspace(workspace_arena,
                                     PrimitiveWorkspaceFamily::compact, stream);
  workspace->ensure(required_bytes);
  auto *workspace_base = static_cast<std::uint8_t *>(workspace->data());
  void *prefix = workspace_base;
  void *tile_counts = workspace_base + tile_counts_offset;
  void *flags_arg = flags;
  std::uint64_t flags_offset_arg = flags_offset;
  std::uint64_t flags_stride_arg = flags_stride;
  void *prefix_arg = prefix;
  void *tile_counts_arg = tile_counts;
  std::uint32_t count_arg = static_cast<std::uint32_t>(num_items);
  std::vector<void *> rank_args{&flags_arg, &flags_offset_arg,
                                &flags_stride_arg, &prefix_arg,
                                &tile_counts_arg, &count_arg};
  const unsigned tile_grid = static_cast<unsigned>(tile_count);
  std::size_t scan_bytes = 0;
  if (!reuse_prefix) {
    CUDAContext::get_instance().launch(
        kernels().compact_rank_tiles, "cuda_driver_compact_rank_tiles",
        rank_args, {}, tile_grid, kBlockDim, 0, stream);
    scan_bytes = tile_count > 1
                     ? driver_inclusive_scan_strided(
                           tile_counts, static_cast<int>(tile_count),
                           CudaTransformValueType::i32, 0,
                           sizeof(std::int32_t), false, stream, workspace_arena)
                     : 0;
  }

  void *values_arg = values;
  std::uint64_t values_offset_arg = values_offset;
  std::uint64_t values_stride_arg = values_stride;
  void *output_arg = output;
  std::uint64_t output_offset_arg = output_offset;
  std::uint64_t output_stride_arg = output_stride;
  void *count_output_arg = count;
  std::uint64_t count_offset_arg = count_offset;
  std::uint32_t words_arg = static_cast<std::uint32_t>(item_words);
  std::vector<void *> scatter_args{
      &values_arg,        &values_offset_arg, &values_stride_arg,
      &prefix_arg,        &tile_counts_arg,   &output_arg,
      &output_offset_arg,
      &output_stride_arg, &count_output_arg,  &count_offset_arg,
      &count_arg,         &words_arg};
  const unsigned grid = (count_arg + kBlockDim - 1u) / kBlockDim;
  CUDAContext::get_instance().launch(
      kernels().compact_scatter_tiled, "cuda_driver_compact_scatter",
      scatter_args, {}, grid, kBlockDim, 0, stream);
  return workspace->allocated_bytes() + scan_bytes;
}

std::size_t driver_bucket_builder_strided(
    void *keys,
    void *values,
    void *offsets,
    void *output,
    void *cursor,
    int num_items,
    int num_bins,
    int item_words,
    std::size_t keys_offset,
    std::size_t keys_stride,
    std::size_t values_offset,
    std::size_t values_stride,
    std::size_t offsets_offset,
    std::size_t offsets_stride,
    std::size_t output_offset,
    std::size_t output_stride,
    void *stream,
    PrimitiveWorkspaceArena *workspace_arena) {
  TI_ERROR_IF(num_items < 0 || num_bins <= 0 || item_words <= 0,
              "CUDA Driver bucket builder received an invalid size.");
  const std::size_t item_bytes =
      static_cast<std::size_t>(item_words) * sizeof(std::uint32_t);
  TI_ERROR_IF(
      !keys || !values || !offsets || !output || !cursor ||
          keys_stride < sizeof(std::int32_t) || values_stride < item_bytes ||
          offsets_stride < sizeof(std::int32_t) || output_stride < item_bytes,
      "CUDA Driver bucket builder received an invalid pointer or "
      "stride.");
  driver_zero_strided(offsets, 1, CudaTransformValueType::i32, offsets_offset,
                      offsets_stride, stream);
  driver_histogram_strided(
      keys, offsets, num_items, num_bins, CudaTransformValueType::i32,
      CudaTransformValueType::i32, keys_offset, keys_stride,
      offsets_offset + offsets_stride, offsets_stride, stream);
  const std::size_t scan_bytes = driver_inclusive_scan_strided(
      offsets, num_bins + 1, CudaTransformValueType::i32, offsets_offset,
      offsets_stride, false, stream, workspace_arena);

  void *offsets_arg = offsets;
  std::uint64_t offsets_offset_arg = offsets_offset;
  std::uint64_t offsets_stride_arg = offsets_stride;
  void *cursor_arg = cursor;
  std::uint64_t cursor_offset_arg = 0;
  std::uint64_t cursor_stride_arg = sizeof(std::int32_t);
  std::uint32_t bins_arg = static_cast<std::uint32_t>(num_bins);
  std::vector<void *> copy_args{
      &offsets_arg, &offsets_offset_arg, &offsets_stride_arg,
      &cursor_arg,  &cursor_offset_arg,  &cursor_stride_arg,
      &bins_arg};
  const unsigned bins_grid = (bins_arg + kBlockDim - 1u) / kBlockDim;
  CUDAContext::get_instance().launch(kernels().copy_i32,
                                     "cuda_driver_bucket_cursor", copy_args, {},
                                     bins_grid, kBlockDim, 0, stream);

  if (num_items > 0) {
    void *keys_arg = keys;
    std::uint64_t keys_offset_arg = keys_offset;
    std::uint64_t keys_stride_arg = keys_stride;
    void *values_arg = values;
    std::uint64_t values_offset_arg = values_offset;
    std::uint64_t values_stride_arg = values_stride;
    void *output_arg = output;
    std::uint64_t output_offset_arg = output_offset;
    std::uint64_t output_stride_arg = output_stride;
    std::uint32_t count_arg = static_cast<std::uint32_t>(num_items);
    std::uint32_t words_arg = static_cast<std::uint32_t>(item_words);
    std::vector<void *> scatter_args{
        &keys_arg,   &keys_offset_arg,   &keys_stride_arg,
        &values_arg, &values_offset_arg, &values_stride_arg,
        &output_arg, &output_offset_arg, &output_stride_arg,
        &cursor_arg, &count_arg,         &bins_arg,
        &words_arg};
    const unsigned grid = (count_arg + kBlockDim - 1u) / kBlockDim;
    CUDAContext::get_instance().launch(
        kernels().bucket_scatter, "cuda_driver_bucket_scatter", scatter_args,
        {}, grid, kBlockDim, 0, stream);
  }
  return scan_bytes;
}

std::size_t driver_grouped_reduce_strided(void *keys,
                                          void *values,
                                          void *output,
                                          int num_items,
                                          int num_groups,
                                          CudaTransformValueType value_type,
                                          std::size_t keys_offset,
                                          std::size_t keys_stride,
                                          std::size_t values_offset,
                                          std::size_t values_stride,
                                          std::size_t output_offset,
                                          std::size_t output_stride,
                                          void *stream) {
  TI_ERROR_IF(num_items < 0 || num_groups <= 0,
              "CUDA Driver grouped reduce received an invalid size.");
  driver_zero_strided(output, num_groups, value_type, output_offset,
                      output_stride, stream);
  return driver_scatter_add_strided(values, keys, output, num_items, num_groups,
                                    value_type, values_offset, values_stride,
                                    keys_offset, keys_stride, output_offset,
                                    output_stride, stream);
}

void driver_sparse_diagonal_apply_f32(void *inverse_diagonal,
                                      void *input,
                                      void *output,
                                      int num_items,
                                      void *stream) {
  TI_ERROR_IF(num_items <= 0 || !inverse_diagonal || !input || !output,
              "CUDA Driver sparse diagonal apply received an invalid size "
              "or pointer.");
  void *inverse_arg = inverse_diagonal;
  void *input_arg = input;
  void *output_arg = output;
  std::uint32_t count_arg = static_cast<std::uint32_t>(num_items);
  std::vector<void *> args{
      &inverse_arg, &input_arg, &output_arg, &count_arg};
  const unsigned grid = (count_arg + kBlockDim - 1u) / kBlockDim;
  CUDAContext::get_instance().launch(
      kernels().sparse_diagonal_apply_f32,
      "cuda_driver_sparse_diagonal_apply_f32", args, {}, grid, kBlockDim, 0,
      stream);
}

void driver_sparse_diagonal_refresh_f32(void *values,
                                        void *diagonal_offsets,
                                        void *staging_inverse,
                                        void *status,
                                        int rows,
                                        int nnz,
                                        void *stream) {
  TI_ERROR_IF(rows <= 0 || nnz <= 0 || !values || !diagonal_offsets ||
                  !staging_inverse || !status,
              "CUDA Driver sparse diagonal refresh received invalid "
              "geometry or a null pointer.");
  void *values_arg = values;
  void *offsets_arg = diagonal_offsets;
  void *staging_arg = staging_inverse;
  void *status_arg = status;
  std::uint32_t rows_arg = static_cast<std::uint32_t>(rows);
  std::uint32_t nnz_arg = static_cast<std::uint32_t>(nnz);
  std::vector<void *> args{&values_arg, &offsets_arg, &staging_arg,
                           &status_arg, &rows_arg, &nnz_arg};
  const unsigned grid = (rows_arg + kBlockDim - 1u) / kBlockDim;
  CUDAContext::get_instance().launch(
      kernels().sparse_diagonal_refresh_f32,
      "cuda_driver_sparse_diagonal_refresh_f32", args, {}, grid,
      kBlockDim, 0, stream);
}

void driver_sparse_block_cholesky_refresh_f32(
    void *values,
    void *diagonal_block_offsets,
    void *staging_factors,
    void *status,
    int block_rows,
    int block_nnz,
    int block_size,
    void *stream) {
  TI_ERROR_IF(block_rows <= 0 || block_nnz <= 0 ||
                  (block_size != 2 && block_size != 3 && block_size != 6 &&
                   block_size != 12) ||
                  !values || !diagonal_block_offsets || !staging_factors ||
                  !status,
              "CUDA Driver sparse block Cholesky refresh received invalid "
              "geometry or a null pointer.");
  void *values_arg = values;
  void *offsets_arg = diagonal_block_offsets;
  void *staging_arg = staging_factors;
  void *status_arg = status;
  std::uint32_t block_rows_arg =
      static_cast<std::uint32_t>(block_rows);
  std::uint32_t block_nnz_arg =
      static_cast<std::uint32_t>(block_nnz);
  std::uint32_t block_size_arg =
      static_cast<std::uint32_t>(block_size);
  std::vector<void *> args{&values_arg, &offsets_arg, &staging_arg,
                           &status_arg, &block_rows_arg, &block_nnz_arg,
                           &block_size_arg};
  const unsigned grid =
      (block_rows_arg + kBlockDim - 1u) / kBlockDim;
  CUDAContext::get_instance().launch(
      kernels().sparse_block_cholesky_refresh_f32,
      "cuda_driver_sparse_block_cholesky_refresh_f32", args, {}, grid,
      kBlockDim, 0, stream);
}

void driver_sparse_block_diagonal_apply_f32(void *factor_blocks,
                                            void *input,
                                            void *output,
                                            int block_rows,
                                            int block_size,
                                            void *stream) {
  TI_ERROR_IF(block_rows <= 0 ||
                  (block_size != 2 && block_size != 3 && block_size != 6 &&
                   block_size != 12) ||
                  !factor_blocks || !input || !output,
              "CUDA Driver sparse block diagonal apply received invalid "
              "geometry or a null pointer.");
  void *factor_arg = factor_blocks;
  void *input_arg = input;
  void *output_arg = output;
  std::uint32_t block_rows_arg =
      static_cast<std::uint32_t>(block_rows);
  std::uint32_t block_size_arg =
      static_cast<std::uint32_t>(block_size);
  std::vector<void *> args{&factor_arg, &input_arg, &output_arg,
                           &block_rows_arg, &block_size_arg};
  const unsigned grid =
      (block_rows_arg + kBlockDim - 1u) / kBlockDim;
  CUDAContext::get_instance().launch(
      kernels().sparse_block_diagonal_apply_f32,
      "cuda_driver_sparse_block_diagonal_apply_f32", args, {}, grid,
      kBlockDim, 0, stream);
}

void driver_sparse_minres_scalar_f32(void *initial_residual_squared,
                                     void *rhs_squared,
                                     void *dot,
                                     void *state,
                                     float absolute_tolerance,
                                     float relative_tolerance,
                                     int stage,
                                     bool limit_reached,
                                     bool has_preconditioner,
                                     bool stop_on_estimate,
                                     void *stream) {
  TI_ERROR_IF(!initial_residual_squared || !rhs_squared || !dot || !state ||
                  stage < 0 || stage > 4,
              "CUDA Driver sparse MINRES scalar stage received an invalid "
              "pointer or stage.");
  void *initial_arg = initial_residual_squared;
  void *rhs_arg = rhs_squared;
  void *dot_arg = dot;
  void *state_arg = state;
  std::uint32_t stage_arg = static_cast<std::uint32_t>(stage);
  std::uint32_t limit_arg = limit_reached ? 1u : 0u;
  std::uint32_t preconditioner_arg = has_preconditioner ? 1u : 0u;
  std::uint32_t stop_arg = stop_on_estimate ? 1u : 0u;
  std::vector<void *> args{&initial_arg, &rhs_arg, &dot_arg, &state_arg,
                           &absolute_tolerance, &relative_tolerance,
                           &stage_arg, &limit_arg, &preconditioner_arg,
                           &stop_arg};
  CUDAContext::get_instance().launch(
      kernels().sparse_minres_scalar_f32,
      "cuda_driver_sparse_minres_scalar_f32", args, {}, 1, 1, 0, stream);
}

void driver_sparse_minres_vector_state_f32(void *source,
                                           void *destination,
                                           void *state,
                                           int num_items,
                                           int coefficient_index,
                                           bool add,
                                           void *stream) {
  TI_ERROR_IF(!source || !destination || !state || num_items <= 0 ||
                  coefficient_index < 0 || coefficient_index >= 25,
              "CUDA Driver sparse MINRES vector stage received invalid "
              "geometry or a null pointer.");
  void *source_arg = source;
  void *destination_arg = destination;
  void *state_arg = state;
  std::uint32_t count_arg = static_cast<std::uint32_t>(num_items);
  std::uint32_t coefficient_arg =
      static_cast<std::uint32_t>(coefficient_index);
  std::uint32_t add_arg = add ? 1u : 0u;
  std::vector<void *> args{&source_arg, &destination_arg, &state_arg,
                           &count_arg, &coefficient_arg, &add_arg};
  const unsigned grid = (count_arg + kBlockDim - 1u) / kBlockDim;
  CUDAContext::get_instance().launch(
      kernels().sparse_minres_vector_state_f32,
      "cuda_driver_sparse_minres_vector_state_f32", args, {}, grid,
      kBlockDim, 0, stream);
}

void driver_sparse_minres_commit_f32(void *v,
                                     void *r1,
                                     void *r2,
                                     void *lanczos_residual,
                                     void *w_older,
                                     void *w_old,
                                     void *w,
                                     void *solution,
                                     void *state,
                                     int num_items,
                                     void *stream) {
  TI_ERROR_IF(!v || !r1 || !r2 || !lanczos_residual || !w_older ||
                  !w_old || !w || !solution || !state || num_items <= 0,
              "CUDA Driver sparse MINRES commit received invalid geometry "
              "or a null pointer.");
  void *v_arg = v;
  void *r1_arg = r1;
  void *r2_arg = r2;
  void *lanczos_arg = lanczos_residual;
  void *w_older_arg = w_older;
  void *w_old_arg = w_old;
  void *w_arg = w;
  void *solution_arg = solution;
  void *state_arg = state;
  std::uint32_t count_arg = static_cast<std::uint32_t>(num_items);
  std::vector<void *> args{&v_arg, &r1_arg, &r2_arg, &lanczos_arg,
                           &w_older_arg, &w_old_arg, &w_arg, &solution_arg,
                           &state_arg, &count_arg};
  const unsigned grid = (count_arg + kBlockDim - 1u) / kBlockDim;
  CUDAContext::get_instance().launch(
      kernels().sparse_minres_commit_f32,
      "cuda_driver_sparse_minres_commit_f32", args, {}, grid, kBlockDim,
      0, stream);
}

void driver_sparse_bicgstab_scalar_f32(void *initial_residual_squared,
                                       void *rhs_squared,
                                       void *dot0,
                                       void *dot1,
                                       void *state,
                                       float absolute_tolerance,
                                       float relative_tolerance,
                                       int stage,
                                       bool limit_reached,
                                       void *stream) {
  TI_ERROR_IF(!initial_residual_squared || !rhs_squared || !dot0 || !dot1 ||
                  !state || stage < 0 || stage > 6,
              "CUDA Driver sparse BiCGSTAB scalar stage received an "
              "invalid pointer or stage.");
  void *initial_arg = initial_residual_squared;
  void *rhs_arg = rhs_squared;
  void *dot0_arg = dot0;
  void *dot1_arg = dot1;
  void *state_arg = state;
  std::uint32_t stage_arg = static_cast<std::uint32_t>(stage);
  std::uint32_t limit_arg = limit_reached ? 1u : 0u;
  std::vector<void *> args{&initial_arg, &rhs_arg, &dot0_arg, &dot1_arg,
                           &state_arg, &absolute_tolerance,
                           &relative_tolerance, &stage_arg, &limit_arg};
  CUDAContext::get_instance().launch(
      kernels().sparse_bicgstab_scalar_f32,
      "cuda_driver_sparse_bicgstab_scalar_f32", args, {}, 1, 1, 0,
      stream);
}

void driver_sparse_bicgstab_direction_f32(void *residual,
                                          void *direction,
                                          void *operator_direction,
                                          void *state,
                                          int num_items,
                                          void *stream) {
  TI_ERROR_IF(!residual || !direction || !operator_direction || !state ||
                  num_items <= 0,
              "CUDA Driver sparse BiCGSTAB direction received invalid "
              "geometry or a null pointer.");
  void *residual_arg = residual;
  void *direction_arg = direction;
  void *operator_direction_arg = operator_direction;
  void *state_arg = state;
  std::uint32_t count_arg = static_cast<std::uint32_t>(num_items);
  std::vector<void *> args{&residual_arg, &direction_arg,
                           &operator_direction_arg, &state_arg,
                           &count_arg};
  const unsigned grid = (count_arg + kBlockDim - 1u) / kBlockDim;
  CUDAContext::get_instance().launch(
      kernels().sparse_bicgstab_direction_f32,
      "cuda_driver_sparse_bicgstab_direction_f32", args, {}, grid,
      kBlockDim, 0, stream);
}

void driver_sparse_bicgstab_intermediate_f32(void *residual,
                                             void *operator_direction,
                                             void *intermediate,
                                             void *state,
                                             int num_items,
                                             void *stream) {
  TI_ERROR_IF(!residual || !operator_direction || !intermediate || !state ||
                  num_items <= 0,
              "CUDA Driver sparse BiCGSTAB intermediate received invalid "
              "geometry or a null pointer.");
  void *residual_arg = residual;
  void *operator_direction_arg = operator_direction;
  void *intermediate_arg = intermediate;
  void *state_arg = state;
  std::uint32_t count_arg = static_cast<std::uint32_t>(num_items);
  std::vector<void *> args{&residual_arg, &operator_direction_arg,
                           &intermediate_arg, &state_arg, &count_arg};
  const unsigned grid = (count_arg + kBlockDim - 1u) / kBlockDim;
  CUDAContext::get_instance().launch(
      kernels().sparse_bicgstab_intermediate_f32,
      "cuda_driver_sparse_bicgstab_intermediate_f32", args, {}, grid,
      kBlockDim, 0, stream);
}

void driver_sparse_bicgstab_commit_f32(void *solution_direction,
                                       void *solution_intermediate,
                                       void *intermediate,
                                       void *operator_intermediate,
                                       void *solution,
                                       void *residual,
                                       void *state,
                                       int num_items,
                                       void *stream) {
  TI_ERROR_IF(!solution_direction || !solution_intermediate ||
                  !intermediate || !operator_intermediate || !solution ||
                  !residual || !state || num_items <= 0,
              "CUDA Driver sparse BiCGSTAB commit received invalid "
              "geometry or a null pointer.");
  void *solution_direction_arg = solution_direction;
  void *solution_intermediate_arg = solution_intermediate;
  void *intermediate_arg = intermediate;
  void *operator_intermediate_arg = operator_intermediate;
  void *solution_arg = solution;
  void *residual_arg = residual;
  void *state_arg = state;
  std::uint32_t count_arg = static_cast<std::uint32_t>(num_items);
  std::vector<void *> args{
      &solution_direction_arg, &solution_intermediate_arg,
      &intermediate_arg, &operator_intermediate_arg, &solution_arg,
      &residual_arg, &state_arg, &count_arg};
  const unsigned grid = (count_arg + kBlockDim - 1u) / kBlockDim;
  CUDAContext::get_instance().launch(
      kernels().sparse_bicgstab_commit_f32,
      "cuda_driver_sparse_bicgstab_commit_f32", args, {}, grid,
      kBlockDim, 0, stream);
}

void driver_sparse_bicgstab_reconcile_f32(void *true_residual,
                                          void *residual,
                                          void *shadow_residual,
                                          void *direction,
                                          void *operator_direction,
                                          void *solution,
                                          void *state,
                                          int num_items,
                                          void *stream) {
  TI_ERROR_IF(!true_residual || !residual || !shadow_residual || !direction ||
                  !operator_direction || !solution || !state ||
                  num_items <= 0,
              "CUDA Driver sparse BiCGSTAB reconcile received invalid "
              "geometry or a null pointer.");
  void *true_residual_arg = true_residual;
  void *residual_arg = residual;
  void *shadow_residual_arg = shadow_residual;
  void *direction_arg = direction;
  void *operator_direction_arg = operator_direction;
  void *solution_arg = solution;
  void *state_arg = state;
  std::uint32_t count_arg = static_cast<std::uint32_t>(num_items);
  std::vector<void *> args{
      &true_residual_arg, &residual_arg, &shadow_residual_arg,
      &direction_arg, &operator_direction_arg, &solution_arg,
      &state_arg, &count_arg};
  const unsigned grid = (count_arg + kBlockDim - 1u) / kBlockDim;
  CUDAContext::get_instance().launch(
      kernels().sparse_bicgstab_reconcile_f32,
      "cuda_driver_sparse_bicgstab_reconcile_f32", args, {}, grid,
      kBlockDim, 0, stream);
}

void driver_sparse_gmres_multi_dot_f32(void *basis,
                                       void *work,
                                       void *partials,
                                       void *projection,
                                       void *state,
                                       int num_items,
                                       int basis_stride,
                                       int basis_count,
                                       int group_count,
                                       void *stream) {
  TI_ERROR_IF(!basis || !work || !partials || !projection || !state ||
                  num_items <= 0 || basis_stride < num_items ||
                  basis_count <= 0 || basis_count > 32 ||
                  group_count <= 0 || group_count > 65535,
              "CUDA Driver sparse GMRES multi-dot received invalid "
              "geometry or a null pointer.");
  void *basis_arg = basis;
  void *work_arg = work;
  void *partials_arg = partials;
  void *projection_arg = projection;
  void *state_arg = state;
  std::uint32_t count_arg = static_cast<std::uint32_t>(num_items);
  std::uint32_t stride_arg = static_cast<std::uint32_t>(basis_stride);
  std::uint32_t basis_count_arg =
      static_cast<std::uint32_t>(basis_count);
  std::uint32_t group_count_arg =
      static_cast<std::uint32_t>(group_count);
  std::vector<void *> partial_args{
      &basis_arg, &work_arg, &partials_arg, &state_arg, &count_arg,
      &stride_arg, &basis_count_arg, &group_count_arg};
  CUDAContext::get_instance().launch(
      kernels().sparse_gmres_multi_dot_partial_f32,
      "cuda_driver_sparse_gmres_multi_dot_partial_f32", partial_args, {},
      group_count_arg, kBlockDim, 0, stream);
  std::vector<void *> final_args{
      &partials_arg, &projection_arg, &state_arg, &group_count_arg,
      &basis_count_arg};
  CUDAContext::get_instance().launch(
      kernels().sparse_gmres_multi_dot_final_f32,
      "cuda_driver_sparse_gmres_multi_dot_final_f32", final_args, {},
      basis_count_arg, kBlockDim, 0, stream);
}

void driver_sparse_gmres_projection_f32(void *basis,
                                        void *work,
                                        void *projection,
                                        void *hessenberg,
                                        void *state,
                                        int num_items,
                                        int basis_stride,
                                        int restart,
                                        int step,
                                        int pass,
                                        void *stream) {
  TI_ERROR_IF(!basis || !work || !projection || !hessenberg || !state ||
                  num_items <= 0 || basis_stride < num_items ||
                  (restart != 8 && restart != 16 && restart != 32) ||
                  step < 0 || step >= restart || pass < 0 || pass > 1,
              "CUDA Driver sparse GMRES projection received invalid "
              "geometry, controls, or a null pointer.");
  void *basis_arg = basis;
  void *work_arg = work;
  void *projection_arg = projection;
  void *hessenberg_arg = hessenberg;
  void *state_arg = state;
  std::uint32_t count_arg = static_cast<std::uint32_t>(num_items);
  std::uint32_t stride_arg = static_cast<std::uint32_t>(basis_stride);
  std::uint32_t restart_arg = static_cast<std::uint32_t>(restart);
  std::uint32_t step_arg = static_cast<std::uint32_t>(step);
  std::uint32_t pass_arg = static_cast<std::uint32_t>(pass);
  std::vector<void *> args{
      &basis_arg, &work_arg, &projection_arg, &hessenberg_arg, &state_arg,
      &count_arg, &stride_arg, &restart_arg, &step_arg, &pass_arg};
  const unsigned grid = (count_arg + kBlockDim - 1u) / kBlockDim;
  CUDAContext::get_instance().launch(
      kernels().sparse_gmres_projection_f32,
      "cuda_driver_sparse_gmres_projection_f32", args, {}, grid,
      kBlockDim, 0, stream);
}

void driver_sparse_gmres_basis_f32(void *source,
                                   void *basis,
                                   void *current,
                                   void *state,
                                   int num_items,
                                   int basis_stride,
                                   int row,
                                   int mode,
                                   void *stream) {
  TI_ERROR_IF(!source || !basis || !current || !state || num_items <= 0 ||
                  basis_stride < num_items || row < 0 || row > 32 ||
                  mode < 0 || mode > 2,
              "CUDA Driver sparse GMRES basis update received invalid "
              "geometry, controls, or a null pointer.");
  void *source_arg = source;
  void *basis_arg = basis;
  void *current_arg = current;
  void *state_arg = state;
  std::uint32_t count_arg = static_cast<std::uint32_t>(num_items);
  std::uint32_t stride_arg = static_cast<std::uint32_t>(basis_stride);
  std::uint32_t row_arg = static_cast<std::uint32_t>(row);
  std::uint32_t mode_arg = static_cast<std::uint32_t>(mode);
  std::vector<void *> args{&source_arg, &basis_arg, &current_arg,
                           &state_arg, &count_arg, &stride_arg, &row_arg,
                           &mode_arg};
  const unsigned grid = (count_arg + kBlockDim - 1u) / kBlockDim;
  CUDAContext::get_instance().launch(
      kernels().sparse_gmres_basis_f32,
      "cuda_driver_sparse_gmres_basis_f32", args, {}, grid, kBlockDim, 0,
      stream);
}

void driver_sparse_gmres_combine_f32(void *basis,
                                     void *coefficients,
                                     void *update,
                                     void *state,
                                     int num_items,
                                     int basis_stride,
                                     void *stream) {
  TI_ERROR_IF(!basis || !coefficients || !update || !state ||
                  num_items <= 0 || basis_stride < num_items,
              "CUDA Driver sparse GMRES basis combination received invalid "
              "geometry or a null pointer.");
  void *basis_arg = basis;
  void *coefficients_arg = coefficients;
  void *update_arg = update;
  void *state_arg = state;
  std::uint32_t count_arg = static_cast<std::uint32_t>(num_items);
  std::uint32_t stride_arg = static_cast<std::uint32_t>(basis_stride);
  std::vector<void *> args{&basis_arg, &coefficients_arg, &update_arg,
                           &state_arg, &count_arg, &stride_arg};
  const unsigned grid = (count_arg + kBlockDim - 1u) / kBlockDim;
  CUDAContext::get_instance().launch(
      kernels().sparse_gmres_combine_f32,
      "cuda_driver_sparse_gmres_combine_f32", args, {}, grid, kBlockDim, 0,
      stream);
}

void driver_sparse_gmres_scalar_f32(void *initial_residual_squared,
                                    void *rhs_squared,
                                    void *dot0,
                                    void *dot1,
                                    void *hessenberg,
                                    void *cosines,
                                    void *sines,
                                    void *g,
                                    void *coefficients,
                                    void *state,
                                    float absolute_tolerance,
                                    float relative_tolerance,
                                    int restart,
                                    int max_iterations,
                                    int stage,
                                    int step,
                                    bool limit_reached,
                                    void *stream) {
  TI_ERROR_IF(!initial_residual_squared || !rhs_squared || !dot0 || !dot1 ||
                  !hessenberg || !cosines || !sines || !g ||
                  !coefficients || !state ||
                  (restart != 8 && restart != 16 && restart != 32) ||
                  max_iterations < 0 || stage < 0 || stage > 4 ||
                  step < 0 || step >= restart,
              "CUDA Driver sparse GMRES scalar stage received invalid "
              "controls or a null pointer.");
  void *initial_arg = initial_residual_squared;
  void *rhs_arg = rhs_squared;
  void *dot0_arg = dot0;
  void *dot1_arg = dot1;
  void *hessenberg_arg = hessenberg;
  void *cosines_arg = cosines;
  void *sines_arg = sines;
  void *g_arg = g;
  void *coefficients_arg = coefficients;
  void *state_arg = state;
  std::uint32_t restart_arg = static_cast<std::uint32_t>(restart);
  std::uint32_t max_iterations_arg =
      static_cast<std::uint32_t>(max_iterations);
  std::uint32_t stage_arg = static_cast<std::uint32_t>(stage);
  std::uint32_t step_arg = static_cast<std::uint32_t>(step);
  std::uint32_t limit_arg = limit_reached ? 1u : 0u;
  std::vector<void *> args{
      &initial_arg, &rhs_arg, &dot0_arg, &dot1_arg, &hessenberg_arg,
      &cosines_arg, &sines_arg, &g_arg, &coefficients_arg, &state_arg,
      &absolute_tolerance, &relative_tolerance, &restart_arg,
      &max_iterations_arg, &stage_arg, &step_arg, &limit_arg};
  CUDAContext::get_instance().launch(
      kernels().sparse_gmres_scalar_f32,
      "cuda_driver_sparse_gmres_scalar_f32", args, {}, 1, 1, 0, stream);
}

void driver_sparse_assembly_pack_validate(void *triplet_rows,
                                          void *triplet_columns,
                                          void *triplet_values,
                                          void *sorted_keys,
                                          void *sorted_values,
                                          void *active_count,
                                          void *control,
                                          int num_items,
                                          int rows,
                                          int columns,
                                          void *stream) {
  TI_ERROR_IF(num_items <= 0 || rows <= 0 || columns <= 0,
              "CUDA Driver sparse assembly pack received an invalid size.");
  TI_ERROR_IF(!triplet_rows || !triplet_columns || !triplet_values ||
                  !sorted_keys || !sorted_values || !active_count || !control,
              "CUDA Driver sparse assembly pack received a null pointer.");
  void *triplet_rows_arg = triplet_rows;
  void *triplet_columns_arg = triplet_columns;
  void *triplet_values_arg = triplet_values;
  void *sorted_keys_arg = sorted_keys;
  void *sorted_values_arg = sorted_values;
  void *active_count_arg = active_count;
  void *control_arg = control;
  std::uint32_t count_arg = static_cast<std::uint32_t>(num_items);
  std::uint32_t rows_arg = static_cast<std::uint32_t>(rows);
  std::uint32_t columns_arg = static_cast<std::uint32_t>(columns);
  std::vector<void *> args{
      &triplet_rows_arg, &triplet_columns_arg, &triplet_values_arg,
      &sorted_keys_arg,  &sorted_values_arg,   &active_count_arg,
      &control_arg,      &count_arg,           &rows_arg,
      &columns_arg};
  const unsigned grid = (count_arg + kBlockDim - 1u) / kBlockDim;
  CUDAContext::get_instance().launch(
      kernels().sparse_assembly_pack_validate,
      "cuda_driver_sparse_assembly_pack_validate", args, {}, grid, kBlockDim,
      0, stream);
}

void driver_sparse_assembly_pack_packed_validate(void *packed_triplets,
                                                 void *sorted_keys,
                                                 void *sorted_values,
                                                 void *active_count,
                                                 void *control,
                                                 int capacity,
                                                 int rows,
                                                 int columns,
                                                 void *stream) {
  TI_ERROR_IF(capacity <= 0 || rows <= 0 || columns <= 0,
              "CUDA Driver packed sparse assembly received an invalid size.");
  TI_ERROR_IF(!packed_triplets || !sorted_keys || !sorted_values ||
                  !active_count || !control,
              "CUDA Driver packed sparse assembly received a null pointer.");
  void *packed_triplets_arg = packed_triplets;
  void *sorted_keys_arg = sorted_keys;
  void *sorted_values_arg = sorted_values;
  void *active_count_arg = active_count;
  void *control_arg = control;
  std::uint32_t capacity_arg = static_cast<std::uint32_t>(capacity);
  std::uint32_t rows_arg = static_cast<std::uint32_t>(rows);
  std::uint32_t columns_arg = static_cast<std::uint32_t>(columns);
  std::vector<void *> args{
      &packed_triplets_arg, &sorted_keys_arg, &sorted_values_arg,
      &active_count_arg,    &control_arg,     &capacity_arg,
      &rows_arg,            &columns_arg};
  const unsigned grid = (capacity_arg + kBlockDim - 1u) / kBlockDim;
  CUDAContext::get_instance().launch(
      kernels().sparse_assembly_pack_packed_validate,
      "cuda_driver_sparse_assembly_pack_packed_validate", args, {}, grid,
      kBlockDim, 0, stream);
}

void driver_sparse_assembly_mark_segments(void *sorted_keys,
                                          void *segment_ids,
                                          void *active_count,
                                          void *control,
                                          int capacity,
                                          void *stream) {
  TI_ERROR_IF(capacity <= 0 || !sorted_keys || !segment_ids ||
                  !active_count || !control,
              "CUDA Driver sparse assembly segment marking received an "
              "invalid size or pointer.");
  void *sorted_keys_arg = sorted_keys;
  void *segment_ids_arg = segment_ids;
  void *active_count_arg = active_count;
  void *control_arg = control;
  std::uint32_t count_arg = static_cast<std::uint32_t>(capacity);
  std::vector<void *> args{
      &sorted_keys_arg, &segment_ids_arg, &active_count_arg, &control_arg,
      &count_arg};
  const unsigned grid = (count_arg + kBlockDim - 1u) / kBlockDim;
  CUDAContext::get_instance().launch(
      kernels().sparse_assembly_mark_segments,
      "cuda_driver_sparse_assembly_mark_segments", args, {}, grid, kBlockDim,
      0, stream);
}

void driver_sparse_assembly_scatter_segments(void *sorted_keys,
                                             void *segment_ids,
                                             void *unique_keys,
                                             void *segment_offsets,
                                             void *active_count,
                                             void *control,
                                             int capacity,
                                             void *stream) {
  TI_ERROR_IF(capacity <= 0 || !sorted_keys || !segment_ids || !unique_keys ||
                  !segment_offsets || !active_count || !control,
              "CUDA Driver sparse assembly segment scatter received an "
              "invalid size or pointer.");
  void *sorted_keys_arg = sorted_keys;
  void *segment_ids_arg = segment_ids;
  void *unique_keys_arg = unique_keys;
  void *segment_offsets_arg = segment_offsets;
  void *active_count_arg = active_count;
  void *control_arg = control;
  std::uint32_t count_arg = static_cast<std::uint32_t>(capacity);
  std::vector<void *> args{
      &sorted_keys_arg,     &segment_ids_arg, &unique_keys_arg,
      &segment_offsets_arg, &active_count_arg, &control_arg,
      &count_arg};
  const unsigned grid = (count_arg + kBlockDim - 1u) / kBlockDim;
  CUDAContext::get_instance().launch(
      kernels().sparse_assembly_scatter_segments,
      "cuda_driver_sparse_assembly_scatter_segments", args, {}, grid,
      kBlockDim, 0, stream);
}

void driver_sparse_assembly_reduce_segments(void *sorted_values,
                                            void *segment_offsets,
                                            void *unique_values,
                                            void *active_count,
                                            void *control,
                                            int capacity,
                                            void *stream) {
  TI_ERROR_IF(capacity <= 0 || !sorted_values || !segment_offsets ||
                  !unique_values || !active_count || !control,
              "CUDA Driver sparse assembly segment reduction received an "
              "invalid size or pointer.");
  void *sorted_values_arg = sorted_values;
  void *segment_offsets_arg = segment_offsets;
  void *unique_values_arg = unique_values;
  void *active_count_arg = active_count;
  void *control_arg = control;
  std::uint32_t count_arg = static_cast<std::uint32_t>(capacity);
  std::vector<void *> args{
      &sorted_values_arg, &segment_offsets_arg, &unique_values_arg,
      &active_count_arg, &control_arg, &count_arg};
  const unsigned grid = (count_arg + kBlockDim - 1u) / kBlockDim;
  CUDAContext::get_instance().launch(
      kernels().sparse_assembly_reduce_segments,
      "cuda_driver_sparse_assembly_reduce_segments", args, {}, grid,
      kBlockDim, 0, stream);
}

void driver_sparse_assembly_emit_csr(void *unique_keys,
                                     void *row_offsets,
                                     void *column_indices,
                                     void *active_count,
                                     void *control,
                                     int capacity,
                                     int rows,
                                     int columns,
                                     void *stream) {
  TI_ERROR_IF(capacity <= 0 || rows <= 0 || columns <= 0 || !unique_keys ||
                  !row_offsets || !column_indices || !active_count ||
                  !control,
              "CUDA Driver sparse assembly CSR emit received an invalid size "
              "or pointer.");
  void *unique_keys_arg = unique_keys;
  void *row_offsets_arg = row_offsets;
  void *column_indices_arg = column_indices;
  void *active_count_arg = active_count;
  void *control_arg = control;
  std::uint32_t capacity_arg = static_cast<std::uint32_t>(capacity);
  std::uint32_t rows_arg = static_cast<std::uint32_t>(rows);
  std::uint32_t columns_arg = static_cast<std::uint32_t>(columns);
  std::vector<void *> args{
      &unique_keys_arg, &row_offsets_arg, &column_indices_arg,
      &active_count_arg, &control_arg, &capacity_arg, &rows_arg,
      &columns_arg};
  const unsigned grid = (capacity_arg + kBlockDim - 1u) / kBlockDim;
  CUDAContext::get_instance().launch(
      kernels().sparse_assembly_emit_csr,
      "cuda_driver_sparse_assembly_emit_csr", args, {}, grid, kBlockDim, 0,
      stream);
}

void driver_sparse_assembly_finalize_control(void *active_count,
                                             void *control,
                                             int capacity,
                                             void *stream) {
  TI_ERROR_IF(capacity <= 0 || !active_count || !control,
              "CUDA Driver sparse assembly finalize received an invalid size "
              "or pointer.");
  void *active_count_arg = active_count;
  void *control_arg = control;
  std::uint32_t capacity_arg = static_cast<std::uint32_t>(capacity);
  std::vector<void *> args{
      &active_count_arg, &control_arg, &capacity_arg};
  CUDAContext::get_instance().launch(
      kernels().sparse_assembly_finalize_control,
      "cuda_driver_sparse_assembly_finalize_control", args, {}, 1, 1, 0,
      stream);
}

void driver_sparse_csr_spmv_f32(void *row_offsets,
                                void *column_indices,
                                void *values,
                                void *input,
                                void *output,
                                int rows,
                                void *stream) {
  TI_ERROR_IF(rows <= 0 || !row_offsets || !column_indices || !values ||
                  !input || !output,
              "CUDA Driver CSR SpMV received an invalid size or pointer.");
  void *row_offsets_arg = row_offsets;
  void *column_indices_arg = column_indices;
  void *values_arg = values;
  void *input_arg = input;
  void *output_arg = output;
  std::uint32_t rows_arg = static_cast<std::uint32_t>(rows);
  std::vector<void *> args{&row_offsets_arg, &column_indices_arg, &values_arg,
                           &input_arg, &output_arg, &rows_arg};
  const unsigned grid = (rows_arg + kBlockDim - 1u) / kBlockDim;
  CUDAContext::get_instance().launch(
      kernels().sparse_csr_spmv_f32, "cuda_driver_sparse_csr_spmv_f32", args,
      {}, grid, kBlockDim, 0, stream);
}

void driver_sparse_bsr_spmv_f32(void *row_offsets,
                                void *column_indices,
                                void *values,
                                void *input,
                                void *output,
                                int block_rows,
                                int block_size,
                                void *stream) {
  TI_ERROR_IF(block_rows <= 0 || block_size <= 0 || !row_offsets ||
                  !column_indices || !values || !input || !output,
              "CUDA Driver BSR SpMV received an invalid size or pointer.");
  TI_ERROR_IF(block_rows >
                  (std::numeric_limits<std::uint32_t>::max)() /
                      static_cast<std::uint32_t>(block_size),
              "CUDA Driver BSR SpMV row count overflow.");
  void *row_offsets_arg = row_offsets;
  void *column_indices_arg = column_indices;
  void *values_arg = values;
  void *input_arg = input;
  void *output_arg = output;
  std::uint32_t block_rows_arg = static_cast<std::uint32_t>(block_rows);
  std::uint32_t block_size_arg = static_cast<std::uint32_t>(block_size);
  const std::uint32_t rows = block_rows_arg * block_size_arg;
  std::vector<void *> args{&row_offsets_arg, &column_indices_arg, &values_arg,
                           &input_arg, &output_arg, &block_rows_arg,
                           &block_size_arg};
  const unsigned grid = (rows + kBlockDim - 1u) / kBlockDim;
  CUDAContext::get_instance().launch(
      kernels().sparse_bsr_spmv_f32, "cuda_driver_sparse_bsr_spmv_f32", args,
      {}, grid, kBlockDim, 0, stream);
}

std::size_t driver_stable_radix_sort_strided(
    void *keys,
    void *values,
    int num_items,
    CudaDriverSortKeyType key_type,
    int value_words,
    std::size_t keys_offset,
    std::size_t keys_stride,
    std::size_t values_offset,
    std::size_t values_stride,
    bool has_values,
    int nan_policy,
    void *stream,
    PrimitiveWorkspaceArena *workspace_arena) {
  TI_ERROR_IF(num_items < 0 || (has_values && value_words <= 0),
              "CUDA Driver stable sort received an invalid size.");
  TI_ERROR_IF(nan_policy != 0 && nan_policy != 1,
              "CUDA Driver stable sort received an invalid NaN policy.");
  const std::size_t key_size = sort_key_type_size(key_type);
  const std::size_t key_words = key_size / sizeof(std::uint32_t);
  const std::size_t value_size =
      static_cast<std::size_t>(std::max(value_words, 0)) *
      sizeof(std::uint32_t);
  TI_ERROR_IF((num_items > 0 && (!keys || (has_values && !values))) ||
                  keys_stride < key_size ||
                  (has_values && values_stride < value_size),
              "CUDA Driver stable sort received an invalid pointer or stride.");
  if (num_items <= 1) {
    return 0;
  }

  const std::size_t n = static_cast<std::size_t>(num_items);
  auto checked_bytes = [n](std::size_t item_size) {
    TI_ERROR_IF(item_size != 0 &&
                    n > std::numeric_limits<std::size_t>::max() / item_size,
                "CUDA Driver stable sort workspace size overflow.");
    return n * item_size;
  };
  std::size_t cursor = 0;
  auto reserve = [&cursor](std::size_t bytes) {
    constexpr std::size_t alignment = 256;
    TI_ERROR_IF(
        cursor > std::numeric_limits<std::size_t>::max() - (alignment - 1),
        "CUDA Driver stable sort workspace alignment overflow.");
    cursor = (cursor + alignment - 1) & ~(alignment - 1);
    TI_ERROR_IF(bytes > std::numeric_limits<std::size_t>::max() - cursor,
                "CUDA Driver stable sort workspace size overflow.");
    const std::size_t offset = cursor;
    cursor += bytes;
    return offset;
  };

  const std::size_t sortable_a_offset = reserve(checked_bytes(key_size));
  const std::size_t sortable_b_offset = reserve(checked_bytes(key_size));
  const std::size_t indices_a_offset =
      reserve(checked_bytes(sizeof(std::uint32_t)));
  const std::size_t indices_b_offset =
      reserve(checked_bytes(sizeof(std::uint32_t)));
  const std::size_t local_ranks_offset =
      reserve(checked_bytes(sizeof(std::int32_t)));
  const std::uint32_t radix_block_count =
      (static_cast<std::uint32_t>(num_items) + kRadixItemsPerBlock - 1u) /
      kRadixItemsPerBlock;
  std::array<std::uint32_t, 8> histogram_counts{};
  std::array<std::size_t, 8> histogram_offsets{};
  std::size_t histogram_level_count = 0;
  const std::size_t expected_histogram_level_count =
      radix_histogram_level_count(radix_block_count);
  std::uint32_t histogram_count = radix_block_count;
  for (;;) {
    TI_ASSERT(histogram_level_count < histogram_counts.size());
    TI_ERROR_IF(
        histogram_count >
            std::numeric_limits<std::size_t>::max() /
                (kRadixDigits * sizeof(std::uint32_t)),
        "CUDA Driver stable sort histogram workspace size overflow.");
    histogram_counts[histogram_level_count] = histogram_count;
    histogram_offsets[histogram_level_count] = reserve(
        static_cast<std::size_t>(histogram_count) * kRadixDigits *
        sizeof(std::uint32_t));
    ++histogram_level_count;
    // One scan block already produces the complete prefix for a digit when
    // the level fits in a tile.  Adding a one-element parent in that case
    // launches a redundant scan followed by a uniform-add that cannot touch
    // any tile other than zero.
    if (histogram_count <= kScanTileItems) {
      break;
    }
    histogram_count =
        (histogram_count + kScanTileItems - 1u) / kScanTileItems;
  }
  TI_ASSERT(histogram_level_count == expected_histogram_level_count);
  const std::size_t keys_output_offset = reserve(checked_bytes(key_size));
  const std::size_t values_output_offset =
      has_values ? reserve(checked_bytes(value_size)) : cursor;

  auto workspace = acquire_workspace(
      workspace_arena, PrimitiveWorkspaceFamily::ordering, stream);
  workspace->ensure(cursor);
  auto *base = static_cast<std::uint8_t *>(workspace->data());
  void *sortable_a = base + sortable_a_offset;
  void *sortable_b = base + sortable_b_offset;
  void *indices_a = base + indices_a_offset;
  void *indices_b = base + indices_b_offset;
  void *local_ranks = base + local_ranks_offset;
  std::array<void *, 8> histogram_levels{};
  for (std::size_t level = 0; level < histogram_level_count; ++level) {
    histogram_levels[level] = base + histogram_offsets[level];
  }
  void *keys_output = base + keys_output_offset;
  void *values_output =
      has_values ? static_cast<void *>(base + values_output_offset) : nullptr;

  auto &kernel = kernels();
  const std::size_t type_index = sort_key_type_index(key_type);
  const std::size_t width_index = key_size == sizeof(std::uint32_t) ? 0 : 1;
  std::uint32_t count_arg = static_cast<std::uint32_t>(num_items);
  const unsigned grid = (count_arg + kBlockDim - 1u) / kBlockDim;

  void *keys_arg = keys;
  std::uint64_t keys_offset_arg = keys_offset;
  std::uint64_t keys_stride_arg = keys_stride;
  void *sortable_arg = sortable_a;
  void *indices_arg = indices_a;
  std::int32_t nan_policy_arg = nan_policy;
  std::vector<void *> init_args{
      &keys_arg,    &keys_offset_arg, &keys_stride_arg, &sortable_arg,
      &indices_arg, &count_arg,       &nan_policy_arg};
  CUDAContext::get_instance().launch(kernel.radix_init[type_index],
                                     "cuda_driver_radix_init", init_args, {},
                                     grid, kBlockDim, 0, stream);

  const std::uint32_t bit_count = static_cast<std::uint32_t>(key_size * 8);
  std::uint32_t radix_block_count_arg = radix_block_count;
  for (std::uint32_t bit = 0; bit < bit_count;
       bit += kRadixBitsPerPass) {
    void *keys_in_arg = sortable_a;
    void *local_ranks_arg = local_ranks;
    void *block_histogram_arg = histogram_levels[0];
    std::vector<void *> rank_args{&keys_in_arg, &local_ranks_arg,
                                  &block_histogram_arg, &count_arg,
                                  &radix_block_count_arg, &bit};
    CUDAContext::get_instance().launch(
        kernel.radix_rank4[width_index], "cuda_driver_radix_rank4", rank_args,
        {}, radix_block_count, kBlockDim, 0, stream);

    for (std::size_t level = 0; level < histogram_level_count; ++level) {
      void *histogram_arg = histogram_levels[level];
      std::uint32_t histogram_count_arg = histogram_counts[level];
      std::uint32_t tile_count_arg =
          (histogram_count_arg + kScanTileItems - 1u) / kScanTileItems;
      void *tile_sums_arg = level + 1 < histogram_level_count
                                ? histogram_levels[level + 1]
                                : nullptr;
      std::vector<void *> scan_args{&histogram_arg, &histogram_count_arg,
                                    &tile_sums_arg, &tile_count_arg};
      const unsigned scan_grid = kRadixDigits * tile_count_arg;
      CUDAContext::get_instance().launch(
          kernel.radix_hist_scan, "cuda_driver_radix_hist_scan", scan_args,
          {}, scan_grid, kBlockDim, 0, stream);
    }
    for (std::size_t level = histogram_level_count; level > 1; --level) {
      const std::size_t target_level = level - 2;
      const std::size_t prefix_level = level - 1;
      void *histogram_arg = histogram_levels[target_level];
      std::uint32_t histogram_count_arg = histogram_counts[target_level];
      void *tile_prefix_arg = histogram_levels[prefix_level];
      std::uint32_t tile_count_arg = histogram_counts[prefix_level];
      std::vector<void *> uniform_args{
          &histogram_arg, &histogram_count_arg, &tile_prefix_arg,
          &tile_count_arg};
      const unsigned blocks_per_digit =
          (histogram_count_arg + kBlockDim - 1u) / kBlockDim;
      CUDAContext::get_instance().launch(
          kernel.radix_hist_uniform, "cuda_driver_radix_hist_uniform",
          uniform_args, {}, kRadixDigits * blocks_per_digit, kBlockDim, 0,
          stream);
    }

    void *indices_in_arg = indices_a;
    void *keys_out_arg = sortable_b;
    void *indices_out_arg = indices_b;
    std::vector<void *> scatter_args{
        &keys_in_arg,          &indices_in_arg,       &local_ranks_arg,
        &block_histogram_arg, &keys_out_arg,         &indices_out_arg,
        &count_arg,           &radix_block_count_arg, &bit};
    CUDAContext::get_instance().launch(
        kernel.radix_scatter4[width_index], "cuda_driver_radix_scatter4",
        scatter_args, {}, grid, kBlockDim, 0, stream);
    std::swap(sortable_a, sortable_b);
    std::swap(indices_a, indices_b);
  }

  auto launch_gather = [&](void *src, std::size_t src_offset,
                           std::size_t src_stride, void *dst,
                           std::uint32_t item_words) {
    void *src_arg = src;
    std::uint64_t src_offset_arg = src_offset;
    std::uint64_t src_stride_arg = src_stride;
    void *indices_arg = indices_a;
    void *dst_arg = dst;
    std::vector<void *> args{&src_arg,     &src_offset_arg, &src_stride_arg,
                             &indices_arg, &dst_arg,        &count_arg,
                             &item_words};
    CUDAContext::get_instance().launch(kernel.radix_gather_words,
                                       "cuda_driver_radix_gather", args, {},
                                       grid, kBlockDim, 0, stream);
  };
  auto launch_copy = [&](void *src, void *dst, std::size_t dst_offset,
                         std::size_t dst_stride, std::uint32_t item_words) {
    void *src_arg = src;
    void *dst_arg = dst;
    std::uint64_t dst_offset_arg = dst_offset;
    std::uint64_t dst_stride_arg = dst_stride;
    std::vector<void *> args{&src_arg,        &dst_arg,   &dst_offset_arg,
                             &dst_stride_arg, &count_arg, &item_words};
    CUDAContext::get_instance().launch(kernel.radix_copy_words,
                                       "cuda_driver_radix_copy", args, {}, grid,
                                       kBlockDim, 0, stream);
  };

  launch_gather(keys, keys_offset, keys_stride, keys_output,
                static_cast<std::uint32_t>(key_words));
  if (has_values) {
    launch_gather(values, values_offset, values_stride, values_output,
                  static_cast<std::uint32_t>(value_words));
  }
  launch_copy(keys_output, keys, keys_offset, keys_stride,
              static_cast<std::uint32_t>(key_words));
  if (has_values) {
    launch_copy(values_output, values, values_offset, values_stride,
                static_cast<std::uint32_t>(value_words));
  }
  return workspace->allocated_bytes();
}

}  // namespace taichi::lang::cuda
