#pragma once

#include <cstddef>

#include "taichi/rhi/cuda/primitives/linear_ptx.h"

namespace taichi::lang {
class PrimitiveWorkspaceArena;
}

namespace taichi::lang::cuda {

enum class CudaHierarchicalReduceOp : int {
  sum = 0,
  min = 1,
  max = 2,
};

enum class CudaDriverSortKeyType : int {
  u32 = 0,
  i32 = 1,
  f32 = 2,
  u64 = 3,
  i64 = 4,
  f64 = 5,
};

bool driver_hierarchical_available();

// Cold Graph preparation and exact, caller-owned scratch. Recording never
// acquires the mutable Program arena; the Graph retains the bound allocation.
std::size_t driver_scan_workspace_bytes(int num_items,
                                        CudaTransformValueType value_type);
void driver_prepare_scan();
void driver_inclusive_scan_with_workspace(void *data,
                                          int num_items,
                                          CudaTransformValueType value_type,
                                          void *workspace,
                                          void *stream);

std::size_t driver_inclusive_scan_strided(
    void *data,
    int num_items,
    CudaTransformValueType value_type,
    std::size_t offset,
    std::size_t stride,
    bool reverse,
    void *stream,
    PrimitiveWorkspaceArena *workspace_arena);

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
                                  PrimitiveWorkspaceArena *workspace_arena);

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
                                     void *stream);

std::size_t driver_add_strided(void *src,
                               void *dst,
                               int num_items,
                               CudaTransformValueType value_type,
                               std::size_t src_offset,
                               std::size_t src_stride,
                               std::size_t dst_offset,
                               std::size_t dst_stride,
                               void *stream);

std::size_t driver_add_scaled_strided(void *src,
                                      void *dst,
                                      int num_items,
                                      CudaTransformValueType value_type,
                                      std::size_t src_offset,
                                      std::size_t src_stride,
                                      std::size_t dst_offset,
                                      std::size_t dst_stride,
                                      double scale,
                                      void *stream);

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
                                       void *stream);

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
                                      void *stream);

std::size_t driver_zero_strided(void *dst,
                                int num_items,
                                CudaTransformValueType value_type,
                                std::size_t dst_offset,
                                std::size_t dst_stride,
                                void *stream);

void driver_sparse_diagonal_apply_f32(void *inverse_diagonal,
                                      void *input,
                                      void *output,
                                      int num_items,
                                      void *stream);

void driver_sparse_diagonal_refresh_f32(void *values,
                                        void *diagonal_offsets,
                                        void *staging_inverse,
                                        void *status,
                                        int rows,
                                        int nnz,
                                        void *stream);

void driver_sparse_block_cholesky_refresh_f32(void *values,
                                              void *diagonal_block_offsets,
                                              void *staging_factors,
                                              void *status,
                                              int block_rows,
                                              int block_nnz,
                                              int block_size,
                                              void *stream);

void driver_sparse_block_diagonal_apply_f32(void *factor_blocks,
                                            void *input,
                                            void *output,
                                            int block_rows,
                                            int block_size,
                                            void *stream);

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
                                     void *stream);

void driver_sparse_minres_vector_state_f32(void *source,
                                           void *destination,
                                           void *state,
                                           int num_items,
                                           int coefficient_index,
                                           bool add,
                                           void *stream);

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
                                     void *stream);

void driver_sparse_bicgstab_scalar_f32(void *initial_residual_squared,
                                       void *rhs_squared,
                                       void *dot0,
                                       void *dot1,
                                       void *state,
                                       float absolute_tolerance,
                                       float relative_tolerance,
                                       int stage,
                                       bool limit_reached,
                                       void *stream);

void driver_sparse_bicgstab_direction_f32(void *residual,
                                          void *direction,
                                          void *operator_direction,
                                          void *state,
                                          int num_items,
                                          void *stream);

void driver_sparse_bicgstab_intermediate_f32(void *residual,
                                             void *operator_direction,
                                             void *intermediate,
                                             void *state,
                                             int num_items,
                                             void *stream);

void driver_sparse_bicgstab_commit_f32(void *solution_direction,
                                       void *solution_intermediate,
                                       void *intermediate,
                                       void *operator_intermediate,
                                       void *solution,
                                       void *residual,
                                       void *state,
                                       int num_items,
                                       void *stream);

void driver_sparse_bicgstab_reconcile_f32(void *true_residual,
                                          void *residual,
                                          void *shadow_residual,
                                          void *direction,
                                          void *operator_direction,
                                          void *solution,
                                          void *state,
                                          int num_items,
                                          void *stream);

void driver_sparse_gmres_multi_dot_f32(void *basis,
                                       void *work,
                                       void *partials,
                                       void *projection,
                                       void *state,
                                       int num_items,
                                       int basis_stride,
                                       int basis_count,
                                       int group_count,
                                       void *stream);

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
                                        void *stream);

void driver_sparse_gmres_basis_f32(void *source,
                                   void *basis,
                                   void *current,
                                   void *state,
                                   int num_items,
                                   int basis_stride,
                                   int row,
                                   int mode,
                                   void *stream);

void driver_sparse_gmres_combine_f32(void *basis,
                                     void *coefficients,
                                     void *update,
                                     void *state,
                                     int num_items,
                                     int basis_stride,
                                     void *stream);

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
                                    void *stream);

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
                                   bool reuse_prefix = false);

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
    PrimitiveWorkspaceArena *workspace_arena);

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
                                          void *stream);

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
                                          void *stream);

void driver_sparse_assembly_pack_packed_validate(void *packed_triplets,
                                                 void *sorted_keys,
                                                 void *sorted_values,
                                                 void *active_count,
                                                 void *control,
                                                 int capacity,
                                                 int rows,
                                                 int columns,
                                                 void *stream);

void driver_sparse_assembly_mark_segments(void *sorted_keys,
                                          void *segment_ids,
                                          void *active_count,
                                          void *control,
                                          int capacity,
                                          void *stream);

void driver_sparse_assembly_scatter_segments(void *sorted_keys,
                                             void *segment_ids,
                                             void *unique_keys,
                                             void *segment_offsets,
                                             void *active_count,
                                             void *control,
                                             int capacity,
                                             void *stream);

void driver_sparse_assembly_reduce_segments(void *sorted_values,
                                            void *segment_offsets,
                                            void *unique_values,
                                            void *active_count,
                                            void *control,
                                            int capacity,
                                            void *stream);

void driver_sparse_assembly_emit_csr(void *unique_keys,
                                     void *row_offsets,
                                     void *column_indices,
                                     void *active_count,
                                     void *control,
                                     int capacity,
                                     int rows,
                                     int columns,
                                     void *stream);

void driver_sparse_assembly_finalize_control(void *active_count,
                                             void *control,
                                             int capacity,
                                             void *stream);

void driver_sparse_csr_spmv_f32(void *row_offsets,
                                void *column_indices,
                                void *values,
                                void *input,
                                void *output,
                                int rows,
                                void *stream);

void driver_sparse_bsr_spmv_f32(void *row_offsets,
                                void *column_indices,
                                void *values,
                                void *input,
                                void *output,
                                int block_rows,
                                int block_size,
                                void *stream);

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
    PrimitiveWorkspaceArena *workspace_arena);

}  // namespace taichi::lang::cuda
