#pragma once

#include <optional>
#include <set>
#include <string>
#include <vector>

#include "taichi/ir/type.h"
#include "taichi/aot/graph_data.h"

namespace taichi::lang {
class Kernel;
class GraphBuilder;
class SparseMatrix;
class CuSparseMatrix;
struct CudaCublasLtCapturePlan;

class Node {
 public:
  Node() = default;
  virtual ~Node() = default;
  Node(const Node &) = delete;
  Node &operator=(const Node &) = delete;
  Node(Node &&) = default;
  Node &operator=(Node &&) = default;

  virtual void compile(
      std::vector<aot::CompiledDispatch> &compiled_dispatches) = 0;
};

class Dispatch : public Node {
 public:
  explicit Dispatch(
      Kernel *kernel,
      const std::vector<aot::Arg> &args,
      std::optional<aot::Arg> indirect_dispatch_arg = std::nullopt,
      std::optional<aot::CudaBoundedDispatchMetadata>
          cuda_bounded_dispatch = std::nullopt,
      std::string dispatch_label = {},
      std::uint64_t logical_dispatch_id =
          std::numeric_limits<std::uint64_t>::max())
      : kernel_(kernel),
        symbolic_args_(args),
        indirect_dispatch_arg_(std::move(indirect_dispatch_arg)),
        cuda_bounded_dispatch_(std::move(cuda_bounded_dispatch)),
        dispatch_label_(std::move(dispatch_label)),
        logical_dispatch_id_(logical_dispatch_id) {
  }

  void compile(
      std::vector<aot::CompiledDispatch> &compiled_dispatches) override;

  aot::CompiledDispatch compile_dispatch() const;

  Kernel *kernel() const {
    return kernel_;
  }

  const std::vector<aot::Arg> &symbolic_args() const {
    return symbolic_args_;
  }

  bool is_indirect() const {
    return indirect_dispatch_arg_.has_value();
  }

  bool is_cuda_bounded() const {
    return cuda_bounded_dispatch_.has_value();
  }

  bool is_cpu_bounded() const {
    return cpu_bounded_dispatch_.has_value();
  }

  void set_cpu_bounded_dispatch(aot::CpuBoundedDispatchMetadata metadata) {
    cpu_bounded_dispatch_ = std::move(metadata);
  }

  const std::string &dispatch_label() const {
    return dispatch_label_;
  }

  std::uint64_t logical_dispatch_id() const {
    return logical_dispatch_id_;
  }

 private:
  mutable bool serialized_{false};
  Kernel *kernel_{nullptr};
  std::vector<aot::Arg> symbolic_args_;
  std::optional<aot::Arg> indirect_dispatch_arg_;
  std::optional<aot::CudaBoundedDispatchMetadata> cuda_bounded_dispatch_;
  std::optional<aot::CpuBoundedDispatchMetadata> cpu_bounded_dispatch_;
  std::string dispatch_label_;
  std::uint64_t logical_dispatch_id_;
};

class Sequential : public Node {
 public:
  explicit Sequential(GraphBuilder *graph) : owning_graph_(graph) {
  }

  void append(Node *node);

  void dispatch(Kernel *kernel,
                const std::vector<aot::Arg> &args,
                const std::string &dispatch_label = {});

  void dispatch_indirect(Kernel *kernel,
                         const std::vector<aot::Arg> &args,
                         const aot::Arg &dispatch_packet,
                         const std::string &dispatch_label = {});

  void dispatch_cuda_bounded(Kernel *kernel,
                             const std::vector<aot::Arg> &args,
                             const aot::Arg &extent,
                             std::uint32_t capacity,
                             std::uint32_t block_dim,
                             bool adaptive_grid,
                             bool grouped_update = false,
                             const std::string &dispatch_label = {});

  void dispatch_cpu_bounded(Kernel *kernel,
                            const std::vector<aot::Arg> &args,
                            const aot::Arg &extent,
                            std::uint32_t capacity,
                            const std::string &dispatch_label = {});

  void compile(
      std::vector<aot::CompiledDispatch> &compiled_dispatches) override;

 private:
  std::vector<Node *> sequence_;
  GraphBuilder *owning_graph_{nullptr};
};

class GraphBuilder {
 public:
  explicit GraphBuilder();
  ~GraphBuilder();

  // TODO: compile() can take in Arch argument
  std::unique_ptr<aot::CompiledGraph> compile();

  Node *new_dispatch_node(Kernel *kernel,
                          const std::vector<aot::Arg> &args,
                          const std::string &dispatch_label = {});

  Node *new_indirect_dispatch_node(Kernel *kernel,
                                   const std::vector<aot::Arg> &args,
                                   const aot::Arg &dispatch_packet,
                                   const std::string &dispatch_label = {});

  Node *new_cuda_bounded_dispatch_node(
      Kernel *kernel,
      const std::vector<aot::Arg> &args,
      const aot::Arg &extent,
      std::uint32_t capacity,
      std::uint32_t block_dim,
      bool adaptive_grid,
      bool grouped_update = false,
      const std::string &dispatch_label = {});

  Node *new_cpu_bounded_dispatch_node(
      Kernel *kernel,
      const std::vector<aot::Arg> &args,
      const aot::Arg &extent,
      std::uint32_t capacity,
      const std::string &dispatch_label = {});

  Sequential *new_sequential_node();

  void dispatch(Kernel *kernel,
                const std::vector<aot::Arg> &args,
                const std::string &dispatch_label = {});

  void dispatch_indirect(Kernel *kernel,
                         const std::vector<aot::Arg> &args,
                         const aot::Arg &dispatch_packet,
                         const std::string &dispatch_label = {});

  void dispatch_cuda_bounded(Kernel *kernel,
                             const std::vector<aot::Arg> &args,
                             const aot::Arg &extent,
                             std::uint32_t capacity,
                             std::uint32_t block_dim,
                             bool adaptive_grid,
                             bool grouped_update = false,
                             const std::string &dispatch_label = {});

  void dispatch_cpu_bounded(Kernel *kernel,
                            const std::vector<aot::Arg> &args,
                            const aot::Arg &extent,
                            std::uint32_t capacity,
                            const std::string &dispatch_label = {});

  // Private JIT-only feasibility proof. This is intentionally not an AOT
  // node and does not broaden the serialized Graph contract.
  void dispatch_cuda_capture_cusparse_spmv(SparseMatrix *matrix,
                                           Program *program,
                                           const aot::Arg &input,
                                           const aot::Arg &output);

  void dispatch_cuda_capture_cusparse_spmm(
      CuSparseMatrix *matrix,
      Program *program,
      const aot::Arg &input,
      const aot::Arg &output,
      int rhs_count,
      int algorithm,
      bool owned_plan = false,
      std::int64_t expected_workspace = -1);

  void dispatch_cuda_capture_cusparse_triangular(CuSparseMatrix *matrix,
                                                 Program *program,
                                                 const aot::Arg &input,
                                                 const aot::Arg &output,
                                                 int rhs_count,
                                                 int fill_mode,
                                                 bool unit_diagonal,
                                                 bool transpose);

  void dispatch_cuda_capture_cufft(std::uint64_t plan_handle,
                                   Program *program,
                                   const aot::Arg &input,
                                   const aot::Arg &output,
                                   int direction,
                                   std::size_t input_scalars,
                                   std::size_t output_scalars);

  // A frozen provider source preserves topology without an executable plan.
  // Forge must replace it at cold materialization before any submission.
  void dispatch_cuda_capture_description(
      Program *program,
      const std::string &kind,
      const std::vector<aot::Arg> &arguments);

  Sequential *seq() const;

  void dispatch_cuda_capture_cublaslt(
      Program *program,
      const CudaCublasLtCapturePlan &plan,
      const std::vector<aot::Arg> &arguments);

  void dispatch_cuda_capture_scan(Program *program,
                                  const aot::Arg &values,
                                  const aot::Arg &workspace,
                                  int num_items,
                                  int value_type);

  void dispatch_cuda_capture_addon(
      Program *program,
      std::uint64_t invoke_address,
      const std::string &payload,
      std::size_t stream_offset,
      const std::vector<aot::Arg> &arguments,
      const std::vector<std::size_t> &pointer_offsets,
      const std::vector<std::size_t> &scalar_counts,
      const std::vector<bool> &writable,
      std::uint64_t error_address);

  void enable_two_map_composer() {
    map_composer_max_group_size_ = 2;
  }

  bool two_map_composer_enabled() const {
    return map_composer_max_group_size_ >= 2;
  }

  void set_map_composer_max_group_size(std::uint32_t max_group_size);

  void set_map_composer_allowed_groups(
      const std::vector<std::vector<std::uint64_t>> &groups);

  // Private JIT-only physical schedule selected by a complete Forge recipe.
  // Indices address the post-composition dispatch vector; callers must keep
  // map composition disabled when installing this schedule.
  void set_cuda_parallel_dispatch_groups(
      const std::vector<std::vector<std::uint32_t>> &groups);

  std::uint32_t map_composer_max_group_size() const {
    return map_composer_max_group_size_;
  }

  std::optional<aot::CompiledDispatch> try_compose_maps(
      const std::vector<const Dispatch *> &sources,
      const std::vector<const aot::CompiledDispatch *> &compiled_sources);

 private:
  void register_arg(const aot::Arg &arg);

  std::unique_ptr<Sequential> seq_{nullptr};
  std::unordered_map<std::string, aot::Arg> all_args_;
  std::vector<std::unique_ptr<Node>> all_nodes_;
  std::vector<std::unique_ptr<Kernel>> composed_kernels_;
  std::uint32_t map_composer_max_group_size_{1};
  std::set<std::vector<std::uint64_t>> map_composer_allowed_groups_;
  std::vector<std::vector<std::uint32_t>> cuda_parallel_dispatch_groups_;
  std::uint64_t next_logical_dispatch_id_{0};
};

}  // namespace taichi::lang
