#pragma once
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <limits>
#include <mutex>
#include <optional>
#include <vector>
#include <string>
#include <unordered_map>
#include <utility>
#include "taichi/analysis/graph_kernel_metadata.h"
#include "taichi/ir/type.h"
#include "taichi/ir/type_factory.h"
#include "taichi/program/callable.h"
#include "taichi/aot/module_data.h"
#include "taichi/program/compile_config.h"
#include "taichi/program/runtime_resource_registry.h"
#include "taichi/struct/snode_tree.h"
#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

template <typename T, typename G>
T taichi_union_cast_with_different_sizes(G g);

namespace taichi::lang {
class AotModuleBuilder;
class Ndarray;
class Texture;
class Matrix;
class Kernel;
class CompiledKernelData;
class KernelExecutionHandle;
class Program;
namespace storage {
class RuntimeStorageArgument;
}

namespace aot {

// Preserve serialized values. Acceleration structures are JIT-only bindings;
// AOT export rejects them rather than serializing a process-local resource.
enum class ArgKind {
  kScalar,
  kMatrix,
  kNdarray,
  kTexture,
  kRWTexture,
  kUnknown,
  kAccelerationStructure
};

/**
 * Symbolic argument used in building `Dispatch` nodes in the `Graph`.
 */
struct Arg {
  ArgKind tag;
  std::string name;
  // Ndarray: element_dtype = dtype + element_shape
  // Texture: element_shape carries [width, height, depth] info
  //          dtype_id carries channel_format info
  PrimitiveTypeID dtype_id;
  size_t field_dim;
  std::vector<int> element_shape;

  // For texture
  size_t num_channels;  // TODO: maybe rename field_dim and merge?

  // For serialization & deserialization
  explicit Arg()
      : tag(ArgKind::kUnknown),
        name(""),
        dtype_id(PrimitiveTypeID::unknown),
        field_dim(0),
        element_shape({}),
        num_channels(0) {
  }

  explicit Arg(ArgKind tag,
               const std::string &name,

               PrimitiveTypeID dtype_id,
               size_t field_dim,
               const std::vector<int> &element_shape)
      : tag(tag),
        name(name),
        dtype_id(dtype_id),
        field_dim(field_dim),
        element_shape(element_shape),
        num_channels(0) {
  }

  // Python/C++ interface that's user facing.
  explicit Arg(ArgKind tag,
               const std::string &name,
               const DataType &dtype,
               size_t dim = 0,
               const std::vector<int> &element_shape = {})
      : tag(tag), name(name), element_shape(element_shape) {
    field_dim = 0;
    num_channels = 0;
    if (tag == ArgKind::kTexture || tag == ArgKind::kRWTexture) {
      num_channels = dim;
    } else {
      field_dim = dim;
    }
    DataType scalar_dtype = dtype;
    if (dtype->is<TensorType>()) {
      TI_ERROR_IF(tag == ArgKind::kScalar,
                  "Scalar Graph argument {} cannot use a TensorType dtype",
                  name);
      TI_ERROR_IF(tag == ArgKind::kTexture || tag == ArgKind::kRWTexture,
                  "Texture Graph arguments cannot use a TensorType dtype");
      const auto inferred_shape = dtype->as<TensorType>()->get_shape();
      TI_ERROR_IF(inferred_shape.empty() || inferred_shape.size() > 2,
                  "Graph argument {} supports rank-1 vector and rank-2 matrix "
                  "tensor types only, but got shape {}",
                  name, inferred_shape);
      TI_ERROR_IF(!this->element_shape.empty() &&
                      this->element_shape != inferred_shape,
                  "Graph argument {} specifies conflicting tensor shapes",
                  name);
      this->element_shape = inferred_shape;
      scalar_dtype = dtype->as<TensorType>()->get_element_type();
    }
    TI_ERROR_IF(!scalar_dtype->is<PrimitiveType>(),
                "Graph argument {} requires a primitive or tensor dtype",
                name);
    dtype_id = scalar_dtype->as<PrimitiveType>()->type;
  }

  DataType dtype() const {
    return PrimitiveType::get(dtype_id);
  }

  DataType element_dtype() const {
    TI_ERROR_IF(tag == ArgKind::kTexture || tag == ArgKind::kRWTexture,
                "Texture Graph arguments do not expose an element dtype");
    DataType scalar_dtype = dtype();
    if (element_shape.empty()) {
      return scalar_dtype;
    }
    return TypeFactory::get_instance().create_tensor_type(element_shape,
                                                           scalar_dtype);
  }

  bool operator==(const Arg &other) const {
    return tag == other.tag && name == other.name &&
           field_dim == other.field_dim && dtype_id == other.dtype_id &&
           element_shape == other.element_shape &&
           num_channels == other.num_channels;
  }

  bool operator!=(const Arg &other) const {
    return !(*this == other);
  }

  TI_IO_DEF(name, dtype_id, field_dim, tag, element_shape, num_channels);
};

/**
 * Runtime value used in graph execution.
 */
struct TI_DLL_EXPORT IValue {
 public:
  uint64 val;
  ArgKind tag;
  const storage::RuntimeStorageArgument *runtime_storage{nullptr};
  Program *resource_owner{nullptr};

  static IValue acceleration_structure(Program &owner, std::uint64_t handle) {
    IValue value(handle, ArgKind::kAccelerationStructure);
    value.resource_owner = &owner;
    return value;
  }

  static IValue create(const Ndarray &ndarray) {
    return IValue(reinterpret_cast<intptr_t>(&ndarray), ArgKind::kNdarray);
  }

  static IValue create(const Ndarray &ndarray,
                       const storage::RuntimeStorageArgument &runtime_storage) {
    IValue value(reinterpret_cast<intptr_t>(&ndarray), ArgKind::kNdarray);
    value.runtime_storage = &runtime_storage;
    return value;
  }
  static IValue create(
      const storage::RuntimeStorageArgument &runtime_storage) {
    IValue value(0, ArgKind::kNdarray);
    value.runtime_storage = &runtime_storage;
    return value;
  }

  static IValue create(const Texture &tex) {
    return IValue(reinterpret_cast<intptr_t>(&tex), ArgKind::kTexture);
  }

  static IValue create(const Matrix &matrix) {
    return IValue(reinterpret_cast<intptr_t>(&matrix), ArgKind::kMatrix);
  }

  template <typename T,
            typename = std::enable_if_t<!std::is_same<T, Ndarray>::value, void>>
  static IValue create(T v) {
    return IValue(taichi_union_cast_with_different_sizes<uint64>(v),
                  ArgKind::kScalar);
  }

 private:
  IValue(uint64 val, ArgKind tag) : val(val), tag(tag) {
  }
};

class TI_DLL_EXPORT Kernel : public CallableBase {
 public:
  // Rule of 5 to make MSVC happy
  Kernel() = default;
  virtual ~Kernel() = default;
  Kernel(const Kernel &) = delete;
  Kernel &operator=(const Kernel &) = delete;
  Kernel(Kernel &&) = default;
  Kernel &operator=(Kernel &&) = default;

  /**
   * @brief Launches the kernel to the device
   *
   * This does not manage the device to host synchronization.
   *
   * @param ctx Host context
   */
  virtual void launch(LaunchContextBuilder &ctx) = 0;
};

struct GraphSourceDispatchMetadata {
  std::string kernel_name;
  std::string dispatch_label;
  std::vector<Arg> symbolic_args;
  GraphKernelMetadata graph_metadata;
  // JIT-only creation-order identity inside one GraphBuilder. Region lineage
  // is attached by the Forge Graph IR so independently compiled CGraph
  // segments cannot accidentally share a tuning identity.
  std::uint64_t logical_dispatch_id{
      std::numeric_limits<std::uint64_t>::max()};
  // JIT-only compiler identity for the unfused logical source kernel. This
  // scopes offline executable recipes to code, not just to a same-shaped
  // Graph position/name/effect description. It is deliberately excluded from
  // the AOT CGraph v1 schema.
  std::string logical_kernel_identity;
};

struct CudaBoundedDispatchMetadata {
  Arg extent_arg;
  std::uint32_t capacity{0};
  std::uint32_t block_dim{0};
  bool adaptive_grid{false};
  bool grouped_update{false};
};

struct CpuBoundedDispatchMetadata {
  Arg extent_arg;
  std::uint32_t capacity{0};
};

// Capture-local provider storage. Finalization runs after stream capture and
// before instantiation, never on replay. The graph/frame retires this storage
// only after its launches complete. No AOT or public deployment schema.
class CudaGraphCaptureResources {
 public:
  virtual ~CudaGraphCaptureResources() = default;
  virtual void finalize() = 0;
  virtual std::uint64_t requested_device_bytes() const = 0;
  virtual void release(bool backend_safe) noexcept = 0;
};

// JIT-only contract for one provider command that can be prepared outside a
// CUDA stream capture and then recorded on the capture stream. Concrete
// provider recipes remain internal and are deliberately excluded from the
// public/AOT graph schema.
class CudaGraphCaptureCommand {
 public:
  virtual ~CudaGraphCaptureCommand() = default;

  virtual const char *kind() const = 0;
  virtual Program *program() const = 0;
  virtual bool supports(
      const std::unordered_map<std::string, IValue> &args,
      Program &program) const = 0;
  // Prepare descriptors, private workspace and analysis only. No user binding
  // may be modified and no mathematical operation may be executed: feedback
  // bindings otherwise advance the Graph state before its first submission.
  virtual void prepare(const std::unordered_map<std::string, IValue> &args,
                       Program &program) = 0;
  virtual void record(const std::unordered_map<std::string, IValue> &args,
                      Program &program,
                      void *stream) = 0;
  virtual bool requires_exact_bindings() const {
    return true;
  }
  // Opt-in only for a fixed provider plan whose captured parameters can live
  // in separate immutable binding frames. Queried at construction, not replay.
  virtual bool supports_binding_frames() const {
    return false;
  }
  virtual std::shared_ptr<void> retain_binding_frame_plan(Program &program) {
    return {};
  }
  // Cold capture handoff. A provider may snapshot host parameters which its
  // library otherwise mutates when recording another binding. Each captured
  // graph/frame retains these bytes until its queued launches have retired.
  virtual std::shared_ptr<CudaGraphCaptureResources> take_capture_resources() {
    return {};
  }
};

struct CompiledDispatch {
  std::string kernel_name;
  // JIT-only invocation metadata. AOT payloads remain source-compatible and
  // simply load this as empty.
  std::string dispatch_label;
  std::vector<Arg> symbolic_args;
  Kernel *compiled_kernel{nullptr};
  taichi::lang::Kernel *ti_kernel{nullptr};
  // JIT-only metadata. AOT serialization intentionally remains unchanged:
  // loaded module fields have a module-owned lifecycle rather than Program
  // SNodeTree identities.
  std::vector<SNodeTreeDependency> snode_tree_dependencies;
  GraphKernelMetadata graph_metadata;
  // JIT-only lineage lets diagnostics retain the two logical source maps
  // after one physical dispatch replaces them. AOT serialization is unchanged.
  std::vector<GraphSourceDispatchMetadata> source_dispatches;
  std::uint32_t compiled_task_count{
      std::numeric_limits<std::uint32_t>::max()};
  // JIT-only Vulkan dispatch packet. It is intentionally excluded from
  // TI_IO_DEF until the AOT module ABI can represent indirect dispatch.
  std::optional<Arg> indirect_dispatch_arg;
  // JIT-only CUDA device-count launch metadata. This remains separate from
  // Vulkan indirect dispatch because its capture, replay, and capability
  // contracts are backend-specific.
  std::optional<CudaBoundedDispatchMetadata> cuda_bounded_dispatch;
  // JIT-only CPU scheduler metadata; omitted from the public AOT schema.
  std::optional<CpuBoundedDispatchMetadata> cpu_bounded_dispatch;
  // JIT-only provider capture recipe. Provider-specific state is retained by
  // this shared command object and never enters the serialized Graph schema.
  std::shared_ptr<CudaGraphCaptureCommand> cuda_capture_command;

  TI_IO_DEF(kernel_name, symbolic_args);
};

struct CompiledGraphJITCachedKernel {
  std::string kernel_key;
  // Keeps the compiled payload alive independently of the frontend Kernel
  // shell and compilation-cache entry. SNode retirement marks the handle
  // inactive; Graph generation validation still rejects any new stale submit.
  std::shared_ptr<KernelExecutionHandle> execution_handle;
  int llvm_launch_id{-1};
  // UINT32_MAX is the unknown sentinel. Keeping this beside llvm_launch_id
  // consumes the struct's existing alignment padding on 64-bit builds.
  std::uint32_t task_count{std::numeric_limits<std::uint32_t>::max()};
};

struct CompiledGraphRuntimeArgPlan {
  ArgKind tag{ArgKind::kUnknown};
  std::string name;
  std::vector<int> arg_id;
  PrimitiveTypeID dtype_id{PrimitiveTypeID::unknown};
  size_t field_dim{0};
  int type_size{0};
  int arg_buffer_offset{-1};
  std::vector<int> element_shape;
  std::vector<int> ndarray_data_ptr_key;
  std::vector<int> ndarray_grad_ptr_key;
  std::vector<int> ndarray_shape_offsets;
};

struct CompiledGraphDispatchRuntimePlan {
  bool cpu_fast_path{false};
  std::vector<CompiledGraphRuntimeArgPlan> args;
  std::optional<std::string> bounded_extent_name;
  std::vector<int> bounded_extent_arg_id;
  std::uint32_t bounded_capacity{0};
};

struct CompiledGraphCudaState;

struct CompiledGraphCudaStateDeleter {
  void operator()(CompiledGraphCudaState *state) const noexcept;
};

struct CompiledGraphVulkanState;

struct CompiledGraphVulkanStateDeleter {
  void operator()(CompiledGraphVulkanState *state) const noexcept;
};

enum class CompiledGraphBackend : uint8_t {
  none,
  cuda,
  vulkan,
};

enum class CompiledGraphExecutionPath : uint8_t {
  none,
  ordinary_fallback,
  cuda_capture,
  cuda_exact_replay,
  cuda_patched_replay,
  cuda_masked_capture,
  cuda_masked_replay,
  cuda_masked_patched_replay,
  cuda_device_update_nested_capture,
  cuda_device_update_nested_replay,
  cuda_device_update_nested_patched_replay,
  vulkan_record,
  vulkan_replay,
  vulkan_patched_replay,
};

enum class CompiledGraphFallbackReason : uint8_t {
  none,
  debug_mode,
  insufficient_dispatches,
  unsupported_arguments,
  resource_unavailable,
  structural_unsupported,
  transient_driver_failure,
  fatal_driver_failure,
  retry_backoff,
  runtime_mode,
  replay_slot_saturated,
};

// Internal diagnostics snapshot. Counters are maintained with integer writes
// on the serialized graph-cache path; labels are constructed only when a
// caller explicitly requests the debug snapshot.
struct CompiledGraphStats {
  CompiledGraphBackend backend{CompiledGraphBackend::none};
  CompiledGraphExecutionPath last_path{CompiledGraphExecutionPath::none};
  CompiledGraphFallbackReason last_fallback_reason{
      CompiledGraphFallbackReason::none};
  uint64_t attempts{0};
  uint64_t ordinary_fallbacks{0};
  uint64_t capture_attempts{0};
  uint64_t captures{0};
  uint64_t exact_replays{0};
  uint64_t patched_replays{0};
  uint64_t masked_captures{0};
  uint64_t masked_replays{0};
  uint64_t masked_patched_replays{0};
  uint64_t recaptures{0};
  uint64_t records{0};
  uint64_t replays{0};
  uint64_t structural_fallbacks{0};
  uint64_t transient_failures{0};
  uint64_t retry_backoff_fallbacks{0};
  uint64_t replay_slot_saturation_fallbacks{0};
  uint64_t capture_exceptions{0};
  uint64_t zero_arg_captures{0};
  uint64_t asynchronous_control_updates{0};
  uint64_t deferred_replay_waits{0};
  uint64_t peak_deferred_replay_batches{0};
  uint64_t known_persistent_argument_bytes{0};
  uint64_t known_bounded_control_bytes{0};
  uint32_t known_bounded_update_groups{0};
  uint32_t known_bounded_updater_dispatches{0};
  uint32_t known_bounded_grouped_payloads{0};
  uint32_t known_bounded_producer_fused_groups{0};
  uint32_t known_bounded_payloads{0};
  uint64_t last_bounded_useful_lanes{0};
  uint64_t last_bounded_physical_blocks{0};
  uint64_t last_bounded_physical_threads{0};
  uint64_t last_bounded_baseline_blocks{0};
  uint32_t last_bounded_zero_payloads{0};
  bool bounded_physical_observation_available{false};
  uint64_t bounded_update_replays{0};
  uint64_t bounded_update_state_changes{0};
  uint64_t bounded_update_cache_hits{0};
  uint64_t bounded_node_api_calls{0};
  uint32_t known_bounded_max_group_size{0};
  uint64_t last_driver_error{0};
  uint32_t retry_backoff_remaining{0};
  uint64_t effect_reads{0};
  uint64_t effect_writes{0};
  uint64_t dependency_barriers{0};
  uint64_t exit_barriers{0};
  uint64_t barrier_deferrals{0};
  uint64_t rar_elisions{0};
  uint32_t consecutive_transient_failures{0};
  bool zero_arg_eligible{false};
};

// Ephemeral report metadata. Keeping this wrapper separate prevents opt-in
// diagnostics from enlarging every persistent CUDA/Vulkan stats object.
struct CompiledGraphDebugSnapshot {
  CompiledGraphStats stats;
  uint64_t known_compiled_tasks{0};
  uint32_t known_compiled_dispatches{0};
  uint32_t runtime_binding_plan_slots{0};
  uint32_t backend_replay_signature_slots{0};
  uint32_t backend_replay_signature_slot_capacity{0};
  // Public snapshots never enable diagnostics. Detailed counters are complete
  // only when private debug instrumentation was enabled before execution;
  // path/fallback enums and static metadata remain available immediately.
  bool diagnostics_previously_enabled{false};
  bool diagnostics_counters_complete{true};
  bool replay_attribution_enabled{false};
  uint64_t replay_calls{0};
  uint64_t replay_total_ns{0};
  uint64_t replay_snode_guard_ns{0};
  uint64_t replay_resource_guard_ns{0};
  uint64_t replay_cuda_submission_lock_ns{0};
  uint64_t replay_cache_wait_ns{0};
  uint64_t replay_binding_plan_ns{0};
  uint64_t replay_resource_retain_ns{0};
  uint64_t replay_snode_validation_ns{0};
  uint64_t replay_backend_ns{0};
  uint64_t replay_signature_ns{0};
  uint64_t replay_binding_plan_hits{0};
  uint64_t replay_binding_plan_misses{0};
  uint64_t replay_signature_hits{0};
  uint64_t replay_signature_misses{0};
  uint64_t replay_snode_guard_acquisitions{0};
  uint64_t replay_snode_guard_elisions{0};
};

struct CompiledGraphRuntimeResourceIdentity {
  std::string name;
  RuntimeResourceHandle handle;
  const void *object{nullptr};
  std::uint64_t opaque_handle{0};
};

// Reusable, generation-qualified resource binding for stable Graph replay.
// The plan stores only high-level identities and object references; submission
// leases are reacquired for every replay so cached state never extends a
// resource lifetime on its own.
struct CompiledGraphRuntimeBindingPlan {
  Program *program{nullptr};
  bool initialized{false};
  uint64_t revision{0};
  std::vector<CompiledGraphRuntimeResourceIdentity> identities;
  std::vector<Ndarray *> ndarrays;
  std::vector<const storage::RuntimeStorageArgument *> runtime_storage;
  std::vector<Texture *> textures;

  void clear() {
    program = nullptr;
    initialized = false;
    revision = 0;
    identities.clear();
    ndarrays.clear();
    runtime_storage.clear();
    textures.clear();
  }
};

struct CompiledGraphReplayAttribution {
  uint64_t calls{0};
  uint64_t total_ns{0};
  uint64_t snode_guard_ns{0};
  uint64_t resource_guard_ns{0};
  uint64_t cuda_submission_lock_ns{0};
  uint64_t cache_wait_ns{0};
  uint64_t binding_plan_ns{0};
  uint64_t resource_retain_ns{0};
  uint64_t snode_validation_ns{0};
  uint64_t backend_ns{0};
  uint64_t signature_ns{0};
  uint64_t binding_plan_hits{0};
  uint64_t binding_plan_misses{0};
  uint64_t signature_hits{0};
  uint64_t signature_misses{0};
  uint64_t snode_guard_acquisitions{0};
  uint64_t snode_guard_elisions{0};
};

struct CompiledGraphStructuredResult {
  bool submitted{false};
  std::uint32_t strategy{0};
  std::uint32_t logical_iterations{0};
  std::int32_t predicate{0};
  std::int32_t counter{0};
  std::int32_t status{0};
  std::int32_t initial_status{0};
  std::uint32_t encoded_iterations{0};
  std::uint32_t indirect_dispatches{0};
  std::uint32_t controller_dispatches{0};
  std::uint32_t controller_invocations{0};
  std::uint32_t zero_dispatches{0};
  std::uint32_t control_bytes{0};
  std::uint32_t observation_bytes{0};
};

struct CompiledGraphNestedStructuredResult {
  bool submitted{false};
  std::uint32_t inner_region_count{0};
  std::uint32_t outer_logical_iterations{0};
  std::uint32_t outer_encoded_iterations{0};
  std::int32_t outer_initial_predicate{0};
  std::int32_t outer_final_predicate{0};
  std::int32_t outer_initial_counter{0};
  std::int32_t outer_final_counter{0};
  std::int32_t outer_initial_status{0};
  std::int32_t outer_final_status{0};
  std::vector<std::uint32_t> inner_logical_iterations;
  std::vector<std::uint32_t> inner_encoded_iterations;
  std::vector<std::int32_t> inner_initial_counters;
  std::vector<std::int32_t> inner_final_counters;
  std::vector<std::int32_t> inner_final_predicates;
  std::vector<std::int32_t> inner_initial_statuses;
  std::vector<std::int32_t> inner_final_statuses;
  std::uint32_t indirect_dispatches{0};
  std::uint32_t controller_dispatches{0};
  std::uint32_t controller_invocations{0};
  std::uint32_t zero_dispatches{0};
  std::uint32_t control_bytes{0};
  std::uint32_t observation_bytes{0};
};

// One ordered leaf while in a depth-2 outer while. Dispatch boundaries use
// the flattened CompiledGraph coordinate space. `dispatch_end` is exclusive
// and includes the terminal condition copy repeated after every body step.
struct CompiledGraphNestedInnerControl {
  Ndarray *predicate{nullptr};
  Ndarray *counter{nullptr};
  Ndarray *status{nullptr};
  std::size_t condition_dispatch_begin{0};
  std::size_t body_dispatch_begin{0};
  std::size_t dispatch_end{0};
  int max_iterations{0};
  int chunk_size{0};
};

// A transient capture error must not permanently disable an otherwise valid
// graph. Retry periodically with bounded exponential backoff; structural
// incompatibility remains disabled because repeating it cannot recover.
class CompiledGraphCaptureRetryState {
 public:
  static constexpr uint32_t kMaxBackoffInvocations = 32;

  bool should_attempt() noexcept {
    if (structurally_disabled_) {
      return false;
    }
    if (retry_backoff_remaining_ > 0) {
      --retry_backoff_remaining_;
      return false;
    }
    return true;
  }

  void record_success() noexcept {
    consecutive_transient_failures_ = 0;
    retry_backoff_remaining_ = 0;
  }

  void record_structural_failure() noexcept {
    structurally_disabled_ = true;
  }

  void record_transient_failure() noexcept {
    consecutive_transient_failures_ =
        std::min<uint32_t>(consecutive_transient_failures_ + 1, 32);
    const uint32_t shift =
        std::min<uint32_t>(consecutive_transient_failures_ - 1, 5);
    retry_backoff_remaining_ = uint32_t{1} << shift;
  }

  bool structurally_disabled() const noexcept {
    return structurally_disabled_;
  }

  uint32_t consecutive_transient_failures() const noexcept {
    return consecutive_transient_failures_;
  }

  uint32_t retry_backoff_remaining() const noexcept {
    return retry_backoff_remaining_;
  }

 private:
  bool structurally_disabled_{false};
  uint32_t consecutive_transient_failures_{0};
  uint32_t retry_backoff_remaining_{0};
};

class CompiledGraphReplayIdentity {
 public:
  CompiledGraphReplayIdentity()
      : value_(next_value_.fetch_add(1, std::memory_order_relaxed)) {
    // Zero is reserved as an invalid/unregistered token. Reaching it requires
    // exhausting the complete 64-bit process lifetime, at which point
    // continuing would reintroduce identity reuse.
    TI_ASSERT(value_ != 0);
  }
  CompiledGraphReplayIdentity(const CompiledGraphReplayIdentity &) = delete;
  CompiledGraphReplayIdentity &operator=(
      const CompiledGraphReplayIdentity &) = delete;
  CompiledGraphReplayIdentity(CompiledGraphReplayIdentity &&) = delete;
  CompiledGraphReplayIdentity &operator=(
      CompiledGraphReplayIdentity &&) = delete;

  uint64_t value() const noexcept {
    return value_;
  }

 private:
  inline static std::atomic<uint64_t> next_value_{1};
  uint64_t value_{0};
};

struct CompiledGraphJITCache {
  CompiledGraphJITCache() = default;
  ~CompiledGraphJITCache();
  CompiledGraphJITCache(const CompiledGraphJITCache &) = delete;
  CompiledGraphJITCache &operator=(const CompiledGraphJITCache &) = delete;

  // Release all state tied to the current Program/Device while that runtime is
  // still alive. This is intentionally separate from the destructor: Python
  // GC may destroy a graph after ti.reset() has already finalized its Program.
  void clear_runtime_state();
  // A stale SNode-dependent Graph cannot submit again, but it must keep its
  // compiled executable leases until the Graph itself is released. Backend
  // replay and generation bindings are still discarded immediately.
  void retire_snode_tree_runtime_state();
  CompiledGraphDebugSnapshot debug_graph_stats();

  std::vector<CompiledGraphJITCachedKernel> kernels;
  std::vector<std::shared_ptr<KernelExecutionHandle>> retired_execution_handles;
  std::vector<CompiledGraphDispatchRuntimePlan> runtime_arg_plans;
  Program *validated_snode_tree_program{nullptr};
  std::uint64_t validated_snode_tree_epoch{0};
  std::unique_ptr<CompiledGraphCudaState, CompiledGraphCudaStateDeleter>
      cuda_graph_state;
  // CUDA executable instances are driver-heavy and may pin allocation
  // generations. Keep one MRU plus one lazy alternate for the common A/B
  // ping-pong case; larger binding sets fall back to bounded patching.
  std::vector<
      std::unique_ptr<CompiledGraphCudaState, CompiledGraphCudaStateDeleter>>
      cuda_graph_state_alternates;
  std::unique_ptr<CompiledGraphVulkanState, CompiledGraphVulkanStateDeleter>
      vulkan_graph_state;
  // Single-dispatch Vulkan graphs intentionally stay on the ordinary path.
  // Keep their cheap classification in the cache so diagnostics can explain
  // the decision without constructing replay slots or compiling twice.
  CompiledGraphStats vulkan_inline_stats;
  // A small generation-qualified MRU covers recurring ping-pong/triple-buffer
  // bindings without turning Graph into an unbounded resource owner. Entries
  // contain non-owning identities; every submission reacquires its leases.
  std::vector<CompiledGraphRuntimeBindingPlan> runtime_binding_plans;
  uint64_t next_runtime_binding_plan_revision{1};
  std::atomic<bool> stable_replay_optimization_enabled{true};
  CompiledGraphReplayAttribution replay_attribution;
  // Detailed counters are opt-in. Failure recovery keeps its own bounded
  // backoff state, so ordinary graph replay does not pay for diagnostics until
  // an internal caller requests _graph_stats.
  bool graph_diagnostics_enabled{false};
  // Once diagnostics are enabled after a GPU execution, lifetime counters
  // can never be reconstructed. Keep that incompleteness sticky across later
  // snapshots until runtime state is cleared.
  bool graph_diagnostics_counters_complete{true};
  // GfxRuntime may retain replay state after this cache is destroyed. A
  // monotonic token, rather than the reusable cache-object address, prevents a
  // later cache at the same host address from inheriting that state.
  CompiledGraphReplayIdentity graph_replay_identity;
  uint64_t graph_replay_token() const noexcept {
    return graph_replay_identity.value();
  }
  // One replay mutates the cached kernels, argument plans and optional graph
  // capture state as a transaction. Do not expose partially updated state to
  // another caller sharing this cache.
  std::mutex run_mutex;
};

struct TI_DLL_EXPORT CompiledGraph {
  std::vector<CompiledDispatch> dispatches;
  std::unordered_map<std::string, aot::Arg> args;
  std::vector<SNodeTreeDependency> snode_tree_dependencies;
  // JIT-only dispatch ranges captured as sibling CUDA Graph branches. Each
  // group is ordered and contiguous; all groups together cover one contiguous
  // interval followed by at least one serial join/suffix dispatch. The public
  // AOT schema intentionally remains linear.
  std::vector<std::vector<std::uint32_t>> cuda_parallel_dispatch_groups;
  // JIT-only ownership for synthetic kernels created by Graph composition.
  // Shared ownership preserves CompiledGraph's existing copy contract.
  std::vector<std::shared_ptr<taichi::lang::Kernel>> owned_jit_kernels;

  CompiledGraph() = default;
  explicit CompiledGraph(std::vector<CompiledDispatch> compiled_dispatches);
  CompiledGraph(std::vector<CompiledDispatch> compiled_dispatches,
                std::unordered_map<std::string, aot::Arg> graph_args);
  CompiledGraph(const CompiledGraph &) = default;
  CompiledGraph &operator=(const CompiledGraph &) = default;

  bool has_cuda_parallel_dispatch_groups() const {
    return !cuda_parallel_dispatch_groups.empty();
  }
  CompiledGraph(CompiledGraph &&) = default;
  CompiledGraph &operator=(CompiledGraph &&) = default;

  bool has_indirect_dispatches() const;
  bool has_dispatch_labels() const;
  bool has_cuda_capture_commands() const;
  bool cuda_capture_commands_require_exact_bindings() const;

  void run(const std::unordered_map<std::string, IValue> &args) const;
  void jit_run(const CompileConfig &compile_config,
               const std::unordered_map<std::string, IValue> &args) const;
  void jit_run_cached(const CompileConfig &compile_config,
                      const std::unordered_map<std::string, IValue> &args,
                      CompiledGraphJITCache &cache,
                      bool cuda_concurrent_batch_lane = false) const;
  bool jit_run_bounded_cuda_cached(
      const CompileConfig &compile_config,
      const std::unordered_map<std::string, IValue> &args,
      CompiledGraphJITCache &cache,
      Ndarray *predicate,
      int max_iterations,
      bool continue_while_nonzero) const;
  bool jit_run_bounded_cuda_masked_cached(
      const CompileConfig &compile_config,
      const std::unordered_map<std::string, IValue> &args,
      CompiledGraphJITCache &cache,
      Ndarray *predicate,
      int max_iterations,
      bool continue_while_nonzero) const;
  bool jit_submit_bounded_cuda_nested_cached(
      const CompileConfig &compile_config,
      const std::unordered_map<std::string, IValue> &args,
      CompiledGraphJITCache &cache,
      Ndarray *outer_predicate,
      Ndarray *outer_counter,
      Ndarray *outer_status,
      Ndarray *inner_predicate,
      Ndarray *inner_counter,
      Ndarray *inner_status,
      std::size_t outer_condition_dispatch_count,
      std::size_t inner_condition_dispatch_begin,
      std::size_t inner_body_dispatch_begin,
      std::size_t outer_suffix_dispatch_begin,
      int outer_max_iterations,
      int inner_max_iterations,
      bool allow_device_update) const;
  bool jit_submit_bounded_cuda_nested_sequence_cached(
      const CompileConfig &compile_config,
      const std::unordered_map<std::string, IValue> &args,
      CompiledGraphJITCache &cache,
      Ndarray *outer_predicate,
      Ndarray *outer_counter,
      Ndarray *outer_status,
      const std::vector<CompiledGraphNestedInnerControl> &inner_controls,
      std::size_t outer_condition_dispatch_count,
      int outer_max_iterations,
      bool allow_device_update) const;
  CompiledGraphStructuredResult jit_run_bounded_vulkan_cached(
      const CompileConfig &compile_config,
      const std::unordered_map<std::string, IValue> &args,
      CompiledGraphJITCache &cache,
      Ndarray *predicate,
      Ndarray *counter,
      Ndarray *status,
      std::size_t initial_dispatch_count,
      int max_iterations,
      bool execute_initial_dispatches,
      std::uint32_t strategy,
      bool wait_for_result = true) const;
  bool jit_submit_bounded_vulkan_compound_cached(
      const CompileConfig &compile_config,
      const std::unordered_map<std::string, IValue> &args,
      CompiledGraphJITCache &cache,
      Ndarray *predicate,
      Ndarray *counter,
      Ndarray *status,
      std::size_t initial_dispatch_count,
      const std::vector<int> &chunk_iterations,
      const std::vector<std::uint32_t> &strategies) const;
  CompiledGraphNestedStructuredResult jit_run_bounded_vulkan_nested_cached(
      const CompileConfig &compile_config,
      const std::unordered_map<std::string, IValue> &args,
      CompiledGraphJITCache &cache,
      Ndarray *outer_predicate,
      Ndarray *outer_counter,
      Ndarray *outer_status,
      Ndarray *inner_predicate,
      Ndarray *inner_counter,
      Ndarray *inner_status,
      std::size_t outer_condition_dispatch_count,
      std::size_t inner_condition_dispatch_begin,
      std::size_t inner_body_dispatch_begin,
      std::size_t outer_suffix_dispatch_begin,
      int outer_max_iterations,
      int inner_max_iterations,
      int inner_chunk_size,
      bool wait_for_result = true) const;
  CompiledGraphNestedStructuredResult
  jit_run_bounded_vulkan_nested_sequence_cached(
      const CompileConfig &compile_config,
      const std::unordered_map<std::string, IValue> &args,
      CompiledGraphJITCache &cache,
      Ndarray *outer_predicate,
      Ndarray *outer_counter,
      Ndarray *outer_status,
      const std::vector<CompiledGraphNestedInnerControl> &inner_controls,
      std::size_t outer_condition_dispatch_count,
      int outer_max_iterations,
      bool wait_for_result = true) const;
  bool jit_run_conditional_cuda_cached(
      const CompileConfig &compile_config,
      const std::unordered_map<std::string, IValue> &args,
      CompiledGraphJITCache &cache,
      Ndarray *selector,
      const std::vector<int> &branch_dispatch_counts,
      int conditional_type,
      int default_branch) const;
  bool jit_run_conditional_cuda_masked_cached(
      const CompileConfig &compile_config,
      const std::unordered_map<std::string, IValue> &args,
      CompiledGraphJITCache &cache,
      Ndarray *selector,
      const std::vector<int> &branch_dispatch_counts,
      int conditional_type,
      int default_branch) const;

  TI_IO_DEF(dispatches);

  // Internal helper shared by graph replay backends.
  static void init_runtime_context(
      const std::vector<Arg> &paramter_list,
      const std::unordered_map<std::string, IValue> &args,
      LaunchContextBuilder &ctx);
};

}  // namespace aot
}  // namespace taichi::lang
