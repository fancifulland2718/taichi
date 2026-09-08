#pragma once

#include <string>
#include <vector>
#include <assert.h>
#include <cstdint>
#include <memory>
#include <functional>
#include <limits>

// https://gcc.gnu.org/wiki/Visibility
#if defined _WIN32 || defined _WIN64 || defined __CYGWIN__
#ifdef __GNUC__
#define RHI_DLL_EXPORT __attribute__((dllexport))
#else
#define RHI_DLL_EXPORT __declspec(dllexport)
#endif  //  __GNUC__
#else
#define RHI_DLL_EXPORT __attribute__((visibility("default")))
#endif  // defined _WIN32 || defined _WIN64 || defined __CYGWIN__

// Unreachable
#if __cplusplus > 202002L  // C++23
#include <utility>
#define RHI_UNREACHABLE std::unreachable();
#else            // C++20 and below
#ifdef __GNUC__  // GCC, Clang, ICC
#define RHI_UNREACHABLE __builtin_unreachable();
#else  // MSVC
#define RHI_UNREACHABLE __assume(false);
#endif
#endif

// Not implemented
#define RHI_NOT_IMPLEMENTED         \
  assert(false && "Not supported"); \
  RHI_UNREACHABLE

#include "taichi/rhi/device_capability.h"
#include "taichi/rhi/arch.h"
#include "taichi/rhi/backend_error.h"

namespace taichi::lang {

enum class RhiResult {
  success = 0,
  error = -1,
  invalid_usage = -2,
  not_supported = -3,
  out_of_memory = -4,
};

const std::string rhi_result_to_string(RhiResult result);

constexpr size_t kBufferSizeEntireSize = std::numeric_limits<size_t>::max();

#define MAKE_ENUM_FLAGS(name)                  \
  inline name operator|(name a, name b) {      \
    return static_cast<name>(int(a) | int(b)); \
  }                                            \
  inline name operator&(name a, name b) {      \
    return static_cast<name>(int(a) & int(b)); \
  }                                            \
  inline bool operator&&(name a, name b) {     \
    return (int(a) & int(b)) != 0;             \
  }

enum class BlendOp : uint32_t { add, subtract, reverse_subtract, min, max };

enum class BlendFactor : uint32_t {
  zero,
  one,
  src_color,
  one_minus_src_color,
  dst_color,
  one_minus_dst_color,
  src_alpha,
  one_minus_src_alpha,
  dst_alpha,
  one_minus_dst_alpha
};

class Device;
struct DeviceAllocation;
struct DevicePtr;

// TODO: Figure out how to support images. Temporary solutions is to have all
// opque types such as images work as an allocation
using DeviceAllocationId = uint64_t;

struct RHI_DLL_EXPORT DeviceAllocation {
  Device *device{nullptr};
  DeviceAllocationId alloc_id{0};
  // TODO: Shall we include size here?

  DevicePtr get_ptr(uint64_t offset = 0) const;

  bool operator==(const DeviceAllocation &other) const {
    return other.device == device && other.alloc_id == alloc_id;
  }

  bool operator!=(const DeviceAllocation &other) const {
    return !(*this == other);
  }
};

struct RHI_DLL_EXPORT DeviceAllocationGuard : public DeviceAllocation {
  explicit DeviceAllocationGuard(DeviceAllocation alloc)
      : DeviceAllocation(alloc) {
  }
  DeviceAllocationGuard(const DeviceAllocationGuard &) = delete;
  ~DeviceAllocationGuard();
};

using DeviceAllocationUnique = std::unique_ptr<DeviceAllocationGuard>;

struct RHI_DLL_EXPORT DeviceImageGuard : public DeviceAllocation {
  explicit DeviceImageGuard(DeviceAllocation alloc) : DeviceAllocation(alloc) {
  }
  DeviceImageGuard(const DeviceAllocationGuard &) = delete;
  ~DeviceImageGuard();
};

using DeviceImageUnique = std::unique_ptr<DeviceImageGuard>;

struct RHI_DLL_EXPORT DevicePtr : public DeviceAllocation {
  uint64_t offset{0};

  bool operator==(const DevicePtr &other) const {
    return other.device == device && other.alloc_id == alloc_id &&
           other.offset == offset;
  }

  bool operator!=(const DevicePtr &other) const {
    return !(*this == other);
  }
};

constexpr DeviceAllocation kDeviceNullAllocation{};
constexpr DevicePtr kDeviceNullPtr{};

enum class ImageFilter : uint8_t {
  nearest,
  linear,
};

enum class ImageAddressMode : uint8_t {
  repeat,
  mirrored_repeat,
  clamp_to_edge,
};

// Immutable sampled-image binding state. Taichi textures currently expose one
// mip level and normalized coordinates, so mip/compare/anisotropy controls are
// intentionally not part of this first portable contract.
struct ImageSamplerConfig {
  ImageFilter min_filter{ImageFilter::linear};
  ImageFilter mag_filter{ImageFilter::linear};
  ImageAddressMode address_mode_u{ImageAddressMode::repeat};
  ImageAddressMode address_mode_v{ImageAddressMode::repeat};
  ImageAddressMode address_mode_w{ImageAddressMode::repeat};

  bool operator==(const ImageSamplerConfig &other) const {
    return min_filter == other.min_filter && mag_filter == other.mag_filter &&
           address_mode_u == other.address_mode_u &&
           address_mode_v == other.address_mode_v &&
           address_mode_w == other.address_mode_w;
  }

  bool operator!=(const ImageSamplerConfig &other) const {
    return !(*this == other);
  }
};

// A set of shader resources (that is bound at once)
class RHI_DLL_EXPORT ShaderResourceSet {
 public:
  virtual ~ShaderResourceSet() = default;

  /**
   * Bind a RW subregion of a buffer resource (StorgeBuffer / SSBO)
   * @params[in] binding The binding index of the resource
   * @params[in] ptr The Device Pointer that is going to be bound
   * @params[in] size The size of the bound region of the buffer
   */
  virtual ShaderResourceSet &rw_buffer(uint32_t binding,
                                       DevicePtr ptr,
                                       size_t size) = 0;

  /**
   * Bind an entire RW buffer resource (StorgeBuffer / SSBO)
   * @params[in] binding The binding index of the resource
   * @params[in] alloc The Device Allocation that is going to be bound
   */
  virtual ShaderResourceSet &rw_buffer(uint32_t binding,
                                       DeviceAllocation alloc) = 0;

  /**
   * Bind a read-only subregion of a buffer resource (Constants / UBO)
   * @params[in] binding The binding index of the resource
   * @params[in] ptr The Device Pointer that is going to be bound
   * @params[in] size The size of the bound region of the buffer
   */
  virtual ShaderResourceSet &buffer(uint32_t binding,
                                    DevicePtr ptr,
                                    size_t size) = 0;

  /**
   * Bind an entire read-only buffer resource (Constants / UBO)
   * @params[in] binding The binding index of the resource
   * @params[in] alloc The Device Allocation that is going to be bound
   */
  virtual ShaderResourceSet &buffer(uint32_t binding,
                                    DeviceAllocation alloc) = 0;

  /**
   * Bind a read-only image resource (SRV / Texture)
   * @params[in] binding The binding index of the resource
   * @params[in] alloc The Device Allocation that is going to be bound
   * @params[in] sampler_config The texture sampling configuration
   */
  virtual ShaderResourceSet &image(uint32_t binding,
                                   DeviceAllocation alloc,
                                   ImageSamplerConfig sampler_config) {
    RHI_NOT_IMPLEMENTED;
  }

  /**
   * Bind a RW image resource (UAV / Storage Image)
   * @params binding The binding index of the resource
   * @params alloc The Device Allocation that is going to be bound
   */
  virtual ShaderResourceSet &rw_image(uint32_t binding,
                                      DeviceAllocation alloc,
                                      int lod) {
    RHI_NOT_IMPLEMENTED
  }

  /**
   * C-2.5 (2026-05): Bind an array of storage buffers under a single
   * binding (descriptorCount=N). The SPIR-V variable side declares an
   * OpTypeArray of SSBO struct and accesses chunk[k] via OpAccessChain on
   * the array index. Used by the chunked pointer-SNode allocator.
   * @params binding The binding index (single binding holds N buffers)
   * @params allocs The Device Allocations that map to chunk[0..N-1]
   */
  virtual ShaderResourceSet &rw_buffer_array(
      uint32_t binding,
      const std::vector<DeviceAllocation> &allocs) {
    RHI_NOT_IMPLEMENTED
  }

  // Optional backend hook for a graph-owned stable resource-set handle.
  // patch_existing requires the owning replay completion to be ready.
  virtual RhiResult prepare_for_replay(bool patch_existing) {
    (void)patch_existing;
    return RhiResult::not_supported;
  }
};

// A set of states / resources for rasterization
class RHI_DLL_EXPORT RasterResources {
 public:
  virtual ~RasterResources() = default;

  /**
   * Set a vertex buffer for the rasterization
   * @params ptr The Device Pointer to the vertices data
   * @params binding The binding index of the vertex buffer
   */
  virtual RasterResources &vertex_buffer(DevicePtr ptr, uint32_t binding = 0) {
    RHI_NOT_IMPLEMENTED
  }

  /**
   * Set an index buffer for the rasterization
   * @params ptr The Device Pointer to the vertices data
   * @params index_width The index data width (in bits).
   *                     index_width = 32 -> uint32 index
   *                     index_width = 16 -> uint16 index
   */
  virtual RasterResources &index_buffer(DevicePtr ptr, size_t index_width) {
    RHI_NOT_IMPLEMENTED
  }
};

enum class PipelineSourceType {
  spirv_binary,
  spirv_src,
  glsl_src,
  hlsl_src,
  dxil_binary,
  llvm_ir_src,
  llvm_ir_binary,
  metal_src,
  metal_ir
};

enum class PipelineStageType {
  compute,
  fragment,
  vertex,
  tesselation_control,
  tesselation_eval,
  geometry,
  task,
  mesh,
  raytracing
};

// FIXME: Drop the plural form?
enum class TopologyType : int { Triangles = 0, Lines = 1, Points = 2 };

enum class PolygonMode : int {
  Fill = 0,
  Line = 1,
  Point = 2,
};

enum class BufferFormat : uint32_t {
#define PER_BUFFER_FORMAT(x) x,
#include "taichi/inc/rhi_constants.inc.h"
#undef PER_BUFFER_FORMAT
};

class RHI_DLL_EXPORT Pipeline{public : virtual ~Pipeline(){}};

using UPipeline = std::unique_ptr<Pipeline>;

// Buffer synchronization is expressed in backend-neutral execution and access
// domains. Backends that do not expose fine-grained barriers may conservatively
// fall back to buffer_barrier(); Vulkan uses this contract to distinguish
// shader data dependencies from command-processor reads such as indirect
// dispatch arguments.
enum class BufferBarrierStage : uint32_t {
  None = 0,
  Transfer = 1,
  Compute = 2,
  IndirectCommand = 4,
  ConditionalCommand = 8,
  Host = 16,
};

MAKE_ENUM_FLAGS(BufferBarrierStage)

enum class BufferBarrierAccess : uint32_t {
  None = 0,
  TransferRead = 1,
  TransferWrite = 2,
  ShaderRead = 4,
  ShaderWrite = 8,
  IndirectCommandRead = 16,
  ConditionalCommandRead = 32,
  HostRead = 64,
  HostWrite = 128,
};

MAKE_ENUM_FLAGS(BufferBarrierAccess)

struct BufferTransition {
  BufferBarrierStage source_stage{BufferBarrierStage::None};
  BufferBarrierAccess source_access{BufferBarrierAccess::None};
  BufferBarrierStage destination_stage{BufferBarrierStage::None};
  BufferBarrierAccess destination_access{BufferBarrierAccess::None};
};

enum class ImageDimension {
#define PER_IMAGE_DIMENSION(x) x,
#include "taichi/inc/rhi_constants.inc.h"
#undef PER_IMAGE_DIMENSION
};

enum class ImageLayout {
#define PER_IMAGE_LAYOUT(x) x,
#include "taichi/inc/rhi_constants.inc.h"
#undef PER_IMAGE_LAYOUT
};

struct BufferImageCopyParams {
  uint32_t buffer_row_length{0};
  uint32_t buffer_image_height{0};
  uint32_t image_mip_level{0};
  struct {
    uint32_t x{0};
    uint32_t y{0};
    uint32_t z{0};
  } image_offset;
  struct {
    uint32_t x{1};
    uint32_t y{1};
    uint32_t z{1};
  } image_extent;
  uint32_t image_base_layer{0};
  uint32_t image_layer_count{1};
  uint32_t image_aspect_flag{1};
};

struct ImageCopyParams {
  uint32_t width{1};
  uint32_t height{1};
  uint32_t depth{1};
  struct {
    int32_t x{0};
    int32_t y{0};
    int32_t z{0};
  } source_offset;
  struct {
    int32_t x{0};
    int32_t y{0};
    int32_t z{0};
  } destination_offset;
  uint32_t source_mip_level{0};
  uint32_t destination_mip_level{0};
  uint32_t source_base_layer{0};
  uint32_t destination_base_layer{0};
  uint32_t layer_count{1};
  uint32_t image_aspect_flag{1};
};

struct ImageBlitParams {
  struct Region {
    struct {
      int32_t x{0};
      int32_t y{0};
      int32_t z{0};
    } offset;
    struct {
      uint32_t x{1};
      uint32_t y{1};
      uint32_t z{1};
    } extent;
    uint32_t mip_level{0};
    uint32_t base_layer{0};
  } source, destination;
  uint32_t layer_count{1};
  uint32_t image_aspect_flag{1};
  bool linear_filter{false};
};

class RHI_DLL_EXPORT CommandList {
 public:
  // Optional cold finalization of a secondary compute command list. The
  // returned appender records into a primary on the same device/stream family
  // outside a render pass. Each parent retains the commands and resource owner
  // until completion; the owner must not reference this command list.
  virtual std::function<void(CommandList *)> finalize_secondary(
      std::shared_ptr<void> resource_owner) {
    return {};
  }
  virtual ~CommandList() {
  }

  /**
   * Bind a pipeline to the command list.
   * Doing so resets all bound resources.
   * @params[in] pipeline The pipeline to be bound
   */
  virtual void bind_pipeline(Pipeline *p) noexcept = 0;

  /**
   * Bind a ShaderResourceSet to a set index.
   * - If the set index is already bound, the previous binding will be
   *   overwritten.
   * - A set index can only be bound with a single ShaderResourceSet.
   * - If the input set is empty, this command is a no-op.
   * @params[in] res The ShaderResourceSet to be bound.
   * @params[in] set_index The index the resources will be bound to.
   * @return The binding result code
   *         `success` If the binding succeded
   *         `invalid_usage` If `res` is incompatible with current pipeline
   *         `not_supported` If some bindings are not supported by the backend
   *         `out_of_memory` If binding failed due to OOM conditions
   *         `error` If binding failed due to other reasons
   */
  virtual RhiResult bind_shader_resources(ShaderResourceSet *res,
                                          int set_index = 0) noexcept = 0;

  /**
   * Bind RasterResources to the command list.
   * - If the input resource is empty, this command is a no-op.
   * @params res The RasterResources to be bound.
   * @return The binding result code
   *         `success` If the binding succeded
   *         `invalid_usage` If `res` is incompatible with current pipeline
   *         `not_supported` If some bindings are not supported by the backend
   *         `error` If binding failed due to other reasons
   */
  virtual RhiResult bind_raster_resources(RasterResources *res) noexcept = 0;

  /**
   * Insert a memory barrier into the command list.
   * The barrier affects a continous region of memory.
   * Changes to memory before the barrier will be visible to accesses after the
   * barrier (API command ordering). i.e. Command later to this barrier will see
   * the changes made by commands before this barrier.
   * This barrier is limited in scope to the Stream that the command list is
   * submitted to. Other Streams or Devices may not observe this barrier.
   * @params[in] ptr The pointer to the start of the region
   * @params[in] size The size of the memory region.
   *                  Size is clamped to the underlying buffer size.
   */
  virtual void buffer_barrier(DevicePtr ptr, size_t size) noexcept = 0;

  /**
   * Insert a buffer barrier with explicit producer/consumer domains.
   * Backends without a fine-grained implementation conservatively use the
   * legacy full buffer barrier.
   */
  virtual void buffer_transition(DevicePtr ptr,
                                 size_t size,
                                 const BufferTransition &transition) noexcept {
    (void)transition;
    buffer_barrier(ptr, size);
  }

  /**
   * Insert a memory barrier into the command list.
   * The barrier affects an entire buffer.
   * Behaviour is the same as `buffer_barrier(DevicePtr, size_t)`
   * @params[in] alloc The memory allocation of this barrier
   */
  virtual void buffer_barrier(DeviceAllocation alloc) noexcept = 0;

  /**
   * Insert a memory barrier into the command list.
   * The barrier affects all global memory.
   * Behaviour is the same as `buffer_barrier(DevicePtr, size_t)`
   * @params[in] alloc The memory allocation of this barrier
   */
  virtual void memory_barrier() noexcept = 0;

  /**
   * Insert a buffer copy operation into the command list.
   * @params[in] src The source Device Pointer
   * @params[in] dst The destination Device Pointer
   * @params[in] size The size of the region to be copied.
   *                  The size will be clamped to the minimum between
   *                  `dst.size - dst.offset` and `src.size - src.offset`
   */
  virtual void buffer_copy(DevicePtr dst,
                           DevicePtr src,
                           size_t size) noexcept = 0;

  /**
   * Insert a memory region fill operation into the command list
   * The memory region will be filled with the given (bit precise) value.
   * - (Encouraged behavior) If the `data` is 0, the underlying API might
   *   provide a faster code path.
   * - (Encouraged behavior) If the `size` is -1 (max of size_t) the underlying
   *   API might provide a faster code path.
   * @params[in] ptr The start of the memory region.
   * - ptr.offset will be aligned down to a multiple of 4 bytes.
   * @params[in] size The size of the region.
   * - The size will be clamped to the underlying buffer's size.
   */
  virtual void buffer_fill(DevicePtr ptr,
                           size_t size,
                           uint32_t data) noexcept = 0;

  /**
   * Enqueues a compute operation with {X, Y, Z} amount of workgroups.
   * The block size / workgroup size is pre-determined within the pipeline.
   * - This is only valid if the pipeline has a predetermined block size
   * - This API has a device-dependent variable max values for X, Y, Z
   * - The currently bound pipeline will be dispatched
   * - The enqueued operation starts in CommandList API ordering.
   * - The enqueued operation may end out-of-order, but it respects barriers
   * @params[in] x The number of workgroups in X dimension
   * @params[in] y The number of workgroups in Y dimension
   * @params[in] z The number of workgroups in Y dimension
   * @return The status of this operation
   * - `success` if the operation is successful
   * - `invalid_operation` if the current pipeline has variable block size
   * - `not_supported` if the requested X, Y, or Z is not supported
   */
  virtual RhiResult dispatch(uint32_t x,
                             uint32_t y = 1,
                             uint32_t z = 1) noexcept = 0;

  /**
   * Enqueues a compute operation whose {X, Y, Z} workgroup counts are read
   * from three consecutive uint32 values in `indirect`.
   * - `indirect.offset` must be aligned to 4 bytes.
   * - The backing allocation must have at least 12 bytes available from the
   *   offset and be created with `AllocUsage::Indirect`.
   * - Backends without indirect-dispatch support return `not_supported`.
   */
  virtual RhiResult dispatch_indirect(DevicePtr indirect) noexcept {
    return RhiResult::not_supported;
  }

  /**
   * Begins a conditional command region controlled by one uint32 predicate.
   * Compute dispatches inside the region execute when the predicate is
   * non-zero, or zero when `inverted` is true. Backends without conditional
   * command support return `not_supported`.
   */
  virtual RhiResult begin_conditional(DevicePtr predicate,
                                      bool inverted = false) noexcept {
    return RhiResult::not_supported;
  }

  virtual RhiResult end_conditional() noexcept {
    return RhiResult::not_supported;
  }

  struct ComputeSize {
    uint32_t x{0};
    uint32_t y{0};
    uint32_t z{0};
  };

  /**
   * Enqueues a compute operation with `grid_size` amount of threads.
   * The workgroup size is dynamic and specified through `block_size`
   * - This is only valid if the pipeline has a predetermined block size
   * - This API has a device-dependent variable max values for `grid_size`
   * - This API has a device-dependent supported values for `block_size`
   * - The currently bound pipeline will be dispatched
   * - The enqueued operation starts in CommandList API ordering.
   * - The enqueued operation may end out-of-order, but it respects barriers
   * @params[in] grid_size The number of threads dispatch
   * @params[in] block_size The shape of each block / workgroup / threadsgroup
   * @return The status of this operation
   * - `success` if the operation is successful
   * - `invalid_operation` if the current pipeline has variable block size
   * - `not_supported` if the requested sizes are not supported
   * - `error` if the operation failed due to other reasons
   */
  virtual RhiResult dispatch(ComputeSize grid_size,
                             ComputeSize block_size) noexcept {
    return RhiResult::not_supported;
  }

  // Profiler support
  virtual void begin_profiler_scope(const std::string &kernel_name) {
  }

  virtual void end_profiler_scope() {
  }

  // These are not implemented in compute only device
  virtual void begin_renderpass(int x0,
                                int y0,
                                int x1,
                                int y1,
                                uint32_t num_color_attachments,
                                DeviceAllocation *color_attachments,
                                bool *color_clear,
                                std::vector<float> *clear_colors,
                                DeviceAllocation *depth_attachment,
                                bool depth_clear) {
    RHI_NOT_IMPLEMENTED
  }
  virtual void end_renderpass() {
    RHI_NOT_IMPLEMENTED
  }
  // Coordinates use the render target's top-left origin. The raster viewport
  // and scissor are updated together so draws cannot escape a reserved UI
  // region.
  virtual void set_raster_viewport_and_scissor(int x0,
                                               int y0,
                                               int x1,
                                               int y1) {
    RHI_NOT_IMPLEMENTED
  }
  virtual void draw(uint32_t num_verticies, uint32_t start_vertex = 0) {
    RHI_NOT_IMPLEMENTED
  }
  virtual void draw_instance(uint32_t num_verticies,
                             uint32_t num_instances,
                             uint32_t start_vertex = 0,
                             uint32_t start_instance = 0) {
    RHI_NOT_IMPLEMENTED
  }
  virtual void set_line_width(float width) {
    RHI_NOT_IMPLEMENTED
  }
  virtual void draw_indexed(uint32_t num_indicies,
                            int32_t vertex_offset = 0,
                            uint32_t start_index = 0) {
    RHI_NOT_IMPLEMENTED
  }
  virtual void draw_indexed_instance(uint32_t num_indicies,
                                     uint32_t num_instances,
                                     int32_t vertex_offset = 0,
                                     uint32_t start_index = 0,
                                     uint32_t start_instance = 0) {
    RHI_NOT_IMPLEMENTED
  }
  virtual RhiResult draw_indirect(DevicePtr indirect,
                                  uint32_t draw_count,
                                  uint32_t stride) noexcept {
    return RhiResult::not_supported;
  }
  virtual RhiResult draw_indexed_indirect(DevicePtr indirect,
                                          uint32_t draw_count,
                                          uint32_t stride) noexcept {
    return RhiResult::not_supported;
  }
  virtual RhiResult draw_indirect_count(DevicePtr indirect,
                                        DevicePtr count,
                                        uint32_t max_draw_count,
                                        uint32_t stride) noexcept {
    return RhiResult::not_supported;
  }
  virtual RhiResult draw_indexed_indirect_count(
      DevicePtr indirect,
      DevicePtr count,
      uint32_t max_draw_count,
      uint32_t stride) noexcept {
    return RhiResult::not_supported;
  }
  virtual RhiResult draw_mesh_tasks(uint32_t group_count_x,
                                    uint32_t group_count_y,
                                    uint32_t group_count_z,
                                    bool task_shader) noexcept {
    return RhiResult::not_supported;
  }
  virtual void image_transition(DeviceAllocation img,
                                ImageLayout old_layout,
                                ImageLayout new_layout) {
    RHI_NOT_IMPLEMENTED
  }
  virtual void buffer_to_image(DeviceAllocation dst_img,
                               DevicePtr src_buf,
                               ImageLayout img_layout,
                               const BufferImageCopyParams &params) {
    RHI_NOT_IMPLEMENTED
  }
  virtual void image_to_buffer(DevicePtr dst_buf,
                               DeviceAllocation src_img,
                               ImageLayout img_layout,
                               const BufferImageCopyParams &params) {
    RHI_NOT_IMPLEMENTED
  }
  virtual void copy_image(DeviceAllocation dst_img,
                          DeviceAllocation src_img,
                          ImageLayout dst_img_layout,
                          ImageLayout src_img_layout,
                          const ImageCopyParams &params) {
    RHI_NOT_IMPLEMENTED
  }
  virtual void blit_image(DeviceAllocation dst_img,
                          DeviceAllocation src_img,
                          ImageLayout dst_img_layout,
                          ImageLayout src_img_layout,
                          const ImageBlitParams &params) {
    RHI_NOT_IMPLEMENTED
  }
};

struct PipelineSourceDesc {
  PipelineSourceType type;
  const void *data{nullptr};
  size_t size{0};
  PipelineStageType stage{PipelineStageType::compute};
};

// FIXME: this probably isn't backend-neutral enough
enum class AllocUsage : int {
  None = 0,
  Storage = 1,
  Uniform = 2,
  Vertex = 4,
  Index = 8,
  Upload = 16,
  Indirect = 32,
  Conditional = 64,
  // Vulkan ray acceleration structures consume device addresses and require
  // distinct usage bits for their build inputs and opaque storage. Backends
  // without a matching hardware primitive may safely ignore these flags.
  AccelerationStructureBuildInput = 128,
  AccelerationStructureStorage = 256,
  DeviceAddress = 512,
};

MAKE_ENUM_FLAGS(AllocUsage)

class RHI_DLL_EXPORT StreamSemaphoreObject {
 public:
  virtual ~StreamSemaphoreObject() {
  }

  // Optional completion token support. Backends that can expose a submission
  // fence override these; others return false so callers can fall back to their
  // legacy stream/device sync path. This keeps Stream's vtable unchanged while
  // allowing higher layers to avoid over-conservative wait_idle() when possible.
  virtual bool is_ready() const {
    return false;
  }

  virtual bool wait() const {
    return false;
  }
};

using StreamSemaphore = std::shared_ptr<StreamSemaphoreObject>;

// Opt-in, stream-ordered GPU timing owned by one submission ticket. Backends
// keep their native event/query resources alive in this object until the
// ticket releases it. The snapshot is intentionally unavailable until the
// enclosing completion is ready; callers must never substitute host wall time.
struct StreamGpuTimingSnapshot {
  bool available{false};
  std::uint64_t duration_ns{0};
  bool exact{false};
  bool measurement_path_changed{false};
  std::uint64_t stream_id{0};
  std::uint64_t driver_owned_bytes{0};
  bool driver_owned_bytes_known{false};
  std::string status{"unavailable"};
};

class RHI_DLL_EXPORT StreamGpuTimingObject {
 public:
  virtual ~StreamGpuTimingObject() = default;
  virtual StreamGpuTimingSnapshot snapshot() const = 0;
};

using StreamGpuTiming = std::shared_ptr<StreamGpuTimingObject>;

class RHI_DLL_EXPORT Stream {
 public:
  // Unsupported backends return null without altering their primary path.
  virtual std::unique_ptr<CommandList> new_secondary_command_list() {
    return nullptr;
  }
  virtual ~Stream() {
  }

  /**
   * Allocates a new CommandList object from the stream.
   * @params[out] out_cmdlist The allocated command list.
   * @return The status of this operation.
   * - `success` If allocation succeeded.
   * - `out_of_memory` If allocation failed due to lack of device or host
   * memory.
   */
  virtual RhiResult new_command_list(CommandList **out_cmdlist) noexcept = 0;

  inline std::pair<std::unique_ptr<CommandList>, RhiResult>
  new_command_list_unique() {
    CommandList *cmdlist{nullptr};
    RhiResult res = this->new_command_list(&cmdlist);
    return std::make_pair(std::unique_ptr<CommandList>(cmdlist), res);
  }

  virtual StreamSemaphore submit(
      CommandList *cmdlist,
      const std::vector<StreamSemaphore> &wait_semaphores = {}) = 0;
  virtual StreamSemaphore submit_synced(
      CommandList *cmdlist,
      const std::vector<StreamSemaphore> &wait_semaphores = {}) = 0;

  // Optional host transaction batching. Backends that support it may defer
  // submit() calls between these boundaries and publish the ordered command
  // lists with fewer native queue calls. Returned per-submit completion
  // tokens must remain valid and complete no earlier than their work.
  virtual void begin_submission_batch() {
  }

  virtual StreamSemaphore end_submission_batch() {
    return nullptr;
  }

  // Optional diagnostic timing scope. Implementations must enqueue both
  // boundaries in stream order and retain independent native resources for
  // every in-flight scope. The default keeps unsupported backends honest.
  virtual StreamGpuTiming begin_gpu_timing() {
    return nullptr;
  }

  virtual void end_gpu_timing(const StreamGpuTiming &timing) {
    (void)timing;
  }

  // Backends may place timestamp markers into a command list already owned by
  // the enclosing submission. The default preserves the separate-marker
  // implementation for streams that do not support inline timing.
  virtual StreamGpuTiming begin_gpu_timing_inline(CommandList *cmdlist) {
    (void)cmdlist;
    return begin_gpu_timing();
  }

  virtual void end_gpu_timing_inline(const StreamGpuTiming &timing,
                                     CommandList *cmdlist) {
    (void)cmdlist;
    end_gpu_timing(timing);
  }

  virtual void command_sync() = 0;
};

class RHI_DLL_EXPORT PipelineCache {
 public:
  virtual ~PipelineCache() = default;

  /**
   * Get the pointer to the raw data of the cache.
   * - Can return `nullptr` if cache is invalid or empty.
   */
  virtual void *data() noexcept {
    return nullptr;
  }

  /**
   * Get the size of the cache (in bytes).
   */
  virtual size_t size() const noexcept {
    return 0;
  }
};

using UPipelineCache = std::unique_ptr<PipelineCache>;

class RHI_DLL_EXPORT Device {
  DeviceCapabilityConfig caps_{};

 public:
  virtual ~Device() {};

  void set_backend_fault_reporter(
      std::shared_ptr<BackendFaultReporter> reporter) noexcept;
  static void clear_backend_fault_reporter(
      const std::shared_ptr<BackendFaultReporter> &reporter) noexcept;
  std::shared_ptr<BackendFaultReporter> backend_fault_reporter() const
      noexcept;
  bool backend_calls_safe() const noexcept;
  void throw_if_backend_submission_disallowed(const char *operation) const;
  void report_backend_error(const BackendRuntimeError &error,
                            std::uint64_t submission_sequence = 0) const
      noexcept;
  [[noreturn]] void raise_backend_error(std::int64_t backend_code,
                                        std::string operation,
                                        std::string message) const;

  struct AllocParams {
    uint64_t size{0};
    bool host_write{false};
    bool host_read{false};
    bool export_sharing{false};
    AllocUsage usage{AllocUsage::Storage};
  };

  virtual RhiResult allocate_memory(const AllocParams &params,
                                    DeviceAllocation *out_devalloc) = 0;

  virtual void dealloc_memory(DeviceAllocation handle) = 0;

  virtual uint64_t get_memory_physical_pointer(DeviceAllocation handle) {
    // FIXME: (penguinliong) This method reports the actual device memory
    // address, it's used for bindless (like argument buffer on Metal). If the
    // backend doesn't have access to physical memory address, it should return
    // null and it depends on the backend implementation to use the address in
    // argument binders.
    return 0;
  }

  // C-2.4.c: maximum number of storage buffer descriptors that may share a
  // single binding slot (descriptorCount in a VkDescriptorSetLayoutBinding).
  // Used by ChunkedDeviceNodeAllocator (vulkan_pointer_allocator_kind=chunked
  // with max_chunks>1) to fail fast when the user-requested chunk count exceeds
  // the physical device limit. Default UINT32_MAX (= unknown / unlimited) for
  // backends that don't expose this information; Vulkan overrides it with
  // VkPhysicalDeviceLimits::maxPerStageDescriptorStorageBuffers.
  virtual uint32_t get_max_storage_buffer_descriptors_per_binding()
      const noexcept {
    return UINT32_MAX;
  }

  virtual bool supports_conditional_commands() const noexcept {
    return false;
  }

  /**
   * Create a Pipeline Cache, which acclerates backend API's pipeline creation.
   * @params[out] out_cache The created pipeline cache.
   * - If operation failed this will be set to `nullptr`
   * @params[in] initial_size Size of the initial data, can be 0.
   * @params[in] initial_data The initial data, can be nullptr.
   * - This data can be used to load back the cache from previous invocations.
   * - The backend API may ignore this data or deem it incompatible.
   * @return The status of this operation.
   * - `success` if the pipeline cache is created successfully.
   * - `out_of_memory` if operation failed due to lack of device or host memory.
   * - `error` if operation failed due to other errors.
   */
  virtual RhiResult create_pipeline_cache(
      PipelineCache **out_cache,
      size_t initial_size = 0,
      const void *initial_data = nullptr) noexcept {
    *out_cache = nullptr;
    return RhiResult::not_supported;
  }

  inline std::pair<UPipelineCache, RhiResult> create_pipeline_cache_unique(
      size_t initial_size = 0,
      const void *initial_data = nullptr) noexcept {
    PipelineCache *cache{nullptr};
    RhiResult res =
        this->create_pipeline_cache(&cache, initial_size, initial_data);
    return std::make_pair(UPipelineCache(cache), res);
  }

  /**
   * Create a Pipeline. A Pipeline is a program that can be dispatched into a
   * stream through a command list.
   * @params[out] out_pipeline The created pipeline.
   * @params[in] src The source description of the pipeline.
   * @params[in] name The name of such pipeline, for debug purposes.
   * @params[in] cache The pipeline cache to use, can be nullptr.
   * @return The status of this operation.
   * - `success` if the pipeline is created successfully.
   * - `out_of_memory` if operation failed due to lack of device or host memory.
   * - `invalid_usage` if the specified source is incompatible or invalid.
   * - `not_supported` if the pipeline uses features the device can't support.
   * - `error` if the operation failed due to other reasons.
   */
  virtual RhiResult create_pipeline(
      Pipeline **out_pipeline,
      const PipelineSourceDesc &src,
      std::string name = "Pipeline",
      PipelineCache *cache = nullptr) noexcept = 0;

  inline std::pair<UPipeline, RhiResult> create_pipeline_unique(
      const PipelineSourceDesc &src,
      std::string name = "Pipeline",
      PipelineCache *cache = nullptr) noexcept {
    Pipeline *pipeline{nullptr};
    RhiResult res = this->create_pipeline(&pipeline, src, name, cache);
    return std::make_pair(UPipeline(pipeline), res);
  }

  inline std::pair<DeviceAllocationUnique, RhiResult> allocate_memory_unique(
      const AllocParams &params) {
    DeviceAllocation alloc;
    RhiResult res = allocate_memory(params, &alloc);
    if (res != RhiResult::success) {
      return std::make_pair(nullptr, res);
    }
    return std::make_pair(std::make_unique<DeviceAllocationGuard>(alloc), res);
  }

  /**
   * Upload data to device allocations immediately.
   * - This is a synchronous operation, function returns when upload is complete
   * - The host data pointers must be valid and large enough for the size of the
   * copy, otherwise this function might segfault
   * - `device_ptr`, `data`, and `sizes` must contain `count` number of valid
   * values
   * @params[in] device_ptr The array to destination device pointers.
   * @params[in] data The array to source host pointers.
   * @params[in] sizes The array to sizes of data/copy.
   * @params[in] count The number of uploads to perform.
   * @return The status of this operation
   * - `success` if the upload is successful.
   * - `out_of_memory` if operation failed due to lack of device or host memory.
   * - `invalid_usage` if the specified source is incompatible or invalid.
   * - `error` if the operation failed due to other reasons.
   */
  virtual RhiResult upload_data(DevicePtr *device_ptr,
                                const void **data,
                                size_t *size,
                                int num_alloc = 1) noexcept;

  /**
   * Read data from device allocations back to host immediately.
   * - This is a synchronous operation, function returns when readback is
   * complete
   * - The host data pointers must be valid and large enough for the size of the
   * copy, otherwise this function might segfault
   * - `device_ptr`, `data`, and `sizes` must contain `count` number of valid
   * values
   * @params[in] device_ptr The array to source device pointers.
   * @params[in] data The array to destination host pointers.
   * @params[in] sizes The array to sizes of data/copy.
   * @params[in] count The number of readbacks to perform.
   * @params[in] wait_sema The semaphores to wait for before the copy is
   * initiated.
   * @return The status of this operation
   * - `success` if the upload is successful.
   * - `out_of_memory` if operation failed due to lack of device or host memory.
   * - `invalid_usage` if the specified source is incompatible or invalid.
   * - `error` if the operation failed due to other reasons.
   */
  virtual RhiResult readback_data(
      DevicePtr *device_ptr,
      void **data,
      size_t *size,
      int num_alloc = 1,
      const std::vector<StreamSemaphore> &wait_sema = {}) noexcept;

  /**
   * Read several device ranges through one caller-owned host-readable staging
   * allocation. The copy is synchronous, but the staging allocation may be
   * retained and reused by a higher-level transfer plan.
   *
   * This conservative first contract accepts only four-byte aligned source
   * offsets and sizes, matching the portable buffer-copy requirements used by
   * Graph scalar observations. Other requests return `not_supported` so the
   * caller can use readback_data() without changing semantics.
   */
  RhiResult readback_data_packed(
      DevicePtr *device_ptr,
      void **data,
      size_t *size,
      int num_alloc,
      DevicePtr staging,
      size_t staging_size,
      const std::vector<StreamSemaphore> &wait_sema = {}) noexcept;

  // Each thraed will acquire its own stream
  virtual Stream *get_compute_stream() = 0;

  // Wait for all tasks to complete (task from all streams)
  virtual void wait_idle() = 0;

  /**
   * Create a new shader resource set
   * @return The new shader resource set pointer
   */
  virtual ShaderResourceSet *create_resource_set() = 0;

  /**
   * Create a new shader resource set (wrapped in unique ptr)
   * @return The new shader resource set unique pointer
   */
  inline std::unique_ptr<ShaderResourceSet> create_resource_set_unique() {
    return std::unique_ptr<ShaderResourceSet>(this->create_resource_set());
  }

  /**
   * Map a range within a DeviceAllocation memory into host address space.
   *
   * @param[in] ptr The Device Pointer to map.
   * @param[in] size The size of the mapped region.
   * @param[out] mapped_ptr Outputs the pointer to the mapped region.
   * @return The result status.
   *         `success` when the mapping is successful.
   *         `invalid_usage` when the memory is not host visible.
   *         `invalid_usage` when trying to map the memory multiple times.
   *         `invalid_usage` when `ptr.offset + size` is out-of-bounds.
   *         `error` when the mapping failed for other reasons.
   */
  virtual RhiResult map_range(DevicePtr ptr,
                              uint64_t size,
                              void **mapped_ptr) = 0;

  /**
   * Map an entire DeviceAllocation into host address space.
   * @param[in] ptr The Device Pointer to map.
   * @param[in] size The size of the mapped region.
   * @param[out] mapped_ptr Outputs the pointer to the mapped region.
   * @return The result status.
   *         `success` when the mapping is successful.
   *         `invalid_usage` when the memory is not host visible.
   *         `invalid_usage` when trying to map the memory multiple times.
   *         `invalid_usage` when `ptr.offset + size` is out-of-bounds.
   *         `error` when the mapping failed for other reasons.
   */
  virtual RhiResult map(DeviceAllocation alloc, void **mapped_ptr) = 0;

  /**
   * Unmap a previously mapped DevicePtr or DeviceAllocation.
   * @param[in] ptr The DevicePtr to unmap.
   */
  virtual void unmap(DevicePtr ptr) = 0;

  /**
   * Unmap a previously mapped DevicePtr or DeviceAllocation.
   * @param[in] alloc The DeviceAllocation to unmap
   */
  virtual void unmap(DeviceAllocation alloc) = 0;

  // Directly share memory in the form of alias
  static DeviceAllocation share_to(DeviceAllocation *alloc, Device *target);

  // Strictly intra device copy (synced)
  virtual void memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) = 0;

  // Copy memory inter or intra devices (synced)
  enum class MemcpyCapability { Direct, RequiresStagingBuffer, RequiresHost };

  static MemcpyCapability check_memcpy_capability(DevicePtr dst,
                                                  DevicePtr src,
                                                  uint64_t size);

  static void memcpy_direct(DevicePtr dst, DevicePtr src, uint64_t size);

  static void memcpy_via_staging(DevicePtr dst,
                                 DevicePtr staging,
                                 DevicePtr src,
                                 uint64_t size);

  static void memcpy_via_host(DevicePtr dst,
                              void *host_buffer,
                              DevicePtr src,
                              uint64_t size);

  // Get all supported capabilities of the current created device.
  virtual Arch arch() const = 0;
  inline const DeviceCapabilityConfig &get_caps() const {
    return caps_;
  }
  inline void set_caps(DeviceCapabilityConfig &&caps) {
    caps_ = std::move(caps);
  }

  // Profiler support
  virtual void profiler_sync() {
  }

  virtual size_t profiler_get_sampler_count() {
    return 0;
  }

  virtual std::vector<std::pair<std::string, double>>
  profiler_flush_sampled_time() {
    return std::vector<std::pair<std::string, double>>();
  }
};

struct SurfaceImage {
  StreamSemaphore image_available{nullptr};
  DeviceAllocation image;
  uint32_t image_index{0};
};

class RHI_DLL_EXPORT Surface {
 public:
  virtual ~Surface() {
  }

  virtual StreamSemaphore acquire_next_image() = 0;
  virtual DeviceAllocation get_target_image() = 0;
  virtual SurfaceImage acquire_surface_image() {
    SurfaceImage surface_image;
    surface_image.image_available = acquire_next_image();
    surface_image.image = get_target_image();
    return surface_image;
  }
  virtual bool try_acquire_surface_image(SurfaceImage *surface_image) {
    *surface_image = acquire_surface_image();
    return true;
  }
  virtual void present_image(
      const std::vector<StreamSemaphore> &wait_semaphores = {}) = 0;
  virtual void present_surface_image(
      const SurfaceImage &surface_image,
      const std::vector<StreamSemaphore> &wait_semaphores = {}) {
    (void)surface_image;
    present_image(wait_semaphores);
  }
  virtual std::pair<uint32_t, uint32_t> get_size() = 0;
  virtual int get_image_count() = 0;
  virtual BufferFormat image_format() = 0;
  virtual void resize(uint32_t width, uint32_t height) = 0;
};

struct VertexInputBinding {
  uint32_t binding{0};
  size_t stride{0};
  bool instance{false};
};

struct VertexInputAttribute {
  uint32_t location{0};
  uint32_t binding{0};
  BufferFormat format;
  uint32_t offset{0};
};

struct SurfaceConfig {
  // VSync:
  // - true: will attempt to wait for V-Blank
  // - when adaptive is true: when supported, if a V-Blank is missed, instead of
  //   waiting, a tearing may appear, reduces overall latency
  bool vsync{false};
  bool adaptive{true};
  uint32_t width{1};
  uint32_t height{1};
  void *native_surface_handle{nullptr};
};

enum class ImageAllocUsage : int {
  None = 0,
  Storage = 1,
  Sampled = 2,
  Attachment = 4,
};
inline ImageAllocUsage operator|(ImageAllocUsage a, ImageAllocUsage b) {
  return static_cast<ImageAllocUsage>(static_cast<int>(a) |
                                      static_cast<int>(b));
}
inline bool operator&(ImageAllocUsage a, ImageAllocUsage b) {
  return static_cast<int>(a) & static_cast<int>(b);
}

struct ImageParams {
  ImageDimension dimension;
  BufferFormat format;
  ImageLayout initial_layout{ImageLayout::undefined};
  uint32_t x{1};
  uint32_t y{1};
  uint32_t z{1};
  bool export_sharing{false};
  ImageAllocUsage usage{ImageAllocUsage::Storage | ImageAllocUsage::Sampled |
                        ImageAllocUsage::Attachment};
  ImageSamplerConfig sampler_config{};
};

struct BlendFunc {
  BlendOp op{BlendOp::add};
  BlendFactor src_factor{BlendFactor::src_alpha};
  BlendFactor dst_factor{BlendFactor::one_minus_src_alpha};
};

struct BlendingParams {
  bool enable{true};
  BlendFunc color;
  BlendFunc alpha;
};

struct RasterParams {
  TopologyType prim_topology{TopologyType::Triangles};
  PolygonMode polygon_mode{PolygonMode::Fill};
  bool front_face_cull{false};
  bool back_face_cull{false};
  bool depth_test{false};
  bool depth_write{false};
  std::vector<BlendingParams> blending{};
};

class RHI_DLL_EXPORT GraphicsDevice : public Device {
 public:
  virtual std::unique_ptr<Pipeline> create_raster_pipeline(
      const std::vector<PipelineSourceDesc> &src,
      const RasterParams &raster_params,
      const std::vector<VertexInputBinding> &vertex_inputs,
      const std::vector<VertexInputAttribute> &vertex_attrs,
      std::string name = "Pipeline") = 0;

  virtual Stream *get_graphics_stream() = 0;

  /**
   * Create a new raster resources set
   * @return The new RasterResources pointer
   */
  virtual RasterResources *create_raster_resources() = 0;

  /**
   * Create a new raster resources set (wrapped in unique ptr)
   * @return The new RasterResources unique pointer
   */
  inline std::unique_ptr<RasterResources> create_raster_resources_unique() {
    return std::unique_ptr<RasterResources>(this->create_raster_resources());
  }

  virtual std::unique_ptr<Surface> create_surface(
      const SurfaceConfig &config) = 0;
  // You are not expected to call this directly. If you want to use this image
  // in a taichi kernel, you usually want to create the image via
  // `GfxRuntime::create_image`. `GfxRuntime` is available in `ProgramImpl`
  // of GPU backends.
  virtual DeviceAllocation create_image(const ImageParams &params) = 0;
  inline DeviceImageUnique create_image_unique(const ImageParams &params) {
    return std::make_unique<DeviceImageGuard>(this->create_image(params));
  }
  virtual void destroy_image(DeviceAllocation handle) = 0;

  virtual void image_transition(DeviceAllocation img,
                                ImageLayout old_layout,
                                ImageLayout new_layout);
  virtual void buffer_to_image(DeviceAllocation dst_img,
                               DevicePtr src_buf,
                               ImageLayout img_layout,
                               const BufferImageCopyParams &params);
  virtual void image_to_buffer(DevicePtr dst_buf,
                               DeviceAllocation src_img,
                               ImageLayout img_layout,
                               const BufferImageCopyParams &params);
};

}  // namespace taichi::lang

template <>
class fmt::formatter<taichi::lang::RhiResult> {
 public:
  constexpr auto parse(format_parse_context &ctx) {
    return ctx.begin();
  }
  template <typename Context>
  constexpr auto format(taichi::lang::RhiResult const &res,
                        Context &ctx) const {
    return format_to(ctx.out(), taichi::lang::rhi_result_to_string(res));
  }
};
