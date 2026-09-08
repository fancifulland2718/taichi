#pragma once

#include "taichi/rhi/device.h"
#include "taichi/rhi/common/runtime_telemetry.h"
#include "taichi/rhi/vulkan/vulkan_api.h"
#include "taichi/rhi/vulkan/vulkan_utils.h"
#include "taichi/common/ref_counted_pool.h"

#include "vk_mem_alloc.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace taichi::lang {
namespace vulkan {

using std::unordered_map;

class VulkanDevice;
class VulkanResourceBinder;
class VulkanStream;

struct VulkanProfilerSampler {
  std::string kernel_name;
  vkapi::IVkQueryPool query_pool;
  // Assigned only after the command buffer has been submitted successfully.
  vkapi::IVkFence fence;
};

struct SpirvCodeView {
  const uint32_t *data = nullptr;
  size_t size = 0;
  VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;

  SpirvCodeView() = default;

  explicit SpirvCodeView(const std::vector<uint32_t> &code)
      : data(code.data()), size(code.size() * sizeof(uint32_t)) {
  }
};

struct VulkanRenderPassDesc {
  std::vector<std::pair<VkFormat, bool>> color_attachments;
  VkFormat depth_attachment{VK_FORMAT_UNDEFINED};
  bool clear_depth{false};
  VkImageLayout color_final_layout{VK_IMAGE_LAYOUT_PRESENT_SRC_KHR};

  bool operator==(const VulkanRenderPassDesc &other) const {
    if (other.depth_attachment != depth_attachment) {
      return false;
    }
    if (other.clear_depth != clear_depth) {
      return false;
    }
    if (other.color_final_layout != color_final_layout) {
      return false;
    }
    return other.color_attachments == color_attachments;
  }
};

struct RenderPassDescHasher {
  std::size_t operator()(const VulkanRenderPassDesc &desc) const {
    size_t hash = std::hash<uint64_t>()((uint64_t(desc.depth_attachment) << 1) |
                                        uint64_t(desc.clear_depth));
    rhi_impl::hash_combine(
        hash, std::hash<uint32_t>()(
                  static_cast<uint32_t>(desc.color_final_layout)));
    for (auto &pair : desc.color_attachments) {
      size_t hash_pair = std::hash<uint64_t>()((uint64_t(pair.first) << 1) |
                                               uint64_t(pair.second));
      rhi_impl::hash_combine(hash, hash_pair);
    }
    return hash;
  }
};

struct VulkanFramebufferDesc {
  std::vector<vkapi::IVkImageView> attachments{};
  uint32_t width{0};
  uint32_t height{0};
  vkapi::IVkRenderPass renderpass{nullptr};

  bool operator==(const VulkanFramebufferDesc &other) const {
    return width == other.width && height == other.height &&
           renderpass == other.renderpass && attachments == other.attachments;
  }
};

class VulkanResourceSet : public ShaderResourceSet {
 public:
  struct Buffer {
    vkapi::IVkBuffer buffer{nullptr};
    VkDeviceSize offset{0};
    VkDeviceSize size{0};

    bool operator==(const Buffer &rhs) const {
      return buffer == rhs.buffer && offset == rhs.offset && size == rhs.size;
    }

    bool operator!=(const Buffer &rhs) const {
      return !(*this == rhs);
    }
  };

  struct Image {
    vkapi::IVkImageView view{nullptr};

    bool operator==(const Image &rhs) const {
      return view == rhs.view;
    }

    bool operator!=(const Image &rhs) const {
      return view != rhs.view;
    }
  };

  struct Texture {
    vkapi::IVkImageView view{nullptr};
    vkapi::IVkSampler sampler{nullptr};

    bool operator==(const Texture &rhs) const {
      return view == rhs.view && sampler == rhs.sampler;
    }

    bool operator!=(const Texture &rhs) const {
      return !(*this == rhs);
    }
  };

  // C-2.5 (2026-05): descriptor array of storage buffers. Each entry maps
  // to one VkDescriptorBufferInfo with offset=0, range=VK_WHOLE_SIZE; the
  // SPIR-V side accesses chunk[k] via OpAccessChain on the array variable.
  // descriptorCount = buffers.size() at finalize time.
  struct BufferArray {
    std::vector<vkapi::IVkBuffer> buffers;

    bool operator==(const BufferArray &rhs) const {
      if (buffers.size() != rhs.buffers.size()) {
        return false;
      }
      for (size_t i = 0; i < buffers.size(); ++i) {
        if (buffers[i] != rhs.buffers[i]) {
          return false;
        }
      }
      return true;
    }

    bool operator!=(const BufferArray &rhs) const {
      return !(*this == rhs);
    }
  };

  struct AccelerationStructure {
    vkapi::IVkAccelerationStructureKHR acceleration_structure{nullptr};

    bool operator==(const AccelerationStructure &rhs) const {
      return acceleration_structure == rhs.acceleration_structure;
    }

    bool operator!=(const AccelerationStructure &rhs) const {
      return !(*this == rhs);
    }
  };

  struct Binding {
    VkDescriptorType type{VK_DESCRIPTOR_TYPE_MAX_ENUM};
    std::variant<Buffer, Image, Texture, BufferArray, AccelerationStructure> res{
        Buffer()};

    bool operator==(const Binding &other) const {
      return other.type == type && other.res == res;
    }

    bool operator!=(const Binding &other) const {
      return other.type != type || other.res != res;
    }

    size_t hash() const {
      size_t hash = 0;
      rhi_impl::hash_combine(hash, int(type));
      if (const Buffer *buf = std::get_if<Buffer>(&res)) {
        rhi_impl::hash_combine(hash, (void *)buf->buffer.get());
        rhi_impl::hash_combine(hash, size_t(buf->offset));
        rhi_impl::hash_combine(hash, size_t(buf->size));
      } else if (const Image *img = std::get_if<Image>(&res)) {
        rhi_impl::hash_combine(hash, (void *)img->view.get());
      } else if (const Texture *tex = std::get_if<Texture>(&res)) {
        rhi_impl::hash_combine(hash, (void *)tex->view.get());
        rhi_impl::hash_combine(hash, (void *)tex->sampler.get());
      } else if (const BufferArray *ba = std::get_if<BufferArray>(&res)) {
        rhi_impl::hash_combine(hash, ba->buffers.size());
        for (const auto &b : ba->buffers) {
          rhi_impl::hash_combine(hash, (void *)b.get());
        }
      } else if (const AccelerationStructure *as =
                     std::get_if<AccelerationStructure>(&res)) {
        rhi_impl::hash_combine(
            hash, (void *)as->acceleration_structure.get());
      }
      return hash;
    }
  };

  static uint32_t descriptor_count(const Binding &binding) {
    if (const auto *array = std::get_if<BufferArray>(&binding.res)) {
      return static_cast<uint32_t>(array->buffers.size());
    }
    return 1;
  }

  // This hashes the Set Layout
  struct SetLayoutHasher {
    std::size_t operator()(const VulkanResourceSet &set) const {
      // NOTE: Bindings in this case is ordered, we can use non-commutative
      // operations
      size_t hash = 0;
      for (const auto &pair : set.bindings_) {
        rhi_impl::hash_combine(hash, pair.first);
        // We only care about type in this case
        rhi_impl::hash_combine(hash, pair.second.type);
        rhi_impl::hash_combine(hash, descriptor_count(pair.second));
      }
      return hash;
    }
  };

  // This compares the layout of two sets
  struct SetLayoutCmp {
    bool operator()(const VulkanResourceSet &lhs,
                    const VulkanResourceSet &rhs) const {
      if (lhs.bindings_.size() != rhs.bindings_.size()) {
        return false;
      }
      for (auto &lhs_pair : lhs.bindings_) {
        auto rhs_binding_iter = rhs.bindings_.find(lhs_pair.first);
        if (rhs_binding_iter == rhs.bindings_.end()) {
          return false;
        }
        const Binding &rhs_binding = rhs_binding_iter->second;
        if (rhs_binding.type != lhs_pair.second.type ||
            descriptor_count(rhs_binding) !=
                descriptor_count(lhs_pair.second)) {
          return false;
        }
      }
      return true;
    }
  };

  // This hashes the entire set (including resources)
  struct DescSetHasher {
    std::size_t operator()(const VulkanResourceSet &set) const {
      size_t hash = 0;
      for (const auto &pair : set.bindings_) {
        rhi_impl::hash_combine(hash, pair.first);
        hash ^= pair.second.hash() + 0x9e3779b9 + (hash << 6) + (hash >> 2);
      }
      return hash;
    }
  };

  // This compares two sets (including resources)
  struct SetCmp {
    bool operator()(const VulkanResourceSet &lhs,
                    const VulkanResourceSet &rhs) const {
      return lhs.bindings_ == rhs.bindings_;
    }
  };

  explicit VulkanResourceSet(VulkanDevice *device);
  VulkanResourceSet(const VulkanResourceSet &other) = default;
  ~VulkanResourceSet() override;

  ShaderResourceSet &rw_buffer(uint32_t binding,
                               DevicePtr ptr,
                               size_t size) final;
  ShaderResourceSet &rw_buffer(uint32_t binding, DeviceAllocation alloc) final;
  ShaderResourceSet &buffer(uint32_t binding, DevicePtr ptr, size_t size) final;
  ShaderResourceSet &buffer(uint32_t binding, DeviceAllocation alloc) final;
  ShaderResourceSet &image(uint32_t binding,
                           DeviceAllocation alloc,
                           ImageSamplerConfig sampler_config) final;
  ShaderResourceSet &rw_image(uint32_t binding,
                              DeviceAllocation alloc,
                              int lod) final;

  // C-2.5 (2026-05): descriptor array of storage buffers (single binding,
  // descriptorCount=N). All buffers must use VK_WHOLE_SIZE range.
  ShaderResourceSet &rw_buffer_array(
      uint32_t binding,
      const std::vector<DeviceAllocation> &allocs) final;

  // Vulkan-only descriptor used by explicit ray-query providers. This is not
  // part of the backend-neutral ShaderResourceSet contract because other
  // backends expose different acceleration-structure object models.
  VulkanResourceSet &acceleration_structure(
      uint32_t binding,
      vkapi::IVkAccelerationStructureKHR acceleration_structure);

  rhi_impl::RhiReturn<vkapi::IVkDescriptorSet> finalize();
  RhiResult prepare_for_replay(bool patch_existing) final;

  vkapi::IVkDescriptorSetLayout get_layout() {
    return layout_;
  }

  const std::map<uint32_t, Binding> &get_bindings() const {
    return bindings_;
  }

 private:
  rhi_impl::RhiReturn<vkapi::IVkDescriptorSet> finalize_impl(
      bool replay_dedicated,
      bool patch_existing);
  void set_binding(uint32_t binding, Binding new_binding);

  std::map<uint32_t, Binding> bindings_;
  VulkanDevice *device_;

  vkapi::IVkDescriptorSetLayout layout_{nullptr};
  vkapi::IVkDescriptorSet set_{nullptr};

  bool dirty_{true};
};

class VulkanRasterResources : public RasterResources {
 public:
  explicit VulkanRasterResources(VulkanDevice *device) : device_(device) {
  }

  struct BufferBinding {
    vkapi::IVkBuffer buffer{nullptr};
    size_t offset{0};
  };

  std::unordered_map<uint32_t, BufferBinding> vertex_buffers;
  BufferBinding index_binding;
  VkIndexType index_type{VK_INDEX_TYPE_MAX_ENUM};

  ~VulkanRasterResources() override = default;

  RasterResources &vertex_buffer(DevicePtr ptr, uint32_t binding = 0) final;
  RasterResources &index_buffer(DevicePtr ptr, size_t index_width) final;

 private:
  VulkanDevice *device_;
};

class VulkanPipelineCache : public PipelineCache {
 public:
  VulkanPipelineCache(VulkanDevice *device,
                      size_t initial_size,
                      const void *initial_data);
  ~VulkanPipelineCache() override;

  void *data() noexcept final;
  size_t size() const noexcept final;

  vkapi::IVkPipelineCache vk_pipeline_cache() {
    return cache_;
  }

  bool is_valid() const {
    return cache_ != nullptr;
  }

 private:
  VulkanDevice *device_{nullptr};
  vkapi::IVkPipelineCache cache_{nullptr};
  mutable std::mutex mutex_;
  std::vector<uint8_t> data_shadow_;
};

// VulkanPipeline maps to a vkapi::IVkPipeline, or a SPIR-V module (a GLSL
// compute shader).
class VulkanPipeline : public Pipeline {
 public:
  struct Params {
    VulkanDevice *device{nullptr};
    std::vector<SpirvCodeView> code;
    std::string name{"Pipeline"};
    vkapi::IVkPipelineCache cache{nullptr};
  };

  explicit VulkanPipeline(const Params &params);
  explicit VulkanPipeline(
      const Params &params,
      const RasterParams &raster_params,
      const std::vector<VertexInputBinding> &vertex_inputs,
      const std::vector<VertexInputAttribute> &vertex_attrs);
  ~VulkanPipeline() override;

  vkapi::IVkPipelineLayout pipeline_layout() const {
    return pipeline_layout_;
  }

  vkapi::IVkPipeline pipeline() const {
    return pipeline_;
  }

  vkapi::IVkPipeline graphics_pipeline(
      const VulkanRenderPassDesc &renderpass_desc,
      vkapi::IVkRenderPass renderpass);

  vkapi::IVkPipeline graphics_pipeline_dynamic(
      const VulkanRenderPassDesc &renderpass_desc);

  const std::string &name() const {
    return name_;
  }

  bool is_graphics() const {
    return graphics_pipeline_template_ != nullptr;
  }

  std::unordered_map<uint32_t, VulkanResourceSet> &
  get_resource_set_templates() {
    return set_templates_;
  }

 private:
  void create_descriptor_set_layout(const Params &params);
  void create_shader_stages(const Params &params);
  void create_pipeline_layout();
  void create_compute_pipeline(const Params &params);
  void create_graphics_pipeline(
      const RasterParams &raster_params,
      const std::vector<VertexInputBinding> &vertex_inputs,
      const std::vector<VertexInputAttribute> &vertex_attrs);

  static VkShaderModule create_shader_module(VkDevice device,
                                             const SpirvCodeView &code);

  struct GraphicsPipelineTemplate {
    VkPipelineViewportStateCreateInfo viewport_state{};
    std::vector<VkVertexInputBindingDescription> input_bindings;
    std::vector<VkVertexInputAttributeDescription> input_attrs;
    VkPipelineVertexInputStateCreateInfo input{};
    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    VkPipelineMultisampleStateCreateInfo multisampling{};
    VkPipelineDepthStencilStateCreateInfo depth_stencil{};
    VkPipelineColorBlendStateCreateInfo color_blending{};
    std::vector<VkPipelineColorBlendAttachmentState> blend_attachments{};
    std::vector<VkDynamicState> dynamic_state_enables = {
        VK_DYNAMIC_STATE_LINE_WIDTH, VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic_state{};
    VkGraphicsPipelineCreateInfo pipeline_info{};
  };

  VulkanDevice &ti_device_;          // not owned
  VkDevice device_{VK_NULL_HANDLE};  // not owned

  std::string name_;

  std::vector<VkPipelineShaderStageCreateInfo> shader_stages_;

  std::unique_ptr<GraphicsPipelineTemplate> graphics_pipeline_template_;
  std::mutex graphics_pipeline_mutex_;
  std::unordered_map<vkapi::IVkRenderPass, vkapi::IVkPipeline>
      graphics_pipeline_;

  // For KHR_dynamic_rendering
  std::unordered_map<VulkanRenderPassDesc,
                     vkapi::IVkPipeline,
                     RenderPassDescHasher>
      graphics_pipeline_dynamic_;

  std::unordered_map<uint32_t, VulkanResourceSet> set_templates_;
  std::vector<vkapi::IVkDescriptorSetLayout> set_layouts_;
  std::vector<VkShaderModule> shader_modules_;
  vkapi::IVkPipeline pipeline_{VK_NULL_HANDLE};
  vkapi::IVkPipelineLayout pipeline_layout_{VK_NULL_HANDLE};
  vkapi::IVkPipelineCache cache_{nullptr};
};

class VulkanCommandList : public CommandList {
 public:
  VulkanCommandList(VulkanDevice *ti_device,
                    VulkanStream *stream,
                    vkapi::IVkCommandBuffer buffer);
  ~VulkanCommandList() override;

  void bind_pipeline(Pipeline *p) noexcept final;
  RhiResult bind_shader_resources(ShaderResourceSet *res,
                                  int set_index = 0) noexcept final;
  RhiResult bind_raster_resources(RasterResources *res) noexcept final;
  void buffer_barrier(DevicePtr ptr, size_t size) noexcept final;
  void buffer_transition(DevicePtr ptr,
                         size_t size,
                         const BufferTransition &transition) noexcept final;
  void buffer_barrier(DeviceAllocation alloc) noexcept final;
  void memory_barrier() noexcept final;
  void buffer_copy(DevicePtr dst, DevicePtr src, size_t size) noexcept final;
  void buffer_fill(DevicePtr ptr, size_t size, uint32_t data) noexcept final;
  void push_constants(const void *data, uint32_t size) noexcept;
  RhiResult dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1) noexcept final;
  RhiResult dispatch_indirect(DevicePtr indirect) noexcept final;
  RhiResult begin_conditional(DevicePtr predicate,
                              bool inverted = false) noexcept final;
  RhiResult end_conditional() noexcept final;
  void begin_renderpass(int x0,
                        int y0,
                        int x1,
                        int y1,
                        uint32_t num_color_attachments,
                        DeviceAllocation *color_attachments,
                        bool *color_clear,
                        std::vector<float> *clear_colors,
                        DeviceAllocation *depth_attachment,
                        bool depth_clear) override;

  // Offscreen render passes keep their color target in attachment layout so
  // the caller can record an explicit post-pass transition. Swapchain callers
  // retain the historical present-src default.
  void set_next_renderpass_color_final_layout(ImageLayout layout);
  void end_renderpass() override;
  void set_raster_viewport_and_scissor(int x0,
                                       int y0,
                                       int x1,
                                       int y1) override;
  void draw(uint32_t num_verticies, uint32_t start_vertex = 0) override;
  void draw_instance(uint32_t num_verticies,
                     uint32_t num_instances,
                     uint32_t start_vertex = 0,
                     uint32_t start_instance = 0) override;
  void draw_indexed(uint32_t num_indicies,
                    int32_t vertex_offset = 0,
                    uint32_t start_index = 0) override;
  void draw_indexed_instance(uint32_t num_indicies,
                             uint32_t num_instances,
                             int32_t vertex_offset = 0,
                             uint32_t start_index = 0,
                             uint32_t start_instance = 0) override;
  RhiResult draw_indirect(DevicePtr indirect,
                          uint32_t draw_count,
                          uint32_t stride) noexcept override;
  RhiResult draw_indexed_indirect(DevicePtr indirect,
                                  uint32_t draw_count,
                                  uint32_t stride) noexcept override;
  RhiResult draw_indirect_count(DevicePtr indirect,
                                DevicePtr count,
                                uint32_t max_draw_count,
                                uint32_t stride) noexcept override;
  RhiResult draw_indexed_indirect_count(
      DevicePtr indirect,
      DevicePtr count,
      uint32_t max_draw_count,
      uint32_t stride) noexcept override;
  RhiResult draw_mesh_tasks(uint32_t group_count_x,
                            uint32_t group_count_y,
                            uint32_t group_count_z,
                            bool task_shader) noexcept override;
  void set_line_width(float width) override;
  void image_transition(DeviceAllocation img,
                        ImageLayout old_layout,
                        ImageLayout new_layout) override;
  void buffer_to_image(DeviceAllocation dst_img,
                       DevicePtr src_buf,
                       ImageLayout img_layout,
                       const BufferImageCopyParams &params) override;
  void image_to_buffer(DevicePtr dst_buf,
                       DeviceAllocation src_img,
                       ImageLayout img_layout,
                       const BufferImageCopyParams &params) override;

  void copy_image(DeviceAllocation dst_img,
                  DeviceAllocation src_img,
                  ImageLayout dst_img_layout,
                  ImageLayout src_img_layout,
                  const ImageCopyParams &params) override;

  void blit_image(DeviceAllocation dst_img,
                  DeviceAllocation src_img,
                  ImageLayout dst_img_layout,
                  ImageLayout src_img_layout,
                  const ImageBlitParams &params) override;

  vkapi::IVkRenderPass current_renderpass();

  // Vulkan specific functions
  vkapi::IVkCommandBuffer finalize();

  vkapi::IVkCommandBuffer vk_command_buffer();

  // External compute recording changes the bound pipeline. Keep the external
  // objects with the command buffer and invalidate only recording-time state.
  VkCommandBuffer begin_external_compute(vkapi::IDeviceObj owner);
  void execute_secondary(vkapi::IVkCommandBuffer buffer);
  std::function<void(CommandList *)> finalize_secondary(
      std::shared_ptr<void> resource_owner) override;

  // Profiler support
  void begin_profiler_scope(const std::string &kernel_name) override;
  void end_profiler_scope() override;
  std::vector<VulkanProfilerSampler> take_completed_profiler_samplers();

  // Ticket-owned timing support. These markers do not enter the global
  // kernel-profiler sampler queue and therefore remain replay/ticket local.
  void write_runtime_timestamp(const vkapi::IVkQueryPool &query_pool,
                               std::uint32_t query,
                               VkPipelineStageFlagBits stage,
                               bool reset);

 private:
  void apply_raster_viewport_and_scissor();

  bool finalized_{false};
  VulkanDevice *ti_device_;
  VulkanStream *stream_;
  VkDevice device_;
  vkapi::IVkCommandBuffer buffer_;
  VulkanPipeline *current_pipeline_{nullptr};
  bool conditional_active_{false};

  struct ProfilerScope {
    std::string kernel_name;
    vkapi::IVkQueryPool query_pool;
  };
  std::vector<ProfilerScope> profiler_scopes_;
  // Each completed scope reserves one device-level sampler slot. The
  // reservation is released by the device after its result is collected, or
  // by this command list if it is discarded before submission.
  size_t profiler_sampler_reservations_{0};
  std::vector<VulkanProfilerSampler> completed_profiler_samplers_;

  // Renderpass & raster pipeline
  std::vector<vkapi::IVkImage> current_dynamic_targets_;
  VulkanRenderPassDesc current_renderpass_desc_;
  vkapi::IVkRenderPass current_renderpass_{VK_NULL_HANDLE};
  vkapi::IVkFramebuffer current_framebuffer_{VK_NULL_HANDLE};
  int32_t viewport_x_{0}, viewport_y_{0};
  uint32_t viewport_width_{0}, viewport_height_{0};
};

enum class VulkanSurfaceResult {
  kSuccess,
  kSuboptimal,
  kOutOfDate,
  kDeviceLost,
  kError,
};

constexpr VulkanSurfaceResult classify_vulkan_surface_result(VkResult result) {
  if (result == VK_SUCCESS) {
    return VulkanSurfaceResult::kSuccess;
  }
  if (result == VK_SUBOPTIMAL_KHR) {
    return VulkanSurfaceResult::kSuboptimal;
  }
  if (result == VK_ERROR_OUT_OF_DATE_KHR) {
    return VulkanSurfaceResult::kOutOfDate;
  }
  if (result == VK_ERROR_DEVICE_LOST) {
    return VulkanSurfaceResult::kDeviceLost;
  }
  return VulkanSurfaceResult::kError;
}

class VulkanSurface : public Surface {
 public:
  VulkanSurface(VulkanDevice *device, const SurfaceConfig &config);
  ~VulkanSurface() override;

  StreamSemaphore acquire_next_image() override;
  DeviceAllocation get_target_image() override;
  SurfaceImage acquire_surface_image() override;
  bool try_acquire_surface_image(SurfaceImage *surface_image) override;

  void present_image(
      const std::vector<StreamSemaphore> &wait_semaphores = {}) override;
  void present_surface_image(
      const SurfaceImage &surface_image,
      const std::vector<StreamSemaphore> &wait_semaphores = {}) override;
  std::vector<StreamSemaphore> take_present_waits_after_acquire(
      uint32_t image_index);
  std::pair<uint32_t, uint32_t> get_size() override;
  int get_image_count() override;
  BufferFormat image_format() override;
  void resize(uint32_t width, uint32_t height) override;

  bool needs_swapchain_recreate() const {
    return swapchain_needs_recreate_;
  }

  bool device_lost() const {
    return device_lost_;
  }

 private:
  void create_swap_chain();
  void destroy_swap_chain();
  void create_offscreen_images();
  void destroy_offscreen_images();
  bool handle_surface_result(VkResult result, const char *operation);

  SurfaceConfig config_;

  VulkanDevice *device_{nullptr};
  VkSurfaceKHR surface_{VK_NULL_HANDLE};
  VkSwapchainKHR swapchain_{VK_NULL_HANDLE};
  std::vector<vkapi::IVkSemaphore> image_available_;
  uint32_t image_available_index_{0};
  BufferFormat image_format_{BufferFormat::unknown};

  uint32_t image_index_{0};

  uint32_t width_{0};
  uint32_t height_{0};

  std::vector<DeviceAllocation> swapchain_images_;
  std::vector<std::vector<StreamSemaphore>> present_waits_by_image_;
  bool swapchain_needs_recreate_{false};
  bool device_lost_{false};
};

struct DescPool {
  VkDescriptorPool pool;
  // Threads share descriptor sets
  RefCountedPool<vkapi::IVkDescriptorSet, true> sets;

  explicit DescPool(VkDescriptorPool pool) : pool(pool) {
  }
};

class VulkanStreamSemaphoreObject : public StreamSemaphoreObject {
 public:
  explicit VulkanStreamSemaphoreObject(
      std::shared_ptr<BackendFaultReporter> fault_reporter,
      vkapi::IVkSemaphore sema,
      vkapi::IVkFence fence = nullptr,
      BackendWaitTelemetry *wait_telemetry = nullptr)
      : vkapi_ref(sema),
        fence_ref(fence),
        fault_reporter_(std::move(fault_reporter)),
        wait_telemetry_(wait_telemetry) {
  }
  ~VulkanStreamSemaphoreObject() override {
  }

  bool is_ready() const override;
  bool wait() const override;

  vkapi::IVkSemaphore vkapi_ref{nullptr};
  vkapi::IVkFence fence_ref{nullptr};

 private:
  std::shared_ptr<BackendFaultReporter> fault_reporter_;
  BackendWaitTelemetry *wait_telemetry_{nullptr};
};

using VulkanQueueLockTelemetry = SampledLockTelemetry<std::mutex>;

struct VulkanRuntimeTelemetrySnapshot {
  BackendWaitTelemetry::Snapshot wait;
  VulkanQueueLockTelemetry::Snapshot queue_lock;
};

struct VulkanQueueSubmissionSnapshot {
  std::uint64_t queue_submit_calls{0};
  std::uint64_t submitted_command_buffers{0};
  std::uint64_t batched_queue_submit_calls{0};
  std::uint64_t batched_command_buffers{0};
};

class VulkanStream : public Stream {
 public:
  VulkanStream(VulkanDevice &device,
               VkQueue queue,
               uint32_t queue_family_index);
  ~VulkanStream() override;

  RhiResult new_command_list(CommandList **out_cmdlist) noexcept final;
  std::unique_ptr<CommandList> new_secondary_command_list() override;
  StreamSemaphore submit(
      CommandList *cmdlist,
      const std::vector<StreamSemaphore> &wait_semaphores = {}) override;
  // Backend-specific submission seam for external interop. The returned
  // semaphore remains the completion token owned by this stream; additional
  // signal semaphores are signaled by the same queue submission and retained
  // until its fence completes.
  StreamSemaphore submit_with_semaphores(
      CommandList *cmdlist,
      const std::vector<StreamSemaphore> &wait_semaphores,
      const std::vector<StreamSemaphore> &signal_semaphores);
  StreamSemaphore submit_synced(
      CommandList *cmdlist,
      const std::vector<StreamSemaphore> &wait_semaphores = {}) override;

  void begin_submission_batch() override;
  StreamSemaphore end_submission_batch() override;
  StreamGpuTiming begin_gpu_timing() override;
  void end_gpu_timing(const StreamGpuTiming &timing) override;
  StreamGpuTiming begin_gpu_timing_inline(CommandList *cmdlist) override;
  void end_gpu_timing_inline(const StreamGpuTiming &timing,
                             CommandList *cmdlist) override;

  void command_sync() override;

  std::size_t debug_in_flight_command_buffer_count();

 private:
  struct TrackedCmdbuf {
    vkapi::IVkFence fence;
    std::vector<vkapi::IVkCommandBuffer> buffers;
    std::vector<vkapi::IDeviceObj> submit_refs;
  };

  struct PendingBatchSubmission {
    vkapi::IVkCommandBuffer buffer;
    std::vector<VkSemaphore> wait_semaphores;
    std::vector<VkPipelineStageFlags> wait_stages;
    std::vector<VkSemaphore> signal_semaphores;
    std::vector<vkapi::IDeviceObj> submit_refs;
    std::vector<VulkanProfilerSampler> profiler_samplers;
  };

  void retire_completed_cmdbuffers();
  void apply_in_flight_backpressure();

  VulkanDevice &device_;
  VkQueue queue_;
  uint32_t queue_family_index_;
  std::uint64_t stream_id_{0};

  // Command pools are per-thread
  vkapi::IVkCommandPool command_pool_;
  std::mutex submission_mutex_;
  std::vector<TrackedCmdbuf> submitted_cmdbuffers_;
  std::size_t submission_batch_depth_{0};
  vkapi::IVkFence submission_batch_fence_;
  std::vector<PendingBatchSubmission> pending_batch_submissions_;
  StreamSemaphore submission_batch_completion_;
};

struct VulkanCooperativeMatrixProperty {
  std::uint32_t m{0};
  std::uint32_t n{0};
  std::uint32_t k{0};
  VkComponentTypeKHR a_type{VK_COMPONENT_TYPE_MAX_ENUM_KHR};
  VkComponentTypeKHR b_type{VK_COMPONENT_TYPE_MAX_ENUM_KHR};
  VkComponentTypeKHR c_type{VK_COMPONENT_TYPE_MAX_ENUM_KHR};
  VkComponentTypeKHR result_type{VK_COMPONENT_TYPE_MAX_ENUM_KHR};
  VkScopeKHR scope{VK_SCOPE_MAX_ENUM_KHR};
  bool saturating_accumulation{false};
};

struct VulkanCapabilities {
  uint32_t vk_api_version{0};
  // C-2.4.c: VkPhysicalDeviceLimits::maxPerStageDescriptorStorageBuffers,
  // 0 = not yet probed.
  uint32_t max_per_stage_descriptor_storage_buffers{0};
  uint32_t max_descriptor_set_storage_buffers{0};
  uint32_t max_per_stage_resources{0};
  bool physical_device_features2{false};
  bool external_memory{false};
  bool external_semaphore{false};
  bool wide_line{false};
  bool surface{false};
  bool present{false};
  bool dynamic_rendering{false};
  bool present_mode_fifo_latest_ready{false};
  bool descriptor_update_after_bind{false};
  bool descriptor_indexing{false};
  bool descriptor_storage_buffer_array_non_uniform_indexing{false};
  bool descriptor_storage_buffer_update_after_bind{false};
  bool descriptor_binding_partially_bound{false};
  bool descriptor_binding_variable_count{false};
  bool runtime_descriptor_array{false};
  bool descriptor_update_unused_while_pending{false};
  std::uint32_t max_update_after_bind_descriptors_in_all_pools{0};
  std::uint32_t
      max_per_stage_descriptor_update_after_bind_storage_buffers{0};
  std::uint32_t max_descriptor_set_update_after_bind_storage_buffers{0};
  bool conditional_rendering{false};
  bool buffer_device_address{false};
  bool acceleration_structure{false};
  bool ray_query{false};
  bool cooperative_matrix{false};
  bool multi_draw_indirect{false};
  bool draw_indirect_first_instance{false};
  bool draw_indirect_count{false};
  std::uint32_t max_draw_indirect_count{0};
  bool mesh_shader{false};
  bool task_shader{false};
  std::array<std::uint32_t, 3> max_task_work_group_count{0, 0, 0};
  std::uint32_t max_task_work_group_total_count{0};
  std::uint32_t max_task_work_group_invocations{0};
  std::array<std::uint32_t, 3> max_mesh_work_group_count{0, 0, 0};
  std::uint32_t max_mesh_work_group_total_count{0};
  std::uint32_t max_mesh_work_group_invocations{0};
  std::uint32_t max_mesh_output_vertices{0};
  std::uint32_t max_mesh_output_primitives{0};
  std::uint32_t subgroup_size{0};
  VkShaderStageFlags cooperative_matrix_supported_stages{0};
  std::vector<VulkanCooperativeMatrixProperty> cooperative_matrix_properties;
};

class TI_DLL_EXPORT VulkanDevice : public GraphicsDevice {
 public:
  using InteropAllocationReleaseCallback =
      void (*)(VulkanDevice *, DeviceAllocationId, uint64_t);
  using InteropDeviceReleaseCallback = void (*)(VulkanDevice *);

  struct Params {
    PFN_vkGetInstanceProcAddr get_proc_addr{nullptr};
    VkInstance instance{VK_NULL_HANDLE};
    VkPhysicalDevice physical_device{VK_NULL_HANDLE};
    VkDevice device{VK_NULL_HANDLE};
    VkQueue compute_queue{VK_NULL_HANDLE};
    uint32_t compute_queue_family_index{0};
    VkQueue graphics_queue{VK_NULL_HANDLE};
    uint32_t graphics_queue_family_index{0};
  };

  VulkanDevice();
  void init_vulkan_structs(Params &params);
  ~VulkanDevice() override;

  Arch arch() const override {
    return Arch::vulkan;
  }

  RhiResult create_pipeline_cache(
      PipelineCache **out_cache,
      size_t initial_size = 0,
      const void *initial_data = nullptr) noexcept final;

  RhiResult create_pipeline(Pipeline **out_pipeline,
                            const PipelineSourceDesc &src,
                            std::string name,
                            PipelineCache *cache) noexcept final;

  void set_default_pipeline_cache(PipelineCache *cache) noexcept;

  RhiResult allocate_memory(const AllocParams &params,
                            DeviceAllocation *out_devalloc) override;
  void dealloc_memory(DeviceAllocation handle) override;

  uint64_t get_memory_physical_pointer(DeviceAllocation handle) override;

  ShaderResourceSet *create_resource_set() final;

  RasterResources *create_raster_resources() final;

  RhiResult map_range(DevicePtr ptr, uint64_t size, void **mapped_ptr) final;
  RhiResult map(DeviceAllocation alloc, void **mapped_ptr) final;

  void unmap(DevicePtr ptr) final;
  void unmap(DeviceAllocation alloc) final;

  RhiResult upload_data(DevicePtr *device_ptr,
                        const void **data,
                        size_t *size,
                        int num_alloc = 1) noexcept final;

  RhiResult readback_data(
      DevicePtr *device_ptr,
      void **data,
      size_t *size,
      int num_alloc = 1,
      const std::vector<StreamSemaphore> &wait_sema = {}) noexcept final;

  // Strictly intra device copy
  void memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) override;

  Stream *get_compute_stream() override;
  Stream *get_graphics_stream() override;

  void wait_idle() override;

  std::pair<size_t, size_t> debug_stream_cache_counts();

  VulkanRuntimeTelemetrySnapshot runtime_telemetry_snapshot() const noexcept;
  VulkanQueueSubmissionSnapshot queue_submission_snapshot() const noexcept;

  BackendWaitTelemetry *backend_wait_telemetry() noexcept {
    return &backend_wait_telemetry_;
  }

  std::unique_ptr<Pipeline> create_raster_pipeline(
      const std::vector<PipelineSourceDesc> &src,
      const RasterParams &raster_params,
      const std::vector<VertexInputBinding> &vertex_inputs,
      const std::vector<VertexInputAttribute> &vertex_attrs,
      std::string name = "Pipeline") override;

  std::unique_ptr<Surface> create_surface(const SurfaceConfig &config) override;

  DeviceAllocation create_image(const ImageParams &params) override;
  void destroy_image(DeviceAllocation handle) override;

  // Vulkan specific functions
  VkInstance vk_instance() const {
    return instance_;
  }

  VkDevice vk_device() const {
    return device_;
  }

  VkPhysicalDevice vk_physical_device() const {
    return physical_device_;
  }

  uint32_t compute_queue_family_index() const {
    return compute_queue_family_index_;
  }

  uint32_t graphics_queue_family_index() const {
    return graphics_queue_family_index_;
  }

  bool image_blit_supported(BufferFormat source_format,
                            BufferFormat destination_format,
                            bool linear_filter) const;

  uint32_t queue_timestamp_valid_bits(uint32_t queue_family_index) const;

  VkQueue graphics_queue() const {
    return graphics_queue_;
  }

  VkQueue compute_queue() const {
    return compute_queue_;
  }

  // Cold provider initialization may submit LUT uploads on the shared queue.
  std::unique_lock<std::mutex> acquire_external_compute_queue_lock() {
    return acquire_queue_lock(compute_queue_);
  }

  std::tuple<VkDeviceMemory, size_t, size_t> get_vkmemory_offset_size(
      const DeviceAllocation &alloc) const;

  vkapi::IVkBuffer get_vkbuffer(const DeviceAllocation &alloc) const;

  size_t get_vkbuffer_size(const DeviceAllocation &alloc) const;

  uint64_t allocation_generation(DeviceAllocation handle) const {
    return get_alloc_internal(handle).generation;
  }

  AllocUsage allocation_usage(DeviceAllocation handle) const {
    return get_alloc_internal(handle).usage;
  }

  VkDeviceAddress get_buffer_device_address(DeviceAllocation handle) const;

  void set_interop_cleanup_callbacks(
      InteropAllocationReleaseCallback allocation_release,
      InteropDeviceReleaseCallback device_release) {
    std::lock_guard<std::mutex> lock(interop_cleanup_mutex_);
    TI_ASSERT(interop_allocation_release_ == nullptr ||
              interop_allocation_release_ == allocation_release);
    TI_ASSERT(interop_device_release_ == nullptr ||
              interop_device_release_ == device_release);
    interop_allocation_release_ = allocation_release;
    interop_device_release_ = device_release;
  }

  std::tuple<vkapi::IVkImage, vkapi::IVkImageView, VkFormat> get_vk_image(
      const DeviceAllocation &alloc) const;

  DeviceAllocation import_vkbuffer(vkapi::IVkBuffer buffer,
                                   size_t size,
                                   VkDeviceMemory memory,
                                   VkDeviceSize offset,
                                   AllocUsage usage = AllocUsage::Storage);

  DeviceAllocation import_vk_image(vkapi::IVkImage image,
                                   vkapi::IVkImageView view,
                                   VkImageLayout layout);

  vkapi::IVkImageView get_vk_imageview(const DeviceAllocation &alloc) const;

  vkapi::IVkImageView get_vk_lod_imageview(const DeviceAllocation &alloc,
                                           int lod) const;
  vkapi::IVkSampler get_sampler(const ImageSamplerConfig &config);
  vkapi::IVkSampler get_default_sampler();
  std::size_t image_sampler_cache_size() const {
    std::lock_guard<std::mutex> lock(descriptor_mutex_);
    return image_samplers_.size();
  }

  vkapi::IVkRenderPass get_renderpass(const VulkanRenderPassDesc &desc);

  vkapi::IVkFramebuffer get_framebuffer(const VulkanFramebufferDesc &desc);

  vkapi::IVkDescriptorSetLayout get_desc_set_layout(VulkanResourceSet &set);
  rhi_impl::RhiReturn<vkapi::IVkDescriptorSet> alloc_desc_set(
      vkapi::IVkDescriptorSetLayout layout);

  constexpr VulkanCapabilities &vk_caps() {
    return vk_caps_;
  }
  constexpr const VulkanCapabilities &vk_caps() const {
    return vk_caps_;
  }

  // C-2.4.c: expose maxPerStageDescriptorStorageBuffers via the generic Device
  // API so backend-agnostic code (e.g. ChunkedDeviceNodeAllocator) can probe
  // it without dynamic_cast.
  uint32_t get_max_storage_buffer_descriptors_per_binding()
      const noexcept override {
    auto v = vk_caps_.max_per_stage_descriptor_storage_buffers;
    return v == 0u ? UINT32_MAX : v;
  }
  bool supports_conditional_commands() const noexcept override {
    return vk_caps_.conditional_rendering;
  }

  void set_descriptor_set_cache_enabled(bool enabled) {
    std::lock_guard<std::mutex> lock(descriptor_mutex_);
    descriptor_set_cache_enabled_ = enabled;
    if (!enabled) {
      desc_set_cache_.clear();
      desc_set_cache_lru_.clear();
    }
  }

  void set_descriptor_set_cache_options(int capacity, bool lru) {
    std::lock_guard<std::mutex> lock(descriptor_mutex_);
    desc_set_cache_capacity_ = capacity > 0 ? static_cast<size_t>(capacity)
                                            : size_t{1024};
    if (descriptor_set_cache_lru_ != lru) {
      // Existing cache entries hold iterators into the current LRU list.
      // Rebuild lazily by dropping the CPU-side cache when the policy changes.
      desc_set_cache_.clear();
      desc_set_cache_lru_.clear();
    }
    descriptor_set_cache_lru_ = lru;
    while (desc_set_cache_.size() > desc_set_cache_capacity_) {
      if (desc_set_cache_lru_.empty()) {
        desc_set_cache_.clear();
        break;
      }
      desc_set_cache_.erase(desc_set_cache_lru_.front());
      desc_set_cache_lru_.pop_front();
    }
  }

  bool descriptor_set_cache_enabled() const {
    std::lock_guard<std::mutex> lock(descriptor_mutex_);
    return descriptor_set_cache_enabled_;
  }

  vkapi::IVkDescriptorSet find_cached_desc_set(const VulkanResourceSet &set);
  void cache_desc_set(const VulkanResourceSet &set,
                      vkapi::IVkDescriptorSet desc_set);
  size_t descriptor_set_cache_hits() const {
    std::lock_guard<std::mutex> lock(descriptor_mutex_);
    return desc_set_cache_hits_;
  }
  size_t descriptor_set_cache_misses() const {
    std::lock_guard<std::mutex> lock(descriptor_mutex_);
    return desc_set_cache_misses_;
  }
  size_t descriptor_set_cache_evictions() const {
    std::lock_guard<std::mutex> lock(descriptor_mutex_);
    return desc_set_cache_evictions_;
  }

  const VkPhysicalDeviceProperties &get_vk_physical_device_props() const {
    return vk_device_properties_;
  }

  // Profiler support
  void profiler_reserve_samplers(size_t count) {
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    const size_t retained =
        profiler_pending_sampler_count_ + sampled_records_.size();
    TI_ERROR_IF(count > profiler_record_capacity_ ||
                    retained > profiler_record_capacity_ - count,
                "Vulkan kernel profiler reached its {}-record memory budget. "
                "Synchronize and clear kernel profiler info periodically.",
                profiler_record_capacity_);
    profiler_pending_sampler_count_ += count;
  }

  void profiler_discard_reserved_samplers(size_t count) {
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    TI_ASSERT(profiler_pending_sampler_count_ >= count);
    profiler_pending_sampler_count_ -= count;
  }

  void profiler_add_samplers(std::vector<VulkanProfilerSampler> samplers) {
    if (samplers.empty()) {
      return;
    }
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    samplers_.insert(samplers_.end(),
                     std::make_move_iterator(samplers.begin()),
                     std::make_move_iterator(samplers.end()));
  }

  size_t profiler_get_sampler_count() override {
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    return profiler_pending_sampler_count_;
  }

  void profiler_sync() override;
  void profiler_sync_fences(const std::vector<vkapi::IVkFence> &fences);
  std::vector<std::pair<std::string, double>> profiler_flush_sampled_time()
      override;

 private:
  friend class VulkanResourceSet;
  friend class VulkanStream;
  friend VulkanSurface;

  std::unique_lock<std::mutex> acquire_queue_lock(VkQueue queue);
  void create_vma_allocator();
  [[nodiscard]] RhiResult new_descriptor_pool_locked();
  void update_descriptor_sets_locked(
      const std::vector<VkWriteDescriptorSet> &desc_writes);
  bool should_touch_desc_set_cache_lru_locked() const;

  VulkanCapabilities vk_caps_;
  VkPhysicalDeviceProperties vk_device_properties_;

  VkInstance instance_{VK_NULL_HANDLE};
  VkDevice device_{VK_NULL_HANDLE};
  VkPhysicalDevice physical_device_{VK_NULL_HANDLE};
  VmaAllocator allocator_{nullptr};
  VmaAllocator allocator_export_{nullptr};

  VkQueue compute_queue_{VK_NULL_HANDLE};
  uint32_t compute_queue_family_index_{0};
  std::mutex compute_queue_mutex_;
  VulkanQueueLockTelemetry compute_queue_lock_telemetry_;

  VkQueue graphics_queue_{VK_NULL_HANDLE};
  uint32_t graphics_queue_family_index_{0};
  std::mutex graphics_queue_mutex_;
  VulkanQueueLockTelemetry graphics_queue_lock_telemetry_;
  BackendWaitTelemetry backend_wait_telemetry_;
  std::atomic<std::uint64_t> queue_submit_calls_{0};
  std::atomic<std::uint64_t> submitted_command_buffers_{0};
  std::atomic<std::uint64_t> batched_queue_submit_calls_{0};
  std::atomic<std::uint64_t> batched_command_buffers_{0};

  struct ThreadLocalStreams;
  std::shared_ptr<ThreadLocalStreams> compute_streams_{nullptr};
  std::shared_ptr<ThreadLocalStreams> graphics_streams_{nullptr};

  // Memory allocation
  struct AllocationInternal {
    // Allocation info from VMA or set by `import_vkbuffer`
    VmaAllocationInfo alloc_info;
    // VkBuffer handle (reference counted)
    vkapi::IVkBuffer buffer{nullptr};
    // Buffer Device Address
    VkDeviceAddress addr{0};
    // If mapped, the currently mapped address
    void *mapped{nullptr};
    VkDeviceSize mapped_offset{0};
    VkDeviceSize mapped_size{VK_WHOLE_SIZE};
    bool host_read{false};
    bool host_write{false};
    AllocUsage usage{AllocUsage::None};
    // Is the allocation external (imported) or not (VMA)
    bool external{false};
    uint64_t generation{0};
  };

  // Images / Image views
  struct ImageAllocInternal {
    bool external{false};
    VmaAllocationInfo alloc_info{};
    vkapi::IVkImage image{nullptr};
    vkapi::IVkImageView view{nullptr};
    std::vector<vkapi::IVkImageView> view_lods{};
  };

  // Since we use the pointer to AllocationInternal as the `alloc_id`,
  // **pointer stability** is important.
  rhi_impl::SyncedPtrStableObjectList<AllocationInternal> allocations_;
  rhi_impl::SyncedPtrStableObjectList<ImageAllocInternal> image_allocations_;
  std::atomic<uint64_t> allocation_generation_counter_{1};
  std::mutex interop_cleanup_mutex_;
  InteropAllocationReleaseCallback interop_allocation_release_{nullptr};
  InteropDeviceReleaseCallback interop_device_release_{nullptr};

  // Renderpass
  unordered_map<VulkanRenderPassDesc,
                vkapi::IVkRenderPass,
                RenderPassDescHasher>
      renderpass_pools_;
  std::mutex renderpass_mutex_;

  // Descriptor layouts, cache metadata, and the default sampler are shared
  // device-level owner state. Allocation remains separately serialized by its
  // VkDescriptorPool, while each descriptor-set update is serialized by that
  // descriptor set itself.
  mutable std::mutex descriptor_mutex_;
  std::mutex descriptor_pool_mutex_;
  unordered_map<VulkanResourceSet,
                vkapi::IVkDescriptorSetLayout,
                VulkanResourceSet::SetLayoutHasher,
                VulkanResourceSet::SetLayoutCmp>
      desc_set_layouts_;

  struct CachedDescriptorSet {
    vkapi::IVkDescriptorSet set{nullptr};
    std::list<VulkanResourceSet>::iterator lru_it;
    bool has_lru_entry{false};
  };

  unordered_map<VulkanResourceSet,
                CachedDescriptorSet,
                VulkanResourceSet::DescSetHasher,
                VulkanResourceSet::SetCmp>
      desc_set_cache_;
  // LRU eviction removes only the cache's shared_ptr. Submitted command buffers
  // keep every bound IVkDescriptorSet in DeviceObjVkCommandBuffer::refs until
  // VulkanStream::command_sync() retires them, and descriptor sets are owned by
  // the descriptor pool for the device lifetime. Therefore cache eviction does
  // not require a fence wait; ctx buffer reuse is handled separately in
  // GfxRuntime's pooled/ring buffer lifecycle.
  std::list<VulkanResourceSet> desc_set_cache_lru_;
  size_t desc_set_cache_capacity_{1024};
  bool descriptor_set_cache_lru_{true};
  size_t desc_set_cache_hits_{0};
  size_t desc_set_cache_misses_{0};
  size_t desc_set_cache_evictions_{0};
  vkapi::IVkDescriptorPool desc_pool_{nullptr};
  bool descriptor_set_cache_enabled_{false};
  std::vector<std::pair<ImageSamplerConfig, vkapi::IVkSampler>>
      image_samplers_;
  PipelineCache *default_pipeline_cache_{nullptr};

  // Internal implementaion functions
  inline static AllocationInternal &get_alloc_internal(
      const DeviceAllocation &alloc) {
    return *reinterpret_cast<AllocationInternal *>(alloc.alloc_id);
  }

  inline static ImageAllocInternal &get_image_alloc_internal(
      const DeviceAllocation &alloc) {
    return *reinterpret_cast<ImageAllocInternal *>(alloc.alloc_id);
  }

  RhiResult map_internal(AllocationInternal &alloc_int,
                         size_t offset,
                         size_t size,
                         void **mapped_ptr);

  void profiler_collect_samplers(std::vector<VulkanProfilerSampler> samplers);

  // Profiler support
  std::mutex profiler_mutex_;
  size_t profiler_pending_sampler_count_{0};
  size_t profiler_record_capacity_{131072};
  static constexpr size_t kMaximumProfilerRecordCapacity = 1048576;

  std::vector<VulkanProfilerSampler> samplers_;
  std::vector<std::pair<std::string, double>> sampled_records_;
};

}  // namespace vulkan
}  // namespace taichi::lang
