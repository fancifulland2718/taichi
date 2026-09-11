#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <list>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "taichi/rhi/vulkan/vulkan_common.h"
#include "taichi/rhi/vulkan/vulkan_utils.h"
#include "taichi/util/environ_config.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/rhi/vulkan/vulkan_device.h"

#include "spirv_reflect.h"

namespace taichi::lang {
namespace vulkan {

using namespace rhi_impl;

const BidirMap<BufferFormat, VkFormat> buffer_format_map = {
    {BufferFormat::r8, VK_FORMAT_R8_UNORM},
    {BufferFormat::rg8, VK_FORMAT_R8G8_UNORM},
    {BufferFormat::rgba8, VK_FORMAT_R8G8B8A8_UNORM},
    {BufferFormat::rgba8srgb, VK_FORMAT_R8G8B8A8_SRGB},
    {BufferFormat::bgra8, VK_FORMAT_B8G8R8A8_UNORM},
    {BufferFormat::bgra8srgb, VK_FORMAT_B8G8R8A8_SRGB},
    {BufferFormat::r8u, VK_FORMAT_R8_UINT},
    {BufferFormat::rg8u, VK_FORMAT_R8G8_UINT},
    {BufferFormat::rgba8u, VK_FORMAT_R8G8B8A8_UINT},
    {BufferFormat::r8i, VK_FORMAT_R8_SINT},
    {BufferFormat::rg8i, VK_FORMAT_R8G8_SINT},
    {BufferFormat::rgba8i, VK_FORMAT_R8G8B8A8_SINT},
    {BufferFormat::r16, VK_FORMAT_R16_UNORM},
    {BufferFormat::rg16, VK_FORMAT_R16G16_UNORM},
    {BufferFormat::rgb16, VK_FORMAT_R16G16B16_UNORM},
    {BufferFormat::rgba16, VK_FORMAT_R16G16B16A16_UNORM},
    {BufferFormat::r16u, VK_FORMAT_R16_UINT},
    {BufferFormat::rg16u, VK_FORMAT_R16G16_UINT},
    {BufferFormat::rgb16u, VK_FORMAT_R16G16B16_UINT},
    {BufferFormat::rgba16u, VK_FORMAT_R16G16B16A16_UINT},
    {BufferFormat::r16i, VK_FORMAT_R16_SINT},
    {BufferFormat::rg16i, VK_FORMAT_R16G16_SINT},
    {BufferFormat::rgb16i, VK_FORMAT_R16G16B16_SINT},
    {BufferFormat::rgba16i, VK_FORMAT_R16G16B16A16_SINT},
    {BufferFormat::r16f, VK_FORMAT_R16_SFLOAT},
    {BufferFormat::rg16f, VK_FORMAT_R16G16_SFLOAT},
    {BufferFormat::rgb16f, VK_FORMAT_R16G16B16_SFLOAT},
    {BufferFormat::rgba16f, VK_FORMAT_R16G16B16A16_SFLOAT},
    {BufferFormat::r32u, VK_FORMAT_R32_UINT},
    {BufferFormat::rg32u, VK_FORMAT_R32G32_UINT},
    {BufferFormat::rgb32u, VK_FORMAT_R32G32B32_UINT},
    {BufferFormat::rgba32u, VK_FORMAT_R32G32B32A32_UINT},
    {BufferFormat::r32i, VK_FORMAT_R32_SINT},
    {BufferFormat::rg32i, VK_FORMAT_R32G32_SINT},
    {BufferFormat::rgb32i, VK_FORMAT_R32G32B32_SINT},
    {BufferFormat::rgba32i, VK_FORMAT_R32G32B32A32_SINT},
    {BufferFormat::r32f, VK_FORMAT_R32_SFLOAT},
    {BufferFormat::rg32f, VK_FORMAT_R32G32_SFLOAT},
    {BufferFormat::rgb32f, VK_FORMAT_R32G32B32_SFLOAT},
    {BufferFormat::rgba32f, VK_FORMAT_R32G32B32A32_SFLOAT},
    {BufferFormat::depth16, VK_FORMAT_D16_UNORM},
    {BufferFormat::depth24stencil8, VK_FORMAT_D24_UNORM_S8_UINT},
    {BufferFormat::depth32f, VK_FORMAT_D32_SFLOAT}};

RhiReturn<VkFormat> buffer_format_ti_to_vk(BufferFormat f) {
  if (!buffer_format_map.exists(f)) {
    RHI_LOG_ERROR("BufferFormat cannot be mapped to vk");
    return {RhiResult::not_supported, VK_FORMAT_UNDEFINED};
  }
  return {RhiResult::success, buffer_format_map.at(f)};
}

RhiReturn<BufferFormat> buffer_format_vk_to_ti(VkFormat f) {
  if (!buffer_format_map.exists(f)) {
    RHI_LOG_ERROR("VkFormat cannot be mapped to ti");
    return {RhiResult::not_supported, BufferFormat::unknown};
  }
  return {RhiResult::success, buffer_format_map.backend2rhi.at(f)};
}

const BidirMap<ImageLayout, VkImageLayout> image_layout_map = {
    {ImageLayout::undefined, VK_IMAGE_LAYOUT_UNDEFINED},
    {ImageLayout::shader_read, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
    {ImageLayout::shader_write, VK_IMAGE_LAYOUT_GENERAL},
    {ImageLayout::shader_read_write, VK_IMAGE_LAYOUT_GENERAL},
    {ImageLayout::color_attachment, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
    {ImageLayout::color_attachment_read,
     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
    {ImageLayout::depth_attachment,
     VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL},
    {ImageLayout::depth_attachment_read,
     VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL},
    {ImageLayout::transfer_dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL},
    {ImageLayout::transfer_src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL},
    {ImageLayout::present_src, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR}};

VkImageLayout image_layout_ti_to_vk(ImageLayout layout) {
  if (!image_layout_map.exists(layout)) {
    RHI_LOG_ERROR("ImageLayout cannot be mapped to vk");
    return VK_IMAGE_LAYOUT_UNDEFINED;
  }
  return image_layout_map.at(layout);
}

const BidirMap<BlendOp, VkBlendOp> blend_op_map = {
    {BlendOp::add, VK_BLEND_OP_ADD},
    {BlendOp::subtract, VK_BLEND_OP_SUBTRACT},
    {BlendOp::reverse_subtract, VK_BLEND_OP_REVERSE_SUBTRACT},
    {BlendOp::min, VK_BLEND_OP_MIN},
    {BlendOp::max, VK_BLEND_OP_MAX}};

RhiReturn<VkBlendOp> blend_op_ti_to_vk(BlendOp op) {
  if (!blend_op_map.exists(op)) {
    RHI_LOG_ERROR("BlendOp cannot be mapped to vk");
    return {RhiResult::not_supported, VK_BLEND_OP_ADD};
  }
  return {RhiResult::success, blend_op_map.at(op)};
}

const BidirMap<BlendFactor, VkBlendFactor> blend_factor_map = {
    {BlendFactor::zero, VK_BLEND_FACTOR_ZERO},
    {BlendFactor::one, VK_BLEND_FACTOR_ONE},
    {BlendFactor::src_color, VK_BLEND_FACTOR_SRC_COLOR},
    {BlendFactor::one_minus_src_color, VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR},
    {BlendFactor::dst_color, VK_BLEND_FACTOR_DST_COLOR},
    {BlendFactor::one_minus_dst_color, VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR},
    {BlendFactor::src_alpha, VK_BLEND_FACTOR_SRC_ALPHA},
    {BlendFactor::one_minus_src_alpha, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA},
    {BlendFactor::dst_alpha, VK_BLEND_FACTOR_DST_ALPHA},
    {BlendFactor::one_minus_dst_alpha, VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA},
};

RhiReturn<VkBlendFactor> blend_factor_ti_to_vk(BlendFactor factor) {
  if (!blend_factor_map.exists(factor)) {
    RHI_LOG_ERROR("BlendFactor cannot be mapped to vk");
    return {RhiResult::not_supported, VK_BLEND_FACTOR_ONE};
  }
  return {RhiResult::success, blend_factor_map.at(factor)};
}

VulkanPipelineCache::VulkanPipelineCache(VulkanDevice *device,
                                         size_t initial_size,
                                         const void *initial_data)
    : device_(device) {
  cache_ = vkapi::create_pipeline_cache(device_->vk_device(), 0, initial_size,
                                        initial_data);
}

VulkanPipelineCache ::~VulkanPipelineCache() {
}

void *VulkanPipelineCache::data() noexcept {
  std::lock_guard<std::mutex> data_lock(mutex_);
  if (!cache_) {
    return nullptr;
  }

  std::lock_guard<std::mutex> cache_lock(cache_->mutex);

  try {
    constexpr int kMaxSnapshotAttempts = 3;
    for (int attempt = 0; attempt < kMaxSnapshotAttempts; ++attempt) {
      size_t queried_size = 0;
      VkResult result = vkGetPipelineCacheData(device_->vk_device(),
                                                cache_->cache, &queried_size,
                                                nullptr);
      if (result != VK_SUCCESS) {
        char message[128];
        std::snprintf(message, sizeof(message),
                      "failed to query pipeline cache size: VkResult %d",
                      static_cast<int>(result));
        RHI_LOG_ERROR(message);
        data_shadow_.clear();
        return nullptr;
      }
      if (queried_size == 0) {
        data_shadow_.clear();
        return nullptr;
      }

      std::vector<uint8_t> snapshot(queried_size);
      size_t written_size = snapshot.size();
      result = vkGetPipelineCacheData(device_->vk_device(), cache_->cache,
                                      &written_size, snapshot.data());
      if (result == VK_SUCCESS) {
        snapshot.resize(written_size);
        data_shadow_.swap(snapshot);
        return data_shadow_.empty() ? nullptr : data_shadow_.data();
      }
      if (result != VK_INCOMPLETE) {
        char message[128];
        std::snprintf(message, sizeof(message),
                      "failed to read pipeline cache data: VkResult %d",
                      static_cast<int>(result));
        RHI_LOG_ERROR(message);
        data_shadow_.clear();
        return nullptr;
      }
    }
  } catch (std::bad_alloc &) {
    RHI_LOG_ERROR("out of memory while snapshotting pipeline cache");
  }

  RHI_LOG_ERROR("pipeline cache snapshot did not stabilize");
  data_shadow_.clear();
  return nullptr;
}

size_t VulkanPipelineCache::size() const noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  return data_shadow_.size();
}

VulkanPipeline::VulkanPipeline(const Params &params)
    : ti_device_(*params.device),
      device_(params.device->vk_device()),
      name_(params.name),
      cache_(params.cache) {
  create_descriptor_set_layout(params);
  create_shader_stages(params);
  create_pipeline_layout();
  create_compute_pipeline(params);

  for (VkShaderModule shader_module : shader_modules_) {
    vkDestroyShaderModule(device_, shader_module, kNoVkAllocCallbacks);
  }
  shader_modules_.clear();
}

VulkanPipeline::VulkanPipeline(
    const Params &params,
    const RasterParams &raster_params,
    const std::vector<VertexInputBinding> &vertex_inputs,
    const std::vector<VertexInputAttribute> &vertex_attrs)
    : ti_device_(*params.device),
      device_(params.device->vk_device()),
      name_(params.name),
      cache_(params.cache) {
  this->graphics_pipeline_template_ =
      std::make_unique<GraphicsPipelineTemplate>();

  create_descriptor_set_layout(params);
  create_shader_stages(params);
  create_pipeline_layout();
  create_graphics_pipeline(raster_params, vertex_inputs, vertex_attrs);
}

VulkanPipeline::~VulkanPipeline() {
  for (VkShaderModule shader_module : shader_modules_) {
    vkDestroyShaderModule(device_, shader_module, kNoVkAllocCallbacks);
  }
  shader_modules_.clear();
}

VkShaderModule VulkanPipeline::create_shader_module(VkDevice device,
                                                    const SpirvCodeView &code) {
  VkShaderModuleCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.codeSize = code.size;
  create_info.pCode = code.data;

  VkShaderModule shader_module;
  VkResult res = vkCreateShaderModule(device, &create_info, kNoVkAllocCallbacks,
                                      &shader_module);
  RHI_THROW_UNLESS(res == VK_SUCCESS,
                   std::runtime_error("vkCreateShaderModule failed"));
  return shader_module;
}

vkapi::IVkPipeline VulkanPipeline::graphics_pipeline(
    const VulkanRenderPassDesc &renderpass_desc,
    vkapi::IVkRenderPass renderpass) {
  std::lock_guard<std::mutex> lock(graphics_pipeline_mutex_);
  if (graphics_pipeline_.find(renderpass) != graphics_pipeline_.end()) {
    return graphics_pipeline_.at(renderpass);
  }

  vkapi::IVkPipeline pipeline = vkapi::create_graphics_pipeline(
      device_, &graphics_pipeline_template_->pipeline_info, renderpass,
      pipeline_layout_, cache_);

  graphics_pipeline_[renderpass] = pipeline;

  return pipeline;
}

vkapi::IVkPipeline VulkanPipeline::graphics_pipeline_dynamic(
    const VulkanRenderPassDesc &renderpass_desc) {
  std::lock_guard<std::mutex> lock(graphics_pipeline_mutex_);
  if (graphics_pipeline_dynamic_.find(renderpass_desc) !=
      graphics_pipeline_dynamic_.end()) {
    return graphics_pipeline_dynamic_.at(renderpass_desc);
  }

  std::vector<VkFormat> color_attachment_formats;
  for (const auto &color_attachment : renderpass_desc.color_attachments) {
    color_attachment_formats.push_back(color_attachment.first);
  }

  VkPipelineRenderingCreateInfoKHR rendering_info{};
  rendering_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
  rendering_info.pNext = nullptr;
  rendering_info.viewMask = 0;
  rendering_info.colorAttachmentCount =
      renderpass_desc.color_attachments.size();
  rendering_info.pColorAttachmentFormats = color_attachment_formats.data();
  rendering_info.depthAttachmentFormat = renderpass_desc.depth_attachment;
  rendering_info.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

  vkapi::IVkPipeline pipeline = vkapi::create_graphics_pipeline_dynamic(
      device_, &graphics_pipeline_template_->pipeline_info, &rendering_info,
      pipeline_layout_, cache_);

  graphics_pipeline_dynamic_[renderpass_desc] = pipeline;

  return pipeline;
}

void VulkanPipeline::create_descriptor_set_layout(const Params &params) {
  for (auto &code_view : params.code) {
    SpvReflectShaderModule module;
    SpvReflectResult result =
        spvReflectCreateShaderModule(code_view.size, code_view.data, &module);
    RHI_THROW_UNLESS(result == SPV_REFLECT_RESULT_SUCCESS,
                     std::runtime_error("spvReflectCreateShaderModule failed"));

    uint32_t set_count = 0;
    result = spvReflectEnumerateDescriptorSets(&module, &set_count, nullptr);
    RHI_THROW_UNLESS(result == SPV_REFLECT_RESULT_SUCCESS,
                     std::runtime_error("Failed to enumerate number of sets"));
    std::vector<SpvReflectDescriptorSet *> desc_sets(set_count);
    result = spvReflectEnumerateDescriptorSets(&module, &set_count,
                                               desc_sets.data());
    RHI_THROW_UNLESS(
        result == SPV_REFLECT_RESULT_SUCCESS,
        std::runtime_error("spvReflectEnumerateDescriptorSets failed"));

    for (SpvReflectDescriptorSet *desc_set : desc_sets) {
      uint32_t set_index = desc_set->set;
      if (set_templates_.find(set_index) == set_templates_.end()) {
        set_templates_.insert({set_index, VulkanResourceSet(&ti_device_)});
      }
      VulkanResourceSet &set = set_templates_.at(set_index);

      for (int i = 0; i < desc_set->binding_count; i++) {
        SpvReflectDescriptorBinding *desc_binding = desc_set->bindings[i];

        if (desc_binding->descriptor_type ==
            SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
          RHI_THROW_UNLESS(
              desc_binding->count > 0,
              std::invalid_argument(
                  "Runtime-sized storage-buffer descriptor arrays require "
                  "an explicit variable-count provider"));
          if (desc_binding->count > 1) {
            set.rw_buffer_array(
                desc_binding->binding,
                std::vector<DeviceAllocation>(desc_binding->count,
                                              kDeviceNullAllocation));
          } else {
            set.rw_buffer(desc_binding->binding, kDeviceNullPtr, 0);
          }
        } else if (desc_binding->descriptor_type ==
                   SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER) {
          set.buffer(desc_binding->binding, kDeviceNullPtr, 0);
        } else if (desc_binding->descriptor_type ==
                   SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
          set.image(desc_binding->binding, kDeviceNullAllocation, {});
        } else if (desc_binding->descriptor_type ==
                   SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
          set.rw_image(desc_binding->binding, kDeviceNullAllocation, {});
        } else if (desc_binding->descriptor_type ==
                   SPV_REFLECT_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR) {
          set.acceleration_structure(desc_binding->binding, nullptr);
        } else {
          RHI_LOG_ERROR("Unrecognized binding ignored");
        }
      }
    }

    // Handle special vertex shaders stuff
    // if (code_view.stage == VK_SHADER_STAGE_VERTEX_BIT) {
    //   uint32_t attrib_count;
    //   result =
    //       spvReflectEnumerateInputVariables(&module, &attrib_count, nullptr);
    //   RHI_ASSERT(result == SPV_REFLECT_RESULT_SUCCESS);
    //   std::vector<SpvReflectInterfaceVariable *> attribs(attrib_count);
    //   result = spvReflectEnumerateInputVariables(&module, &attrib_count,
    //                                               attribs.data());
    //   RHI_ASSERT(result == SPV_REFLECT_RESULT_SUCCESS);

    //   for (SpvReflectInterfaceVariable *attrib : attribs) {
    //     uint32_t location = attrib->location;
    //     SpvReflectTypeDescription *type = attrib->type_description;
    //     TI_WARN("attrib {}:{}", location, type->type_name);
    //   }
    // }

    if (code_view.stage == VK_SHADER_STAGE_FRAGMENT_BIT) {
      uint32_t render_target_count = 0;
      result = spvReflectEnumerateOutputVariables(&module, &render_target_count,
                                                  nullptr);
      RHI_THROW_UNLESS(
          result == SPV_REFLECT_RESULT_SUCCESS,
          std::runtime_error("Failed to enumerate number of output vars"));

      std::vector<SpvReflectInterfaceVariable *> variables(render_target_count);
      result = spvReflectEnumerateOutputVariables(&module, &render_target_count,
                                                  variables.data());

      RHI_THROW_UNLESS(
          result == SPV_REFLECT_RESULT_SUCCESS,
          std::runtime_error("spvReflectEnumerateOutputVariables failed"));

      render_target_count = 0;

      for (auto var : variables) {
        // We want to remove auxiliary outputs such as frag depth
        if (static_cast<int>(var->built_in) == -1) {
          render_target_count++;
        }
      }

      graphics_pipeline_template_->blend_attachments.resize(
          render_target_count);

      VkPipelineColorBlendAttachmentState default_state{};
      default_state.colorWriteMask =
          VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
      default_state.blendEnable = VK_FALSE;

      std::fill(graphics_pipeline_template_->blend_attachments.begin(),
                graphics_pipeline_template_->blend_attachments.end(),
                default_state);
    }
    spvReflectDestroyShaderModule(&module);
  }

  // A program can have no binding sets at all.
  if (set_templates_.size()) {
    // We need to verify the set layouts are all continous
    uint32_t max_set = 0;
    for (auto &[index, layout_template] : set_templates_) {
      max_set = std::max(index, max_set);
    }
    RHI_THROW_UNLESS(
        max_set + 1 == set_templates_.size(),
        std::invalid_argument("Sets must be continous & start with 0"));

    set_layouts_.resize(set_templates_.size(), nullptr);
    for (auto &[index, layout_template] : set_templates_) {
      set_layouts_[index] = ti_device_.get_desc_set_layout(layout_template);
    }
  }
}

void VulkanPipeline::create_shader_stages(const Params &params) {
  for (auto &code_view : params.code) {
    VkPipelineShaderStageCreateInfo &shader_stage_info =
        shader_stages_.emplace_back();

    VkShaderModule shader_module = create_shader_module(device_, code_view);

    shader_stage_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stage_info.stage = code_view.stage;
    shader_stage_info.module = shader_module;
    shader_stage_info.pName = "main";

    shader_modules_.push_back(shader_module);
  }
}

void VulkanPipeline::create_pipeline_layout() {
  VkPushConstantRange push_constant_range{};
  push_constant_range.stageFlags = VK_SHADER_STAGE_ALL;
  push_constant_range.offset = 0;
  push_constant_range.size = 128;
  pipeline_layout_ =
      vkapi::create_pipeline_layout(device_, set_layouts_, 1,
                                    &push_constant_range);
}

void VulkanPipeline::create_compute_pipeline(const Params &params) {
  std::array<char, 512> msg_buf;
  RHI_DEBUG_SNPRINTF(msg_buf.data(), msg_buf.size(),
                     "Compiling Vulkan pipeline %s", params.name.data());
  RHI_LOG_DEBUG(msg_buf.data());
  pipeline_ = vkapi::create_compute_pipeline(device_, 0, shader_stages_[0],
                                             pipeline_layout_, params.cache);
}

void VulkanPipeline::create_graphics_pipeline(
    const RasterParams &raster_params,
    const std::vector<VertexInputBinding> &vertex_inputs,
    const std::vector<VertexInputAttribute> &vertex_attrs) {
  const bool mesh_pipeline = std::any_of(
      shader_stages_.begin(), shader_stages_.end(),
      [](const VkPipelineShaderStageCreateInfo &stage) {
        return stage.stage == VK_SHADER_STAGE_MESH_BIT_EXT;
      });
  // Use dynamic viewport state. These two are just dummies
  VkViewport viewport{};
  viewport.width = 1;
  viewport.height = 1;
  viewport.x = 0;
  viewport.y = 0;
  viewport.minDepth = 0.0;
  viewport.maxDepth = 1.0;

  VkRect2D scissor{/*offset*/ {0, 0}, /*extent*/ {1, 1}};

  VkPipelineViewportStateCreateInfo &viewport_state =
      graphics_pipeline_template_->viewport_state;
  viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewport_state.viewportCount = 1;
  viewport_state.pViewports = &viewport;
  viewport_state.scissorCount = 1;
  viewport_state.pScissors = &scissor;

  for (const VertexInputBinding &binding : vertex_inputs) {
    VkVertexInputBindingDescription &desc =
        graphics_pipeline_template_->input_bindings.emplace_back();
    desc.binding = binding.binding;
    desc.stride = binding.stride;
    desc.inputRate = binding.instance ? VK_VERTEX_INPUT_RATE_INSTANCE
                                      : VK_VERTEX_INPUT_RATE_VERTEX;
  }

  for (const VertexInputAttribute &attr : vertex_attrs) {
    VkVertexInputAttributeDescription &desc =
        graphics_pipeline_template_->input_attrs.emplace_back();
    desc.binding = attr.binding;
    desc.location = attr.location;
    auto [result, vk_format] = buffer_format_ti_to_vk(attr.format);
    RHI_ASSERT(result == RhiResult::success);
    desc.format = vk_format;
    assert(desc.format != VK_FORMAT_UNDEFINED);
    desc.offset = attr.offset;
  }

  VkPipelineVertexInputStateCreateInfo &vertex_input =
      graphics_pipeline_template_->input;
  vertex_input.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertex_input.pNext = nullptr;
  vertex_input.flags = 0;
  vertex_input.vertexBindingDescriptionCount =
      graphics_pipeline_template_->input_bindings.size();
  vertex_input.pVertexBindingDescriptions =
      graphics_pipeline_template_->input_bindings.data();
  vertex_input.vertexAttributeDescriptionCount =
      graphics_pipeline_template_->input_attrs.size();
  vertex_input.pVertexAttributeDescriptions =
      graphics_pipeline_template_->input_attrs.data();

  VkPipelineInputAssemblyStateCreateInfo &input_assembly =
      graphics_pipeline_template_->input_assembly;
  input_assembly.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  static const std::unordered_map<TopologyType, VkPrimitiveTopology>
      topo_types = {
          {TopologyType::Triangles, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST},
          {TopologyType::Lines, VK_PRIMITIVE_TOPOLOGY_LINE_LIST},
          {TopologyType::Points, VK_PRIMITIVE_TOPOLOGY_POINT_LIST},
      };
  input_assembly.topology = topo_types.at(raster_params.prim_topology);
  input_assembly.primitiveRestartEnable = VK_FALSE;

  static const std::unordered_map<PolygonMode, VkPolygonMode> polygon_modes = {
      {PolygonMode::Fill, VK_POLYGON_MODE_FILL},
      {PolygonMode::Line, VK_POLYGON_MODE_LINE},
      {PolygonMode::Point, VK_POLYGON_MODE_POINT},
  };

  VkPipelineRasterizationStateCreateInfo &rasterizer =
      graphics_pipeline_template_->rasterizer;
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = polygon_modes.at(raster_params.polygon_mode);
  rasterizer.lineWidth = 1.0f;
  rasterizer.cullMode = 0;
  if (raster_params.front_face_cull) {
    rasterizer.cullMode |= VK_CULL_MODE_FRONT_BIT;
  }
  if (raster_params.back_face_cull) {
    rasterizer.cullMode |= VK_CULL_MODE_BACK_BIT;
  }
  rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterizer.depthBiasEnable = VK_FALSE;

  VkPipelineMultisampleStateCreateInfo &multisampling =
      graphics_pipeline_template_->multisampling;
  multisampling.sType =
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineDepthStencilStateCreateInfo &depth_stencil =
      graphics_pipeline_template_->depth_stencil;
  depth_stencil.sType =
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depth_stencil.depthTestEnable = raster_params.depth_test;
  depth_stencil.depthWriteEnable = raster_params.depth_write;
  depth_stencil.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
  depth_stencil.depthBoundsTestEnable = VK_FALSE;
  depth_stencil.stencilTestEnable = VK_FALSE;

  VkPipelineColorBlendStateCreateInfo &color_blending =
      graphics_pipeline_template_->color_blending;
  color_blending.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  color_blending.logicOpEnable = VK_FALSE;
  color_blending.logicOp = VK_LOGIC_OP_COPY;
  color_blending.attachmentCount =
      graphics_pipeline_template_->blend_attachments.size();
  color_blending.pAttachments =
      graphics_pipeline_template_->blend_attachments.data();
  color_blending.blendConstants[0] = 0.0f;
  color_blending.blendConstants[1] = 0.0f;
  color_blending.blendConstants[2] = 0.0f;
  color_blending.blendConstants[3] = 0.0f;

  if (raster_params.blending.size()) {
    if (raster_params.blending.size() != color_blending.attachmentCount) {
      std::array<char, 256> buf;
      RHI_DEBUG_SNPRINTF(
          buf.data(), buf.size(),
          "RasterParams::blending (size=%u) must either be zero sized "
          "or match the number of fragment shader outputs (size=%u).",
          uint32_t(raster_params.blending.size()),
          uint32_t(color_blending.attachmentCount));
      RHI_LOG_ERROR(buf.data());
      RHI_ASSERT(false);
    }

    for (int i = 0; i < raster_params.blending.size(); i++) {
      auto &state = graphics_pipeline_template_->blend_attachments[i];
      auto &ti_param = raster_params.blending[i];
      state.blendEnable = ti_param.enable;
      if (ti_param.enable) {
        {
          auto [res, op] = blend_op_ti_to_vk(ti_param.color.op);
          RHI_ASSERT(res == RhiResult::success);
          state.colorBlendOp = op;
        }
        {
          auto [res, factor] = blend_factor_ti_to_vk(ti_param.color.src_factor);
          RHI_ASSERT(res == RhiResult::success);
          state.srcColorBlendFactor = factor;
        }
        {
          auto [res, factor] = blend_factor_ti_to_vk(ti_param.color.dst_factor);
          RHI_ASSERT(res == RhiResult::success);
          state.dstColorBlendFactor = factor;
        }
        {
          auto [res, op] = blend_op_ti_to_vk(ti_param.alpha.op);
          RHI_ASSERT(res == RhiResult::success);
          state.alphaBlendOp = op;
        }
        {
          auto [res, factor] = blend_factor_ti_to_vk(ti_param.alpha.src_factor);
          RHI_ASSERT(res == RhiResult::success);
          state.srcAlphaBlendFactor = factor;
        }
        {
          auto [res, factor] = blend_factor_ti_to_vk(ti_param.alpha.dst_factor);
          RHI_ASSERT(res == RhiResult::success);
          state.dstAlphaBlendFactor = factor;
        }
        state.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
      }
    }
  }

  VkPipelineDynamicStateCreateInfo &dynamic_state =
      graphics_pipeline_template_->dynamic_state;
  dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamic_state.pNext = nullptr;
  dynamic_state.pDynamicStates =
      graphics_pipeline_template_->dynamic_state_enables.data();
  dynamic_state.dynamicStateCount =
      graphics_pipeline_template_->dynamic_state_enables.size();

  VkGraphicsPipelineCreateInfo &pipeline_info =
      graphics_pipeline_template_->pipeline_info;
  pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipeline_info.stageCount = shader_stages_.size();
  pipeline_info.pStages = shader_stages_.data();
  pipeline_info.pVertexInputState = mesh_pipeline ? nullptr : &vertex_input;
  pipeline_info.pInputAssemblyState =
      mesh_pipeline ? nullptr : &input_assembly;
  pipeline_info.pViewportState = &viewport_state;
  pipeline_info.pRasterizationState = &rasterizer;
  pipeline_info.pMultisampleState = &multisampling;
  pipeline_info.pDepthStencilState = &depth_stencil;
  pipeline_info.pColorBlendState = &color_blending;
  pipeline_info.pDynamicState = &dynamic_state;
  pipeline_info.renderPass = VK_NULL_HANDLE;  // Filled in later
  pipeline_info.subpass = 0;
  pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
}

VulkanResourceSet::VulkanResourceSet(VulkanDevice *device) : device_(device) {
}

VulkanResourceSet::~VulkanResourceSet() {
}

void VulkanResourceSet::set_binding(uint32_t binding, Binding new_binding) {
  auto it = bindings_.find(binding);
  if (it != bindings_.end() && it->second == new_binding) {
    return;
  }
  bindings_[binding] = std::move(new_binding);
  dirty_ = true;
}

ShaderResourceSet &VulkanResourceSet::rw_buffer(uint32_t binding,
                                                DevicePtr ptr,
                                                size_t size) {
  vkapi::IVkBuffer buffer =
      (ptr != kDeviceNullPtr) ? device_->get_vkbuffer(ptr) : nullptr;
  set_binding(binding, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        Buffer{buffer, ptr.offset, size}});
  return *this;
}

ShaderResourceSet &VulkanResourceSet::rw_buffer(uint32_t binding,
                                                DeviceAllocation alloc) {
  return rw_buffer(binding, alloc.get_ptr(0), VK_WHOLE_SIZE);
}

ShaderResourceSet &VulkanResourceSet::buffer(uint32_t binding,
                                             DevicePtr ptr,
                                             size_t size) {
  vkapi::IVkBuffer buffer =
      (ptr != kDeviceNullPtr) ? device_->get_vkbuffer(ptr) : nullptr;
  set_binding(binding, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                        Buffer{buffer, ptr.offset, size}});
  return *this;
}

ShaderResourceSet &VulkanResourceSet::buffer(uint32_t binding,
                                             DeviceAllocation alloc) {
  return buffer(binding, alloc.get_ptr(0), VK_WHOLE_SIZE);
}

ShaderResourceSet &VulkanResourceSet::image(uint32_t binding,
                                            DeviceAllocation alloc,
                                            ImageSamplerConfig sampler_config) {
  vkapi::IVkSampler sampler = nullptr;
  vkapi::IVkImageView view = nullptr;

  if (alloc != kDeviceNullAllocation) {
    sampler = device_->get_sampler(sampler_config);
    view = device_->get_vk_imageview(alloc);
  }

  set_binding(binding, {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        Texture{view, sampler}});

  return *this;
}

ShaderResourceSet &VulkanResourceSet::rw_image(uint32_t binding,
                                               DeviceAllocation alloc,
                                               int lod) {
  vkapi::IVkImageView view = (alloc != kDeviceNullAllocation)
                                 ? device_->get_vk_lod_imageview(alloc, lod)
                                 : nullptr;

  set_binding(binding, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, Image{view}});

  return *this;
}

ShaderResourceSet &VulkanResourceSet::rw_buffer_array(
    uint32_t binding,
    const std::vector<DeviceAllocation> &allocs) {
  // C-2.5 (2026-05): single binding, descriptorCount=N storage buffers.
  // Each buffer is bound with VK_WHOLE_SIZE; SPIR-V side selects chunk[k]
  // via OpAccessChain on the array variable. Empty allocs is treated as
  // an empty descriptor write (caller's responsibility to ensure non-empty
  // when shader actually reads).
  BufferArray ba;
  ba.buffers.reserve(allocs.size());
  for (const auto &alloc : allocs) {
    vkapi::IVkBuffer buffer =
        (alloc != kDeviceNullAllocation)
            ? device_->get_vkbuffer(alloc.get_ptr(0))
            : nullptr;
    ba.buffers.push_back(buffer);
  }
  set_binding(binding, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, std::move(ba)});
  return *this;
}

VulkanResourceSet &VulkanResourceSet::acceleration_structure(
    uint32_t binding,
    vkapi::IVkAccelerationStructureKHR acceleration_structure) {
  set_binding(binding,
              {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
               AccelerationStructure{std::move(acceleration_structure)}});
  return *this;
}

RhiReturn<vkapi::IVkDescriptorSet> VulkanResourceSet::finalize() {
  return finalize_impl(/*replay_dedicated=*/false,
                       /*patch_existing=*/false);
}

RhiResult VulkanResourceSet::prepare_for_replay(bool patch_existing) {
  if (patch_existing &&
      !device_->vk_caps().descriptor_update_after_bind) {
    return RhiResult::not_supported;
  }
  return finalize_impl(/*replay_dedicated=*/true, patch_existing).result;
}

RhiReturn<vkapi::IVkDescriptorSet> VulkanResourceSet::finalize_impl(
    bool replay_dedicated,
    bool patch_existing) {
  if (!dirty_ && set_) {
    // If nothing changed directly return the set
    return {RhiResult::success, set_};
  }

  if (bindings_.size() <= 0) {
    // A set can't be empty
    return {RhiResult::invalid_usage, nullptr};
  }

  if (!replay_dedicated && device_->descriptor_set_cache_enabled()) {
    if (auto cached_set = device_->find_cached_desc_set(*this)) {
      set_ = cached_set;
      layout_ = set_->ref_layout;
      dirty_ = false;
      return {RhiResult::success, set_};
    }
  }

  vkapi::IVkDescriptorSetLayout new_layout =
      device_->get_desc_set_layout(*this);
  if (new_layout != layout_) {
    if (patch_existing) {
      return {RhiResult::invalid_usage, nullptr};
    }
    // Layout changed, reset `set`.
    set_ = nullptr;
    layout_ = new_layout;
  }
  if (patch_existing && !set_) {
    return {RhiResult::invalid_usage, nullptr};
  }

  if (set_) {
    std::lock_guard<std::mutex> descriptor_set_lock(set_->mutex);
    // Normal resource sets remain immutable while command buffers own them.
    // A replay-dedicated set may be patched only after its slot fence is ready.
    if (!patch_existing && set_->recording_use_count != 0) {
      set_ = nullptr;
    }
  }

  if (!set_) {
    // If set_ is null, create a new one
    auto [status, new_set] = device_->alloc_desc_set(layout_);
    if (status != RhiResult::success) {
      return {status, nullptr};
    }
    set_ = new_set;
  }

  std::forward_list<VkDescriptorBufferInfo> buffer_infos;
  std::forward_list<VkDescriptorImageInfo> image_infos;
  std::forward_list<VkWriteDescriptorSetAccelerationStructureKHR>
      acceleration_structure_infos;
  // C-2.5 (2026-05): per-BufferArray storage. Each binding owns its own
  // contiguous std::vector; the std::list keeps inner vectors stable in
  // memory across emplaces so VkWriteDescriptorSet::pBufferInfo remains
  // valid until vkUpdateDescriptorSets.
  std::list<std::vector<VkDescriptorBufferInfo>> buffer_array_infos;
  std::vector<VkWriteDescriptorSet> desc_writes;

  {
    std::unique_lock<std::mutex> descriptor_set_lock(set_->mutex);
    // A cache hit in another resource set may bind this descriptor between the
    // earlier availability check and this update lock. Retry with a replacement
    // instead of rewriting the newly recorded set.
    if (!patch_existing && set_->recording_use_count != 0) {
      descriptor_set_lock.unlock();
      set_ = nullptr;
      return finalize_impl(replay_dedicated,
                           /*patch_existing=*/false);
    }
    set_->ref_binding_objs.clear();

    for (auto &pair : bindings_) {
      uint32_t binding = pair.first;
      VkDescriptorType type = pair.second.type;
      auto &resource = pair.second.res;

      VkWriteDescriptorSet &write = desc_writes.emplace_back();
      write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      write.pNext = nullptr;
      write.dstSet = set_->set;
      write.dstBinding = binding;
      write.dstArrayElement = 0;
      write.descriptorCount = 1;
      write.descriptorType = type;
      write.pImageInfo = nullptr;
      write.pBufferInfo = nullptr;
      write.pTexelBufferView = nullptr;

      if (Buffer *buf = std::get_if<Buffer>(&resource)) {
        VkDescriptorBufferInfo &buffer_info = buffer_infos.emplace_front();
        buffer_info.buffer =
            buf->buffer ? buf->buffer->buffer : VK_NULL_HANDLE;
        buffer_info.offset = buf->offset;
        buffer_info.range = buf->size;

        write.pBufferInfo = &buffer_info;
        if (buf->buffer) {
          set_->ref_binding_objs.push_back(buf->buffer);
        }
      } else if (Image *img = std::get_if<Image>(&resource)) {
        VkDescriptorImageInfo &image_info = image_infos.emplace_front();
        image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        image_info.imageView = img->view ? img->view->view : VK_NULL_HANDLE;
        image_info.sampler = VK_NULL_HANDLE;

        write.pImageInfo = &image_info;
        if (img->view) {
          set_->ref_binding_objs.push_back(img->view);
        }
      } else if (Texture *tex = std::get_if<Texture>(&resource)) {
        VkDescriptorImageInfo &image_info = image_infos.emplace_front();
        image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        image_info.imageView = tex->view ? tex->view->view : VK_NULL_HANDLE;
        image_info.sampler =
            tex->sampler ? tex->sampler->sampler : VK_NULL_HANDLE;

        write.pImageInfo = &image_info;
        if (tex->view) {
          set_->ref_binding_objs.push_back(tex->view);
        }
        if (tex->sampler) {
          set_->ref_binding_objs.push_back(tex->sampler);
        }
      } else if (BufferArray *ba = std::get_if<BufferArray>(&resource)) {
        // C-2.5: emit N contiguous VkDescriptorBufferInfo entries; each
        // buffer is bound with offset=0, range=VK_WHOLE_SIZE. Empty array
        // is rejected by Vulkan (descriptorCount must be > 0).
        auto &infos = buffer_array_infos.emplace_back();
        infos.resize(ba->buffers.size());
        for (size_t i = 0; i < ba->buffers.size(); ++i) {
          const auto &b = ba->buffers[i];
          infos[i].buffer = b ? b->buffer : VK_NULL_HANDLE;
          infos[i].offset = 0;
          infos[i].range = VK_WHOLE_SIZE;
          if (b) {
            set_->ref_binding_objs.push_back(b);
          }
        }
        write.descriptorCount = static_cast<uint32_t>(infos.size());
        write.pBufferInfo = infos.data();
      } else if (AccelerationStructure *as =
                     std::get_if<AccelerationStructure>(&resource)) {
        auto &as_info = acceleration_structure_infos.emplace_front();
        as_info.sType =
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
        as_info.pNext = nullptr;
        as_info.accelerationStructureCount = 1;
        as_info.pAccelerationStructures = as->acceleration_structure
                                              ? &as->acceleration_structure->accel
                                              : nullptr;
        write.pNext = &as_info;
        if (as->acceleration_structure) {
          set_->ref_binding_objs.push_back(as->acceleration_structure);
        }
      } else {
        RHI_LOG_ERROR("Ignoring unsupported Descriptor Type");
      }
    }

    device_->update_descriptor_sets_locked(desc_writes);
  }

  dirty_ = false;
  if (!replay_dedicated && device_->descriptor_set_cache_enabled()) {
    device_->cache_desc_set(*this, set_);
  }

  return {RhiResult::success, set_};
}

RasterResources &VulkanRasterResources::vertex_buffer(DevicePtr ptr,
                                                      uint32_t binding) {
  vkapi::IVkBuffer buffer =
      (ptr != kDeviceNullPtr) ? device_->get_vkbuffer(ptr) : nullptr;
  if (buffer == nullptr) {
    vertex_buffers.erase(binding);
  } else {
    vertex_buffers[binding] = {buffer, ptr.offset};
  }
  return *this;
}

RasterResources &VulkanRasterResources::index_buffer(DevicePtr ptr,
                                                     size_t index_width) {
  vkapi::IVkBuffer buffer =
      (ptr != kDeviceNullPtr) ? device_->get_vkbuffer(ptr) : nullptr;
  if (buffer == nullptr) {
    index_binding = BufferBinding();
    index_type = VK_INDEX_TYPE_MAX_ENUM;
  } else {
    index_binding = {buffer, ptr.offset};
    if (index_width == 32) {
      index_type = VK_INDEX_TYPE_UINT32;
    } else if (index_width == 16) {
      index_type = VK_INDEX_TYPE_UINT16;
    }
  }
  return *this;
}

VulkanCommandList::VulkanCommandList(VulkanDevice *ti_device,
                                     VulkanStream *stream,
                                     vkapi::IVkCommandBuffer buffer)
    : ti_device_(ti_device),
      stream_(stream),
      device_(ti_device->vk_device()),
      buffer_(buffer) {
  VkCommandBufferBeginInfo info{};
  info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  info.pNext = nullptr;
  info.pInheritanceInfo = nullptr;
  info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

  VkCommandBufferInheritanceInfo inheritance{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO};
  if (buffer->level == VK_COMMAND_BUFFER_LEVEL_SECONDARY) {
    info.pInheritanceInfo = &inheritance;
  }

  vkBeginCommandBuffer(buffer->buffer, &info);
}

VulkanCommandList::~VulkanCommandList() {
  if (profiler_sampler_reservations_ != 0) {
    ti_device_->profiler_discard_reserved_samplers(
        profiler_sampler_reservations_);
  }
}

void VulkanCommandList::bind_pipeline(Pipeline *p) noexcept {
  auto pipeline = static_cast<VulkanPipeline *>(p);

  if (current_pipeline_ == pipeline)
    return;

  if (pipeline->is_graphics()) {
    vkapi::IVkPipeline vk_pipeline =
        ti_device_->vk_caps().dynamic_rendering
            ? pipeline->graphics_pipeline_dynamic(current_renderpass_desc_)
            : pipeline->graphics_pipeline(current_renderpass_desc_,
                                          current_renderpass_);
    vkCmdBindPipeline(buffer_->buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      vk_pipeline->pipeline);

    apply_raster_viewport_and_scissor();
    vkCmdSetLineWidth(buffer_->buffer, 1.0f);
    buffer_->refs.push_back(vk_pipeline);
  } else {
    auto vk_pipeline = pipeline->pipeline();
    vkCmdBindPipeline(buffer_->buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      vk_pipeline->pipeline);
    buffer_->refs.push_back(vk_pipeline);
  }

  current_pipeline_ = pipeline;
}

RhiResult VulkanCommandList::bind_shader_resources(ShaderResourceSet *res,
                                                   int set_index) noexcept {
  VulkanResourceSet *set = static_cast<VulkanResourceSet *>(res);
  if (set->get_bindings().size() <= 0) {
    return RhiResult::success;
  }

  auto [status, vk_set] = set->finalize();
  if (status != RhiResult::success) {
    return status;
  }

  vkapi::IVkDescriptorSetLayout set_layout = set->get_layout();

  if (current_pipeline_->pipeline_layout()->ref_desc_layouts.empty() ||
      current_pipeline_->pipeline_layout()->ref_desc_layouts[set_index] !=
          set_layout) {
    // WARN: we have a layout mismatch
    RHI_LOG_ERROR("Layout mismatch");

    auto &templates = current_pipeline_->get_resource_set_templates();
    VulkanResourceSet &set_template = templates.at(set_index);

    for (const auto &template_binding : set_template.get_bindings()) {
      std::array<char, 256> msg_buf;
      RHI_DEBUG_SNPRINTF(msg_buf.data(), msg_buf.size(),
                         "Template binding %d: (VkDescriptorType) %d",
                         template_binding.first, template_binding.second.type);
      RHI_LOG_ERROR(msg_buf.data());
    }

    for (const auto &binding : set->get_bindings()) {
      std::array<char, 256> msg_buf;
      RHI_DEBUG_SNPRINTF(msg_buf.data(), msg_buf.size(),
                         "Binding %d: (VkDescriptorType) %d", binding.first,
                         binding.second.type);
      RHI_LOG_ERROR(msg_buf.data());
    }

    return RhiResult::invalid_usage;
  }

  VkPipelineLayout pipeline_layout =
      current_pipeline_->pipeline_layout()->layout;
  VkPipelineBindPoint bind_point = current_pipeline_->is_graphics()
                                       ? VK_PIPELINE_BIND_POINT_GRAPHICS
                                       : VK_PIPELINE_BIND_POINT_COMPUTE;

  {
    std::lock_guard<std::mutex> descriptor_set_lock(vk_set->mutex);
    ++vk_set->recording_use_count;
    vkCmdBindDescriptorSets(buffer_->buffer, bind_point, pipeline_layout,
                            /*firstSet=*/set_index,
                            /*descriptorSetCount=*/1, &vk_set->set,
                            /*dynamicOffsetCount=*/0,
                            /*pDynamicOffsets=*/nullptr);
  }
  buffer_->descriptor_sets_in_use.push_back(vk_set);
  buffer_->refs.push_back(vk_set);

  return RhiResult::success;
}

RhiResult VulkanCommandList::bind_raster_resources(
    RasterResources *_res) noexcept {
  VulkanRasterResources *res = static_cast<VulkanRasterResources *>(_res);

  if (!current_pipeline_->is_graphics()) {
    return RhiResult::invalid_usage;
  }

  if (res->index_binding.buffer != nullptr) {
    // We have a valid index buffer
    if (res->index_type >= VK_INDEX_TYPE_MAX_ENUM) {
      return RhiResult::not_supported;
    }

    vkapi::IVkBuffer index_buffer = res->index_binding.buffer;
    vkCmdBindIndexBuffer(buffer_->buffer, index_buffer->buffer,
                         res->index_binding.offset, res->index_type);
    buffer_->refs.push_back(index_buffer);
  }

  for (auto &[binding, buffer] : res->vertex_buffers) {
    VkDeviceSize offset_vk = buffer.offset;
    vkCmdBindVertexBuffers(buffer_->buffer, binding, 1, &buffer.buffer->buffer,
                           &offset_vk);
    buffer_->refs.push_back(buffer.buffer);
  }

  return RhiResult::success;
}

namespace {

VkPipelineStageFlags buffer_barrier_stage_to_vk(
    BufferBarrierStage stages) noexcept {
  VkPipelineStageFlags result = 0;
  if (int(stages & BufferBarrierStage::Transfer)) {
    result |= VK_PIPELINE_STAGE_TRANSFER_BIT;
  }
  if (int(stages & BufferBarrierStage::Compute)) {
    result |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  }
  if (int(stages & BufferBarrierStage::IndirectCommand)) {
    result |= VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
  }
  if (int(stages & BufferBarrierStage::ConditionalCommand)) {
    result |= VK_PIPELINE_STAGE_CONDITIONAL_RENDERING_BIT_EXT;
  }
  if (int(stages & BufferBarrierStage::Host)) {
    result |= VK_PIPELINE_STAGE_HOST_BIT;
  }
  return result;
}

VkAccessFlags buffer_barrier_access_to_vk(
    BufferBarrierAccess accesses) noexcept {
  VkAccessFlags result = 0;
  if (int(accesses & BufferBarrierAccess::TransferRead)) {
    result |= VK_ACCESS_TRANSFER_READ_BIT;
  }
  if (int(accesses & BufferBarrierAccess::TransferWrite)) {
    result |= VK_ACCESS_TRANSFER_WRITE_BIT;
  }
  if (int(accesses & BufferBarrierAccess::ShaderRead)) {
    result |= VK_ACCESS_SHADER_READ_BIT;
  }
  if (int(accesses & BufferBarrierAccess::ShaderWrite)) {
    result |= VK_ACCESS_SHADER_WRITE_BIT;
  }
  if (int(accesses & BufferBarrierAccess::IndirectCommandRead)) {
    result |= VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
  }
  if (int(accesses & BufferBarrierAccess::ConditionalCommandRead)) {
    result |= VK_ACCESS_CONDITIONAL_RENDERING_READ_BIT_EXT;
  }
  if (int(accesses & BufferBarrierAccess::HostRead)) {
    result |= VK_ACCESS_HOST_READ_BIT;
  }
  if (int(accesses & BufferBarrierAccess::HostWrite)) {
    result |= VK_ACCESS_HOST_WRITE_BIT;
  }
  return result;
}

}  // namespace

void VulkanCommandList::buffer_barrier(DevicePtr ptr, size_t size) noexcept {
  const BufferTransition transition{
      BufferBarrierStage::Transfer | BufferBarrierStage::Compute,
      BufferBarrierAccess::TransferRead |
          BufferBarrierAccess::TransferWrite |
          BufferBarrierAccess::ShaderRead |
          BufferBarrierAccess::ShaderWrite,
      BufferBarrierStage::Transfer | BufferBarrierStage::Compute,
      BufferBarrierAccess::TransferRead |
          BufferBarrierAccess::TransferWrite |
          BufferBarrierAccess::ShaderRead |
          BufferBarrierAccess::ShaderWrite,
  };
  buffer_transition(ptr, size, transition);
}

void VulkanCommandList::buffer_transition(
    DevicePtr ptr,
    size_t size,
    const BufferTransition &transition) noexcept {
  if (ptr.device != ti_device_ || ptr.alloc_id == 0) {
    RHI_LOG_ERROR("Buffer transition requires a live allocation on this device");
    return;
  }
  auto buffer = ti_device_->get_vkbuffer(ptr);
  size_t buffer_size = ti_device_->get_vkbuffer_size(ptr);

  // Clamp to buffer size
  if (ptr.offset > buffer_size) {
    return;
  }

  if (saturate_uadd<size_t>(ptr.offset, size) > buffer_size) {
    size = VK_WHOLE_SIZE;
  }

  VkBufferMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  barrier.pNext = nullptr;
  barrier.buffer = buffer->buffer;
  barrier.offset = ptr.offset;
  barrier.size = size;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.srcAccessMask =
      buffer_barrier_access_to_vk(transition.source_access);
  barrier.dstAccessMask =
      buffer_barrier_access_to_vk(transition.destination_access);

  const VkPipelineStageFlags source_stage =
      buffer_barrier_stage_to_vk(transition.source_stage);
  const VkPipelineStageFlags destination_stage =
      buffer_barrier_stage_to_vk(transition.destination_stage);
  if (source_stage == 0 || destination_stage == 0) {
    RHI_LOG_ERROR("Buffer transition stages must not be empty");
    return;
  }

  vkCmdPipelineBarrier(
      buffer_->buffer, source_stage, destination_stage,
      /*dependencyFlags=*/0, /*memoryBarrierCount=*/0, nullptr,
      /*bufferMemoryBarrierCount=*/1,
      /*pBufferMemoryBarriers=*/&barrier,
      /*imageMemoryBarrierCount=*/0,
      /*pImageMemoryBarriers=*/nullptr);
  buffer_->refs.push_back(buffer);
}

void VulkanCommandList::buffer_barrier(DeviceAllocation alloc) noexcept {
  buffer_barrier(DevicePtr{alloc, 0}, std::numeric_limits<size_t>::max());
}

void VulkanCommandList::memory_barrier() noexcept {
  VkMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.pNext = nullptr;
  barrier.srcAccessMask =
      (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
  barrier.dstAccessMask =
      (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

  vkCmdPipelineBarrier(
      buffer_->buffer,
      /*srcStageMask=*/
      VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      /*dstStageMask=*/VK_PIPELINE_STAGE_TRANSFER_BIT |
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      /*srcStageMask=*/0, /*memoryBarrierCount=*/1, &barrier,
      /*bufferMemoryBarrierCount=*/0,
      /*pBufferMemoryBarriers=*/nullptr,
      /*imageMemoryBarrierCount=*/0,
      /*pImageMemoryBarriers=*/nullptr);
}

void VulkanCommandList::buffer_copy(DevicePtr dst,
                                    DevicePtr src,
                                    size_t size) noexcept {
  size_t src_size = ti_device_->get_vkbuffer_size(src);
  size_t dst_size = ti_device_->get_vkbuffer_size(dst);

  // Clamp to minimum available size
  if (saturate_uadd<size_t>(src.offset, size) > src_size) {
    size = saturate_usub<size_t>(src_size, src.offset);
  }
  if (saturate_uadd<size_t>(dst.offset, size) > dst_size) {
    size = saturate_usub<size_t>(dst_size, dst.offset);
  }

  if (size == 0) {
    return;
  }

  VkBufferCopy copy_region{};
  copy_region.srcOffset = src.offset;
  copy_region.dstOffset = dst.offset;
  copy_region.size = size;

  auto src_buffer = ti_device_->get_vkbuffer(src);
  auto dst_buffer = ti_device_->get_vkbuffer(dst);
  vkCmdCopyBuffer(buffer_->buffer, src_buffer->buffer, dst_buffer->buffer,
                  /*regionCount=*/1, &copy_region);
  buffer_->refs.push_back(src_buffer);
  buffer_->refs.push_back(dst_buffer);
}

void VulkanCommandList::buffer_fill(DevicePtr ptr,
                                    size_t size,
                                    uint32_t data) noexcept {
  // Align to 4 bytes
  ptr.offset = ptr.offset & size_t(-4);

  auto buffer = ti_device_->get_vkbuffer(ptr);
  size_t buffer_size = ti_device_->get_vkbuffer_size(ptr);

  // Check for overflow
  if (ptr.offset > buffer_size) {
    return;
  }

  if (saturate_uadd<size_t>(ptr.offset, size) > buffer_size) {
    size = VK_WHOLE_SIZE;
  }

  vkCmdFillBuffer(buffer_->buffer, buffer->buffer, ptr.offset, size, data);
  buffer_->refs.push_back(buffer);
}

void VulkanCommandList::push_constants(const void *data,
                                       uint32_t size) noexcept {
  if (!current_pipeline_ || !data || size == 0) {
    return;
  }
  TI_ASSERT(size <= 128);
  VkPipelineLayout pipeline_layout =
      current_pipeline_->pipeline_layout()->layout;
  vkCmdPushConstants(buffer_->buffer, pipeline_layout, VK_SHADER_STAGE_ALL, 0,
                     size, data);
}

RhiResult VulkanCommandList::dispatch(uint32_t x,
                                      uint32_t y,
                                      uint32_t z) noexcept {
  auto &dev_props = ti_device_->get_vk_physical_device_props();
  if (x > dev_props.limits.maxComputeWorkGroupCount[0] ||
      y > dev_props.limits.maxComputeWorkGroupCount[1] ||
      z > dev_props.limits.maxComputeWorkGroupCount[2]) {
    return RhiResult::not_supported;
  }
  vkCmdDispatch(buffer_->buffer, x, y, z);
  return RhiResult::success;
}

RhiResult VulkanCommandList::dispatch_indirect(DevicePtr indirect) noexcept {
  constexpr size_t kDispatchIndirectCommandSize = 3 * sizeof(uint32_t);
  if (indirect.device != ti_device_ || indirect.alloc_id == 0) {
    return RhiResult::invalid_usage;
  }
  if (!int(ti_device_->allocation_usage(indirect) & AllocUsage::Indirect)) {
    return RhiResult::invalid_usage;
  }
  if ((indirect.offset & (sizeof(uint32_t) - 1)) != 0) {
    return RhiResult::invalid_usage;
  }
  const size_t buffer_size = ti_device_->get_vkbuffer_size(indirect);
  if (indirect.offset > buffer_size ||
      kDispatchIndirectCommandSize > buffer_size - indirect.offset) {
    return RhiResult::invalid_usage;
  }

  auto indirect_buffer = ti_device_->get_vkbuffer(indirect);
  vkCmdDispatchIndirect(buffer_->buffer, indirect_buffer->buffer,
                        indirect.offset);
  buffer_->refs.push_back(indirect_buffer);
  return RhiResult::success;
}

RhiResult VulkanCommandList::begin_conditional(DevicePtr predicate,
                                               bool inverted) noexcept {
  constexpr size_t kConditionalPredicateSize = sizeof(uint32_t);
  if (!ti_device_->vk_caps().conditional_rendering) {
    return RhiResult::not_supported;
  }
  if (conditional_active_ || predicate.device != ti_device_ ||
      predicate.alloc_id == 0) {
    return RhiResult::invalid_usage;
  }
  if (!int(ti_device_->allocation_usage(predicate) &
           AllocUsage::Conditional)) {
    return RhiResult::invalid_usage;
  }
  if ((predicate.offset & (alignof(uint32_t) - 1)) != 0) {
    return RhiResult::invalid_usage;
  }
  const size_t buffer_size = ti_device_->get_vkbuffer_size(predicate);
  if (predicate.offset > buffer_size ||
      kConditionalPredicateSize > buffer_size - predicate.offset) {
    return RhiResult::invalid_usage;
  }

  auto predicate_buffer = ti_device_->get_vkbuffer(predicate);
  VkConditionalRenderingBeginInfoEXT info{};
  info.sType = VK_STRUCTURE_TYPE_CONDITIONAL_RENDERING_BEGIN_INFO_EXT;
  info.buffer = predicate_buffer->buffer;
  info.offset = predicate.offset;
  info.flags =
      inverted ? VK_CONDITIONAL_RENDERING_INVERTED_BIT_EXT : 0;
  vkCmdBeginConditionalRenderingEXT(buffer_->buffer, &info);
  buffer_->refs.push_back(predicate_buffer);
  conditional_active_ = true;
  return RhiResult::success;
}

RhiResult VulkanCommandList::end_conditional() noexcept {
  if (!ti_device_->vk_caps().conditional_rendering) {
    return RhiResult::not_supported;
  }
  if (!conditional_active_) {
    return RhiResult::invalid_usage;
  }
  vkCmdEndConditionalRenderingEXT(buffer_->buffer);
  conditional_active_ = false;
  return RhiResult::success;
}

vkapi::IVkCommandBuffer VulkanCommandList::vk_command_buffer() {
  return buffer_;
}

VkCommandBuffer VulkanCommandList::begin_external_compute(
    vkapi::IDeviceObj owner) {
  buffer_->refs.push_back(std::move(owner));
  current_pipeline_ = nullptr;
  return buffer_->buffer;
}

void VulkanCommandList::execute_secondary(vkapi::IVkCommandBuffer buffer) {
  buffer_->refs.push_back(buffer);
  current_pipeline_ = nullptr;
  vkCmdExecuteCommands(buffer_->buffer, 1, &buffer->buffer);
}

std::function<void(CommandList *)> VulkanCommandList::finalize_secondary(
    std::shared_ptr<void> resource_owner) {
  struct Resources : vkapi::DeviceObj {
    std::shared_ptr<void> owner;
  };
  auto buffer = finalize();
  TI_ERROR_IF(buffer->level != VK_COMMAND_BUFFER_LEVEL_SECONDARY,
              "Only secondary commands may be finalized for embedding");
  auto resources = std::make_shared<Resources>();
  resources->owner = std::move(resource_owner);
  buffer->refs.push_back(std::move(resources));
  return [buffer = std::move(buffer)](CommandList *parent) {
    static_cast<VulkanCommandList *>(parent)->execute_secondary(buffer);
  };
}

void VulkanCommandList::set_next_renderpass_color_final_layout(
    ImageLayout layout) {
  TI_ERROR_IF(layout != ImageLayout::color_attachment &&
                  layout != ImageLayout::present_src,
              "Vulkan render-pass color final layout must be color_attachment "
              "or present_src.");
  current_renderpass_desc_.color_final_layout =
      image_layout_ti_to_vk(layout);
}

void VulkanCommandList::begin_renderpass(int x0,
                                         int y0,
                                         int x1,
                                         int y1,
                                         uint32_t num_color_attachments,
                                         DeviceAllocation *color_attachments,
                                         bool *color_clear,
                                         std::vector<float> *clear_colors,
                                         DeviceAllocation *depth_attachment,
                                         bool depth_clear) {
  VulkanRenderPassDesc &rp_desc = current_renderpass_desc_;
  current_renderpass_desc_.color_attachments.clear();
  rp_desc.clear_depth = depth_clear;

  VkRect2D render_area{/*offset*/ {x0, y0},
                       /*extent*/ {uint32_t(x1 - x0), uint32_t(y1 - y0)}};

  viewport_width_ = render_area.extent.width;
  viewport_height_ = render_area.extent.height;
  viewport_x_ = render_area.offset.x;
  viewport_y_ = render_area.offset.y;

  // Dynamic rendering codepath
  if (ti_device_->vk_caps().dynamic_rendering) {
    current_dynamic_targets_.clear();

    std::vector<VkRenderingAttachmentInfoKHR> color_attachment_infos(
        num_color_attachments);
    for (uint32_t i = 0; i < num_color_attachments; i++) {
      auto [image, view, format] =
          ti_device_->get_vk_image(color_attachments[i]);
      bool clear = color_clear[i];
      rp_desc.color_attachments.emplace_back(format, clear);

      VkRenderingAttachmentInfoKHR &attachment_info = color_attachment_infos[i];
      attachment_info.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
      attachment_info.pNext = nullptr;
      attachment_info.imageView = view->view;
      attachment_info.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      attachment_info.resolveMode = VK_RESOLVE_MODE_NONE;
      attachment_info.resolveImageView = VK_NULL_HANDLE;
      attachment_info.resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      attachment_info.loadOp =
          clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
      attachment_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      if (clear) {
        attachment_info.clearValue.color = {
            {clear_colors[i][0], clear_colors[i][1], clear_colors[i][2],
             clear_colors[i][3]}};
      }

      current_dynamic_targets_.push_back(image);
    }

    VkRenderingInfoKHR render_info{};
    render_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
    render_info.pNext = nullptr;
    render_info.flags = 0;
    render_info.renderArea = render_area;
    render_info.layerCount = 1;
    render_info.viewMask = 0;
    render_info.colorAttachmentCount = num_color_attachments;
    render_info.pColorAttachments = color_attachment_infos.data();
    render_info.pDepthAttachment = nullptr;
    render_info.pStencilAttachment = nullptr;

    VkRenderingAttachmentInfo depth_attachment_info{};
    if (depth_attachment) {
      auto [image, view, format] = ti_device_->get_vk_image(*depth_attachment);
      rp_desc.depth_attachment = format;

      depth_attachment_info.sType =
          VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
      depth_attachment_info.pNext = nullptr;
      depth_attachment_info.imageView = view->view;
      depth_attachment_info.imageLayout =
          image_layout_ti_to_vk(ImageLayout::depth_attachment);
      depth_attachment_info.resolveMode = VK_RESOLVE_MODE_NONE;
      depth_attachment_info.resolveImageView = VK_NULL_HANDLE;
      depth_attachment_info.resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      depth_attachment_info.loadOp = depth_clear ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                                 : VK_ATTACHMENT_LOAD_OP_LOAD;
      depth_attachment_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      depth_attachment_info.clearValue.depthStencil = {0.0, 0};

      render_info.pDepthAttachment = &depth_attachment_info;

      current_dynamic_targets_.push_back(image);
    } else {
      rp_desc.depth_attachment = VK_FORMAT_UNDEFINED;
    }

    vkCmdBeginRenderingKHR(buffer_->buffer, &render_info);

    return;
  }

  // VkRenderpass & VkFramebuffer codepath
  bool has_depth = false;

  if (depth_attachment) {
    auto [image, view, format] = ti_device_->get_vk_image(*depth_attachment);
    rp_desc.depth_attachment = format;
    has_depth = true;
  } else {
    rp_desc.depth_attachment = VK_FORMAT_UNDEFINED;
  }

  std::vector<VkClearValue> clear_values(num_color_attachments +
                                         (has_depth ? 1 : 0));

  VulkanFramebufferDesc fb_desc;

  for (uint32_t i = 0; i < num_color_attachments; i++) {
    auto [image, view, format] = ti_device_->get_vk_image(color_attachments[i]);
    rp_desc.color_attachments.emplace_back(format, color_clear[i]);
    fb_desc.attachments.push_back(view);
    clear_values[i].color =
        VkClearColorValue{{clear_colors[i][0], clear_colors[i][1],
                           clear_colors[i][2], clear_colors[i][3]}};
  }

  if (has_depth) {
    auto [depth_image, depth_view, depth_format] =
        ti_device_->get_vk_image(*depth_attachment);
    clear_values[num_color_attachments].depthStencil =
        VkClearDepthStencilValue{0.0, 0};
    fb_desc.attachments.push_back(depth_view);
  }

  current_renderpass_ = ti_device_->get_renderpass(rp_desc);

  fb_desc.width = x1 - x0;
  fb_desc.height = y1 - y0;
  fb_desc.renderpass = current_renderpass_;

  current_framebuffer_ = ti_device_->get_framebuffer(fb_desc);

  VkRenderPassBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  begin_info.pNext = nullptr;
  begin_info.renderPass = current_renderpass_->renderpass;
  begin_info.framebuffer = current_framebuffer_->framebuffer;
  begin_info.renderArea = render_area;
  begin_info.clearValueCount = clear_values.size();
  begin_info.pClearValues = clear_values.data();

  vkCmdBeginRenderPass(buffer_->buffer, &begin_info,
                       VK_SUBPASS_CONTENTS_INLINE);
  buffer_->refs.push_back(current_renderpass_);
  buffer_->refs.push_back(current_framebuffer_);
}

void VulkanCommandList::end_renderpass() {
  if (ti_device_->vk_caps().dynamic_rendering) {
    vkCmdEndRenderingKHR(buffer_->buffer);

    if (0) {
      std::vector<VkImageMemoryBarrier> memory_barriers(
          current_dynamic_targets_.size());
      for (int i = 0; i < current_dynamic_targets_.size(); i++) {
        VkImageMemoryBarrier &barrier = memory_barriers[i];
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.pNext = nullptr;
        barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        // FIXME: Change this spec to stay in color attachment
        barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = current_dynamic_targets_[i]->image;
        barrier.subresourceRange.aspectMask =
            VK_IMAGE_ASPECT_COLOR_BIT | VK_IMAGE_ASPECT_DEPTH_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
      }

      vkCmdPipelineBarrier(buffer_->buffer,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                           VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           /*dependencyFlags=*/0, /*memoryBarrierCount=*/0,
                           /*pMemoryBarriers=*/nullptr,
                           /*bufferMemoryBarrierCount=*/0,
                           /*pBufferMemoryBarriers=*/nullptr,
                           /*imageMemoryBarrierCount=*/memory_barriers.size(),
                           /*pImageMemoryBarriers=*/memory_barriers.data());
    }
    current_dynamic_targets_.clear();

    return;
  }

  vkCmdEndRenderPass(buffer_->buffer);

  current_renderpass_ = VK_NULL_HANDLE;
  current_framebuffer_ = VK_NULL_HANDLE;
}

void VulkanCommandList::draw(uint32_t num_verticies, uint32_t start_vertex) {
  vkCmdDraw(buffer_->buffer, num_verticies, /*instanceCount=*/1, start_vertex,
            /*firstInstance=*/0);
}

void VulkanCommandList::draw_instance(uint32_t num_verticies,
                                      uint32_t num_instances,
                                      uint32_t start_vertex,
                                      uint32_t start_instance) {
  vkCmdDraw(buffer_->buffer, num_verticies, num_instances, start_vertex,
            start_instance);
}

void VulkanCommandList::draw_indexed(uint32_t num_indicies,
                                     int32_t vertex_offset,
                                     uint32_t start_index) {
  vkCmdDrawIndexed(buffer_->buffer, num_indicies, /*instanceCount=*/1,
                   start_index, vertex_offset,
                   /*firstInstance=*/0);
}

void VulkanCommandList::draw_indexed_instance(uint32_t num_indicies,
                                              uint32_t num_instances,
                                              int32_t vertex_offset,
                                              uint32_t start_index,
                                              uint32_t start_instance) {
  vkCmdDrawIndexed(buffer_->buffer, num_indicies, num_instances, start_index,
                   vertex_offset, start_instance);
}

namespace {

RhiResult validate_draw_indirect_buffer(VulkanDevice *device,
                                        DevicePtr indirect,
                                        uint32_t draw_count,
                                        uint32_t stride,
                                        size_t command_size) noexcept {
  if (indirect.device != device || indirect.alloc_id == 0 || draw_count == 0) {
    return RhiResult::invalid_usage;
  }
  if (!int(device->allocation_usage(indirect) & AllocUsage::Indirect) ||
      (indirect.offset & (sizeof(uint32_t) - 1)) != 0 ||
      stride < command_size || (stride & (sizeof(uint32_t) - 1)) != 0) {
    return RhiResult::invalid_usage;
  }
  const auto &caps = device->vk_caps();
  if (draw_count > caps.max_draw_indirect_count) {
    return RhiResult::not_supported;
  }
  if (draw_count > 1 && !caps.multi_draw_indirect) {
    return RhiResult::not_supported;
  }
  const size_t buffer_size = device->get_vkbuffer_size(indirect);
  if (indirect.offset > buffer_size) {
    return RhiResult::invalid_usage;
  }
  const size_t count_minus_one = static_cast<size_t>(draw_count - 1);
  if (count_minus_one >
      ((std::numeric_limits<size_t>::max)() - command_size) / stride) {
    return RhiResult::invalid_usage;
  }
  const size_t required = count_minus_one * stride + command_size;
  if (required > buffer_size - indirect.offset) {
    return RhiResult::invalid_usage;
  }
  return RhiResult::success;
}

RhiResult validate_draw_count_buffer(VulkanDevice *device,
                                     DevicePtr count) noexcept {
  if (count.device != device || count.alloc_id == 0 ||
      !int(device->allocation_usage(count) & AllocUsage::Indirect) ||
      (count.offset & (sizeof(uint32_t) - 1)) != 0) {
    return RhiResult::invalid_usage;
  }
  const size_t buffer_size = device->get_vkbuffer_size(count);
  if (count.offset > buffer_size ||
      sizeof(uint32_t) > buffer_size - count.offset) {
    return RhiResult::invalid_usage;
  }
  return RhiResult::success;
}

}  // namespace

RhiResult VulkanCommandList::draw_indirect(DevicePtr indirect,
                                           uint32_t draw_count,
                                           uint32_t stride) noexcept {
  if (!current_pipeline_ || !current_pipeline_->is_graphics() ||
      current_renderpass_ == VK_NULL_HANDLE) {
    return RhiResult::invalid_usage;
  }
  const auto result = validate_draw_indirect_buffer(
      ti_device_, indirect, draw_count, stride, sizeof(VkDrawIndirectCommand));
  if (result != RhiResult::success) {
    return result;
  }
  auto buffer = ti_device_->get_vkbuffer(indirect);
  vkCmdDrawIndirect(buffer_->buffer, buffer->buffer, indirect.offset,
                    draw_count, stride);
  buffer_->refs.push_back(buffer);
  return RhiResult::success;
}

RhiResult VulkanCommandList::draw_indexed_indirect(
    DevicePtr indirect,
    uint32_t draw_count,
    uint32_t stride) noexcept {
  if (!current_pipeline_ || !current_pipeline_->is_graphics() ||
      current_renderpass_ == VK_NULL_HANDLE) {
    return RhiResult::invalid_usage;
  }
  const auto result = validate_draw_indirect_buffer(
      ti_device_, indirect, draw_count, stride,
      sizeof(VkDrawIndexedIndirectCommand));
  if (result != RhiResult::success) {
    return result;
  }
  auto buffer = ti_device_->get_vkbuffer(indirect);
  vkCmdDrawIndexedIndirect(buffer_->buffer, buffer->buffer, indirect.offset,
                           draw_count, stride);
  buffer_->refs.push_back(buffer);
  return RhiResult::success;
}

RhiResult VulkanCommandList::draw_indirect_count(
    DevicePtr indirect,
    DevicePtr count,
    uint32_t max_draw_count,
    uint32_t stride) noexcept {
  if (!ti_device_->vk_caps().draw_indirect_count || !current_pipeline_ ||
      !current_pipeline_->is_graphics() ||
      current_renderpass_ == VK_NULL_HANDLE) {
    return RhiResult::not_supported;
  }
  const auto indirect_result = validate_draw_indirect_buffer(
      ti_device_, indirect, max_draw_count, stride,
      sizeof(VkDrawIndirectCommand));
  if (indirect_result != RhiResult::success) {
    return indirect_result;
  }
  const auto count_result = validate_draw_count_buffer(ti_device_, count);
  if (count_result != RhiResult::success) {
    return count_result;
  }
  auto indirect_buffer = ti_device_->get_vkbuffer(indirect);
  auto count_buffer = ti_device_->get_vkbuffer(count);
  if (ti_device_->vk_caps().vk_api_version >= VK_API_VERSION_1_2) {
    vkCmdDrawIndirectCount(buffer_->buffer, indirect_buffer->buffer,
                           indirect.offset, count_buffer->buffer, count.offset,
                           max_draw_count, stride);
  } else {
    vkCmdDrawIndirectCountKHR(buffer_->buffer, indirect_buffer->buffer,
                              indirect.offset, count_buffer->buffer,
                              count.offset, max_draw_count, stride);
  }
  buffer_->refs.push_back(indirect_buffer);
  buffer_->refs.push_back(count_buffer);
  return RhiResult::success;
}

RhiResult VulkanCommandList::draw_indexed_indirect_count(
    DevicePtr indirect,
    DevicePtr count,
    uint32_t max_draw_count,
    uint32_t stride) noexcept {
  if (!ti_device_->vk_caps().draw_indirect_count || !current_pipeline_ ||
      !current_pipeline_->is_graphics() ||
      current_renderpass_ == VK_NULL_HANDLE) {
    return RhiResult::not_supported;
  }
  const auto indirect_result = validate_draw_indirect_buffer(
      ti_device_, indirect, max_draw_count, stride,
      sizeof(VkDrawIndexedIndirectCommand));
  if (indirect_result != RhiResult::success) {
    return indirect_result;
  }
  const auto count_result = validate_draw_count_buffer(ti_device_, count);
  if (count_result != RhiResult::success) {
    return count_result;
  }
  auto indirect_buffer = ti_device_->get_vkbuffer(indirect);
  auto count_buffer = ti_device_->get_vkbuffer(count);
  if (ti_device_->vk_caps().vk_api_version >= VK_API_VERSION_1_2) {
    vkCmdDrawIndexedIndirectCount(
        buffer_->buffer, indirect_buffer->buffer, indirect.offset,
        count_buffer->buffer, count.offset, max_draw_count, stride);
  } else {
    vkCmdDrawIndexedIndirectCountKHR(
        buffer_->buffer, indirect_buffer->buffer, indirect.offset,
        count_buffer->buffer, count.offset, max_draw_count, stride);
  }
  buffer_->refs.push_back(indirect_buffer);
  buffer_->refs.push_back(count_buffer);
  return RhiResult::success;
}

RhiResult VulkanCommandList::draw_mesh_tasks(uint32_t group_count_x,
                                              uint32_t group_count_y,
                                              uint32_t group_count_z,
                                              bool task_shader) noexcept {
  const auto &caps = ti_device_->vk_caps();
  if (!caps.mesh_shader || (task_shader && !caps.task_shader) ||
      vkCmdDrawMeshTasksEXT == nullptr || current_pipeline_ == nullptr ||
      !current_pipeline_->is_graphics() ||
      current_renderpass_ == VK_NULL_HANDLE || group_count_x == 0 ||
      group_count_y == 0 || group_count_z == 0) {
    return RhiResult::not_supported;
  }
  const auto &limits = task_shader ? caps.max_task_work_group_count
                                   : caps.max_mesh_work_group_count;
  const std::uint64_t total = static_cast<std::uint64_t>(group_count_x) *
                              group_count_y * group_count_z;
  const std::uint64_t total_limit =
      task_shader ? caps.max_task_work_group_total_count
                  : caps.max_mesh_work_group_total_count;
  if (group_count_x > limits[0] || group_count_y > limits[1] ||
      group_count_z > limits[2] || total > total_limit) {
    return RhiResult::invalid_usage;
  }
  vkCmdDrawMeshTasksEXT(buffer_->buffer, group_count_x, group_count_y,
                        group_count_z);
  return RhiResult::success;
}

void VulkanCommandList::image_transition(DeviceAllocation img,
                                         ImageLayout old_layout_,
                                         ImageLayout new_layout_) {
  auto [image, view, format] = ti_device_->get_vk_image(img);

  VkImageLayout old_layout = image_layout_ti_to_vk(old_layout_);
  VkImageLayout new_layout = image_layout_ti_to_vk(new_layout_);

  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = old_layout;
  barrier.newLayout = new_layout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image->image;
  if (format == VK_FORMAT_D16_UNORM || format == VK_FORMAT_D32_SFLOAT) {
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
  } else if (format == VK_FORMAT_D24_UNORM_S8_UINT) {
    barrier.subresourceRange.aspectMask =
        VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
  } else {
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  }
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;

  VkPipelineStageFlags source_stage;
  VkPipelineStageFlags destination_stage;

  // Layout alone does not identify the shader stage: these images are shared
  // by compute and graphics. TOP_OF_PIPE cannot make shader writes available
  // or delay shader reads. ALL_COMMANDS covers the queue's supported stages;
  // it does not introduce a host wait or a queue-family ownership transfer.
  static const std::unordered_map<VkImageLayout, VkPipelineStageFlags> stages{
      {VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT},
      {VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT},
      {VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT},
      {VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT},
      {VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT},
      {VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT},
      {VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
       VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
           VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT},
      {VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT},
      {VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT}};

  static const std::unordered_map<VkImageLayout, VkAccessFlags> access{
      {VK_IMAGE_LAYOUT_UNDEFINED, 0},
      {VK_IMAGE_LAYOUT_GENERAL,
       VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT},
      {VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT},
      {VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT},
      {VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT},
      {VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
       VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
           VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT},
      {VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
           VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT},
      {VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_SHADER_READ_BIT},
      {VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_MEMORY_READ_BIT}};

  if (stages.find(old_layout) == stages.end() ||
      stages.find(new_layout) == stages.end()) {
    throw std::invalid_argument("unsupported layout transition!");
  }
  source_stage = stages.at(old_layout);
  destination_stage = stages.at(new_layout);

  if (access.find(old_layout) == access.end() ||
      access.find(new_layout) == access.end()) {
    throw std::invalid_argument("unsupported layout transition!");
  }
  barrier.srcAccessMask = access.at(old_layout);
  barrier.dstAccessMask = access.at(new_layout);

  vkCmdPipelineBarrier(buffer_->buffer, source_stage, destination_stage, 0, 0,
                       nullptr, 0, nullptr, 1, &barrier);
  buffer_->refs.push_back(image);
}

inline void buffer_image_copy_ti_to_vk(VkBufferImageCopy &copy_info,
                                       size_t offset,
                                       const BufferImageCopyParams &params) {
  copy_info.bufferOffset = offset;
  copy_info.bufferRowLength = params.buffer_row_length;
  copy_info.bufferImageHeight = params.buffer_image_height;
  copy_info.imageExtent.width = params.image_extent.x;
  copy_info.imageExtent.height = params.image_extent.y;
  copy_info.imageExtent.depth = params.image_extent.z;
  copy_info.imageOffset.x = params.image_offset.x;
  copy_info.imageOffset.y = params.image_offset.y;
  copy_info.imageOffset.z = params.image_offset.z;
  copy_info.imageSubresource.aspectMask =
      params.image_aspect_flag;  // FIXME: add option in BufferImageCopyParams
                                 // to support copying depth images
                                 // FIXED: added an option in
                                 // BufferImageCopyParams as image_aspect_flag
                                 // by yuhaoLong(mocki)
  copy_info.imageSubresource.baseArrayLayer = params.image_base_layer;
  copy_info.imageSubresource.layerCount = params.image_layer_count;
  copy_info.imageSubresource.mipLevel = params.image_mip_level;
}

void VulkanCommandList::buffer_to_image(DeviceAllocation dst_img,
                                        DevicePtr src_buf,
                                        ImageLayout img_layout,
                                        const BufferImageCopyParams &params) {
  VkBufferImageCopy copy_info{};
  buffer_image_copy_ti_to_vk(copy_info, src_buf.offset, params);

  auto [image, view, format] = ti_device_->get_vk_image(dst_img);
  auto buffer = ti_device_->get_vkbuffer(src_buf);

  vkCmdCopyBufferToImage(buffer_->buffer, buffer->buffer, image->image,
                         image_layout_ti_to_vk(img_layout), 1, &copy_info);
  buffer_->refs.push_back(image);
  buffer_->refs.push_back(buffer);
}

void VulkanCommandList::image_to_buffer(DevicePtr dst_buf,
                                        DeviceAllocation src_img,
                                        ImageLayout img_layout,
                                        const BufferImageCopyParams &params) {
  VkBufferImageCopy copy_info{};
  buffer_image_copy_ti_to_vk(copy_info, dst_buf.offset, params);

  auto [image, view, format] = ti_device_->get_vk_image(src_img);
  auto buffer = ti_device_->get_vkbuffer(dst_buf);

  vkCmdCopyImageToBuffer(buffer_->buffer, image->image,
                         image_layout_ti_to_vk(img_layout), buffer->buffer, 1,
                         &copy_info);
  buffer_->refs.push_back(image);
  buffer_->refs.push_back(buffer);
}

void VulkanCommandList::copy_image(DeviceAllocation dst_img,
                                   DeviceAllocation src_img,
                                   ImageLayout dst_img_layout,
                                   ImageLayout src_img_layout,
                                   const ImageCopyParams &params) {
  VkImageCopy copy{};
  copy.srcSubresource.aspectMask = params.image_aspect_flag;
  copy.srcSubresource.mipLevel = params.source_mip_level;
  copy.srcSubresource.baseArrayLayer = params.source_base_layer;
  copy.srcSubresource.layerCount = params.layer_count;
  copy.srcOffset = {params.source_offset.x, params.source_offset.y,
                    params.source_offset.z};
  copy.dstSubresource.aspectMask = params.image_aspect_flag;
  copy.dstSubresource.mipLevel = params.destination_mip_level;
  copy.dstSubresource.baseArrayLayer = params.destination_base_layer;
  copy.dstSubresource.layerCount = params.layer_count;
  copy.dstOffset = {params.destination_offset.x, params.destination_offset.y,
                    params.destination_offset.z};
  copy.extent.width = params.width;
  copy.extent.height = params.height;
  copy.extent.depth = params.depth;

  auto [dst_vk_image, dst_view, dst_format] = ti_device_->get_vk_image(dst_img);
  auto [src_vk_image, src_view, src_format] = ti_device_->get_vk_image(src_img);

  vkCmdCopyImage(buffer_->buffer, src_vk_image->image,
                 image_layout_ti_to_vk(src_img_layout), dst_vk_image->image,
                 image_layout_ti_to_vk(dst_img_layout), 1, &copy);

  buffer_->refs.push_back(dst_vk_image);
  buffer_->refs.push_back(src_vk_image);
}

void VulkanCommandList::blit_image(DeviceAllocation dst_img,
                                   DeviceAllocation src_img,
                                   ImageLayout dst_img_layout,
                                   ImageLayout src_img_layout,
                                   const ImageBlitParams &params) {
  VkImageBlit blit{};
  blit.srcSubresource.aspectMask = params.image_aspect_flag;
  blit.srcSubresource.mipLevel = params.source.mip_level;
  blit.srcSubresource.baseArrayLayer = params.source.base_layer;
  blit.srcSubresource.layerCount = params.layer_count;
  blit.srcOffsets[0] = {params.source.offset.x, params.source.offset.y,
                        params.source.offset.z};
  blit.srcOffsets[1] = {
      params.source.offset.x + static_cast<int32_t>(params.source.extent.x),
      params.source.offset.y + static_cast<int32_t>(params.source.extent.y),
      params.source.offset.z + static_cast<int32_t>(params.source.extent.z)};
  blit.dstSubresource.aspectMask = params.image_aspect_flag;
  blit.dstSubresource.mipLevel = params.destination.mip_level;
  blit.dstSubresource.baseArrayLayer = params.destination.base_layer;
  blit.dstSubresource.layerCount = params.layer_count;
  blit.dstOffsets[0] = {params.destination.offset.x,
                        params.destination.offset.y,
                        params.destination.offset.z};
  blit.dstOffsets[1] = {
      params.destination.offset.x +
          static_cast<int32_t>(params.destination.extent.x),
      params.destination.offset.y +
          static_cast<int32_t>(params.destination.extent.y),
      params.destination.offset.z +
          static_cast<int32_t>(params.destination.extent.z)};

  auto [dst_vk_image, dst_view, dst_format] = ti_device_->get_vk_image(dst_img);
  auto [src_vk_image, src_view, src_format] = ti_device_->get_vk_image(src_img);

  vkCmdBlitImage(buffer_->buffer, src_vk_image->image,
                 image_layout_ti_to_vk(src_img_layout), dst_vk_image->image,
                 image_layout_ti_to_vk(dst_img_layout), 1, &blit,
                 params.linear_filter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);

  buffer_->refs.push_back(dst_vk_image);
  buffer_->refs.push_back(src_vk_image);
}

void VulkanCommandList::set_line_width(float width) {
  if (ti_device_->vk_caps().wide_line) {
    vkCmdSetLineWidth(buffer_->buffer, width);
  }
}

vkapi::IVkRenderPass VulkanCommandList::current_renderpass() {
  if (ti_device_->vk_caps().dynamic_rendering) {
    vkapi::IVkRenderPass rp =
        ti_device_->get_renderpass(current_renderpass_desc_);
    buffer_->refs.push_back(rp);
    return rp;
  }
  return current_renderpass_;
}

vkapi::IVkCommandBuffer VulkanCommandList::finalize() {
  if (!finalized_) {
    TI_ASSERT(!conditional_active_);
    vkEndCommandBuffer(buffer_->buffer);
    finalized_ = true;
  }
  return buffer_;
}

namespace {

constexpr std::size_t kVulkanMaxInFlightCommandBuffersPerStream = 64;
std::atomic<std::uint64_t> next_vulkan_stream_registry_id{1};

}  // namespace

struct VulkanDevice::ThreadLocalStreams {
  struct ThreadExitRegistration {
    ThreadExitRegistration(std::weak_ptr<ThreadLocalStreams> streams,
                           std::thread::id thread_id);
    ~ThreadExitRegistration();

    std::weak_ptr<ThreadLocalStreams> streams;
    std::thread::id thread_id;
  };

  ThreadLocalStreams(std::uint64_t registry_id,
                     std::thread::id owner_thread_id)
      : id(registry_id), owner_thread_id(owner_thread_id) {
  }

  void register_thread_exit(
      const std::shared_ptr<ThreadLocalStreams> &owner,
      std::thread::id thread_id);
  void retire(std::thread::id thread_id) noexcept;

  const std::uint64_t id;
  const std::thread::id owner_thread_id;
  std::mutex map_mutex;
  unordered_map<std::thread::id, std::unique_ptr<VulkanStream>> map;
};

VulkanDevice::ThreadLocalStreams::ThreadExitRegistration::
    ThreadExitRegistration(std::weak_ptr<ThreadLocalStreams> streams_,
                           std::thread::id thread_id_)
    : streams(std::move(streams_)), thread_id(thread_id_) {
}

VulkanDevice::ThreadLocalStreams::ThreadExitRegistration::
    ~ThreadExitRegistration() {
  if (auto owner = streams.lock()) {
    owner->retire(thread_id);
  }
}

void VulkanDevice::ThreadLocalStreams::register_thread_exit(
    const std::shared_ptr<ThreadLocalStreams> &owner,
    std::thread::id thread_id) {
  if (thread_id == owner_thread_id) {
    return;
  }
  thread_local unordered_map<
      std::uint64_t, std::unique_ptr<ThreadExitRegistration>>
      registrations;
  for (auto it = registrations.begin(); it != registrations.end();) {
    if (it->second->streams.expired()) {
      it = registrations.erase(it);
    } else {
      ++it;
    }
  }
  if (registrations.find(id) == registrations.end()) {
    registrations.emplace(
        id, std::make_unique<ThreadExitRegistration>(owner, thread_id));
  }
}

void VulkanDevice::ThreadLocalStreams::retire(
    std::thread::id thread_id) noexcept {
  std::unique_ptr<VulkanStream> stream;
  {
    std::lock_guard<std::mutex> lock(map_mutex);
    const auto found = map.find(thread_id);
    if (found == map.end()) {
      return;
    }
    stream = std::move(found->second);
    map.erase(found);
  }
  try {
    stream->command_sync();
  } catch (...) {
    // Thread-exit cleanup must not terminate the process. The stream already
    // reports backend failures through the owning device before throwing.
  }
}

VulkanDevice::VulkanDevice()
    : compute_streams_(std::make_shared<ThreadLocalStreams>(
          next_vulkan_stream_registry_id.fetch_add(
              1, std::memory_order_relaxed),
          std::this_thread::get_id())),
      graphics_streams_(std::make_shared<ThreadLocalStreams>(
          next_vulkan_stream_registry_id.fetch_add(
              1, std::memory_order_relaxed),
          std::this_thread::get_id())) {
  const int configured_profiler_records = get_environ_config(
      "TI_KERNEL_PROFILER_MAX_RECORDS", 131072);
  profiler_record_capacity_ = std::clamp<size_t>(
      static_cast<size_t>(std::max(1, configured_profiler_records)), 1,
      kMaximumProfilerRecordCapacity);

  DeviceCapabilityConfig caps{};
  caps.set(DeviceCapability::spirv_version, 0x10000);
  set_caps(std::move(caps));
}

void VulkanDevice::init_vulkan_structs(Params &params) {
  instance_ = params.instance;
  device_ = params.device;
  physical_device_ = params.physical_device;
  compute_queue_ = params.compute_queue;
  compute_queue_family_index_ = params.compute_queue_family_index;
  graphics_queue_ = params.graphics_queue;
  graphics_queue_family_index_ = params.graphics_queue_family_index;

  create_vma_allocator();
  {
    std::lock_guard<std::mutex> lock(descriptor_pool_mutex_);
    RHI_ASSERT(new_descriptor_pool_locked() == RhiResult::success &&
               "Failed to allocate initial descriptor pool");
  }

  vkGetPhysicalDeviceProperties(physical_device_, &vk_device_properties_);
}

VulkanDevice::~VulkanDevice() {
  // Note: Ideally whoever allocated the buffer & image should be responsible
  // for deallocation as well.
  // These manual deallocations work as last resort for the case where we
  // have GGUI window whose lifetime is controlled by Python but
  // shares the same underlying VulkanDevice with Program. In an extreme
  // edge case when Python shuts down and program gets destructed before
  // GGUI Window, buffers and images allocated through GGUI window won't
  // be properly deallocated before VulkanDevice destruction. This isn't
  // the most proper fix but is less intrusive compared to other
  // approaches.
  if (backend_calls_safe()) {
    const VkResult wait_result = vkDeviceWaitIdle(device_);
    if (wait_result != VK_SUCCESS) {
      try {
        BackendRuntimeError error(
            Arch::vulkan, static_cast<std::int64_t>(wait_result),
            "vkDeviceWaitIdle", "Vulkan device wait failed during teardown");
        report_backend_error(error);
      } catch (...) {
        // A destructor must not replace the first backend failure.
      }
    }
  }

  InteropDeviceReleaseCallback interop_device_release = nullptr;
  {
    std::lock_guard<std::mutex> lock(interop_cleanup_mutex_);
    interop_device_release = interop_device_release_;
  }
  if (interop_device_release != nullptr) {
    interop_device_release(this);
  }

  allocations_.clear();
  image_allocations_.clear();

  compute_streams_.reset();
  graphics_streams_.reset();

  renderpass_pools_.clear();
  desc_set_cache_.clear();
  desc_set_cache_lru_.clear();
  desc_set_layouts_.clear();
  desc_pool_ = nullptr;
  image_samplers_.clear();

  vmaDestroyAllocator(allocator_);
  vmaDestroyAllocator(allocator_export_);
}

RhiResult VulkanDevice::create_pipeline_cache(
    PipelineCache **out_cache,
    size_t initial_size,
    const void *initial_data) noexcept {
  *out_cache = nullptr;
  try {
    auto *cache = new VulkanPipelineCache(this, initial_size, initial_data);
    if (!cache->is_valid()) {
      delete cache;
      return RhiResult::error;
    }
    *out_cache = cache;
  } catch (std::bad_alloc &) {
    return RhiResult::out_of_memory;
  }
  return RhiResult::success;
}

RhiResult VulkanDevice::create_pipeline(Pipeline **out_pipeline,
                                        const PipelineSourceDesc &src,
                                        std::string name,
                                        PipelineCache *cache) noexcept {
  if (src.type != PipelineSourceType::spirv_binary ||
      src.stage != PipelineStageType::compute) {
    return RhiResult::invalid_usage;
  }

  if (src.data == nullptr || src.size == 0) {
    RHI_LOG_ERROR("pipeline source cannot be empty");
    return RhiResult::invalid_usage;
  }

  SpirvCodeView code;
  code.data = (uint32_t *)src.data;
  code.size = src.size;
  code.stage = VK_SHADER_STAGE_COMPUTE_BIT;

  VulkanPipeline::Params params;
  params.code = {code};
  params.device = this;
  params.name = name;
  params.cache =
      cache ? static_cast<VulkanPipelineCache *>(cache)->vk_pipeline_cache()
            : nullptr;

  try {
    *out_pipeline = new VulkanPipeline(params);
  } catch (std::invalid_argument &e) {
    *out_pipeline = nullptr;
    RHI_LOG_ERROR(e.what());
    return RhiResult::invalid_usage;
  } catch (std::runtime_error &e) {
    *out_pipeline = nullptr;
    RHI_LOG_ERROR(e.what());
    return RhiResult::error;
  } catch (std::bad_alloc &e) {
    *out_pipeline = nullptr;
    RHI_LOG_ERROR(e.what());
    return RhiResult::out_of_memory;
  }

  return RhiResult::success;
}

void VulkanDevice::set_default_pipeline_cache(PipelineCache *cache) noexcept {
  default_pipeline_cache_ = cache;
}

RhiResult VulkanDevice::allocate_memory(const AllocParams &params,
                                        DeviceAllocation *out_devalloc) {
  AllocationInternal &alloc = allocations_.acquire();
  alloc.generation =
      allocation_generation_counter_.fetch_add(1, std::memory_order_relaxed);
  alloc.usage = params.usage;

  RHI_ASSERT(params.size > 0);

  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.pNext = nullptr;
  buffer_info.size = params.size;
  // FIXME: How to express this in a backend-neutral way?
  buffer_info.usage =
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  if (int(params.usage & AllocUsage::Storage)) {
    buffer_info.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  }
  if (int(params.usage & AllocUsage::Uniform)) {
    buffer_info.usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  }
  if (int(params.usage & AllocUsage::Vertex)) {
    buffer_info.usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  }
  if (int(params.usage & AllocUsage::Index)) {
    buffer_info.usage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
  }
  if (int(params.usage & AllocUsage::Indirect)) {
    buffer_info.usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
  }
  if (int(params.usage & AllocUsage::Conditional)) {
    buffer_info.usage |= VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT;
  }
  if (int(params.usage & AllocUsage::AccelerationStructureBuildInput)) {
    buffer_info.usage |=
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
  }
  if (int(params.usage & AllocUsage::AccelerationStructureStorage)) {
    buffer_info.usage |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
  }
  const bool needs_device_address =
      int(params.usage & AllocUsage::DeviceAddress) ||
      int(params.usage & AllocUsage::AccelerationStructureBuildInput) ||
      int(params.usage & AllocUsage::AccelerationStructureStorage);
  if (needs_device_address) {
    if (!vk_caps().buffer_device_address) {
      allocations_.release(&alloc);
      return RhiResult::not_supported;
    }
    buffer_info.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  }

  uint32_t queue_family_indices[] = {compute_queue_family_index_,
                                     graphics_queue_family_index_};

  if (compute_queue_family_index_ == graphics_queue_family_index_) {
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  } else {
    buffer_info.sharingMode = VK_SHARING_MODE_CONCURRENT;
    buffer_info.queueFamilyIndexCount = 2;
    buffer_info.pQueueFamilyIndices = queue_family_indices;
  }

  VkExternalMemoryBufferCreateInfo external_mem_buffer_create_info = {};
  external_mem_buffer_create_info.sType =
      VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
  external_mem_buffer_create_info.pNext = nullptr;

#if defined(_WIN32) || defined(_WIN64)
  external_mem_buffer_create_info.handleTypes =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
  external_mem_buffer_create_info.handleTypes =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif

  bool export_sharing = params.export_sharing && vk_caps().external_memory;

  VmaAllocationCreateInfo alloc_info{};
  if (export_sharing) {
    alloc_info.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    buffer_info.pNext = &external_mem_buffer_create_info;
  }
#ifdef __APPLE__
  // weird behavior on apple: these flags are needed even if either read or
  // write is required
  if (params.host_read || params.host_write) {
#else
  if (params.host_read && params.host_write) {
#endif  //__APPLE__
    // This should be the unified memory on integrated GPUs
    alloc_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    alloc_info.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
#ifdef __APPLE__
    // weird behavior on apple: if coherent bit is not set, then the memory
    // writes between map() and unmap() cannot be seen by gpu
    alloc_info.preferredFlags |= VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
#endif  //__APPLE__
  } else if (params.host_read) {
    alloc_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    alloc_info.preferredFlags = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  } else if (params.host_write) {
    alloc_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    if (int(params.usage & AllocUsage::Upload)) {
      alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    } else {
      alloc_info.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    }
  } else {
    alloc_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  }

  if (get_caps().get(DeviceCapability::spirv_has_physical_storage_buffer) &&
      ((buffer_info.usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) ||
       (buffer_info.usage &
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR) ||
       (buffer_info.usage &
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR) ||
       (buffer_info.usage & VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR))) {
    buffer_info.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_KHR;
  }

  alloc.buffer = vkapi::create_buffer(
      device_, export_sharing ? allocator_export_ : allocator_, &buffer_info,
      &alloc_info);
  if (alloc.buffer == nullptr) {
    return RhiResult::out_of_memory;
  }

  vmaGetAllocationInfo(alloc.buffer->allocator, alloc.buffer->allocation,
                       &alloc.alloc_info);
  alloc.host_read = params.host_read;
  alloc.host_write = params.host_write;
  alloc.mapped = nullptr;
  alloc.mapped_offset = 0;
  alloc.mapped_size = VK_WHOLE_SIZE;

  if (get_caps().get(DeviceCapability::spirv_has_physical_storage_buffer) ||
      needs_device_address) {
    VkBufferDeviceAddressInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR;
    info.buffer = alloc.buffer->buffer;
    info.pNext = nullptr;
    alloc.addr = vkGetBufferDeviceAddressKHR(device_, &info);
  }

  *out_devalloc = DeviceAllocation{this, (uint64_t)&alloc};
  return RhiResult::success;
}

bool VulkanDevice::image_blit_supported(BufferFormat source_format,
                                        BufferFormat destination_format,
                                        bool linear_filter) const {
  auto [source_result, source_vk_format] =
      buffer_format_ti_to_vk(source_format);
  auto [destination_result, destination_vk_format] =
      buffer_format_ti_to_vk(destination_format);
  if (source_result != RhiResult::success ||
      destination_result != RhiResult::success) {
    return false;
  }
  VkFormatProperties source_properties{};
  VkFormatProperties destination_properties{};
  vkGetPhysicalDeviceFormatProperties(physical_device_, source_vk_format,
                                      &source_properties);
  vkGetPhysicalDeviceFormatProperties(physical_device_, destination_vk_format,
                                      &destination_properties);
  const VkFormatFeatureFlags source_required =
      VK_FORMAT_FEATURE_BLIT_SRC_BIT |
      (linear_filter ? VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT : 0);
  return (source_properties.optimalTilingFeatures & source_required) ==
             source_required &&
         (destination_properties.optimalTilingFeatures &
          VK_FORMAT_FEATURE_BLIT_DST_BIT) != 0;
}

VkDeviceAddress VulkanDevice::get_buffer_device_address(
    DeviceAllocation handle) const {
  return get_alloc_internal(handle).addr;
}

RhiResult VulkanDevice::map_internal(AllocationInternal &alloc_int,
                                     size_t offset,
                                     size_t size,
                                     void **mapped_ptr) {
  if (alloc_int.mapped != nullptr) {
    RHI_LOG_ERROR("Memory can not be mapped multiple times");
    return RhiResult::invalid_usage;
  }

  if (size != VK_WHOLE_SIZE && alloc_int.alloc_info.size < offset + size) {
    RHI_LOG_ERROR("Mapping out of range");
    return RhiResult::invalid_usage;
  }

  VkResult res;
  if (alloc_int.buffer->allocator) {
    res = vmaMapMemory(alloc_int.buffer->allocator,
                       alloc_int.buffer->allocation, &alloc_int.mapped);
    if (res == VK_SUCCESS && alloc_int.host_read) {
      vmaInvalidateAllocation(alloc_int.buffer->allocator,
                              alloc_int.buffer->allocation, offset, size);
    }
    alloc_int.mapped = (uint8_t *)(alloc_int.mapped) + offset;
  } else {
    res = vkMapMemory(device_, alloc_int.alloc_info.deviceMemory,
                      alloc_int.alloc_info.offset + offset, size, 0,
                      &alloc_int.mapped);
    if (res == VK_SUCCESS && alloc_int.host_read) {
      const VkDeviceSize atom =
          vk_device_properties_.limits.nonCoherentAtomSize;
      const VkDeviceSize mapped_begin = alloc_int.alloc_info.offset + offset;
      const VkDeviceSize mapped_end =
          size == VK_WHOLE_SIZE
              ? alloc_int.alloc_info.offset + alloc_int.alloc_info.size
              : mapped_begin + size;
      const VkDeviceSize range_begin = mapped_begin / atom * atom;
      const VkDeviceSize range_end =
          std::min((mapped_end + atom - 1) / atom * atom,
                   alloc_int.alloc_info.offset + alloc_int.alloc_info.size);
      VkMappedMemoryRange range{};
      range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
      range.memory = alloc_int.alloc_info.deviceMemory;
      range.offset = range_begin;
      range.size = range_end - range_begin;
      vkInvalidateMappedMemoryRanges(device_, 1, &range);
    }
  }

  if (alloc_int.mapped == nullptr || res == VK_ERROR_MEMORY_MAP_FAILED) {
    RHI_LOG_ERROR(
        "cannot map memory, potentially because the memory is not "
        "accessible from the host: ensure your memory is allocated with "
        "`host_read=true` or `host_write=true` (or `host_access=true` in C++ "
        "wrapper)");
    return RhiResult::invalid_usage;
  } else if (res != VK_SUCCESS) {
    std::array<char, 256> msg_buf;
    RHI_DEBUG_SNPRINTF(
        msg_buf.data(), msg_buf.size(),
        "failed to map memory for unknown reasons. VkResult = %d", res);
    RHI_LOG_ERROR(msg_buf.data());
    return RhiResult::error;
  }

  *mapped_ptr = alloc_int.mapped;
  alloc_int.mapped_offset = offset;
  alloc_int.mapped_size = size;

  return RhiResult::success;
}

void VulkanDevice::dealloc_memory(DeviceAllocation handle) {
  const uint64_t generation = allocation_generation(handle);
  InteropAllocationReleaseCallback interop_allocation_release = nullptr;
  {
    std::lock_guard<std::mutex> lock(interop_cleanup_mutex_);
    interop_allocation_release = interop_allocation_release_;
  }
  if (interop_allocation_release != nullptr) {
    interop_allocation_release(this, handle.alloc_id, generation);
  }
  allocations_.release(&get_alloc_internal(handle));
}

ShaderResourceSet *VulkanDevice::create_resource_set() {
  return new VulkanResourceSet(this);
}

RasterResources *VulkanDevice::create_raster_resources() {
  return new VulkanRasterResources(this);
}

uint64_t VulkanDevice::get_memory_physical_pointer(DeviceAllocation handle) {
  return uint64_t(get_alloc_internal(handle).addr);
}

RhiResult VulkanDevice::map_range(DevicePtr ptr,
                                  uint64_t size,
                                  void **mapped_ptr) {
  return map_internal(get_alloc_internal(ptr), ptr.offset, size, mapped_ptr);
}

RhiResult VulkanDevice::map(DeviceAllocation alloc, void **mapped_ptr) {
  return map_internal(get_alloc_internal(alloc), 0, VK_WHOLE_SIZE, mapped_ptr);
}

void VulkanDevice::unmap(DevicePtr ptr) {
  return this->VulkanDevice::unmap(DeviceAllocation(ptr));
}

void VulkanDevice::unmap(DeviceAllocation alloc) {
  AllocationInternal &alloc_int = get_alloc_internal(alloc);

  if (alloc_int.mapped == nullptr) {
    RHI_LOG_ERROR("Unmapping memory that is not mapped");
    return;
  }

  if (alloc_int.buffer->allocator) {
    if (alloc_int.host_write) {
      vmaFlushAllocation(alloc_int.buffer->allocator,
                         alloc_int.buffer->allocation,
                         alloc_int.mapped_offset, alloc_int.mapped_size);
    }
    vmaUnmapMemory(alloc_int.buffer->allocator, alloc_int.buffer->allocation);
  } else {
    if (alloc_int.host_write) {
      const VkDeviceSize atom =
          vk_device_properties_.limits.nonCoherentAtomSize;
      const VkDeviceSize mapped_begin =
          alloc_int.alloc_info.offset + alloc_int.mapped_offset;
      const VkDeviceSize mapped_end =
          alloc_int.mapped_size == VK_WHOLE_SIZE
              ? alloc_int.alloc_info.offset + alloc_int.alloc_info.size
              : mapped_begin + alloc_int.mapped_size;
      const VkDeviceSize range_begin = mapped_begin / atom * atom;
      const VkDeviceSize range_end =
          std::min((mapped_end + atom - 1) / atom * atom,
                   alloc_int.alloc_info.offset + alloc_int.alloc_info.size);
      VkMappedMemoryRange range{};
      range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
      range.memory = alloc_int.alloc_info.deviceMemory;
      range.offset = range_begin;
      range.size = range_end - range_begin;
      vkFlushMappedMemoryRanges(device_, 1, &range);
    }
    vkUnmapMemory(device_, alloc_int.alloc_info.deviceMemory);
  }

  alloc_int.mapped = nullptr;
  alloc_int.mapped_offset = 0;
  alloc_int.mapped_size = VK_WHOLE_SIZE;
}

RhiResult VulkanDevice::upload_data(DevicePtr *device_ptr,
                                    const void **data,
                                    size_t *size,
                                    int num_alloc) noexcept {
  if (!device_ptr || !data || !size || num_alloc < 0) {
    return RhiResult::invalid_usage;
  }
  if (num_alloc == 0) {
    return RhiResult::success;
  }

  std::vector<DeviceAllocationUnique> stagings;
  stagings.reserve(num_alloc);
  for (int i = 0; i < num_alloc; i++) {
    if (device_ptr[i].device != this || !data[i]) {
      return RhiResult::invalid_usage;
    }
    auto [staging, res] = allocate_memory_unique(
        {size[i], /*host_write=*/true, /*host_read=*/false,
         /*export_sharing=*/false, AllocUsage::Upload});
    if (res != RhiResult::success) {
      return res;
    }

    void *mapped{nullptr};
    res = map(*staging, &mapped);
    if (res != RhiResult::success) {
      return res;
    }
    std::memcpy(mapped, data[i], size[i]);
    unmap(*staging);
    stagings.push_back(std::move(staging));
  }

  Stream *stream = get_compute_stream();
  auto [cmdlist, res] = stream->new_command_list_unique();
  if (res != RhiResult::success) {
    return res;
  }
  for (int i = 0; i < num_alloc; i++) {
    cmdlist->buffer_copy(device_ptr[i], stagings[i]->get_ptr(0), size[i]);
  }
  stream->submit_synced(cmdlist.get());
  return RhiResult::success;
}

RhiResult VulkanDevice::readback_data(
    DevicePtr *device_ptr,
    void **data,
    size_t *size,
    int num_alloc,
    const std::vector<StreamSemaphore> &wait_sema) noexcept {
  if (!device_ptr || !data || !size || num_alloc < 0) {
    return RhiResult::invalid_usage;
  }
  if (num_alloc == 0) {
    return RhiResult::success;
  }

  Stream *stream = get_compute_stream();
  auto [cmdlist, res] = stream->new_command_list_unique();
  if (res != RhiResult::success) {
    return res;
  }

  std::vector<DeviceAllocationUnique> stagings;
  stagings.reserve(num_alloc);
  for (int i = 0; i < num_alloc; i++) {
    if (device_ptr[i].device != this || !data[i]) {
      return RhiResult::invalid_usage;
    }
    auto [staging, alloc_res] = allocate_memory_unique(
        {size[i], /*host_write=*/false, /*host_read=*/true,
         /*export_sharing=*/false, AllocUsage::None});
    if (alloc_res != RhiResult::success) {
      return alloc_res;
    }

    cmdlist->buffer_copy(staging->get_ptr(0), device_ptr[i], size[i]);
    stagings.push_back(std::move(staging));
  }
  stream->submit_synced(cmdlist.get(), wait_sema);

  for (int i = 0; i < num_alloc; i++) {
    void *mapped{nullptr};
    res = map(*stagings[i], &mapped);
    if (res != RhiResult::success) {
      return res;
    }
    std::memcpy(data[i], mapped, size[i]);
    unmap(*stagings[i]);
  }
  return RhiResult::success;
}

void VulkanDevice::memcpy_internal(DevicePtr dst,
                                   DevicePtr src,
                                   uint64_t size) {
  // Compute queues are guaranteed to support transfer commands. A dedicated
  // transfer family would also require queue-family ownership transitions and
  // a measured copy/compute overlap policy; keep small internal copies on the
  // existing externally synchronized compute queue until such a workload is
  // demonstrated.
  Stream *stream = get_compute_stream();
  auto [cmd, res] = stream->new_command_list_unique();
  TI_ASSERT(res == RhiResult::success);
  cmd->buffer_copy(dst, src, size);
  stream->submit_synced(cmd.get());
}

Stream *VulkanDevice::get_compute_stream() {
  const auto streams = compute_streams_;
  const auto thread_id = std::this_thread::get_id();
  VulkanStream *result = nullptr;
  bool inserted = false;
  {
    std::lock_guard<std::mutex> lock(streams->map_mutex);
    auto iter = streams->map.find(thread_id);
    if (iter == streams->map.end()) {
      auto stream = std::make_unique<VulkanStream>(
          *this, compute_queue_, compute_queue_family_index_);
      result = stream.get();
      streams->map.emplace(thread_id, std::move(stream));
      inserted = true;
    } else {
      result = iter->second.get();
    }
  }
  if (inserted) {
    streams->register_thread_exit(streams, thread_id);
  }
  return result;
}

uint32_t VulkanDevice::queue_timestamp_valid_bits(
    uint32_t queue_family_index) const {
  uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &count, nullptr);
  if (queue_family_index >= count) {
    return 0;
  }
  std::vector<VkQueueFamilyProperties> properties(count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &count,
                                           properties.data());
  return properties[queue_family_index].timestampValidBits;
}

void VulkanCommandList::begin_profiler_scope(const std::string &kernel_name) {
  auto pool = vkapi::create_query_pool(ti_device_->vk_device());
  vkCmdResetQueryPool(buffer_->buffer, pool->query_pool, 0, 2);
  vkCmdWriteTimestamp(buffer_->buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                      pool->query_pool, 0);
  profiler_scopes_.push_back({kernel_name, pool});
}

void VulkanCommandList::end_profiler_scope() {
  RHI_ASSERT(!profiler_scopes_.empty() &&
             "Profiler scope ended without a matching begin");
  auto scope = std::move(profiler_scopes_.back());
  profiler_scopes_.pop_back();
  auto pool = scope.query_pool;
  ti_device_->profiler_reserve_samplers(1);
  vkCmdWriteTimestamp(buffer_->buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                      pool->query_pool, 1);
  buffer_->refs.push_back(pool);
  completed_profiler_samplers_.push_back(
      {std::move(scope.kernel_name), pool, nullptr});
  ++profiler_sampler_reservations_;
}

void VulkanCommandList::write_runtime_timestamp(
    const vkapi::IVkQueryPool &query_pool,
    std::uint32_t query,
    VkPipelineStageFlagBits stage,
    bool reset) {
  TI_ERROR_IF(!query_pool, "Vulkan runtime timestamp query pool is null");
  if (reset) {
    vkCmdResetQueryPool(buffer_->buffer, query_pool->query_pool, 0, 2);
  }
  vkCmdWriteTimestamp(buffer_->buffer, stage, query_pool->query_pool, query);
  buffer_->refs.push_back(query_pool);
}

std::vector<VulkanProfilerSampler>
VulkanCommandList::take_completed_profiler_samplers() {
  // Keep the device-level reservations: they become owned by the submitted
  // samplers and are released when their results are collected.
  profiler_sampler_reservations_ = 0;
  return std::move(completed_profiler_samplers_);
}

void VulkanDevice::profiler_collect_samplers(
    std::vector<VulkanProfilerSampler> samplers) {
  if (samplers.empty()) {
    return;
  }

  std::vector<std::pair<std::string, double>> records;
  records.reserve(samplers.size());
  for (auto &sampler : samplers) {
    TI_ASSERT(sampler.query_pool != nullptr);
    auto query_pool = sampler.query_pool->query_pool;

    uint64_t t[2];
    const VkResult query_result = vkGetQueryPoolResults(
        vk_device(), query_pool, 0, 2, sizeof(uint64_t) * 2, &t,
        sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
    if (query_result != VK_SUCCESS) {
      raise_backend_error(static_cast<std::int64_t>(query_result),
                          "vkGetQueryPoolResults",
                          "Failed to get Vulkan profiler query results");
    }
    double duration_ms =
        (t[1] - t[0]) * vk_device_properties_.limits.timestampPeriod /
        1000000.0;
    records.push_back(std::make_pair(std::move(sampler.kernel_name),
                                     duration_ms));
  }

  {
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    TI_ASSERT(profiler_pending_sampler_count_ >= samplers.size());
    profiler_pending_sampler_count_ -= samplers.size();
    sampled_records_.insert(sampled_records_.end(),
                            std::make_move_iterator(records.begin()),
                            std::make_move_iterator(records.end()));
  }
}

void VulkanDevice::profiler_sync_fences(
    const std::vector<vkapi::IVkFence> &fences) {
  if (fences.empty()) {
    return;
  }

  std::vector<VulkanProfilerSampler> completed_samplers;
  {
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    std::vector<VulkanProfilerSampler> still_pending;
    still_pending.reserve(samplers_.size());
    for (auto &sampler : samplers_) {
      bool completed = std::find(fences.begin(), fences.end(), sampler.fence) !=
                       fences.end();
      if (completed) {
        completed_samplers.push_back(std::move(sampler));
      } else {
        still_pending.push_back(std::move(sampler));
      }
    }
    samplers_ = std::move(still_pending);
  }
  profiler_collect_samplers(std::move(completed_samplers));
}

void VulkanDevice::profiler_sync() {
  std::vector<VulkanProfilerSampler> samplers;
  {
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    samplers.swap(samplers_);
  }

  std::vector<vkapi::IVkFence> fences;
  fences.reserve(samplers.size());
  for (const auto &sampler : samplers) {
    TI_ASSERT(sampler.fence != nullptr);
    if (std::find(fences.begin(), fences.end(), sampler.fence) ==
        fences.end()) {
      fences.push_back(sampler.fence);
    }
  }
  for (const auto &fence : fences) {
    std::lock_guard<std::mutex> lock(fence->mutex);
    VkResult result = VK_SUCCESS;
    {
      ScopedBackendWaitTelemetry wait_scope(&backend_wait_telemetry_);
      result = vkWaitForFences(fence->device, /*fenceCount=*/1,
                               &fence->fence, VK_TRUE, UINT64_MAX);
    }
    if (result != VK_SUCCESS) {
      raise_backend_error(static_cast<std::int64_t>(result),
                          "vkWaitForFences",
                          "Failed to wait for a Vulkan profiler fence");
    }
  }

  profiler_collect_samplers(std::move(samplers));
}

std::vector<std::pair<std::string, double>>
VulkanDevice::profiler_flush_sampled_time() {
  std::vector<std::pair<std::string, double>> records;
  {
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    records.swap(sampled_records_);
  }
  return records;
}

Stream *VulkanDevice::get_graphics_stream() {
  const auto streams = graphics_streams_;
  const auto thread_id = std::this_thread::get_id();
  VulkanStream *result = nullptr;
  bool inserted = false;
  {
    std::lock_guard<std::mutex> lock(streams->map_mutex);
    auto iter = streams->map.find(thread_id);
    if (iter == streams->map.end()) {
      auto stream = std::make_unique<VulkanStream>(
          *this, graphics_queue_, graphics_queue_family_index_);
      result = stream.get();
      streams->map.emplace(thread_id, std::move(stream));
      inserted = true;
    } else {
      result = iter->second.get();
    }
  }
  if (inserted) {
    streams->register_thread_exit(streams, thread_id);
  }
  return result;
}

void VulkanDevice::wait_idle() {
  std::scoped_lock lock(compute_streams_->map_mutex,
                        graphics_streams_->map_mutex);
  for (auto &entry : compute_streams_->map) {
    entry.second->command_sync();
  }
  for (auto &entry : graphics_streams_->map) {
    entry.second->command_sync();
  }
}

std::pair<size_t, size_t> VulkanDevice::debug_stream_cache_counts() {
  std::scoped_lock lock(compute_streams_->map_mutex,
                        graphics_streams_->map_mutex);
  return {compute_streams_->map.size(), graphics_streams_->map.size()};
}

std::unique_lock<std::mutex> VulkanDevice::acquire_queue_lock(VkQueue queue) {
  TI_ASSERT(queue != VK_NULL_HANDLE);
  // Compute and graphics can alias queue index 0. Resolve compute first so
  // aliased handles share one mutex and one telemetry source while distinct
  // queues remain independent.
  if (queue == compute_queue_) {
    return compute_queue_lock_telemetry_.acquire(compute_queue_mutex_);
  }
  TI_ASSERT(queue == graphics_queue_);
  return graphics_queue_lock_telemetry_.acquire(graphics_queue_mutex_);
}

VulkanRuntimeTelemetrySnapshot VulkanDevice::runtime_telemetry_snapshot()
    const noexcept {
  const auto compute = compute_queue_lock_telemetry_.snapshot();
  const auto graphics = graphics_queue_lock_telemetry_.snapshot();
  return {
      backend_wait_telemetry_.snapshot(),
      {
          compute.sampled_acquisitions + graphics.sampled_acquisitions,
          compute.contended_acquisitions + graphics.contended_acquisitions,
          compute.sampled_wait_ns + graphics.sampled_wait_ns,
      },
  };
}

void VulkanCommandList::set_raster_viewport_and_scissor(int x0,
                                                         int y0,
                                                         int x1,
                                                         int y1) {
  RHI_ASSERT(x0 >= 0 && y0 >= 0 && x1 > x0 && y1 > y0);
  viewport_x_ = x0;
  viewport_y_ = y0;
  viewport_width_ = static_cast<uint32_t>(x1 - x0);
  viewport_height_ = static_cast<uint32_t>(y1 - y0);
  if (current_pipeline_ != nullptr && current_pipeline_->is_graphics()) {
    apply_raster_viewport_and_scissor();
  }
}

void VulkanCommandList::apply_raster_viewport_and_scissor() {
  VkViewport viewport{};
  viewport.x = static_cast<float>(viewport_x_);
  viewport.y = static_cast<float>(viewport_y_);
  viewport.width = static_cast<float>(viewport_width_);
  viewport.height = static_cast<float>(viewport_height_);
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  VkRect2D scissor{/*offset=*/{viewport_x_, viewport_y_},
                   /*extent=*/{viewport_width_, viewport_height_}};
  vkCmdSetViewport(buffer_->buffer, 0, 1, &viewport);
  vkCmdSetScissor(buffer_->buffer, 0, 1, &scissor);
}

VulkanQueueSubmissionSnapshot
VulkanDevice::queue_submission_snapshot() const noexcept {
  return {
      queue_submit_calls_.load(std::memory_order_relaxed),
      submitted_command_buffers_.load(std::memory_order_relaxed),
      batched_queue_submit_calls_.load(std::memory_order_relaxed),
      batched_command_buffers_.load(std::memory_order_relaxed),
  };
}

bool VulkanStreamSemaphoreObject::is_ready() const {
  if (!fence_ref) {
    return false;
  }
  if (fault_reporter_) {
    fault_reporter_->throw_if_submission_disallowed("Vulkan fence query");
  }
  std::lock_guard<std::mutex> lock(fence_ref->mutex);
  VkResult res = vkGetFenceStatus(fence_ref->device, fence_ref->fence);
  if (res == VK_SUCCESS) {
    return true;
  }
  if (res == VK_NOT_READY) {
    return false;
  }
  BackendRuntimeError error(Arch::vulkan, static_cast<std::int64_t>(res),
                            "vkGetFenceStatus",
                            "Failed to query Vulkan fence status");
  if (fault_reporter_) {
    fault_reporter_->report_backend_error(error, 0);
  }
  throw error;
}

bool VulkanStreamSemaphoreObject::wait() const {
  if (!fence_ref) {
    return false;
  }
  if (fault_reporter_) {
    fault_reporter_->throw_if_submission_disallowed("Vulkan fence wait");
  }
  std::lock_guard<std::mutex> lock(fence_ref->mutex);
  VkResult result = VK_SUCCESS;
  {
    ScopedBackendWaitTelemetry wait_scope(wait_telemetry_);
    result = vkWaitForFences(fence_ref->device, /*fenceCount=*/1,
                             &fence_ref->fence, VK_TRUE, UINT64_MAX);
  }
  if (result != VK_SUCCESS) {
    BackendRuntimeError error(Arch::vulkan,
                              static_cast<std::int64_t>(result),
                              "vkWaitForFences",
                              "Failed to wait for Vulkan fence");
    if (fault_reporter_) {
      fault_reporter_->report_backend_error(error, 0);
    }
    throw error;
  }
  return true;
}

RhiResult VulkanStream::new_command_list(CommandList **out_cmdlist) noexcept {
  vkapi::IVkCommandBuffer buffer =
      vkapi::allocate_command_buffer(command_pool_);

  if (buffer == nullptr) {
    return RhiResult::out_of_memory;
  }

  *out_cmdlist = new VulkanCommandList(&device_, this, buffer);
  return RhiResult::success;
}

std::unique_ptr<CommandList> VulkanStream::new_secondary_command_list() {
  auto buffer = vkapi::allocate_command_buffer(
      command_pool_, VK_COMMAND_BUFFER_LEVEL_SECONDARY);
  TI_ERROR_IF(!buffer, "Vulkan secondary command allocation failed");
  return std::make_unique<VulkanCommandList>(&device_, this, std::move(buffer));
}

void VulkanStream::retire_completed_cmdbuffers() {
  if (submitted_cmdbuffers_.empty()) {
    return;
  }
  std::vector<vkapi::IVkFence> completed_fences;
  std::vector<TrackedCmdbuf> still_submitted;
  still_submitted.reserve(submitted_cmdbuffers_.size());
  for (auto &tracked : submitted_cmdbuffers_) {
    std::lock_guard<std::mutex> lock(tracked.fence->mutex);
    VkResult res = vkGetFenceStatus(tracked.fence->device,
                                    tracked.fence->fence);
    if (res == VK_SUCCESS) {
      completed_fences.push_back(tracked.fence);
      continue;
    }
    if (res == VK_NOT_READY) {
      still_submitted.push_back(std::move(tracked));
      continue;
    }
    device_.raise_backend_error(static_cast<std::int64_t>(res),
                                "vkGetFenceStatus",
                                "Failed to retire a Vulkan command buffer");
  }
  submitted_cmdbuffers_ = std::move(still_submitted);
  // Profiler samplers are tracked separately from command buffers. If an
  // already-complete submission is retired here, command_sync() can no longer
  // discover its fence, so collect its query results before dropping that
  // ownership edge.
  device_.profiler_sync_fences(completed_fences);
}
void VulkanStream::apply_in_flight_backpressure() {
  if (submitted_cmdbuffers_.size() <
      kVulkanMaxInFlightCommandBuffersPerStream) {
    return;
  }

  // A producer can otherwise enqueue faster than the device completes work
  // forever, retaining one command buffer, fence, semaphore and resource set
  // per submission. Waiting for only the oldest fence keeps normal async
  // execution unchanged while bounding that host-side backlog.
  const auto &oldest_fence = submitted_cmdbuffers_.front().fence;
  VkResult result = VK_SUCCESS;
  {
    std::lock_guard<std::mutex> lock(oldest_fence->mutex);
    ScopedBackendWaitTelemetry wait_scope(device_.backend_wait_telemetry());
    result = vkWaitForFences(oldest_fence->device, /*fenceCount=*/1,
                             &oldest_fence->fence, VK_TRUE, UINT64_MAX);
  }
  if (result != VK_SUCCESS) {
    device_.raise_backend_error(
        static_cast<std::int64_t>(result), "vkWaitForFences",
        "Failed to apply Vulkan in-flight submission backpressure");
  }
  retire_completed_cmdbuffers();
  TI_ASSERT(submitted_cmdbuffers_.size() <
            kVulkanMaxInFlightCommandBuffersPerStream);
}

StreamSemaphore VulkanStream::submit(
    CommandList *cmdlist_,
    const std::vector<StreamSemaphore> &wait_semaphores) {
  return submit_with_semaphores(cmdlist_, wait_semaphores, {});
}

StreamSemaphore VulkanStream::submit_with_semaphores(
    CommandList *cmdlist_,
    const std::vector<StreamSemaphore> &wait_semaphores,
    const std::vector<StreamSemaphore> &signal_semaphores) {
  device_.throw_if_backend_submission_disallowed("Vulkan queue submit");
  std::lock_guard<std::mutex> submission_lock(submission_mutex_);
  VulkanCommandList *cmdlist = static_cast<VulkanCommandList *>(cmdlist_);
  vkapi::IVkCommandBuffer buffer = cmdlist->finalize();
  auto profiler_samplers = cmdlist->take_completed_profiler_samplers();

  /*
  if (in_flight_cmdlists_.find(buffer) != in_flight_cmdlists_.end()) {
    TI_ERROR("Can not submit command list that is still in-flight");
    return;
  }
  */

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &buffer->buffer;

  std::vector<VkSemaphore> vk_wait_semaphores;
  std::vector<VkPipelineStageFlags> vk_wait_stages;
  std::vector<vkapi::IDeviceObj> submit_refs;

  for (const StreamSemaphore &sema_ : wait_semaphores) {
    auto sema = std::static_pointer_cast<VulkanStreamSemaphoreObject>(sema_);
    vk_wait_semaphores.push_back(sema->vkapi_ref->semaphore);
    vk_wait_stages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    submit_refs.push_back(sema->vkapi_ref);
  }

  submit_info.pWaitSemaphores = vk_wait_semaphores.data();
  submit_info.waitSemaphoreCount = vk_wait_semaphores.size();
  submit_info.pWaitDstStageMask = vk_wait_stages.data();

  auto semaphore = vkapi::create_semaphore(buffer->device, 0);
  submit_refs.push_back(semaphore);

  std::vector<VkSemaphore> vk_signal_semaphores;
  vk_signal_semaphores.reserve(signal_semaphores.size() + 1);
  vk_signal_semaphores.push_back(semaphore->semaphore);
  for (const StreamSemaphore &sema_ : signal_semaphores) {
    auto sema = std::static_pointer_cast<VulkanStreamSemaphoreObject>(sema_);
    TI_ERROR_IF(!sema || !sema->vkapi_ref,
                "Vulkan submission received an invalid signal semaphore");
    vk_signal_semaphores.push_back(sema->vkapi_ref->semaphore);
    submit_refs.push_back(sema->vkapi_ref);
  }
  submit_info.signalSemaphoreCount =
      static_cast<uint32_t>(vk_signal_semaphores.size());
  submit_info.pSignalSemaphores = vk_signal_semaphores.data();

  if (submission_batch_depth_ != 0) {
    if (!submission_batch_fence_) {
      submission_batch_fence_ = vkapi::create_fence(buffer->device, 0);
    }
    auto completion = std::make_shared<VulkanStreamSemaphoreObject>(
        device_.backend_fault_reporter(), semaphore,
        submission_batch_fence_, device_.backend_wait_telemetry());
    pending_batch_submissions_.push_back(
        PendingBatchSubmission{
            buffer,
            std::move(vk_wait_semaphores),
            std::move(vk_wait_stages),
            std::move(vk_signal_semaphores),
            std::move(submit_refs),
            std::move(profiler_samplers)});
    submission_batch_completion_ = completion;
    return completion;
  }

  auto fence = vkapi::create_fence(buffer->device, 0);

  // Resource tracking, check previously submitted commands
  retire_completed_cmdbuffers();
  apply_in_flight_backpressure();

  VkResult submit_result = VK_SUCCESS;
  {
    auto queue_lock = device_.acquire_queue_lock(queue_);
    submit_result = vkQueueSubmit(queue_, /*submitCount=*/1, &submit_info,
                                  /*fence=*/fence->fence);
  }
  if (submit_result != VK_SUCCESS) {
    device_.raise_backend_error(
        static_cast<std::int64_t>(submit_result), "vkQueueSubmit",
        "Vulkan queue submission failed");
  }
  device_.queue_submit_calls_.fetch_add(1, std::memory_order_relaxed);
  device_.submitted_command_buffers_.fetch_add(
      1, std::memory_order_relaxed);
  submitted_cmdbuffers_.push_back(
      TrackedCmdbuf{fence, {buffer}, std::move(submit_refs)});
  for (auto &sampler : profiler_samplers) {
    sampler.fence = fence;
  }
  device_.profiler_add_samplers(std::move(profiler_samplers));
  return std::make_shared<VulkanStreamSemaphoreObject>(
      device_.backend_fault_reporter(), semaphore, fence,
      device_.backend_wait_telemetry());
}

void VulkanStream::begin_submission_batch() {
  device_.throw_if_backend_submission_disallowed(
      "Vulkan submission batch begin");
  std::lock_guard<std::mutex> submission_lock(submission_mutex_);
  if (submission_batch_depth_++ == 0) {
    TI_ASSERT(pending_batch_submissions_.empty());
    submission_batch_fence_.reset();
    submission_batch_completion_.reset();
  }
}

StreamSemaphore VulkanStream::end_submission_batch() {
  device_.throw_if_backend_submission_disallowed(
      "Vulkan submission batch end");
  std::lock_guard<std::mutex> submission_lock(submission_mutex_);
  TI_ERROR_IF(submission_batch_depth_ == 0,
              "Vulkan submission batch is not active");
  if (--submission_batch_depth_ != 0) {
    return nullptr;
  }
  if (pending_batch_submissions_.empty()) {
    submission_batch_fence_.reset();
    return std::exchange(submission_batch_completion_, nullptr);
  }

  std::vector<VkSubmitInfo> submit_infos(
      pending_batch_submissions_.size());
  for (std::size_t i = 0; i < pending_batch_submissions_.size(); ++i) {
    auto &pending = pending_batch_submissions_[i];
    auto &info = submit_infos[i];
    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    info.commandBufferCount = 1;
    info.pCommandBuffers = &pending.buffer->buffer;
    info.waitSemaphoreCount =
        static_cast<std::uint32_t>(pending.wait_semaphores.size());
    info.pWaitSemaphores = pending.wait_semaphores.data();
    info.pWaitDstStageMask = pending.wait_stages.data();
    info.signalSemaphoreCount =
        static_cast<std::uint32_t>(pending.signal_semaphores.size());
    info.pSignalSemaphores = pending.signal_semaphores.data();
  }

  retire_completed_cmdbuffers();
  apply_in_flight_backpressure();
  VkResult submit_result = VK_SUCCESS;
  {
    auto queue_lock = device_.acquire_queue_lock(queue_);
    submit_result = vkQueueSubmit(
        queue_, static_cast<std::uint32_t>(submit_infos.size()),
        submit_infos.data(), submission_batch_fence_->fence);
  }
  if (submit_result != VK_SUCCESS) {
    pending_batch_submissions_.clear();
    submission_batch_fence_.reset();
    submission_batch_completion_.reset();
    device_.raise_backend_error(
        static_cast<std::int64_t>(submit_result), "vkQueueSubmit",
        "Vulkan batched queue submission failed");
  }
  device_.queue_submit_calls_.fetch_add(1, std::memory_order_relaxed);
  device_.submitted_command_buffers_.fetch_add(
      pending_batch_submissions_.size(), std::memory_order_relaxed);
  device_.batched_queue_submit_calls_.fetch_add(
      1, std::memory_order_relaxed);
  device_.batched_command_buffers_.fetch_add(
      pending_batch_submissions_.size(), std::memory_order_relaxed);

  TrackedCmdbuf tracked;
  tracked.fence = submission_batch_fence_;
  tracked.buffers.reserve(pending_batch_submissions_.size());
  for (auto &pending : pending_batch_submissions_) {
    tracked.buffers.push_back(std::move(pending.buffer));
    tracked.submit_refs.insert(
        tracked.submit_refs.end(),
        std::make_move_iterator(pending.submit_refs.begin()),
        std::make_move_iterator(pending.submit_refs.end()));
    for (auto &sampler : pending.profiler_samplers) {
      sampler.fence = submission_batch_fence_;
    }
    device_.profiler_add_samplers(
        std::move(pending.profiler_samplers));
  }
  submitted_cmdbuffers_.push_back(std::move(tracked));
  pending_batch_submissions_.clear();
  submission_batch_fence_.reset();
  return std::exchange(submission_batch_completion_, nullptr);
}

StreamSemaphore VulkanStream::submit_synced(
    CommandList *cmdlist,
    const std::vector<StreamSemaphore> &wait_semaphores) {
  auto sema = submit(cmdlist, wait_semaphores);
  command_sync();
  return sema;
}

void VulkanStream::command_sync() {
  device_.throw_if_backend_submission_disallowed("Vulkan stream wait");
  std::vector<vkapi::IVkFence> fences;
  {
    std::lock_guard<std::mutex> submission_lock(submission_mutex_);
    TI_ERROR_IF(submission_batch_depth_ != 0,
                "Cannot synchronize a Vulkan stream inside an active "
                "submission batch");
    fences.reserve(submitted_cmdbuffers_.size());
    for (const auto &tracked : submitted_cmdbuffers_) {
      fences.push_back(tracked.fence);
    }
    for (const auto &fence : fences) {
      std::lock_guard<std::mutex> lock(fence->mutex);
      VkResult result = VK_SUCCESS;
      {
        ScopedBackendWaitTelemetry wait_scope(
            device_.backend_wait_telemetry());
        result = vkWaitForFences(fence->device, /*fenceCount=*/1,
                                 &fence->fence, VK_TRUE, UINT64_MAX);
      }
      if (result != VK_SUCCESS) {
        device_.raise_backend_error(
            static_cast<std::int64_t>(result), "vkWaitForFences",
            "Failed to wait for a Vulkan stream fence");
      }
    }
    retire_completed_cmdbuffers();
  }
  // The fences above make only this stream's query results available. Do not
  // drain or wait for profiler work submitted by another stream.
  device_.profiler_sync_fences(fences);
}

std::size_t VulkanStream::debug_in_flight_command_buffer_count() {
  std::lock_guard<std::mutex> submission_lock(submission_mutex_);
  retire_completed_cmdbuffers();
  std::size_t count = 0;
  for (const auto &submission : submitted_cmdbuffers_) {
    count += submission.buffers.size();
  }
  return count;
}

std::unique_ptr<Pipeline> VulkanDevice::create_raster_pipeline(
    const std::vector<PipelineSourceDesc> &src,
    const RasterParams &raster_params,
    const std::vector<VertexInputBinding> &vertex_inputs,
    const std::vector<VertexInputAttribute> &vertex_attrs,
    std::string name) {
  VulkanPipeline::Params params;
  params.code = {};
  params.device = this;
  params.name = name;
  params.cache = default_pipeline_cache_
                     ? static_cast<VulkanPipelineCache *>(
                           default_pipeline_cache_)->vk_pipeline_cache()
                     : nullptr;

  for (auto &src_desc : src) {
    SpirvCodeView &code = params.code.emplace_back();
    code.data = (uint32_t *)src_desc.data;
    code.size = src_desc.size;
    code.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    if (src_desc.stage == PipelineStageType::fragment) {
      code.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    } else if (src_desc.stage == PipelineStageType::vertex) {
      code.stage = VK_SHADER_STAGE_VERTEX_BIT;
    } else if (src_desc.stage == PipelineStageType::geometry) {
      code.stage = VK_SHADER_STAGE_GEOMETRY_BIT;
    } else if (src_desc.stage == PipelineStageType::tesselation_control) {
      code.stage = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
    } else if (src_desc.stage == PipelineStageType::tesselation_eval) {
      code.stage = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
    } else if (src_desc.stage == PipelineStageType::task) {
      code.stage = VK_SHADER_STAGE_TASK_BIT_EXT;
    } else if (src_desc.stage == PipelineStageType::mesh) {
      code.stage = VK_SHADER_STAGE_MESH_BIT_EXT;
    }
  }

  return std::make_unique<VulkanPipeline>(params, raster_params, vertex_inputs,
                                          vertex_attrs);
}

std::unique_ptr<Surface> VulkanDevice::create_surface(
    const SurfaceConfig &config) {
  return std::make_unique<VulkanSurface>(this, config);
}

std::tuple<VkDeviceMemory, size_t, size_t>
VulkanDevice::get_vkmemory_offset_size(const DeviceAllocation &alloc) const {
  auto &buffer_alloc = get_alloc_internal(alloc);
  return std::make_tuple(buffer_alloc.alloc_info.deviceMemory,
                         buffer_alloc.alloc_info.offset,
                         buffer_alloc.alloc_info.size);
}

vkapi::IVkBuffer VulkanDevice::get_vkbuffer(
    const DeviceAllocation &alloc) const {
  const AllocationInternal &alloc_int = get_alloc_internal(alloc);

  return alloc_int.buffer;
}

size_t VulkanDevice::get_vkbuffer_size(const DeviceAllocation &alloc) const {
  const AllocationInternal &alloc_int = get_alloc_internal(alloc);

  return alloc_int.alloc_info.size;
}

std::tuple<vkapi::IVkImage, vkapi::IVkImageView, VkFormat>
VulkanDevice::get_vk_image(const DeviceAllocation &alloc) const {
  const ImageAllocInternal &alloc_int = get_image_alloc_internal(alloc);

  return std::make_tuple(alloc_int.image, alloc_int.view,
                         alloc_int.image->format);
}

vkapi::IVkFramebuffer VulkanDevice::get_framebuffer(
    const VulkanFramebufferDesc &desc) {
  // We won't pool framebuffer and resuse it, as doing so requires hashing the
  // referenced IVkImageView objects, which might destruct unless we hold strong
  // references. Thus doing so is way too ugly, and Vulkan is moving towards
  // dynamic rendering anyways.
  vkapi::IVkFramebuffer framebuffer = vkapi::create_framebuffer(
      0, desc.renderpass, desc.attachments, desc.width, desc.height, 1);

  return framebuffer;
}

vkapi::IVkSampler VulkanDevice::get_sampler(
    const ImageSamplerConfig &config) {
  std::lock_guard<std::mutex> lock(descriptor_mutex_);
  for (const auto &[key, sampler] : image_samplers_) {
    if (key == config) {
      return sampler;
    }
  }

  const auto to_filter = [](ImageFilter filter) {
    return filter == ImageFilter::nearest ? VK_FILTER_NEAREST
                                          : VK_FILTER_LINEAR;
  };
  const auto to_address_mode = [](ImageAddressMode mode) {
    switch (mode) {
      case ImageAddressMode::repeat:
        return VK_SAMPLER_ADDRESS_MODE_REPEAT;
      case ImageAddressMode::mirrored_repeat:
        return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
      case ImageAddressMode::clamp_to_edge:
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    }
    TI_NOT_IMPLEMENTED;
  };

  VkSamplerCreateInfo sampler_info{};
  sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  sampler_info.magFilter = to_filter(config.mag_filter);
  sampler_info.minFilter = to_filter(config.min_filter);
  sampler_info.addressModeU = to_address_mode(config.address_mode_u);
  sampler_info.addressModeV = to_address_mode(config.address_mode_v);
  sampler_info.addressModeW = to_address_mode(config.address_mode_w);
  sampler_info.anisotropyEnable = VK_FALSE;
  sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  sampler_info.unnormalizedCoordinates = VK_FALSE;
  sampler_info.compareEnable = VK_FALSE;
  sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
  sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;

  auto sampler = vkapi::create_sampler(device_, sampler_info);
  image_samplers_.emplace_back(config, sampler);
  return sampler;
}

vkapi::IVkSampler VulkanDevice::get_default_sampler() {
  return get_sampler({});
}

DeviceAllocation VulkanDevice::import_vkbuffer(vkapi::IVkBuffer buffer,
                                               size_t size,
                                               VkDeviceMemory memory,
                                               VkDeviceSize offset,
                                               AllocUsage usage) {
  AllocationInternal &alloc_int = allocations_.acquire();
  alloc_int.generation =
      allocation_generation_counter_.fetch_add(1, std::memory_order_relaxed);

  alloc_int.external = true;
  alloc_int.usage = usage;
  alloc_int.buffer = buffer;
  alloc_int.mapped = nullptr;
  const bool import_needs_device_address =
      int(usage & AllocUsage::DeviceAddress) ||
      int(usage & AllocUsage::AccelerationStructureBuildInput) ||
      int(usage & AllocUsage::AccelerationStructureStorage);
  if (get_caps().get(DeviceCapability::spirv_has_physical_storage_buffer) ||
      import_needs_device_address) {
    VkBufferDeviceAddressInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    info.buffer = buffer->buffer;
    info.pNext = nullptr;
    alloc_int.addr = vkGetBufferDeviceAddress(device_, &info);
  }

  alloc_int.alloc_info.size = size;
  alloc_int.alloc_info.deviceMemory = memory;
  alloc_int.alloc_info.offset = offset;

  return DeviceAllocation{this, reinterpret_cast<uint64_t>(&alloc_int)};
}

DeviceAllocation VulkanDevice::import_vk_image(vkapi::IVkImage image,
                                               vkapi::IVkImageView view,
                                               VkImageLayout layout) {
  ImageAllocInternal &alloc_int = image_allocations_.acquire();

  alloc_int.external = true;
  alloc_int.image = image;
  alloc_int.view = view;
  alloc_int.view_lods.emplace_back(view);

  return DeviceAllocation{this, reinterpret_cast<uint64_t>(&alloc_int)};
}

vkapi::IVkImageView VulkanDevice::get_vk_imageview(
    const DeviceAllocation &alloc) const {
  return std::get<1>(get_vk_image(alloc));
}

vkapi::IVkImageView VulkanDevice::get_vk_lod_imageview(
    const DeviceAllocation &alloc,
    int lod) const {
  return get_image_alloc_internal(alloc).view_lods[lod];
}

DeviceAllocation VulkanDevice::create_image(const ImageParams &params) {
  ImageAllocInternal &alloc = image_allocations_.acquire();

  int num_mip_levels = 1;

  bool is_depth = params.format == BufferFormat::depth16 ||
                  params.format == BufferFormat::depth24stencil8 ||
                  params.format == BufferFormat::depth32f;

  VkImageCreateInfo image_info{};
  image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_info.pNext = nullptr;
  if (params.dimension == ImageDimension::d1D) {
    image_info.imageType = VK_IMAGE_TYPE_1D;
  } else if (params.dimension == ImageDimension::d2D) {
    image_info.imageType = VK_IMAGE_TYPE_2D;
  } else if (params.dimension == ImageDimension::d3D) {
    image_info.imageType = VK_IMAGE_TYPE_3D;
  }
  image_info.extent.width = params.x;
  image_info.extent.height = params.y;
  image_info.extent.depth = params.z;
  image_info.mipLevels = num_mip_levels;
  image_info.arrayLayers = 1;
  auto [result, vk_format] = buffer_format_ti_to_vk(params.format);
  assert(result == RhiResult::success);
  image_info.format = vk_format;
  image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
  image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  image_info.usage =
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  if (params.usage & ImageAllocUsage::Sampled) {
    image_info.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
  }

  if (is_depth) {
    if (params.usage & ImageAllocUsage::Storage) {
      image_info.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    }
    if (params.usage & ImageAllocUsage::Attachment) {
      image_info.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    }
  } else {
    if (params.usage & ImageAllocUsage::Storage) {
      image_info.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    }
    if (params.usage & ImageAllocUsage::Attachment) {
      image_info.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    }
  }
  image_info.samples = VK_SAMPLE_COUNT_1_BIT;

  uint32_t queue_family_indices[] = {compute_queue_family_index_,
                                     graphics_queue_family_index_};

  if (compute_queue_family_index_ == graphics_queue_family_index_) {
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  } else {
    image_info.sharingMode = VK_SHARING_MODE_CONCURRENT;
    image_info.queueFamilyIndexCount = 2;
    image_info.pQueueFamilyIndices = queue_family_indices;
  }

  bool export_sharing = params.export_sharing && vk_caps_.external_memory;

  VkExternalMemoryImageCreateInfo external_mem_image_create_info = {};
  if (export_sharing) {
    external_mem_image_create_info.sType =
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    external_mem_image_create_info.pNext = nullptr;

#if defined(_WIN32) || defined(_WIN64)
    external_mem_image_create_info.handleTypes =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    external_mem_image_create_info.handleTypes =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif
    image_info.pNext = &external_mem_image_create_info;
  }

  VmaAllocationCreateInfo alloc_info{};
  if (params.export_sharing) {
    alloc_info.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
  }
  alloc_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;

  alloc.image = vkapi::create_image(
      device_, export_sharing ? allocator_export_ : allocator_, &image_info,
      &alloc_info);
  vmaGetAllocationInfo(alloc.image->allocator, alloc.image->allocation,
                       &alloc.alloc_info);

  VkImageViewCreateInfo view_info{};
  view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.pNext = nullptr;
  if (params.dimension == ImageDimension::d1D) {
    view_info.viewType = VK_IMAGE_VIEW_TYPE_1D;
  } else if (params.dimension == ImageDimension::d2D) {
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  } else if (params.dimension == ImageDimension::d3D) {
    view_info.viewType = VK_IMAGE_VIEW_TYPE_3D;
  }
  view_info.format = image_info.format;
  view_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.subresourceRange.aspectMask =
      is_depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
  view_info.subresourceRange.baseMipLevel = 0;
  view_info.subresourceRange.levelCount = num_mip_levels;
  view_info.subresourceRange.baseArrayLayer = 0;
  view_info.subresourceRange.layerCount = 1;

  alloc.view = vkapi::create_image_view(device_, alloc.image, &view_info);

  for (int i = 0; i < num_mip_levels; i++) {
    view_info.subresourceRange.baseMipLevel = i;
    view_info.subresourceRange.levelCount = 1;
    alloc.view_lods.push_back(
        vkapi::create_image_view(device_, alloc.image, &view_info));
  }

  DeviceAllocation handle{this, reinterpret_cast<uint64_t>(&alloc)};

  if (params.initial_layout != ImageLayout::undefined) {
    image_transition(handle, ImageLayout::undefined, params.initial_layout);
  }

  return handle;
}

void VulkanDevice::destroy_image(DeviceAllocation handle) {
  image_allocations_.release(&get_image_alloc_internal(handle));
}

vkapi::IVkRenderPass VulkanDevice::get_renderpass(
    const VulkanRenderPassDesc &desc) {
  std::lock_guard<std::mutex> lock(renderpass_mutex_);
  if (renderpass_pools_.find(desc) != renderpass_pools_.end()) {
    return renderpass_pools_.at(desc);
  }

  std::vector<VkAttachmentDescription> attachments;
  std::vector<VkAttachmentReference> color_attachments;

  VkAttachmentReference depth_attachment{};

  uint32_t i = 0;
  for (auto &[format, clear] : desc.color_attachments) {
    VkAttachmentDescription &description = attachments.emplace_back();
    description.flags = 0;
    description.format = format;
    description.samples = VK_SAMPLE_COUNT_1_BIT;
    description.loadOp =
        clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
    description.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    description.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    description.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    description.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    description.finalLayout = desc.color_final_layout;

    VkAttachmentReference &ref = color_attachments.emplace_back();
    ref.attachment = i;
    ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    i += 1;
  }

  if (desc.depth_attachment != VK_FORMAT_UNDEFINED) {
    VkAttachmentDescription &description = attachments.emplace_back();
    description.flags = 0;
    description.format = desc.depth_attachment;
    description.samples = VK_SAMPLE_COUNT_1_BIT;
    description.loadOp = desc.clear_depth ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                          : VK_ATTACHMENT_LOAD_OP_LOAD;
    description.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    description.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    description.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    description.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    description.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    depth_attachment.attachment = i;
    depth_attachment.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  }

  VkSubpassDescription subpass{};
  subpass.flags = 0;
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.inputAttachmentCount = 0;
  subpass.pInputAttachments = nullptr;
  subpass.colorAttachmentCount = color_attachments.size();
  subpass.pColorAttachments = color_attachments.data();
  subpass.pResolveAttachments = nullptr;
  subpass.pDepthStencilAttachment = desc.depth_attachment == VK_FORMAT_UNDEFINED
                                        ? nullptr
                                        : &depth_attachment;
  subpass.preserveAttachmentCount = 0;
  subpass.pPreserveAttachments = nullptr;

  VkRenderPassCreateInfo renderpass_info{};
  renderpass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderpass_info.pNext = nullptr;
  renderpass_info.flags = 0;
  renderpass_info.attachmentCount = attachments.size();
  renderpass_info.pAttachments = attachments.data();
  renderpass_info.subpassCount = 1;
  renderpass_info.pSubpasses = &subpass;
  renderpass_info.dependencyCount = 0;
  renderpass_info.pDependencies = nullptr;

  vkapi::IVkRenderPass renderpass =
      vkapi::create_render_pass(device_, &renderpass_info);

  renderpass_pools_.insert({desc, renderpass});

  return renderpass;
}

vkapi::IVkDescriptorSetLayout VulkanDevice::get_desc_set_layout(
    VulkanResourceSet &set) {
  std::lock_guard<std::mutex> lock(descriptor_mutex_);
  auto it = desc_set_layouts_.find(set);
  if (it != desc_set_layouts_.end()) {
    return it->second;
  }

  std::vector<VkDescriptorSetLayoutBinding> bindings;
  std::vector<VkDescriptorBindingFlags> binding_flags;
  bool has_update_after_bind_binding = false;
  for (const auto &pair : set.get_bindings()) {
    const auto descriptor_count =
        VulkanResourceSet::descriptor_count(pair.second);
    if (descriptor_count == 0) {
      RHI_LOG_ERROR("Descriptor array bindings must not be empty");
      return nullptr;
    }
    bindings.push_back(VkDescriptorSetLayoutBinding{
        /*binding=*/pair.first, pair.second.type, descriptor_count,
        VK_SHADER_STAGE_ALL,
        /*pImmutableSamplers=*/nullptr});
    const bool is_buffer_array =
        std::holds_alternative<VulkanResourceSet::BufferArray>(
            pair.second.res);
    const bool is_patchable_buffer =
        pair.second.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
        pair.second.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    const auto flags =
        ((is_buffer_array &&
          vk_caps().descriptor_storage_buffer_update_after_bind) ||
         (!is_buffer_array && vk_caps().descriptor_update_after_bind &&
          is_patchable_buffer))
            ? VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
                  (vk_caps().descriptor_binding_partially_bound
                       ? VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT
                       : VkDescriptorBindingFlags{0})
            : VkDescriptorBindingFlags{0};
    binding_flags.push_back(flags);
    has_update_after_bind_binding |= flags != 0;
  }

  VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags_info{};
  binding_flags_info.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
  binding_flags_info.bindingCount = binding_flags.size();
  binding_flags_info.pBindingFlags = binding_flags.data();

  VkDescriptorSetLayoutCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  create_info.pNext =
      has_update_after_bind_binding ? &binding_flags_info : nullptr;
  create_info.flags =
      has_update_after_bind_binding
          ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT
          : VkDescriptorSetLayoutCreateFlags{0};
  create_info.bindingCount = bindings.size();
  create_info.pBindings = bindings.data();

  auto layout = vkapi::create_descriptor_set_layout(device_, &create_info);
  auto inserted = desc_set_layouts_.emplace(set, layout);
  return inserted.first->second;
}

vkapi::IVkDescriptorSet VulkanDevice::find_cached_desc_set(
    const VulkanResourceSet &set) {
  std::lock_guard<std::mutex> lock(descriptor_mutex_);
  if (!descriptor_set_cache_enabled_) {
    return nullptr;
  }
  auto it = desc_set_cache_.find(set);
  if (it == desc_set_cache_.end()) {
    ++desc_set_cache_misses_;
    return nullptr;
  }
  ++desc_set_cache_hits_;
  if (should_touch_desc_set_cache_lru_locked()) {
    auto &entry = it->second;
    if (entry.has_lru_entry) {
      desc_set_cache_lru_.splice(desc_set_cache_lru_.end(),
                                 desc_set_cache_lru_, entry.lru_it);
    } else {
      desc_set_cache_lru_.push_back(it->first);
      entry.lru_it = std::prev(desc_set_cache_lru_.end());
      entry.has_lru_entry = true;
    }
  }
  return it->second.set;
}

void VulkanDevice::cache_desc_set(const VulkanResourceSet &set,
                                  vkapi::IVkDescriptorSet desc_set) {
  std::lock_guard<std::mutex> lock(descriptor_mutex_);
  if (!descriptor_set_cache_enabled_ || !desc_set) {
    return;
  }
  auto existing = desc_set_cache_.find(set);
  if (existing != desc_set_cache_.end()) {
    existing->second.set = desc_set;
    if (should_touch_desc_set_cache_lru_locked()) {
      auto &entry = existing->second;
      if (entry.has_lru_entry) {
        desc_set_cache_lru_.splice(desc_set_cache_lru_.end(),
                                   desc_set_cache_lru_, entry.lru_it);
      } else {
        desc_set_cache_lru_.push_back(existing->first);
        entry.lru_it = std::prev(desc_set_cache_lru_.end());
        entry.has_lru_entry = true;
      }
    }
    return;
  }
  if (desc_set_cache_.size() >= desc_set_cache_capacity_) {
    if (descriptor_set_cache_lru_) {
      if (desc_set_cache_lru_.empty() && !desc_set_cache_.empty()) {
        desc_set_cache_evictions_ += desc_set_cache_.size();
        desc_set_cache_.clear();
      }
      size_t evict_count = desc_set_cache_capacity_ / 4u;
      if (evict_count == 0) {
        evict_count = 1;
      }
      while (evict_count-- > 0 && !desc_set_cache_lru_.empty()) {
        desc_set_cache_.erase(desc_set_cache_lru_.front());
        desc_set_cache_lru_.pop_front();
        ++desc_set_cache_evictions_;
      }
    } else {
      desc_set_cache_evictions_ += desc_set_cache_.size();
      desc_set_cache_.clear();
      desc_set_cache_lru_.clear();
    }
  }
  auto [inserted, _] = desc_set_cache_.emplace(set, CachedDescriptorSet{});
  inserted->second.set = desc_set;
  if (descriptor_set_cache_lru_) {
    desc_set_cache_lru_.push_back(inserted->first);
    inserted->second.lru_it = std::prev(desc_set_cache_lru_.end());
    inserted->second.has_lru_entry = true;
  }
}

bool VulkanDevice::should_touch_desc_set_cache_lru_locked() const {
  if (!descriptor_set_cache_lru_) {
    return false;
  }
  const size_t touch_threshold =
      std::max<size_t>(size_t{1}, desc_set_cache_capacity_ * 3u / 4u);
  return desc_set_cache_.size() >= touch_threshold;
}

RhiReturn<vkapi::IVkDescriptorSet> VulkanDevice::alloc_desc_set(
    vkapi::IVkDescriptorSetLayout layout) {
  std::lock_guard<std::mutex> lock(descriptor_pool_mutex_);
  // This returns nullptr if can't allocate (OOM or pool is full)
  vkapi::IVkDescriptorSet set =
      vkapi::allocate_descriptor_sets(desc_pool_, layout);

  if (set == nullptr) {
    RhiResult status = new_descriptor_pool_locked();
    // Allocating new descriptor pool failed
    if (status != RhiResult::success) {
      return {status, nullptr};
    }
    set = vkapi::allocate_descriptor_sets(desc_pool_, layout);
  }

  return {RhiResult::success, set};
}

void VulkanDevice::update_descriptor_sets_locked(
    const std::vector<VkWriteDescriptorSet> &desc_writes) {
  vkUpdateDescriptorSets(device_, desc_writes.size(), desc_writes.data(),
                         /*descriptorCopyCount=*/0,
                         /*pDescriptorCopies=*/nullptr);
}

void VulkanDevice::create_vma_allocator() {
  VmaAllocatorCreateInfo allocatorInfo = {};
  allocatorInfo.vulkanApiVersion = vk_caps().vk_api_version;
  allocatorInfo.physicalDevice = physical_device_;
  allocatorInfo.device = device_;
  allocatorInfo.instance = instance_;

  VmaVulkanFunctions vk_vma_functions{};
  vk_vma_functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
  vk_vma_functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

  // Let VMA grab the functions by itself
  /*
  volkLoadDeviceTable(&table, device_);
  vk_vma_functions.vkGetPhysicalDeviceProperties =
      PFN_vkGetPhysicalDeviceProperties(vkGetInstanceProcAddr(
          volkGetLoadedInstance(), "vkGetPhysicalDeviceProperties"));
  vk_vma_functions.vkGetPhysicalDeviceMemoryProperties =
      PFN_vkGetPhysicalDeviceMemoryProperties(vkGetInstanceProcAddr(
          volkGetLoadedInstance(), "vkGetPhysicalDeviceMemoryProperties"));
  vk_vma_functions.vkAllocateMemory = table.vkAllocateMemory;
  vk_vma_functions.vkFreeMemory = table.vkFreeMemory;
  vk_vma_functions.vkMapMemory = table.vkMapMemory;
  vk_vma_functions.vkUnmapMemory = table.vkUnmapMemory;
  vk_vma_functions.vkFlushMappedMemoryRanges = table.vkFlushMappedMemoryRanges;
  vk_vma_functions.vkInvalidateMappedMemoryRanges =
      table.vkInvalidateMappedMemoryRanges;
  vk_vma_functions.vkBindBufferMemory = table.vkBindBufferMemory;
  vk_vma_functions.vkBindImageMemory = table.vkBindImageMemory;
  vk_vma_functions.vkGetBufferMemoryRequirements =
      table.vkGetBufferMemoryRequirements;
  vk_vma_functions.vkGetImageMemoryRequirements =
      table.vkGetImageMemoryRequirements;
  vk_vma_functions.vkCreateBuffer = table.vkCreateBuffer;
  vk_vma_functions.vkDestroyBuffer = table.vkDestroyBuffer;
  vk_vma_functions.vkCreateImage = table.vkCreateImage;
  vk_vma_functions.vkDestroyImage = table.vkDestroyImage;
  vk_vma_functions.vkCmdCopyBuffer = table.vkCmdCopyBuffer;
  vk_vma_functions.vkGetBufferMemoryRequirements2KHR =
      table.vkGetBufferMemoryRequirements2KHR;
  vk_vma_functions.vkGetImageMemoryRequirements2KHR =
      table.vkGetImageMemoryRequirements2KHR;
  vk_vma_functions.vkBindBufferMemory2KHR = table.vkBindBufferMemory2KHR;
  vk_vma_functions.vkBindImageMemory2KHR = table.vkBindImageMemory2KHR;
  vk_vma_functions.vkGetPhysicalDeviceMemoryProperties2KHR =
      (PFN_vkGetPhysicalDeviceMemoryProperties2KHR)(std::max(
          vkGetInstanceProcAddr(volkGetLoadedInstance(),
                                "vkGetPhysicalDeviceMemoryProperties2KHR"),
          vkGetInstanceProcAddr(volkGetLoadedInstance(),
                                "vkGetPhysicalDeviceMemoryProperties2")));
  vk_vma_functions.vkGetDeviceBufferMemoryRequirements =
      table.vkGetDeviceBufferMemoryRequirements;
  vk_vma_functions.vkGetDeviceImageMemoryRequirements =
      table.vkGetDeviceImageMemoryRequirements;
  */

  allocatorInfo.pVulkanFunctions = &vk_vma_functions;

  if (get_caps().get(DeviceCapability::spirv_has_physical_storage_buffer) ||
      vk_caps().buffer_device_address) {
    allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
  }

  vmaCreateAllocator(&allocatorInfo, &allocator_);

  VkPhysicalDeviceMemoryProperties properties;
  vkGetPhysicalDeviceMemoryProperties(physical_device_, &properties);

  std::vector<VkExternalMemoryHandleTypeFlags> flags(
      properties.memoryTypeCount);

  for (int i = 0; i < properties.memoryTypeCount; i++) {
    auto flag = properties.memoryTypes[i].propertyFlags;
    if (flag & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
#if defined(_WIN32) || defined(_WIN64)
      flags[i] = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
      flags[i] = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
    } else {
      flags[i] = 0;
    }
  }

  allocatorInfo.pTypeExternalMemoryHandleTypes = flags.data();

  vmaCreateAllocator(&allocatorInfo, &allocator_export_);
}

RhiResult VulkanDevice::new_descriptor_pool_locked() {
  std::vector<VkDescriptorPoolSize> pool_sizes{
      {VK_DESCRIPTOR_TYPE_SAMPLER, 64},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 256},
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 256},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},
      {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 256},
      {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 256},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 256},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 512},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 128},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 128},
      {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 128}};
  if (vk_caps().acceleration_structure) {
    pool_sizes.push_back(
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 64});
  }
  VkDescriptorPoolCreateInfo pool_info = {};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  if (vk_caps().descriptor_update_after_bind ||
      vk_caps().descriptor_storage_buffer_update_after_bind) {
    pool_info.flags |= VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
  }
  pool_info.maxSets = 64;
  pool_info.poolSizeCount = pool_sizes.size();
  pool_info.pPoolSizes = pool_sizes.data();
  auto new_desc_pool = vkapi::create_descriptor_pool(device_, &pool_info);

  if (!new_desc_pool) {
    return RhiResult::out_of_memory;
  }

  desc_pool_ = new_desc_pool;

  return RhiResult::success;
}

VkPresentModeKHR choose_swap_present_mode(
    const std::vector<VkPresentModeKHR> &available_present_modes,
    bool vsync,
    bool adaptive,
    bool fifo_latest_ready_supported) {
  (void)vsync;
  (void)adaptive;
  auto find_present_mode = [&](VkPresentModeKHR mode) {
    for (const auto &available_present_mode : available_present_modes) {
      if (available_present_mode == mode) {
        return true;
      }
    }
    return false;
  };

  if (find_present_mode(VK_PRESENT_MODE_MAILBOX_KHR)) {
    return VK_PRESENT_MODE_MAILBOX_KHR;
  }
  if (fifo_latest_ready_supported &&
      find_present_mode(VK_PRESENT_MODE_FIFO_LATEST_READY_KHR)) {
    return VK_PRESENT_MODE_FIFO_LATEST_READY_KHR;
  }
  if (find_present_mode(VK_PRESENT_MODE_FIFO_KHR)) {
    return VK_PRESENT_MODE_FIFO_KHR;
  }
  if (find_present_mode(VK_PRESENT_MODE_IMMEDIATE_KHR)) {
    return VK_PRESENT_MODE_IMMEDIATE_KHR;
  }

  if (available_present_modes.size() == 0) {
    throw std::runtime_error("no avialble present modes");
  }

  return available_present_modes[0];
}

uint32_t choose_swapchain_image_count(
    const VkSurfaceCapabilitiesKHR &capabilities) {
  uint32_t requested_image_count =
      std::max<uint32_t>(capabilities.minImageCount, 4);
  if (capabilities.maxImageCount == 0) {
    return requested_image_count;
  }
  return std::min<uint32_t>(capabilities.maxImageCount, requested_image_count);
}

VulkanSurface::VulkanSurface(VulkanDevice *device, const SurfaceConfig &config)
    : config_(config), device_(device) {
  width_ = config.width;
  height_ = config.height;

  if (config.native_surface_handle) {
    surface_ = (VkSurfaceKHR)config.native_surface_handle;

    create_swap_chain();
    swapchain_needs_recreate_ = swapchain_ == VK_NULL_HANDLE;
  } else {
    create_offscreen_images();
  }
}

void VulkanSurface::create_offscreen_images() {
  ImageParams params = {ImageDimension::d2D,
                        BufferFormat::rgba8,
                        ImageLayout::present_src,
                        width_,
                        height_,
                        1,
                        false};
  image_format_ = BufferFormat::rgba8;
  swapchain_images_.push_back(device_->create_image(params));
  swapchain_images_.push_back(device_->create_image(params));
}

void VulkanSurface::destroy_offscreen_images() {
  for (auto &img : swapchain_images_) {
    device_->destroy_image(img);
  }
  swapchain_images_.clear();
}

void VulkanSurface::create_swap_chain() {
  auto choose_surface_format =
      [](const std::vector<VkSurfaceFormatKHR> &availableFormats) {
        for (const auto &availableFormat : availableFormats) {
          if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM &&
              availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return availableFormat;
          }
        }
        return availableFormats[0];
      };

  VkSurfaceCapabilitiesKHR capabilities;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device_->vk_physical_device(),
                                            surface_, &capabilities);
  VkBool32 supported = false;
  vkGetPhysicalDeviceSurfaceSupportKHR(device_->vk_physical_device(),
                                       device_->graphics_queue_family_index(),
                                       surface_, &supported);

  if (!supported) {
    RHI_LOG_ERROR("Selected queue does not support presenting");
    return;
  }

  uint32_t formatCount;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device_->vk_physical_device(), surface_,
                                       &formatCount, nullptr);
  std::vector<VkSurfaceFormatKHR> surface_formats(formatCount);
  vkGetPhysicalDeviceSurfaceFormatsKHR(device_->vk_physical_device(), surface_,
                                       &formatCount, surface_formats.data());

  VkSurfaceFormatKHR surface_format = choose_surface_format(surface_formats);

  uint32_t present_mode_count;
  std::vector<VkPresentModeKHR> present_modes;
  vkGetPhysicalDeviceSurfacePresentModesKHR(
      device_->vk_physical_device(), surface_, &present_mode_count, nullptr);

  if (present_mode_count != 0) {
    present_modes.resize(present_mode_count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(device_->vk_physical_device(),
                                              surface_, &present_mode_count,
                                              present_modes.data());
  }
  VkPresentModeKHR present_mode =
      choose_swap_present_mode(
          present_modes, config_.vsync, config_.adaptive,
          device_->vk_caps().present_mode_fifo_latest_ready);

  VkExtent2D extent = {uint32_t(width_), uint32_t(height_)};
  extent.width =
      std::max(capabilities.minImageExtent.width,
               std::min(capabilities.maxImageExtent.width, extent.width));
  extent.height =
      std::max(capabilities.minImageExtent.height,
               std::min(capabilities.maxImageExtent.height, extent.height));
  {
    std::array<char, 512> msg_buf;
    RHI_DEBUG_SNPRINTF(msg_buf.data(), msg_buf.size(),
                       "Creating suface of %u x %u, present mode %d",
                       extent.width, extent.height, present_mode);
    RHI_LOG_DEBUG(msg_buf.data());
  }
  VkImageUsageFlags usage =
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

  this->width_ = extent.width;
  this->height_ = extent.height;

  VkSwapchainCreateInfoKHR createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  createInfo.pNext = nullptr;
  createInfo.flags = 0;
  createInfo.surface = surface_;
  uint32_t requested_image_count = choose_swapchain_image_count(capabilities);
  createInfo.minImageCount = requested_image_count;
  createInfo.imageFormat = surface_format.format;
  createInfo.imageColorSpace = surface_format.colorSpace;
  createInfo.imageExtent = extent;
  createInfo.imageArrayLayers = 1;
  createInfo.imageUsage = usage;
  createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  createInfo.queueFamilyIndexCount = 0;
  createInfo.pQueueFamilyIndices = nullptr;
  createInfo.preTransform = capabilities.currentTransform;
  createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  createInfo.presentMode = present_mode;
  createInfo.clipped = VK_TRUE;
  createInfo.oldSwapchain = VK_NULL_HANDLE;

  if (vkCreateSwapchainKHR(device_->vk_device(), &createInfo,
                           kNoVkAllocCallbacks, &swapchain_) != VK_SUCCESS) {
    RHI_LOG_ERROR("Failed to create swapchain");
    return;
  }

  uint32_t num_images;
  vkGetSwapchainImagesKHR(device_->vk_device(), swapchain_, &num_images,
                          nullptr);
  std::vector<VkImage> swapchain_images(num_images);
  vkGetSwapchainImagesKHR(device_->vk_device(), swapchain_, &num_images,
                          swapchain_images.data());

  auto [result, image_format] = buffer_format_vk_to_ti(surface_format.format);
  RHI_ASSERT(result == RhiResult::success);
  image_format_ = image_format;

  for (VkImage img : swapchain_images) {
    vkapi::IVkImage image = vkapi::create_image(
        device_->vk_device(), img, surface_format.format, VK_IMAGE_TYPE_2D,
        VkExtent3D{uint32_t(width_), uint32_t(height_), 1}, 1u, 1u, usage);

    VkImageViewCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    create_info.image = image->image;
    create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    create_info.format = image->format;
    create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    create_info.subresourceRange.baseMipLevel = 0;
    create_info.subresourceRange.levelCount = 1;
    create_info.subresourceRange.baseArrayLayer = 0;
    create_info.subresourceRange.layerCount = 1;

    vkapi::IVkImageView view =
        vkapi::create_image_view(device_->vk_device(), image, &create_info);

    swapchain_images_.push_back(
        device_->import_vk_image(image, view, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR));
  }

  image_available_.clear();
  image_available_.reserve(swapchain_images_.size());
  for (size_t i = 0; i < swapchain_images_.size(); ++i) {
    image_available_.push_back(vkapi::create_semaphore(device_->vk_device(), 0));
  }
  image_available_index_ = 0;
  present_waits_by_image_.clear();
  present_waits_by_image_.resize(swapchain_images_.size());
}

void VulkanSurface::destroy_swap_chain() {
  if (swapchain_ == VK_NULL_HANDLE) {
    return;
  }
  for (auto &alloc : swapchain_images_) {
    std::get<1>(device_->get_vk_image(alloc)) = nullptr;
    device_->destroy_image(alloc);
  }
  image_available_.clear();
  present_waits_by_image_.clear();
  swapchain_images_.clear();
  vkDestroySwapchainKHR(device_->vk_device(), swapchain_, nullptr);
  swapchain_ = VK_NULL_HANDLE;
}

int VulkanSurface::get_image_count() {
  return swapchain_images_.size();
}

bool VulkanSurface::handle_surface_result(VkResult result,
                                          const char *operation) {
  switch (classify_vulkan_surface_result(result)) {
    case VulkanSurfaceResult::kSuccess:
      return true;
    case VulkanSurfaceResult::kSuboptimal:
      swapchain_needs_recreate_ = true;
      return true;
    case VulkanSurfaceResult::kOutOfDate:
      swapchain_needs_recreate_ = true;
      return false;
    case VulkanSurfaceResult::kDeviceLost:
      if (!device_lost_) {
        char message[160];
        std::snprintf(message, sizeof(message), "Vulkan device lost while %s",
                      operation);
        RHI_LOG_ERROR(message);
      }
      device_lost_ = true;
      device_->raise_backend_error(static_cast<std::int64_t>(result),
                                   operation,
                                   "Vulkan surface operation lost the device");
    case VulkanSurfaceResult::kError:
      char message[160];
      std::snprintf(message, sizeof(message),
                    "Vulkan surface %s failed with VkResult %d", operation,
                    static_cast<int>(result));
      RHI_LOG_ERROR(message);
      return false;
  }
  return false;
}

VulkanSurface::~VulkanSurface() {
  if (config_.native_surface_handle) {
    destroy_swap_chain();
  } else {
    destroy_offscreen_images();
  }
}

void VulkanSurface::resize(uint32_t width, uint32_t height) {
  if (device_lost_) {
    return;
  }
  if (config_.native_surface_handle) {
    destroy_swap_chain();
  } else {
    destroy_offscreen_images();
  }
  this->width_ = width;
  this->height_ = height;
  if (config_.native_surface_handle) {
    create_swap_chain();
    swapchain_needs_recreate_ = swapchain_ == VK_NULL_HANDLE;
  } else {
    create_offscreen_images();
  }
}

std::pair<uint32_t, uint32_t> VulkanSurface::get_size() {
  return std::make_pair(width_, height_);
}

StreamSemaphore VulkanSurface::acquire_next_image() {
  SurfaceImage surface_image = acquire_surface_image();
  return surface_image.image_available;
}

SurfaceImage VulkanSurface::acquire_surface_image() {
  SurfaceImage surface_image;
  if (!config_.native_surface_handle) {
    image_index_ = (image_index_ + 1) % uint32_t(swapchain_images_.size());
    surface_image.image = swapchain_images_[image_index_];
    surface_image.image_index = image_index_;
    return surface_image;
  } else {
    device_->throw_if_backend_submission_disallowed(
        "Vulkan swapchain image acquire");
    if (swapchain_needs_recreate_ || device_lost_ ||
        swapchain_ == VK_NULL_HANDLE || image_available_.empty()) {
      return surface_image;
    }
    auto image_available = image_available_[image_available_index_];
    VkResult res = VK_SUCCESS;
    {
      ScopedBackendWaitTelemetry wait_scope(
          device_->backend_wait_telemetry());
      res = vkAcquireNextImageKHR(
          device_->vk_device(), swapchain_, uint64_t(4 * 1e9),
          image_available->semaphore, VK_NULL_HANDLE, &image_index_);
    }
    if (!handle_surface_result(res, "acquiring the next swapchain image")) {
      return surface_image;
    }
    image_available_index_ =
        (image_available_index_ + 1) % uint32_t(image_available_.size());
    surface_image.image_available =
        std::make_shared<VulkanStreamSemaphoreObject>(
            device_->backend_fault_reporter(), image_available, nullptr,
            device_->backend_wait_telemetry());
    surface_image.image = swapchain_images_[image_index_];
    surface_image.image_index = image_index_;
    return surface_image;
  }
}

bool VulkanSurface::try_acquire_surface_image(SurfaceImage *surface_image) {
  if (!config_.native_surface_handle) {
    *surface_image = acquire_surface_image();
    return true;
  }

  device_->throw_if_backend_submission_disallowed(
      "Vulkan swapchain image acquire");

  if (swapchain_needs_recreate_ || device_lost_ ||
      swapchain_ == VK_NULL_HANDLE || image_available_.empty()) {
    return false;
  }

  auto image_available = image_available_[image_available_index_];
  VkResult res = vkAcquireNextImageKHR(
      device_->vk_device(), swapchain_, 0, image_available->semaphore,
      VK_NULL_HANDLE, &image_index_);
  if (res == VK_NOT_READY || res == VK_TIMEOUT) {
    return false;
  }
  if (!handle_surface_result(res, "acquiring the next swapchain image")) {
    return false;
  }

  image_available_index_ =
      (image_available_index_ + 1) % uint32_t(image_available_.size());
  surface_image->image_available =
      std::make_shared<VulkanStreamSemaphoreObject>(
          device_->backend_fault_reporter(), image_available, nullptr,
          device_->backend_wait_telemetry());
  surface_image->image = swapchain_images_[image_index_];
  surface_image->image_index = image_index_;
  return true;
}

std::vector<StreamSemaphore> VulkanSurface::take_present_waits_after_acquire(
    uint32_t image_index) {
  if (image_index >= present_waits_by_image_.size()) {
    return {};
  }
  auto waits = std::move(present_waits_by_image_[image_index]);
  present_waits_by_image_[image_index].clear();
  return waits;
}

DeviceAllocation VulkanSurface::get_target_image() {
  return swapchain_images_[image_index_];
}

BufferFormat VulkanSurface::image_format() {
  return image_format_;
}

void VulkanSurface::present_image(
    const std::vector<StreamSemaphore> &wait_semaphores) {
  if (!config_.native_surface_handle) {
    return;
  }
  SurfaceImage surface_image;
  surface_image.image = swapchain_images_[image_index_];
  surface_image.image_index = image_index_;
  present_surface_image(surface_image, wait_semaphores);
}

void VulkanSurface::present_surface_image(
    const SurfaceImage &surface_image,
    const std::vector<StreamSemaphore> &wait_semaphores) {
  if (!config_.native_surface_handle) {
    return;
  }
  device_->throw_if_backend_submission_disallowed("Vulkan queue present");
  if (device_lost_) {
    return;
  }
  if (swapchain_ == VK_NULL_HANDLE) {
    RHI_LOG_ERROR("Cannot present image without a valid Vulkan swapchain");
    return;
  }

  std::vector<VkSemaphore> vk_wait_semaphores;
  std::vector<StreamSemaphore> tracked_wait_semaphores;

  // Already transitioned to `present_src` at the end of the render pass.
  // device_->image_transition(get_target_image(),
  // ImageLayout::color_attachment,
  //                          ImageLayout::present_src);

  for (const StreamSemaphore &sema_ : wait_semaphores) {
    if (!sema_) {
      continue;
    }
    auto sema = std::static_pointer_cast<VulkanStreamSemaphoreObject>(sema_);
    vk_wait_semaphores.push_back(sema->vkapi_ref->semaphore);
    tracked_wait_semaphores.push_back(sema_);
  }

  VkPresentInfoKHR presentInfo{};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.waitSemaphoreCount = vk_wait_semaphores.size();
  presentInfo.pWaitSemaphores = vk_wait_semaphores.data();
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = &swapchain_;
  presentInfo.pImageIndices = &surface_image.image_index;
  presentInfo.pResults = nullptr;

  VkResult present_result = VK_SUCCESS;
  {
    auto queue_lock =
        device_->acquire_queue_lock(device_->graphics_queue());
    present_result = vkQueuePresentKHR(device_->graphics_queue(), &presentInfo);
  }
  handle_surface_result(present_result, "presenting a swapchain image");
  if (surface_image.image_index < present_waits_by_image_.size()) {
    present_waits_by_image_[surface_image.image_index] =
        std::move(tracked_wait_semaphores);
  }
}

namespace {

std::atomic<std::uint64_t> next_vulkan_timing_stream_id{1};

class VulkanStreamGpuTimingObject final : public StreamGpuTimingObject {
 public:
  VulkanStreamGpuTimingObject(VulkanDevice *device,
                              std::uint32_t timestamp_valid_bits,
                              double timestamp_period_ns,
                              std::uint64_t stream_id)
      : device_(device),
        query_pool_(vkapi::create_query_pool(device->vk_device())),
        timestamp_valid_bits_(timestamp_valid_bits),
        timestamp_period_ns_(timestamp_period_ns),
        stream_id_(stream_id) {
    TI_ASSERT(device_ != nullptr);
    TI_ASSERT(query_pool_ != nullptr);
  }

  const vkapi::IVkQueryPool &query_pool() const {
    return query_pool_;
  }

  void mark_ended() noexcept {
    ended_.store(true, std::memory_order_release);
  }

  StreamGpuTimingSnapshot snapshot() const override {
    StreamGpuTimingSnapshot result;
    result.measurement_path_changed = true;
    result.stream_id = stream_id_;
    if (!ended_.load(std::memory_order_acquire)) {
      result.status = "not_ended";
      return result;
    }
    if (timestamp_valid_bits_ == 0) {
      result.status = "unsupported";
      return result;
    }

    std::uint64_t timestamps[2]{};
    const VkResult query_result = vkGetQueryPoolResults(
        device_->vk_device(), query_pool_->query_pool, 0, 2,
        sizeof(timestamps), timestamps, sizeof(std::uint64_t),
        VK_QUERY_RESULT_64_BIT);
    if (query_result == VK_NOT_READY) {
      result.status = "not_ready";
      return result;
    }
    if (query_result != VK_SUCCESS) {
      device_->raise_backend_error(
          static_cast<std::int64_t>(query_result), "vkGetQueryPoolResults",
          "Failed to read ticket-owned Vulkan GPU timestamps");
    }

    const std::uint64_t mask =
        timestamp_valid_bits_ >= 64
            ? (std::numeric_limits<std::uint64_t>::max)()
            : (std::uint64_t{1} << timestamp_valid_bits_) - 1;
    const std::uint64_t start = timestamps[0] & mask;
    const std::uint64_t end = timestamps[1] & mask;
    const std::uint64_t ticks =
        end >= start ? end - start : (mask - start) + end + 1;
    const long double duration_ns =
        static_cast<long double>(ticks) * timestamp_period_ns_;
    if (duration_ns < 0.0L ||
        duration_ns >
            static_cast<long double>(
                (std::numeric_limits<std::uint64_t>::max)())) {
      result.status = "overflow";
      return result;
    }
    result.available = true;
    result.duration_ns = static_cast<std::uint64_t>(duration_ns + 0.5L);
    result.exact = true;
    result.status = "instrumented_exact";
    // VkQueryPool storage is driver opaque. Never report a fabricated byte
    // count in Forge-owned memory accounting.
    result.driver_owned_bytes_known = false;
    return result;
  }

 private:
  VulkanDevice *device_{nullptr};
  vkapi::IVkQueryPool query_pool_;
  std::uint32_t timestamp_valid_bits_{0};
  long double timestamp_period_ns_{0.0L};
  std::uint64_t stream_id_{0};
  std::atomic<bool> ended_{false};
};

}  // namespace

VulkanStream::VulkanStream(VulkanDevice &device,
                           VkQueue queue,
                           uint32_t queue_family_index)
    : device_(device),
      queue_(queue),
      queue_family_index_(queue_family_index),
      stream_id_(next_vulkan_timing_stream_id.fetch_add(
          1, std::memory_order_relaxed)) {
  command_pool_ = vkapi::create_command_pool(
      device_.vk_device(), VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      queue_family_index);
}

StreamGpuTiming VulkanStream::begin_gpu_timing() {
  TI_ERROR_IF(submission_batch_depth_ == 0,
              "Vulkan GPU timing requires an active submission batch");
  const std::uint32_t valid_bits =
      device_.queue_timestamp_valid_bits(queue_family_index_);
  if (valid_bits == 0) {
    return nullptr;
  }
  auto timing = std::make_shared<VulkanStreamGpuTimingObject>(
      &device_, valid_bits,
      static_cast<double>(
          device_.get_vk_physical_device_props().limits.timestampPeriod),
      stream_id_);
  auto [cmdlist, result] = new_command_list_unique();
  TI_ERROR_IF(result != RhiResult::success,
              "Unable to allocate Vulkan GPU timing begin command list");
  static_cast<VulkanCommandList *>(cmdlist.get())
      ->write_runtime_timestamp(timing->query_pool(), 0,
                                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, true);
  submit(cmdlist.get());
  return timing;
}

StreamGpuTiming VulkanStream::begin_gpu_timing_inline(
    CommandList *cmdlist) {
  TI_ERROR_IF(submission_batch_depth_ == 0,
              "Vulkan GPU timing requires an active submission batch");
  TI_ERROR_IF(cmdlist == nullptr,
              "Vulkan inline GPU timing requires a command list");
  const std::uint32_t valid_bits =
      device_.queue_timestamp_valid_bits(queue_family_index_);
  if (valid_bits == 0) {
    return nullptr;
  }
  auto timing = std::make_shared<VulkanStreamGpuTimingObject>(
      &device_, valid_bits,
      static_cast<double>(
          device_.get_vk_physical_device_props().limits.timestampPeriod),
      stream_id_);
  auto *vulkan_cmdlist = dynamic_cast<VulkanCommandList *>(cmdlist);
  TI_ERROR_IF(vulkan_cmdlist == nullptr,
              "Vulkan inline GPU timing requires a Vulkan command list");
  vulkan_cmdlist->write_runtime_timestamp(
      timing->query_pool(), 0, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, true);
  return timing;
}

void VulkanStream::end_gpu_timing(const StreamGpuTiming &timing) {
  if (!timing) {
    return;
  }
  TI_ERROR_IF(submission_batch_depth_ == 0,
              "Vulkan GPU timing requires an active submission batch");
  auto vulkan_timing =
      std::dynamic_pointer_cast<VulkanStreamGpuTimingObject>(timing);
  TI_ERROR_IF(!vulkan_timing,
              "Vulkan stream received a timing object from another backend");
  auto [cmdlist, result] = new_command_list_unique();
  TI_ERROR_IF(result != RhiResult::success,
              "Unable to allocate Vulkan GPU timing end command list");
  static_cast<VulkanCommandList *>(cmdlist.get())
      ->write_runtime_timestamp(vulkan_timing->query_pool(), 1,
                                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, false);
  submit(cmdlist.get());
  vulkan_timing->mark_ended();
}

void VulkanStream::end_gpu_timing_inline(const StreamGpuTiming &timing,
                                         CommandList *cmdlist) {
  if (!timing) {
    return;
  }
  TI_ERROR_IF(submission_batch_depth_ == 0,
              "Vulkan GPU timing requires an active submission batch");
  TI_ERROR_IF(cmdlist == nullptr,
              "Vulkan inline GPU timing requires a command list");
  auto vulkan_timing =
      std::dynamic_pointer_cast<VulkanStreamGpuTimingObject>(timing);
  TI_ERROR_IF(!vulkan_timing,
              "Vulkan stream received a timing object from another backend");
  auto *vulkan_cmdlist = dynamic_cast<VulkanCommandList *>(cmdlist);
  TI_ERROR_IF(vulkan_cmdlist == nullptr,
              "Vulkan inline GPU timing requires a Vulkan command list");
  vulkan_cmdlist->write_runtime_timestamp(
      vulkan_timing->query_pool(), 1, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
      false);
  vulkan_timing->mark_ended();
}

VulkanStream::~VulkanStream() {
}

}  // namespace vulkan
}  // namespace taichi::lang
