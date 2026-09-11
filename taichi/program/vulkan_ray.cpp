#include "taichi/program/program.h"
#include "taichi/program/storage_view.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>

#if defined(TI_WITH_VULKAN)
#include "taichi/rhi/vulkan/vulkan_device.h"

namespace taichi::lang {
namespace {

constexpr std::uint32_t kRayQueryWorkgroupSize = 128;

static const std::uint32_t kRayQueryTrianglesSpv[] =
#include "taichi/program/vulkan_sort_shaders/ray_query_triangles.comp.spv.h"
    ;

static const std::uint32_t kRayQueryTrianglesTypedSpv[] =
#include "taichi/program/vulkan_sort_shaders/ray_query_triangles_typed.comp.spv.h"
    ;

static const std::uint32_t kRayQueryTrianglesSubrangeSpv[] =
#include "taichi/program/vulkan_sort_shaders/ray_query_triangles_subrange.comp.spv.h"
    ;
static const std::uint32_t kRayQueryTrianglesTypedSubrangeSpv[] =
#include "taichi/program/vulkan_sort_shaders/ray_query_triangles_typed_subrange.comp.spv.h"
    ;

// Scene and independent TLAS share query code, but each owns its pipelines and
// resource sets. The owner's existing mutex protects prepare and record.
class RayQueryPipelines {
 public:
  void clear() {
    bindings_ = {};
    pipelines_ = {};
  }

  void prepare(vulkan::VulkanDevice *device, unsigned variant) {
    if (pipelines_[variant]) {
      return;
    }
    const std::array<std::pair<const std::uint32_t *, std::size_t>, 4> code{
        {{kRayQueryTrianglesSpv, sizeof(kRayQueryTrianglesSpv)},
         {kRayQueryTrianglesTypedSpv, sizeof(kRayQueryTrianglesTypedSpv)},
         {kRayQueryTrianglesSubrangeSpv, sizeof(kRayQueryTrianglesSubrangeSpv)},
         {kRayQueryTrianglesTypedSubrangeSpv,
          sizeof(kRayQueryTrianglesTypedSubrangeSpv)}}};
    PipelineSourceDesc source{PipelineSourceType::spirv_binary,
                              code[variant].first, code[variant].second,
                              PipelineStageType::compute};
    auto [pipeline, result] = device->create_pipeline_unique(
        source, "vulkan_ray_query_triangles_" + std::to_string(variant));
    TI_ERROR_IF(result != RhiResult::success || !pipeline,
                "Failed to create Vulkan ray query pipeline: {}.", result);
    std::unique_ptr<ShaderResourceSet> bindings(device->create_resource_set());
    pipelines_[variant] = std::move(pipeline);
    bindings_[variant] = std::move(bindings);
  }

  void record(CommandList *commands,
              const vkapi::IVkAccelerationStructureKHR &tlas,
              const VulkanRayQueryCommand &packet) {
    const bool typed = (packet.variant & 1u) != 0;
    auto *bindings = static_cast<vulkan::VulkanResourceSet *>(
        bindings_[packet.variant].get());
    commands->buffer_barrier(packet.shader_bindings[0].pointer);
    bindings->acceleration_structure(0, tlas);
    for (unsigned i = 0; i < (typed ? 3u : 2u); ++i) {
      const auto &binding = packet.shader_bindings[i];
      bindings->rw_buffer(i + 1, binding.pointer, binding.bytes);
    }
    commands->bind_pipeline(pipelines_[packet.variant].get());
    const auto bind_result = commands->bind_shader_resources(bindings, 0);
    TI_ERROR_IF(bind_result != RhiResult::success,
                "Failed to bind Vulkan ray query resources: {}.", bind_result);
    auto *vk_commands = static_cast<vulkan::VulkanCommandList *>(commands);
    vk_commands->push_constants(packet.parameters.data(),
                                (packet.variant & 2u) ? 16 : 4);
    const auto dispatch_result = commands->dispatch(static_cast<std::uint32_t>(
        (packet.ray_count + kRayQueryWorkgroupSize - 1) /
        kRayQueryWorkgroupSize));
    TI_ERROR_IF(dispatch_result != RhiResult::success,
                "Failed to dispatch Vulkan ray query: {}.", dispatch_result);
    commands->buffer_barrier(packet.shader_bindings[1].pointer);
    if (typed) {
      commands->buffer_barrier(packet.shader_bindings[2].pointer);
    }
  }

 private:
  std::array<std::unique_ptr<Pipeline>, 4> pipelines_;
  std::array<std::unique_ptr<ShaderResourceSet>, 4> bindings_;
};

template <typename Function>
Function load_vulkan_device_function(VkDevice device, const char *name) {
  auto function = reinterpret_cast<Function>(vkGetDeviceProcAddr(device, name));
  TI_ERROR_IF(function == nullptr,
              "Vulkan ray provider could not load required function '{}'.",
              name);
  return function;
}

std::size_t checked_mul(std::size_t lhs,
                        std::size_t rhs,
                        const char *description) {
  TI_ERROR_IF(lhs != 0 && rhs > (std::numeric_limits<std::size_t>::max)() / lhs,
              "Vulkan ray {} size overflow.", description);
  return lhs * rhs;
}

VkDeviceAddress aligned_address(VkDeviceAddress address,
                                VkDeviceSize alignment) {
  TI_ASSERT(alignment != 0 && (alignment & (alignment - 1)) == 0);
  return (address + alignment - 1) & ~(alignment - 1);
}

}  // namespace

class VulkanTriangleRayScene {
 public:
  VulkanTriangleRayScene(Program *program,
                         std::size_t vertex_count,
                         std::size_t triangle_count)
      : program_(program),
        vertex_count_(vertex_count),
        triangle_count_(triangle_count) {
    TI_ERROR_IF(program_ == nullptr,
                "Vulkan triangle ray scene requires a live Program.");
    device_ = static_cast<vulkan::VulkanDevice *>(
        program_->get_compute_device());
    TI_ERROR_IF(device_ == nullptr ||
                    !device_->vk_caps().acceleration_structure ||
                    !device_->vk_caps().ray_query,
                "Vulkan triangle ray scene requires acceleration-structure "
                "and ray-query support.");
    TI_ERROR_IF(vertex_count_ == 0 ||
                    vertex_count_ > static_cast<std::size_t>(
                                        (std::numeric_limits<
                                            std::uint32_t>::max)()),
                "Vulkan triangle ray vertex_count must be in [1, UINT32_MAX].");
    TI_ERROR_IF(triangle_count_ == 0 ||
                    triangle_count_ > static_cast<std::size_t>(
                                          (std::numeric_limits<
                                              std::uint32_t>::max)()),
                "Vulkan triangle ray triangle_count must be in [1, "
                "UINT32_MAX].");

    get_build_sizes_ = load_vulkan_device_function<
        PFN_vkGetAccelerationStructureBuildSizesKHR>(
        device_->vk_device(), "vkGetAccelerationStructureBuildSizesKHR");
    get_as_address_ = load_vulkan_device_function<
        PFN_vkGetAccelerationStructureDeviceAddressKHR>(
        device_->vk_device(), "vkGetAccelerationStructureDeviceAddressKHR");
    cmd_build_ =
        load_vulkan_device_function<PFN_vkCmdBuildAccelerationStructuresKHR>(
            device_->vk_device(), "vkCmdBuildAccelerationStructuresKHR");

    VkPhysicalDeviceAccelerationStructurePropertiesKHR as_properties{};
    as_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &as_properties;
    vkGetPhysicalDeviceProperties2(device_->vk_physical_device(), &properties);
    scratch_alignment_ =
        as_properties.minAccelerationStructureScratchOffsetAlignment;
    TI_ERROR_IF(scratch_alignment_ == 0,
                "Vulkan ray provider reported zero scratch alignment.");

    vertex_bytes_ = checked_mul(vertex_count_, 3 * sizeof(float), "vertex");
    index_bytes_ =
        checked_mul(triangle_count_, 3 * sizeof(std::uint32_t), "index");
    vertex_buffer_ = allocate(
        vertex_bytes_, AllocUsage::AccelerationStructureBuildInput |
                           AllocUsage::DeviceAddress);
    index_buffer_ = allocate(
        index_bytes_, AllocUsage::AccelerationStructureBuildInput |
                          AllocUsage::DeviceAddress);

    create_blas();
    create_tlas();
    create_query_pipeline();
  }

  ~VulkanTriangleRayScene() {
    query_.clear();
    tlas_.reset();
    blas_.reset();
    release(tlas_scratch_);
    release(blas_scratch_);
    release(tlas_storage_);
    release(blas_storage_);
    release(instance_buffer_);
    release(index_buffer_);
    release(vertex_buffer_);
  }

  VulkanTriangleRayScene(const VulkanTriangleRayScene &) = delete;
  VulkanTriangleRayScene &operator=(const VulkanTriangleRayScene &) = delete;

  VulkanTriangleRaySceneMemoryStatistics memory_statistics() const {
    VulkanTriangleRaySceneMemoryStatistics result;
    result.geometry_input_requested_bytes =
        vertex_bytes_ + index_bytes_ + instance_buffer_bytes_;
    result.acceleration_structure_requested_bytes =
        blas_storage_bytes_ + tlas_storage_bytes_;
    result.build_scratch_requested_bytes =
        blas_scratch_bytes_ + tlas_scratch_bytes_;
    result.known_requested_bytes =
        result.geometry_input_requested_bytes +
        result.acceleration_structure_requested_bytes +
        result.build_scratch_requested_bytes;
    result.known_allocation_count = 7;
    return result;
  }

  void record_build(CommandList *command_list,
                    DeviceAllocation source_vertices,
                    DeviceAllocation source_indices) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto *vk_commands = static_cast<vulkan::VulkanCommandList *>(command_list);
    auto command_buffer = vk_commands->vk_command_buffer();

    command_list->buffer_copy(vertex_buffer_.get_ptr(),
                              source_vertices.get_ptr(), vertex_bytes_);
    command_list->buffer_copy(index_buffer_.get_ptr(), source_indices.get_ptr(),
                              index_bytes_);

    VkMemoryBarrier input_barrier{};
    input_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    input_barrier.srcAccessMask =
        VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT;
    input_barrier.dstAccessMask =
        VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(
        command_buffer->buffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1,
        &input_barrier, 0, nullptr, 0, nullptr);

    const auto blas_geometry = make_blas_geometry();
    auto blas_build = make_blas_build_info(blas_geometry);
    const VkAccelerationStructureBuildRangeInfoKHR blas_range{
        static_cast<std::uint32_t>(triangle_count_), 0, 0, 0};
    const VkAccelerationStructureBuildRangeInfoKHR *blas_ranges[] = {
        &blas_range};
    cmd_build_(command_buffer->buffer, 1, &blas_build, blas_ranges);

    VkMemoryBarrier blas_barrier{};
    blas_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    blas_barrier.srcAccessMask =
        VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    blas_barrier.dstAccessMask =
        VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(
        command_buffer->buffer,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1,
        &blas_barrier, 0, nullptr, 0, nullptr);

    const auto tlas_geometry = make_tlas_geometry();
    auto tlas_build = make_tlas_build_info(tlas_geometry);
    const VkAccelerationStructureBuildRangeInfoKHR tlas_range{1, 0, 0, 0};
    const VkAccelerationStructureBuildRangeInfoKHR *tlas_ranges[] = {
        &tlas_range};
    cmd_build_(command_buffer->buffer, 1, &tlas_build, tlas_ranges);

    VkMemoryBarrier query_barrier{};
    query_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    query_barrier.srcAccessMask =
        VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    query_barrier.dstAccessMask =
        VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(
        command_buffer->buffer,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &query_barrier, 0, nullptr,
        0, nullptr);

    retain_build_resources(command_buffer);
  }

  void record_refit(CommandList *command_list,
                    DeviceAllocation source_vertices) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto *vk_commands = static_cast<vulkan::VulkanCommandList *>(command_list);
    auto command_buffer = vk_commands->vk_command_buffer();

    command_list->buffer_copy(vertex_buffer_.get_ptr(),
                              source_vertices.get_ptr(), vertex_bytes_);

    VkMemoryBarrier input_barrier{};
    input_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    input_barrier.srcAccessMask =
        VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT;
    input_barrier.dstAccessMask =
        VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(
        command_buffer->buffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1,
        &input_barrier, 0, nullptr, 0, nullptr);

    const auto geometry = make_blas_geometry();
    auto build = make_blas_build_info(
        geometry, VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR);
    const VkAccelerationStructureBuildRangeInfoKHR range{
        static_cast<std::uint32_t>(triangle_count_), 0, 0, 0};
    const VkAccelerationStructureBuildRangeInfoKHR *ranges[] = {&range};
    cmd_build_(command_buffer->buffer, 1, &build, ranges);

    VkMemoryBarrier query_barrier{};
    query_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    query_barrier.srcAccessMask =
        VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    query_barrier.dstAccessMask =
        VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(command_buffer->buffer,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &query_barrier, 0, nullptr, 0, nullptr);

    retain_build_resources(command_buffer);
  }

  std::size_t vertex_count() const {
    return vertex_count_;
  }

  void prepare_query_variant(unsigned variant) {
    std::lock_guard<std::mutex> lock(mutex_);
    query_.prepare(device_, variant);
  }

  void record_query(CommandList *command_list,
                    const VulkanRayQueryCommand &packet) {
    std::lock_guard<std::mutex> lock(mutex_);
    query_.record(command_list, tlas_, packet);
  }

 private:
  DeviceAllocation allocate(std::size_t bytes,
                            AllocUsage usage,
                            bool host_write = false) {
    Device::AllocParams params;
    params.size = bytes;
    params.host_write = host_write;
    params.usage = usage;
    DeviceAllocation allocation{kDeviceNullAllocation};
    const auto result = device_->allocate_memory(params, &allocation);
    TI_ERROR_IF(result != RhiResult::success,
                "Failed to allocate Vulkan ray buffer ({} bytes): "
                "RhiResult({}).",
                bytes, result);
    return allocation;
  }

  void release(DeviceAllocation &allocation) noexcept {
    if (allocation != kDeviceNullAllocation && device_ != nullptr) {
      device_->dealloc_memory(allocation);
      allocation = kDeviceNullAllocation;
    }
  }

  VkDeviceAddress scratch_address(DeviceAllocation allocation) const {
    return aligned_address(device_->get_buffer_device_address(allocation),
                           scratch_alignment_);
  }

  std::size_t scratch_allocation_bytes(VkDeviceSize bytes) const {
    TI_ERROR_IF(bytes > (std::numeric_limits<std::size_t>::max)() -
                            scratch_alignment_,
                "Vulkan ray scratch size overflow.");
    return static_cast<std::size_t>(bytes + scratch_alignment_);
  }

  VkAccelerationStructureGeometryKHR make_blas_geometry() const {
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
    triangles.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress =
        device_->get_buffer_device_address(vertex_buffer_);
    triangles.vertexStride = 3 * sizeof(float);
    triangles.maxVertex = static_cast<std::uint32_t>(vertex_count_ - 1);
    triangles.indexType = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress =
        device_->get_buffer_device_address(index_buffer_);

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geometry.geometry.triangles = triangles;
    return geometry;
  }

  VkAccelerationStructureBuildGeometryInfoKHR make_blas_build_info(
      const VkAccelerationStructureGeometryKHR &geometry,
      VkBuildAccelerationStructureModeKHR mode =
          VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR) const {
    VkAccelerationStructureBuildGeometryInfoKHR info{};
    info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                 VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    info.mode = mode;
    if (mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR) {
      info.srcAccelerationStructure = blas_ ? blas_->accel : VK_NULL_HANDLE;
    }
    info.dstAccelerationStructure = blas_ ? blas_->accel : VK_NULL_HANDLE;
    info.geometryCount = 1;
    info.pGeometries = &geometry;
    if (blas_scratch_ != kDeviceNullAllocation) {
      info.scratchData.deviceAddress = scratch_address(blas_scratch_);
    }
    return info;
  }

  VkAccelerationStructureGeometryKHR make_tlas_geometry() const {
    VkAccelerationStructureGeometryInstancesDataKHR instances{};
    instances.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instances.arrayOfPointers = VK_FALSE;
    instances.data.deviceAddress =
        device_->get_buffer_device_address(instance_buffer_);

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances = instances;
    return geometry;
  }

  VkAccelerationStructureBuildGeometryInfoKHR make_tlas_build_info(
      const VkAccelerationStructureGeometryKHR &geometry) const {
    VkAccelerationStructureBuildGeometryInfoKHR info{};
    info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    info.flags =
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    info.dstAccelerationStructure = tlas_ ? tlas_->accel : VK_NULL_HANDLE;
    info.geometryCount = 1;
    info.pGeometries = &geometry;
    if (tlas_scratch_ != kDeviceNullAllocation) {
      info.scratchData.deviceAddress = scratch_address(tlas_scratch_);
    }
    return info;
  }

  void create_blas() {
    const auto geometry = make_blas_geometry();
    auto build_info = make_blas_build_info(geometry);
    const auto primitive_count = static_cast<std::uint32_t>(triangle_count_);
    VkAccelerationStructureBuildSizesInfoKHR sizes{};
    sizes.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    get_build_sizes_(device_->vk_device(),
                     VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                     &build_info, &primitive_count, &sizes);
    blas_storage_bytes_ =
        static_cast<std::size_t>(sizes.accelerationStructureSize);
    blas_storage_ =
        allocate(blas_storage_bytes_, AllocUsage::AccelerationStructureStorage);
    blas_ = vkapi::create_acceleration_structure(
        0, device_->get_vkbuffer(blas_storage_.get_ptr()), 0,
        sizes.accelerationStructureSize,
        VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR);
    TI_ERROR_IF(!blas_, "Failed to create Vulkan triangle BLAS.");
    blas_scratch_bytes_ = scratch_allocation_bytes(
        std::max(sizes.buildScratchSize, sizes.updateScratchSize));
    blas_scratch_ = allocate(blas_scratch_bytes_,
                             AllocUsage::Storage | AllocUsage::DeviceAddress);
  }

  void create_tlas() {
    VkAccelerationStructureDeviceAddressInfoKHR address_info{};
    address_info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    address_info.accelerationStructure = blas_->accel;
    const VkDeviceAddress blas_address =
        get_as_address_(device_->vk_device(), &address_info);
    TI_ERROR_IF(blas_address == 0,
                "Vulkan triangle BLAS returned a null device address.");

    instance_buffer_bytes_ = sizeof(VkAccelerationStructureInstanceKHR);
    instance_buffer_ = allocate(
        instance_buffer_bytes_,
        AllocUsage::AccelerationStructureBuildInput | AllocUsage::DeviceAddress,
        true);
    VkAccelerationStructureInstanceKHR instance{};
    instance.transform.matrix[0][0] = 1.0f;
    instance.transform.matrix[1][1] = 1.0f;
    instance.transform.matrix[2][2] = 1.0f;
    instance.instanceCustomIndex = 0;
    instance.mask = 0xff;
    instance.instanceShaderBindingTableRecordOffset = 0;
    instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    instance.accelerationStructureReference = blas_address;
    void *mapped = nullptr;
    const auto map_result = device_->map(instance_buffer_, &mapped);
    TI_ERROR_IF(map_result != RhiResult::success || mapped == nullptr,
                "Failed to map Vulkan TLAS instance buffer: RhiResult({}).",
                map_result);
    std::memcpy(mapped, &instance, sizeof(instance));
    device_->unmap(instance_buffer_);

    const auto geometry = make_tlas_geometry();
    auto build_info = make_tlas_build_info(geometry);
    const std::uint32_t primitive_count = 1;
    VkAccelerationStructureBuildSizesInfoKHR sizes{};
    sizes.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    get_build_sizes_(device_->vk_device(),
                     VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                     &build_info, &primitive_count, &sizes);
    tlas_storage_bytes_ =
        static_cast<std::size_t>(sizes.accelerationStructureSize);
    tlas_storage_ =
        allocate(tlas_storage_bytes_, AllocUsage::AccelerationStructureStorage);
    tlas_ = vkapi::create_acceleration_structure(
        0, device_->get_vkbuffer(tlas_storage_.get_ptr()), 0,
        sizes.accelerationStructureSize,
        VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR);
    TI_ERROR_IF(!tlas_, "Failed to create Vulkan triangle TLAS.");
    tlas_scratch_bytes_ = scratch_allocation_bytes(sizes.buildScratchSize);
    tlas_scratch_ = allocate(tlas_scratch_bytes_,
                             AllocUsage::Storage | AllocUsage::DeviceAddress);
  }

  void create_query_pipeline() {
    query_.prepare(device_, 0);
  }

  void retain_build_resources(
      const vkapi::IVkCommandBuffer &command_buffer) const {
    const std::array<DeviceAllocation, 7> allocations{
        vertex_buffer_, index_buffer_, instance_buffer_, blas_storage_,
        tlas_storage_,  blas_scratch_,  tlas_scratch_};
    for (const auto allocation : allocations) {
      command_buffer->refs.push_back(
          device_->get_vkbuffer(allocation.get_ptr()));
    }
    command_buffer->refs.push_back(blas_);
    command_buffer->refs.push_back(tlas_);
  }

  Program *program_{nullptr};
  vulkan::VulkanDevice *device_{nullptr};
  std::size_t vertex_count_{0};
  std::size_t triangle_count_{0};
  std::size_t vertex_bytes_{0};
  std::size_t index_bytes_{0};
  std::size_t instance_buffer_bytes_{0};
  std::size_t blas_storage_bytes_{0};
  std::size_t tlas_storage_bytes_{0};
  std::size_t blas_scratch_bytes_{0};
  std::size_t tlas_scratch_bytes_{0};
  VkDeviceSize scratch_alignment_{1};
  DeviceAllocation vertex_buffer_{kDeviceNullAllocation};
  DeviceAllocation index_buffer_{kDeviceNullAllocation};
  DeviceAllocation instance_buffer_{kDeviceNullAllocation};
  DeviceAllocation blas_storage_{kDeviceNullAllocation};
  DeviceAllocation tlas_storage_{kDeviceNullAllocation};
  DeviceAllocation blas_scratch_{kDeviceNullAllocation};
  DeviceAllocation tlas_scratch_{kDeviceNullAllocation};
  vkapi::IVkAccelerationStructureKHR blas_{nullptr};
  vkapi::IVkAccelerationStructureKHR tlas_{nullptr};
  RayQueryPipelines query_;
  PFN_vkGetAccelerationStructureBuildSizesKHR get_build_sizes_{nullptr};
  PFN_vkGetAccelerationStructureDeviceAddressKHR get_as_address_{nullptr};
  PFN_vkCmdBuildAccelerationStructuresKHR cmd_build_{nullptr};
  std::mutex mutex_;
};

enum class VulkanRayResourceKind {
  kTriangleBlas,
  kInstanceTlas,
};

class VulkanRayResource {
 public:
  virtual ~VulkanRayResource() = default;
  virtual VulkanRayResourceKind kind() const = 0;
  virtual VulkanTriangleRaySceneMemoryStatistics memory_statistics() const = 0;
};

class VulkanTriangleBlasResource final : public VulkanRayResource {
 public:
  VulkanTriangleBlasResource(Program *program,
                             std::size_t vertex_count,
                             std::size_t triangle_count)
      : program_(program),
        vertex_count_(vertex_count),
        triangle_count_(triangle_count) {
    TI_ERROR_IF(program_ == nullptr,
                "Vulkan triangle BLAS requires a live Program.");
    device_ = static_cast<vulkan::VulkanDevice *>(
        program_->get_compute_device());
    TI_ERROR_IF(device_ == nullptr ||
                    !device_->vk_caps().acceleration_structure ||
                    !device_->vk_caps().ray_query,
                "Vulkan triangle BLAS requires acceleration-structure and "
                "ray-query support.");
    TI_ERROR_IF(vertex_count_ == 0 ||
                    vertex_count_ > static_cast<std::size_t>(
                                        (std::numeric_limits<
                                            std::uint32_t>::max)()),
                "Vulkan triangle BLAS vertex_count must be in [1, "
                "UINT32_MAX].");
    TI_ERROR_IF(triangle_count_ == 0 ||
                    triangle_count_ > static_cast<std::size_t>(
                                          (std::numeric_limits<
                                              std::uint32_t>::max)()),
                "Vulkan triangle BLAS triangle_count must be in [1, "
                "UINT32_MAX].");
    get_build_sizes_ = load_vulkan_device_function<
        PFN_vkGetAccelerationStructureBuildSizesKHR>(
        device_->vk_device(), "vkGetAccelerationStructureBuildSizesKHR");
    get_as_address_ = load_vulkan_device_function<
        PFN_vkGetAccelerationStructureDeviceAddressKHR>(
        device_->vk_device(), "vkGetAccelerationStructureDeviceAddressKHR");
    cmd_build_ =
        load_vulkan_device_function<PFN_vkCmdBuildAccelerationStructuresKHR>(
            device_->vk_device(), "vkCmdBuildAccelerationStructuresKHR");

    VkPhysicalDeviceAccelerationStructurePropertiesKHR as_properties{};
    as_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &as_properties;
    vkGetPhysicalDeviceProperties2(device_->vk_physical_device(), &properties);
    scratch_alignment_ =
        as_properties.minAccelerationStructureScratchOffsetAlignment;
    TI_ERROR_IF(scratch_alignment_ == 0,
                "Vulkan triangle BLAS reported zero scratch alignment.");

    vertex_bytes_ = checked_mul(vertex_count_, 3 * sizeof(float), "vertex");
    index_bytes_ =
        checked_mul(triangle_count_, 3 * sizeof(std::uint32_t), "index");
    vertex_buffer_ = allocate(
        vertex_bytes_, AllocUsage::AccelerationStructureBuildInput |
                           AllocUsage::DeviceAddress);
    index_buffer_ = allocate(
        index_bytes_, AllocUsage::AccelerationStructureBuildInput |
                          AllocUsage::DeviceAddress);
    create_acceleration_structure();
  }

  ~VulkanTriangleBlasResource() override {
    blas_.reset();
    release(scratch_);
    release(storage_);
    release(index_buffer_);
    release(vertex_buffer_);
  }

  VulkanRayResourceKind kind() const override {
    return VulkanRayResourceKind::kTriangleBlas;
  }

  VulkanTriangleRaySceneMemoryStatistics memory_statistics() const override {
    VulkanTriangleRaySceneMemoryStatistics result;
    result.geometry_input_requested_bytes = vertex_bytes_ + index_bytes_;
    result.acceleration_structure_requested_bytes = storage_bytes_;
    result.build_scratch_requested_bytes = scratch_bytes_;
    result.known_requested_bytes = result.geometry_input_requested_bytes +
                                   result.acceleration_structure_requested_bytes +
                                   result.build_scratch_requested_bytes;
    result.known_allocation_count = 4;
    return result;
  }

  std::size_t vertex_count() const {
    return vertex_count_;
  }

  std::size_t triangle_count() const {
    return triangle_count_;
  }

  VkDeviceAddress device_address() const {
    VkAccelerationStructureDeviceAddressInfoKHR info{};
    info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    info.accelerationStructure = blas_->accel;
    const VkDeviceAddress address =
        get_as_address_(device_->vk_device(), &info);
    TI_ERROR_IF(address == 0,
                "Vulkan triangle BLAS returned a null device address.");
    return address;
  }

  void record_build(CommandList *command_list,
                    DeviceAllocation source_vertices,
                    DeviceAllocation source_indices,
                    bool update) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto *vk_commands = static_cast<vulkan::VulkanCommandList *>(command_list);
    auto command_buffer = vk_commands->vk_command_buffer();

    VkMemoryBarrier reuse_barrier{};
    reuse_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    reuse_barrier.srcAccessMask =
        VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    reuse_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(
        command_buffer->buffer,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &reuse_barrier, 0, nullptr, 0,
        nullptr);

    command_list->buffer_copy(vertex_buffer_.get_ptr(),
                              source_vertices.get_ptr(), vertex_bytes_);
    if (!update) {
      command_list->buffer_copy(index_buffer_.get_ptr(),
                                source_indices.get_ptr(), index_bytes_);
    }

    VkMemoryBarrier input_barrier{};
    input_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    input_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    input_barrier.dstAccessMask =
        VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(
        command_buffer->buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1,
        &input_barrier, 0, nullptr, 0, nullptr);

    const auto geometry = make_geometry();
    auto build = make_build_info(
        geometry, update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR
                         : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR);
    const VkAccelerationStructureBuildRangeInfoKHR range{
        static_cast<std::uint32_t>(triangle_count_), 0, 0, 0};
    const VkAccelerationStructureBuildRangeInfoKHR *ranges[] = {&range};
    cmd_build_(command_buffer->buffer, 1, &build, ranges);

    VkMemoryBarrier completion_barrier{};
    completion_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    completion_barrier.srcAccessMask =
        VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    completion_barrier.dstAccessMask =
        VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(
        command_buffer->buffer,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &completion_barrier, 0, nullptr, 0, nullptr);
    retain(command_buffer);
  }

  void retain(const vkapi::IVkCommandBuffer &command_buffer) const {
    const std::array<DeviceAllocation, 4> allocations{
        vertex_buffer_, index_buffer_, storage_, scratch_};
    for (const auto allocation : allocations) {
      command_buffer->refs.push_back(
          device_->get_vkbuffer(allocation.get_ptr()));
    }
    command_buffer->refs.push_back(blas_);
  }

 private:
  DeviceAllocation allocate(std::size_t bytes, AllocUsage usage) {
    Device::AllocParams params;
    params.size = bytes;
    params.usage = usage;
    DeviceAllocation allocation{kDeviceNullAllocation};
    const auto result = device_->allocate_memory(params, &allocation);
    TI_ERROR_IF(result != RhiResult::success,
                "Failed to allocate Vulkan triangle BLAS buffer ({} bytes): "
                "RhiResult({}).",
                bytes, result);
    return allocation;
  }

  void release(DeviceAllocation &allocation) noexcept {
    if (allocation != kDeviceNullAllocation && device_ != nullptr) {
      device_->dealloc_memory(allocation);
      allocation = kDeviceNullAllocation;
    }
  }

  VkDeviceAddress scratch_address() const {
    return aligned_address(device_->get_buffer_device_address(scratch_),
                           scratch_alignment_);
  }

  std::size_t scratch_allocation_bytes(VkDeviceSize bytes) const {
    TI_ERROR_IF(bytes > (std::numeric_limits<std::size_t>::max)() -
                            scratch_alignment_,
                "Vulkan triangle BLAS scratch size overflow.");
    return static_cast<std::size_t>(bytes + scratch_alignment_);
  }

  VkAccelerationStructureGeometryKHR make_geometry() const {
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
    triangles.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress =
        device_->get_buffer_device_address(vertex_buffer_);
    triangles.vertexStride = 3 * sizeof(float);
    triangles.maxVertex = static_cast<std::uint32_t>(vertex_count_ - 1);
    triangles.indexType = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress =
        device_->get_buffer_device_address(index_buffer_);
    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geometry.geometry.triangles = triangles;
    return geometry;
  }

  VkAccelerationStructureBuildGeometryInfoKHR make_build_info(
      const VkAccelerationStructureGeometryKHR &geometry,
      VkBuildAccelerationStructureModeKHR mode) const {
    VkAccelerationStructureBuildGeometryInfoKHR info{};
    info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                 VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    info.mode = mode;
    if (mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR) {
      info.srcAccelerationStructure = blas_->accel;
    }
    info.dstAccelerationStructure = blas_->accel;
    info.geometryCount = 1;
    info.pGeometries = &geometry;
    info.scratchData.deviceAddress = scratch_address();
    return info;
  }

  void create_acceleration_structure() {
    const auto geometry = make_geometry();
    VkAccelerationStructureBuildGeometryInfoKHR build{};
    build.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                  VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    build.geometryCount = 1;
    build.pGeometries = &geometry;
    const auto primitive_count = static_cast<std::uint32_t>(triangle_count_);
    VkAccelerationStructureBuildSizesInfoKHR sizes{};
    sizes.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    get_build_sizes_(device_->vk_device(),
                     VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build,
                     &primitive_count, &sizes);
    storage_bytes_ =
        static_cast<std::size_t>(sizes.accelerationStructureSize);
    storage_ = allocate(storage_bytes_, AllocUsage::AccelerationStructureStorage);
    blas_ = vkapi::create_acceleration_structure(
        0, device_->get_vkbuffer(storage_.get_ptr()), 0,
        sizes.accelerationStructureSize,
        VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR);
    TI_ERROR_IF(!blas_, "Failed to create independent Vulkan triangle BLAS.");
    scratch_bytes_ = scratch_allocation_bytes(
        std::max(sizes.buildScratchSize, sizes.updateScratchSize));
    scratch_ = allocate(scratch_bytes_,
                        AllocUsage::Storage | AllocUsage::DeviceAddress);
  }

  Program *program_{nullptr};
  vulkan::VulkanDevice *device_{nullptr};
  std::size_t vertex_count_{0};
  std::size_t triangle_count_{0};
  std::size_t vertex_bytes_{0};
  std::size_t index_bytes_{0};
  std::size_t storage_bytes_{0};
  std::size_t scratch_bytes_{0};
  VkDeviceSize scratch_alignment_{1};
  DeviceAllocation vertex_buffer_{kDeviceNullAllocation};
  DeviceAllocation index_buffer_{kDeviceNullAllocation};
  DeviceAllocation storage_{kDeviceNullAllocation};
  DeviceAllocation scratch_{kDeviceNullAllocation};
  vkapi::IVkAccelerationStructureKHR blas_{nullptr};
  PFN_vkGetAccelerationStructureBuildSizesKHR get_build_sizes_{nullptr};
  PFN_vkGetAccelerationStructureDeviceAddressKHR get_as_address_{nullptr};
  PFN_vkCmdBuildAccelerationStructuresKHR cmd_build_{nullptr};
  std::mutex mutex_;
};

class VulkanInstanceTlasResource final : public VulkanRayResource {
 public:
  VulkanInstanceTlasResource(
      Program *program,
      std::vector<std::shared_ptr<VulkanTriangleBlasResource>> blases)
      : program_(program), blases_(std::move(blases)) {
    TI_ERROR_IF(program_ == nullptr,
                "Vulkan instance TLAS requires a live Program.");
    TI_ERROR_IF(blases_.empty(),
                "Vulkan instance TLAS requires at least one BLAS instance.");
    device_ = static_cast<vulkan::VulkanDevice *>(
        program_->get_compute_device());
    TI_ERROR_IF(device_ == nullptr ||
                    !device_->vk_caps().acceleration_structure ||
                    !device_->vk_caps().ray_query,
                "Vulkan instance TLAS requires acceleration-structure and "
                "ray-query support.");
    get_build_sizes_ = load_vulkan_device_function<
        PFN_vkGetAccelerationStructureBuildSizesKHR>(
        device_->vk_device(), "vkGetAccelerationStructureBuildSizesKHR");
    cmd_build_ =
        load_vulkan_device_function<PFN_vkCmdBuildAccelerationStructuresKHR>(
            device_->vk_device(), "vkCmdBuildAccelerationStructuresKHR");

    VkPhysicalDeviceAccelerationStructurePropertiesKHR as_properties{};
    as_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &as_properties;
    vkGetPhysicalDeviceProperties2(device_->vk_physical_device(), &properties);
    scratch_alignment_ =
        as_properties.minAccelerationStructureScratchOffsetAlignment;
    TI_ERROR_IF(scratch_alignment_ == 0,
                "Vulkan instance TLAS reported zero scratch alignment.");
    TI_ERROR_IF(blases_.size() > as_properties.maxInstanceCount ||
                    blases_.size() > static_cast<std::size_t>(
                                         (std::numeric_limits<
                                             std::uint32_t>::max)()),
                "Vulkan instance TLAS instance count exceeds the device "
                "limit.");

    instance_bytes_ = checked_mul(blases_.size(),
                                  sizeof(VkAccelerationStructureInstanceKHR),
                                  "instance input");
    instance_buffer_ = allocate(
        instance_bytes_, AllocUsage::AccelerationStructureBuildInput |
                             AllocUsage::DeviceAddress);
    create_acceleration_structure();
    create_query_pipeline();
  }

  ~VulkanInstanceTlasResource() override {
    query_.clear();
    tlas_.reset();
    release(scratch_);
    release(storage_);
    release(instance_buffer_);
  }

  VulkanRayResourceKind kind() const override {
    return VulkanRayResourceKind::kInstanceTlas;
  }

  VulkanTriangleRaySceneMemoryStatistics memory_statistics() const override {
    VulkanTriangleRaySceneMemoryStatistics result;
    result.geometry_input_requested_bytes = instance_bytes_;
    result.acceleration_structure_requested_bytes = storage_bytes_;
    result.build_scratch_requested_bytes = scratch_bytes_;
    result.known_requested_bytes = result.geometry_input_requested_bytes +
                                   result.acceleration_structure_requested_bytes +
                                   result.build_scratch_requested_bytes;
    result.known_allocation_count = 3;
    return result;
  }

  std::size_t instance_count() const {
    return blases_.size();
  }

  const std::vector<std::shared_ptr<VulkanTriangleBlasResource>> &blases()
      const {
    return blases_;
  }

  void record_build(CommandList *command_list,
                    const std::vector<VulkanRayInstanceInfo> &instances,
                    bool update) {
    std::lock_guard<std::mutex> lock(mutex_);
    TI_ERROR_IF(instances.size() != blases_.size(),
                "Vulkan instance TLAS build must preserve instance count {}.",
                blases_.size());
    std::vector<VkAccelerationStructureInstanceKHR> packed(instances.size());
    for (std::size_t index = 0; index < instances.size(); ++index) {
      const auto &source = instances[index];
      TI_ERROR_IF(source.mask > 0xff || source.custom_index > 0xffffff,
                  "Vulkan instance mask/custom index exceeds packed limits.");
      for (float value : source.transform) {
        TI_ERROR_IF(!std::isfinite(value),
                    "Vulkan instance transform must be finite.");
      }
      auto &destination = packed[index];
      std::memcpy(destination.transform.matrix, source.transform.data(),
                  sizeof(destination.transform.matrix));
      destination.instanceCustomIndex = source.custom_index;
      destination.mask = source.mask;
      destination.instanceShaderBindingTableRecordOffset = 0;
      destination.flags =
          VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
      destination.accelerationStructureReference =
          blases_[index]->device_address();
    }

    auto *vk_commands = static_cast<vulkan::VulkanCommandList *>(command_list);
    auto command_buffer = vk_commands->vk_command_buffer();
    VkMemoryBarrier reuse_barrier{};
    reuse_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    reuse_barrier.srcAccessMask =
        VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    reuse_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(
        command_buffer->buffer,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &reuse_barrier, 0, nullptr, 0,
        nullptr);

    const auto instance_vk_buffer =
        device_->get_vkbuffer(instance_buffer_.get_ptr());
    constexpr std::size_t kMaxUpdateBytes = 65536;
    const auto *bytes = reinterpret_cast<const std::uint8_t *>(packed.data());
    for (std::size_t offset = 0; offset < instance_bytes_;) {
      const std::size_t chunk =
          std::min(kMaxUpdateBytes, instance_bytes_ - offset);
      vkCmdUpdateBuffer(command_buffer->buffer, instance_vk_buffer->buffer,
                        offset, chunk, bytes + offset);
      offset += chunk;
    }

    VkMemoryBarrier input_barrier{};
    input_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    input_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    input_barrier.dstAccessMask =
        VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(
        command_buffer->buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1,
        &input_barrier, 0, nullptr, 0, nullptr);

    const auto geometry = make_geometry();
    auto build = make_build_info(
        geometry, update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR
                         : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR);
    const VkAccelerationStructureBuildRangeInfoKHR range{
        static_cast<std::uint32_t>(instances.size()), 0, 0, 0};
    const VkAccelerationStructureBuildRangeInfoKHR *ranges[] = {&range};
    cmd_build_(command_buffer->buffer, 1, &build, ranges);

    VkMemoryBarrier query_barrier{};
    query_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    query_barrier.srcAccessMask =
        VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    query_barrier.dstAccessMask =
        VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(
        command_buffer->buffer,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        0, 1, &query_barrier, 0, nullptr, 0, nullptr);
    retain(command_buffer);
  }

  void prepare_query_variant(unsigned variant) {
    std::lock_guard<std::mutex> lock(mutex_);
    query_.prepare(device_, variant);
  }

  void record_query(CommandList *command_list,
                    const VulkanRayQueryCommand &packet) {
    std::lock_guard<std::mutex> lock(mutex_);
    query_.record(command_list, tlas_, packet);
    retain(static_cast<vulkan::VulkanCommandList *>(command_list)
               ->vk_command_buffer());
  }

  void bind_for_kernel(ShaderResourceSet *bindings,
                       int binding,
                       CommandList *command_list) {
    std::lock_guard<std::mutex> lock(mutex_);
    TI_ERROR_IF(bindings == nullptr || command_list == nullptr || binding < 0,
                "Vulkan TLAS kernel binding is invalid.");
    auto *vulkan_bindings = static_cast<vulkan::VulkanResourceSet *>(bindings);
    auto *vulkan_commands =
        static_cast<vulkan::VulkanCommandList *>(command_list);
    vulkan_bindings->acceleration_structure(binding, tlas_);
    retain(vulkan_commands->vk_command_buffer());
  }

 private:
  DeviceAllocation allocate(std::size_t bytes, AllocUsage usage) {
    Device::AllocParams params;
    params.size = bytes;
    params.usage = usage;
    DeviceAllocation allocation{kDeviceNullAllocation};
    const auto result = device_->allocate_memory(params, &allocation);
    TI_ERROR_IF(result != RhiResult::success,
                "Failed to allocate Vulkan instance TLAS buffer ({} bytes): "
                "RhiResult({}).",
                bytes, result);
    return allocation;
  }

  void release(DeviceAllocation &allocation) noexcept {
    if (allocation != kDeviceNullAllocation && device_ != nullptr) {
      device_->dealloc_memory(allocation);
      allocation = kDeviceNullAllocation;
    }
  }

  VkDeviceAddress scratch_address() const {
    return aligned_address(device_->get_buffer_device_address(scratch_),
                           scratch_alignment_);
  }

  std::size_t scratch_allocation_bytes(VkDeviceSize bytes) const {
    TI_ERROR_IF(bytes > (std::numeric_limits<std::size_t>::max)() -
                            scratch_alignment_,
                "Vulkan instance TLAS scratch size overflow.");
    return static_cast<std::size_t>(bytes + scratch_alignment_);
  }

  VkAccelerationStructureGeometryKHR make_geometry() const {
    VkAccelerationStructureGeometryInstancesDataKHR instances{};
    instances.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instances.arrayOfPointers = VK_FALSE;
    instances.data.deviceAddress =
        device_->get_buffer_device_address(instance_buffer_);
    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances = instances;
    return geometry;
  }

  VkAccelerationStructureBuildGeometryInfoKHR make_build_info(
      const VkAccelerationStructureGeometryKHR &geometry,
      VkBuildAccelerationStructureModeKHR mode) const {
    VkAccelerationStructureBuildGeometryInfoKHR info{};
    info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                 VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    info.mode = mode;
    if (mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR) {
      info.srcAccelerationStructure = tlas_->accel;
    }
    info.dstAccelerationStructure = tlas_->accel;
    info.geometryCount = 1;
    info.pGeometries = &geometry;
    info.scratchData.deviceAddress = scratch_address();
    return info;
  }

  void create_acceleration_structure() {
    const auto geometry = make_geometry();
    VkAccelerationStructureBuildGeometryInfoKHR build{};
    build.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                  VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    build.geometryCount = 1;
    build.pGeometries = &geometry;
    const auto primitive_count =
        static_cast<std::uint32_t>(blases_.size());
    VkAccelerationStructureBuildSizesInfoKHR sizes{};
    sizes.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    get_build_sizes_(device_->vk_device(),
                     VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build,
                     &primitive_count, &sizes);
    storage_bytes_ =
        static_cast<std::size_t>(sizes.accelerationStructureSize);
    storage_ = allocate(storage_bytes_, AllocUsage::AccelerationStructureStorage);
    tlas_ = vkapi::create_acceleration_structure(
        0, device_->get_vkbuffer(storage_.get_ptr()), 0,
        sizes.accelerationStructureSize,
        VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR);
    TI_ERROR_IF(!tlas_, "Failed to create independent Vulkan instance TLAS.");
    scratch_bytes_ = scratch_allocation_bytes(
        std::max(sizes.buildScratchSize, sizes.updateScratchSize));
    scratch_ = allocate(scratch_bytes_,
                        AllocUsage::Storage | AllocUsage::DeviceAddress);
  }

  void create_query_pipeline() {
    query_.prepare(device_, 0);
  }

  void retain(const vkapi::IVkCommandBuffer &command_buffer) const {
    const std::array<DeviceAllocation, 3> allocations{
        instance_buffer_, storage_, scratch_};
    for (const auto allocation : allocations) {
      command_buffer->refs.push_back(
          device_->get_vkbuffer(allocation.get_ptr()));
    }
    command_buffer->refs.push_back(tlas_);
    for (const auto &blas : blases_) {
      blas->retain(command_buffer);
    }
  }

  Program *program_{nullptr};
  vulkan::VulkanDevice *device_{nullptr};
  std::vector<std::shared_ptr<VulkanTriangleBlasResource>> blases_;
  std::size_t instance_bytes_{0};
  std::size_t storage_bytes_{0};
  std::size_t scratch_bytes_{0};
  VkDeviceSize scratch_alignment_{1};
  DeviceAllocation instance_buffer_{kDeviceNullAllocation};
  DeviceAllocation storage_{kDeviceNullAllocation};
  DeviceAllocation scratch_{kDeviceNullAllocation};
  vkapi::IVkAccelerationStructureKHR tlas_{nullptr};
  RayQueryPipelines query_;
  PFN_vkGetAccelerationStructureBuildSizesKHR get_build_sizes_{nullptr};
  PFN_vkCmdBuildAccelerationStructuresKHR cmd_build_{nullptr};
  std::mutex mutex_;
};

bool Program::vulkan_ray_query_available() const {
  if (compile_config().arch != Arch::vulkan || !program_impl_) {
    return false;
  }
  auto *device = static_cast<vulkan::VulkanDevice *>(
      const_cast<Program *>(this)->get_compute_device());
  return device && device->vk_caps().buffer_device_address &&
         device->vk_caps().acceleration_structure &&
         device->vk_caps().ray_query;
}

std::unordered_map<std::string, std::uint64_t>
Program::vulkan_ray_query_properties() const {
  std::unordered_map<std::string, std::uint64_t> result{
      {"available", 0},
      {"buffer_device_address", 0},
      {"acceleration_structure", 0},
      {"ray_query", 0},
      {"max_geometry_count", 0},
      {"max_instance_count", 0},
      {"max_primitive_count", 0},
      {"max_per_stage_descriptor_acceleration_structures", 0},
      {"max_per_stage_descriptor_update_after_bind_acceleration_structures", 0},
      {"max_descriptor_set_acceleration_structures", 0},
      {"max_descriptor_set_update_after_bind_acceleration_structures", 0},
      {"min_acceleration_structure_scratch_offset_alignment", 0},
  };
  if (compile_config().arch != Arch::vulkan || !program_impl_) {
    return result;
  }
  auto *device = static_cast<vulkan::VulkanDevice *>(
      const_cast<Program *>(this)->get_compute_device());
  if (device == nullptr) {
    return result;
  }
  const auto &caps = device->vk_caps();
  result["buffer_device_address"] = caps.buffer_device_address;
  result["acceleration_structure"] = caps.acceleration_structure;
  result["ray_query"] = caps.ray_query;
  result["available"] = vulkan_ray_query_available();
  if (!caps.acceleration_structure) {
    return result;
  }

  VkPhysicalDeviceAccelerationStructurePropertiesKHR properties{};
  properties.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
  VkPhysicalDeviceProperties2 properties2{};
  properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  properties2.pNext = &properties;
  vkGetPhysicalDeviceProperties2(device->vk_physical_device(), &properties2);
  result["max_geometry_count"] = properties.maxGeometryCount;
  result["max_instance_count"] = properties.maxInstanceCount;
  result["max_primitive_count"] = properties.maxPrimitiveCount;
  result["max_per_stage_descriptor_acceleration_structures"] =
      properties.maxPerStageDescriptorAccelerationStructures;
  result["max_per_stage_descriptor_update_after_bind_acceleration_structures"] =
      properties.maxPerStageDescriptorUpdateAfterBindAccelerationStructures;
  result["max_descriptor_set_acceleration_structures"] =
      properties.maxDescriptorSetAccelerationStructures;
  result["max_descriptor_set_update_after_bind_acceleration_structures"] =
      properties.maxDescriptorSetUpdateAfterBindAccelerationStructures;
  result["min_acceleration_structure_scratch_offset_alignment"] =
      properties.minAccelerationStructureScratchOffsetAlignment;
  return result;
}

std::uint64_t Program::create_vulkan_triangle_ray_scene(
    Ndarray *vertices,
    Ndarray *indices,
    std::size_t vertex_count,
    std::size_t triangle_count) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_ray_query_available(),
              "Vulkan triangle ray scenes require "
              "VK_KHR_acceleration_structure and VK_KHR_ray_query.");
  TI_ERROR_IF(!vertices || !indices,
              "Vulkan triangle ray scene received a null ndarray.");
  const auto check_array = [](const char *name, Ndarray *array, DataType dtype,
                              std::size_t item_count, std::size_t width) {
    const auto element_shape = array->get_element_shape();
    const bool scalar_layout =
        element_shape.empty() &&
        array->get_nelement() == checked_mul(item_count, width, name) &&
        array->get_element_size() == sizeof(std::uint32_t);
    const bool vector_layout =
        element_shape == std::vector<int>{static_cast<int>(width)} &&
        array->get_nelement() == item_count &&
        array->get_element_size() == width * sizeof(std::uint32_t);
    TI_ERROR_IF(array->get_element_data_type() != dtype ||
                    (!scalar_layout && !vector_layout),
                "Vulkan triangle ray {} must be a compact scalar ndarray with "
                "shape (N, {}) or an AOS vector-{} ndarray with shape (N,).",
                name, width, width);
  };
  check_array("vertices", vertices, PrimitiveType::f32, vertex_count, 3);
  check_array("indices", indices, PrimitiveType::i32, triangle_count, 3);
  TI_ERROR_IF(vertices->owning_program() != this ||
                  indices->owning_program() != this,
              "Vulkan triangle ray geometry must belong to the active runtime.");

  auto leases = acquire_ndarray_leases({vertices, indices});
  pin_ndarray_launch_leases(leases);
  auto scene = std::make_shared<VulkanTriangleRayScene>(
      this, vertex_count, triangle_count);
  const auto vertex_allocation = vertices->get_device_allocation();
  const auto index_allocation = indices->get_device_allocation();
  enqueue_compute_op_lambda(
      [scene, vertex_allocation, index_allocation](Device *,
                                                   CommandList *commands) {
        scene->record_build(commands, vertex_allocation, index_allocation);
      },
      {});
  mark_runtime_submission_pending();

  std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
  TI_ERROR_IF(next_vulkan_ray_scene_handle_ == 0,
              "Vulkan ray scene handle space exhausted.");
  const std::uint64_t handle = next_vulkan_ray_scene_handle_++;
  vulkan_ray_scenes_.emplace(handle, std::move(scene));
  return handle;
}

VulkanRayQueryCommand Program::prepare_vulkan_ray_query(std::uint64_t handle,
                                                        bool instance_tlas,
                                                        Ndarray *rays,
                                                        Ndarray *hits,
                                                        std::size_t ray_count,
                                                        Ndarray *hit_indices) {
  TI_ERROR_IF(!rays || !hits, "Vulkan ray query requires rays and hits.");
  const auto ray_storage = storage::describe_ndarray_storage(*rays);
  const auto hit_storage = storage::describe_ndarray_storage(*hits);
  const auto index_storage =
      hit_indices ? storage::describe_ndarray_storage(*hit_indices)
                  : storage::DenseStorageBuildResult{};
  TI_ERROR_IF(!ray_storage || !hit_storage || (hit_indices && !index_storage),
              "Vulkan ray query requires describable dense storage.");
  return prepare_vulkan_ray_query_storage(
      handle, instance_tlas, *ray_storage.descriptor, *hit_storage.descriptor,
      ray_count, hit_indices ? &*index_storage.descriptor : nullptr);
}

VulkanRayQueryCommand Program::prepare_vulkan_ray_query_storage(
    std::uint64_t handle,
    bool instance_tlas,
    const storage::DenseStorageDescriptor &rays,
    const storage::DenseStorageDescriptor &hits,
    std::size_t ray_count,
    const storage::DenseStorageDescriptor *hit_indices) {
  TI_ERROR_IF(
      compile_config().arch != Arch::vulkan || ray_count == 0 ||
          ray_count > (std::numeric_limits<std::uint32_t>::max)(),
      "Vulkan ray query requires Vulkan storage and a positive uint32 count.");
  auto check_storage = [ray_count](const storage::DenseStorageDescriptor &value,
                                   unsigned width, DataType dtype,
                                   const char *name) {
    const auto shape = value.index_shape();
    const auto element = value.element_shape();
    const bool scalar =
        element.empty() &&
        shape == std::vector<std::int64_t>{static_cast<std::int64_t>(ray_count),
                                           width};
    const bool vector = element == std::vector<std::int64_t>{width} &&
                        shape == std::vector<std::int64_t>{
                                     static_cast<std::int64_t>(ray_count)};
    TI_ERROR_IF(value.scalar_type() != dtype || (!scalar && !vector),
                "Vulkan ray {} requires dtype {} and {}-wide records.", name,
                dtype->to_string(), width);
  };
  check_storage(rays, 8, PrimitiveType::f32, "rays");
  check_storage(hits, 4, PrimitiveType::f32, "hits");
  std::vector<const storage::DenseStorageDescriptor *> descriptors{&rays,
                                                                   &hits};
  std::vector<bool> writable{false, true};
  if (hit_indices) {
    const auto dtype = hit_indices->scalar_type();
    TI_ERROR_IF(dtype != PrimitiveType::i32 && dtype != PrimitiveType::u32,
                "Vulkan ray hit_indices requires dtype i32 or u32.");
    check_storage(*hit_indices, 4, dtype, "hit_indices");
    descriptors.push_back(hit_indices);
    writable.push_back(true);
  }
  VulkanRayQueryCommand packet;
  packet.storage = prepare_native_storage(descriptors, writable);
  packet.scene_handle = handle;
  packet.instance_tlas = instance_tlas;
  packet.ray_count = ray_count;
  packet.parameters[0] = static_cast<std::uint32_t>(ray_count);
  packet.variant = hit_indices ? 1u : 0u;
  auto *device = static_cast<vulkan::VulkanDevice *>(get_compute_device());
  const auto &limits = device->get_vk_physical_device_props().limits;
  TI_ERROR_IF(
      (ray_count + kRayQueryWorkgroupSize - 1) / kRayQueryWorkgroupSize >
          limits.maxComputeWorkGroupCount[0],
      "Vulkan ray count exceeds the device's one-dimensional dispatch limit.");
  // Descriptor offsets obey the device limit. std430 vec4 records additionally
  // need 16-byte alignment; scalar subrange addressing handles the remainder.
  const auto alignment =
      std::max<VkDeviceSize>(16, limits.minStorageBufferOffsetAlignment);
  for (std::size_t i = 0; i < descriptors.size(); ++i) {
    const auto &binding = packet.storage->binding(i);
    const auto expected =
        checked_mul(ray_count, (i == 0 ? 8 : 4) * sizeof(float), "query bytes");
    TI_ERROR_IF(
        binding.bytes != expected || (binding.pointer.offset & 3u),
        "Vulkan ray storage must be a compact four-byte-aligned record range.");
    for (std::size_t j = 0; j < i; ++j) {
      const auto &other = packet.storage->binding(j);
      TI_ERROR_IF(
          static_cast<const DeviceAllocation &>(binding.pointer) ==
                  static_cast<const DeviceAllocation &>(other.pointer) &&
              binding.pointer.offset < other.pointer.offset + other.bytes &&
              other.pointer.offset < binding.pointer.offset + binding.bytes,
          "Vulkan ray output ranges must not alias rays or each other.");
    }
    const auto remainder = binding.pointer.offset % alignment;
    auto shader_binding = binding;
    shader_binding.pointer.offset -= remainder;
    TI_ERROR_IF(remainder > limits.maxStorageBufferRange ||
                    binding.bytes > limits.maxStorageBufferRange - remainder,
                "Vulkan ray descriptor range exceeds maxStorageBufferRange.");
    shader_binding.bytes += remainder;
    packet.shader_bindings[i] = shader_binding;
    packet.parameters[i + 1] =
        static_cast<std::uint32_t>(remainder / sizeof(float));
    if (remainder) {
      packet.variant |= 2u;
    }
  }
  // Build only the selected shader variant at the cold boundary.
  auto submission_guard = acquire_runtime_resource_submission_guard();
  std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
  if (instance_tlas) {
    const auto found = vulkan_ray_resources_.find(handle);
    TI_ERROR_IF(found == vulkan_ray_resources_.end(),
                "Vulkan instance TLAS handle is stale or closed.");
    auto resource =
        std::dynamic_pointer_cast<VulkanInstanceTlasResource>(found->second);
    TI_ERROR_IF(!resource, "Vulkan ray resource is not an instance TLAS.");
    resource->prepare_query_variant(packet.variant);
  } else {
    const auto found = vulkan_ray_scenes_.find(handle);
    TI_ERROR_IF(found == vulkan_ray_scenes_.end(),
                "Vulkan triangle ray scene handle is stale or closed.");
    found->second->prepare_query_variant(packet.variant);
  }
  return packet;
}

std::size_t Program::execute_vulkan_ray_query(
    const VulkanRayQueryCommand &packet) {
  with_prepared_native_storage(*packet.storage, [&] {
    if (packet.instance_tlas) {
      std::shared_ptr<VulkanInstanceTlasResource> resource;
      {
        std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
        const auto found = vulkan_ray_resources_.find(packet.scene_handle);
        TI_ERROR_IF(found == vulkan_ray_resources_.end(),
                    "Vulkan instance TLAS handle is stale or closed.");
        resource = std::dynamic_pointer_cast<VulkanInstanceTlasResource>(
            found->second);
      }
      TI_ERROR_IF(!resource, "Vulkan ray resource is not an instance TLAS.");
      enqueue_compute_op_lambda(
          [resource, packet](Device *, CommandList *commands) {
            resource->record_query(commands, packet);
          },
          {});
    } else {
      std::shared_ptr<VulkanTriangleRayScene> scene;
      {
        std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
        const auto found = vulkan_ray_scenes_.find(packet.scene_handle);
        TI_ERROR_IF(found == vulkan_ray_scenes_.end(),
                    "Vulkan triangle ray scene handle is stale or closed.");
        scene = found->second;
      }
      enqueue_compute_op_lambda(
          [scene, packet](Device *, CommandList *commands) {
            scene->record_query(commands, packet);
          },
          {});
    }
    mark_runtime_submission_pending();
  });
  return 0;
}

std::size_t Program::vulkan_triangle_ray_query(std::uint64_t handle,
                                               Ndarray *rays,
                                               Ndarray *hits,
                                               std::size_t ray_count,
                                               Ndarray *hit_indices) {
  return execute_vulkan_ray_query(prepare_vulkan_ray_query(
      handle, false, rays, hits, ray_count, hit_indices));
}

std::size_t Program::vulkan_triangle_ray_refit(std::uint64_t handle,
                                               Ndarray *vertices,
                                               std::size_t vertex_count) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vertices,
              "Vulkan triangle ray refit received a null vertex ndarray.");
  TI_ERROR_IF(vertex_count == 0 ||
                  vertex_count > static_cast<std::size_t>(
                                     (std::numeric_limits<
                                         std::uint32_t>::max)()),
              "Vulkan triangle ray refit vertex_count must be in [1, "
              "UINT32_MAX].");
  const auto element_shape = vertices->get_element_shape();
  const bool scalar_layout =
      element_shape.empty() &&
      vertices->get_nelement() ==
          checked_mul(vertex_count, std::size_t{3}, "refit vertices") &&
      vertices->get_element_size() == sizeof(float);
  const bool vector_layout =
      element_shape == std::vector<int>{3} &&
      vertices->get_nelement() == vertex_count &&
      vertices->get_element_size() == 3 * sizeof(float);
  TI_ERROR_IF(vertices->get_element_data_type() != PrimitiveType::f32 ||
                  (!scalar_layout && !vector_layout),
              "Vulkan triangle ray refit vertices must be a compact scalar "
              "f32 ndarray with shape (N, 3) or an AOS vector-3 ndarray "
              "with shape (N,).");
  TI_ERROR_IF(vertices->owning_program() != this,
              "Vulkan triangle ray refit vertices must belong to the active "
              "runtime.");

  std::shared_ptr<VulkanTriangleRayScene> scene;
  {
    std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
    const auto found = vulkan_ray_scenes_.find(handle);
    TI_ERROR_IF(found == vulkan_ray_scenes_.end(),
                "Vulkan triangle ray scene handle is stale or closed.");
    scene = found->second;
  }
  TI_ERROR_IF(vertex_count != scene->vertex_count(),
              "Vulkan triangle ray refit must preserve vertex_count {}.",
              scene->vertex_count());
  auto leases = acquire_ndarray_leases({vertices});
  pin_ndarray_launch_leases(leases);
  const auto vertex_allocation = vertices->get_device_allocation();
  enqueue_compute_op_lambda(
      [scene, vertex_allocation](Device *, CommandList *commands) {
        scene->record_refit(commands, vertex_allocation);
      },
      {});
  mark_runtime_submission_pending();
  return 0;
}

std::uint64_t Program::create_vulkan_triangle_blas_resource(
    std::size_t vertex_count,
    std::size_t triangle_count) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_ray_query_available(),
              "Vulkan triangle BLAS resources require "
              "VK_KHR_acceleration_structure and VK_KHR_ray_query.");
  auto resource = std::make_shared<VulkanTriangleBlasResource>(
      this, vertex_count, triangle_count);
  std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
  TI_ERROR_IF(next_vulkan_ray_resource_handle_ == 0,
              "Vulkan ray resource handle space exhausted.");
  const std::uint64_t handle = next_vulkan_ray_resource_handle_++;
  vulkan_ray_resources_.emplace(handle, std::move(resource));
  return handle;
}

std::size_t Program::vulkan_triangle_blas_build(std::uint64_t handle,
                                                Ndarray *vertices,
                                                Ndarray *indices,
                                                std::size_t vertex_count,
                                                std::size_t triangle_count,
                                                bool update) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vertices || (!update && !indices),
              "Vulkan triangle BLAS build received a null ndarray.");
  const auto check_array = [](const char *name, Ndarray *array, DataType dtype,
                              std::size_t item_count, std::size_t width) {
    const auto element_shape = array->get_element_shape();
    const bool scalar_layout =
        element_shape.empty() &&
        array->get_nelement() == checked_mul(item_count, width, name) &&
        array->get_element_size() == sizeof(std::uint32_t);
    const bool vector_layout =
        element_shape == std::vector<int>{static_cast<int>(width)} &&
        array->get_nelement() == item_count &&
        array->get_element_size() == width * sizeof(std::uint32_t);
    TI_ERROR_IF(array->get_element_data_type() != dtype ||
                    (!scalar_layout && !vector_layout),
                "Vulkan triangle BLAS {} must be a compact scalar ndarray "
                "with shape (N, {}) or an AOS vector-{} ndarray with shape "
                "(N,).",
                name, width, width);
  };
  check_array("vertices", vertices, PrimitiveType::f32, vertex_count, 3);
  if (!update) {
    check_array("indices", indices, PrimitiveType::i32, triangle_count, 3);
  }
  TI_ERROR_IF(vertices->owning_program() != this ||
                  (!update && indices->owning_program() != this),
              "Vulkan triangle BLAS geometry must belong to the active "
              "runtime.");

  std::shared_ptr<VulkanTriangleBlasResource> resource;
  {
    std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
    const auto found = vulkan_ray_resources_.find(handle);
    TI_ERROR_IF(found == vulkan_ray_resources_.end(),
                "Vulkan triangle BLAS handle is stale or closed.");
    resource =
        std::dynamic_pointer_cast<VulkanTriangleBlasResource>(found->second);
  }
  TI_ERROR_IF(!resource, "Vulkan ray resource is not a triangle BLAS.");
  TI_ERROR_IF(vertex_count != resource->vertex_count() ||
                  triangle_count != resource->triangle_count(),
              "Vulkan triangle BLAS build must preserve vertex_count {} and "
              "triangle_count {}.",
              resource->vertex_count(), resource->triangle_count());

  auto leases = update ? acquire_ndarray_leases({vertices})
                       : acquire_ndarray_leases({vertices, indices});
  pin_ndarray_launch_leases(leases);
  const auto vertex_allocation = vertices->get_device_allocation();
  const auto index_allocation =
      update ? kDeviceNullAllocation : indices->get_device_allocation();
  enqueue_compute_op_lambda(
      [resource, vertex_allocation, index_allocation,
       update](Device *, CommandList *commands) {
        resource->record_build(commands, vertex_allocation, index_allocation,
                               update);
      },
      {});
  mark_runtime_submission_pending();
  return 0;
}

std::uint64_t Program::create_vulkan_instance_tlas_resource(
    const std::vector<std::uint64_t> &blas_handles) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_ray_query_available(),
              "Vulkan instance TLAS resources require "
              "VK_KHR_acceleration_structure and VK_KHR_ray_query.");
  TI_ERROR_IF(blas_handles.empty(),
              "Vulkan instance TLAS requires at least one BLAS handle.");
  std::vector<std::shared_ptr<VulkanTriangleBlasResource>> blases;
  blases.reserve(blas_handles.size());
  {
    std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
    for (const auto handle : blas_handles) {
      const auto found = vulkan_ray_resources_.find(handle);
      TI_ERROR_IF(found == vulkan_ray_resources_.end(),
                  "Vulkan instance TLAS received a stale or closed BLAS "
                  "handle.");
      auto blas =
          std::dynamic_pointer_cast<VulkanTriangleBlasResource>(found->second);
      TI_ERROR_IF(!blas,
                  "Vulkan instance TLAS dependencies must be triangle BLAS "
                  "resources.");
      blases.push_back(std::move(blas));
    }
  }
  auto resource =
      std::make_shared<VulkanInstanceTlasResource>(this, std::move(blases));
  std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
  TI_ERROR_IF(next_vulkan_ray_resource_handle_ == 0,
              "Vulkan ray resource handle space exhausted.");
  const std::uint64_t handle = next_vulkan_ray_resource_handle_++;
  vulkan_ray_resources_.emplace(handle, std::move(resource));
  return handle;
}

std::size_t Program::vulkan_instance_tlas_build(
    std::uint64_t handle,
    const std::vector<VulkanRayInstanceInfo> &instances,
    bool update) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  std::shared_ptr<VulkanInstanceTlasResource> resource;
  {
    std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
    const auto found = vulkan_ray_resources_.find(handle);
    TI_ERROR_IF(found == vulkan_ray_resources_.end(),
                "Vulkan instance TLAS handle is stale or closed.");
    resource =
        std::dynamic_pointer_cast<VulkanInstanceTlasResource>(found->second);
  }
  TI_ERROR_IF(!resource, "Vulkan ray resource is not an instance TLAS.");
  TI_ERROR_IF(instances.size() != resource->instance_count(),
              "Vulkan instance TLAS build must preserve instance count {}.",
              resource->instance_count());
  enqueue_compute_op_lambda(
      [resource, instances, update](Device *, CommandList *commands) {
        resource->record_build(commands, instances, update);
      },
      {});
  mark_runtime_submission_pending();
  return 0;
}

std::size_t Program::vulkan_instance_tlas_query(std::uint64_t handle,
                                                Ndarray *rays,
                                                Ndarray *hits,
                                                std::size_t ray_count,
                                                Ndarray *hit_indices) {
  return execute_vulkan_ray_query(prepare_vulkan_ray_query(
      handle, true, rays, hits, ray_count, hit_indices));
}

VulkanTriangleRaySceneMemoryStatistics
Program::vulkan_ray_resource_memory_statistics(std::uint64_t handle) {
  std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
  const auto found = vulkan_ray_resources_.find(handle);
  TI_ERROR_IF(found == vulkan_ray_resources_.end(),
              "Vulkan ray resource handle is stale or closed.");
  return found->second->memory_statistics();
}

std::unordered_map<std::string, std::uint64_t>
Program::vulkan_ray_kernel_resource_properties(std::uint64_t handle) {
  std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
  const auto found = vulkan_ray_resources_.find(handle);
  TI_ERROR_IF(found == vulkan_ray_resources_.end(),
              "Vulkan ray kernel resource handle is stale or closed.");
  auto resource =
      std::dynamic_pointer_cast<VulkanInstanceTlasResource>(found->second);
  TI_ERROR_IF(!resource,
              "Vulkan ray kernel resources must be top-level instance AS.");
  return {{"handle", handle},
          {"top_level", 1},
          {"read_only", 1},
          {"exact_generation", 1},
          {"instance_count", resource->instance_count()}};
}

void Program::vulkan_bind_ray_kernel_resource(
    std::uint64_t handle,
    ShaderResourceSet *bindings,
    int binding,
    CommandList *command_list) {
  std::shared_ptr<VulkanInstanceTlasResource> resource;
  {
    std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
    const auto found = vulkan_ray_resources_.find(handle);
    TI_ERROR_IF(found == vulkan_ray_resources_.end(),
                "Vulkan ray kernel resource handle is stale or closed.");
    resource =
        std::dynamic_pointer_cast<VulkanInstanceTlasResource>(found->second);
  }
  TI_ERROR_IF(!resource,
              "Vulkan ray kernel resources must be top-level instance AS.");
  resource->bind_for_kernel(bindings, binding, command_list);
}

VulkanTriangleRaySceneMemoryStatistics
Program::vulkan_triangle_ray_scene_memory_statistics(
    std::uint64_t handle) {
  std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
  const auto found = vulkan_ray_scenes_.find(handle);
  TI_ERROR_IF(found == vulkan_ray_scenes_.end(),
              "Vulkan triangle ray scene handle is stale or closed.");
  return found->second->memory_statistics();
}

std::unordered_map<std::string, std::uint64_t>
Program::debug_vulkan_ray_resource_stats() {
  std::uint64_t live = 0;
  std::uint64_t queued_for_completion = 0;
  std::uint64_t independent_live = 0;
  std::uint64_t blas_live = 0;
  std::uint64_t tlas_live = 0;
  std::uint64_t independent_queued_for_completion = 0;
  {
    std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
    live = vulkan_ray_scenes_.size();
    queued_for_completion = vulkan_ray_scene_retirements_.size();
    independent_live = vulkan_ray_resources_.size();
    independent_queued_for_completion = vulkan_ray_resource_retirements_.size();
    for (const auto &[handle, resource] : vulkan_ray_resources_) {
      if (resource->kind() == VulkanRayResourceKind::kTriangleBlas) {
        ++blas_live;
      } else {
        ++tlas_live;
      }
    }
  }
  const auto completion_retained =
      runtime_completion_resource_count(kVulkanRaySceneResourceKind);
  const auto independent_completion_retained =
      runtime_completion_resource_count(kVulkanRayResourceKind);
  return {{"live", live},
          {"retiring", queued_for_completion + completion_retained},
          {"queued_for_completion", queued_for_completion},
          {"completion_retained", completion_retained},
          {"independent_live", independent_live},
          {"blas_live", blas_live},
          {"tlas_live", tlas_live},
          {"independent_retiring", independent_queued_for_completion +
                                       independent_completion_retained},
          {"independent_queued_for_completion",
           independent_queued_for_completion},
          {"independent_completion_retained",
           independent_completion_retained}};
}

void Program::destroy_vulkan_triangle_ray_scene(std::uint64_t handle) {
  std::shared_ptr<VulkanTriangleRayScene> scene;
  bool record_retirement = false;
  {
    auto submission_guard = acquire_runtime_resource_submission_guard();
    std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
    const auto found = vulkan_ray_scenes_.find(handle);
    if (found == vulkan_ray_scenes_.end()) {
      return;
    }
    scene = found->second;
    vulkan_ray_scenes_.erase(found);
    if (!runtime_has_fatal_fault() &&
        runtime_submission_pending_.load(std::memory_order_acquire)) {
      vulkan_ray_scene_retirements_.push_back(std::move(scene));
      record_retirement = true;
    }
  }
  if (record_retirement) {
    record_runtime_completion();
  }
}

void Program::destroy_vulkan_ray_resource(std::uint64_t handle) {
  std::shared_ptr<VulkanRayResource> resource;
  bool record_retirement = false;
  {
    auto submission_guard = acquire_runtime_resource_submission_guard();
    std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
    const auto found = vulkan_ray_resources_.find(handle);
    if (found == vulkan_ray_resources_.end()) {
      return;
    }
    resource = found->second;
    vulkan_ray_resources_.erase(found);
    if (!runtime_has_fatal_fault() &&
        runtime_submission_pending_.load(std::memory_order_acquire)) {
      vulkan_ray_resource_retirements_.push_back(std::move(resource));
      record_retirement = true;
    }
  }
  if (record_retirement) {
    record_runtime_completion();
  }
}

void Program::vulkan_clear_ray_scenes() {
  std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
  vulkan_ray_scenes_.clear();
  vulkan_ray_scene_retirements_.clear();
  vulkan_ray_resources_.clear();
  vulkan_ray_resource_retirements_.clear();
}

}  // namespace taichi::lang

#else

namespace taichi::lang {

bool Program::vulkan_ray_query_available() const {
  return false;
}

std::unordered_map<std::string, std::uint64_t>
Program::vulkan_ray_query_properties() const {
  return {{"available", 0},
          {"buffer_device_address", 0},
          {"acceleration_structure", 0},
          {"ray_query", 0},
          {"max_geometry_count", 0},
          {"max_instance_count", 0},
          {"max_primitive_count", 0},
          {"max_per_stage_descriptor_acceleration_structures", 0},
          {"max_per_stage_descriptor_update_after_bind_acceleration_structures", 0},
          {"max_descriptor_set_acceleration_structures", 0},
          {"max_descriptor_set_update_after_bind_acceleration_structures", 0},
          {"min_acceleration_structure_scratch_offset_alignment", 0}};
}

std::uint64_t Program::create_vulkan_triangle_ray_scene(
    Ndarray *,
    Ndarray *,
    std::size_t,
    std::size_t) {
  TI_ERROR("Vulkan ray query requires TI_WITH_VULKAN=ON.");
}

VulkanRayQueryCommand Program::prepare_vulkan_ray_query(std::uint64_t,
                                                        bool,
                                                        Ndarray *,
                                                        Ndarray *,
                                                        std::size_t,
                                                        Ndarray *) {
  TI_ERROR("Vulkan ray query requires TI_WITH_VULKAN=ON.");
}

VulkanRayQueryCommand Program::prepare_vulkan_ray_query_storage(
    std::uint64_t,
    bool,
    const storage::DenseStorageDescriptor &,
    const storage::DenseStorageDescriptor &,
    std::size_t,
    const storage::DenseStorageDescriptor *) {
  TI_ERROR("Vulkan ray query requires TI_WITH_VULKAN=ON.");
}

std::size_t Program::execute_vulkan_ray_query(const VulkanRayQueryCommand &) {
  TI_ERROR("Vulkan ray query requires TI_WITH_VULKAN=ON.");
}

std::size_t Program::vulkan_triangle_ray_query(std::uint64_t,
                                               Ndarray *,
                                               Ndarray *,
                                               std::size_t,
                                               Ndarray *) {
  TI_ERROR("Vulkan ray query requires TI_WITH_VULKAN=ON.");
}

std::size_t Program::vulkan_triangle_ray_refit(std::uint64_t,
                                               Ndarray *,
                                               std::size_t) {
  TI_ERROR("Vulkan ray refit requires TI_WITH_VULKAN=ON.");
}

VulkanTriangleRaySceneMemoryStatistics
Program::vulkan_triangle_ray_scene_memory_statistics(std::uint64_t) {
  TI_ERROR("Vulkan ray query requires TI_WITH_VULKAN=ON.");
}

std::uint64_t Program::create_vulkan_triangle_blas_resource(std::size_t,
                                                            std::size_t) {
  TI_ERROR("Vulkan ray query requires TI_WITH_VULKAN=ON.");
}

std::size_t Program::vulkan_triangle_blas_build(std::uint64_t,
                                                Ndarray *,
                                                Ndarray *,
                                                std::size_t,
                                                std::size_t,
                                                bool) {
  TI_ERROR("Vulkan ray query requires TI_WITH_VULKAN=ON.");
}

std::uint64_t Program::create_vulkan_instance_tlas_resource(
    const std::vector<std::uint64_t> &) {
  TI_ERROR("Vulkan ray query requires TI_WITH_VULKAN=ON.");
}

std::size_t Program::vulkan_instance_tlas_build(
    std::uint64_t,
    const std::vector<VulkanRayInstanceInfo> &,
    bool) {
  TI_ERROR("Vulkan ray query requires TI_WITH_VULKAN=ON.");
}

std::size_t Program::vulkan_instance_tlas_query(std::uint64_t,
                                                Ndarray *,
                                                Ndarray *,
                                                std::size_t,
                                                Ndarray *) {
  TI_ERROR("Vulkan ray query requires TI_WITH_VULKAN=ON.");
}

VulkanTriangleRaySceneMemoryStatistics
Program::vulkan_ray_resource_memory_statistics(std::uint64_t) {
  TI_ERROR("Vulkan ray query requires TI_WITH_VULKAN=ON.");
}

std::unordered_map<std::string, std::uint64_t>
Program::vulkan_ray_kernel_resource_properties(std::uint64_t) {
  TI_ERROR("Vulkan ray query requires TI_WITH_VULKAN=ON.");
}

void Program::vulkan_bind_ray_kernel_resource(std::uint64_t,
                                              ShaderResourceSet *,
                                              int,
                                              CommandList *) {
  TI_ERROR("Vulkan ray query requires TI_WITH_VULKAN=ON.");
}

std::unordered_map<std::string, std::uint64_t>
Program::debug_vulkan_ray_resource_stats() {
  return {{"live", 0},
          {"retiring", 0},
          {"queued_for_completion", 0},
          {"completion_retained", 0},
          {"independent_live", 0},
          {"blas_live", 0},
          {"tlas_live", 0},
          {"independent_retiring", 0},
          {"independent_queued_for_completion", 0},
          {"independent_completion_retained", 0}};
}

void Program::destroy_vulkan_triangle_ray_scene(std::uint64_t) {
}

void Program::destroy_vulkan_ray_resource(std::uint64_t) {
}

void Program::vulkan_clear_ray_scenes() {
}

}  // namespace taichi::lang

#endif
